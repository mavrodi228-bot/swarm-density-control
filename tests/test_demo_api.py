"""Jury access remains usable while private records and training stay isolated."""
import base64
import io
import json
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

import agmen_api


TOKEN = "private-test-token-with-more-than-24-characters"
AUTH = {"X-AGMEN-Token": TOKEN}
DEMO = "/demo/api"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "private")
    monkeypatch.setenv("AGMEN_API_TOKEN", TOKEN)
    monkeypatch.setenv("AGMEN_DATA_DIR", str(tmp_path))
    app = agmen_api.create_app(data_dir=tmp_path)
    with TestClient(app) as connection:
        yield connection
    app.state.executor.shutdown(wait=True)


def finish(client, identifier, prefix=DEMO, headers=None):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get(f"{prefix}/missions/{identifier}", headers=headers)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] in {"completed", "failed"}:
            assert record["status"] == "completed", record
            return record
        time.sleep(.01)
    pytest.fail("The bounded demo job did not complete")


def make_mission(client, prefix=DEMO, headers=None, **overrides):
    config = {"name": "Jury mission", "drone_count": 5, "steps": 20,
              "max_speed": 5, "safe_distance": .5, "obstacle": "none",
              "width_m": 20, "height_m": 20, "wait_for_arrival": False}
    config.update(overrides)
    response = client.post(f"{prefix}/missions", json=config, headers=headers)
    assert response.status_code == 202, response.text
    return finish(client, response.json()["id"], prefix, headers)


def upload(client, prefix=DEMO, headers=None):
    image = Image.fromarray(np.random.default_rng(9).integers(0, 255, (320, 320, 3), dtype=np.uint8))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    body = {"name": "Uploaded jury territory", "width_m": 20, "height_m": 20,
            "image_data_url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}
    assert len(json.dumps(body)) > 256_000
    response = client.post(f"{prefix}/maps", json=body, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def test_demo_is_keyless_but_private_archives_and_maps_remain_hidden(client):
    private = make_mission(client, "/api", AUTH, name="Private research")
    private_map = upload(client, "/api", AUTH)
    assert client.get("/api/bootstrap").status_code == 401
    assert client.get(f"/api/missions/{private['id']}").status_code == 401
    assert client.get("/api/maps").status_code == 401
    assert client.get("/api/edge/status").status_code == 401
    bootstrap = client.get(DEMO + "/bootstrap").json()
    assert bootstrap["missions"] == []
    assert bootstrap["profile"]["name"] == "public"
    assert bootstrap["profile"]["demo"] is True
    assert bootstrap["profile"]["token_required"] is False
    assert bootstrap["profile"]["training_allowed"] is False
    assert bootstrap["profile"]["limits"]["max_drones"] == 100
    assert client.post(DEMO + "/missions", json={"drone_count": 101}).status_code == 422
    assert client.post(DEMO + "/missions", json={"steps": 151}).status_code == 422
    assert client.get(DEMO + "/health").json()["demo"] is True
    assert client.get(DEMO + "/maps").json() == []
    assert client.get(f"{DEMO}/missions/{private['id']}", headers=AUTH).status_code == 404
    assert client.get(f"{DEMO}/missions/{private['id']}/export").status_code == 404
    assert client.get(f"{DEMO}/missions/{private['id']}/next-stage-options").status_code == 404
    assert client.get(f"{DEMO}/maps/{private_map['id']}/image").status_code == 404
    public = make_mission(client)
    assert client.get(f"/api/missions/{public['id']}", headers=AUTH).status_code == 404
    assert len(client.get("/api/missions", headers=AUTH).json()) == 1
    assert client.get(f"{DEMO}/missions/{public['id']}/export?format=csv").status_code == 200
    assert TOKEN not in client.get(DEMO + "/bootstrap").text


def test_mounted_origin_body_limit_and_map_urls_preserve_canonical_disk_state(client):
    record = upload(client, headers={"origin": "http://testserver"})
    assert record["image_url"] == f"{DEMO}/maps/{record['id']}/image"
    assert client.get(record["image_url"]).headers["content-type"] == "image/png"
    assert client.get(f"{DEMO}/maps/{record['id']}").json()["image_url"] == record["image_url"]
    calibrated = client.post(f"{DEMO}/maps/{record['id']}/calibrate", json={"width_m": 25, "height_m": 20})
    assert calibrated.json()["image_url"] == record["image_url"]
    analyzed = client.post(f"{DEMO}/maps/{record['id']}/analyze", json={"method": "water"})
    assert analyzed.json()["image_url"] == record["image_url"]
    assert client.post(DEMO + "/missions", json={}, headers={"origin": "https://untrusted.invalid"}).status_code == 403
    assert client.post(DEMO + "/missions", json={}, headers={"origin": "http://localhost:5173"}).status_code == 403
    demo = client.app.state.demo_app
    canonical = json.loads((demo.state.maps.directory / f"{record['id']}.json").read_text(encoding="utf-8"))
    assert canonical["image_url"] == f"/api/maps/{record['id']}/image"
    mission = make_mission(client, map_id=record["id"])
    assert mission["territory"]["image_url"] == record["image_url"]
    assert mission["config"]["map_snapshot"]["image_url"] == record["image_url"]
    stored = json.loads((demo.state.data_directory / "missions" / f"{mission['id']}.json").read_text(encoding="utf-8"))
    assert stored["territory"]["image_url"] == canonical["image_url"]
    assert client.get(f"{DEMO}/missions/{mission['id']}/export").json()["territory"]["image_url"] == record["image_url"]


def test_demo_lab_does_not_resolve_private_runs_or_checkpoints(client):
    identifier = "b" * 32
    folder = client.app.state.data_directory / "lab" / "runs" / identifier
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"id": identifier, "status": "completed", "validated": True,
        "created_at": "2026-01-01", "history": [], "logs": ["PRIVATE LAB RESULT"]}), encoding="utf-8")
    assert client.get("/api/lab/runs", headers=AUTH).json()[0]["id"] == identifier
    assert client.get(DEMO + "/lab/runs").json() == []
    models = client.get(DEMO + "/lab/models").json()
    assert [model["id"] for model in models] == ["bundled"]
    for route in ("runs", f"runs/{identifier}/resume", f"runs/{identifier}/stop"):
        assert client.post(f"{DEMO}/lab/{route}", json={}).status_code == 403
    assert client.get(f"{DEMO}/lab/runs/{identifier}").status_code == 403
    assert client.get(f"{DEMO}/lab/models/{identifier}/download").status_code == 403
    assert client.post(DEMO + "/missions", json={"model_id": identifier}).status_code == 422
    assert not (client.app.state.demo_app.state.data_directory / "lab").exists()


def test_demo_stage_and_edge_use_only_demo_state(client):
    territory = upload(client)
    source = make_mission(client, map_id=territory["id"], wait_for_arrival=True,
                          targets=[{"x": .4, "y": .5, "radius": .25, "weight": 1}])
    assert source["arrival"]["all_arrived"]
    options = client.get(f"{DEMO}/missions/{source['id']}/next-stage-options").json()
    assert options["allowed"]
    assert options["source_territory"]["image_url"] == territory["image_url"]
    response = client.post(f"{DEMO}/missions/{source['id']}/next-stage", json={
        "name": "Detailed jury stage", "scenario": "distribution", "steps": 20,
        "map_id": territory["id"], "origin_x_m": 0, "origin_y_m": 0})
    assert response.status_code == 202, response.text
    stage = finish(client, response.json()["id"])
    assert stage["parent_id"] == source["id"]
    assert stage["territory"]["image_url"] == territory["image_url"]
    package = client.post(DEMO + "/edge/packages", json={"mission_id": source["id"], "drone_id": 1, "policy": "continue"})
    assert package.status_code == 201, package.text
    package_id = package.json()["id"]
    assert client.get(f"{DEMO}/edge/packages/{package_id}/download").content.startswith(b"PK")
    response = client.post(DEMO + "/edge/devices", json={"package_id": package_id})
    assert response.status_code == 201, response.text
    device_id = response.json()["id"]
    assert client.post(f"{DEMO}/edge/devices/{device_id}/connectivity", json={"internet": False}).status_code == 200
    assert client.post(f"{DEMO}/edge/devices/{device_id}/run", json={"steps": 3}).status_code == 202
    for _ in range(500):
        device = client.get(f"{DEMO}/edge/devices/{device_id}").json()
        if device["job_status"] == "idle":
            break
        time.sleep(.01)
    assert device["step"] == 3 and device["mode"] == "OFFLINE"
    assert client.post(f"{DEMO}/edge/devices/{device_id}/sync", json={}).status_code == 200
    assert client.get(f"{DEMO}/edge/devices/{device_id}/journal").status_code == 200
    assert client.get("/api/edge/status", headers=AUTH).json()["devices"] == []
    assert client.get(f"/api/edge/devices/{device_id}", headers=AUTH).status_code == 404
    assert client.app.state.demo_app.state.edge_store.directory == client.app.state.data_directory / "demo" / "edge"


def test_demo_and_private_jobs_share_the_same_bounded_worker(client, monkeypatch):
    demo = client.app.state.demo_app
    assert demo.state.executor is client.app.state.executor
    assert demo.state.compute_resources[1] is client.app.state.compute_resources[1]
    release = threading.Event()
    def blocked(*args, **kwargs):
        release.wait(10)
        return {"summary": {}}
    monkeypatch.setattr(agmen_api, "simulate", blocked)
    try:
        assert client.post("/api/missions", json={"steps": 20}, headers=AUTH).status_code == 202
        assert client.post(DEMO + "/missions", json={"steps": 20}).status_code == 202
        assert client.post(DEMO + "/missions", json={"steps": 20}).status_code == 202
        assert client.post(DEMO + "/missions", json={"steps": 20}).status_code == 429
    finally:
        release.set()


def test_demo_rotates_more_than_twenty_runs_without_touching_private_archive(client):
    private = make_mission(client, "/api", AUTH)
    private_path = client.app.state.data_directory / "missions" / f"{private['id']}.json"
    original_private = private_path.read_bytes()
    identifiers = [make_mission(client)["id"] for _ in range(22)]
    saved = client.get(DEMO + "/missions").json()
    assert len(saved) == 20
    assert {record["id"] for record in saved} == set(identifiers[-20:])
    assert client.get(f"{DEMO}/missions/{identifiers[0]}").status_code == 404
    assert len(list((client.app.state.demo_app.state.data_directory / "missions").glob("*.json"))) == 20
    assert private_path.read_bytes() == original_private
    assert client.get(f"/api/missions/{private['id']}", headers=AUTH).status_code == 200


def test_demo_rotation_preserves_parent_of_the_new_stage(client):
    source = make_mission(client, wait_for_arrival=True,
                          targets=[{"x": .4, "y": .5, "radius": .25, "weight": 1}])
    demo = client.app.state.demo_app
    # Make the real source the oldest entry; it still must survive its child's
    # admission while another terminal demo record is rotated.
    demo.state.records[source["id"]]["created_at"] = "2020-01-01"
    for index in range(19):
        identifier = f"{index:012x}"
        record = {"id": identifier, "kind": "mission", "status": "failed", "name": "Expired demo",
                  "created_at": f"2021-01-{index+1:02d}", "config": {}}
        demo.state.records[identifier] = record
        (demo.state.data_directory / "missions" / f"{identifier}.json").write_text(json.dumps(record), encoding="utf-8")
    response = client.post(f"{DEMO}/missions/{source['id']}/next-stage", json={
        "scenario": "distribution", "steps": 20, "origin_x_m": 0, "origin_y_m": 0,
        "width_m": 20, "height_m": 20})
    assert response.status_code == 202, response.text
    stage = finish(client, response.json()["id"])
    assert stage["parent_id"] == source["id"]
    assert client.get(f"{DEMO}/missions/{source['id']}").status_code == 200
    assert client.get(f"{DEMO}/missions/{'0'*12}").status_code == 404


def test_vite_origin_is_inherited_only_from_a_local_parent(tmp_path):
    app = agmen_api.create_app(profile="local", data_dir=tmp_path / "dev")
    with TestClient(app) as connection:
        mission = make_mission(connection, headers={"origin": "http://localhost:5173"})
        assert mission["simulation_only"] is True
    app.state.executor.shutdown(wait=True)


def test_external_https_demo_image_url_is_public_on_a_private_server(tmp_path, monkeypatch):
    hostname = "agmen-petr234512.msk1.amvera.tech"
    monkeypatch.setenv("AGMEN_PROFILE", "private")
    monkeypatch.setenv("AGMEN_API_TOKEN", TOKEN)
    monkeypatch.setenv("AGMEN_ALLOWED_HOSTS", hostname)
    app = agmen_api.create_app(data_dir=tmp_path)
    with TestClient(app, base_url=f"https://{hostname}") as connection:
        territory = upload(connection, headers={"origin": f"https://{hostname}"})
        assert territory["image_url"].startswith("/demo/api/maps/")
        assert connection.get(f"https://{hostname}{territory['image_url']}").status_code == 200
        assert connection.get("/api/maps").status_code == 401
        assert connection.get(DEMO + "/bootstrap").json()["profile"]["training_allowed"] is False
    app.state.executor.shutdown(wait=True)
