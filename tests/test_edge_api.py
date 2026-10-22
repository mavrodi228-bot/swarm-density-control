import time
import zipfile
import io

from fastapi.testclient import TestClient
import pytest

from agmen_api import create_app


@pytest.fixture
def edge_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "local")
    monkeypatch.setenv("AGMEN_DATA_DIR", str(tmp_path))
    app = create_app(tmp_path / "missions")
    with TestClient(app) as client:
        yield client
    app.state.executor.shutdown(wait=True)


def wait_mission(client, identifier):
    for _ in range(500):
        record = client.get(f"/api/missions/{identifier}").json()
        if record["status"] in {"completed", "failed"}:
            assert record["status"] == "completed", record
            return record
        time.sleep(.01)
    pytest.fail("Mission timeout")


def run_steps(client, identifier, steps):
    response = client.post(f"/api/edge/devices/{identifier}/run", json={"steps": steps})
    assert response.status_code == 202, response.text
    for _ in range(1000):
        snapshot = client.get(f"/api/edge/devices/{identifier}").json()
        if snapshot["job_status"] == "idle":
            assert not snapshot.get("error"), snapshot
            return snapshot
        time.sleep(.01)
    pytest.fail("EDGE timeout")


def make_device(client, policy="continue", coordination="independent"):
    mission = client.post("/api/missions", json={"drone_count": 5, "steps": 20, "obstacle": "none"}).json()
    wait_mission(client, mission["id"])
    response = client.post("/api/edge/packages", json={"mission_id": mission["id"], "drone_id": 1, "policy": policy, "coordination": coordination})
    assert response.status_code == 201, response.text
    package = response.json()
    response = client.post("/api/edge/devices", json={"package_id": package["id"]})
    assert response.status_code == 201, response.text
    return package, response.json()


def test_package_build_download_load_and_real_offline_execution(edge_client):
    client = edge_client
    assert client.get("/api/edge/status").json()["devices"] == []
    package, initial = make_device(client)
    archive = client.get(f"/api/edge/packages/{package['id']}/download")
    assert archive.status_code == 200
    assert zipfile.is_zipfile(io.BytesIO(archive.content))
    assert initial["package_loaded"] is True
    identifier = initial["id"]
    online = run_steps(client, identifier, 5)
    assert online["step"] == 5
    response = client.post(f"/api/edge/devices/{identifier}/connectivity", json={"internet": False})
    assert response.status_code == 200, response.text
    offline = run_steps(client, identifier, 5)
    assert offline["step"] == 10
    assert offline["mode"] == "OFFLINE"
    assert offline["connectivity"]["ground_radio"] is True
    assert offline["telemetry_kind"] == "confirmed"
    assert offline["position_m"] != online["position_m"]
    assert offline["journal_entries"] > online["journal_entries"]


def test_ground_radio_loss_keeps_last_known_telemetry_until_sync(edge_client):
    client = edge_client
    _, initial = make_device(client, policy="continue")
    identifier = initial["id"]
    previous = run_steps(client, identifier, 3)
    response = client.post(f"/api/edge/devices/{identifier}/connectivity", json={"ground_radio": False, "peer_radio": False})
    assert response.status_code == 200
    lost = run_steps(client, identifier, 4)
    assert lost["mode"] == "LINK LOST"
    assert lost["telemetry_kind"] == "last_known"
    assert lost["telemetry"] == previous["telemetry"]
    assert lost["position_m"] != previous["position_m"]
    assert client.post(f"/api/edge/devices/{identifier}/sync").status_code == 409
    assert client.post(f"/api/edge/devices/{identifier}/connectivity", json={"ground_radio": True}).status_code == 200
    synced = client.post(f"/api/edge/devices/{identifier}/sync")
    assert synced.status_code == 200, synced.text
    assert synced.json()["telemetry_kind"] == "confirmed"
    assert synced.json()["last_sync_at"]
    assert synced.json()["ground_receipt"]["step"] == 7
    assert synced.json()["synced_entries"] >= 1
    store = client.app.state.edge_store
    assert (store.ground_dir / f"{identifier}.json").is_file()
    receipt = client.get(f"/api/edge/devices/{identifier}/journal")
    assert receipt.status_code == 200
    journal = receipt.json()["snapshot"]["journal"]
    assert len({row["seq"] for row in journal}) == len(journal)
    assert receipt.json()["snapshot"]["step"] == 7


def test_edge_api_validates_unknown_devices_and_mission_ownership(edge_client):
    client = edge_client
    assert client.get("/api/edge/devices/not-an-id").status_code == 404
    assert client.post("/api/edge/devices", json={"package_id": "a"*32}).status_code == 404
    assert client.post("/api/edge/packages", json={"mission_id": "a"*12, "drone_id": 1}).status_code == 404
    package, initial = make_device(client)
    assert client.post(f"/api/edge/devices/{initial['id']}/run", json={"steps": 301}).status_code == 422
    assert client.post(f"/api/edge/devices/{initial['id']}/connectivity", json={"gnss": False}).status_code == 200
    result = run_steps(client, initial["id"], 2)
    assert result["position_m"] == initial["position_m"]


def test_edge_state_and_ground_receipt_survive_application_restart(edge_client):
    client = edge_client
    _, initial = make_device(client)
    identifier = initial["id"]
    before = run_steps(client, identifier, 4)
    assert client.post(f"/api/edge/devices/{identifier}/sync", json={}).status_code == 200
    store = client.app.state.edge_store
    restarted_app = create_app(store.directory.parent / "missions")
    with TestClient(restarted_app) as restarted:
        response = restarted.get(f"/api/edge/devices/{identifier}")
        assert response.status_code == 200, response.text
        after = response.json()
        assert after["step"] == 4
        assert after["position_m"] == before["position_m"]
        assert after["ground_receipt"]["step"] == 4
        assert after["last_sync_at"]
    restarted_app.state.executor.shutdown(wait=True)
