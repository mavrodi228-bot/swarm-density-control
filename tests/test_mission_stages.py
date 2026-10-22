"""Stage boundaries preserve physical state, identities, map registration and limits."""
import base64
import copy
import io
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

import agmen_api
from mission_stages import validate_stage_reachability


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "local")
    monkeypatch.setenv("AGMEN_DATA_DIR", str(tmp_path))
    app = agmen_api.create_app(tmp_path / "missions")
    with TestClient(app) as connection:
        yield connection
    app.state.executor.shutdown(wait=True)


def completed(client, identifier, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/missions/{identifier}").json()
        if record["status"] in {"completed", "failed"}:
            assert record["status"] == "completed", record
            return record
        time.sleep(.02)
    pytest.fail("Stage did not complete within its bounded test budget")


def arrived_source(client, **overrides):
    config = {"name": "Arrival source", "drone_count": 5, "steps": 20, "max_speed": 5,
              "safe_distance": .5, "obstacle": "none", "scenario": "distribution",
              "width_m": 20, "height_m": 20,
              "targets": [{"x": .4, "y": .5, "radius": .25, "weight": 1}]}
    config.update(overrides)
    response = client.post("/api/missions", json=config)
    assert response.status_code == 202, response.text
    result = completed(client, response.json()["id"])
    assert result["summary"]["mission_completed"], result["summary"]
    assert result["arrival"]["all_arrived"]
    return result


def stage_payload(source, **overrides):
    body = {"name": "Detailed stage", "scenario": "distribution", "origin_x_m": 0,
            "origin_y_m": 0, "width_m": source["workspace"]["width_m"],
            "height_m": source["workspace"]["height_m"], "steps": 20}
    body.update(overrides)
    return body


def test_arrival_default_progress_and_trusted_fields_stay_server_owned(client):
    source = arrived_source(client)
    assert source["config"]["wait_for_arrival"] is True
    assert source["config"]["max_completion_steps"] == 1200
    assert source["progress"]["all_arrived"] is True
    assert source["progress"]["actual_steps"] == source["summary"]["actual_steps"]
    for key, value in {"initial_positions_m": [[1, 1]] * 5, "source_drone_ids": [1, 2, 3, 4, 5],
                       "initial_batteries": [100] * 5, "max_completion_steps": 10000}.items():
        assert client.post("/api/missions", json={key: value}).status_code == 422
    assert client.post("/api/missions", json={"drone_count": 1}).status_code == 422


def test_registered_stage_preserves_positions_velocities_battery_ids_and_parent(client):
    source = arrived_source(client)
    options = client.get(f"/api/missions/{source['id']}/next-stage-options").json()
    assert options["allowed"], options
    assert len(options["source_agents"]) == 5
    assert {s["id"] for s in options["scenarios"]} == {"distribution", "patrol", "survey", "scan"}
    registration = options["suggested_region"]
    response = client.post(f"/api/missions/{source['id']}/next-stage", json=stage_payload(source, **registration))
    assert response.status_code == 202, response.text
    child = completed(client, response.json()["id"])
    assert child["parent_id"] == source["id"]
    assert child["stage"]["parent_mission_id"] == source["id"]
    assert child["stage"]["index"] == 2
    assert child["stage"]["registration"]["metres_per_metre"] == 1
    assert child["stage"]["registration"]["gps_registered"] is False
    np.testing.assert_allclose(child["final_state"]["source_drone_ids"], source["final_state"]["source_drone_ids"])
    initial = child["frames"][0]["agents"]
    actual_positions = np.array([[a["x"] * child["workspace"]["width_m"], a["y"] * child["workspace"]["height_m"]] for a in initial])
    np.testing.assert_allclose(actual_positions + [registration["origin_x_m"], registration["origin_y_m"]], source["final_state"]["positions_m"], atol=2e-8)
    np.testing.assert_allclose([a["battery"] for a in initial], source["final_state"]["batteries"], atol=2e-8)
    np.testing.assert_allclose([a["speed"] for a in initial], np.linalg.norm(source["final_state"]["velocities_mps"], axis=1), atol=2e-8)
    assert source["id"] in client.app.state.records
    assert client.get(f"/api/missions/{source['id']}").json()["frames"] == source["frames"]
    assert client.get(f"/api/missions/{source['id']}/next-stage-options").json()["next_stages"][0]["id"] == child["id"]
    assert client.post(f"/api/missions/{source['id']}/events", json={"kind": "drone_failure", "drone_id": 1}).status_code == 409


def test_stage_rejects_nonarrival_failed_or_held_participants(client):
    source = arrived_source(client)
    stored = client.app.state.records[source["id"]]
    original = copy.deepcopy(stored)
    for mutate in (lambda r: r.update(status="running"),
                   lambda r: r["arrival"].update(all_arrived=False),
                   lambda r: r["frames"][-1]["agents"][0].update(status="failed"),
                   lambda r: r["frames"][-1]["agents"][0].update(status="hold"),
                   lambda r: r["frames"][-1]["agents"][0].update(link="lost")):
        stored.clear()
        stored.update(copy.deepcopy(original))
        mutate(stored)
        assert not client.get(f"/api/missions/{source['id']}/next-stage-options").json()["allowed"]
        assert client.post(f"/api/missions/{source['id']}/next-stage", json=stage_payload(source)).status_code == 409


def test_no_automatic_spawn_repair_or_cross_origin_stage_write(client):
    source = arrived_source(client)
    count = len(client.get("/api/missions").json())
    url = f"/api/missions/{source['id']}/next-stage"
    assert client.post(url, json=stage_payload(source, origin_x_m=100)).status_code == 422
    assert client.post(url, json=stage_payload(source), headers={"origin": "https://untrusted.invalid"}).status_code == 403
    blocked = [{"kind": "rectangle", "x": 0, "y": 0, "width": 1, "height": 1}]
    assert client.post(url, json=stage_payload(source, restricted_zones=blocked)).status_code == 422
    assert len(client.get("/api/missions").json()) == count


def test_current_map_calibration_controls_detail_geometry(client):
    source = arrived_source(client)
    image = io.BytesIO()
    Image.new("RGB", (24, 24), "white").save(image, format="PNG")
    uploaded = client.post("/api/maps", json={"name": "Registered detail", "width_m": 20, "height_m": 20,
        "image_data_url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()}).json()
    response = client.post(f"/api/missions/{source['id']}/next-stage", json=stage_payload(source, map_id=uploaded["id"], width_m=100, height_m=100))
    assert response.status_code == 202, response.text
    child = completed(client, response.json()["id"])
    assert child["workspace"]["width_m"] == 20
    assert child["workspace"]["height_m"] == 20
    assert child["territory"]["id"] == uploaded["id"]


def test_zone_subset_can_continue_less_than_five_without_lowering_creation_minimum(client):
    source = arrived_source(client, targets=[{"x": .45, "y": .28, "radius": .2, "weight": 1}, {"x": .45, "y": .75, "radius": .2, "weight": 1}])
    ids = [a["source_drone_id"] for a in source["frames"][-1]["agents"] if a["target_zone_index"] == 0]
    assert 0 < len(ids) < 5
    response = client.post(f"/api/missions/{source['id']}/next-stage", json=stage_payload(source, target_zone_index=0))
    assert response.status_code == 202, response.text
    child = completed(client, response.json()["id"])
    assert child["config"]["drone_count"] == len(ids)
    assert child["config"]["source_drone_ids"] == ids
    assert client.post("/api/missions", json={"drone_count": len(ids)}).status_code == 422


def test_chunk_continuation_uses_same_assigned_goals_and_actual_state(client):
    response = client.post("/api/missions", json={"name": "Long plane", "drone_count": 5, "steps": 20,
        "width_m": 1000, "height_m": 1000, "max_speed": 1, "safe_distance": .5, "obstacle": "none", "scenario": "distribution"})
    assert response.status_code == 202
    source = completed(client, response.json()["id"])
    assert source["summary"]["termination_reason"] == "step_limit", source["summary"]
    options = client.get(f"/api/missions/{source['id']}/next-stage-options").json()
    assert options["can_continue"] and not options["allowed"]
    response = client.post(f"/api/missions/{source['id']}/continue", json={"steps": 20})
    assert response.status_code == 202, response.text
    child = completed(client, response.json()["id"])
    assert child["stage"]["kind"] == "arrival_continuation"
    np.testing.assert_allclose(child["config"]["initial_positions_m"], source["final_state"]["positions_m"])
    np.testing.assert_allclose(child["config"]["initial_batteries"], source["final_state"]["batteries"])
    np.testing.assert_allclose(child["config"]["initial_assigned_slots_m"], source["final_state"]["assigned_slots_m"])
    assert child["config"]["initial_target_zone_indices"] == source["final_state"]["target_zone_indices"]
    assert child["final_state"]["elapsed_steps"] > source["final_state"]["elapsed_steps"]


def test_grid_reachability_rejects_disconnected_destination_before_queueing():
    config = agmen_api.MissionConfig(drone_count=5, steps=20, width_m=20, height_m=20,
        max_speed=3, safe_distance=.5, obstacle="none", scenario="distribution").model_dump()
    config["initial_positions_m"] = [[2, y] for y in (2, 5, 8, 11, 14)]
    validate_stage_reachability(config)
    config["restricted_zones"] = [{"kind": "rectangle", "x": .45, "y": 0, "width": .1, "height": 1}]
    with pytest.raises(ValueError, match="отделена"):
        validate_stage_reachability(config)


def test_grid_reachability_checks_final_frozen_goal_between_five_step_samples():
    config = dict(drone_count=1, scenario="scan", steps=22, width_m=120, height_m=90,
                  max_speed=5, safe_distance=.5, obstacle="none", controller="baseline", seed=42,
                  initial_positions_m=[[10, 20]], restricted_zones=[
                      {"kind": "rectangle", "x": .8, "y": 0, "width": .01, "height": 1}])
    # At step 20 the slot is reachable on the left side of the wall. The
    # simulator freezes the actual step-22 destination on the other side.
    from agmen_simulation import _zones
    before_freeze = copy.deepcopy(config)
    before_freeze["targets"] = _zones(config, 20)
    validate_stage_reachability(before_freeze)
    with pytest.raises(ValueError, match="отделена"):
        validate_stage_reachability(config)


def test_source_restrictions_and_child_archive_survive_a_new_api_instance(client, tmp_path):
    source = arrived_source(client, restricted_zones=[{"kind": "rectangle", "x": .9, "y": .9, "width": .05, "height": .05}])
    response = client.post(f"/api/missions/{source['id']}/next-stage", json=stage_payload(source))
    assert response.status_code == 202, response.text
    child = completed(client, response.json()["id"])
    assert child["stage"]["inherited_restriction_count"] > 0
    assert any(z.get("id", "").startswith("source-restriction-") for z in child["restricted_zones"])
    assert np.all(np.asarray(child["obstacle"])[np.asarray(source["obstacle"], dtype=bool)])
    restored = agmen_api.create_app(tmp_path / "missions")
    with TestClient(restored) as connection:
        loaded = connection.get(f"/api/missions/{child['id']}").json()
        assert loaded["parent_id"] == source["id"]
        assert loaded["stage"] == child["stage"]
        assert loaded["frames"] == child["frames"]
    restored.state.executor.shutdown(wait=True)


def test_stage_creation_obeys_public_profile_budget(client, tmp_path, monkeypatch):
    source = arrived_source(client)
    monkeypatch.setenv("AGMEN_PROFILE", "public")
    public = agmen_api.create_app(tmp_path / "missions")
    with TestClient(public) as connection:
        url = f"/api/missions/{source['id']}/next-stage"
        assert connection.post(url, json=stage_payload(source, steps=151)).status_code == 422
        response = connection.post(url, json=stage_payload(source, steps=20))
        assert response.status_code == 202, response.text
        child = completed(connection, response.json()["id"])
        assert child["config"]["max_completion_steps"] == 600
    public.state.executor.shutdown(wait=True)
