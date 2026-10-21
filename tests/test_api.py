"""API integration tests: physical states, durable jobs, faults, and boundaries."""
import json
import math
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

import agmen_api


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "local")
    app = agmen_api.create_app(tmp_path)
    with TestClient(app) as connection:
        yield connection
    app.state.executor.shutdown(wait=True)


def completed(client, identifier, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get(f"/api/missions/{identifier}").json()
        if result["status"] in {"completed", "failed"}:
            assert result["status"] == "completed", result
            return result
        time.sleep(.02)
    pytest.fail("Mission did not complete")


def mission(client, **overrides):
    config = dict(name="API integration", drone_count=5, steps=20, max_speed=2,
                  safe_distance=1, obstacle="none", scenario="clusters")
    config.update(overrides)
    response = client.post("/api/missions", json=config)
    assert response.status_code == 202, response.text
    return completed(client, response.json()["id"])


def test_simulated_states_are_consistent_with_integration_and_density(client):
    result = mission(client)
    assert result["simulation_only"] is True
    assert len(result["frames"]) == 21
    assert result["frames"][0]["agents"][0]["battery"] == 100
    previous = None
    for frame in result["frames"]:
        density = np.array(frame["density"])
        assert density.shape == (24, 24)
        assert abs(density.sum() - 1) < .00001
        assert abs(np.array(frame["target"]).sum() - 1) < .00001
        agents = frame["agents"]
        assert len(agents) == 5
        positions = np.array([[a["x"] * 80, a["y"] * 60] for a in agents])
        assert np.all(positions >= 0)
        assert np.all(positions <= [80, 60])
        if previous is not None:
            speeds = np.linalg.norm(positions - previous, axis=1) / .2
            assert max(speeds) <= 2 + 1e-6
            assert np.allclose(speeds, [a["speed"] for a in agents])
        previous = positions
    assert result["summary"]["controller_composition"]["field_weight"] == .35
    assert result["summary"]["convergence_time_s"] is None or result["summary"]["convergence_time_s"] >= 0


def test_failure_replay_preserves_physical_prefix_and_holds_vehicle(client):
    original = mission(client)
    response = client.post(f"/api/missions/{original['id']}/events", json={"kind": "drone_failure", "drone_id": 1, "step": 7})
    assert response.status_code == 202
    replay = completed(client, original["id"])
    for step in range(7):
        assert replay["frames"][step]["agents"] == original["frames"][step]["agents"]
    assert replay["frames"][7]["agents"][0]["status"] == "failed"
    position = replay["frames"][7]["agents"][0]
    assert all(f["agents"][0]["x"] == position["x"] and f["agents"][0]["y"] == position["y"] for f in replay["frames"][7:])
    assert replay["summary"]["active_drones"] == 4


def test_exports_are_real_artifacts(client):
    run = mission(client)
    base = f"/api/missions/{run['id']}/export?format="
    assert client.get(base + "json").json()["id"] == run["id"]
    assert "tv_distance" in client.get(base + "csv").text
    assert client.get(base + "png").content.startswith(b"\x89PNG")
    assert b"<svg" in client.get(base + "svg").content
    assert client.get(base + "pdf").content.startswith(b"%PDF")


def test_supports_more_than_legacy_hundred_vehicles(client):
    result = mission(client, drone_count=125)
    assert len(result["frames"][-1]["agents"]) == 125


def test_validation_origin_and_path_boundaries(client):
    assert client.get("/api/health").json()["app"] == "AGMEN"
    assert client.get("/api/bootstrap").json()["controllers"][0]["available"]
    assert client.post("/api/missions", json={"drone_count": 501}).status_code == 422
    assert client.post("/api/missions", json={"targets": [{"x": .5, "y": .5, "weight": -1}]}).status_code == 422
    assert client.post("/api/missions", json={"steps": 20}, headers={"origin": "https://attacker.invalid"}).status_code == 403
    assert client.get("/api/missions/not-an-id").status_code == 404
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/api/health", headers={"host": "attacker.invalid"}).status_code == 400


def test_private_token_and_public_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "private")
    monkeypatch.setenv("AGMEN_API_TOKEN", "a-strong-development-token-12345")
    app = agmen_api.create_app(tmp_path / "private")
    with TestClient(app) as connection:
        assert connection.get("/api/health").status_code == 200
        assert connection.get("/api/bootstrap").status_code == 401
        assert connection.get("/api/bootstrap", headers={"X-AGMEN-Token": "a-strong-development-token-12345"}).status_code == 200
    app.state.executor.shutdown(wait=True)
    monkeypatch.setenv("AGMEN_PROFILE", "public")
    public = agmen_api.create_app(tmp_path / "public")
    with TestClient(public) as connection:
        assert connection.post("/api/missions", json={"drone_count": 101}).status_code == 422
    public.state.executor.shutdown(wait=True)


def test_bounded_background_queue(client, monkeypatch):
    release = threading.Event()
    def blocked(*args, **kwargs):
        release.wait(10)
        return {"summary": {}}
    monkeypatch.setattr(agmen_api, "simulate", blocked)
    try:
        for _ in range(3):
            assert client.post("/api/missions", json={"steps": 20}).status_code == 202
        assert client.post("/api/missions", json={"steps": 20}).status_code == 429
    finally:
        release.set()


def test_interrupted_run_is_recovered_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "local")
    identifier = "a" * 12
    (tmp_path / f"{identifier}.json").write_text(json.dumps({"id": identifier, "kind": "mission", "status": "running", "created_at": "2026-01-01"}), encoding="utf-8")
    app = agmen_api.create_app(tmp_path)
    with TestClient(app) as connection:
        record = connection.get(f"/api/missions/{identifier}").json()
        assert record["status"] == "failed"
        assert "restarted" in record["error"]
    app.state.executor.shutdown(wait=True)


def test_controller_comparison_and_math_parameter_search(client):
    cases = [("experiments", {"counts": [5, 10], "controllers": ["baseline"], "steps": 20}),
             ("optimization", {"gains": [.5, 1.0], "steps": 20})]
    for endpoint, body in cases:
        response = client.post(f"/api/{endpoint}", json=body)
        assert response.status_code == 202, response.text
        identifier = response.json()["id"]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            result = client.get(f"/api/{endpoint}/{identifier}").json()
            if result["status"] in {"completed", "failed"}:
                break
            time.sleep(.02)
        assert result["status"] == "completed", result
        assert len(result["results"]) == 2
        assert all(r["compute_ms"] > 0 and len(r["history"]) == 21 for r in result["results"])
        if endpoint == "optimization":
            assert result["best"]["mae"] == min(r["mae"] for r in result["results"])
            assert "grid search" in result["summary"]["method"]


def test_memory_admission_rejects_before_starting_new_job(client, monkeypatch):
    monkeypatch.setenv("AGMEN_MAX_RSS_MB", "1")
    assert client.post("/api/missions", json={"steps": 20}).status_code == 503


def test_external_deployment_host_keeps_loopback_health_and_explicit_https_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "public")
    monkeypatch.setenv("AGMEN_ALLOWED_HOSTS", "agmen.example")
    monkeypatch.delenv("AGMEN_ALLOWED_ORIGINS", raising=False)
    app = agmen_api.create_app(tmp_path)
    with TestClient(app, base_url="http://agmen.example") as connection:
        assert connection.get("/api/health", headers={"host": "127.0.0.1:7860"}).status_code == 200
        assert connection.get("/api/health", headers={"host": "localhost:7860"}).status_code == 200
        assert connection.get("/api/health", headers={"host": "unknown.example"}).status_code == 400
        body = {"steps": 20, "drone_count": 5, "obstacle": "none"}
        headers = {"origin": "https://agmen.example"}
        assert connection.post("/api/missions", json=body, headers=headers).status_code == 403
        monkeypatch.setenv("AGMEN_ALLOWED_ORIGINS", "https://agmen.example")
        response = connection.post("/api/missions", json=body, headers=headers)
        assert response.status_code == 202, response.text
        completed(connection, response.json()["id"])
        assert connection.post("/api/missions", json=body, headers={"origin": "https://other.example"}).status_code == 403
    app.state.executor.shutdown(wait=True)
