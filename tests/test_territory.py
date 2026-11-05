"""Image territory storage, confirmed geometry, and actual motion boundaries."""
import base64
import io
import time

import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from agmen_api import MissionConfig, create_app
from agmen_simulation import simulate
from edge import EdgeRuntime, build_package, load_package
from fleet_core import _segment_intersects_rect
from territory import (MapStore, MapUpload, MapAnalysis, PolygonZone,
                       restricted_mask, mask_rectangles)


def image_url(image=None):
    if image is None:
        image = Image.new("RGB", (320, 240), "#eeeecc")
        ImageDraw.Draw(image).rectangle((120, 40, 180, 170), fill="#202010")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGMEN_PROFILE", "local")
    monkeypatch.setenv("AGMEN_DATA_DIR", str(tmp_path))
    app = create_app(tmp_path / "missions")
    with TestClient(app) as connection:
        yield connection
    app.state.executor.shutdown(wait=True)


def completed(client, identifier):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = client.get(f"/api/missions/{identifier}").json()
        if result["status"] in {"completed", "failed"}:
            assert result["status"] == "completed", result
            return result
        time.sleep(.02)
    pytest.fail("Territory mission did not complete")


def test_analysis_is_advisory_preserves_image_orientation_and_survives_restart(tmp_path):
    store = MapStore(tmp_path)
    record = store.create(MapUpload(name="Test site", image_data_url=image_url(), width_m=120, height_m=90))
    assert record["analysis"]["requires_confirmation"] is True
    assert record["analysis"]["recognized_objects"] is False
    assert len(record["suggestions"]) == 1
    suggestion = record["suggestions"][0]
    assert abs(suggestion["x"] - 120/320) < .02
    assert abs(suggestion["y"] - 40/240) < .02
    assert suggestion["source"] == "automatic" and suggestion["confidence"] is None
    restored = MapStore(tmp_path).get(record["id"])
    assert restored == record
    assert MapStore(tmp_path).list() == [record]
    assert store.analyze(record["id"], MapAnalysis(method="water"))["analysis"]["method"] == "water"


def test_map_api_upload_body_larger_than_old_global_limit_and_calibration(client):
    noisy = Image.fromarray(np.random.default_rng(2).integers(0, 255, (360, 360, 3), dtype=np.uint8))
    uploaded = client.post("/api/maps", json={"name": "Large PNG", "image_data_url": image_url(noisy)})
    assert uploaded.status_code == 201, uploaded.text
    record = uploaded.json()
    identifier = record["id"]
    assert client.get(f"/api/maps/{identifier}/image").headers["content-type"] == "image/png"
    assert client.get("/api/maps").json()[0]["id"] == identifier
    calibrated = client.post(f"/api/maps/{identifier}/calibrate", json={"width_m": 120, "height_m": 90})
    assert calibrated.status_code == 200
    assert calibrated.json()["width_m"] == 120
    assert client.post(f"/api/maps/{identifier}/analyze", json={"method": "contrast", "sensitivity": .7}).status_code == 200
    assert client.get("/api/maps/not-a-map/image").status_code == 404


@pytest.mark.parametrize("value", [
    "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode(),
    image_url().replace("image/png", "image/jpeg"),
    "data:image/png;base64," + "A" * (4 * 1024 * 1024 * 4 // 3 + 8),
], ids=["corrupt", "svg", "mime-mismatch", "oversized"])
def test_invalid_corrupt_mismatched_and_oversized_images_rejected(client, value):
    response = client.post("/api/maps", json={"name": "Bad upload", "image_data_url": value})
    assert response.status_code == 422, response.text
    assert client.get("/api/maps").json() == []


def test_shape_rasterization_covers_geometry_and_physical_circle():
    circle = {"kind": "circle", "x": .5, "y": .5, "radius": .2}
    mask = restricted_mask([circle], n=24, width=120, height=60)
    # Radius is 12 m: normalized x radius .1, normalized y radius .2.
    assert mask[12, 12]
    assert mask[8, 12]
    assert not mask[12, 7]
    polygon = {"kind": "polygon", "points": [{"x": .2, "y": .2}, {"x": .8, "y": .2}, {"x": .5, "y": .8}]}
    shape_mask = restricted_mask([polygon])
    assert shape_mask[10, 12]
    assert not shape_mask[20, 2]
    rectangles = mask_rectangles(shape_mask, 80, 60)
    for row in range(24):
        for col in range(24):
            x, y = (col+.5)*80/24, (row+.5)*60/24
            assert bool(shape_mask[row, col]) == any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in rectangles)
    with pytest.raises(ValueError):
        PolygonZone(kind="polygon", points=[{"x": .1, "y": .1}, {"x": .8, "y": .8}, {"x": .1, "y": .8}, {"x": .8, "y": .1}])


def test_map_candidates_not_implicitly_applied_and_scale_snapshot_is_immutable(client):
    uploaded = client.post("/api/maps", json={"name": "Local area", "image_data_url": image_url(), "width_m": 120, "height_m": 90}).json()
    response = client.post("/api/missions", json={"drone_count": 5, "steps": 20, "obstacle": "none", "map_id": uploaded["id"]})
    assert response.status_code == 202, response.text
    record = completed(client, response.json()["id"])
    assert record["workspace"]["width_m"] == 120 and record["workspace"]["height_m"] == 90
    assert record["territory"]["id"] == uploaded["id"]
    assert not np.asarray(record["obstacle"]).any()
    assert record["restricted_zones"] == []
    client.post(f"/api/maps/{uploaded['id']}/calibrate", json={"width_m": 200, "height_m": 150})
    assert client.get(f"/api/missions/{record['id']}").json()["config"]["map_snapshot"]["width_m"] == 120
    assert client.post("/api/missions", json={"map_id": "a"*12}).status_code == 422


def test_confirmed_shapes_reject_swept_crossings_in_simulation_and_offline_edge(tmp_path):
    zones = [{"kind": "circle", "x": .45, "y": .5, "radius": .16},
             {"kind": "polygon", "points": [{"x": .55, "y": .15}, {"x": .7, "y": .18}, {"x": .6, "y": .38}]}]
    config = MissionConfig(drone_count=5, steps=100, max_speed=6, obstacle="none", width_m=120, height_m=90,
                           restricted_zones=zones).model_dump()
    record = {"id": "abcd1234abcd", "name": "Restricted site", "status": "completed", "config": config, **simulate(config)}
    width, height = 120, 90
    rectangles = [(r["x"]*width, r["y"]*height, (r["x"]+r["width"])*width, (r["y"]+r["height"])*height) for r in record["obstacle_rects"]]
    frames = record["frames"]
    for previous, current in zip(frames, frames[1:]):
        for before, after in zip(previous["agents"], current["agents"]):
            p = np.asarray([before["x"]*width, before["y"]*height])
            q = np.asarray([after["x"]*width, after["y"]*height])
            assert not any(_segment_intersects_rect(p, q, rectangle) for rectangle in rectangles)
    archive = tmp_path / "territory-mission.zip"
    build_package(record, 1, output_path=archive)
    package = load_package(archive)
    assert package["map"]["workspace"]["width_m"] == width
    assert package["map"]["restricted_zones"] == config["restricted_zones"]
    runtime = EdgeRuntime(package, tmp_path / "runtime")
    runtime.set_connectivity(internet=False)
    previous = np.asarray(runtime.snapshot()["position_m"])
    for _ in range(80):
        snapshot = runtime.run_steps(1)
        current = np.asarray(snapshot["position_m"])
        assert not any(_segment_intersects_rect(previous, current, rectangle) for rectangle in rectangles)
        previous = current
    assert snapshot["mode"] == "OFFLINE"
    assert snapshot["metrics"]["constraint_violations"] == 0


def test_invalid_restrictions_and_target_in_confirmed_zone_fail_before_queueing(client):
    rectangle = {"kind": "rectangle", "x": 0, "y": 0, "width": 1, "height": 1}
    assert client.post("/api/missions", json={"restricted_zones": [rectangle]}).status_code == 422
    rectangle.update(x=.4, y=.4, width=.2, height=.2)
    response = client.post("/api/missions", json={"restricted_zones": [rectangle], "targets": [{"x": .5, "y": .5}]})
    assert response.status_code == 422
    assert "target" in response.json()["detail"].lower()
    assert client.post("/api/missions", json={"restricted_zones": [{"kind": "rectangle", "x": .9, "y": .9, "width": .2, "height": .2}]}).status_code == 422
    assert client.get("/api/missions").json() == []


def test_confirmed_territory_api_mission_can_be_packaged_for_edge(client):
    territory = client.post("/api/maps", json={"name": "Confirmed area", "image_data_url": image_url(),
                                               "width_m": 125, "height_m": 85}).json()
    zone = {"id": "manual-zone", "kind": "rectangle", "x": .4, "y": .2, "width": .1, "height": .4}
    response = client.post("/api/missions", json={"map_id": territory["id"], "restricted_zones": [zone],
                                                  "drone_count": 5, "steps": 20, "obstacle": "none"})
    assert response.status_code == 202, response.text
    record = completed(client, response.json()["id"])
    assert np.asarray(record["obstacle"]).any()
    assert record["restriction_policy"]["swept_segment_checks"] is True
    response = client.post("/api/edge/packages", json={"mission_id": record["id"], "drone_id": 1})
    assert response.status_code == 201, response.text
    response = client.post("/api/edge/devices", json={"package_id": response.json()["id"]})
    assert response.status_code == 201, response.text
    device = response.json()
    assert device["workspace"]["width_m"] == 125
    assert device["workspace"]["height_m"] == 85
