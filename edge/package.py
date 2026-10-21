"""Bounded JSON-only mission archives; integrity checked before content loading."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import uuid
import zipfile

import numpy as np

SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
REQUIRED_MEMBERS = {"mission.json", "map.json", "fields.json"}
ALLOWED_MEMBERS = REQUIRED_MEMBERS | {"model.json", "manifest.json"}
POLICIES = {"continue", "hold", "return"}
COORDINATION = {"independent", "peer"}


class PackageError(ValueError):
    pass


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_read(data):
    def reject_constant(value):
        raise PackageError("Non-finite numeric value in mission package")
    try:
        return json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise PackageError("Invalid JSON in mission package") from error


def _vector(value, length, label):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise PackageError(f"Invalid {label}") from None
    if array.shape != (length,) or not np.isfinite(array).all():
        raise PackageError(f"Invalid {label}")
    return array


def _validate_content(package):
    mission, local_map, fields = package["mission"], package["map"], package["fields"]
    if not all(isinstance(value, dict) for value in (mission, local_map, fields)):
        raise PackageError("Mission package sections must be objects")
    if mission.get("policy") not in POLICIES or mission.get("coordination") not in COORDINATION:
        raise PackageError("Unsupported autonomous policy or coordination mode")
    if not isinstance(mission.get("drone_id"), int) or not 1 <= mission["drone_id"] <= 500:
        raise PackageError("Invalid device identity")
    if mission.get("controller") not in {"baseline", "neural", "hybrid"}:
        raise PackageError("Unsupported controller")
    if mission.get("simulation_only") is not True or local_map.get("geographic_origin") is not None:
        raise PackageError("Only local planar simulation packages are supported")
    workspace = local_map.get("workspace", {})
    n = workspace.get("grid_size")
    if not isinstance(n, int) or n not in {24, 32, 48, 64, 96}:
        raise PackageError("Invalid bounded map grid")
    for key, low, high in (("width_m", 1, 10000), ("height_m", 1, 10000), ("dt", .01, 1)):
        value = workspace.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise PackageError("Invalid local workspace")
    obstacle = np.asarray(local_map.get("obstacle"))
    if obstacle.shape != (n, n) or not np.isin(obstacle, [False, True, 0, 1]).all():
        raise PackageError("Missing or invalid local obstacle map")
    if obstacle.all():
        raise PackageError("Mission map contains no free cells")
    for name in ("initial_density", "target_density", "flux_x", "flux_y"):
        array = np.asarray(fields.get(name), dtype=float)
        if array.shape != (n, n) or not np.isfinite(array).all():
            raise PackageError(f"Invalid {name} field")
        if name.endswith("density") and (np.any(array < 0) or abs(float(array.sum()) - 1) > 1e-4):
            raise PackageError(f"Invalid normalized {name}")
    width, height = workspace["width_m"], workspace["height_m"]
    rectangles = local_map.get("obstacle_rectangles_m")
    if not isinstance(rectangles, list) or len(rectangles) > 64:
        raise PackageError("Invalid obstacle boundaries")
    for rectangle in rectangles:
        x0, y0, x1, y1 = _vector(rectangle, 4, "obstacle rectangle")
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise PackageError("Obstacle rectangle lies outside the local map")
    for name in ("initial_position_m", "home_position_m", "assigned_slot_m"):
        point = _vector(mission.get(name), 2, name)
        if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
            raise PackageError("Mission coordinate lies outside its geofence")
        if any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1 for x0, y0, x1, y1 in rectangles):
            raise PackageError("Mission coordinate lies inside an obstacle")
    limits = mission.get("limits", {})
    for name, low, high in (("max_speed_mps", .1, 20), ("max_accel_mps2", .1, 20), ("altitude_m", 0, 120), ("max_steps", 1, 10000), ("low_battery_percent", 1, 50)):
        value = limits.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise PackageError(f"Invalid {name} limit")
    battery = mission.get("initial_battery_percent")
    if not isinstance(battery, (int, float)) or not math.isfinite(battery) or not 0 <= battery <= 100:
        raise PackageError("Invalid initial battery state")
    tolerance = mission.get("completion", {}).get("arrival_tolerance_m")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not .1 <= tolerance <= 10:
        raise PackageError("Invalid mission completion tolerance")
    parameters = mission.get("controller_parameters", {})
    for name in ("slot_gain", "field_weight", "slot_weight", "obstacle_margin_m", "min_separation_m", "safety_gain"):
        value = parameters.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise PackageError("Invalid controller parameters")
    model = package.get("model")
    if mission["controller"] != "baseline" and model is None:
        raise PackageError("Neural controller requires included model weights")
    if model is not None:
        if not isinstance(model, dict) or model.get("architecture") != "SwarmUNet" or model.get("encoding") != "float32-json-v1":
            raise PackageError("Unsupported model encoding")
        tensors = model.get("state_dict", {})
        if not isinstance(tensors, dict) or not 1 <= len(tensors) <= 128:
            raise PackageError("Invalid neural tensors")
        count = 0
        for name, tensor in tensors.items():
            if not isinstance(name, str) or len(name) > 120 or not isinstance(tensor, dict):
                raise PackageError("Invalid neural tensor name")
            shape = tensor.get("shape")
            if not isinstance(shape, list) or not 1 <= len(shape) <= 4 or any(not isinstance(size, int) or not 1 <= size <= 128 for size in shape):
                raise PackageError("Invalid neural tensor shape")
            expected = math.prod(shape)
            count += expected
            if count > 200000 or not isinstance(tensor.get("values"), list) or len(tensor["values"]) != expected:
                raise PackageError("Neural tensor exceeds the model budget")
            values = np.asarray(tensor["values"], dtype=float)
            if not np.isfinite(values).all():
                raise PackageError("Non-finite neural tensor")
    return package


def build_package(mission: dict, drone_id: int, policy="continue", coordination="independent", output_path=None):
    """Compile the initial recorded assignment, never replay future fleet state."""
    if mission.get("status") != "completed" or not mission.get("frames"):
        raise PackageError("A completed recorded mission is required")
    if policy not in POLICIES or coordination not in COORDINATION:
        raise PackageError("Unsupported policy or coordination mode")
    if isinstance(drone_id, bool) or not isinstance(drone_id, int):
        raise PackageError("drone_id must be an integer")
    from agmen_simulation import _targets, _zones
    from fleet_core import FleetSimConfig, build_target_slots, assign_drones_to_slots
    from swarm_core import expert_flux
    config = dict(mission["config"])
    frame = mission["frames"][0]
    agents = frame["agents"]
    if not 1 <= drone_id <= len(agents):
        raise PackageError("The selected drone is absent from the mission")
    workspace = dict(mission.get("workspace", {"width_m": 80., "height_m": 60., "dt": .2, "grid_size": 24}))
    width, height = workspace["width_m"], workspace["height_m"]
    if (width, height) != (80., 60.):
        raise PackageError("Recorded mission must use the original 80 x 60 m plane")
    rectangles = [[rect["x"] * width, rect["y"] * height, (rect["x"] + rect["width"]) * width, (rect["y"] + rect["height"]) * height] for rect in mission.get("obstacle_rects", [])]
    positions = np.asarray([[agent["x"] * width, agent["y"] * height] for agent in agents])
    zones = _zones(config, 0)
    for event in mission.get("events", []):
        if event.get("step") == 0 and event.get("kind") == "target_change":
            zones = event.get("targets") or [{"x": .22, "y": .77, "radius": .16, "weight": 1.0}]
    slots, labels = build_target_slots(_targets(zones, len(agents)), width, height, rectangles)
    assignment = assign_drones_to_slots(positions, slots)
    selected_slot = slots[assignment[drone_id - 1]]
    obstacle = np.asarray(mission["obstacle"], dtype=bool)
    initial = np.asarray(frame["density"], dtype=float)
    target = np.asarray(frame["target"], dtype=float)
    # Recorded field values are rounded for web transfer. Normalize exactly for
    # the original expert solver and offline validation.
    initial = initial / initial.sum()
    target = target / target.sum()
    jx, jy = expert_flux(initial, target, obstacle)
    controller = config.get("controller", "baseline")
    model_data = None
    if controller != "baseline":
        from ml_lab import get_inference_model
        from model import predict_flux
        model_id = config.get("model_id", "bundled")
        selected = get_inference_model(model_id)
        if selected is None:
            raise PackageError("Selected neural weights are unavailable")
        model, device = selected
        nx, ny = predict_flux(model, initial, target, obstacle, device)
        jx, jy = (nx, ny) if controller == "neural" else (.7 * nx + .3 * jx, .7 * ny + .3 * jy)
        model_data = {"architecture": "SwarmUNet", "encoding": "float32-json-v1", "id": model_id, "state_dict": {name: {"shape": list(value.shape), "values": value.detach().cpu().numpy().ravel().tolist()} for name, value in model.state_dict().items()}}
    cfg = FleetSimConfig()
    own = agents[drone_id - 1]
    package = {
        "mission": {
            "mission_id": str(mission["id"]), "drone_id": drone_id,
            "name": mission.get("name", "AGMEN mission"), "source_config": config,
            "policy": policy, "coordination": coordination, "controller": controller,
            "controller_version": "density-flux-1", "model_id": config.get("model_id", "bundled") if model_data else None,
            "initial_position_m": positions[drone_id - 1].tolist(), "home_position_m": positions[drone_id - 1].tolist(),
            "initial_battery_percent": float(own.get("battery", 100)),
            "assigned_slot_m": selected_slot.tolist(), "assigned_zone": labels[assignment[drone_id - 1]],
            "route_m": [positions[drone_id - 1].tolist(), selected_slot.tolist()],
            "limits": {"max_speed_mps": config["max_speed"], "max_accel_mps2": cfg.max_accel_mps2, "altitude_m": 20., "max_steps": config["steps"], "low_battery_percent": 20.},
            "controller_parameters": {name: getattr(cfg, name) for name in ("slot_gain", "field_weight", "slot_weight", "obstacle_margin_m", "min_separation_m", "safety_gain")},
            "completion": {"arrival_tolerance_m": 1.0, "on_time_limit": "return"},
            "logging": {"max_entries": 2048, "max_history": 1024}, "simulation_only": True,
        },
        "map": {"coordinate_system": "local_planar_metres", "geographic_origin": None, "workspace": workspace,
                "obstacle": obstacle.tolist(), "obstacle_rectangles_m": rectangles},
        "fields": {"initial_density": initial.tolist(), "target_density": target.tolist(), "flux_x": jx.tolist(), "flux_y": jy.tolist()},
    }
    package["mission"]["controller_parameters"]["min_separation_m"] = config["safe_distance"]
    if model_data:
        package["model"] = model_data
    _validate_content(package)
    members = {f"{name}.json": _json_bytes(value) for name, value in package.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    identity = uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(_json_bytes(hashes)).hexdigest()).hex
    manifest = {"schema_version": SCHEMA_VERSION, "id": identity, "mission_id": str(mission["id"]), "drone_id": drone_id,
                "controller_version": "density-flux-1", "files": hashes, "simulation_only": True}
    package["manifest"] = manifest
    if output_path is None:
        return package
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members["manifest.json"] = _json_bytes(manifest)
    if any(len(data) > MAX_MEMBER_BYTES for data in members.values()) or sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise PackageError("Package exceeds its storage budget")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, data)
    return {"id": identity, "mission_id": str(mission["id"]), "drone_id": drone_id, "policy": policy,
            "coordination": coordination, "path": str(path.resolve()), "manifest": manifest,
            "size_bytes": path.stat().st_size, "simulation_only": True}


def load_package(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise PackageError("Mission archive is missing or exceeds its size budget")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or not REQUIRED_MEMBERS.union({"manifest.json"}).issubset(names) or set(names) - ALLOWED_MEMBERS:
                raise PackageError("Missing, duplicate, or unexpected archive member")
            if any(entry.file_size > MAX_MEMBER_BYTES or entry.flag_bits & 1 or entry.is_dir() for entry in entries) or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
                raise PackageError("Archive exceeds the allowed extraction budget")
            contents = {entry.filename: archive.read(entry) for entry in entries}
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise PackageError("Cannot read mission archive") from error
    manifest = _json_read(contents.pop("manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("simulation_only") is not True:
        raise PackageError("Unsupported mission package manifest")
    hashes = manifest.get("files", {})
    if not isinstance(hashes, dict) or set(hashes) != set(contents):
        raise PackageError("Manifest does not describe all package members")
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise PackageError(f"Package integrity check failed for {name}")
    expected_id = uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(_json_bytes(hashes)).hexdigest()).hex
    if manifest.get("id") != expected_id:
        raise PackageError("Mission package identifier does not match its content")
    package = {name.removesuffix(".json"): _json_read(data) for name, data in contents.items()}
    package["manifest"] = manifest
    if manifest.get("mission_id") != package.get("mission", {}).get("mission_id") or manifest.get("drone_id") != package.get("mission", {}).get("drone_id"):
        raise PackageError("Manifest mission identity does not match its content")
    try:
        _validate_content(package)
    except (TypeError, KeyError, ValueError, OverflowError) as error:
        if isinstance(error, PackageError):
            raise
        raise PackageError("Invalid mission package schema") from error
    return package
