"""Persisted per-device planar controller; imports no network transports.

An independent device samples a precomputed density flux and follows its own
assigned slot. Peer mode only recalculates from explicitly supplied observations;
it never reads other runtimes or the ground station's current fleet state.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from fleet_core import (
    FleetSimConfig, bilinear_sample, slot_guidance_velocity,
    obstacle_boundary_velocity, positions_to_density, separation_velocity,
    _clip_vectors, _prevent_rect_crossing, _segment_intersects_rect,
)
from swarm_core import expert_flux
from .package import load_package, _validate_content, _json_bytes, PackageError


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path, text):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.005 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class EdgeRuntime:
    """A virtual onboard computer. state_path is its persistent local directory."""
    MAX_JOURNAL = 2048
    MAX_HISTORY = 1024

    def __init__(self, package, state_path):
        self.lock = threading.RLock()
        self.package = load_package(package) if isinstance(package, (str, Path)) else copy.deepcopy(package)
        _validate_content(self.package)
        manifest = self.package.get("manifest", {})
        files = manifest.get("files", {})
        for section in ("mission", "map", "fields", "model"):
            if section in self.package and hashlib.sha256(_json_bytes(self.package[section])).hexdigest() != files.get(f"{section}.json"):
                raise PackageError("In-memory mission package failed its integrity check")
        self.mission = self.package["mission"]
        self.map = self.package["map"]
        self.fields = self.package["fields"]
        self.package_id = manifest["id"]
        self.directory = Path(state_path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.obstacle = np.asarray(self.map["obstacle"], dtype=bool)
        self.target = np.asarray(self.fields["target_density"], dtype=float)
        self.rectangles = self.map["obstacle_rectangles_m"]
        workspace, limits = self.map["workspace"], self.mission["limits"]
        self.width, self.height, self.dt = workspace["width_m"], workspace["height_m"], workspace["dt"]
        self.cfg = FleetSimConfig(
            drone_count=1, workspace_width_m=self.width, workspace_height_m=self.height,
            grid_size=workspace["grid_size"], dt=self.dt,
            max_speed_mps=limits["max_speed_mps"], max_accel_mps2=limits["max_accel_mps2"],
            **self.mission["controller_parameters"],
        )
        self.model = self._load_local_model()
        # Compute locally once even for independent operation. The mathematical
        # solver and optional inference are the original repository functions.
        self.initial_flux = self._flux(np.asarray(self.fields["initial_density"], dtype=float))
        if self.state_path.is_file():
            if self.state_path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Persistent EDGE state exceeds its storage budget")
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError("Persistent EDGE state is invalid") from error
            self._validate_state()
            self._event("runtime_restart", "Local mission and journal restored from persistent storage")
        else:
            position = list(self.mission["initial_position_m"])
            self.state = {
                "schema_version": 1, "package_id": self.package_id,
                "mission_id": self.mission["mission_id"], "drone_id": self.mission["drone_id"],
                "position_m": position, "velocity_mps": list(self.mission.get("initial_velocity_mps", [0., 0.])), "step": 0,
                "battery": self.mission["initial_battery_percent"],
                "connectivity": {"internet": True, "ground_radio": True, "peer_radio": True, "gnss": True},
                "navigation_valid": True, "mode": "ONLINE", "status": "running",
                "last_confirmed_telemetry": None, "last_sync": None,
                "ground_link_lost_at_step": None, "ground_link_lost_at": None,
                "recovery_steps": None, "journal": [], "journal_seq": 0, "history": [],
                "recovery_time_s": None, "task_completed": False,
                "constraint_violations": 0, "emergency_stops": 0,
                "peer_observations_used": 0, "created_at": _now(), "updated_at": _now(),
                "simulation_only": True,
            }
            self._event("package_loaded", "Verified local map, assignment and controller loaded")
            self._confirm_telemetry()
        self._persist()

    def _load_local_model(self):
        weights = self.package.get("model")
        if weights is None:
            return None
        try:
            import torch
            from model import SwarmUNet
        except ImportError as error:
            raise ValueError("This package requires the local PyTorch inference runtime") from error
        torch.set_num_threads(2)
        model = SwarmUNet()
        expected = model.state_dict()
        encoded = weights["state_dict"]
        if set(encoded) != set(expected):
            raise PackageError("Neural checkpoint architecture does not match SwarmUNet")
        state = {}
        for name, original in expected.items():
            record = encoded[name]
            if tuple(record["shape"]) != tuple(original.shape):
                raise PackageError("Neural checkpoint tensor shape mismatch")
            array = np.asarray(record["values"], dtype=np.float32).reshape(record["shape"])
            state[name] = torch.from_numpy(array)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _flux(self, density):
        jx, jy = expert_flux(density, self.target, self.obstacle)
        if self.model is not None:
            from model import predict_flux
            nx, ny = predict_flux(self.model, density, self.target, self.obstacle)
            return (nx, ny) if self.mission["controller"] == "neural" else (.7 * nx + .3 * jx, .7 * ny + .3 * jy)
        return jx, jy

    def _validate_state(self):
        state = self.state
        if state.get("schema_version") != 1 or state.get("package_id") != self.package_id:
            raise ValueError("Persistent EDGE state belongs to a different mission package")
        if state.get("mission_id") != self.mission["mission_id"] or state.get("drone_id") != self.mission["drone_id"]:
            raise ValueError("Persistent EDGE device identity does not match its package")
        if not isinstance(state.get("step"), int) or not 0 <= state["step"] <= 1000000:
            raise ValueError("Invalid persistent mission step")
        position, velocity = np.asarray(state.get("position_m"), float), np.asarray(state.get("velocity_mps"), float)
        if position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("Invalid persistent local navigation state")
        if not 0 <= position[0] <= self.width or not 0 <= position[1] <= self.height:
            raise ValueError("Persistent position is outside its geofence")
        if not isinstance(state.get("battery"), (int, float)) or not 0 <= state["battery"] <= 100:
            raise ValueError("Invalid persistent battery state")
        if not isinstance(state.get("journal"), list) or len(state["journal"]) > self.MAX_JOURNAL or not isinstance(state.get("history"), list) or len(state["history"]) > self.MAX_HISTORY:
            raise ValueError("Persistent journal exceeds its bounds")
        if not isinstance(state.get("journal_seq"), int) or state["journal_seq"] < 0 or not isinstance(state.get("navigation_valid"), bool):
            raise ValueError("Invalid persistent journal or navigation state")
        if not isinstance(state.get("connectivity"), dict) or set(state["connectivity"]) != {"internet", "ground_radio", "peer_radio", "gnss"} or not all(isinstance(value, bool) for value in state["connectivity"].values()):
            raise ValueError("Invalid persistent connectivity state")

    def _event(self, kind, message, **details):
        self.state["journal_seq"] += 1
        self.state["journal"].append({"seq": self.state["journal_seq"], "step": self.state["step"], "time": _now(), "kind": kind, "message": message, **details})
        self.state["journal"] = self.state["journal"][-self.MAX_JOURNAL:]

    def _persist(self):
        self.state["updated_at"] = _now()
        _atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        _atomic_write(self.directory / "journal.jsonl", "".join(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n" for entry in self.state["journal"]))

    def _mode(self):
        links = self.state["connectivity"]
        if not links["ground_radio"] and not links["peer_radio"]:
            return "LINK_LOST"
        return "ONLINE" if links["internet"] and links["ground_radio"] else "OFFLINE"

    def _confirm_telemetry(self):
        if self.state["connectivity"]["ground_radio"]:
            position = self.state["position_m"]
            self.state["last_confirmed_telemetry"] = {"step": self.state["step"], "x": position[0] / self.width, "y": position[1] / self.height,
                "position_m": list(position), "velocity_mps": list(self.state["velocity_mps"]), "battery": self.state["battery"], "status": self.state["status"], "received_at": _now(), "source": "confirmed_simulated_ground_radio"}

    def set_connectivity(self, **values):
        if set(values) - {"internet", "ground_radio", "peer_radio", "gnss"} or not all(isinstance(value, bool) for value in values.values()):
            raise ValueError("Connectivity values must be internet, ground_radio, peer_radio or gnss booleans")
        with self.lock:
            recovery_started = time.perf_counter()
            previous_ground = self.state["connectivity"]["ground_radio"]
            self.state["connectivity"].update(values)
            current_ground = self.state["connectivity"]["ground_radio"]
            if previous_ground and not current_ground:
                self.state["ground_link_lost_at_step"] = self.state["step"]
                self.state["ground_link_lost_at"] = _now()
            elif not previous_ground and current_ground:
                self.state["recovery_steps"] = self.state["step"] - (self.state["ground_link_lost_at_step"] or 0)
            self.state["mode"] = self._mode()
            self._event("connectivity", "Local connection state changed", connectivity=dict(self.state["connectivity"]))
            self._confirm_telemetry()
            self._persist()
            if not previous_ground and current_ground:
                self.sync()
                self.state["recovery_time_s"] = time.perf_counter() - recovery_started
                self._persist()
                return self.snapshot()
            return self.snapshot()

    def set_battery(self, percent):
        if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("Battery must be a finite percentage from 0 to 100")
        with self.lock:
            self.state["battery"] = float(percent)
            self._event("battery", "Simulated local battery state changed", percent=percent)
            self._persist()
            return self.snapshot()

    def set_navigation(self, valid):
        if not isinstance(valid, bool):
            raise ValueError("Navigation validity must be boolean")
        with self.lock:
            self.state["navigation_valid"] = valid
            self._event("navigation", "Local navigation validity changed", valid=valid)
            self._persist()
            return self.snapshot()

    def _observations(self, supplied):
        if self.mission["coordination"] != "peer" or not self.state["connectivity"]["peer_radio"]:
            return []
        if supplied is None:
            return []
        if not isinstance(supplied, list) or len(supplied) > 100:
            raise ValueError("At most 100 explicit peer observations are permitted")
        points, used = [], set()
        for observation in supplied:
            if not isinstance(observation, dict):
                raise ValueError("Invalid peer observation")
            identity, step = observation.get("drone_id"), observation.get("step")
            point = np.asarray(observation.get("position_m"), dtype=float)
            if not isinstance(identity, int) or not isinstance(step, int) or point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Invalid peer navigation observation")
            # Old or future packets are not current measurements. Own state is
            # always obtained locally, never accepted from a peer packet.
            if identity == self.mission["drone_id"] or identity in used or step > self.state["step"] or self.state["step"] - step > 10:
                continue
            if not 0 <= point[0] <= self.width or not 0 <= point[1] <= self.height:
                continue
            used.add(identity)
            points.append(point)
        return points

    def _desired_status(self):
        links, limits = self.state["connectivity"], self.mission["limits"]
        if not self.state["navigation_valid"] or not links["gnss"]:
            return "holding", "Trusted local navigation is unavailable"
        if self.state["battery"] <= 0:
            return "battery_depleted", "Simulated battery depleted; stop locally"
        if self.state["battery"] <= limits["low_battery_percent"]:
            return "returning", "Low battery: follow preloaded return route"
        if self.state["status"] in {"completed", "returned"}:
            return self.state["status"], "Mission already finished"
        if self.mission.get("completion", {}).get("wait_for_arrival") and not links["ground_radio"]:
            if self.mission["policy"] == "hold":
                return "holding", "Ground radio lost: preloaded hold policy"
            if self.mission["policy"] == "return":
                return "returning", "Ground radio lost: preloaded return policy"
        if self.state["step"] >= limits["max_steps"]:
            if self.mission.get("completion", {}).get("wait_for_arrival"):
                return "awaiting_extension", "Computation budget reached before arrival; mission remains incomplete"
            return "returning", "Preloaded mission time limit reached"
        if not links["ground_radio"]:
            if self.mission["policy"] == "hold":
                return "holding", "Ground radio lost: preloaded hold policy"
            if self.mission["policy"] == "return":
                return "returning", "Ground radio lost: preloaded return policy"
        return "running", "Execute the assigned local mission"

    def run_steps(self, steps, peer_observations=None):
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 500:
            raise ValueError("A local execution segment must contain 1 to 500 steps")
        with self.lock:
            for _ in range(steps):
                before = np.asarray(self.state["position_m"], dtype=float)
                velocity = np.asarray(self.state["velocity_mps"], dtype=float)
                status, reason = self._desired_status()
                if status != self.state["status"]:
                    self._event("policy", reason, status=status)
                self.state["status"] = status
                peers = self._observations(peer_observations)
                self.state["peer_observations_used"] = len(peers)
                desired = np.zeros((1, 2))
                if status in {"running", "returning"}:
                    destination = np.asarray(self.mission["home_position_m"] if status == "returning" else self.mission["assigned_slot_m"])
                    completion = self.mission["completion"]
                    zone = completion.get("assigned_zone_m")
                    inside_zone = not completion.get("wait_for_arrival") or np.linalg.norm(before - [zone["x"], zone["y"]]) <= zone["radius"] + 1e-8
                    if np.linalg.norm(destination - before) <= completion["arrival_tolerance_m"] and (status == "returning" or inside_zone):
                        minimum = self.mission.get("completion", {}).get("minimum_steps", 0)
                        if status == "returning" or not self.mission.get("completion", {}).get("wait_for_arrival") or self.state["step"] >= minimum:
                            self.state["status"] = "returned" if status == "returning" else "completed"
                            if status == "running":
                                self.state["task_completed"] = True
                            self._event("mission", "Arrived at the preloaded destination", status=self.state["status"])
                    else:
                        jx, jy = self.initial_flux
                        if peers and status == "running":
                            observed = np.asarray([before, *peers])
                            density = positions_to_density(observed, self.cfg.grid_size, self.width, self.height, self.obstacle)
                            jx, jy = self._flux(density)
                        sample = np.asarray([[bilinear_sample(jx, before[None], self.width, self.height)[0], bilinear_sample(jy, before[None], self.width, self.height)[0]]])
                        norm = float(np.linalg.norm(sample))
                        macro = _clip_vectors(sample / (norm + 1e-9) * (.55 * self.cfg.max_speed_mps), .75 * self.cfg.max_speed_mps)
                        slot = slot_guidance_velocity(before[None], destination[None], self.rectangles, self.cfg)
                        avoidance = obstacle_boundary_velocity(before[None], self.rectangles, self.width, self.height, self.cfg.obstacle_margin_m, .85 * self.cfg.max_speed_mps)
                        # Return routing uses the original slot obstacle planner;
                        # a target-density flux would point away from home.
                        terminal_approach = (self.mission.get("completion", {}).get("wait_for_arrival")
                                             and self.state["step"] >= self.mission["completion"]["minimum_steps"])
                        desired = slot + avoidance if status == "returning" or terminal_approach else self.cfg.field_weight * macro + self.cfg.slot_weight * slot + avoidance
                        if terminal_approach and np.linalg.norm(desired) <= .05 and self.rectangles:
                            # The corner router can select its current expanded
                            # corner as a waypoint. Move outward through the same
                            # controller; never project or teleport the device.
                            candidates = []
                            margin = self.cfg.obstacle_margin_m
                            for x0, y0, x1, y1 in self.rectangles:
                                for x, y, sx, sy in ((x0-margin, y0-margin, -1, -1), (x0-margin, y1+margin, -1, 1),
                                                     (x1+margin, y0-margin, 1, -1), (x1+margin, y1+margin, 1, 1)):
                                    distance = np.linalg.norm(before - [x, y])
                                    if distance < .45:
                                        goal = np.clip(before + np.asarray([sx, sy]) * .7, [.5, .5], [self.width-.5, self.height-.5])
                                        if not any(_segment_intersects_rect(before, goal, rectangle, .15) for rectangle in self.rectangles):
                                            candidates.append((distance, goal))
                            if candidates:
                                goal = min(candidates, key=lambda candidate: candidate[0])[1]
                                desired = slot_guidance_velocity(before[None], goal[None], self.rectangles, self.cfg) + avoidance
                                self._event("navigation", "Applied an outward corner navigation goal", goal_m=goal.tolist())
                        if peers:
                            separation, _, _ = separation_velocity(np.asarray([before, *peers]), self.cfg.min_separation_m, self.cfg.safety_gain, .95 * self.cfg.max_speed_mps)
                            desired += separation[:1]
                        desired = _clip_vectors(desired, self.cfg.max_speed_mps)
                        desired = velocity[None] + _clip_vectors(desired - velocity[None], self.cfg.max_accel_mps2 * self.dt)
                        desired = _clip_vectors(desired, self.cfg.max_speed_mps)
                proposed = before[None] + desired * self.dt
                moved = _prevent_rect_crossing(before[None], proposed, self.rectangles, self.width, self.height)[0]
                # Confirmed territory restrictions cover whole swept segments,
                # including steps whose endpoints both lie outside an obstacle.
                if (self.map.get("restricted_zones") or self.mission.get("completion", {}).get("wait_for_arrival")) and any(
                        _segment_intersects_rect(before, moved, rectangle, .15) for rectangle in self.rectangles):
                    moved = before.copy()
                if not np.allclose(moved, proposed[0], atol=1e-10):
                    self.state["emergency_stops"] += 1
                actual_velocity = (moved - before) / self.dt
                speed = float(np.linalg.norm(actual_velocity))
                bad = speed > self.cfg.max_speed_mps + 1e-7 or not (0 <= moved[0] <= self.width and 0 <= moved[1] <= self.height)
                bad = bad or any(x0 <= moved[0] <= x1 and y0 <= moved[1] <= y1 for x0, y0, x1, y1 in self.rectangles)
                self.state["constraint_violations"] += int(bad)
                self.state["position_m"], self.state["velocity_mps"] = moved.tolist(), actual_velocity.tolist()
                self.state["battery"] = max(0., self.state["battery"] - self.dt * (.012 + .003 * speed ** 2))
                self.state["step"] += 1
                self.state["mode"] = self._mode()
                metrics = self._metrics(peers)
                self.state["history"].append({"step": self.state["step"], "time_s": self.state["step"] * self.dt, "speed_mps": speed, "battery": self.state["battery"], "mode": self.state["mode"], "status": self.state["status"], **metrics})
                self.state["history"] = self.state["history"][-self.MAX_HISTORY:]
                self._event("telemetry", "Executed one local simulation integration step", position_m=list(self.state["position_m"]), status=self.state["status"])
                self._confirm_telemetry()
                self._persist()
            return self.snapshot()

    run = run_steps

    def _metrics(self, peers=None):
        position = np.asarray(self.state["position_m"])
        distance = float(np.linalg.norm(position - np.asarray(self.mission["assigned_slot_m"])))
        initial_distance = float(np.linalg.norm(np.asarray(self.mission["initial_position_m"]) - np.asarray(self.mission["assigned_slot_m"])))
        density = positions_to_density(np.asarray([position, *(peers or [])]), self.cfg.grid_size, self.width, self.height, self.obstacle)
        return {"distance_to_target_m": distance, "task_fraction": float(np.clip(1 - distance / max(initial_distance, 1e-9), 0, 1)),
                "task_completion_fraction": float(self.state.get("task_completed", False)),
                "distribution_mae": float(np.mean(np.abs(density - self.target))), "distribution_scope": "own_and_observed_peers" if peers else "own_local_position",
                "constraint_violations": self.state["constraint_violations"], "emergency_stops": self.state["emergency_stops"], "recovery_steps": self.state["recovery_steps"],
                "radio_outage_steps": self.state["recovery_steps"], "recovery_time_s": self.state.get("recovery_time_s")}

    def sync(self):
        with self.lock:
            if not self.state["connectivity"]["ground_radio"]:
                raise ValueError("Ground radio is unavailable; the local journal remains unsynchronized")
            self._confirm_telemetry()
            self._event("sync", "Local journal acknowledged by the simulated ground station; no internet required")
            self.state["last_sync"] = {"step": self.state["step"], "journal_seq": self.state["journal_seq"], "entries": len(self.state["journal"]), "time": _now(), "transport": "simulated_ground_radio", "cloud_upload": False}
            self._persist()
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
            state["id"] = self.package_id
            state["workspace"] = copy.deepcopy(self.map["workspace"])
            state["x"], state["y"] = state["position_m"][0] / self.width, state["position_m"][1] / self.height
            state["mode"] = self._mode()
            state["metrics"] = self._metrics()
            if state["history"] and state["history"][-1]["step"] == state["step"]:
                for key in ("distribution_mae", "distribution_scope"):
                    state["metrics"][key] = state["history"][-1][key]
            state["journal_entries"] = len(state["journal"])
            confirmed = state["last_confirmed_telemetry"]
            state["telemetry_stale"] = not state["connectivity"]["ground_radio"] or confirmed is None or confirmed["step"] < state["step"]
            state["last_confirmed_ground_telemetry"] = confirmed
            state["simulated_position"] = {"x": state["x"], "y": state["y"], "source": "local_simulation", "ground_confirmed": not state["telemetry_stale"]}
            state["predicted_position"] = None
            if state["telemetry_stale"] and confirmed is not None:
                elapsed = (state["step"] - confirmed["step"]) * self.dt
                estimate = np.asarray(confirmed["position_m"]) + np.asarray(confirmed.get("velocity_mps", [0., 0.])) * elapsed
                estimate = np.clip(estimate, [0., 0.], [self.width, self.height])
                state["predicted_position"] = {
                    "x": float(estimate[0] / self.width), "y": float(estimate[1] / self.height),
                    "source": "dead_reckoning_from_last_confirmed", "age_s": elapsed,
                    "uncertainty_m": min(math.hypot(self.width, self.height), .5 + .5 * self.cfg.max_accel_mps2 * elapsed ** 2),
                    "uncertainty_kind": "kinematic_envelope_estimate", "obstacle_guarantee": False,
                }
            state["controller_version"] = self.mission["controller_version"]
            state["policy"], state["coordination"] = self.mission["policy"], self.mission["coordination"]
            state["sync_pending_entries"] = max(0, state["journal_seq"] - (state["last_sync"]["journal_seq"] if state["last_sync"] else 0))
            lost = state["ground_link_lost_at_step"]
            state["time_since_link_loss_s"] = (state["step"] - lost) * self.dt if lost is not None and not state["connectivity"]["ground_radio"] else 0.
            return state
