"""Deterministic planar simulation built from the repository's fleet controller.

No aircraft, GPS, radios, or batteries are connected. Coordinates are normalized
for rendering; integration and safety use metres in a calibrated local plane,
defaulting to 80 x 60 m. Image axes are x right, y down, without geographic origin.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from fleet_core import (
    FleetSimConfig, TargetPoint, assign_drones_to_slots, build_target_slots,
    density_control_commands, initial_positions, obstacle_mask, obstacle_rectangles,
    positions_to_density, target_density_from_points, _prevent_rect_crossing, _segment_intersects_rect,
)
from swarm_core import expert_flux
from territory import restricted_mask, mask_rectangles, free_slots, initial_free_positions

WIDTH, HEIGHT, DT, GRID = 80.0, 60.0, 0.2, 24
CONTROLLERS = {"baseline": "Математический эталон", "neural": "Нейросеть", "hybrid": "Гибрид 70/30"}
OBSTACLES = {"none": "Нет", "central": "Центральный блок", "corridor": "Коридор"}


def _zones(config: dict, step: int) -> list[dict]:
    if config.get("targets"):
        return config["targets"]
    scenario = config["scenario"]
    progress = step / max(config["steps"], 1)
    if scenario == "clusters":
        return [{"x": .77, "y": y, "radius": .09, "weight": 1.0} for y in (.24, .76)]
    if scenario == "patrol":
        angle = 2 * math.pi * progress
        return [{"x": .70 + .15 * math.cos(angle), "y": .50 + .27 * math.sin(angle), "radius": .12, "weight": 1.0}]
    if scenario == "survey":
        return [{"x": x, "y": y, "radius": .09, "weight": 1.0} for x in (.25, .73, .88) for y in (.23, .77)]
    if scenario == "scan":
        return [{"x": .30 + .55 * progress, "y": y, "radius": .075, "weight": 1.0} for y in (.17, .39, .61, .83)]
    if scenario == "return":
        return [{"x": .16, "y": .50, "radius": .22, "weight": 1.0}]
    return [{"x": .78, "y": .50, "radius": .18, "weight": 1.0}]


def _targets(zones: list[dict], count: int, width=WIDTH, height=HEIGHT) -> list[TargetPoint]:
    weights = np.asarray([z.get("weight", 1) for z in zones], float)
    raw = weights / weights.sum() * count
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[:count - int(counts.sum())]:
        counts[i] += 1
    return [TargetPoint(f"T{i+1}", z["x"] * width, z["y"] * height,
                        int(n), z.get("radius", .12) * min(width, height))
            for i, (z, n) in enumerate(zip(zones, counts)) if n > 0]


def _safety(positions: np.ndarray, distance: float) -> tuple[float, int]:
    if len(positions) < 2:
        return 0.0, 0
    d = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    d = d[np.triu_indices(len(positions), k=1)]
    return float(d.min()), int(np.count_nonzero(d < distance))


def _array(a: np.ndarray) -> list:
    return np.asarray(a).round(9).tolist()


def _geometry(config: dict):
    width, height = config.get("width_m", WIDTH), config.get("height_m", HEIGHT)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and 1 <= value <= 10000 for value in (width, height)):
        raise ValueError("Invalid bounded local workspace")
    rects = obstacle_rectangles(OBSTACLES[config["obstacle"]], width, height)
    obstacle = obstacle_mask(GRID, OBSTACLES[config["obstacle"]])
    if config.get("restricted_zones"):
        custom = restricted_mask(config["restricted_zones"], GRID, width, height)
        obstacle = obstacle | custom
        rects += mask_rectangles(custom, width, height)
    if obstacle.all() or len(rects) > 256:
        raise ValueError("Restrictions leave no usable bounded local map")
    return width, height, rects, obstacle


def validate_continuation(config: dict) -> None:
    """Validate server-owned continuation coordinates without moving any point.

    API callers must not be allowed to supply these trusted initial-state fields.
    Existing close neighbours remain measured safety violations; exact duplicate
    physical points, blocked positions, and invalid state are rejected.
    """
    count = config["drone_count"]
    width, height, rectangles, _ = _geometry(config)
    for key in ("initial_positions_m", "initial_velocities_mps", "initial_assigned_slots_m"):
        if config.get(key) is None:
            continue
        try:
            array = np.asarray(config[key], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid {key}") from error
        if array.shape != (count, 2) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}; expected one finite 2D state per vehicle")
        if key == "initial_velocities_mps":
            if np.any(np.linalg.norm(array, axis=1) > config["max_speed"] + 1e-7):
                raise ValueError("Initial velocity exceeds the mission speed limit")
        else:
            if np.any(array < .15 - 1e-8) or np.any(array > np.asarray([width-.15, height-.15]) + 1e-8):
                raise ValueError("Initial physical coordinate lies outside the local geofence")
            if any(x0-.15 < p[0] < x1+.15 and y0-.15 < p[1] < y1+.15 for p in array for x0, y0, x1, y1 in rectangles):
                raise ValueError("Initial physical coordinate lies inside an obstacle or safety margin")
            if key == "initial_positions_m" and len(np.unique(np.round(array, 8), axis=0)) != count:
                raise ValueError("Initial physical positions must not overlap exactly")
    if config.get("initial_batteries") is not None:
        battery = np.asarray(config["initial_batteries"], dtype=float)
        if battery.shape != (count,) or not np.isfinite(battery).all() or np.any(battery < 0) or np.any(battery > 100):
            raise ValueError("Invalid initial battery state")
    ids = config.get("source_drone_ids", list(range(1, count+1)))
    if not isinstance(ids, list) or len(ids) != count or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in ids) or len(set(ids)) != count:
        raise ValueError("Source vehicle identities must be unique positive integers")
    for key, allowed in (("initial_statuses", {"active", "failed", "hold"}), ("initial_links", {"online", "lost"})):
        value = config.get(key)
        if value is not None and (not isinstance(value, list) or len(value) != count or any(item not in allowed for item in value)):
            raise ValueError(f"Invalid {key}")
    indices = config.get("initial_target_zone_indices")
    if indices is not None and (not isinstance(indices, list) or len(indices) != count or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)):
        raise ValueError("Invalid initial target zone membership")
    if indices is not None and any(i >= len(config.get("targets") or _zones(config, 0)) for i in indices):
        raise ValueError("Initial target membership lies outside the target list")
    if (config.get("initial_assigned_slots_m") is None) != (indices is None):
        raise ValueError("Continuation slots and zone membership must be supplied together")
    elapsed = config.get("elapsed_steps", 0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
        raise ValueError("Elapsed physical steps must be a nonnegative integer")


def _arrival_state(positions, statuses, zone_indices, zones, width, height, velocities=None, hold=False):
    distances = np.full(len(positions), np.nan)
    arrived = np.zeros(len(positions), dtype=bool)
    zone_progress = []
    for index, zone in enumerate(zones):
        members = zone_indices == index
        if members.any():
            centre = np.asarray([zone["x"]*width, zone["y"]*height])
            distances[members] = np.linalg.norm(positions[members]-centre, axis=1)
            arrived[members] = ((statuses[members] == "active") &
                                (distances[members] <= zone["radius"]*min(width, height) + 1e-7))
            if hold:
                arrived[members] &= np.linalg.norm(velocities[members], axis=1) <= .15
        zone_progress.append({"index": index, "x": zone["x"], "y": zone["y"], "radius": zone["radius"],
                              "required_count": int(members.sum()), "arrived_count": int(arrived[members].sum())})
    total = len(positions)
    return arrived, distances, {"arrived_count": int(arrived.sum()), "total_count": total,
                                "remaining_count": total-int(arrived.sum()), "all_arrived": bool(arrived.all()),
                                "zones": zone_progress, "criterion": "inside_assigned_physical_zone"}


def simulate(config: dict, model=None, device="cpu", events: list[dict] | None = None,
             slot_gain: float = 1.0, include_frames: bool = True, progress_callback=None) -> dict[str, Any]:
    """Run the original density/slot/safety controller and record actual states.

    Event replays start from the seed and recompute the trajectory. This preserves
    the physical prefix (timings may vary). Failures and lost links hold the affected
    virtual vehicle; the remaining vehicles receive newly allocated target slots.
    """
    width, height, rects, obstacle = _geometry(config)
    count, nominal_steps = config["drone_count"], int(config["steps"])
    wait_for_arrival = bool(config.get("wait_for_arrival", False))
    resource_limit = min(2400, max(nominal_steps, 300_000 // max(1, count)))
    estimated = nominal_steps + math.ceil(math.hypot(width, height) / max(config["max_speed"], .1) / DT * 8)
    requested_limit = config.get("max_completion_steps")
    maximum_steps = min(resource_limit, int(requested_limit if requested_limit is not None else max(nominal_steps+200, estimated))) if wait_for_arrival else nominal_steps
    if maximum_steps < nominal_steps:
        raise ValueError("Completion step limit must cover the minimum mission horizon")
    frame_stride = max(1, math.ceil(maximum_steps / 380)) if wait_for_arrival else 1
    validate_continuation(config)
    cfg = FleetSimConfig(drone_count=config["drone_count"], planar_steps=config["steps"],
                         workspace_width_m=width, workspace_height_m=height,
                         grid_size=GRID, dt=DT, max_speed_mps=config["max_speed"],
                         min_separation_m=config["safe_distance"], seed=config["seed"],
                         controller=CONTROLLERS[config["controller"]],
                         obstacle_preset=OBSTACLES[config["obstacle"]], slot_gain=slot_gain)
    if config["controller"] != "baseline" and model is None:
        raise ValueError("Trained neural checkpoint is unavailable; select the mathematical baseline.")
    restrictions = config.get("restricted_zones") or []
    if config.get("initial_positions_m") is not None:
        pos = np.asarray(config["initial_positions_m"], dtype=float).copy()
    elif restrictions:
        pos = initial_free_positions(cfg.drone_count, rects, width, height, cfg.seed)
    else:
        pos = initial_positions(cfg.drone_count, width, height, "Сетка слева", rects,
                                cfg.seed, min_sep=min(.8, config["safe_distance"] * .5))
    if config["scenario"] == "return" and config.get("initial_positions_m") is None:
        pos[:, 0] = width - pos[:, 0]
        if restrictions:
            pos, _ = free_slots(pos, rects, width, height)
    vel = np.asarray(config.get("initial_velocities_mps") if config.get("initial_velocities_mps") is not None else np.zeros_like(pos), dtype=float).copy()
    battery = np.asarray(config.get("initial_batteries") if config.get("initial_batteries") is not None else np.full(cfg.drone_count, 100.0), dtype=float).copy()
    statuses = np.asarray(config.get("initial_statuses") or ["active"]*count, dtype=object).copy()
    links = np.asarray(config.get("initial_links") or ["online"]*count, dtype=object).copy()
    source_ids = config.get("source_drone_ids", list(range(1, count+1)))
    events = sorted(events or [], key=lambda e: e["step"])
    override = None
    old_zone_key = None
    assigned = None
    zone_indices = np.full(count, -1, dtype=int)
    all_slots = pos.copy()
    initial_assigned = None
    stage_hold = bool(config.get("stage_hold", False))
    if stage_hold:
        cfg.field_weight, cfg.slot_weight = 0., 1.
    hold_zones = [{"x": float(p[0]/width), "y": float(p[1]/height),
                   "radius": float((config.get("arrival_tolerance_m") or 1.0)/min(width, height)), "weight": 1.}
                  for p in pos] if stage_hold else None
    relocated_slots = 0
    frames, history = [], []
    started = time.perf_counter()
    target = np.zeros((GRID, GRID))
    termination_reason = "fixed_horizon" if not wait_for_arrival else "step_limit"
    best_distance, best_count, last_progress, consecutive_arrived = math.inf, -1, 0, 0
    elapsed_offset = int(config.get("elapsed_steps", 0))
    terminal_ready = bool(wait_for_arrival and config.get("initial_assigned_slots_m") is not None)
    previous_arrived = np.zeros(count, dtype=bool)
    corner_escapes = 0
    last_frame = None
    for step in range(maximum_steps + 1):
        changed = False
        for event in events:
            if event["step"] != step:
                continue
            if event["kind"] == "target_change":
                override = event.get("targets") or [{"x": .22, "y": .77, "radius": .16, "weight": 1.0}]
                changed = True
            else:
                idx = event["drone_id"] - 1
                statuses[idx] = "failed" if event["kind"] == "drone_failure" else "hold"
                if event["kind"] == "link_loss":
                    links[idx] = "lost"
                vel[idx] = 0
                changed = True
        active = np.flatnonzero(statuses == "active")
        # Moving target stages are held for 5 integration steps for stable assignment.
        moving = config["scenario"] in {"patrol", "scan"} and not config.get("targets") and not stage_hold
        zone_step = (min(step, nominal_steps) // 5) * 5 if moving and wait_for_arrival else ((step // 5) * 5 if moving else 0)
        if moving and wait_for_arrival and step >= nominal_steps:
            zone_step = nominal_steps
        zones = hold_zones or override or _zones(config, zone_step)
        targets_frozen = not moving or step >= nominal_steps or override is not None
        terminal_start = wait_for_arrival and not stage_hold and step >= nominal_steps and not terminal_ready
        zone_key = str(zones)
        if zone_key != old_zone_key or changed or terminal_start:
            target_points = _targets(zones, max(1, len(active)), width, height)
            target = target_density_from_points(GRID, target_points, width, height, obstacle)
            if len(active):
                slot_targets = [TargetPoint(t.label, t.x_m, t.y_m, t.count, t.radius_m*.72) for t in target_points] if terminal_start else target_points
                slots, labels = build_target_slots(slot_targets, width, height, rects)
                if restrictions or wait_for_arrival:
                    slots, moved = free_slots(slots, rects, width, height)
                    relocated_slots += moved
                assignment = assign_drones_to_slots(pos[active], slots)
                if terminal_start and np.all(zone_indices[active] >= 0):
                    # Interior slots do not change an aircraft's target group.
                    grouped = np.empty(len(active), dtype=int)
                    complete_groups = True
                    for group in range(len(zones)):
                        members = np.flatnonzero(zone_indices[active] == group)
                        available = np.asarray([j for j, label in enumerate(labels) if label == f"T{group+1}"], dtype=int)
                        if len(members) != len(available):
                            complete_groups = False
                            break
                        if len(members):
                            grouped[members] = available[assign_drones_to_slots(pos[active[members]], slots[available])]
                    if complete_groups:
                        assignment = grouped
                assigned = slots[assignment]
                zone_indices[active] = [int(labels[index][1:])-1 for index in assignment]
                if stage_hold:
                    assigned = pos[active].copy() if initial_assigned is None else all_slots[active]
                    zone_indices[active] = active
                elif step == 0 and config.get("initial_assigned_slots_m") is not None:
                    all_slots = np.asarray(config["initial_assigned_slots_m"], dtype=float).copy()
                    zone_indices = np.asarray(config["initial_target_zone_indices"], dtype=int)
                    if np.any(zone_indices >= len(zones)):
                        raise ValueError("Continuation membership lies outside the frozen target list")
                    assigned = all_slots[active].copy()
                all_slots[active] = assigned
                if initial_assigned is None:
                    initial_assigned = all_slots.copy()
            old_zone_key = zone_key
            if terminal_start:
                terminal_ready = True
            last_progress = step
            best_distance = math.inf
        compute_ms = 0.0
        rho = positions_to_density(pos, GRID, width, height, obstacle)
        flux = expert_flux(rho, target, obstacle)
        if step and len(active):
            if wait_for_arrival and not stage_hold and terminal_ready:
                blend = min(1., max(0., (elapsed_offset+step-nominal_steps)/50))
                cfg.field_weight, cfg.slot_weight = .35*(1-blend), .65+.35*blend
            cmd, _, flux, _, _, compute_ms = density_control_commands(
                pos[active], vel[active], assigned, target, obstacle, rects, cfg, model, device)
            if wait_for_arrival and terminal_ready and rects:
                # The inherited corner router can choose its current waypoint
                # exactly, producing a zero command at an expanded rectangle
                # corner. A nearby outward navigation goal breaks that fixed
                # point through the same acceleration/safety controller.
                navigation_slots = assigned.copy()
                escaped = False
                for local_index, drone in enumerate(active):
                    if previous_arrived[drone] or np.linalg.norm(cmd[local_index]) > .05:
                        continue
                    point = pos[drone]
                    candidates = []
                    margin = cfg.obstacle_margin_m
                    for x0, y0, x1, y1 in rects:
                        for x, y, sx, sy in ((x0-margin, y0-margin, -1, -1), (x0-margin, y1+margin, -1, 1),
                                             (x1+margin, y0-margin, 1, -1), (x1+margin, y1+margin, 1, 1)):
                            distance = np.linalg.norm(point-np.asarray([x, y]))
                            if distance < .45:
                                goal = np.clip(point + np.asarray([sx, sy])*.7, [.5, .5], [width-.5, height-.5])
                                if not any(_segment_intersects_rect(point, goal, rectangle, .15) for rectangle in rects):
                                    candidates.append((distance, goal))
                    if candidates:
                        navigation_slots[local_index] = min(candidates, key=lambda candidate: candidate[0])[1]
                        escaped = True
                        corner_escapes += 1
                if escaped:
                    cmd, _, flux, _, _, retry_ms = density_control_commands(
                        pos[active], vel[active], navigation_slots, target, obstacle, rects, cfg, model, device)
                    compute_ms += retry_ms
            new = _prevent_rect_crossing(pos[active], pos[active] + cmd * DT, rects, width, height)
            # Core integration checks endpoints. Confirmed restrictions additionally
            # reject an entire swept segment, including thin crossed obstacles.
            if restrictions or wait_for_arrival:
                for index, before in enumerate(pos[active]):
                    if any(_segment_intersects_rect(before, new[index], rectangle, .15) for rectangle in rects):
                        new[index] = before
            vel[active] = (new - pos[active]) / DT
            pos[active] = new
        speed = np.linalg.norm(vel, axis=1)
        if step:
            battery -= DT * (.012 + .003 * speed * speed) * (statuses != "failed")
        battery = np.clip(battery, 0, 100)
        rho = positions_to_density(pos, GRID, width, height, obstacle)
        minimum, violations = _safety(pos, cfg.min_separation_m)
        metrics = {"mae": float(np.abs(rho - target).mean()),
                   "tv_distance": float(.5 * np.abs(rho - target).sum()),
                   "compute_ms": compute_ms, "min_distance": minimum, "violations": violations}
        arrived, distances, arrival = _arrival_state(pos, statuses, zone_indices, zones, width, height, vel, stage_hold)
        previous_arrived = arrived.copy()
        arrival["targets_frozen"] = targets_frozen
        metrics.update(arrived_count=arrival["arrived_count"], remaining_count=arrival["remaining_count"])
        history.append({"step": step, "time": round(step * DT, 4), **metrics, "mean_speed": float(speed.mean())})
        if wait_for_arrival:
            outside = sum(max(0., distances[i]-zones[zone_indices[i]]["radius"]*min(width, height))
                          for i in active if np.isfinite(distances[i]) and 0 <= zone_indices[i] < len(zones))
            if outside < best_distance-.05 or arrival["arrived_count"] > best_count:
                best_distance, best_count, last_progress = outside, arrival["arrived_count"], step
            consecutive_arrived = consecutive_arrived+1 if arrival["all_arrived"] and targets_frozen else 0
            if step >= nominal_steps:
                if consecutive_arrived >= 3:
                    termination_reason = "all_arrived"
                elif np.any(statuses != "active"):
                    termination_reason = "vehicle_unavailable"
                elif targets_frozen and step-last_progress >= int(config.get("stall_window_steps", 150)):
                    termination_reason = "stalled"
            stop = step >= nominal_steps and termination_reason != "step_limit"
        else:
            stop = step == maximum_steps
        if progress_callback is not None and (step % 20 == 0 or stop or step == maximum_steps):
            progress_callback({"actual_steps": step, "planned_steps": nominal_steps,
                               "completion_step_limit": maximum_steps, "time_s": round(step*DT, 4),
                               "arrived_count": arrival["arrived_count"], "total_count": count,
                               "remaining_count": arrival["remaining_count"], "all_arrived": arrival["all_arrived"],
                               "targets_frozen": targets_frozen})
        if include_frames:
            agents = [{"id": i + 1, "x": float(pos[i, 0] / width), "y": float(pos[i, 1] / height),
                       "heading": float(math.atan2(vel[i, 1], vel[i, 0])), "speed": float(speed[i]),
                       "battery": float(battery[i]), "link": links[i], "status": statuses[i],
                       "altitude": 0.0 if statuses[i] == "failed" else 20.0,
                       "arrived": bool(arrived[i]), "source_drone_id": source_ids[i],
                       "target_zone_index": int(zone_indices[i]) if 0 <= zone_indices[i] < len(zones) else None,
                       "distance_to_target_m": float(distances[i]) if np.isfinite(distances[i]) else None}
                      for i in range(cfg.drone_count)]
            last_frame = {"step": step, "time": round(step * DT, 4), "agents": agents,
                           "density": _array(rho), "target": _array(target), "metrics": metrics,
                           "flux_x": _array(flux[0]), "flux_y": _array(flux[1]), "arrival": arrival}
            if step % frame_stride == 0 or step in {0, nominal_steps, maximum_steps} or changed or stop:
                frames.append(last_frame)
        if stop:
            break
    if include_frames and last_frame is not None and (not frames or frames[-1]["step"] != last_frame["step"]):
        frames.append(last_frame)
    final = history[-1]
    initial = history[0]["mae"]
    summary = {**{k: final[k] for k in ("mae", "tv_distance", "min_distance")},
               "compute_ms": float(np.mean([r["compute_ms"] for r in history[1:]])),
               "compute_total_ms": (time.perf_counter() - started) * 1000,
               "violations": int(sum(r["violations"] for r in history)),
               "max_violations": max(r["violations"] for r in history),
               "initial_mae": initial, "improvement_pct": 100 * (initial - final["mae"]) / max(initial, 1e-12),
               "duration_s": step * DT, "drone_count": cfg.drone_count,
               "frames": len(frames) if include_frames else len(history), "converged": final["tv_distance"] < .15,
               "convergence_time_s": next((r["time"] for r in history if r["tv_distance"] < .15), None),
               "active_drones": int(np.sum(statuses == "active")),
               "mission_completed": termination_reason == "all_arrived",
               "completion_status": "arrived" if termination_reason == "all_arrived" else ("incomplete" if wait_for_arrival else "fixed_horizon"),
               "termination_reason": termination_reason, "planned_steps": nominal_steps, "actual_steps": step,
               "arrived_count": arrival["arrived_count"], "total_count": count, "remaining_count": arrival["remaining_count"],
               "arrival_time_s": (elapsed_offset+step)*DT if termination_reason == "all_arrived" else None,
               "frame_stride": frame_stride, "integration_samples": len(history), "completion_step_limit": maximum_steps,
               "terminal_guidance": {"enabled": terminal_ready and not stage_hold,
                                     "method": "interior_slots_with_smooth_position_guidance",
                                     "interior_radius_ratio": .72, "blend_steps": 50,
                                     "physical_corner_escape_steps": corner_escapes},
               "controller_composition": {"field_weight": cfg.field_weight, "slot_weight": cfg.slot_weight,
                                          "neural_flux_weight": .7 if config["controller"] == "hybrid" else int(config["controller"] == "neural")},
               "battery_model": "100% minus dt*(0.012+0.003*speed^2); illustrative only",
               "safety_note": "Violations are measured point-pair samples in the planar model; no collision guarantee."}
    return {"frames": frames, "target": _array(target), "obstacle": obstacle.tolist(),
            "obstacle_rects": [{"x": x0/width, "y": y0/height, "width": (x1-x0)/width, "height": (y1-y0)/height}
                               for x0, y0, x1, y1 in rects],
            "history": history, "events": events, "summary": summary, "arrival": arrival,
            "workspace": {"width_m": width, "height_m": height, "dt": DT, "grid_size": GRID},
            "territory": config.get("map_snapshot"), "restricted_zones": restrictions,
            "restriction_policy": {"method": "conservative_grid_cell_cover", "grid_size": GRID,
                                   "target_slots_projected_to_free_cells": relocated_slots,
                                   "swept_segment_checks": bool(restrictions)},
            "initial_assigned_slots_m": _array(initial_assigned) if initial_assigned is not None else [],
            "final_state": {"positions_m": _array(pos), "velocities_mps": _array(vel), "batteries": _array(battery),
                            "statuses": statuses.tolist(), "links": links.tolist(), "source_drone_ids": source_ids,
                            "assigned_slots_m": _array(all_slots), "target_zone_indices": zone_indices.tolist(),
                            "target_zones": zones, "elapsed_steps": elapsed_offset+step},
            "simulation_only": True}
