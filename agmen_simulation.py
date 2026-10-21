"""Deterministic planar simulation built from the repository's fleet controller.

No aircraft, GPS, radios, or batteries are connected. Coordinates are normalized
for rendering; integration and safety calculations use metres in an 80 x 60 m area.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from fleet_core import (
    FleetSimConfig, TargetPoint, assign_drones_to_slots, build_target_slots,
    density_control_commands, initial_positions, obstacle_mask, obstacle_rectangles,
    positions_to_density, target_density_from_points, _prevent_rect_crossing,
)
from swarm_core import expert_flux

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


def _targets(zones: list[dict], count: int) -> list[TargetPoint]:
    weights = np.asarray([z.get("weight", 1) for z in zones], float)
    raw = weights / weights.sum() * count
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[:count - int(counts.sum())]:
        counts[i] += 1
    return [TargetPoint(f"T{i+1}", z["x"] * WIDTH, z["y"] * HEIGHT,
                        int(n), z.get("radius", .12) * min(WIDTH, HEIGHT))
            for i, (z, n) in enumerate(zip(zones, counts)) if n > 0]


def _safety(positions: np.ndarray, distance: float) -> tuple[float, int]:
    if len(positions) < 2:
        return 0.0, 0
    d = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    d = d[np.triu_indices(len(positions), k=1)]
    return float(d.min()), int(np.count_nonzero(d < distance))


def _array(a: np.ndarray) -> list:
    return np.asarray(a).round(9).tolist()


def simulate(config: dict, model=None, device="cpu", events: list[dict] | None = None,
             slot_gain: float = 1.0, include_frames: bool = True) -> dict[str, Any]:
    """Run the original density/slot/safety controller and record actual states.

    Event replays start from the seed and recompute the trajectory. This preserves
    the physical prefix (timings may vary). Failures and lost links hold the affected
    virtual vehicle; the remaining vehicles receive newly allocated target slots.
    """
    cfg = FleetSimConfig(drone_count=config["drone_count"], planar_steps=config["steps"],
                         grid_size=GRID, dt=DT, max_speed_mps=config["max_speed"],
                         min_separation_m=config["safe_distance"], seed=config["seed"],
                         controller=CONTROLLERS[config["controller"]],
                         obstacle_preset=OBSTACLES[config["obstacle"]], slot_gain=slot_gain)
    if config["controller"] != "baseline" and model is None:
        raise ValueError("Trained neural checkpoint is unavailable; select the mathematical baseline.")
    rects = obstacle_rectangles(cfg.obstacle_preset, WIDTH, HEIGHT)
    obstacle = obstacle_mask(GRID, cfg.obstacle_preset)
    pos = initial_positions(cfg.drone_count, WIDTH, HEIGHT, "Сетка слева", rects,
                            cfg.seed, min_sep=min(.8, config["safe_distance"] * .5))
    if config["scenario"] == "return":
        pos[:, 0] = WIDTH - pos[:, 0]
    vel = np.zeros_like(pos)
    battery = np.full(cfg.drone_count, 100.0)
    statuses = np.full(cfg.drone_count, "active", dtype=object)
    links = np.full(cfg.drone_count, "online", dtype=object)
    events = sorted(events or [], key=lambda e: e["step"])
    override = None
    old_zone_key = None
    assigned = None
    frames, history = [], []
    started = time.perf_counter()
    target = np.zeros((GRID, GRID))
    for step in range(config["steps"] + 1):
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
        zone_step = (step // 5) * 5 if config["scenario"] in {"patrol", "scan"} else 0
        zones = override or _zones(config, zone_step)
        zone_key = str(zones)
        if zone_key != old_zone_key or changed:
            target_points = _targets(zones, max(1, len(active)))
            target = target_density_from_points(GRID, target_points, WIDTH, HEIGHT, obstacle)
            if len(active):
                slots, _ = build_target_slots(target_points, WIDTH, HEIGHT, rects)
                assigned = slots[assign_drones_to_slots(pos[active], slots)]
            old_zone_key = zone_key
        compute_ms = 0.0
        rho = positions_to_density(pos, GRID, WIDTH, HEIGHT, obstacle)
        flux = expert_flux(rho, target, obstacle)
        if step and len(active):
            cmd, _, flux, _, _, compute_ms = density_control_commands(
                pos[active], vel[active], assigned, target, obstacle, rects, cfg, model, device)
            new = _prevent_rect_crossing(pos[active], pos[active] + cmd * DT, rects, WIDTH, HEIGHT)
            vel[active] = (new - pos[active]) / DT
            pos[active] = new
        speed = np.linalg.norm(vel, axis=1)
        if step:
            battery -= DT * (.012 + .003 * speed * speed) * (statuses != "failed")
        battery = np.clip(battery, 0, 100)
        rho = positions_to_density(pos, GRID, WIDTH, HEIGHT, obstacle)
        minimum, violations = _safety(pos, cfg.min_separation_m)
        metrics = {"mae": float(np.abs(rho - target).mean()),
                   "tv_distance": float(.5 * np.abs(rho - target).sum()),
                   "compute_ms": compute_ms, "min_distance": minimum, "violations": violations}
        history.append({"step": step, "time": round(step * DT, 4), **metrics, "mean_speed": float(speed.mean())})
        if include_frames:
            agents = [{"id": i + 1, "x": float(pos[i, 0] / WIDTH), "y": float(pos[i, 1] / HEIGHT),
                       "heading": float(math.atan2(vel[i, 1], vel[i, 0])), "speed": float(speed[i]),
                       "battery": float(battery[i]), "link": links[i], "status": statuses[i],
                       "altitude": 0.0 if statuses[i] == "failed" else 20.0}
                      for i in range(cfg.drone_count)]
            frames.append({"step": step, "time": round(step * DT, 4), "agents": agents,
                           "density": _array(rho), "target": _array(target), "metrics": metrics,
                           "flux_x": _array(flux[0]), "flux_y": _array(flux[1])})
    final = history[-1]
    initial = history[0]["mae"]
    summary = {**{k: final[k] for k in ("mae", "tv_distance", "min_distance")},
               "compute_ms": float(np.mean([r["compute_ms"] for r in history[1:]])),
               "compute_total_ms": (time.perf_counter() - started) * 1000,
               "violations": int(sum(r["violations"] for r in history)),
               "max_violations": max(r["violations"] for r in history),
               "initial_mae": initial, "improvement_pct": 100 * (initial - final["mae"]) / max(initial, 1e-12),
               "duration_s": config["steps"] * DT, "drone_count": cfg.drone_count,
               "frames": len(history), "converged": final["tv_distance"] < .15,
               "convergence_time_s": next((r["time"] for r in history if r["tv_distance"] < .15), None),
               "active_drones": int(np.sum(statuses == "active")),
               "controller_composition": {"field_weight": cfg.field_weight, "slot_weight": cfg.slot_weight,
                                          "neural_flux_weight": .7 if config["controller"] == "hybrid" else int(config["controller"] == "neural")},
               "battery_model": "100% minus dt*(0.012+0.003*speed^2); illustrative only",
               "safety_note": "Violations are measured point-pair samples in the planar model; no collision guarantee."}
    return {"frames": frames, "target": _array(target), "obstacle": obstacle.tolist(),
            "obstacle_rects": [{"x": x0/WIDTH, "y": y0/HEIGHT, "width": (x1-x0)/WIDTH, "height": (y1-y0)/HEIGHT}
                               for x0, y0, x1, y1 in rects],
            "history": history, "events": events, "summary": summary,
            "workspace": {"width_m": WIDTH, "height_m": HEIGHT, "dt": DT, "grid_size": GRID},
            "simulation_only": True}
