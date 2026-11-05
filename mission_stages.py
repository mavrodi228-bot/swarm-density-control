"""Trusted continuation of simulated missions in explicitly registered local maps.

Image coordinates never establish GPS coordinates. A detailed map's origin is a
translation in the source mission's physical plane; metres are never rescaled.
"""
from __future__ import annotations

from collections import deque
import copy
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from territory import RestrictedZone, restricted_mask, mask_rectangles, free_slots


class NextStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    scenario: Literal["distribution", "patrol", "survey", "scan"]
    map_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    width_m: float | None = Field(default=None, ge=10, le=10000)
    height_m: float | None = Field(default=None, ge=10, le=10000)
    origin_x_m: float = Field(ge=-10000, le=10000)
    origin_y_m: float = Field(ge=-10000, le=10000)
    restricted_zones: list[RestrictedZone] = Field(default_factory=list, max_length=32)
    steps: int = Field(default=100, ge=20, le=300)
    max_speed: float | None = Field(default=None, ge=.1, le=15)
    controller: Literal["baseline", "neural", "hybrid"] | None = None
    target_zone_index: int | None = Field(default=None, ge=0, le=31)


class ContinueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int | None = Field(default=None, ge=20, le=300)


SCENARIOS = [
    {"id": "distribution", "label": "Распределение в области", "description": "Разместить прибывшую группу в выбранной области подробной карты."},
    {"id": "patrol", "label": "Патрулирование", "description": "Следовать за целевой областью, движущейся по замкнутому маршруту."},
    {"id": "survey", "label": "Обследование", "description": "Распределить группу по контрольным участкам подробной карты."},
    {"id": "scan", "label": "Сканирование", "description": "Перемещать контрольные области через подробную карту."},
]


def source_state(record: dict) -> dict:
    workspace = record.get("workspace") or {}
    width, height = float(workspace.get("width_m", 0)), float(workspace.get("height_m", 0))
    frames = record.get("frames") or []
    agents = frames[-1].get("agents", []) if frames else []
    if width <= 0 or height <= 0 or not agents:
        raise ValueError("В миссии нет сохранённого физического состояния аппаратов")
    final = record.get("final_state") or {}
    positions = final.get("positions_m") or [[a["x"] * width, a["y"] * height] for a in agents]
    velocities = final.get("velocities_mps") or [[a.get("speed", 0) * math.cos(a.get("heading", 0)), a.get("speed", 0) * math.sin(a.get("heading", 0))] for a in agents]
    batteries = final.get("batteries") or [a.get("battery", 0) for a in agents]
    identifiers = final.get("source_drone_ids") or [a.get("source_drone_id", a["id"]) for a in agents]
    if not all(len(values) == len(agents) for values in (positions, velocities, batteries, identifiers)):
        raise ValueError("Сохранённое состояние аппаратов неполно")
    return {"width_m": width, "height_m": height, "agents": agents, "positions": positions,
            "velocities": velocities, "batteries": batteries, "source_drone_ids": identifiers,
            "source_step": frames[-1].get("step", 0), "source_time_s": frames[-1].get("time", 0),
            "elapsed_steps": final.get("elapsed_steps", frames[-1].get("step", 0))}


def operational(record: dict) -> tuple[bool, str]:
    try:
        state = source_state(record)
    except (ValueError, TypeError, KeyError):
        return False, "Нет сохранённых координат. Сначала рассчитайте миссию."
    if any(a.get("status") != "active" or a.get("link") != "online" for a in state["agents"]):
        return False, "Есть неисправный аппарат или аппарат без связи. Продолжение группы заблокировано."
    if any(battery <= 0 for battery in state["batteries"]):
        return False, "У одного из аппаратов исчерпан модельный заряд."
    return True, ""


def stage_eligibility(record: dict) -> tuple[bool, str]:
    if record.get("status") != "completed":
        return False, "Дождитесь завершения текущего расчёта."
    ready, reason = operational(record)
    if not ready:
        return ready, reason
    arrival = record.get("arrival") or {}
    agents = source_state(record)["agents"]
    if not arrival.get("all_arrived") or not (record.get("summary") or {}).get("mission_completed"):
        return False, "Новый этап доступен после фактического прибытия всех аппаратов в назначенные области."
    if not all(a.get("arrived") is True for a in agents):
        return False, "Не подтверждено прибытие каждого аппарата. Пересчитайте исходную миссию."
    return True, "Все аппараты прибыли. Можно подготовить подробную карту следующего этапа."


def continuation_eligibility(record: dict) -> tuple[bool, str]:
    if record.get("status") != "completed":
        return False, "Сначала дождитесь завершения текущего расчёта."
    ready, reason = operational(record)
    if not ready:
        return False, reason
    summary = record.get("summary") or {}
    if summary.get("termination_reason") != "step_limit" or summary.get("mission_completed"):
        return False, "Продолжение доступно для незавершённой миссии, исчерпавшей вычислительный бюджет."
    if not record.get("final_state"):
        return False, "Для этой старой миссии нет состояния продолжения."
    return True, "Можно продолжить движение от последнего положения с теми же целями."


def suggested_region(record: dict, state: dict) -> dict:
    points = [tuple(p) for p in state["positions"]]
    for zone in (record.get("arrival") or {}).get("zones", []):
        radius = float(zone.get("radius", .1)) * min(state["width_m"], state["height_m"])
        x, y = zone["x"] * state["width_m"], zone["y"] * state["height_m"]
        points.extend([(x - radius, y - radius), (x + radius, y + radius)])
    padding = max(3.0, float(record["config"].get("safe_distance", 1)) * 2)
    x0, y0 = max(0.0, min(p[0] for p in points) - padding), max(0.0, min(p[1] for p in points) - padding)
    x1, y1 = min(state["width_m"], max(p[0] for p in points) + padding), min(state["height_m"], max(p[1] for p in points) + padding)
    width, height = max(10.0, x1 - x0), max(10.0, y1 - y0)
    x0, y0 = min(x0, state["width_m"] - width), min(y0, state["height_m"] - height)
    return {"origin_x_m": round(x0, 6), "origin_y_m": round(y0, 6), "width_m": round(width, 6), "height_m": round(height, 6)}


def stage_options(record: dict, children: list[dict]) -> dict:
    allowed, reason = stage_eligibility(record)
    can_continue, continue_reason = continuation_eligibility(record)
    try:
        state = source_state(record)
    except (ValueError, TypeError, KeyError):
        state = None
    participants = []
    if state:
        for a, p, battery, source_id in zip(state["agents"], state["positions"], state["batteries"], state["source_drone_ids"]):
            participants.append({"id": a["id"], "source_drone_id": source_id, "x_m": p[0], "y_m": p[1],
                                 "battery": battery, "target_zone_index": a.get("target_zone_index"), "arrived": a.get("arrived", False)})
    return {"allowed": allowed, "reason": reason, "parent_id": record["id"],
            "can_continue": can_continue, "continue_reason": continue_reason,
            "arrival": record.get("arrival"), "source_workspace": record.get("workspace"),
            "source_agents": participants, "suggested_region": suggested_region(record, state) if state else None,
            "source_territory": record.get("territory"), "zones": (record.get("arrival") or {}).get("zones", []),
            "scenarios": [{**s, "recommended": allowed and s["id"] in {"distribution", "survey"},
                           "available": allowed, "reason": "Перед запуском проверяются запреты и связность маршрута на подробной карте." if allowed else reason} for s in SCENARIOS],
            "next_stages": children,
            "coordinate_note": "Локальная плоскость: X вправо, Y вниз. Укажите начало подробной карты в метрах исходной карты. Масштаб 1:1 м; GPS по изображению не определяется."}


def _inherited_restrictions(record: dict, ox: float, oy: float, width: float, height: float) -> list[dict]:
    source_width, source_height = record["workspace"]["width_m"], record["workspace"]["height_m"]
    zones = []
    for rect in record.get("obstacle_rects", []):
        x0, y0 = max(0.0, rect["x"] * source_width - ox), max(0.0, rect["y"] * source_height - oy)
        x1 = min(width, (rect["x"] + rect["width"]) * source_width - ox)
        y1 = min(height, (rect["y"] + rect["height"]) * source_height - oy)
        if x1 > x0 + 1e-8 and y1 > y0 + 1e-8:
            zones.append({"kind": "rectangle", "x": x0 / width, "y": y0 / height,
                          "width": (x1 - x0) / width, "height": (y1 - y0) / height,
                          "id": f"source-restriction-{len(zones) + 1}"})
    return zones


def build_stage(record: dict, request: NextStageConfig, map_snapshot: dict | None) -> tuple[dict, dict]:
    allowed, reason = stage_eligibility(record)
    if not allowed:
        raise ValueError(reason)
    state = source_state(record)
    suggested = suggested_region(record, state)
    width = map_snapshot["width_m"] if map_snapshot else request.width_m or suggested["width_m"]
    height = map_snapshot["height_m"] if map_snapshot else request.height_m or suggested["height_m"]
    selected = [i for i, agent in enumerate(state["agents"]) if request.target_zone_index is None or agent.get("target_zone_index") == request.target_zone_index]
    if not selected:
        raise ValueError("В выбранной целевой области нет аппаратов")
    ox, oy = request.origin_x_m, request.origin_y_m
    positions = [[state["positions"][i][0] - ox, state["positions"][i][1] - oy] for i in selected]
    for position in positions:
        if not .15 <= position[0] <= width - .15 or not .15 <= position[1] <= height - .15:
            raise ValueError("Аппараты находятся вне подробной карты или у её края. Исправьте начало координат и физические размеры; позиции не перемещаются автоматически.")
    maximum = request.max_speed or record["config"]["max_speed"]
    velocities = [state["velocities"][i] for i in selected]
    if any(math.hypot(*v) > maximum + 1e-6 for v in velocities):
        raise ValueError("Максимальная скорость следующего этапа ниже текущей скорости аппарата")
    inherited = _inherited_restrictions(record, ox, oy, width, height)
    config = {**copy.deepcopy(record["config"]), "name": request.name or f"Следующий этап: {request.scenario}",
              "drone_count": len(selected), "scenario": request.scenario, "steps": request.steps,
              "controller": request.controller or record["config"]["controller"], "max_speed": maximum,
              "width_m": width, "height_m": height, "map_id": request.map_id, "map_snapshot": map_snapshot,
              "obstacle": "none", "targets": None, "wait_for_arrival": True,
              "restricted_zones": [*inherited, *(z.model_dump(exclude_none=True) for z in request.restricted_zones)],
              "initial_positions_m": positions, "initial_velocities_mps": velocities,
              "initial_batteries": [state["batteries"][i] for i in selected],
              "source_drone_ids": [state["source_drone_ids"][i] for i in selected],
              "initial_statuses": ["active"] * len(selected), "initial_links": ["online"] * len(selected), "elapsed_steps": 0}
    for key in ("initial_assigned_slots_m", "initial_target_zone_indices", "stage_hold"):
        config.pop(key, None)
    metadata = {"kind": "next_stage", "parent_id": record["id"], "source_step": state["source_step"],
                "source_time_s": state["source_time_s"], "source_drone_ids": config["source_drone_ids"],
                "source_target_zone_index": request.target_zone_index, "inherited_restriction_count": len(inherited),
                "registration": {"origin_x_m": ox, "origin_y_m": oy, "width_m": width, "height_m": height,
                                 "metres_per_metre": 1, "axes": "x_right_y_down", "gps_registered": False}}
    metadata.update(parent_mission_id=record["id"], index=(record.get("stage") or {}).get("index", 1) + 1)
    return config, metadata


def build_continuation(record: dict, request: ContinueConfig) -> tuple[dict, dict]:
    allowed, reason = continuation_eligibility(record)
    if not allowed:
        raise ValueError(reason)
    state, final = source_state(record), record["final_state"]
    config = copy.deepcopy(record["config"])
    config.update(name=f"{record['name'][:78]} · продолжение", steps=request.steps or config["steps"],
                  wait_for_arrival=True, initial_positions_m=state["positions"],
                  initial_velocities_mps=state["velocities"], initial_batteries=state["batteries"],
                  source_drone_ids=state["source_drone_ids"], initial_statuses=["active"] * len(state["agents"]),
                  initial_links=["online"] * len(state["agents"]), elapsed_steps=state["elapsed_steps"],
                  targets=copy.deepcopy(final["target_zones"]),
                  initial_assigned_slots_m=copy.deepcopy(final["assigned_slots_m"]),
                  initial_target_zone_indices=copy.deepcopy(final["target_zone_indices"]))
    metadata = {"kind": "arrival_continuation", "parent_id": record["id"], "source_step": state["source_step"],
                "source_time_s": state["source_time_s"], "elapsed_source_steps": state["elapsed_steps"],
                "source_drone_ids": state["source_drone_ids"], "registration": {"origin_x_m": 0, "origin_y_m": 0,
                    "width_m": state["width_m"], "height_m": state["height_m"], "metres_per_metre": 1,
                    "axes": "x_right_y_down", "gps_registered": False}}
    metadata.update(parent_mission_id=record["id"], index=(record.get("stage") or {}).get("index", 1))
    return config, metadata


def validate_stage_reachability(config: dict) -> None:
    """Conservative grid connectivity preflight; this is not a collision proof."""
    from agmen_simulation import GRID, OBSTACLES, _targets, _zones
    from fleet_core import obstacle_mask, obstacle_rectangles, build_target_slots, assign_drones_to_slots
    width, height = config["width_m"], config["height_m"]
    mask = obstacle_mask(GRID, OBSTACLES[config["obstacle"]])
    custom = restricted_mask(config.get("restricted_zones") or [], GRID, width, height)
    mask = mask | custom
    rects = obstacle_rectangles(OBSTACLES[config["obstacle"]], width, height) + mask_rectangles(custom, width, height)
    labels = np.full(mask.shape, -1, dtype=int)
    component = 0
    for y in range(GRID):
        for x in range(GRID):
            if mask[y, x] or labels[y, x] >= 0:
                continue
            queue = deque([(y, x)])
            labels[y, x] = component
            while queue:
                py, px = queue.popleft()
                for yy, xx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                    if 0 <= yy < GRID and 0 <= xx < GRID and not mask[yy, xx] and labels[yy, xx] < 0:
                        labels[yy, xx] = component
                        queue.append((yy, xx))
            component += 1
    def locate(point):
        return labels[min(GRID - 1, max(0, int(point[1] / height * GRID))), min(GRID - 1, max(0, int(point[0] / width * GRID)))]
    positions = np.asarray(config["initial_positions_m"], dtype=float)
    source_components = np.asarray([locate(point) for point in positions])
    if np.any(source_components < 0):
        raise ValueError("Подтверждённая запретная область накрывает текущее положение аппарата")
    sample_steps = sorted({*range(0, config["steps"] + 1, 5), config["steps"]})
    for step in sample_steps:
        points = _targets(_zones(config, step), len(positions), width, height)
        slots, _ = build_target_slots(points, width, height, rects)
        slots, _ = free_slots(slots, rects, width, height)
        assigned = slots[assign_drones_to_slots(positions, slots)]
        if any(locate(slot) != component for slot, component in zip(assigned, source_components)):
            raise ValueError("Целевая область сценария отделена запретами от аппаратов. Измените подробную карту или выберите другой сценарий.")
