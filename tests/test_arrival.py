"""Arrival-driven completion uses actual integrated states, never a time cutoff."""
import math

import numpy as np
import pytest

from agmen_simulation import simulate, validate_continuation, _zones
from fleet_core import _segment_intersects_rect


def config(**changes):
    value = dict(drone_count=5, steps=20, max_speed=3., safe_distance=1.2,
                 seed=42, controller="baseline", obstacle="none", scenario="distribution",
                 wait_for_arrival=True, max_completion_steps=1200)
    value.update(changes)
    return value


def test_wait_extends_short_horizon_until_every_vehicle_physically_enters_its_zone():
    result = simulate(config())
    summary = result["summary"]
    assert summary["mission_completed"] is True
    assert summary["completion_status"] == "arrived"
    assert summary["actual_steps"] > summary["planned_steps"] == 20
    assert summary["termination_reason"] == "all_arrived"
    assert summary["arrived_count"] == summary["total_count"] == 5
    assert summary["remaining_count"] == 0
    assert summary["duration_s"] == summary["actual_steps"]*.2
    for agent in result["frames"][-1]["agents"]:
        zone = result["arrival"]["zones"][agent["target_zone_index"]]
        distance = math.hypot((agent["x"]-zone["x"])*80, (agent["y"]-zone["y"])*60)
        assert distance <= zone["radius"]*60 + 1e-7
        assert agent["arrived"] is True and agent["status"] == "active"
    for previous, current in zip(result["frames"], result["frames"][1:]):
        dt = current["time"]-previous["time"]
        for before, after in zip(previous["agents"], current["agents"]):
            displacement = math.hypot((after["x"]-before["x"])*80, (after["y"]-before["y"])*60)
            assert displacement <= 3*dt + 1e-7


def test_default_25_fleet_reaches_zone_around_central_obstacle_with_real_guidance():
    progress = []
    result = simulate(config(drone_count=25, obstacle="central"), progress_callback=progress.append)
    assert result["summary"]["mission_completed"] is True
    assert result["summary"]["arrived_count"] == 25
    assert result["summary"]["terminal_guidance"]["enabled"] is True
    assert result["summary"]["terminal_guidance"]["physical_corner_escape_steps"] > 0
    assert len(result["frames"]) <= 400
    assert result["frames"][-1]["step"] == result["summary"]["actual_steps"]
    assert any(frame["step"] == 20 for frame in result["frames"])
    assert progress[0]["actual_steps"] == 0
    assert progress[-1]["all_arrived"] is True
    assert progress[-1]["actual_steps"] == result["summary"]["actual_steps"]
    # This short result is sampled; verify the full trajectory at stride 1 in a
    # separate run with a bounded cap near its actual completion step.
    full = simulate(config(drone_count=25, obstacle="central", max_completion_steps=300))
    rectangle = (36.8, 20.4, 45.6, 40.2)
    for previous, current in zip(full["frames"], full["frames"][1:]):
        for before, after in zip(previous["agents"], current["agents"]):
            p = np.asarray([before["x"]*80, before["y"]*60])
            q = np.asarray([after["x"]*80, after["y"]*60])
            assert not _segment_intersects_rect(p, q, rectangle)


@pytest.mark.parametrize("kind", ["drone_failure", "link_loss"])
def test_unavailable_vehicle_inside_zone_never_counts_as_success(kind):
    initial = [[15, 27], [17, 27], [19, 27], [15, 30], [18, 30]]
    result = simulate(config(initial_positions_m=initial, targets=[{"x": .22, "y": .5, "radius": .3, "weight": 1}]),
                      events=[{"kind": kind, "drone_id": 1, "step": 7}])
    assert result["frames"][0]["agents"][0]["arrived"] is True
    assert result["summary"]["mission_completed"] is False
    assert result["summary"]["termination_reason"] == "vehicle_unavailable"
    assert result["frames"][-1]["agents"][0]["arrived"] is False
    assert result["arrival"]["remaining_count"] >= 1


def test_unreachable_wall_stops_honestly_without_crossing_or_arriving():
    wall = {"kind": "rectangle", "x": .46, "y": 0., "width": .08, "height": 1.}
    result = simulate(config(restricted_zones=[wall], max_completion_steps=500, stall_window_steps=80))
    assert result["summary"]["mission_completed"] is False
    assert result["summary"]["termination_reason"] == "stalled"
    assert result["summary"]["actual_steps"] < 500
    assert result["arrival"]["arrived_count"] == 0
    assert all(agent["x"] < .46 for frame in result["frames"] for agent in frame["agents"])


@pytest.mark.parametrize("scenario", ["patrol", "scan"])
def test_moving_targets_freeze_at_nominal_horizon_while_physical_motion_continues(scenario):
    settings = config(scenario=scenario, max_completion_steps=40)
    result = simulate(settings)
    expected = _zones(settings, settings["steps"])
    after = [frame for frame in result["frames"] if frame["step"] >= 20]
    assert len(after) > 2
    for frame in after:
        assert frame["arrival"]["targets_frozen"] is True
        assert [{key: zone[key] for key in ("x", "y", "radius")} for zone in frame["arrival"]["zones"]] == [
            {key: zone[key] for key in ("x", "y", "radius")} for zone in expected]
        np.testing.assert_allclose(frame["target"], after[0]["target"])
    assert result["frames"][-1]["agents"] != result["frames"][20]["agents"]


def test_capped_chunk_can_continue_same_physics_ids_velocity_and_battery():
    settings = config(max_completion_steps=40, source_drone_ids=[11, 12, 13, 14, 15])
    first = simulate(settings)
    assert first["summary"]["mission_completed"] is False
    assert first["summary"]["termination_reason"] == "step_limit"
    state = first["final_state"]
    continuation = dict(settings, initial_positions_m=state["positions_m"], initial_velocities_mps=state["velocities_mps"],
                        initial_batteries=state["batteries"], initial_statuses=state["statuses"], initial_links=state["links"],
                        initial_assigned_slots_m=state["assigned_slots_m"], initial_target_zone_indices=state["target_zone_indices"],
                        targets=state["target_zones"], elapsed_steps=state["elapsed_steps"])
    second = simulate(continuation)
    uninterrupted = simulate(dict(settings, max_completion_steps=80))
    np.testing.assert_allclose(second["final_state"]["positions_m"], uninterrupted["final_state"]["positions_m"], atol=1e-7)
    np.testing.assert_allclose(second["final_state"]["velocities_mps"], uninterrupted["final_state"]["velocities_mps"], atol=1e-7)
    np.testing.assert_allclose(second["final_state"]["batteries"], uninterrupted["final_state"]["batteries"], atol=1e-7)
    assert second["final_state"]["elapsed_steps"] == 80
    assert [agent["source_drone_id"] for agent in second["frames"][0]["agents"]] == [11, 12, 13, 14, 15]
    assert second["frames"][0]["agents"][0]["battery"] < 100


@pytest.mark.parametrize("change", [
    {"initial_positions_m": [[0, 10]]*5},
    {"initial_positions_m": [[20, 20]]*5},
    {"initial_positions_m": [[50, 10], [20, 20], [21, 21], [22, 22], [float("nan"), 23]]},
    {"source_drone_ids": [1, 1, 2, 3, 4]},
    {"initial_velocities_mps": [[4, 0]]*5},
    {"initial_batteries": [100, 100, -1, 100, 100]},
], ids=["outside", "coincident", "nonfinite", "duplicate-identity", "overspeed", "invalid-battery"])
def test_invalid_physical_continuations_are_rejected_without_projection(change):
    settings = config(**change)
    with pytest.raises(ValueError):
        validate_continuation(settings)


def test_single_vehicle_hold_and_legacy_fixed_horizon_preserve_real_initial_state():
    hold = simulate(config(drone_count=1, stage_hold=True, initial_positions_m=[[40, 30]],
                           initial_velocities_mps=[[0, 0]], initial_batteries=[70], source_drone_ids=[23],
                           arrival_tolerance_m=None))
    assert hold["summary"]["mission_completed"] is True
    assert hold["frames"][0]["agents"][0]["battery"] == 70
    assert hold["frames"][-1]["agents"][0]["battery"] < 70
    assert hold["frames"][-1]["agents"][0]["source_drone_id"] == 23
    np.testing.assert_array_equal(hold["final_state"]["positions_m"], [[40, 30]])
    legacy_config = config()
    legacy_config.pop("wait_for_arrival")
    legacy = simulate(legacy_config)
    assert len(legacy["frames"]) == 21
    assert legacy["summary"]["actual_steps"] == 20
    assert legacy["summary"]["completion_status"] == "fixed_horizon"
    assert legacy["summary"]["terminal_guidance"]["enabled"] is False


def test_continuation_rejects_obstacle_instead_of_relocating_and_return_does_not_mirror_start():
    with pytest.raises(ValueError, match="obstacle"):
        validate_continuation(config(drone_count=1, obstacle="central", initial_positions_m=[[40, 30]]))
    result = simulate(config(drone_count=1, scenario="return", wait_for_arrival=False,
                             initial_positions_m=[[65, 30]], initial_velocities_mps=[[.5, 0]],
                             initial_batteries=[67], source_drone_ids=[9]))
    first = result["frames"][0]["agents"][0]
    assert first["x"]*80 == 65 and first["y"]*60 == 30
    assert first["speed"] == .5 and first["battery"] == 67 and first["source_drone_id"] == 9
