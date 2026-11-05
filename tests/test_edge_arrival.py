"""Preloaded EDGE missions keep their arrival policy beyond a nominal horizon."""
import numpy as np
import pytest

from agmen_simulation import simulate
from edge import EdgeRuntime, build_package


def mission(**changes):
    config = {"drone_count": 5, "steps": 20, "max_speed": 3., "safe_distance": 1.,
              "seed": 19, "controller": "baseline", "obstacle": "none",
              "scenario": "distribution", "wait_for_arrival": True,
              "max_completion_steps": 1200, **changes}
    return {"id": "abcd1234abcd", "name": "Arrival regression", "status": "completed",
            "config": config, **simulate(config)}


def test_edge_continues_past_nominal_horizon_and_reaches_destination_offline(tmp_path):
    record = mission()
    package = build_package(record, 1)
    assert package["mission"]["limits"]["max_steps"] > record["config"]["steps"]
    assert package["mission"]["completion"]["wait_for_arrival"] is True
    runtime = EdgeRuntime(package, tmp_path)
    runtime.set_connectivity(internet=False)
    nominal = runtime.run_steps(20)
    assert nominal["status"] == "running"
    assert nominal["metrics"]["task_completion_fraction"] == 0
    final = runtime.run_steps(400)
    assert final["status"] == "completed"
    assert final["metrics"]["task_completion_fraction"] == 1
    assert np.linalg.norm(np.asarray(final["position_m"]) - package["mission"]["assigned_slot_m"]) <= 1


def test_edge_exhausted_budget_never_claims_arrival_or_returns_early(tmp_path):
    record = mission(max_completion_steps=20)
    runtime = EdgeRuntime(build_package(record, 1), tmp_path)
    result = runtime.run_steps(25)
    assert result["status"] == "awaiting_extension"
    assert result["metrics"]["task_completion_fraction"] == 0
    before = result["position_m"]
    assert runtime.run_steps(3)["position_m"] == before


def test_edge_stage_preserves_source_identity_velocity_and_battery(tmp_path):
    positions = [[4, y] for y in (10, 20, 30, 40, 50)]
    velocities = [[.2, .3]] * 5
    record = mission(initial_positions_m=positions, initial_velocities_mps=velocities,
                     initial_batteries=[72.] * 5, source_drone_ids=[11, 12, 13, 14, 15])
    record["stage"] = {"index": 2, "parent_mission_id": "1234abcd1234"}
    package = build_package(record, 2)
    assert package["mission"]["source_drone_id"] == 12
    assert package["mission"]["stage"]["index"] == 2
    runtime = EdgeRuntime(package, tmp_path)
    assert runtime.state["battery"] == 72
    assert np.allclose(runtime.state["position_m"], positions[1])
    assert np.allclose(runtime.state["velocity_mps"], velocities[1])


def test_edge_slot_tolerance_does_not_claim_arrival_outside_its_target_zone(tmp_path):
    record = mission(drone_count=25, max_speed=5.)
    package = build_package(record, 19)
    zone = package["mission"]["completion"]["assigned_zone_m"]
    runtime = EdgeRuntime(package, tmp_path)
    result = runtime.run_steps(300)
    assert result["status"] == "completed"
    assert np.linalg.norm(np.asarray(result["position_m"]) - [zone["x"], zone["y"]]) <= zone["radius"] + 1e-8


def test_radio_return_policy_has_priority_over_arrival_computation_budget(tmp_path):
    runtime = EdgeRuntime(build_package(mission(max_completion_steps=20), 1, policy="return"), tmp_path)
    runtime.run_steps(25)
    runtime.set_connectivity(ground_radio=False, peer_radio=False)
    assert runtime.run_steps(1)["status"] == "returning"


@pytest.mark.parametrize("drone_id", [3, 4])
def test_edge_physically_escapes_expanded_obstacle_corners(tmp_path, drone_id):
    from fleet_core import _segment_intersects_rect

    record = mission(drone_count=25, seed=42, safe_distance=1.2, obstacle="central")
    package = build_package(record, drone_id)
    runtime = EdgeRuntime(package, tmp_path)
    positions = [np.asarray(runtime.state["position_m"])]
    for _ in range(400):
        result = runtime.run_steps(1)
        positions.append(np.asarray(result["position_m"]))
        if result["status"] == "completed":
            break
    assert result["status"] == "completed"
    zone = package["mission"]["completion"]["assigned_zone_m"]
    assert np.linalg.norm(positions[-1] - [zone["x"], zone["y"]]) <= zone["radius"] + 1e-8
    moves = np.diff(positions, axis=0)
    assert np.linalg.norm(moves, axis=1).max() <= package["mission"]["limits"]["max_speed_mps"] * .2 + 1e-8
    assert all(not _segment_intersects_rect(before, after, rectangle, .15)
               for before, after in zip(positions, positions[1:])
               for rectangle in package["map"]["obstacle_rectangles_m"])
    assert result["metrics"]["constraint_violations"] == 0
