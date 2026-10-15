import numpy as np
from fleet_core import FleetSimConfig, TargetPoint, run_fleet_simulation, validate_targets


def test_1_9_distribution_runs():
    cfg=FleetSimConfig(drone_count=10,planar_steps=30,landing=False,seed=1,max_speed_mps=5.0)
    targets=[TargetPoint("A",62,15,1,2.0),TargetPoint("B",62,44,9,8.0)]
    r=run_fleet_simulation(cfg,targets,model=None)
    assert len(r.final_positions)==10
    assert np.isfinite(r.final_positions).all()
    assert r.commands.drone_id.nunique()==10
    assert r.history.phase.eq("PLANAR").sum()==30


def test_target_sum_validation():
    try:
        validate_targets([TargetPoint("A",10,10,2)],10,80,60)
    except ValueError:
        return
    raise AssertionError("validation must fail")


if __name__=='__main__':
    test_1_9_distribution_runs(); test_target_sum_validation(); print('fleet tests OK')
