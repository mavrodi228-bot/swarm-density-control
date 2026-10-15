"""Regression checks for the civilian, offline research simulation layer."""
from pathlib import Path
import tempfile
import pandas as pd
from fleet_core import FleetSimConfig, TargetPoint, run_fleet_simulation
from mission_scenarios import tasks_from_dataframe, default_tasks, simulate_tasks, save_simulation_package

def main():
    cfg=FleetSimConfig(drone_count=4,workspace_width_m=60,workspace_height_m=45,
                       obstacle_preset='Нет',controller='Математический эталон',
                       planar_steps=12,flight_altitude_m=4,landing=False)
    targets=[TargetPoint('A',45,10,1,3),TargetPoint('B',43,32,3,6)]
    mission=run_fleet_simulation(cfg,targets)
    tasks=tasks_from_dataframe(default_tasks(targets),[t.label for t in targets])
    okay=simulate_tasks(mission,tasks)
    assert len(okay.coverage)==4
    assert okay.summary['Сценариев завершено']==4
    partial=simulate_tasks(mission,tasks,link_loss_s=6,failsafe='HOLD')
    assert partial.summary['Сценариев прервано']==4
    assert 'LINK_LOSS' in partial.events['kind'].tolist()
    with tempfile.TemporaryDirectory() as p:
        file=save_simulation_package(mission,tasks,partial,p)
        assert Path(file).is_file()
        assert 'NOT_FOR_FLIGHT' in Path(file).read_text('utf-8')
    try:
        bad=default_tasks(targets).drop(0)
        tasks_from_dataframe(bad,[t.label for t in targets])
        raise AssertionError('Expected validation error')
    except ValueError:
        pass
    print('mission scenarios tests OK')
if __name__ == '__main__': main()
