from __future__ import annotations

import threading
import time
import numpy as np

from fleet_core import (
    FleetSimConfig, TargetPoint, validate_targets, build_target_slots, assign_drones_to_slots,
    target_density_from_points, obstacle_mask, obstacle_rectangles, density_control_commands,
    leader_follower_commands,
)
from mavlink_bridge import MavlinkFleetManager


class RealMissionController:
    """Closes the loop: telemetry -> density/assignment -> velocity commands -> MAVLink."""
    def __init__(self,fleet:MavlinkFleetManager,model=None,device="cpu"):
        self.fleet=fleet; self.model=model; self.device=device
        self.cfg=None; self.targets=None; self.slots=None; self.assignment=None
        self.target_density=None; self.obstacle=None; self.rects=None
        self.velocities=None; self.running=False; self.thread=None; self.status="IDLE"; self.last_metrics={}

    def configure(self,cfg:FleetSimConfig,targets:list[TargetPoint]):
        if len(self.fleet.vehicles)!=cfg.drone_count:
            raise ValueError(f"Подключено {len(self.fleet.vehicles)} БПЛА, а миссия рассчитана на {cfg.drone_count}.")
        self.cfg=cfg
        self.targets=validate_targets(targets,cfg.drone_count,cfg.workspace_width_m,cfg.workspace_height_m)
        self.rects=obstacle_rectangles(cfg.obstacle_preset,cfg.workspace_width_m,cfg.workspace_height_m)
        self.obstacle=obstacle_mask(cfg.grid_size,cfg.obstacle_preset)
        self.target_density=target_density_from_points(cfg.grid_size,self.targets,cfg.workspace_width_m,cfg.workspace_height_m,self.obstacle)
        self.slots,_=build_target_slots(self.targets,cfg.workspace_width_m,cfg.workspace_height_m,self.rects)
        self.assignment=None; self.velocities=np.zeros((cfg.drone_count,2),float)
        self.fleet.set_workspace_offset(cfg.workspace_width_m/2,cfg.workspace_height_m/2)
        self.status="CONFIGURED"

    def _positions(self):
        pts=[]; names=[]
        for v in self.fleet.vehicles:
            xy=self.fleet.local_xy(v.telemetry)
            if xy is None: return None,None
            pts.append(xy); names.append(v.name)
        return np.asarray(pts,float),names

    def start(self):
        if self.cfg is None: raise RuntimeError("Сначала настройте миссию")
        if self.running: return
        pos,names=self._positions()
        if pos is None: raise RuntimeError("Нет координат GPS у всех подключённых БПЛА")
        self.assignment=assign_drones_to_slots(pos,self.slots)
        self.running=True; self.status="RUNNING"
        self.fleet.enter_external_control()
        self.thread=threading.Thread(target=self._loop,name="real-density-controller",daemon=True); self.thread.start()

    def _loop(self):
        period=max(0.12,self.cfg.dt)
        while self.running:
            t0=time.perf_counter()
            try:
                pos,names=self._positions()
                if pos is None:
                    self.status="WAITING FOR GPS"; time.sleep(period); continue
                assigned=self.slots[self.assignment]
                if self.cfg.guidance_mode=="Leader–Follower":
                    cmd,min_dist,unsafe,ms=leader_follower_commands(pos,self.velocities,assigned,self.cfg,self.rects)
                else:
                    cmd,_,_,min_dist,unsafe,ms=density_control_commands(pos,self.velocities,assigned,self.target_density,self.obstacle,self.rects,self.cfg,self.model,self.device)
                self.velocities=cmd
                self.fleet.set_velocity_xy({name:(float(cmd[i,0]),float(cmd[i,1])) for i,name in enumerate(names)},self.cfg.max_speed_mps)
                err=np.linalg.norm(pos-assigned,axis=1)
                self.last_metrics={"mean_target_error_m":float(err.mean()),"min_separation_m":float(min_dist),"unsafe_pairs":int(unsafe),"compute_ms":float(ms)}
                if np.all(err<1.5):
                    self.status="TARGET REACHED / HOLD"
                    self.fleet.hold_all(); self.running=False; break
                self.status="RUNNING"
            except Exception as exc:
                self.status=f"ERROR: {exc}"; self.fleet.hold_all(); self.running=False; break
            time.sleep(max(0.0,period-(time.perf_counter()-t0)))

    def hold(self):
        self.running=False
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.0)
        self.fleet.hold_all(); self.status="HOLD"

    def stop(self): self.hold()
