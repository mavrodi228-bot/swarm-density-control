from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from swarm_core import gaussian_density, expert_flux, make_obstacle

EPS = 1e-9


@dataclass
class TargetPoint:
    label: str
    x_m: float
    y_m: float
    count: int
    radius_m: float = 4.0


@dataclass
class FleetSimConfig:
    drone_count: int = 10
    workspace_width_m: float = 80.0
    workspace_height_m: float = 60.0
    flight_altitude_m: float = 20.0
    initial_preset: str = "Линия слева"
    obstacle_preset: str = "Центральный блок"
    controller: str = "Гибрид 70/30"
    guidance_mode: str = "Density control"
    grid_size: int = 48
    planar_steps: int = 220
    dt: float = 0.20
    max_speed_mps: float = 4.0
    max_accel_mps2: float = 3.0
    climb_rate_mps: float = 2.0
    descent_rate_mps: float = 1.5
    min_separation_m: float = 2.5
    safety_gain: float = 3.2
    slot_gain: float = 1.00
    field_weight: float = 0.35
    slot_weight: float = 0.65
    obstacle_margin_m: float = 3.0
    position_noise_m: float = 0.0
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.0
    landing: bool = True
    seed: int = 42


@dataclass
class FleetSimulationResult:
    config: FleetSimConfig
    targets: list[TargetPoint]
    target_slots: np.ndarray
    assignment: np.ndarray
    initial_positions: np.ndarray
    final_positions: np.ndarray
    final_altitudes: np.ndarray
    trajectories: np.ndarray
    altitude_history: np.ndarray
    history: pd.DataFrame
    commands: pd.DataFrame
    target_density: np.ndarray
    final_density: np.ndarray
    obstacle: np.ndarray
    obstacle_rects: list[tuple[float, float, float, float]]
    summary: dict


def validate_targets(targets: Iterable[TargetPoint], drone_count: int, width: float, height: float) -> list[TargetPoint]:
    clean: list[TargetPoint] = []
    total = 0
    for idx, t in enumerate(targets):
        count = int(t.count)
        if count < 0:
            raise ValueError("Количество БПЛА в точке не может быть отрицательным.")
        if count == 0:
            continue
        x, y = float(t.x_m), float(t.y_m)
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"Цель {t.label or idx+1}: координаты должны находиться внутри рабочей области.")
        radius = max(0.25, float(t.radius_m))
        clean.append(TargetPoint(t.label or f"T{idx+1}", x, y, count, radius))
        total += count
    if not clean:
        raise ValueError("Нужно задать хотя бы одну целевую точку.")
    if total != int(drone_count):
        raise ValueError(f"Сумма БПЛА по целям ({total}) должна равняться размеру группы ({drone_count}).")
    return clean


def targets_from_dataframe(data, drone_count: int, width: float, height: float) -> list[TargetPoint]:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("Таблица целей пуста.")
    # Gradio can return named or integer columns.
    desired = ["Точка", "X, м", "Y, м", "БПЛА", "Радиус, м"]
    if all(c in df.columns for c in desired):
        cols = desired
    else:
        if df.shape[1] < 5:
            raise ValueError("Таблица целей должна содержать 5 столбцов.")
        cols = list(df.columns[:5])
    targets = []
    for _, row in df.iterrows():
        vals = [row[c] for c in cols]
        if all(pd.isna(v) or str(v).strip() == "" for v in vals):
            continue
        try:
            label = str(vals[0]).strip() if not pd.isna(vals[0]) else ""
            x = float(vals[1]); y = float(vals[2]); count = int(float(vals[3])); radius = float(vals[4])
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать строку цели: {list(vals)}") from exc
        targets.append(TargetPoint(label, x, y, count, radius))
    return validate_targets(targets, drone_count, width, height)


def obstacle_rectangles(preset: str, width: float, height: float) -> list[tuple[float, float, float, float]]:
    if preset == "Нет":
        return []
    if preset == "Центральный блок":
        return [(0.46*width, 0.34*height, 0.57*width, 0.67*height)]
    if preset == "Коридор":
        return [
            (0.44*width, 0.12*height, 0.54*width, 0.43*height),
            (0.44*width, 0.58*height, 0.54*width, 0.90*height),
        ]
    if preset == "Два блока":
        return [
            (0.42*width, 0.18*height, 0.54*width, 0.42*height),
            (0.52*width, 0.60*height, 0.64*width, 0.82*height),
        ]
    if preset == "Стена с проходом":
        return [
            (0.47*width, 0.0, 0.53*width, 0.42*height),
            (0.47*width, 0.60*height, 0.53*width, height),
        ]
    raise ValueError(f"Неизвестный тип препятствия: {preset}")


def obstacle_mask(n: int, preset: str) -> np.ndarray:
    return make_obstacle(n, preset)


def _point_in_rect(p: np.ndarray, r: tuple[float,float,float,float], margin: float = 0.0) -> bool:
    x0,y0,x1,y1 = r
    return (x0-margin <= p[0] <= x1+margin) and (y0-margin <= p[1] <= y1+margin)


def _is_free(p: np.ndarray, rects, width, height, margin=0.0) -> bool:
    if p[0] < margin or p[0] > width-margin or p[1] < margin or p[1] > height-margin:
        return False
    return not any(_point_in_rect(p, r, margin) for r in rects)


def initial_positions(count: int, width: float, height: float, preset: str, rects, seed: int = 42, min_sep: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(count)
    if n < 1:
        raise ValueError("Число БПЛА должно быть >= 1")

    if preset == "Линия слева" and n <= 18:
        ys = np.linspace(0.12*height, 0.88*height, n)
        xs = np.full(n, 0.12*width)
        pos = np.column_stack([xs, ys])
    elif preset in {"Линия слева", "Сетка слева"}:
        # For larger groups a literal single line would violate the requested separation.
        # We therefore expand it into a launch grid in the left part of the workspace.
        span_x, span_y = 0.34*width, 0.84*height
        cols = max(1, int(round(math.sqrt(n*span_x/max(span_y,1e-6)))))
        rows = int(math.ceil(n/cols))
        while cols*rows < n:
            cols += 1; rows = int(math.ceil(n/cols))
        xs = np.linspace(0.06*width, 0.40*width, cols)
        ys = np.linspace(0.08*height, 0.92*height, rows)
        pos = np.array([(x,y) for x in xs for y in ys], dtype=float)[:n]
    elif preset == "Кластер слева":
        pos = np.column_stack([
            rng.normal(0.16*width, 0.05*width, n),
            rng.normal(0.50*height, 0.16*height, n),
        ])
        pos[:,0] = np.clip(pos[:,0], 0.04*width, 0.34*width)
        pos[:,1] = np.clip(pos[:,1], 0.06*height, 0.94*height)
    elif preset == "Случайно":
        pos = np.empty((0,2), float)
    else:
        raise ValueError(f"Неизвестное начальное расположение: {preset}")

    # Repair points in obstacles and (for random preset) generate by rejection.
    accepted: list[np.ndarray] = []
    candidates = list(pos) if len(pos) else []
    tries = 0
    while len(accepted) < n and tries < 100000:
        tries += 1
        if candidates:
            p = np.asarray(candidates.pop(0), float)
            if not _is_free(p, rects, width, height, margin=0.4):
                p = np.array([rng.uniform(0.04*width,0.34*width), rng.uniform(0.06*height,0.94*height)])
        else:
            p = np.array([rng.uniform(0.05*width,0.35*width), rng.uniform(0.05*height,0.95*height)])
        if not _is_free(p, rects, width, height, margin=0.35):
            continue
        # soft separation: relax if density is too high
        required = min_sep * 0.95
        if accepted and min(np.linalg.norm(p-q) for q in accepted) < required:
            if tries < 5000:
                continue
        accepted.append(p)
    if len(accepted) < n:
        raise RuntimeError("Не удалось разместить БПЛА в выбранной области без препятствий.")
    return np.vstack(accepted[:n])


def build_target_slots(targets: list[TargetPoint], width: float, height: float, rects=None) -> tuple[np.ndarray, list[str]]:
    slots = []
    labels = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    rects = rects or []
    for t in targets:
        k = int(t.count)
        if k == 1:
            candidates = [np.array([t.x_m, t.y_m], float)]
        else:
            candidates = []
            for i in range(k):
                rr = t.radius_m * math.sqrt((i + 0.35) / k)
                th = i * golden
                candidates.append(np.array([t.x_m + rr*math.cos(th), t.y_m + rr*math.sin(th)], float))
        for p in candidates:
            p[0] = np.clip(p[0], 0.5, width-0.5)
            p[1] = np.clip(p[1], 0.5, height-0.5)
            if any(_point_in_rect(p, r, 0.5) for r in rects):
                # Pull slot toward target center first, then small deterministic spiral.
                v = np.array([t.x_m, t.y_m]) - p
                for a in np.linspace(0.2, 1.0, 5):
                    q = p + a*v
                    if not any(_point_in_rect(q, r, 0.5) for r in rects):
                        p = q; break
            slots.append(p)
            labels.append(t.label)
    return np.vstack(slots), labels


def assign_drones_to_slots(positions: np.ndarray, slots: np.ndarray) -> np.ndarray:
    if len(positions) != len(slots):
        raise ValueError("Число позиций и целевых мест должно совпадать.")
    cost = np.linalg.norm(positions[:,None,:] - slots[None,:,:], axis=2)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.empty(len(positions), dtype=int)
    assignment[rows] = cols
    return assignment


def target_density_from_points(n: int, targets: list[TargetPoint], width: float, height: float, obstacle: np.ndarray) -> np.ndarray:
    centers=[]; sigmas=[]; weights=[]
    scale = max(width, height)
    for t in targets:
        centers.append((t.x_m/width, t.y_m/height))
        sigma_m = max(1.0, t.radius_m*0.55)
        sigmas.append(sigma_m/scale)
        weights.append(max(1,t.count))
    # gaussian_density assumes same scale x/y; compensate by using max dimension. Good enough for rectangular workspace.
    return gaussian_density(n, centers, sigmas, weights, obstacle)


def positions_to_density(positions: np.ndarray, n: int, width: float, height: float, obstacle: np.ndarray, smooth_sigma: float = 1.25) -> np.ndarray:
    x = np.clip(positions[:,0] / max(width,EPS), 0, 0.999999)
    y = np.clip(positions[:,1] / max(height,EPS), 0, 0.999999)
    H, _, _ = np.histogram2d(y, x, bins=n, range=[[0,1],[0,1]])
    H = gaussian_filter(H.astype(float), smooth_sigma, mode="nearest")
    H[obstacle] = 0.0
    s = H.sum()
    if s <= EPS:
        H[~obstacle] = 1.0
        s = H.sum()
    return H/s


def bilinear_sample(field: np.ndarray, positions: np.ndarray, width: float, height: float) -> np.ndarray:
    n, m = field.shape
    gx = np.clip(positions[:,0] / max(width,EPS) * (m-1), 0, m-1-1e-9)
    gy = np.clip(positions[:,1] / max(height,EPS) * (n-1), 0, n-1-1e-9)
    x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0+1, m-1); y1 = np.minimum(y0+1, n-1)
    tx = gx-x0; ty=gy-y0
    return ((1-tx)*(1-ty)*field[y0,x0] + tx*(1-ty)*field[y0,x1] +
            (1-tx)*ty*field[y1,x0] + tx*ty*field[y1,x1])


def _clip_vectors(v: np.ndarray, limit: float) -> np.ndarray:
    mag = np.linalg.norm(v, axis=1)
    scale = np.ones_like(mag)
    mask = mag > limit
    scale[mask] = limit/(mag[mask]+EPS)
    return v*scale[:,None]


def separation_velocity(positions: np.ndarray, min_sep: float, gain: float, max_component: float) -> tuple[np.ndarray, float, int]:
    """Vectorised local separation field for up to hundreds of agents."""
    n=len(positions)
    if n < 2:
        return np.zeros_like(positions,dtype=float), 999.0, 0
    diff=positions[:,None,:]-positions[None,:,:]
    dist=np.linalg.norm(diff,axis=2)
    np.fill_diagonal(dist,np.inf)
    min_d=float(np.min(dist))
    unsafe_pairs=int(np.sum(np.triu(dist < min_sep,k=1)))
    trigger=max(min_sep*1.8,min_sep+0.5)
    mask=dist < trigger
    direction=np.zeros_like(diff)
    direction[mask]=diff[mask]/(dist[mask,None]+EPS)
    strength=np.zeros_like(dist)
    strength[mask]=gain*(trigger-dist[mask])/trigger
    danger=dist < min_sep
    strength[danger]*=1.8
    rep=np.sum(direction*strength[:,:,None],axis=1)
    return _clip_vectors(rep,max_component),min_d,unsafe_pairs

def obstacle_boundary_velocity(positions: np.ndarray, rects, width: float, height: float, margin: float, gain: float) -> np.ndarray:
    out = np.zeros_like(positions, dtype=float)
    safe = max(margin, 0.5)
    for i,p in enumerate(positions):
        # Workspace boundary repulsion.
        if p[0] < safe: out[i,0] += gain*(safe-p[0])/safe
        if p[0] > width-safe: out[i,0] -= gain*(p[0]-(width-safe))/safe
        if p[1] < safe: out[i,1] += gain*(safe-p[1])/safe
        if p[1] > height-safe: out[i,1] -= gain*(p[1]-(height-safe))/safe
        for x0,y0,x1,y1 in rects:
            qx = np.clip(p[0], x0, x1); qy = np.clip(p[1], y0, y1)
            delta = p - np.array([qx,qy])
            d = float(np.linalg.norm(delta))
            inside = x0 <= p[0] <= x1 and y0 <= p[1] <= y1
            if inside:
                # Push toward nearest side.
                ds = [(p[0]-x0, np.array([-1.,0.])), (x1-p[0],np.array([1.,0.])),
                      (p[1]-y0,np.array([0.,-1.])), (y1-p[1],np.array([0.,1.]))]
                _, direction = min(ds, key=lambda z:z[0])
                out[i] += direction*gain*2.0
            elif d < safe:
                direction = delta/(d+EPS)
                out[i] += direction*gain*(safe-d)/safe
    return out



def _segment_intersects_rect(p: np.ndarray, q: np.ndarray, r: tuple[float,float,float,float], margin: float = 0.0) -> bool:
    """Liang-Barsky style segment/AABB intersection."""
    x0,y0,x1,y1=r
    x0-=margin; y0-=margin; x1+=margin; y1+=margin
    d=q-p
    tmin,tmax=0.0,1.0
    for a,b,lo,hi in ((p[0],d[0],x0,x1),(p[1],d[1],y0,y1)):
        if abs(b)<1e-12:
            if a<lo or a>hi: return False
            continue
        t1=(lo-a)/b; t2=(hi-a)/b
        if t1>t2: t1,t2=t2,t1
        tmin=max(tmin,t1); tmax=min(tmax,t2)
        if tmin>tmax: return False
    return True


def route_waypoint(p: np.ndarray, target: np.ndarray, rects, width: float, height: float, margin: float) -> np.ndarray:
    """Return target or a temporary corner waypoint if a forbidden rectangle blocks line-of-sight."""
    blockers=[r for r in rects if _segment_intersects_rect(p,target,r,margin)]
    if not blockers: return target
    # Use nearest blocking rectangle along the path.
    r=min(blockers,key=lambda rr: np.linalg.norm(p-np.array([(rr[0]+rr[2])/2,(rr[1]+rr[3])/2])))
    x0,y0,x1,y1=r; m=max(0.8,margin)
    corners=[np.array([x0-m,y0-m]),np.array([x0-m,y1+m]),np.array([x1+m,y0-m]),np.array([x1+m,y1+m])]
    valid=[]
    for c in corners:
        c[0]=np.clip(c[0],0.5,width-0.5); c[1]=np.clip(c[1],0.5,height-0.5)
        if any(_point_in_rect(c,rr,0.25) for rr in rects): continue
        # Penalise candidates whose second leg still cuts through many obstacles.
        intersections=sum(_segment_intersects_rect(c,target,rr,margin*0.6) for rr in rects)
        cost=np.linalg.norm(p-c)+np.linalg.norm(c-target)+intersections*max(width,height)
        valid.append((cost,c))
    return min(valid,key=lambda z:z[0])[1] if valid else target


def slot_guidance_velocity(positions: np.ndarray, assigned_slots: np.ndarray, rects, cfg: FleetSimConfig) -> np.ndarray:
    out=np.zeros_like(positions,float)
    for i,(p,t) in enumerate(zip(positions,assigned_slots)):
        w=route_waypoint(p,t,rects,cfg.workspace_width_m,cfg.workspace_height_m,cfg.obstacle_margin_m)
        delta=w-p
        # Near the actual target, use the true target rather than a stale corner waypoint.
        if np.linalg.norm(t-p)<max(2.0,cfg.obstacle_margin_m): delta=t-p
        out[i]=cfg.slot_gain*delta
    return _clip_vectors(out,0.75*cfg.max_speed_mps)


def enforce_pairwise_velocity_safety(positions: np.ndarray, cmd: np.ndarray, min_sep: float, dt: float, max_speed: float) -> np.ndarray:
    """Vectorised velocity-level safety projection.

    It suppresses pairwise closing motion predicted to violate the requested separation
    on the next integration step. Four bounded iterations are sufficient for the
    1–100-agent range used by the application.
    """
    n=len(positions)
    if n < 2:
        return _clip_vectors(cmd,max_speed)
    v=cmd.copy()
    trigger=max(min_sep*1.55,min_sep+0.55)
    rel=positions[:,None,:]-positions[None,:,:]
    dist=np.linalg.norm(rel,axis=2)
    eye=np.eye(n,dtype=bool)
    safe_dist=np.where(eye,1.0,np.maximum(dist,1e-6))
    normal=rel/safe_dist[:,:,None]
    # deterministic normals for exact overlaps are uncommon; leave zero and rely on separation layer
    for _ in range(4):
        vrel=v[:,None,:]-v[None,:,:]
        pred_rel=rel+vrel*dt
        pred_d=np.linalg.norm(pred_rel,axis=2)
        mask=(~eye)&(dist<trigger)&(pred_d<min_sep)
        required=np.zeros_like(dist)
        required[mask]=(min_sep-pred_d[mask])/max(dt,1e-3)+0.15
        inside=(~eye)&(dist<min_sep)
        required[inside]=np.maximum(required[inside],0.75*(min_sep-dist[inside])/max(dt,1e-3))
        # Half-to-each vehicle is naturally represented by summing symmetric pair corrections.
        corr=0.78*required[:,:,None]*normal
        v += np.sum(corr,axis=1)
        v=_clip_vectors(v,max_speed)
    return v

def density_control_commands(
    positions: np.ndarray,
    velocities: np.ndarray,
    assigned_slots: np.ndarray,
    target_density: np.ndarray,
    obstacle: np.ndarray,
    rects,
    cfg: FleetSimConfig,
    model=None,
    device="cpu",
):
    t0=time.perf_counter()
    rho = positions_to_density(positions, cfg.grid_size, cfg.workspace_width_m, cfg.workspace_height_m, obstacle)
    ex,ey = expert_flux(rho, target_density, obstacle)
    if model is not None and cfg.controller != "Математический эталон":
        from model import predict_flux
        nx,ny = predict_flux(model, rho, target_density, obstacle, device=device)
        if cfg.controller == "Нейросеть":
            jx,jy=nx,ny
        else:
            jx,jy=0.70*nx+0.30*ex, 0.70*ny+0.30*ey
    else:
        jx,jy=ex,ey

    sx = bilinear_sample(jx, positions, cfg.workspace_width_m, cfg.workspace_height_m)
    sy = bilinear_sample(jy, positions, cfg.workspace_width_m, cfg.workspace_height_m)
    macro = np.column_stack([sx,sy])
    # Flux values are small; direction is physically meaningful, magnitude is rescaled to a practical speed envelope.
    mm = np.linalg.norm(macro,axis=1)
    positive = mm[mm>1e-12]
    ref = float(np.median(positive)) if len(positive) else 1.0
    macro = macro/(ref+EPS) * (0.55*cfg.max_speed_mps)
    macro = _clip_vectors(macro, 0.75*cfg.max_speed_mps)

    slot = slot_guidance_velocity(positions, assigned_slots, rects, cfg)
    sep,min_dist,unsafe = separation_velocity(positions, cfg.min_separation_m, cfg.safety_gain, 0.95*cfg.max_speed_mps)
    obs = obstacle_boundary_velocity(positions, rects, cfg.workspace_width_m, cfg.workspace_height_m,
                                     cfg.obstacle_margin_m, 0.85*cfg.max_speed_mps)
    cmd = cfg.field_weight*macro + cfg.slot_weight*slot + sep + obs
    cmd += np.array([cfg.wind_x_mps,cfg.wind_y_mps]) * -0.20  # feed-forward compensation estimate
    cmd = _clip_vectors(cmd, cfg.max_speed_mps)
    # Acceleration limiter
    dv = cmd - velocities
    max_dv = cfg.max_accel_mps2*cfg.dt
    dv = _clip_vectors(dv, max_dv)
    cmd = velocities + dv
    cmd = _clip_vectors(cmd, cfg.max_speed_mps)
    cmd = enforce_pairwise_velocity_safety(positions, cmd, cfg.min_separation_m, cfg.dt, cfg.max_speed_mps)
    ms=(time.perf_counter()-t0)*1000
    return cmd, rho, (jx,jy), min_dist, unsafe, ms


def leader_follower_commands(positions: np.ndarray, velocities: np.ndarray, assigned_slots: np.ndarray, cfg: FleetSimConfig, rects):
    t0=time.perf_counter()
    # Leader = UAV 1. It moves to its assigned target. Followers retain initial slot offsets relative to leader target frame.
    leader=0
    cmd=np.zeros_like(positions)
    cmd[leader]=0.8*(assigned_slots[leader]-positions[leader])
    leader_target=assigned_slots[leader]
    desired_offsets=assigned_slots-leader_target
    for i in range(1,len(positions)):
        desired=positions[leader]+desired_offsets[i]
        cmd[i]=0.65*(desired-positions[i])+0.25*cmd[leader]
    sep,min_dist,unsafe=separation_velocity(positions,cfg.min_separation_m,cfg.safety_gain,0.95*cfg.max_speed_mps)
    obs=obstacle_boundary_velocity(positions,rects,cfg.workspace_width_m,cfg.workspace_height_m,cfg.obstacle_margin_m,0.85*cfg.max_speed_mps)
    cmd=cmd+sep+obs
    cmd=_clip_vectors(cmd,cfg.max_speed_mps)
    dv=_clip_vectors(cmd-velocities,cfg.max_accel_mps2*cfg.dt)
    cmd=_clip_vectors(velocities+dv,cfg.max_speed_mps)
    cmd=enforce_pairwise_velocity_safety(positions,cmd,cfg.min_separation_m,cfg.dt,cfg.max_speed_mps)
    return cmd,min_dist,unsafe,(time.perf_counter()-t0)*1000


def _prevent_rect_crossing(pos: np.ndarray, new: np.ndarray, rects, width, height, margin=0.15):
    out=new.copy()
    out[:,0]=np.clip(out[:,0],margin,width-margin)
    out[:,1]=np.clip(out[:,1],margin,height-margin)
    for i,p in enumerate(out):
        for r in rects:
            if _point_in_rect(p,r,margin):
                # Reject the step if it enters a forbidden rectangle.
                out[i]=pos[i]
                break
    return out


def _target_occupancy(positions: np.ndarray, targets: list[TargetPoint]) -> dict[str,int]:
    result={t.label:0 for t in targets}
    if not len(positions): return result
    for p in positions:
        d=[np.linalg.norm(p-np.array([t.x_m,t.y_m]))/max(t.radius_m,0.25) for t in targets]
        idx=int(np.argmin(d)); t=targets[idx]
        if np.linalg.norm(p-np.array([t.x_m,t.y_m])) <= max(t.radius_m*1.25,1.5):
            result[t.label]+=1
    return result


def run_fleet_simulation(cfg: FleetSimConfig, targets: list[TargetPoint], model=None, device="cpu") -> FleetSimulationResult:
    if not (1 <= int(cfg.drone_count) <= 100):
        raise ValueError("В этой версии симулятора число БПЛА должно быть от 1 до 100.")
    targets=validate_targets(targets,cfg.drone_count,cfg.workspace_width_m,cfg.workspace_height_m)
    rects=obstacle_rectangles(cfg.obstacle_preset,cfg.workspace_width_m,cfg.workspace_height_m)
    obstacle=obstacle_mask(cfg.grid_size,cfg.obstacle_preset)
    positions=initial_positions(cfg.drone_count,cfg.workspace_width_m,cfg.workspace_height_m,cfg.initial_preset,rects,cfg.seed,max(0.6,cfg.min_separation_m*0.55))
    initial=positions.copy()
    slots,slot_labels=build_target_slots(targets,cfg.workspace_width_m,cfg.workspace_height_m,rects)
    assignment=assign_drones_to_slots(positions,slots)
    assigned_slots=slots[assignment]
    target_density=target_density_from_points(cfg.grid_size,targets,cfg.workspace_width_m,cfg.workspace_height_m,obstacle)

    rng=np.random.default_rng(cfg.seed+100)
    velocities=np.zeros_like(positions)
    alt=np.zeros(cfg.drone_count,float)
    takeoff_steps=max(1,int(math.ceil(cfg.flight_altitude_m/max(cfg.climb_rate_mps,0.1)/cfg.dt)))
    landing_steps=max(1,int(math.ceil(cfg.flight_altitude_m/max(cfg.descent_rate_mps,0.1)/cfg.dt))) if cfg.landing else 0
    total=takeoff_steps+cfg.planar_steps+landing_steps
    trajectories=np.zeros((total+1,cfg.drone_count,2),float); trajectories[0]=positions
    altitude_history=np.zeros((total+1,cfg.drone_count),float)
    rows=[]; cmds=[]
    current_density=positions_to_density(positions,cfg.grid_size,cfg.workspace_width_m,cfg.workspace_height_m,obstacle)

    def add_row(step,phase,compute_ms,min_dist,unsafe,cmd):
        nonlocal current_density
        err=np.linalg.norm(positions-assigned_slots,axis=1)
        dist_mae=float(np.mean(np.abs(current_density-target_density)))
        rows.append({
            "step":step,"phase":phase,"mean_target_error_m":float(err.mean()),
            "p95_target_error_m":float(np.quantile(err,.95)),"max_target_error_m":float(err.max()),
            "distribution_mae":dist_mae,"distribution_mae_x1e4":dist_mae*1e4,
            "min_separation_m":float(min_dist),"unsafe_pairs":int(unsafe),
            "mean_speed_mps":float(np.linalg.norm(cmd,axis=1).mean()),
            "max_speed_mps":float(np.linalg.norm(cmd,axis=1).max()),"compute_ms":float(compute_ms),
            "mean_altitude_m":float(alt.mean()),
        })

    add_row(0,"GROUND",999.0,999.0,0,np.zeros_like(positions))
    step=0
    # Vertical takeoff: horizontal coordinates are intentionally unchanged. This makes the 2D assumption explicit.
    for k in range(takeoff_steps):
        step+=1
        alt=np.minimum(cfg.flight_altitude_m,alt+cfg.climb_rate_mps*cfg.dt)
        trajectories[step]=positions; altitude_history[step]=alt
        add_row(step,"TAKEOFF",0.0,999.0,0,np.zeros_like(positions))
        for i in range(cfg.drone_count):
            cmds.append({"step":step,"phase":"TAKEOFF","drone_id":i+1,"x_m":positions[i,0],"y_m":positions[i,1],"z_m":alt[i],"vx_mps":0.0,"vy_mps":0.0,"vz_mps":cfg.climb_rate_mps,"target_x_m":assigned_slots[i,0],"target_y_m":assigned_slots[i,1]})

    last_min=999.0; last_unsafe=0
    for k in range(cfg.planar_steps):
        step+=1
        if cfg.guidance_mode == "Leader–Follower":
            cmd,last_min,last_unsafe,ms=leader_follower_commands(positions,velocities,assigned_slots,cfg,rects)
            current_density=positions_to_density(positions,cfg.grid_size,cfg.workspace_width_m,cfg.workspace_height_m,obstacle)
        else:
            cmd,current_density,_,last_min,last_unsafe,ms=density_control_commands(
                positions,velocities,assigned_slots,target_density,obstacle,rects,cfg,model,device)
        # environmental drift acts on the plant, not on the controller
        plant_v=cmd+np.array([cfg.wind_x_mps,cfg.wind_y_mps])
        new=positions+plant_v*cfg.dt
        if cfg.position_noise_m>0:
            new+=rng.normal(0.0,cfg.position_noise_m,size=new.shape)
        new=_prevent_rect_crossing(positions,new,rects,cfg.workspace_width_m,cfg.workspace_height_m)
        velocities=(new-positions)/cfg.dt
        positions=new
        current_density=positions_to_density(positions,cfg.grid_size,cfg.workspace_width_m,cfg.workspace_height_m,obstacle)
        trajectories[step]=positions; altitude_history[step]=alt
        add_row(step,"PLANAR",ms,last_min,last_unsafe,cmd)
        for i in range(cfg.drone_count):
            cmds.append({"step":step,"phase":"PLANAR","drone_id":i+1,"x_m":positions[i,0],"y_m":positions[i,1],"z_m":alt[i],"vx_mps":cmd[i,0],"vy_mps":cmd[i,1],"vz_mps":0.0,"target_x_m":assigned_slots[i,0],"target_y_m":assigned_slots[i,1]})

    if cfg.landing:
        velocities[:]=0
        for k in range(landing_steps):
            step+=1
            alt=np.maximum(0.0,alt-cfg.descent_rate_mps*cfg.dt)
            trajectories[step]=positions; altitude_history[step]=alt
            add_row(step,"LANDING",0.0,last_min,last_unsafe,np.zeros_like(positions))
            for i in range(cfg.drone_count):
                cmds.append({"step":step,"phase":"LANDING","drone_id":i+1,"x_m":positions[i,0],"y_m":positions[i,1],"z_m":alt[i],"vx_mps":0.0,"vy_mps":0.0,"vz_mps":-cfg.descent_rate_mps,"target_x_m":assigned_slots[i,0],"target_y_m":assigned_slots[i,1]})

    final_density=positions_to_density(positions,cfg.grid_size,cfg.workspace_width_m,cfg.workspace_height_m,obstacle)
    hist=pd.DataFrame(rows)
    cmd_df=pd.DataFrame(cmds)
    occupancy=_target_occupancy(positions,targets)
    planar=hist[hist.phase=="PLANAR"]
    final_err=np.linalg.norm(positions-assigned_slots,axis=1)
    summary={
        "Число БПЛА":int(cfg.drone_count),
        "Рабочая высота, м":float(cfg.flight_altitude_m),
        "Средняя ошибка до назначенной позиции, м":float(final_err.mean()),
        "95-й процентиль ошибки, м":float(np.quantile(final_err,.95)),
        "Итоговая MAE распределения":float(np.mean(np.abs(final_density-target_density))),
        "Минимальная дистанция за горизонтальный этап, м":float(planar.min_separation_m.min()) if len(planar) else 999.0,
        "Максимум опасных пар за шаг":int(planar.unsafe_pairs.max()) if len(planar) else 0,
        "Среднее время расчёта шага, мс":float(planar.compute_ms.mean()) if len(planar) else 0.0,
        "Максимальное время расчёта шага, мс":float(planar.compute_ms.max()) if len(planar) else 0.0,
        "Доля БПЛА ближе 2 м к назначенной позиции, %":float(100*np.mean(final_err<=2.0)),
        "Распределение по целям (факт)":occupancy,
        "Распределение по целям (задано)":{t.label:int(t.count) for t in targets},
    }
    return FleetSimulationResult(
        cfg,targets,slots,assignment,initial,positions.copy(),alt.copy(),trajectories[:step+1],altitude_history[:step+1],
        hist,cmd_df,target_density,final_density,obstacle,rects,summary
    )


def config_to_dict(cfg: FleetSimConfig):
    return asdict(cfg)
