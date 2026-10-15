from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.fft import dctn, idctn


EPS = 1e-12


@dataclass
class SimConfig:
    grid_size: int = 48
    steps: int = 80
    dt: float = 0.16
    diffusion: float = 0.0008
    noise: float = 0.0
    controller: str = "Гибрид 70/30"
    agent_count: int = 10000
    seed: int = 42
    initial_preset: str = "Слева"
    target_preset: str = "Кольцо"
    obstacle_preset: str = "Центральный блок"


@dataclass
class SimulationResult:
    initial: np.ndarray
    target: np.ndarray
    obstacle: np.ndarray
    final: np.ndarray
    last_flux_x: np.ndarray
    last_flux_y: np.ndarray
    history: pd.DataFrame
    density_frames: list[np.ndarray]
    expert_flux_x: np.ndarray
    expert_flux_y: np.ndarray
    predicted_flux_x: np.ndarray
    predicted_flux_y: np.ndarray
    config: SimConfig


def make_grid(n: int):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    return np.meshgrid(x, y)


def _normalise_density(rho: np.ndarray, obstacle: np.ndarray | None = None) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64).copy()
    rho[~np.isfinite(rho)] = 0.0
    rho = np.clip(rho, 0.0, None)
    if obstacle is not None:
        rho[np.asarray(obstacle, dtype=bool)] = 0.0
    s = float(rho.sum())
    if s <= EPS:
        if obstacle is None:
            rho[:] = 1.0
        else:
            free = ~np.asarray(obstacle, dtype=bool)
            rho[free] = 1.0
        s = float(rho.sum())
    return rho / s


def gaussian_density(n: int, centers, sigmas, weights=None, obstacle=None):
    X, Y = make_grid(n)
    if weights is None:
        weights = np.ones(len(centers))
    rho = np.zeros((n, n), dtype=np.float64)
    for (cx, cy), sigma, w in zip(centers, sigmas, weights):
        rho += float(w) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * float(sigma) ** 2))
    return _normalise_density(rho, obstacle)


def ring_density(n: int, cx=0.72, cy=0.52, radius=0.18, width=0.05, obstacle=None):
    X, Y = make_grid(n)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rho = np.exp(-((r - radius) ** 2) / (2.0 * width ** 2))
    return _normalise_density(rho, obstacle)


def stripe_density(n: int, x0=0.72, width=0.09, obstacle=None):
    X, _ = make_grid(n)
    rho = np.exp(-((X - x0) ** 2) / (2.0 * width ** 2))
    return _normalise_density(rho, obstacle)


def make_obstacle(n: int, preset: str) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    if preset == "Нет":
        return m
    if preset == "Центральный блок":
        m[int(0.34*n):int(0.67*n), int(0.46*n):int(0.57*n)] = True
    elif preset == "Коридор":
        m[int(0.12*n):int(0.43*n), int(0.44*n):int(0.54*n)] = True
        m[int(0.58*n):int(0.90*n), int(0.44*n):int(0.54*n)] = True
    elif preset == "Два блока":
        m[int(0.18*n):int(0.42*n), int(0.42*n):int(0.54*n)] = True
        m[int(0.60*n):int(0.82*n), int(0.52*n):int(0.64*n)] = True
    elif preset == "Стена с проходом":
        c0, c1 = int(0.47*n), int(0.53*n)
        m[:, c0:c1] = True
        m[int(0.42*n):int(0.60*n), c0:c1] = False
    else:
        raise ValueError(f"Unknown obstacle preset: {preset}")
    return m


def make_initial(n: int, preset: str, obstacle: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    if preset == "Слева":
        return gaussian_density(n, [(0.20, 0.50)], [0.10], obstacle=obstacle)
    if preset == "Снизу":
        return gaussian_density(n, [(0.50, 0.20)], [0.11], obstacle=obstacle)
    if preset == "Два кластера":
        return gaussian_density(n, [(0.20, 0.28), (0.23, 0.72)], [0.08, 0.08], [1, 1], obstacle)
    if preset == "Случайно":
        centers = [(rng.uniform(0.10, 0.38), rng.uniform(0.15, 0.85)) for _ in range(3)]
        sigmas = [rng.uniform(0.05, 0.11) for _ in centers]
        return gaussian_density(n, centers, sigmas, obstacle=obstacle)
    raise ValueError(f"Unknown initial preset: {preset}")


def make_target(n: int, preset: str, obstacle: np.ndarray, seed: int = 42):
    if preset == "Круг справа":
        return gaussian_density(n, [(0.78, 0.50)], [0.11], obstacle=obstacle)
    if preset == "Кольцо":
        return ring_density(n, 0.76, 0.50, 0.17, 0.045, obstacle)
    if preset == "Два центра":
        return gaussian_density(n, [(0.75, 0.28), (0.75, 0.72)], [0.075, 0.075], [1, 1], obstacle)
    if preset == "Полоса":
        return stripe_density(n, 0.76, 0.07, obstacle)
    if preset == "Центр":
        return gaussian_density(n, [(0.50, 0.50)], [0.12], obstacle=obstacle)
    raise ValueError(f"Unknown target preset: {preset}")


def solve_poisson_neumann(rhs: np.ndarray, dx: float) -> np.ndarray:
    """Solve Δφ = rhs with homogeneous Neumann-like spectral discretisation.

    The zero mode is fixed to zero. rhs must have approximately zero mean.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    n, m = rhs.shape
    f = dctn(rhs, type=2, norm="ortho")
    ky = np.arange(n)
    kx = np.arange(m)
    lam_y = 2.0 * (np.cos(np.pi * ky / n) - 1.0) / (dx * dx)
    lam_x = 2.0 * (np.cos(np.pi * kx / m) - 1.0) / (dx * dx)
    lam = lam_y[:, None] + lam_x[None, :]
    phi_hat = np.zeros_like(f)
    mask = np.abs(lam) > 1e-14
    phi_hat[mask] = f[mask] / lam[mask]
    phi_hat[0, 0] = 0.0
    return idctn(phi_hat, type=2, norm="ortho")


def expert_flux(rho: np.ndarray, target: np.ndarray, obstacle: np.ndarray):
    n = rho.shape[0]
    dx = 1.0 / (n - 1)
    error = rho - target
    error = error - error.mean()
    phi = solve_poisson_neumann(error, dx)
    gy, gx = np.gradient(phi, dx, edge_order=2)
    jx, jy = gx, gy
    jx = jx.astype(np.float64)
    jy = jy.astype(np.float64)
    jx[obstacle] = 0.0
    jy[obstacle] = 0.0
    # Gentle limiter to avoid rare spectral spikes around masked obstacles.
    mag = np.sqrt(jx*jx + jy*jy)
    lim = max(np.quantile(mag, 0.995) * 2.0, 1e-8)
    scale = np.minimum(1.0, lim / (mag + EPS))
    return jx * scale, jy * scale


def _face_flux_and_divergence(jx: np.ndarray, jy: np.ndarray, obstacle: np.ndarray):
    n, m = jx.shape
    dx = 1.0 / (m - 1)
    dy = 1.0 / (n - 1)

    fx = np.zeros((n, m + 1), dtype=np.float64)
    fy = np.zeros((n + 1, m), dtype=np.float64)

    fx[:, 1:m] = 0.5 * (jx[:, :-1] + jx[:, 1:])
    fy[1:n, :] = 0.5 * (jy[:-1, :] + jy[1:, :])

    # No-flow across obstacle boundaries.
    block_x = obstacle[:, :-1] | obstacle[:, 1:]
    block_y = obstacle[:-1, :] | obstacle[1:, :]
    fx[:, 1:m][block_x] = 0.0
    fy[1:n, :][block_y] = 0.0

    div = (fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy
    return fx, fy, div


def diffusion_flux(rho: np.ndarray, diffusion: float, obstacle: np.ndarray):
    if diffusion <= 0:
        z = np.zeros_like(rho)
        return z, z
    n = rho.shape[0]
    dx = 1.0 / (n - 1)
    gy, gx = np.gradient(rho, dx, edge_order=2)
    jx = -diffusion * gx
    jy = -diffusion * gy
    jx[obstacle] = 0.0
    jy[obstacle] = 0.0
    return jx, jy


def step_density(rho, control_jx, control_jy, obstacle, dt, diffusion=0.0, noise=0.0, rng=None):
    djx, djy = diffusion_flux(rho, diffusion, obstacle)
    jx = control_jx + djx
    jy = control_jy + djy
    _, _, div = _face_flux_and_divergence(jx, jy, obstacle)
    raw = rho - dt * div
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng()
        raw = raw + rng.normal(0.0, noise / rho.size, size=rho.shape)
    raw[obstacle] = 0.0
    mass_before_clip = float(raw.sum())
    raw = np.clip(raw, 0.0, None)
    rho_new = _normalise_density(raw, obstacle)
    return rho_new, jx, jy, abs(mass_before_clip - 1.0)


def center_of_mass(rho: np.ndarray):
    n = rho.shape[0]
    X, Y = make_grid(n)
    s = max(float(rho.sum()), EPS)
    return float((X*rho).sum()/s), float((Y*rho).sum()/s)


def entropy(rho: np.ndarray):
    p = np.clip(rho.ravel(), EPS, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def compute_metrics(rho, target, obstacle, jx, jy, step_idx, mass_error, elapsed_ms):
    err = rho - target
    mae = float(np.mean(np.abs(err)))
    tv = float(0.5 * np.sum(np.abs(err)))
    cx, cy = center_of_mass(rho)
    tx, ty = center_of_mass(target)
    com_dist = float(math.hypot(cx - tx, cy - ty))
    flux_mag = np.sqrt(jx*jx + jy*jy)
    obstacle_mass = float(rho[obstacle].sum()) if obstacle.any() else 0.0
    return {
        "step": int(step_idx),
        "mae": mae,
        "mae_x1e4": mae * 1e4,
        "tv_distance": tv,
        "com_distance": com_dist,
        "entropy": entropy(rho),
        "mean_flux": float(flux_mag.mean()),
        "max_flux": float(flux_mag.max()),
        "mass_error": float(mass_error),
        "obstacle_mass": obstacle_mass,
        "compute_ms": float(elapsed_ms),
    }


def sample_agents(rho: np.ndarray, count: int, seed: int = 42, render_cap: int | None = None):
    rng = np.random.default_rng(seed)
    n = rho.shape[0]
    draw_n = min(int(count), int(render_cap)) if render_cap else int(count)
    idx = rng.choice(n*n, size=draw_n, replace=True, p=rho.ravel())
    iy, ix = np.divmod(idx, n)
    jitter = rng.uniform(-0.45, 0.45, size=(draw_n, 2))
    x = np.clip((ix + 0.5 + jitter[:, 0]) / n, 0, 1)
    y = np.clip((iy + 0.5 + jitter[:, 1]) / n, 0, 1)
    return np.column_stack([x, y])


def run_simulation(config: SimConfig, model=None, device="cpu", frame_stride=5) -> SimulationResult:
    n = int(config.grid_size)
    obstacle = make_obstacle(n, config.obstacle_preset)
    rho = make_initial(n, config.initial_preset, obstacle, config.seed)
    target = make_target(n, config.target_preset, obstacle, config.seed)
    initial = rho.copy()
    rng = np.random.default_rng(config.seed)

    frames = [rho.copy()]
    rows = []
    last_jx = np.zeros_like(rho)
    last_jy = np.zeros_like(rho)

    # Store the first expert/predicted flux comparison for diagnostics.
    efx, efy = expert_flux(rho, target, obstacle)
    pfx, pfy = efx.copy(), efy.copy()
    if model is not None:
        from model import predict_flux
        pfx, pfy = predict_flux(model, rho, target, obstacle, device=device)

    for k in range(config.steps + 1):
        if k == 0:
            rows.append(compute_metrics(rho, target, obstacle, last_jx, last_jy, k, 0.0, 0.0))
            continue
        t0 = time.perf_counter()
        ex, ey = expert_flux(rho, target, obstacle)
        if model is not None and config.controller != "Математический эталон":
            from model import predict_flux
            nx, ny = predict_flux(model, rho, target, obstacle, device=device)
            if config.controller == "Нейросеть":
                jx, jy = nx, ny
            else:  # robust hybrid for live demonstration
                jx = 0.70 * nx + 0.30 * ex
                jy = 0.70 * ny + 0.30 * ey
        else:
            jx, jy = ex, ey

        rho, last_jx, last_jy, mass_err = step_density(
            rho, jx, jy, obstacle, config.dt, config.diffusion, config.noise, rng
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(compute_metrics(rho, target, obstacle, last_jx, last_jy, k, mass_err, elapsed_ms))
        if k % max(1, frame_stride) == 0 or k == config.steps:
            frames.append(rho.copy())

    hist = pd.DataFrame(rows)
    return SimulationResult(
        initial=initial,
        target=target,
        obstacle=obstacle,
        final=rho,
        last_flux_x=last_jx,
        last_flux_y=last_jy,
        history=hist,
        density_frames=frames,
        expert_flux_x=efx,
        expert_flux_y=efy,
        predicted_flux_x=pfx,
        predicted_flux_y=pfy,
        config=config,
    )


def save_result_bundle(result: SimulationResult, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(
        out / "fields.npz",
        initial=result.initial,
        target=result.target,
        obstacle=result.obstacle.astype(np.uint8),
        final=result.final,
        flux_x=result.last_flux_x,
        flux_y=result.last_flux_y,
    )
    (out / "config.json").write_text(json.dumps(asdict(result.config), ensure_ascii=False, indent=2), encoding="utf-8")
    return out
