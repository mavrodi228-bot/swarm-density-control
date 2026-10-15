from __future__ import annotations
from pathlib import Path
import io, json, time, zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swarm_core import sample_agents, expert_flux
from model import predict_flux


def density_figure(rho, obstacle=None, title="Плотность роя", agents=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(rho, origin="lower", extent=[0,1,0,1], interpolation="bilinear")
    if obstacle is not None and obstacle.any():
        masked = np.ma.masked_where(~obstacle, obstacle)
        ax.imshow(masked, origin="lower", extent=[0,1,0,1], alpha=.60, cmap="gray_r", vmin=0, vmax=1)
    if agents is not None and len(agents):
        ax.scatter(agents[:,0], agents[:,1], s=4, alpha=.30)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Нормированная плотность")
    fig.tight_layout()
    return fig


def flux_figure(rho, jx, jy, obstacle=None, title="Управляющее поле потока"):
    n = rho.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(rho, origin="lower", extent=[0,1,0,1], interpolation="bilinear", alpha=.80)
    step = max(2, n//16)
    yy, xx = np.mgrid[0:n:step, 0:n:step]
    x = xx/(n-1); y = yy/(n-1)
    u = jx[::step,::step]; v = jy[::step,::step]
    mag = np.sqrt(u*u+v*v)
    denom = np.quantile(mag, .90)+1e-12
    ax.quiver(x,y,u/denom,v/denom,angles="xy",scale_units="xy",scale=14,width=.004)
    if obstacle is not None and obstacle.any():
        masked = np.ma.masked_where(~obstacle, obstacle)
        ax.imshow(masked, origin="lower", extent=[0,1,0,1], alpha=.60, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def metrics_figure(history: pd.DataFrame, title="Качество управления по шагам"):
    fig, ax = plt.subplots(figsize=(7.0,4.5))
    ax.plot(history["step"], history["mae_x1e4"], label="MAE × 10⁴", linewidth=2)
    ax.plot(history["step"], history["tv_distance"], label="TV-distance", linewidth=2)
    ax.plot(history["step"], history["com_distance"], label="Расстояние центров", linewidth=2)
    ax.set_xlabel("Шаг моделирования")
    ax.set_ylabel("Метрика")
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig


def mass_figure(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.0,4.0))
    ax.plot(history["step"], history["mass_error"], label="Ошибка сохранения массы")
    ax.plot(history["step"], history["obstacle_mass"], label="Масса внутри препятствий")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Абсолютная ошибка")
    ax.set_title("Численная устойчивость и физические ограничения")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig


def correlation_matrix(history: pd.DataFrame):
    cols = ["mae_x1e4","com_distance","entropy","mean_flux","max_flux","compute_ms"]
    df = history[cols].copy()
    # Retain only varying columns to avoid meaningless NaN blocks.
    keep = [c for c in cols if float(df[c].std()) > 1e-12]
    corr = df[keep].corr()
    return corr


def correlation_figure(history: pd.DataFrame):
    corr = correlation_matrix(history)
    fig, ax = plt.subplots(figsize=(6.6,5.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
    ax.set_title("Корреляционная матрица показателей")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Pearson r")
    fig.tight_layout()
    return fig, corr


def prediction_scatter_figure(ex, ey, px, py):
    e = np.concatenate([ex.ravel(), ey.ravel()])
    p = np.concatenate([px.ravel(), py.ravel()])
    mask = np.abs(e) > np.quantile(np.abs(e), .10)
    e,p = e[mask],p[mask]
    if len(e) > 7000:
        idx = np.linspace(0,len(e)-1,7000).astype(int)
        e,p=e[idx],p[idx]
    mae = float(np.mean(np.abs(e-p))) if len(e) else 0.0
    corr = float(np.corrcoef(e,p)[0,1]) if len(e)>2 and np.std(e)>0 and np.std(p)>0 else 0.0
    fig, ax = plt.subplots(figsize=(6.3,5.0))
    ax.scatter(e,p,s=8,alpha=.28)
    lo = min(e.min(initial=-1e-5),p.min(initial=-1e-5)); hi=max(e.max(initial=1e-5),p.max(initial=1e-5))
    ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1.5)
    ax.set_xlabel("Эталонный поток")
    ax.set_ylabel("Прогноз нейросети")
    ax.set_title(f"Нейросеть vs математический эталон\nMAE={mae:.3e}; r={corr:.3f}")
    ax.grid(alpha=.25)
    fig.tight_layout()
    return fig, {"flux_mae": mae, "flux_corr": corr}


def benchmark_controller(model, rho, target, obstacle, device="cpu", counts=(100,1000,10000,100000,1000000), repeats=4):
    rows=[]
    for n_agents in counts:
        t_density=[]; t_control=[]
        # Estimate density from a materialized sample, then run fixed-grid controller.
        for r in range(repeats):
            t0=time.perf_counter()
            pts=sample_agents(rho, n_agents, seed=100+r)
            # bin back to the fixed grid
            bins=rho.shape[0]
            hist,_,_=np.histogram2d(pts[:,1],pts[:,0],bins=bins,range=[[0,1],[0,1]])
            hist=hist/hist.sum()
            t_density.append((time.perf_counter()-t0)*1000)
            t1=time.perf_counter()
            _=predict_flux(model,hist,target,obstacle,device=device)
            t_control.append((time.perf_counter()-t1)*1000)
        rows.append({"agents":int(n_agents),"density_estimation_ms":float(np.median(t_density)),"field_control_ms":float(np.median(t_control))})
    return pd.DataFrame(rows)


def benchmark_figure(df):
    fig, ax=plt.subplots(figsize=(7.0,4.5))
    ax.plot(df["agents"],df["density_estimation_ms"],marker="o",label="Оценка плотности из агентов")
    ax.plot(df["agents"],df["field_control_ms"],marker="o",label="Нейросетевой field-control")
    ax.set_xscale("log")
    ax.set_xlabel("Число виртуальных агентов")
    ax.set_ylabel("Время, мс")
    ax.set_title("Масштабируемость: агенты → фиксированная сетка плотности")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig


def training_figure(history_json):
    p=Path(history_json)
    if not p.exists():
        return None
    h=pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    fig,ax=plt.subplots(figsize=(6.6,4.0))
    ax.plot(h["epoch"],h["loss"],marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.set_title("Обучение нейросетевого аппроксиматора потока")
    ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def make_export_zip(result, corr_df, benchmark_df, extra_summary, out_path):
    out_path=Path(out_path)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_path,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("metrics.csv", result.history.to_csv(index=False))
        z.writestr("correlation_matrix.csv", corr_df.to_csv())
        z.writestr("scalability.csv", benchmark_df.to_csv(index=False))
        z.writestr("summary.json", json.dumps(extra_summary,ensure_ascii=False,indent=2))
        buf=io.BytesIO()
        np.savez_compressed(buf,initial=result.initial,target=result.target,final=result.final,obstacle=result.obstacle.astype(np.uint8),flux_x=result.last_flux_x,flux_y=result.last_flux_y)
        z.writestr("fields.npz",buf.getvalue())
    return str(out_path)
