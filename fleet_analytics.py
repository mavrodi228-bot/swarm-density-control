from __future__ import annotations

from pathlib import Path
import json
import time
import zipfile
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from fleet_core import FleetSimulationResult, FleetSimConfig, TargetPoint, run_fleet_simulation


def mission_map_figure(result: FleetSimulationResult, title="Траектории и целевое распределение"):
    cfg=result.config
    fig,ax=plt.subplots(figsize=(8.0,5.8))
    for x0,y0,x1,y1 in result.obstacle_rects:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,alpha=.35,hatch="//",label="Препятствие" if not any(p.get_label()=="Препятствие" for p in ax.patches) else None))
    for t in result.targets:
        ax.add_patch(Circle((t.x_m,t.y_m),t.radius_m,fill=False,linestyle="--",linewidth=1.5))
        ax.text(t.x_m,t.y_m+max(1.2,t.radius_m*1.05),f"{t.label}: {t.count}",ha="center",fontsize=9)
    ax.scatter(result.target_slots[:,0],result.target_slots[:,1],marker="x",s=35,label="Целевые места")
    n=len(result.final_positions)
    stride=max(1,n//20)
    for i in range(0,n,stride):
        tr=result.trajectories[:,i,:]
        ax.plot(tr[:,0],tr[:,1],linewidth=1,alpha=.45)
    ax.scatter(result.initial_positions[:,0],result.initial_positions[:,1],s=26,marker="o",label="Старт")
    ax.scatter(result.final_positions[:,0],result.final_positions[:,1],s=34,marker="^",label="Финиш")
    for i,(x,y) in enumerate(result.final_positions):
        if n<=20: ax.text(x+0.35,y+0.35,str(i+1),fontsize=7)
    ax.set_xlim(0,cfg.workspace_width_m); ax.set_ylim(0,cfg.workspace_height_m)
    ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("X (восток), м"); ax.set_ylabel("Y (север), м"); ax.set_title(title)
    ax.grid(alpha=.2); ax.legend(loc="best",fontsize=8)
    fig.tight_layout(); return fig


def density_pair_figure(result: FleetSimulationResult):
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.4),sharex=True,sharey=True)
    for ax,rho,title in zip(axs,[result.target_density,result.final_density],["Заданное распределение","Полученное распределение"]):
        im=ax.imshow(rho,origin="lower",extent=[0,result.config.workspace_width_m,0,result.config.workspace_height_m],aspect="auto")
        for x0,y0,x1,y1 in result.obstacle_rects:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,alpha=.45,hatch="//"))
        ax.set_title(title); ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.tight_layout(); return fig


def mission_metrics_figure(result: FleetSimulationResult):
    h=result.history[result.history.phase=="PLANAR"].copy()
    if h.empty: h=result.history.copy()
    h["planar_step"]=np.arange(len(h))
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    ax.plot(h.planar_step,h.mean_target_error_m,label="Средняя ошибка назначения, м",linewidth=2)
    # Put density MAE on a comparable relative scale for readability.
    if h.distribution_mae.iloc[0]>0:
        rel=100*h.distribution_mae/h.distribution_mae.iloc[0]
        ax.plot(h.planar_step,rel,label="MAE плотности, % от начальной",linewidth=2)
    ax.set_xlabel("Шаг горизонтального управления"); ax.set_ylabel("Ошибка")
    ax.set_title("Сходимость к заданному распределению")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def safety_figure(result: FleetSimulationResult):
    h=result.history[result.history.phase=="PLANAR"].copy()
    h["planar_step"]=np.arange(len(h))
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.plot(h.planar_step,h.min_separation_m,label="Минимальная дистанция",linewidth=2)
    ax.axhline(result.config.min_separation_m,linestyle="--",label="Заданный минимум")
    ax.set_xlabel("Шаг горизонтального управления"); ax.set_ylabel("Дистанция, м")
    ax.set_title("Контроль межбортовой дистанции")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def altitude_phase_figure(result: FleetSimulationResult):
    z=result.altitude_history.mean(axis=1)
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    ax.plot(np.arange(len(z)),z,linewidth=2)
    ax.axhline(result.config.flight_altitude_m,linestyle="--",label="Рабочая высота")
    ax.set_xlabel("Шаг моделирования"); ax.set_ylabel("Средняя высота, м")
    ax.set_title("Трёхэтапная логика: взлёт → движение в плоскости → посадка")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def command_figure(result: FleetSimulationResult, drone_id=1):
    c=result.commands[(result.commands.drone_id==drone_id)&(result.commands.phase=="PLANAR")].copy()
    if c.empty: c=result.commands[result.commands.drone_id==drone_id].copy()
    c["k"]=np.arange(len(c))
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    ax.plot(c.k,c.vx_mps,label="$v_x$",linewidth=1.8)
    ax.plot(c.k,c.vy_mps,label="$v_y$",linewidth=1.8)
    ax.set_xlabel("Шаг"); ax.set_ylabel("Команда скорости, м/с")
    ax.set_title(f"Команды горизонтальной скорости для БПЛА {drone_id}")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def target_count_table(result: FleetSimulationResult) -> pd.DataFrame:
    planned=result.summary["Распределение по целям (задано)"]
    actual=result.summary["Распределение по целям (факт)"]
    return pd.DataFrame([{"Цель":k,"Задано":planned.get(k,0),"Фактически в области":actual.get(k,0)} for k in planned])


def benchmark_small_fleet(base_cfg: FleetSimConfig, targets_factory, model=None, counts=(5,10,20,50,100), repeats=2, steps=80):
    rows=[]
    for n in counts:
        vals=[]
        for r in range(repeats):
            cfg=FleetSimConfig(**{**base_cfg.__dict__,"drone_count":int(n),"planar_steps":int(steps),"landing":False,"seed":base_cfg.seed+r})
            targets=targets_factory(n,cfg.workspace_width_m,cfg.workspace_height_m)
            t0=time.perf_counter(); res=run_fleet_simulation(cfg,targets,model=model); total_ms=(time.perf_counter()-t0)*1000
            planar=res.history[res.history.phase=="PLANAR"]
            vals.append((float(planar.compute_ms.mean()),total_ms,float(res.summary["Средняя ошибка до назначенной позиции, м"]),float(res.summary["Минимальная дистанция за горизонтальный этап, м"])))
        arr=np.array(vals)
        rows.append({"drones":n,"controller_step_ms":float(arr[:,0].mean()),"experiment_total_ms":float(arr[:,1].mean()),"final_error_m":float(arr[:,2].mean()),"min_separation_m":float(arr[:,3].mean())})
    return pd.DataFrame(rows)


def scalability_figure(df: pd.DataFrame):
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    ax.plot(df.drones,df.controller_step_ms,marker="o",linewidth=2,label="Расчёт команд на шаг")
    ax.set_xlabel("Количество БПЛА"); ax.set_ylabel("Время, мс")
    ax.set_title("Масштабируемость практического контроллера (1–100 БПЛА)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def noise_robustness(base_cfg: FleetSimConfig, targets, model=None, noises=(0.0,0.03,0.08,0.15)):
    rows=[]
    for q in noises:
        cfg=FleetSimConfig(**{**base_cfg.__dict__,"position_noise_m":float(q),"landing":False})
        r=run_fleet_simulation(cfg,targets,model=model)
        rows.append({"position_noise_m":q,"final_error_m":r.summary["Средняя ошибка до назначенной позиции, м"],"density_mae":r.summary["Итоговая MAE распределения"],"min_separation_m":r.summary["Минимальная дистанция за горизонтальный этап, м"]})
    return pd.DataFrame(rows)


def noise_figure(df: pd.DataFrame):
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.plot(df.position_noise_m,df.final_error_m,marker="o",linewidth=2)
    ax.set_xlabel("СКО случайной ошибки положения за шаг, м"); ax.set_ylabel("Итоговая ошибка назначения, м")
    ax.set_title("Устойчивость к ошибкам положения")
    ax.grid(alpha=.25); fig.tight_layout(); return fig


def save_fleet_export(result: FleetSimulationResult, out_path: str|Path, extra_files: dict[str,bytes|str]|None=None):
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_path,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("history.csv",result.history.to_csv(index=False))
        z.writestr("commands.csv",result.commands.to_csv(index=False))
        z.writestr("summary.json",json.dumps(result.summary,ensure_ascii=False,indent=2))
        z.writestr("targets.json",json.dumps([t.__dict__ for t in result.targets],ensure_ascii=False,indent=2))
        z.writestr("config.json",json.dumps(result.config.__dict__,ensure_ascii=False,indent=2))
        buf=io.BytesIO(); np.savez_compressed(buf,initial_positions=result.initial_positions,final_positions=result.final_positions,target_slots=result.target_slots,trajectories=result.trajectories,target_density=result.target_density,final_density=result.final_density)
        z.writestr("simulation_arrays.npz",buf.getvalue())
        for name,data in (extra_files or {}).items():
            z.writestr(name,data)
    return str(out_path)
