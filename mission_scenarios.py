"""Offline-first *research simulator* for civilian post-arrival tasks.

Only simulates perception/coverage and communication failures. This module DOES NOT
encode MAVLink commands, create uploadable autopilot missions, or control aircraft.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ACTIONS = ("Наблюдение", "Площадная съёмка", "Обход периметра", "Ожидание")
FAILSAFES = ("HOLD", "RETURN_REQUEST", "LAND_REQUEST")
COLUMNS = ["Точка", "Задача", "Длительность, с", "Интервал снимков, с"]

@dataclass(frozen=True)
class AreaTask:
    label: str
    action: str
    duration_s: float
    capture_interval_s: float

@dataclass
class ScenarioResult:
    events: pd.DataFrame
    coverage: pd.DataFrame
    frames: pd.DataFrame
    communication_loss_s: float | None
    failsafe: str
    summary: dict


def tasks_from_dataframe(data, target_labels):
    df = pd.DataFrame(data)
    if df.empty: raise ValueError("Задайте хотя бы одну задачу.")
    if all(c in df.columns for c in COLUMNS): df = df[COLUMNS]
    elif df.shape[1] >= 4: df = df.iloc[:, :4]; df.columns=COLUMNS
    else: raise ValueError("Нужны четыре столбца таблицы задач.")
    labels = set(str(t) for t in target_labels)
    tasks={}
    for _,r in df.iterrows():
        label=str(r[COLUMNS[0]]).strip()
        if not label or label.lower()=="nan": continue
        if label not in labels: raise ValueError(f"Неизвестная точка {label}.")
        if label in tasks: raise ValueError(f"Повтор задачи для точки {label}.")
        action=str(r[COLUMNS[1]]).strip()
        if action not in ACTIONS: raise ValueError(f"Неизвестный тип задачи: {action}.")
        seconds=float(r[COLUMNS[2]])
        interval=float(r[COLUMNS[3]])
        if not math.isfinite(seconds) or not (1<=seconds<=3600): raise ValueError("Длительность: 1–3600 с.")
        if not math.isfinite(interval) or not (0.2<=interval<=3600): raise ValueError("Интервал: 0.2–3600 с.")
        tasks[label]=AreaTask(label,action,seconds,interval)
    if set(tasks) != labels: raise ValueError("Каждой целевой точке должна соответствовать одна задача.")
    return tasks


def default_tasks(targets):
    return pd.DataFrame([
        [t.label, "Площадная съёмка" if i%2==0 else "Наблюдение", 30.0, 5.0]
        for i,t in enumerate(targets)
    ], columns=COLUMNS)


def simulate_tasks(mission, tasks, *, link_loss_s=None, failsafe="HOLD", sample_dt_s=1.0):
    """Simulates jobs AFTER fleet motion in the original GCS.

    Sensor 'coverage' is a virtual statistic based on abstract footprint cells.
    Positions remain at final assigned slots; the program cannot claim actual
    images, real obstacle avoidance, or a physically executable scan route.
    """
    if failsafe not in FAILSAFES: raise ValueError("Неизвестная failsafe-политика")
    if sample_dt_s<=0 or sample_dt_s>60: raise ValueError("Некорректный шаг моделирования")
    targets={t.label:t for t in mission.targets}
    if set(tasks) != set(targets): raise ValueError("Не для всех точек определены задачи")
    if link_loss_s is not None and (link_loss_s<0 or not math.isfinite(link_loss_s)):
        raise ValueError("Время потери связи должно быть неотрицательным")
    events=[]; records=[]; frames=[]
    assignment_by_slot=mission.assignment
    # slot order follows allocation by target label in build_target_slots
    target_for_drone=[]
    cursor=0; slot_to_label=[]
    for t in mission.targets: slot_to_label.extend([t.label]*t.count)
    target_for_drone=[slot_to_label[int(j)] for j in assignment_by_slot]
    completed=0; interrupted=0
    for i,label in enumerate(target_for_drone):
        t=targets[label]; task=tasks[label]
        nsteps=max(1,int(math.ceil(task.duration_s / sample_dt_s)))
        seen=set(); captures=0; active=0
        # Explicit no-position movement: scan represents simulated payload activity.
        # 64 abstract image sectors, no geographic mapping accuracy claims.
        for k in range(nsteps):
            at=min(k*sample_dt_s,task.duration_s)
            lost=(link_loss_s is not None and at>=link_loss_s)
            if lost:
                interrupted+=1
                events.append(dict(drone=i+1,target=label,kind="LINK_LOSS",time_s=at,detail=f"{failsafe}: учебный запрос; не команда автопилоту"))
                break
            active+=1
            if task.action=="Площадная съёмка":
                sector=min(63,int(at/max(task.duration_s,1e-9)*64))
                seen.add(sector)
            elif task.action=="Обход периметра":
                seen.add(min(63,int(at/max(task.duration_s,1e-9)*64)))
            elif task.action=="Наблюдение":
                seen.add(0)
            if task.action!="Ожидание" and (k==0 or (at - math.floor(at/task.capture_interval_s)*task.capture_interval_s) <sample_dt_s*0.51):
                captures+=1
                events.append(dict(drone=i+1,target=label,kind="VIRTUAL_CAPTURE",time_s=at,detail="Имитированная регистрация кадра (нет изображения с БПЛА)"))
            frames.append(dict(drone=i+1,target=label,time_s=at,phase="SIMULATED_TASK",link="ONLINE",action=task.action,coverage_proxy_pct=100*len(seen)/(64 if task.action in ACTIONS[1:3] else 1)))
        else:
            completed+=1
            events.append(dict(drone=i+1,target=label,kind="TASK_COMPLETE",time_s=task.duration_s,detail="Учебный сценарий завершён"))
        if link_loss_s is not None and link_loss_s < task.duration_s:
            frames.append(dict(drone=i+1,target=label,time_s=link_loss_s,phase="FAILSAFE_REQUEST",link="LOST",action=failsafe,coverage_proxy_pct=100*len(seen)/(64 if task.action in ACTIONS[1:3] else 1)))
        records.append(dict(drone=i+1,target=label,task=task.action,scheduled_s=task.duration_s,simulated_steps=active,virtual_captures=captures,coverage_proxy_pct=100*len(seen)/(64 if task.action in ACTIONS[1:3] else 1),status="INTERRUPTED" if link_loss_s is not None and link_loss_s<task.duration_s else "COMPLETE"))
    return ScenarioResult(pd.DataFrame(events),pd.DataFrame(records),pd.DataFrame(frames),link_loss_s,failsafe,{
        "Аппаратов в сценарии":len(target_for_drone),"Сценариев завершено":completed,"Сценариев прервано":interrupted,
        "Виртуальных кадров":int(sum(r['virtual_captures'] for r in records)),
        "Средняя модельная доля охвата, %":round(float(np.mean([r['coverage_proxy_pct'] for r in records])),2),
        "Режим":"ТОЛЬКО СИМУЛЯЦИЯ — нет команды автопилоту",
    })


def map_figure(mission, scenario=None, background=None):
    fig,ax=plt.subplots(figsize=(8.5,5.8))
    cfg=mission.config
    if background:
        p=Path(background)
        if p.stat().st_size>20*1024*1024: raise ValueError("Размер карты больше 20 МБ")
        with Image.open(p) as orig:
            if orig.format not in ("JPEG","PNG","WEBP"):
                raise ValueError("Карта должна быть PNG/JPG/WEBP")
            if orig.width*orig.height>20_000_000: raise ValueError("Слишком большое разрешение изображения")
            img=ImageOps.exif_transpose(orig).convert("RGB")
            img.thumbnail((2000,1400))
            ax.imshow(img, extent=(0,cfg.workspace_width_m,0,cfg.workspace_height_m),origin="upper",alpha=0.64,zorder=0)
    for x0,y0,x1,y1 in mission.obstacle_rects:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=True,fc="#dd6b50",alpha=0.28,ec="#c43820",lw=1.5))
    p=mission.final_positions
    ax.scatter(p[:,0],p[:,1],c="#2359d7",s=26,zorder=5)
    for t in mission.targets:
        ax.scatter(t.x_m,t.y_m,s=110,facecolors="none",edgecolors="#e08620",linewidths=2)
        ax.annotate(f"{t.label} · {t.count}", (t.x_m,t.y_m), xytext=(5,7),textcoords="offset points",fontsize=10,color="#905000")
    ax.set(xlim=(0,cfg.workspace_width_m),ylim=(0,cfg.workspace_height_m),xlabel="X, м (локальная модель)",ylabel="Y, м (локальная модель)",title="Карта задач после прибытия в зоны")
    ax.set_aspect('equal');ax.grid(alpha=.15);fig.tight_layout()
    return fig


def save_simulation_package(mission, tasks, scenario, out_dir):
    """Research-only, explicitly NOT an onboard/offline flight plan."""
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    meta={"kind":"RESEARCH_SIMULATION_NOT_FOR_FLIGHT","format_version":1,
          "disclaimer":"НЕЛЬЗЯ загружать в автопилот. Нет геопривязки, валидации связи, динамики или failsafe.",
          "workspace_m":{"width":mission.config.workspace_width_m,"height":mission.config.workspace_height_m},
          "task_definitions":[vars(t) for t in tasks.values()],"summary":scenario.summary,
          "loss_link":{"simulated_at_s":scenario.communication_loss_s,"requested_policy":scenario.failsafe}}
    (root/'scenario_simulation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    scenario.events.to_csv(root/'scenario_events.csv',index=False)
    scenario.coverage.to_csv(root/'scenario_coverage.csv',index=False)
    scenario.frames.to_csv(root/'scenario_frames.csv',index=False)
    (root/'READ_BEFORE_USE.txt').write_text('Файлы только для исследовательского моделирования. Это НЕ полётный план и НЕ команды БПЛА.\n',encoding='utf-8')
    return root/'scenario_simulation.json'
