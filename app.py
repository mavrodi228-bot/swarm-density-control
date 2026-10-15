from __future__ import annotations

from pathlib import Path
import json
import logging
import os

import gradio as gr
import numpy as np
import pandas as pd

from swarm_core import SimConfig, run_simulation, sample_agents
from model import load_model
from analytics import (
    density_figure, flux_figure, metrics_figure, mass_figure,
    correlation_figure, prediction_scatter_figure, benchmark_controller,
    benchmark_figure, training_figure, make_export_zip,
)
from fleet_core import FleetSimConfig, TargetPoint, targets_from_dataframe, run_fleet_simulation
from fleet_analytics import (
    mission_map_figure, density_pair_figure, mission_metrics_figure, safety_figure,
    altitude_phase_figure, command_figure, target_count_table, save_fleet_export,
)
from mavlink_bridge import MavlinkFleetManager
from real_mission import RealMissionController
from mission_scenarios import COLUMNS as SCENARIO_COLUMNS, default_tasks, tasks_from_dataframe, simulate_tasks, map_figure, save_simulation_package

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "model_data" / "swarm_unet.pt"
TRAIN_HIST = ROOT / "model_data" / "training_history.json"
DEVICE = "cpu"
MODEL = load_model(WEIGHTS, device=DEVICE)
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename=ROOT / "logs" / "operations.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LAST_DENSITY = {"result": None, "corr": None, "bench": None, "summary": None}
LAST_FLEET = {"result": None}
LAST_SCENARIO = {"result": None, "tasks": None}
FLEET_LINK = MavlinkFleetManager(rate_hz=10.0)
REAL_MISSION = RealMissionController(FLEET_LINK, model=MODEL, device=DEVICE)

TARGET_COLUMNS = ["Точка", "X, м", "Y, м", "БПЛА", "Радиус, м"]
CONNECTION_COLUMNS = ["Имя", "Endpoint", "Использовать"]


def default_targets(n=10, width=80.0, height=60.0):
    n=int(n); width=float(width); height=float(height)
    if n <= 1:
        rows=[["A",0.76*width,0.50*height,1,2.0]]
    else:
        rows=[
            ["A",0.76*width,0.25*height,1,2.0],
            ["B",0.76*width,0.72*height,n-1,min(10.0,max(4.0,0.12*min(width,height)))],
        ]
    return pd.DataFrame(rows,columns=TARGET_COLUMNS)


def update_target_template(n,width,height):
    return default_targets(n,width,height), f"Шаблон обновлён: 1 БПЛА в точке A и {max(0,int(n)-1)} в области B."


def run_fleet_ui(n,width,height,alt,initial,obstacle,controller,guidance,targets_df,steps,dt,max_speed,max_accel,min_sep,noise,wind_x,wind_y,seed,continue_tasks):
    try:
        cfg=FleetSimConfig(
            drone_count=int(n),workspace_width_m=float(width),workspace_height_m=float(height),
            flight_altitude_m=float(alt),initial_preset=initial,obstacle_preset=obstacle,
            controller=controller,guidance_mode=guidance,planar_steps=int(steps),dt=float(dt),
            max_speed_mps=float(max_speed),max_accel_mps2=float(max_accel),min_separation_m=float(min_sep),
            position_noise_m=float(noise),wind_x_mps=float(wind_x),wind_y_mps=float(wind_y),seed=int(seed),
            landing=not bool(continue_tasks),
        )
        targets=targets_from_dataframe(targets_df,cfg.drone_count,cfg.workspace_width_m,cfg.workspace_height_m)
        result=run_fleet_simulation(cfg,targets,model=MODEL,device=DEVICE)
        LAST_FLEET["result"]=result
        LAST_SCENARIO.update(result=None,tasks=None)
        logging.info("simulation n=%s controller=%s guidance=%s summary=%s",n,controller,guidance,result.summary)
        figs=[mission_map_figure(result),density_pair_figure(result),mission_metrics_figure(result),safety_figure(result),altitude_phase_figure(result),command_figure(result,1)]
        summary_df=pd.DataFrame([{"Показатель":k,"Значение":json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v} for k,v in result.summary.items()])
        count_df=target_count_table(result)
        tail=result.history.tail(15)
        status=(f"Готово: {cfg.drone_count} БПЛА. Средняя ошибка назначения {result.summary['Средняя ошибка до назначенной позиции, м']:.2f} м; "
                f"минимальная дистанция {result.summary['Минимальная дистанция за горизонтальный этап, м']:.2f} м; "
                f"средний расчёт шага {result.summary['Среднее время расчёта шага, мс']:.2f} мс.")
        return (*figs,summary_df,count_df,tail,status)
    except Exception as exc:
        logging.exception("simulation failed")
        return (None,None,None,None,None,None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),f"Ошибка: {exc}")


def export_fleet_ui():
    r=LAST_FLEET["result"]
    if r is None: return None,"Сначала запустите миссию."
    p=ROOT/"outputs"/"fleet_mission_export.zip"
    save_fleet_export(r,p)
    return str(p),"Экспорт готов: история, команды каждому БПЛА, конфигурация, цели и массивы траекторий."


# ---------- Post-arrival civilian mission scenarios (simulation only) ----------
def tasks_template_ui():
    r=LAST_FLEET["result"]
    if r is None:
        return pd.DataFrame(columns=SCENARIO_COLUMNS), "Сначала запустите первый этап полёта."
    return default_tasks(r.targets), "Таблица задач создана по целевым точкам последней миссии."


def run_tasks_ui(task_df,link_loss_enabled,link_loss_time,failsafe,background):
    try:
        r=LAST_FLEET["result"]
        if r is None: raise RuntimeError("Сначала выполните 1-й этап во вкладке «Симулятор миссии».")
        if r.config.landing: raise RuntimeError("В первой вкладке включите продолжение задачами (без LANDING) и повторите симуляцию.")
        tasks=tasks_from_dataframe(task_df,[t.label for t in r.targets])
        scenario=simulate_tasks(r,tasks,link_loss_s=float(link_loss_time) if link_loss_enabled else None,failsafe=failsafe)
        LAST_SCENARIO.update(result=scenario,tasks=tasks)
        fig=map_figure(r,scenario,background)
        summary=pd.DataFrame([{"Показатель":k,"Значение":v} for k,v in scenario.summary.items()])
        return fig,scenario.coverage,scenario.events.tail(80),summary,("Сценарий рассчитан. Карта — фоновое изображение без привязки к GPS. "
                 "Потеря связи и задача сенсоров только моделируются; физический БПЛА этим модулем не управляется.")
    except Exception as exc:
        logging.exception("scenario error")
        return None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),f"Ошибка сценария: {exc}"


def export_tasks_ui():
    r=LAST_FLEET["result"]; s=LAST_SCENARIO["result"]
    if r is None or s is None: return None,"Сначала смоделируйте вторую фазу миссии."
    p=save_simulation_package(r,LAST_SCENARIO["tasks"],s,ROOT/"outputs"/"offline_research_simulation")
    return str(p),"JSON для научного моделирования, не для загрузки в автопилот."


# ---------- Legacy density laboratory ----------
def run_density_ui(initial_preset,target_preset,obstacle_preset,controller,agent_count,steps,dt,diffusion,noise,seed):
    cfg=SimConfig(grid_size=48,steps=int(steps),dt=float(dt),diffusion=float(diffusion),noise=float(noise),controller=controller,
                  agent_count=int(agent_count),seed=int(seed),initial_preset=initial_preset,target_preset=target_preset,obstacle_preset=obstacle_preset)
    result=run_simulation(cfg,model=MODEL,device=DEVICE,frame_stride=max(1,int(steps)//12))
    agents0=sample_agents(result.initial,int(agent_count),seed=int(seed),render_cap=2500)
    agentsf=sample_agents(result.final,int(agent_count),seed=int(seed)+1,render_cap=2500)
    f_init=density_figure(result.initial,result.obstacle,"Начальная плотность",agents0)
    f_target=density_figure(result.target,result.obstacle,"Целевая плотность")
    f_final=density_figure(result.final,result.obstacle,"Итоговая плотность",agentsf)
    f_flux=flux_figure(result.final,result.last_flux_x,result.last_flux_y,result.obstacle)
    f_metrics=metrics_figure(result.history); f_mass=mass_figure(result.history)
    f_corr,corr=correlation_figure(result.history)
    f_scatter,pred_stats=prediction_scatter_figure(result.expert_flux_x,result.expert_flux_y,result.predicted_flux_x,result.predicted_flux_y)
    bench=benchmark_controller(MODEL,result.initial,result.target,result.obstacle,device=DEVICE,counts=(100,1000,10000,100000),repeats=2)
    f_bench=benchmark_figure(bench)
    h=result.history
    summary={"MAE начальное":float(h.iloc[0].mae),"MAE итоговое":float(h.iloc[-1].mae),"Снижение MAE, %":float(100*(1-h.iloc[-1].mae/max(h.iloc[0].mae,1e-15))),
             "TV-distance итог":float(h.iloc[-1].tv_distance),"Среднее время шага, мс":float(h.iloc[1:].compute_ms.mean()) if len(h)>1 else 0,**pred_stats}
    LAST_DENSITY.update({"result":result,"corr":corr,"bench":bench,"summary":summary})
    return f_init,f_target,f_final,f_flux,f_metrics,f_mass,f_corr,f_scatter,f_bench,pd.DataFrame({"Показатель":list(summary.keys()),"Значение":list(summary.values())}),f"Готово. MAE снизилась на {summary['Снижение MAE, %']:.1f}%."


def export_density_ui():
    if LAST_DENSITY["result"] is None: return None,"Сначала запустите плотностную модель."
    p=ROOT/"outputs"/"density_experiment_export.zip"
    make_export_zip(LAST_DENSITY["result"],LAST_DENSITY["corr"],LAST_DENSITY["bench"],LAST_DENSITY["summary"],p)
    return str(p),"Экспорт плотностного эксперимента готов."


# ---------- MAVLink / real fleet ----------
def _parse_connections(df):
    if not isinstance(df,pd.DataFrame): df=pd.DataFrame(df,columns=CONNECTION_COLUMNS)
    entries=[]
    for _,row in df.iterrows():
        vals=list(row.values)
        if len(vals)<3: continue
        name=str(vals[0]).strip(); endpoint=str(vals[1]).strip(); enabled=vals[2]
        enabled = bool(enabled) and str(enabled).lower() not in {"false","0","nan","none",""}
        if enabled and name and endpoint: entries.append((name,endpoint))
    return entries


def connect_fleet_ui(mode,connections,confirmed):
    try:
        entries=_parse_connections(connections)
        if not entries: raise ValueError("Не задано ни одного активного endpoint.")
        rows=FLEET_LINK.connect(entries,mode=mode,operator_confirmed=bool(confirmed),timeout=5.0)
        logging.info("fleet connected mode=%s endpoints=%s",mode,entries)
        msg=f"Подключено: {len(rows)}. Режим: {mode}."
        if mode=="REAL" and not FLEET_LINK.real_enabled:
            msg += " Команды полёта пока заблокированы: SWARM_ENABLE_REAL=1 не установлен."
        return pd.DataFrame(rows),msg
    except Exception as exc:
        logging.exception("fleet connection failed")
        return pd.DataFrame(),f"Ошибка подключения: {exc}"


def refresh_fleet_ui():
    try:
        return pd.DataFrame(FLEET_LINK.telemetry_rows()), f"Связь обновлена. {REAL_MISSION.status}. {FLEET_LINK.last_error}".strip()
    except Exception as exc: return pd.DataFrame(),f"Ошибка: {exc}"


def configure_real_ui():
    try:
        r=LAST_FLEET["result"]
        if r is None: raise RuntimeError("Сначала запустите и проверьте миссию на вкладке «Симулятор миссии».")
        REAL_MISSION.configure(r.config,r.targets)
        return f"Миссия загружена для {r.config.drone_count} БПЛА. Цели: {r.summary['Распределение по целям (задано)']}."
    except Exception as exc: return f"Ошибка: {exc}"


def arm_real_ui():
    try: FLEET_LINK.arm_all(); logging.info("ARM ALL"); return "ARM ALL отправлено."
    except Exception as exc: return f"Команда заблокирована/ошибка: {exc}"


def takeoff_real_ui():
    try:
        r=LAST_FLEET["result"]
        if r is None: raise RuntimeError("Нет последней миссии")
        FLEET_LINK.takeoff_all(r.config.flight_altitude_m); logging.info("TAKEOFF ALL alt=%s",r.config.flight_altitude_m)
        return f"TAKEOFF отправлено: {r.config.flight_altitude_m:.1f} м. После выхода на высоту запустите распределение."
    except Exception as exc: return f"Команда заблокирована/ошибка: {exc}"


def start_real_ui():
    try: REAL_MISSION.start(); logging.info("REAL DISTRIBUTION START"); return "Контур управления запущен: телеметрия → плотность → поле → индивидуальные velocity setpoints."
    except Exception as exc: return f"Не удалось запустить: {exc}"


def hold_real_ui():
    try: REAL_MISSION.hold(); logging.info("HOLD ALL"); return "HOLD: горизонтальные команды обнулены, внешний контур остановлен."
    except Exception as exc: return f"Ошибка: {exc}"


def land_real_ui():
    try: REAL_MISSION.stop(); FLEET_LINK.land_all(); logging.info("LAND ALL"); return "LAND ALL отправлено."
    except Exception as exc: return f"Команда заблокирована/ошибка: {exc}"


def disconnect_real_ui():
    try: REAL_MISSION.stop(); FLEET_LINK.disconnect(); logging.info("DISCONNECT"); return pd.DataFrame(),"Соединения закрыты."
    except Exception as exc: return pd.DataFrame(),f"Ошибка: {exc}"


CSS="""
.gradio-container {max-width: 1550px !important; margin: auto;}
.hero {padding:18px 22px;border:1px solid rgba(120,120,120,.25);border-radius:18px;margin-bottom:14px;}
.warn {padding:12px;border-left:4px solid #d97706;background:rgba(217,119,6,.08);}
"""

with gr.Blocks(title="Swarm Density Control GCS") as demo:
    gr.Markdown("""
<div class='hero'>
<h1>Swarm Density Control GCS</h1>
<p><b>Симулятор и наземная станция верхнеуровневого управления группой 1–100 мультироторных БПЛА.</b><br>
Оператор задаёт рабочую высоту и распределение по точкам, плотностный контроллер строит коллективное поле, а нижний слой переводит его в индивидуальные команды скорости каждому аппарату.</p>
</div>
""")

    with gr.Tab("1. Симулятор миссии"):
        with gr.Row():
            with gr.Column(scale=1):
                n=gr.Slider(1,100,10,step=1,label="Количество БПЛА")
                with gr.Row():
                    width=gr.Number(80.0,label="Ширина области, м")
                    height=gr.Number(60.0,label="Высота области, м")
                alt=gr.Number(20.0,label="Рабочая высота полёта, м")
                initial=gr.Dropdown(["Линия слева","Сетка слева","Кластер слева","Случайно"],value="Линия слева",label="Начальное расположение")
                obstacle=gr.Dropdown(["Нет","Центральный блок","Коридор","Два блока","Стена с проходом"],value="Центральный блок",label="Препятствия на горизонтальной плоскости")
                controller=gr.Radio(["Гибрид 70/30","Нейросеть","Математический эталон"],value="Гибрид 70/30",label="Плотностный контроллер")
                guidance=gr.Radio(["Density control","Leader–Follower"],value="Density control",label="Способ группового управления")
                targets=gr.Dataframe(value=default_targets(),headers=TARGET_COLUMNS,datatype=["str","number","number","number","number"],row_count=(2,"dynamic"),column_count=(5,"fixed"),interactive=True,label="Целевое распределение")
                template_btn=gr.Button("Сделать шаблон 1 / (N−1)")
                template_status=gr.Markdown()
                template_btn.click(update_target_template,inputs=[n,width,height],outputs=[targets,template_status])
            with gr.Column(scale=1):
                steps=gr.Slider(40,500,300,step=10,label="Шагов горизонтального управления")
                dt=gr.Slider(0.08,0.5,0.20,step=0.01,label="Шаг времени, с")
                max_speed=gr.Slider(0.5,10.0,4.0,step=0.1,label="Максимальная горизонтальная скорость, м/с")
                max_accel=gr.Slider(0.5,8.0,3.0,step=0.1,label="Максимальное ускорение, м/с²")
                min_sep=gr.Slider(0.5,10.0,2.5,step=0.1,label="Минимальная межбортовая дистанция, м")
                noise=gr.Slider(0.0,0.5,0.0,step=0.01,label="Случайная ошибка положения за шаг, м")
                wind_x=gr.Slider(-3.0,3.0,0.0,step=0.1,label="Ветер по X, м/с")
                wind_y=gr.Slider(-3.0,3.0,0.0,step=0.1,label="Ветер по Y, м/с")
                seed=gr.Number(42,precision=0,label="Seed")
                continue_tasks=gr.Checkbox(True,label="Продолжить задачами в зонах (не выполнять посадку после распределения)")
                run_fleet=gr.Button("Выполнить распределение и перейти к сценарию",variant="primary")
                fleet_status=gr.Markdown("Готов к моделированию.")
        with gr.Row():
            p_map=gr.Plot(label="Миссия и траектории")
            p_density=gr.Plot(label="Плотность")
        with gr.Row():
            p_conv=gr.Plot(label="Сходимость")
            p_safe=gr.Plot(label="Безопасная дистанция")
        with gr.Row():
            p_alt=gr.Plot(label="Высота и этапы")
            p_cmd=gr.Plot(label="Команды БПЛА 1")
        with gr.Row():
            fleet_summary=gr.Dataframe(interactive=False,label="Итоговые показатели")
            target_counts=gr.Dataframe(interactive=False,label="Распределение по целям")
        fleet_tail=gr.Dataframe(interactive=False,label="Последние шаги")
        run_fleet.click(run_fleet_ui,inputs=[n,width,height,alt,initial,obstacle,controller,guidance,targets,steps,dt,max_speed,max_accel,min_sep,noise,wind_x,wind_y,seed,continue_tasks],outputs=[p_map,p_density,p_conv,p_safe,p_alt,p_cmd,fleet_summary,target_counts,fleet_tail,fleet_status])

    with gr.Tab("2. Сценарии после прибытия"):
        gr.Markdown("""### От координаты — к гражданской задаче
Выполните распределение на вкладке **«Симулятор миссии»**. После этого каждой точке назначается действие:
**наблюдение, площадь съёмки, виртуальный обход периметра или ожидание**. Ведётся журнал событий и модельная оценка охвата.

**Ограничение:** площадная съёмка и обход периметра имитируют выполнение сенсорной задачи; модель не формирует реальных маршрутов и не подтверждает получение изображений.
Потеря связи приводит только к модельной записи запроса HOLD / RETURN / LAND; команд реальному аппарату эта вкладка не посылает.
""")
        with gr.Row():
            with gr.Column(scale=1):
                task_df=gr.Dataframe(headers=SCENARIO_COLUMNS,datatype=["str","str","number","number"],row_count=(2,"dynamic"),column_count=(4,"fixed"),interactive=True,label="Задачи для точек")
                fill_tasks=gr.Button("Загрузить точки последней миссии")
                map_file=gr.File(label="Фоновая карта PNG/JPG/WEBP (без геопривязки)",file_types=["image"],type="filepath")
                loss_enabled=gr.Checkbox(False,label="Имитировать потерю связи")
                link_loss_time=gr.Number(15.0,label="Секунда потери связи от начала задачи")
                failsafe_task=gr.Radio(["HOLD","RETURN_REQUEST","LAND_REQUEST"],value="HOLD",label="Предпочтение при отказе (симуляция)")
                run_task=gr.Button("Моделировать действия после прибытия",variant="primary")
                tasks_status=gr.Markdown()
                export_task=gr.Button("Экспорт исследовательского сценария JSON")
                export_task_file=gr.File(label="Файл модели")
            with gr.Column(scale=2):
                scenario_map=gr.Plot(label="Схематическая карта зоны и расположения")
                scenario_summary=gr.Dataframe(interactive=False,label="Результаты")
        with gr.Row():
            scenario_coverage=gr.Dataframe(interactive=False,label="Виртуальное выполнение задач")
            scenario_events=gr.Dataframe(interactive=False,label="События")
        fill_tasks.click(tasks_template_ui,outputs=[task_df,tasks_status])
        run_task.click(run_tasks_ui,inputs=[task_df,loss_enabled,link_loss_time,failsafe_task,map_file],outputs=[scenario_map,scenario_coverage,scenario_events,scenario_summary,tasks_status])
        export_task.click(export_tasks_ui,outputs=[export_task_file,tasks_status])

    with gr.Tab("3. Реальные БПЛА / SITL"):
        gr.Markdown("""
<div class='warn'><b>Логика реального полёта:</b> ARM → TAKEOFF до общей рабочей высоты → внешний горизонтальный контроллер → HOLD → LAND.
Исследуемая модель управляет только движением в горизонтальной плоскости. Для реального режима запись команд дополнительно защищена переменной <code>SWARM_ENABLE_REAL=1</code> и подтверждением оператора. Сначала проверяйте миссию в симуляторе/SITL.</div>
""")
        with gr.Row():
            with gr.Column(scale=1):
                link_mode=gr.Radio(["SITL","REAL"],value="SITL",label="Режим подключения")
                connections=gr.Dataframe(value=pd.DataFrame([["UAV-01","udp:127.0.0.1:14550",True]],columns=CONNECTION_COLUMNS),headers=CONNECTION_COLUMNS,datatype=["str","str","bool"],row_count=(1,"dynamic"),column_count=(3,"fixed"),interactive=True,label="MAVLink endpoints")
                confirm=gr.Checkbox(False,label="Подтверждаю: зона полёта свободна, аппараты проверены, ручной перехват доступен")
                connect_btn=gr.Button("Подключить телеметрию")
                configure_btn=gr.Button("Загрузить последнюю проверенную миссию")
                with gr.Row():
                    arm_btn=gr.Button("ARM ALL")
                    takeoff_btn=gr.Button("TAKEOFF ALL")
                start_btn=gr.Button("START DISTRIBUTION",variant="primary")
                with gr.Row():
                    hold_btn=gr.Button("HOLD",variant="stop")
                    land_btn=gr.Button("LAND ALL")
                disconnect_btn=gr.Button("Отключить")
                link_status=gr.Markdown("Нет подключений.")
            with gr.Column(scale=2):
                telemetry=gr.Dataframe(interactive=False,label="Телеметрия флота")
                refresh_btn=gr.Button("Обновить телеметрию")
                gr.Markdown("""
**Что происходит при START DISTRIBUTION:** координаты каждого аппарата переводятся в общую локальную плоскость; из них строится текущая плотность; контроллер вычисляет поле движения; поле интерполируется в координатах каждого БПЛА; добавляются назначение на конкретную целевую позицию, межбортовое разведение и обход препятствий; затем каждому автопилоту передаётся собственная команда горизонтальной скорости через MAVLink.
""")
        connect_btn.click(connect_fleet_ui,inputs=[link_mode,connections,confirm],outputs=[telemetry,link_status])
        refresh_btn.click(refresh_fleet_ui,outputs=[telemetry,link_status])
        configure_btn.click(configure_real_ui,outputs=link_status)
        arm_btn.click(arm_real_ui,outputs=link_status)
        takeoff_btn.click(takeoff_real_ui,outputs=link_status)
        start_btn.click(start_real_ui,outputs=link_status)
        hold_btn.click(hold_real_ui,outputs=link_status)
        land_btn.click(land_real_ui,outputs=link_status)
        disconnect_btn.click(disconnect_real_ui,outputs=[telemetry,link_status])

    with gr.Tab("4. Плотностная лаборатория"):
        gr.Markdown("Исходная научная модель: управление полем плотности без явного списка отдельных аппаратов.")
        with gr.Row():
            with gr.Column(scale=1):
                d_initial=gr.Dropdown(["Слева","Снизу","Два кластера","Случайно"],value="Слева",label="Начальная конфигурация")
                d_target=gr.Dropdown(["Круг справа","Кольцо","Два центра","Полоса","Центр"],value="Кольцо",label="Целевая конфигурация")
                d_obstacle=gr.Dropdown(["Нет","Центральный блок","Коридор","Два блока","Стена с проходом"],value="Центральный блок",label="Препятствия")
                d_controller=gr.Radio(["Нейросеть","Гибрид 70/30","Математический эталон"],value="Гибрид 70/30",label="Контроллер")
                d_agents=gr.Slider(100,100000,10000,step=100,label="Виртуальное число агентов")
                d_steps=gr.Slider(20,160,80,step=10,label="Шаги")
                d_dt=gr.Slider(0.04,0.25,0.16,step=0.01,label="Шаг времени")
                d_diff=gr.Slider(0,0.003,0.0008,step=0.0001,label="Диффузия")
                d_noise=gr.Slider(0,0.02,0,step=0.001,label="Возмущение")
                d_seed=gr.Number(42,precision=0,label="Seed")
                d_run=gr.Button("Запустить плотностный эксперимент",variant="primary")
                d_status=gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row(): d_p1=gr.Plot(); d_p2=gr.Plot()
                with gr.Row(): d_p3=gr.Plot(); d_p4=gr.Plot()
        with gr.Row(): d_p5=gr.Plot(); d_p6=gr.Plot()
        with gr.Row(): d_p7=gr.Plot(); d_p8=gr.Plot()
        d_p9=gr.Plot(); d_summary=gr.Dataframe(interactive=False)
        d_run.click(run_density_ui,inputs=[d_initial,d_target,d_obstacle,d_controller,d_agents,d_steps,d_dt,d_diff,d_noise,d_seed],outputs=[d_p1,d_p2,d_p3,d_p4,d_p5,d_p6,d_p7,d_p8,d_p9,d_summary,d_status])

    with gr.Tab("5. Экспорт и документация"):
        gr.Markdown("Экспорт позволяет сохранить воспроизводимый журнал эксперимента и индивидуальные команды для каждого виртуального БПЛА.")
        export_fleet=gr.Button("Экспорт последней миссии")
        export_fleet_file=gr.File(label="ZIP миссии")
        export_fleet_status=gr.Markdown()
        export_fleet.click(export_fleet_ui,outputs=[export_fleet_file,export_fleet_status])
        export_density=gr.Button("Экспорт плотностного эксперимента")
        export_density_file=gr.File(label="ZIP плотностной модели")
        export_density_status=gr.Markdown()
        export_density.click(export_density_ui,outputs=[export_density_file,export_density_status])
        gr.Markdown("""
### Архитектура
`Оператор → целевое распределение → плотностный контроллер → поле J/v → назначение индивидуальных целевых мест → слой безопасности → команды vx/vy → MAVLink → автопилоты`.

### Ограничение исследования
Математическая часть статьи рассматривает **горизонтальную плоскость**. Высота не входит в поле плотности: взлёт и посадка реализованы отдельными дискретными стадиями, а на основном этапе принимается `z = H = const`. Для самолётных БПЛА требуется отдельная модель с ограничениями минимальной скорости, радиуса разворота и курса; текущая реальная интеграция ориентирована на мультироторные аппараты.
""")
        train_fig=training_figure(TRAIN_HIST)
        if train_fig is not None: gr.Plot(value=train_fig,label="Обучение U-Net")

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(os.getenv("PORT", "7860")), show_error=True, css=CSS, theme=gr.themes.Soft())
