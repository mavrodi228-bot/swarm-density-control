from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from model import load_model
from fleet_core import FleetSimConfig, TargetPoint, run_fleet_simulation
from fleet_analytics import (
    mission_map_figure,density_pair_figure,mission_metrics_figure,safety_figure,
    altitude_phase_figure,command_figure,benchmark_small_fleet,scalability_figure,
    noise_robustness,noise_figure,target_count_table,
)

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; OUT=ROOT/'outputs'; FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
MODEL=load_model(ROOT/'model_data'/'swarm_unet.pt',device='cpu')


def savefig(fig,name):
    p=FIG/name; fig.savefig(p,dpi=210,bbox_inches='tight'); plt.close(fig); return p


def targets_1_9(n=10,w=80,h=60):
    if n==1: return [TargetPoint('A',0.76*w,0.5*h,1,2.0)]
    a=max(1,int(round(0.1*n)))
    if a>=n: a=1
    return [TargetPoint('A',0.76*w,0.25*h,a,max(2.0,0.045*min(w,h))),TargetPoint('B',0.76*w,0.72*h,n-a,max(4.0,0.12*min(w,h)))]


def architecture_figure():
    fig,ax=plt.subplots(figsize=(12,3.2)); ax.axis('off')
    labels=['Оператор\n(цели, высота, доли)','Плотность ρ(x,y)\nи U-Net / baseline','Поле движения\nJ(x,y), v(x,y)','Дискретизация\nна N аппаратов','Safety layer\nразведение + препятствия','MAVLink\nvx, vy каждому БПЛА']
    xs=np.linspace(.02,.84,len(labels))
    for i,(x,l) in enumerate(zip(xs,labels)):
        box=FancyBboxPatch((x,.32),.145,.36,boxstyle='round,pad=0.015',transform=ax.transAxes,fill=False,linewidth=1.5)
        ax.add_patch(box); ax.text(x+.0725,.50,l,ha='center',va='center',transform=ax.transAxes,fontsize=9)
        if i<len(labels)-1: ax.annotate('',xy=(xs[i+1],.50),xytext=(x+.145,.50),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.4))
    ax.set_title('Архитектура перехода от распределения к индивидуальным командам',fontsize=13)
    return fig


def real_architecture_figure():
    fig,ax=plt.subplots(figsize=(11.5,4.3)); ax.axis('off')
    rows=[
        (.78,'Наземная станция','Gradio UI / оператор / журнал миссии'),
        (.56,'Контур группового управления','телеметрия → плотность → поле → назначение → safety'),
        (.34,'MAVLink bridge','SET_POSITION_TARGET_LOCAL_NED, heartbeat, GPS, battery'),
        (.12,'Автопилоты БПЛА','PX4 / ArduPilot: стабилизация и низкоуровневое управление'),
    ]
    for y,title,desc in rows:
        b=FancyBboxPatch((.15,y),.70,.13,boxstyle='round,pad=0.015',transform=ax.transAxes,fill=False,linewidth=1.6)
        ax.add_patch(b); ax.text(.20,y+.065,title,va='center',transform=ax.transAxes,fontweight='bold'); ax.text(.43,y+.065,desc,va='center',transform=ax.transAxes,fontsize=9)
    for (y1,_,_),(y2,_,_) in zip(rows[:-1],rows[1:]):
        ax.annotate('',xy=(.50,y2+.13),xytext=(.50,y1),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='<->',lw=1.3))
    ax.set_title('Интеграция симулятора с реальными мультироторными БПЛА',fontsize=13)
    return fig


def comparison_figure(df):
    fig,ax=plt.subplots(figsize=(8,4.6))
    x=np.arange(len(df)); ax.bar(x,df['final_error_m'])
    ax.set_xticks(x,df['mode'],rotation=18,ha='right'); ax.set_ylabel('Средняя итоговая ошибка, м'); ax.set_title('Сравнение способов группового управления')
    ax.grid(axis='y',alpha=.25); fig.tight_layout(); return fig


def main():
    cfg=FleetSimConfig(drone_count=10,workspace_width_m=80,workspace_height_m=60,flight_altitude_m=20,
                       initial_preset='Линия слева',obstacle_preset='Центральный блок',controller='Гибрид 70/30',
                       guidance_mode='Density control',planar_steps=300,dt=.2,max_speed_mps=4,max_accel_mps2=3,
                       min_separation_m=2.5,position_noise_m=0,wind_x_mps=0,wind_y_mps=0,landing=True,seed=42)
    targets=targets_1_9(10,80,60)
    print('baseline...',flush=True); base=run_fleet_simulation(cfg,targets,MODEL); print('baseline done',flush=True)
    base.history.to_csv(OUT/'fleet_baseline_history.csv',index=False)
    base.commands.to_csv(OUT/'fleet_baseline_commands.csv',index=False)
    target_count_table(base).to_csv(OUT/'fleet_target_counts.csv',index=False)
    (OUT/'fleet_baseline_summary.json').write_text(json.dumps(base.summary,ensure_ascii=False,indent=2),encoding='utf-8')

    savefig(architecture_figure(),'20_fleet_architecture.png')
    savefig(mission_map_figure(base),'21_mission_1_9.png')
    savefig(density_pair_figure(base),'22_density_1_9.png')
    savefig(mission_metrics_figure(base),'23_convergence_1_9.png')
    savefig(safety_figure(base),'24_separation_1_9.png')
    savefig(altitude_phase_figure(base),'25_three_phase_altitude.png')
    savefig(command_figure(base,1),'26_uav1_velocity_commands.png')
    savefig(real_architecture_figure(),'30_real_mavlink_architecture.png')

    # Controller comparison on exactly the same mission.
    comp=[]
    variants=[('Density: эталон','Математический эталон','Density control'),('Density: нейросеть','Нейросеть','Density control'),('Density: гибрид','Гибрид 70/30','Density control'),('Leader–Follower','Гибрид 70/30','Leader–Follower')]
    print('comparison...',flush=True)
    for label,ctl,guidance in variants:
        print(' ',label,flush=True)
        c=FleetSimConfig(**{**cfg.__dict__,'controller':ctl,'guidance_mode':guidance,'landing':False})
        r=run_fleet_simulation(c,targets,MODEL)
        comp.append({'mode':label,'final_error_m':r.summary['Средняя ошибка до назначенной позиции, м'],'density_mae':r.summary['Итоговая MAE распределения'],'min_separation_m':r.summary['Минимальная дистанция за горизонтальный этап, м'],'compute_ms':r.summary['Среднее время расчёта шага, мс']})
    print('comparison done',flush=True)
    comp=pd.DataFrame(comp); comp.to_csv(OUT/'fleet_controller_comparison.csv',index=False)
    savefig(comparison_figure(comp),'27_controller_comparison.png')

    # Scalability is now deliberately focused on small/medium practical groups, 5..100.
    bcfg=FleetSimConfig(**{**cfg.__dict__,'landing':False,'planar_steps':30})
    print('benchmark...',flush=True)
    bench=benchmark_small_fleet(bcfg,targets_1_9,MODEL,counts=(5,10,20,50,100),repeats=1,steps=30)
    print('benchmark done',flush=True)
    bench.to_csv(OUT/'fleet_scalability_1_100.csv',index=False)
    savefig(scalability_figure(bench),'28_scalability_1_100.png')

    print('noise...',flush=True)
    noise=noise_robustness(FleetSimConfig(**{**cfg.__dict__,'landing':False,'planar_steps':300}),targets,MODEL,noises=(0,.05,.12,.25))
    print('noise done',flush=True)
    noise.to_csv(OUT/'fleet_noise_robustness.csv',index=False)
    savefig(noise_figure(noise),'29_noise_robustness_fleet.png')

    summary={'baseline':base.summary,'controller_comparison':comp.to_dict(orient='records'),'scalability':bench.to_dict(orient='records'),'noise':noise.to_dict(orient='records')}
    (OUT/'fleet_experiment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
