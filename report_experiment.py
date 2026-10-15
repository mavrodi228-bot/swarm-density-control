from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swarm_core import SimConfig, run_simulation, sample_agents
from model import load_model
from analytics import (
    density_figure, flux_figure, correlation_figure, prediction_scatter_figure,
    benchmark_controller, benchmark_figure, training_figure
)

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; OUT=ROOT/'outputs'
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
model=load_model(ROOT/'model_data'/'swarm_unet.pt')

base=dict(grid_size=48,steps=100,dt=.16,diffusion=.0008,noise=0.0,agent_count=10000,seed=42,
          initial_preset='Слева',target_preset='Кольцо',obstacle_preset='Центральный блок')
results={}
for ctrl in ['Математический эталон','Нейросеть','Гибрид 70/30']:
    cfg=SimConfig(controller=ctrl,**base)
    results[ctrl]=run_simulation(cfg,model=model,frame_stride=10)

# Save main fields using hybrid result
r=results['Гибрид 70/30']
ag0=sample_agents(r.initial,10000,42,render_cap=2200)
agf=sample_agents(r.final,10000,43,render_cap=2200)
for name,fig in [
    ('01_initial_density.png',density_figure(r.initial,r.obstacle,'Начальное распределение роя',ag0)),
    ('02_target_density.png',density_figure(r.target,r.obstacle,'Целевое распределение')),
    ('03_final_density.png',density_figure(r.final,r.obstacle,'Результат гибридного управления',agf)),
    ('04_control_flux.png',flux_figure(r.final,r.last_flux_x,r.last_flux_y,r.obstacle,'Управляющее поле потока J')),
]:
    fig.savefig(FIG/name,dpi=180,bbox_inches='tight'); plt.close(fig)

# Comparative MAE
fig,ax=plt.subplots(figsize=(7.4,4.6))
for ctrl,res in results.items():
    ax.plot(res.history.step,res.history.mae_x1e4,label=ctrl,linewidth=2)
ax.set_xlabel('Шаг моделирования'); ax.set_ylabel('MAE × 10⁴'); ax.set_title('Сходимость к целевому распределению')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'05_mae_comparison.png',dpi=180); plt.close(fig)

# TV distance
fig,ax=plt.subplots(figsize=(7.4,4.6))
for ctrl,res in results.items():
    ax.plot(res.history.step,res.history.tv_distance,label=ctrl,linewidth=2)
ax.set_xlabel('Шаг моделирования'); ax.set_ylabel('TV-distance'); ax.set_title('Остаточная разница распределений')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'06_tv_comparison.png',dpi=180); plt.close(fig)

# Mass conservation
fig,ax=plt.subplots(figsize=(7.4,4.2))
h=r.history
ax.plot(h.step,h.mass_error,label='Ошибка массы до нормировки')
ax.plot(h.step,h.obstacle_mass,label='Масса внутри препятствий')
ax.set_yscale('symlog',linthresh=1e-12); ax.set_xlabel('Шаг'); ax.set_ylabel('Абсолютная ошибка')
ax.set_title('Контроль физических ограничений'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'07_mass_constraints.png',dpi=180); plt.close(fig)

# Correlation
fig,corr=correlation_figure(r.history); fig.savefig(FIG/'08_correlation_matrix.png',dpi=180,bbox_inches='tight'); plt.close(fig)
corr.to_csv(OUT/'correlation_matrix.csv')

# NN vs expert
fig,pstats=prediction_scatter_figure(r.expert_flux_x,r.expert_flux_y,r.predicted_flux_x,r.predicted_flux_y)
fig.savefig(FIG/'09_nn_vs_expert.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Benchmark
bench=benchmark_controller(model,r.initial,r.target,r.obstacle,counts=(100,1000,10000,100000,1000000),repeats=3)
bench.to_csv(OUT/'scalability.csv',index=False)
fig=benchmark_figure(bench); fig.savefig(FIG/'10_scalability.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Training history
fig=training_figure(ROOT/'model_data'/'training_history.json')
if fig:
    fig.savefig(FIG/'11_training_loss.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Flux error map
exmag=np.sqrt(r.expert_flux_x**2+r.expert_flux_y**2)
pmag=np.sqrt(r.predicted_flux_x**2+r.predicted_flux_y**2)
err=np.sqrt((r.expert_flux_x-r.predicted_flux_x)**2+(r.expert_flux_y-r.predicted_flux_y)**2)
fig,ax=plt.subplots(figsize=(6.3,5.0)); im=ax.imshow(err,origin='lower',extent=[0,1,0,1]); ax.set_title('Пространственная ошибка нейросетевого поля'); ax.set_xlabel('x'); ax.set_ylabel('y'); fig.colorbar(im,ax=ax,label='|J_NN - J_ref|'); fig.tight_layout(); fig.savefig(FIG/'12_flux_error_map.png',dpi=180); plt.close(fig)

# Architecture schematic
fig,ax=plt.subplots(figsize=(10,4.2)); ax.axis('off')
boxes=[(.04,.40,.17,.26,'Текущее поле\nρ(x,y,t)'),(.28,.40,.18,.26,'U-Net\n3 канала → 2'),(.53,.40,.18,.26,'Поток управления\nJ=(Jx,Jy)'),(.78,.40,.18,.26,'Следующий шаг\nρ(x,y,t+Δt)')]
for x,y,w,hh,txt in boxes:
    rect=plt.Rectangle((x,y),w,hh,fill=False,linewidth=2); ax.add_patch(rect); ax.text(x+w/2,y+hh/2,txt,ha='center',va='center',fontsize=12)
for a,b in [(.21,.28),(.46,.53),(.71,.78)]: ax.annotate('',xy=(b,.53),xytext=(a,.53),arrowprops=dict(arrowstyle='->',lw=2))
ax.text(.13,.20,'+ целевая плотность ρ*\n+ карта препятствий',ha='center',va='center',fontsize=11)
ax.annotate('',xy=(.34,.40),xytext=(.16,.28),arrowprops=dict(arrowstyle='->',lw=1.6))
ax.text(.62,.15,'Консервативное обновление:\nизменение плотности определяется потоком\nчерез границы соседних ячеек',ha='center',va='center',fontsize=11)
ax.set_title('Архитектура программно-математической модели',fontsize=15)
fig.tight_layout(); fig.savefig(FIG/'00_architecture.png',dpi=180,bbox_inches='tight'); plt.close(fig)


# Multi-scenario robustness
scenario_specs=[
    ('S1: слева→кольцо','Слева','Кольцо','Центральный блок',42),
    ('S2: два→два','Два кластера','Два центра','Коридор',11),
    ('S3: снизу→полоса','Снизу','Полоса','Стена с проходом',23),
    ('S4: случайно→круг','Случайно','Круг справа','Два блока',77),
    ('S5: слева→центр','Слева','Центр','Нет',5),
]
robust_rows=[]
for label,ip,tp,op,seed in scenario_specs:
    cfg=SimConfig(grid_size=48,steps=90,dt=.16,diffusion=.0008,noise=0.0,controller='Гибрид 70/30',agent_count=10000,seed=seed,initial_preset=ip,target_preset=tp,obstacle_preset=op)
    rr=run_simulation(cfg,model=model,frame_stride=20)
    h=rr.history
    robust_rows.append({'scenario':label,'initial_mae':h.mae.iloc[0],'final_mae':h.mae.iloc[-1],'mae_reduction_pct':100*(1-h.mae.iloc[-1]/h.mae.iloc[0]),'final_com_distance':h.com_distance.iloc[-1]})
robust=pd.DataFrame(robust_rows); robust.to_csv(OUT/'scenario_robustness.csv',index=False)
fig,ax=plt.subplots(figsize=(8.0,4.8)); ax.bar(robust['scenario'],robust['mae_reduction_pct']); ax.set_ylabel('Снижение MAE, %'); ax.set_title('Проверка гибридного контроллера на разных сценариях'); ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/'13_scenario_robustness.png',dpi=180); plt.close(fig)

# Disturbance test
noise_rows=[]
for noise_level in [0.0,0.002,0.005,0.01]:
    cfg=SimConfig(controller='Гибрид 70/30',noise=noise_level,**{k:v for k,v in base.items() if k!='noise'})
    rr=run_simulation(cfg,model=model,frame_stride=20)
    h=rr.history
    noise_rows.append({'noise':noise_level,'final_mae':h.mae.iloc[-1],'reduction_pct':100*(1-h.mae.iloc[-1]/h.mae.iloc[0])})
noise_df=pd.DataFrame(noise_rows); noise_df.to_csv(OUT/'noise_robustness.csv',index=False)
fig,ax=plt.subplots(figsize=(6.8,4.3)); ax.plot(noise_df['noise'],noise_df['final_mae']*1e4,marker='o',linewidth=2); ax.set_xlabel('Амплитуда случайного возмущения'); ax.set_ylabel('Итоговая MAE × 10⁴'); ax.set_title('Устойчивость к случайным возмущениям'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/'14_noise_robustness.png',dpi=180); plt.close(fig)

# Summary table
rows=[]
for ctrl,res in results.items():
    h=res.history
    rows.append({
        'controller':ctrl,
        'initial_mae':h.mae.iloc[0],
        'final_mae':h.mae.iloc[-1],
        'mae_reduction_pct':100*(1-h.mae.iloc[-1]/h.mae.iloc[0]),
        'final_tv':h.tv_distance.iloc[-1],
        'final_com_distance':h.com_distance.iloc[-1],
        'mean_step_ms':h.compute_ms.iloc[1:].mean(),
    })
summary=pd.DataFrame(rows); summary.to_csv(OUT/'controller_comparison.csv',index=False)
r.history.to_csv(OUT/'hybrid_metrics.csv',index=False)
meta={'flux_mae':pstats['flux_mae'],'flux_corr':pstats['flux_corr'],'summary':rows,'benchmark':bench.to_dict(orient='records')}
(OUT/'experiment_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print('\nNN flux stats',pstats)
print('\nBenchmark\n',bench.to_string(index=False))
