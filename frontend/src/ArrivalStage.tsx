import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { ArrowRight, Check, Crosshair, LoaderCircle, Map, Route, ScanLine, Target, Waypoints } from "lucide-react";
import { api } from "./api";
import { TerritoryBackground, TerritoryLayers, TerritoryPanel, useTerritory } from "./TerritoryMap";

export type Arrival = {
  all_arrived: boolean; arrived_count: number; total_count: number; remaining_count: number;
  targets_frozen?: boolean;
  zones?: { index: number; x: number; y: number; radius: number; required_count: number; arrived_count: number }[];
};
export type StageMission = {
  id: string; name: string; status: string; arrival?: Arrival;
  config?: { steps: number; max_speed: number; controller: string; model_id?: string };
  stage?: { parent_mission_id?: string; index?: number };
};
type Region = { origin_x_m: number; origin_y_m: number; width_m: number; height_m: number };
type SourceAgent = { id: number; source_drone_id?: number; x_m: number; y_m: number; battery: number };
type Scenario = { id: string; label: string; description: string; recommended?: boolean; available: boolean; reason?: string };
type StageOptions = {
  allowed: boolean; reason?: string; parent_id: string; arrival?: Arrival;
  source_workspace: { width_m: number; height_m: number } | null;
  source_agents: SourceAgent[]; suggested_region: Region | null;
  scenarios: Scenario[]; next_stages?: { id: string; name: string; status: string }[];
  coordinate_note?: string;
  can_continue?: boolean; continue_reason?: string;
};
type Props = {
  mission: StageMission; online: boolean; atArrival: boolean;
  onJumpToArrival: () => void; onError: (message: string) => void;
  onCreated: (mission: StageMission) => void;
};
const icons = [Target, Route, Crosshair, ScanLine];

export function ArrivalStage({ mission, online, atArrival, onJumpToArrival, onError, onCreated }: Props) {
  const storageKey = `agmen-destination-${mission.id}`;
  const planKey = `agmen-stage-plan-${mission.id}`;
  const existingMap = useRef(localStorage.getItem(storageKey) !== null).current;
  const initialPlan = useRef((() => {
    try { return JSON.parse(localStorage.getItem(planKey) || "null"); } catch { return null; }
  })()).current;
  const [options, setOptions] = useState<StageOptions | null>(null);
  const [scenario, setScenario] = useState<string>(initialPlan?.scenario || "distribution");
  const [origin, setOrigin] = useState({ x: Number(initialPlan?.origin?.x || 0), y: Number(initialPlan?.origin?.y || 0) });
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const initialized = useRef(false);
  const editor = useTerritory(onError, online, undefined, storageKey);
  const svg = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!online || mission.status !== "completed") return;
    let live = true;
    setLoading(true);
    api<StageOptions>(`/api/missions/${mission.id}/next-stage-options`).then(value => {
      if (!live) return;
      setOptions(value);
      if (!initialized.current) {
        if (!initialPlan && value.suggested_region) setOrigin({ x: value.suggested_region.origin_x_m, y: value.suggested_region.origin_y_m });
        if (!existingMap && value.suggested_region) editor.restore(null, [], value.suggested_region);
        initialized.current = true;
      }
    }).catch(error => { if (live) onError(error.message); }).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [online, mission.id, mission.status]);
  useEffect(() => {
    if (initialized.current) localStorage.setItem(planKey, JSON.stringify({ scenario, origin }));
  }, [scenario, origin, planKey, options]);

  const registrationValid = Number.isFinite(origin.x) && Number.isFinite(origin.y);
  const agents = options?.source_agents || [];
  const outside = agents.filter(agent => agent.x_m - origin.x < 0 || agent.x_m - origin.x > editor.width || agent.y_m - origin.y < 0 || agent.y_m - origin.y > editor.height);
  const ready = online && !!options?.allowed && atArrival && registrationValid && !outside.length && !editor.busy && !busy;

  function point(event: PointerEvent<SVGSVGElement>) {
    const p = event.currentTarget.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
    const matrix = event.currentTarget.getScreenCTM();
    const local = matrix ? p.matrixTransform(matrix.inverse()) : p;
    return { x: Math.max(0, Math.min(1, (local.x - 44) / 880)), y: Math.max(0, Math.min(1, (local.y - 28) / 578)) };
  }
  async function startStage() {
    if (!ready || !options) return;
    setBusy(true);
    try {
      const next = await api<StageMission>(`/api/missions/${mission.id}/next-stage`, {
        name: `${mission.name.slice(0, 65)} · этап ${(mission.stage?.index || 1) + 1}`,
        scenario, map_id: editor.mapId, width_m: editor.width, height_m: editor.height,
        origin_x_m: origin.x, origin_y_m: origin.y, restricted_zones: editor.shapes,
        steps: mission.config?.steps || 100, max_speed: mission.config?.max_speed || 3,
        controller: mission.config?.controller || "baseline",
      });
      onCreated(next);
    } catch (error) { onError((error as Error).message); }
    finally { setBusy(false); }
  }
  async function continueArrival() {
    if (!options?.can_continue || !online || busy) return;
    setBusy(true);
    try { onCreated(await api<StageMission>(`/api/missions/${mission.id}/continue`, {})); }
    catch (error) { onError((error as Error).message); }
    finally { setBusy(false); }
  }
  const complete = atArrival && !!mission.arrival?.all_arrived;
  return <section className="panel arrival-stage" aria-label="После прибытия">
    <div className="panel-header">
      <div className="section-title"><Waypoints size={17} /> После прибытия <span className="subtle">Следующий участок</span></div>
      <button className="secondary-button small" aria-expanded={expanded} onClick={() => setExpanded(!expanded)} disabled={mission.status !== "completed"}>
        {expanded ? "Свернуть план" : "Подготовить следующий этап"} <ArrowRight size={14} />
      </button>
    </div>
    <div className="arrival-intro">
      <div className={`arrival-step ${complete ? "ready" : ""}`}><span>{complete ? <Check size={15} /> : "01"}</span><div><strong>Прибытие всей группы</strong><p>{mission.status === "failed" ? "Расчёт прерван. Исправьте ошибку и повторите запуск." : mission.status !== "completed" ? "Рассчитываем движение и проверяем прибытие группы." : mission.arrival ? !atArrival && mission.arrival.all_arrived ? "Перелёт рассчитан. Перейдите к прибытию для следующего этапа." : `${mission.arrival.arrived_count} из ${mission.arrival.total_count} аппаратов в назначенных зонах` : "Для этой записи прибытие не измерялось. Выполните новый расчёт с ожиданием группы."}</p></div></div>
      <div className="arrival-step"><span>02</span><div><strong>Подробная карта участка</strong><p>Загрузите вторую карту и задайте её положение относительно первой.</p></div></div>
      <div className="arrival-step"><span>03</span><div><strong>Дальнейшее движение</strong><p>Выберите сценарий. Начальные позиции и заряд сохранятся.</p></div></div>
    </div>
    {mission.status === "completed" && options && !options.allowed && <p className="stage-notice" role="status">{options.reason || "Следующий этап станет доступен после прибытия всех аппаратов."}</p>}
    {options?.can_continue && <div className="stage-notice"><span>{options.continue_reason || "Продолжим движение из последних позиций с сохранением заряда и целей."}</span><button className="primary-button small" disabled={!online || busy} onClick={continueArrival}>{busy ? <LoaderCircle className="spin" size={15} /> : <ArrowRight size={15} />} Продолжить до цели</button></div>}
    {loading && <p className="stage-notice"><LoaderCircle className="spin" size={15} /> Подготовка конечного участка…</p>}
    {expanded && options && <div className="stage-planner">
      {mission.arrival?.all_arrived && !atArrival && <div className="stage-notice"><span>Расчёт прибытия готов. Перейдите к концу перелёта, чтобы начать следующий этап.</span><button className="secondary-button small" onClick={onJumpToArrival}>Перейти к прибытию</button></div>}
      <div className="stage-scenarios" role="radiogroup" aria-label="Сценарий после прибытия">
        {options.scenarios.map((choice, i) => { const Icon = icons[i % icons.length]; return <button key={choice.id} role="radio" aria-checked={scenario === choice.id} aria-label={`После прибытия: ${choice.label}`} className={`stage-scenario ${scenario === choice.id ? "selected" : ""}`} disabled={busy || !choice.available} onClick={() => setScenario(choice.id)}><Icon size={20} /><strong>{choice.label}</strong><span>{choice.description}</span>{choice.recommended && <small>Подходит участку</small>}{!choice.available && <small>{choice.reason}</small>}</button>; })}
      </div>
      <div className="stage-registration">
        <div><strong>Привязка второй карты</strong><p>Левый верхний угол участка на первой карте, в метрах. Оси направлены вправо и вниз.</p></div>
        <label className="field"><span>Начало участка X, м</span><input type="number" step=".1" aria-label="Начало участка X, м" value={origin.x} disabled={busy} onChange={event => setOrigin(current => ({ ...current, x: +event.target.value }))} /></label>
        <label className="field"><span>Начало участка Y, м</span><input type="number" step=".1" aria-label="Начало участка Y, м" value={origin.y} disabled={busy} onChange={event => setOrigin(current => ({ ...current, y: +event.target.value }))} /></label>
        <button className="secondary-button small" disabled={busy} onClick={() => {
          const region = options.suggested_region;
          if (!region) return;
          setOrigin({ x: region.origin_x_m, y: region.origin_y_m });
          if (!editor.mapId) editor.restore(null, editor.shapes, region);
        }}>Подобрать границы</button>
      </div>
      <div className="destination-map-panel">
        <div className="destination-map-heading"><Map size={17} /><strong>Карта конечного участка</strong><span>{editor.width} × {editor.height} м</span></div>
        <TerritoryPanel editor={editor} disabled={busy || !online} onActivate={() => {}} />
        <svg ref={svg} viewBox="0 0 960 650" className={`destination-map ${editor.tool !== "select" ? "draw-mode" : ""}`} role="img" aria-label="Карта конечного участка с позициями прибывших аппаратов" tabIndex={0}
          onPointerDown={event => { if (!busy && online && !editor.busy) editor.down(point(event), event); }}
          onPointerMove={event => { if (!busy && online && !editor.busy) editor.move(point(event)); }}
          onPointerUp={() => editor.up()} onPointerCancel={() => editor.activate("select")}
          onKeyDown={event => { if (event.key === "Escape") editor.activate("select"); if (event.key === "Enter" && !busy && online && editor.tool === "polygon") editor.finishPolygon(); }}>
          <rect x="44" y="28" width="880" height="578" fill="var(--map-bg)" stroke="var(--map-border)" />
          <TerritoryBackground editor={editor} />
          <ScenarioPreview scenario={scenario} width={editor.width} height={editor.height} />
          <TerritoryLayers editor={editor} />
          {registrationValid && agents.map(agent => {
            const x = (agent.x_m - origin.x) / editor.width, y = (agent.y_m - origin.y) / editor.height;
            if (x < 0 || x > 1 || y < 0 || y > 1) return null;
            return <g key={agent.id} className="stage-source-agent" aria-label={`Прибывший UAV-${agent.source_drone_id || agent.id}`}><circle cx={44 + x * 880} cy={28 + y * 578} r="5" /><title>UAV-{agent.source_drone_id || agent.id} · {agent.battery.toFixed(1)}%</title></g>;
          })}
          <text x="44" y="632" fill="var(--muted)" fontSize="14">0</text><text x="870" y="632" fill="var(--muted)" fontSize="14">{editor.width} м</text>
        </svg>
        <div className="stage-map-legend"><span><i /> Фактические позиции при прибытии</span><span>Пунктир — схема будущих целей; траектории рассчитываются после запуска.</span></div>
      </div>
      {!!outside.length && <p className="stage-notice warning" role="alert">За границами второй карты: {outside.length} из {agents.length} аппаратов. Скорректируйте начало или размеры участка; позиции не будут перенесены автоматически.</p>}
      <div className="stage-submit"><p>{options.coordinate_note || "Следующий этап продолжит движение из сохранённых конечных позиций. Ограничения первой карты сохраняются."}</p><button className="primary-button" disabled={!ready || !options.scenarios.some(value => value.id === scenario && value.available)} onClick={startStage}>{busy ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />} Начать следующий этап</button></div>
      {!!options.next_stages?.length && <p className="stage-notice">Создано продолжений этой миссии: {options.next_stages.length}. Все этапы доступны в архиве миссий.</p>}
    </div>}
  </section>;
}

function ScenarioPreview({ scenario, width, height }: { scenario: string; width: number; height: number }) {
  const zone = (x: number, y: number, radius: number, key: number) => <ellipse key={key} cx={44 + x * 880} cy={28 + y * 578} rx={radius * Math.min(width, height) / width * 880} ry={radius * Math.min(width, height) / height * 578} />;
  return <g className="stage-scenario-preview" pointerEvents="none">
    {scenario === "distribution" && zone(.78, .5, .18, 0)}
    {scenario === "patrol" && <ellipse cx={44 + .7 * 880} cy={28 + .5 * 578} rx={.15 * 880} ry={.27 * 578} />}
    {scenario === "survey" && [.25, .73, .88].flatMap((x, i) => [.23, .77].map((y, j) => zone(x, y, .09, i * 2 + j)))}
    {scenario === "scan" && [.17, .39, .61, .83].map((y, i) => <line key={i} x1={44 + .3 * 880} x2={44 + .85 * 880} y1={28 + y * 578} y2={28 + y * 578} />)}
  </g>;
}
