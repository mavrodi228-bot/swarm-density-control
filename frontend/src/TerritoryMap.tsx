import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, PointerEvent } from "react";
import { Circle, ImagePlus, LoaderCircle, Map, MousePointer2, Pentagon, ScanSearch, Square, Trash2, Undo2, Check, X, RotateCcw } from "lucide-react";
import { api, requestBlob } from "./api";

export type MapPoint = { x: number; y: number };
export type RestrictedZone = {
  id?: string;
  kind: "rectangle" | "circle" | "polygon";
  x?: number; y?: number; width?: number; height?: number; radius?: number;
  points?: MapPoint[];
};
type Suggestion = RestrictedZone & { id: string; method?: string; source?: string; confidence?: number | null };
type Territory = {
  id: string; name: string; width_m: number; height_m: number;
  image_url: string; suggestions: Suggestion[];
  analysis?: { method?: string; warning?: string; note?: string };
};
export type TerritoryTool = "select" | "rectangle" | "circle" | "polygon";
export type TerritoryWorkspace = { width_m: number; height_m: number };
export type TerritoryEditReason = "map" | "shape" | "reset";
type StoredTerritory = {
  map_id: string | null; restricted_zones: RestrictedZone[];
  workspace?: TerritoryWorkspace; workspace_snapshot?: TerritoryWorkspace;
  suggestions_cleared?: boolean;
};
const uid = () => crypto.randomUUID().replaceAll("-", "").slice(0, 12);
const clamp = (v: number) => Math.max(.005, Math.min(.995, v));
const validDimension = (value: number) => Number.isFinite(value) && value >= 10 && value <= 10000;
function validWorkspace(value: unknown): TerritoryWorkspace | undefined {
  if (!value || typeof value !== "object") return;
  const candidate = value as TerritoryWorkspace;
  if (validDimension(candidate.width_m) && validDimension(candidate.height_m)) return { width_m: candidate.width_m, height_m: candidate.height_m };
}
const empty = (storageKey: string): StoredTerritory => {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || "null");
    return {
      map_id: typeof value?.map_id === "string" ? value.map_id : null,
      restricted_zones: Array.isArray(value?.restricted_zones) ? value.restricted_zones : [],
      workspace: validWorkspace(value?.workspace), workspace_snapshot: validWorkspace(value?.workspace_snapshot),
      suggestions_cleared: value?.suggestions_cleared === true,
    };
  }
  catch { return { map_id: null, restricted_zones: [] }; }
};

export function useTerritory(onError: (message: string) => void, online: boolean, onEdit?: (reason: TerritoryEditReason) => void, storageKey = "agmen-territory") {
  const [initial] = useState(() => empty(storageKey));
  const [maps, setMaps] = useState<Territory[]>([]);
  const [map, setMap] = useState<Territory | null>(null);
  const [mapId, setMapId] = useState<string | null>(initial.map_id);
  const [shapes, setShapes] = useState<RestrictedZone[]>(initial.restricted_zones);
  const [tool, setTool] = useState<TerritoryTool>("select");
  const [draft, setDraft] = useState<RestrictedZone | null>(null);
  const [vertices, setVertices] = useState<MapPoint[]>([]);
  const [image, setImage] = useState("");
  const [busy, setBusy] = useState(false);
  const [workspace, setWorkspace] = useState<TerritoryWorkspace>(initial.workspace_snapshot || initial.workspace || { width_m: 80, height_m: 60 });
  const { width_m: width, height_m: height } = workspace;
  const [draftWidth, setDraftWidth] = useState(width);
  const [draftHeight, setDraftHeight] = useState(height);
  // A mission owns its dimensions even if its source map is calibrated later.
  const snapshot = useRef<{ mapId: string | null; workspace: TerritoryWorkspace } | null>(initial.workspace_snapshot ? { mapId: initial.map_id, workspace: initial.workspace_snapshot } : null);
  const [method, setMethod] = useState("dark");
  const [sensitivity, setSensitivity] = useState(.5);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [resetNotice, setResetNotice] = useState("");
  const anchor = useRef<MapPoint | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const generation = useRef(0);
  const zoneGeneration = useRef(0);
  const catalogGeneration = useRef(0);
  const [loadVersion, setLoadVersion] = useState(0);
  const [suggestionsCleared, setSuggestionsCleared] = useState(initial.suggestions_cleared || false);
  const suppressSuggestions = useRef(suggestionsCleared);

  function suppress(value: boolean) { suppressSuggestions.current = value; setSuggestionsCleared(value); }
  function currentMap(value: Territory): Territory { return suppressSuggestions.current ? { ...value, suggestions: [] } : value; }
  function invalidate(reload = false) {
    generation.current += 1; setBusy(false);
    if (reload) setLoadVersion(value => value + 1);
  }
  async function refreshMaps() {
    if (!online) return;
    const request = ++catalogGeneration.current;
    try {
      const values = await api<Territory[]>("/api/maps");
      if (request === catalogGeneration.current) setMaps(values);
    } catch (error) { if (request === catalogGeneration.current) onError((error as Error).message); }
  }
  function notifyMapsChanged() { window.dispatchEvent(new Event("agmen-maps-changed")); }

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify({
      map_id: mapId, restricted_zones: shapes, workspace,
      workspace_snapshot: snapshot.current?.mapId === mapId ? snapshot.current.workspace : undefined,
      suggestions_cleared: suggestionsCleared,
    }));
  }, [mapId, shapes, workspace, suggestionsCleared, storageKey]);
  useEffect(() => {
    void refreshMaps();
    const changed = () => { void refreshMaps(); };
    window.addEventListener("agmen-maps-changed", changed);
    return () => { catalogGeneration.current += 1; window.removeEventListener("agmen-maps-changed", changed); };
  }, [online]);
  useEffect(() => () => { generation.current += 1; catalogGeneration.current += 1; }, []);
  useEffect(() => {
    if (!mapId || !online) { if (!mapId) { setMap(null); setImage(""); } return; }
    let active = true;
    const context = generation.current;
    let objectURL = "";
    (async () => {
      try {
        const metadata = await api<Territory>(`/api/maps/${mapId}`);
        const blob = await requestBlob(metadata.image_url);
        if (!active || context !== generation.current) return;
        objectURL = URL.createObjectURL(blob);
        const effective = snapshot.current?.mapId === mapId ? snapshot.current.workspace : validWorkspace(metadata);
        setMap(currentMap(metadata)); setImage(objectURL);
        if (effective) { setWorkspace(effective); setDraftWidth(effective.width_m); setDraftHeight(effective.height_m); }
      } catch (error) { if (active && context === generation.current) onError((error as Error).message); }
    })();
    return () => { active = false; if (objectURL) URL.revokeObjectURL(objectURL); };
  }, [mapId, online, loadVersion]);

  function activate(next: TerritoryTool) { setTool(next); setDraft(null); setVertices([]); setResetNotice(""); anchor.current = null; }
  function applyWorkspace(value: TerritoryWorkspace) {
    setWorkspace(value); setDraftWidth(value.width_m); setDraftHeight(value.height_m);
  }
  function restore(id: string | null | undefined, zones: RestrictedZone[] | undefined, dimensions?: TerritoryWorkspace) {
    invalidate(true); if ((id || null) !== mapId) suppress(false);
    const nextId = id || null, effective = validWorkspace(dimensions);
    snapshot.current = effective ? { mapId: nextId, workspace: effective } : null;
    if (effective) applyWorkspace(effective);
    else if (map?.id === nextId) applyWorkspace({ width_m: map.width_m, height_m: map.height_m });
    setMapId(nextId); setShapes(zones || []); activate("select");
  }
  function clearZones() {
    zoneGeneration.current += 1; suppress(true);
    setShapes([]); setMap(value => value ? { ...value, suggestions: [] } : value); activate("select");
    setResetNotice("Зоны очищены. Карта и масштаб сохранены.");
    onEdit?.("shape");
  }
  function resetMap() {
    invalidate(true); zoneGeneration.current += 1; snapshot.current = null; suppress(false);
    setMapId(null); setMap(null); setImage(""); setShapes([]); applyWorkspace({ width_m: 80, height_m: 60 });
    setMethod("dark"); setSensitivity(.5); setShowSuggestions(true); setExpanded(true); activate("select");
    setResetNotice("Новое поле 80 × 60 м готово. Загрузите другую карту или измените размеры.");
    if (fileInput.current) fileInput.current.value = "";
    onEdit?.("reset");
  }
  function setWidth(value: number) {
    setDraftWidth(value);
    if (!mapId && validDimension(value)) { snapshot.current = null; setWorkspace(current => ({ ...current, width_m: value })); onEdit?.("map"); }
  }
  function setHeight(value: number) {
    setDraftHeight(value);
    if (!mapId && validDimension(value)) { snapshot.current = null; setWorkspace(current => ({ ...current, height_m: value })); onEdit?.("map"); }
  }
  function append(shape: RestrictedZone) {
    if (shapes.length >= 32) { onError("Можно добавить не более 32 запретных зон."); return; }
    setShapes(current => [...current, { ...shape, id: uid() }]);
    onEdit?.("shape");
  }
  function down(point: MapPoint, event: PointerEvent<SVGSVGElement>) {
    if (tool === "select") return false;
    event.preventDefault();
    const p = { x: clamp(point.x), y: clamp(point.y) };
    if (tool === "polygon") {
      if (vertices.length >= 3 && Math.hypot(p.x - vertices[0].x, p.y - vertices[0].y) < .025) {
        finishPolygon();
      } else if (vertices.length < 24) setVertices(current => [...current, p]);
      else onError("У многоугольника может быть не более 24 вершин.");
      return true;
    }
    anchor.current = p;
    setDraft(tool === "circle" ? { kind: "circle", ...p, radius: .005 } : { kind: "rectangle", ...p, width: .005, height: .005 });
    event.currentTarget.setPointerCapture(event.pointerId);
    return true;
  }
  function move(point: MapPoint) {
    if (!anchor.current || tool === "select" || tool === "polygon") return;
    const p = { x: clamp(point.x), y: clamp(point.y) }, a = anchor.current;
    if (tool === "rectangle") setDraft({ kind: "rectangle", x: Math.min(a.x, p.x), y: Math.min(a.y, p.y), width: Math.abs(p.x - a.x), height: Math.abs(p.y - a.y) });
    else {
      const minimum = Math.min(width, height);
      const radius = Math.min(Math.hypot((p.x - a.x) * width, (p.y - a.y) * height) / minimum, a.x * width / minimum, (1 - a.x) * width / minimum, a.y * height / minimum, (1 - a.y) * height / minimum, .45);
      setDraft({ kind: "circle", ...a, radius: Math.max(.005, radius) });
    }
  }
  function up() {
    if (!anchor.current) return;
    if (draft && ((draft.kind === "circle" && (draft.radius || 0) >= .005) || ((draft.width || 0) >= .005 && (draft.height || 0) >= .005))) append(draft);
    anchor.current = null; setDraft(null);
  }
  function finishPolygon() {
    if (vertices.length < 3) { onError("Отметьте минимум три вершины запретной зоны."); return; }
    append({ kind: "polygon", points: vertices }); setVertices([]);
  }
  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = "";
    if (!file) return;
    if (!/^image\/(png|jpeg|webp)$/.test(file.type)) { onError("Выберите карту в формате PNG, JPEG или WEBP."); return; }
    if (file.size > 4 * 1024 * 1024) { onError("Размер карты должен быть не больше 4 МБ."); return; }
    const dimensions = validWorkspace({ width_m: draftWidth, height_m: draftHeight });
    if (!dimensions) { onError("Укажите ширину и высоту территории от 10 до 10 000 метров."); return; }
    invalidate();
    const context = generation.current, zonesAtStart = zoneGeneration.current;
    setBusy(true);
    try {
      const dataURL = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file); });
      if (context !== generation.current) return;
      const created = await api<Territory>("/api/maps", { name: file.name, image_data_url: dataURL, ...dimensions }, { timeoutMs: 30000 });
      notifyMapsChanged();
      if (context !== generation.current) return;
      if (zonesAtStart === zoneGeneration.current) suppress(false);
      snapshot.current = null; applyWorkspace(validWorkspace(created) || dimensions);
      setMaps(current => [created, ...current.filter(item => item.id !== created.id)]); setMap(currentMap(created)); setMapId(created.id); setLoadVersion(value => value + 1); setExpanded(true); setShowSuggestions(true);
      if (created.id !== mapId) setShapes([]);
      activate("select");
      onEdit?.("map");
      // The detected regions remain suggestions until the user accepts them.
    } catch (error) { if (context === generation.current) onError((error as Error).message); }
    finally { if (context === generation.current) setBusy(false); }
  }
  async function analyze() {
    if (!map) return;
    const context = generation.current, zonesAtStart = zoneGeneration.current;
    setBusy(true);
    try {
      const analyzed = await api<Territory>(`/api/maps/${map.id}/analyze`, { method, sensitivity }, { timeoutMs: 30000 });
      notifyMapsChanged();
      if (context !== generation.current || zonesAtStart !== zoneGeneration.current) return;
      suppress(false);
      setMap(analyzed); setShowSuggestions(true);
    } catch (error) { if (context === generation.current) onError((error as Error).message); }
    finally { if (context === generation.current) setBusy(false); }
  }
  async function calibrate() {
    if (!map) return;
    const dimensions = validWorkspace({ width_m: draftWidth, height_m: draftHeight });
    if (!dimensions) { onError("Укажите ширину и высоту территории от 10 до 10 000 метров."); return; }
    const context = generation.current;
    setBusy(true);
    try {
      const calibrated = await api<Territory>(`/api/maps/${map.id}/calibrate`, dimensions);
      notifyMapsChanged();
      if (context !== generation.current) return;
      snapshot.current = null; applyWorkspace(validWorkspace(calibrated) || dimensions);
      setMap(currentMap(calibrated)); setMaps(current => current.map(item => item.id === calibrated.id ? calibrated : item)); onEdit?.("map");
    }
    catch (error) { if (context === generation.current) onError((error as Error).message); }
    finally { if (context === generation.current) setBusy(false); }
  }
  function accept(suggestion: Suggestion) {
    const { source: _source, method: _method, confidence: _confidence, ...shape } = suggestion;
    append(shape); setMap(current => current ? { ...current, suggestions: current.suggestions.filter(item => item.id !== suggestion.id) } : current);
  }
  return { maps, map, mapId, shapes, tool, draft, vertices, image, busy, width, height, draftWidth, draftHeight, method, sensitivity, showSuggestions, expanded, fileInput, storageKey, resetNotice,
    setMapId: (id: string | null) => {
      invalidate(true); suppress(false); activate("select");
      snapshot.current = null;
      if (id !== mapId) setShapes([]);
      const selected = maps.find(item => item.id === id);
      if (selected) applyWorkspace({ width_m: selected.width_m, height_m: selected.height_m });
      else applyWorkspace(workspace);
      setMapId(id); onEdit?.("map");
    }, setShapes, setWidth, setHeight, setMethod, setSensitivity, setShowSuggestions, setExpanded,
    activate, restore, clearZones, resetMap, refreshMaps, down, move, up, finishPolygon, upload, analyze, calibrate, accept,
    reject: (id: string) => setMap(current => current ? { ...current, suggestions: current.suggestions.filter(item => item.id !== id) } : current),
    remove: (index: number) => { setShapes(current => current.filter((_, i) => i !== index)); onEdit?.("shape"); },
  };
}
type Editor = ReturnType<typeof useTerritory>;

export function TerritoryPanel({ editor: e, disabled = false, onActivate, additionalZoneCount = 0 }: { editor: Editor; disabled?: boolean; onActivate: () => void; additionalZoneCount?: number }) {
  const locked = disabled || e.busy;
  const hasZones = additionalZoneCount > 0 || e.shapes.length > 0 || e.vertices.length > 0 || e.draft !== null || (e.map?.suggestions?.length || 0) > 0;
  return <div className="territory-panel">
    <div className="territory-heading">
      <button className="text-button" onClick={() => e.setExpanded(!e.expanded)} aria-expanded={e.expanded}><Map size={17} /> {e.map ? e.map.name : "Своя территория"} <span>{e.expanded ? "−" : "+"}</span></button>
      <span className="subtle">{e.map ? `${e.width} × ${e.height} м` : "Загрузите карту или рисуйте на поле"}</span>
      <button className="secondary-button small" disabled={locked} onClick={() => e.fileInput.current?.click()}><ImagePlus size={15} /> Загрузить карту</button>
      <input ref={e.fileInput} type="file" accept="image/png,image/jpeg,image/webp" onChange={e.upload} hidden disabled={locked} aria-label="Файл карты территории" />
    </div>
    <div className="territory-actions" role="toolbar" aria-label="Инструменты запретных зон">
      {([{ id: "select", label: "Выбор", icon: MousePointer2 }, { id: "rectangle", label: "Прямоугольник", icon: Square }, { id: "circle", label: "Круг", icon: Circle }, { id: "polygon", label: "Многоугольник", icon: Pentagon }] as const).map(({ id, label, icon: Icon }) => <button key={id} className={`secondary-button small ${e.tool === id ? "active" : ""}`} disabled={locked} onClick={() => { e.activate(id); onActivate(); }} aria-pressed={e.tool === id} aria-label={`Запретная зона: ${label}`}><Icon size={15} /> {label}</button>)}
      <button className="icon-button" title="Удалить последнюю запретную зону" aria-label="Удалить последнюю запретную зону" disabled={locked || !e.shapes.length} onClick={() => e.remove(e.shapes.length - 1)}><Undo2 size={16} /></button>
      {e.tool === "polygon" && <button className="secondary-button small" disabled={locked || e.vertices.length < 3} onClick={e.finishPolygon}><Check size={14} /> Завершить контур</button>}
      <span className="shape-tag">{e.shapes.length} / 32 запретов</span>
    </div>
    <div className="territory-reset-controls">
      <div className="territory-reset-actions">
        <button className="secondary-button small" disabled={disabled || !hasZones} onClick={e.clearZones}><Trash2 size={14} /> Очистить зоны</button>
        <button className="secondary-button small territory-new-map" disabled={disabled} onClick={e.resetMap}><RotateCcw size={14} /> Новая карта</button>
      </div>
      <p>«Очистить зоны» оставляет подложку и размеры. «Новая карта» открывает пустое поле. Сохранённые карты доступны в списке.</p>
    </div>
    {e.resetNotice && <p className="territory-reset-notice" role="status">{e.resetNotice}</p>}
    {e.tool !== "select" && <p className="map-notice">{e.tool === "polygon" ? "Отмечайте вершины на карте. Замкните контур кнопкой или нажмите рядом с первой точкой." : "Нажмите и протяните по карте, чтобы задать запретную зону."} Зоны применятся при следующем расчёте.</p>}
    {e.expanded && <div className="territory-settings">
      {e.maps.length > 0 && <label className="field"><span>Сохранённая карта</span><select aria-label="Сохранённая карта" value={e.mapId || ""} disabled={locked} onChange={event => e.setMapId(event.target.value || null)}><option value="">Локальная сетка без подложки</option>{e.maps.map(map => <option key={map.id} value={map.id}>{map.name}</option>)}</select></label>}
      <div className="map-calibration"><label className="field"><span>Ширина территории, м</span><input aria-label="Ширина территории, м" type="number" min="10" max="10000" value={e.draftWidth} disabled={locked} onChange={event => e.setWidth(+event.target.value)} /></label><label className="field"><span>Высота территории, м</span><input aria-label="Высота территории, м" type="number" min="10" max="10000" value={e.draftHeight} disabled={locked} onChange={event => e.setHeight(+event.target.value)} /></label>{e.map && <button className="secondary-button small" onClick={e.calibrate} disabled={locked || !validDimension(e.draftWidth) || !validDimension(e.draftHeight)}>Применить масштаб</button>}</div>
      <p className="map-notice">PNG, JPEG или WEBP до 4 МБ. Укажите реальные размеры участка: географическая привязка по одному изображению не определяется.</p>
      {e.map && <><div className="map-analysis-controls"><label className="field"><span>Предварительный анализ</span><select aria-label="Метод анализа карты" value={e.method} disabled={locked} onChange={event => e.setMethod(event.target.value)}><option value="dark">Тёмные области</option><option value="water">Участки водного цвета</option><option value="contrast">Контрастные объекты</option></select></label><label className="field"><span>Чувствительность</span><input aria-label="Чувствительность анализа" type="range" min=".1" max=".9" step=".1" value={e.sensitivity} disabled={locked} onChange={event => e.setSensitivity(+event.target.value)} /></label><button className="secondary-button small" disabled={locked} onClick={e.analyze}>{e.busy ? <LoaderCircle size={15} className="spin" /> : <ScanSearch size={15} />} Найти препятствия</button></div>
        <p className="map-notice">Предложения основаны на цвете и контрасте. Они не подтверждают опасность объекта. Проверьте каждую область перед добавлением в запреты.</p>
        <label className="territory-toggle"><input type="checkbox" checked={e.showSuggestions} disabled={locked} onChange={event => e.setShowSuggestions(event.target.checked)} /> Показывать предложения анализа ({e.map.suggestions?.length || 0})</label>
        <div className="territory-shapes">{e.map.suggestions?.map((suggestion, i) => <div className="suggestion-row" key={suggestion.id}><span className="shape-tag">Предложение {i + 1}</span><button className="secondary-button small" disabled={locked || e.shapes.length >= 32} onClick={() => e.accept(suggestion)}><Check size={13} /> Добавить в запреты</button><button className="icon-button" title={`Отклонить предложение ${i + 1}`} aria-label={`Отклонить предложение ${i + 1}`} disabled={locked} onClick={() => e.reject(suggestion.id)}><X size={15} /></button></div>)}</div></>}
      {e.shapes.length > 0 && <div className="territory-shapes">{e.shapes.map((shape, i) => <div className="shape-row" key={shape.id || i}><span>Запрет {i + 1} · {shape.kind === "rectangle" ? "прямоугольник" : shape.kind === "circle" ? "круг" : "многоугольник"}</span><button className="icon-button" aria-label={`Удалить запрет ${i + 1}`} title={`Удалить запрет ${i + 1}`} disabled={locked} onClick={() => e.remove(i)}><Trash2 size={14} /></button></div>)}</div>}
    </div>}
  </div>;
}

function Shape({ value, width, height, className, label }: { value: RestrictedZone; width: number; height: number; className: string; label?: string }) {
  const x = (v: number) => 44 + v * 880, y = (v: number) => 28 + v * 578;
  let geometry;
  if (value.kind === "rectangle") geometry = <rect x={x(value.x || 0)} y={y(value.y || 0)} width={(value.width || 0) * 880} height={(value.height || 0) * 578} />;
  else if (value.kind === "circle") geometry = <ellipse cx={x(value.x || 0)} cy={y(value.y || 0)} rx={(value.radius || 0) * Math.min(width, height) / width * 880} ry={(value.radius || 0) * Math.min(width, height) / height * 578} />;
  else geometry = <polygon points={(value.points || []).map(p => `${x(p.x)},${y(p.y)}`).join(" ")} />;
  return <g className={className} pointerEvents="none">{geometry}{label && <text x={x(value.x ?? value.points?.[0]?.x ?? 0)} y={y(value.y ?? value.points?.[0]?.y ?? 0) - 5}>{label}</text>}</g>;
}
export function TerritoryBackground({ editor }: { editor: Editor }) {
  return editor.image ? <image className="map-background" href={editor.image} x="44" y="28" width="880" height="578" preserveAspectRatio="none" pointerEvents="none" /> : null;
}
export function TerritoryLayers({ editor: e }: { editor: Editor }) {
  return <g className="territory-layers">
    {e.showSuggestions && e.map?.suggestions?.map((value, i) => <Shape key={value.id} value={value} width={e.width} height={e.height} className="terrain-suggestion" label={`? ${i + 1}`} />)}
    {e.shapes.map((value, i) => <Shape key={value.id || i} value={value} width={e.width} height={e.height} className="terrain-restricted" label={`ЗАПРЕТ ${i + 1}`} />)}
    {e.draft && <Shape value={e.draft} width={e.width} height={e.height} className="terrain-draft" />}
    {e.vertices.length > 0 && <g className="terrain-draft" pointerEvents="none"><polyline points={e.vertices.map(p => `${44 + p.x * 880},${28 + p.y * 578}`).join(" ")} fill="none" />{e.vertices.map((p, i) => <circle key={i} cx={44 + p.x * 880} cy={28 + p.y * 578} r={i ? 3 : 6} />)}</g>}
  </g>;
}
