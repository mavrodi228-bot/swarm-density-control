import { cloneElement, isValidElement, useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  Battery,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  Crosshair,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Maximize2,
  Moon,
  MousePointer2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Settings2,
  ShieldCheck,
  SkipBack,
  Square,
  Sun,
  Target,
  TriangleAlert,
  Unplug,
  Waypoints,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";

type Agent = {
  id: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  battery: number;
  link: string;
  status: string;
};
type Metrics = {
  mae: number;
  tv_distance: number;
  compute_ms: number;
  min_distance: number;
  violations: number;
};
type Frame = {
  step: number;
  time: number;
  agents: Agent[];
  metrics: Metrics;
  density: number[][];
  target?: number[][];
  flux_x?: number[][];
  flux_y?: number[][];
};
type Config = {
  name: string;
  drone_count: number;
  controller: string;
  scenario: string;
  steps: number;
  seed: number;
  max_speed: number;
  safe_distance: number;
  obstacle: string;
  targets?: Zone[];
  model_id?: string;
};
type Zone = { x: number; y: number; radius: number; weight: number };
type Mission = {
  id: string;
  name: string;
  status: string;
  config?: Config;
  frames?: Frame[];
  target?: number[][];
  obstacle_rects?: { x: number; y: number; width: number; height: number }[];
  history?: Record<string, number>[];
  events?: Record<string, unknown>[];
  summary?: Record<string, number | boolean>;
  error?: string;
};
type Bootstrap = {
  controllers: { id: string; label: string; available: boolean }[];
  scenarios: { id: string; label: string }[];
  missions: Mission[];
  profile?:
    | string
    | {
        name: string;
        training_allowed?: boolean;
        limits?: { max_drones: number; max_steps: number };
      };
};
type Run = {
  id: string;
  status: string;
  history: Record<string, number>[];
  logs: string[];
  error?: string;
  metrics?: Record<string, number>;
  config?: Record<string, unknown>;
};
type Device = {
  id: string;
  status: string;
  job_status: string;
  package_loaded: boolean;
  package_id: string;
  controller_version: string;
  connectivity: {
    internet: boolean;
    ground_radio: boolean;
    peer_radio: boolean;
    gnss: boolean;
    low_battery?: boolean;
  };
  telemetry: {
    x: number;
    y: number;
    battery: number;
    step: number;
    velocity_mps?: number[];
  } | null;
  telemetry_age_s: number;
  telemetry_kind: string;
  journal_entries: number;
  last_sync_at: string | null;
  mode?: string;
  error?: string;
  history?: Record<string, unknown>[];
  journal?: { seq: number; step: number; kind: string; message: string }[];
  predicted_position?: { x: number; y: number; uncertainty_m: number };
  ground_receipt?: {
    step: number;
    journal_entries: number;
    synced_entries: number;
  };
  sync_pending_entries?: number;
};
type EdgePackage = { id: string; drone_id: number; mission_id: string };
const defaults: Config = {
  name: "Миссия «Горизонт»",
  drone_count: 48,
  controller: "baseline",
  scenario: "clusters",
  steps: 120,
  seed: 42,
  max_speed: 3,
  safe_distance: 1.2,
  obstacle: "central",
};
const labels: Record<string, string> = {
  baseline: "Математический",
  neural: "Нейросетевой",
  hybrid: "Гибридный",
  distribution: "Распределение",
  clusters: "Целевые области",
  patrol: "Патрулирование",
  survey: "Исследование",
  scan: "Сканирование",
  return: "Возврат",
  queued: "В очереди",
  running: "Вычисляется",
  completed: "Готова",
  failed: "Ошибка",
  stopped: "Остановлено",
  stopping: "Остановка",
  confirmed: "Подтверждённая",
  last_known: "Последняя известная",
  predicted: "Прогноз",
};
const fmt = (n: unknown, d = 2) =>
  typeof n === "number" && Number.isFinite(n) ? n.toFixed(d) : "—";
function authHeaders(): Record<string, string> {
  const token = sessionStorage.getItem("agmen-token");
  return token ? { "X-AGMEN-Token": token } : {};
}
async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      ...authHeaders(),
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    let m = t;
    try {
      const j = JSON.parse(t);
      m =
        typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* text error */
    }
    throw new Error(m || `HTTP ${r.status}`);
  }
  return r.json();
}
async function download(path: string, name: string) {
  const r = await fetch(path, { headers: authHeaders() });
  if (!r.ok) throw new Error(`Ошибка экспорта: HTTP ${r.status}`);
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}
function IconButton({
  title,
  onClick,
  children,
  active = false,
  disabled = false,
}: {
  title: string;
  onClick: () => void;
  children: ReactNode;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      className={`icon-button ${active ? "active" : ""}`}
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {isValidElement(children)
        ? cloneElement(children as ReactElement<{ "aria-label"?: string }>, {
            "aria-label": label,
          })
        : children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
function Chart({
  data,
  keys = ["mae"],
  colors = ["#3bd8d0", "#7187ff"],
  height = 150,
}: {
  data: Record<string, number>[];
  keys?: string[];
  colors?: string[];
  height?: number;
}) {
  const width = 600,
    pad = 28;
  const vals = data.flatMap((d) =>
    keys.map((k) => d[k]).filter(Number.isFinite),
  );
  const min = Math.min(0, ...vals),
    max = Math.max(0.001, ...vals);
  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="График измеренных значений"
      >
        <defs>
          <linearGradient id="chartFade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={colors[0]} stopOpacity=".16" />
            <stop offset="1" stopColor={colors[0]} stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1={pad}
              x2={width - 8}
              y1={pad + t * (height - pad * 2)}
              y2={pad + t * (height - pad * 2)}
              stroke="var(--line)"
              strokeDasharray="3 5"
            />
            <text
              x="0"
              y={pad + t * (height - pad * 2) + 3}
              fill="var(--muted)"
              fontSize="9"
            >
              {fmt(max - t * (max - min), max > 10 ? 0 : 3)}
            </text>
          </g>
        ))}
        {keys.map((k, i) => {
          const pts = data
            .map(
              (d, j) =>
                `${pad + (j / Math.max(1, data.length - 1)) * (width - pad - 8)},${height - pad - ((d[k] - min) / (max - min)) * (height - pad * 2)}`,
            )
            .join(" ");
          return (
            <g key={k}>
              {i === 0 && data.length > 1 && (
                <polygon
                  points={`${pad},${height - pad} ${pts} ${width - 8},${height - pad}`}
                  fill="url(#chartFade)"
                />
              )}
              <polyline
                points={pts}
                fill="none"
                stroke={colors[i % colors.length]}
                strokeWidth="2"
                strokeLinejoin="round"
              />
            </g>
          );
        })}
        <text x={pad} y={height - 4} fill="var(--muted)" fontSize="9">
          0
        </text>
        <text x={width - 42} y={height - 4} fill="var(--muted)" fontSize="9">
          {data.length ? data.length - 1 : 0} шаг
        </text>
        {!data.length && (
          <text
            x="300"
            y={height / 2}
            textAnchor="middle"
            fill="var(--muted)"
            fontSize="12"
          >
            Здесь появятся результаты вычислений
          </text>
        )}
      </svg>
    </div>
  );
}

function App() {
  const [page, setPage] = useState("mission");
  const [theme, setTheme] = useState(
    localStorage.getItem("agmen-theme") || "dark",
  );
  const [pro, setPro] = useState(localStorage.getItem("agmen-pro") === "true");
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [mission, setMission] = useState<Mission | null>(null);
  const [missions, setMissions] = useState<Mission[]>([]);
  const [config, setConfig] = useState<Config>(defaults);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<number[]>([]);
  const [zones, setZones] = useState<Zone[]>([]);
  const [tool, setTool] = useState("select");
  const [layers, setLayers] = useState({
    target: true,
    density: false,
    trails: true,
    velocity: false,
  });
  const [bottom, setBottom] = useState("fleet");
  const [fleetExpanded, setFleetExpanded] = useState(false);
  const [help, setHelp] = useState(false);
  const [toast, setToast] = useState("");
  const started = useRef(false);
  const mapRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [drawing, setDrawing] = useState<Zone | null>(null);
  const [mapZoom, setMapZoom] = useState(1);
  const [runs, setRuns] = useState<Run[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [labConfig, setLabConfig] = useState({
    epochs: 10,
    learning_rate: 0.001,
    batch_size: 8,
    samples: 64,
    optimizer: "AdamW",
    checkpoint: "bundled" as string | null,
    seed: 42,
    grid_size: 24,
  });
  const [edge, setEdge] = useState<{
    devices: Device[];
    packages: EdgePackage[];
  }>({ devices: [], packages: [] });
  const [edgePolicy, setEdgePolicy] = useState("hold");
  const [edgeCoord, setEdgeCoord] = useState("independent");
  const [edgeDrone, setEdgeDrone] = useState(1);
  const [edgeBusy, setEdgeBusy] = useState(false);
  const [edgeDevice, setEdgeDevice] = useState<string>("");
  const [experiment, setExperiment] = useState<Record<string, unknown> | null>(
    null,
  );
  const [optimization, setOptimization] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [token, setToken] = useState(
    sessionStorage.getItem("agmen-token") || "",
  );
  const frames = mission?.frames || [];
  const frame = frames[Math.min(frameIndex, frames.length - 1)];
  const metrics = frame?.metrics;
  const calculating =
    mission?.status === "running" || mission?.status === "queued";
  const device =
    edge.devices.find((d) => d.id === edgeDevice) || edge.devices[0];
  const profile = typeof boot?.profile === "object" ? boot.profile : undefined;
  const maxDrones = profile?.limits?.max_drones || 500;
  const maxSteps = profile?.limits?.max_steps || 300;
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("agmen-theme", theme);
  }, [theme]);
  useEffect(() => {
    localStorage.setItem("agmen-pro", String(pro));
  }, [pro]);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    (async () => {
      try {
        const b = await api<Bootstrap>("/api/bootstrap");
        setBoot(b);
        setMissions(b.missions || []);
        setOnline(true);
        const saved = localStorage.getItem("agmen-mission");
        const initial =
          b.missions?.find((m) => m.id === saved) || b.missions?.[0];
        if (initial) {
          const m = await api<Mission>(`/api/missions/${initial.id}`);
          setMission(m);
          if (m.config) {
            setConfig(m.config);
            setZones(m.config.targets || []);
          }
        }
      } catch (e) {
        setError(String((e as Error).message));
        setOnline(false);
      }
    })();
  }, []);
  useEffect(() => {
    if (!mission?.id || !calculating) return;
    const t = setInterval(async () => {
      try {
        const m = await api<Mission>(`/api/missions/${mission.id}`);
        setMission(m);
        if (m.status === "completed") {
          setFrameIndex(0);
          setPlaying(true);
          setMissions((prev) => [m, ...prev.filter((x) => x.id !== m.id)]);
          setToast("Расчёт завершён. Воспроизведение рассчитанной траектории.");
        }
        if (m.status === "failed") setError(m.error || "Ошибка расчёта миссии");
      } catch (e) {
        setError((e as Error).message);
      }
    }, 900);
    return () => clearInterval(t);
  }, [mission?.id, calculating]);
  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const t = setInterval(
      () =>
        setFrameIndex((i) => {
          if (i >= frames.length - 1) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        }),
      Math.max(20, 90 / speed),
    );
    return () => clearInterval(t);
  }, [playing, frames.length, speed]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 5000);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    if (page !== "lab") return;
    let live = true;
    const refresh = async () => {
      try {
        const [rs, ms] = await Promise.all([
          api<Run[]>("/api/lab/runs"),
          api<Record<string, unknown>[]>("/api/lab/models"),
        ]);
        if (live) {
          setRuns(rs);
          setModels(ms);
          setRun((r) =>
            r ? rs.find((x) => x.id === r.id) || r : rs[0] || null,
          );
        }
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    };
    refresh();
    const t = setInterval(refresh, 1800);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [page]);
  useEffect(() => {
    if (page !== "edge") return;
    let live = true;
    const refresh = async () => {
      try {
        const s = await api<{ devices: Device[]; packages: EdgePackage[] }>(
          "/api/edge/status",
        );
        if (live) setEdge(s);
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    };
    refresh();
    const t = setInterval(refresh, 1200);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [page]);
  useEffect(() => {
    if (page !== "mission" || !online) return;
    api<Record<string, unknown>[]>("/api/lab/models")
      .then(setModels)
      .catch(() => {});
    api<Bootstrap>("/api/bootstrap")
      .then((b) => {
        setBoot(b);
        setMissions(b.missions || []);
      })
      .catch(() => {});
  }, [page, online]);
  useEffect(() => {
    if (
      !experiment?.id ||
      !["queued", "running"].includes(String(experiment.status))
    )
      return;
    const t = setInterval(
      () =>
        api<Record<string, unknown>>(`/api/experiments/${experiment.id}`)
          .then(setExperiment)
          .catch((e) => setError(e.message)),
      1500,
    );
    return () => clearInterval(t);
  }, [experiment?.id, experiment?.status]);
  useEffect(() => {
    if (
      !optimization?.id ||
      !["queued", "running"].includes(String(optimization.status))
    )
      return;
    const t = setInterval(
      () =>
        api<Record<string, unknown>>(`/api/optimization/${optimization.id}`)
          .then(setOptimization)
          .catch((e) => setError(e.message)),
      1500,
    );
    return () => clearInterval(t);
  }, [optimization?.id, optimization?.status]);
  useEffect(() => {
    if (page !== "analytics" || !pro) return;
    api<Record<string, unknown>[]>("/api/experiments")
      .then((s) => setExperiment((e) => e || s[0] || null))
      .catch(() => {});
    api<Record<string, unknown>[]>("/api/optimization")
      .then((s) => setOptimization((e) => e || s[0] || null))
      .catch(() => {});
  }, [page, pro]);
  async function createMission() {
    setError("");
    setBusy(true);
    setPlaying(false);
    try {
      const m = await api<Mission>("/api/missions", {
        ...config,
        targets: zones.length ? zones : undefined,
      });
      setMission(m);
      setFrameIndex(0);
      setSelected([]);
      localStorage.setItem("agmen-mission", m.id);
      setMissions((prev) => [m, ...prev.filter((x) => x.id !== m.id)]);
      if (m.status === "completed") setPlaying(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function loadMission(id: string) {
    try {
      const m = await api<Mission>(`/api/missions/${id}`);
      setMission(m);
      setFrameIndex(0);
      setPlaying(false);
      if (m.config) {
        setConfig(m.config);
        setZones(m.config.targets || []);
      }
      localStorage.setItem("agmen-mission", id);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function failDrone(kind: string) {
    if (!mission || !selected.length) return;
    try {
      setPlaying(false);
      const m = await api<Mission>(`/api/missions/${mission.id}/events`, {
        kind,
        drone_id: selected[0],
        step: Math.max(
          1,
          Math.min(frame?.step || 1, (mission.config?.steps || 120) - 1),
        ),
      });
      setMission(m);
      setToast("Событие принято. Пересчитывается оставшаяся часть миссии.");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function point(e: React.PointerEvent<SVGSVGElement>) {
    const p = e.currentTarget.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const matrix = e.currentTarget.getScreenCTM();
    const out = matrix ? p.matrixTransform(matrix.inverse()) : p;
    return {
      x: Math.max(
        0.02,
        Math.min(0.98, ((out.x - 44) / 880 - 0.5) / mapZoom + 0.5),
      ),
      y: Math.max(
        0.02,
        Math.min(0.98, ((out.y - 28) / 578 - 0.5) / mapZoom + 0.5),
      ),
    };
  }
  function startDraw(e: React.PointerEvent<SVGSVGElement>) {
    if (tool !== "target") return;
    const p = point(e);
    drag.current = p;
    setDrawing({ ...p, radius: 0.02, weight: 1 });
    e.currentTarget.setPointerCapture(e.pointerId);
  }
  function finishDraw() {
    if (drag.current && drawing) {
      if (zones.length >= 32) setError("Достигнут предел: 32 целевые зоны");
      else {
        setZones((z) => [...z, drawing]);
        setToast(
          "Целевая зона добавлена. Запустите расчёт, чтобы применить её.",
        );
      }
    }
    drag.current = null;
    setDrawing(null);
  }
  async function train() {
    setBusy(true);
    setError("");
    try {
      const r = await api<Run>("/api/lab/runs", labConfig);
      setRun(r);
      setRuns((x) => [r, ...x]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function edgeAction(fn: () => Promise<unknown>, success: string) {
    setEdgeBusy(true);
    setError("");
    try {
      await fn();
      setEdge(await api("/api/edge/status"));
      setToast(success);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setEdgeBusy(false);
    }
  }
  const nav = [
    { id: "mission", label: "Миссии", icon: Waypoints },
    { id: "edge", label: "AGMEN EDGE", icon: Cpu },
    { id: "lab", label: "AI лаборатория", icon: FlaskConical },
    { id: "analytics", label: "Аналитика", icon: BarChart3 },
  ];
  const navigation = (id: string) => {
    setPage(id);
    setError("");
  };
  return (
    <div className="app-shell">
      <aside className="rail">
        <button
          className="emblem"
          aria-label="AGMEN — миссии"
          onClick={() => navigation("mission")}
        >
          <svg viewBox="0 0 38 38">
            <path
              d="M19 3 35 32H3L19 3Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
            />
            <path d="m19 13 10 19H9l10-19Z" fill="currentColor" />
            <path d="M5 27h28" stroke="var(--rail)" strokeWidth="3" />
          </svg>
        </button>
        <span className="rail-caption">CONTROL</span>
        <nav>
          {nav.map((n) => (
            <button
              key={n.id}
              title={n.label}
              aria-label={n.label}
              className={`nav-item ${page === n.id ? "selected" : ""}`}
              onClick={() => navigation(n.id)}
            >
              <n.icon size={21} />
              <span>{n.label}</span>
              {page === n.id && <i />}
            </button>
          ))}
        </nav>
        <div className="rail-bottom">
          <button
            className={`nav-item ${page === "settings" ? "selected" : ""}`}
            title="Настройки"
            aria-label="Настройки"
            onClick={() => navigation("settings")}
          >
            <Settings2 size={20} />
            <span>Настройки</span>
          </button>
          <button
            className="avatar"
            onClick={() => setHelp(true)}
            title="О платформе"
          >
            A
          </button>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="brand">
            AGMEN<span>SWARM INTELLIGENCE</span>
          </div>
          <div className="breadcrumb">
            Рабочее пространство
            <ChevronRight size={13} />
            <strong>
              {nav.find((n) => n.id === page)?.label || "Настройки"}
            </strong>
          </div>
          <div className="top-actions">
            <span className={`engine-status ${online ? "" : "disconnected"}`}>
              <i />
              {online ? "LOCAL ENGINE" : "НЕТ СОЕДИНЕНИЯ"}
            </span>
            <div className="mode-switch">
              <button
                className={!pro ? "selected" : ""}
                onClick={() => setPro(false)}
              >
                Basic
              </button>
              <button
                className={pro ? "selected" : ""}
                onClick={() => setPro(true)}
              >
                Pro
              </button>
            </div>
            <IconButton
              title="Сменить тему"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </IconButton>
            <IconButton title="Справка" onClick={() => setHelp(true)}>
              <CircleHelp size={18} />
            </IconButton>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span /> RESEARCH & SIMULATION PLATFORM
              </div>
              <h1>
                {page === "mission"
                  ? "Центр управления миссией"
                  : page === "edge"
                    ? "AGMEN EDGE CONTROL"
                    : page === "lab"
                      ? "AI лаборатория"
                      : page === "analytics"
                        ? "Аналитика миссий"
                        : "Настройки платформы"}
              </h1>
              <p>
                {page === "mission"
                  ? "От целевого распределения — к согласованному движению роя."
                  : page === "edge"
                    ? "Миссия на борту. Локальное выполнение. Независимость от сети."
                    : page === "lab"
                      ? "Обучение и проверка нейросетевого контроллера на локальном процессоре."
                      : page === "analytics"
                        ? "Измеренные результаты, сравнение контроллеров и экспорт исследований."
                        : "Ваше исследовательское пространство, ваши параметры."}
              </p>
            </div>
            <div className="heading-meta">
              <span className="small-badge">
                <ShieldCheck size={13} /> СИМУЛЯЦИЯ
              </span>
              <span className="version">WORKSPACE / 01</span>
            </div>
          </div>
          {error && (
            <div role="alert" className="error-banner">
              <TriangleAlert size={18} />
              <span>{error}</span>
              <button onClick={() => setError("")} aria-label="Закрыть ошибку">
                <X size={16} />
              </button>
            </div>
          )}
          {page === "mission" && (
            <>
              <section className="metrics-grid">
                <Metric
                  icon={<Waypoints size={18} />}
                  label="Аппаратов в рое"
                  value={
                    frame
                      ? String(frame.agents.length)
                      : String(config.drone_count)
                  }
                  suffix="UAV"
                  detail={
                    frame
                      ? `${frame.agents.filter((a) => a.status === "active").length} активных аппаратов`
                      : "Готовы к конфигурации"
                  }
                  tone="cyan"
                />
                <Metric
                  icon={<Target size={18} />}
                  label="Ошибка распределения"
                  value={fmt(metrics?.mae, 4)}
                  suffix="MAE"
                  detail={
                    mission?.summary
                      ? `${fmt(mission.summary.improvement_pct, 1)}% улучшение к старту`
                      : "Метрика плотности ρ − ρ*"
                  }
                  tone="blue"
                />
                <Metric
                  icon={<Activity size={18} />}
                  label="Время контроллера"
                  value={fmt(metrics?.compute_ms, 1)}
                  suffix="ms"
                  detail="Фактическое время вычисления"
                  tone="violet"
                />
                <Metric
                  icon={<ShieldCheck size={18} />}
                  label="Минимальная дистанция"
                  value={fmt(metrics?.min_distance, 2)}
                  suffix="m"
                  detail={
                    metrics
                      ? `${metrics.violations} нарушений на текущем шаге`
                      : "Проверяется на каждом шаге"
                  }
                  tone="green"
                />
              </section>
              <div className="mission-layout">
                <section className="panel map-panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <span className="title-dot" />
                      Поле миссии
                      <span className="subtle">
                        Локальная плоскость · 80 × 60 m
                      </span>
                    </div>
                    <div className="map-header-actions">
                      <span
                        className={`status-tag ${calculating ? "warning" : ""}`}
                      >
                        <i />
                        {calculating
                          ? "Расчёт траекторий"
                          : playing
                            ? "Воспроизведение"
                            : frames.length
                              ? "Расчёт завершён"
                              : "Готов к запуску"}
                      </span>
                      <IconButton
                        title="Развернуть поле"
                        onClick={() => {
                          const el = mapRef.current?.parentElement;
                          if (el) {
                            if (document.fullscreenElement)
                              document.exitFullscreen();
                            else el.requestFullscreen?.();
                          }
                        }}
                      >
                        <Maximize2 size={15} />
                      </IconButton>
                    </div>
                  </div>
                  <div
                    className={`map-viewport ${tool === "target" ? "draw-mode" : ""}`}
                  >
                    <div className="map-toolbar">
                      <IconButton
                        title="Выбрать аппарат"
                        active={tool === "select"}
                        onClick={() => setTool("select")}
                      >
                        <MousePointer2 size={17} />
                      </IconButton>
                      <IconButton
                        title="Нарисовать целевую зону: протяните мышью"
                        active={tool === "target"}
                        onClick={() => setTool("target")}
                      >
                        <Crosshair size={18} />
                      </IconButton>
                      <div className="toolbar-divider" />
                      <IconButton
                        title="Показать целевую плотность"
                        active={layers.target}
                        onClick={() =>
                          setLayers({ ...layers, target: !layers.target })
                        }
                      >
                        <Target size={18} />
                      </IconButton>
                      <IconButton
                        title="Показать плотность роя"
                        active={layers.density}
                        onClick={() =>
                          setLayers({ ...layers, density: !layers.density })
                        }
                      >
                        <Layers3 size={17} />
                      </IconButton>
                      <IconButton
                        title="Показать траектории"
                        active={layers.trails}
                        onClick={() =>
                          setLayers({ ...layers, trails: !layers.trails })
                        }
                      >
                        <Waypoints size={18} />
                      </IconButton>
                      <IconButton
                        title="Показать векторы скорости"
                        active={layers.velocity}
                        onClick={() =>
                          setLayers({ ...layers, velocity: !layers.velocity })
                        }
                      >
                        <ArrowRight size={18} />
                      </IconButton>
                    </div>
                    <div className="map-corner">
                      <span className="live-dot" />{" "}
                      {frame
                        ? `T + ${fmt(frame.time, 1)} s`
                        : "ГОТОВ К РАСЧЁТУ"}
                      <small>Локальная система координат</small>
                    </div>
                    <svg
                      ref={mapRef}
                      className="mission-map"
                      viewBox="0 0 960 650"
                      aria-label="Интерактивное поле симуляции"
                      onPointerDown={startDraw}
                      onPointerMove={(e) => {
                        if (drag.current) {
                          const p = point(e);
                          setDrawing({
                            ...drag.current,
                            radius: Math.min(
                              0.35,
                              Math.max(
                                0.02,
                                Math.hypot(
                                  p.x - drag.current.x,
                                  p.y - drag.current.y,
                                ),
                              ),
                            ),
                            weight: 1,
                          });
                        }
                      }}
                      onPointerUp={finishDraw}
                    >
                      <defs>
                        <pattern
                          id="mapgrid"
                          width="55"
                          height="48.166"
                          patternUnits="userSpaceOnUse"
                        >
                          <path
                            d="M 55 0 L 0 0 0 48.166"
                            fill="none"
                            stroke="var(--map-grid)"
                            strokeWidth=".7"
                          />
                          <circle cx="0" cy="0" r="1.1" fill="var(--map-dot)" />
                        </pattern>
                        <radialGradient id="zoneGlow">
                          <stop
                            offset="0"
                            stopColor="#30d3cd"
                            stopOpacity=".25"
                          />
                          <stop
                            offset="1"
                            stopColor="#30d3cd"
                            stopOpacity="0"
                          />
                        </radialGradient>
                        <pattern
                          id="obstacle"
                          width="8"
                          height="8"
                          patternUnits="userSpaceOnUse"
                          patternTransform="rotate(45)"
                        >
                          <line
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="8"
                            stroke="#dfa36b"
                            strokeWidth="1"
                            opacity=".35"
                          />
                        </pattern>
                        <clipPath id="plot">
                          <rect x="44" y="28" width="880" height="578" rx="2" />
                        </clipPath>
                      </defs>
                      <rect
                        x="44"
                        y="28"
                        width="880"
                        height="578"
                        fill="url(#mapgrid)"
                        stroke="var(--map-border)"
                      />
                      {Array.from({ length: 9 }, (_, i) => (
                        <g key={i}>
                          <text
                            x={44 + i * 110}
                            y="629"
                            fill="var(--muted)"
                            fontSize="10"
                            textAnchor="middle"
                          >
                            {i * 10}
                          </text>
                          {i < 7 && (
                            <text
                              x="27"
                              y={32 + i * 96.33}
                              fill="var(--muted)"
                              fontSize="10"
                              textAnchor="end"
                            >
                              {i * 10}
                            </text>
                          )}
                        </g>
                      ))}
                      <text x="925" y="629" fill="var(--muted)" fontSize="9">
                        m / X
                      </text>
                      <text x="17" y="20" fill="var(--muted)" fontSize="9">
                        Y / m
                      </text>
                      <g
                        clipPath="url(#plot)"
                        transform={`translate(${484 * (1 - mapZoom)},${317 * (1 - mapZoom)}) scale(${mapZoom})`}
                      >
                        {layers.target &&
                          (frame?.target || mission?.target)?.flatMap(
                            (row, y) =>
                              row.map((v, x) => {
                                const matrix =
                                  frame?.target || mission?.target || [];
                                const max = Math.max(...matrix.flat(), 0.001);
                                return v / max > 0.03 ? (
                                  <rect
                                    key={`t${x}-${y}`}
                                    x={44 + (x * 880) / row.length}
                                    y={28 + (y * 578) / matrix.length}
                                    width={880 / row.length + 0.3}
                                    height={578 / matrix.length + 0.3}
                                    fill="#24c7ba"
                                    opacity={Math.min(0.36, (v / max) * 0.36)}
                                  />
                                ) : null;
                              }),
                          )}
                        {layers.density &&
                          frame?.density?.flatMap((row, y) =>
                            row.map((v, x) => {
                              const max = Math.max(
                                ...frame.density.flat(),
                                0.001,
                              );
                              return v / max > 0.03 ? (
                                <rect
                                  key={`d${x}-${y}`}
                                  x={44 + (x * 880) / row.length}
                                  y={28 + (y * 578) / frame.density.length}
                                  width={880 / row.length + 0.3}
                                  height={578 / frame.density.length + 0.3}
                                  fill="#6d8cff"
                                  opacity={Math.min(0.55, (v / max) * 0.55)}
                                />
                              ) : null;
                            }),
                          )}
                        {mission?.obstacle_rects?.map((o, i) => (
                          <g key={i}>
                            <rect
                              x={44 + o.x * 880}
                              y={28 + o.y * 578}
                              width={o.width * 880}
                              height={o.height * 578}
                              fill="url(#obstacle)"
                              stroke="#b17f4d"
                              strokeDasharray="4 3"
                              opacity=".75"
                            />
                            <text
                              x={44 + (o.x + o.width / 2) * 880}
                              y={28 + (o.y + o.height / 2) * 578}
                              fill="#cd9a66"
                              fontSize="8"
                              textAnchor="middle"
                            >
                              NO-FLY
                            </text>
                          </g>
                        ))}
                        {zones.map((z, i) => (
                          <g key={i}>
                            <ellipse
                              cx={44 + z.x * 880}
                              cy={28 + z.y * 578}
                              rx={z.radius * 880}
                              ry={z.radius * 578}
                              fill="url(#zoneGlow)"
                              stroke="#44d5cb"
                              strokeDasharray="5 4"
                              strokeWidth="1.2"
                            />
                            <text
                              x={44 + z.x * 880}
                              y={28 + z.y * 578}
                              textAnchor="middle"
                              fill="#67e5da"
                              fontSize="10"
                            >
                              ЗОНА {i + 1}
                            </text>
                          </g>
                        ))}
                        {drawing && (
                          <ellipse
                            cx={44 + drawing.x * 880}
                            cy={28 + drawing.y * 578}
                            rx={drawing.radius * 880}
                            ry={drawing.radius * 578}
                            fill="url(#zoneGlow)"
                            stroke="#44d5cb"
                            strokeDasharray="4 4"
                          />
                        )}
                        {layers.trails &&
                          frame?.agents.map((a, i) => (
                            <polyline
                              key={`trail${a.id}`}
                              points={frames
                                .slice(
                                  Math.max(0, frameIndex - 14),
                                  frameIndex + 1,
                                )
                                .map((f) => f.agents[i])
                                .filter(Boolean)
                                .map(
                                  (p) => `${44 + p.x * 880},${28 + p.y * 578}`,
                                )
                                .join(" ")}
                              fill="none"
                              stroke={
                                selected.includes(a.id) ? "#e5ffff" : "#36a5a9"
                              }
                              strokeWidth={selected.includes(a.id) ? 1.8 : 0.8}
                              opacity={selected.includes(a.id) ? 0.75 : 0.38}
                            />
                          ))}
                        {frame?.agents.map((a) => (
                          <g
                            key={a.id}
                            className="drone drone-marker"
                            role="button"
                            tabIndex={0}
                            aria-label={`Выбрать UAV-${String(a.id).padStart(3, "0")}`}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                setSelected([a.id]);
                              }
                            }}
                            transform={`translate(${44 + a.x * 880},${28 + a.y * 578})`}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (tool === "select")
                                setSelected((s) =>
                                  e.shiftKey
                                    ? s.includes(a.id)
                                      ? s.filter((i) => i !== a.id)
                                      : [...s, a.id]
                                    : s.includes(a.id)
                                      ? []
                                      : [a.id],
                                );
                            }}
                          >
                            <title>{`UAV-${String(a.id).padStart(3, "0")} · ${fmt(a.battery, 0)}% · ${fmt(a.speed, 1)} m/s`}</title>
                            <circle r="12" fill="transparent" />
                            {selected.includes(a.id) && (
                              <circle
                                r="12"
                                fill="#4ac8d01a"
                                stroke="#8cf6ed"
                                strokeWidth="1"
                              />
                            )}
                            {layers.velocity && (
                              <line
                                x1="0"
                                y1="0"
                                x2={Math.cos(a.heading) * a.speed * 10}
                                y2={Math.sin(a.heading) * a.speed * 10}
                                stroke="#77e1db"
                                strokeWidth="1"
                                opacity=".7"
                              />
                            )}
                            <g
                              transform={`rotate(${(a.heading * 180) / Math.PI + 90})`}
                            >
                              <path
                                d="M0 -6.5 4.5 5 0 2.8 -4.5 5Z"
                                fill={
                                  a.status === "failed"
                                    ? "#f27878"
                                    : a.link === "lost"
                                      ? "#f3bd70"
                                      : selected.includes(a.id)
                                        ? "#edffff"
                                        : "#71e1d8"
                                }
                                stroke="var(--map-bg)"
                                strokeWidth=".8"
                              />
                              <path
                                d="M-6 1H6M-3 -3 3 5M3 -3 -3 5"
                                stroke={
                                  a.status === "failed" ? "#f27878" : "#71e1d8"
                                }
                                strokeWidth=".7"
                                opacity=".8"
                              />
                            </g>
                            {selected.includes(a.id) && (
                              <text x="15" y="-8" fill="#bcfffa" fontSize="9">
                                UAV-{String(a.id).padStart(3, "0")}
                              </text>
                            )}
                          </g>
                        ))}
                      </g>
                    </svg>
                    {!frames.length && !calculating && (
                      <div className="map-empty">
                        <div className="empty-orbit">
                          <Waypoints size={30} />
                        </div>
                        <h3>Пространство для вашей миссии</h3>
                        <p>
                          Настройте рой и запустите расчёт.
                          <br />
                          Добавьте собственные целевые области инструментом{" "}
                          <Crosshair size={13} />.
                        </p>
                        <button className="text-button" onClick={createMission}>
                          Рассчитать первую миссию <ArrowRight size={14} />
                        </button>
                      </div>
                    )}
                    {calculating && (
                      <div className="calculation-indicator">
                        <LoaderCircle size={16} className="spin" /> Контроллер
                        рассчитывает движение роя…
                      </div>
                    )}
                    <div className="map-bottom">
                      <div className="map-legend">
                        <span>
                          <i className="legend-drone" />
                          Аппарат
                        </span>
                        <span>
                          <i className="legend-zone" />
                          Целевая плотность
                        </span>
                        {layers.density && (
                          <span>
                            <i className="legend-density" />
                            Плотность роя
                          </span>
                        )}
                        <span>
                          <i className="legend-obstacle" />
                          Ограничение
                        </span>
                      </div>
                      <div className="zoom-controls">
                        <button
                          aria-label="Уменьшить"
                          onClick={() =>
                            setMapZoom((z) => Math.max(1, z - 0.25))
                          }
                        >
                          −
                        </button>
                        <span>{Math.round(mapZoom * 100)}%</span>
                        <button
                          aria-label="Увеличить"
                          onClick={() =>
                            setMapZoom((z) => Math.min(2, z + 0.25))
                          }
                        >
                          +
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="playback">
                    <div className="playback-buttons">
                      <IconButton
                        title="В начало"
                        disabled={!frames.length}
                        onClick={() => {
                          setFrameIndex(0);
                          setPlaying(false);
                        }}
                      >
                        <SkipBack size={16} />
                      </IconButton>
                      <button
                        className="play-control"
                        aria-label={playing ? "Пауза" : "Воспроизвести"}
                        disabled={!frames.length || calculating}
                        onClick={() => {
                          if (frameIndex === frames.length - 1)
                            setFrameIndex(0);
                          setPlaying(!playing);
                        }}
                      >
                        {playing ? (
                          <Pause size={16} fill="currentColor" />
                        ) : (
                          <Play size={16} fill="currentColor" />
                        )}
                      </button>
                      <IconButton
                        title="Остановить воспроизведение"
                        disabled={!frames.length}
                        onClick={() => {
                          setPlaying(false);
                          setFrameIndex(0);
                        }}
                      >
                        <Square size={13} />
                      </IconButton>
                    </div>
                    <span className="timecode">
                      {fmt(frame?.time || 0, 1)}
                      <small> / {fmt(frames.at(-1)?.time || 0, 1)} s</small>
                    </span>
                    <input
                      className="timeline"
                      type="range"
                      min="0"
                      max={Math.max(0, frames.length - 1)}
                      value={frameIndex}
                      aria-label="Шаг воспроизведения"
                      onChange={(e) => {
                        setFrameIndex(+e.target.value);
                        setPlaying(false);
                      }}
                      disabled={!frames.length}
                    />
                    <select
                      className="speed-select"
                      aria-label="Скорость воспроизведения"
                      value={speed}
                      onChange={(e) => setSpeed(+e.target.value)}
                    >
                      {[0.5, 1, 2, 4].map((s) => (
                        <option key={s} value={s}>
                          {s}×
                        </option>
                      ))}
                    </select>
                    <span className="step-count">
                      {frame?.step || 0}{" "}
                      <span>/ {frames.at(-1)?.step || config.steps}</span>
                    </span>
                  </div>
                </section>
                <aside className="panel config-panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <Settings2 size={15} />
                      Параметры миссии
                    </div>
                    <span className="panel-index">01</span>
                  </div>
                  <div className="config-body">
                    <Field label="Название миссии">
                      <input
                        value={config.name}
                        onChange={(e) =>
                          setConfig({ ...config, name: e.target.value })
                        }
                        maxLength={100}
                      />
                    </Field>
                    <Field label="Сценарий">
                      <select
                        value={config.scenario}
                        onChange={(e) =>
                          setConfig({ ...config, scenario: e.target.value })
                        }
                      >
                        {(
                          boot?.scenarios ||
                          Object.keys(labels)
                            .filter((k) =>
                              [
                                "distribution",
                                "clusters",
                                "patrol",
                                "survey",
                                "scan",
                                "return",
                              ].includes(k),
                            )
                            .map((id) => ({ id, label: labels[id] }))
                        ).map((s) => (
                          <option value={s.id} key={s.id}>
                            {s.label || labels[s.id]}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <div className="field">
                      <span>Контроллер</span>
                      <div className="controller-options">
                        {["baseline", "neural", "hybrid"].map((c, i) => {
                          const unavailable =
                            boot?.controllers.find((x) => x.id === c)
                              ?.available === false;
                          return (
                            <button
                              disabled={unavailable}
                              title={
                                unavailable
                                  ? "Модель недоступна: обучите контроллер в AI лаборатории"
                                  : labels[c]
                              }
                              className={
                                config.controller === c ? "selected" : ""
                              }
                              key={c}
                              onClick={() =>
                                setConfig({ ...config, controller: c })
                              }
                            >
                              {i === 0 ? (
                                <Activity size={16} />
                              ) : i === 1 ? (
                                <Cpu size={16} />
                              ) : (
                                <Layers3 size={16} />
                              )}
                              <span>
                                {i === 0
                                  ? "Math"
                                  : i === 1
                                    ? "Neural"
                                    : "Hybrid"}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                      <small>
                        {labels[config.controller]} контроллер поля плотности
                      </small>
                    </div>
                    {config.controller !== "baseline" && (
                      <Field label="Нейросетевая модель">
                        <select
                          value={config.model_id || "bundled"}
                          onChange={(e) =>
                            setConfig({ ...config, model_id: e.target.value })
                          }
                        >
                          {models
                            .filter(
                              (m) =>
                                m.available &&
                                (m.validated || m.id === "bundled"),
                            )
                            .map((m) => (
                              <option value={String(m.id)} key={String(m.id)}>
                                {String(m.name || m.id)}
                              </option>
                            ))}
                        </select>
                      </Field>
                    )}
                    <div className="range-field">
                      <div>
                        <label htmlFor="drone-count">
                          Количество аппаратов
                        </label>
                        <strong>
                          {config.drone_count}
                          <small> UAV</small>
                        </strong>
                      </div>
                      <input
                        id="drone-count"
                        type="range"
                        min="5"
                        max={maxDrones}
                        step="1"
                        value={config.drone_count}
                        onChange={(e) =>
                          setConfig({ ...config, drone_count: +e.target.value })
                        }
                      />
                      <div className="range-extents">
                        <span>5</span>
                        <span>{maxDrones}</span>
                      </div>
                    </div>
                    <div className="two-fields">
                      <Field label="Скорость, m/s">
                        <input
                          type="number"
                          min="0.1"
                          max="15"
                          step="0.1"
                          value={config.max_speed}
                          onChange={(e) =>
                            setConfig({ ...config, max_speed: +e.target.value })
                          }
                        />
                      </Field>
                      <Field label="Дистанция, m">
                        <input
                          type="number"
                          min="0.25"
                          max="8"
                          step="0.05"
                          value={config.safe_distance}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              safe_distance: +e.target.value,
                            })
                          }
                        />
                      </Field>
                    </div>
                    <Field label="Длительность расчёта">
                      <select
                        value={config.steps}
                        onChange={(e) =>
                          setConfig({ ...config, steps: +e.target.value })
                        }
                      >
                        {[20, 40, 80, 120, 150, 200, 300]
                          .filter((s) => s <= maxSteps)
                          .map((s) => (
                            <option value={s} key={s}>
                              {s} шагов симуляции
                            </option>
                          ))}
                      </select>
                    </Field>
                    {pro && (
                      <div className="advanced">
                        <span className="eyebrow">РАСШИРЕННЫЕ ПАРАМЕТРЫ</span>
                        <Field label="Препятствия">
                          <select
                            value={config.obstacle}
                            onChange={(e) =>
                              setConfig({ ...config, obstacle: e.target.value })
                            }
                          >
                            <option value="none">Без препятствий</option>
                            <option value="central">
                              Центральная запретная зона
                            </option>
                            <option value="corridor">Коридор</option>
                          </select>
                        </Field>
                        <Field label="Seed — воспроизводимость">
                          <input
                            type="number"
                            min="0"
                            max="2147483547"
                            value={config.seed}
                            onChange={(e) =>
                              setConfig({ ...config, seed: +e.target.value })
                            }
                          />
                        </Field>
                      </div>
                    )}
                    {zones.length > 0 && (
                      <div className="zone-count">
                        <Target size={14} />
                        {zones.length} пользовательских зон
                        <button
                          onClick={() => setZones([])}
                          title="Очистить зоны"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    )}
                    <div className="config-note">
                      <ShieldCheck size={15} />
                      <p>
                        Исследовательская 2D симуляция.
                        <br />
                        Физические аппараты не подключены.
                      </p>
                    </div>
                    <button
                      className="primary-button launch-button"
                      disabled={busy || calculating || !online}
                      onClick={createMission}
                    >
                      {busy || calculating ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <Play size={15} fill="currentColor" />
                      )}
                      {busy || calculating
                        ? "Рассчитываем миссию…"
                        : "Рассчитать миссию"}
                      <span aria-hidden="true">↗</span>
                    </button>
                    <p className="launch-caption">
                      Результат сохраняется локально
                    </p>
                  </div>
                </aside>
              </div>
              <section className="panel fleet-panel">
                <div className="panel-header">
                  <div className="tabs">
                    <button
                      className={bottom === "fleet" ? "selected" : ""}
                      onClick={() => setBottom("fleet")}
                    >
                      Аппараты <span>{frame?.agents.length || 0}</span>
                    </button>
                    <button
                      className={bottom === "metrics" ? "selected" : ""}
                      onClick={() => setBottom("metrics")}
                    >
                      Динамика плотности
                    </button>
                    <button
                      className={bottom === "events" ? "selected" : ""}
                      onClick={() => setBottom("events")}
                    >
                      Журнал миссии <span>{mission?.events?.length || 0}</span>
                    </button>
                  </div>
                  <span className="subtle hide-small">
                    {selected.length
                      ? `Выбрано: ${selected.map((id) => "UAV-" + String(id).padStart(3, "0")).join(", ")}`
                      : "Выберите аппарат на поле · Shift для группы"}
                  </span>
                </div>
                {bottom === "fleet" ? (
                  <>
                    <div className="fleet-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Аппарат</th>
                            <th>Состояние</th>
                            <th>Положение, m</th>
                            <th>Скорость</th>
                            <th>Заряд · модель</th>
                            <th>Канал · симуляция</th>
                            {pro && <th>Событие</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {frame?.agents
                            .slice(
                              0,
                              selected.length || fleetExpanded ? 500 : 8,
                            )
                            .filter(
                              (a) =>
                                !selected.length || selected.includes(a.id),
                            )
                            .map((a) => (
                              <tr
                                key={a.id}
                                className={
                                  selected.includes(a.id) ? "row-selected" : ""
                                }
                                onClick={() => setSelected([a.id])}
                              >
                                <td>
                                  <span className="drone-cell">
                                    <Waypoints size={14} />
                                    <strong>
                                      UAV-{String(a.id).padStart(3, "0")}
                                    </strong>
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`table-status ${a.status === "failed" ? "red" : ""}`}
                                  >
                                    <i />
                                    {a.status === "failed"
                                      ? "Отказ"
                                      : "Выполняет миссию"}
                                  </span>
                                </td>
                                <td className="mono">
                                  {fmt(a.x * 80, 1)} / {fmt(a.y * 60, 1)}
                                </td>
                                <td className="mono">
                                  {fmt(a.speed, 2)}{" "}
                                  <span className="subtle">m/s</span>
                                </td>
                                <td>
                                  <span className="battery-cell">
                                    <Battery size={15} />
                                    <span className="battery-bar">
                                      <i style={{ width: `${a.battery}%` }} />
                                    </span>
                                    <span className="mono">
                                      {fmt(a.battery, 0)}%
                                    </span>
                                  </span>
                                </td>
                                <td>
                                  <span
                                    className={`radio-cell ${a.link === "lost" ? "amber" : ""}`}
                                  >
                                    <Radio size={13} />
                                    {a.link === "lost"
                                      ? "Связь потеряна"
                                      : "Стабильный"}
                                  </span>
                                </td>
                                {pro && (
                                  <td>
                                    <button
                                      className="text-button warning-text"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setSelected([a.id]);
                                        setToast(
                                          "Аппарат выбран. Используйте действия под таблицей.",
                                        );
                                      }}
                                    >
                                      Выбрать
                                    </button>
                                  </td>
                                )}
                              </tr>
                            ))}
                        </tbody>
                      </table>
                      {frame && frame.agents.length > 8 && !selected.length && (
                        <button
                          className="fleet-expand"
                          onClick={() => setFleetExpanded(!fleetExpanded)}
                        >
                          {fleetExpanded
                            ? "Свернуть список"
                            : `Показать все аппараты (${frame.agents.length})`}{" "}
                          <ChevronRight size={12} />
                        </button>
                      )}
                      {!frame && (
                        <div className="table-empty">
                          Телеметрия аппаратов появится после первого расчёта
                          миссии.
                        </div>
                      )}
                    </div>
                    {pro && selected.length > 0 && (
                      <div className="fleet-actions">
                        <span>
                          Исследование устойчивости · UAV-{selected[0]}
                        </span>
                        <button
                          className="secondary-button small"
                          disabled={calculating}
                          onClick={() => failDrone("drone_failure")}
                        >
                          Смоделировать отказ
                        </button>
                        <button
                          className="secondary-button small"
                          disabled={calculating}
                          onClick={() => failDrone("link_loss")}
                        >
                          Потеря радиоканала
                        </button>
                        <button
                          className="text-button"
                          onClick={() => setSelected([])}
                        >
                          Снять выбор
                        </button>
                      </div>
                    )}
                  </>
                ) : bottom === "metrics" ? (
                  <div className="chart-section">
                    <div>
                      <h3>Ошибка распределения плотности</h3>
                      <p>MAE между текущим и целевым полем на каждом шаге</p>
                    </div>
                    <Chart data={mission?.history || []} />
                  </div>
                ) : (
                  <div className="event-log">
                    {mission?.events?.length ? (
                      mission.events.map((e, i) => (
                        <div key={i}>
                          <span className="mono">
                            {String(e.step ?? i).padStart(4, "0")}
                          </span>
                          <Radio size={13} />
                          <span>
                            {String(
                              e.kind ||
                                e.type ||
                                e.message ||
                                JSON.stringify(e),
                            )}
                          </span>
                          <span className="subtle">
                            {e.drone_id ? `UAV-${e.drone_id}` : ""}
                          </span>
                        </div>
                      ))
                    ) : (
                      <p className="subtle">
                        Событий отказа нет. В режиме Pro можно исследовать отказ
                        аппарата и потерю радиосвязи.
                      </p>
                    )}
                  </div>
                )}
              </section>
            </>
          )}
          {page === "lab" && (
            <>
              <div className="lab-intro">
                <div className="lab-intro-icon">
                  <Cpu size={26} />
                </div>
                <div>
                  <h3>Обучайте. Сравнивайте. Проверяйте.</h3>
                  <p>
                    Локальное обучение модели управляющего поля по
                    математическому baseline. Loss и MAE — реальные результаты
                    каждого запуска.
                  </p>
                </div>
                <span className="small-badge">PYTORCH · CPU</span>
              </div>
              <div className="lab-layout">
                <section className="panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <Activity size={16} />
                      Обучение контроллера
                    </div>
                    <span
                      className={`status-tag ${run?.status === "running" ? "warning" : ""}`}
                    >
                      {labels[run?.status || ""] ||
                        run?.status ||
                        "Ожидание запуска"}
                    </span>
                  </div>
                  <div className="lab-chart">
                    <div className="chart-title">
                      <div>
                        <span className="eyebrow">TRAINING HISTORY</span>
                        <h2>
                          {run?.history?.length || 0}
                          <small>
                            {" "}
                            / {String(
                              run?.config?.epochs || labConfig.epochs,
                            )}{" "}
                            эпох
                          </small>
                        </h2>
                      </div>
                      <div className="chart-key">
                        <span>
                          <i style={{ background: "#3bd8d0" }} />
                          Train loss
                        </span>
                        <span>
                          <i style={{ background: "#7187ff" }} />
                          Validation loss
                        </span>
                      </div>
                    </div>
                    <Chart
                      data={run?.history || []}
                      keys={["train_loss", "val_loss"]}
                      height={250}
                    />
                    <div className="lab-metrics">
                      <div>
                        <span>Train loss</span>
                        <strong>
                          {fmt(run?.history?.at(-1)?.train_loss, 5)}
                        </strong>
                      </div>
                      <div>
                        <span>Validation loss</span>
                        <strong>
                          {fmt(run?.history?.at(-1)?.val_loss, 5)}
                        </strong>
                      </div>
                      <div>
                        <span>MAE</span>
                        <strong>{fmt(run?.history?.at(-1)?.mae, 5)}</strong>
                      </div>
                      <div>
                        <span>Время эпохи</span>
                        <strong>
                          {fmt(run?.history?.at(-1)?.epoch_seconds, 2)} s
                        </strong>
                      </div>
                    </div>
                  </div>
                  <div className="panel-header">
                    <div className="section-title">Журнал обучения</div>
                    <span className="mono subtle">
                      {run?.id?.slice(0, 16) || "—"}
                    </span>
                  </div>
                  <div className="terminal">
                    {run?.logs?.length ? (
                      run.logs.map((l, i) => (
                        <div key={i}>
                          <span>{String(i + 1).padStart(3, "0")}</span>
                          {typeof l === "string" ? l : JSON.stringify(l)}
                        </div>
                      ))
                    ) : (
                      <div>
                        <span>001</span>Готов к запуску локального обучения.
                      </div>
                    )}
                    {run?.error && <div className="red">{run.error}</div>}
                  </div>
                </section>
                <aside className="panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <Settings2 size={15} />
                      Конфигурация обучения
                    </div>
                  </div>
                  <div className="config-body">
                    <Field label="Начальная модель">
                      <select
                        value={labConfig.checkpoint || ""}
                        onChange={(e) =>
                          setLabConfig({
                            ...labConfig,
                            checkpoint: e.target.value || null,
                          })
                        }
                      >
                        <option value="">Новая модель</option>
                        <option value="bundled">Встроенная модель</option>
                        {models
                          .filter((m) => m.id !== "bundled")
                          .map((m) => (
                            <option key={String(m.id)} value={String(m.id)}>
                              {String(m.name || m.id)}
                            </option>
                          ))}
                      </select>
                    </Field>
                    <div className="two-fields">
                      <Field label="Эпохи">
                        <input
                          type="number"
                          min="1"
                          max="50"
                          value={labConfig.epochs}
                          onChange={(e) =>
                            setLabConfig({
                              ...labConfig,
                              epochs: +e.target.value,
                            })
                          }
                        />
                      </Field>
                      <Field label="Batch size">
                        <input
                          type="number"
                          min="1"
                          max="64"
                          value={labConfig.batch_size}
                          onChange={(e) =>
                            setLabConfig({
                              ...labConfig,
                              batch_size: +e.target.value,
                            })
                          }
                        />
                      </Field>
                    </div>
                    <Field label="Learning rate">
                      <input
                        type="number"
                        min="0.000001"
                        max="0.1"
                        step="0.0001"
                        value={labConfig.learning_rate}
                        onChange={(e) =>
                          setLabConfig({
                            ...labConfig,
                            learning_rate: +e.target.value,
                          })
                        }
                      />
                    </Field>
                    <Field label="Оптимизатор">
                      <select
                        value={labConfig.optimizer}
                        onChange={(e) =>
                          setLabConfig({
                            ...labConfig,
                            optimizer: e.target.value,
                          })
                        }
                      >
                        {["AdamW", "Adam", "SGD"].map((s) => (
                          <option key={s}>{s}</option>
                        ))}
                      </select>
                    </Field>
                    <div className="two-fields">
                      <Field label="Образцов">
                        <input
                          type="number"
                          min="16"
                          max="1024"
                          value={labConfig.samples}
                          onChange={(e) =>
                            setLabConfig({
                              ...labConfig,
                              samples: +e.target.value,
                            })
                          }
                        />
                      </Field>
                      <Field label="Сетка">
                        <select
                          value={labConfig.grid_size}
                          onChange={(e) =>
                            setLabConfig({
                              ...labConfig,
                              grid_size: +e.target.value,
                            })
                          }
                        >
                          {[24, 32, 48].map((n) => (
                            <option key={n} value={n}>
                              {n} × {n}
                            </option>
                          ))}
                        </select>
                      </Field>
                    </div>
                    <button
                      className="primary-button"
                      disabled={
                        busy ||
                        profile?.training_allowed === false ||
                        ["running", "queued", "stopping"].includes(
                          run?.status || "",
                        )
                      }
                      onClick={train}
                    >
                      <Play size={15} />
                      {profile?.training_allowed === false
                        ? "Обучение доступно локально"
                        : "Обучить модель"}
                    </button>
                    {run &&
                      ["running", "queued", "stopped"].includes(run.status) && (
                        <button
                          className="secondary-button"
                          onClick={async () => {
                            try {
                              setRun(
                                await api<Run>(
                                  `/api/lab/runs/${run.id}/${run.status === "stopped" ? "resume" : "stop"}`,
                                  {},
                                ),
                              );
                            } catch (e) {
                              setError((e as Error).message);
                            }
                          }}
                        >
                          {run.status === "stopped" ? (
                            <Play size={14} />
                          ) : (
                            <Square size={13} />
                          )}{" "}
                          {run.status === "stopped"
                            ? "Продолжить обучение"
                            : "Остановить после эпохи"}
                        </button>
                      )}
                    {run?.status === "completed" && (
                      <button
                        className="secondary-button"
                        onClick={() => {
                          setConfig({
                            ...config,
                            controller: "neural",
                            model_id: run.id,
                          });
                          setPage("mission");
                          setToast(
                            "Обученная модель выбрана для следующей миссии",
                          );
                        }}
                      >
                        <Check size={14} />
                        Использовать модель в миссии
                      </button>
                    )}
                    <div className="config-note">
                      <Cpu size={16} />
                      <p>
                        Обучение выполняется на вашем компьютере и сохраняет
                        checkpoint локально.
                      </p>
                    </div>
                  </div>
                </aside>
              </div>
              <section className="panel">
                <div className="panel-header">
                  <div className="section-title">
                    <Box size={16} />
                    История запусков
                  </div>
                  <span className="subtle">
                    {runs.length} запусков · {models.length} моделей
                  </span>
                </div>
                <div className="run-list">
                  {runs.length ? (
                    runs.map((r) => (
                      <button
                        className={run?.id === r.id ? "selected" : ""}
                        key={r.id}
                        onClick={() => setRun(r)}
                      >
                        <div>
                          <Cpu size={18} />
                          <strong>{r.id.slice(0, 20)}</strong>
                          <span className="subtle">
                            {r.history?.length || 0} эпох
                          </span>
                        </div>
                        <span className="status-tag">
                          {labels[r.status] || r.status}
                        </span>
                        <ChevronRight size={16} />
                      </button>
                    ))
                  ) : (
                    <div className="table-empty">
                      Первый запуск появится здесь после старта обучения.
                    </div>
                  )}
                </div>
              </section>
            </>
          )}
          {page === "edge" && (
            <>
              <div className="edge-architecture panel">
                <div className="ground-node">
                  <div className="node-icon">
                    <Radio size={25} />
                  </div>
                  <div>
                    <strong>GROUND STATION</strong>
                    <span>Планирование · Mission Package · Аналитика</span>
                  </div>
                  <span className="status-tag">
                    <i />
                    Локальный сервер
                  </span>
                </div>
                <div
                  className={`connection-line ${device && !device.connectivity.ground_radio ? "lost" : ""}`}
                >
                  <i />
                  <span>
                    {device
                      ? device.connectivity.ground_radio
                        ? "Радиоканал доступен"
                        : "Радиоканал потерян"
                      : "Загрузите пакет на виртуальный борт"}
                  </span>
                  <i />
                </div>
                <div className="edge-node">
                  <div className="node-icon">
                    <Cpu size={25} />
                  </div>
                  <div>
                    <strong>AGMEN EDGE</strong>
                    <span>Контроллер · Локальная карта · Журнал</span>
                  </div>
                  <span className="small-badge">VIRTUAL LINUX</span>
                </div>
              </div>
              <div className="edge-layout">
                <section className="panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <Cpu size={16} />
                      Бортовые вычислители
                    </div>
                    <span className="subtle">
                      {edge.devices.length} устройств
                    </span>
                  </div>
                  {!edge.devices.length ? (
                    <div className="edge-empty">
                      <Box size={38} />
                      <h3>Миссия начинается с пакета</h3>
                      <p>
                        Выберите рассчитанную миссию, подготовьте автономный
                        пакет
                        <br />и загрузите его в виртуальный бортовой
                        вычислитель.
                      </p>
                    </div>
                  ) : (
                    <>
                      <div className="device-tabs">
                        {edge.devices.map((d) => (
                          <button
                            key={d.id}
                            className={device?.id === d.id ? "selected" : ""}
                            onClick={() => setEdgeDevice(d.id)}
                          >
                            <Cpu size={15} />
                            {d.id.slice(0, 18)}
                            <i
                              className={
                                d.connectivity.ground_radio ? "online" : "lost"
                              }
                            />
                          </button>
                        ))}
                      </div>
                      {device && (
                        <div className="device-detail">
                          <div className="device-title">
                            <div>
                              <span className="eyebrow">
                                VIRTUAL ONBOARD COMPUTER
                              </span>
                              <h2>{device.id.slice(0, 22)}</h2>
                            </div>
                            <span
                              className={`status-tag ${!device.connectivity.ground_radio ? "warning" : ""}`}
                            >
                              <i />
                              {device.mode ||
                                (!device.connectivity.ground_radio
                                  ? "LINK LOST"
                                  : device.connectivity.internet
                                    ? "ONLINE"
                                    : "OFFLINE")}
                            </span>
                          </div>
                          <div className="connectivity-grid">
                            {[
                              { id: "internet", label: "Интернет", icon: Wifi },
                              {
                                id: "ground_radio",
                                label: "Наземный радиоканал",
                                icon: Radio,
                              },
                              {
                                id: "peer_radio",
                                label: "Междроновая связь",
                                icon: Waypoints,
                              },
                              {
                                id: "gnss",
                                label: "Навигация",
                                icon: Crosshair,
                              },
                            ].map((c) => (
                              <button
                                disabled={
                                  edgeBusy || device.job_status === "running"
                                }
                                className={
                                  device.connectivity[
                                    c.id as keyof Device["connectivity"]
                                  ]
                                    ? "connected"
                                    : "disconnected"
                                }
                                key={c.id}
                                onClick={() =>
                                  edgeAction(
                                    () =>
                                      api(
                                        `/api/edge/devices/${device.id}/connectivity`,
                                        {
                                          [c.id]:
                                            !device.connectivity[
                                              c.id as keyof Device["connectivity"]
                                            ],
                                        },
                                      ),
                                    "Состояние канала изменено",
                                  )
                                }
                              >
                                <c.icon size={18} />
                                <span>
                                  {c.label}
                                  <small>
                                    {device.connectivity[
                                      c.id as keyof Device["connectivity"]
                                    ]
                                      ? "Доступен"
                                      : "Недоступен"}
                                  </small>
                                </span>
                                <span className="toggle-track">
                                  <i />
                                </span>
                              </button>
                            ))}
                          </div>
                          <div className="edge-readings">
                            <div>
                              <span>Исполнение</span>
                              <strong>
                                {(
                                  {
                                    running: "Выполняет задание",
                                    holding: "Удержание позиции",
                                    returning: "Возврат",
                                    returned: "Возвращён",
                                    completed: "Задание завершено",
                                    battery_depleted: "Заряд исчерпан",
                                  } as Record<string, string>
                                )[device.status] || device.status}
                              </strong>
                            </div>
                            <div>
                              <span>Контроллер</span>
                              <strong>
                                {device.controller_version || "—"}
                              </strong>
                            </div>
                            <div>
                              <span>Mission Package</span>
                              <strong className="cyan">
                                {device.package_loaded
                                  ? "Загружен и проверен"
                                  : "Не загружен"}
                              </strong>
                            </div>
                            <div>
                              <span>Локальный журнал</span>
                              <strong>{device.journal_entries} записей</strong>
                            </div>
                          </div>
                          <div
                            className={`telemetry-box ${device.telemetry_kind !== "confirmed" ? "stale" : ""}`}
                          >
                            <div>
                              <Radio size={16} />
                              <strong>
                                {labels[device.telemetry_kind] ||
                                  device.telemetry_kind}{" "}
                                телеметрия
                              </strong>
                              <span>
                                {fmt(device.telemetry_age_s, 1)} s симуляции
                              </span>
                            </div>
                            <p>
                              {device.telemetry
                                ? `X ${fmt(device.telemetry.x * 80, 2)} m  /  Y ${fmt(device.telemetry.y * 60, 2)} m  ·  Заряд ${fmt(device.telemetry.battery, 1)}%  ·  Шаг ${device.telemetry.step}`
                                : "Нет подтверждённых координат"}
                            </p>
                            {device.telemetry_kind !== "confirmed" && (
                              <small>
                                Координаты не обновляются без связи. Последняя
                                известная позиция не является текущей.
                              </small>
                            )}
                          </div>
                          <EdgeMap device={device} />
                          {device.error && (
                            <p className="edge-error">{device.error}</p>
                          )}
                          <div className="device-actions">
                            <button
                              className="primary-button"
                              disabled={
                                edgeBusy || device.job_status === "running"
                              }
                              onClick={() =>
                                edgeAction(
                                  () =>
                                    api(`/api/edge/devices/${device.id}/run`, {
                                      steps: 20,
                                    }),
                                  "Локальное исполнение запущено на 20 шагов",
                                )
                              }
                            >
                              <Play size={15} />
                              Выполнить 20 шагов
                            </button>
                            <button
                              className="secondary-button"
                              disabled={
                                edgeBusy || !device.connectivity.ground_radio
                              }
                              onClick={() =>
                                edgeAction(
                                  () =>
                                    api(
                                      `/api/edge/devices/${device.id}/sync`,
                                      {},
                                    ),
                                  "Телеметрия и журнал синхронизированы",
                                )
                              }
                            >
                              <RotateCcw size={15} />
                              Синхронизировать
                            </button>
                          </div>
                          <div className="edge-faults">
                            <button
                              className="secondary-button small"
                              disabled={
                                edgeBusy || device.job_status === "running"
                              }
                              onClick={() =>
                                edgeAction(
                                  () =>
                                    api(
                                      `/api/edge/devices/${device.id}/connectivity`,
                                      { internet: false },
                                    ),
                                  "Интернет отключён. Наземный радиоканал моделируется отдельно.",
                                )
                              }
                            >
                              <WifiOff size={14} />
                              Потеря интернета
                            </button>
                            <button
                              className="secondary-button small"
                              disabled={
                                edgeBusy || device.job_status === "running"
                              }
                              onClick={() =>
                                edgeAction(
                                  () =>
                                    api(
                                      `/api/edge/devices/${device.id}/connectivity`,
                                      {
                                        ground_radio: false,
                                        peer_radio: false,
                                      },
                                    ),
                                  "Смоделирована потеря радиосвязи",
                                )
                              }
                            >
                              <Unplug size={14} />
                              Потеря связи
                            </button>
                            <button
                              className="secondary-button small"
                              disabled={
                                edgeBusy || device.job_status === "running"
                              }
                              onClick={() =>
                                edgeAction(
                                  () =>
                                    api(
                                      `/api/edge/devices/${device.id}/connectivity`,
                                      { low_battery: true },
                                    ),
                                  "Смоделирован низкий заряд",
                                )
                              }
                            >
                              <Battery size={14} />
                              Низкий заряд
                            </button>
                          </div>
                          <EdgeHistory device={device} />
                          <div className="last-sync">
                            <Check size={14} />
                            Последняя синхронизация:{" "}
                            {device.last_sync_at
                              ? new Date(device.last_sync_at).toLocaleString(
                                  "ru-RU",
                                )
                              : "не выполнялась"}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </section>
                <aside className="panel">
                  <div className="panel-header">
                    <div className="section-title">
                      <Box size={16} />
                      Mission Package Builder
                    </div>
                  </div>
                  <div className="config-body">
                    <Field label="Исходная миссия">
                      <select
                        value={mission?.id || ""}
                        onChange={(e) => loadMission(e.target.value)}
                      >
                        <option value="" disabled>
                          Сначала рассчитайте миссию
                        </option>
                        {missions.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name} · {labels[m.status] || m.status}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Назначить аппарату">
                      <input
                        type="number"
                        min="1"
                        max={mission?.config?.drone_count || 500}
                        value={edgeDrone}
                        onChange={(e) => setEdgeDrone(+e.target.value)}
                      />
                    </Field>
                    <Field label="При потере связи">
                      <select
                        value={edgePolicy}
                        onChange={(e) => setEdgePolicy(e.target.value)}
                      >
                        <option value="hold">
                          Удерживать безопасную позицию
                        </option>
                        <option value="return">
                          Возврат к начальной точке
                        </option>
                        <option value="continue">
                          Продолжить разрешённое задание
                        </option>
                      </select>
                    </Field>
                    <Field label="Групповая координация">
                      <select
                        value={edgeCoord}
                        onChange={(e) => setEdgeCoord(e.target.value)}
                      >
                        <option value="independent">
                          Независимые назначенные зоны
                        </option>
                        <option value="peer">
                          Локальная междроновая связь
                        </option>
                      </select>
                    </Field>
                    <div className="package-contents">
                      <span>
                        <Check size={13} />
                        Карта и целевая плотность
                      </span>
                      <span>
                        <Check size={13} />
                        Контроллер и параметры миссии
                      </span>
                      <span>
                        <Check size={13} />
                        Ограничения и безопасная политика
                      </span>
                      <span>
                        <Check size={13} />
                        Версия и проверка целостности
                      </span>
                    </div>
                    <button
                      className="primary-button"
                      disabled={
                        edgeBusy || !mission || mission.status !== "completed"
                      }
                      onClick={() =>
                        edgeAction(async () => {
                          await api("/api/edge/packages", {
                            mission_id: mission!.id,
                            drone_id: edgeDrone,
                            policy: edgePolicy,
                            coordination: edgeCoord,
                          });
                        }, "Автономный пакет подготовлен")
                      }
                    >
                      <Box size={15} />
                      Подготовить автономную миссию
                    </button>
                    <p className="launch-caption">
                      Пакет исполняется локально без облака
                    </p>
                  </div>
                  <div className="panel-header">
                    <div className="section-title">Подготовленные пакеты</div>
                    <span className="subtle">{edge.packages.length}</span>
                  </div>
                  <div className="package-list">
                    {edge.packages.length ? (
                      edge.packages.map((p) => (
                        <div key={p.id}>
                          <div>
                            <Box size={15} />
                            <span>
                              <strong>{p.id.slice(0, 20)}</strong>
                              <small>
                                UAV-{p.drone_id} · Проверка при загрузке
                              </small>
                            </span>
                          </div>
                          <div>
                            <button
                              className="icon-button"
                              title="Скачать автономный пакет"
                              onClick={() =>
                                download(
                                  `/api/edge/packages/${p.id}/download`,
                                  `agmen-edge-${p.id}.zip`,
                                ).catch((e) => setError(e.message))
                              }
                            >
                              <ArrowDownToLine size={15} />
                            </button>
                            <button
                              className="text-button"
                              disabled={edgeBusy}
                              onClick={() =>
                                edgeAction(async () => {
                                  const d = await api<Device>(
                                    "/api/edge/devices",
                                    { package_id: p.id },
                                  );
                                  setEdgeDevice(d.id);
                                }, "Пакет загружен в виртуальный борт")
                              }
                            >
                              Загрузить <ArrowRight size={13} />
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="subtle">Пакеты ещё не созданы.</p>
                    )}
                  </div>
                </aside>
              </div>
              <div className="edge-notice">
                <ShieldCheck size={18} />
                <p>
                  <strong>Симуляция автономного исполнения</strong> · Интернет,
                  наземная радиосвязь, междроновая связь и навигация
                  моделируются независимо. Реальные БПЛА не подключены. При
                  полной потере связи борт использует только локальные данные.
                </p>
              </div>
            </>
          )}
          {page === "analytics" && (
            <>
              <section className="panel analytics-top">
                <div className="panel-header">
                  <div className="section-title">
                    <BarChart3 size={17} />
                    Результаты исследования
                  </div>
                  <select
                    value={mission?.id || ""}
                    onChange={(e) => loadMission(e.target.value)}
                  >
                    <option value="" disabled>
                      Выберите рассчитанную миссию
                    </option>
                    {missions.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="analytics-body">
                  <div className="analytics-summary">
                    <span className="eyebrow">DENSITY CONTROL PERFORMANCE</span>
                    <h2>{mission?.name || "Ваша первая миссия"}</h2>
                    <p>
                      {mission?.config
                        ? `${labels[mission.config.controller]} контроллер · ${mission.config.drone_count} аппаратов · seed ${mission.config.seed}`
                        : "Рассчитайте миссию, чтобы исследовать сходимость контроллера."}
                    </p>
                    <div className="summary-values">
                      <div>
                        <strong>{fmt(mission?.summary?.mae, 5)}</strong>
                        <span>Итоговая MAE</span>
                      </div>
                      <div>
                        <strong>
                          {fmt(mission?.summary?.improvement_pct, 1)}
                          <small>%</small>
                        </strong>
                        <span>Изменение к старту</span>
                      </div>
                      <div>
                        <strong>
                          {fmt(mission?.summary?.duration_s, 1)}
                          <small> s</small>
                        </strong>
                        <span>Время симуляции</span>
                      </div>
                    </div>
                  </div>
                  <div className="analytics-chart">
                    <div className="chart-title">
                      <h3>Сходимость распределения</h3>
                      <span className="chart-key">
                        <i style={{ background: "#3bd8d0" }} /> MAE
                      </span>
                    </div>
                    <Chart data={mission?.history || []} height={225} />
                  </div>
                </div>
                <div className="export-row">
                  <div>
                    <ArrowDownToLine size={17} />
                    <strong>Экспорт результатов</strong>
                    <span>Фактические данные и графики выбранной миссии</span>
                  </div>
                  <div>
                    {["json", "csv", "svg", "png", "pdf"].map((f) =>
                      mission ? (
                        <button
                          key={f}
                          className="secondary-button small"
                          onClick={() =>
                            download(
                              `/api/missions/${mission.id}/export?format=${f}`,
                              `agmen-${mission.id}.${f}`,
                            ).catch((e) => setError(e.message))
                          }
                        >
                          {f.toUpperCase()}
                          <ArrowDownToLine size={12} />
                        </button>
                      ) : (
                        <button
                          className="secondary-button small"
                          disabled
                          key={f}
                        >
                          {f.toUpperCase()}
                        </button>
                      ),
                    )}
                  </div>
                </div>
              </section>
              <div className="analytics-two">
                <section className="panel">
                  <div className="panel-header">
                    <div className="section-title">Вариационное расстояние</div>
                    <span className="subtle">TV distance</span>
                  </div>
                  <div className="small-chart">
                    <Chart
                      data={mission?.history || []}
                      keys={["tv_distance"]}
                      colors={["#7187ff"]}
                      height={180}
                    />
                  </div>
                </section>
                <section className="panel">
                  <div className="panel-header">
                    <div className="section-title">Вычислительная нагрузка</div>
                    <span className="subtle">ms / шаг</span>
                  </div>
                  <div className="small-chart">
                    <Chart
                      data={mission?.history || []}
                      keys={["compute_ms"]}
                      colors={["#e1b77e"]}
                      height={180}
                    />
                  </div>
                </section>
              </div>
              <section className="panel">
                <div className="panel-header">
                  <div className="section-title">Архив миссий</div>
                  <span className="subtle">
                    {missions.length} сохранено локально
                  </span>
                </div>
                <div className="fleet-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Миссия</th>
                        <th>Контроллер</th>
                        <th>Аппаратов</th>
                        <th>MAE</th>
                        <th>Состояние</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {missions.map((m) => (
                        <tr key={m.id}>
                          <td>
                            <strong>{m.name}</strong>
                          </td>
                          <td>{labels[m.config?.controller || ""] || "—"}</td>
                          <td>{m.config?.drone_count || "—"}</td>
                          <td className="mono">{fmt(m.summary?.mae, 5)}</td>
                          <td>
                            <span className="status-tag">
                              {labels[m.status] || m.status}
                            </span>
                          </td>
                          <td>
                            <button
                              className="text-button"
                              onClick={() => loadMission(m.id)}
                            >
                              Открыть <ArrowRight size={13} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!missions.length && (
                    <div className="table-empty">
                      Пока нет сохранённых миссий.
                    </div>
                  )}
                </div>
              </section>
              {pro && (
                <div className="analytics-two">
                  <section className="panel experiment-card">
                    <FlaskConical size={24} />
                    <h3>Сравнение контроллеров</h3>
                    <p>
                      Воспроизводимый эксперимент baseline, neural и hybrid с
                      одинаковой начальной конфигурацией.
                    </p>
                    <button
                      className="secondary-button"
                      disabled={
                        busy ||
                        ["queued", "running"].includes(
                          String(experiment?.status),
                        )
                      }
                      onClick={async () => {
                        setBusy(true);
                        try {
                          setExperiment(
                            await api("/api/experiments", {
                              counts: [config.drone_count],
                              controllers: (
                                boot?.controllers || [
                                  { id: "baseline", available: true },
                                ]
                              )
                                .filter((c) => c.available)
                                .map((c) => c.id),
                              steps: Math.min(config.steps, 80),
                              seed: config.seed,
                            }),
                          );
                          setToast("Эксперимент запущен и сохранён");
                        } catch (e) {
                          setError((e as Error).message);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      Запустить эксперимент <ArrowRight size={14} />
                    </button>
                    {experiment && <ResearchResult result={experiment} />}
                  </section>
                  <section className="panel experiment-card">
                    <Activity size={24} />
                    <h3>Оптимизация параметров</h3>
                    <p>
                      Численный подбор slot_gain на фиксированном baseline: 25
                      аппаратов, 60 шагов, исходный сценарий.
                    </p>
                    <button
                      className="secondary-button"
                      disabled={
                        busy ||
                        ["queued", "running"].includes(
                          String(optimization?.status),
                        )
                      }
                      onClick={async () => {
                        setBusy(true);
                        try {
                          setOptimization(
                            await api("/api/optimization", {
                              gains: [0.5, 1, 1.5],
                              steps: 60,
                              seed: config.seed,
                            }),
                          );
                          setToast("Оптимизация запущена и сохранена");
                        } catch (e) {
                          setError((e as Error).message);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      Рассчитать оптимизацию <ArrowRight size={14} />
                    </button>
                    {optimization && <ResearchResult result={optimization} />}
                  </section>
                </div>
              )}
            </>
          )}
          {page === "settings" && (
            <div className="settings-layout">
              <section className="panel settings-card">
                <span className="eyebrow">PERSONAL WORKSPACE</span>
                <h2>Интерфейс и окружение</h2>
                <div className="setting-row">
                  <div>
                    <strong>Цветовая тема</strong>
                    <p>Выбор сохраняется в этом браузере</p>
                  </div>
                  <div className="mode-switch">
                    <button
                      className={theme === "dark" ? "selected" : ""}
                      onClick={() => setTheme("dark")}
                    >
                      <Moon size={14} />
                      Тёмная
                    </button>
                    <button
                      className={theme === "light" ? "selected" : ""}
                      onClick={() => setTheme("light")}
                    >
                      <Sun size={14} />
                      Светлая
                    </button>
                  </div>
                </div>
                <div className="setting-row">
                  <div>
                    <strong>Профессиональный режим</strong>
                    <p>Seed, препятствия, события отказа и эксперименты</p>
                  </div>
                  <button
                    aria-label="Профессиональный режим"
                    className={`switch-button ${pro ? "on" : ""}`}
                    onClick={() => setPro(!pro)}
                  >
                    <i />
                  </button>
                </div>
                <div className="setting-row token-row">
                  <div>
                    <strong>Токен частного сервера</strong>
                    <p>Хранится только до закрытия вкладки</p>
                  </div>
                  <div className="token-input">
                    <input
                      aria-label="Токен частного сервера"
                      type="password"
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      placeholder="X-AGMEN-Token"
                      autoComplete="off"
                    />
                    <button
                      className="secondary-button small"
                      onClick={async () => {
                        sessionStorage.setItem("agmen-token", token);
                        try {
                          const b = await api<Bootstrap>("/api/bootstrap");
                          setBoot(b);
                          setMissions(b.missions || []);
                          setOnline(true);
                          setError("");
                          setToast("Соединение с сервером установлено");
                        } catch (e) {
                          setOnline(false);
                          setError((e as Error).message);
                        }
                      }}
                    >
                      Подключить
                    </button>
                  </div>
                </div>
                <div className="setting-row">
                  <div>
                    <strong>Сбросить параметры миссии</strong>
                    <p>Вернуться к исходной конфигурации</p>
                  </div>
                  <button
                    className="secondary-button small"
                    onClick={() => {
                      setConfig(defaults);
                      setZones([]);
                      setToast("Параметры миссии сброшены");
                    }}
                  >
                    <RotateCcw size={14} />
                    Сбросить
                  </button>
                </div>
              </section>
              <section className="panel settings-card">
                <span className="eyebrow">ENGINE INFORMATION</span>
                <h2>Локальный исследовательский контур</h2>
                <dl className="system-info">
                  <div>
                    <dt>Сервер</dt>
                    <dd className={online ? "cyan" : "red"}>
                      {online ? "Доступен" : "Нет соединения"}
                    </dd>
                  </div>
                  <div>
                    <dt>Профиль</dt>
                    <dd>
                      {typeof boot?.profile === "string"
                        ? boot.profile
                        : String(boot?.profile?.name || "local")}
                    </dd>
                  </div>
                  <div>
                    <dt>Координаты</dt>
                    <dd>Нормализованная 2D плоскость</dd>
                  </div>
                  <div>
                    <dt>Размер области</dt>
                    <dd>80 × 60 метров</dd>
                  </div>
                  <div>
                    <dt>Математика</dt>
                    <dd>Управление полем плотности</dd>
                  </div>
                  <div>
                    <dt>Автономный контур</dt>
                    <dd>AGMEN EDGE · локальный runtime</dd>
                  </div>
                  <div>
                    <dt>Физические аппараты</dt>
                    <dd>Не подключены</dd>
                  </div>
                </dl>
              </section>
            </div>
          )}
          <footer className="footer">
            <span>
              <span className="footer-mark">△</span>AGMEN
              <span className="footer-divider" />
              Research. Coordinate. Evolve.
            </span>
            <span>
              Локальная симуляция <i /> 2D density control
            </span>
          </footer>
        </main>
      </div>
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          {toast}
          <button onClick={() => setToast("")} aria-label="Закрыть уведомление">
            <X size={14} />
          </button>
        </div>
      )}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="help-modal panel"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close icon-button"
              onClick={() => setHelp(false)}
              aria-label="Закрыть справку"
            >
              <X size={19} />
            </button>
            <span className="eyebrow">AGMEN / MISSION CONTROL</span>
            <h2>
              Согласованное движение.
              <br />
              Измеримый результат.
            </h2>
            <p>
              AGMEN исследует распределение группы БПЛА через управление полем
              плотности. Все траектории и метрики вычисляет локальный
              Python-движок.
            </p>
            <ol>
              <li>
                Выберите сценарий и контроллер, настройте число аппаратов.
              </li>
              <li>При необходимости нарисуйте целевые области на поле.</li>
              <li>Нажмите «Рассчитать миссию» и исследуйте траектории.</li>
              <li>
                В AGMEN EDGE подготовьте пакет и проверьте автономное исполнение
                при потере связи.
              </li>
            </ol>
            <div className="config-note">
              <ShieldCheck size={17} />
              <p>
                Это исследовательская симуляция в локальной плоскости.
                Отображаемые заряд и связь являются моделируемой телеметрией.
              </p>
            </div>
            <button className="primary-button" onClick={() => setHelp(false)}>
              Начать исследование <ArrowRight size={15} />
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
function Metric({
  icon,
  label,
  value,
  suffix,
  detail,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  suffix: string;
  detail: string;
  tone: string;
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-label">
        <span>{label}</span>
        <div className="metric-icon">{icon}</div>
      </div>
      <div className="metric-value">
        {value}
        <span>{suffix}</span>
      </div>
      <div className="metric-detail">
        <i />
        {detail}
      </div>
    </article>
  );
}
function EdgeMap({ device }: { device: Device }) {
  const p = device.telemetry;
  const estimate = device.predicted_position;
  return (
    <div className="edge-map">
      <div className="edge-map-label">
        <Crosshair size={13} />
        Наблюдение наземной станции<span>80 × 60 m</span>
      </div>
      <svg
        viewBox="0 0 600 230"
        role="img"
        aria-label="Подтверждённая, устаревшая и прогнозируемая позиция аппарата"
      >
        <defs>
          <pattern
            id="edge-grid"
            width="35"
            height="25"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M35 0H0V25"
              stroke="var(--map-grid)"
              fill="none"
              strokeWidth=".6"
            />
          </pattern>
          <clipPath id="edge-clip">
            <rect x="20" y="10" width="560" height="200" />
          </clipPath>
        </defs>
        <rect
          x="20"
          y="10"
          width="560"
          height="200"
          fill="url(#edge-grid)"
          stroke="var(--map-border)"
        />
        <g clipPath="url(#edge-clip)">
          {p && (
            <g transform={`translate(${20 + p.x * 560},${10 + p.y * 200})`}>
              <circle
                r="9"
                fill={
                  device.telemetry_kind === "confirmed"
                    ? "#3bd8d020"
                    : "#d6a45c16"
                }
                stroke={
                  device.telemetry_kind === "confirmed" ? "#3bd8d0" : "#d6a45c"
                }
                strokeWidth="1.3"
              />
              <circle
                r="3"
                fill={
                  device.telemetry_kind === "confirmed" ? "#3bd8d0" : "#d6a45c"
                }
              />
            </g>
          )}
          {estimate && (
            <g
              transform={`translate(${20 + estimate.x * 560},${10 + estimate.y * 200})`}
            >
              <ellipse
                rx={Math.min(180, (estimate.uncertainty_m / 80) * 560)}
                ry={Math.min(85, (estimate.uncertainty_m / 60) * 200)}
                fill="#d6a45c08"
                stroke="#bc9c62"
                strokeDasharray="3 5"
                strokeWidth=".7"
              />
              <path
                d="M0 -8 6 5 0 2 -6 5Z"
                fill="none"
                stroke="#d5ad72"
                strokeDasharray="2 2"
                strokeWidth="1.2"
              />
            </g>
          )}
        </g>
        <text x="20" y="225" fontSize="8" fill="var(--muted)">
          0 / 0
        </text>
        <text x="545" y="225" fontSize="8" fill="var(--muted)">
          80 / 60 m
        </text>
        {!p && (
          <text
            x="300"
            y="115"
            textAnchor="middle"
            fontSize="11"
            fill="var(--muted)"
          >
            Подтверждённая телеметрия пока не получена
          </text>
        )}
      </svg>
      <div className="edge-map-legend">
        <span>
          <i
            className={
              device.telemetry_kind === "confirmed" ? "confirmed" : "stale"
            }
          />
          {device.telemetry_kind === "confirmed"
            ? "Подтверждённая позиция"
            : "Последняя известная позиция"}
        </span>
        {estimate && (
          <span>
            <i className="prediction" />
            Прогноз по последней скорости
          </span>
        )}
      </div>
      {estimate && (
        <p>
          Прогноз не является телеметрией. Область неопределённости:{" "}
          {fmt(estimate.uncertainty_m, 1)} m. Соблюдение препятствий для
          прогноза не гарантируется.
        </p>
      )}
    </div>
  );
}
function EdgeHistory({ device }: { device: Device }) {
  const rows = device.history || [];
  const modes = ["ONLINE", "OFFLINE", "LINK_LOST"];
  return (
    <div className="edge-history">
      <div className="edge-history-heading">
        <strong>Результаты локального симулятора</strong>
        <span>{rows.length} измерений</span>
      </div>
      <div className="fleet-table">
        <table>
          <thead>
            <tr>
              <th>Режим</th>
              <th>Шагов</th>
              <th>Средняя MAE *</th>
              <th>Выполнение</th>
            </tr>
          </thead>
          <tbody>
            {modes.map((mode) => {
              const hs = rows.filter(
                (h) => String(h.mode).replaceAll(" ", "_") === mode,
              );
              if (!hs.length) return null;
              const mae =
                hs.reduce((s, h) => s + Number(h.distribution_mae || 0), 0) /
                hs.length;
              return (
                <tr key={mode}>
                  <td>{mode.replaceAll("_", " ")}</td>
                  <td className="mono">{hs.length}</td>
                  <td className="mono">{fmt(mae, 5)}</td>
                  <td className="mono">
                    {fmt(Number(hs.at(-1)?.task_fraction || 0) * 100, 1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length && (
          <div className="table-empty">
            Выполните первый сегмент автономной миссии.
          </div>
        )}
      </div>
      <p className="edge-footnote">
        * Локальная оценка по собственному аппарату и доступным
        peer-наблюдениям. Не является полной плотностью роя. Выполнение —
        прогресс до назначенной цели.
      </p>
      <div className="edge-sync-stat">
        <Check size={13} />
        <span>
          На наземной станции: {device.ground_receipt?.journal_entries || 0}{" "}
          записей · Ожидают синхронизации: {device.sync_pending_entries || 0}
        </span>
      </div>
      {device.ground_receipt && (
        <button
          className="text-button"
          onClick={() =>
            download(
              `/api/edge/devices/${device.id}/journal`,
              `agmen-ground-journal-${device.id}.json`,
            ).catch(() => {})
          }
        >
          Скачать синхронизированный журнал <ArrowDownToLine size={12} />
        </button>
      )}
      <details>
        <summary>Бортовой журнал · {device.journal_entries} записей</summary>
        <div className="edge-journal">
          {device.journal?.slice(-8).map((j) => (
            <div key={j.seq}>
              <span className="mono">{String(j.step).padStart(3, "0")}</span>
              <span>{j.kind}</span>
              <p>{j.message}</p>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
function ResearchResult({ result }: { result: Record<string, unknown> }) {
  const rows = (result.results || []) as Record<string, unknown>[];
  const histories = rows.map(
    (r) => (r.history || []) as Record<string, number>[],
  );
  const chart = Array.from(
    { length: Math.max(0, ...histories.map((h) => h.length)) },
    (_, i) =>
      Object.fromEntries(histories.map((h, j) => [`s${j}`, h[i]?.mae || 0])),
  );
  return (
    <div className="experiment-result">
      <div className="experiment-result-status">
        <span
          className={`status-tag ${["queued", "running"].includes(String(result.status)) ? "warning" : ""}`}
        >
          {["queued", "running"].includes(String(result.status)) && (
            <LoaderCircle className="spin" size={10} />
          )}{" "}
          {labels[String(result.status)] || String(result.status)}
        </span>
        <span className="mono subtle">{String(result.id).slice(0, 12)}</span>
      </div>
      {Boolean(result.error) && <p className="red">{String(result.error)}</p>}
      {rows.length > 0 && (
        <>
          <Chart
            data={chart}
            keys={histories.map((_, i) => "s" + i)}
            colors={["#3bd8d0", "#7187ff", "#d3a96e", "#a58ed1"]}
            height={150}
          />
          <div className="fleet-table">
            <table>
              <thead>
                <tr>
                  <th>{rows[0].gain ? "slot_gain" : "Контроллер"}</th>
                  <th>MAE</th>
                  <th>Время, ms</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>
                      {r.gain
                        ? fmt(r.gain, 2)
                        : labels[String(r.controller)] || String(r.controller)}
                    </td>
                    <td className="mono">{fmt(r.mae, 5)}</td>
                    <td className="mono">{fmt(r.compute_total_ms, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Boolean(result.best) && (
            <p className="cyan">
              Лучший slot_gain:{" "}
              {fmt((result.best as Record<string, number>).gain, 2)} · MAE{" "}
              {fmt((result.best as Record<string, number>).mae, 5)}
            </p>
          )}
        </>
      )}
    </div>
  );
}
export default App;
