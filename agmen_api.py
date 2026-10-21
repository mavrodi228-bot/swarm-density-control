"""AGMEN's local, simulation-only HTTP application.

Run ``python -m uvicorn agmen_api:app --host 127.0.0.1 --port 8000``.
The service never imports or invokes the repository's MAVLink bridge.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hmac
import io
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agmen_simulation import CONTROLLERS, simulate
from territory import (MapStore, MapUpload, MapAnalysis, MapCalibration, RestrictedZone,
                       restricted_mask, mask_rectangles, initial_free_positions)

LOG = logging.getLogger("agmen")
ROOT = Path(__file__).resolve().parent
ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=0.02, le=.98)
    y: float = Field(ge=.02, le=.98)
    radius: float = Field(default=.12, ge=.02, le=.4)
    weight: float = Field(default=1, gt=0, le=100)


class MissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="Новая миссия", min_length=1, max_length=100)
    drone_count: int = Field(default=25, ge=5, le=500)
    controller: Literal["baseline", "neural", "hybrid"] = "baseline"
    model_id: str = Field(default="bundled", pattern=r"^(bundled|[a-f0-9]{32})$")
    scenario: Literal["distribution", "clusters", "patrol", "survey", "scan", "return"] = "distribution"
    steps: int = Field(default=100, ge=20, le=300)
    seed: int = Field(default=42, ge=0, le=2147483547)
    max_speed: float = Field(default=5, ge=.1, le=15)
    safe_distance: float = Field(default=1.5, ge=.25, le=8)
    obstacle: Literal["none", "central", "corridor"] = "central"
    targets: list[Target] | None = Field(default=None, min_length=1, max_length=32)
    map_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    width_m: float = Field(default=80, ge=10, le=10000)
    height_m: float = Field(default=60, ge=10, le=10000)
    restricted_zones: list[RestrictedZone] = Field(default_factory=list, max_length=32)


class EventConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["drone_failure", "link_loss", "target_change"]
    drone_id: int | None = Field(default=None, ge=1, le=500)
    step: int | None = Field(default=None, ge=1, le=299)
    targets: list[Target] | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_drone(self):
        if self.kind != "target_change" and self.drone_id is None:
            raise ValueError("drone_id is required for a vehicle event")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    counts: list[int] = Field(default=[10, 25, 50], min_length=1, max_length=6)
    controllers: list[Literal["baseline", "neural", "hybrid"]] = Field(default=["baseline"], min_length=1, max_length=3)
    steps: int = Field(default=60, ge=20, le=150)
    seed: int = Field(default=42, ge=0, le=2147483547)

    @model_validator(mode="after")
    def check_counts(self):
        if any(n < 5 or n > 500 for n in self.counts):
            raise ValueError("counts must be between 5 and 500")
        if len(set(self.counts)) != len(self.counts) or len(set(self.controllers)) != len(self.controllers):
            raise ValueError("counts and controllers must not contain duplicates")
        return self


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gains: list[float] = Field(default=[.5, 1, 1.5], min_length=1, max_length=8)
    steps: int = Field(default=60, ge=20, le=150)
    seed: int = Field(default=42, ge=0, le=2147483547)

    @model_validator(mode="after")
    def check_gains(self):
        if any(g < .1 or g > 3 for g in self.gains):
            raise ValueError("slot gains must be between 0.1 and 3")
        return self


def _inference(controller: str, model_id: str = "bundled"):
    if controller == "baseline":
        return None, "cpu"
    try:
        from ml_lab import get_inference_model
        result = get_inference_model(model_id)
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(503, "Neural runtime/checkpoint is unavailable. Select the mathematical baseline.") from exc
    if result is None:
        raise HTTPException(503, "Trained neural checkpoint is unavailable. Select the mathematical baseline.")
    return result


def _controllers():
    try:
        from ml_lab import model_available
        available = model_available()
    except (ImportError, OSError, RuntimeError):
        available = False
    return [{"id": k, "label": v, "available": k == "baseline" or available} for k, v in CONTROLLERS.items()]


def create_app(state_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="AGMEN Simulation API", version="1.0.0")
    configured_hosts = [host.strip() for host in os.getenv("AGMEN_ALLOWED_HOSTS", "").split(",") if host.strip()]
    # External deployment domains extend the local health-check hosts. Docker
    # and the desktop launcher must still reach the service through loopback.
    allowed_hosts = list(dict.fromkeys(["localhost", "127.0.0.1", "testserver", *configured_hosts]))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts, www_redirect=False)
    data_directory = Path(os.getenv("AGMEN_DATA_DIR", os.getenv("AGMEN_STATE_DIR", str(ROOT / "data"))))
    directory = Path(state_dir) if state_dir is not None else data_directory / "missions"
    directory.mkdir(parents=True, exist_ok=True)
    maps = MapStore((Path(state_dir) / "maps") if state_dir is not None else data_directory / "maps")
    app.state.maps = maps
    profile_name = os.getenv("AGMEN_PROFILE", "local").lower()
    if profile_name == "demo":
        profile_name = "public"
    if profile_name not in {"local", "private", "public"}:
        raise ValueError("AGMEN_PROFILE must be local, private, or public")
    token = os.getenv("AGMEN_API_TOKEN", os.getenv("AGMEN_TOKEN", ""))
    if profile_name == "private" and len(token) < 24:
        raise ValueError("Private profile requires AGMEN_API_TOKEN of at least 24 characters")
    limits = {"max_drones": 100 if profile_name == "public" else 500,
              "max_steps": 150 if profile_name == "public" else 300,
              "max_jobs": 3, "max_missions": 20}
    profile = {"name": profile_name, "simulation_only": True,
               "token_required": profile_name == "private", "limits": limits,
               "training_allowed": profile_name != "public"}
    lock = threading.RLock()
    plot_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agmen-sim")
    capacity = threading.BoundedSemaphore(limits["max_jobs"])
    records: dict[str, dict] = {}
    app.state.records, app.state.executor = records, executor

    def persist(record: dict):
        target = directory / f"{record['id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:60]:
        if not ID_PATTERN.fullmatch(path.stem) or path.stat().st_size > 120_000_000:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("id") != path.stem or record.get("kind") not in {"mission", "experiment", "optimization"}:
                continue
            if record.get("status") in {"queued", "running"}:
                record.update(status="failed", error="Server restarted during this job; submit a new run.")
                persist(record)
            records[record["id"]] = record
        except (ValueError, OSError):
            LOG.warning("Could not restore state file %s", path.name)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if profile_name == "private" and request.url.path != "/api/health":
                received = request.headers.get("x-agmen-token", "")
                if not received:
                    received = request.headers.get("authorization", "").removeprefix("Bearer ")
                if not hmac.compare_digest(received.encode("utf-8"), token.encode("utf-8")):
                    return JSONResponse({"detail": "Authentication required: provide X-AGMEN-Token."}, status_code=401)
            if request.method in {"POST", "PUT", "PATCH"}:
                origin = request.headers.get("origin")
                if origin:
                    accepted_origins = {str(request.base_url).rstrip("/")}
                    accepted_origins.update(value.strip().rstrip("/") for value in os.getenv("AGMEN_ALLOWED_ORIGINS", "").split(",") if value.strip())
                    if profile_name == "local":
                        accepted_origins.update({"http://localhost:5173", "http://127.0.0.1:5173"})
                    if origin.rstrip("/") not in accepted_origins:
                        return JSONResponse({"detail": "Cross-origin writes are not permitted."}, status_code=403)
                body_limit = 8 * 1024 * 1024 if request.method == "POST" and request.url.path == "/api/maps" else 256_000
                chunks, length = [], 0
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > body_limit:
                        return JSONResponse({"detail": f"Request body exceeds {body_limit} bytes."}, status_code=413)
                    chunks.append(chunk)
                request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    def get_record(identifier: str, kind="mission") -> dict:
        if not ID_PATTERN.fullmatch(identifier):
            raise HTTPException(404, "Run not found")
        with lock:
            record = records.get(identifier)
            if not record or record["kind"] != kind:
                raise HTTPException(404, "Run not found")
            return record

    app.state.get_mission = lambda identifier: get_record(identifier, "mission")

    def brief(record: dict) -> dict:
        return {key: record.get(key) for key in ("id", "name", "kind", "status", "created_at", "updated_at", "config", "summary", "error", "results", "best") if key in record}

    def check_config(config: dict):
        if config["drone_count"] > limits["max_drones"] or config["steps"] > limits["max_steps"]:
            raise HTTPException(422, f"{profile_name} profile permits at most {limits['max_drones']} drones and {limits['max_steps']} steps")
        _inference(config["controller"], config.get("model_id", "bundled"))
        if config.get("map_id"):
            try:
                snapshot = maps.get(config["map_id"])
            except FileNotFoundError as error:
                raise HTTPException(422, "Selected territory is unavailable; upload or select a saved territory") from error
            config.update(width_m=snapshot["width_m"], height_m=snapshot["height_m"], map_snapshot=snapshot)
        width, height = config["width_m"], config["height_m"]
        zones = config.get("restricted_zones") or []
        for zone in zones:
            if zone["kind"] == "circle":
                rx, ry = zone["radius"] * min(width, height) / width, zone["radius"] * min(width, height) / height
                if zone["x"] < rx or zone["x"] + rx > 1 + 1e-9 or zone["y"] < ry or zone["y"] + ry > 1 + 1e-9:
                    raise HTTPException(422, "Circle restriction must fit inside the calibrated territory")
        if zones:
            from fleet_core import obstacle_mask, obstacle_rectangles
            from agmen_simulation import GRID, OBSTACLES
            custom = restricted_mask(zones, GRID, width, height)
            combined = custom | obstacle_mask(GRID, OBSTACLES[config["obstacle"]])
            if combined.all():
                raise HTTPException(422, "Restrictions cover the entire territory")
            for target in config.get("targets") or []:
                if custom[min(GRID-1, int(target["y"]*GRID)), min(GRID-1, int(target["x"]*GRID))]:
                    raise HTTPException(422, "A target lies inside a restricted grid cell; move the target or change the restriction")
            try:
                rects = obstacle_rectangles(OBSTACLES[config["obstacle"]], width, height) + mask_rectangles(custom, width, height)
                if len(rects) > 256:
                    raise ValueError("Restrictions exceed the planning budget; simplify the shapes")
                initial_free_positions(config["drone_count"], rects, width, height, config["seed"])
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

    def memory_admission():
        import psutil
        maximum = int(os.getenv("AGMEN_MAX_RSS_MB", "1536"))
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if rss_mb > maximum:
            raise HTTPException(503, "Memory admission limit reached; restart the app or increase AGMEN_MAX_RSS_MB")

    def submit_background(work):
        """Reserve the same bounded compute queue for local EDGE execution."""
        memory_admission()
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Simulation queue is full. Wait for an active run to finish.")
        def bounded():
            try:
                return work()
            finally:
                capacity.release()
        return executor.submit(bounded)

    app.state.submit_background = submit_background
    app.state.agmen_profile = profile

    def enqueue(kind: str, name: str, config: dict, work) -> dict:
        memory_admission()
        with lock:
            if sum(r["kind"] == kind for r in records.values()) >= limits["max_missions"]:
                raise HTTPException(409, "Saved run limit reached. Archive state files before starting further runs.")
            if not capacity.acquire(blocking=False):
                raise HTTPException(429, "Simulation queue is full. Wait for an active run to finish.")
            identifier = uuid.uuid4().hex[:12]
            record = {"id": identifier, "kind": kind, "name": name, "status": "queued",
                      "config": config, "created_at": now(), "updated_at": now()}
            records[identifier] = record
            try:
                persist(record)
            except OSError:
                records.pop(identifier, None)
                capacity.release()
                raise HTTPException(507, "Unable to persist run; check disk space and state directory permissions")
            result = brief(record)
        submit(record, work)
        return result

    def submit(record: dict, work):
        def run():
            try:
                with lock:
                    record.update(status="running", updated_at=now())
                    persist(record)
                output = work()
                with lock:
                    record.update(output)
                    record.update(status="completed", updated_at=now())
                    persist(record)
            except Exception as exc:
                LOG.exception("AGMEN job %s failed", record["id"])
                with lock:
                    message = exc.detail if isinstance(exc, HTTPException) else str(exc)
                    record.update(status="failed", updated_at=now(), error=message[:500])
                    try:
                        persist(record)
                    except OSError:
                        LOG.exception("Unable to save failed run")
            finally:
                capacity.release()
        executor.submit(run)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "app": "AGMEN", "service": "AGMEN", "version": "1.0.0", "simulation_only": True, "profile": profile_name}

    @app.get("/api/bootstrap")
    def bootstrap():
        return {"controllers": _controllers(), "profile": profile,
                "scenarios": [{"id": i, "label": label} for i, label in [
                    ("distribution", "Распределение"), ("clusters", "Два кластера"), ("patrol", "Патрулирование"),
                    ("survey", "Обследование"), ("scan", "Сканирование"), ("return", "Возврат")]],
                "missions": list_missions()}

    @app.get("/api/maps")
    def list_maps():
        return maps.list()

    @app.post("/api/maps", status_code=201)
    def upload_map(upload: MapUpload):
        try:
            return maps.create(upload)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except OSError as error:
            raise HTTPException(507, "Unable to save territory; check disk space and data directory") from error

    @app.get("/api/maps/{identifier}")
    def get_map(identifier: str):
        try:
            return maps.get(identifier)
        except FileNotFoundError as error:
            raise HTTPException(404, "Territory not found") from error

    @app.get("/api/maps/{identifier}/image")
    def map_image(identifier: str):
        get_map(identifier)
        return FileResponse(maps.image_path(identifier), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/maps/{identifier}/analyze")
    def analyze_map(identifier: str, settings: MapAnalysis):
        try:
            return maps.analyze(identifier, settings)
        except FileNotFoundError as error:
            raise HTTPException(404, "Territory not found") from error
        except OSError as error:
            raise HTTPException(507, "Unable to update territory analysis") from error

    @app.post("/api/maps/{identifier}/calibrate")
    def calibrate_map(identifier: str, dimensions: MapCalibration):
        try:
            return maps.calibrate(identifier, dimensions)
        except FileNotFoundError as error:
            raise HTTPException(404, "Territory not found") from error
        except OSError as error:
            raise HTTPException(507, "Unable to save territory dimensions") from error

    @app.get("/api/missions")
    def list_missions():
        with lock:
            return [brief(r) for r in sorted(records.values(), key=lambda r: r["created_at"], reverse=True) if r["kind"] == "mission"]

    @app.post("/api/missions", status_code=202)
    def create_mission(config: MissionConfig):
        data = config.model_dump()
        check_config(data)
        return enqueue("mission", config.name, data, lambda: simulate(data, *_inference(config.controller, config.model_id)))

    @app.get("/api/missions/{identifier}")
    def mission(identifier: str):
        with lock:
            return dict(get_record(identifier))

    @app.post("/api/missions/{identifier}/events", status_code=202)
    def inject_event(identifier: str, event: EventConfig):
        memory_admission()
        with lock:
            record = get_record(identifier)
            if record["status"] != "completed":
                raise HTTPException(409, "Wait for the mission to complete before replaying an event")
            data = record["config"]
            event_data = event.model_dump(exclude_none=True)
            event_data.setdefault("step", max(1, data["steps"] // 2))
            if event_data["step"] >= data["steps"] or event_data.get("drone_id", 1) > data["drone_count"]:
                raise HTTPException(422, "Event step or vehicle id is outside this mission")
            if len(record.get("events", [])) >= 16:
                raise HTTPException(422, "At most 16 fault/target events are permitted")
            events = [*record.get("events", []), event_data]
            if not capacity.acquire(blocking=False):
                raise HTTPException(429, "Simulation queue is full")
            previous_events = record.get("events", [])
            previous_updated = record["updated_at"]
            record.update(status="queued", updated_at=now(), events=events)
            try:
                persist(record)
            except OSError as exc:
                record.update(status="completed", updated_at=previous_updated, events=previous_events)
                capacity.release()
                raise HTTPException(507, "Unable to persist event replay; check disk space") from exc
            response = brief(record)
        submit(record, lambda: simulate(data, *_inference(data["controller"], data.get("model_id", "bundled")), events=events))
        return response

    @app.get("/api/missions/{identifier}/export")
    def export(identifier: str, format: Literal["json", "csv", "png", "svg", "pdf"] = "json"):
        record = get_record(identifier)
        if record["status"] != "completed":
            raise HTTPException(409, "A completed run is required for export")
        filename = f"agmen-{identifier}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        if format == "json":
            return Response(json.dumps(record, ensure_ascii=False, allow_nan=False), media_type="application/json", headers=headers)
        if format == "csv":
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(record["history"][0]))
            writer.writeheader()
            writer.writerows(record["history"])
            return Response("\ufeff" + stream.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)
        with plot_lock:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            frames, history = record["frames"], record["history"]
            width, height = record["workspace"]["width_m"], record["workspace"]["height_m"]
            axes[0].imshow(frames[-1]["density"], origin="upper", extent=[0, width, height, 0], cmap="Blues", aspect="auto")
            for i in range(record["config"]["drone_count"]):
                axes[0].plot([f["agents"][i]["x"] * width for f in frames], [f["agents"][i]["y"] * height for f in frames], alpha=.35, linewidth=.65)
            for rect in record["obstacle_rects"]:
                from matplotlib.patches import Rectangle
                axes[0].add_patch(Rectangle((rect["x"]*width, rect["y"]*height), rect["width"]*width, rect["height"]*height, color="#333333"))
            axes[0].set(xlabel="X (m)", ylabel="Y (m)", title="Simulated trajectories and final density")
            axes[1].plot([h["time"] for h in history], [h["tv_distance"] for h in history], label="Total variation distance")
            axes[1].set(xlabel="Simulation time (s)", ylabel="Density error", title=f"{record['config']['controller']} / {record['config']['drone_count']} virtual UAVs")
            axes[1].grid(alpha=.2)
            axes[1].legend()
            fig.suptitle("AGMEN · Simulation only · Battery and links are modeled", fontsize=11)
            fig.tight_layout()
            output = io.BytesIO()
            fig.savefig(output, format=format, dpi=150)
            plt.close(fig)
        return Response(output.getvalue(), media_type={"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[format], headers=headers)

    @app.get("/api/experiments")
    def experiments():
        with lock:
            return [brief(r) for r in records.values() if r["kind"] == "experiment"]

    @app.get("/api/experiments/{identifier}")
    def experiment(identifier: str):
        return get_record(identifier, "experiment")

    @app.post("/api/experiments", status_code=202)
    def create_experiment(config: ExperimentConfig):
        if profile_name == "public" and (max(config.counts) > 100 or len(config.counts)*len(config.controllers) > 6 or config.steps > 80):
            raise HTTPException(422, "Public experiments permit at most 100 drones, 6 combinations, and 80 steps")
        for controller in config.controllers:
            _inference(controller)
        def work():
            results = []
            for count in config.counts:
                for controller in config.controllers:
                    data = MissionConfig(drone_count=count, controller=controller, steps=config.steps, seed=config.seed).model_dump()
                    result = simulate(data, *_inference(controller), include_frames=False)
                    results.append({"drone_count": count, "controller": controller, **result["summary"], "history": result["history"]})
            return {"results": results, "summary": {"combinations": len(results), "simulation_only": True}}
        return enqueue("experiment", "Сравнение контроллеров", config.model_dump(), work)

    @app.get("/api/optimization")
    def optimizations():
        with lock:
            return [brief(r) for r in records.values() if r["kind"] == "optimization"]

    @app.get("/api/optimization/{identifier}")
    def optimization(identifier: str):
        return get_record(identifier, "optimization")

    @app.post("/api/optimization", status_code=202)
    def optimize(config: OptimizationConfig):
        if profile_name == "public" and (len(config.gains) > 3 or config.steps > 80):
            raise HTTPException(422, "Public optimization permits at most 3 gains and 80 steps")
        def work():
            results = []
            for gain in config.gains:
                data = MissionConfig(steps=config.steps, seed=config.seed).model_dump()
                run = simulate(data, slot_gain=gain, include_frames=False)
                results.append({"gain": gain, **run["summary"], "history": run["history"]})
            best = min(results, key=lambda r: r["mae"])
            return {"results": results, "best": {"gain": best["gain"], "mae": best["mae"]},
                    "summary": {"method": "bounded grid search of original slot_gain", "objective": "final MAE", "evaluations": len(results)}}
        return enqueue("optimization", "Оптимизация baseline", config.model_dump(), work)

    try:
        from ml_lab import router
        app.include_router(router)
    except ImportError as exc:
        LOG.warning("ML Laboratory is unavailable: %s", exc)

    try:
        from edge.api import router as edge_router
        app.include_router(edge_router)
    except ImportError as exc:
        LOG.warning("EDGE module is unavailable: %s", exc)

    dist = ROOT / "frontend" / "dist"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "API endpoint not found")
        resolved = (dist / path).resolve()
        if resolved.is_relative_to(dist.resolve()) and resolved.is_file():
            return FileResponse(resolved)
        index = dist / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<h1>AGMEN API is running</h1><p>Build the frontend first: npm --prefix frontend install &amp;&amp; npm --prefix frontend run build.</p><a href='/docs'>API documentation</a>", status_code=503)

    return app


app = create_app()
