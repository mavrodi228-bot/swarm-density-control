"""HTTP adapter for local virtual EDGE computers; no hardware transport."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Literal
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from edge.package import build_package, load_package
from edge.runtime import EdgeRuntime

router = APIRouter(prefix="/api/edge", tags=["AGMEN EDGE"])
LOG = logging.getLogger("agmen.edge.api")
SAFE_ID = re.compile(r"^[a-f0-9]{12,32}$")
_INIT_LOCK = threading.Lock()


class PackageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mission_id: str = Field(pattern=r"^[a-f0-9]{12}$")
    drone_id: int = Field(ge=1, le=500)
    policy: Literal["hold", "return", "continue"] = "continue"
    coordination: Literal["independent", "peer"] = "independent"


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_id: str = Field(pattern=r"^[a-f0-9]{12,32}$")


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = Field(default=20, ge=1, le=300, strict=True)


class ConnectivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    internet: bool | None = None
    ground_radio: bool | None = None
    peer_radio: bool | None = None
    gnss: bool | None = None
    low_battery: bool | None = None


class Store:
    def __init__(self, directory: Path, profile: dict):
        self.directory, self.profile = directory, profile
        self.lock = threading.RLock()
        self.packages = {}
        self.devices = {}
        self.registrations = {}
        self.running = set()
        self.device_errors = {}
        self.receipts = {}
        self.errors = []
        self.package_dir = directory / "packages"
        self.device_dir = directory / "devices"
        self.ground_dir = directory / "ground_sync"
        self.package_dir.mkdir(parents=True, exist_ok=True)
        self.device_dir.mkdir(parents=True, exist_ok=True)
        self.ground_dir.mkdir(parents=True, exist_ok=True)
        for path in self.package_dir.glob("*.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                identifier = metadata["id"]
                if not SAFE_ID.fullmatch(identifier) or path.stem != identifier:
                    continue
                archive = self.package_dir / f"{identifier}.zip"
                load_package(archive)
                self.packages[identifier] = {**metadata, "path": str(archive)}
            except (KeyError, ValueError, OSError) as exc:
                self.errors.append(f"Invalid stored package {path.name}: {str(exc)[:120]}")
        for path in self.device_dir.glob("*/registration.json"):
            try:
                registration = json.loads(path.read_text(encoding="utf-8"))
                identifier, package_id = registration["id"], registration["package_id"]
                if not SAFE_ID.fullmatch(identifier) or path.parent.name != identifier or package_id not in self.packages:
                    continue
                runtime = EdgeRuntime(load_package(self.packages[package_id]["path"]), path.parent)
                self.devices[identifier] = runtime
                self.registrations[identifier] = registration
                receipt_path = self.ground_dir / f"{identifier}.json"
                if receipt_path.exists() and receipt_path.stat().st_size < 4_000_000:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    self.receipts[identifier] = {"received_at": receipt["received_at"], "step": receipt["snapshot"]["step"],
                                                 "synced_entries": receipt.get("synced_entries", 0),
                                                 "journal_entries": len(receipt["snapshot"].get("journal", []))}
            except (KeyError, ValueError, OSError) as exc:
                self.errors.append(f"Invalid stored device {path.parent.name}: {str(exc)[:120]}")

    def public_package(self, metadata):
        return {k: v for k, v in metadata.items() if k not in {"path", "arrays", "checkpoint"}}

    def package(self, identifier):
        if not SAFE_ID.fullmatch(identifier) or identifier not in self.packages:
            raise HTTPException(404, "Autonomous mission package not found")
        return self.packages[identifier]

    def device(self, identifier):
        if not SAFE_ID.fullmatch(identifier) or identifier not in self.devices:
            raise HTTPException(404, "Virtual EDGE computer not found")
        return self.devices[identifier]

    def snapshot(self, identifier):
        runtime = self.device(identifier)
        raw = runtime.snapshot()
        registration = self.registrations[identifier]
        metadata = self.packages[registration["package_id"]]
        telemetry = raw.get("last_confirmed_telemetry")
        stale = raw.get("telemetry_stale", not raw.get("connectivity", {}).get("ground_radio", True))
        step = raw.get("step", 0)
        telemetry_step = (telemetry or {}).get("step", 0)
        last_sync = raw.get("last_sync") or {}
        return {**raw, "id": identifier, "package_id": registration["package_id"],
                "mission_id": metadata["mission_id"], "drone_id": metadata["drone_id"],
                "mode": raw.get("mode", "ONLINE").replace("_", " "),
                "controller_version": metadata.get("manifest", {}).get("controller_version", "density-flux-1"),
                "telemetry": telemetry,
                "telemetry_kind": "last_known" if stale else "confirmed",
                "telemetry_age_s": max(0, step - telemetry_step) * .2,
                "last_sync_at": last_sync.get("time"),
                "package_loaded": True, "simulation_only": True,
                "error": self.device_errors.get(identifier),
                "ground_receipt": self.receipts.get(identifier),
                "synced_entries": (self.receipts.get(identifier) or {}).get("synced_entries", 0),
                "job_status": "running" if identifier in self.running else "idle"}

    def receive(self, identifier):
        """Copy acknowledged local logs/results to separate ground storage."""
        snapshot = self.snapshot(identifier)
        received_at = datetime.now(timezone.utc).isoformat()
        destination = self.ground_dir / f"{identifier}.json"
        previous = {}
        if destination.exists():
            previous = json.loads(destination.read_text(encoding="utf-8"))
        previous_snapshot = previous.get("snapshot", {})
        journal_by_seq = {event["seq"]: event for event in previous_snapshot.get("journal", [])}
        new_events = {event["seq"]: event for event in snapshot.get("journal", [])}
        synced_entries = len(set(new_events) - set(journal_by_seq))
        journal_by_seq.update(new_events)
        history_by_step = {row["step"]: row for row in previous_snapshot.get("history", [])}
        history_by_step.update({row["step"]: row for row in snapshot.get("history", [])})
        journal_truncated = len(journal_by_seq) > 16000
        snapshot["journal"] = [journal_by_seq[seq] for seq in sorted(journal_by_seq)[-16000:]]
        snapshot["history"] = [history_by_step[step] for step in sorted(history_by_step)[-4000:]]
        payload = {"device_id": identifier, "package_id": snapshot["package_id"],
                   "received_at": received_at, "transport": "simulated_ground_radio", "snapshot": snapshot,
                   "synced_entries": synced_entries, "journal_retention_limit": 16000, "journal_truncated": journal_truncated}
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
        self.receipts[identifier] = {"received_at": received_at, "step": snapshot["step"],
                                     "synced_entries": synced_entries, "journal_entries": len(snapshot["journal"])}


def _store(request: Request) -> Store:
    with _INIT_LOCK:
        store = getattr(request.app.state, "edge_store", None)
        if store is None:
            root = Path(os.getenv("AGMEN_DATA_DIR", os.getenv("AGMEN_STATE_DIR", str(Path(__file__).resolve().parent.parent / "data"))))
            store = Store(root / "edge", request.app.state.agmen_profile)
            request.app.state.edge_store = store
        return store


@router.get("/status")
def status(request: Request):
    store = _store(request)
    with store.lock:
        return {"simulation_only": True, "devices": [store.snapshot(i) for i in store.devices],
                "packages": [store.public_package(p) for p in store.packages.values()],
                "errors": store.errors, "architecture": "ground station + local virtual EDGE runtimes; standalone CLI available"}


@router.post("/packages", status_code=201)
def create_package(body: PackageConfig, request: Request):
    store = _store(request)
    mission = request.app.state.get_mission(body.mission_id)
    if mission["status"] != "completed":
        raise HTTPException(409, "Complete the simulation before preparing an autonomous mission package")
    if body.drone_id > mission["config"]["drone_count"]:
        raise HTTPException(422, "Vehicle id is outside the mission")
    with store.lock:
        maximum = 10 if store.profile["name"] == "public" else 100
        if len(store.packages) >= maximum:
            raise HTTPException(409, "Autonomous package storage limit reached")
        provisional = store.package_dir / f"{uuid.uuid4().hex}.zip"
        try:
            metadata = build_package(mission, body.drone_id, body.policy, body.coordination, provisional)
            identifier = metadata["id"]
            if not SAFE_ID.fullmatch(identifier):
                raise ValueError("Package builder produced an invalid identifier")
            destination = store.package_dir / f"{identifier}.zip"
            if destination != provisional:
                provisional.replace(destination)
            # Validate exactly the artifact that will be offered for download.
            load_package(destination)
            metadata = {**metadata, "path": str(destination), "created_at": datetime.now(timezone.utc).isoformat()}
            serializable = store.public_package(metadata)
            (store.package_dir / f"{identifier}.json").write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
            store.packages[identifier] = metadata
            return serializable
        except (ValueError, OSError) as exc:
            if provisional.exists():
                provisional.unlink()
            raise HTTPException(422, f"Unable to build autonomous package: {exc}") from exc


@router.get("/packages/{identifier}/download")
def download_package(identifier: str, request: Request):
    store = _store(request)
    metadata = store.package(identifier)
    return FileResponse(metadata["path"], media_type="application/zip", filename=f"agmen-edge-{identifier}.zip")


@router.post("/devices", status_code=201)
def create_device(body: DeviceConfig, request: Request):
    store = _store(request)
    metadata = store.package(body.package_id)
    with store.lock:
        maximum = 5 if store.profile["name"] == "public" else 30
        if len(store.devices) >= maximum:
            raise HTTPException(409, "Virtual EDGE computer limit reached")
        identifier = uuid.uuid4().hex
        directory = store.device_dir / identifier
        try:
            package = load_package(metadata["path"])
            runtime = EdgeRuntime(package, directory)
            registration = {"id": identifier, "package_id": body.package_id, "created_at": datetime.now(timezone.utc).isoformat()}
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
            store.devices[identifier] = runtime
            store.registrations[identifier] = registration
            return store.snapshot(identifier)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, f"Unable to load autonomous package: {exc}") from exc


@router.get("/devices/{identifier}")
def device(identifier: str, request: Request):
    store = _store(request)
    with store.lock:
        return store.snapshot(identifier)


@router.get("/devices/{identifier}/journal")
def ground_journal(identifier: str, request: Request):
    store = _store(request)
    with store.lock:
        store.device(identifier)
        path = store.ground_dir / f"{identifier}.json"
        if not path.exists():
            raise HTTPException(409, "No ground receipt yet; restore the ground radio and synchronize the device")
        return FileResponse(path, media_type="application/json", filename=f"agmen-edge-{identifier}-ground.json")


@router.post("/devices/{identifier}/run", status_code=202)
def run_device(identifier: str, body: RunConfig, request: Request):
    store = _store(request)
    with store.lock:
        runtime = store.device(identifier)
        if identifier in store.running:
            raise HTTPException(409, "This virtual computer is already running")
        if store.profile["name"] == "public" and body.steps > 50:
            raise HTTPException(422, "Public EDGE runs permit at most 50 steps per request")
        snapshot = runtime.snapshot()
        if store.profile["name"] == "public" and snapshot.get("step", 0) + body.steps > 1000:
            raise HTTPException(422, "Public EDGE computers permit at most 1000 cumulative steps")
        store.running.add(identifier)
        store.device_errors.pop(identifier, None)
        # Only available peer-radio observations are passed. Ground-server fleet
        # state and unobserved future positions are never supplied to a device.
        peer_observations = []
        if snapshot.get("connectivity", {}).get("peer_radio"):
            own = store.registrations[identifier]
            own_package = store.packages[own["package_id"]]
            for peer_id, peer_runtime in store.devices.items():
                if peer_id == identifier:
                    continue
                peer_package = store.packages[store.registrations[peer_id]["package_id"]]
                if peer_package["mission_id"] != own_package["mission_id"] or peer_package["drone_id"] == own_package["drone_id"]:
                    continue
                observed = peer_runtime.snapshot()
                if observed.get("connectivity", {}).get("peer_radio"):
                    peer_observations.append({"drone_id": peer_package["drone_id"], "position_m": observed["position_m"], "step": observed["step"]})
        def execute():
            try:
                runtime.run_steps(body.steps, peer_observations=peer_observations)
            except Exception as exc:
                LOG.exception("Virtual EDGE execution failed for %s", identifier)
                with store.lock:
                    store.device_errors[identifier] = str(exc)[:300]
            finally:
                with store.lock:
                    store.running.discard(identifier)
        try:
            request.app.state.submit_background(execute)
        except Exception:
            store.running.discard(identifier)
            raise
        return store.snapshot(identifier)


@router.post("/devices/{identifier}/connectivity")
def connectivity(identifier: str, body: ConnectivityConfig, request: Request):
    store = _store(request)
    with store.lock:
        runtime = store.device(identifier)
        if identifier in store.running:
            raise HTTPException(409, "Wait for the current local execution segment to finish")
        values = body.model_dump(exclude_none=True)
        low_battery = values.pop("low_battery", None)
        previous_sync = runtime.snapshot().get("last_sync")
        try:
            if low_battery is not None:
                runtime.set_battery(12.0 if low_battery else 100.0)
            if values:
                runtime.set_connectivity(**values)
            if runtime.snapshot().get("last_sync") != previous_sync:
                store.receive(identifier)
            return store.snapshot(identifier)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(507, "Ground receipt could not be saved; the local device retains its journal") from exc


@router.post("/devices/{identifier}/sync")
def synchronize(identifier: str, request: Request):
    store = _store(request)
    with store.lock:
        runtime = store.device(identifier)
        if identifier in store.running:
            raise HTTPException(409, "Wait for the current local execution segment to finish")
        try:
            runtime.sync()
            store.receive(identifier)
            return store.snapshot(identifier)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(507, "Ground receipt could not be saved; the local device retains its journal") from exc
