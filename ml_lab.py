"""Isolated, resumable synthetic-data training for the AGMEN AI Laboratory.

The API does not import PyTorch until inference or training is requested. Each
training run executes in a separate CPU subprocess, with one worker per data
folder. No model is promoted to live vehicle control by this module.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r"^[a-f0-9]{32}$")
ACTIVE_STATUSES = {"queued", "running", "stopping"}
_API_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()
_INFERENCE_CACHE = {}
_WORKER_PROCESSES = {}
router = APIRouter(prefix="/api/lab", tags=["AI Laboratory"])


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=5, ge=1, le=50, strict=True)
    learning_rate: float = Field(default=0.0018, ge=0.000001, le=0.1, allow_inf_nan=False)
    batch_size: int = Field(default=8, ge=1, le=64, strict=True)
    samples: int = Field(default=128, ge=16, le=1024, strict=True)
    optimizer: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    checkpoint: str | None = "bundled"
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    grid_size: Literal[24, 32, 48] = 24


def _now():
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    configured = os.environ.get("AGMEN_DATA_DIR") or os.environ.get("AGMEN_STATE_DIR")
    return (Path(configured).expanduser().resolve() if configured else ROOT / "data") / "lab"


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Training run not found")
    return _data_dir() / "runs" / run_id


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        _replace_file(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _replace_file(source, target):
    # Windows can briefly deny replacement while another process is reading
    # the destination. Keep atomic replacement and retry that transient lock.
    for attempt in range(20):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01 * (attempt + 1))


def _read_json(path: Path):
    # Atomic replacement prevents partial JSON, but Windows may also deny an
    # open while the worker replaces the file. Retry only that sharing lock;
    # missing files and malformed JSON must retain their distinct meanings.
    for attempt in range(20):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01 * (attempt + 1))
    return json.loads(text)


def _read_run(run_id: str):
    path = _run_dir(run_id) / "run.json"
    try:
        run = _read_json(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training run not found") from None
    except PermissionError as error:
        raise HTTPException(status_code=503, detail="Training state is temporarily locked. Retry shortly.") from error
    except (ValueError, UnicodeError) as error:
        raise HTTPException(status_code=500, detail="Training run state is corrupt; the saved file was preserved.") from error
    if not isinstance(run, dict) or run.get("id") != run_id or not isinstance(run.get("status"), str) or run["status"] not in ACTIVE_STATUSES | {"completed", "stopped", "failed"}:
        raise HTTPException(status_code=500, detail="Training run state is invalid; the saved file was preserved.")
    if run["status"] in ACTIVE_STATUSES and (path.parent / "stop.request").exists():
        run["status"] = "stopping"
    return run


def _log(run, message):
    run["logs"] = (run.get("logs", []) + [f"{_now()} {message}"])[-120:]
    run["updated_at"] = _now()


def _save_run(run):
    _atomic_json(_run_dir(run["id"]) / "run.json", run)


def _public_profile():
    return os.environ.get("AGMEN_PROFILE", "local").strip().lower() in {"demo", "public"}


def _require_training():
    if _public_profile():
        raise HTTPException(status_code=403, detail="Training is disabled in the public demo. Use the local application.")
    if importlib.util.find_spec("torch") is None:
        raise HTTPException(status_code=503, detail="PyTorch is not installed; the local AI training runtime is unavailable.")


def _memory_limit_mb():
    try:
        return max(128, int(os.environ.get("AGMEN_MAX_RSS_MB", "1536")))
    except ValueError:
        return 1536


def _rss_mb():
    """Best-effort current worker RSS without a mandatory monitoring package."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1048576
    else:
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576
        except (OSError, ValueError, IndexError):
            pass
    return None


def _memory_admission(config):
    # Conservative workload estimate; the runtime also checks actual RSS. This
    # is cooperative admission, not an operating-system memory guarantee.
    estimated_mb = 320 + config.batch_size * config.grid_size ** 2 * 2400 / 1048576
    if estimated_mb > _memory_limit_mb():
        raise HTTPException(status_code=422, detail="Training exceeds the configured memory budget. Reduce batch size or grid size.")


def model_available(model_id="bundled") -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        return _checkpoint_path(model_id).is_file()
    except HTTPException:
        return False


def get_inference_model(model_id="bundled"):
    """Load bundled or validated lab weights; never choose arbitrary file paths."""
    if not model_available(model_id):
        return None
    with _INFERENCE_LOCK:
        if model_id not in _INFERENCE_CACHE:
            import torch
            from model import load_model
            torch.set_num_threads(2)
            # Keep the inference cache bounded when users train many models.
            if len(_INFERENCE_CACHE) >= 3:
                _INFERENCE_CACHE.pop(next(iter(_INFERENCE_CACHE)))
            _INFERENCE_CACHE[model_id] = (load_model(_checkpoint_path(model_id), "cpu"), "cpu")
        return _INFERENCE_CACHE[model_id]


def _pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _release_slot(run_id):
    path = _data_dir() / "worker.lock"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_id") == run_id:
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def _recover_worker():
    for run_id, process in list(_WORKER_PROCESSES.items()):
        if process.poll() is not None:
            _WORKER_PROCESSES.pop(run_id, None)
    path = _data_dir() / "worker.lock"
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return
    run_id = owner.get("run_id", "")
    if _pid_alive(owner.get("pid")):
        # The worker writes its terminal status after computation ends, just
        # before interpreter shutdown. Allow a new run during that brief exit.
        try:
            if _read_run(run_id)["status"] not in ACTIVE_STATUSES:
                _release_slot(run_id)
        except HTTPException as error:
            if error.status_code != 404:
                raise
        return
    if RUN_ID.fullmatch(run_id):
        try:
            run = _read_run(run_id)
            if run["status"] in ACTIVE_STATUSES:
                run["status"] = "stopped" if (_run_dir(run_id) / "checkpoint.pt").is_file() else "failed"
                run["error"] = "Worker exited unexpectedly. Resume the saved checkpoint." if run["status"] == "stopped" else "Worker exited before saving a checkpoint."
                _log(run, run["error"])
                _save_run(run)
        except HTTPException as error:
            if error.status_code != 404:
                raise
    path.unlink(missing_ok=True)


def _claim_slot(run_id):
    _data_dir().mkdir(parents=True, exist_ok=True)
    _recover_worker()
    path = _data_dir() / "worker.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="A training worker is already active. Stop it before starting another run.") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "pid": os.getpid()}, stream)


def _launch(run):
    run_id = run["id"]
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stop.request").unlink(missing_ok=True)
    run["status"] = "queued"
    run["error"] = None
    _log(run, "CPU training worker queued (2 computation threads).")
    _save_run(run)
    try:
        with (run_dir / "worker.log").open("ab") as log:
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "worker", run_id],
                cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, **kwargs,
            )
        _WORKER_PROCESSES[process.pid] = process
        _atomic_json(_data_dir() / "worker.lock", {"run_id": run_id, "pid": process.pid})
    except Exception:
        run["status"] = "failed"
        run["error"] = "Could not start the training worker."
        _log(run, run["error"])
        _save_run(run)
        _release_slot(run_id)
        raise HTTPException(status_code=503, detail=run["error"]) from None
    return _read_run(run_id)


def _checkpoint_path(checkpoint):
    if checkpoint == "bundled":
        path = ROOT / "model_data" / "swarm_unet.pt"
    elif isinstance(checkpoint, str) and RUN_ID.fullmatch(checkpoint):
        source = _read_run(checkpoint)
        if source["status"] != "completed" or not source.get("validated"):
            raise HTTPException(status_code=422, detail="Select a completed, validated model checkpoint.")
        path = _run_dir(checkpoint) / "best.pt"
    else:
        raise HTTPException(status_code=422, detail="checkpoint must be bundled, a completed model ID, or null")
    if not path.is_file():
        raise HTTPException(status_code=422, detail="The selected model weights are unavailable.")
    return path


@router.get("/models")
def list_models():
    history_path = ROOT / "model_data" / "training_history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = []
    bundled = {
        "id": "bundled", "name": "SwarmUNet · bundled", "source": "bundled",
        "available": model_available(), "validated": False,
        "validation_note": "Original bundled weights; no held-out evaluation was supplied.",
        "metrics": {"train_loss": history[-1].get("loss")} if history else {},
        "history": history, "architecture": "SwarmUNet", "hardware_promoted": False,
    }
    models = [bundled]
    for run in list_runs():
        if run.get("validated") and (_run_dir(run["id"]) / "best.pt").is_file():
            models.append({
                "id": run["id"], "name": f"SwarmUNet · {run['id'][:8]}",
                "source": "laboratory", "available": True, "validated": True,
                "metrics": run.get("metrics", {}), "created_at": run["created_at"],
                "architecture": "SwarmUNet", "hardware_promoted": False,
            })
    return models


@router.get("/runs")
def list_runs():
    with _API_LOCK:
        _recover_worker()
        folder = _data_dir() / "runs"
        runs = []
        if folder.is_dir():
            for item in folder.iterdir():
                if item.is_dir() and RUN_ID.fullmatch(item.name):
                    try:
                        runs.append(_read_run(item.name))
                    except HTTPException as error:
                        if error.status_code != 404:
                            raise
        return sorted(runs, key=lambda run: run["created_at"], reverse=True)


@router.post("/runs", status_code=202)
def create_run(config: TrainingConfig):
    _require_training()
    _memory_admission(config)
    if config.checkpoint is not None:
        _checkpoint_path(config.checkpoint)
    with _API_LOCK:
        run_id = uuid.uuid4().hex
        _claim_slot(run_id)
        run = {
            "id": run_id, "status": "queued", "config": config.model_dump(),
            "history": [], "logs": [], "error": None, "model_id": None,
            "created_at": _now(), "updated_at": _now(), "metrics": {},
            "validated": False, "hardware_promoted": False,
            "progress": {"epoch": 0, "batch": 0},
        }
        return _launch(run)


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    with _API_LOCK:
        _recover_worker()
        return _read_run(run_id)


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    _require_training()
    with _API_LOCK:
        _recover_worker()
        run = _read_run(run_id)
        if run["status"] in ACTIVE_STATUSES:
            # The worker owns run.json while active; a separate marker avoids
            # racing an epoch update. The read API projects stopping below.
            (_run_dir(run_id) / "stop.request").touch()
            run["status"] = "stopping"
        return run


@router.post("/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: str):
    _require_training()
    with _API_LOCK:
        _recover_worker()
        run = _read_run(run_id)
        if run["status"] not in {"stopped", "failed"}:
            raise HTTPException(status_code=409, detail="Only a stopped or failed run can be resumed.")
        if not (_run_dir(run_id) / "checkpoint.pt").is_file():
            raise HTTPException(status_code=409, detail="This run has no resumable checkpoint.")
        _claim_slot(run_id)
        return _launch(run)


def _atomic_torch_save(torch, value, path):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temp)
        _replace_file(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _split_indices(samples, seed):
    """One seeded permutation yields strictly disjoint train/validation/test sets."""
    import numpy as np
    indices = np.random.default_rng(seed ^ 0xA6E3).permutation(samples).tolist()
    held_out = max(2, int(samples * 0.15))
    return indices[:samples - 2 * held_out], indices[samples - 2 * held_out:samples - held_out], indices[samples - held_out:]


def _loss(torch, prediction, truth):
    mse = torch.mean((prediction - truth) ** 2)
    dx = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]) - (truth[:, :, :, 1:] - truth[:, :, :, :-1])
    dy = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]) - (truth[:, :, 1:, :] - truth[:, :, :-1, :])
    return mse + 0.08 * 0.5 * (torch.mean(dx ** 2) + torch.mean(dy ** 2))


def _evaluate(torch, model, loader):
    model.eval()
    total_loss = total_mae = total_mse = 0.0
    examples = 0
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x)
            count = x.shape[0]
            total_loss += float(_loss(torch, prediction, y)) * count
            total_mae += float(torch.mean(torch.abs(prediction - y))) * count
            total_mse += float(torch.mean((prediction - y) ** 2)) * count
            examples += count
    return {"loss": total_loss / examples, "mae": total_mae / examples, "mse": total_mse / examples, "samples": examples}


def _train_worker(run_id):
    run = _read_run(run_id)
    run_dir = _run_dir(run_id)
    try:
        # Child process only: API/simulator responsiveness is independent of
        # model training, and native thread pools have a fixed small budget.
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset
        from model import SwarmUNet, load_model
        from train_model import SyntheticSwarmDataset

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        config = run["config"]
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        model = SwarmUNet()
        optimizer_type = getattr(torch.optim, config["optimizer"])
        optimizer_kwargs = {"lr": config["learning_rate"]}
        if config["optimizer"] == "AdamW":
            optimizer_kwargs["weight_decay"] = 1e-5
        elif config["optimizer"] == "SGD":
            optimizer_kwargs["momentum"] = 0.9
        optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        epoch = batch_offset = epoch_seen = 0
        epoch_loss_sum = 0.0
        best_val = float("inf")
        best_state = None
        elapsed_epoch = 0.0
        checkpoint_path = run_dir / "checkpoint.pt"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch, batch_offset = checkpoint["epoch"], checkpoint["batch_offset"]
            epoch_loss_sum, epoch_seen = checkpoint["epoch_loss_sum"], checkpoint["epoch_seen"]
            best_val, best_state = checkpoint["best_val"], checkpoint["best_model_state_dict"]
            elapsed_epoch = checkpoint.get("elapsed_epoch", 0.0)
            run["history"] = checkpoint["history"]
            torch.set_rng_state(checkpoint["torch_rng_state"])
            _log(run, f"Resumed epoch {epoch + 1}, batch {batch_offset}; optimizer and scheduler restored.")
        elif config["checkpoint"] is not None:
            model.load_state_dict(load_model(_checkpoint_path(config["checkpoint"])).state_dict())

        dataset = SyntheticSwarmDataset(config["samples"], config["grid_size"], config["seed"])
        train_ids, val_ids, test_ids = _split_indices(config["samples"], config["seed"])
        val_loader = DataLoader(Subset(dataset, val_ids), batch_size=config["batch_size"], num_workers=0)
        test_loader = DataLoader(Subset(dataset, test_ids), batch_size=config["batch_size"], num_workers=0)
        run["dataset"] = {"train_samples": len(train_ids), "validation_samples": len(val_ids), "test_samples": len(test_ids), "seed": config["seed"], "disjoint": True, "synthetic": True}
        run["status"] = "running"
        _log(run, f"Training on {len(train_ids)} synthetic examples; validation {len(val_ids)}; test {len(test_ids)}. CPU only.")
        _save_run(run)

        def checkpoint():
            _atomic_torch_save(torch, {
                "format_version": 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "best_model_state_dict": best_state, "best_val": best_val,
                "epoch": epoch, "batch_offset": batch_offset, "epoch_loss_sum": epoch_loss_sum,
                "epoch_seen": epoch_seen, "elapsed_epoch": elapsed_epoch,
                "history": run["history"], "torch_rng_state": torch.get_rng_state(),
                "config": config,
            }, checkpoint_path)

        checkpoint()
        while epoch < config["epochs"]:
            started = time.monotonic()
            generator = torch.Generator().manual_seed(config["seed"] + epoch * 101)
            train_loader = DataLoader(Subset(dataset, train_ids), batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
            model.train()
            last_progress_write = time.monotonic()
            for batch_index, (x, y) in enumerate(train_loader):
                if batch_index < batch_offset:
                    continue
                resident_mb = _rss_mb()
                memory_exceeded = resident_mb is not None and resident_mb > _memory_limit_mb()
                if (run_dir / "stop.request").exists() or memory_exceeded:
                    elapsed_epoch += time.monotonic() - started
                    checkpoint()
                    run["status"] = "stopped"
                    run["progress"] = {"epoch": epoch, "batch": batch_offset}
                    if memory_exceeded:
                        run["error"] = f"Worker exceeded the cooperative {_memory_limit_mb()} MiB RSS budget."
                    _log(run, run["error"] or "Stopped; model, optimizer, scheduler, best weights and batch position saved.")
                    _save_run(run)
                    return
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(torch, model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss; lower the learning rate.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                epoch_loss_sum += float(loss.detach()) * x.shape[0]
                epoch_seen += x.shape[0]
                batch_offset = batch_index + 1
                run["progress"] = {"epoch": epoch, "batch": batch_offset, "batches": len(train_loader)}
                run["worker"] = {"threads": 2, "rss_mb": round(resident_mb, 1) if resident_mb is not None else None, "max_rss_mb": _memory_limit_mb(), "memory_limit_kind": "cooperative"}
                if time.monotonic() - last_progress_write > 1:
                    _save_run(run)
                    last_progress_write = time.monotonic()
            validation = _evaluate(torch, model, val_loader)
            if not all(math.isfinite(value) for value in validation.values()):
                raise RuntimeError("Validation produced non-finite metrics.")
            if validation["loss"] < best_val:
                best_val = validation["loss"]
                best_state = copy.deepcopy(model.state_dict())
                _atomic_torch_save(torch, {"model_state_dict": best_state, "validation": validation, "epoch": epoch + 1, "config": config}, run_dir / "best.pt")
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            run["history"].append({
                "epoch": epoch + 1, "train_loss": epoch_loss_sum / epoch_seen,
                "val_loss": validation["loss"], "mae": validation["mae"],
                "lr": learning_rate, "epoch_seconds": elapsed_epoch + time.monotonic() - started,
            })
            _log(run, f"Epoch {epoch + 1}/{config['epochs']}: train {run['history'][-1]['train_loss']:.6f}, validation {validation['loss']:.6f}, MAE {validation['mae']:.6f}.")
            epoch += 1
            batch_offset = epoch_seen = 0
            epoch_loss_sum = elapsed_epoch = 0.0
            run["progress"] = {"epoch": epoch, "batch": 0}
            checkpoint()
            _save_run(run)
        model.load_state_dict(best_state)
        test = _evaluate(torch, model, test_loader)
        if not all(math.isfinite(value) for value in test.values()):
            raise RuntimeError("Test evaluation produced non-finite metrics.")
        run["metrics"] = {"best_val_loss": best_val, "test_loss": test["loss"], "test_mse": test["mse"], "test_mae": test["mae"], "test_samples": test["samples"], "synthetic": True}
        run["status"] = "completed"
        run["validated"] = True
        run["model_id"] = run_id
        _log(run, "Completed. Best validation checkpoint evaluated on held-out synthetic test data; no hardware promotion.")
        _save_run(run)
    except BaseException as error:
        run["status"] = "failed"
        # Do not expose local filesystem paths or native runtime traces via API.
        message = str(error) if isinstance(error, RuntimeError) and "non-finite" in str(error) else type(error).__name__
        run["error"] = f"Training worker failed: {message}. See local worker log for details."
        _log(run, run["error"])
        _save_run(run)
        import traceback
        traceback.print_exc()
    finally:
        _release_slot(run_id)


def main():
    parser = argparse.ArgumentParser(description="AGMEN isolated AI Laboratory worker")
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("run_id")
    arguments = parser.parse_args()
    _run_dir(arguments.run_id)
    _train_worker(arguments.run_id)


if __name__ == "__main__":
    main()
