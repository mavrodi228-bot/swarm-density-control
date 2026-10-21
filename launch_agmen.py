"""Install once, start a local AGMEN server, and open the application."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def load_settings() -> None:
    settings = ROOT / ".env"
    if settings.exists():
        for line in settings.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name and not name.startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    os.environ.setdefault("AGMEN_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("AGMEN_PROFILE", "local")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["SWARM_ENABLE_REAL"] = "0"


def ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=2) as response:
            payload = json.load(response)
            return response.status == 200 and (payload.get("app") == "AGMEN" or payload.get("service", "").lower().startswith("agmen"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def prepare_runtime() -> Path:
    runtime = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not runtime.exists():
        if not (3, 11) <= sys.version_info[:2] <= (3, 13):
            raise RuntimeError("Use Python 3.12 (recommended), 3.11 or 3.13 for the CPU PyTorch runtime.")
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    requirements = ROOT / "requirements-runtime.txt"
    digest = hashlib.sha256(requirements.read_bytes() + (ROOT / "requirements-core.txt").read_bytes() + (ROOT / "constraints.txt").read_bytes()).hexdigest()
    marker = runtime.parent / ".agmen-installed"
    if not marker.exists() or marker.read_text(encoding="utf-8") != digest:
        print("Installing AGMEN dependencies. Internet is needed for this first installation.", flush=True)
        subprocess.run([str(runtime), "-m", "pip", "install", "-r", str(requirements)], cwd=ROOT, check=True)
        marker.write_text(digest, encoding="utf-8")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise RuntimeError("Built frontend is missing. Run npm ci and npm run build in frontend/.")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--foreground", action="store_true", help="Keep the server in this terminal")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    load_settings()
    port = int(os.getenv("PORT", "7860"))
    if not 1024 <= port <= 65535:
        raise ValueError("PORT must be between 1024 and 65535")
    url = f"http://127.0.0.1:{port}"
    if args.serve:
        import uvicorn
        uvicorn.run("agmen_api:app", host=os.getenv("HOST", "127.0.0.1"), port=port, workers=1, log_level="info")
        return 0
    runtime = prepare_runtime()
    if not ready(url):
        command = [str(runtime), str(ROOT / "launch_agmen.py"), "--serve"]
        if args.foreground:
            if not args.no_browser:
                def open_when_ready():
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        if ready(url):
                            webbrowser.open(url)
                            return
                        time.sleep(.3)
                threading.Thread(target=open_when_ready, daemon=True).start()
            return subprocess.call(command, cwd=ROOT)
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "agmen-server.log").open("ab") as output:
            options: dict = {"cwd": str(ROOT), "stdout": output, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(command, **options)
        (logs / "server.json").write_text(json.dumps({"pid": process.pid, "url": url}), encoding="utf-8")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if ready(url):
                break
            if process.poll() is not None:
                raise RuntimeError(f"Server stopped. See {logs / 'agmen-server.log'}")
            time.sleep(0.3)
        else:
            raise RuntimeError(f"Server did not become ready. See {logs / 'agmen-server.log'}")
    print(f"AGMEN is running: {url}")
    print("Stop with STOP_AGMEN.cmd, or use --foreground and Ctrl+C.")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"AGMEN: {error}", file=sys.stderr)
        raise SystemExit(1)
