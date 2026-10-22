"""Stop only the AGMEN process recorded by this project's launcher."""
import json
from pathlib import Path
import psutil

root = Path(__file__).resolve().parent
record = root / "logs" / "server.json"
if record.exists():
    try:
        server = psutil.Process(json.loads(record.read_text(encoding="utf-8"))["pid"])
        command = server.cmdline()
        launcher = str(root / "launch_agmen.py")
        if launcher not in command or "--serve" not in command:
            raise RuntimeError("PID belongs to another process; no process was stopped")
        children = server.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        server.terminate()
        _, alive = psutil.wait_procs([server, *children], timeout=8)
        for process in alive:
            process.kill()
        _, alive = psutil.wait_procs(alive, timeout=3)
        if alive:
            raise RuntimeError("Some AGMEN processes are still stopping; the PID record was preserved")
        print("AGMEN stopped. Saved missions and checkpoints are in data/.")
    except psutil.NoSuchProcess:
        print("AGMEN is already stopped.")
    record.unlink(missing_ok=True)
else:
    print("No server started by this launcher was found.")
