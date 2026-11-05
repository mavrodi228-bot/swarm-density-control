"""Generate reproducible real simulation packages and measured offline comparison."""
from pathlib import Path
import json
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agmen_simulation import simulate
from edge import build_package, EdgeRuntime, load_package

out = ROOT / "examples" / "edge"
out.mkdir(parents=True, exist_ok=True)
config = {"name": "EDGE independent offline example", "drone_count": 10, "controller": "baseline", "scenario": "distribution", "steps": 150, "seed": 42, "max_speed": 5., "safe_distance": 1.5, "obstacle": "none"}
mission = {"id": "000000000042", "name": config["name"], "status": "completed", "config": config, **simulate(config)}
for policy in ("continue", "hold", "return"):
    build_package(mission, 1, policy, "independent", out / f"mission-{policy}.zip")
comparison = {"simulation_only": True, "config": config, "scope": "One assigned virtual vehicle, original controller, no peer observations", "results": []}
with tempfile.TemporaryDirectory() as temporary:
    for mode in ("ONLINE", "OFFLINE", "LINK_LOST"):
        runtime = EdgeRuntime(load_package(out / "mission-continue.zip"), Path(temporary) / mode)
        runtime.run_steps(20)
        if mode != "ONLINE":
            runtime.set_connectivity(internet=False)
        if mode == "LINK_LOST":
            runtime.set_connectivity(ground_radio=False, peer_radio=False)
        started = time.perf_counter()
        result = runtime.run_steps(60)
        elapsed = (time.perf_counter() - started) * 1000
        comparison["results"].append({"mode": mode, "step": result["step"], "position_m": result["position_m"], "status": result["status"], "battery_model_percent": result["battery"], "metrics": result["metrics"], "journal_entries": result["journal_entries"], "compute_and_persist_ms": elapsed, "last_confirmed_step": result["last_confirmed_telemetry"]["step"], "telemetry_stale": result["telemetry_stale"]})
    positions = [row["position_m"] for row in comparison["results"]]
    comparison["same_independent_trajectory_endpoint"] = all(point == positions[0] for point in positions)
(out / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=True, indent=2))
