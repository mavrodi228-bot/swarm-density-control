"""Standalone offline simulator entry point: python -m edge."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .package import load_package
from .runtime import EdgeRuntime


def main():
    parser = argparse.ArgumentParser(description="AGMEN EDGE local planar simulation; no aircraft transport")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check package schema and SHA256 member integrity")
    validate.add_argument("package", type=Path)
    run = commands.add_parser("run", help="Execute the preloaded controller locally and persist its journal")
    run.add_argument("package", type=Path)
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--steps", type=int, default=20)
    run.add_argument("--offline", action="store_true", help="Disable internet independently of the ground radio")
    run.add_argument("--link-lost", action="store_true", help="Disable ground and peer radios")
    arguments = parser.parse_args()
    try:
        package = load_package(arguments.package)
        if arguments.command == "validate":
            result = {"valid": True, "manifest": package["manifest"], "simulation_only": True}
        else:
            runtime = EdgeRuntime(package, arguments.state)
            if arguments.offline:
                runtime.set_connectivity(internet=False)
            if arguments.link_lost:
                runtime.set_connectivity(ground_radio=False, peer_radio=False)
            result = runtime.run_steps(arguments.steps)
            # Full history/journal are persisted in --state; keep stdout compact.
            result.pop("history", None)
            result.pop("journal", None)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"AGMEN EDGE: {error}\n")


if __name__ == "__main__":
    main()
