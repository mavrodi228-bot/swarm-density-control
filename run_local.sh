#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python launch_agmen.py "$@"
fi
exec python3 launch_agmen.py "$@"
