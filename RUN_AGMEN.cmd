@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch_agmen.py %*
) else (
  py -3.12 launch_agmen.py %*
)
if errorlevel 1 (
  echo Install Python 3.12 from python.org if Python was not found.
  pause
)
