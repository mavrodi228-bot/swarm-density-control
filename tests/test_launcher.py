"""The local server must outlive its launching command, with no extra console."""
import os
import subprocess
import sys

import pytest

import launch_agmen


def test_background_launch_escapes_host_job_and_hides_console(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **options: calls.append((command, options)) or "child")
    with (tmp_path / "server.log").open("ab") as output:
        assert launch_agmen.start_background([sys.executable, "--version"], output) == "child"
    options = calls[0][1]
    assert options["stdin"] == subprocess.DEVNULL
    if os.name == "nt":
        assert options["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB
        assert options["creationflags"] & subprocess.DETACHED_PROCESS
        assert not options["creationflags"] & subprocess.CREATE_NEW_CONSOLE
    else:
        assert options["start_new_session"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows process lifetime regression")
def test_detached_child_survives_launcher_exit(tmp_path):
    import psutil
    marker = tmp_path / "alive.txt"
    # A separate launcher exits immediately. Its detached child writes afterwards.
    child_code = "import time;from pathlib import Path;time.sleep(.5);Path({!r}).write_text('alive');time.sleep(15)".format(str(marker))
    code = (
        "import launch_agmen,os,sys;"
        "output=open(os.devnull,'wb');"
        f"p=launch_agmen.start_background([sys.executable,'-c',{child_code!r}],output);"
        "print(p.pid,flush=True)"
    )
    launched = subprocess.run([sys.executable, "-c", code], cwd=launch_agmen.ROOT,
                              capture_output=True, text=True, timeout=10, check=True)
    child = psutil.Process(int(launched.stdout.strip()))
    try:
        import time
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.read_text() == "alive"
        assert child.is_running()
    finally:
        for process in reversed(child.children(recursive=True)):
            process.terminate()
        child.terminate()
        child.wait(timeout=5)
