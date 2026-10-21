"""Build the installable local distribution without environments, data or secrets."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
if not (root / "frontend/dist/index.html").is_file():
    raise SystemExit("Build frontend first: npm --prefix frontend run build")
listed = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
paths = sorted({Path(name.decode("utf-8")) for name in listed.split(b"\0") if name})
release = root / "release"
release.mkdir(exist_ok=True)
archive = release / "AGMEN-Windows-Local.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for relative in paths:
        path = root / relative
        if not path.is_file() or any(part in {".git", ".venv", ".venv-clean", "node_modules", "data", "logs", "release", "__pycache__", ".pytest_cache", "test-results"} for part in relative.parts):
            continue
        if relative.name == ".env" or relative.suffix in {".log", ".pyc"}:
            continue
        output.write(path, "AGMEN/" + relative.as_posix())
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
(release / "AGMEN-Windows-Local.zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
print(f"{archive} ({archive.stat().st_size:,} bytes)\nSHA256 {checksum}")
