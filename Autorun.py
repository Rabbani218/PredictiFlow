#!/usr/bin/env python3
"""
Autorun.py

Orkestra setup & run helper untuk PredictiFlow.

Fitur utama:
- Membuat virtual environment di folder `.venv` (jika belum ada)
- Meng-upgrade pip/setuptools/wheel dan mencoba memasang semua requirements yang ditemukan
- Coba jalankan `docker compose up --build` bila Docker tersedia
- Jika Docker tidak tersedia, jalankan backend (uvicorn) dan frontend (streamlit) lokal dan monitoring proses
- Self-healing sederhana: retry, upgrade pip, retry install paket tertentu, dan restart proses yang crash beberapa kali

Catatan: skrip ini tidak menggantikan Docker/Conda untuk kasus pembangunan binary wheels (numpy). Jika pemasangan pip gagal untuk paket yang berat, skrip akan memberi saran untuk menggunakan Docker atau conda.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "autorun.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def run_cmd(cmd: List[str], cwd: Path | None = None, env=None, check: bool = True, timeout: int | None = None):
    logging.info("Run: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, cwd=cwd, env=env, check=check, capture_output=True, text=True, timeout=timeout)
        if res.stdout:
            logging.debug(res.stdout)
        if res.stderr:
            logging.debug(res.stderr)
        return res.returncode, res.stdout, res.stderr
    except subprocess.CalledProcessError as e:
        logging.error("Command failed (%s): %s", e.returncode, e.stderr)
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        logging.exception("Command error: %s", e)
        return 2, "", str(e)


def create_venv(path: Path = VENV_DIR):
    if path.exists():
        logging.info("Virtualenv already exists at %s", path)
        return
    logging.info("Creating virtualenv at %s", path)
    subprocess.check_call([sys.executable, "-m", "venv", str(path)])
    logging.info("Virtualenv created")


def get_venv_python(path: Path = VENV_DIR) -> str:
    if os.name == "nt":
        return str(path / "Scripts" / "python.exe")
    return str(path / "bin" / "python")


def upgrade_pip(python: str):
    logging.info("Upgrading pip, setuptools and wheel in venv")
    run_cmd([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_requirements(python: str, req_files: List[Path]):
    for f in req_files:
        if not f.exists():
            logging.info("Requirements file not found: %s — skip", f)
            continue
        logging.info("Installing from %s", f)
        code, out, err = run_cmd([python, "-m", "pip", "install", "-r", str(f)])
        if code != 0:
            logging.warning("pip install failed for %s (code=%s). Attempting heal steps.", f, code)
            # Healing steps
            upgrade_pip(python)
            logging.info("Retry install (prefer binary wheels where possible)")
            # Try again with only-binary for problematic packages (best-effort)
            code2, out2, err2 = run_cmd([python, "-m", "pip", "install", "-r", str(f), "--only-binary=:all:"])
            if code2 != 0:
                logging.error("Retry also failed for %s — see logs. Suggest using Docker or conda for heavy packages.", f)
            else:
                logging.info("Install succeeded on retry with --only-binary for %s", f)
        else:
            logging.info("Installed requirements from %s", f)


def start_process(cmd: List[str], cwd: Path | None = None, env=None) -> subprocess.Popen:
    logging.info("Starting process: %s", " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, env=env)
    return p


def run_docker_compose():
    if shutil.which("docker") is None or shutil.which("docker-compose") is None and shutil.which("docker") is None:
        logging.warning("Docker or docker-compose not found on PATH. Skipping docker-compose run.")
        return False
    # Prefer `docker compose` if supported
    if shutil.which("docker"):
        cmd = ["docker", "compose", "up", "--build", "--remove-orphans"]
    else:
        cmd = ["docker-compose", "up", "--build", "--remove-orphans"]
    logging.info("Running: %s", " ".join(cmd))
    # spawn and stream logs
    p = subprocess.Popen(cmd, cwd=ROOT)
    try:
        p.wait()
    except KeyboardInterrupt:
        logging.info("Stopping docker compose")
        p.terminate()
        p.wait()
    return True


def run_local_stack(python: str):
    procs = {}
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure the backend package (backend/app) is importable as top-level `app` by
    # adding the backend folder to PYTHONPATH and launching uvicorn from that folder.
    backend_path = str(ROOT / "backend")
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = backend_path + (os.pathsep + prev_pythonpath if prev_pythonpath else "")

    # Ensure uvicorn and streamlit are installed
    logging.info("Ensuring uvicorn and streamlit are present in venv")
    run_cmd([python, "-m", "pip", "install", "uvicorn[standard]", "streamlit", "plotly"])

    # Start backend from the backend folder so that `app` is a top-level package
    backend_cmd = [python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    procs['backend'] = start_process(backend_cmd, cwd=ROOT / "backend", env=env)

    # Start frontend
    frontend_cmd = [python, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"]
    procs['frontend'] = start_process(frontend_cmd, cwd=ROOT, env=env)

    # Optionally start an RQ worker if redis is available
    if shutil.which("redis-server") is not None or os.environ.get("REDIS_URL"):
        logging.info("Redis seems available, attempting to start RQ worker (if installed)")
        run_cmd([python, "-m", "pip", "install", "rq", "redis"])  # best-effort
        worker_cmd = [python, "-m", "rq", "worker", "default"]
        try:
            procs['worker'] = start_process(worker_cmd, cwd=ROOT, env=env)
        except Exception:
            logging.exception("Failed to start RQ worker")

    logging.info("Local stack started. Monitoring processes... (Ctrl-C to quit)")

    try:
        while True:
            for name, p in list(procs.items()):
                ret = p.poll()
                if ret is not None:
                    logging.warning("Process %s exited with code %s — attempting restart (one retry)", name, ret)
                    # single retry
                    if name == 'backend':
                        procs[name] = start_process(backend_cmd, cwd=ROOT, env=env)
                    elif name == 'frontend':
                        procs[name] = start_process(frontend_cmd, cwd=ROOT, env=env)
                    elif name == 'worker':
                        procs[name] = start_process(worker_cmd, cwd=ROOT, env=env)
            time.sleep(3)
    except KeyboardInterrupt:
        logging.info("Shutting down processes...")
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass


def run_tests():
    logging.info("Running test suite (pytest)")
    if shutil.which("pytest") is None:
        # use venv python -m pytest
        python = get_venv_python()
        return run_cmd([python, "-m", "pytest", "-q"])  # quiet
    else:
        return run_cmd(["pytest", "-q"])  # local pytest


def quick_summary():
    logging.info("Project root: %s", ROOT)
    logging.info("Venv dir: %s", VENV_DIR)
    logging.info("Python: %s", sys.executable)


def main():
    parser = argparse.ArgumentParser(description="Autorun helper for PredictiFlow")
    parser.add_argument("action", nargs="?", default="all", choices=["all", "venv", "install", "docker", "local", "tests"], help="action to run")
    args = parser.parse_args()

    quick_summary()

    if args.action in ("all", "venv"):
        create_venv()

    python = get_venv_python()

    if args.action in ("all", "install"):
        upgrade_pip(python)
        reqs = [ROOT / "dev-requirements.txt", ROOT / "backend" / "requirements.txt", ROOT / "frontend" / "requirements.txt"]
        install_requirements(python, reqs)

    if args.action in ("all", "docker"):
        ok = run_docker_compose()
        if ok:
            logging.info("Docker compose started — attach to logs to inspect services")
            return

    if args.action in ("all", "local"):
        run_local_stack(python)

    if args.action == "tests":
        code, out, err = run_tests()
        if code == 0:
            logging.info("Tests passed")
        else:
            logging.error("Tests failed: see output")


if __name__ == "__main__":
    main()
