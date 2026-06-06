#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


APP_NAME = "Geospatial TIFF ML Platform"
STREAMLIT_APP = "app.py"
DEFAULT_PORT = 8501


def is_frozen():
    return getattr(sys, "frozen", False)


def bundle_dir():
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


def install_home():
    if is_frozen():
        base_dir = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        home_dir = os.path.join(base_dir, APP_NAME)
    else:
        home_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(home_dir, exist_ok=True)
    os.makedirs(os.path.join(home_dir, "outputs"), exist_ok=True)
    return home_dir


def ensure_sample_data(source_dir, target_dir):
    source_test_data = os.path.join(source_dir, "test_data")
    target_test_data = os.path.join(target_dir, "test_data")
    if os.path.isdir(source_test_data) and not os.path.exists(target_test_data):
        shutil.copytree(source_test_data, target_test_data)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(url, timeout_seconds=90):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def show_error(message):
    try:
        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    except Exception:
        print(message, file=sys.stderr)


def main():
    source_dir = bundle_dir()
    home_dir = install_home()
    ensure_sample_data(source_dir, home_dir)

    python_exe = os.path.join(source_dir, "python_embed", "python.exe")
    app_path = os.path.join(source_dir, STREAMLIT_APP)
    log_path = os.path.join(home_dir, "launcher.log")

    if not os.path.exists(python_exe):
        show_error(f"Missing bundled Python runtime:\n{python_exe}")
        return 1
    if not os.path.exists(app_path):
        show_error(f"Missing bundled app entry:\n{app_path}")
        return 1

    port = find_free_port() if not os.environ.get("GEOSPATIAL_ML_PORT") else int(os.environ["GEOSPATIAL_ML_PORT"])
    url = f"http://127.0.0.1:{port}"

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting app at {url}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=home_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

        if not wait_for_server(url):
            time.sleep(1)
            if proc.poll() is not None:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as reader:
                        tail = reader.read()[-3000:]
                except Exception:
                    tail = "Unable to read launcher log."
                show_error("Application failed to start.\n\n" + tail)
                return proc.returncode or 1

        webbrowser.open(url)
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
