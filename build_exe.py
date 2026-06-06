#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
RELEASE_DIR = BASE_DIR / "release"
APP_NAME = "Geospatial TIFF ML Platform"
EXE_NAME = f"{APP_NAME}.exe"
LAUNCHER = BASE_DIR / "launcher.py"

DATA_ITEMS = [
    ("app.py", "."),
    ("pages", "pages"),
    ("src", "src"),
    ("test_data", "test_data"),
    ("python_embed", "python_embed"),
]


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pyinstaller_command():
    candidates = [sys.executable, shutil.which("python"), shutil.which("py")]
    for candidate in candidates:
        if not candidate:
            continue
        probe = [candidate, "-m", "PyInstaller", "--version"]
        try:
            subprocess.run(probe, check=True, capture_output=True, text=True)
            return [candidate, "-m", "PyInstaller"]
        except Exception:
            continue

    pyinstaller = shutil.which("pyinstaller")
    if pyinstaller:
        return [pyinstaller]

    raise RuntimeError(
        "PyInstaller is not installed. Install it first with:\n"
        "  python -m pip install pyinstaller\n"
        "or:\n"
        "  python_embed\\python.exe -m pip install pyinstaller"
    )


def build():
    if not LAUNCHER.exists():
        raise FileNotFoundError(f"Missing launcher script: {LAUNCHER}")

    cmd = pyinstaller_command() + [
        "--noconfirm",
        "--clean",
        "--onefile",
        "--noconsole",
        "--name",
        APP_NAME,
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
    ]

    for source, target in DATA_ITEMS:
        source_path = BASE_DIR / source
        if source_path.exists():
            cmd.extend(["--add-data", f"{source_path};{target}"])

    cmd.append(str(LAUNCHER))
    run(cmd)


def copy_release():
    DIST_EXE = DIST_DIR / EXE_NAME
    if not DIST_EXE.exists():
        raise FileNotFoundError(f"Build output not found: {DIST_EXE}")

    RELEASE_DIR.mkdir(exist_ok=True)
    target_exe = RELEASE_DIR / EXE_NAME
    shutil.copy2(DIST_EXE, target_exe)
    print(f"Release EXE: {target_exe}")


def main():
    print("=" * 60)
    print("Geospatial TIFF ML Platform - EXE Builder")
    print("=" * 60)
    build()
    copy_release()
    print("Build complete.")


if __name__ == "__main__":
    main()
