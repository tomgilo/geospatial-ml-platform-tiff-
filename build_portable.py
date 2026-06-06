#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build portable Python environment.
Downloads Python 3.12 embeddable, installs pip, installs all deps directly.
No venv needed -- everything lives inside python_embed/.
Run once on the build machine, then zip the whole folder for distribution.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "python_embed")
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TRUSTED = "pypi.tuna.tsinghua.edu.cn"
ALT_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"
ALT_TRUSTED = "mirrors.aliyun.com"

PY_VERSION = "3.12.8"
EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
EMBED_ZIP = os.path.join(os.environ.get("TEMP", "/tmp"), f"python-{PY_VERSION}-embed-amd64.zip")
GETPIP_URL = "https://bootstrap.pypa.io/get-pip.py"
GETPIP_PY = os.path.join(os.environ.get("TEMP", "/tmp"), "get-pip.py")


def run(cmd, check=False):
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore")
    if check and result.returncode != 0:
        print(f"  [FAIL] {' '.join(cmd)}")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        print(f"  [OK]   {' '.join(cmd[:3])}...")
    return result


def download(url, dest):
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {dest}")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def extract_embeddable():
    """Download and extract Python embeddable package."""
    if os.path.exists(os.path.join(EMBED_DIR, "python.exe")):
        print("  Python embeddable already present.")
        return os.path.join(EMBED_DIR, "python.exe")

    print("\n[1/4] Downloading Python embeddable package...")
    download(EMBED_URL, EMBED_ZIP)

    print("  Extracting...")
    os.makedirs(EMBED_DIR, exist_ok=True)
    with zipfile.ZipFile(EMBED_ZIP, "r") as z:
        z.extractall(EMBED_DIR)

    print("  Fixing python312._pth for pip support...")
    pth_files = glob.glob(os.path.join(EMBED_DIR, "python*._pth"))
    for pth in pth_files:
        with open(pth, "r", encoding="utf-8") as f:
            lines = f.readlines()
        has_import_site = any("import site" in line for line in lines)
        if not has_import_site:
            with open(pth, "a", encoding="utf-8") as f:
                f.write("import site\n")
            print(f"    Added 'import site' to {os.path.basename(pth)}")

    print("  Cleaning up zip...")
    os.remove(EMBED_ZIP)

    python_exe = os.path.join(EMBED_DIR, "python.exe")
    print(f"  Python embeddable ready: {python_exe}")
    return python_exe


def install_pip(python_exe):
    """Install pip into the embeddable Python."""
    pip_exe = os.path.join(EMBED_DIR, "Scripts", "pip.exe")
    if os.path.exists(pip_exe):
        print("  pip already installed.")
        return

    print("\n[2/4] Installing pip...")
    download(GETPIP_URL, GETPIP_PY)
    run([python_exe, GETPIP_PY], check=True)
    os.remove(GETPIP_PY)
    print("  pip installed.")


def install_packages(python_exe):
    """Install all requirements into the embeddable Python."""
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    pkgs = [
        "streamlit>=1.28", "numpy>=1.24", "pandas>=2.0",
        "matplotlib>=3.7", "seaborn>=0.12", "rasterio>=1.3",
        "scikit-learn>=1.3", "xgboost>=2.0", "lightgbm>=4.1",
        "joblib>=1.3", "shap>=0.44", "pillow>=10.0",
        "openpyxl>=3.1", "plotly>=5.18",
    ]

    pip_exe = os.path.join(EMBED_DIR, "Scripts", "pip.exe")

    print("\n[3/4] Installing packages (Tsinghua mirror)...")
    run([pip_exe, "install", "--upgrade", "pip",
         "-i", PIP_MIRROR, "--trusted-host", TRUSTED], check=False)

    if os.path.exists(req_file):
        r = run([pip_exe, "install", "-r", req_file,
                 "-i", PIP_MIRROR, "--trusted-host", TRUSTED], check=False)
        if r.returncode != 0:
            print("  Tsinghua failed, trying Aliyun...")
            run([pip_exe, "install", "-r", req_file,
                 "-i", ALT_MIRROR, "--trusted-host", ALT_TRUSTED], check=True)
    else:
        r = run([pip_exe, "install"] + pkgs +
                ["-i", PIP_MIRROR, "--trusted-host", TRUSTED], check=False)
        if r.returncode != 0:
            print("  Tsinghua failed, trying Aliyun...")
            run([pip_exe, "install"] + pkgs +
                ["-i", ALT_MIRROR, "--trusted-host", ALT_TRUSTED], check=True)

    print("  Packages installed.")


def create_shortcut():
    """Create desktop shortcut using PowerShell."""
    print("\n[4/4] Creating desktop shortcut...")
    try:
        desktop = os.path.expandvars(r"%USERPROFILE%\Desktop")
        lnk = os.path.join(desktop, "Geospatial TIFF ML Platform.lnk")
        bat_path = os.path.join(BASE_DIR, "启动应用.bat")
        if not os.path.exists(bat_path):
            bat_path = os.path.join(BASE_DIR, "StartApp.bat")

        ps_cmd = (
            f'$WshShell = New-Object -ComObject WScript.Shell; '
            f'$shortcut = $WshShell.CreateShortcut("{lnk}"); '
            f'$shortcut.TargetPath = "{bat_path}"; '
            f'$shortcut.WorkingDirectory = "{BASE_DIR}"; '
            f'$shortcut.Description = "Geospatial TIFF ML Platform"; '
            f'$shortcut.Save()'
        )
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-Command", ps_cmd], capture_output=True, timeout=10)
        print(f"  Shortcut: {lnk}")
    except Exception as e:
        print(f"  Skipped: {e}")


def main():
    print("=" * 50)
    print("  Geospatial TIFF ML Platform")
    print("  Portable Environment Builder")
    print("=" * 50)
    print()
    print("This will download Python embeddable + install all deps.")
    print("After this, you can ZIP the whole folder for distribution.")
    print()
    print("Starting in 3 seconds...")
    import time
    time.sleep(3)
    print()

    python_exe = extract_embeddable()
    install_pip(python_exe)
    install_packages(python_exe)
    create_shortcut()

    print("\n" + "=" * 50)
    print("  DONE!")
    print("=" * 50)
    print()
    print("Next: Double-click '启动应用.bat' to start.")
    print("For distribution: ZIP the entire folder and share.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
