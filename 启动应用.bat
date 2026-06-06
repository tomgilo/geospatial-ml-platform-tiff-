@echo off
cd /d "%~dp0"

REM Prefer the packaged EXE when it exists.
if exist "release\Geospatial TIFF ML Platform.exe" (
    start "" "%~dp0release\Geospatial TIFF ML Platform.exe"
    goto :end
)

if exist "dist\Geospatial TIFF ML Platform.exe" (
    start "" "%~dp0dist\Geospatial TIFF ML Platform.exe"
    goto :end
)

REM Use embedded Python directly
set "PY=python_embed\python.exe"
if exist "%PY%" (
    start "" "%PY%" -m streamlit run app.py
    echo App starting at http://localhost:8501
    timeout /t 3 >nul
    goto :end
)

REM If no embedded Python, try building
if exist "build_portable.py" (
    echo Building portable environment first...
    python build_portable.py
    if exist "%PY%" (
        start "" "%PY%" -m streamlit run app.py
        echo App starting at http://localhost:8501
        timeout /t 3 >nul
    ) else (
        echo Build failed. Please run SETUP.bat or install Python manually.
        pause
    )
) else (
    echo python_embed not found and build_portable.py missing.
    pause
)

:end
