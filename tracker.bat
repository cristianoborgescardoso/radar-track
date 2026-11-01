@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%VENV_PYTHON%" (
    echo [setup] Creating isolated virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if exist "%VENV_PIP%" (
        "%VENV_PIP%" install --upgrade pip -q
        "%VENV_PIP%" install -q -r "%SCRIPT_DIR%requirements.txt"
    )
)

set "PYTHON_BIN=python"
if exist "%VENV_PYTHON%" set "PYTHON_BIN=%VENV_PYTHON%"

if "%~1"=="" (
    "%PYTHON_BIN%" "%SCRIPT_DIR%radar_tracker\tracker.py" "plots.json" "tracks.json"
) else if "%~2"=="" (
    "%PYTHON_BIN%" "%SCRIPT_DIR%radar_tracker\tracker.py" "%~1" "tracks.json"
) else (
    "%PYTHON_BIN%" "%SCRIPT_DIR%radar_tracker\tracker.py" %*
)
exit /b %ERRORLEVEL%
