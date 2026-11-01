#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${1:-plots.json}"
OUTPUT_FILE="${2:-tracks.json}"
shift $(( $# >= 2 ? 2 : $# ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Create isolated virtual environment and install dependencies automatically if missing
if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] Creating isolated virtual environment in ${VENV_DIR}..." >&2
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv "${VENV_DIR}"
    else
        python -m venv "${VENV_DIR}"
    fi

    if [ -x "${VENV_DIR}/bin/pip" ]; then
        "${VENV_DIR}/bin/pip" install --upgrade pip -q
        "${VENV_DIR}/bin/pip" install -q -r "${SCRIPT_DIR}/requirements.txt"
    elif [ -x "${VENV_DIR}/Scripts/pip.exe" ]; then
        "${VENV_DIR}/Scripts/pip.exe" install --upgrade pip -q
        "${VENV_DIR}/Scripts/pip.exe" install -q -r "${SCRIPT_DIR}/requirements.txt"
    fi
fi

# Select the Python binary from .venv (compatible with Linux/macOS and Windows/Git Bash)
if [ -x "${VENV_DIR}/bin/python" ]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
    PYTHON_BIN="${VENV_DIR}/Scripts/python.exe"
else
    PYTHON_BIN="python3"
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m radar_tracker "${INPUT_FILE}" "${OUTPUT_FILE}" "$@"
