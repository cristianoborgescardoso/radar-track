#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SCRIPTS="${HERE}/scripts"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
elif [[ -x "${ROOT}/.venv/Scripts/python.exe" ]]; then
    PY="${ROOT}/.venv/Scripts/python.exe"
else
    PY="python3"
fi

mkdir -p "${HERE}/outputs"

echo "=================================================================="
echo "External Real-World Validation (Fortem R20 / AERPAW Opt2 Corridor)"
echo "=================================================================="

"${PY}" "${SCRIPTS}/convert_fortem.py" --airborne-only "$@"

bash "${ROOT}/tracker.sh" \
    "${HERE}/data/aerpaw_opt2_subset/plots.json" \
    "${HERE}/outputs/tracks.json" \
    --config "${HERE}/fortem_r20.properties"

"${PY}" "${SCRIPTS}/evaluate_fortem.py"

echo "=================================================================="
echo "Validation completed successfully! Output: validations/outputs/"
echo "=================================================================="
