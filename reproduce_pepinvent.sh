#!/bin/bash
set -euo pipefail

# Reproduce the PepINVENT comparison experiment.
#
# Usage:
#   ./reproduce_pepinvent.sh bladder [seed]
#   ./reproduce_pepinvent.sh brain   [seed]
#
# Required:
#   Run this script from an environment containing REINVENT/PepINVENT
#   and PepFun2.
#
#   DEEPIMMUNO_PYTHON must point to the Python executable of the
#   TensorFlow/DeepImmuno environment.
#
# Optional:
#   EPOCHS=1000
#   MODE=direct
#   REINVENT_EXE=reinvent

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${1:-bladder}"
SEED="${2:-0}"

EPOCHS="${EPOCHS:-1000}"
MODE="${MODE:-direct}"
REINVENT_EXE="${REINVENT_EXE:-reinvent}"

PRIOR="$ROOT/code_R2_pepinvent/priors/pepinvent.prior"

if [[ -z "${DEEPIMMUNO_PYTHON:-}" ]]; then
    echo "ERROR: DEEPIMMUNO_PYTHON is not set."
    echo
    echo "Set it to the Python executable of the TensorFlow/DeepImmuno environment."
    echo "Example:"
    echo "  export DEEPIMMUNO_PYTHON=/path/to/tf/environment/bin/python"
    exit 1
fi

if [[ ! -f "$DEEPIMMUNO_PYTHON" ]]; then
    echo "ERROR: DeepImmuno Python executable not found:"
    echo "  $DEEPIMMUNO_PYTHON"
    exit 1
fi

if [[ ! -f "$PRIOR" ]]; then
    echo "ERROR: PepINVENT prior not found:"
    echo "  $PRIOR"
    exit 1
fi

if ! command -v "$REINVENT_EXE" >/dev/null 2>&1; then
    echo "ERROR: REINVENT executable not found:"
    echo "  $REINVENT_EXE"
    echo
    echo "Activate the environment described by environment_pepinvent.yml."
    exit 1
fi

case "$DATASET" in
    bladder)
        SCRIPT="$ROOT/code_R2_pepinvent/run_pepinvent_bladder3mask_epochs.py"
        ;;
    brain)
        SCRIPT="$ROOT/code_R2_pepinvent/run_pepinvent_brain3mask_epochs.py"
        ;;
    *)
        echo "ERROR: dataset must be 'bladder' or 'brain'."
        exit 1
        ;;
esac

echo "============================================================"
echo "PepINVENT reproduction"
echo "============================================================"
echo "Dataset:             $DATASET"
echo "Seed:                $SEED"
echo "Epochs:              $EPOCHS"
echo "Mode:                $MODE"
echo "Prior:               $PRIOR"
echo "REINVENT:            $REINVENT_EXE"
echo "DeepImmuno Python:   $DEEPIMMUNO_PYTHON"
echo "============================================================"

python "$SCRIPT" \
    --prior "$PRIOR" \
    --mode "$MODE" \
    --seed "$SEED" \
    --reinvent "$REINVENT_EXE" \
    --num-epochs "$EPOCHS" \
    --checkpoint-every-epochs 50 \
    --mask-count 3 \
    --masks-per-peptide 1 \
    --reinvent-log-level verbose
