#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --output=R-%x_%j.out
#SBATCH --job-name=timing_table

set -euo pipefail
module purge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -u make_computational_time_table.py \
    --config timing_table_config.txt \
    --seed-start 0 \
    --seed-end 10 \
    --epoch 1000 \
    --output-prefix bladder_computational_time \
    --show-sd
