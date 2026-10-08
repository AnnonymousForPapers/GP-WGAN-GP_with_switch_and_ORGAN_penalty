#!/bin/bash

#SBATCH --partition=general
#SBATCH --mem=64G
#SBATCH --output=R-%x_%j.out

set -euo pipefail
module purge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python make_evaluation_table.py \
    --config tableS2_config.txt \
    --aggregation sample \
    --sample-seed 0 \
    --output-prefix tableS2_seed0
