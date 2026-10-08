#!/bin/bash
set -euo pipefail

# Reproduce the main multi-seed experiments used in the manuscript.
#
# Seven unique models are trained for seeds 0-10.
# MolGAN variants are evaluated using both last and best checkpoints.
#
# Optional environment variables:
#   START_SEED=0
#   END_SEED=10
#   EPOCHS=1000
#   NUM_SAMPLES=10000
#   RUN_PEPSYSCO=1
#   TCGA_CSV=/path/to/TCGA_BLCA_WT_mutant_peptides_unique.csv

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_SEED="${START_SEED:-0}"
END_SEED="${END_SEED:-10}"
EPOCHS="${EPOCHS:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
RUN_PEPSYSCO="${RUN_PEPSYSCO:-1}"

run_last_only () {
    local model="$1"
    local seed="$2"

    EPOCHS="$EPOCHS" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    CHECKPOINT=last \
    RUN_PEPSYSCO="$RUN_PEPSYSCO" \
    "$ROOT/reproduce_experiment.sh" "$model" "$seed"
}

run_last_and_best () {
    local model="$1"
    local seed="$2"

    # Train once and evaluate model_last.pth.
    EPOCHS="$EPOCHS" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    CHECKPOINT=last \
    RUN_PEPSYSCO="$RUN_PEPSYSCO" \
    "$ROOT/reproduce_experiment.sh" "$model" "$seed"

    # Reuse the same training run and evaluate model_best.pth.
    EPOCHS="$EPOCHS" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    CHECKPOINT=best \
    RUN_PEPSYSCO="$RUN_PEPSYSCO" \
    SKIP_TRAIN=1 \
    "$ROOT/reproduce_experiment.sh" "$model" "$seed"
}

for seed in $(seq "$START_SEED" "$END_SEED"); do
    echo
    echo "################################################################"
    echo "Seed $seed"
    echo "################################################################"

    # WGAN-GP: last checkpoint
    run_last_only wgan "$seed"

    # MolGAN lambda_M = 0.5: last and best
    run_last_and_best molgan050 "$seed"

    # MolGAN lambda_M = 0: last and best
    run_last_and_best molgan0 "$seed"

    # MolGAN lambda_M = 0 with revised ORGAN penalty: last and best
    run_last_and_best molgan0_organ "$seed"

    # GD-WGAN-GP gamma_max = 0.25: last
    run_last_only gd_gamma025 "$seed"

    # GD-WGAN-GP gamma_max = 1: last
    run_last_only gd_gamma1 "$seed"

    # GD-WGAN-GP with revised ORGAN penalty: last
    run_last_only gd_organ "$seed"
done

echo
echo "============================================================"
echo "All requested seeds completed."
echo "============================================================"
