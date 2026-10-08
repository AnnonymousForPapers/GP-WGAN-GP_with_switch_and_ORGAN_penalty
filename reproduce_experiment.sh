#!/bin/bash
set -euo pipefail

# Reproduce one bladder-cancer experiment from training through evaluation.
#
# Usage:
#   bash reproduce_experiment.sh <model> <seed>
#
# Examples:
#   bash reproduce_experiment.sh gd_gamma1 0
#   bash reproduce_experiment.sh gd_gamma025 0
#   bash reproduce_experiment.sh gd_organ 0
#
# Optional environment variables:
#   EPOCHS=1000
#   NUM_SAMPLES=10000
#   CHECKPOINT=last
#   TCGA_CSV=/path/to/TCGA_BLCA_WT_mutant_peptides_unique.csv
#   RUN_PEPSYSCO=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-gd_gamma1}"
SEED="${2:-0}"

EPOCHS="${EPOCHS:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
CHECKPOINT="${CHECKPOINT:-last}"
RUN_PEPSYSCO="${RUN_PEPSYSCO:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

TRAIN_DIR="$ROOT/code_R2_FixPad"
EVAL_DIR="$ROOT/unified_peptide_evaluation"
TRAINING_DATA="$ROOT/data/neoepitopes/Bladder.4.0_test_mut.csv"
DEEPIMMUNO_WEIGHTS="$ROOT/weights/Immunogenicity_Predictor"

case "$MODEL" in
    wgan)
        TRAIN_SCRIPT="WGAN-GP_GPU_FixPad_SeedEpochBestLast.py"
        RESULT_NAME="WGAN-GP_FixPad_seed${SEED}"
        ;;

    molgan050)
        TRAIN_SCRIPT="Goal-directed_WGAN_wclip_MolGAN050_DScalar_Loss_MyPredictor_GPU_Run.py"
        RESULT_NAME="GD_WGAN_wclip_MolGAN050_DScalar_Loss_MyPredictor_FixPad_seed${SEED}"
        ;;

    molgan0)
        TRAIN_SCRIPT="Goal-directed_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor_GPU_FixPad_SeedEpochBestLast.py"
        RESULT_NAME="GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor_FixPad_seed${SEED}"
        ;;

    molgan0_organ)
        TRAIN_SCRIPT="Goal-directed_WGAN_wclip_OR_PlusOne_MolGAN_Gamma0_DScalar_Loss_MyPredictor_GPU_Run.py"
        RESULT_NAME="GD_WGAN_wclip_OR_PlusOne_MolGAN_Gamma0_DScalar_Loss_MyPredictor_FixPad_seed${SEED}"
        ;;

    gd_gamma025)
        TRAIN_SCRIPT="Goal-directed_WGAN-GP_Gamma0_25_GPU_FixPad_SeedEpochBestLast.py"
        RESULT_NAME="Goal-directed_WGAN-GP_Gamma0_25_FixPad_seed${SEED}"
        ;;

    gd_gamma1)
        TRAIN_SCRIPT="Goal-directed_WGAN-GP_GPU_FixPad_SeedEpochBestLast.py"
        RESULT_NAME="Goal-directed_WGAN-GP_FixPad_seed${SEED}"
        ;;

    gd_organ)
        TRAIN_SCRIPT="Goal-directed_WGAN-GP_ORGAN_PlusOne_GPU_Run.py"
        RESULT_NAME="Goal-directed_WGAN-GP_ORGAN_PlusOne_FixPad_seed${SEED}"
        ;;

    *)
        echo "Unknown model: $MODEL"
        echo "Supported: wgan molgan050 molgan0 molgan0_organ gd_gamma025 gd_gamma1 gd_organ"
        exit 1
        ;;
esac

# ----------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------

for required in \
    "$TRAIN_DIR/$TRAIN_SCRIPT" \
    "$TRAINING_DATA" \
    "$ROOT/data/DeepImmuno/after_pca.txt" \
    "$ROOT/data/DeepImmuno/hla2paratopeTable_aligned.txt"
do
    if [[ ! -e "$required" ]]; then
        echo "Required file not found: $required"
        exit 1
    fi
done

if [[ ! -d "$DEEPIMMUNO_WEIGHTS" ]]; then
    echo "Immunogenicity-predictor checkpoint not found:"
    echo "  $DEEPIMMUNO_WEIGHTS"
    exit 1
fi

echo "============================================================"
echo "Reproduction configuration"
echo "============================================================"
echo "Model:       $MODEL"
echo "Seed:        $SEED"
echo "Epochs:      $EPOCHS"
echo "Samples:     $NUM_SAMPLES"
echo "Checkpoint:  $CHECKPOINT"
echo

# ----------------------------------------------------------------------
# 1. Train
# ----------------------------------------------------------------------

echo "============================================================"
echo "Stage 1: training"
echo "============================================================"

MODEL_DIR="$ROOT/result/$RESULT_NAME/epoch$EPOCHS"

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "Training skipped; using existing model directory:"
    echo "  $MODEL_DIR"
else
    cd "$TRAIN_DIR"

    python "$TRAIN_SCRIPT" \
        --num_epochs "$EPOCHS" \
        --seed "$SEED"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Expected model directory does not exist:"
    echo "  $MODEL_DIR"
    exit 1
fi

# ----------------------------------------------------------------------
# 2. Generate 10,000 peptides and run the main evaluation
# ----------------------------------------------------------------------

echo
echo "============================================================"
echo "Stage 2: generation and main predictor evaluation"
echo "============================================================"

OUTPUT_DIR="$MODEL_DIR/evaluation_model_${CHECKPOINT}_seed${SEED}"

PREDICTORS=(
    deepimmuno
    iedb_immunogenicity
    netmhcpan40
    netmhcpan41
    pepmatch
    similarity
)

EXTRA_ARGS=()

# TCGA-BLCA is run when the externally prepared reference file is supplied.
if [[ -n "${TCGA_CSV:-}" ]]; then
    PREDICTORS+=(tcga_blca)
    EXTRA_ARGS+=(--tcga-csv "$TCGA_CSV")
fi

cd "$EVAL_DIR"

python -u run_all_evaluations.py \
    --model-dir "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --architecture auto \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --data-root "$ROOT" \
    --bladder-csv "$TRAINING_DATA" \
    --deepimmuno-weights "$DEEPIMMUNO_WEIGHTS" \
    --tf-python "$(command -v python)" \
    --iedb-immunogenicity-backend api \
    --netmhcpan-backend api \
    --predictors "${PREDICTORS[@]}" \
    "${EXTRA_ARGS[@]}"

# ----------------------------------------------------------------------
# 3. Novelty/diversity evaluation
# ----------------------------------------------------------------------

echo
echo "============================================================"
echo "Stage 3: novelty/diversity evaluation"
echo "============================================================"

python -u novelty/run_novelty_all_evaluations.py \
    --root "$OUTPUT_DIR" \
    --training "$TRAINING_DATA" \
    --batch-size 256

# ----------------------------------------------------------------------
# 4. PepSySco evaluation
# ----------------------------------------------------------------------

if [[ "$RUN_PEPSYSCO" == "1" ]]; then
    echo
    echo "============================================================"
    echo "Stage 4: PepSySco evaluation"
    echo "============================================================"

    python -u pepsysco_api/evaluate_all/run_pepsysco_all_evaluations.py \
        --root "$OUTPUT_DIR" \
        --lock-file "$ROOT/.iedb_api.lock" \
        --poll-seconds 30 \
        --http-timeout 120 \
        --retry-seconds 60
else
    echo
    echo "Stage 4: PepSySco skipped because RUN_PEPSYSCO=$RUN_PEPSYSCO"
fi

echo
echo "============================================================"
echo "Reproduction completed"
echo "============================================================"
echo "Training output:"
echo "  $MODEL_DIR"
echo
echo "Evaluation output:"
echo "  $OUTPUT_DIR"
