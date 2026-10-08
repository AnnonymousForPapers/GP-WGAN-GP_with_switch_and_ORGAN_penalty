#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL_STEPS = 1000
CHECKPOINT_EVERY = 50
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
SIGMA = 80
LEARNING_RATE = 0.00005
DISTANCE_THRESHOLD = -20

DF_TYPE = "IdenticalMurckoScaffold"
DF_SCORE_THRESHOLD = 0.4
DF_BUCKET_SIZE = 25
DF_SIMILARITY_THRESHOLD = 0.4
DF_PENALTY = 0.5

# Repository root, independent of the current working directory.
DATA_ROOT = Path(__file__).resolve().parents[1]
REAL_PEPTIDE_CSV = DATA_ROOT / "data/neoepitopes/Bladder.4.0_test_mut.csv"


def q(x):
    return str(x).replace("\\", "\\\\")


def scoring_block(score_script, with_scorer, output_dir):
    extra = ""
    if with_scorer:
        extra = (
            f' --state-file "{q(output_dir / "scorer_state.pth")}"'
            f' --real-peptide-csv "{q(REAL_PEPTIDE_CSV)}"'
        )

    return "\n".join([
        "[stage.scoring]",
        'type = "arithmetic_mean"',
        "",
        "[[stage.scoring.component]]",
        "[stage.scoring.component.ExternalProcess]",
        "",
        "[[stage.scoring.component.ExternalProcess.endpoint]]",
        'name = "DeepImmuno"',
        "weight = 1.0",
        f'params.executable = "{q(sys.executable)}"',
        f'params.args = "-u {q(score_script)} --data-root ../{extra}"',
        'params.property = "predictions"',
        "",
    ])


def make_training_config(prior, train_masks, score_script, output_dir, with_scorer):
    lines = [
        'run_type = "staged_learning"',
        "",
        'device = "cuda:0"',
        f'tb_logdir = "{q(output_dir / "tb")}"',
        f'json_out_config = "{q(output_dir / "resolved_training_config.json")}"',
        "",
        "[parameters]",
        f'prior_file = "{q(prior)}"',
        f'agent_file = "{q(prior)}"',
        f'smiles_file = "{q(train_masks)}"',
        'sample_strategy = "multinomial"',
        f"distance_threshold = {DISTANCE_THRESHOLD}",
        f'summary_csv_prefix = "{q(output_dir / "training")}"',
        f"batch_size = {TRAIN_BATCH_SIZE}",
        "randomize_smiles = false",
        "purge_memories = false",
        "",
        "[learning_strategy]",
        'type = "dap"',
        f"sigma = {SIGMA}",
        f"rate = {LEARNING_RATE}",
        "",
        "[diversity_filter]",
        f'type = "{DF_TYPE}"',
        f"bucket_size = {DF_BUCKET_SIZE}",
        f"minscore = {DF_SCORE_THRESHOLD}",
        f"minsimilarity = {DF_SIMILARITY_THRESHOLD}",
        f"penalty_multiplier = {DF_PENALTY}",
        "",
    ]

    for step in range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY):
        lines.extend([
            "[[stage]]",
            f'chkpt_file = "{q(output_dir / f"model_step_{step:04d}.chkpt")}"',
            'termination = "simple"',
            "max_score = 1.0",
            f"min_steps = {CHECKPOINT_EVERY}",
            f"max_steps = {CHECKPOINT_EVERY}",
        ])
        lines.append(scoring_block(score_script, with_scorer, output_dir))

    return "\n".join(lines) + "\n"


def make_sampling_config(model_file, eval_masks, output_csv):
    return "\n".join([
        'run_type = "sampling"',
        "",
        'device = "cuda:0"',
        "",
        "[parameters]",
        f'model_file = "{q(model_file)}"',
        f'smiles_file = "{q(eval_masks)}"',
        'sample_strategy = "multinomial"',
        "temperature = 1.0",
        f'output_file = "{q(output_csv)}"',
        f"num_smiles = {EVAL_BATCH_SIZE}",
        "unique_molecules = false",
        "randomize_smiles = false",
        "",
    ])


def run_cmd(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_checkpoint(
    reinvent_exe,
    checkpoint,
    step,
    eval_masks,
    output_dir,
    eval_script,
    base_seed,
):
    sample_csv = output_dir / f"df_PepINVENT_all_step{step}.csv"
    sample_toml = output_dir / f"sampling_step_{step:04d}.toml"
    sample_log = output_dir / f"sampling_step_{step:04d}.log"

    sample_toml.write_text(
        make_sampling_config(checkpoint, eval_masks, sample_csv),
        encoding="utf-8",
    )

    eval_seed = base_seed + step // CHECKPOINT_EVERY

    # Sampling uses REINVENT4 environment.
    run_cmd([
        reinvent_exe,
        "-s", str(eval_seed),
        "-l", str(sample_log),
        str(sample_toml),
    ])

    eval_csv = output_dir / f"checkpoint_evaluation_step{step}.csv"
    eval_json = output_dir / f"checkpoint_evaluation_step{step}.json"

    # DeepImmuno evaluation uses the existing tf environment.
    run_cmd([
        sys.executable,
        str(eval_script),
        "--sample-csv", str(sample_csv),
        "--output-csv", str(eval_csv),
        "--output-json", str(eval_json),
        "--data-root", "../",
    ])

    result = json.loads(eval_json.read_text(encoding="utf-8"))
    result["step"] = step
    result["checkpoint"] = str(checkpoint)

    print(
        "Checkpoint evaluation: "
        f"step={step}, "
        f"mean_imm_score={result['mean_immunogenicity_score']:.6f}, "
        f"unique={result['unique_count']}/{EVAL_BATCH_SIZE}, "
        f"unique_ratio={result['unique_ratio']:.6f}, "
        f"sum={result['checkpoint_score_sum']:.6f}"
    )

    return result



def _find_training_summary_csvs(output_dir):
    """
    REINVENT writes summary CSV files using summary_csv_prefix.
    Collect all CSV files that begin with 'training' and exclude our own
    checkpoint/evaluation files.
    """
    files = []
    for p in sorted(output_dir.glob("training*.csv")):
        if p.is_file():
            files.append(p)
    return files


def _pick_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def print_and_save_training_statistics(output_dir):
    """
    Print one compact PepINVENT summary per RL step and save the analogous
    arrays/statistics.

    We do not invent GAN-only quantities such as d_real_loss or W_dist.
    The closest PepINVENT/REINVENT quantities are reported instead.
    """
    csvs = _find_training_summary_csvs(output_dir)
    if not csvs:
        print("[WARNING] No REINVENT training summary CSV was found.")
        print("          REINVENT's native training.log still contains its own step output.")
        return None

    frames = []
    for p in csvs:
        try:
            d = pd.read_csv(p)
            if len(d):
                frames.append(d)
        except Exception as e:
            print(f"[WARNING] Could not read {p}: {e}")

    if not frames:
        print("[WARNING] REINVENT training summary CSV files were empty.")
        return None

    df = pd.concat(frames, ignore_index=True)

    step_col = _pick_column(df, ["step", "Step", "STEP"])
    score_col = _pick_column(df, ["total_score", "Total_Score", "score", "Score"])
    agent_col = _pick_column(df, ["AGENT", "agent", "agent_nll", "Agent"])
    prior_col = _pick_column(df, ["PRIOR", "prior", "prior_nll", "Prior"])
    aug_col = _pick_column(df, ["AUGMENTED_NLL", "augmented_nll", "Augmented_NLL"])
    smiles_col = _pick_column(df, ["SMILES", "smiles", "Smiles"])
    valid_col = _pick_column(df, ["valid", "Valid", "is_valid"])

    if step_col is None:
        print("[WARNING] Could not find a step column in REINVENT training CSV.")
        print("Columns:", list(df.columns))
        return None

    rows = []
    print()
    print("=" * 90)
    print("PEPINVENT TRAINING STATISTICS BY RL STEP")
    print("=" * 90)

    # REINVENT may number steps from 0. Preserve exactly what is stored.
    for step, g in df.groupby(step_col, sort=True):
        row = {"step": int(step)}

        if score_col is not None:
            scores = pd.to_numeric(g[score_col], errors="coerce").dropna()
            row["mean_score"] = float(scores.mean()) if len(scores) else np.nan
            row["max_score"] = float(scores.max()) if len(scores) else np.nan
        else:
            row["mean_score"] = np.nan
            row["max_score"] = np.nan

        if smiles_col is not None:
            s = g[smiles_col].dropna().astype(str)
            row["unique_ratio"] = float(s.nunique() / len(s)) if len(s) else np.nan
            row["n_samples"] = int(len(s))
        else:
            row["unique_ratio"] = np.nan
            row["n_samples"] = int(len(g))

        if valid_col is not None:
            v = g[valid_col]
            if v.dtype == bool:
                row["valid_ratio"] = float(v.mean())
            else:
                vv = pd.to_numeric(v, errors="coerce").dropna()
                row["valid_ratio"] = float(vv.mean()) if len(vv) else np.nan
        else:
            row["valid_ratio"] = np.nan

        for out_name, col in [
            ("agent_nll", agent_col),
            ("prior_nll", prior_col),
            ("augmented_nll", aug_col),
        ]:
            if col is not None:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                row[out_name] = float(vals.mean()) if len(vals) else np.nan
            else:
                row[out_name] = np.nan

        rows.append(row)

        def ff(x):
            return "NA" if pd.isna(x) else f"{x:.4f}"

        print(
            f"Step {int(step)}/{TOTAL_STEPS}: "
            f"mean_score={ff(row['mean_score'])}, "
            f"max_score={ff(row['max_score'])}, "
            f"unique_ratio={ff(row['unique_ratio'])}, "
            f"valid_ratio={ff(row['valid_ratio'])}, "
            f"agent_NLL={ff(row['agent_nll'])}, "
            f"prior_NLL={ff(row['prior_nll'])}, "
            f"augmented_NLL={ff(row['augmented_nll'])}"
        )

    summary = pd.DataFrame(rows).sort_values("step")
    summary.to_csv(output_dir / "training_step_statistics.csv", index=False)

    # Save numpy arrays analogous to the user's WGAN outputs.
    for col in [
        "mean_score",
        "max_score",
        "unique_ratio",
        "valid_ratio",
        "agent_nll",
        "prior_nll",
        "augmented_nll",
    ]:
        np.save(
            output_dir / f"{col}.npy",
            summary[col].to_numpy(dtype=float),
        )

    return summary


def print_final_summary(output_dir, step_summary, best, seed, total_runtime):
    print()
    print("=" * 90)
    print("FINAL PEPINVENT TRAINING SUMMARY")
    print("=" * 90)
    print(f"Seed: {seed}")
    print(f"Total RL steps: {TOTAL_STEPS}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Score multiplier (sigma): {SIGMA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}")

    if step_summary is not None and len(step_summary):
        last = step_summary.iloc[-1]

        def ff(x):
            return "NA" if pd.isna(x) else f"{float(x):.6f}"

        print()
        print("Final training-step statistics:")
        print(f"  step: {int(last['step'])}")
        print(f"  mean_score: {ff(last['mean_score'])}")
        print(f"  max_score: {ff(last['max_score'])}")
        print(f"  unique_ratio: {ff(last['unique_ratio'])}")
        print(f"  valid_ratio: {ff(last['valid_ratio'])}")
        print(f"  agent_NLL: {ff(last['agent_nll'])}")
        print(f"  prior_NLL: {ff(last['prior_nll'])}")
        print(f"  augmented_NLL: {ff(last['augmented_nll'])}")

    print()
    print("Best checkpoint:")
    print(f"  step: {best['step']}")
    print(
        "  mean_immunogenicity_score: "
        f"{best['mean_immunogenicity_score']:.6f}"
    )
    print(f"  unique_ratio: {best['unique_ratio']:.6f}")
    print(f"  score_sum: {best['checkpoint_score_sum']:.6f}")
    print(f"  saved as: {output_dir / 'model_best.chkpt'}")

    print()
    print("Last checkpoint:")
    print(f"  step: {TOTAL_STEPS}")
    print(f"  saved as: {output_dir / 'model_last.chkpt'}")

    print()
    print(f"Program runtime: {total_runtime:.2f} s")

    with open(output_dir / "RunTime.txt", "w", encoding="utf-8") as f:
        f.write(f"Program Runtime: {total_runtime:.6f}(s)\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument(
        "--mode",
        choices=["direct", "with_scorer"],
        default="direct",
    )
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--reinvent", default="reinvent")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    start_whole = time.perf_counter()

    here = Path(__file__).resolve().parent
    prior = Path(args.prior).resolve()

    if not prior.exists():
        raise FileNotFoundError(f"PepINVENT prior not found: {prior}")

    if args.output_dir is None:
        output_dir = (
            DATA_ROOT
            / "result"
            / f"PepINVENT_DeepImmuno_{args.mode}_seed{args.seed}"
            / f"step{TOTAL_STEPS}"
        ).resolve()
    else:
        output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    score_script = (
        here / (
            "pepinvent_deepimmuno_direct_score.py"
            if args.mode == "direct"
            else "pepinvent_deepimmuno_with_scorer_score.py"
        )
    ).resolve()

    eval_script = (here / "evaluate_pepinvent_checkpoint.py").resolve()

    if not score_script.exists():
        raise FileNotFoundError(f"Scoring script not found: {score_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    train_masks = output_dir / "pepinvent_9_10_masks_train.smi"
    train_masks.write_text(
        "?|?|?|?|?|?|?|?|?\n"
        "?|?|?|?|?|?|?|?|?|?\n",
        encoding="utf-8",
    )

    eval_masks = output_dir / "pepinvent_9_10_masks_eval64.smi"
    eval_rows = [
        "?|?|?|?|?|?|?|?|?" if i % 2 == 0
        else "?|?|?|?|?|?|?|?|?|?"
        for i in range(EVAL_BATCH_SIZE)
    ]
    eval_masks.write_text("\n".join(eval_rows) + "\n", encoding="utf-8")

    training_toml = output_dir / "training_20x50steps.toml"
    training_toml.write_text(
        make_training_config(
            prior,
            train_masks,
            score_script,
            output_dir,
            args.mode == "with_scorer",
        ),
        encoding="utf-8",
    )

    if not args.skip_training:
        run_cmd([
            args.reinvent,
            "-s", str(args.seed),
            "-l", str(output_dir / "training.log"),
            str(training_toml),
        ])

    # Print/save the closest PepINVENT equivalents of the per-epoch
    # statistics printed by the provided WGAN code.
    step_summary = print_and_save_training_statistics(output_dir)

    best = None
    all_rows = []

    for step in range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY):
        checkpoint = output_dir / f"model_step_{step:04d}.chkpt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Expected checkpoint not found: {checkpoint}"
            )

        result = evaluate_checkpoint(
            args.reinvent,
            checkpoint,
            step,
            eval_masks,
            output_dir,
            eval_script,
            args.seed,
        )
        all_rows.append(result)

        if best is None or result["checkpoint_score_sum"] > best["checkpoint_score_sum"]:
            best = result
            shutil.copy2(checkpoint, output_dir / "model_best.chkpt")

            with open(output_dir / "best_model_info.txt", "w", encoding="utf-8") as f:
                f.write(f"seed: {args.seed}\n")
                f.write(f"num_steps_requested: {TOTAL_STEPS}\n")
                f.write(f"best_step: {best['step']}\n")
                f.write(
                    "mean_immunogenicity_score: "
                    f"{best['mean_immunogenicity_score']:.10f}\n"
                )
                f.write(f"unique_ratio: {best['unique_ratio']:.10f}\n")
                f.write(
                    "checkpoint_score_sum: "
                    f"{best['checkpoint_score_sum']:.10f}\n"
                )

            print(
                f"[BEST MODEL UPDATED] step={best['step']}, "
                f"sum={best['checkpoint_score_sum']:.6f}"
            )

    last_checkpoint = output_dir / f"model_step_{TOTAL_STEPS:04d}.chkpt"
    shutil.copy2(last_checkpoint, output_dir / "model_last.chkpt")

    pd.DataFrame(all_rows).to_csv(
        output_dir / "checkpoint_selection_history.csv",
        index=False,
    )

    print(
        f"Best checkpoint: step={best['step']}, "
        f"mean_imm_score={best['mean_immunogenicity_score']:.6f}, "
        f"unique_ratio={best['unique_ratio']:.6f}, "
        f"sum={best['checkpoint_score_sum']:.6f}"
    )
    print(f"Last checkpoint saved at step {TOTAL_STEPS}")

    stop_whole = time.perf_counter()
    print_final_summary(
        output_dir,
        step_summary,
        best,
        args.seed,
        stop_whole - start_whole,
    )


if __name__ == "__main__":
    main()
