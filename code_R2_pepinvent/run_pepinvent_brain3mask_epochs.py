#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pepinvent_runtime_env import make_runtime_env
from pepinvent_chuckles import (
    load_peptides,
    prepare_all_training_masks,
    prepare_random_masks,
)

TOTAL_EPOCHS = 1000
CHECKPOINT_EVERY_EPOCHS = 50
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
REAL_PEPTIDE_CSV = DATA_ROOT / "data/neoepitopes/Brain.4.0_test_mut.csv"


def q(x):
    return str(x).replace("\\", "\\\\")
    
def run_cmd_training(cmd, log_file, output_dir, runtime_overrides=False):
    print("+", " ".join(map(str, cmd)), flush=True)

    env = make_runtime_env() if runtime_overrides else None

    log_file = Path(log_file)
    output_dir = Path(output_dir)

    if log_file.exists():
        log_file.unlink()

    # Let REINVENT write its normal detailed log itself.
    cmd = list(cmd)
    cmd[1:1] = ["-l", str(log_file)]

    process = subprocess.Popen(
        cmd,
        env=env,
    )

    printed_global_steps = set()
    stage_no = 1

    start_time = time.perf_counter()
    previous_print_time = start_time

    while process.poll() is None:

        csv_file = output_dir / f"training_{stage_no}.csv"

        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)

                if "step" in df.columns and len(df) > 0:

                    for local_step in sorted(df["step"].dropna().unique()):

                        local_step = int(local_step)

                        global_step = (
                            (stage_no - 1)
                            * CHECKPOINT_EVERY_EPOCHS
                            + local_step
                        )

                        if global_step in printed_global_steps:
                            continue

                        g = df[df["step"] == local_step]

                        # Only print once the complete batch of 32
                        # has been written to the CSV.
                        if len(g) < TRAIN_BATCH_SIZE:
                            continue

                        now = time.perf_counter()
                        step_time = now - previous_print_time
                        previous_print_time = now

                        mean_score = pd.to_numeric(
                            g["Score"],
                            errors="coerce",
                        ).mean()

                        max_score = pd.to_numeric(
                            g["Score"],
                            errors="coerce",
                        ).max()

                        agent_nll = pd.to_numeric(
                            g["Agent"],
                            errors="coerce",
                        ).mean()

                        prior_nll = pd.to_numeric(
                            g["Prior"],
                            errors="coerce",
                        ).mean()

                        target_nll = pd.to_numeric(
                            g["Target"],
                            errors="coerce",
                        ).mean()

                        unique_rate = (
                            g["SMILES"].astype(str).nunique()
                            / len(g)
                        )

                        valid_rate = pd.to_numeric(
                            g["SMILES_state"],
                            errors="coerce",
                        ).mean()

                        mean_deep = pd.to_numeric(
                            g["DeepImmuno (raw)"],
                            errors="coerce",
                        ).mean()

                        print(
                            f"Epoch {global_step}/{TOTAL_EPOCHS}: "
                            f"mean_score={mean_score:.4f}, "
                            f"max_score={max_score:.4f}, "
                            f"Agent_NLL={agent_nll:.4f}, "
                            f"Prior_NLL={prior_nll:.4f}, "
                            f"Target_NLL={target_nll:.4f}, "
                            f"DeepImmuno={mean_deep:.4f}, "
                            f"unq_rate={unique_rate:.4f}, "
                            f"valid_rate={valid_rate:.4f}, "
                            f"Time={step_time:.2f}(s)",
                            flush=True,
                        )

                        printed_global_steps.add(global_step)

            except (pd.errors.EmptyDataError, PermissionError):
                pass

        # If stage N has finished and stage N+1 CSV exists,
        # advance to the next stage.
        next_csv = output_dir / f"training_{stage_no + 1}.csv"

        if next_csv.exists():
            stage_no += 1

        time.sleep(0.5)

    # Final pass, because REINVENT may finish immediately after writing
    # the last batch.
    while True:
        csv_file = output_dir / f"training_{stage_no}.csv"

        if not csv_file.exists():
            break

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            break

        for local_step in sorted(df["step"].dropna().unique()):
            local_step = int(local_step)

            global_step = (
                (stage_no - 1)
                * CHECKPOINT_EVERY_EPOCHS
                + local_step
            )

            if global_step in printed_global_steps:
                continue

            g = df[df["step"] == local_step]

            if len(g) < TRAIN_BATCH_SIZE:
                continue

            mean_score = pd.to_numeric(
                g["Score"], errors="coerce"
            ).mean()

            max_score = pd.to_numeric(
                g["Score"], errors="coerce"
            ).max()

            agent_nll = pd.to_numeric(
                g["Agent"], errors="coerce"
            ).mean()

            prior_nll = pd.to_numeric(
                g["Prior"], errors="coerce"
            ).mean()

            target_nll = pd.to_numeric(
                g["Target"], errors="coerce"
            ).mean()

            unique_rate = (
                g["SMILES"].astype(str).nunique()
                / len(g)
            )

            valid_rate = pd.to_numeric(
                g["SMILES_state"],
                errors="coerce",
            ).mean()

            mean_deep = pd.to_numeric(
                g["DeepImmuno (raw)"],
                errors="coerce",
            ).mean()

            print(
                f"Epoch {global_step}/{TOTAL_EPOCHS}: "
                f"mean_score={mean_score:.4f}, "
                f"max_score={max_score:.4f}, "
                f"Agent_NLL={agent_nll:.4f}, "
                f"Prior_NLL={prior_nll:.4f}, "
                f"Target_NLL={target_nll:.4f}, "
                f"DeepImmuno={mean_deep:.4f}, "
                f"unq_rate={unique_rate:.4f}, "
                f"valid_rate={valid_rate:.4f}",
                flush=True,
            )

            printed_global_steps.add(global_step)

        next_csv = output_dir / f"training_{stage_no + 1}.csv"

        if next_csv.exists():
            stage_no += 1
        else:
            break

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            cmd,
        )

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
        f"params.args = '-u {q(score_script)} --data-root ../{extra}'",
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

    for epoch in range(CHECKPOINT_EVERY_EPOCHS, TOTAL_EPOCHS + 1, CHECKPOINT_EVERY_EPOCHS):
        # REINVENT max_steps is a hard limit that terminates the ENTIRE
        # staged-learning run.  Do not use max_steps=50 for periodic
        # checkpoints.  Instead, force a normal stage transition after
        # exactly CHECKPOINT_EVERY_EPOCHS updates.
        #
        # SimpleTerminator uses zero-based `step` and tests:
        #     step > min_steps and score >= max_score
        # so for 50 reported updates (steps 0..49), min_steps must be 48.
        stage_min_steps = max(0, CHECKPOINT_EVERY_EPOCHS - 2)

        lines.extend([
            "[[stage]]",
            f'chkpt_file = "{q(output_dir / f"model_epoch_{epoch:04d}.chkpt")}"',
            'termination = "simple"',
            # DeepImmuno/REINVENT total scores are non-negative.  Thus,
            # after stage_min_steps this condition is guaranteed for any
            # finite score, causing a normal stage transition/checkpoint.
            "max_score = 0.0",
            f"min_steps = {stage_min_steps}",
            # Must be larger than the planned 50-step stage.  Hitting
            # max_steps would terminate all remaining stages.
            f"max_steps = {TOTAL_EPOCHS}",
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


def run_cmd(cmd, runtime_overrides=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = make_runtime_env() if runtime_overrides else None
    subprocess.run(cmd, check=True, env=env)


def evaluate_checkpoint(
    reinvent_exe,
    checkpoint,
    epoch,
    eval_masks,
    output_dir,
    eval_script,
    base_seed,
    reinvent_log_level,
):
    sample_csv = output_dir / f"df_PepINVENT_all_epoch{epoch}.csv"
    sample_toml = output_dir / f"sampling_epoch_{epoch:04d}.toml"
    sample_log = output_dir / f"sampling_epoch_{epoch:04d}.log"

    sample_toml.write_text(
        make_sampling_config(checkpoint, eval_masks, sample_csv),
        encoding="utf-8",
    )

    eval_seed = base_seed + epoch // CHECKPOINT_EVERY_EPOCHS

    # Sampling uses REINVENT4 environment.
    run_cmd([
        reinvent_exe,
        "--log-level", reinvent_log_level,
        "-s", str(eval_seed),
        "-l", str(sample_log),
        str(sample_toml),
    ], runtime_overrides=True)

    eval_csv = output_dir / f"checkpoint_evaluation_epoch{epoch}.csv"
    eval_json = output_dir / f"checkpoint_evaluation_epoch{epoch}.json"

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
    result["epoch"] = epoch
    result["checkpoint"] = str(checkpoint)

    print(
        "Checkpoint evaluation: "
        f"epoch={epoch}, "
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
    Print one compact PepINVENT summary per RL epoch and save the analogous
    arrays/statistics.

    We do not invent GAN-only quantities such as d_real_loss or W_dist.
    The closest PepINVENT/REINVENT quantities are reported instead.
    """
    csvs = _find_training_summary_csvs(output_dir)
    if not csvs:
        print("[WARNING] No REINVENT training summary CSV was found.")
        print("          REINVENT's native training.log still contains its own RL-epoch/step output.")
        return None

    frames = []
    for p in csvs:
        try:
            d = pd.read_csv(p)
            if not len(d):
                continue

            # REINVENT writes one CSV per stage:
            # training_1.csv, training_2.csv, ...
            m = re.search(r"training_(\\d+)\\.csv$", p.name)
            stage_no = int(m.group(1)) if m else len(frames) + 1

            local_step_col = _pick_column(d, ["step", "Step", "STEP"])
            if local_step_col is None:
                print(f"[WARNING] Could not find step column in {p}; skipping.")
                continue

            local_step = pd.to_numeric(
                d[local_step_col], errors="coerce"
            )

            # REINVENT reports Step = step+1, so local steps are normally
            # 1..50.  Convert them to 1..1000 across stages.
            d["_global_epoch"] = (
                (stage_no - 1) * CHECKPOINT_EVERY_EPOCHS + local_step
            )
            frames.append(d)
        except Exception as e:
            print(f"[WARNING] Could not read {p}: {e}")

    if not frames:
        print("[WARNING] REINVENT training summary CSV files were empty.")
        return None

    df = pd.concat(frames, ignore_index=True)

    step_col = "_global_epoch"
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
    print("PEPINVENT TRAINING STATISTICS BY RL EPOCH")
    print("=" * 90)

    # REINVENT may number steps from 0. Preserve exactly what is stored.
    for epoch, g in df.groupby(step_col, sort=True):
        row = {"epoch": int(epoch)}

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
            f"Epoch {int(epoch)}/{TOTAL_EPOCHS}: "
            f"mean_score={ff(row['mean_score'])}, "
            f"max_score={ff(row['max_score'])}, "
            f"unique_ratio={ff(row['unique_ratio'])}, "
            f"valid_ratio={ff(row['valid_ratio'])}, "
            f"agent_NLL={ff(row['agent_nll'])}, "
            f"prior_NLL={ff(row['prior_nll'])}, "
            f"augmented_NLL={ff(row['augmented_nll'])}"
        )

    summary = pd.DataFrame(rows).sort_values("epoch")
    summary.to_csv(output_dir / "training_epoch_statistics.csv", index=False)

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
    print(f"Total RL epochs: {TOTAL_EPOCHS}")
    print(f"Training batch size: {TRAIN_BATCH_SIZE}")
    print(f"Score multiplier (sigma): {SIGMA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}")

    if step_summary is not None and len(step_summary):
        last = step_summary.iloc[-1]

        def ff(x):
            return "NA" if pd.isna(x) else f"{float(x):.6f}"

        print()
        print("Final training-epoch statistics:")
        print(f"  epoch: {int(last['epoch'])}")
        print(f"  mean_score: {ff(last['mean_score'])}")
        print(f"  max_score: {ff(last['max_score'])}")
        print(f"  unique_ratio: {ff(last['unique_ratio'])}")
        print(f"  valid_ratio: {ff(last['valid_ratio'])}")
        print(f"  agent_NLL: {ff(last['agent_nll'])}")
        print(f"  prior_NLL: {ff(last['prior_nll'])}")
        print(f"  augmented_NLL: {ff(last['augmented_nll'])}")

    print()
    print("Best checkpoint:")
    print(f"  epoch: {best['epoch']}")
    print(
        "  mean_immunogenicity_score: "
        f"{best['mean_immunogenicity_score']:.6f}"
    )
    print(f"  unique_ratio: {best['unique_ratio']:.6f}")
    print(f"  score_sum: {best['checkpoint_score_sum']:.6f}")
    print(f"  saved as: {output_dir / 'model_best.chkpt'}")

    print()
    print("Last checkpoint:")
    print(f"  epoch: {TOTAL_EPOCHS}")
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
    ap.add_argument("--mask-count", type=int, default=3)
    ap.add_argument("--masks-per-peptide", type=int, default=1)
    ap.add_argument("--num-epochs", type=int, default=1000)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--reinvent-log-level", default="verbose", choices=["critical", "error", "warning", "info", "debug", "verbose"])
    args = ap.parse_args()

    global TOTAL_EPOCHS, CHECKPOINT_EVERY_EPOCHS
    TOTAL_EPOCHS = int(args.num_epochs)
    CHECKPOINT_EVERY_EPOCHS = int(args.checkpoint_every_epochs)

    if TOTAL_EPOCHS <= 0:
        raise ValueError("--num-epochs must be > 0")
    if CHECKPOINT_EVERY_EPOCHS <= 0:
        raise ValueError("--checkpoint-every-epochs must be > 0")
    if TOTAL_EPOCHS % CHECKPOINT_EVERY_EPOCHS != 0:
        raise ValueError(
            "--num-epochs must be divisible by --checkpoint-every-epochs "
            "for this checkpoint-selection implementation."
        )

    start_whole = time.perf_counter()

    here = Path(__file__).resolve().parent
    prior = Path(args.prior).resolve()

    if not prior.exists():
        raise FileNotFoundError(f"PepINVENT prior not found: {prior}")

    if args.output_dir is None:
        output_dir = (
            DATA_ROOT
            / "result_brain"
            / f"PepINVENT_DeepImmuno_{args.mode}_seed{args.seed}"
            / f"epoch{TOTAL_EPOCHS}"
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

    if args.mask_count != 3:
        print(
            f"[NOTE] mask_count={args.mask_count}; the requested/default experiment uses 3."
        )

    train_masks = output_dir / "pepinvent_brain_train_masks.smi"
    train_map_csv = output_dir / "pepinvent_brain_train_masks.csv"

    prep_info = prepare_all_training_masks(
        REAL_PEPTIDE_CSV,
        train_masks,
        train_map_csv,
        seed=args.seed,
        mask_count=args.mask_count,
        masks_per_peptide=args.masks_per_peptide,
    )

    print("=" * 90)
    print("PEPINVENT RL INPUT PREPARATION")
    print("=" * 90)
    print(f"Brain CSV: {REAL_PEPTIDE_CSV}")
    print(f"Detected peptide column: {prep_info['peptide_column']}")
    print(
        "Valid unique natural 9/10-mer brain peptides: "
        f"{prep_info['num_unique_source_peptides']}"
    )
    print(
        "Masked RL inputs: "
        f"{prep_info['num_masked_inputs']} "
        f"({args.mask_count} masked residues each)"
    )
    print(f"Mask mapping CSV: {train_map_csv}")
    print(f"REINVENT log level: {args.reinvent_log_level}")
    print("REINVENT torch device: cuda:0")
    print(f"RL source-pool size: {prep_info['num_masked_inputs']}")
    print("RL samples per update: 32 total (32 sources x 1 completion)")
    print("No installed REINVENT .py source file is edited by this workflow")
    print("PepINVENT output constraint: runtime-only Natural20 canonical amino acids")
    print("Natural20 likelihood: runtime-constrained/renormalized for Agent and Prior NLL")
    print(f"Checkpoint stages: every {CHECKPOINT_EVERY_EPOCHS} updates; "
          f"{TOTAL_EPOCHS // CHECKPOINT_EVERY_EPOCHS} stages total")
    print()

    # Load the same complete source pool for checkpoint evaluation.  A fresh
    # random batch of 64 source peptides + 3 random positions is generated at
    # each checkpoint, analogous to the fresh GAN batch used by the original
    # checkpoint rule.
    brain_peptides, _ = load_peptides(REAL_PEPTIDE_CSV)

    training_toml = output_dir / "training_epochs.toml"
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
        print(
            "\nUsing runtime-only PepINVENT Natural20 + exact-batch overrides. "
            "Installed REINVENT source files are not modified.",
            flush=True,
        )

        run_cmd_training([
            args.reinvent,
            "-d", "cuda:0",
            "--log-level", args.reinvent_log_level,
            "-s", str(args.seed),
            str(training_toml),
        ],
            log_file=output_dir / "training.log",
            output_dir=output_dir,
            runtime_overrides=True,
        )

    # Print/save the closest PepINVENT equivalents of the per-epoch
    # statistics printed by the provided WGAN code.
    step_summary = print_and_save_training_statistics(output_dir)

    best = None
    all_rows = []

    expected_checkpoints = [
        output_dir / f"model_epoch_{epoch:04d}.chkpt"
        for epoch in range(
            CHECKPOINT_EVERY_EPOCHS,
            TOTAL_EPOCHS + 1,
            CHECKPOINT_EVERY_EPOCHS,
        )
    ]
    missing_checkpoints = [p for p in expected_checkpoints if not p.exists()]
    if missing_checkpoints:
        existing = sorted(output_dir.glob("model_epoch_*.chkpt"))
        raise FileNotFoundError(
            "Training finished without all requested checkpoints. "
            f"Missing {len(missing_checkpoints)} checkpoint(s): "
            + ", ".join(p.name for p in missing_checkpoints[:10])
            + (" ..." if len(missing_checkpoints) > 10 else "")
            + "\nExisting checkpoints: "
            + ", ".join(p.name for p in existing)
        )

    print(
        f"Verified {len(expected_checkpoints)} checkpoints: "
        f"epoch {CHECKPOINT_EVERY_EPOCHS} through {TOTAL_EPOCHS}"
    )

    for epoch in range(
        CHECKPOINT_EVERY_EPOCHS,
        TOTAL_EPOCHS + 1,
        CHECKPOINT_EVERY_EPOCHS,
    ):
        checkpoint = output_dir / f"model_epoch_{epoch:04d}.chkpt"

        eval_masks = output_dir / f"pepinvent_eval_masks_epoch{epoch:04d}.smi"
        eval_map_csv = output_dir / f"pepinvent_eval_masks_epoch{epoch:04d}.csv"

        prepare_random_masks(
            brain_peptides,
            EVAL_BATCH_SIZE,
            eval_masks,
            eval_map_csv,
            seed=args.seed + 100000 + epoch,
            mask_count=args.mask_count,
        )

        result = evaluate_checkpoint(
            args.reinvent,
            checkpoint,
            epoch,
            eval_masks,
            output_dir,
            eval_script,
            args.seed,
        args.reinvent_log_level,
        )
        all_rows.append(result)

        if best is None or result["checkpoint_score_sum"] > best["checkpoint_score_sum"]:
            best = result
            shutil.copy2(checkpoint, output_dir / "model_best.chkpt")

            with open(output_dir / "best_model_info.txt", "w", encoding="utf-8") as f:
                f.write(f"seed: {args.seed}\n")
                f.write(f"num_epochs_requested: {TOTAL_EPOCHS}\n")
                f.write(f"best_epoch: {best['epoch']}\n")
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
                f"[BEST MODEL UPDATED] epoch={best['epoch']}, "
                f"sum={best['checkpoint_score_sum']:.6f}"
            )

    last_checkpoint = output_dir / f"model_epoch_{TOTAL_EPOCHS:04d}.chkpt"
    shutil.copy2(last_checkpoint, output_dir / "model_last.chkpt")

    pd.DataFrame(all_rows).to_csv(
        output_dir / "checkpoint_selection_history.csv",
        index=False,
    )

    print(
        f"Best checkpoint: epoch={best['epoch']}, "
        f"mean_imm_score={best['mean_immunogenicity_score']:.6f}, "
        f"unique_ratio={best['unique_ratio']:.6f}, "
        f"sum={best['checkpoint_score_sum']:.6f}"
    )
    print(f"Last checkpoint saved at epoch {TOTAL_EPOCHS}")

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
