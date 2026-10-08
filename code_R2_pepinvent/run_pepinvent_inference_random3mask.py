#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pepinvent_runtime_env import make_runtime_env
from pepinvent_chuckles import load_bladder_peptides, prepare_random_masks
from pepinvent_deepimmuno_direct_score import (
    smiles_to_natural_peptide,
    score_sequences_with_tf,
)


def q(x):
    return str(x).replace("\\", "\\\\")


def find_smiles_column(df):
    for c in ["SMILES", "smiles", "Smiles", "output", "Output"]:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not identify generated SMILES column. Columns: {list(df.columns)}"
    )


def make_sampling_config(model_file, masks_file, output_file, n):
    return "\n".join([
        'run_type = "sampling"',
        "",
        'device = "cuda:0"',
        "",
        "[parameters]",
        f'model_file = "{q(model_file)}"',
        f'smiles_file = "{q(masks_file)}"',
        'sample_strategy = "multinomial"',
        "temperature = 1.0",
        f'output_file = "{q(output_file)}"',
        f"num_smiles = {int(n)}",
        # Keep one row per masked input so source/mask metadata stays aligned.
        "unique_molecules = false",
        "randomize_smiles = false",
        "",
    ]) + "\n"


def run_cmd(cmd, runtime_overrides=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = make_runtime_env() if runtime_overrides else None
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_best.chkpt or model_last.chkpt")
    ap.add_argument("--reinvent", default="reinvent")
    ap.add_argument(
        "--bladder-csv",
        default="../data/neoepitopes/Bladder.4.0_test_mut.csv",
    )
    ap.add_argument("--data-root", default="../")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--mask-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    model = Path(args.model).resolve()
    if not model.exists():
        raise FileNotFoundError(model)

    bladder_csv = Path(args.bladder_csv).resolve()
    peptides, peptide_col = load_bladder_peptides(bladder_csv)

    if args.output_dir is None:
        out = model.parent / f"inference_random3mask_n{args.num_samples}_seed{args.seed}"
    else:
        out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    masks_smi = out / "inference_masks.smi"
    mask_map = out / "inference_masks.csv"

    prepare_random_masks(
        peptides,
        args.num_samples,
        masks_smi,
        mask_map,
        seed=args.seed,
        mask_count=args.mask_count,
    )

    print("=" * 90)
    print("PEPINVENT INFERENCE INPUT")
    print("=" * 90)
    print(f"Model: {model}")
    print(f"Bladder CSV: {bladder_csv}")
    print(f"Detected peptide column: {peptide_col}")
    print(f"Source pool: {len(peptides)} unique natural 9/10-mers")
    print(f"Requested samples: {args.num_samples}")
    print(f"Masked residues per sample: {args.mask_count}")
    print(
        "Each inference row independently samples one source peptide and "
        f"{args.mask_count} distinct mask positions."
    )

    raw_csv = out / "pepinvent_raw_sampling.csv"
    toml = out / "sampling.toml"
    log = out / "sampling.log"

    toml.write_text(
        make_sampling_config(model, masks_smi, raw_csv, args.num_samples),
        encoding="utf-8",
    )


    run_cmd([
        args.reinvent,
        "-d", "cuda:0",
        "-s", str(args.seed),
        "-l", str(log),
        str(toml),
    ], runtime_overrides=True)

    raw = pd.read_csv(raw_csv)
    meta = pd.read_csv(mask_map)

    if len(raw) != len(meta):
        raise RuntimeError(
            "REINVENT sampling row count does not match the masked-input count: "
            f"{len(raw)} vs {len(meta)}. "
            "The script keeps unique_molecules=false specifically to preserve alignment."
        )

    smiles_col = find_smiles_column(raw)
    generated_smiles = raw[smiles_col].astype(str).tolist()

    generated_peptides = []
    for smi in generated_smiles:
        try:
            generated_peptides.append(smiles_to_natural_peptide(smi))
        except Exception:
            generated_peptides.append(None)

    valid_idx = [i for i, p in enumerate(generated_peptides) if p is not None]
    scores = np.zeros(len(generated_peptides), dtype=np.float32)

    if valid_idx:
        valid_peptides = [generated_peptides[i] for i in valid_idx]
        valid_scores = score_sequences_with_tf(
            valid_peptides,
            args.data_root,
            Path(__file__).resolve().parent / "deepimmuno_tf_score_sequences.py",
        )
        for i, score in zip(valid_idx, valid_scores):
            scores[i] = float(score)

    result = meta.copy()
    result["generated_smiles"] = generated_smiles
    result["generated_peptide"] = [
        p if p is not None else "" for p in generated_peptides
    ]
    result["valid_natural_9_10mer"] = [p is not None for p in generated_peptides]
    result["DeepImmuno_score"] = scores

    # Carry useful REINVENT columns through if present.
    for c in raw.columns:
        if c not in result.columns and c != smiles_col:
            result[f"reinvent_{c}"] = raw[c].values

    result_csv = out / "generated_peptides_scored.csv"
    result.to_csv(result_csv, index=False)

    valid = [p for p in generated_peptides if p is not None]
    unique = set(valid)

    summary = {
        "model": str(model),
        "seed": args.seed,
        "num_requested": args.num_samples,
        "mask_count": args.mask_count,
        "num_valid_natural_9_10mer": len(valid),
        "num_unique_valid_peptides": len(unique),
        "unique_rate_over_requested": len(unique) / args.num_samples,
        "valid_rate": len(valid) / args.num_samples,
        "mean_DeepImmuno_over_all_requested": float(np.mean(scores)),
        "max_DeepImmuno": float(np.max(scores)) if len(scores) else 0.0,
    }

    if valid_idx:
        summary["mean_DeepImmuno_valid_only"] = float(
            np.mean(scores[valid_idx])
        )
    else:
        summary["mean_DeepImmuno_valid_only"] = 0.0

    (out / "inference_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("PEPINVENT INFERENCE SUMMARY")
    print("=" * 90)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Output CSV: {result_csv}")


if __name__ == "__main__":
    main()
