#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pepinvent_deepimmuno_direct_score import (
    DeepImmuno,
    smiles_to_natural_peptide,
)

EVAL_BATCH_SIZE = 64


def find_smiles_column(df):
    for candidate in ["SMILES", "smiles", "Smiles", "output", "Output"]:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(
        f"Could not identify SMILES column. Columns: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--data-root", default="../")
    args = ap.parse_args()

    df = pd.read_csv(args.sample_csv)
    smiles_col = find_smiles_column(df)
    smiles = df[smiles_col].astype(str).tolist()

    if len(smiles) < EVAL_BATCH_SIZE:
        raise RuntimeError(
            f"Sampling returned {len(smiles)} rows; expected at least {EVAL_BATCH_SIZE}."
        )

    smiles = smiles[:EVAL_BATCH_SIZE]

    deepimmuno = DeepImmuno(args.data_root)

    peptides = [smiles_to_natural_peptide(s) for s in smiles]
    valid_idx = [i for i, p in enumerate(peptides) if p is not None]

    imm_scores = np.zeros(EVAL_BATCH_SIZE, dtype=np.float32)
    if valid_idx:
        valid_peptides = [peptides[i] for i in valid_idx]
        vals = deepimmuno.score(valid_peptides)
        for i, score in zip(valid_idx, vals):
            imm_scores[i] = float(score)

    unique_peptides = set(p for p in peptides if p is not None)
    unique_count = len(unique_peptides)
    unique_ratio = unique_count / float(EVAL_BATCH_SIZE)

    mean_imm = float(np.mean(imm_scores))
    checkpoint_score = mean_imm + unique_ratio

    out_df = pd.DataFrame({
        "SMILES": smiles,
        "peptide": [p if p is not None else "" for p in peptides],
        "HLA": ["HLA-A*0201"] * EVAL_BATCH_SIZE,
        "immunogenicity": imm_scores,
        "valid_natural_9_10mer": [p is not None for p in peptides],
    })
    out_df.to_csv(args.output_csv, index=False)

    result = {
        "mean_immunogenicity_score": mean_imm,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "checkpoint_score_sum": checkpoint_score,
    }

    Path(args.output_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
