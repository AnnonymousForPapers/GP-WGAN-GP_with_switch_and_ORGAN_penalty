#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script expects quantify_peptide_novelty.py in the same folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantify_peptide_novelty import (
    load_unique_peptides,
    nearest_training_neighbors,
    build_summary,
    write_summary_csv,
)

import pandas as pd


def find_evaluation_dirs(root: Path) -> list[Path]:
    """Find evaluator output folders recursively."""
    dirs = []
    for p in root.rglob("generated_filtered_unique.csv"):
        if p.is_file():
            dirs.append(p.parent.resolve())
    return sorted(set(dirs))


def is_complete(eval_dir: Path) -> bool:
    out_dir = eval_dir / "novelty_vs_training"
    required = [
        out_dir / "novelty_per_peptide.csv",
        out_dir / "novelty_summary.csv",
        out_dir / "novelty_summary.json",
        out_dir / "min_edit_distance_distribution.csv",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def process_one(
    eval_dir: Path,
    training_path: Path,
    training_peptides: list[str],
    training_info: dict,
    batch_size: int,
):
    generated_path = eval_dir / "generated_filtered_unique.csv"
    out_dir = eval_dir / "novelty_vs_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated, generated_info = load_unique_peptides(generated_path)

    print(f"Generated unique peptides: {len(generated):,}", flush=True)
    print(f"Training unique peptides:  {len(training_peptides):,}", flush=True)

    result = nearest_training_neighbors(
        generated,
        training_peptides,
        batch_size=batch_size,
    )

    per_peptide_path = out_dir / "novelty_per_peptide.csv"
    result.to_csv(per_peptide_path, index=False)

    summary = build_summary(
        result,
        generated_info,
        training_info,
    )

    summary_json = out_dir / "novelty_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    summary_csv = out_dir / "novelty_summary.csv"
    write_summary_csv(summary, summary_csv)

    edit_dist = (
        result["min_edit_distance"]
        .value_counts()
        .sort_index()
        .rename_axis("min_edit_distance")
        .reset_index(name="count")
    )
    edit_dist["percent"] = 100.0 * edit_dist["count"] / len(result)
    edit_dist.to_csv(
        out_dir / "min_edit_distance_distribution.csv",
        index=False,
    )

    print(
        f"[DONE] exact overlap={summary['exact_overlap_percent']:.2f}% | "
        f"mean min edit distance={summary['min_edit_distance_mean']:.3f} | "
        f"mean NN similarity={summary['nearest_neighbor_similarity_mean']:.4f} | "
        f"mean identity={summary['sequence_identity_percent_mean']:.2f}%",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Recursively run peptide novelty analysis for every unified "
            "evaluator output folder under a root directory."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root result folder to search recursively.",
    )
    ap.add_argument(
        "--training",
        required=True,
        help="Training/reference peptide dataset.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Generated peptides per distance-computation block.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of evaluation folders to process.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute folders even when novelty outputs already exist.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    training_path = Path(args.training).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Root folder does not exist: {root}")
    if not training_path.exists():
        raise SystemExit(f"Training file does not exist: {training_path}")

    training_peptides, training_info = load_unique_peptides(training_path)

    eval_dirs = find_evaluation_dirs(root)
    if args.limit is not None:
        eval_dirs = eval_dirs[:max(0, int(args.limit))]

    print("=" * 90)
    print("RECURSIVE PEPTIDE NOVELTY EVALUATION")
    print("=" * 90)
    print(f"Root:              {root}")
    print(f"Training dataset:  {training_path}")
    print(f"Training peptides: {len(training_peptides):,}")
    print(f"Evaluation folders found: {len(eval_dirs):,}")
    print(f"Batch size:        {args.batch_size:,}")
    print("=" * 90)

    if not eval_dirs:
        print("No generated_filtered_unique.csv files were found.")
        return 0

    done = 0
    skipped = 0
    failed = 0

    for i, eval_dir in enumerate(eval_dirs, start=1):
        print("")
        print("=" * 90)
        print(f"[{i}/{len(eval_dirs)}] {eval_dir}")
        print("=" * 90)

        if not args.force and is_complete(eval_dir):
            print("[SKIPPED] novelty_vs_training output already complete.")
            skipped += 1
            continue

        try:
            process_one(
                eval_dir,
                training_path,
                training_peptides,
                training_info,
                batch_size=args.batch_size,
            )
            done += 1
        except Exception as exc:
            failed += 1
            print(
                f"[FAILED] {type(exc).__name__}: {exc}",
                flush=True,
            )

    print("")
    print("=" * 90)
    print("FINISHED")
    print(f"Completed: {done:,}")
    print(f"Skipped:   {skipped:,}")
    print(f"Failed:    {failed:,}")
    print("=" * 90)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
