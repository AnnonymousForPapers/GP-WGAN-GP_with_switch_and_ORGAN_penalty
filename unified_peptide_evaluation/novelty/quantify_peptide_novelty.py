#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
except ImportError as exc:
    raise ImportError(
        "This script requires rapidfuzz. Install it with:\n"
        "  pip install rapidfuzz"
    ) from exc


NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_colname(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_peptide_column(df: pd.DataFrame) -> str:
    aliases = [
        "peptide",
        "generated_peptide",
        "sequence",
        "seq",
        "peptide_sequence",
    ]
    normalized = {normalize_colname(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_colname(alias)
        if key in normalized:
            return normalized[key]
    raise KeyError(
        "Could not find a peptide column. "
        f"Available columns: {list(df.columns)}"
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # TXT: first try as tabular text. If that does not yield a useful peptide
    # column, treat each non-empty line as one peptide.
    if suffix == ".txt":
        try:
            df = pd.read_csv(path, sep="\t")
            find_peptide_column(df)
            return df
        except Exception:
            vals = [
                x.strip()
                for x in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if x.strip()
            ]
            return pd.DataFrame({"peptide": vals})

    # Generic fallback.
    return pd.read_csv(path)


def clean_peptide(raw, lengths=(9, 10)):
    if pd.isna(raw):
        return None

    pep = str(raw).strip().upper()

    # Same placeholder convention used by the unified evaluator:
    # reject sequences with >=2 '-' characters, then remove a single '-'.
    if pep.count("-") >= 2:
        return None
    pep = pep.replace("-", "")

    if len(pep) not in lengths:
        return None
    if any(ch not in NATURAL_AA for ch in pep):
        return None

    return pep


def load_unique_peptides(path: Path, lengths=(9, 10)) -> Tuple[list[str], dict]:
    df = read_table(path)
    pcol = find_peptide_column(df)

    raw_n = len(df)

    cleaned = []
    for x in df[pcol]:
        p = clean_peptide(x, lengths=lengths)
        if p is not None:
            cleaned.append(p)

    valid_n = len(cleaned)
    unique = list(dict.fromkeys(cleaned))

    info = {
        "path": str(path.resolve()),
        "peptide_column": str(pcol),
        "rows_raw": int(raw_n),
        "rows_valid_9_10mer": int(valid_n),
        "unique_valid_9_10mer": int(len(unique)),
    }

    return unique, info


def aligned_identity_min_edit(a: str, b: str) -> Tuple[float, int, int]:
    """
    Global sequence identity for a minimum-edit alignment.

    Optimization priority:
      1. minimum Levenshtein edit cost,
      2. among equally minimal-edit alignments, maximize exact matches,
      3. among remaining ties, minimize alignment length.

    Returns:
      identity_fraction, exact_matches, alignment_length

    identity_fraction = exact_matches / alignment_length
    """
    n, m = len(a), len(b)

    # Each DP cell stores: (edit_cost, negative_matches, alignment_length)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0)

    for i in range(1, n + 1):
        cost, neg_matches, alen = dp[i - 1][0]
        dp[i][0] = (cost + 1, neg_matches, alen + 1)

    for j in range(1, m + 1):
        cost, neg_matches, alen = dp[0][j - 1]
        dp[0][j] = (cost + 1, neg_matches, alen + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Diagonal: exact match or substitution.
            c0, nm0, l0 = dp[i - 1][j - 1]
            is_match = a[i - 1] == b[j - 1]
            diag = (
                c0 + (0 if is_match else 1),
                nm0 - (1 if is_match else 0),
                l0 + 1,
            )

            # Delete a[i-1].
            c1, nm1, l1 = dp[i - 1][j]
            delete = (c1 + 1, nm1, l1 + 1)

            # Insert b[j-1].
            c2, nm2, l2 = dp[i][j - 1]
            insert = (c2 + 1, nm2, l2 + 1)

            dp[i][j] = min(diag, delete, insert)

    cost, neg_matches, alignment_length = dp[n][m]
    matches = -neg_matches

    identity = (
        float(matches) / float(alignment_length)
        if alignment_length > 0
        else float("nan")
    )

    return identity, matches, alignment_length


def nearest_training_neighbors(
    generated: list[str],
    training: list[str],
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Find the minimum-Levenshtein-distance training peptide for every generated
    peptide without allocating the full generated x training matrix.

    RapidFuzz computes the expensive pairwise distance block in compiled code.
    """
    if not generated:
        return pd.DataFrame()
    if not training:
        raise ValueError("Training peptide set is empty.")

    rows = []

    for start in range(0, len(generated), batch_size):
        stop = min(start + batch_size, len(generated))
        batch = generated[start:stop]

        # Shape = [batch_size, n_training].
        # dtype is chosen by RapidFuzz; processing only one batch at a time
        # keeps RAM bounded.
        dist = process.cdist(
            batch,
            training,
            scorer=Levenshtein.distance,
            workers=-1,
        )

        nearest_idx = np.argmin(dist, axis=1)
        nearest_dist = dist[np.arange(len(batch)), nearest_idx]

        for local_i, pep in enumerate(batch):
            train_pep = training[int(nearest_idx[local_i])]
            d = int(nearest_dist[local_i])

            max_len = max(len(pep), len(train_pep))
            nn_similarity = 1.0 - (float(d) / float(max_len))

            identity, matches, alignment_length = aligned_identity_min_edit(
                pep, train_pep
            )

            rows.append({
                "peptide": pep,
                "nearest_training_peptide": train_pep,
                "exact_training_overlap": bool(d == 0),
                "min_edit_distance": d,
                "nearest_neighbor_similarity": nn_similarity,
                "sequence_identity_to_nearest": identity,
                "sequence_identity_percent": 100.0 * identity,
                "aligned_exact_matches": int(matches),
                "alignment_length": int(alignment_length),
                "novelty_score": 1.0 - nn_similarity,
            })

        print(
            f"Processed {stop:,}/{len(generated):,} generated peptides",
            flush=True,
        )

    return pd.DataFrame(rows)


def safe_mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(x.mean()) if len(x) else float("nan")


def safe_median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(x.median()) if len(x) else float("nan")


def safe_min(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(x.min()) if len(x) else float("nan")


def safe_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(x.max()) if len(x) else float("nan")


def build_summary(
    result: pd.DataFrame,
    generated_info: dict,
    training_info: dict,
) -> dict:
    n = len(result)

    if n == 0:
        raise ValueError("No generated peptides were evaluated.")

    exact_n = int(result["exact_training_overlap"].sum())

    edit_counts = {}
    for d in [0, 1, 2, 3]:
        edit_counts[str(d)] = int((result["min_edit_distance"] == d).sum())
    edit_counts["4_or_more"] = int(
        (result["min_edit_distance"] >= 4).sum()
    )

    summary = {
        "generated_input": generated_info,
        "training_input": training_info,
        "n_generated_unique_evaluated": int(n),
        "n_training_unique": int(training_info["unique_valid_9_10mer"]),

        "exact_overlap_count": exact_n,
        "exact_overlap_percent": 100.0 * exact_n / n,
        "non_exact_novel_count": int(n - exact_n),
        "non_exact_novel_percent": 100.0 * (n - exact_n) / n,

        "min_edit_distance_mean": safe_mean(result["min_edit_distance"]),
        "min_edit_distance_median": safe_median(result["min_edit_distance"]),
        "min_edit_distance_min": safe_min(result["min_edit_distance"]),
        "min_edit_distance_max": safe_max(result["min_edit_distance"]),
        "min_edit_distance_counts": edit_counts,

        "nearest_neighbor_similarity_mean": safe_mean(
            result["nearest_neighbor_similarity"]
        ),
        "nearest_neighbor_similarity_median": safe_median(
            result["nearest_neighbor_similarity"]
        ),

        "sequence_identity_percent_mean": safe_mean(
            result["sequence_identity_percent"]
        ),
        "sequence_identity_percent_median": safe_median(
            result["sequence_identity_percent"]
        ),

        "pct_sequence_identity_ge_100": 100.0 * float(
            (result["sequence_identity_percent"] >= 100.0).sum()
        ) / n,
        "pct_sequence_identity_ge_90": 100.0 * float(
            (result["sequence_identity_percent"] >= 90.0).sum()
        ) / n,
        "pct_sequence_identity_ge_80": 100.0 * float(
            (result["sequence_identity_percent"] >= 80.0).sum()
        ) / n,

        "pct_min_edit_distance_ge_1": 100.0 * float(
            (result["min_edit_distance"] >= 1).sum()
        ) / n,
        "pct_min_edit_distance_ge_2": 100.0 * float(
            (result["min_edit_distance"] >= 2).sum()
        ) / n,
        "pct_min_edit_distance_ge_3": 100.0 * float(
            (result["min_edit_distance"] >= 3).sum()
        ) / n,

        "novelty_score_mean": safe_mean(result["novelty_score"]),
        "novelty_score_median": safe_median(result["novelty_score"]),
    }

    return summary


def write_summary_csv(summary: dict, path: Path):
    flat = {
        "n_generated_unique_evaluated":
            summary["n_generated_unique_evaluated"],
        "n_training_unique":
            summary["n_training_unique"],
        "exact_overlap_count":
            summary["exact_overlap_count"],
        "exact_overlap_percent":
            summary["exact_overlap_percent"],
        "non_exact_novel_count":
            summary["non_exact_novel_count"],
        "non_exact_novel_percent":
            summary["non_exact_novel_percent"],
        "min_edit_distance_mean":
            summary["min_edit_distance_mean"],
        "min_edit_distance_median":
            summary["min_edit_distance_median"],
        "nearest_neighbor_similarity_mean":
            summary["nearest_neighbor_similarity_mean"],
        "nearest_neighbor_similarity_median":
            summary["nearest_neighbor_similarity_median"],
        "sequence_identity_percent_mean":
            summary["sequence_identity_percent_mean"],
        "sequence_identity_percent_median":
            summary["sequence_identity_percent_median"],
        "pct_sequence_identity_ge_100":
            summary["pct_sequence_identity_ge_100"],
        "pct_sequence_identity_ge_90":
            summary["pct_sequence_identity_ge_90"],
        "pct_sequence_identity_ge_80":
            summary["pct_sequence_identity_ge_80"],
        "pct_min_edit_distance_ge_1":
            summary["pct_min_edit_distance_ge_1"],
        "pct_min_edit_distance_ge_2":
            summary["pct_min_edit_distance_ge_2"],
        "pct_min_edit_distance_ge_3":
            summary["pct_min_edit_distance_ge_3"],
        "novelty_score_mean":
            summary["novelty_score_mean"],
        "novelty_score_median":
            summary["novelty_score_median"],
    }

    for key, value in summary["min_edit_distance_counts"].items():
        flat[f"min_edit_distance_count_{key}"] = value

    pd.DataFrame([flat]).to_csv(path, index=False)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Quantify generated-peptide novelty relative to a training set "
            "using exact overlap, minimum edit distance, nearest-neighbor "
            "similarity, and sequence identity."
        )
    )

    ap.add_argument(
        "--generated",
        required=True,
        help=(
            "Generated peptide CSV/TSV/TXT/XLSX. For unified evaluator output, "
            "use generated_filtered_unique.csv."
        ),
    )

    ap.add_argument(
        "--training",
        required=True,
        help="Training/reference peptide CSV/TSV/TXT/XLSX.",
    )

    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: <generated parent>/novelty_vs_training"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Generated peptides per pairwise-distance block.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    generated_path = Path(args.generated).expanduser().resolve()
    training_path = Path(args.training).expanduser().resolve()

    if not generated_path.exists():
        raise FileNotFoundError(generated_path)
    if not training_path.exists():
        raise FileNotFoundError(training_path)

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (
            generated_path.parent / "novelty_vs_training"
        ).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    generated, generated_info = load_unique_peptides(generated_path)
    training, training_info = load_unique_peptides(training_path)

    print("=" * 78)
    print("PEPTIDE NOVELTY ANALYSIS")
    print("=" * 78)
    print(f"Generated: {generated_path}")
    print(
        f"  raw rows={generated_info['rows_raw']:,}, "
        f"valid 9/10mers={generated_info['rows_valid_9_10mer']:,}, "
        f"unique={generated_info['unique_valid_9_10mer']:,}"
    )
    print(f"Training:  {training_path}")
    print(
        f"  raw rows={training_info['rows_raw']:,}, "
        f"valid 9/10mers={training_info['rows_valid_9_10mer']:,}, "
        f"unique={training_info['unique_valid_9_10mer']:,}"
    )
    print(f"Batch size: {args.batch_size:,}")
    print("=" * 78)

    result = nearest_training_neighbors(
        generated,
        training,
        batch_size=args.batch_size,
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

    # Edit-distance histogram/table for direct reporting.
    edit_dist = (
        result["min_edit_distance"]
        .value_counts()
        .sort_index()
        .rename_axis("min_edit_distance")
        .reset_index(name="count")
    )
    edit_dist["percent"] = (
        100.0 * edit_dist["count"] / len(result)
    )
    edit_dist.to_csv(
        out_dir / "min_edit_distance_distribution.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"Exact training overlap: "
        f"{summary['exact_overlap_count']:,}/"
        f"{summary['n_generated_unique_evaluated']:,} "
        f"({summary['exact_overlap_percent']:.2f}%)"
    )
    print(
        f"Non-exact novel peptides: "
        f"{summary['non_exact_novel_count']:,}/"
        f"{summary['n_generated_unique_evaluated']:,} "
        f"({summary['non_exact_novel_percent']:.2f}%)"
    )
    print(
        f"Minimum edit distance: "
        f"mean={summary['min_edit_distance_mean']:.3f}, "
        f"median={summary['min_edit_distance_median']:.3f}"
    )
    print(
        f"Nearest-neighbor similarity: "
        f"mean={summary['nearest_neighbor_similarity_mean']:.4f}, "
        f"median={summary['nearest_neighbor_similarity_median']:.4f}"
    )
    print(
        f"Sequence identity to nearest neighbor: "
        f"mean={summary['sequence_identity_percent_mean']:.2f}%, "
        f"median={summary['sequence_identity_percent_median']:.2f}%"
    )
    print(
        f"Identity >=90%: "
        f"{summary['pct_sequence_identity_ge_90']:.2f}%"
    )
    print(
        f"Minimum edit distance >=2: "
        f"{summary['pct_min_edit_distance_ge_2']:.2f}%"
    )
    print()
    print(f"Per-peptide results: {per_peptide_path}")
    print(f"Summary JSON:        {summary_json}")
    print(f"Summary CSV:         {summary_csv}")
    print("=" * 78)


if __name__ == "__main__":
    main()
