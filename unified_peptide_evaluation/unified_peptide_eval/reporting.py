from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def netmhcpan_counts(df, version):
    col = f"netmhcpan_{version}_Aff_nM"
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return {
        "n_scored": int(len(x)),
        "ic50_lt_150": int((x < 150).sum()),
        "ic50_150_to_lt_500": int(((x >= 150) & (x < 500)).sum()),
        "ic50_ge_500": int((x >= 500).sum()),
        "ic50_lt_500": int((x < 500).sum()),
        "mean_ic50_nM": float(x.mean()) if len(x) else None,
        "median_ic50_nM": float(x.median()) if len(x) else None,
    }


def save_ic50_plot(df, version, output_path):
    col = f"netmhcpan_{version}_Aff_nM"
    if col not in df.columns:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    vals = [int((x < 150).sum()), int(((x >= 150) & (x < 500)).sum()), int((x >= 500).sum())]
    labels = ["IC50 < 150", "150 <= IC50 < 500", "IC50 >= 500"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(3), vals)
    ax.bar_label(bars, padding=3)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Number of peptides")
    ax.set_title(f"NetMHCpan {version} BA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(raw_df, filtered_all, unique_df, merged, architecture, checkpoint, seed, predictor_status):
    summary = {
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "generated_raw": int(len(raw_df)),
        "valid_9_10mer_before_dedup": int(len(filtered_all)),
        "unique_valid_9_10mer": int(len(unique_df)),
        "predictor_status": predictor_status,
    }
    if "deepimmuno_score" in merged.columns:
        x = pd.to_numeric(merged["deepimmuno_score"], errors="coerce").dropna()
        summary["deepimmuno"] = {
            "n_scored": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "median": float(x.median()) if len(x) else None,
            "min": float(x.min()) if len(x) else None,
            "max": float(x.max()) if len(x) else None,
        }
    if "iedb_immunogenicity_score" in merged.columns:
        x = pd.to_numeric(merged["iedb_immunogenicity_score"], errors="coerce").dropna()
        summary["iedb_immunogenicity"] = {
            "n_scored": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "median": float(x.median()) if len(x) else None,
            "score_gt_0": int((x > 0).sum()),
        }
    for v in ["4.0", "4.1"]:
        c = netmhcpan_counts(merged, v)
        if c:
            summary[f"netmhcpan_{v}"] = c
    if "pepmatch_mismatches" in merged.columns:
        x = pd.to_numeric(merged["pepmatch_mismatches"], errors="coerce")
        matched = x.dropna()
        summary["pepmatch"] = {
            "n_queries": int(len(x)),
            "n_matched_within_3": int(x.notna().sum()),
            "n_no_match_within_3": int(x.isna().sum()),
            "exact_match": int((x == 0).sum()),
            "best_match_1_mismatch": int((x == 1).sum()),
            "best_match_2_mismatches": int((x == 2).sum()),
            "best_match_3_mismatches": int((x == 3).sum()),
            "mean_best_mismatches_among_matched": float(matched.mean()) if len(matched) else None,
        }
    if "pepsysco_score" in merged.columns:
        x = pd.to_numeric(merged["pepsysco_score"], errors="coerce").dropna()
        summary["pepsysco"] = {
            "n_scored": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "median": float(x.median()) if len(x) else None,
            "min": float(x.min()) if len(x) else None,
            "max": float(x.max()) if len(x) else None,
        }

    if "tcga_blca_exact_match" in merged.columns:
        flag = merged["tcga_blca_exact_match"].fillna(False).astype(bool)
        lengths = merged["peptide"].astype(str).str.len()
        n = int(len(flag))
        m = int(flag.sum())
        summary["tcga_blca"] = {
            "n_queries": n,
            "exact_match_unique_peptides": m,
            "exact_match_rate_percent": float(100.0 * m / n) if n else 0.0,
            "matched_9mers": int((flag & lengths.eq(9)).sum()),
            "matched_10mers": int((flag & lengths.eq(10)).sum()),
        }
    if "bladder_similarity_max" in merged.columns:
        x = pd.to_numeric(merged["bladder_similarity_max"], errors="coerce").dropna()
        summary["bladder_similarity"] = {
            "n_scored": int(len(x)),
            "mean_max_similarity": float(x.mean()) if len(x) else None,
            "median_max_similarity": float(x.median()) if len(x) else None,
            "exact_matches": int((x == 1.0).sum()),
            "one_mismatch_or_closer_9mer_approx": int((x >= (8 / 9)).sum()),
        }
    return summary
