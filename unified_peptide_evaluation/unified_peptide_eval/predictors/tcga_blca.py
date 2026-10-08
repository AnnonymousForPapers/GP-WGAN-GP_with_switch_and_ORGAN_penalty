from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .base import BasePredictor, PredictorContext


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_peptide(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "")
    return s


def _is_valid_peptide(s):
    return bool(s) and all(a in STANDARD_AA for a in s)


def _join_unique(series):
    vals = sorted({str(x).strip() for x in series if pd.notna(x) and str(x).strip()})
    return ";".join(vals)


def _split_semicolon_values(series):
    vals = set()
    for x in series:
        if pd.isna(x):
            continue
        for item in str(x).split(";"):
            item = item.strip()
            if item:
                vals.add(item)
    return sorted(vals)


class TCGABLCAExactMatchPredictor(BasePredictor):
    """Exact-match generated 9/10-mers against TCGA-BLCA mutant peptides.

    The detailed output preserves every matching TCGA reference row.  The
    DataFrame returned to the unified pipeline contains exactly one row per
    generated peptide so it can be merged into all_predictions.csv.
    """

    name = "tcga_blca"
    version = "exact-mutant-peptide-match-v1"

    def __init__(self, tcga_csv, keep_lengths=(9, 10)):
        self.tcga_csv = str(tcga_csv)
        self.keep_lengths = tuple(int(x) for x in keep_lengths)

    def _load_reference(self):
        path = Path(self.tcga_csv).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"TCGA-BLCA reference CSV not found: {path}. "
                "Pass --tcga-csv /path/to/TCGA_BLCA_WT_mutant_peptides_unique.csv"
            )

        df = pd.read_csv(path, dtype=str, low_memory=False)
        required = ["wt_peptide", "mutant_peptide", "peptide_length"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"TCGA file is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df["wt_peptide"] = df["wt_peptide"].map(_clean_peptide)
        df["mutant_peptide"] = df["mutant_peptide"].map(_clean_peptide)
        df["tcga_peptide_length"] = df["mutant_peptide"].str.len()
        df = df[df["mutant_peptide"].map(_is_valid_peptide)].copy()
        if self.keep_lengths:
            df = df[df["tcga_peptide_length"].isin(self.keep_lengths)].copy()
        return path, df

    def predict(self, peptides, context: PredictorContext):
        tcga_path, tcga_df = self._load_reference()
        if tcga_df.empty:
            raise RuntimeError(f"No valid TCGA-BLCA mutant {self.keep_lengths}-mer peptides in {tcga_path}")

        # The unified pipeline already supplies unique valid 9/10-mers.  Clean
        # once more here so this module also behaves safely when called alone.
        gen = pd.DataFrame({"peptide": list(peptides)})
        gen["peptide"] = gen["peptide"].map(_clean_peptide)
        gen["generated_peptide_length"] = gen["peptide"].str.len()
        gen = gen[
            gen["peptide"].map(_is_valid_peptide)
            & gen["generated_peptide_length"].isin(self.keep_lengths)
        ].drop_duplicates("peptide", keep="first").reset_index(drop=True)

        matches = gen.merge(
            tcga_df,
            how="inner",
            left_on="peptide",
            right_on="mutant_peptide",
            suffixes=("_generated", "_tcga"),
        )

        out_dir = context.output_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the detailed exact-match table in the same spirit as the
        # user's standalone script.
        preferred = [
            "peptide", "generated_peptide_length", "wt_peptide", "mutant_peptide",
            "peptide_length", "mutation_position_in_peptide", "genes", "mutations",
            "n_patients", "n_samples", "patient_ids", "sample_ids", "protein_accessions",
        ]
        if not matches.empty:
            front = [c for c in preferred if c in matches.columns]
            rest = [c for c in matches.columns if c not in front]
            detailed_matches = matches[front + rest].copy()
        else:
            detailed_matches = matches.copy()

        matched_set = set(matches["peptide"]) if not matches.empty else set()
        all_annotated = gen.copy()
        all_annotated["exact_TCGA_BLCA_match"] = all_annotated["peptide"].isin(matched_set)
        unmatched = all_annotated[~all_annotated["exact_TCGA_BLCA_match"]].copy()

        detailed_matches.to_csv(out_dir / "generated_vs_TCGA_BLCA_exact_matches.csv", index=False)
        all_annotated.to_csv(out_dir / "generated_vs_TCGA_BLCA_all.csv", index=False)
        unmatched.to_csv(out_dir / "generated_vs_TCGA_BLCA_unmatched.csv", index=False)

        # Build one compact row per generated peptide for all_predictions.csv.
        compact_rows = []
        for p in gen["peptide"]:
            sub = matches[matches["peptide"] == p] if not matches.empty else matches
            row = {
                "peptide": p,
                "tcga_blca_exact_match": bool(len(sub)),
                "tcga_blca_match_rows": int(len(sub)),
            }
            if len(sub):
                field_map = {
                    "wt_peptide": "tcga_blca_wt_peptides",
                    "mutant_peptide": "tcga_blca_mutant_peptides",
                    "genes": "tcga_blca_genes",
                    "mutations": "tcga_blca_mutations",
                    "mutation_position_in_peptide": "tcga_blca_mutation_positions",
                    "patient_ids": "tcga_blca_patient_ids",
                    "sample_ids": "tcga_blca_sample_ids",
                    "protein_accessions": "tcga_blca_protein_accessions",
                }
                for src, dst in field_map.items():
                    if src in sub.columns:
                        row[dst] = _join_unique(sub[src])

                if "patient_ids" in sub.columns:
                    row["tcga_blca_unique_patients"] = len(_split_semicolon_values(sub["patient_ids"]))
                elif "n_patients" in sub.columns:
                    nums = pd.to_numeric(sub["n_patients"], errors="coerce").dropna()
                    row["tcga_blca_reported_n_patients_max"] = int(nums.max()) if len(nums) else None

                if "sample_ids" in sub.columns:
                    row["tcga_blca_unique_samples"] = len(_split_semicolon_values(sub["sample_ids"]))
                elif "n_samples" in sub.columns:
                    nums = pd.to_numeric(sub["n_samples"], errors="coerce").dropna()
                    row["tcga_blca_reported_n_samples_max"] = int(nums.max()) if len(nums) else None
            compact_rows.append(row)

        compact = pd.DataFrame(compact_rows)
        compact.to_csv(out_dir / "TCGA_BLCA_prediction.csv", index=False)

        total_unique = int(gen["peptide"].nunique())
        matched_unique = int(len(matched_set))
        unique_9 = int(gen.loc[gen["generated_peptide_length"] == 9, "peptide"].nunique())
        unique_10 = int(gen.loc[gen["generated_peptide_length"] == 10, "peptide"].nunique())
        matched_9 = int(compact.loc[
            compact["tcga_blca_exact_match"] & compact["peptide"].str.len().eq(9), "peptide"
        ].nunique()) if len(compact) else 0
        matched_10 = int(compact.loc[
            compact["tcga_blca_exact_match"] & compact["peptide"].str.len().eq(10), "peptide"
        ].nunique()) if len(compact) else 0
        exact_rate = 100.0 * matched_unique / total_unique if total_unique else 0.0

        summary = f"""Generated peptide vs TCGA-BLCA exact-match summary

TCGA-BLCA reference file:
{tcga_path}

Unique generated peptides evaluated:
{total_unique:,}

Unique generated 9-mers:
{unique_9:,}

Unique generated 10-mers:
{unique_10:,}

TCGA reference rows used:
{len(tcga_df):,}

Unique TCGA mutant peptides:
{tcga_df['mutant_peptide'].nunique():,}

Exact-match result rows:
{len(matches):,}

Unique generated peptides with at least one exact TCGA-BLCA mutant-peptide match:
{matched_unique:,}

Unique matched generated 9-mers:
{matched_9:,}

Unique matched generated 10-mers:
{matched_10:,}

Exact-match percentage among unique generated peptides:
{exact_rate:.6f}%

Interpretation:
An exact match means that the generated peptide sequence is identical to a
mutation-containing peptide reconstructed from a somatic missense mutation
observed in the TCGA-BLCA cohort represented by the reference file.

It does NOT by itself prove natural processing, HLA presentation,
immunogenicity, tumor specificity, or experimental validation.
"""
        (out_dir / "generated_vs_TCGA_BLCA_summary.txt").write_text(summary, encoding="utf-8")
        print(summary, flush=True)
        return compact

    def manifest(self):
        x = super().manifest()
        x.update({"tcga_csv": self.tcga_csv, "keep_lengths": list(self.keep_lengths), "match": "exact"})
        return x
