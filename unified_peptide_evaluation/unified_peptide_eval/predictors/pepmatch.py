from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .base import BasePredictor, PredictorContext
from .iedb_nextgen import DEFAULT_BASE_URL, IEDBNextGenClient, find_table


def _first_column(df, aliases):
    lookup = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    for a in aliases:
        key = a.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


class PEPMatchPredictor(BasePredictor):
    name = "pepmatch"
    version = "IEDB-NG"

    def __init__(self, mismatch=3, proteome="Human", best_match=True,
                 chunk_size=500, timeout=900, poll_seconds=2.0,
                 api_base_url=DEFAULT_BASE_URL):
        mismatch = int(mismatch)
        if mismatch < 0 or mismatch > 5:
            raise ValueError("PEPMatch mismatch must be between 0 and 5")
        self.mismatch = mismatch
        self.proteome = proteome
        self.best_match = bool(best_match)
        self.chunk_size = int(chunk_size)
        self.api_base_url = api_base_url
        self.client = IEDBNextGenClient(
            base_url=api_base_url, timeout=timeout, poll_seconds=poll_seconds
        )

    def predict(self, peptides, context: PredictorContext):
        peptides = list(dict.fromkeys(str(p).strip().upper() for p in peptides))
        out_dir = context.output_dir / "pepmatch"
        raw_dir = out_dir / "raw_api"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for start in range(0, len(peptides), self.chunk_size):
            chunk = peptides[start:start + self.chunk_size]
            result = self.client.run(
                "pepmatch",
                "\n".join(chunk),
                {
                    "mismatch": self.mismatch,
                    "proteome": self.proteome,
                    "best_match": self.best_match,
                },
                raw_dir=raw_dir,
                label=f"chunk{start // self.chunk_size:04d}",
            )
            df = find_table(
                result,
                required_any=["input_sequence", "matched_sequence", "mismatches"],
            )
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raw.to_csv(out_dir / "PEPMatch_raw_combined.csv", index=False)

        # A completely empty PEPMatch result is valid: it means that none of
        # the submitted peptides had a Human-proteome match within the requested
        # mismatch threshold.  Preserve every query peptide and mark it as
        # unmatched instead of treating the empty result as a parser failure.
        if raw.empty:
            out = pd.DataFrame({"peptide": peptides})
            out["pepmatch_matched_sequence"] = np.nan
            out["pepmatch_protein_id"] = np.nan
            out["pepmatch_protein_name"] = np.nan
            out["pepmatch_gene"] = np.nan
            out["pepmatch_mismatches"] = np.nan
            out["pepmatch_mutated_positions"] = np.nan
            out["pepmatch_exact_match"] = False
            out["pepmatch_within_1_mismatch"] = False
            out["pepmatch_within_2_mismatches"] = False
            out["pepmatch_within_3_mismatches"] = False
            out[f"pepmatch_no_match_within_{self.mismatch}"] = True
            out.to_csv(out_dir / "PEPMatch_prediction.csv", index=False)
            return out

        qcol = _first_column(raw, [
            "input_sequence", "input sequence", "query_sequence", "query sequence",
            "peptide", "sequence",
        ])
        mcol = _first_column(raw, ["matched_sequence", "matched sequence", "match_sequence", "match sequence"])
        mmcol = _first_column(raw, ["mismatches", "mismatch", "num_mismatches", "number_of_mismatches"])
        pidcol = _first_column(raw, ["protein_id", "protein id", "protein_accession", "accession"])
        pnamecol = _first_column(raw, ["protein_name", "protein name"])
        genecol = _first_column(raw, ["gene", "gene_name", "gene name"])
        mutcol = _first_column(raw, ["mutated_positions", "mutated positions", "mismatch_positions", "mismatch positions"])

        if qcol is None:
            raise RuntimeError(f"Could not identify PEPMatch input-sequence column. Columns={list(raw.columns)}")

        norm = pd.DataFrame({"peptide": raw[qcol].astype(str).str.strip().str.upper()})
        norm["pepmatch_matched_sequence"] = raw[mcol] if mcol else np.nan
        norm["pepmatch_protein_id"] = raw[pidcol] if pidcol else np.nan
        norm["pepmatch_protein_name"] = raw[pnamecol] if pnamecol else np.nan
        norm["pepmatch_gene"] = raw[genecol] if genecol else np.nan
        norm["pepmatch_mismatches"] = pd.to_numeric(raw[mmcol], errors="coerce") if mmcol else np.nan
        norm["pepmatch_mutated_positions"] = raw[mutcol] if mutcol else np.nan

        # Best-match mode is used for the merged one-row-per-peptide table. If the
        # API ever returns more than one row, keep the lowest-mismatch row.
        norm = norm.sort_values("pepmatch_mismatches", na_position="last").drop_duplicates("peptide", keep="first")
        out = pd.DataFrame({"peptide": peptides}).merge(norm, on="peptide", how="left")
        mm = pd.to_numeric(out["pepmatch_mismatches"], errors="coerce")
        out["pepmatch_exact_match"] = mm.eq(0)
        out["pepmatch_within_1_mismatch"] = mm.le(1).fillna(False)
        out["pepmatch_within_2_mismatches"] = mm.le(2).fillna(False)
        out["pepmatch_within_3_mismatches"] = mm.le(3).fillna(False)
        out[f"pepmatch_no_match_within_{self.mismatch}"] = mm.isna()
        out.to_csv(out_dir / "PEPMatch_prediction.csv", index=False)
        return out

    def manifest(self):
        x = super().manifest()
        x.update({
            "backend": "IEDB Next-Generation API",
            "api_base_url": self.api_base_url,
            "mismatch": self.mismatch,
            "proteome": self.proteome,
            "best_match": self.best_match,
            "chunk_size": self.chunk_size,
        })
        return x
