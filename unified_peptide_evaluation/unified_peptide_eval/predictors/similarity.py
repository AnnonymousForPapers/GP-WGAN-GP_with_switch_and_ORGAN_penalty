from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BasePredictor, PredictorContext
from ..utils import find_peptide_column, read_table, natural_peptide_or_none


class BladderSimilarityPredictor(BasePredictor):
    name = "bladder_similarity"
    version = "SequenceMatcher-compatible"

    def __init__(self, reference_csv, top_k=5):
        self.reference_csv = str(reference_csv)
        self.top_k = int(top_k)

    def predict(self, peptides, context: PredictorContext):
        ref_df = read_table(self.reference_csv)
        col = find_peptide_column(ref_df)
        refs = [natural_peptide_or_none(x) for x in ref_df[col]]
        refs = [x for x in refs if x]
        if not refs:
            raise RuntimeError(f"No valid reference peptides in {self.reference_csv}")

        out_dir = context.output_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, max_scores = [], []
        top5_path = out_dir / "similarity_top5.txt"
        with top5_path.open("w", encoding="utf-8") as fh:
            for i, p in enumerate(peptides, 1):
                pairs = [(r, SequenceMatcher(None, p, r).ratio()) for r in refs]
                pairs.sort(key=lambda x: x[1], reverse=True)
                scores = [x[1] for x in pairs]
                mean_score = float(np.mean(scores))
                max_score = float(scores[0])
                max_scores.append(max_score)
                row = {
                    "peptide": p,
                    "bladder_similarity_mean": mean_score,
                    "bladder_similarity_max": max_score,
                    "bladder_exact_match": bool(max_score == 1.0),
                }
                for k, (rp, sc) in enumerate(pairs[:self.top_k], 1):
                    row[f"similarity_top{k}_peptide"] = rp
                    row[f"similarity_top{k}_score"] = sc
                rows.append(row)
                fh.write(f"{p}:{pairs[:self.top_k]}\n")
                if i % 100 == 0:
                    print(f"Similarity: {i}/{len(peptides)}", flush=True)
        np.save(out_dir / "MaxSimilarity.npy", np.asarray(max_scores, dtype=float))
        out = pd.DataFrame(rows)
        out.to_csv(out_dir / "similarity.csv", index=False)
        return out

    def manifest(self):
        x = super().manifest(); x.update({"reference_csv": self.reference_csv, "top_k": self.top_k}); return x
