from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pandas as pd
from .base import BasePredictor, PredictorContext


class DeepImmunoPredictor(BasePredictor):
    name = "deepimmuno"
    version = "uploaded-compatible"

    def __init__(self, tf_python=None, batch_size=1024, weights_path=None):
        self.tf_python = tf_python or "python"
        self.batch_size = int(batch_size)
        self.weights_path = weights_path

    def predict(self, peptides, context: PredictorContext):
        out_dir = context.output_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        inp, out = out_dir / "input.csv", out_dir / "deepimmuno.csv"
        pd.DataFrame({"peptide": list(peptides)}).to_csv(inp, index=False)
        worker = Path(__file__).resolve().parents[1] / "deepimmuno_worker.py"
        python = self.tf_python if Path(self.tf_python).exists() else sys.executable
        cmd = [python, str(worker), "--input", str(inp), "--output", str(out),
               "--data-root", str(context.data_root), "--batch-size", str(self.batch_size)]
        if self.weights_path:
            cmd.extend(["--weights", str(self.weights_path)])
        subprocess.run(cmd, check=True)
        result = pd.read_csv(out)
        compat = result.rename(columns={"deepimmuno_score": "immunogenicity"})
        compat.to_csv(out_dir / "deepimmuno_scored.txt", sep="\t", index=False)
        return result

    def manifest(self):
        x = super().manifest(); x.update({"tf_python": self.tf_python, "batch_size": self.batch_size, "weights_path": self.weights_path}); return x
