from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class PredictorContext:
    output_dir: Path
    data_root: Path
    hla: str = "HLA-A*02:01"
    seed: int = 0


class BasePredictor:
    name = "base"
    version = None

    def predict(self, peptides, context: PredictorContext) -> pd.DataFrame:
        raise NotImplementedError

    def manifest(self):
        return {"name": self.name, "version": self.version, "class": type(self).__name__}


class PredictorUnavailable(RuntimeError):
    """Raised when a requested predictor is not available in the current backend/environment."""
    pass
