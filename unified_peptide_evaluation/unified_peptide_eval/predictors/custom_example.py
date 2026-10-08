"""Template showing how to add another predictor without editing the generators."""
import pandas as pd
from .base import BasePredictor, PredictorContext


class ExamplePredictor(BasePredictor):
    name = "example_predictor"
    version = "1.0"

    def predict(self, peptides, context: PredictorContext):
        # Replace this with a local model, CLI tool, or web/API call.
        # Keep one row per peptide and return a DataFrame containing 'peptide'.
        return pd.DataFrame({
            "peptide": list(peptides),
            "example_score": [0.0] * len(peptides),
        })
