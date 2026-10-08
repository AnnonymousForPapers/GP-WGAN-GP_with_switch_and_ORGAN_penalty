import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pepinvent_chuckles import (
    decode_natural20_linear_smiles,
    validate_natural20_sequence,
)

AA = set("ARNDCQEGHILKMFPSTWYV")
AA_ORDER = "ARNDCQEGHILKMFPSTWYV-"
HLA_NAME = "HLA-A*0201"


TF_PYTHON = os.environ.get("DEEPIMMUNO_PYTHON")

if not TF_PYTHON:
    raise RuntimeError(
        "DEEPIMMUNO_PYTHON is not set. "
        "Set it to the Python executable of the TensorFlow/DeepImmuno environment."
    )


def score_with_deepimmuno(peptides, data_root):
    """Run only DeepImmuno/TensorFlow in the dedicated tf environment."""
    if not peptides:
        return np.asarray([], dtype=np.float32)

    helper_script = (
        Path(__file__).resolve().parent
        / "deepimmuno_tf_score_sequences.py"
    )

    proc = subprocess.run(
        [
            TF_PYTHON,
            str(helper_script),
            str(data_root),
        ],
        input=json.dumps({"peptides": peptides}),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "DeepImmuno TensorFlow subprocess failed.\n"
            f"Command: {proc.args}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    try:
        result = json.loads(proc.stdout)
        scores = result["scores"]
    except Exception as e:
        raise RuntimeError(
            "Could not parse DeepImmuno TensorFlow output.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        ) from e

    return np.asarray(scores, dtype=np.float32)


def smiles_to_natural_peptide(smiles):
    """
    Convert PepINVENT output to a Natural20 9/10-mer.

    Preferred path:
        deterministic decoding using the same Natural20 CHUCKLES
        fragments used during constrained generation.

    PepFun is only an optional fallback.
    """

    # Preferred deterministic Natural20 decoder
    seq = decode_natural20_linear_smiles(smiles)
    seq = validate_natural20_sequence(seq)

    if seq is not None:
        return seq

    # Optional PepFun fallback
    try:
        from pepfun.extra import readProperties, peptideFromSMILES
    except Exception:
        return None

    if Chem.MolFromSmiles(smiles) is None:
        return None

    try:
        props = readProperties()
        seq = peptideFromSMILES(smiles, props)
    except Exception:
        return None

    if hasattr(seq, "sequence"):
        seq = seq.sequence

    seq = (
        str(seq)
        .strip()
        .replace("-", "")
        .replace(" ", "")
        .upper()
    )

    return validate_natural20_sequence(seq)


def read_smiles_from_stdin():
    raw = sys.stdin.read()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def emit_predictions(scores):
    print(json.dumps({
        "version": 4,
        "payload": {"predictions": [float(x) for x in scores]}
    }))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + 0.3 * self.block(x)


class Scorer(nn.Module):
    def __init__(self, hidden=128, n_chars=21, seq_len=10):
        super().__init__()
        self.conv1 = nn.Conv1d(n_chars, hidden, 1)
        self.blocks = nn.Sequential(
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden),
            ResBlock(hidden), ResBlock(hidden)
        )
        self.fc = nn.Linear(seq_len * hidden, 1)
        self.hidden = hidden
        self.seq_len = seq_len

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.conv1(x)
        x = self.blocks(x)
        x = x.reshape(-1, self.hidden * self.seq_len)
        return torch.sigmoid(self.fc(x)).squeeze(-1)


def one_hot(peptides, device):
    eye = torch.eye(21, device=device)
    out = []
    for peptide in peptides:
        p = peptide
        if len(p) == 9:
            p = p[:5] + "-" + p[5:]
        ids = [AA_ORDER.index(a) for a in p]
        out.append(eye[torch.tensor(ids, device=device)])
    return torch.stack(out)


def load_real_pool(csv_path):
    if csv_path is None or not Path(csv_path).exists():
        return []
    df = pd.read_csv(csv_path)
    if "peptide" not in df.columns:
        return []
    pool = []
    for x in df["peptide"].dropna():
        x = str(x).strip().upper()
        if len(x) in (9, 10) and set(x).issubset(AA):
            pool.append(x)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../", help="Matches data_file = ../ in the provided WGAN/DeepImmuno code.")
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--real-peptide-csv", default="../data/neoepitopes/Bladder.4.0_test_mut.csv")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-steps", type=int, default=1)
    args = ap.parse_args()

    smiles = read_smiles_from_stdin()
    sequences = [smiles_to_natural_peptide(s) for s in smiles]
    valid_seq = [x for x in sequences if x is not None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Scorer(hidden=args.hidden).to(device)
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device))

    real_pool = load_real_pool(args.real_peptide_csv)
    n_real = min(len(valid_seq), len(real_pool))
    real_seq = random.sample(real_pool, n_real) if n_real else []
    train_seq = valid_seq + real_seq

    if train_seq:
        targets_np = score_with_deepimmuno(train_seq, args.data_root)
        x = one_hot(train_seq, device)
        y = torch.tensor(targets_np, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        for _ in range(max(1, args.train_steps)):
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), state_path)

    scores = np.zeros(len(smiles), dtype=np.float32)
    valid_idx = [i for i, seq in enumerate(sequences) if seq is not None]
    if valid_idx:
        model.eval()
        with torch.no_grad():
            pred = model(one_hot([sequences[i] for i in valid_idx], device))
        for i, s in zip(valid_idx, pred.detach().cpu().numpy()):
            scores[i] = float(s)

    emit_predictions(scores)


if __name__ == "__main__":
    main()
