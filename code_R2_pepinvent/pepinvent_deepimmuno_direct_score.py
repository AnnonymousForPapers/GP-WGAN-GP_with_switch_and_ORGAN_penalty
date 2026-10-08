#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem

from pepinvent_chuckles import (
    decode_natural20_linear_smiles,
    validate_natural20_sequence,
)

AA = set("ARNDCQEGHILKMFPSTWYV")
TF_PYTHON = os.environ.get("DEEPIMMUNO_PYTHON")

if not TF_PYTHON:
    raise RuntimeError(
        "DEEPIMMUNO_PYTHON is not set. "
        "Set it to the Python executable of the TensorFlow/DeepImmuno environment."
    )


def log(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} [DeepImmuno] {message}", file=sys.stderr, flush=True)


def smiles_to_natural_peptide(smiles):
    """
    Convert generated PepINVENT output to a Natural20 9/10-mer.

    First use exact deterministic decoding from the same fixed Natural20
    CHUCKLES fragment table used by constrained generation.  This is the
    preferred path for this experiment.

    PepFun is kept only as a fallback for molecules that do not match the
    exact linear Natural20 grammar.
    """

    # Preferred path: exact Natural20 fragment decoding.
    seq = decode_natural20_linear_smiles(smiles)
    seq = validate_natural20_sequence(seq)
    if seq is not None:
        return seq

    # Fallback: generic PepFun conversion.
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


def score_sequences_with_tf(peptides, data_root, helper_script):
    if not peptides:
        return []

    log(
        f"Launching TensorFlow scorer for {len(peptides)} valid peptide(s): "
        f"{TF_PYTHON} {helper_script}"
    )
    t0 = time.perf_counter()

    proc = subprocess.run(
        [TF_PYTHON, str(helper_script), str(data_root)],
        input=json.dumps({"peptides": peptides}),
        text=True,
        capture_output=True,
    )

    elapsed = time.perf_counter() - t0

    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            log(f"[tf] {line}")

    if proc.returncode != 0:
        log(
            f"TensorFlow scorer FAILED after {elapsed:.3f}s "
            f"(return code {proc.returncode})"
        )
        if proc.stdout:
            log(f"TensorFlow stdout (first 1000 chars): {proc.stdout[:1000]}")
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr
        )

    log(f"TensorFlow scorer returned successfully in {elapsed:.3f}s")

    result = json.loads(proc.stdout)
    scores = [float(x) for x in result["scores"]]

    if scores:
        log(
            f"DeepImmuno valid-only scores: mean={np.mean(scores):.6f}, "
            f"min={np.min(scores):.6f}, max={np.max(scores):.6f}"
        )
    return scores


def read_smiles_from_stdin():
    return [x.strip() for x in sys.stdin.read().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    helper = here / "deepimmuno_tf_score_sequences.py"

    batch_start = time.perf_counter()
    log("START external scoring batch")

    smiles = read_smiles_from_stdin()
    log(f"Received {len(smiles)} generated SMILES from REINVENT")

    t0 = time.perf_counter()
    sequences = []
    exceptions = 0
    direct_decoded = 0

    for s in smiles:
        try:
            direct = decode_natural20_linear_smiles(s)
            direct = validate_natural20_sequence(direct)

            if direct is not None:
                sequences.append(direct)
                direct_decoded += 1
            else:
                sequences.append(smiles_to_natural_peptide(s))
        except Exception as e:
            exceptions += 1
            sequences.append(None)
            log(f"SMILES conversion exception: {type(e).__name__}: {e}")

    conversion_time = time.perf_counter() - t0
    valid_idx = [i for i, p in enumerate(sequences) if p is not None]

    log(
        f"SMILES->peptide finished: valid={len(valid_idx)}/{len(smiles)}, "
        f"direct_natural20={direct_decoded}, "
        f"invalid={len(smiles)-len(valid_idx)}, exceptions={exceptions}, "
        f"time={conversion_time:.3f}s"
    )

    if valid_idx:
        log("First valid peptides: " + ", ".join(sequences[i] for i in valid_idx[:5]))
    else:
        log("No valid natural 9/10-mer peptides in this batch")

    scores = np.zeros(len(smiles), dtype=np.float32)

    if valid_idx:
        valid_peptides = [sequences[i] for i in valid_idx]
        valid_scores = score_sequences_with_tf(
            valid_peptides, args.data_root, helper
        )
        for i, score in zip(valid_idx, valid_scores):
            scores[i] = float(score)

    total_time = time.perf_counter() - batch_start
    log(
        f"END external scoring batch: returned={len(scores)}, "
        f"mean_all={np.mean(scores) if len(scores) else 0.0:.6f}, "
        f"max_all={np.max(scores) if len(scores) else 0.0:.6f}, "
        f"total_time={total_time:.3f}s"
    )

    # stdout MUST remain pure JSON for REINVENT ExternalProcess.
    print(json.dumps({
        "version": 4,
        "payload": {"predictions": [float(x) for x in scores]}
    }))


if __name__ == "__main__":
    main()
