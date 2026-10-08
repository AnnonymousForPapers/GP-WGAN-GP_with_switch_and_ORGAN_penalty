from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from ..utils import find_peptide_column, natural_peptide_or_none, read_table


def _runtime_env(module_dir: Path):
    env = dict(os.environ)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(module_dir) if not old else str(module_dir) + os.pathsep + old
    env["PEPINVENT_RUNTIME_OVERRIDES"] = "1"
    return env


def _find_smiles_column(df):
    for c in ["SMILES", "smiles", "Smiles", "output", "Output"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not identify PepINVENT SMILES column: {list(df.columns)}")


def _make_sampling_config(checkpoint, masks_file, output_csv, num_samples, device):
    return "\n".join([
        'run_type = "sampling"',
        '',
        f'device = "{device}"',
        '',
        '[parameters]',
        f'model_file = "{checkpoint}"',
        f'smiles_file = "{masks_file}"',
        'sample_strategy = "multinomial"',
        'temperature = 1.0',
        f'output_file = "{output_csv}"',
        f'num_smiles = {int(num_samples)}',
        'unique_molecules = false',
        'randomize_smiles = false',
        '',
    ])


def generate_pepinvent(
    checkpoint: str | Path,
    num_samples: int,
    seed: int,
    output_dir: str | Path,
    bladder_csv: str | Path,
    reinvent_exe: str = "reinvent",
    device: str = "cuda:0",
    mask_count: int = 3,
):
    """Generate PepINVENT samples using the same random-mask convention as the uploaded training workflow."""
    from pepinvent_chuckles import prepare_random_masks, load_bladder_peptides, decode_natural20_linear_smiles

    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    peptides, _ = load_bladder_peptides(bladder_csv)
    masks = output_dir / "pepinvent_eval_masks.smi"
    mask_map = output_dir / "pepinvent_eval_masks.csv"
    prepare_random_masks(peptides, num_samples, masks, mask_map, seed=seed, mask_count=mask_count)

    sample_csv = output_dir / "pepinvent_sampling_raw.csv"
    toml = output_dir / "pepinvent_sampling.toml"
    log = output_dir / "pepinvent_sampling.log"
    toml.write_text(_make_sampling_config(checkpoint, masks, sample_csv, num_samples, device), encoding="utf-8")

    module_dir = Path(__file__).resolve().parents[1]
    cmd = [reinvent_exe, "-s", str(seed), "-l", str(log), str(toml)]
    subprocess.run(cmd, check=True, env=_runtime_env(module_dir))

    df = pd.read_csv(sample_csv)
    sm_col = _find_smiles_column(df)
    smiles = df[sm_col].astype(str).tolist()
    decoded = [decode_natural20_linear_smiles(s) for s in smiles]
    decoded = [natural_peptide_or_none(s) for s in decoded]
    out = pd.DataFrame({
        "SMILES": smiles,
        "peptide": [x if x is not None else "" for x in decoded],
        "HLA": ["HLA-A*0201"] * len(smiles),
        "immunogenicity": [1] * len(smiles),
    })
    return out, {"architecture": "pepinvent", "num_samples_returned": len(out)}
