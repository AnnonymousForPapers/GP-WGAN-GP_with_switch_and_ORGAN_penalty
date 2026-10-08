from __future__ import annotations

import contextlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

AA20 = set("ARNDCQEGHILKMFPSTWYV")
AA_ORDER = "ARNDCQEGHILKMFPSTWYV-"


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def natural_peptide_or_none(seq: object):
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return None
    s = str(seq).strip().upper()
    if not s:
        return None
    if s.count("-") >= 2:
        return None
    s = s.replace("-", "")
    if len(s) not in (9, 10):
        return None
    if not set(s).issubset(AA20):
        return None
    return s


def find_peptide_column(df: pd.DataFrame) -> str:
    candidates = [
        "peptide", "Peptide", "PEPTIDE", "sequence", "Sequence", "SEQUENCE",
        "epitope", "Epitope", "mut_peptide", "Mut_peptide", "MT_pep",
        "mutant_peptide",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    best_col, best_count = None, -1
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        count = sum(natural_peptide_or_none(x) is not None for x in vals)
        if count > best_count:
            best_col, best_count = c, count
    if best_col is None or best_count <= 0:
        raise RuntimeError(f"Could not identify a peptide column. Columns: {list(df.columns)}")
    return best_col


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def resolve_data_root(model_dir: Path, explicit: Optional[str] = None) -> Path:
    """Find a root containing DeepImmuno data and weights."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = []
    p = model_dir.resolve()
    candidates.extend([p] + list(p.parents))
    candidates.extend([
    ])
    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if (c / "data/DeepImmuno/after_pca.txt").exists() and (
            c / "weights/Immunogenicity_Predictor"
        ).exists():
            return c
    # Return the most plausible root; DeepImmuno stage will give a clear error if missing.
    for c in candidates:
        if (c / "data").exists():
            return c
    return model_dir.resolve()


def resolve_bladder_csv(data_root: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        data_root / "data/neoepitopes/Bladder.4.0_test_mut.csv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]



def resolve_tcga_csv(data_root: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    names = [
        "TCGA_BLCA_WT_mutant_peptides_unique.csv",
        "TCGA_BLCA_WT_mutant_peptides.csv",
    ]
    roots = [
        data_root,
        data_root / "data",
        data_root / "data/neoepitopes",
        data_root / "data/TCGA",
    ]
    for root in roots:
        for name in names:
            c = root / name
            if c.exists():
                return c.resolve()
    # Return the preferred path so a predictor failure names the expected file.
    return (data_root / "data/TCGA/TCGA_BLCA_WT_mutant_peptides_unique.csv").resolve()

def _checkpoint_epoch(path: Path) -> int:
    m = re.search(r"model_epoch_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def resolve_checkpoint(model_dir: str | Path, choice: str = "auto") -> Path:
    model_dir = Path(model_dir).expanduser().resolve()
    if Path(choice).expanduser().is_file():
        return Path(choice).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    choice = str(choice).lower()
    if choice not in {"auto", "best", "last"}:
        candidate = model_dir / choice
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    preferred = []
    if choice in {"auto", "best"}:
        preferred += [model_dir / "model_best.pth", model_dir / "model_best.chkpt"]
    if choice in {"auto", "last"}:
        preferred += [model_dir / "model_last.pth", model_dir / "model_last.chkpt"]
    for p in preferred:
        if p.exists():
            return p.resolve()

    epoch_files = list(model_dir.glob("model_epoch_*.pth")) + list(model_dir.glob("model_epoch_*.chkpt"))
    if epoch_files:
        return sorted(epoch_files, key=_checkpoint_epoch)[-1].resolve()
    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. Expected model_best/model_last or model_epoch_* files."
    )


def unwrap_state_dict(obj):
    """Extract a state_dict from common checkpoint wrappers and strip DataParallel prefixes."""
    if isinstance(obj, dict):
        for key in [
            "generator_state_dict", "model_state_dict", "state_dict", "generator", "model", "G"
        ]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a recognizable PyTorch state_dict.")
    out = {}
    for k, v in obj.items():
        k2 = k[7:] if k.startswith("module.") else k
        if k2.startswith("G."):
            k2 = k2[2:]
        out[k2] = v
    return out


class RunLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def print(self, *args, **kwargs):
        msg = " ".join(str(x) for x in args)
        print(msg, **kwargs)
        print(msg, file=self._fh, flush=True)

    @contextlib.contextmanager
    def stage(self, index: int, total: int, name: str):
        self.print("\n" + "=" * 78)
        self.print(f"[{index}/{total}] {name}")
        self.print("=" * 78)
        start = time.perf_counter()
        try:
            yield
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.print(f"[FAILED] {name} ({elapsed:.2f}s): {type(e).__name__}: {e}")
            raise
        else:
            elapsed = time.perf_counter() - start
            self.print(f"[DONE] {name} ({elapsed:.2f}s)")


class OptionalStageError(RuntimeError):
    pass


def merge_on_peptide(base: pd.DataFrame, addon: pd.DataFrame) -> pd.DataFrame:
    if addon is None or addon.empty:
        return base
    if "peptide" not in addon.columns:
        raise ValueError("Predictor result must contain a 'peptide' column.")
    addon = addon.drop_duplicates(subset="peptide", keep="first")
    return base.merge(addon, on="peptide", how="left", suffixes=("", "_dup"))
