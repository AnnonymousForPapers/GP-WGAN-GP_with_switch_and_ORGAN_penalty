from __future__ import annotations

import io
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BasePredictor, PredictorContext
from .iedb_nextgen import DEFAULT_BASE_URL, IEDBNextGenClient, find_table


def _pick(df, aliases):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


class IEDBImmunogenicityPredictor(BasePredictor):
    """IEDB Class-I pMHC immunogenicity predictor.

    Default backend: IEDB Next-Generation API (tool_group="mhci").

    The API receives the already-filtered peptide sequences *as-is* by using
    peptide_length_range=None, HLA-A*02:01 by default, and an immunogenicity
    predictor block such as::

        {"type": "immunogenicity", "mask_choice": "default"}

    The legacy standalone predictor is retained only as an optional fallback
    for offline/air-gapped runs.
    """

    name = "iedb_immunogenicity"
    version = "IEDB-NG"

    def __init__(
        self,
        backend="api",
        mask_choice="default",
        position_to_mask=None,
        chunk_size=500,
        timeout=900,
        poll_seconds=2.0,
        api_base_url=DEFAULT_BASE_URL,
        script=None,
        python_exe="python2",
        command_template=None,
    ):
        backend = str(backend).lower()
        if backend not in {"api", "local", "auto"}:
            raise ValueError("IEDB immunogenicity backend must be api/local/auto")
        if mask_choice not in {"default", "custom", "by_allele"}:
            raise ValueError("mask_choice must be default/custom/by_allele")
        if mask_choice == "custom" and not position_to_mask:
            raise ValueError("custom mask_choice requires position_to_mask, e.g. '2,5,9'")

        self.backend = backend
        self.mask_choice = mask_choice
        self.position_to_mask = position_to_mask
        self.chunk_size = int(chunk_size)
        self.api_base_url = str(api_base_url).rstrip("/")
        self.client = IEDBNextGenClient(
            base_url=self.api_base_url,
            timeout=timeout,
            poll_seconds=poll_seconds,
        )

        # Optional legacy local fallback.
        self.script = script
        self.python_exe = python_exe
        self.command_template = command_template

    def _api_predict(self, peptides, context: PredictorContext):
        peptides = list(dict.fromkeys(str(p).strip().upper() for p in peptides))
        out_dir = context.output_dir / self.name
        raw_dir = out_dir / "raw_api"
        out_dir.mkdir(parents=True, exist_ok=True)

        predictor_spec = {
            "type": "immunogenicity",
            "mask_choice": self.mask_choice,
        }
        if self.mask_choice == "custom":
            predictor_spec["position_to_mask"] = str(self.position_to_mask)

        frames = []
        for start in range(0, len(peptides), self.chunk_size):
            chunk = peptides[start:start + self.chunk_size]
            result = self.client.run(
                "mhci",
                "\n".join(chunk),
                {
                    "alleles": context.hla,
                    # Peptides are already 9/10-mers. None means use each input
                    # peptide as-is rather than tiling it into new peptides.
                    "peptide_length_range": None,
                    "predictors": [predictor_spec],
                },
                raw_dir=raw_dir,
                label=f"chunk{start // self.chunk_size:04d}",
            )
            df = find_table(
                result,
                required_any=["peptide", "immunogenicity_score", "immunogenicity score"],
            )
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raw.to_csv(out_dir / "IEDB_immunogenicity_raw_combined.csv", index=False)

        pcol = _pick(raw, [
            "peptide", "input_sequence", "input sequence", "sequence", "query_sequence"
        ])
        scol = _pick(raw, [
            "immunogenicity_score", "immunogenicity score", "immunogenicity",
            "pmhc_immunogenicity_score", "score"
        ])
        acol = _pick(raw, ["allele", "mhc_allele", "hla", "mhc"])

        # Be defensive to minor API display-name changes: prefer any column that
        # contains both "immun" and "score" if the canonical alias was absent.
        if scol is None:
            for c in raw.columns:
                lc = str(c).lower()
                if "immun" in lc and "score" in lc:
                    scol = c
                    break

        if pcol is None or scol is None:
            raise RuntimeError(
                "Could not identify IEDB immunogenicity peptide/score columns. "
                f"Columns={list(raw.columns)}"
            )

        norm = pd.DataFrame({
            "peptide": raw[pcol].astype(str).str.strip().str.upper(),
            "iedb_immunogenicity_score": pd.to_numeric(raw[scol], errors="coerce"),
        })
        if acol is not None:
            norm["iedb_immunogenicity_allele"] = raw[acol].astype(str)
        else:
            norm["iedb_immunogenicity_allele"] = context.hla

        norm = norm.drop_duplicates("peptide", keep="first")
        out = pd.DataFrame({"peptide": peptides}).merge(norm, on="peptide", how="left")
        out.to_csv(out_dir / "IEDB_immunogenicity_prediction.csv", index=False)
        return out

    # ------------------------- optional legacy local backend -------------------------
    def _parse_local_scores(self, text, peptides):
        for sep in ["\t", ",", r"\s+"]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            except Exception:
                continue
            if df.empty:
                continue
            pcol = next((c for c in df.columns if "peptide" in str(c).lower() or "sequence" in str(c).lower()), None)
            scol = next((c for c in df.columns if "score" in str(c).lower() or "immun" in str(c).lower()), None)
            if pcol is not None and scol is not None:
                out = pd.DataFrame({
                    "peptide": df[pcol].astype(str).str.strip().str.upper(),
                    "iedb_immunogenicity_score": pd.to_numeric(df[scol], errors="coerce"),
                }).dropna(subset=["iedb_immunogenicity_score"])
                if len(out):
                    return out.drop_duplicates("peptide")

        rows = []
        pep_set = set(peptides)
        for line in text.splitlines():
            toks = line.strip().replace(",", " ").split()
            pep = next((t.upper() for t in toks if t.upper() in pep_set), None)
            if pep is None:
                continue
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except Exception:
                    pass
            if nums:
                rows.append((pep, nums[-1]))
        if rows:
            return pd.DataFrame(rows, columns=["peptide", "iedb_immunogenicity_score"]).drop_duplicates("peptide")
        raise RuntimeError("Could not parse standalone IEDB immunogenicity output")

    @staticmethod
    def _run_local_command(cmd, cwd, stdout_path, stderr_path):
        p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        stdout_path.write_text(p.stdout or "", encoding="utf-8")
        stderr_path.write_text(p.stderr or "", encoding="utf-8")
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
        return p.stdout

    def _local_predict(self, peptides, context: PredictorContext):
        out_dir = context.output_dir / self.name / "local_fallback"
        out_dir.mkdir(parents=True, exist_ok=True)
        input_txt = out_dir / "peptides.txt"
        output_txt = out_dir / "tool_output.txt"
        input_txt.write_text("\n".join(peptides) + "\n", encoding="utf-8")

        if self.command_template:
            script = str(Path(self.script).resolve()) if self.script else ""
            rendered = self.command_template.format(input=input_txt, output=output_txt, script=script)
            cmd = shlex.split(rendered)
            stdout = self._run_local_command(cmd, out_dir, out_dir / "stdout.txt", out_dir / "stderr.txt")
            text = output_txt.read_text(encoding="utf-8", errors="replace") if output_txt.exists() else stdout
            out = self._parse_local_scores(text, peptides)
        else:
            if not self.script:
                raise FileNotFoundError(
                    "Local IEDB immunogenicity fallback requires --iedb-immunogenicity-script "
                    "or --iedb-immunogenicity-command"
                )
            script = Path(self.script).expanduser().resolve()
            if not script.exists():
                raise FileNotFoundError(script)
            attempts = [
                [self.python_exe, str(script), str(input_txt)],
                [self.python_exe, str(script), "-f", str(input_txt)],
                [self.python_exe, str(script), "--file", str(input_txt)],
                [self.python_exe, str(script), "-i", str(input_txt)],
            ]
            errors = []
            out = None
            for ai, cmd in enumerate(attempts, 1):
                try:
                    stdout = self._run_local_command(
                        cmd, script.parent,
                        out_dir / f"stdout_attempt{ai}.txt",
                        out_dir / f"stderr_attempt{ai}.txt",
                    )
                    out = self._parse_local_scores(stdout, peptides)
                    break
                except Exception as e:
                    errors.append(f"{cmd}: {type(e).__name__}: {e}")
            if out is None:
                raise RuntimeError("Could not invoke local IEDB immunogenicity predictor:\n" + "\n".join(errors))

        out["iedb_immunogenicity_allele"] = context.hla
        out.to_csv(context.output_dir / self.name / "IEDB_immunogenicity_prediction.csv", index=False)
        return out

    def predict(self, peptides, context: PredictorContext):
        peptides = list(dict.fromkeys(str(p).strip().upper() for p in peptides))
        if self.backend == "api":
            return self._api_predict(peptides, context)
        if self.backend == "local":
            return self._local_predict(peptides, context)

        # auto: prefer the current API, but preserve an offline fallback if configured.
        try:
            return self._api_predict(peptides, context)
        except Exception:
            if self.script or self.command_template:
                return self._local_predict(peptides, context)
            raise

    def manifest(self):
        x = super().manifest()
        x.update({
            "backend": self.backend,
            "api_base_url": self.api_base_url,
            "tool_group": "mhci",
            "hla": "from PredictorContext",
            "peptide_length_range": None,
            "predictor_type": "immunogenicity",
            "mask_choice": self.mask_choice,
            "position_to_mask": self.position_to_mask,
            "chunk_size": self.chunk_size,
            "local_script": self.script,
        })
        return x
