from __future__ import annotations

import io
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BasePredictor, PredictorContext

API_URL = "https://tools-cluster-interface.iedb.org/tools_api/mhci/"


def _api_hla(hla):
    h = hla.replace("HLA-A*0201", "HLA-A*02:01")
    if h.startswith("HLA-") and "*" in h and ":" not in h:
        left, digits = h.split("*", 1)
        if len(digits) == 4 and digits.isdigit():
            h = left + "*" + digits[:2] + ":" + digits[2:]
    return h


def _local_hla(hla):
    return _api_hla(hla).replace("*", "")


class NetMHCpanPredictor(BasePredictor):
    name = "netmhcpan"

    def __init__(self, version="4.1", backend="api", executable=None, chunk_size=200,
                 timeout=600, retries=3, api_url=API_URL):
        if version not in {"4.0", "4.1"}:
            raise ValueError("NetMHCpan version must be 4.0 or 4.1")
        self.version = version
        self.backend = backend
        self.executable = executable
        self.chunk_size = int(chunk_size)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.api_url = api_url

    @property
    def output_name(self):
        return f"netmhcpan_{self.version.replace('.', '_')}"

    def predict(self, peptides, context: PredictorContext):
        if self.backend == "local" or (self.backend == "auto" and self.executable):
            return self._predict_local(peptides, context)
        return self._predict_api(peptides, context)

    def _post(self, payload):
        data = urllib.parse.urlencode(payload).encode("utf-8")
        attempt = 0
        while True:
            attempt += 1
            try:
                req = urllib.request.Request(self.api_url, data=data, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(
                    f"IEDB NetMHCpan API request failed on attempt {attempt}: {e}. "
                    f"Retrying in 60s...",
                    flush=True,
                )
                time.sleep(60)

    def _predict_api(self, peptides, context):
        out_dir = context.output_dir / self.output_name
        raw_dir = out_dir / "raw_api"
        raw_dir.mkdir(parents=True, exist_ok=True)
        hla = _api_hla(context.hla)
        rows = []
        seqs = list(peptides)
        for length in (9, 10):
            indexed = [(i, p) for i, p in enumerate(seqs) if len(p) == length]
            for ci in range(0, len(indexed), self.chunk_size):
                chunk = indexed[ci:ci + self.chunk_size]
                pending = list(chunk)
                returned_indices = set()
                result_attempt = 0

                while pending:
                    result_attempt += 1
                    fasta = "\n".join(f">p{i}\n{p}" for i, p in pending) + "\n"

                    # Official API: netmhcpan-4.0/4.1 resolves to BA; netmhcpan_ba is its explicit alias.
                    response_text = self._post({
                        "method": f"netmhcpan-{self.version}",
                        "sequence_text": fasta,
                        "allele": hla,
                        "length": str(length),
                    })

                    retry_suffix = "" if result_attempt == 1 else f"_retry{result_attempt - 1}"
                    raw_path = raw_dir / (
                        f"length{length}_chunk{ci // self.chunk_size:04d}{retry_suffix}.tsv"
                    )
                    raw_path.write_text(response_text, encoding="utf-8")

                    try:
                        df = pd.read_csv(io.StringIO(response_text), sep="\t")
                    except pd.errors.EmptyDataError:
                        df = pd.DataFrame()
                    except Exception as e:
                        print(
                            f"Could not parse IEDB response in {raw_path}: {e}. "
                            "Retrying the same missing peptides...",
                            flush=True,
                        )
                        time.sleep(60)
                        continue

                    found_this_attempt = set()

                    if not df.empty:
                        if "peptide" not in df.columns:
                            print(
                                f"IEDB response missing peptide column. "
                                f"Columns={list(df.columns)}. "
                                "Retrying the same missing peptides...",
                                flush=True,
                            )
                            time.sleep(60)
                            continue

                        if "seq_num" in df.columns:
                            for _, r in df.iterrows():
                                try:
                                    sn = int(r["seq_num"]) - 1
                                except Exception:
                                    continue
                                if sn < 0 or sn >= len(pending):
                                    continue
                                orig_i, orig_pep = pending[sn]
                                if orig_i in returned_indices:
                                    continue
                                rows.append(self._normalize_row(orig_i, orig_pep, r, hla))
                                returned_indices.add(orig_i)
                                found_this_attempt.add(orig_i)
                        else:
                            for _, r in df.iterrows():
                                p = str(r["peptide"])
                                candidates = [
                                    x for x in pending
                                    if x[1] == p and x[0] not in returned_indices
                                ]
                                if candidates:
                                    orig_i, orig_pep = candidates[0]
                                    rows.append(self._normalize_row(orig_i, orig_pep, r, hla))
                                    returned_indices.add(orig_i)
                                    found_this_attempt.add(orig_i)

                    pending = [x for x in pending if x[0] not in found_this_attempt]

                    if pending:
                        print(
                            f"NetMHCpan {self.version}: {len(pending)} peptide(s) still missing "
                            f"for length {length}, chunk {ci // self.chunk_size}; "
                            f"resending only missing peptides "
                            f"(result attempt {result_attempt + 1})...",
                            flush=True,
                        )
                        time.sleep(60)
        out = pd.DataFrame(rows)
        if out.empty:
            raise RuntimeError("NetMHCpan API returned no parsed rows")
        out = out.sort_values("_input_index").drop_duplicates("_input_index", keep="first")
        compat = out.drop(columns=["_input_index"], errors="ignore").copy()
        compat["Aff(nM)"] = compat[f"netmhcpan_{self.version}_Aff_nM"]
        compat["PercentileRank"] = compat[f"netmhcpan_{self.version}_percentile"]
        compat.to_csv(out_dir / f"NetMHCpan_{self.version}_prediction.csv", index=False)
        return out.drop(columns=["_input_index"], errors="ignore")

    def _normalize_row(self, idx, peptide, r, hla):
        def val(*names):
            for n in names:
                if n in r.index:
                    return r[n]
            return np.nan
        return {
            "_input_index": idx,
            "peptide": peptide,
            f"netmhcpan_{self.version}_Aff_nM": pd.to_numeric(val("ic50", "affinity", "Aff(nM)"), errors="coerce"),
            f"netmhcpan_{self.version}_percentile": pd.to_numeric(val("percentile", "percentile_rank", "%Rank_BA", "%Rank"), errors="coerce"),
            f"netmhcpan_{self.version}_rank_label": val("rank", "BindLevel", "bind_level"),
            f"netmhcpan_{self.version}_allele": val("allele") if "allele" in r.index else hla,
        }

    def _predict_local(self, peptides, context):
        if not self.executable:
            raise ValueError("Local NetMHCpan backend requires --netmhcpanXX-exe")
        exe = Path(self.executable)
        if not exe.exists():
            raise FileNotFoundError(exe)
        out_dir = context.output_dir / self.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        inp = out_dir / "peptides.txt"
        xls = out_dir / "netmhcpan.xls"
        inp.write_text("\n".join(peptides) + "\n", encoding="utf-8")
        cmd = [str(exe), "-p", str(inp), "-a", _local_hla(context.hla), "-BA", "-xls", "-xlsfile", str(xls)]
        subprocess.run(cmd, check=True, cwd=str(exe.parent))
        # NetMHCpan xls output is tabular text. Detect the header row rather than assuming an exact version layout.
        lines = xls.read_text(encoding="utf-8", errors="replace").splitlines()
        header_i = next((i for i, x in enumerate(lines) if "Peptide" in x and ("Aff" in x or "Rank" in x)), None)
        if header_i is None:
            raise RuntimeError(f"Could not detect header in {xls}")
        df = pd.read_csv(io.StringIO("\n".join(lines[header_i:])), sep="\t")
        pcol = next(c for c in df.columns if str(c).lower() == "peptide")
        affcol = next((c for c in df.columns if "aff" in str(c).lower() and "nm" in str(c).lower()), None)
        rankcol = next((c for c in df.columns if "rank" in str(c).lower() and "ba" in str(c).lower()), None)
        out = pd.DataFrame({
            "peptide": df[pcol].astype(str),
            f"netmhcpan_{self.version}_Aff_nM": pd.to_numeric(df[affcol], errors="coerce") if affcol else np.nan,
            f"netmhcpan_{self.version}_percentile": pd.to_numeric(df[rankcol], errors="coerce") if rankcol else np.nan,
        }).drop_duplicates("peptide")
        compat = out.copy()
        compat["Aff(nM)"] = compat[f"netmhcpan_{self.version}_Aff_nM"]
        compat["PercentileRank"] = compat[f"netmhcpan_{self.version}_percentile"]
        compat.to_csv(out_dir / f"NetMHCpan_{self.version}_prediction.csv", index=False)
        return out

    def manifest(self):
        x = super().manifest(); x.update({"backend": self.backend, "executable": self.executable, "api_url": self.api_url,
                                         "chunk_size": self.chunk_size}); return x
