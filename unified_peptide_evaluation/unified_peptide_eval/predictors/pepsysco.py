from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .base import BasePredictor, PredictorContext, PredictorUnavailable


BASE_URL = "https://api-nextgen-tools.iedb.org/api/v1"
DEFAULT_PIPELINE_SPEC_ID = "3366c695-26ae-4966-ac5f-1082a1edfd10"
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _norm_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _pick(df, aliases):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for a in aliases:
        key = a.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _normalize_results(raw: pd.DataFrame, peptides):
    pcol = _pick(raw, ["peptide", "input_sequence", "input sequence", "sequence"])
    scol = _pick(raw, [
        "pepsysco_score", "pepsysco score", "pepsysco", "score",
        "synthesis_score", "peptide_synthesis_score", "prediction",
    ])
    if pcol is None or scol is None:
        raise RuntimeError(
            "Could not identify PepSySco peptide/score columns. "
            f"Columns={list(raw.columns)}"
        )

    norm = pd.DataFrame({
        "peptide": raw[pcol].astype(str).str.strip().str.upper(),
        "pepsysco_score": pd.to_numeric(raw[scol], errors="coerce"),
    }).drop_duplicates("peptide", keep="first")

    return pd.DataFrame({"peptide": peptides}).merge(
        norm, on="peptide", how="left"
    )


def _validate_peptides(peptides):
    out = []
    seen = set()

    for raw in peptides:
        pep = str(raw).strip().upper()
        if not pep:
            continue

        if any(ch not in CANONICAL_AA for ch in pep):
            raise ValueError(
                "PepSySco accepts only canonical amino acids; "
                f"invalid peptide: {pep!r}"
            )

        if pep not in seen:
            seen.add(pep)
            out.append(pep)

    if not out:
        raise ValueError("No peptide sequences were provided.")

    return out


def _request_json(
    url,
    method="GET",
    payload=None,
    timeout=120,
    retry_seconds=60,
):
    """
    Same retry behavior as the standalone pepsysco_api.py:

      * each HTTP request has its own timeout;
      * HTTP/network/empty/non-JSON failures retry indefinitely;
      * wait retry_seconds between failed attempts.
    """
    body = None
    headers = {
        "accept": "application/json",
        "User-Agent": "unified-peptide-eval-pepsysco/1.0",
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    attempt = 0

    while True:
        attempt += 1

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type", "")
                text = response.read().decode("utf-8", errors="replace")

            if not text.strip():
                raise RuntimeError(
                    f"HTTP {status} returned an empty response "
                    f"(Content-Type={content_type!r})"
                )

            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"HTTP {status} returned non-JSON content: "
                    f"{text[:500]!r}"
                ) from exc

        except Exception as exc:
            print(
                f"[HTTP FAILED] attempt={attempt}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(
                f"Retrying in {retry_seconds} seconds...",
                flush=True,
            )
            time.sleep(retry_seconds)


def _contains_pepsysco(obj):
    if isinstance(obj, dict):
        if str(obj.get("tool_group", "")).lower() == "pepsysco":
            return True
        return any(_contains_pepsysco(v) for v in obj.values())

    if isinstance(obj, list):
        return any(_contains_pepsysco(v) for v in obj)

    return False


def _verify_pipeline_spec(
    spec_id,
    base_url=BASE_URL,
    timeout=120,
    retry_seconds=60,
):
    url = f"{base_url.rstrip('/')}/pipeline_spec/{spec_id}"

    print(f"[SPEC] Checking {url}", flush=True)

    spec = _request_json(
        url,
        timeout=timeout,
        retry_seconds=retry_seconds,
    )

    if not _contains_pepsysco(spec):
        raise RuntimeError(
            f"Pipeline spec {spec_id} does not contain "
            "tool_group='pepsysco'."
        )

    print("[SPEC] PepSySco configuration verified.", flush=True)
    return spec


def _find_result_rows(obj):
    """
    Parse the real IEDB Next-Generation PepSySco typed-table result.

    Current result format:
      type = "peptide_table"

      table columns include:
        sequence_name
        peptide
        score

      The score column has:
        source = "pepsysco"
        display_name = "Pepsysco Score"
    """
    rows = []

    if not isinstance(obj, dict):
        return rows

    data = obj.get("data") or {}
    tables = data.get("results") or []

    for table in tables:
        if not isinstance(table, dict):
            continue

        if table.get("type") != "peptide_table":
            continue

        columns = table.get("table_columns") or []
        table_data = table.get("table_data") or []

        names = []
        display_names = []
        sources = []

        for col in columns:
            if isinstance(col, dict):
                names.append(str(col.get("name", "")))
                display_names.append(str(col.get("display_name", "")))
                sources.append(str(col.get("source", "")))
            else:
                names.append(str(col))
                display_names.append("")
                sources.append("")

        def find_col(*aliases):
            aliases = {_norm_key(x) for x in aliases}
            for i, (name, display) in enumerate(
                zip(names, display_names)
            ):
                if (
                    _norm_key(name) in aliases
                    or _norm_key(display) in aliases
                ):
                    return i
            return None

        peptide_i = find_col("peptide")
        name_i = find_col("sequence_name", "sequence name")

        # Prefer the column explicitly sourced from PepSySco.
        score_i = None

        for i, source in enumerate(sources):
            if _norm_key(source) == "pepsysco":
                if _norm_key(names[i]) in {
                    "score",
                    "pepsyscoscore",
                    "pepsysco",
                }:
                    score_i = i
                    break

        # Compatible fallback if IEDB changes only the metadata slightly.
        if score_i is None:
            score_i = find_col(
                "score",
                "Pepsysco Score",
                "pepsysco_score",
                "pepsysco",
            )

        if peptide_i is None or score_i is None:
            continue

        for row in table_data:
            if not isinstance(row, (list, tuple)):
                continue

            if len(row) <= max(peptide_i, score_i):
                continue

            peptide = str(row[peptide_i]).strip().upper()

            try:
                score = float(row[score_i])
            except (TypeError, ValueError):
                continue

            sequence_name = ""
            if name_i is not None and len(row) > name_i:
                sequence_name = str(row[name_i])

            rows.append({
                "sequence name": sequence_name,
                "peptide": peptide,
                "Pepsysco Score": score,
            })

    return rows


class PepSyScoPredictor(BasePredictor):
    """
    PepSySco adapter for the unified evaluator.

    Supported backends:
      auto
          Use --pepsysco-results-csv if supplied, otherwise use a configured
          --pepsysco-command. If neither is supplied, skip PepSySco.

          API use remains explicit because the main evaluator's shared IEDB
          lock identifies PepSySco as an internet job when
          --pepsysco-backend api is selected.

      csv
          Import a downloaded PepSySco CSV.

      command
          Run a user-supplied local command template.

      api
          Same behavior as the working standalone pepsysco_api.py:
            * verify the PepSySco pipeline specification;
            * submit one /pipeline job containing all evaluator peptides;
            * retry failed HTTP requests indefinitely every 60 s;
            * poll the same results_uri indefinitely every 30 s;
            * save request/submission/raw-result JSON;
            * parse the IEDB peptide_table;
            * return peptide + pepsysco_score to the evaluator.

    The evaluator should be run with:
        --pepsysco-backend api
    """

    name = "pepsysco"
    version = "IEDB PepSySco"

    def __init__(
        self,
        backend="auto",
        results_csv=None,
        command_template=None,
        api_base_url=BASE_URL,
        pipeline_spec_id=DEFAULT_PIPELINE_SPEC_ID,
        api_timeout=120,
        poll_seconds=30,
        retry_seconds=60,
        verify_spec=True,
    ):
        self.backend = str(backend).lower()

        self.results_csv = (
            Path(results_csv).expanduser().resolve()
            if results_csv
            else None
        )

        self.command_template = command_template

        self.api_base_url = str(api_base_url).rstrip("/")
        self.pipeline_spec_id = str(pipeline_spec_id)
        self.api_timeout = int(api_timeout)
        self.poll_seconds = int(poll_seconds)
        self.retry_seconds = int(retry_seconds)
        self.verify_spec = bool(verify_spec)

    def _from_csv(self, peptides, out_dir: Path):
        if not self.results_csv or not self.results_csv.exists():
            raise PredictorUnavailable(
                "PepSySco CSV backend requires --pepsysco-results-csv "
                "pointing to a CSV downloaded from the IEDB PepSySco web tool."
            )

        raw = pd.read_csv(self.results_csv)

        raw.to_csv(
            out_dir / "PepSySco_raw_combined.csv",
            index=False,
        )

        out = _normalize_results(raw, peptides)

        out.to_csv(
            out_dir / "PepSySco_prediction.csv",
            index=False,
        )

        return out

    def _from_command(self, peptides, out_dir: Path):
        if not self.command_template:
            raise PredictorUnavailable(
                "PepSySco command backend requires --pepsysco-command. "
                "The template may use {input} and {output}."
            )

        input_path = out_dir / "PepSySco_input.txt"
        output_path = out_dir / "PepSySco_command_output.csv"

        input_path.write_text(
            "\n".join(peptides) + "\n",
            encoding="utf-8",
        )

        cmd = self.command_template.format(
            input=str(input_path),
            output=str(output_path),
        )

        subprocess.run(
            shlex.split(cmd),
            check=True,
        )

        if not output_path.exists():
            raise RuntimeError(
                "PepSySco command finished but did not create expected "
                f"output: {output_path}"
            )

        raw = pd.read_csv(output_path)

        raw.to_csv(
            out_dir / "PepSySco_raw_combined.csv",
            index=False,
        )

        out = _normalize_results(raw, peptides)

        out.to_csv(
            out_dir / "PepSySco_prediction.csv",
            index=False,
        )

        return out

    def _from_api(self, peptides, out_dir: Path):
        peptides = _validate_peptides(peptides)

        # Same input text saved by the standalone client / postprocessor.
        input_path = out_dir / "PepSySco_input.txt"
        input_path.write_text(
            "\n".join(peptides) + "\n",
            encoding="utf-8",
        )

        if self.verify_spec:
            spec = _verify_pipeline_spec(
                self.pipeline_spec_id,
                base_url=self.api_base_url,
                timeout=self.api_timeout,
                retry_seconds=self.retry_seconds,
            )

            (out_dir / "pipeline_spec.json").write_text(
                json.dumps(spec, indent=2),
                encoding="utf-8",
            )

        payload = {
            "pipeline_id": "",
            "run_stage_range": [1, 1],
            "stages": [
                {
                    "stage_number": 1,
                    "tool_group": "pepsysco",
                    "input_sequence_text": "\n".join(peptides),
                    "input_parameters": {},
                }
            ],
        }

        (out_dir / "pepsysco_request.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(
            f"[SUBMIT] PepSySco peptides: {len(peptides):,}",
            flush=True,
        )

        submission = _request_json(
            f"{self.api_base_url}/pipeline",
            method="POST",
            payload=payload,
            timeout=self.api_timeout,
            retry_seconds=self.retry_seconds,
        )

        (out_dir / "pepsysco_submission.json").write_text(
            json.dumps(submission, indent=2),
            encoding="utf-8",
        )

        if submission.get("errors"):
            raise RuntimeError(
                "IEDB PepSySco submission errors: "
                f"{submission['errors']}"
            )

        result_id = submission.get("result_id")
        results_uri = submission.get("results_uri")

        if not results_uri and result_id:
            results_uri = (
                f"{self.api_base_url}/results/{result_id}"
            )

        if not results_uri:
            raise RuntimeError(
                "PepSySco submission did not return "
                "result_id/results_uri."
            )

        print(
            f"[SUBMITTED] pipeline_id={submission.get('pipeline_id')}",
            flush=True,
        )
        print(
            f"[SUBMITTED] result_id={result_id}",
            flush=True,
        )
        print(
            f"[SUBMITTED] results_uri={results_uri}",
            flush=True,
        )

        poll = 0
        start = time.monotonic()
        last_status = None

        while True:
            poll += 1

            result = _request_json(
                results_uri,
                timeout=self.api_timeout,
                retry_seconds=self.retry_seconds,
            )

            status = str(
                result.get("status", "unknown")
            ).lower()

            elapsed = time.monotonic() - start

            if (
                status != last_status
                or poll == 1
                or poll % 10 == 0
            ):
                print(
                    f"[POLL {poll}] "
                    f"elapsed={elapsed:.1f}s "
                    f"status={status}",
                    flush=True,
                )
                last_status = status

            if status == "done":
                (out_dir / "pepsysco_raw_result.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

                rows = _find_result_rows(result)

                dedup = {}
                for row in rows:
                    dedup[row["peptide"]] = row

                rows = [
                    dedup[p]
                    for p in peptides
                    if p in dedup
                ]

                if not rows:
                    raise RuntimeError(
                        "PepSySco finished, but no "
                        "peptide/Pepsysco Score rows could be parsed. "
                        "Inspect pepsysco_raw_result.json."
                    )

                # Save the same human-readable API result CSV as the
                # standalone pepsysco_api.py.
                raw = pd.DataFrame(
                    rows,
                    columns=[
                        "sequence name",
                        "peptide",
                        "Pepsysco Score",
                    ],
                )

                raw.to_csv(
                    out_dir / "pepsysco_predictions.csv",
                    index=False,
                )

                raw.to_csv(
                    out_dir / "PepSySco_raw_combined.csv",
                    index=False,
                )

                # Convert to the normalized evaluator output:
                # peptide, pepsysco_score
                out = _normalize_results(raw, peptides)

                out.to_csv(
                    out_dir / "PepSySco_prediction.csv",
                    index=False,
                )

                scored = int(
                    out["pepsysco_score"].notna().sum()
                )

                print(
                    f"[DONE] PepSySco scored "
                    f"{scored:,}/{len(peptides):,} peptide(s)",
                    flush=True,
                )

                return out

            if status in {
                "error",
                "failed",
                "failure",
            }:
                (out_dir / "pepsysco_raw_result.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

                raise RuntimeError(
                    "IEDB PepSySco job ended with "
                    f"status={status}"
                )

            data = result.get("data") or {}

            if (
                isinstance(data, dict)
                and data.get("errors")
            ):
                (out_dir / "pepsysco_raw_result.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

                raise RuntimeError(
                    "IEDB PepSySco job errors: "
                    f"{data['errors']}"
                )

            # Deliberately no overall poll timeout.
            time.sleep(self.poll_seconds)

    def predict(
        self,
        peptides,
        context: PredictorContext,
    ):
        peptides = _validate_peptides(peptides)

        out_dir = context.output_dir / "pepsysco"
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        backend = self.backend

        if backend == "auto":
            if (
                self.results_csv
                and self.results_csv.exists()
            ):
                backend = "csv"

            elif self.command_template:
                backend = "command"

            else:
                raise PredictorUnavailable(
                    "PepSySco auto mode has no CSV or command configured. "
                    "To use the working IEDB Next-Generation API, run the "
                    "evaluator with --pepsysco-backend api. Keeping API use "
                    "explicit allows the evaluator's shared IEDB API lock "
                    "to serialize PepSySco with the other IEDB jobs."
                )

        if backend == "csv":
            return self._from_csv(
                peptides,
                out_dir,
            )

        if backend == "command":
            return self._from_command(
                peptides,
                out_dir,
            )

        if backend == "api":
            return self._from_api(
                peptides,
                out_dir,
            )

        raise ValueError(
            f"Unknown PepSySco backend: {backend}"
        )

    def manifest(self):
        x = super().manifest()

        x.update({
            "backend": self.backend,
            "results_csv": (
                str(self.results_csv)
                if self.results_csv
                else None
            ),
            "command_configured": bool(
                self.command_template
            ),
            "api_base_url": self.api_base_url,
            "pipeline_spec_id": self.pipeline_spec_id,
            "api_timeout": self.api_timeout,
            "poll_seconds": self.poll_seconds,
            "retry_seconds": self.retry_seconds,
            "verify_spec": self.verify_spec,
            "api_note": (
                "API backend uses the working IEDB "
                "Next-Generation POST /pipeline flow "
                "with tool_group='pepsysco'."
            ),
        })

        return x
