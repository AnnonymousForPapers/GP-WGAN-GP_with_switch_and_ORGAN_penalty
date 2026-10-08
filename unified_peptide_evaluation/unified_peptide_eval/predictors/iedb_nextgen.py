from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_BASE_URL = "https://api-nextgen-tools.iedb.org/api/v1"


def _slug(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def table_to_dataframe(table):
    columns = table.get("table_columns") or []
    data = table.get("table_data") or []
    names = []
    seen = {}
    for i, col in enumerate(columns):
        raw = col.get("name") or col.get("display_name") or f"column_{i}"
        name = _slug(raw) or f"column_{i}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        names.append(name)
    if not names and data:
        names = [f"column_{i}" for i in range(len(data[0]))]
    # Be defensive if server metadata and row width briefly differ.
    width = max([len(r) for r in data], default=len(names))
    if len(names) < width:
        names += [f"column_{i}" for i in range(len(names), width)]
    rows = [list(r) + [None] * (len(names) - len(r)) for r in data]
    return pd.DataFrame(rows, columns=names)


def result_tables(result_json):
    data = result_json.get("data") or {}
    out = []
    for table in data.get("results") or []:
        if "table_data" in table:
            out.append((table.get("type", "table"), table_to_dataframe(table)))
    return out


class IEDBNextGenClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=900, poll_seconds=2.0,
                 retries=5, max_poll_seconds=None):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = int(timeout)
        self.poll_seconds = float(poll_seconds)
        self.retries = int(retries)
        self.max_poll_seconds = (
            None if max_poll_seconds is None else float(max_poll_seconds)
        )

    def _request_json(self, url, method="GET", payload=None):
        body = None
        headers = {
            "accept": "application/json",
            "User-Agent": "unified-peptide-evaluation/1.0",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        attempt = 0

        while True:
            attempt += 1
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method=method
                )

                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    status = getattr(r, "status", None) or r.getcode()
                    content_type = r.headers.get("Content-Type", "")
                    response_text = r.read().decode("utf-8", errors="replace")

                if not response_text.strip():
                    raise RuntimeError(
                        f"HTTP {status} returned an empty response "
                        f"(Content-Type={content_type!r})"
                    )

                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    preview = response_text[:500].replace("\n", "\\n")
                    raise RuntimeError(
                        f"HTTP {status} returned non-JSON content "
                        f"(Content-Type={content_type!r}): {preview!r}"
                    ) from e

            except urllib.error.HTTPError as e:
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                preview = error_body[:500].replace("\n", "\\n")
                error = (
                    f"HTTP {e.code} {e.reason}; response={preview!r}"
                )
                print(
                    f"IEDB Next-Generation API request failed on attempt "
                    f"{attempt}: {error}. Retrying in 60s...",
                    flush=True,
                )
                time.sleep(60)

            except Exception as e:
                print(
                    f"IEDB Next-Generation API request failed on attempt "
                    f"{attempt}: {type(e).__name__}: {e}. "
                    f"Retrying in 60s...",
                    flush=True,
                )
                time.sleep(60)

    def run(self, tool_group, input_sequence_text, input_parameters, raw_dir=None, label="request"):
        payload = {
            "pipeline_id": "",
            "run_stage_range": [1, 1],
            "stages": [{
                "stage_number": 1,
                "tool_group": tool_group,
                "input_sequence_text": input_sequence_text,
                "input_parameters": input_parameters,
            }],
        }
        raw_dir = Path(raw_dir) if raw_dir is not None else None
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{label}_request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        response = self._request_json(f"{self.base_url}/pipeline", method="POST", payload=payload)
        if raw_dir is not None:
            (raw_dir / f"{label}_submission.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
        errors = response.get("errors") or []
        if errors:
            raise RuntimeError(f"IEDB {tool_group} submission errors: {errors}")
        result_id = response.get("result_id")
        results_uri = response.get("results_uri") or (f"{self.base_url}/results/{result_id}" if result_id else None)
        if not results_uri:
            raise RuntimeError(f"IEDB {tool_group} response did not contain result_id/results_uri: {response}")

        start = time.monotonic()
        poll_no = 0
        while True:
            poll_no += 1
            result = self._request_json(results_uri)
            status = str(result.get("status", "")).lower()
            if raw_dir is not None and (status != "pending" or poll_no == 1):
                (raw_dir / f"{label}_result_{status or 'unknown'}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
            if status == "done":
                return result
            if status in {"error", "failed", "failure"}:
                raise RuntimeError(f"IEDB {tool_group} job failed: {result}")
            data = result.get("data") or {}
            if data.get("errors"):
                raise RuntimeError(f"IEDB {tool_group} job errors: {data['errors']}")
            if (
                self.max_poll_seconds is not None
                and time.monotonic() - start > self.max_poll_seconds
            ):
                raise TimeoutError(
                    f"IEDB {tool_group} job did not finish within "
                    f"{self.max_poll_seconds}s"
                )
            time.sleep(self.poll_seconds)


def find_table(result_json, required_any=(), preferred_type=None):
    tables = result_tables(result_json)
    if not tables:
        raise RuntimeError("IEDB result contained no tabular result")
    if preferred_type:
        for t, df in tables:
            if t == preferred_type:
                return df
    required_any = {_slug(x) for x in required_any}
    for _, df in tables:
        if required_any.intersection(df.columns):
            return df
    # The first non-input table is generally the tool result.
    for t, df in tables:
        if t != "input_sequence_table":
            return df
    return tables[0][1]
