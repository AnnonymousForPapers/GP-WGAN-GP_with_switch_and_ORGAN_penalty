#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

BASE_URL = "https://api-nextgen-tools.iedb.org/api/v1"
DEFAULT_PIPELINE_SPEC_ID = "3366c695-26ae-4966-ac5f-1082a1edfd10"
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _norm_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def request_json(url, method="GET", payload=None, timeout=120, retry_seconds=60):
    body = None
    headers = {
        "accept": "application/json",
        "User-Agent": "pepsysco-api-client/1.0",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status = getattr(r, "status", None) or r.getcode()
                ctype = r.headers.get("Content-Type", "")
                text = r.read().decode("utf-8", errors="replace")

            if not text.strip():
                raise RuntimeError(
                    f"HTTP {status} returned an empty response "
                    f"(Content-Type={ctype!r})"
                )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"HTTP {status} returned non-JSON content: {text[:500]!r}"
                ) from exc
        except Exception as exc:
            print(
                f"[HTTP FAILED] attempt={attempt}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(f"Retrying in {retry_seconds} seconds...", flush=True)
            time.sleep(retry_seconds)


def contains_pepsysco(obj):
    if isinstance(obj, dict):
        if str(obj.get("tool_group", "")).lower() == "pepsysco":
            return True
        return any(contains_pepsysco(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_pepsysco(v) for v in obj)
    return False


def verify_pipeline_spec(spec_id, base_url=BASE_URL, timeout=120, retry_seconds=60):
    url = f"{base_url.rstrip('/')}/pipeline_spec/{spec_id}"
    print(f"[SPEC] Checking {url}", flush=True)
    spec = request_json(url, timeout=timeout, retry_seconds=retry_seconds)
    if not contains_pepsysco(spec):
        raise RuntimeError(
            f"Pipeline spec {spec_id} does not contain tool_group='pepsysco'."
        )
    print("[SPEC] PepSySco configuration verified.", flush=True)
    return spec


def find_result_rows(obj):
    rows = []
    if isinstance(obj, dict):
        km = {_norm_key(k): k for k in obj}

        peptide_key = None
        for candidate in ("peptide", "peptidesequence", "sequence"):
            if candidate in km:
                peptide_key = km[candidate]
                break

        score_key = None
        for candidate in ("pepsyscoscore", "pepsysco"):
            if candidate in km:
                score_key = km[candidate]
                break

        if peptide_key is not None and score_key is not None:
            pep = str(obj.get(peptide_key, "")).strip().upper()
            try:
                score = float(obj.get(score_key))
            except (TypeError, ValueError):
                score = None

            if pep and score is not None:
                name = ""
                for candidate in ("sequencename", "name", "sequenceid"):
                    if candidate in km:
                        name = str(obj.get(km[candidate], ""))
                        break
                rows.append({
                    "sequence name": name,
                    "peptide": pep,
                    "Pepsysco Score": score,
                })

        for value in obj.values():
            rows.extend(find_result_rows(value))

    elif isinstance(obj, list):
        for value in obj:
            rows.extend(find_result_rows(value))

    return rows


def validate_peptides(peptides):
    out = []
    seen = set()
    for raw in peptides:
        pep = str(raw).strip().upper()
        if not pep:
            continue
        if any(ch not in CANONICAL_AA for ch in pep):
            raise ValueError(
                f"PepSySco accepts only canonical amino acids; invalid peptide: {pep!r}"
            )
        if pep not in seen:
            seen.add(pep)
            out.append(pep)
    if not out:
        raise ValueError("No peptide sequences were provided.")
    return out


@contextmanager
def optional_file_lock(lock_path):
    if not lock_path:
        yield
        return

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as f:
        print(f"[IEDB API LOCK] Waiting: {lock_path}", flush=True)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        print("[IEDB API LOCK] Acquired", flush=True)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            print("[IEDB API LOCK] Released", flush=True)


def submit_pepsysco(
    peptides,
    base_url=BASE_URL,
    poll_seconds=30,
    timeout=120,
    retry_seconds=60,
    output_dir=None,
):
    peptides = validate_peptides(peptides)

    payload = {
        "pipeline_id": "",
        "run_stage_range": [1, 1],
        "stages": [{
            "stage_number": 1,
            "tool_group": "pepsysco",
            "input_sequence_text": "\n".join(peptides),
            "input_parameters": {},
        }],
    }

    outdir = Path(output_dir) if output_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "pepsysco_request.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    print(f"[SUBMIT] PepSySco peptides: {len(peptides):,}", flush=True)
    submission = request_json(
        f"{base_url.rstrip('/')}/pipeline",
        method="POST",
        payload=payload,
        timeout=timeout,
        retry_seconds=retry_seconds,
    )

    if outdir:
        (outdir / "pepsysco_submission.json").write_text(
            json.dumps(submission, indent=2), encoding="utf-8"
        )

    if submission.get("errors"):
        raise RuntimeError(f"IEDB PepSySco submission errors: {submission['errors']}")

    result_id = submission.get("result_id")
    results_uri = submission.get("results_uri")
    if not results_uri and result_id:
        results_uri = f"{base_url.rstrip('/')}/results/{result_id}"
    if not results_uri:
        raise RuntimeError("Submission did not return result_id/results_uri.")

    print(f"[SUBMITTED] pipeline_id={submission.get('pipeline_id')}", flush=True)
    print(f"[SUBMITTED] result_id={result_id}", flush=True)
    print(f"[SUBMITTED] results_uri={results_uri}", flush=True)

    poll = 0
    start = time.monotonic()
    last_status = None

    while True:
        poll += 1
        result = request_json(
            results_uri,
            timeout=timeout,
            retry_seconds=retry_seconds,
        )
        status = str(result.get("status", "unknown")).lower()
        elapsed = time.monotonic() - start

        if status != last_status or poll == 1 or poll % 10 == 0:
            print(f"[POLL {poll}] elapsed={elapsed:.1f}s status={status}", flush=True)
            last_status = status

        if status == "done":
            if outdir:
                (outdir / "pepsysco_raw_result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )

            rows = find_result_rows(result)
            dedup = {}
            for row in rows:
                dedup[row["peptide"]] = row
            rows = [dedup[p] for p in peptides if p in dedup]

            if not rows:
                raise RuntimeError(
                    "PepSySco finished, but no peptide/Pepsysco Score rows "
                    "could be parsed. Inspect pepsysco_raw_result.json."
                )
            return rows

        if status in {"error", "failed", "failure"}:
            if outdir:
                (outdir / "pepsysco_raw_result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
            raise RuntimeError(f"IEDB PepSySco job ended with status={status}")

        data = result.get("data") or {}
        if isinstance(data, dict) and data.get("errors"):
            raise RuntimeError(f"IEDB PepSySco job errors: {data['errors']}")

        time.sleep(poll_seconds)


def read_peptides(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            km = {_norm_key(x): x for x in fields}
            col = None
            for candidate in ("peptide", "peptidesequence", "sequence"):
                if candidate in km:
                    col = km[candidate]
                    break
            if col is None:
                raise ValueError(f"No peptide column found. Columns: {fields}")
            return validate_peptides(row.get(col, "") for row in reader)

    return validate_peptides(
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(">")
    )


def write_csv(rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence name", "peptide", "Pepsysco Score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input")
    src.add_argument("--peptides", nargs="+")
    ap.add_argument("--output-dir", default="pepsysco_test_output")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--http-timeout", type=int, default=120)
    ap.add_argument("--retry-seconds", type=int, default=60)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--pipeline-spec-id", default=DEFAULT_PIPELINE_SPEC_ID)
    ap.add_argument("--no-verify-spec", action="store_true")
    ap.add_argument("--lock-file", default=None)
    args = ap.parse_args()

    peptides = read_peptides(args.input) if args.input else validate_peptides(args.peptides)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with optional_file_lock(args.lock_file):
        if not args.no_verify_spec:
            spec = verify_pipeline_spec(
                args.pipeline_spec_id,
                base_url=args.base_url,
                timeout=args.http_timeout,
                retry_seconds=args.retry_seconds,
            )
            (outdir / "pipeline_spec.json").write_text(
                json.dumps(spec, indent=2), encoding="utf-8"
            )

        rows = submit_pepsysco(
            peptides,
            base_url=args.base_url,
            poll_seconds=args.poll_seconds,
            timeout=args.http_timeout,
            retry_seconds=args.retry_seconds,
            output_dir=outdir,
        )

    csv_path = outdir / "pepsysco_predictions.csv"
    write_csv(rows, csv_path)

    print("")
    print("=" * 70)
    print("PepSySco finished successfully")
    print(f"Requested peptides: {len(peptides):,}")
    print(f"Scored peptides:    {len(rows):,}")
    print(f"Output CSV:         {csv_path}")
    print("=" * 70)
    for row in rows:
        print(f"{row['peptide']}\t{row['Pepsysco Score']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
