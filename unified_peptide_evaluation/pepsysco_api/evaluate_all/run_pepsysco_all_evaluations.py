#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# This script expects the fixed standalone PepSySco client to be named
# pepsysco_api.py and to be in the same directory as this script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pepsysco_api import optional_file_lock, submit_pepsysco


def find_evaluation_dirs(root: Path) -> list[Path]:
    """
    Recursively find evaluator output folders by locating
    generated_filtered_unique.csv.
    """
    dirs = []
    for path in root.rglob("generated_filtered_unique.csv"):
        if path.is_file():
            dirs.append(path.parent.resolve())
    return sorted(set(dirs))


def pick_peptide_column(df: pd.DataFrame) -> str:
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }
    for candidate in ("peptide", "sequence", "peptide_sequence"):
        if candidate in normalized:
            return normalized[candidate]
    raise RuntimeError(
        f"Could not find peptide column. Columns={list(df.columns)}"
    )


def load_eval_peptides(eval_dir: Path) -> list[str]:
    path = eval_dir / "generated_filtered_unique.csv"
    df = pd.read_csv(path)
    col = pick_peptide_column(df)

    peptides = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    return list(dict.fromkeys(p for p in peptides if p))


def write_prediction_files(eval_dir: Path, rows: list[dict]) -> pd.DataFrame:
    out_dir = eval_dir / "pepsysco"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_style = pd.DataFrame(rows)
    raw_style.to_csv(
        out_dir / "pepsysco_predictions.csv",
        index=False,
    )
    raw_style.to_csv(
        out_dir / "PepSySco_raw_combined.csv",
        index=False,
    )

    normalized = pd.DataFrame({
        "peptide": raw_style["peptide"].astype(str).str.strip().str.upper(),
        "pepsysco_score": pd.to_numeric(
            raw_style["Pepsysco Score"],
            errors="coerce",
        ),
    }).drop_duplicates("peptide", keep="first")

    normalized.to_csv(
        out_dir / "PepSySco_prediction.csv",
        index=False,
    )
    return normalized


def load_existing_prediction(eval_dir: Path) -> pd.DataFrame | None:
    path = eval_dir / "pepsysco" / "PepSySco_prediction.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "peptide" not in df.columns or "pepsysco_score" not in df.columns:
        return None

    out = df[["peptide", "pepsysco_score"]].copy()
    out["peptide"] = out["peptide"].astype(str).str.strip().str.upper()
    out["pepsysco_score"] = pd.to_numeric(
        out["pepsysco_score"],
        errors="coerce",
    )
    return out.drop_duplicates("peptide", keep="first")


def merge_into_all_predictions(
    eval_dir: Path,
    predictions: pd.DataFrame,
) -> tuple[int, int]:
    all_path = eval_dir / "all_predictions.csv"

    if all_path.exists():
        all_df = pd.read_csv(all_path)
    else:
        # If the evaluator has not yet written all_predictions.csv, construct
        # a minimal one from generated_filtered_unique.csv so the PepSySco
        # result is still preserved.
        src = pd.read_csv(eval_dir / "generated_filtered_unique.csv")
        pcol = pick_peptide_column(src)
        all_df = pd.DataFrame({
            "peptide": src[pcol].astype(str).str.strip().str.upper()
        })

    if "peptide" not in all_df.columns:
        raise RuntimeError(
            f"{all_path} has no peptide column. Columns={list(all_df.columns)}"
        )

    all_df["peptide"] = all_df["peptide"].astype(str).str.strip().str.upper()

    # Replace an older/incomplete PepSySco column cleanly instead of creating
    # _x/_y suffixes.
    all_df = all_df.drop(columns=["pepsysco_score"], errors="ignore")
    all_df = all_df.merge(
        predictions[["peptide", "pepsysco_score"]],
        on="peptide",
        how="left",
    )

    all_df.to_csv(all_path, index=False)

    scored = int(all_df["pepsysco_score"].notna().sum())
    total = int(len(all_df))
    return scored, total


def update_summary(eval_dir: Path, scored: int, total: int) -> None:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[NOTE] Could not read {summary_path}: {exc}",
            flush=True,
        )
        return

    status = summary.setdefault("predictor_status", {})
    status["pepsysco"] = {
        "status": "done",
        "rows": int(total),
        "scored": int(scored),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    # Keep summary.csv synchronized with summary.json.
    pd.json_normalize(summary, sep=".").to_csv(
        eval_dir / "summary.csv",
        index=False,
    )


def prediction_complete(
    predictions: pd.DataFrame | None,
    peptides: list[str],
) -> bool:
    if predictions is None:
        return False

    expected = set(peptides)
    got = set(
        predictions.loc[
            predictions["pepsysco_score"].notna(),
            "peptide",
        ].astype(str)
    )
    return expected.issubset(got)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run IEDB PepSySco for every unified-evaluator output folder "
            "under a root directory and merge pepsysco_score into "
            "all_predictions.csv."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root folder to search recursively.",
    )
    ap.add_argument(
        "--lock-file",
        default=None,
        help=(
            "Shared IEDB flock path. If omitted, use .iedb_api.lock "
            "in the parent directory of --root."
        ),
    )
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--http-timeout", type=int, default=120)
    ap.add_argument("--retry-seconds", type=int, default=60)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Resubmit PepSySco even if a complete prediction CSV exists.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of evaluation folders to process.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root folder does not exist: {root}")

    lock_file = (
        Path(args.lock_file).expanduser().resolve()
        if args.lock_file
        else root.parent / ".iedb_api.lock"
    )

    eval_dirs = find_evaluation_dirs(root)
    if args.limit is not None:
        eval_dirs = eval_dirs[: max(0, int(args.limit))]

    print("=" * 78)
    print("PepSySco recursive evaluator post-processing")
    print(f"Root: {root}")
    print(f"Evaluation folders found: {len(eval_dirs):,}")
    print(f"Shared IEDB lock: {lock_file}")
    print("=" * 78)

    if not eval_dirs:
        print("No generated_filtered_unique.csv files found.")
        return 0

    done = 0
    reused = 0
    failed = 0

    for i, eval_dir in enumerate(eval_dirs, start=1):
        print("")
        print("=" * 78)
        print(f"[{i}/{len(eval_dirs)}] {eval_dir}")
        print("=" * 78)

        try:
            peptides = load_eval_peptides(eval_dir)
            print(f"Unique evaluator peptides: {len(peptides):,}")

            existing = load_existing_prediction(eval_dir)

            if not args.force and prediction_complete(existing, peptides):
                print(
                    "[REUSE] Complete PepSySco_prediction.csv already exists; "
                    "no API resubmission."
                )
                scored, total = merge_into_all_predictions(
                    eval_dir,
                    existing,
                )
                update_summary(eval_dir, scored, total)
                print(
                    f"[DONE] all_predictions.csv updated: "
                    f"{scored:,}/{total:,} PepSySco scores"
                )
                reused += 1
                continue

            with optional_file_lock(lock_file):
                rows = submit_pepsysco(
                    peptides,
                    poll_seconds=args.poll_seconds,
                    timeout=args.http_timeout,
                    retry_seconds=args.retry_seconds,
                    output_dir=eval_dir / "pepsysco",
                )

            pred = write_prediction_files(eval_dir, rows)
            scored, total = merge_into_all_predictions(eval_dir, pred)
            update_summary(eval_dir, scored, total)

            print(
                f"[DONE] PepSySco merged into all_predictions.csv: "
                f"{scored:,}/{total:,} scored"
            )
            done += 1

        except Exception as exc:
            failed += 1
            print(
                f"[FAILED] {type(exc).__name__}: {exc}",
                flush=True,
            )

    print("")
    print("=" * 78)
    print("FINISHED")
    print(f"New PepSySco runs: {done:,}")
    print(f"Reused existing results: {reused:,}")
    print(f"Failed folders: {failed:,}")
    print("=" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
