#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from unified_peptide_eval.generators import generate_from_pytorch_checkpoint, generate_pepinvent
from unified_peptide_eval.predictors import (
    PredictorContext,
    PredictorUnavailable,
    DeepImmunoPredictor,
    NetMHCpanPredictor,
    IEDBImmunogenicityPredictor,
    BladderSimilarityPredictor,
    PEPMatchPredictor,
    PepSyScoPredictor,
    TCGABLCAExactMatchPredictor,
)
from unified_peptide_eval.reporting import build_summary, save_ic50_plot
from unified_peptide_eval.utils import (
    RunLogger,
    find_peptide_column,
    merge_on_peptide,
    natural_peptide_or_none,
    read_table,
    resolve_bladder_csv,
    resolve_checkpoint,
    resolve_data_root,
    resolve_tcga_csv,
    set_seed,
    write_json,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified peptide generation + evaluation for GAN/LSTM/Transformer/D3PM/PepINVENT."
    )
    ap.add_argument("--model-dir", default=None, help="Directory containing model_best/model_last/model_epoch checkpoint.")
    ap.add_argument("--checkpoint", default="auto", help="auto, best, last, an explicit filename, or an explicit path.")
    ap.add_argument("--architecture", default="auto", choices=["auto", "gan", "lstm", "transformer", "d3pm", "pepinvent"])
    ap.add_argument("--input-peptides", default=None, help="Skip model generation and evaluate an existing CSV/TSV/TXT peptide file.")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generation-batch-size", type=int, default=512)
    ap.add_argument("--d3pm-batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0 for PyTorch models.")
    ap.add_argument("--q-heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--bladder-csv", default=None)
    ap.add_argument("--tcga-csv", default=None, help="TCGA_BLCA_WT_mutant_peptides_unique.csv; auto-detected under data-root if omitted.")
    ap.add_argument("--hla", default="HLA-A*02:01")

    ap.add_argument("--predictors", nargs="+", default=["all"],
                    help="all, none, or any of: deepimmuno iedb_immunogenicity netmhcpan40 netmhcpan41 pepmatch pepsysco tcga_blca similarity")
    ap.add_argument("--strict", action="store_true", help="Stop if an optional predictor fails; default is continue and log failure.")

    ap.add_argument("--tf-python", default=None)
    ap.add_argument("--deepimmuno-batch-size", type=int, default=1024)
    ap.add_argument("--deepimmuno-weights", default=None,
                    help="Optional DeepImmuno weight file/checkpoint/directory. Default: <data-root>/weights/Immunogenicity_Predictor.")

    ap.add_argument("--netmhcpan-backend", default="api", choices=["api", "local", "auto"])
    ap.add_argument("--netmhcpan40-exe", default=None)
    ap.add_argument("--netmhcpan41-exe", default=None)
    ap.add_argument("--iedb-api-chunk-size", type=int, default=200)
    ap.add_argument("--iedb-api-timeout", type=int, default=600)
    ap.add_argument("--iedb-nextgen-api-base", default="https://api-nextgen-tools.iedb.org/api/v1")
    ap.add_argument("--iedb-nextgen-timeout", type=int, default=900)
    ap.add_argument("--iedb-nextgen-poll-seconds", type=float, default=2.0)
    ap.add_argument("--pepmatch-mismatch", type=int, default=3, choices=range(0, 6))
    ap.add_argument("--pepmatch-proteome", default="Human",
                    choices=["Human", "Mouse", "Cow", "Dog", "Horse", "Pig", "Rabbit", "Rat"])
    ap.add_argument("--pepmatch-chunk-size", type=int, default=500)
    ap.add_argument("--pepsysco-backend", default="auto", choices=["auto", "csv", "command", "api"],
                    help="PepSySco backend. The current IEDB pipeline API does not accept pepsysco; auto uses CSV or command if configured.")
    ap.add_argument("--pepsysco-results-csv", default=None,
                    help="CSV downloaded from the IEDB PepSySco web tool; used by csv/auto backend.")
    ap.add_argument("--pepsysco-command", default=os.environ.get("PEPSYSCO_COMMAND"),
                    help="Optional local command template with {input} and {output} placeholders.")

    ap.add_argument("--iedb-immunogenicity-backend", default="api", choices=["api", "local", "auto"],
                    help="Default api uses IEDB Next-Generation T-cell Class I API; local retains the legacy standalone tool.")
    ap.add_argument("--iedb-immunogenicity-mask-choice", default="default",
                    choices=["default", "custom", "by_allele"])
    ap.add_argument("--iedb-immunogenicity-position-to-mask", default=None,
                    help="Only for --iedb-immunogenicity-mask-choice custom, e.g. 2,5,9")
    ap.add_argument("--iedb-immunogenicity-chunk-size", type=int, default=500)
    # Optional offline legacy fallback. These are not needed for the default API backend.
    ap.add_argument("--iedb-immunogenicity-script", default=os.environ.get("IEDB_IMMUNOGENICITY_SCRIPT"))
    ap.add_argument("--iedb-immunogenicity-python", default=os.environ.get("IEDB_IMMUNOGENICITY_PYTHON", "python2"))
    ap.add_argument("--iedb-immunogenicity-command", default=os.environ.get("IEDB_IMMUNOGENICITY_COMMAND"),
                    help="Optional legacy local command template with {input}, {output}, {script}.")

    ap.add_argument("--reinvent", default="reinvent")
    ap.add_argument("--pepinvent-device", default="cuda:0")
    ap.add_argument("--pepinvent-mask-count", type=int, default=3)
    ap.add_argument("--pepinvent-runtime-dir", default=None,
                    help="Directory containing sitecustomize.py + pepinvent_runtime_overrides.py from your runtime-only Natural20 bundle.")
    return ap.parse_args()


def filter_generated(raw_df):
    rows = []
    for i, raw in enumerate(raw_df["peptide"].tolist()):
        p = natural_peptide_or_none(raw)
        if p is None:
            continue
        rows.append({
            "source_row": i,
            "raw_peptide": str(raw),
            "peptide": p,
            "HLA": "HLA-A*0201",
            "immunogenicity": 1,
        })
    all_valid = pd.DataFrame(rows)
    unique = all_valid.drop_duplicates("peptide", keep="first").reset_index(drop=True) if len(all_valid) else all_valid.copy()
    return all_valid, unique


def load_existing(path):
    path = Path(path)
    if path.suffix.lower() == ".txt":
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            vals = [x.strip() for x in path.read_text().splitlines() if x.strip()]
            return pd.DataFrame({"peptide": vals})
    else:
        df = read_table(path)
    col = find_peptide_column(df)
    out = df.copy()
    if col != "peptide":
        out = out.rename(columns={col: "peptide"})
    return out


def predictor_names(args):
    names = []
    for x in args.predictors:
        if x.lower() == "all":
            names.extend(["deepimmuno", "iedb_immunogenicity", "netmhcpan40", "netmhcpan41", "pepmatch", "pepsysco", "tcga_blca", "similarity"])
        elif x.lower() == "none":
            continue
        else:
            names.append(x.lower())
    return list(dict.fromkeys(names))


def make_predictor(name, args, bladder_csv, tcga_csv):
    if name == "deepimmuno":
        return DeepImmunoPredictor(args.tf_python, args.deepimmuno_batch_size, args.deepimmuno_weights)
    if name == "iedb_immunogenicity":
        return IEDBImmunogenicityPredictor(
            backend=args.iedb_immunogenicity_backend,
            mask_choice=args.iedb_immunogenicity_mask_choice,
            position_to_mask=args.iedb_immunogenicity_position_to_mask,
            chunk_size=args.iedb_immunogenicity_chunk_size,
            timeout=args.iedb_nextgen_timeout,
            poll_seconds=args.iedb_nextgen_poll_seconds,
            api_base_url=args.iedb_nextgen_api_base,
            script=args.iedb_immunogenicity_script,
            python_exe=args.iedb_immunogenicity_python,
            command_template=args.iedb_immunogenicity_command,
        )
    if name == "netmhcpan40":
        return NetMHCpanPredictor("4.0", args.netmhcpan_backend, args.netmhcpan40_exe,
                                 args.iedb_api_chunk_size, args.iedb_api_timeout)
    if name == "netmhcpan41":
        return NetMHCpanPredictor("4.1", args.netmhcpan_backend, args.netmhcpan41_exe,
                                 args.iedb_api_chunk_size, args.iedb_api_timeout)
    if name == "pepmatch":
        return PEPMatchPredictor(
            mismatch=args.pepmatch_mismatch, proteome=args.pepmatch_proteome, best_match=True,
            chunk_size=args.pepmatch_chunk_size, timeout=args.iedb_nextgen_timeout,
            poll_seconds=args.iedb_nextgen_poll_seconds, api_base_url=args.iedb_nextgen_api_base,
        )
    if name == "pepsysco":
        return PepSyScoPredictor(
            backend=args.pepsysco_backend,
            results_csv=args.pepsysco_results_csv,
            command_template=args.pepsysco_command,
        )
    if name == "tcga_blca":
        return TCGABLCAExactMatchPredictor(tcga_csv, keep_lengths=(9, 10))
    if name == "similarity":
        return BladderSimilarityPredictor(bladder_csv)
    raise ValueError(f"Unknown predictor: {name}")


def copy_pepinvent_runtime_files(runtime_dir: Path):
    """Make original runtime-only Natural20 hooks available to the REINVENT child process."""
    if runtime_dir is None:
        return None
    runtime_dir = Path(runtime_dir).expanduser().resolve()
    if not runtime_dir.exists():
        raise FileNotFoundError(runtime_dir)
    # The adapter receives the directory separately through PYTHONPATH (see monkey-patch below).
    return runtime_dir



def discover_file(roots, filename):
    for root in roots:
        if root is None:
            continue
        root = Path(root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            continue
        direct = root / filename
        if direct.exists():
            return direct.parent if filename == "pepinvent_runtime_overrides.py" else direct
        try:
            for m in root.rglob(filename):
                if filename == "pepinvent_runtime_overrides.py":
                    if (m.parent / "sitecustomize.py").exists():
                        return m.parent
                else:
                    return m
        except Exception:
            pass
    return None


@contextmanager
def iedb_api_lock(lock_path: Path, logger):
    """
    Cross-process / cross-job lock for IEDB internet predictors.

    All jobs that use the same shared data_root wait on the same lock file.
    The lock is automatically released if the Python process exits.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_file:
        logger.print(f"[IEDB API LOCK] Waiting: {lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        logger.print("[IEDB API LOCK] Acquired")

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            logger.print("[IEDB API LOCK] Released")


def predictor_uses_internet(name, args):
    """Return True only when this predictor will use an internet API."""
    if name == "iedb_immunogenicity":
        return args.iedb_immunogenicity_backend in {"api", "auto"}

    if name in {"netmhcpan40", "netmhcpan41"}:
        if args.netmhcpan_backend == "local":
            return False

        if args.netmhcpan_backend == "auto":
            exe = (
                args.netmhcpan40_exe
                if name == "netmhcpan40"
                else args.netmhcpan41_exe
            )
            return not bool(exe)

        return True

    if name == "pepmatch":
        return True

    if name == "pepsysco":
        return args.pepsysco_backend == "api"

    return False


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.input_peptides and not args.model_dir:
        raise SystemExit("Provide --model-dir or --input-peptides")

    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else Path(args.input_peptides).expanduser().resolve().parent
    data_root = resolve_data_root(model_dir, args.data_root)
    bladder_csv = resolve_bladder_csv(data_root, args.bladder_csv)
    tcga_csv = resolve_tcga_csv(data_root, args.tcga_csv)
    if args.iedb_immunogenicity_backend in {"local", "auto"} and not args.iedb_immunogenicity_script:
        found = discover_file([data_root, model_dir.parent], "predict_immunogenicity.py")
        if found:
            args.iedb_immunogenicity_script = str(found)
    if not args.pepinvent_runtime_dir:
        found_runtime = discover_file([data_root, model_dir.parent], "pepinvent_runtime_overrides.py")
        if found_runtime:
            args.pepinvent_runtime_dir = str(found_runtime)

    checkpoint = None
    if args.input_peptides:
        run_name = Path(args.input_peptides).stem
    else:
        checkpoint = resolve_checkpoint(model_dir, args.checkpoint)
        run_name = f"{checkpoint.stem}_seed{args.seed}"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_dir / f"evaluation_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "evaluation.log")
    iedb_lock_path = data_root / ".iedb_api.lock"

    pred_names = predictor_names(args)
    stage_total = 3 + len(pred_names)  # generation, filtering, predictors, final report
    stage_i = 1
    predictor_status = {}
    manifests = []
    architecture = args.architecture
    generation_meta = {}

    logger.print("Unified peptide evaluation")
    logger.print(f"Seed: {args.seed}")
    logger.print(f"Model directory: {model_dir}")
    logger.print(f"Checkpoint: {checkpoint}")
    logger.print(f"Data root: {data_root}")
    logger.print(f"Bladder reference: {bladder_csv}")
    logger.print(f"TCGA-BLCA reference: {tcga_csv}")
    logger.print(f"Output: {output_dir}")
    logger.print(f"Requested predictors: {', '.join(pred_names)}")

    try:
        with logger.stage(stage_i, stage_total, "Generate/load peptide sequences"):
            stage_i += 1
            if args.input_peptides:
                raw_df = load_existing(args.input_peptides)
                architecture = "external_file"
                generation_meta = {"input_peptides": str(Path(args.input_peptides).resolve())}
            else:
                pepinvent = args.architecture == "pepinvent" or (args.architecture == "auto" and checkpoint.suffix == ".chkpt")
                if pepinvent:
                    # Ensure the package's canonical helper is importable under the name expected by the uploaded code.
                    pkg = HERE / "unified_peptide_eval"
                    if str(pkg) not in sys.path:
                        sys.path.insert(0, str(pkg))
                    runtime_dir = copy_pepinvent_runtime_files(args.pepinvent_runtime_dir)
                    if runtime_dir:
                        old = os.environ.get("PYTHONPATH", "")
                        os.environ["PYTHONPATH"] = str(runtime_dir) if not old else str(runtime_dir) + os.pathsep + old
                    raw_df, generation_meta = generate_pepinvent(
                        checkpoint, args.num_samples, args.seed, output_dir / "generation_pepinvent",
                        bladder_csv, args.reinvent, args.pepinvent_device, args.pepinvent_mask_count,
                    )
                    architecture = "pepinvent"
                else:
                    # D3PM uses a smaller default batch because every sample traverses all diffusion steps.
                    batch = args.generation_batch_size
                    if args.architecture == "d3pm" or "D3PM" in model_dir.name.upper():
                        batch = args.d3pm_batch_size
                    peptides, architecture, generation_meta = generate_from_pytorch_checkpoint(
                        checkpoint, args.architecture, args.num_samples, args.seed, batch, args.device,
                        args.q_heads, args.kv_heads,
                    )
                    raw_df = pd.DataFrame({
                        "peptide": peptides,
                        "HLA": ["HLA-A*0201"] * len(peptides),
                        "immunogenicity": [1] * len(peptides),
                    })
            raw_df.to_csv(output_dir / "generated_raw.csv", index=False)
            raw_df.to_csv(output_dir / f"generated_raw_seed{args.seed}_batch{len(raw_df)}.txt", sep="\t", index=False)
            logger.print(f"Architecture: {architecture}")
            logger.print(f"Raw sequences: {len(raw_df):,}")
            logger.print(f"Generation metadata: {generation_meta}")

        with logger.stage(stage_i, stage_total, "Filter placeholders / validate 9-10mers / deduplicate"):
            stage_i += 1
            filtered_all, unique_df = filter_generated(raw_df)
            filtered_all.to_csv(output_dir / "generated_filtered_all.csv", index=False)
            unique_df.to_csv(output_dir / "generated_filtered_unique.csv", index=False)
            unique_df.to_csv(output_dir / f"generated_filtered_unique_seed{args.seed}_batch{len(raw_df)}.txt", sep="\t", index=False)
            logger.print(f"Valid 9/10-mers before deduplication: {len(filtered_all):,}")
            logger.print(f"Unique valid 9/10-mers: {len(unique_df):,}")
            logger.print(f"Unique raw peptide strings before filtering: {raw_df['peptide'].astype(str).nunique():,}")
            if unique_df.empty:
                raise RuntimeError("No valid unique 9/10-mer peptides remain after filtering")

        merged = unique_df[["peptide"]].copy()
        context = PredictorContext(output_dir=output_dir, data_root=data_root, hla=args.hla, seed=args.seed)
        for name in pred_names:
            title = {
                "deepimmuno": "DeepImmuno scoring",
                "iedb_immunogenicity": f"IEDB Class-I immunogenicity ({args.iedb_immunogenicity_backend})",
                "netmhcpan40": "NetMHCpan 4.0 binding affinity",
                "netmhcpan41": "NetMHCpan 4.1 binding affinity",
                "pepmatch": f"IEDB PEPMatch against {args.pepmatch_proteome} proteome (<={args.pepmatch_mismatch} mismatches)",
                "pepsysco": f"IEDB PepSySco synthesis score ({args.pepsysco_backend})",
                "tcga_blca": "Exact match to TCGA-BLCA mutant peptides",
                "similarity": "Similarity to bladder-cancer peptide reference",
            }.get(name, name)
            with logger.stage(stage_i, stage_total, title):
                stage_i += 1
                predictor = make_predictor(name, args, bladder_csv, tcga_csv)
                manifests.append(predictor.manifest())
                try:
                    if predictor_uses_internet(name, args):
                        with iedb_api_lock(iedb_lock_path, logger):
                            result = predictor.predict(
                                unique_df["peptide"].tolist(), context
                            )
                    else:
                        result = predictor.predict(
                            unique_df["peptide"].tolist(), context
                        )

                    merged = merge_on_peptide(merged, result)
                    predictor_status[name] = {"status": "done", "rows": int(len(result))}
                    logger.print(f"Scored rows: {len(result):,}")
                except PredictorUnavailable as e:
                    predictor_status[name] = {"status": "skipped", "reason": str(e)}
                    logger.print(f"[SKIPPED] {e}")
                except Exception as e:
                    predictor_status[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                    logger.print(f"[PREDICTOR FAILED, CONTINUING] {type(e).__name__}: {e}")
                    if args.strict:
                        raise

        with logger.stage(stage_i, stage_total, "Merge outputs / statistics / final summary"):
            merged.to_csv(output_dir / "all_predictions.csv", index=False)
            for v in ["4.0", "4.1"]:
                try:
                    save_ic50_plot(merged, v, output_dir / f"IC50_compare_NetMHCpan_{v}.png")
                except Exception as e:
                    logger.print(f"[NOTE] Could not create NetMHCpan {v} plot: {e}")
            summary = build_summary(raw_df, filtered_all, unique_df, merged, architecture, checkpoint,
                                    args.seed, predictor_status)
            summary["generation_meta"] = generation_meta
            summary["data_root"] = str(data_root)
            summary["bladder_csv"] = str(bladder_csv)
            summary["tcga_csv"] = str(tcga_csv)
            write_json(output_dir / "summary.json", summary)
            pd.json_normalize(summary, sep=".").to_csv(output_dir / "summary.csv", index=False)
            write_json(output_dir / "predictor_manifest.json", manifests)
            logger.print(json.dumps(summary, indent=2, default=str))

        logger.print("\n" + "=" * 78)
        logger.print("ALL REQUESTED STAGES FINISHED")
        logger.print("=" * 78)
        logger.print(f"Results: {output_dir}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
