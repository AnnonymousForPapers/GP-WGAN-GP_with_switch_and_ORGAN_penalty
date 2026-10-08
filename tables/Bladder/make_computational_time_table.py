#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "result"

# Table column -> compatible saved timer filenames, in priority order.
#
# Some older WGAN-GP scripts save the epoch-level accumulated forward times
# with an "_each.npy" suffix even though each stored entry is already the
# sum for that epoch. Support both naming conventions.
COMPONENT_FILES = {
    "critic_update": [
        "Dupdate_time_sum.npy",
        "Dupdate_time_each.npy",
    ],
    "generator_update": [
        "Gupdate_time_sum.npy",
        "Gupdate_time_each.npy",
    ],
    "reward_update": [
        "Supdate_time_sum.npy",
        "Supdate_time_each.npy",
    ],
    "critic_forward": [
        "Dpred_time_sum.npy",
        "Dpred_time_each.npy",
    ],
    "generator_forward": [
        "Ggen_time_sum.npy",
        "Ggen_time_each.npy",
    ],
    "reward_forward": [
        "Sgen_time_sum.npy",
        "Sgen_time_each.npy",
    ],
    "predictor_forward": [
        "CNNPred_time_sum.npy",
        "CNNPred_time_each.npy",
    ],
}

DISPLAY_COLUMNS = [
    "critic_update",
    "generator_update",
    "reward_update",
    "critic_forward",
    "generator_forward",
    "reward_forward",
    "predictor_forward",
    "others",
    "total",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build the computational-time LaTeX table from timing files saved "
            "inside each result/model/seed folder."
        )
    )
    ap.add_argument("--config", default="timing_table_config.txt")
    ap.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    ap.add_argument("--epoch", type=int, default=1000)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=0)
    ap.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Explicit seeds; overrides --seed-start/--seed-end.",
    )
    ap.add_argument(
        "--output-prefix",
        default="computational_time_table",
    )
    ap.add_argument(
        "--show-sd",
        action="store_true",
        help=(
            "If multiple seeds are selected, display mean +/- SD across seeds. "
            "Default matches the supplied table style and shows the mean only."
        ),
    )
    ap.add_argument(
        "--caption",
        default=(
            "Comparison of computational time of different peptide-generation "
            "algorithms. Times are reported in minutes."
        ),
    )
    ap.add_argument(
        "--label",
        default="bladder-imm-time-table",
    )
    return ap.parse_args()


def parse_config(path: Path):
    """
    Config format:
      enabled | displayed name | result-folder pattern

    Example:
      1 | WGAN-GP | WGAN-GP_FixPad_seed{seed}
    """
    rows = []
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 3:
            raise ValueError(
                f"{path}:{line_no}: expected "
                "enabled | displayed name | result-folder pattern"
            )

        enabled = parts[0].lower() in {
            "1", "true", "yes", "y", "on"
        }
        if not enabled:
            continue

        rows.append({
            "display_name": parts[1],
            "folder": parts[2],
        })

    if not rows:
        raise ValueError(f"No enabled algorithms found in {path}")
    return rows


def resolve_folder(pattern: str, seed: int) -> str:
    if "{seed}" in pattern:
        return pattern.format(seed=seed)
    if re.search(r"_seed\d+$", pattern):
        return re.sub(r"_seed\d+$", f"_seed{seed}", pattern)
    return f"{pattern}_seed{seed}"


def candidate_epoch_dirs(seed_root: Path, epoch: int):
    dirs = [
        seed_root / f"epoch{epoch}",
        seed_root / f"step{epoch}",
        seed_root,
    ]
    return [p for p in dirs if p.exists()]


def find_named_file(seed_root: Path, epoch: int, filename: str) -> Optional[Path]:
    # Normal/direct locations.
    for d in candidate_epoch_dirs(seed_root, epoch):
        p = d / filename
        if p.is_file():
            return p

    # Support historical concatenated naming such as epoch1000RunRime.txt.
    for prefix in [f"epoch{epoch}", f"step{epoch}"]:
        p = seed_root / f"{prefix}{filename}"
        if p.is_file():
            return p

    # Conservative recursive fallback.
    if seed_root.exists():
        matches = sorted(
            [p for p in seed_root.rglob(filename) if p.is_file()],
            key=lambda p: (
                0 if f"epoch{epoch}" in p.parts else 1,
                0 if f"step{epoch}" in p.parts else 1,
                len(p.parts),
                str(p),
            ),
        )
        if matches:
            return matches[0]

    return None


def find_component_file(
    seed_root: Path,
    epoch: int,
    filenames,
) -> Optional[Path]:
    """Return the first existing compatible timing file by priority."""
    for filename in filenames:
        path = find_named_file(seed_root, epoch, filename)
        if path is not None:
            return path
    return None


def load_component_seconds(path: Path) -> float:
    """
    Timing arrays contain accumulated seconds per epoch. This includes both
    *_sum.npy and historical *_each.npy files whose entries are already
    epoch-level sums. Summing the array gives total time across epochs.
    """
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    vals = arr.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(vals.sum())


def load_runtime_npy_seconds(path: Path) -> float:
    """
    Recover total training time from runtime.npy.

    Clean files contain one runtime per epoch and are summed.

    Legacy GAN/WGAN training code accidentally saved an interleaved array:
        reward loss, epoch runtime, reward loss, epoch runtime, ...
    This is detected when runtime.npy has exactly twice the number of entries
    in a saved component timing array. In that case only arr[1::2] is summed.
    """
    arr = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(arr).any():
        return float("nan")

    component_files = [
        "Dupdate_time_sum.npy",
        "Dupdate_time_each.npy",
        "Gupdate_time_sum.npy",
        "Gupdate_time_each.npy",
        "Ggen_time_sum.npy",
        "Ggen_time_each.npy",
        "Dpred_time_sum.npy",
        "Dpred_time_each.npy",
        "CNNPred_time_sum.npy",
        "CNNPred_time_each.npy",
        "Supdate_time_sum.npy",
        "Supdate_time_each.npy",
        "Sgen_time_sum.npy",
        "Sgen_time_each.npy",
    ]

    expected_epochs = None
    for filename in component_files:
        comp_path = path.parent / filename
        if not comp_path.is_file():
            continue
        try:
            comp = np.asarray(
                np.load(comp_path, allow_pickle=False)
            ).reshape(-1)
            if comp.size:
                expected_epochs = int(comp.size)
                break
        except Exception:
            continue

    if (
        expected_epochs is not None
        and arr.size == 2 * expected_epochs
    ):
        values = arr[1::2]
        values = values[np.isfinite(values)]
        seconds = float(values.sum())
        print(
            f"           NOTE: detected legacy GAN/WGAN interleaved "
            f"runtime.npy; summing runtime[1::2] = {seconds:.3f} s."
        )
        return seconds

    values = arr[np.isfinite(arr)]
    seconds = float(values.sum())
    print(
        f"           NOTE: clean per-epoch runtime.npy; "
        f"summing {values.size} entries = {seconds:.3f} s."
    )
    return seconds


def parse_runtime_text(path: Path):
    """
    Return (seconds, kind).

    Priority:
      1. Total Training Time
      2. Training Time
      3. Program Runtime

    D3PM's 'Final 10000 Sampling Time' is deliberately NOT included when
    Training Time is available.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    patterns = [
        (
            "training",
            r"Total\s+Training\s+(?:Time|Runtime|RumTime)\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?",
        ),
        (
            "training",
            r"Training\s+(?:Time|Runtime|RumTime)\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?",
        ),
        (
            "program",
            r"Program\s+(?:Runtime|RumTime)\s*:\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?",
        ),
    ]

    for kind, pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1]), kind

    raise ValueError(
        f"Could not find Training Time or Program Runtime in {path}"
    )


def find_total_runtime(seed_root: Path, epoch: int):
    """
    Preferred source order:
      1. RunTime.txt
      2. RunRime.txt   (historical typo)
      3. runtime.npy

    Returns:
      seconds, kind, source_path
    where kind is 'training', 'program', or 'runtime_npy'.
    """
    for name in ["RunTime.txt", "RunRime.txt"]:
        path = find_named_file(seed_root, epoch, name)
        if path is None:
            continue
        try:
            seconds, kind = parse_runtime_text(path)
            return seconds, kind, path
        except ValueError:
            pass

    path = find_named_file(seed_root, epoch, "runtime.npy")
    if path is not None:
        return (
            load_runtime_npy_seconds(path),
            "runtime_npy",
            path,
        )

    return float("nan"), "missing", None


def read_one_seed(seed_root: Path, epoch: int):
    result = {}

    total_seconds, total_kind, total_source = find_total_runtime(
        seed_root, epoch
    )
    result["total_seconds"] = total_seconds
    result["total_kind"] = total_kind
    result["total_source"] = (
        str(total_source) if total_source else ""
    )

    component_seconds = {}
    for key, filenames in COMPONENT_FILES.items():
        path = find_component_file(seed_root, epoch, filenames)
        if path is None:
            component_seconds[key] = float("nan")
            result[f"{key}_source"] = ""
            print(
                f"           NOTE: no timing file found for {key}; "
                f"checked {filenames}"
            )
            continue

        component_seconds[key] = load_component_seconds(path)
        result[f"{key}_source"] = str(path)
        print(
            f"           {key}: {path.name} "
            f"({component_seconds[key]:.3f} s)"
        )

    # Convert component timers to minutes.
    for key, seconds in component_seconds.items():
        result[key] = (
            seconds / 60.0
            if math.isfinite(seconds)
            else float("nan")
        )

    result["total"] = (
        total_seconds / 60.0
        if math.isfinite(total_seconds)
        else float("nan")
    )

    # "Others" is only meaningful if component timing exists. For models
    # such as D3PM/PepINVENT with no component timer files, leave it blank.
    known = [
        x for x in component_seconds.values()
        if math.isfinite(x)
    ]

    if math.isfinite(total_seconds) and known:
        other_seconds = total_seconds - sum(known)

        # Small negative values can arise from timer/rounding noise.
        if other_seconds >= -1e-6:
            result["others"] = max(0.0, other_seconds) / 60.0
        else:
            print(
                f"           WARNING: component timers exceed total runtime "
                f"by {-other_seconds:.3f} s in {seed_root}; "
                "leaving Others blank."
            )
            result["others"] = float("nan")
    else:
        result["others"] = float("nan")

    return result


def mean_sd(values):
    x = pd.to_numeric(
        pd.Series(values), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return float("nan"), float("nan"), 0

    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return mean, sd, len(x)


def format_value(mean, sd, n, show_sd=False, program_runtime=False):
    if math.isnan(mean):
        return "-"

    value = f"{mean:.2f}"

    if show_sd and n > 1 and not math.isnan(sd):
        value += rf"$\pm${sd:.2f}"

    if program_runtime:
        value += r"\textsuperscript{\dag}"

    return value


def latex_table(summary_rows, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\begin{center}",
        r"\begin{tabular}{llll|llll|l|l}",
        (
            r"\multirow{2}{*}{\bf Algorithm}  "
            r"&\multicolumn{3}{c}{\bf Weight update time (min.)} "
            r"&\multicolumn{4}{c}{\bf Forward pass time (min.)} "
            r"&\multicolumn{1}{c}{\bf \makecell{Others\\(min.)}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Total\\training\\time (min.)}} \\"
        ),
        r"\cline{2-8}",
        (
            r"&\multicolumn{1}{c}{\bf \makecell{Critic}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Generator}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Reward}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Critic}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Generator}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Reward}} "
            r"&\multicolumn{1}{c}{\bf \makecell{Predictor}}&& \\ \hline"
        ),
    ]

    any_program_runtime = False

    for row in summary_rows:
        any_program_runtime |= row["uses_program_runtime"]

        cells = [
            row["Algorithm"],
            row["critic_update_fmt"],
            row["generator_update_fmt"],
            row["reward_update_fmt"],
            row["critic_forward_fmt"],
            row["generator_forward_fmt"],
            row["reward_forward_fmt"],
            row["predictor_forward_fmt"],
            row["others_fmt"],
            row["total_fmt"],
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{tabular}")

    caption_text = caption
    if any_program_runtime:
        caption_text += (
            r" \textsuperscript{\dag}For PepINVENT (or any row for which only "
            r"Program Runtime is available), the reported total corresponds to "
            r"the complete program runtime rather than a separately recorded "
            r"training-only time."
        )

    lines.extend([
        rf"\caption{{{caption_text}}}",
        rf"\label{{{label}}}",
        r"\end{center}",
        r"\end{table}",
        "",
    ])

    return "\n".join(lines)


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()

    algorithms = parse_config(config_path)

    if args.seeds:
        seeds = args.seeds
    else:
        if args.seed_end < args.seed_start:
            raise SystemExit("--seed-end must be >= --seed-start")
        seeds = list(range(args.seed_start, args.seed_end + 1))

    per_seed_rows = []
    summary_rows = []

    print(f"Config:      {config_path}")
    print(f"Result root: {result_root}")
    print(f"Epoch:       {args.epoch}")
    print(f"Seeds:       {seeds}")
    print()

    for alg in algorithms:
        print("=" * 90)
        print(f"Algorithm: {alg['display_name']}")
        print("=" * 90)

        alg_rows = []

        for seed in seeds:
            folder_name = resolve_folder(alg["folder"], seed)
            seed_root = result_root / folder_name

            print(f"[seed {seed}] {seed_root}")

            if not seed_root.exists():
                print("           WARNING: folder not found")
                row = {
                    "Algorithm": alg["display_name"],
                    "folder": folder_name,
                    "seed": seed,
                    **{k: float("nan") for k in DISPLAY_COLUMNS},
                    "total_kind": "missing",
                    "total_source": "",
                }
            else:
                timing = read_one_seed(seed_root, args.epoch)
                row = {
                    "Algorithm": alg["display_name"],
                    "folder": folder_name,
                    "seed": seed,
                    **timing,
                }

                print(
                    f"           total={row['total']:.2f} min "
                    f"kind={row['total_kind']} "
                    f"source={row['total_source']}"
                    if math.isfinite(row["total"])
                    else "           total=NOT FOUND"
                )

            per_seed_rows.append(row)
            alg_rows.append(row)

        out = {
            "Algorithm": alg["display_name"],
        }

        # If any selected seed uses Program Runtime, mark the total value.
        uses_program_runtime = any(
            r.get("total_kind") == "program"
            for r in alg_rows
        )
        out["uses_program_runtime"] = uses_program_runtime

        for col in DISPLAY_COLUMNS:
            mean, sd, n = mean_sd(
                [r.get(col, float("nan")) for r in alg_rows]
            )
            out[f"{col}_mean"] = mean
            out[f"{col}_sd"] = sd
            out[f"{col}_n"] = n
            out[f"{col}_fmt"] = format_value(
                mean,
                sd,
                n,
                show_sd=args.show_sd,
                program_runtime=(
                    col == "total" and uses_program_runtime
                ),
            )

        summary_rows.append(out)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = Path(
        f"{args.output_prefix}_per_seed.csv"
    )
    per_seed_df.to_csv(per_seed_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(
        f"{args.output_prefix}_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    tex = latex_table(
        summary_rows,
        args.caption,
        args.label,
    )
    tex_path = Path(f"{args.output_prefix}.tex")
    tex_path.write_text(tex, encoding="utf-8")

    print()
    print("=" * 90)
    print("FINISHED")
    print("=" * 90)
    print(f"Per-seed CSV: {per_seed_path.resolve()}")
    print(f"Summary CSV:  {summary_path.resolve()}")
    print(f"LaTeX table:  {tex_path.resolve()}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
