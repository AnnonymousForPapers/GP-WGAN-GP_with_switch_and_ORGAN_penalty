#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "result"
DEFAULT_OUT_ROOT = REPO_ROOT / "code_R2_FixPad_seeds"


METRICS = {
    'deepimmuno': {
        'header': r'\makecell{Average\\DeepImmuno\\imm. score}',
        'kind': 'prediction_mean',
        'columns': ['DeepImmuno_score', 'deepimmuno_score'],
        'decimals': 2,
    },
    'iedb_immunogenicity': {
        'header': r'\makecell{Average\\IEDB\\imm. score}',
        'kind': 'prediction_mean',
        'columns': ['iedb_immunogenicity_score', 'IEDB_immunogenicity_score', 'IEDB_score'],
        'decimals': 2,
    },
    'netmhcpan40_affinity': {
        'header': r'\makecell{Average\\NetMHCpan 4.0\\affinity (nM)}',
        'kind': 'prediction_mean',
        'columns': ['NetMHCpan_4.0_Aff_nM', 'NetMHCpan_4_0_Aff_nM', 'netmhcpan_4.0_aff_nm'],
        'decimals': 2,
    },
    'netmhcpan40_binder_pct': {
        'header': r'\makecell{NetMHCpan 4.0\\unique binders\\(<500 nM) (\%)}',
        'kind': 'prediction_threshold_pct',
        'columns': ['NetMHCpan_4.0_Aff_nM', 'NetMHCpan_4_0_Aff_nM', 'netmhcpan_4.0_aff_nm'],
        'threshold': 500.0,
        'comparison': 'lt',
        'decimals': 2,
    },
    'netmhcpan41_affinity': {
        'header': r'\makecell{Average\\NetMHCpan 4.1\\affinity (nM)}',
        'kind': 'prediction_mean',
        'columns': ['NetMHCpan_4.1_Aff_nM', 'NetMHCpan_4_1_Aff_nM', 'netmhcpan_4.1_aff_nm'],
        'decimals': 2,
    },
    'netmhcpan41_binder_pct': {
        'header': r'\makecell{NetMHCpan 4.1\\unique binders\\(<500 nM) (\%)}',
        'kind': 'prediction_threshold_pct',
        'columns': ['NetMHCpan_4.1_Aff_nM', 'NetMHCpan_4_1_Aff_nM', 'netmhcpan_4.1_aff_nm'],
        'threshold': 500.0,
        'comparison': 'lt',
        'decimals': 2,
    },
    'netmhcpan40_rank': {
        'header': r'\makecell{Average\\NetMHCpan 4.0\\\%Rank}',
        'kind': 'prediction_mean',
        'columns': ['NetMHCpan_4.0_Rank', 'NetMHCpan_4.0_rank', 'NetMHCpan_4_0_Rank'],
        'decimals': 2,
    },
    'pepmatch_exact_pct': {
        'header': r'\makecell{PEPMatch\\exact match (\%)}',
        'kind': 'pepmatch_mismatch_pct',
        'mismatches': 0,
        'decimals': 2,
    },
    'pepmatch_1_mismatch_pct': {
        'header': r'\makecell{PEPMatch\\1 mismatch (\%)}',
        'kind': 'pepmatch_mismatch_pct',
        'mismatches': 1,
        'decimals': 2,
    },
    'pepmatch_2_mismatch_pct': {
        'header': r'\makecell{PEPMatch\\2 mismatches (\%)}',
        'kind': 'pepmatch_mismatch_pct',
        'mismatches': 2,
        'decimals': 2,
    },
    'pepmatch_3_mismatch_pct': {
        'header': r'\makecell{PEPMatch\\3 mismatches (\%)}',
        'kind': 'pepmatch_mismatch_pct',
        'mismatches': 3,
        'decimals': 2,
    },
    'pepmatch_exact_count': {
        'header': r'\makecell{PEPMatch\\exact match\\count}',
        'kind': 'pepmatch_mismatch_count',
        'mismatches': 0,
        'decimals': 2,
    },
    'pepmatch_1_mismatch_count': {
        'header': r'\makecell{PEPMatch\\1 mismatch\\count}',
        'kind': 'pepmatch_mismatch_count',
        'mismatches': 1,
        'decimals': 2,
    },
    'pepmatch_2_mismatch_count': {
        'header': r'\makecell{PEPMatch\\2 mismatches\\count}',
        'kind': 'pepmatch_mismatch_count',
        'mismatches': 2,
        'decimals': 2,
    },
    'pepmatch_3_mismatch_count': {
        'header': r'\makecell{PEPMatch\\3 mismatches\\count}',
        'kind': 'pepmatch_mismatch_count',
        'mismatches': 3,
        'decimals': 2,
    },
    'pepsysco': {
        'header': r'\makecell{Average\\PepSySco\\score}',
        'kind': 'prediction_mean',
        'columns': ['pepsysco_score', 'PepSySco_score', 'Pepsysco Score',
                    'pepsysco_probability', 'PepSySco_probability'],
        'decimals': 2,
        'blank_if_missing': True,
    },
    'tcga_blca_match_pct': {
        'header': r'\makecell{TCGA-BLCA\\exact match (\%)}',
        'kind': 'boolean_pct',
        'columns': ['tcga_blca_exact_match', 'exact_TCGA_BLCA_match', 'tcga_exact_match'],
        'decimals': 2,
    },
    'pairwise_edit_distance': {
        'header': r'\makecell{Mean pairwise\\edit distance\\among generated peptides}',
        'kind': 'cached_pairwise_edit_distance',
        'summary_key': 'pairwise_edit_distance_mean',
        'sd_key': 'pairwise_edit_distance_sd',
        'n_key': 'pairwise_edit_distance_n_pairs',
        'decimals': 2,
    },
    'mean_edit_distance': {
        'header': r'\makecell{Mean minimum\\edit distance\\to training set}',
        'kind': 'novelty_summary',
        'summary_key': 'min_edit_distance_mean',
        'decimals': 2,
    },
    'median_edit_distance': {
        'header': r'\makecell{Median minimum\\edit distance\\to training set}',
        'kind': 'novelty_summary',
        'summary_key': 'min_edit_distance_median',
        'decimals': 2,
    },
    'nearest_neighbor_similarity': {
        'header': r'\makecell{Mean nearest-neighbor\\similarity}',
        'kind': 'novelty_summary',
        'summary_key': 'nearest_neighbor_similarity_mean',
        'decimals': 3,
    },
    'sequence_identity': {
        'header': r'\makecell{Mean sequence identity\\to nearest training\\peptide (\%)}',
        'kind': 'novelty_summary',
        'summary_key': 'sequence_identity_percent_mean',
        'decimals': 2,
    },
    'training_exact_overlap_pct': {
        'header': r'\makecell{Exact training-set\\overlap (\%)}',
        'kind': 'novelty_summary',
        'summary_key': 'exact_overlap_percent',
        'decimals': 2,
    },
    'training_non_exact_novel_pct': {
        'header': r'\makecell{Non-exact novel\\peptides (\%)}',
        'kind': 'novelty_summary',
        'summary_key': 'non_exact_novel_percent',
        'decimals': 2,
    },
    'identity_ge_90_pct': {
        'header': r'\makecell{Nearest-neighbor\\identity $\geq$90\% (\%)}',
        'kind': 'novelty_summary',
        'summary_key': 'pct_sequence_identity_ge_90',
        'decimals': 2,
    },
    'edit_distance_ge_2_pct': {
        'header': r'\makecell{Minimum edit\\distance $\geq$2 (\%)}',
        'kind': 'novelty_summary',
        'summary_key': 'pct_min_edit_distance_ge_2',
        'decimals': 2,
    },
    'novelty_score': {
        'header': r'\makecell{Mean novelty\\score}',
        'kind': 'novelty_summary',
        'summary_key': 'novelty_score_mean',
        'decimals': 3,
    },
    'valid_9_10_pct': {
        'header': r'\makecell{Percent. of\\peptides with\\9--10-mer (\%)}',
        'kind': 'valid_pct',
        'decimals': 2,
    },
    'non_repeated_pct': {
        'header': r'\makecell{Percent. of\\non-repeated\\9--10-mer peptides (\%)}',
        'kind': 'unique_pct',
        'decimals': 2,
    },
    'training_time_min': {
        'header': r'\makecell{Total\\training\\time (min.)}',
        'kind': 'training_time',
        'decimals': 2,
    },
}

DEFAULT_COLUMNS = [
    'iedb_immunogenicity',
    'netmhcpan41_binder_pct',
    'pepmatch_exact_count',
    'pepmatch_1_mismatch_count',
    'pepmatch_2_mismatch_count',
    'pepmatch_3_mismatch_count',
    'pepsysco',
    'tcga_blca_match_pct',
    'mean_edit_distance',
    'nearest_neighbor_similarity',
    'sequence_identity',
    'training_exact_overlap_pct',
]


def parse_args():
    ap = argparse.ArgumentParser(
        description='Aggregate unified peptide-evaluation outputs across seeds and generate a LaTeX table.'
    )
    ap.add_argument('--config', default='table_config.txt')
    ap.add_argument('--result-root', default=str(DEFAULT_RESULT_ROOT))
    ap.add_argument('--out-root', default=str(DEFAULT_OUT_ROOT))
    ap.add_argument(
        '--aggregation',
        choices=['seed', 'sample'],
        default='seed',
        help=(
            'Aggregation mode. seed (default) preserves the existing behavior: '
            'compute one metric per seed and report mean +/- SD across seeds. '
            'sample uses one selected seed per model and reports mean +/- SD '
            'across peptide-level samples within that seed.'
        ),
    )
    ap.add_argument(
        '--sample-seed',
        type=int,
        default=0,
        help=(
            'Fallback seed used in --aggregation sample mode. A model-specific '
            'sample seed may instead be supplied as the sixth field in the '
            'config line.'
        ),
    )
    ap.add_argument(
        '--sample-count-std',
        action='store_true',
        help=(
            'In --aggregation sample mode, also report an estimated SD for '
            'PEPMatch count metrics using sqrt(n*p*(1-p)). By default, one-seed '
            'count metrics are reported as integer counts without +/- SD.'
        ),
    )
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--seed-end', type=int, default=50)
    ap.add_argument('--seeds', nargs='*', type=int, default=None,
                    help='Optional explicit seeds. If supplied, overrides --seed-start/--seed-end.')
    ap.add_argument('--epoch', type=int, default=1000,
                    help='Used when looking for epoch1000/step1000 beneath each seed folder.')
    ap.add_argument('--columns', nargs='+', default=None,
                    help='Override COLUMNS=... in the config file.')
    ap.add_argument('--output-prefix', default='evaluation_table_predictors')
    ap.add_argument('--caption', default=(
        'Comparison of peptide-generation algorithms. Values are mean $\\pm$ standard deviation '
        'across the selected random seeds.'
    ))
    ap.add_argument('--label', default='tab:peptide_generation_comparison')
    ap.add_argument('--strict', action='store_true',
                    help='Stop on a missing evaluation file/metric/training-time file instead of recording NaN.')
    return ap.parse_args()


def parse_config(path: Path) -> Tuple[List[Dict[str, str]], Optional[List[str]]]:
    algorithms = []
    config_columns = None

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        if '=' in line and '|' not in line:
            key, value = [x.strip() for x in line.split('=', 1)]
            if key.upper() == 'COLUMNS':
                config_columns = [x.strip() for x in value.split(',') if x.strip()]
            continue

        parts = [x.strip() for x in line.split('|')]
        if len(parts) < 3:
            raise ValueError(
                f'{path}:{line_no}: expected at least 3 pipe-separated fields: '
                'enabled | display_name | folder_name_or_template'
            )

        enabled = parts[0].lower() in {'1', 'true', 'yes', 'y', 'on'}
        if not enabled:
            continue

        sample_seed = None
        if len(parts) >= 6 and parts[5]:
            try:
                sample_seed = int(parts[5])
            except ValueError as e:
                raise ValueError(
                    f'{path}:{line_no}: sixth field (sample seed) must be an integer; '
                    f'got {parts[5]!r}'
                ) from e

        algorithms.append({
            'display_name': parts[1],
            'folder': parts[2],
            'checkpoint': parts[3].lower() if len(parts) >= 4 and parts[3] else 'best',
            'out_glob': parts[4] if len(parts) >= 5 else '',
            'sample_seed': sample_seed,
        })

    if not algorithms:
        raise ValueError(f'No enabled algorithms found in {path}')
    return algorithms, config_columns


def resolve_seed_folder(folder_template: str, seed: int) -> str:
    if '{seed}' in folder_template:
        return folder_template.format(seed=seed)
    if re.search(r'_seed\d+$', folder_template):
        return re.sub(r'_seed\d+$', f'_seed{seed}', folder_template)
    return f'{folder_template}_seed{seed}'


def resolve_seed_root(
    result_root: Path,
    folder_template: str,
    seed: int,
) -> Tuple[str, Path]:
    """
    Resolve either a normal seeded model folder or a fixed dataset folder.

    If the config folder has no explicit seed marker and the literal folder
    already exists under result_root, use it directly. Otherwise preserve
    the historical _seedN behavior.
    """
    if '{seed}' in folder_template or re.search(r'_seed\d+$', folder_template):
        folder_name = resolve_seed_folder(folder_template, seed)
        return folder_name, result_root / folder_name

    literal = result_root / folder_template
    if literal.is_dir():
        return folder_template, literal

    folder_name = resolve_seed_folder(folder_template, seed)
    return folder_name, result_root / folder_name


def find_eval_dir(seed_root: Path, checkpoint: str, seed: int, epoch: int) -> Optional[Path]:
    checkpoint = checkpoint.lower()
    if checkpoint not in {'best', 'last'}:
        # Also allow an exact checkpoint stem such as model_epoch500.
        checkpoint_stem = checkpoint
        if checkpoint_stem.endswith('.pth') or checkpoint_stem.endswith('.pt') or checkpoint_stem.endswith('.chkpt'):
            checkpoint_stem = Path(checkpoint_stem).stem
    else:
        checkpoint_stem = f'model_{checkpoint}'

    eval_name = f'evaluation_{checkpoint_stem}_seed{seed}'
    direct_candidates = [
        seed_root / f'epoch{epoch}' / eval_name,
        seed_root / f'step{epoch}' / eval_name,
        seed_root / eval_name,
    ]
    for p in direct_candidates:
        if p.is_dir():
            return p

    matches = sorted([p for p in seed_root.rglob(eval_name) if p.is_dir()]) if seed_root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer epoch/step requested by the user, then shortest path.
        preferred = [p for p in matches if f'epoch{epoch}' in p.parts or f'step{epoch}' in p.parts]
        if preferred:
            return sorted(preferred, key=lambda x: (len(x.parts), str(x)))[0]
        return sorted(matches, key=lambda x: (len(x.parts), str(x)))[0]

    # Dataset-style fallback, e.g.
    #   result/bladder_dataset/evaluation_Bladder.4.0_test_mut
    if seed_root.exists():
        dataset_candidates = sorted(
            p for p in seed_root.rglob('evaluation_*')
            if p.is_dir()
        )

        if len(dataset_candidates) == 1:
            print(
                f'           NOTE: using dataset-style evaluation directory: '
                f'{dataset_candidates[0]}'
            )
            return dataset_candidates[0]

        if len(dataset_candidates) > 1:
            direct = [p for p in dataset_candidates if p.parent == seed_root]
            if len(direct) == 1:
                print(
                    f'           NOTE: using dataset-style evaluation directory: '
                    f'{direct[0]}'
                )
                return direct[0]

            preferred = [
                p for p in dataset_candidates
                if f'epoch{epoch}' in p.parts or f'step{epoch}' in p.parts
            ]
            if len(preferred) == 1:
                print(
                    f'           NOTE: using dataset-style evaluation directory: '
                    f'{preferred[0]}'
                )
                return preferred[0]

            print(
                f'           WARNING: found multiple dataset-style evaluation '
                f'directories under {seed_root}; refusing to guess: '
                f'{[str(p) for p in dataset_candidates]}'
            )

    return None


def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r'seed\{?\d*\}?', '', s)
    s = re.sub(r'\bs\{?\d+\}?\b', '', s)
    s = s.replace('fixpad', '')
    return re.sub(r'[^a-z0-9]+', '', s)


def seed_matches_filename(name: str, seed: int) -> bool:
    low = name.lower()
    patterns = [
        rf'(?:^|[_-])seed{seed}(?:[_\-.]|$)',
        rf'(?:^|[_-])s{seed}(?:[_\-.]|$)',
    ]
    return any(re.search(p, low) for p in patterns)


def find_out_file(out_root: Path, out_glob: str, folder_template: str, seed: int) -> Optional[Path]:
    if out_glob:
        pattern = out_glob.format(seed=seed)
        p = Path(pattern)
        if p.is_absolute():
            matches = [Path(x) for x in glob.glob(pattern)]
        else:
            matches = [Path(x) for x in glob.glob(str(out_root / pattern))]
        matches = sorted([x for x in matches if x.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    if not out_root.exists():
        return None

    all_out = [p for p in out_root.rglob('*.out') if p.is_file() and seed_matches_filename(p.name, seed)]
    if not all_out:
        return None

    target = normalize_name(folder_template)
    scored = []
    for p in all_out:
        cand = normalize_name(p.name)
        # Longest common-prefix ratio plus substring bonus.
        common = 0
        for a, b in zip(target, cand):
            if a != b:
                break
            common += 1
        prefix_score = common / max(1, len(target))
        substring_bonus = 1.0 if target and (target in cand or cand in target) else 0.0
        scored.append((substring_bonus + prefix_score, p.stat().st_mtime, p))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def parse_training_time_minutes(path: Path) -> float:
    text = path.read_text(errors='replace')

    # Preferred line from the supplied training .out file:
    # Training Time: 11415.961209499277(s)
    matches = re.findall(r'Training\s+Time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?',
                         text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1]) / 60.0

    # Conservative fallbacks for other training scripts.
    fallback_patterns = [
        r'Total\s+Training\s+Time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?',
        r'Training\s+time\s*\(s\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
    ]
    for pattern in fallback_patterns:
        m = re.findall(pattern, text, flags=re.IGNORECASE)
        if m:
            return float(m[-1]) / 60.0

    raise ValueError(f'Could not find a final Training Time in {path}')



def _parse_runtime_text_with_kind(path: Path):
    """
    Parse a result-folder runtime text file.

    Returns:
        (minutes, kind)

    kind:
        'training' -> Training Time / Total Training Time
        'program'  -> Program Runtime

    Priority is training-only time first. This means D3PM's
    'Final 10000 Sampling Time' is not added to Training Time.
    """
    raw = path.read_text(encoding='utf-8', errors='replace')

    patterns = [
        (
            'training',
            r'Total\s+Training\s+(?:Time|Runtime|RumTime)\s*:\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?',
        ),
        (
            'training',
            r'Training\s+(?:Time|Runtime|RumTime)\s*:\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?',
        ),
        (
            'program',
            r'Program\s+(?:Runtime|RumTime)\s*:\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*\(?s(?:ec(?:onds?)?)?\)?',
        ),
    ]

    for kind, pattern in patterns:
        matches = re.findall(pattern, raw, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1]) / 60.0, kind

    raise ValueError(
        f'Could not find Training Time, Total Training Time, '
        f'or Program Runtime in {path}'
    )


def _runtime_candidate_paths(seed_root: Path, epoch: int):
    """
    Candidate result-folder runtime text files.

    Supports both the corrected RunTime.txt name and the historical
    RunRime.txt typo.
    """
    epoch_dir = seed_root / f'epoch{epoch}'
    step_dir = seed_root / f'step{epoch}'

    candidates = [
        epoch_dir / 'RunTime.txt',
        epoch_dir / 'RunRime.txt',
        step_dir / 'RunTime.txt',
        step_dir / 'RunRime.txt',
        seed_root / f'epoch{epoch}RunTime.txt',
        seed_root / f'epoch{epoch}RunRime.txt',
        seed_root / f'step{epoch}RunTime.txt',
        seed_root / f'step{epoch}RunRime.txt',
        seed_root / 'RunTime.txt',
        seed_root / 'RunRime.txt',
    ]

    # Deduplicate without changing priority.
    seen = set()
    out = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _runtime_npy_candidate_paths(seed_root: Path, epoch: int):
    return [
        seed_root / f'epoch{epoch}' / 'runtime.npy',
        seed_root / f'step{epoch}' / 'runtime.npy',
        seed_root / 'runtime.npy',
    ]


def _read_runtime_npy_minutes(path: Path) -> float:
    """
    Recover total training time from runtime.npy.

    Two formats are supported:

    1. Clean per-epoch runtime array
       Example: D3PM's array_runtime.
       -> total training time = sum(all finite entries)

    2. Legacy GAN/WGAN array8 bug
       In the GAN/WGAN training scripts, array8 receives BOTH:
           array8.append(np.mean(S_losses))
           array8.append(stop_epoch - start_epoch)
       and is then saved as runtime.npy.

       We detect this format by comparing runtime.npy length with one of the
       component timing arrays (Dupdate_time_sum.npy, Gupdate_time_sum.npy,
       etc.). If runtime.npy has exactly twice as many entries, then:
           runtime[0::2] = reward-loss values
           runtime[1::2] = epoch runtimes
       and only runtime[1::2] is summed.

    Values are stored in seconds and converted to minutes here.
    """
    arr = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=float,
    ).reshape(-1)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValueError(f'No finite runtime values in {path}')

    # Detect the legacy GAN/WGAN interleaved array8 format from the actual
    # saved component-array lengths rather than from the folder name.
    component_files = [
        'Dupdate_time_sum.npy',
        'Gupdate_time_sum.npy',
        'Ggen_time_sum.npy',
        'Dpred_time_sum.npy',
        'CNNPred_time_sum.npy',
        'Supdate_time_sum.npy',
        'Sgen_time_sum.npy',
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
            if comp.size > 0:
                expected_epochs = int(comp.size)
                break
        except Exception:
            continue

    if (
        expected_epochs is not None
        and arr.size == 2 * expected_epochs
    ):
        runtime_values = arr[1::2]
        runtime_values = runtime_values[np.isfinite(runtime_values)]

        if runtime_values.size == 0:
            raise ValueError(
                f'Legacy interleaved runtime.npy detected but no finite '
                f'epoch runtime entries were found in {path}'
            )

        seconds = float(runtime_values.sum())

        print(
            f'           NOTE: detected legacy GAN/WGAN interleaved '
            f'runtime.npy ({arr.size} entries for {expected_epochs} epochs); '
            f'summing runtime[1::2] = {seconds:.3f} s.'
        )
        return seconds / 60.0

    # Clean format: runtime.npy contains one runtime value per epoch.
    clean = arr[np.isfinite(arr)]
    seconds = float(clean.sum())

    print(
        f'           NOTE: treating {path} as a clean per-epoch runtime array; '
        f'summing {clean.size} values = {seconds:.3f} s.'
    )

    return seconds / 60.0


def result_training_time_minutes(
    seed_root: Path,
    epoch: int,
    out_file: Optional[Path] = None,
):
    """
    Return:
        minutes, source_kind, source_path

    Source priority:
      1. result-folder RunTime.txt
      2. concatenated epoch1000RunTime.txt form
      3. runtime.npy
      4. legacy matching SLURM .out fallback

    source_kind:
      'training'
      'program'
      'runtime_npy'
      'slurm_out'
    """
    for path in _runtime_candidate_paths(seed_root, epoch):
        if not path.is_file():
            continue
        try:
            minutes, kind = _parse_runtime_text_with_kind(path)
            return float(minutes), kind, path
        except ValueError:
            continue

    for path in _runtime_npy_candidate_paths(seed_root, epoch):
        if path.is_file():
            return (
                _read_runtime_npy_minutes(path),
                'runtime_npy',
                path,
            )

    if out_file is not None and Path(out_file).is_file():
        return (
            parse_training_time_minutes(Path(out_file)),
            'slurm_out',
            Path(out_file),
        )

    raise FileNotFoundError(
        f'No result-folder RunTime.txt/RunRime.txt/runtime.npy and '
        f'no usable SLURM .out file for {seed_root}'
    )


def normalize_colname(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def find_prediction_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    exact = {str(c): c for c in df.columns}
    for c in candidates:
        if c in exact:
            return exact[c]

    normalized = {normalize_colname(c): c for c in df.columns}
    for c in candidates:
        key = normalize_colname(c)
        if key in normalized:
            return normalized[key]
    return None


def numeric_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float('nan')


def bool_series(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to True/False/NaN."""
    def convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer, float, np.floating)) and not pd.isna(x):
            if float(x) == 1.0:
                return True
            if float(x) == 0.0:
                return False
        s = str(x).strip().lower()
        if s in {'true', 't', 'yes', 'y', '1'}:
            return True
        if s in {'false', 'f', 'no', 'n', '0'}:
            return False
        return np.nan
    return series.map(convert)


def pepmatch_exact_mismatch_count(df: pd.DataFrame, mismatch_count: int) -> float:
    """
    Count unique evaluated 9-10-mer peptides whose best Human PEPMatch
    result has exactly 0, 1, 2, or 3 sequence mismatches.
    """
    if len(df) == 0:
        return float('nan')

    mismatch_col = find_prediction_column(
        df,
        ['pepmatch_mismatches', 'PEPMatch_mismatches',
         'pepmatch_mismatch_count', 'mismatches']
    )

    if mismatch_col is not None:
        vals = pd.to_numeric(df[mismatch_col], errors='coerce')
        return float((vals == mismatch_count).sum())

    # Fallback for outputs that contain only cumulative boolean flags.
    current_candidates = {
        0: ['pepmatch_exact_match'],
        1: ['pepmatch_within_1_mismatch'],
        2: ['pepmatch_within_2_mismatches'],
        3: ['pepmatch_within_3_mismatches'],
    }
    current_col = find_prediction_column(df, current_candidates[mismatch_count])
    if current_col is None:
        raise KeyError(
            'Could not find pepmatch_mismatches or compatible PEPMatch flags. '
            f'Available columns: {list(df.columns)}'
        )

    current = bool_series(df[current_col])

    if mismatch_count == 0:
        return float((current == True).sum())

    previous_candidates = {
        1: ['pepmatch_exact_match'],
        2: ['pepmatch_within_1_mismatch'],
        3: ['pepmatch_within_2_mismatches'],
    }
    previous_col = find_prediction_column(df, previous_candidates[mismatch_count])
    if previous_col is None:
        raise KeyError(
            f'Found cumulative flag {current_col}, but not the preceding cumulative flag. '
            f'Available columns: {list(df.columns)}'
        )

    previous = bool_series(df[previous_col])
    exact_k = (current == True) & ~(previous == True)
    return float(exact_k.sum())


def pepmatch_exact_mismatch_pct(df: pd.DataFrame, mismatch_count: int) -> float:
    """
    Percentage of ALL unique evaluated 9-10-mer peptides whose best Human
    PEPMatch hit has exactly 0, 1, 2, or 3 sequence mismatches.
    """
    n_total = len(df)
    if n_total == 0:
        return float('nan')

    mismatch_col = find_prediction_column(
        df,
        ['pepmatch_mismatches', 'PEPMatch_mismatches',
         'pepmatch_mismatch_count', 'mismatches']
    )

    if mismatch_col is not None:
        vals = pd.to_numeric(df[mismatch_col], errors='coerce')
        return 100.0 * float((vals == mismatch_count).sum()) / n_total

    # Fallback for outputs that contain only cumulative flags.
    current_candidates = {
        0: ['pepmatch_exact_match'],
        1: ['pepmatch_within_1_mismatch'],
        2: ['pepmatch_within_2_mismatches'],
        3: ['pepmatch_within_3_mismatches'],
    }
    current_col = find_prediction_column(df, current_candidates[mismatch_count])
    if current_col is None:
        raise KeyError(
            'Could not find pepmatch_mismatches or compatible PEPMatch flags. '
            f'Available columns: {list(df.columns)}'
        )

    current = bool_series(df[current_col])

    if mismatch_count == 0:
        return 100.0 * float((current == True).sum()) / n_total

    previous_candidates = {
        1: ['pepmatch_exact_match'],
        2: ['pepmatch_within_1_mismatch'],
        3: ['pepmatch_within_2_mismatches'],
    }
    previous_col = find_prediction_column(df, previous_candidates[mismatch_count])
    if previous_col is None:
        raise KeyError(
            f'Found cumulative flag {current_col}, but not the preceding cumulative flag. '
            f'Available columns: {list(df.columns)}'
        )

    previous = bool_series(df[previous_col])
    exact_k = (current == True) & ~(previous == True)
    return 100.0 * float(exact_k.sum()) / n_total


def pairwise_mean_edit_distance(eval_dir: Path) -> float:
    """
    Exact low-RAM version of the calculation used in 7_Edit_distance.py.

    For peptide i, compare it with peptides i..N-1. Therefore:
      * each unordered pair is counted once;
      * each peptide is compared with itself once (distance 0).

    RAM-saving changes:
      * read only the peptide column with csv.DictReader instead of pandas;
      * do not create peptides[i:] list slices inside the O(N^2) loop;
      * accumulate only one running integer sum (no distance list/matrix).
    """
    path = eval_dir / 'generated_filtered_unique.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    # Read only the one column needed for this metric.
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        pep_col = None
        normalized_fields = {normalize_colname(c): c for c in fieldnames}
        for candidate in ['peptide', 'generated_peptide', 'sequence']:
            key = normalize_colname(candidate)
            if key in normalized_fields:
                pep_col = normalized_fields[key]
                break

        if pep_col is None:
            raise KeyError(
                f'No peptide column found in {path}. Available columns: {fieldnames}'
            )

        peptides = []
        append_peptide = peptides.append
        for row in reader:
            value = row.get(pep_col)
            if value is None:
                continue
            value = value.strip()
            if value:
                append_peptide(value)

    n = len(peptides)
    if n == 0:
        return float('nan')

    try:
        import Levenshtein as _lev
        distance = _lev.distance
    except ImportError:
        try:
            from rapidfuzz.distance import Levenshtein as _rf_lev
            distance = _rf_lev.distance
        except ImportError as e:
            raise ImportError(
                'Mean edit distance requires python-Levenshtein or rapidfuzz. '
                'Install one with: pip install python-Levenshtein'
            ) from e

    # Exactly N(N+1)/2 comparisons because self-comparisons are included.
    n_pairs = n * (n + 1) // 2
    total_distance = 0

    # IMPORTANT: use indexes instead of `for b in peptides[i:]`.
    # The latter allocates a new list slice for every i.
    for i in range(n):
        a = peptides[i]
        for j in range(i, n):
            total_distance += distance(a, peptides[j])

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f'           edit distance progress: {i + 1:,}/{n:,}')

    return float(total_distance) / n_pairs




def read_pepsysco_mean(eval_dir: Path) -> float:
    """
    Read PepSySco scores produced by the automated API post-processing workflow.

    Preferred sources, in order:
      1. <evaluation folder>/all_predictions.csv
      2. <evaluation folder>/pepsysco/PepSySco_prediction.csv
      3. Other CSV files under <evaluation folder>/pepsysco/

    Scores are restricted to generated_filtered_unique.csv. If the preferred
    API-derived source is incomplete, return NaN instead of averaging a partial
    result.
    """
    generated_path = eval_dir / 'generated_filtered_unique.csv'
    if not generated_path.exists():
        raise FileNotFoundError(generated_path)

    generated = pd.read_csv(generated_path)
    generated_peptide_col = find_prediction_column(
        generated,
        ['peptide', 'generated_peptide', 'sequence']
    )
    if generated_peptide_col is None:
        raise KeyError(
            f'No peptide column found in {generated_path}. '
            f'Available columns: {list(generated.columns)}'
        )

    wanted_list = (
        generated[generated_peptide_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    wanted_list = wanted_list[wanted_list.ne('')].drop_duplicates().tolist()
    wanted = set(wanted_list)

    if not wanted:
        return float('nan')

    def mean_from_df(df: pd.DataFrame, source_name: str) -> Optional[float]:
        peptide_col = find_prediction_column(
            df, ['peptide', 'sequence', 'Peptide', 'Sequence']
        )
        score_col = find_prediction_column(
            df, METRICS['pepsysco']['columns']
        )
        if peptide_col is None or score_col is None:
            return None

        part = pd.DataFrame({
            'peptide': df[peptide_col].astype(str).str.strip().str.upper(),
            'pepsysco_score': pd.to_numeric(df[score_col], errors='coerce'),
        })
        part = part[part['peptide'].isin(wanted)]
        part = part.drop_duplicates('peptide', keep='first')

        if part.empty:
            return None

        scored = part.loc[
            part['pepsysco_score'].notna(), 'peptide'
        ].nunique()

        if scored != len(wanted):
            print(
                f'           NOTE: PepSySco source {source_name} has '
                f'{scored:,}/{len(wanted):,} scores; '
                'average PepSySco score is left missing rather than '
                'computed from a partial set.'
            )
            return float('nan')

        return numeric_mean(part['pepsysco_score'])

    # 1. Preferred: API postprocessor merges pepsysco_score here.
    all_predictions = eval_dir / 'all_predictions.csv'
    if all_predictions.exists():
        try:
            df = pd.read_csv(all_predictions)
            value = mean_from_df(df, str(all_predictions))
            if value is not None:
                return value
        except Exception as e:
            print(
                f'           NOTE: Could not use PepSySco from '
                f'{all_predictions}: {type(e).__name__}: {e}'
            )

    pepsysco_dir = eval_dir / 'pepsysco'

    # 2. Normalized API output from run_pepsysco_all_evaluations.py.
    api_prediction = pepsysco_dir / 'PepSySco_prediction.csv'
    if api_prediction.exists():
        try:
            df = pd.read_csv(api_prediction)
            value = mean_from_df(df, str(api_prediction))
            if value is not None:
                return value
        except Exception as e:
            print(
                f'           NOTE: Could not use PepSySco from '
                f'{api_prediction}: {type(e).__name__}: {e}'
            )

    # 3. Backward-compatible fallback for older/manual PepSySco CSVs.
    if pepsysco_dir.is_dir():
        csv_files = sorted(
            p for p in pepsysco_dir.glob('*.csv')
            if p.is_file() and p.name != 'PepSySco_prediction.csv'
        )
        frames = []

        for path in csv_files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            peptide_col = find_prediction_column(
                df, ['peptide', 'sequence', 'Peptide', 'Sequence']
            )
            score_col = find_prediction_column(
                df, METRICS['pepsysco']['columns']
            )
            if peptide_col is None or score_col is None:
                continue

            part = pd.DataFrame({
                'peptide': df[peptide_col].astype(str).str.strip().str.upper(),
                'pepsysco_score': pd.to_numeric(df[score_col], errors='coerce'),
            })
            part = part[
                part['peptide'].ne('') &
                part['pepsysco_score'].notna() &
                part['peptide'].isin(wanted)
            ]
            if len(part):
                frames.append(part)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates('peptide', keep='first')
            scored = combined['peptide'].nunique()
            if scored != len(wanted):
                print(
                    f'           NOTE: PepSySco fallback CSVs contain '
                    f'{scored:,}/{len(wanted):,} evaluated peptides; '
                    'average PepSySco score is left missing.'
                )
                return float('nan')

            return numeric_mean(combined['pepsysco_score'])

    raise FileNotFoundError(
        f'No usable PepSySco API result found in {all_predictions} '
        f'or under {pepsysco_dir}'
    )



def read_novelty_summary_metric(eval_dir: Path, summary_key: str) -> float:
    """
    Read a metric produced by quantify_peptide_novelty.py /
    run_novelty_all_evaluations.py.

    Preferred:
        <evaluation folder>/novelty_vs_training/novelty_summary.csv

    Fallback:
        <evaluation folder>/novelty_vs_training/novelty_summary.json

    This intentionally does not recompute edit distance or similarity here.
    The table therefore reports exactly the values produced by the dedicated
    novelty-vs-training analysis.
    """
    novelty_dir = eval_dir / 'novelty_vs_training'
    csv_path = novelty_dir / 'novelty_summary.csv'
    json_path = novelty_dir / 'novelty_summary.json'

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f'Empty novelty summary: {csv_path}')

        col = find_prediction_column(df, [summary_key])
        if col is None:
            raise KeyError(
                f'Novelty metric {summary_key!r} not found in {csv_path}. '
                f'Available columns: {list(df.columns)}'
            )

        value = pd.to_numeric(
            pd.Series([df.iloc[0][col]]),
            errors='coerce'
        ).iloc[0]

        if pd.isna(value):
            return float('nan')
        return float(value)

    if json_path.exists():
        import json
        obj = json.loads(json_path.read_text(encoding='utf-8'))

        if summary_key not in obj:
            raise KeyError(
                f'Novelty metric {summary_key!r} not found in {json_path}. '
                f'Available keys: {list(obj.keys())}'
            )

        value = pd.to_numeric(
            pd.Series([obj[summary_key]]),
            errors='coerce'
        ).iloc[0]

        if pd.isna(value):
            return float('nan')
        return float(value)

    raise FileNotFoundError(
        f'No novelty summary found. Expected {csv_path} or {json_path}. '
        'Run run_novelty_all_evaluations.py first.'
    )


def read_seed_metric(
    metric: str,
    eval_dir: Path,
    out_file: Optional[Path],
    seed_root: Optional[Path] = None,
    epoch: int = 1000,
) -> float:
    spec = METRICS[metric]
    kind = spec['kind']

    if metric == 'pepsysco':
        return read_pepsysco_mean(eval_dir)

    if kind == 'prediction_mean':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}. '
                f'Available columns: {list(df.columns)}'
            )
        return numeric_mean(df[col])

    if kind == 'prediction_threshold_pct':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}. '
                f'Available columns: {list(df.columns)}'
            )

        values = pd.to_numeric(
            df[col], errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)

        if len(values) == 0:
            return float('nan')

        missing = int(values.isna().sum())
        if missing:
            print(
                f'           NOTE: {missing}/{len(values)} affinity values are missing/non-numeric '
                f'in {path}; binder percentage is left missing rather than computed from a partial set.'
            )
            return float('nan')

        threshold = float(spec['threshold'])
        comparison = spec.get('comparison', 'lt')

        if comparison == 'lt':
            n_binders = int((values < threshold).sum())
        elif comparison == 'le':
            n_binders = int((values <= threshold).sum())
        else:
            raise ValueError(f'Unsupported comparison: {comparison}')

        return 100.0 * n_binders / len(values)

    if kind == 'pepmatch_mismatch_count':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        return pepmatch_exact_mismatch_count(df, int(spec['mismatches']))

    if kind == 'pepmatch_mismatch_pct':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        return pepmatch_exact_mismatch_pct(df, int(spec['mismatches']))

    if kind == 'boolean_pct':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}. '
                f'Available columns: {list(df.columns)}'
            )

        flags = bool_series(df[col])
        if len(flags) == 0:
            return float('nan')

        return 100.0 * float((flags == True).sum()) / len(flags)

    if kind == 'cached_pairwise_edit_distance':
        mean, _, _ = _read_cached_pairwise_sample_stats(eval_dir)
        return mean

    if kind == 'novelty_summary':
        return read_novelty_summary_metric(
            eval_dir,
            spec['summary_key'],
        )

    if kind in {'valid_pct', 'unique_pct'}:
        raw_path = eval_dir / 'generated_raw.csv'
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        n_raw = len(pd.read_csv(raw_path))
        if n_raw == 0:
            return float('nan')

        if kind == 'valid_pct':
            selected_path = eval_dir / 'generated_filtered_all.csv'
        else:
            selected_path = eval_dir / 'generated_filtered_unique.csv'

        if not selected_path.exists():
            raise FileNotFoundError(selected_path)

        n_selected = len(pd.read_csv(selected_path))
        return 100.0 * n_selected / n_raw

    if kind == 'training_time':
        if seed_root is None:
            if out_file is None:
                raise FileNotFoundError(
                    'No result-folder timing source or matching SLURM .out file'
                )
            return parse_training_time_minutes(out_file)

        value, _, _ = result_training_time_minutes(
            seed_root,
            epoch,
            out_file,
        )
        return value

    raise ValueError(f'Unsupported metric kind: {kind}')



def _series_stats(
    values: pd.Series,
    center: str = 'mean',
) -> Tuple[float, float, int]:
    """
    Center +/- sample SD across individual samples.

    center='mean'   -> arithmetic mean
    center='median' -> median, while SD is still the ordinary sample SD
                       of the underlying sample-level values.
    """
    x = pd.to_numeric(values, errors='coerce').replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    n = len(x)
    if n == 0:
        return float('nan'), float('nan'), 0

    if center == 'median':
        value = float(x.median())
    else:
        value = float(x.mean())

    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    return value, sd, n




def _compute_and_cache_pairwise_stats(
    eval_dir: Path,
    batch_size: int = 256,
) -> Tuple[float, float, int]:
    """
    Compute the original generated-vs-generated pairwise Levenshtein metric
    when no saved cache is available, then save the result for future runs.

    Exact convention preserved:
      * compare i with j for j >= i;
      * each unordered pair is counted once;
      * one self-comparison per peptide is included.

    RapidFuzz is used in blocks when available so the full N x N matrix is
    never stored. If RapidFuzz is unavailable, fall back to the existing
    low-RAM Python-loop implementation already present in this script.
    """
    path = eval_dir / 'generated_filtered_unique.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    pep_col = find_prediction_column(
        df,
        ['peptide', 'generated_peptide', 'sequence'],
    )
    if pep_col is None:
        raise KeyError(
            f'No peptide column found in {path}. '
            f'Available columns: {list(df.columns)}'
        )

    peptides = (
        df[pep_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    peptides = [x for x in peptides.tolist() if x]

    if not peptides:
        mean = float('nan')
        sd = float('nan')
        count = 0
    else:
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein

            n = len(peptides)
            count = 0
            total = 0.0
            total_sq = 0.0

            print(
                f'           pairwise cache missing; computing '
                f'{n:,} generated peptides...',
                flush=True,
            )

            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                block = peptides[start:stop]

                dist = process.cdist(
                    block,
                    peptides,
                    scorer=Levenshtein.distance,
                    workers=-1,
                )

                for local_i, global_i in enumerate(range(start, stop)):
                    vals = np.asarray(
                        dist[local_i, global_i:],
                        dtype=np.float64,
                    )
                    count += int(vals.size)
                    total += float(vals.sum())
                    total_sq += float(np.square(vals).sum())

                print(
                    f'           pairwise edit-distance progress: '
                    f'{stop:,}/{n:,}',
                    flush=True,
                )

            mean = total / count

            if count > 1:
                variance = (
                    total_sq - count * mean * mean
                ) / (count - 1)
                variance = max(0.0, variance)
                sd = math.sqrt(variance)
            else:
                sd = 0.0

        except ImportError:
            print(
                '           rapidfuzz unavailable; using the existing '
                'low-RAM Python pairwise implementation.',
                flush=True,
            )
            mean, sd, count = _pairwise_edit_distance_sample_stats(
                eval_dir
            )

    novelty_dir = eval_dir / 'novelty_vs_training'
    novelty_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'pairwise_edit_distance_mean': float(mean),
        'pairwise_edit_distance_sd': float(sd),
        'pairwise_edit_distance_n_pairs': int(count),
    }

    csv_path = novelty_dir / 'pairwise_edit_distance_summary.csv'
    json_path = novelty_dir / 'pairwise_edit_distance_summary.json'

    pd.DataFrame([stats]).to_csv(csv_path, index=False)

    import json
    json_path.write_text(
        json.dumps(stats, indent=2),
        encoding='utf-8',
    )

    # Also update existing novelty summaries when present so old/new tools
    # can both find the cached fields.
    novelty_csv = novelty_dir / 'novelty_summary.csv'
    if novelty_csv.exists():
        try:
            ndf = pd.read_csv(novelty_csv)
            if len(ndf):
                for key, value in stats.items():
                    ndf.loc[0, key] = value
                ndf.to_csv(novelty_csv, index=False)
        except Exception as e:
            print(
                f'           NOTE: could not update {novelty_csv}: {e}',
                flush=True,
            )

    novelty_json = novelty_dir / 'novelty_summary.json'
    if novelty_json.exists():
        try:
            obj = json.loads(
                novelty_json.read_text(encoding='utf-8')
            )
            if isinstance(obj, dict):
                obj.update(stats)
                novelty_json.write_text(
                    json.dumps(obj, indent=2),
                    encoding='utf-8',
                )
        except Exception as e:
            print(
                f'           NOTE: could not update {novelty_json}: {e}',
                flush=True,
            )

    print(
        f'           pairwise cache saved: {json_path}',
        flush=True,
    )

    return float(mean), float(sd), int(count)


def _read_cached_pairwise_sample_stats(
    eval_dir: Path,
) -> Tuple[float, float, int]:
    """
    Read the cached generated-vs-generated pairwise edit-distance statistics.

    Search order:
      1. novelty_vs_training/pairwise_edit_distance_summary.json
      2. novelty_vs_training/pairwise_edit_distance_summary.csv
      3. pairwise fields merged into novelty_summary.csv/json

    No O(N^2) recomputation is performed by the table generator.
    """
    novelty_dir = eval_dir / 'novelty_vs_training'

    def _coerce_triplet(mean, sd=None, n=None):
        mean_v = pd.to_numeric(
            pd.Series([mean]),
            errors='coerce',
        ).iloc[0]
        sd_v = pd.to_numeric(
            pd.Series([sd]),
            errors='coerce',
        ).iloc[0]
        n_v = pd.to_numeric(
            pd.Series([n]),
            errors='coerce',
        ).iloc[0]

        if pd.isna(mean_v):
            return None

        return (
            float(mean_v),
            float(sd_v) if not pd.isna(sd_v) else float('nan'),
            int(n_v) if not pd.isna(n_v) else 0,
        )

    # 1. Preferred dedicated JSON cache.
    json_path = novelty_dir / 'pairwise_edit_distance_summary.json'
    if json_path.exists():
        import json
        obj = json.loads(json_path.read_text(encoding='utf-8'))
        out = _coerce_triplet(
            obj.get('pairwise_edit_distance_mean'),
            obj.get('pairwise_edit_distance_sd'),
            obj.get('pairwise_edit_distance_n_pairs'),
        )
        if out is not None:
            return out

    # 2. Dedicated CSV cache.
    csv_cache = novelty_dir / 'pairwise_edit_distance_summary.csv'
    if csv_cache.exists():
        df = pd.read_csv(csv_cache)
        if len(df):
            row = df.iloc[0]
            out = _coerce_triplet(
                row.get('pairwise_edit_distance_mean'),
                row.get('pairwise_edit_distance_sd'),
                row.get('pairwise_edit_distance_n_pairs'),
            )
            if out is not None:
                return out

    # 3. Fallback to fields merged into novelty_summary.
    csv_summary = novelty_dir / 'novelty_summary.csv'
    json_summary = novelty_dir / 'novelty_summary.json'

    if csv_summary.exists():
        df = pd.read_csv(csv_summary)
        if len(df) and 'pairwise_edit_distance_mean' in df.columns:
            row = df.iloc[0]
            out = _coerce_triplet(
                row.get('pairwise_edit_distance_mean'),
                row.get('pairwise_edit_distance_sd'),
                row.get('pairwise_edit_distance_n_pairs'),
            )
            if out is not None:
                return out

    if json_summary.exists():
        import json
        obj = json.loads(json_summary.read_text(encoding='utf-8'))
        if 'pairwise_edit_distance_mean' in obj:
            out = _coerce_triplet(
                obj.get('pairwise_edit_distance_mean'),
                obj.get('pairwise_edit_distance_sd'),
                obj.get('pairwise_edit_distance_n_pairs'),
            )
            if out is not None:
                return out

    # No saved result exists: restore the original table-script behavior
    # by computing it now, but cache it so later table runs reuse it.
    return _compute_and_cache_pairwise_stats(eval_dir)


def _pairwise_edit_distance_sample_stats(
    eval_dir: Path,
) -> Tuple[float, float, int]:
    """
    Original generated-vs-generated pairwise edit-distance metric, but in
    sample aggregation mode also report the SD across the actual pairwise
    distance values.

    The same unordered comparisons as the original metric are used:
      i <= j, including one self-comparison per generated peptide.

    RAM remains O(N): distances are accumulated using sum and sum-of-squares;
    the O(N^2) distance values are never stored.
    """
    path = eval_dir / 'generated_filtered_unique.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    pep_col = find_prediction_column(
        df, ['peptide', 'generated_peptide', 'sequence']
    )
    if pep_col is None:
        raise KeyError(
            f'No peptide column found in {path}. '
            f'Available columns: {list(df.columns)}'
        )

    peptides = (
        df[pep_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    peptides = [x for x in peptides.tolist() if x]

    if not peptides:
        return float('nan'), float('nan'), 0

    try:
        import Levenshtein as _lev
        distance = _lev.distance
    except ImportError:
        try:
            from rapidfuzz.distance import Levenshtein as _rf_lev
            distance = _rf_lev.distance
        except ImportError as e:
            raise ImportError(
                'Pairwise edit distance requires python-Levenshtein or rapidfuzz.'
            ) from e

    count = 0
    total = 0.0
    total_sq = 0.0

    n = len(peptides)
    for i in range(n):
        a = peptides[i]
        for j in range(i, n):
            d = float(distance(a, peptides[j]))
            count += 1
            total += d
            total_sq += d * d

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(
                f'           pairwise edit-distance progress: '
                f'{i + 1:,}/{n:,}'
            )

    mean = total / count

    if count > 1:
        variance = (total_sq - count * mean * mean) / (count - 1)
        variance = max(0.0, variance)
        sd = math.sqrt(variance)
    else:
        sd = 0.0

    return float(mean), float(sd), int(count)


def _read_novelty_per_peptide(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / 'novelty_vs_training' / 'novelty_per_peptide.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run run_novelty_all_evaluations.py first.'
        )
    return pd.read_csv(path)


def _pepmatch_exact_indicator(
    df: pd.DataFrame,
    mismatch_count: int,
) -> pd.Series:
    """
    One 0/1 value per evaluated peptide for exactly k mismatches.

    Important handling for PEPMatch no-hit rows:
      * If pepmatch_mismatches is numeric, use it directly.
      * If pepmatch_mismatches is blank/NaN but
        pepmatch_no_match_within_3=True, the peptide contributes 0 for
        exact, 1-, 2-, and 3-mismatch categories.
      * For any remaining rows, fall back to the cumulative PEPMatch
        boolean flags when available.

    This prevents a valid "no match within 3 mismatches" result from being
    treated as missing data in sample aggregation.
    """
    if mismatch_count not in {0, 1, 2, 3}:
        raise ValueError(
            f'PEPMatch exact-mismatch indicator supports only 0..3; '
            f'got {mismatch_count}.'
        )

    mismatch_col = find_prediction_column(
        df,
        [
            'pepmatch_mismatches',
            'PEPMatch_mismatches',
            'pepmatch_mismatch_count',
            'mismatches',
        ],
    )

    out = pd.Series(np.nan, index=df.index, dtype=float)

    # 1. Preferred: explicit numeric best-mismatch count.
    if mismatch_col is not None:
        vals = pd.to_numeric(df[mismatch_col], errors='coerce')
        known = vals.notna()
        out.loc[known] = (
            vals.loc[known] == mismatch_count
        ).astype(float)

    # 2. Explicit "no hit within 3 mismatches" is a valid negative result
    #    for every exact-k category k=0,1,2,3, not missing data.
    no_match_col = find_prediction_column(
        df,
        [
            'pepmatch_no_match_within_3',
            'PEPMatch_no_match_within_3',
            'no_match_within_3',
        ],
    )
    if no_match_col is not None:
        no_match = bool_series(df[no_match_col])
        mask = out.isna() & (no_match == True)
        out.loc[mask] = 0.0

    # 3. For rows still unresolved, derive exact-k membership from the
    #    cumulative boolean flags if those columns are available.
    current_candidates = {
        0: ['pepmatch_exact_match'],
        1: ['pepmatch_within_1_mismatch'],
        2: ['pepmatch_within_2_mismatches'],
        3: ['pepmatch_within_3_mismatches'],
    }

    current_col = find_prediction_column(
        df, current_candidates[mismatch_count]
    )

    if current_col is not None:
        current = bool_series(df[current_col])

        if mismatch_count == 0:
            mask = out.isna() & current.notna()
            out.loc[mask] = current.loc[mask].astype(bool).astype(float)
        else:
            previous_candidates = {
                1: ['pepmatch_exact_match'],
                2: ['pepmatch_within_1_mismatch'],
                3: ['pepmatch_within_2_mismatches'],
            }
            previous_col = find_prediction_column(
                df, previous_candidates[mismatch_count]
            )

            if previous_col is not None:
                previous = bool_series(df[previous_col])
                mask = (
                    out.isna()
                    & current.notna()
                    & previous.notna()
                )
                out.loc[mask] = (
                    (current.loc[mask] == True)
                    & ~(previous.loc[mask] == True)
                ).astype(float)

    # If nothing at all could be interpreted, preserve the old clear error.
    if out.notna().sum() == 0:
        raise KeyError(
            'Could not find usable PEPMatch mismatch information. '
            f'Available columns: {list(df.columns)}'
        )

    return out


def _read_pepsysco_sample_values(eval_dir: Path) -> pd.Series:
    """
    Read one PepSySco score per generated peptide using the same preferred
    sources as the table's aggregate PepSySco reader.
    """
    generated_path = eval_dir / 'generated_filtered_unique.csv'
    if not generated_path.exists():
        raise FileNotFoundError(generated_path)

    generated = pd.read_csv(generated_path)
    gcol = find_prediction_column(
        generated, ['peptide', 'generated_peptide', 'sequence']
    )
    if gcol is None:
        raise KeyError(
            f'No peptide column found in {generated_path}.'
        )

    wanted = pd.DataFrame({
        'peptide': (
            generated[gcol]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    }).drop_duplicates('peptide', keep='first')

    sources = [
        eval_dir / 'all_predictions.csv',
        eval_dir / 'pepsysco' / 'PepSySco_prediction.csv',
    ]

    for path in sources:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        pcol = find_prediction_column(
            df, ['peptide', 'sequence', 'Peptide', 'Sequence']
        )
        scol = find_prediction_column(
            df, METRICS['pepsysco']['columns']
        )
        if pcol is None or scol is None:
            continue

        part = pd.DataFrame({
            'peptide': df[pcol].astype(str).str.strip().str.upper(),
            'value': pd.to_numeric(df[scol], errors='coerce'),
        }).drop_duplicates('peptide', keep='first')

        merged = wanted.merge(part, on='peptide', how='left')
        return merged['value']

    raise FileNotFoundError(
        f'No usable PepSySco sample-level result under {eval_dir}'
    )


def read_metric_sample_stats(
    metric: str,
    eval_dir: Path,
    out_file: Optional[Path],
    seed_root: Optional[Path] = None,
    epoch: int = 1000,
    sample_count_std: bool = False,
) -> Tuple[float, float, int]:
    """
    Compute one table cell from sample-level values inside ONE selected seed.

    This is used only for --aggregation sample. The default --aggregation seed
    path is unchanged.
    """
    spec = METRICS[metric]
    kind = spec['kind']

    if metric == 'pepsysco':
        return _series_stats(
            _read_pepsysco_sample_values(eval_dir)
        )

    if kind == 'prediction_mean':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}. '
                f'Available columns: {list(df.columns)}'
            )

        return _series_stats(df[col])

    if kind == 'prediction_threshold_pct':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}.'
            )

        vals = pd.to_numeric(df[col], errors='coerce')
        if vals.isna().any():
            raise ValueError(
                f'{int(vals.isna().sum())}/{len(vals)} values are missing '
                f'in {path}; refusing partial sample aggregation.'
            )

        threshold = float(spec['threshold'])
        if spec.get('comparison', 'lt') == 'lt':
            indicator = (vals < threshold).astype(float)
        else:
            indicator = (vals <= threshold).astype(float)

        # Percentage over this one selected seed. Do not attach the SD of the
        # underlying 0/1 indicators to the reported percentage.
        pct = 100.0 * float(indicator.mean())
        return pct, float('nan'), 1

    if kind in {'pepmatch_mismatch_count', 'pepmatch_mismatch_pct'}:
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        indicator = _pepmatch_exact_indicator(
            df, int(spec['mismatches'])
        )

        if kind == 'pepmatch_mismatch_count':
            indicator = pd.to_numeric(
                indicator, errors='coerce'
            ).dropna()
            n = len(indicator)
            if n == 0:
                return float('nan'), float('nan'), 0

            count = float(indicator.sum())
            if sample_count_std:
                p = count / n
                sd = math.sqrt(n * p * (1.0 - p))
            else:
                sd = float('nan')
            return count, sd, n

        pct = 100.0 * float(pd.to_numeric(
            indicator, errors='coerce'
        ).dropna().mean())
        return pct, float('nan'), 1

    if kind == 'boolean_pct':
        path = eval_dir / 'all_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        col = find_prediction_column(df, spec['columns'])
        if col is None:
            raise KeyError(
                f'None of {spec["columns"]} found in {path}.'
            )

        flags = bool_series(df[col]).dropna()
        if len(flags) == 0:
            return float('nan'), float('nan'), 0
        pct = 100.0 * float((flags == True).mean())
        return pct, float('nan'), 1

    if kind == 'cached_pairwise_edit_distance':
        return _read_cached_pairwise_sample_stats(eval_dir)

    if kind == 'novelty_summary':
        df = _read_novelty_per_peptide(eval_dir)
        key = spec['summary_key']

        mapping = {
            'min_edit_distance_mean':
                ('min_edit_distance', 'mean'),
            'min_edit_distance_median':
                ('min_edit_distance', 'median'),
            'nearest_neighbor_similarity_mean':
                ('nearest_neighbor_similarity', 'mean'),
            'sequence_identity_percent_mean':
                ('sequence_identity_percent', 'mean'),
            'exact_overlap_percent':
                ('exact_training_overlap', 'bool_pct'),
            'non_exact_novel_percent':
                ('exact_training_overlap', 'not_bool_pct'),
            'pct_sequence_identity_ge_90':
                ('sequence_identity_percent', 'ge90_pct'),
            'pct_min_edit_distance_ge_2':
                ('min_edit_distance', 'ge2_pct'),
            'novelty_score_mean':
                ('novelty_score', 'mean'),
        }

        if key not in mapping:
            raise KeyError(
                f'No sample-level mapping defined for novelty key {key!r}.'
            )

        col, mode = mapping[key]
        if col not in df.columns:
            raise KeyError(
                f'{col!r} not found in novelty_per_peptide.csv. '
                f'Available columns: {list(df.columns)}'
            )

        if mode == 'mean':
            return _series_stats(df[col])
        if mode == 'median':
            return _series_stats(df[col], center='median')

        if mode in {'bool_pct', 'not_bool_pct'}:
            flags = bool_series(df[col]).dropna()
            if len(flags) == 0:
                return float('nan'), float('nan'), 0
            if mode == 'not_bool_pct':
                pct = 100.0 * float((flags == False).mean())
            else:
                pct = 100.0 * float((flags == True).mean())
            return pct, float('nan'), 1

        vals = pd.to_numeric(df[col], errors='coerce')

        if mode == 'ge90_pct':
            known = vals.dropna()
            if len(known) == 0:
                return float('nan'), float('nan'), 0
            pct = 100.0 * float((known >= 90.0).mean())
            return pct, float('nan'), 1

        if mode == 'ge2_pct':
            known = vals.dropna()
            if len(known) == 0:
                return float('nan'), float('nan'), 0
            pct = 100.0 * float((known >= 2.0).mean())
            return pct, float('nan'), 1

    if kind in {'valid_pct', 'unique_pct'}:
        raw_path = eval_dir / 'generated_raw.csv'
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        raw = pd.read_csv(raw_path)
        n_raw = len(raw)
        if n_raw == 0:
            return float('nan'), float('nan'), 0

        selected_path = (
            eval_dir / 'generated_filtered_all.csv'
            if kind == 'valid_pct'
            else eval_dir / 'generated_filtered_unique.csv'
        )
        if not selected_path.exists():
            raise FileNotFoundError(selected_path)

        selected = pd.read_csv(selected_path)

        # These are percentages for the one selected seed, so report the
        # percentage itself without an SD of per-sample 0/1 indicators.
        pct = 100.0 * len(selected) / n_raw
        return float(pct), float('nan'), 1

    if kind == 'training_time':
        if seed_root is None:
            if out_file is None:
                raise FileNotFoundError(
                    'No result-folder timing source or matching SLURM .out file'
                )
            value = parse_training_time_minutes(out_file)
        else:
            value, _, _ = result_training_time_minutes(
                seed_root,
                epoch,
                out_file,
            )
        return float(value), 0.0, 1

    raise ValueError(
        f'Unsupported sample-level metric kind: {kind}'
    )



def format_sample_value(
    center: float,
    sd: float,
    n: int,
    decimals: int,
    blank_if_missing: bool = False,
    integer_center: bool = False,
) -> str:
    """
    Formatting used only for --aggregation sample.

    If an SD is statistically meaningful/available (n > 1 and finite SD),
    report center +/- SD.

    If SD cannot be computed or is not meaningful (for example n <= 1),
    report only the center value.

    Seed aggregation continues to use format_mean_sd() unchanged.
    """
    if math.isnan(center):
        return '' if blank_if_missing else '--'

    center_text = (
        str(int(round(center)))
        if integer_center
        else f'{center:.{decimals}f}'
    )

    if n <= 1 or math.isnan(sd):
        return center_text

    return f'{center_text}$\\pm${sd:.{decimals}f}'


def run_sample_aggregation(
    args,
    algorithms,
    columns,
    result_root: Path,
    out_root: Path,
):
    """
    One selected seed per algorithm; mean +/- SD is computed from samples
    inside that seed instead of across random seeds.
    """
    stat_rows = []
    summary_rows = []

    print(
        'Sample aggregation mode: one selected seed per model; '
        'mean +/- SD is across sample-level values.'
    )
    print(
        f'Fallback sample seed: {args.sample_seed} '
        '(overridden by optional sixth config field)'
    )

    for alg in algorithms:
        seed = (
            alg.get('sample_seed')
            if alg.get('sample_seed') is not None
            else args.sample_seed
        )

        folder_name, seed_root = resolve_seed_root(
            result_root,
            alg['folder'],
            seed,
        )
        eval_dir = find_eval_dir(
            seed_root,
            alg['checkpoint'],
            seed,
            args.epoch,
        )
        out_file = find_out_file(
            out_root,
            alg['out_glob'],
            alg['folder'],
            seed,
        )

        print('\\n' + '=' * 90)
        print(
            f'Algorithm: {alg["display_name"]} | '
            f'selected seed: {seed}'
        )
        print('=' * 90)
        print(
            f'eval={eval_dir if eval_dir else "NOT FOUND"}'
        )

        summary_row = {
            'Algorithm': alg['display_name'],
            '_training_time_program': False,
        }

        # Resolve the timing source once so Program Runtime can be marked
        # in the LaTeX table when training_time_min is requested.
        if 'training_time_min' in columns:
            try:
                _, timing_kind, timing_source = result_training_time_minutes(
                    seed_root,
                    args.epoch,
                    out_file,
                )
                summary_row['_training_time_program'] = (
                    timing_kind == 'program'
                )
                print(
                    f'           training time source={timing_source} '
                    f'kind={timing_kind}'
                )
            except Exception as e:
                print(
                    f'           NOTE training-time source: '
                    f'{type(e).__name__}: {e}'
                )

        for metric in columns:
            try:
                if (
                    eval_dir is None
                    and METRICS[metric]['kind'] != 'training_time'
                ):
                    raise FileNotFoundError(
                        f'No evaluation directory under {seed_root}'
                    )

                center, sd, n = read_metric_sample_stats(
                    metric,
                    eval_dir,
                    out_file,
                    seed_root=seed_root,
                    epoch=args.epoch,
                    sample_count_std=args.sample_count_std,
                )

            except Exception as e:
                center = float('nan')
                sd = float('nan')
                n = 0
                print(
                    f'           WARNING {metric}: '
                    f'{type(e).__name__}: {e}'
                )
                if args.strict:
                    raise

            stat_rows.append({
                'Algorithm': alg['display_name'],
                'folder': folder_name,
                'seed': seed,
                'checkpoint': alg['checkpoint'],
                'eval_dir': str(eval_dir) if eval_dir else '',
                'metric': metric,
                'center': center,
                'sd': sd,
                'n_samples': n,
                'training_time_is_program_runtime': (
                    summary_row.get('_training_time_program', False)
                    if metric == 'training_time_min'
                    else False
                ),
            })

            summary_row[f'{metric}_mean'] = center
            summary_row[f'{metric}_sd'] = sd
            summary_row[f'{metric}_n'] = n
            summary_row[f'{metric}_formatted'] = format_sample_value(
                center,
                sd,
                n,
                METRICS[metric]['decimals'],
                METRICS[metric].get(
                    'blank_if_missing', False
                ),
                integer_center=(
                    METRICS[metric]['kind'] == 'pepmatch_mismatch_count'
                ),
            )

            if (
                metric == 'training_time_min'
                and summary_row.get('_training_time_program', False)
                and summary_row[f'{metric}_formatted'] not in {'', '--'}
            ):
                summary_row[f'{metric}_formatted'] += (
                    r'\textsuperscript{\dag}'
                )

        summary_rows.append(summary_row)

    sample_stats = pd.DataFrame(stat_rows)
    sample_stats_path = Path(
        f'{args.output_prefix}_sample_stats.csv'
    )
    sample_stats.to_csv(
        sample_stats_path,
        index=False,
    )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(
        f'{args.output_prefix}_summary.csv'
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    tex = latex_table(
        summary,
        columns,
        args.caption,
        args.label,
    )
    tex_path = Path(
        f'{args.output_prefix}.tex'
    )
    tex_path.write_text(tex)

    print('\\n' + '=' * 90)
    print('FINISHED - SAMPLE AGGREGATION')
    print('=' * 90)
    print(
        f'Sample statistics: {sample_stats_path.resolve()}'
    )
    print(
        f'Summary CSV:       {summary_path.resolve()}'
    )
    print(
        f'LaTeX table:       {tex_path.resolve()}'
    )
    print('\\nLaTeX preview:\\n')
    print(tex)


def mean_sd(values: pd.Series) -> Tuple[float, float, int]:
    x = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    n = len(x)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    return mean, sd, n


def format_mean_sd(mean: float, sd: float, decimals: int, blank_if_missing: bool = False) -> str:
    if math.isnan(mean):
        return '' if blank_if_missing else '--'
    if math.isnan(sd):
        sd = 0.0
    return f'{mean:.{decimals}f}$\\pm${sd:.{decimals}f}'


def latex_table(summary_df: pd.DataFrame, columns: List[str], caption: str, label: str) -> str:
    align = 'l' + 'l' * len(columns)
    header_cells = [r'\multicolumn{1}{c}{\bf Algorithm}']
    for metric in columns:
        header_cells.append(rf'\multicolumn{{1}}{{c}}{{\bf {METRICS[metric]["header"]}}}')

    lines = [
        r'\begin{table}[t]',
        r'\begin{center}',
        rf'\begin{{tabular}}{{{align}}}',
        ' & '.join(header_cells) + r' \\ \hline',
    ]

    for _, row in summary_df.iterrows():
        cells = [row['Algorithm']]
        for metric in columns:
            cells.append(row[f'{metric}_formatted'])
        lines.append(' & '.join(cells) + r' \\')

    any_program_runtime = (
        '_training_time_program' in summary_df.columns
        and summary_df['_training_time_program'].fillna(False).astype(bool).any()
    )

    caption_text = caption
    if any_program_runtime:
        caption_text += (
            r' \textsuperscript{\dag}The marked value uses Program Runtime '
            r'(e.g., for PepINVENT) because a separate training-only runtime '
            r'was not recorded.'
        )

    lines.extend([
        r'\end{tabular}',
        rf'\caption{{{caption_text}}}',
        rf'\label{{{label}}}',
        r'\end{center}',
        r'\end{table}',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    algorithms, config_columns = parse_config(config_path)
    columns = args.columns or config_columns or DEFAULT_COLUMNS
    unknown = [c for c in columns if c not in METRICS]
    if unknown:
        raise SystemExit(f'Unknown metric(s): {unknown}. Available: {list(METRICS)}')

    if args.seeds:
        seeds = args.seeds
    else:
        if args.seed_end < args.seed_start:
            raise SystemExit('--seed-end must be >= --seed-start')
        seeds = list(range(args.seed_start, args.seed_end + 1))

    rows = []
    print(f'Config:      {config_path}')
    print(f'Result root: {result_root}')
    print(f'OUT root:    {out_root}')
    print(f'Aggregation: {args.aggregation}')
    if args.aggregation == 'seed':
        print(f'Seeds:       {seeds[0]}..{seeds[-1]} ({len(seeds)} seed(s))' if seeds else 'Seeds: none')
    else:
        print(f'Fallback sample seed: {args.sample_seed}')
    print(f'Columns:     {columns}')

    if args.aggregation == 'sample':
        run_sample_aggregation(
            args,
            algorithms,
            columns,
            result_root,
            out_root,
        )
        return

    for alg in algorithms:
        print('\n' + '=' * 90)
        print(f'Algorithm: {alg["display_name"]}')
        print('=' * 90)

        for seed in seeds:
            folder_name, seed_root = resolve_seed_root(
                result_root,
                alg['folder'],
                seed,
            )
            eval_dir = find_eval_dir(seed_root, alg['checkpoint'], seed, args.epoch)
            out_file = find_out_file(out_root, alg['out_glob'], alg['folder'], seed)

            row = {
                'Algorithm': alg['display_name'],
                'folder': folder_name,
                'seed': seed,
                'checkpoint': alg['checkpoint'],
                'eval_dir': str(eval_dir) if eval_dir else '',
                'out_file': str(out_file) if out_file else '',
                'training_time_kind': '',
                'training_time_source': '',
            }

            print(f'[seed {seed:>3}] eval={eval_dir if eval_dir else "NOT FOUND"}')
            if 'training_time_min' in columns:
                try:
                    _, timing_kind, timing_source = result_training_time_minutes(
                        seed_root,
                        args.epoch,
                        out_file,
                    )
                    row['training_time_kind'] = timing_kind
                    row['training_time_source'] = str(timing_source)
                    print(
                        f'           training time source={timing_source} '
                        f'kind={timing_kind}'
                    )
                except Exception as e:
                    print(
                        f'           training time source NOT FOUND: '
                        f'{type(e).__name__}: {e}'
                    )

            for metric in columns:
                try:
                    if eval_dir is None and METRICS[metric]['kind'] != 'training_time':
                        raise FileNotFoundError(f'No evaluation directory under {seed_root}')
                    row[metric] = read_seed_metric(
                        metric,
                        eval_dir,
                        out_file,
                        seed_root=seed_root,
                        epoch=args.epoch,
                    )
                except Exception as e:
                    row[metric] = float('nan')
                    print(f'           WARNING {metric}: {type(e).__name__}: {e}')
                    if args.strict:
                        raise
            rows.append(row)

    per_seed = pd.DataFrame(rows)
    per_seed_path = Path(f'{args.output_prefix}_per_seed.csv')
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows = []
    for alg in algorithms:
        display = alg['display_name']
        sub = per_seed[per_seed['Algorithm'] == display]
        out = {
            'Algorithm': display,
            '_training_time_program': (
                'training_time_kind' in sub.columns
                and (sub['training_time_kind'] == 'program').any()
            ),
        }
        for metric in columns:
            mean, sd, n = mean_sd(sub[metric])
            out[f'{metric}_mean'] = mean
            out[f'{metric}_sd'] = sd
            out[f'{metric}_n'] = n
            out[f'{metric}_formatted'] = format_mean_sd(
                mean,
                sd,
                METRICS[metric]['decimals'],
                METRICS[metric].get('blank_if_missing', False),
            )

            if (
                metric == 'training_time_min'
                and out['_training_time_program']
                and out[f'{metric}_formatted'] not in {'', '--'}
            ):
                out[f'{metric}_formatted'] += (
                    r'\textsuperscript{\dag}'
                )

        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(f'{args.output_prefix}_summary.csv')
    summary.to_csv(summary_path, index=False)

    tex = latex_table(summary, columns, args.caption, args.label)
    tex_path = Path(f'{args.output_prefix}.tex')
    tex_path.write_text(tex)

    print('\n' + '=' * 90)
    print('FINISHED')
    print('=' * 90)
    print(f'Per-seed metrics: {per_seed_path.resolve()}')
    print(f'Summary CSV:      {summary_path.resolve()}')
    print(f'LaTeX table:      {tex_path.resolve()}')
    print('\nLaTeX preview:\n')
    print(tex)


if __name__ == '__main__':
    main()
