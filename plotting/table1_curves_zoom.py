#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULT_ROOT = REPO_ROOT / "result"

COLORS = [
    '#1f77b4',
    '#ff750e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#a64500',
]

# Each MODEL keeps the same line style in every figure.
LINESTYLES = [
    '-',
    '--',
    '-.',
    ':',
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (3, 2, 1, 2, 1, 2)),
    (0, (8, 2)),
    (0, (2, 2)),
]


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            'Plot immunogenicity score, unique rate, and generator/critic losses '
            'for enabled models in table1_config.txt.'
        )
    )
    ap.add_argument('--config', default=str(SCRIPT_DIR / 'table1_config.txt'))
    ap.add_argument('--result-root', default=str(DEFAULT_RESULT_ROOT))
    ap.add_argument('--epoch', type=int, default=1000)
    ap.add_argument(
        '--seeds',
        nargs='*',
        type=int,
        default=None,
        help=(
            'Optional seed list, e.g. --seeds 0 1 2 3. '
            'If omitted, all matching seed folders are discovered automatically.'
        ),
    )
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--linewidth', type=float, default=1.5)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--output-prefix', default='table1_training')
    ap.add_argument(
        '--zoom-ymin',
        type=float,
        default=0.970,
        help='Lower y-axis limit for the separate unique-rate zoom figure (default: 0.970).',
    )
    ap.add_argument(
        '--zoom-ymax',
        type=float,
        default=1.000,
        help='Upper y-axis limit for the separate unique-rate zoom figure (default: 1.000).',
    )
    ap.add_argument('--no-show', action='store_true')
    return ap.parse_args()


def parse_config(path: Path) -> List[Dict[str, str]]:
    """Read enabled algorithm rows from table1_config.txt."""
    algorithms = []

    for line_no, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        # Ignore settings such as COLUMNS=...
        if '=' in line and '|' not in line:
            continue

        parts = [x.strip() for x in line.split('|')]
        if len(parts) < 3:
            raise ValueError(
                f'{path}:{line_no}: expected at least 3 pipe-separated fields: '
                'enabled | display_name | folder_pattern'
            )

        enabled = parts[0].lower() in {'1', 'true', 'yes', 'y', 'on'}
        if not enabled:
            continue

        algorithms.append({
            'display_name': parts[1],
            'folder': parts[2],
        })

    if not algorithms:
        raise ValueError(f'No enabled algorithms found in {path}')

    # "best" and "last" config rows can point to the same training folder.
    # The training-history arrays are identical in that case, so plot once.
    deduplicated = []
    seen_folders = set()

    for alg in algorithms:
        folder = alg['folder']
        if folder in seen_folders:
            continue
        seen_folders.add(folder)
        deduplicated.append(alg)

    return deduplicated


def clean_display_name(name: str) -> str:
    """Remove best/last superscripts because these are training histories."""
    name = re.sub(
        r'\\textsuperscript\{(?:best|last)\}',
        '',
        name,
        flags=re.IGNORECASE,
    )
    return re.sub(r'\s+', ' ', name).strip()


def resolve_seed_folder(folder_template: str, seed: int) -> str:
    if '{seed}' in folder_template:
        return folder_template.format(seed=seed)

    if re.search(r'_seed\d+$', folder_template):
        return re.sub(r'_seed\d+$', f'_seed{seed}', folder_template)

    return f'{folder_template}_seed{seed}'


def discover_seed_folders(
    result_root: Path,
    folder_template: str,
    requested_seeds: Optional[List[int]],
) -> List[Tuple[int, Path]]:
    """Return sorted (seed, folder_path) pairs for one model."""
    if requested_seeds is not None:
        out = []
        for seed in requested_seeds:
            path = result_root / resolve_seed_folder(folder_template, seed)
            if path.is_dir():
                out.append((seed, path))
            else:
                print(f'  missing folder for seed {seed}: {path}')
        return out

    if '{seed}' in folder_template:
        prefix, suffix = folder_template.split('{seed}', 1)
        regex = re.compile(
            '^' + re.escape(prefix) + r'(?P<seed>\d+)' + re.escape(suffix) + '$'
        )
        candidates = result_root.glob(f'{prefix}*{suffix}')

    elif re.search(r'_seed\d+$', folder_template):
        base = re.sub(r'_seed\d+$', '_seed', folder_template)
        regex = re.compile('^' + re.escape(base) + r'(?P<seed>\d+)$')
        candidates = result_root.glob(f'{base}*')

    else:
        path = result_root / folder_template
        return [(0, path)] if path.is_dir() else []

    out = []

    for path in candidates:
        if not path.is_dir():
            continue

        m = regex.match(path.name)
        if m:
            out.append((int(m.group('seed')), path))

    return sorted(out, key=lambda x: x[0])


def load_curve(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=float).squeeze()

    if arr.ndim != 1:
        raise ValueError(
            f'Expected a 1-D history array, got shape {arr.shape}: {path}'
        )

    return arr


def make_model_legend_handle(model_idx: int, label: str) -> Line2D:
    return Line2D(
        [0], [0],
        color=COLORS[model_idx % len(COLORS)],
        linestyle=LINESTYLES[model_idx % len(LINESTYLES)],
        linewidth=2.5,
        label=label,
    )


def plot_metric(
    algorithms: List[Dict[str, str]],
    result_root: Path,
    epoch: int,
    seeds: Optional[List[int]],
    filename: str,
    ylabel: str,
    title: str,
    output_path: Path,
    alpha: float,
    linewidth: float,
    dpi: int,
    show: bool,
):
    # Main plot + separate legend, matching the style of the original script.
    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[5, 2],
    )
    ax = fig.add_subplot(gs[0])

    legend_handles = []
    plotted_models = 0
    plotted_curves = 0

    for model_idx, alg in enumerate(algorithms):
        name = clean_display_name(alg['display_name'])
        folder_template = alg['folder']
        color = COLORS[model_idx % len(COLORS)]
        linestyle = LINESTYLES[model_idx % len(LINESTYLES)]

        seed_folders = discover_seed_folders(
            result_root,
            folder_template,
            seeds,
        )

        print('\n' + '=' * 90)
        print(f'Model:  {name}')
        print(f'Folder: {folder_template}')
        print(f'File:   {filename}')
        print(f'Found:  {len(seed_folders)} seed folder(s)')

        model_has_curve = False

        for seed, seed_folder in seed_folders:
            npy_path = seed_folder / f'epoch{epoch}' / filename

            if not npy_path.is_file():
                print(f'  seed {seed:>3}: missing {npy_path}')
                continue

            try:
                values = load_curve(npy_path)
            except Exception as e:
                print(
                    f'  seed {seed:>3}: could not load {npy_path}: '
                    f'{type(e).__name__}: {e}'
                )
                continue

            x = np.arange(1, len(values) + 1)

            ax.plot(
                x,
                values,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

            print(
                f'  seed {seed:>3}: plotted {len(values)} point(s) '
                f'from {npy_path}'
            )

            model_has_curve = True
            plotted_curves += 1

        if model_has_curve:
            plotted_models += 1
            legend_handles.append(
                make_model_legend_handle(model_idx, name)
            )

    if plotted_curves == 0:
        plt.close(fig)
        raise FileNotFoundError(
            f'No {filename} files were found under {result_root} '
            'for the enabled config models.'
        )

    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(True)

    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=legend_handles,
        loc='center',
        ncol=1,
        fontsize=14,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
    )

    print(f'\nSaved: {output_path.resolve()}')
    print(
        f'Plotted {plotted_curves} curve(s) '
        f'across {plotted_models} model(s).'
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_unique_rate_zoom(
    algorithms: List[Dict[str, str]],
    result_root: Path,
    epoch: int,
    seeds: Optional[List[int]],
    output_path: Path,
    alpha: float,
    linewidth: float,
    dpi: int,
    show: bool,
    ymin: float = 0.970,
    ymax: float = 1.000,
):
    """
    Make a separate zoomed unique-rate figure like the provided example:
    keep the full epoch range, but zoom only the y-axis near 1.0.

    The main unique-rate figure still contains the legend, so this compact
    zoomed figure intentionally omits it.
    """
    if ymax <= ymin:
        raise ValueError('--zoom-ymax must be greater than --zoom-ymin')

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plotted_curves = 0

    for model_idx, alg in enumerate(algorithms):
        folder_template = alg['folder']
        color = COLORS[model_idx % len(COLORS)]
        linestyle = LINESTYLES[model_idx % len(LINESTYLES)]

        seed_folders = discover_seed_folders(
            result_root,
            folder_template,
            seeds,
        )

        for seed, seed_folder in seed_folders:
            npy_path = seed_folder / f'epoch{epoch}' / 'G_unq.npy'

            if not npy_path.is_file():
                continue

            try:
                values = load_curve(npy_path)
            except Exception as e:
                print(
                    f'  seed {seed:>3}: could not load {npy_path}: '
                    f'{type(e).__name__}: {e}'
                )
                continue

            x = np.arange(1, len(values) + 1)
            ax.plot(
                x,
                values,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
            plotted_curves += 1

    if plotted_curves == 0:
        plt.close(fig)
        raise FileNotFoundError(
            'No G_unq.npy files were found for the zoomed unique-rate figure.'
        )

    # Match the attached example: full 0..1000 x-axis, tight y-axis near 1.
    ax.set_xlim(0, epoch)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Unique rate', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
    )

    print(f'\nSaved zoomed unique-rate figure: {output_path.resolve()}')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_losses(
    algorithms: List[Dict[str, str]],
    result_root: Path,
    epoch: int,
    seeds: Optional[List[int]],
    output_path: Path,
    alpha: float,
    linewidth: float,
    dpi: int,
    show: bool,
):
    """
    Plot G_losses.npy and D_losses.npy.

    Generator and critic are shown in separate panels so the color + line-style
    pair can remain a consistent MODEL identifier in every figure.
    """
    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=1,
        height_ratios=[4, 4, 2],
    )

    ax_g = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])
    legend_ax = fig.add_subplot(gs[2])

    legend_handles = []
    plotted_models = 0
    plotted_g = 0
    plotted_d = 0

    for model_idx, alg in enumerate(algorithms):
        name = clean_display_name(alg['display_name'])
        folder_template = alg['folder']
        color = COLORS[model_idx % len(COLORS)]
        linestyle = LINESTYLES[model_idx % len(LINESTYLES)]

        seed_folders = discover_seed_folders(
            result_root,
            folder_template,
            seeds,
        )

        print('\n' + '=' * 90)
        print(f'Loss model: {name}')
        print(f'Folder:     {folder_template}')
        print(f'Found:      {len(seed_folders)} seed folder(s)')

        model_has_loss = False

        for seed, seed_folder in seed_folders:
            epoch_dir = seed_folder / f'epoch{epoch}'
            g_path = epoch_dir / 'G_losses.npy'
            d_path = epoch_dir / 'D_losses.npy'

            if g_path.is_file():
                try:
                    g_values = load_curve(g_path)
                    gx = np.arange(1, len(g_values) + 1)

                    ax_g.plot(
                        gx,
                        g_values,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha,
                        linewidth=linewidth,
                    )

                    print(
                        f'  seed {seed:>3}: generator loss '
                        f'{len(g_values)} point(s)'
                    )

                    plotted_g += 1
                    model_has_loss = True

                except Exception as e:
                    print(
                        f'  seed {seed:>3}: could not load {g_path}: '
                        f'{type(e).__name__}: {e}'
                    )
            else:
                print(
                    f'  seed {seed:>3}: missing {g_path}'
                )

            if d_path.is_file():
                try:
                    d_values = load_curve(d_path)
                    dx = np.arange(1, len(d_values) + 1)

                    ax_d.plot(
                        dx,
                        d_values,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha,
                        linewidth=linewidth,
                    )

                    print(
                        f'  seed {seed:>3}: critic loss '
                        f'{len(d_values)} point(s)'
                    )

                    plotted_d += 1
                    model_has_loss = True

                except Exception as e:
                    print(
                        f'  seed {seed:>3}: could not load {d_path}: '
                        f'{type(e).__name__}: {e}'
                    )
            else:
                print(
                    f'  seed {seed:>3}: missing {d_path}'
                )

        if model_has_loss:
            plotted_models += 1
            legend_handles.append(
                make_model_legend_handle(model_idx, name)
            )

    if plotted_g == 0 and plotted_d == 0:
        plt.close(fig)
        raise FileNotFoundError(
            'No G_losses.npy or D_losses.npy files were found '
            'for the enabled config models.'
        )

    ax_g.set_xlabel('Epoch', fontsize=18)
    ax_g.set_ylabel('Generator loss', fontsize=18)
    ax_g.set_title('Generator Loss vs. Epoch', fontsize=20)
    ax_g.tick_params(labelsize=14)
    ax_g.grid(True)

    ax_d.set_xlabel('Epoch', fontsize=18)
    ax_d.set_ylabel('Critic loss', fontsize=18)
    ax_d.set_title('Critic Loss vs. Epoch', fontsize=20)
    ax_d.tick_params(labelsize=14)
    ax_d.grid(True)

    legend_ax.axis('off')
    legend_ax.legend(
        handles=legend_handles,
        loc='center',
        ncol=1,
        fontsize=14,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
    )

    print(f'\nSaved: {output_path.resolve()}')
    print(
        f'Loss curves: {plotted_g} generator + {plotted_d} critic '
        f'across {plotted_models} model(s).'
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    algorithms = parse_config(config_path)

    print(f'Config:      {config_path}')
    print(f'Result root: {result_root}')
    print(f'Epoch dir:   epoch{args.epoch}')
    print(f'Alpha:       {args.alpha}')
    print(
        f'Models:      {len(algorithms)} '
        'unique training folder pattern(s)'
    )

    if args.seeds is None:
        print(
            'Seeds:       auto-discover all matching seed folders'
        )
    else:
        print(f'Seeds:       {args.seeds}')

    print(
        f'Unique-rate zoom y-range: {args.zoom_ymin:.3f} to {args.zoom_ymax:.3f}'
    )

    plot_metric(
        algorithms=algorithms,
        result_root=result_root,
        epoch=args.epoch,
        seeds=args.seeds,
        filename='G_score.npy',
        ylabel='Predicted imm. score',
        title='Predicted Immunogenicity Score vs. Epoch',
        output_path=Path(
            f'{args.output_prefix}_immunogenicity_scores.png'
        ),
        alpha=args.alpha,
        linewidth=args.linewidth,
        dpi=args.dpi,
        show=not args.no_show,
    )

    plot_metric(
        algorithms=algorithms,
        result_root=result_root,
        epoch=args.epoch,
        seeds=args.seeds,
        filename='G_unq.npy',
        ylabel='Unique rate',
        title='Unique Peptide Rate vs. Epoch',
        output_path=Path(
            f'{args.output_prefix}_unique_rate.png'
        ),
        alpha=args.alpha,
        linewidth=args.linewidth,
        dpi=args.dpi,
        show=not args.no_show,
    )

    plot_unique_rate_zoom(
        algorithms=algorithms,
        result_root=result_root,
        epoch=args.epoch,
        seeds=args.seeds,
        output_path=Path(
            f'{args.output_prefix}_unique_rate_zoom.png'
        ),
        alpha=args.alpha,
        linewidth=args.linewidth,
        dpi=args.dpi,
        show=not args.no_show,
        ymin=args.zoom_ymin,
        ymax=args.zoom_ymax,
    )

    plot_losses(
        algorithms=algorithms,
        result_root=result_root,
        epoch=args.epoch,
        seeds=args.seeds,
        output_path=Path(
            f'{args.output_prefix}_loss.png'
        ),
        alpha=args.alpha,
        linewidth=args.linewidth,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
