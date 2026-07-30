"""
Module: VSLAM-LAB - Datasets - extra-files - sample_vpr.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-24
- Updated: 2026-07-25
- License: GPLv3 License

Downsamples a sequence's rgb.csv to at most --max-images frames, using the VPR distance matrix
D.npy written by `pixi run vpr` (<sequence>/vpr-lab/D.npy) to pick which frames to keep. Reuses
the greedy near-diagonal sampler from Baselines/VPR-LAB/vpr_sampler.py: walking forward from an
anchor frame i, a new keyframe is cut once D[i, j] exceeds a threshold th (higher th -> the walk
survives longer before cutting -> fewer keyframes). Since D is a distance matrix (0 = identical),
thresholds are swept from 0 up to --max-threshold (default 0.6), and the threshold whose
resulting count is the smallest value still >= --max-images
is applied - the "just exceeds" match, minimizing overshoot rather than jumping straight to
whatever undershoots first. th=0 always keeps every frame, so a count >= --max-images is always
reachable as long as --max-images is below the number of frames on hand; if the closest match is
still above --max-images (sweep too coarse), a warning suggests raising --n-thresholds or
--max-threshold. If --max-images is already >= the number of frames on hand, the sequence is left
untouched. The pre-sample rgb.csv is preserved as rgb_raw.csv next to the resampled one.

--interactive opens the same threshold-vs-count curve and original/filtered D-matrix heatmap
comparison as Baselines/VPR-LAB/vpr_sampler.py, adapted to write immediately rather than needing
a separate Save button: each click on the curve updates the heatmap in place for that threshold
and rewrites rgb.csv to that threshold's selection right away (rgb_raw.csv is still the untouched
backup - every interactive session re-starts from it, so clicking never compounds on a previous
session's filtered rgb.csv).

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
Add --revert to restore the rgb_raw.csv original instead of sampling.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_rgb_csv, raw_path, read_csv_rows, write_csv_rows,
    ensure_raw_backup, overwrite_csv_with_backup, revert_csv_from_backup,
)
from run_vpr import sequence_d_matrix

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def sample_indexes(D: np.ndarray, th: float) -> list[int]:
    """Greedy forward sampler (Baselines/VPR-LAB/vpr_sampler.py): walk from anchor i; when
    D[i, j] > th record j-1 as a keyframe and reset anchor to j."""
    indexes = [0]
    i = 0
    j = 0
    while i < D.shape[0] and j < D.shape[1]:
        while j < D.shape[1]:
            if D[i, j] > th:
                indexes.append(j - 1)
                i = j
            j += 1
    return indexes


def sweep_thresholds(D: np.ndarray, max_threshold: float, n_thresholds: int, verbose: bool) -> list[tuple[float, list[int]]]:
    thresholds = np.linspace(0.0, max_threshold, num=max(n_thresholds, 1))
    sweep = []
    for th in thresholds:
        indexes = sample_indexes(D, th)
        if verbose:
            print_info(f"  th={th:.4f} -> {len(indexes)} images")
        sweep.append((float(th), indexes))
    return sweep


def select_for_target(sweep: list[tuple[float, list[int]]], max_images: int) -> tuple[float, list[int]]:
    """The "just exceeds" match from a sweep_thresholds() sweep: the smallest frame count still
    >= max_images (minimizing overshoot), ties broken by the smallest (least aggressive)
    threshold. Assumes at least one candidate reaches max_images - guaranteed for a sweep whose
    th=0 entry covers every frame, as long as max_images doesn't exceed that count."""
    candidates = [(th, indexes) for th, indexes in sweep if len(indexes) >= max_images]
    return min(candidates, key=lambda pair: (len(pair[1]), pair[0]))


def load_D_matrix(dataset_name: str, sequence_name: str, d_matrix_path: Path, n_available: int) -> np.ndarray | None:
    D = np.load(d_matrix_path)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - {d_matrix_path} is not a square matrix (shape {D.shape})")
        return None
    if D.shape[0] != n_available:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - D matrix has {D.shape[0]} rows but rgb.csv has {n_available}")
        return None
    return D


def require_d_matrix(dataset_name: str, sequence_name: str, d_matrix_path: Path) -> bool:
    """True if d_matrix_path exists; otherwise prints the standard "run vpr first" warning and
    returns False. Shared by sample_pair and interactive_pair."""
    if not d_matrix_path.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing {d_matrix_path} (run 'pixi run vpr' first)")
        return False
    return True


def selected_rows(rows: list[list[str]], indexes: list[int]) -> list[list[str]]:
    """rows at indexes, deduplicated/sorted and clamped to rows' bounds. Shared by sample_pair
    and on_click (inside launch_interactive)."""
    valid_indexes = sorted({i for i in indexes if 0 <= i < len(rows)})
    return [rows[i] for i in valid_indexes]


def revert_pair(dataset_name: str, sequence_name: str) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)

    if not revert_csv_from_backup(rgb_csv):
        print_warning(f"Skipping {dataset_name}:{sequence_name} - nothing to revert (no rgb_raw.csv found)")
        return

    print_info(f"Reverted {dataset_name}:{sequence_name}")


def sample_pair(
    dataset_name: str, sequence_name: str, *,
    max_images: int, max_threshold: float, n_thresholds: int, verbose: bool
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    d_matrix_path = sequence_d_matrix(dataset_name, sequence_name)

    if not rgb_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    if rgb_raw.exists():
        print_info(f"Skipping {dataset_name}:{sequence_name} - already sampled (found rgb_raw.csv, use --revert first)")
        return

    if not require_d_matrix(dataset_name, sequence_name, d_matrix_path):
        return

    header, rows = read_csv_rows(rgb_csv)
    n_available = len(rows)

    if max_images >= n_available:
        print_info(f"Skipping {dataset_name}:{sequence_name} - max_images ({max_images}) >= available ({n_available}), nothing to do")
        return

    D = load_D_matrix(dataset_name, sequence_name, d_matrix_path, n_available)
    if D is None:
        return

    sweep = sweep_thresholds(D, max_threshold, n_thresholds, verbose)

    # th=0.0 always yields the full n_available count (D[i,i]=0 never exceeds a 0 threshold on
    # its own row), and n_available > max_images is already guaranteed above, so at least one
    # candidate with count >= max_images always exists - no "unreachable" fallback needed.
    chosen_th, chosen_indexes = select_for_target(sweep, max_images)

    new_rows = selected_rows(rows, chosen_indexes)

    overwrite_csv_with_backup(rgb_csv, header, new_rows)

    if len(new_rows) > max_images:
        print_warning(
            f"{dataset_name}:{sequence_name} - closest achievable count is {len(new_rows)}, "
            f"still above max_images={max_images} at th={chosen_th:.4f} "
            f"(raise --n-thresholds for a tighter match, or --max-threshold to search further)"
        )

    print_info(
        f"Sampled {dataset_name}:{sequence_name} - {len(new_rows)}/{n_available} images kept "
        f"(threshold={chosen_th:.4f})"
    )


def launch_interactive(
    dataset_name: str, sequence_name: str, D: np.ndarray,
    header: list[str], rows: list[list[str]], rgb_csv: Path,
    sweep: list[tuple[float, list[int]]]
) -> None:
    """Same interactive plots as Baselines/VPR-LAB/vpr_sampler.py's launch_interactive: a
    threshold-vs-count curve, and a side-by-side original/filtered D-matrix heatmap. Differs in
    that each click both updates the heatmap in place (rather than opening a new comparison
    window) and immediately rewrites rgb.csv for that threshold (rather than needing a separate
    Save button)."""
    import matplotlib.pyplot as plt
    ths = [th for th, _ in sweep]
    lengths = [len(idx) for _, idx in sweep]
    sweep_by_th = {th: idx for th, idx in sweep}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ths, lengths, marker="o", markersize=3, color="steelblue")
    ax.set_xlabel("Threshold (th)")
    ax.set_ylabel("Number of selected frames")
    ax.set_title(f"{dataset_name}:{sequence_name} - click to select a threshold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(D, cmap="viridis", aspect="auto")
    axes[0].set_title("Original D matrix")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    plt.colorbar(im0, ax=axes[0])

    # Create the filtered-view image/colorbar once (seeded with th=0's full-D view) and only
    # ever update its data/extent/color-limits in place afterward - recreating the colorbar on
    # every click (via axes[1].clear() + a fresh fig2.colorbar(...) call) either piles up extra
    # colorbars or raises a KeyError removing one, depending on matplotlib version quirks.
    initial_indexes = sweep[0][1]
    D_initial = D[np.ix_(initial_indexes, initial_indexes)]
    im1 = axes[1].imshow(D_initial, cmap="viridis", aspect="auto")
    axes[1].set_title("Click the curve to preview a threshold")
    axes[1].set_xlabel("j (subsampled)")
    axes[1].set_ylabel("i (subsampled)")
    cbar1 = fig2.colorbar(im1, ax=axes[1])
    fig2.tight_layout()

    state = {"vline": None}

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        nearest_th = min(sweep_by_th.keys(), key=lambda t: abs(t - event.xdata))
        indexes = sweep_by_th[nearest_th]

        if state["vline"] is not None:
            state["vline"].remove()
        state["vline"] = ax.axvline(nearest_th, color="crimson", linestyle="--", linewidth=1.2, label=f"th={nearest_th:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

        D_filtered = D[np.ix_(indexes, indexes)]
        n = D_filtered.shape[0]
        im1.set_data(D_filtered)
        im1.set_extent((-0.5, n - 0.5, n - 0.5, -0.5))
        im1.set_clim(vmin=float(D_filtered.min()), vmax=float(D_filtered.max()))
        axes[1].set_xlim(-0.5, n - 0.5)
        axes[1].set_ylim(n - 0.5, -0.5)
        axes[1].set_title(f"Filtered D matrix ({len(indexes)} / {D.shape[0]} frames)")
        cbar1.update_normal(im1)
        fig2.suptitle(f"Threshold = {nearest_th:.4f}  |  {len(indexes)} / {D.shape[0]} frames", fontsize=11)
        fig2.canvas.draw_idle()

        new_rows = selected_rows(rows, indexes)
        overwrite_csv_with_backup(rgb_csv, header, new_rows)
        print_info(f"{dataset_name}:{sequence_name} - th={nearest_th:.4f} -> {len(new_rows)} images written to rgb.csv")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig2.show()
    plt.show()


def interactive_pair(
    dataset_name: str, sequence_name: str, *,
    max_threshold: float, n_thresholds: int, verbose: bool
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    d_matrix_path = sequence_d_matrix(dataset_name, sequence_name)

    if not rgb_csv.exists() and not rgb_raw.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    if not require_d_matrix(dataset_name, sequence_name, d_matrix_path):
        return

    # Back up the untouched original once (unless rgb.csv itself is already missing and only the
    # backup remains, e.g. after external interference - nothing to back up from in that case);
    # every interactive session (re-)starts from the backup and resets rgb.csv to match, so
    # clicking always filters the full frame set rather than compounding on a previously-filtered
    # rgb.csv.
    if rgb_csv.exists():
        ensure_raw_backup(rgb_csv)
    header, rows = read_csv_rows(rgb_raw)
    write_csv_rows(rgb_csv, header, rows)
    n_available = len(rows)

    D = load_D_matrix(dataset_name, sequence_name, d_matrix_path, n_available)
    if D is None:
        return

    sweep = sweep_thresholds(D, max_threshold, n_thresholds, verbose)

    print_info(
        f"Opening interactive plot for {dataset_name}:{sequence_name} - click the curve to preview "
        f"and apply a threshold. Close the window to continue."
    )
    launch_interactive(dataset_name, sequence_name, D, header, rows, rgb_csv, sweep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample a sequence's rgb.csv to at most --max-images frames using its VPR D matrix."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to keep")
    parser.add_argument("--max-threshold", type=float, default=0.6, help="Threshold ceiling the sweep never goes above (default: 0.6)")
    parser.add_argument("--n-thresholds", type=int, default=50, help="Number of threshold steps to sweep (default: 50)")
    parser.add_argument("--revert", action="store_true", help="Restore rgb_raw.csv instead of sampling")
    parser.add_argument("--verbose", action="store_true", help="Print the threshold -> image count sweep")
    parser.add_argument("--interactive", action="store_true", help="Open the threshold-vs-count/heatmap plots; click to preview and immediately write rgb.csv for that threshold")
    args = parser.parse_args()

    if not args.revert and not args.interactive and args.max_images is None:
        parser.error("--max-images is required unless --revert or --interactive is given")

    pairs = resolve_sequence_targets_or_exit(args, parser)

    if args.revert:
        action = revert_pair
    elif args.interactive:
        action = functools.partial(
            interactive_pair,
            max_threshold=args.max_threshold, n_thresholds=args.n_thresholds, verbose=args.verbose
        )
    else:
        action = functools.partial(
            sample_pair,
            max_images=args.max_images, max_threshold=args.max_threshold,
            n_thresholds=args.n_thresholds, verbose=args.verbose
        )

    for dataset_name, sequence_name in pairs:
        action(dataset_name, sequence_name)


if __name__ == "__main__":
    main()
