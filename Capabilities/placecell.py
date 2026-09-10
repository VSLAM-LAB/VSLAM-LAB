"""
Module: VSLAM-LAB - Capabilities - placecell.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-09-10
- Updated: 2026-09-10
- License: GPLv3 License

Selects the N least redundant frames of a sequence with placecell's information culler
(Baselines/placecell, github.com/alejandrofontan/placecell) on the VPR distance matrix
written by `pixi run vpr` (<sequence>/vpr-lab/D.npy, faiss squared L2 between MegaLoc
descriptors). The matrix becomes a kernel-only PlaceCell store (S = 1 - D/2, symmetrised,
clipped to PSD, Pearson-centred), and the greedy joint-information culler removes, one at a
time, the alive frame with the least unique information given the other alive frames, until N
remain (CullParameters.target_alive). The first and last frames are always kept. Unlike
sample_vpr.py's forward walk (a threshold on consecutive distances), this ranks every frame
against the whole set, so revisited places are thinned as much as slow motion.

A missing D.npy is generated first with the vpr capability (`pixi run -e vpr-lab vpr`), so
nothing beyond the sequence itself is required.

Two modes:
- Standalone (like sample_vpr.py): `pixi run placecell-select <dataset> [<sequence> ...]
  --n-images N` rewrites the sequence's rgb.csv to the selection, keeping the original as
  rgb_raw.csv (`--revert` restores it). The removal order is written to
  <sequence>/placecell/rgb_placecell.csv.
- Experiment hook (Run/run_functions.py, `rgb_placecell: <n>` in an experiment yaml):
  `--indices <file> --out <csv>` selects among the given source-csv row indices (what
  survived rgb_idx/rgb_step/rgb_max) and writes the selection csv to --out without touching
  the sequence. The csv has one row per candidate frame: frame_idx (row of the csv D.npy was
  computed on), kept (1/0), removal_rank (1 = removed first, empty when kept) and
  unique_information (the frame's score when it was removed, NaN when kept).

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.

Cost: the cull is O(n^3) time and O(n^2) memory in the number of candidate frames (about 3 s and
a few hundred MB at 1180 frames, a minute and ~1 GB at 6000); above --warn-frames (5000) a
warning suggests pre-filtering with rgb_step.
"""

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as `python Capabilities/placecell.py`, sys.path[0] is Capabilities/ and `import placecell`
# would find this very script instead of the installed placecell package: drop it (everything
# from the repo is imported through REPO_ROOT, e.g. Capabilities.vpr).
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, raw_path, read_csv_rows, write_csv_rows,
    overwrite_csv_with_backup, revert_csv_from_backup,
)
from Capabilities.vpr import sequence_d_matrix

PLACECELL_FOLDER = "placecell"
SELECTION_CSV = "rgb_placecell.csv"
SELECTION_HEADER = ["frame_idx", "kept", "removal_rank", "unique_information"]
DEFAULT_WARN_FRAMES = 5000

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def import_placecell():
    """The compiled placecell module (nanobind), installed into the `placecell` pixi environment
    by `pixi run -e placecell install` (pip install of Baselines/placecell)."""
    try:
        import placecell
    except ImportError as e:
        print_warning(f"cannot import placecell ({e}); run 'pixi run -e placecell install' first")
        sys.exit(1)
    return placecell


def ensure_d_matrix(dataset_name: str, sequence_name: str, *, overwrite: bool = False) -> Path | None:
    """<sequence>/vpr-lab/D.npy, generated with the vpr capability when missing (or when
    overwrite is set). None if the matrix is still missing afterwards."""
    d_matrix_path = sequence_d_matrix(dataset_name, sequence_name)
    if not d_matrix_path.exists() or overwrite:
        print_info(f"{dataset_name}:{sequence_name} - {d_matrix_path.name} missing, running 'pixi run vpr' ...")
        cmd = ["pixi", "run", "-e", "vpr-lab", "vpr", dataset_name, sequence_name]
        if overwrite:
            cmd.append("--overwrite")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not d_matrix_path.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - 'pixi run vpr' did not produce {d_matrix_path}")
        return None
    return d_matrix_path


def load_d_submatrix(d_matrix_path: Path, indices: list[int], total_frames: int | None) -> np.ndarray | None:
    """The rows/cols `indices` of D.npy as float32, after checking the matrix is square and (when
    known) covers the frame list it was computed on."""
    D = np.load(d_matrix_path, mmap_mode="r")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        print_warning(f"{d_matrix_path} is not a square matrix (shape {D.shape})")
        return None
    if total_frames is not None and D.shape[0] != total_frames:
        print_warning(f"{d_matrix_path} has {D.shape[0]} rows but the frame list has {total_frames} "
                      f"(recompute it with 'pixi run vpr --overwrite' or revert a sampled rgb.csv first)")
        return None
    if indices and (min(indices) < 0 or max(indices) >= D.shape[0]):
        print_warning(f"frame indices {min(indices)}..{max(indices)} out of range for {d_matrix_path} ({D.shape[0]} rows)")
        return None
    return np.ascontiguousarray(D[np.ix_(indices, indices)], dtype=np.float32)


def select_frames(
    D_sub: np.ndarray, indices: list[int], n_images: int, *,
    centred: bool = True, clip: bool = True, verbosity: str = "warn",
) -> tuple[list[int], list[list]]:
    """Count-driven placecell cull of the frames `indices` (D_sub is D.npy restricted to them, in
    the same order) down to n_images. Returns (kept indices, selection rows following
    SELECTION_HEADER), both in frame order."""
    pc = import_placecell()
    n = len(indices)

    options = pc.PlaceCell.Options()
    options.verbosity = getattr(pc.LogLevel, verbosity)
    options.record = False
    options.name = "rgb_placecell"
    cell = pc.PlaceCell(options)

    kernel_options = pc.PlaceCell.KernelOptions()
    kernel_options.clip_to_psd = clip
    t0 = time.time()
    similarity = pc.similarity_from_distance(D_sub, "squared-euclidean")
    report = cell.set_kernel(similarity, indices, kernel_options)
    print_info(f"kernel: {report.views} frames, min eigenvalue {report.min_eigenvalue:.4f} "
               f"({report.negative_eigenvalues} negative{', clipped' if report.clipped else ''}), "
               f"max asymmetry {report.max_asymmetry:.3f}, {time.time() - t0:.1f} s")

    params = pc.PlaceCell.CullParameters()
    params.target_alive = n_images
    params.min_keyframes = 1
    params.centred = centred
    params.protect_first = True
    params.protect_last = 1
    params.max_per_call = 0
    t0 = time.time()
    cull = cell.cull_keyframes(params, lambda _frame: True)
    print_info(f"cull: removed {len(cull.culled)} of {n} frames in {time.time() - t0:.1f} s "
               f"({cull.alive_after} kept, {'centred' if centred else 'raw'} kernel)")
    if cull.alive_after > n_images:
        print_warning(f"culler stopped at {cull.alive_after} frames (> {n_images} requested); the kernel may be degenerate")

    rank = {view.id: (k + 1, view.unique_information) for k, view in enumerate(cull.culled)}
    kept, rows = [], []
    for idx in indices:
        if idx in rank:
            removal_rank, unique_information = rank[idx]
            rows.append([idx, 0, removal_rank, f"{unique_information:.6f}"])
        else:
            kept.append(idx)
            rows.append([idx, 1, "", "nan"])
    return kept, rows


def parse_indices_file(path: Path) -> list[int]:
    indices = sorted({int(line) for line in path.read_text().split() if line.strip()})
    return indices


def read_selection_csv(path: Path) -> list[int]:
    """Kept frame indices of a selection csv written by this script (used by Run/run_functions.py)."""
    header, rows = read_csv_rows(path)
    idx, kept = header.index("frame_idx"), header.index("kept")
    return sorted(int(row[idx]) for row in rows if row[kept] == "1")


def warn_if_large(n: int, warn_frames: int) -> None:
    if n > warn_frames:
        print_warning(f"{n} candidate frames: the placecell cull is O(n^3) in time and O(n^2) in memory "
                      f"(~{n * n * 8 / 1e9:.1f} GB for the inverse alone); consider pre-filtering with rgb_step")


def select_pair(
    dataset_name: str, sequence_name: str, *, n_images: int, indices_file: Path | None, out_csv: Path | None,
    centred: bool, clip: bool, verbosity: str, warn_frames: int, overwrite: bool,
) -> None:
    """Hook mode when indices_file/out_csv are given (select among those source-csv rows, write
    the selection csv, leave the sequence untouched); standalone mode otherwise (rewrite rgb.csv
    with an rgb_raw.csv backup)."""
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    hook_mode = indices_file is not None

    if not rgb_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return
    if not hook_mode and rgb_raw.exists():
        print_info(f"Skipping {dataset_name}:{sequence_name} - already sampled (found rgb_raw.csv, use --revert first)")
        return

    d_matrix_path = ensure_d_matrix(dataset_name, sequence_name, overwrite=overwrite)
    if d_matrix_path is None:
        sys.exit(1)

    if hook_mode:
        indices = parse_indices_file(indices_file)
        total_frames = None   # the caller's frame list may be a custom rgb_csv; run_functions checks the shape
    else:
        header, rows = read_csv_rows(rgb_csv)
        indices = list(range(len(rows)))
        total_frames = len(rows)

    n = len(indices)
    if n_images >= n:
        print_info(f"{dataset_name}:{sequence_name} - n_images ({n_images}) >= available ({n}), nothing to remove")
        kept, selection = indices, [[idx, 1, "", "nan"] for idx in indices]
    else:
        warn_if_large(n, warn_frames)
        D_sub = load_d_submatrix(d_matrix_path, indices, total_frames)
        if D_sub is None:
            sys.exit(1)
        kept, selection = select_frames(D_sub, indices, n_images, centred=centred, clip=clip, verbosity=verbosity)

    if hook_mode:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv_rows(out_csv, SELECTION_HEADER, selection)
        print_info(f"{dataset_name}:{sequence_name} - kept {len(kept)}/{n} frames -> {out_csv}")
        return

    selection_dir = sequence_path(dataset_name, sequence_name) / PLACECELL_FOLDER
    selection_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(selection_dir / SELECTION_CSV, SELECTION_HEADER, selection)
    if len(kept) < n:
        overwrite_csv_with_backup(rgb_csv, header, [rows[i] for i in kept])
    print_info(f"Sampled {dataset_name}:{sequence_name} - {len(kept)}/{n} images kept "
               f"(removal order in {selection_dir / SELECTION_CSV})")


def revert_pair(dataset_name: str, sequence_name: str) -> None:
    if not revert_csv_from_backup(sequence_rgb_csv(dataset_name, sequence_name)):
        print_warning(f"Skipping {dataset_name}:{sequence_name} - nothing to revert (no rgb_raw.csv found)")
        return
    print_info(f"Reverted {dataset_name}:{sequence_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the N least redundant frames of a sequence with placecell's information culler on its VPR matrix."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--n-images", type=int, default=None, dest="n_images", help="Number of frames to keep")
    parser.add_argument("--indices", type=Path, default=None,
                        help="Hook mode: file with the candidate source-csv row indices (one per line); requires --out")
    parser.add_argument("--out", type=Path, default=None, help="Hook mode: where to write the selection csv")
    parser.add_argument("--raw", action="store_true", help="Use the raw cosine kernel instead of the Pearson-centred one")
    parser.add_argument("--no-clip", action="store_true", dest="no_clip",
                        help="Do not clip the kernel to PSD (D.npy's rotation-min asymmetry makes it indefinite; clipping is needed for meaningful scores)")
    parser.add_argument("--verbosity", default="warn", choices=["off", "error", "warn", "info", "debug", "trace"],
                        help="placecell log level (default: warn)")
    parser.add_argument("--warn-frames", type=int, default=DEFAULT_WARN_FRAMES, dest="warn_frames",
                        help=f"Warn about the O(n^3) cost above this many candidate frames (default: {DEFAULT_WARN_FRAMES})")
    parser.add_argument("--overwrite", action="store_true", help="Recompute D.npy with 'pixi run vpr --overwrite' first")
    parser.add_argument("--revert", action="store_true", help="Restore rgb_raw.csv instead of selecting (standalone mode)")
    parser.add_argument("--prefetch", action="store_true",
                        help="Check the placecell module imports and exit - no sequence targets required (used by the 'install' pixi task)")
    args = parser.parse_args()

    if args.prefetch:
        pc = import_placecell()
        print_info(f"placecell module OK ({pc.__file__})")
        return
    if (args.indices is None) != (args.out is None):
        parser.error("--indices and --out go together")
    if not args.revert and args.n_images is None:
        parser.error("--n-images is required unless --revert is given")
    if args.n_images is not None and args.n_images < 2:
        parser.error("--n-images must be at least 2 (the first and last frames are always kept)")

    pairs = resolve_sequence_targets_or_exit(args, parser)
    if args.indices is not None and len(pairs) != 1:
        parser.error("hook mode (--indices/--out) takes exactly one dataset:sequence")

    if args.revert:
        action = revert_pair
    else:
        action = functools.partial(
            select_pair, n_images=args.n_images, indices_file=args.indices, out_csv=args.out,
            centred=not args.raw, clip=not args.no_clip, verbosity=args.verbosity,
            warn_frames=args.warn_frames, overwrite=args.overwrite,
        )
    for dataset_name, sequence_name in pairs:
        action(dataset_name, sequence_name)


if __name__ == "__main__":
    main()
