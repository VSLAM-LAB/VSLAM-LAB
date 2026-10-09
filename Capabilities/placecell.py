"""
Module: VSLAM-LAB - Capabilities - placecell.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 2.0
- Created: 2026-09-10
- Updated: 2026-10-10
- License: GPLv3 License

Selects the N least redundant frames of a sequence with placecell's information culler on the VPR distance matrix
written by `pixi run vpr` (<sequence>/vpr-lab/D.npy, faiss squared L2 between MegaLoc descriptors). The greedy
joint-information culler removes, one at a time, the frame with the least unique information given the other
remaining frames, until N remain; the first and last frames are always kept. Unlike sample_vpr.py's forward walk (a
threshold on consecutive distances), this ranks every frame against the whole set, so revisited places are thinned as
much as slow motion. The selection itself runs in the capability's repository (github.com/VSLAM-LAB/placecell,
cloned to Capabilities/sources/placecell, with the placecell library as a conda package); this driver runs in the
vslamlab environment, resolves the sequence targets (CLAUDE.md's sequence-target argument convention), makes sure
D.npy exists (generated with the vpr capability when missing) and applies the selection.

Two modes:
- Standalone (like sample_vpr.py): `pixi run placecell-select <dataset> [<sequence> ...] --n-images N` rewrites the
  sequence's rgb.csv to the selection, keeping the original as rgb_raw.csv (`--revert` restores it). The removal
  order is written to <sequence>/placecell/rgb_placecell.csv.
- Experiment hook (`rgb_placecell: <n>` in an experiment yaml): Run/run_functions.py calls run_selection() on the
  source-csv rows that survived rgb_idx/rgb_step/rgb_max and writes the selection csv into the experiment folder,
  without touching the sequence. The csv has one row per candidate frame: frame_idx (row of the csv D.npy was
  computed on), kept (1/0), removal_rank (1 = removed first, empty when kept) and unique_information (the frame's
  score when it was removed, nan when kept).
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from Capabilities.vpr import sequence_d_matrix  # noqa: E402
from path_constants import VSLAM_LAB_DIR  # noqa: E402
from utilities import (  # noqa: E402
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, raw_path, read_csv_rows,
    overwrite_csv_with_backup, revert_csv_from_backup,
)

CAPABILITY = CapabilityVSLAMLAB("placecell", "VSLAM-LAB/placecell")
PLACECELL_FOLDER = "placecell"
SELECTION_CSV = "rgb_placecell.csv"

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def ensure_d_matrix(dataset_name: str, sequence_name: str, *, overwrite: bool = False) -> Path | None:
    """<sequence>/vpr-lab/D.npy, generated with the vpr capability when missing (or when
    overwrite is set). None if the matrix is still missing afterwards."""
    d_matrix_path = sequence_d_matrix(dataset_name, sequence_name)
    if not d_matrix_path.exists() or overwrite:
        print_info(f"{dataset_name}:{sequence_name} - {d_matrix_path.name} missing, running 'pixi run vpr' ...")
        cmd = ["pixi", "run", "-e", "vpr-lab", "vpr", dataset_name, sequence_name]
        if overwrite:
            cmd.append("--overwrite")
        subprocess.run(cmd, cwd=VSLAM_LAB_DIR, check=True)
    if not d_matrix_path.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - 'pixi run vpr' did not produce {d_matrix_path}")
        return None
    return d_matrix_path


def read_selection_csv(path: Path) -> list[int]:
    """Kept frame indices of a selection csv, in frame order."""
    header, rows = read_csv_rows(path)
    idx, kept = header.index("frame_idx"), header.index("kept")
    return sorted(int(row[idx]) for row in rows if row[kept] == "1")


def run_selection(d_matrix_path: Path, n_images: int, out_csv: Path, *, indices: list[int] | None = None,
                  total_frames: int | None = None, extra_args: list[str] | None = None) -> list[int]:
    """Run the capability on D.npy (restricted to `indices` when given) and return the kept frame indices; the
    selection csv stays at out_csv. Exits if the capability produced no selection."""
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.unlink(missing_ok=True)
    args = ["--d-matrix", str(d_matrix_path.resolve()), "--n-images", str(n_images), "--out", str(out_csv),
            *(extra_args or [])]
    indices_file = None
    if indices is not None:
        indices_file = out_csv.with_name(out_csv.stem + "_indices.txt")
        indices_file.write_text("\n".join(str(i) for i in indices) + "\n")
        args += ["--indices", str(indices_file)]
    if total_frames is not None:
        args += ["--total-frames", str(total_frames)]
    try:
        CAPABILITY.run_args(args)
    finally:
        if indices_file is not None:
            indices_file.unlink(missing_ok=True)
    if not out_csv.exists():
        print_warning(f"the placecell capability did not produce {out_csv} (see its output above)")
        sys.exit(1)
    return read_selection_csv(out_csv)


def select_pair(dataset_name: str, sequence_name: str, *, n_images: int, overwrite: bool, extra_args: list[str]) -> None:
    """Standalone mode: rewrite rgb.csv to the selection, with an rgb_raw.csv backup."""
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    if not rgb_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return
    if raw_path(rgb_csv).exists():
        print_info(f"Skipping {dataset_name}:{sequence_name} - already sampled (found rgb_raw.csv, use --revert first)")
        return

    d_matrix_path = ensure_d_matrix(dataset_name, sequence_name, overwrite=overwrite)
    if d_matrix_path is None:
        sys.exit(1)

    header, rows = read_csv_rows(rgb_csv)
    selection_csv = sequence_path(dataset_name, sequence_name) / PLACECELL_FOLDER / SELECTION_CSV
    kept = run_selection(d_matrix_path, n_images, selection_csv, total_frames=len(rows), extra_args=extra_args)
    if len(kept) < len(rows):
        overwrite_csv_with_backup(rgb_csv, header, [rows[i] for i in kept])
    print_info(f"Sampled {dataset_name}:{sequence_name} - {len(kept)}/{len(rows)} images kept "
               f"(removal order in {selection_csv})")


def revert_pair(dataset_name: str, sequence_name: str) -> None:
    if not revert_csv_from_backup(sequence_rgb_csv(dataset_name, sequence_name)):
        print_warning(f"Skipping {dataset_name}:{sequence_name} - nothing to revert (no rgb_raw.csv found)")
        return
    print_info(f"Reverted {dataset_name}:{sequence_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the N least redundant frames of a sequence with placecell's information culler on its VPR "
                    "matrix (runs github.com/VSLAM-LAB/placecell in its own environment)."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--n-images", type=int, default=None, dest="n_images", help="Number of frames to keep")
    # Forwarded to the capability's entry point (vslamlab_placecell.py), only when given
    parser.add_argument("--raw", action="store_true", help="Use the raw cosine kernel instead of the Pearson-centred one")
    parser.add_argument("--no-clip", action="store_true", dest="no_clip",
                        help="Do not clip the kernel to PSD (D.npy's rotation-min asymmetry makes it indefinite; "
                             "clipping is needed for meaningful scores)")
    parser.add_argument("--verbosity", choices=["off", "error", "warn", "info", "debug", "trace"],
                        help="placecell log level (default: warn)")
    parser.add_argument("--warn-frames", type=int, dest="warn_frames",
                        help="Warn about the O(n^3) cost above this many candidate frames (default: 5000)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute D.npy with 'pixi run vpr --overwrite' first")
    parser.add_argument("--revert", action="store_true", help="Restore rgb_raw.csv instead of selecting")
    parser.add_argument("--prefetch", action="store_true",
                        help="Clone + install the capability and check the placecell module imports, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return
    if not args.revert and args.n_images is None:
        parser.error("--n-images is required unless --revert is given")
    if args.n_images is not None and args.n_images < 2:
        parser.error("--n-images must be at least 2 (the first and last frames are always kept)")

    pairs = resolve_sequence_targets_or_exit(args, parser)
    if args.revert:
        action = revert_pair
    else:
        extra_args = [flag for flag, on in (("--raw", args.raw), ("--no-clip", args.no_clip)) if on]
        extra_args += [f"--{flag}={value}" for flag, value in (("verbosity", args.verbosity),
                                                                ("warn-frames", args.warn_frames)) if value is not None]
        action = functools.partial(select_pair, n_images=args.n_images, overwrite=args.overwrite, extra_args=extra_args)
    for dataset_name, sequence_name in pairs:
        action(dataset_name, sequence_name)


if __name__ == "__main__":
    main()
