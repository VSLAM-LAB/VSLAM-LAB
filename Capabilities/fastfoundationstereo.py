"""
Module: VSLAM-LAB - Capabilities - fastfoundationstereo.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 2.0
- Created: 2026-08-12
- Updated: 2026-10-10
- License: GPLv3 License

Stereo depth capability (Fast-FoundationStereo, NVlabs CVPR 2026): per-frame metric depth for the rgb_0/rgb_1 pairs
of one or more sequences. The model code, its environment and the checkpoint live in the capability's repository
(github.com/VSLAM-LAB/fastfoundationstereo, cloned to Capabilities/sources/fastfoundationstereo); this driver runs
in the vslamlab environment, resolves the sequence targets (CLAUDE.md's sequence-target argument convention) into
sequence folders and runs the capability's `inference` task on them.

Artifact (inside each sequence folder): fastfoundationstereo_0/<rgb_0 frame stem>.png, 16-bit, depth (m) =
pixel / depth_factor (default 256), 0 = invalid, plus the fastfoundationstereo_0/.fastfoundationstereo_complete
marker holding the depth_factor. Frames already written are skipped (resume); complete sequences are skipped unless
--overwrite. Neither rgb.csv nor calibration.yaml is modified: when an experiment sets 'depth: fastfoundationstereo',
Run/run_functions.py appends the depth columns to its rgb_exp.csv (running this capability first if the marker is
missing) and registers the depth stream in calibration_exp.yaml. Once a sequence has depth, the dataset's
Datasets/dataset_files/dataset_<name>.yaml modes gain 'rgbd' (and 'rgbd-vi' with 'mono-vi') so rgbd experiments
validate. Requires a stereo calibration.yaml with two pinhole cameras (rgb_0, rgb_1), radtan distortion or none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from path_constants import VSLAM_LAB_DIR  # noqa: E402
from utilities import add_sequence_target_args, make_printers, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

DEPTH_FOLDER_BASE = "fastfoundationstereo"
COMPLETE_MARKER = ".fastfoundationstereo_complete"
CAPABILITY = CapabilityVSLAMLAB("fastfoundationstereo", "VSLAM-LAB/fastfoundationstereo")

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def update_dataset_modes(dataset_name: str) -> None:
    """Once generated depth exists for a sequence, the dataset can run rgbd experiments: add 'rgbd' to the modes
    list in Datasets/dataset_files/dataset_<name>.yaml, plus 'rgbd-vi' when the dataset already supports 'mono-vi'
    (imu present). Follows the existing ordering convention ('rgbd' after 'mono', 'rgbd-vi' after 'mono-vi');
    line-based and idempotent."""
    dataset_yaml = VSLAM_LAB_DIR / "Datasets" / "dataset_files" / f"dataset_{dataset_name}.yaml"
    if not dataset_yaml.exists():
        print_warning(f"{dataset_yaml} not found; cannot add 'rgbd' to the dataset modes")
        return

    lines = dataset_yaml.read_text().splitlines()
    idx = next((i for i, line in enumerate(lines) if line.startswith("modes:")), None)
    modes = yaml.safe_load(lines[idx].split("modes:", 1)[1]) if idx is not None else None
    if not isinstance(modes, list):
        print_warning(f"no parseable 'modes:' list in {dataset_yaml.name}; cannot add 'rgbd'")
        return

    def insert_after(mode: str, anchor: str) -> None:
        if mode not in new_modes:
            pos = new_modes.index(anchor) + 1 if anchor in new_modes else len(new_modes)
            new_modes.insert(pos, mode)

    new_modes = list(modes)
    insert_after("rgbd", "mono")
    if "mono-vi" in new_modes:
        insert_after("rgbd-vi", "mono-vi")
    if new_modes == modes:
        return  # already up to date

    lines[idx] = "modes: [" + ", ".join(f"'{mode}'" for mode in new_modes) + "]"
    dataset_yaml.write_text("\n".join(lines) + "\n")
    print_info(f"{dataset_name} - modes updated to {new_modes} in {dataset_yaml.name}")


def generate_stereo_depth(pairs: list[tuple[str, str]], extra_args: list[str] | None = None,
                          depth_folder_base: str = DEPTH_FOLDER_BASE) -> None:
    """Run the capability on (dataset, sequence) pairs, then add 'rgbd' to the modes of every dataset that now has
    complete depth for one of them."""
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)
    for dataset in sorted({dataset for dataset, _ in pairs}):
        if any((sequence_path(dataset, sequence) / f"{depth_folder_base}_0" / COMPLETE_MARKER).exists()
               for d, sequence in pairs if d == dataset):
            update_dataset_modes(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame metric depth (16-bit PNG) from rgb_0/rgb_1 stereo pairs with Fast-FoundationStereo "
                    "(runs github.com/VSLAM-LAB/fastfoundationstereo in its own environment)."
    )
    add_sequence_target_args(parser)
    # Forwarded to the capability's entry point (vslamlab_fastfoundationstereo.py), only when given
    parser.add_argument("--checkpoint", help="Checkpoint path relative to the capability's weights/ folder")
    parser.add_argument("--device")
    parser.add_argument("--valid_iters", type=int, help="Refinement iterations (lower = faster, slightly less accurate)")
    parser.add_argument("--max_disp", type=int, help="Maximum disparity for volume encoding")
    parser.add_argument("--depth-factor", type=float, dest="depth_factor", help="Depth (m) = png_value / depth_factor (default 256)")
    parser.add_argument("--zfar", type=float, help="Depth beyond this range (m) is stored as invalid (default 100)")
    parser.add_argument("--depth-folder-base", default=DEPTH_FOLDER_BASE, dest="depth_folder_base",
                        help="Depth folder prefix; depth is written to <base>_0 (default: fastfoundationstereo)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute depth even if it already exists")
    parser.add_argument("--prefetch", action="store_true",
                        help="Clone + install the capability and download its checkpoint, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = [f"--{flag}={value}" for flag, value in (
        ("checkpoint", args.checkpoint), ("device", args.device), ("valid_iters", args.valid_iters),
        ("max_disp", args.max_disp), ("depth-factor", args.depth_factor), ("zfar", args.zfar),
        ("depth-folder-base", args.depth_folder_base)) if value is not None]
    if args.overwrite:
        extra_args.append("--overwrite")

    pairs = resolve_sequence_targets_or_exit(args, parser)
    generate_stereo_depth(pairs, extra_args, args.depth_folder_base)


if __name__ == "__main__":
    main()
