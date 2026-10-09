"""
Module: VSLAM-LAB - Capabilities - depth_anything.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2026-09-15
- Updated: 2026-10-10
- License: GPLv3 License

Monocular depth capability (Depth Anything 3, default the metric nested DA3NESTED-GIANT-LARGE-1.1 model): per-frame
metric depth for the rgb_0 frames of one or more sequences, batches of 8 frames processed jointly. The entry point
(vslamlab_depth_anything.py) lives in the da3 baselines' checkout (github.com/VSLAM-LAB/depthanything3,
Baselines/Depth-Anything-3) and runs in its environment; this driver runs in the vslamlab environment, resolves the
sequence targets (CLAUDE.md's sequence-target argument convention) into sequence folders and runs its
`depth-inference` task on them.

Artifact (inside each sequence folder): depth_anything_0/<rgb_0 frame stem>.png, 16-bit, depth (m) = pixel /
depth_factor (default 256), 0 = invalid, plus the depth_anything_0/.depth_anything_complete marker holding the
depth_factor. Frames already written are skipped (resume); complete sequences are skipped unless --overwrite.
Neither rgb.csv nor calibration.yaml is modified: when an experiment sets 'depth: depth_anything',
Run/run_functions.py appends the depth columns to its rgb_exp.csv (running this capability first if the marker is
missing) and registers the depth stream in calibration_exp.yaml. Once a sequence has depth, the dataset's modes gain
'rgbd' (and 'rgbd-vi' with 'mono-vi') so rgbd experiments validate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB, add_rgbd_modes  # noqa: E402
from path_constants import VSLAMLAB_BASELINES  # noqa: E402
from utilities import add_sequence_target_args, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

DEPTH_FOLDER_BASE = "depth_anything"
COMPLETE_MARKER = ".depth_anything_complete"
# Shares the da3 / da3-streaming baselines' checkout and environment
CAPABILITY = CapabilityVSLAMLAB("depth_anything", "VSLAM-LAB/depthanything3",
                                path=VSLAMLAB_BASELINES / "Depth-Anything-3", inference_task="depth-inference")


def generate_mono_depth(pairs: list[tuple[str, str]], extra_args: list[str] | None = None,
                        depth_folder_base: str = DEPTH_FOLDER_BASE) -> None:
    """Run the capability on (dataset, sequence) pairs, then add 'rgbd' to the modes of every dataset that now has
    complete depth for one of them."""
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)
    for dataset in sorted({dataset for dataset, _ in pairs}):
        if any((sequence_path(dataset, sequence) / f"{depth_folder_base}_0" / COMPLETE_MARKER).exists()
               for d, sequence in pairs if d == dataset):
            add_rgbd_modes(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame metric depth (16-bit PNG) for rgb_0 with Depth Anything 3 "
                    "(runs vslamlab_depth_anything.py in the da3 checkout's environment)."
    )
    add_sequence_target_args(parser)
    # Forwarded to the capability's entry point, only when given
    parser.add_argument("--model_id", help="Depth Anything 3 model id (default: DA3NESTED-GIANT-LARGE-1.1, metric)")
    parser.add_argument("--batch_size", type=int, help="Frames per DA3 forward pass, processed jointly (default 8)")
    parser.add_argument("--device")
    parser.add_argument("--depth-factor", type=float, dest="depth_factor", help="Depth (m) = png_value / depth_factor (default 256)")
    parser.add_argument("--depth-folder-base", default=DEPTH_FOLDER_BASE, dest="depth_folder_base",
                        help="Depth folder prefix; depth is written to <base>_0 (default: depth_anything)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute depth even if it already exists")
    parser.add_argument("--prefetch", action="store_true", help="Clone + install the checkout, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = [f"--{flag}={value}" for flag, value in (
        ("model_id", args.model_id), ("batch_size", args.batch_size), ("device", args.device),
        ("depth-factor", args.depth_factor), ("depth-folder-base", args.depth_folder_base)) if value is not None]
    if args.overwrite:
        extra_args.append("--overwrite")

    generate_mono_depth(resolve_sequence_targets_or_exit(args, parser), extra_args, args.depth_folder_base)


if __name__ == "__main__":
    main()
