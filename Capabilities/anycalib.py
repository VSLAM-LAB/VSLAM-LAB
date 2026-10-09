"""
Module: VSLAM-LAB - Capabilities - anycalib.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 2.0
- Created: 2026-08-25
- Updated: 2026-10-10
- License: GPLv3 License

Intrinsics estimation capability (AnyCalib, Tirado-Garin & Civera): estimates each rgb camera's intrinsics from a
median over --n-images evenly spread frames and writes <sequence>/anycalib/calibration.yaml (the sequence's
calibration.yaml with focal_length, principal_point and, for distorted models, distortion_coefficients replaced)
plus <sequence>/anycalib/estimates.csv. The model code, its environment and the weights live in the capability's
repository (github.com/VSLAM-LAB/anycalib, cloned to Capabilities/sources/anycalib); this driver runs in the
vslamlab environment, resolves the sequence targets (CLAUDE.md's sequence-target argument convention) into sequence
folders and runs the capability's `inference` task on them.

The sequence's own calibration.yaml is never modified: when an experiment sets 'calibration: anycalib',
create_calibration_exp_yaml (Run/run_functions.py) swaps the artifact in for the per-experiment calibration_exp.yaml,
running this capability first if the artifact is missing. Complete sequences are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from utilities import add_sequence_target_args, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

CAPABILITY = CapabilityVSLAMLAB("anycalib", "VSLAM-LAB/anycalib")
MODEL_IDS = ("anycalib_pinhole", "anycalib_gen", "anycalib_dist", "anycalib_edit")


def estimate_intrinsics(pairs: list[tuple[str, str]], extra_args: list[str] | None = None) -> None:
    """Run the capability on (dataset, sequence) pairs."""
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics with AnyCalib and write <sequence>/anycalib/calibration.yaml "
                    "(runs github.com/VSLAM-LAB/anycalib in its own environment)."
    )
    add_sequence_target_args(parser)
    # Forwarded to the capability's entry point (vslamlab_anycalib.py), only when given
    parser.add_argument("--n-images", type=int, dest="n_images",
                        help="Images per camera, evenly spread over the sequence, median-aggregated (default 10)")
    parser.add_argument("--cam-id", dest="cam_id",
                        help="AnyCalib camera model for every stream (pinhole, radial:k, kb:k, ...); default: from each "
                             "camera's distortion_type")
    parser.add_argument("--model-id", dest="model_id", choices=MODEL_IDS,
                        help="AnyCalib weights for every stream; default: anycalib_pinhole / anycalib_gen")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if anycalib/calibration.yaml exists")
    parser.add_argument("--prefetch", action="store_true",
                        help="Clone + install the capability and download its weights, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = [f"--{flag}={value}" for flag, value in (
        ("n-images", args.n_images), ("cam-id", args.cam_id), ("model-id", args.model_id), ("device", args.device))
        if value is not None]
    if args.overwrite:
        extra_args.append("--overwrite")

    estimate_intrinsics(resolve_sequence_targets_or_exit(args, parser), extra_args)


if __name__ == "__main__":
    main()
