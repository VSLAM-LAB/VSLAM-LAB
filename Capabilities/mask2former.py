"""
Module: VSLAM-LAB - Capabilities - mask2former.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5, Fable 5)
- Version: 2.0
- Created: 2026-08-08
- Updated: 2026-10-10
- License: GPLv3 License

Static/dynamic mask capability (Mask2Former, HuggingFace transformers, COCO-panoptic checkpoint): a per-frame binary
mask for every rgb frame of one or more sequences, pixels of movable "thing" classes (person, vehicles, animals)
flagged dynamic. The model code, its environment and the checkpoint live in the capability's repository
(github.com/VSLAM-LAB/mask2former, cloned to Capabilities/sources/mask2former); this driver runs in the vslamlab
environment, resolves the sequence targets (CLAUDE.md's sequence-target argument convention) into sequence folders
and runs the capability's `inference` task on them.

Artifact (inside each sequence folder): one mask folder per rgb stream (path_rgb_<i> -> mask2former_<i>/, 8-bit L
PNGs, 1 = static, 0 = dynamic, same filename as the source frame) with a .mask2former_complete marker; complete
streams are skipped unless --overwrite. rgb.csv is never modified: when an experiment sets 'segmentation:
mask2former', Run/run_functions.py appends ts_mask_<i> (ns)/path_mask_<i> columns to its rgb_exp.csv, running this
capability first if masks are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from utilities import add_sequence_target_args, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

CAPABILITY = CapabilityVSLAMLAB("mask2former", "VSLAM-LAB/mask2former")


def generate_masks(pairs: list[tuple[str, str]], extra_args: list[str] | None = None) -> None:
    """Run the capability on (dataset, sequence) pairs."""
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame static(1)/dynamic(0) semantic masks with Mask2Former "
                    "(runs github.com/VSLAM-LAB/mask2former in its own environment)."
    )
    add_sequence_target_args(parser)
    # Forwarded to the capability's entry point (vslamlab_mask2former.py), only when given
    parser.add_argument("--model_id", help="HuggingFace Mask2Former checkpoint id")
    parser.add_argument("--device")
    parser.add_argument("--mask-folder-base", dest="mask_folder_base",
                        help="Mask folder prefix; each rgb stream i writes to <base>_<i> (default: mask2former)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute masks even if they already exist")
    parser.add_argument("--prefetch", action="store_true",
                        help="Clone + install the capability and download its checkpoint, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = [f"--{flag}={value}" for flag, value in (
        ("model_id", args.model_id), ("device", args.device), ("mask-folder-base", args.mask_folder_base))
        if value is not None]
    if args.overwrite:
        extra_args.append("--overwrite")

    generate_masks(resolve_sequence_targets_or_exit(args, parser), extra_args)


if __name__ == "__main__":
    main()
