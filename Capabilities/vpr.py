"""
Module: VSLAM-LAB - Capabilities - vpr.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5)
- Version: 2.0
- Created: 2026-07-23
- Updated: 2026-10-10
- License: GPLv3 License

VPR distance matrix capability: <sequence>/vpr-lab/D.npy (faiss squared L2 between the global descriptors of every
pair of rgb_0 frames, minimum over 0/90/180/270 degree rotations; D_<angle>.npy per rotation), the input of
rgb_vpr (Datasets/extra-files/sample_vpr.py) and rgb_placecell (Capabilities/placecell.py). The extraction code, its
environment and the weights live in the capability's repository (github.com/alejandrofontan/VPR-methods-evaluation,
a fork of gmberton/VPR-methods-evaluation with the vslamlab_vpr.py entry point, cloned to Capabilities/sources/vpr);
this driver runs in the vslamlab environment, resolves the sequence targets (CLAUDE.md's sequence-target argument
convention) into sequence folders and runs the capability's `inference` task on them:
  pixi run vpr <dataset> [<sequence> ...]
  pixi run vpr --datasets d1 d2 ... | --sequences d1 s1 s2 ... | --exp exp.yaml | --configs config.yaml

Sequences that already have a D.npy are skipped unless --overwrite. --method selects the VPR method (default:
megaloc). The Hugging Face token from path_constants is passed to the capability as HF_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from path_constants import HUGGINGFACE_TOKEN  # noqa: E402
from utilities import add_sequence_target_args, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

CAPABILITY = CapabilityVSLAMLAB("vpr", "alejandrofontan/VPR-methods-evaluation")
DEFAULT_METHOD = "megaloc"


def sequence_vpr_dir(dataset_name: str, sequence_name: str) -> Path:
    """<sequence_path>/vpr-lab - where the capability writes D.npy and the per-rotation matrices."""
    return sequence_path(dataset_name, sequence_name) / "vpr-lab"


def sequence_d_matrix(dataset_name: str, sequence_name: str) -> Path:
    return sequence_vpr_dir(dataset_name, sequence_name) / "D.npy"


def compute_d_matrix(pairs: list[tuple[str, str]], extra_args: list[str] | None = None) -> None:
    """Run the capability on (dataset, sequence) pairs."""
    if HUGGINGFACE_TOKEN is not None:
        os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN  # inherited by the capability's pixi run
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the VPR distance matrix <sequence>/vpr-lab/D.npy (runs the VPR capability repository "
                    "in its own environment)."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--method", default=DEFAULT_METHOD, help=f"VPR method to run (default: {DEFAULT_METHOD})")
    parser.add_argument("--overwrite", action="store_true", help="Recompute D.npy even if it already exists for a sequence")
    parser.add_argument("--prefetch", action="store_true", help="Clone + install the capability and download its weights, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = [f"--method={args.method}"] + (["--overwrite"] if args.overwrite else [])
    compute_d_matrix(resolve_sequence_targets_or_exit(args, parser), extra_args)


if __name__ == "__main__":
    main()
