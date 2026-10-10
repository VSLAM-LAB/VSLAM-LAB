"""
Module: VSLAM-LAB - Capabilities - refrax.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 2.0
- Created: 2026-08-27
- Updated: 2026-10-10
- License: GPLv3 License

Flat-port refraction removal capability (Refrax, github.com/cllim118/Refrax): re-renders the rgb_0 frames of
underwater sequences as an in-air pinhole camera of a fronto-parallel scene plane at depth z0, and writes
<sequence>/refrax_0/ (corrected PNGs, mask.png, zoom_sweep.csv, the corrected calibration.yaml and the
.refrax_complete marker). The correction code, its environment and the VSLAM-LAB defaults (configs/vslamlab.yaml:
housing, z0, fit_canvas, crop) live in the capability's repository (github.com/VSLAM-LAB/refrax, a fork of Refrax
with the vslamlab_refrax.py entry point, cloned to Capabilities/sources/refrax); this driver runs in the vslamlab
environment, resolves the sequence targets (CLAUDE.md's sequence-target argument convention) into sequence folders
and runs the capability's `inference` task on them.

The sequence's own rgb.csv and calibration.yaml are never modified: when an experiment sets 'refraction: refrax',
replace_rgb_with_refraction_corrected (Run/run_functions.py) points path_rgb_0 of the per-experiment rgb_exp.csv at
the corrected frames and swaps the artifact's calibration.yaml in, running this capability first if the artifact is
missing. Complete sequences are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Capabilities.CapabilityVSLAMLAB import CapabilityVSLAMLAB  # noqa: E402
from utilities import add_sequence_target_args, resolve_sequence_targets_or_exit, sequence_path  # noqa: E402

CAPABILITY = CapabilityVSLAMLAB("refrax", "VSLAM-LAB/refrax")
FOLDER_BASE = "refrax"
COMPLETE_MARKER = ".refrax_complete"


def remove_refraction(pairs: list[tuple[str, str]], extra_args: list[str] | None = None) -> None:
    """Run the capability on (dataset, sequence) pairs."""
    CAPABILITY.run([sequence_path(dataset, sequence) for dataset, sequence in pairs], extra_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove flat-port refraction from a sequence's rgb_0 frames with Refrax and write "
                    "<sequence>/refrax_0/ (runs github.com/VSLAM-LAB/refrax in its own environment)."
    )
    add_sequence_target_args(parser)
    # Forwarded to the capability's entry point (vslamlab_refrax.py), only when given; defaults in its
    # configs/vslamlab.yaml
    value_flags = {
        "calibration-yaml": "Calibration yaml to read rgb_0 from, relative to the sequence folder (e.g. anycalib/calibration.yaml)",
        "intrinsics": "'in-air' (default: the calibration is the camera's in-air one) or 'in-water' (divided by mu_w)",
        "housing-yaml": "Config yaml with housing:/correction: blocks (default: the capability's configs/vslamlab.yaml)",
        "mu-a": "Refractive index of air", "mu-g": "Refractive index of the port glass",
        "mu-w": "Refractive index of water", "rflat": "Camera centre to port distance (m)",
        "tglass": "Port glass thickness (m)", "z0": "Scene depth (m) the correction map is built for",
        "zoom": "'auto' (default), 'in-bounds' or a number",
        "folder-base": "Output folder prefix; frames are written to <base>_0 (default: refrax)",
    }
    for flag, help_text in value_flags.items():
        parser.add_argument(f"--{flag}", help=help_text)
    parser.add_argument("--n-port", nargs=3, metavar=("NX", "NY", "NZ"), help="Flat-port normal in the camera frame")
    parser.add_argument("--zoom-bounds", nargs=2, metavar=("MIN", "MAX"), help="Zoom search interval (default: 1.0 2.5)")
    toggles = {
        "fit-canvas": "Size the output canvas to the whole corrected image (default)",
        "no-fit-canvas": "Keep the source W x H (pair with --zoom in-bounds)",
        "crop": "Crop outputs to the largest all-valid rectangle",
        "no-crop": "Keep the full canvas, invalid border recorded in mask.png (default)",
        "overwrite": "Recompute even if the artifact already exists",
    }
    for flag, help_text in toggles.items():
        parser.add_argument(f"--{flag}", action="store_true", help=help_text)
    parser.add_argument("--prefetch", action="store_true", help="Clone + install the capability, then exit")
    args = parser.parse_args()

    if args.prefetch:
        CAPABILITY.install()
        return

    extra_args = []
    for flag in value_flags:
        value = getattr(args, flag.replace("-", "_"))
        if value is not None:
            extra_args.append(f"--{flag}={value}")
    for flag in ("n-port", "zoom-bounds"):
        values = getattr(args, flag.replace("-", "_"))
        if values is not None:
            extra_args += [f"--{flag}", *values]
    extra_args += [f"--{flag}" for flag in toggles if getattr(args, flag.replace("-", "_"))]

    remove_refraction(resolve_sequence_targets_or_exit(args, parser), extra_args)


if __name__ == "__main__":
    main()
