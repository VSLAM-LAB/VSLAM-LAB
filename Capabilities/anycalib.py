"""
Module: VSLAM-LAB - Capabilities - anycalib.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-25
- Updated: 2026-08-25
- License: GPLv3 License

Estimates the camera intrinsics of one or more dataset sequences with AnyCalib (Tirado-Garin &
Civera, "AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration",
https://github.com/javrtg/AnyCalib) and writes a complete calibration yaml as a per-sequence
artifact: <sequence>/anycalib/calibration.yaml, plus <sequence>/anycalib/estimates.csv with the
per-image estimates behind it. For every rgb stream in rgb.csv (path_rgb_0 -> rgb_0,
path_rgb_1 -> rgb_1, ...), --n-images frames evenly spread over the sequence are calibrated
independently (single-view, so each frame yields its own estimate) and the per-parameter median
is kept (robust to the odd failed frame; the spread is reported as a standard deviation).

The camera model AnyCalib fits follows what the sequence's calibration.yaml already declares for
that camera (CAM_ID_BY_DISTORTION): a pinhole/undistorted or 'unknown' camera is fitted as
'pinhole' with the 'anycalib_pinhole' weights; 'radtan4'/'radtan5' as Brown-Conrady 'radial:2'/
'radial:3' (radial terms only - the tangential p1/p2 slots are written as 0) and 'equid4' as
Kannala-Brandt 'kb:4', both with the 'anycalib_gen' weights. --cam-id/--model-id override that
for every stream (a forced pinhole fit on a camera that declares distortion leaves its original
distortion_type/distortion_coefficients in place - only the fitted fields are ever replaced).
Principal points are converted from AnyCalib's convention (origin at the
top-left corner of the top-left pixel) to the OpenCV one used throughout VSLAM-LAB (integer
coordinates at pixel centres) by subtracting 0.5.

The written file is the sequence's calibration.yaml with only focal_length, principal_point and
(for distorted models) distortion_coefficients replaced on each rgb_<i> entry - T_BS, fps,
image_dimension, depth fields and the imus block are kept verbatim - and every replaced line
carries a trailing '# anycalib: ...' comment, with a header comment recording the run. A
sequence without any calibration.yaml gets a minimal complete file instead (cam_model pinhole,
identity T_BS, fps from the rgb.csv timestamps). The sequence's own calibration.yaml is never
modified: the run pipeline swaps the artifact in for the per-experiment calibration_exp.yaml
when an experiment sets 'calibration: anycalib' (create_calibration_exp_yaml,
Run/run_functions.py), calling the 'calib-inference' pixi task first if the artifact is missing.
A sequence whose anycalib/calibration.yaml already exists is skipped unless --overwrite is given.

The AnyCalib repo is cloned to Baselines/AnyCalib by the feature's 'fetch-source' pixi task (it is
pure Python, so it is imported from there rather than pip-installed) and the pretrained weights
are fetched from the authors' GitHub release into the torch hub cache by --prefetch (the
'install' pixi task).

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, read_csv_rows, raw_path,
)
from Datasets.DatasetVSLAMLAB_calibration import _get_rgb_yaml_section

ANYCALIB_DIR = REPO_ROOT / "Baselines" / "AnyCalib"
ARTIFACT_FOLDER = "anycalib"
CALIBRATION_FILE = "calibration.yaml"
ESTIMATES_FILE = "estimates.csv"
MODEL_IDS = ("anycalib_pinhole", "anycalib_gen", "anycalib_dist", "anycalib_edit")
DEFAULT_N_IMAGES = 10

# VSLAM-LAB distortion_type -> (AnyCalib cam_id, weights). None = no distortion_type field (a
# pinhole/unknown camera). Kannala-Brandt and Brown-Conrady radial terms share OpenCV's
# conventions, so the fitted coefficients drop straight into distortion_coefficients.
CAM_ID_BY_DISTORTION: dict[str | None, tuple[str, str]] = {
    None: ("pinhole", "anycalib_pinhole"),
    "none": ("pinhole", "anycalib_pinhole"),
    "radtan4": ("radial:2", "anycalib_gen"),
    "radtan5": ("radial:3", "anycalib_gen"),
    "equid4": ("kb:4", "anycalib_gen"),
}

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def rgb_stream_indices(header: list[str]) -> list[int]:
    """Stream indices i for every path_rgb_<i> column in an rgb.csv header (mono: [0],
    stereo: [0, 1], ...). Same helper as mask2former.py's."""
    indices = []
    for col in header:
        suffix = col.removeprefix("path_rgb_")
        if col.startswith("path_rgb_") and suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def import_anycalib():
    """AnyCalib is pure Python: import it straight from the git-clone rather than pip-installing
    it into the environment. ANYCALIB_DIR goes to the front of sys.path on purpose: this file is
    itself importable as a top-level 'anycalib' module when run as a script (sys.path[0] is
    Capabilities/), so the upstream package must shadow it, not the other way round."""
    if not ANYCALIB_DIR.exists():
        print_warning(f"{ANYCALIB_DIR} not found - run 'pixi run -e anycalib install' first")
        sys.exit(1)
    sys.path.insert(0, str(ANYCALIB_DIR))
    from anycalib import AnyCalib  # noqa: E402
    return AnyCalib


def load_model(model_id: str, device: torch.device):
    AnyCalib = import_anycalib()
    return AnyCalib(model_id=model_id).to(device)


def select_indices(n_available: int, n_images: int) -> list[int]:
    """n_images indices evenly spread over [0, n_available) (all of them if fewer exist)."""
    if n_images >= n_available:
        return list(range(n_available))
    return sorted({int(round(i)) for i in np.linspace(0, n_available - 1, num=n_images)})


def cam_id_for(cam_entry: dict | None, cam_id_override: str | None, model_id_override: str | None) -> tuple[str, str]:
    """(cam_id, model_id) to fit a camera with: from its calibration.yaml entry's distortion_type
    (CAM_ID_BY_DISTORTION; unknown types fall back to pinhole with a warning), each overridable
    from the command line."""
    distortion_type = None if cam_entry is None else cam_entry.get("distortion_type")
    if distortion_type in CAM_ID_BY_DISTORTION:
        cam_id, model_id = CAM_ID_BY_DISTORTION[distortion_type]
    else:
        cam_name = "?" if cam_entry is None else cam_entry.get("cam_name", "?")
        print_warning(f"{cam_name}: distortion_type '{distortion_type}' has no AnyCalib camera model mapping; fitting 'pinhole' instead")
        cam_id, model_id = CAM_ID_BY_DISTORTION[None]
    return cam_id_override or cam_id, model_id_override or model_id


def estimate_image(model, device: torch.device, image_path: Path, cam_id: str) -> tuple[np.ndarray, bool]:
    """AnyCalib intrinsics for one image, already in the image's own pixel units
    (predict() undoes its internal resize/crop), plus the fit's success flag."""
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1) / 255
    output = model.predict(tensor, cam_id=cam_id)
    return output["intrinsics"].cpu().numpy().astype(float), bool(output["success"].item())


def intrinsics_to_fields(cam_id: str, intrinsics: np.ndarray) -> dict:
    """Split an AnyCalib intrinsics vector (fx, fy, cx, cy[, k...]) into VSLAM-LAB calibration
    fields. Principal point moves from AnyCalib's corner-origin convention to OpenCV's
    pixel-centre convention (-0.5). radtan slots are [k1, k2, p1, p2(, k3)]: AnyCalib's radial
    model has no tangential terms, so p1 = p2 = 0."""
    fx, fy, cx, cy = intrinsics[:4]
    fields = {"focal_length": [float(fx), float(fy)], "principal_point": [float(cx) - 0.5, float(cy) - 0.5]}
    k = [float(v) for v in intrinsics[4:]]
    family = cam_id.partition(":")[0]
    if family == "radial":
        k = k + [0.0] * (2 - len(k))  # radial:1 -> k2 = 0
        fields["distortion_coefficients"] = [k[0], k[1], 0.0, 0.0] + k[2:3]  # radtan4, or radtan5 when k3 was fitted
    elif family == "kb":
        fields["distortion_coefficients"] = (k + [0.0] * 4)[:4]  # equid4
    elif k:
        fields["distortion_coefficients"] = k
    return fields


def fps_from_timestamps(rows: list[list[str]], header: list[str], stream: int) -> float:
    ts_col = header.index(f"ts_rgb_{stream} (ns)")
    ts = np.array([float(row[ts_col]) for row in rows])
    if len(ts) < 2:
        return 0.0
    dt = np.median(np.diff(ts)) / 1e9
    return round(1.0 / dt, 2) if dt > 0 else 0.0


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}".rstrip("0").rstrip(".") or "0" for v in values) + "]"


def annotate(line_prefix: str, key: str, values: list[float], comment: str) -> str:
    return f"{line_prefix}{key}: {fmt_list(values)}, # anycalib: {comment}"


def patch_calibration_lines(lines: list[str], cam_name: str, fields: dict, comment: str, short_comment: str = "") -> bool:
    """Replace focal_length/principal_point/distortion_coefficients on the cam_name entry of a
    VSLAM-LAB calibration yaml (line-based, so the hand-formatted flow style, comments and every
    other field survive). focal_length carries the full comment, the other patched lines the
    short one (or the full one when none is given). Returns False if the entry is missing."""
    short_comment = short_comment or comment
    try:
        start = next(i for i, line in enumerate(lines) if f"cam_name: {cam_name}," in line or f"cam_name: {cam_name}}}" in line)
        end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    except StopIteration:
        return False

    def find(key: str) -> int | None:
        return next((i for i in range(start, end + 1) if lines[i].lstrip().startswith(f"{key}:")), None)

    for key in ("focal_length", "principal_point"):
        idx = find(key)
        prefix = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())] if idx is not None else "     "
        new_line = annotate(prefix, key, fields[key], comment if key == "focal_length" else short_comment)
        if idx is not None:
            lines[idx] = new_line
        else:  # entry without the field (an 'unknown' camera never written with it): add after cam_model
            anchor = find("cam_model") or start
            lines.insert(anchor + 1, new_line)
            end += 1

    if "distortion_coefficients" in fields:
        idx = find("distortion_coefficients")
        new_line = annotate("     ", "distortion_coefficients", fields["distortion_coefficients"], short_comment)
        if idx is not None:
            lines[idx] = new_line
        else:
            anchor = find("principal_point")
            lines.insert(anchor + 1, new_line)
    return True


def minimal_calibration_lines(dataset_name: str, sequence_name: str, streams: list[int], header: list[str], rows: list[list[str]], seq_path: Path) -> list[str]:
    """A complete minimal calibration yaml for a sequence that has none: one pinhole rgb_<i>
    entry per stream with placeholder intrinsics (patched right after), identity T_BS and fps
    from the timestamps. Uses the same section writer as DatasetVSLAMLAB.write_calibration_yaml."""
    lines = ["%YAML 1.2", "---", "cameras:"]
    for stream in streams:
        first_image = seq_path / rows[0][header.index(f"path_rgb_{stream}")]
        with Image.open(first_image) as im:
            cam_type = "gray" if im.mode in ("L", "1", "I;16", "I") else "rgb"
        camera = {
            "cam_name": f"rgb_{stream}", "cam_type": cam_type, "cam_model": "pinhole",
            "focal_length": [0.0, 0.0], "principal_point": [0.0, 0.0],
            "fps": fps_from_timestamps(rows, header, stream), "T_BS": np.eye(4),
        }
        lines.extend(line for line in _get_rgb_yaml_section(camera, sequence_name, seq_path.parent) if line != "")
        lines.append("")
    return lines


def write_estimates_csv(path: Path, records: list[dict]) -> None:
    keys = ["cam_name", "image", "cam_id", "model_id", "success", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def calibrate_pair(
    dataset_name: str, sequence_name: str, model_loader: Callable[[str], object], device: torch.device, *,
    n_images: int, cam_id_override: str | None, model_id_override: str | None, overwrite: bool,
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    # Prefer the pre-sampling backup when one exists (sample_vpr/synch_gt convention): the full
    # frame list gives the evenly-spread sample the widest coverage.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    seq_path = sequence_path(dataset_name, sequence_name)
    out_dir = seq_path / ARTIFACT_FOLDER
    out_yaml = out_dir / CALIBRATION_FILE
    if out_yaml.exists() and not overwrite:
        print_info(f"Skipping {dataset_name}:{sequence_name} - {out_yaml.relative_to(seq_path)} already exists (use --overwrite to recompute)")
        return

    header, rows = read_csv_rows(source_csv)
    streams = rgb_stream_indices(header)
    if not streams or not rows:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb.csv has no 'path_rgb_<i>' columns or no frames")
        return

    calib_yaml = seq_path / "calibration.yaml"
    if calib_yaml.exists():
        lines = calib_yaml.read_text().splitlines()
        cameras = {cam.get("cam_name"): cam for cam in (yaml.safe_load("\n".join(lines)) or {}).get("cameras", [])}
    else:
        print_info(f"{dataset_name}:{sequence_name} - no calibration.yaml; writing a minimal pinhole one")
        lines = minimal_calibration_lines(dataset_name, sequence_name, streams, header, rows, seq_path)
        cameras = {}

    indices = select_indices(len(rows), n_images)
    records: list[dict] = []
    summary: list[str] = []
    for stream in streams:
        cam_name = f"rgb_{stream}"
        cam_id, model_id = cam_id_for(cameras.get(cam_name), cam_id_override, model_id_override)
        model = model_loader(model_id)
        path_idx = header.index(f"path_rgb_{stream}")

        estimates: list[np.ndarray] = []
        for i in tqdm(indices, desc=f"{dataset_name}:{sequence_name} {cam_name} ({cam_id})"):
            image_rel = rows[i][path_idx]
            intrinsics, success = estimate_image(model, device, seq_path / image_rel, cam_id)
            record = {"cam_name": cam_name, "image": image_rel, "cam_id": cam_id, "model_id": model_id, "success": success,
                      "fx": intrinsics[0], "fy": intrinsics[1], "cx": intrinsics[2], "cy": intrinsics[3]}
            record.update({f"k{j + 1}": v for j, v in enumerate(intrinsics[4:])})
            records.append(record)
            if success:
                estimates.append(intrinsics)

        if not estimates:
            print_warning(f"{dataset_name}:{sequence_name} {cam_name} - AnyCalib failed on every sampled image; not writing a calibration")
            out_dir.mkdir(parents=True, exist_ok=True)
            write_estimates_csv(out_dir / ESTIMATES_FILE, records)
            return

        stacked = np.stack(estimates)
        median, std = np.median(stacked, axis=0), stacked.std(axis=0)
        fields = intrinsics_to_fields(cam_id, median)
        comment = (f"{model_id}/{cam_id}, median of {len(estimates)}/{len(indices)} images "
                   f"(std fx={std[0]:.2f} fy={std[1]:.2f} cx={std[2]:.2f} cy={std[3]:.2f})")
        if not patch_calibration_lines(lines, cam_name, fields, comment, short_comment=f"{model_id}/{cam_id}"):
            print_warning(f"{dataset_name}:{sequence_name} - calibration.yaml has no '{cam_name}' entry; its estimate is only recorded in {ESTIMATES_FILE}")
            continue
        summary.append(f"{cam_name}: f=({fields['focal_length'][0]:.1f}, {fields['focal_length'][1]:.1f}) "
                       f"c=({fields['principal_point'][0]:.1f}, {fields['principal_point'][1]:.1f}) [{cam_id}]")

    stamp = datetime.date.today().isoformat()
    header_idx = next((i for i, line in enumerate(lines) if line.strip() == "---"), -1) + 1
    lines[header_idx:header_idx] = [
        f"# intrinsics: anycalib-generated ({stamp}) - focal_length/principal_point (and distortion_coefficients for distorted",
        f"# models) of every rgb_<i> entry replaced by the median AnyCalib estimate over {len(indices)} images; per-image",
        f"# estimates in {ARTIFACT_FOLDER}/{ESTIMATES_FILE}. Every other field is the sequence's original calibration.",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text("\n".join(lines).rstrip("\n") + "\n")
    write_estimates_csv(out_dir / ESTIMATES_FILE, records)
    print_info(f"{dataset_name}:{sequence_name} - wrote {out_yaml.relative_to(seq_path)}: " + "; ".join(summary))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate a sequence's camera intrinsics with AnyCalib and write <sequence>/anycalib/calibration.yaml."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES, dest="n_images",
                        help=f"Images per camera, evenly spread over the sequence, whose estimates are median-aggregated (default: {DEFAULT_N_IMAGES})")
    parser.add_argument("--cam-id", default=None, dest="cam_id",
                        help="AnyCalib camera model for every stream (pinhole, radial:k, kb:k, ...); default: derived from each camera's "
                             "distortion_type. A forced pinhole fit keeps a camera's original distortion fields untouched")
    parser.add_argument("--model-id", default=None, dest="model_id", choices=MODEL_IDS,
                        help="AnyCalib weights for every stream; default: anycalib_pinhole for pinhole fits, anycalib_gen for distorted ones")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if anycalib/calibration.yaml already exists for a sequence")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download/cache the AnyCalib weights (anycalib_pinhole and anycalib_gen) and exit - no "
                             "sequence targets required (used by the 'install' pixi task)")
    args = parser.parse_args()

    if args.prefetch:
        AnyCalib = import_anycalib()
        for model_id in ("anycalib_pinhole", "anycalib_gen"):
            AnyCalib(model_id=model_id)
        print_info(f"Prefetched AnyCalib weights to {torch.hub.get_dir()}/anycalib")
        return

    pairs = resolve_sequence_targets_or_exit(args, parser)

    device = torch.device(args.device)
    # Lazy and per-weights: a mono pinhole sequence only ever loads anycalib_pinhole.
    model_loader = functools.cache(lambda model_id: load_model(model_id, device))

    for dataset_name, sequence_name in pairs:
        calibrate_pair(
            dataset_name, sequence_name, model_loader, device,
            n_images=args.n_images, cam_id_override=args.cam_id, model_id_override=args.model_id, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
