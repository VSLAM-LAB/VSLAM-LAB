"""
Module: VSLAM-LAB - Capabilities - refrax.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-27
- Updated: 2026-08-28
- License: GPLv3 License

Removes flat-port refraction from the rgb_0 frames of one or more underwater dataset sequences
with Refrax (https://github.com/VSLAM-LAB/Refrax, the successor of refraction_simulator): every frame is
re-rendered as the image an in-air pinhole camera would have taken of a fronto-parallel scene
plane at depth --z0, using the physical housing model (port normal, glass/water refractive
indices, port distance and glass thickness) and the camera's own intrinsics. Two steps per
sequence, both from Refrax's code (imported from Baselines/Refrax):

  1. core.find_best_scale.find_best_scale - the correction zoom (isotropic scale about the
     principal point) that keeps the corrected image at the original magnification, swept over
     --zoom-bounds; --zoom auto (default) takes the unconstrained optimum, --zoom in-bounds the
     best zoom whose map keeps every valid pixel inside the original frame (what Refrax's own
     remove_refraction.py applies), --zoom <float> skips the search. The sampled zoom -> RMSE
     curve is written to zoom_sweep.csv.
  2. remove_refraction.RefractionCorrector - the per-frame remap. With cropping (crop_valid_bbox
     in the housing yaml, or --crop/--no-crop) every output is cropped to the largest all-valid
     rectangle so no black border remains and mask.png is all ones; with --no-crop the full
     frame is kept and mask.png (1 = valid pixel, 0 = outside the valid region, the mask2former
     convention) records the invalid border. mask.png is shared by every frame.

Camera parameters come from the sequence's calibration.yaml (rgb_0 entry: focal_length,
principal_point, image_dimension and, for radtan models, k1/k2/p1/p2 - Refrax applies
lens undistortion on top of the refraction map), never from a config file of the Refrax
repo; --calibration-yaml points at another yaml relative to the sequence folder (e.g.
anycalib/calibration.yaml). Refrax needs the camera's IN-AIR intrinsics, which is what
VSLAM-LAB calibration.yaml files hold (--intrinsics in-air, the default: focal lengths used as
they are, so corrected focal = zoom x calibrated focal). For a sequence whose calibration was
done underwater (a pinhole fit through the flat port, magnified by ~mu_w) pass --intrinsics
in-water: the focal lengths are divided by mu_w first, which is how Refrax's own
lizardisland config was derived (fx = 383.418 / 1.33). Housing parameters, z0, method and
cropping default to Refrax's configs/vslamlab.yaml (method:, housing: and correction: blocks -
edit that file to change them for every run), another Refrax config can be given with
--housing-yaml, and every value can be overridden per flag (--mu-w, --rflat, --z0, --method,
--crop/--no-crop, ...).

Output is a per-sequence artifact: <sequence>/refrax_0/<frame stem>.png (3-channel
PNG, one per rgb_0 frame), mask.png, zoom_sweep.csv, calibration.yaml and the
.refrax_complete marker. calibration.yaml is the sequence's calibration.yaml with
the rgb_0 entry replaced by the corrected camera - cam_model pinhole, focal_length scaled by the
zoom, principal_point shifted by the crop offset, image_dimension of the written PNGs, no
distortion fields - every other field and entry kept verbatim; the marker records every
parameter the run used (zoom, z0, crop box, housing, source intrinsics) as 'key: value' lines.
Frames whose PNG already exists are skipped, so an interrupted run resumes; a sequence whose
marker exists is skipped unless --overwrite is given. Neither rgb.csv nor calibration.yaml of
the sequence is ever modified: when an experiment sets 'refraction: refrax',
create_rgb_exp_csv (Run/run_functions.py) points path_rgb_0 of the per-experiment rgb_exp.csv at
the corrected PNGs, adds ts_mask_0 (ns)/path_mask_0 columns pointing every frame at mask.png
(the mask2former convention, so the validity mask is always available to the baseline) and
replaces the per-experiment calibration_exp.yaml with the artifact's calibration.yaml, calling
the 'refrax-inference' pixi task first if the artifact is missing. If a sample_vpr/synch_gt
rgb_raw.csv backup exists, frames are taken from that full pre-sampling list.

Only rgb_0 is corrected (a stereo rgb_1 stream is left untouched and the run pipeline warns).
Requires a pinhole rgb_0 entry with non-zero intrinsics (a 'cam_model: unknown' sequence must
be calibrated first, e.g. with 'pixi run calib-inference' and --calibration-yaml
anycalib/calibration.yaml) and radtan or no distortion; fisheye/equidistant models are not
supported, and radtan5's k3 is dropped with a warning (Refrax models k1, k2, p1, p2).

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, read_csv_rows, raw_path,
)

SIM_DIR = REPO_ROOT / "Baselines" / "Refrax"
FOLDER_BASE = "refrax"
COMPLETE_MARKER = ".refrax_complete"
MASK_FILE = "mask.png"
CALIBRATION_FILE = "calibration.yaml"
ZOOM_SWEEP_FILE = "zoom_sweep.csv"

# Housing/correction defaults come from Refrax's configs/vslamlab.yaml (method:, housing: and
# correction: z0_fixed/crop_valid_bbox - the camera comes from the sequence's calibration.yaml and
# the paths from its rgb.csv); the constants below are only the fallback when that file is missing.
DEFAULT_HOUSING_YAML = SIM_DIR / "configs" / "vslamlab.yaml"
DEFAULT_HOUSING = {"n_port": [0.0, 0.0, 1.0], "mu_a": 1.0, "mu_g": 1.47, "mu_w": 1.33, "rflat": 0.02, "tglass": 0.002}
DEFAULT_Z0 = 1.0
DEFAULT_ZOOM_BOUNDS = (1.0, 2.5)  # wider than find_best_scale.py's own (1.3, 1.55): a wide-FOV camera optimum can sit above 1.55
METHODS = ("closed_form", "newton")
INTRINSICS_MODES = ("in-water", "in-air")
SUPPORTED_DISTORTION = {None, "none", "radtan4", "radtan5"}

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def import_simulator():
    """Refrax is pure Python: import it straight from the git-clone."""
    if not SIM_DIR.exists():
        print_warning(f"{SIM_DIR} not found - run 'pixi run -e refrax install' first")
        sys.exit(1)
    sys.path.insert(0, str(SIM_DIR))
    from core.find_best_scale import find_best_scale  # noqa: E402
    from remove_refraction import RefractionCorrector  # noqa: E402
    return find_best_scale, RefractionCorrector


def load_camera(dataset_name: str, sequence_name: str, calib_yaml: Path, intrinsics: str, mu_w: float):
    """(camera dict for Refrax, calibration file lines, source description) from the
    rgb_0 entry of calib_yaml, or None (with a warning) if the entry is unusable."""
    if not calib_yaml.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing {calib_yaml}")
        return None
    lines = calib_yaml.read_text().splitlines()
    cameras = {cam.get("cam_name"): cam for cam in (yaml.safe_load("\n".join(lines)) or {}).get("cameras", [])}
    cam = cameras.get("rgb_0")
    if cam is None:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - {calib_yaml.name} has no rgb_0 camera entry")
        return None

    fx, fy = (float(v) for v in cam.get("focal_length", [0.0, 0.0]))
    cx, cy = (float(v) for v in cam.get("principal_point", [0.0, 0.0]))
    if cam.get("cam_model") != "pinhole" or fx <= 0 or fy <= 0:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb_0 in {calib_yaml.name} is not a calibrated pinhole camera "
                      f"(cam_model={cam.get('cam_model')}, focal_length={cam.get('focal_length')}); calibrate it first "
                      f"(e.g. 'pixi run calib-inference' and --calibration-yaml anycalib/calibration.yaml)")
        return None
    dist_type = cam.get("distortion_type")
    if dist_type not in SUPPORTED_DISTORTION:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - distortion_type '{dist_type}' not supported (only radtan/none)")
        return None
    W, H = (int(v) for v in cam["image_dimension"])

    camera = {"W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    coeffs = [float(v) for v in cam.get("distortion_coefficients", [])] if dist_type in ("radtan4", "radtan5") else []
    if coeffs:
        camera.update({"k1": coeffs[0], "k2": coeffs[1], "p1": coeffs[2], "p2": coeffs[3]})
        if len(coeffs) > 4 and coeffs[4] != 0.0:
            print_warning(f"{dataset_name}:{sequence_name} - radtan5 k3={coeffs[4]} ignored (Refrax models k1, k2, p1, p2)")
    if intrinsics == "in-water":
        camera["fx"], camera["fy"] = fx / mu_w, fy / mu_w
    source = (f"{calib_yaml.name} rgb_0 ({intrinsics}): f=({fx:.2f}, {fy:.2f}) c=({cx:.2f}, {cy:.2f})"
              + (f" radtan={coeffs[:4]}" if coeffs else ""))
    return camera, lines, source


def choose_zoom(find_best_scale, camera: dict, housing: dict, z0: float, zoom_spec: str,
                bounds: tuple[float, float], sweep_csv: Path) -> tuple[float, str, float | None]:
    """(zoom, mode, rmse_px) per --zoom: a float is used as is; 'auto'/'in-bounds' run
    Refrax's scale search and write its zoom -> RMSE curve to sweep_csv."""
    try:
        return float(zoom_spec), "fixed", None
    except ValueError:
        pass
    result = find_best_scale(camera["W"], camera["H"], camera["fx"], camera["fy"], camera["cx"], camera["cy"],
                             housing["n_port"], housing["rflat"], housing["tglass"],
                             housing["mu_a"], housing["mu_g"], housing["mu_w"], z0, bounds=bounds)
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zoom", "rmse_px"])
        writer.writerows(zip((f"{s:.6f}" for s in result["s_values"]), (f"{r:.6f}" for r in result["rmse_values"])))

    if zoom_spec == "in-bounds":
        if result["zoom_in_bounds"] is None:
            print_warning(f"no zoom in {bounds} keeps the corrected image inside the frame; using the unconstrained optimum instead")
            zoom, rmse = result["zoom"], result["rmse"]
        else:
            zoom, rmse = result["zoom_in_bounds"], result["rmse_in_bounds"]
    else:
        zoom, rmse = result["zoom"], result["rmse"]
    span = bounds[1] - bounds[0]
    if min(zoom - bounds[0], bounds[1] - zoom) < 0.02 * span:
        print_warning(f"zoom {zoom:.4f} sits at the edge of --zoom-bounds {bounds}; widen the bounds to check it is a true optimum")
    return zoom, zoom_spec, rmse


def fmt_list(values) -> str:
    return "[" + ", ".join(f"{float(v):.6f}".rstrip("0").rstrip(".") or "0" for v in values) + "]"


def patch_calibration_lines(lines: list[str], corrected: dict, comment: str) -> bool:
    """Replace the rgb_0 entry's camera model with the corrected pinhole camera: cam_model,
    focal_length, principal_point and image_dimension rewritten (annotated), distortion lines
    removed. Line-based, so the hand-formatted flow style and every other entry survive.
    Returns False if there is no rgb_0 entry."""
    try:
        start = next(i for i, line in enumerate(lines) if "cam_name: rgb_0," in line or "cam_name: rgb_0}" in line)
        end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    except StopIteration:
        return False

    def find(key: str) -> int | None:
        return next((i for i in range(start, end + 1) if lines[i].lstrip().startswith(f"{key}:")), None)

    def prefix(idx: int | None) -> str:
        return lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())] if idx is not None else "     "

    replacements = {
        "cam_model": "pinhole",
        "focal_length": fmt_list([corrected["fx"], corrected["fy"]]),
        "principal_point": fmt_list([corrected["cx"], corrected["cy"]]),
        "image_dimension": f"[{corrected['W']}, {corrected['H']}]",
    }
    for key, value in replacements.items():
        idx = find(key)
        new_line = f"{prefix(idx)}{key}: {value}, # refrax: {comment if key == 'focal_length' else 'corrected'}"
        if idx is not None:
            lines[idx] = new_line
        else:
            anchor = find("cam_type") if key == "cam_model" else find("cam_model")
            lines.insert((anchor if anchor is not None else start) + 1, new_line)
            end += 1
    for key in ("distortion_type", "distortion_coefficients"):
        idx = find(key)
        if idx is not None:
            del lines[idx]
            end -= 1
    return True


def correct_pair(
    dataset_name: str, sequence_name: str, simulator, *,
    calibration_rel: str, intrinsics: str, housing: dict, z0: float, method: str,
    zoom_spec: str, zoom_bounds: tuple[float, float], crop: bool, folder_base: str, overwrite: bool,
    housing_source: str = "",
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    # Prefer the pre-sampling backup when one exists (sample_vpr/synch_gt convention): it holds
    # the full frame list, so the artifact covers every frame even after rgb.csv was downsampled.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    seq_path = sequence_path(dataset_name, sequence_name)
    out_dir = seq_path / f"{folder_base}_0"
    marker = out_dir / COMPLETE_MARKER
    if marker.exists() and not overwrite:
        print_info(f"Skipping {dataset_name}:{sequence_name} - {out_dir.name} already complete (use --overwrite to recompute)")
        return
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    header, rows = read_csv_rows(source_csv)
    if "path_rgb_0" not in header or not rows:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb.csv has no 'path_rgb_0' column or no frames")
        return
    if "path_rgb_1" in header:
        print_warning(f"{dataset_name}:{sequence_name} - only rgb_0 is corrected; the rgb_1 stream is left untouched")

    loaded = load_camera(dataset_name, sequence_name, seq_path / calibration_rel, intrinsics, housing["mu_w"])
    if loaded is None:
        return
    camera, calib_lines, source = loaded

    path_idx = header.index("path_rgb_0")
    missing = [row[path_idx] for row in rows if not (seq_path / row[path_idx]).exists()]
    if missing:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - {len(missing)}/{len(rows)} frames listed in {source_csv.name} are missing "
                      f"on disk (broken symlinks / unmounted drive?), e.g. {missing[0]}")
        return
    first = cv2.imread(str(seq_path / rows[0][path_idx]))
    if first is None:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - cannot read {rows[0][path_idx]}")
        return
    if first.shape[:2] != (camera["H"], camera["W"]):
        print_warning(f"Skipping {dataset_name}:{sequence_name} - image_dimension {[camera['W'], camera['H']]} in calibration does not "
                      f"match the frames ({first.shape[1]}x{first.shape[0]})")
        return

    find_best_scale, RefractionCorrector = simulator
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom, zoom_mode, zoom_rmse = choose_zoom(find_best_scale, camera, housing, z0, zoom_spec, zoom_bounds, out_dir / ZOOM_SWEEP_FILE)
    corrector = RefractionCorrector(camera, housing, method=method, z0=z0, zoom=zoom, crop_valid_bbox=crop)
    corrected = corrector.corrected_camera()
    print_info(f"{dataset_name}:{sequence_name} - {source}; in-air f=({camera['fx']:.2f}, {camera['fy']:.2f}); "
               f"zoom={zoom:.4f} ({zoom_mode}{'' if zoom_rmse is None else f', rmse={zoom_rmse:.2f} px'}), z0={z0:g} m, {method}; "
               f"corrected f=({corrected['fx']:.2f}, {corrected['fy']:.2f}) c=({corrected['cx']:.2f}, {corrected['cy']:.2f}) "
               f"{corrected['W']}x{corrected['H']}" + (f" (crop {[int(v) for v in corrector.crop_box]})" if crop else ""))

    # mask.png in the mask2former convention (1 = usable pixel), at the output geometry.
    mask = corrector.crop((corrector.mask > 0).astype(np.uint8))
    cv2.imwrite(str(out_dir / MASK_FILE), mask)

    written = skipped = 0
    for row in tqdm(rows, desc=f"{dataset_name}:{sequence_name} refraction"):
        src = seq_path / row[path_idx]
        out_png = out_dir / f"{src.stem}.png"
        if out_png.exists():  # resume support: never recompute an existing frame
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Could not read {src}")
        frame, _, _ = corrector.correct(img)
        cv2.imwrite(str(out_png), corrector.crop(frame))
        written += 1

    stamp = datetime.date.today().isoformat()
    comment = f"zoom {zoom:.4f} x in-air focal ({camera['fx']:.2f}, {camera['fy']:.2f}), z0={z0:g} m"
    if not patch_calibration_lines(calib_lines, corrected, comment):
        print_warning(f"{dataset_name}:{sequence_name} - could not patch the rgb_0 entry; no calibration.yaml written")
        return
    header_idx = next((i for i, line in enumerate(calib_lines) if line.strip() == "---"), -1) + 1
    calib_lines[header_idx:header_idx] = [
        f"# rgb_0: refrax-corrected ({stamp}) - frames in {out_dir.name}/ are re-rendered as an in-air pinhole camera",
        f"# (flat-port refraction removed, {method}, z0={z0:g} m, zoom={zoom:.4f}{', cropped to the valid region' if crop else ''});",
        f"# cam_model/focal_length/principal_point/image_dimension replaced accordingly, distortion removed. Source: {source}.",
        "# Every other field is the sequence's original calibration.",
    ]
    (out_dir / CALIBRATION_FILE).write_text("\n".join(calib_lines).rstrip("\n") + "\n")

    marker.write_text("".join(f"{key}: {value}\n" for key, value in [
        ("zoom", f"{zoom:.6f}"), ("zoom_mode", zoom_mode),
        ("zoom_rmse_px", "" if zoom_rmse is None else f"{zoom_rmse:.4f}"),
        ("zoom_bounds", fmt_list(zoom_bounds)), ("z0", f"{z0:g}"), ("method", method),
        ("crop", "true" if crop else "false"), ("crop_box", [int(v) for v in corrector.crop_box] if corrector.crop_box else "null"),
        ("image_dimension", f"[{corrected['W']}, {corrected['H']}]"),
        ("focal_length", fmt_list([corrected["fx"], corrected["fy"]])),
        ("principal_point", fmt_list([corrected["cx"], corrected["cy"]])),
        ("intrinsics", intrinsics), ("calibration_source", calibration_rel), ("housing_source", housing_source),
        ("source_focal_length_in_air", fmt_list([camera["fx"], camera["fy"]])),
        ("housing_n_port", fmt_list(housing["n_port"])), ("housing_mu_a", housing["mu_a"]), ("housing_mu_g", housing["mu_g"]),
        ("housing_mu_w", housing["mu_w"]), ("housing_rflat", housing["rflat"]), ("housing_tglass", housing["tglass"]),
    ]))
    print_info(f"{dataset_name}:{sequence_name} - wrote {written} corrected frames to {out_dir}"
               + (f" ({skipped} already existed)" if skipped else "") + f", {MASK_FILE}, {CALIBRATION_FILE}")


def resolve_housing(args: argparse.Namespace) -> dict:
    """Housing/correction settings: built-in fallbacks < the housing yaml (--housing-yaml, default
    Refrax's configs/vslamlab.yaml: method:, housing:, correction: z0_fixed / crop_valid_bbox)
    < explicit flags. Returns {housing, z0, method, crop, source}."""
    housing = dict(DEFAULT_HOUSING)
    settings = {"z0": DEFAULT_Z0, "method": "closed_form", "crop": True, "source": "built-in defaults"}
    housing_yaml = Path(args.housing_yaml) if args.housing_yaml else DEFAULT_HOUSING_YAML
    if housing_yaml.exists():
        cfg = yaml.safe_load(housing_yaml.read_text()) or {}
        housing.update({k: v for k, v in (cfg.get("housing") or {}).items() if k in housing})
        corr = cfg.get("correction") or {}
        settings["z0"] = corr.get("z0_fixed", settings["z0"])
        settings["crop"] = bool(corr.get("crop_valid_bbox", settings["crop"]))
        settings["method"] = cfg.get("method", settings["method"])
        settings["source"] = str(housing_yaml)
    elif args.housing_yaml:
        print_warning(f"--housing-yaml {housing_yaml} not found")
        sys.exit(1)
    for key in ("mu_a", "mu_g", "mu_w", "rflat", "tglass"):
        if getattr(args, key) is not None:
            housing[key] = getattr(args, key)
    if args.n_port is not None:
        housing["n_port"] = list(args.n_port)
    if args.z0 is not None:
        settings["z0"] = args.z0
    if args.method is not None:
        settings["method"] = args.method
    if args.crop is not None:
        settings["crop"] = args.crop
    housing["n_port"] = [float(v) for v in housing["n_port"]]
    settings["z0"] = float(settings["z0"])
    settings["housing"] = housing
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove flat-port refraction from a sequence's rgb_0 frames with Refrax."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--calibration-yaml", default=CALIBRATION_FILE, dest="calibration_yaml",
                        help="Calibration yaml to read rgb_0 from, relative to the sequence folder (default: calibration.yaml; "
                             "e.g. anycalib/calibration.yaml)")
    parser.add_argument("--intrinsics", choices=INTRINSICS_MODES, default="in-air",
                        help="What calibration.yaml's focal lengths are: the camera's in-air calibration (default), or an "
                             "underwater (flat-port magnified) calibration, divided by mu_w to get the in-air focal Refrax needs")
    parser.add_argument("--housing-yaml", default=None, dest="housing_yaml",
                        help="Refrax config yaml whose housing:, method: and correction: (z0_fixed, crop_valid_bbox) "
                             f"blocks set the defaults (default: {DEFAULT_HOUSING_YAML.relative_to(REPO_ROOT)})")
    parser.add_argument("--n-port", type=float, nargs=3, default=None, dest="n_port", metavar=("NX", "NY", "NZ"),
                        help=f"Flat-port normal in the camera frame (default: {DEFAULT_HOUSING['n_port']})")
    parser.add_argument("--mu-a", type=float, default=None, dest="mu_a", help=f"Refractive index of air (default: {DEFAULT_HOUSING['mu_a']})")
    parser.add_argument("--mu-g", type=float, default=None, dest="mu_g", help=f"Refractive index of the port glass (default: {DEFAULT_HOUSING['mu_g']})")
    parser.add_argument("--mu-w", type=float, default=None, dest="mu_w", help=f"Refractive index of water (default: {DEFAULT_HOUSING['mu_w']})")
    parser.add_argument("--rflat", type=float, default=None, help=f"Camera centre to port distance (m) (default: {DEFAULT_HOUSING['rflat']})")
    parser.add_argument("--tglass", type=float, default=None, help=f"Port glass thickness (m) (default: {DEFAULT_HOUSING['tglass']})")
    parser.add_argument("--z0", type=float, default=None, help="Scene depth (m) the correction map is built for (default: the housing yaml's z0_fixed)")
    parser.add_argument("--method", choices=METHODS, default=None, help="Map construction (default: the housing yaml's method)")
    parser.add_argument("--zoom", default="auto",
                        help="Correction zoom: 'auto' (find_best_scale optimum, default), 'in-bounds' (best zoom keeping the "
                             "whole valid image inside the frame) or a number")
    parser.add_argument("--zoom-bounds", type=float, nargs=2, default=DEFAULT_ZOOM_BOUNDS, dest="zoom_bounds", metavar=("MIN", "MAX"),
                        help=f"Zoom search interval for --zoom auto/in-bounds (default: {list(DEFAULT_ZOOM_BOUNDS)})")
    parser.add_argument("--crop", action="store_true", default=None, dest="crop",
                        help="Crop every output to the largest all-valid rectangle (default: the housing yaml's crop_valid_bbox)")
    parser.add_argument("--no-crop", action="store_false", dest="crop",
                        help="Keep the full frame (black outside the valid region, recorded in mask.png)")
    parser.add_argument("--folder-base", default=FOLDER_BASE, dest="folder_base",
                        help="Output folder prefix; frames are written to <base>_0 (default: refrax)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if the artifact already exists for a sequence")
    parser.add_argument("--prefetch", action="store_true",
                        help="Check the Refrax clone imports and exit - no sequence targets required "
                             "(used by the 'install' pixi task; there are no weights to download)")
    args = parser.parse_args()

    if args.prefetch:
        import_simulator()
        print_info(f"Refrax importable from {SIM_DIR}")
        return

    pairs = resolve_sequence_targets_or_exit(args, parser)
    settings = resolve_housing(args)
    print_info(f"housing/correction defaults from {settings['source']}: mu_w={settings['housing']['mu_w']}, rflat={settings['housing']['rflat']}, "
               f"tglass={settings['housing']['tglass']}, z0={settings['z0']:g} m, method={settings['method']}, crop={settings['crop']}")
    if args.zoom not in ("auto", "in-bounds"):
        try:
            float(args.zoom)
        except ValueError:
            parser.error(f"--zoom must be 'auto', 'in-bounds' or a number (got '{args.zoom}')")
    simulator = import_simulator()

    for dataset_name, sequence_name in pairs:
        correct_pair(
            dataset_name, sequence_name, simulator,
            calibration_rel=args.calibration_yaml, intrinsics=args.intrinsics, housing=settings["housing"], z0=settings["z0"],
            method=settings["method"], zoom_spec=args.zoom, zoom_bounds=tuple(args.zoom_bounds), crop=settings["crop"],
            folder_base=args.folder_base, overwrite=args.overwrite, housing_source=settings["source"],
        )


if __name__ == "__main__":
    main()
