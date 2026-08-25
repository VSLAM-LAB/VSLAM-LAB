"""
Module: VSLAM-LAB - Datasets - dataset_malaysia_jul2026.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-25
- License: GPLv3 License
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

# The rig's three hardware-synchronized cameras, in rgb_0/rgb_1/rgb_2 order: camera _CAMERAS[i]
# feeds rgb_<i>. Their inter-camera extrinsics were never calibrated, so calibration.yaml writes
# identity T_BS for every camera and the dataset only advertises 'mono' (on rgb_0) - rgb_1/rgb_2
# ship alongside for whoever wants them, but no stereo mode is enabled.
_CAMERAS: Final[tuple[str, ...]] = ("C1", "C2", "C3")

# Per-camera in-air Kalibr calibration (pinhole + radtan), transcribed from the campaign drive's
# p1/calibration/inair/090726_land/C{1,2,3}/calibration-camchain.yaml. Kalibr's radtan
# distortion_coeffs are [k1, k2, p1, p2], which is exactly VSLAM-LAB's radtan4 order. All three
# cameras were calibrated at the same native resolution.
_NATIVE_RESOLUTION: Final[tuple[int, int]] = (1920, 1080)  # (width, height)
_CAMERA_CALIBRATIONS: Final[dict[str, dict[str, list[float]]]] = {
    "C1": {
        "intrinsics": [907.6609199937194, 908.1146818088715, 968.852080866057, 543.0943194128026],
        "distortion": [0.010140395172075422, -0.0021778688736958916, -0.0006327251419563954, 0.0008471120286764555],
    },
    "C2": {
        "intrinsics": [915.7048768182783, 913.7785219949275, 966.3340080340163, 534.9430927193351],
        "distortion": [0.01565595842427522, -0.012776673951956766, -0.0011445041718470593, 0.002501385149897236],
    },
    "C3": {
        "intrinsics": [911.3245005698712, 910.6204950810973, 952.926564193491, 539.2255166777223],
        "distortion": [0.008619144399533709, -0.0007203600415217688, 0.0010402769634429334, -0.0011355993006183648],
    },
}


# "<base>_comb" sequences: the base sequence's three cameras interleaved into one rgb_0 (c1, c2,
# c3 frames of each capture instant back to back), with C2's calibration. Built from the base sequence's already
# resized rgb_0/rgb_1/rgb_2 (per-file relative symlinks, named c<N>_<ts>.png), so requesting a
# *_comb sequence first pulls in its base sequence.
_COMB_SUFFIX: Final = "_comb"
_COMB_CALIBRATION_CAMERA: Final = "C2"


def _is_comb(sequence_name: str) -> bool:
    return sequence_name.endswith(_COMB_SUFFIX)


def _base_sequence(sequence_name: str) -> str:
    return sequence_name[: -len(_COMB_SUFFIX)]


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


class MalaysiaJul2026Dataset(DatasetVSLAMLAB):
    """Malaysia July 2026 coral-reef transect survey dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "malaysia-jul2026") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml) - there is no remote source to fetch from.
        self.sequence_location = self.cfg["sequence_location"]

    def download_sequence_data(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            # Pull in the base sequence (raw check + resize of all three cameras) first.
            self.download_sequence(_base_sequence(sequence_name))
            return

        missing = [self._raw_image_dir(sequence_name, cam) for cam in _CAMERAS]
        missing = [raw_dir for raw_dir in missing if not raw_dir.is_dir()]
        if not missing:
            return
        print_info(
            f"Sequence '{sequence_name}' is marked as 'local'. Please place (or symlink) its "
            f"synchronized raw camera folders at: " + ", ".join(str(p) for p in missing)
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        if _is_comb(sequence_name):
            self._create_comb_rgb_folder(sequence_name)
            return

        for cam_idx, cam in enumerate(_CAMERAS):
            rgb_dir = self._rgb_dir(sequence_name, cam_idx)
            if rgb_dir.exists():
                continue
            raw_dir = self._raw_image_dir(sequence_name, cam)
            if not raw_dir.is_dir():
                raise FileNotFoundError(
                    f"Raw frames for '{sequence_name}' camera {cam} not found at {raw_dir} (sequence marked as 'local')."
                )
            self._resize_folder(raw_dir, rgb_dir, desc=f"    resizing images ({cam} -> {rgb_dir.name})")

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return
        if _is_comb(sequence_name):
            self._create_comb_rgb_csv(sequence_name)
            return

        # Filenames are the real capture timestamps in nanoseconds (e.g. 1000038050210199798.png),
        # identical across the three hardware-synchronized cameras - so one row per common
        # timestamp, joined by filename stem. Any frame missing from one camera is dropped from
        # the row (and reported), rather than silently pairing frames from different moments.
        per_camera: list[dict[str, str]] = []
        for cam_idx in range(len(_CAMERAS)):
            rgb_dir = self._rgb_dir(sequence_name, cam_idx)
            per_camera.append({p.stem: f"{rgb_dir.name}/{p.name}" for p in _image_files(rgb_dir)})
        common = set(per_camera[0]).intersection(*per_camera[1:])
        for cam_idx, files in enumerate(per_camera):
            dropped = len(files) - len(common)
            if dropped:
                print_warning(f"{sequence_name}: rgb_{cam_idx} has {dropped} frames with no match in the other cameras - dropped")

        header: list[str] = []
        for cam_idx in range(len(_CAMERAS)):
            header += [f"ts_rgb_{cam_idx} (ns)", f"path_rgb_{cam_idx}"]
        rows = []
        for stem in sorted(common, key=int):
            row: list[Any] = []
            for files in per_camera:
                row += [int(stem), files[stem]]
            rows.append(row)
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # A *_comb sequence has a single rgb_0 (the three cameras interleaved) calibrated with C2.
        cameras = [_COMB_CALIBRATION_CAMERA] if _is_comb(sequence_name) else list(_CAMERAS)

        rgb: list[dict[str, Any]] = []
        for cam_idx, cam in enumerate(cameras):
            calibration = _CAMERA_CALIBRATIONS[cam]
            fx, fy, cx, cy = (float(v) for v in calibration["intrinsics"])

            # The intrinsics describe the native 1920x1080 frames; rgb_<i> is resized to
            # target_resolution, so rescale them to match. Guard against the declared native
            # size silently drifting from what create_rgb_folder actually resized from (#99).
            self._check_native_resolution(sequence_name, cam_idx)
            focal_length, principal_point = scale_intrinsics(
                (fx, fy), (cx, cy), _NATIVE_RESOLUTION, self.target_resolution
            )
            rgb.append({
                "cam_name": f"rgb_{cam_idx}",
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "distortion_type": "radtan4",
                "distortion_coefficients": [float(v) for v in calibration["distortion"]],
                "focal_length": focal_length,
                "principal_point": principal_point,
                "fps": float(self.rgb_hz),
                # Rig extrinsics are unknown (never calibrated): identity for every camera.
                "T_BS": np.eye(4),
            })
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # No groundtruth exists for this dataset - still write the file (header only, no rows)
        # rather than leaving it missing.
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        write_csv_rows(groundtruth_csv, ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], [])

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: the raw camera folders are
        # user-placed local data (or symlinks onto the campaign drive) with no remote source -
        # deleting them would permanently destroy the only full-resolution copy, with no
        # re-download to recover from.
        return

    def _create_comb_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return
        base = _base_sequence(sequence_name)
        rgb_path.mkdir(parents=True, exist_ok=True)
        for cam_idx in range(len(_CAMERAS)):
            base_rgb_dir = self._rgb_dir(base, cam_idx)
            if not base_rgb_dir.is_dir():
                raise FileNotFoundError(f"Base sequence folder {base_rgb_dir} missing - cannot build {sequence_name}")
            for frame in tqdm(_image_files(base_rgb_dir), desc=f"    linking {base_rgb_dir.name} -> rgb_0 (c{cam_idx + 1}_*)"):
                link = rgb_path / f"c{cam_idx + 1}_{frame.name}"
                # Relative target so the benchmark folder stays relocatable as a whole.
                os.symlink(os.path.relpath(frame, rgb_path), link)

    def _create_comb_rgb_csv(self, sequence_name: str) -> None:
        # Interleaved: the three cameras' frames of one capture instant follow each other
        # (c1, c2, c3), then the next instant. c1 keeps the real capture timestamp; c2/c3 are
        # placed 1/3 and 2/3 of the way towards the *next* capture instant, so the stream stays
        # strictly increasing while tracking the sequence's real frame spacing (the last instant
        # reuses the preceding gap). The original capture timestamp survives in the filename
        # (c<N>_<ts>.png).
        rgb_path = self.rgb_path(sequence_name)
        n_cams = len(_CAMERAS)
        by_ts: dict[int, dict[int, str]] = {}  # capture ts -> {camera index: filename}
        for frame in _image_files(rgb_path):
            cam_idx = int(frame.name[1]) - 1  # c<N>_...
            by_ts.setdefault(int(frame.stem.split("_", 1)[1]), {})[cam_idx] = frame.name

        instants = sorted(by_ts)
        default_gap = int(round(1e9 / self.rgb_hz))
        rows: list[list[Any]] = []
        for i, ts in enumerate(instants):
            if i + 1 < len(instants):
                gap = instants[i + 1] - ts
            elif i > 0:
                gap = ts - instants[i - 1]
            else:
                gap = default_gap
            for cam_idx in sorted(by_ts[ts]):
                rows.append([ts + cam_idx * gap // n_cams, f"{rgb_path.name}/{by_ts[ts][cam_idx]}"])
        write_csv_rows(self.rgb_csv_path(sequence_name), ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def _rgb_dir(self, sequence_name: str, cam_idx: int) -> Path:
        """rgb_0 via the base-class helper, rgb_1/rgb_2 alongside it."""
        if cam_idx == 0:
            return self.rgb_path(sequence_name)
        return self.sequence_path(sequence_name) / f"rgb_{cam_idx}"

    def _raw_image_dir(self, sequence_name: str, cam: str) -> Path:
        """The user-placed raw folder of one camera inside this sequence's folder, named after
        the source's own syncd folder: sequence 'p1_s01', camera 'C2' -> '<sequence_path>/p1_s01_C2'."""
        return self.sequence_path(sequence_name) / f"{sequence_name}_{cam}"

    def _resize_folder(self, raw_dir: Path, out_dir: Path, desc: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target_size = None
        init_size = None
        for file_path in tqdm(_image_files(raw_dir), desc=desc):
            if self.target_resolution is None:
                shutil.copy2(file_path, out_dir / file_path.name)
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = compute_scaled_size(img.size, self.target_resolution)

                if img.size != init_size:
                    print_warning(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(out_dir / file_path.name)

    def _check_native_resolution(self, sequence_name: str, cam_idx: int) -> None:
        """Warn if the first rgb_<i> frame's size disagrees with what _NATIVE_RESOLUTION resizes
        to - the embedded intrinsics would then be scaled for the wrong native size."""
        rgb_dir = self._rgb_dir(sequence_name, cam_idx)
        frames = _image_files(rgb_dir) if rgb_dir.is_dir() else []
        if not frames:
            return
        expected_size = compute_scaled_size(_NATIVE_RESOLUTION, self.target_resolution)
        with Image.open(frames[0]) as img:
            if img.size != expected_size:
                print_warning(
                    f"{rgb_dir.name}/{frames[0].name} is {img.size}, expected {expected_size} from native "
                    f"{_NATIVE_RESOLUTION} - calibration intrinsics may be scaled for the wrong resolution."
                )
