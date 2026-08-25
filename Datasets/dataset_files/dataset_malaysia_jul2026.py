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

# Per-camera in-air Kalibr calibration (pinhole + radtan), transcribed from the campaign drive's
# p1/calibration/inair/090726_land/C{1,2,3}/calibration-camchain.yaml. Kalibr's radtan
# distortion_coeffs are [k1, k2, p1, p2], which is exactly VSLAM-LAB's radtan4 order. All three
# cameras were calibrated at the same native resolution.
_NATIVE_RESOLUTION: Final[tuple[int, int]] = (1920, 1080)  # (width, height)
_CAMERA_CALIBRATIONS: Final[dict[str, dict[str, list[float]]]] = {
    "c1": {
        "intrinsics": [907.6609199937194, 908.1146818088715, 968.852080866057, 543.0943194128026],
        "distortion": [0.010140395172075422, -0.0021778688736958916, -0.0006327251419563954, 0.0008471120286764555],
    },
    "c2": {
        "intrinsics": [915.7048768182783, 913.7785219949275, 966.3340080340163, 534.9430927193351],
        "distortion": [0.01565595842427522, -0.012776673951956766, -0.0011445041718470593, 0.002501385149897236],
    },
    "c3": {
        "intrinsics": [911.3245005698712, 910.6204950810973, 952.926564193491, 539.2255166777223],
        "distortion": [0.008619144399533709, -0.0007203600415217688, 0.0010402769634429334, -0.0011355993006183648],
    },
}


def _split_sequence_name(sequence_name: str) -> tuple[str, str]:
    """'p1_s01_c2' -> ('p1_s01', 'c2'): the survey prefix and the camera id."""
    survey, camera = sequence_name.rsplit("_", 1)
    return survey, camera


class MalaysiaJul2026Dataset(DatasetVSLAMLAB):
    """Malaysia July 2026 coral-reef transect survey dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "malaysia-jul2026") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml) - there is no remote source to fetch from.
        self.sequence_location = self.cfg["sequence_location"]

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_dir = self._raw_image_dir(sequence_name)
        if raw_dir.is_dir():
            return
        print_info(
            f"Sequence '{sequence_name}' is marked as 'local'. Please place (or symlink) its "
            f"synchronized raw frames at {raw_dir}."
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return
        raw_dir = self._raw_image_dir(sequence_name)
        if not raw_dir.is_dir():
            raise FileNotFoundError(
                f"Raw frames for '{sequence_name}' not found at {raw_dir} (sequence marked as 'local')."
            )

        rgb_path.mkdir(parents=True, exist_ok=True)
        target_size = None
        init_size = None
        for file_path in tqdm(sorted(raw_dir.iterdir()), desc="    resizing images"):
            if file_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue

            if self.target_resolution is None:
                shutil.copy2(file_path, rgb_path / file_path.name)
                continue

            with Image.open(file_path) as img:
                img.load()
                if target_size is None:
                    init_size = img.size
                    target_size = compute_scaled_size(img.size, self.target_resolution)

                if img.size != init_size:
                    print_warning(f"{file_path.name} {img.size} != {init_size}")

                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(rgb_path / file_path.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        # Filenames are the real capture timestamps in nanoseconds (e.g. 1000038050210199798.png),
        # so use them directly instead of synthesizing timestamps from rgb_hz.
        rows = sorted(
            [int(file_path.stem), f"rgb_0/{file_path.name}"]
            for file_path in rgb_path.iterdir()
            if file_path.is_file()
        )
        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        _, camera = _split_sequence_name(sequence_name)
        calibration = _CAMERA_CALIBRATIONS[camera]
        fx, fy, cx, cy = (float(v) for v in calibration["intrinsics"])

        # The intrinsics describe the native 1920x1080 frames; rgb_0 is resized to
        # target_resolution, so rescale them to match. Guard against the declared native size
        # silently drifting from what create_rgb_folder actually resized from (issue #99).
        self._check_native_resolution(sequence_name)
        focal_length, principal_point = scale_intrinsics((fx, fy), (cx, cy), _NATIVE_RESOLUTION, self.target_resolution)

        rgb: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(v) for v in calibration["distortion"]],
            "focal_length": focal_length,
            "principal_point": principal_point,
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # No groundtruth exists for this dataset - still write the file (header only, no rows)
        # rather than leaving it missing.
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        write_csv_rows(groundtruth_csv, ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], [])

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: the raw camera folder is
        # user-placed local data (or a symlink onto the campaign drive) with no remote source -
        # deleting it would permanently destroy the only full-resolution copy, with no re-download
        # to recover from.
        return

    def _raw_image_dir(self, sequence_name: str) -> Path:
        """The user-placed raw camera folder inside this sequence's folder, named after the
        source's own syncd folder: sequence 'p1_s01_c2' -> '<sequence_path>/p1_s01_C2'."""
        survey, camera = _split_sequence_name(sequence_name)
        return self.sequence_path(sequence_name) / f"{survey}_{camera.upper()}"

    def _check_native_resolution(self, sequence_name: str) -> None:
        """Warn if the first rgb_0 frame's size disagrees with what _NATIVE_RESOLUTION resizes
        to - the embedded intrinsics would then be scaled for the wrong native size."""
        rgb_path = self.rgb_path(sequence_name)
        first_frame = next((p for p in sorted(rgb_path.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES), None)
        if first_frame is None:
            return
        expected_size = compute_scaled_size(_NATIVE_RESOLUTION, self.target_resolution)
        with Image.open(first_frame) as img:
            if img.size != expected_size:
                print_warning(
                    f"{first_frame.name} is {img.size}, expected {expected_size} from native "
                    f"{_NATIVE_RESOLUTION} - calibration intrinsics may be scaled for the wrong resolution."
                )
