"""
Module: VSLAM-LAB - Datasets - dataset_malaysia.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-12
- License: GPLv3 License
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import compute_scaled_size, make_printers, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

# The standardized folders create_rgb_folder produces - everything else inside a sequence folder
# is the user-placed raw camera folder (e.g. "p1_s01_C2"), whose name varies per sequence.
_STANDARD_FOLDERS = {"rgb_0", "rgb_1", "depth_0"}


class MalaysiaDataset(DatasetVSLAMLAB):
    """Malaysia coral-reef transect survey dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "malaysia") -> None:
        super().__init__(dataset_name)

        # All sequences are local (scalar in the yaml) - there is no remote source to fetch from.
        self.sequence_location = self.cfg["sequence_location"]

    def download_sequence_data(self, sequence_name: str) -> None:
        print_info(
            f"Sequence '{sequence_name}' is marked as 'local'. Please ensure the raw images are "
            f"available at {self.sequence_path(sequence_name)}."
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return
        raw_dir = self._raw_image_dir(sequence_name)

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
        # No calibration exists for these cameras - write the "unknown" block (zero intrinsics,
        # no distortion fields), same for every sequence.
        rgb = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [0.0, 0.0],
            "principal_point": [0.0, 0.0],
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
        # user-placed local data with no remote source - deleting it would permanently destroy the
        # only full-resolution copy, with no re-download to recover from.
        return

    def _raw_image_dir(self, sequence_name: str) -> Path:
        """The user-placed raw camera folder inside this sequence's folder - the only
        subdirectory that isn't part of the standardized layout."""
        sequence_path = self.sequence_path(sequence_name)
        for child in sorted(sequence_path.iterdir()):
            if child.is_dir() and child.name not in _STANDARD_FOLDERS:
                return child
        raise FileNotFoundError(
            f"No raw camera folder found in {sequence_path} - place this sequence's raw images "
            f"in a subfolder there (sequence marked as 'local')."
        )
