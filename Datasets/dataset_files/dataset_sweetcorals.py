"""
Module: VSLAM-LAB - Datasets - dataset_sweetcorals.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-07-22
- Updated: 2026-07-25
- License: GPLv3 License
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml
from PIL import Image

from Datasets.dataset_files.dataset_soneva import HFColmapDatasetMixin
from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import (
    ensure_hf_sequence_download, hf_token, read_colmap_cameras, read_colmap_images, world_to_camera_to_pose,
    write_csv_rows,
)

# Only tabuhan_p1 has been fully processed on the source (color-corrected pinhole images plus a
# colmap reconstruction with real poses) — every other sequence ships only raw, uncalibrated
# fisheye stills with no pose data.
_PINHOLE_SEQUENCE = "tabuhan_p1"

# tabuhan_p1's corrected/images/ folder merges both rig cameras into one flat directory
# (Left frames prefixed GPAA, Right frames GPAB/GPAC) — this prefix isolates the Left camera
# so mono stays a single, consistent viewpoint instead of jumping between two cameras.
_PINHOLE_LEFT_PREFIX = "GPAA"

# sequence_names are kept short (e.g. "tabuhan_p1"); every survey's actual top-level folder in
# the HF repo carries an "indonesia_" prefix and a "_YYYYMMDD" capture-date suffix that doesn't
# derive mechanically from the nickname, so it's kept as an explicit table. tabuhan_p1's folder
# also carries a stray leading underscore in the source repo.
_REMOTE_FOLDER = {
    "banyuwangi_farm": "indonesia_banyuwangi_farm_20250211",
    "pemuteran_p1": "indonesia_pemuteran_p1_20250213",
    "pemuteran_p2": "indonesia_pemuteran_p2_20250213",
    "pemuteran_p3": "indonesia_pemuteran_p3_20250213",
    "tabuhan_p1": "_indonesia_tabuhan_p1_20250210",
    "tabuhan_p2": "indonesia_tabuhan_p2_20250210",
    "tabuhan_p3": "indonesia_tabuhan_p3_20250210",
    "watudodol_p1": "indonesia_watudodol_p1_20250208",
    "watudodol_p2": "indonesia_watudodol_p2_20250208",
    "watudodol_p3": "indonesia_watudodol_p3_20250208",
    "watudodol_p4": "indonesia_watudodol_p4_20250209",
    "watudodol_p5": "indonesia_watudodol_p5_20250209",
    "watudodol_p6": "indonesia_watudodol_p6_20250209",
}

# Every survey (other than tabuhan_p1, handled separately above) is shot with a 2-camera
# (Left/Right) GoPro rig under raw/<tag>_Left|_Right/. mono uses the Left camera as the
# canonical single view, except watudodol_p2 which has no Left data at all and falls back to
# Right. watudodol_p1 also ships an extra continuation folder from a second day, concatenated
# after the main one.
_RAW_CAMERA_SUBFOLDERS = {
    "banyuwangi_farm": ["F1_Left"],
    "pemuteran_p1": ["B1_Left"],
    "pemuteran_p2": ["B2_Left"],
    "pemuteran_p3": ["B3_Left"],
    "tabuhan_p2": ["Q8_Left"],
    "tabuhan_p3": ["Q9_Left"],
    "watudodol_p1": ["Q1_Left", "Q1_Left_extra_20250209"],
    "watudodol_p2": ["Q2_Right"],
    "watudodol_p3": ["Q3_Left"],
    "watudodol_p4": ["Q4_Left"],
    "watudodol_p5": ["Q5_Left"],
    "watudodol_p6": ["Q6_Left"],
}


class SweetcoralsDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):
    """Sweet Corals dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "sweetcorals") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get Hugging Face repo id
        self.hf_repo_id = cfg["hf_repo_id"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = sequence_path / "rgb_0_raw"
        remote_folder = self._remote_sequence_name(sequence_name)

        if sequence_name == _PINHOLE_SEQUENCE:
            remote_dir = f"{remote_folder}/corrected/images"
            ensure_hf_sequence_download(
                self.hf_repo_id, [remote_dir], rgb_path, pattern=f"{_PINHOLE_LEFT_PREFIX}*", token=hf_token(),
            )
            return

        remote_dirs = [f"{remote_folder}/raw/{subfolder}" for subfolder in _RAW_CAMERA_SUBFOLDERS[sequence_name]]
        ensure_hf_sequence_download(self.hf_repo_id, remote_dirs, rgb_path, token=hf_token())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        if sequence_name == _PINHOLE_SEQUENCE:
            cameras = read_colmap_cameras(self._fetch_colmap_file(sequence_name, "cameras.bin"))
            images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))

            # Any registered Left frame tells us which COLMAP camera_id is the Left camera.
            camera_id = next(v[0] for name, v in images.items() if name.startswith(_PINHOLE_LEFT_PREFIX))
            model_name, width, height, params = cameras[camera_id]

            if model_name == "SIMPLE_PINHOLE":
                f, cx, cy = params
                fx, fy = f, f
            else:  # PINHOLE
                fx, fy, cx, cy = params

            # Rescale intrinsics from COLMAP's reference image size to the resized rgb_0 image size.
            with Image.open(next(rgb_path.iterdir())) as img:
                resized_w, resized_h = img.size
            scale_x, scale_y = resized_w / width, resized_h / height

            rgb: dict[str, Any] = {
                "cam_name": "rgb_0",
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "focal_length": [fx * scale_x, fy * scale_y],
                "principal_point": [cx * scale_x, cy * scale_y],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4),
            }
        else:
            # No calibration is published for this sequence's raw fisheye images.
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
        if sequence_name != _PINHOLE_SEQUENCE:
            return

        sequence_path = self.sequence_path(sequence_name)
        rgb_path = self.rgb_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))
        rgb_files = sorted(file_path.name for file_path in rgb_path.iterdir() if file_path.is_file())

        rows = []
        for i, filename in enumerate(rgb_files):
            if filename not in images:
                continue

            _, qvec, tvec = images[filename]
            tx, ty, tz, qx, qy, qz, qw = world_to_camera_to_pose(qvec, tvec)
            ts_ns = int(i * 1e9 / self.rgb_hz)
            rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(groundtruth_csv, ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows)

    @staticmethod
    def _remote_sequence_name(sequence_name: str) -> str:
        """The HFColmapDatasetMixin._fetch_colmap_file() override point - this dataset's remote
        top-level folder names are a hardcoded table rather than looked up dynamically (contrast
        SonevaDataset's HfApi-backed version)."""
        return _REMOTE_FOLDER[sequence_name]
