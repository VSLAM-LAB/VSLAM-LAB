"""
Module: VSLAM-LAB - Datasets - dataset_sweetcorals.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5, Fable 5.1)
- Version: 1.1
- Created: 2026-07-22
- Updated: 2026-09-04
- License: GPLv3 License
"""

from __future__ import annotations

from fnmatch import fnmatch

import numpy as np

from Datasets.dataset_files.dataset_soneva import HFColmapDatasetMixin
from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import ensure_hf_sequence_download, hf_token, read_colmap_images

# Every survey is shot with the same Wildflow two-GoPro rig as soneva (see dataset_soneva.py's
# module comment): two independent, unsynchronized time-lapse cameras, exposed as two independent
# monocular streams - rgb_0 is the Left camera, rgb_1 the Right one - never as a stereo pair.

# Only tabuhan_p1 has been fully processed on the source (color-corrected pinhole images plus a
# colmap reconstruction with real poses) — every other sequence ships only raw, uncalibrated
# fisheye stills with no pose data.
_PINHOLE_SEQUENCE = "tabuhan_p1"

# tabuhan_p1's corrected/images/ folder merges both rig cameras into one flat directory: Left
# frames are prefixed GPAA, Right frames GPAB (27 frames) then GPAC (the rest) - GoPro rolls the
# two-letter prefix over as its file counter wraps, so both belong to the same Right time-lapse
# and sort in capture order. One fnmatch pattern per stream (rgb_0, rgb_1) isolates each camera,
# both for the download (snapshot_download's allow_patterns are fnmatch patterns too) and for
# picking that camera's COLMAP camera_id from images.bin's image names. COLMAP confirms the split:
# GPAA frames are all one camera_id, GPAB+GPAC all the other.
_PINHOLE_STREAM_PATTERNS: tuple[str, ...] = ("GPAA*", "GPA[BC]*")

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

# Every survey (other than tabuhan_p1, handled separately above) ships its two rig cameras under
# raw/<tag>_Left and raw/<tag>_Right. Per sequence, one list of raw/ subfolders per stream, in
# rgb_0/rgb_1 order: Left is rgb_0 (the canonical mono view) and Right is rgb_1 - except
# watudodol_p2, which has no Left data at all, so its only stream (rgb_0) is the Right camera.
# watudodol_p1 also ships an extra continuation folder from a second day for each side,
# concatenated after the main one. (watudodol_p3's Right side is a 69-frame stub in the source.)
_RAW_CAMERA_SUBFOLDERS: dict[str, tuple[list[str], ...]] = {
    "banyuwangi_farm": (["F1_Left"], ["F1_Right"]),
    "pemuteran_p1": (["B1_Left"], ["B1_Right"]),
    "pemuteran_p2": (["B2_Left"], ["B2_Right"]),
    "pemuteran_p3": (["B3_Left"], ["B3_Right"]),
    "tabuhan_p2": (["Q8_Left"], ["Q8_Right"]),
    "tabuhan_p3": (["Q9_Left"], ["Q9_Right"]),
    "watudodol_p1": (["Q1_Left", "Q1_Left_extra_20250209"], ["Q1_Right", "Q1_Right_extra_20250209"]),
    "watudodol_p2": (["Q2_Right"],),
    "watudodol_p3": (["Q3_Left"], ["Q3_Right"]),
    "watudodol_p4": (["Q4_Left"], ["Q4_Right"]),
    "watudodol_p5": (["Q5_Left"], ["Q5_Right"]),
    "watudodol_p6": (["Q6_Left"], ["Q6_Right"]),
}


class SweetcoralsDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):
    """Sweet Corals dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "sweetcorals") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id = self.cfg["hf_repo_id"]

    def download_sequence_data(self, sequence_name: str) -> None:
        remote_folder = self._remote_sequence_name(sequence_name)

        if sequence_name == _PINHOLE_SEQUENCE:
            remote_dir = f"{remote_folder}/corrected/images"
            for cam_idx, pattern in enumerate(_PINHOLE_STREAM_PATTERNS):
                ensure_hf_sequence_download(
                    self.hf_repo_id, [remote_dir], self.rgb_raw_path(sequence_name, cam_idx),
                    pattern=pattern, token=hf_token(),
                )
            return

        for cam_idx, subfolders in enumerate(_RAW_CAMERA_SUBFOLDERS[sequence_name]):
            remote_dirs = [f"{remote_folder}/raw/{subfolder}" for subfolder in subfolders]
            ensure_hf_sequence_download(
                self.hf_repo_id, remote_dirs, self.rgb_raw_path(sequence_name, cam_idx), token=hf_token(),
            )

    def create_calibration_yaml(self, sequence_name: str) -> None:
        rgb = []
        if sequence_name == _PINHOLE_SEQUENCE:
            images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))
            for cam_idx, pattern in enumerate(_PINHOLE_STREAM_PATTERNS):
                # Any registered frame matching this stream's prefix pattern tells us which COLMAP
                # camera_id is its camera.
                camera_id = next(v[0] for name, v in images.items() if fnmatch(name, pattern))
                rgb.append(self._pinhole_rgb_calibration(sequence_name, camera_id, cam_idx))
        else:
            # No calibration is published for this sequence's raw fisheye images - either stream.
            for cam_idx in self._stream_indices(sequence_name):
                rgb.append({
                    "cam_name": f"rgb_{cam_idx}",
                    "cam_type": "rgb",
                    "cam_model": "unknown",
                    "focal_length": [0.0, 0.0],
                    "principal_point": [0.0, 0.0],
                    "fps": float(self.rgb_hz),
                    "T_BS": np.eye(4),
                })

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if sequence_name != _PINHOLE_SEQUENCE:
            # No calibration/pose data is published for this sequence's raw fisheye images -
            # still write the file (header only, no rows) rather than leaving it missing.
            for cam_idx in self._stream_indices(sequence_name):
                self._write_empty_groundtruth(sequence_name, cam_idx)
            return

        # Both cameras are registered in the same COLMAP reconstruction, so groundtruth.csv (rgb_0)
        # and groundtruth_1.csv (rgb_1) share one world frame. The corrected images keep their
        # original filenames in COLMAP, so the frame -> COLMAP image name mapping is the identity.
        images = read_colmap_images(self._fetch_colmap_file(sequence_name, "images.bin"))
        for cam_idx in self._stream_indices(sequence_name):
            self._write_colmap_groundtruth(sequence_name, cam_idx, images, lambda filename: filename)

    @staticmethod
    def _remote_sequence_name(sequence_name: str) -> str:
        """The HFColmapDatasetMixin._fetch_colmap_file() override point - this dataset's remote
        top-level folder names are a hardcoded table rather than looked up dynamically (contrast
        SonevaDataset's HfApi-backed version)."""
        return _REMOTE_FOLDER[sequence_name]

    @staticmethod
    def _stream_indices(sequence_name: str) -> list[int]:
        """The HFColmapDatasetMixin hook: tabuhan_p1 has one stream per prefix pattern, every other
        sequence one per entry of its _RAW_CAMERA_SUBFOLDERS row (two, or one for watudodol_p2)."""
        if sequence_name == _PINHOLE_SEQUENCE:
            return list(range(len(_PINHOLE_STREAM_PATTERNS)))
        return list(range(len(_RAW_CAMERA_SUBFOLDERS[sequence_name])))
