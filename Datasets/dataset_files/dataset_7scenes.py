"""
Module: VSLAM-LAB - Datasets - dataset_7scenes.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-01-06
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Final

import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows

SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
CAMERA_PARAMS: Final = [585.0, 585.0, 320.0, 240.0] # Camera intrinsics (fx, fy, cx, cy)


class SevenscenesDataset(DatasetVSLAMLAB):
    """7-Scenes dataset helper for VSLAM-LAB benchmark."""
    
    def __init__(self, dataset_name: str = "7scenes") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_seq-', ' ') for s in self.sequence_names]

        # Depth factor
        self.depth_factor = self.cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_group = _find_sequence_group(sequence_name)
        compressed_name = sequence_group
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)

        # Variables
        compressed_name = sequence_name.replace(sequence_group + '_', '')
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name

        # Constants
        compressed_file = self.dataset_path / sequence_group / compressed_name_ext
        sequence_path = self.sequence_path(sequence_name)

        if not sequence_path.exists():
            decompressFile(compressed_file, self.dataset_path)
            os.rename(self.dataset_path / decompressed_name, sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        modes = ['color', 'depth']
        folder = {'color': self.rgb_path(sequence_name), 'depth': self.depth_path(sequence_name)}
        for mode in modes:
            folder_path = folder[mode]
            if folder_path.exists():
                continue
            folder_path.mkdir(parents=True, exist_ok=True)
            image_files = glob.glob(str(sequence_path / f'*.{mode}.png'))
            for image_path in image_files:
                image_name = os.path.basename(image_path)
                image_name = image_name.replace("frame-", "")
                image_name = image_name.replace(f"{mode}.", "")
                shutil.move(image_path, folder_path / image_name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        folder = {'color': self.rgb_path(sequence_name), 'depth': self.depth_path(sequence_name)}
        png_files = {}
        for mode, folder_path in folder.items():
            png_files[mode] = sorted(file for file in os.listdir(folder_path) if file.endswith('.png'))

        rows = []
        for iPNG in range(len(png_files['color'])):
            ts_ns = int(1e12 + float(iPNG / self.rgb_hz) * 1e9)
            path_r0 = f"rgb_0/{png_files['color'][iPNG]}"
            path_d = f"depth_0/{png_files['depth'][iPNG]}"
            rows.append([ts_ns, path_r0, ts_ns, path_d])

        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = CAMERA_PARAMS
        rgbd0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])
        
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        pose_files = sorted(glob.glob(str(sequence_path / '*.pose.txt')))

        rows = []
        for iRGB, gt0 in enumerate(pose_files):
            with open(gt0, 'r') as source_file:
                T = [[float(x) for x in line.split()] for line in source_file]
            tx, ty, tz = T[0][3], T[1][3], T[2][3]
            rotation_matrix = np.array([[T[0][0], T[0][1], T[0][2]],
                                        [T[1][0], T[1][1], T[1][2]],
                                        [T[2][0], T[2][1], T[2][2]]])
            qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()
            ts_d_ns = int(1e12 + float(iRGB / self.rgb_hz) * 1e9)
            rows.append([ts_d_ns, tx, ty, tz, qx, qy, qz, qw])

            if BENCHMARK_RETENTION != Retention.FULL:
                os.remove(gt0)

        write_csv_rows(
            self.groundtruth_csv_path(sequence_name), ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows,
        )

    def remove_unused_files(self, sequence_name: str) -> None:
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            sequence_group = _find_sequence_group(sequence_name)
            compressed_name = sequence_name.replace(sequence_group + '_', '')
            group_folder = self.dataset_path / sequence_group

            # This sequence's own sub-zip inside the shared, multi-sequence <group>.zip - safe to
            # remove once this one sequence is done, unlike the group folder below.
            (group_folder / f"{compressed_name}.zip").unlink(missing_ok=True)
            # The shared archive itself - deleting it here (potentially before sibling sequences
            # in the same group are downloaded) is safe: download_sequence_data re-downloads it
            # on demand if a later sequence needs it again.
            (self.dataset_path / f"{sequence_group}.zip").unlink(missing_ok=True)

            # Remove the group folder once it's empty (every sequence in this group has had its
            # own sub-zip cleaned up) - still safe if siblings haven't been processed yet, since
            # their sub-zip would still be present and the folder wouldn't be empty.
            if group_folder.is_dir() and not any(group_folder.iterdir()):
                group_folder.rmdir()


def _find_sequence_group(sequence_name):
    for scene in SCENES:
         if scene in sequence_name:
            return scene