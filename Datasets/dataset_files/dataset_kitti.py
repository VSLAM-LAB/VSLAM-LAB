"""
Module: VSLAM-LAB - Datasets - dataset_kitti.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2024-07-13
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import os
import shutil
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention, VSLAMLAB_BENCHMARK
from utilities import decompressFile, downloadFile, write_csv_rows


class KittiDataset(DatasetVSLAMLAB):
    """KITTI Odometry dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "kitti") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]
        self.url_download_root_gt: str = self.cfg["url_download_root_gt"]

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name = 'data_odometry_gray'
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = 'dataset'
        download_url = self.url_download_root

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)
            downloadFile(self.url_download_root_gt, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)
            decompressFile(self.dataset_path / 'data_odometry_poses.zip', self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for target, image in ((self.rgb_path(sequence_name), 'image_0'), (sequence_path / 'rgb_1', 'image_1')):
            if not target.exists():
                os.makedirs(target)

            raw_dir = self.dataset_path / 'dataset' / 'sequences' / sequence_name / image
            if not raw_dir.exists():
                return

            for png_file in os.listdir(raw_dir):
                if png_file.endswith(".png"):
                    shutil.move(raw_dir / png_file, target / png_file)

            shutil.rmtree(raw_dir)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        times_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'times.txt'

        # Read timestamps
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Collect and sort image filenames
        rgb_path = self.rgb_path(sequence_name)
        rgb_files = sorted(f for f in os.listdir(rgb_path) if (rgb_path / f).is_file())

        header = ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1']
        rows = []
        for t, fname in zip(times, rgb_files):  # pairs safely to the shorter list
            t_ns = int(float(t) * 1e9)
            rows.append([t_ns, f"rgb_0/{fname}", t_ns, f"rgb_1/{fname}"])
        write_csv_rows(rgb_csv, header, rows)
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'calib.txt'

        with open(calibration_txt, 'r') as file:
            calibration_0 = [value for value in file.readline().split()]
            fx_0, fy_0, cx_0, cy_0 = calibration_0[1], calibration_0[6], calibration_0[3], calibration_0[7]
            calibration_1 = [value for value in file.readline().split()]
            fx_1, fy_1, cx_1, cy_1 = calibration_1[1], calibration_1[6], calibration_1[3], calibration_1[7]

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx_0, fy_0], "principal_point": [cx_0, cy_0],
                "fps": self.rgb_hz,
                "T_BS": np.eye(4)}
        
        T_BS_1 = np.eye(4)
        T_BS_1[0, 3] = -float(calibration_1[4]) / float(fx_0)
        rgb1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx_1, fy_1], "principal_point": [cx_1, cy_1],
                "fps": self.rgb_hz,
                "T_BS": T_BS_1}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1])
    
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        out_csv = self.groundtruth_csv_path(sequence_name)
        # Keep your original guard
        sequence_name_int = int(sequence_name)
        if sequence_name_int > 10:
            return

        # Read timestamps
        times_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'times.txt'
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Read trajectory and write CSV
        poses_txt = self.dataset_path / 'dataset' / 'poses' / (sequence_name + '.txt')
        with open(poses_txt, 'r') as src, open(out_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'])

            for idx, line in enumerate(src):
                if idx >= len(times):
                    break  # avoid index error if poses has extra lines
                vals = list(map(float, line.strip().split()))
                # row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]
                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                ts = times[idx]
                ts_ns = int(float(ts)*1e9)
                writer.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=23.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            (VSLAMLAB_BENCHMARK / f"dataset").unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (VSLAMLAB_BENCHMARK / f"data_odometry_gray.zip").unlink(missing_ok=True)
            (VSLAMLAB_BENCHMARK / f"data_odometry_poses.zip").unlink(missing_ok=True)