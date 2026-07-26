"""
Module: VSLAM-LAB - Datasets - dataset_tartanair.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2024-08-01
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Final

import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows

CAMERA_PARAMS: Final = [320.0, 320.0, 320.0, 240.0] # Camera intrinsics (fx, fy, cx, cy)


class TartanairDataset(DatasetVSLAMLAB):
    """TartanAir dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "tartanair") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]
        self.url_download_root_gt: str = self.cfg["url_download_root_gt"]

    def download_sequence_data(self, sequence_name: str) -> None:
        # Variables
        compressed_name = 'tartanair-test-mono-release'
        compressed_name_ext = compressed_name + '.tar.gz'
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
            decompressFile(compressed_file, str(self.dataset_path / compressed_name))

        # Download the gt
        if not (self.dataset_path / 'tartanair_cvpr_gt').exists():
            compressed_name = '3p1sf0eljfwrz4qgbpc6g95xtn2alyfk'
            compressed_name_ext = compressed_name + '.zip'
            decompressed_name = 'tartanair_cvpr_gt'

            compressed_file = self.dataset_path / compressed_name_ext
            decompressed_folder = self.dataset_path / decompressed_name

            download_url = self.url_download_root_gt
            if not compressed_file.exists():
                downloadFile(download_url, self.dataset_path)

            decompressFile(compressed_file, self.dataset_path / decompressed_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        rgb_path.mkdir(parents=True, exist_ok=True)

        rgb_path_0 = self.dataset_path / 'tartanair-test-mono-release' / 'mono' / sequence_name
        if not rgb_path_0.exists():
            return

        for png_file in os.listdir(rgb_path_0):
            if png_file.endswith(".png"):
                shutil.move(rgb_path_0 / png_file, rgb_path / png_file)

        shutil.rmtree(rgb_path_0)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        rows = []
        for filename in rgb_files:
            name, _ = os.path.splitext(filename)
            ts = float(name) / self.rgb_hz
            ts_ns = int(1e10 + ts * 1e9)
            rows.append([ts_ns, f"rgb_0/{filename}"])

        write_csv_rows(self.rgb_csv_path(sequence_name), ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = CAMERA_PARAMS
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_txt = self.dataset_path / "tartanair_cvpr_gt" / "mono_gt" / f"{sequence_name}.txt"

        rows = []
        with open(groundtruth_txt, "r", encoding="utf-8") as fin:
            for frame_idx, line in enumerate(fin):
                parts = line.strip().split()
                ts = frame_idx / float(self.rgb_hz)
                ts_ns = int(1e10 + ts * 1e9)
                tx, ty, tz, qx, qy, qz, qw = parts[:7]
                rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(
            self.groundtruth_csv_path(sequence_name), ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"], rows,
        )

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=8.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            dataset_folder = self.dataset_path / 'tartanair-test-mono-release'
            if dataset_folder.exists():
                shutil.rmtree(dataset_folder)

            gt_folder = self.dataset_path / 'tartanair_cvpr_gt'
            if gt_folder.exists():
                shutil.rmtree(gt_folder)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"tartanair-test-mono-release.tar.gz").unlink(missing_ok=True)
            (self.dataset_path / f"3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip").unlink(missing_ok=True)