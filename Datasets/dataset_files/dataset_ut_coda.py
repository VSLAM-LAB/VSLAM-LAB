"""
Module: VSLAM-LAB - Datasets - dataset_ut_coda.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2025-05-25
- Updated: 2026-07-28
- License: GPLv3 License
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows


class UtCodaDataset(DatasetVSLAMLAB):
    """CODa dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "ut-coda") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [f"seq{s}" for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path: Path = self.sequence_path(sequence_name)
        # Variables
        compressed_name_ext = sequence_name + '.zip'
        download_url = urljoin(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file: Path = self.dataset_path / compressed_name_ext

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not sequence_path.exists():
            decompressFile(compressed_file, sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path_0 = self.rgb_path(sequence_name)
        rgb_path_1 = sequence_path / 'rgb_1'
        rgb_path_0_raw = sequence_path / '2d_rect' / 'cam0' / sequence_name
        rgb_path_1_raw = sequence_path / '2d_rect' / 'cam1' / sequence_name

        for src, tgt in ((rgb_path_0_raw, rgb_path_0), (rgb_path_1_raw, rgb_path_1)):
            if src.is_dir() and not tgt.exists():
                os.symlink(src, tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path: Path = self.sequence_path(sequence_name)
        rgb_csv: Path = self.rgb_csv_path(sequence_name)
        rgb_path_0: Path = self.rgb_path(sequence_name)
        rgb_path_1: Path = sequence_path / 'rgb_1'
        times_txt: Path = sequence_path / 'timestamps' / (sequence_name + '.txt')

        ts = pd.read_csv(times_txt, sep=r"\s+", comment="#", header=None, names=["ts"])

        def extract_frame_id(path, cam_idx=0):
            match = re.search(rf'2d_rect_cam{cam_idx}_\d+_(\d+)\.jpg', path.name)
            return int(match.group(1)) if match else float('inf')

        rgb_files_0 = sorted(rgb_path_0.glob("*.jpg"), key=lambda f: extract_frame_id(f, cam_idx=0))
        rgb_files_1 = sorted(rgb_path_1.glob("*.jpg"), key=lambda f: extract_frame_id(f, cam_idx=1))

        header = ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1']
        df_rgb = pd.DataFrame({
            header[0]: (ts['ts'] * 1e9).astype(np.int64),
            header[1]: [str(f.relative_to(sequence_path)) for f in rgb_files_0],
            header[2]: (ts['ts'] * 1e9).astype(np.int64),
            header[3]: [str(f.relative_to(sequence_path)) for f in rgb_files_1]
        })
        rows = df_rgb[header].astype(object).values.tolist()
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path: Path = self.sequence_path(sequence_name)
        cams = []
        for cam_idx in (0, 1):
            calibration_yaml: Path = sequence_path / 'calibrations' / sequence_name / f'calib_cam{cam_idx}_intrinsics.yaml'

            with open(calibration_yaml, 'r') as file:
                data = yaml.safe_load(file)

            intrinsics = data['projection_matrix']['data']
            fx, fy, cx, cy = intrinsics[0], intrinsics[5], intrinsics[2], intrinsics[6]

            rgb: dict[str, Any] = {
                "cam_name": f"rgb_{cam_idx}",
                "cam_type": "rgb",
                "cam_model": "pinhole",
                "focal_length": [fx, fy],
                "principal_point": [cx, cy],
                "fps": self.rgb_hz,
                "T_BS": np.eye(4),
            }

            if cam_idx == 1:
                rgb["T_BS"][0, 3] = 0.19637310339252168  # baseline from stereo rectification
            cams.append(rgb)

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=cams)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        CAM2ENU = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
        ENU2CAM = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])

        groundtruth_txt_0 = sequence_path / 'poses' / 'dense_global' / (sequence_name + '.txt')
        header = ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']
        rows = []
        with open(groundtruth_txt_0, 'r') as source_file:
            for line in source_file:
                values = np.array([float(x) for x in line.strip().split()])
                ts = values[0]

                # (ENU LiDAR coordinate system -> cam system)
                SE3_ENU = np.eye(4)
                SE3_ENU[:3, 3] = values[1:4]
                SE3_ENU[:3, :3] = R.from_quat(values[[5, 6, 7, 4]]).as_matrix()

                SE3_CAM = ENU2CAM @ SE3_ENU @ CAM2ENU
                tx, ty, tz = SE3_CAM[0, 3], SE3_CAM[1, 3], SE3_CAM[2, 3]
                quat = R.from_matrix(SE3_CAM[:3, :3]).as_quat()
                qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                ts_ns = int(float(ts) * 1e9)
                rows.append([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name):
        sequence_path: Path = self.sequence_path(sequence_name)
        calibration_folder: Path = sequence_path / "calibrations"
        metadata_folder: Path = sequence_path / "metadata"
        timestamps_folder: Path = sequence_path / "timestamps"
        poses_folder: Path = sequence_path / "poses"
        compressed_file: Path = self.dataset_path / (sequence_name + '.zip')

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(calibration_folder, ignore_errors=True)
            shutil.rmtree(timestamps_folder, ignore_errors=True)
            shutil.rmtree(poses_folder, ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(metadata_folder, ignore_errors=True)
            compressed_file.unlink(missing_ok=True)