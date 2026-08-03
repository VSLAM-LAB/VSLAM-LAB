"""
Module: VSLAM-LAB - Datasets - dataset_caves.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-03
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows

CAMERA_PARAMS: Final = [405.6384738851233, 405.588335378204, 189.9054317917407, 139.9149578253755]  # Camera intrinsics (fx, fy, cx, cy)


class CavesDataset(DatasetVSLAMLAB):
    """Underwater Caves Sonar and Vision dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "caves") -> None:
        super().__init__(dataset_name)

        # Get download urls
        self.url_download_root: str = self.cfg["url_download_root"]
        self.url_download_timestamps: str = self.cfg["url_download_timestamps"]
        self.url_download_root_gt: str = self.cfg["url_download_root_gt"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        frames_zip = self.dataset_path / "undistorted_frames.zip"
        if not frames_zip.exists():
            downloaded = Path(downloadFile(self.url_download_root, str(self.dataset_path)))
            downloaded.rename(frames_zip)

        timestamps_txt = self.dataset_path / "undistorted_frames_timestamps.txt"
        if not timestamps_txt.exists():
            downloaded = Path(downloadFile(self.url_download_timestamps, str(self.dataset_path)))
            downloaded.rename(timestamps_txt)

        full_dataset_zip = self.dataset_path / "full_dataset.zip"
        if not full_dataset_zip.exists():
            downloaded = Path(downloadFile(self.url_download_root_gt, str(self.dataset_path)))
            downloaded.rename(full_dataset_zip)

        if not (self.dataset_path / "undistorted_frames").exists() and not self.rgb_path(sequence_name).exists():
            decompressFile(frames_zip, self.dataset_path)

        if not (self.dataset_path / "full_dataset").exists():
            decompressFile(full_dataset_zip, self.dataset_path)

        sequence_path.mkdir(parents=True, exist_ok=True)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if rgb_path.exists():
            return

        raw_frames = self.dataset_path / "undistorted_frames"
        raw_frames.rename(rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        timestamps_txt = self.dataset_path / "undistorted_frames_timestamps.txt"
        rows = []
        with open(timestamps_txt, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if not parts or len(parts) < 2:
                    continue
                name, ts = parts
                ts_ns = int(round(float(ts) * 1e9))
                rows.append([ts_ns, f"rgb_0/{name}"])

        write_csv_rows(rgb_csv, ["ts_rgb_0 (ns)", "path_rgb_0"], rows)

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
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        odometry_txt = self.dataset_path / "full_dataset" / "odometry.txt"
        pose_columns = [
            "field.pose.pose.position.x",
            "field.pose.pose.position.y",
            "field.pose.pose.position.z",
            "field.pose.pose.orientation.x",
            "field.pose.pose.orientation.y",
            "field.pose.pose.orientation.z",
            "field.pose.pose.orientation.w",
        ]

        rows = []
        with open(odometry_txt, "r", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                ts_ns = int(row["%time"])
                pose = [row[col] for col in pose_columns]
                rows.append([ts_ns, *pose])

        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self.dataset_path / "full_dataset", ignore_errors=True)
            (self.dataset_path / "undistorted_frames_timestamps.txt").unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / "undistorted_frames.zip").unlink(missing_ok=True)
            (self.dataset_path / "full_dataset.zip").unlink(missing_ok=True)
