"""
Module: VSLAM-LAB - Datasets - dataset_hamlyn.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-03
- License: GPLv3 License
"""

from __future__ import annotations

import csv
from typing import Any
from zipfile import ZipFile

import numpy as np
from huggingface_hub import hf_hub_download

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import hf_token, write_csv_rows


class HamlynDataset(DatasetVSLAMLAB):
    """Hamlyn rectified stereo endoscopy dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hamlyn") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.hf_repo_id: str = self.cfg["hf_repo_id"]

        # Depth factor
        self.depth_factor: float = self.cfg["depth_factor"]

        # Calibration (intrinsics + stereo extrinsics) ships as one shared calibration.zip in the
        # same HF repo, covering every sequence - downloaded/extracted once, re-read on every
        # create_calibration_yaml call.
        self.master_calibration_path = self.dataset_path / "calibration"

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        file_path = hf_hub_download(repo_id=self.hf_repo_id, filename=f"{sequence_name}.zip",
                                     repo_type='dataset', token=hf_token())
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for raw, tgt in (
            ("image01", self.rgb_path(sequence_name)),
            ("image02", sequence_path / "rgb_1"),
            ("depth01", self.depth_path(sequence_name)),
            ("depth02", sequence_path / "depth_1"),
        ):
            src = sequence_path / raw
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_files = sorted(f.name for f in self.rgb_path(sequence_name).iterdir() if f.is_file())
        rgb1_files = sorted(f.name for f in (sequence_path / "rgb_1").iterdir() if f.is_file())
        depth_files = sorted(f.name for f in self.depth_path(sequence_name).iterdir() if f.is_file())

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1",
                  "ts_depth_0 (ns)", "path_depth_0"]
        rows = []
        for f0, f1, fd in zip(rgb_files, rgb1_files, depth_files, strict=True):
            frame_idx = int(f0.split(".")[0])
            ts_ns = int(1e10 + frame_idx / self.rgb_hz * 1e9)
            rows.append([ts_ns, f"rgb_0/{f0}", ts_ns, f"rgb_1/{f1}", ts_ns, f"depth_0/{fd}"])

        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        self._ensure_calibration_downloaded()
        calib_dir = self.master_calibration_path / sequence_name.removeprefix("rectified")

        with open(calib_dir / "intrinsics.txt", "r", encoding="utf-8") as f:
            row0 = f.readline().split()
            row1 = f.readline().split()
        fx, cx = float(row0[0]), float(row0[2])
        fy, cy = float(row1[1]), float(row1[2])

        # Both rectified cameras share one intrinsics.txt (rectification unifies K between them);
        # extrinsics.txt gives cam_1's pose directly in cam_0's frame (rotation + translation in
        # mm, following the same convention as dataset_kitti.py's stereo T_BS), so cam_0 is a
        # natural choice of body frame (T_BS = identity) with cam_1's T_BS read straight from it.
        with open(calib_dir / "extrinsics.txt", "r", encoding="utf-8") as f:
            rows = [line.split() for line in f if line.strip()]
        R_10 = np.array([[float(v) for v in row[:3]] for row in rows], dtype=float)
        t_10_mm = np.array([float(row[3]) for row in rows], dtype=float)

        T_BS_1 = np.eye(4)
        T_BS_1[:3, :3] = R_10
        T_BS_1[:3, 3] = t_10_mm / 1000.0

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
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.rgb_hz),
            "T_BS": T_BS_1,
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb1], rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        tmp.replace(groundtruth_csv)

    def _ensure_calibration_downloaded(self) -> None:
        marker = self.master_calibration_path / ".download_complete"
        if marker.exists():
            return
        file_path = hf_hub_download(repo_id=self.hf_repo_id, filename="calibration.zip",
                                     repo_type='dataset', token=hf_token())
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)
        marker.touch()
