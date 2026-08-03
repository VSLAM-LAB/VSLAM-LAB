"""
Module: VSLAM-LAB - Datasets - dataset_s3li_vulcano.py
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
from typing import Any

import cv2
import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows


class S3liVulcanoDataset(DatasetVSLAMLAB):
    """S3LI Vulcano dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "s3li_vulcano") -> None:
        super().__init__(dataset_name)

        # Keyed by full sequence_name - see dataset_s3li_vulcano.yaml.
        self.url_download_root: dict[str, str] = self.cfg["url_download_root"]

    def _raw_sequence_path(self, sequence_name: str) -> Path:
        return self.dataset_path / f"{sequence_name}_raw"

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_sequence_path = self._raw_sequence_path(sequence_name)
        marker = self.dataset_path / f".{sequence_name}.download_complete"
        if marker.exists():
            return

        url = self.url_download_root[sequence_name]
        zip_path = self.dataset_path / f"{sequence_name}.zip"

        # downloadFile names the file after the URL's own last path segment - for a HiDrive
        # sharelink that's the literal "download?id=<id>" (no filename in the URL itself), so the
        # downloaded file needs renaming into the {sequence_name}.zip convention every other
        # download_sequence_data in this repo follows.
        if not zip_path.exists():
            downloaded_name = self.dataset_path / url.rsplit("/", 1)[-1]
            downloadFile(url, str(self.dataset_path))
            downloaded_name.rename(zip_path)

        if not raw_sequence_path.exists():
            # The zip's own top-level folder is named after the sequence (e.g. "vegetation/"), so
            # it extracts straight to dataset_path/<sequence_name> - move it aside to <name>_raw so
            # every create_* hook below can read raw input and write the final standardized output
            # without the two colliding on the same filename (both are called e.g. "rgb.csv").
            decompressFile(str(zip_path), str(self.dataset_path))
            (self.dataset_path / sequence_name).rename(raw_sequence_path)

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        raw_sequence_path = self._raw_sequence_path(sequence_name)
        # Vulcano's images (516x386) are already smaller than target_resolution's 640x480 pixel
        # area, so this dataset never resizes - a plain copy either way.
        rgb_paths = (("rgb_0", self.rgb_path(sequence_name)), ("rgb_1", self.sequence_path(sequence_name) / "rgb_1"))
        for raw_name, rgb_path in rgb_paths:
            if rgb_path.exists():
                continue
            shutil.copytree(raw_sequence_path / raw_name, rgb_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Raw rgb.csv has NO header row (unlike groundtruth.csv/imu.csv, which do) - four columns,
        # seconds-precision float timestamps.
        raw_csv = self._raw_sequence_path(sequence_name) / "rgb.csv"
        rgb_csv = self.rgb_csv_path(sequence_name)

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = []
        with open(raw_csv, "r", newline="", encoding="utf-8") as fin:
            for row in csv.reader(fin):
                ts_ns_0 = int(round(float(row[0]) * 1e9))
                ts_ns_1 = int(round(float(row[2]) * 1e9))
                rows.append([ts_ns_0, row[1], ts_ns_1, row[3]])
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        raw_calibration_yaml = self._raw_sequence_path(sequence_name) / "calibration.yaml"
        fs = cv2.FileStorage(str(raw_calibration_yaml), cv2.FILE_STORAGE_READ)

        # Both cameras share one intrinsics block (Camera0.*) - the rig's left/right cameras are
        # the same model, only the extrinsics (baseline) differ.
        fx = fs.getNode("Camera0.fx").real()
        fy = fs.getNode("Camera0.fy").real()
        cx = fs.getNode("Camera0.cx").real()
        cy = fs.getNode("Camera0.cy").real()

        # Stereo.bf is the physical baseline in meters directly (verified against the real
        # download: 9.066208e-02 -> ~9cm, a plausible compact stereo baseline; the conventional
        # ORB-SLAM "bf" = baseline*fx reading would imply an implausible ~0.2mm baseline here).
        baseline = fs.getNode("Stereo.bf").real()
        T_BS_right = np.array(fs.getNode("IMU.T_b_c1").mat().tolist()).reshape((4, 4))
        T_right_left = np.eye(4)
        T_right_left[0, 3] = -baseline
        T_BS_left = T_BS_right @ T_right_left

        sigma_g_c = fs.getNode("IMU.NoiseGyro").real()
        sigma_a_c = fs.getNode("IMU.NoiseAcc").real()
        sigma_gw_c = fs.getNode("IMU.GyroWalk").real()
        sigma_aw_c = fs.getNode("IMU.AccWalk").real()
        fs.release()

        # The raw calibration.yaml's own Camera0.fps happens to match this dataset's real rate
        # (10.0), but the sibling Etna release's copy of the same field turned out to be a stale/
        # copy-pasted value (10.0 there too, despite Etna's own real rate being ~30Hz) - use the
        # dataset's own declared rgb_hz instead of trusting this field, for the same reason.
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": self.rgb_hz,
            "T_BS": T_BS_left,
        }
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": self.rgb_hz,
            "T_BS": T_BS_right,
        }

        # IMU noise-density values from the raw calibration.yaml (XSens MTi-G 10). a_max/g_max/
        # sigma_bg/sigma_ba/g0/a0/s_a have no source in the raw file - generic defaults reused from
        # dataset_madmax.py, another hand-held DLR planetary-analogue rig. fps is the real measured
        # rate from the downloaded vegetation sequence's imu.csv (~400.1 Hz over 95670 samples /
        # 239.18s) - the raw calibration.yaml's own "IMU.Frequency: 100" field does not match the
        # actual data and is not trusted (same reasoning as dataset_hilti2022.py's fps override).
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": sigma_g_c,
            "sigma_a_c": sigma_a_c,
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": sigma_gw_c,
            "sigma_aw_c": sigma_aw_c,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 400.0,
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_imu_csv(self, sequence_name: str) -> None:
        raw_imu_csv = self._raw_sequence_path(sequence_name) / "imu.csv"
        imu_csv = self.imu_csv_path(sequence_name)

        header = [
            "ts (ns)",
            "wx (rad s^-1)",
            "wy (rad s^-1)",
            "wz (rad s^-1)",
            "ax (m s^-2)",
            "ay (m s^-2)",
            "az (m s^-2)",
        ]
        rows = []
        with open(raw_imu_csv, "r", newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            next(reader, None)  # real header row: "timestamp [s],ang_vel_x,...,lin_acc_z"
            for row in reader:
                ts_ns = int(round(float(row[0]) * 1e9))
                rows.append([ts_ns] + row[1:])
        write_csv_rows(imu_csv, header, rows)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # D-GNSS position only (no orientation source) - real header:
        # "timestamp [s],pos_x,pos_y,pos_z,ori_x,ori_y,ori_z,ori_w", with ori_* already written as
        # an identity-quaternion placeholder (0,0,0,1) by the source.
        raw_csv = self._raw_sequence_path(sequence_name) / "groundtruth.csv"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)

        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        rows = []
        with open(raw_csv, "r", newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            next(reader, None)
            for row in reader:
                ts_ns = int(round(float(row[0]) * 1e9))
                rows.append([ts_ns] + row[1:])
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        raw_sequence_path = self._raw_sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            raw_files = (
                "rgb.csv", "calibration.yaml", "imu.csv", "groundtruth.csv",
                "disparity_test.png", "trajectory_plot.png",
            )
            for name in raw_files:
                (raw_sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(raw_sequence_path, ignore_errors=True)
            (self.dataset_path / f"{sequence_name}.zip").unlink(missing_ok=True)
