"""
Module: VSLAM-LAB - Datasets - dataset_s3li_etna.py
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
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, decompressFile, downloadFile, scale_intrinsics, write_csv_rows


class S3liEtnaDataset(DatasetVSLAMLAB):
    """S3LI Etna dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "s3li-etna") -> None:
        super().__init__(dataset_name)

        # Keyed by full sequence_name - see dataset_s3li-etna.yaml. These are direct HiDrive
        # sharelinks, not the DLR page's documented (form-gated) access path - see the yaml's
        # comment on url_download_root.
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
            # Unlike the sibling Vulcano release (whose zip's top-level folder is the bare
            # sequence_name), Etna's zips prefix it - e.g. landmarks.zip's own top-level folder is
            # "s3li_landmarks/", not "landmarks/". Detect whichever new directory decompression
            # actually created rather than assuming the exact name, and move it aside to <name>_raw
            # so every create_* hook below can read raw (native-resolution) input and write the
            # resized/final standardized output without the two colliding on the same filename
            # (both are called e.g. "rgb.csv").
            dirs_before = {p.name for p in self.dataset_path.iterdir() if p.is_dir()}
            decompressFile(str(zip_path), str(self.dataset_path))
            new_dirs = [p.name for p in self.dataset_path.iterdir() if p.is_dir() and p.name not in dirs_before]
            if len(new_dirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one new folder after extracting {zip_path}, found: {new_dirs}"
                )
            (self.dataset_path / new_dirs[0]).rename(raw_sequence_path)

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        raw_sequence_path = self._raw_sequence_path(sequence_name)
        for raw_name, rgb_path in (
            ("rgb_0", self.rgb_path(sequence_name)),
            ("rgb_1", self.sequence_path(sequence_name) / "rgb_1"),
        ):
            if rgb_path.exists():
                continue

            raw_path = raw_sequence_path / raw_name
            if self.target_resolution is None:
                shutil.copytree(raw_path, rgb_path)
                continue

            rgb_path.mkdir(parents=True, exist_ok=True)
            target_size = None
            for file_path in sorted(raw_path.glob("*.png")):
                with Image.open(file_path) as img:
                    if target_size is None:
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(rgb_path / file_path.name)

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
        raw_sequence_path = self._raw_sequence_path(sequence_name)
        raw_calibration_yaml = raw_sequence_path / "calibration.yaml"
        fs = cv2.FileStorage(str(raw_calibration_yaml), cv2.FILE_STORAGE_READ)

        # Both cameras share one intrinsics block (Camera0.*) - the rig's left/right cameras are
        # the same model, only the extrinsics (baseline) differ.
        fx = fs.getNode("Camera0.fx").real()
        fy = fs.getNode("Camera0.fy").real()
        cx = fs.getNode("Camera0.cx").real()
        cy = fs.getNode("Camera0.cy").real()

        # Stereo.bf is the physical baseline in meters directly (verified on the sibling Vulcano
        # release: 9.066208e-02 -> ~9cm, a plausible compact stereo baseline; the conventional
        # ORB-SLAM "bf" = baseline*fx reading would imply an implausible sub-millimeter baseline).
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

        # Camera0.w/Camera0.h in the raw file are unfilled placeholders (0, 0) on the sibling
        # Vulcano release, not a trustworthy native resolution - read the real pre-resize image
        # size directly off disk instead (VSLAM-LAB issue #99 / dataset_soneva.py's same caution).
        with Image.open(next((raw_sequence_path / "rgb_0").glob("*.png"))) as raw_img:
            native_size = raw_img.size
        focal_length, principal_point = scale_intrinsics((fx, fy), (cx, cy), native_size, self.target_resolution)

        # The raw calibration.yaml's own Camera0.fps says 10.0 - that's the sibling Vulcano
        # release's rate, not this dataset's (verified against the downloaded landmarks sequence:
        # 14325 rgb.csv rows over ~540s -> ~26.5 Hz measured, consistent with the paper's stated
        # 30 Hz trigger rate plus some dropped frames, nowhere near 10 Hz) - same kind of stale/
        # copy-pasted field as Vulcano's IMU.Frequency. Use the dataset's own declared rgb_hz
        # instead of trusting this field.
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length,
            "principal_point": principal_point,
            "fps": self.rgb_hz,
            "T_BS": T_BS_left,
        }
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length,
            "principal_point": principal_point,
            "fps": self.rgb_hz,
            "T_BS": T_BS_right,
        }

        # IMU noise-density values from the raw calibration.yaml (XSens MTi-G 10, the same rig
        # hardware as the sibling Vulcano release). a_max/g_max/sigma_bg/sigma_ba/g0/a0/s_a have no
        # source in the raw file - generic defaults reused from dataset_madmax.py, another
        # hand-held DLR planetary-analogue rig. fps is the real measured rate confirmed against the
        # sibling Vulcano release's imu.csv (~400.1 Hz) - the raw calibration.yaml's own
        # "IMU.Frequency" field there did not match its actual data and is not trusted (same
        # reasoning as dataset_hilti2022.py's fps override); verify against this dataset's own
        # imu.csv if that turns out to differ.
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
