"""
Module: VSLAM-LAB - Datasets - dataset_vector.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-04
- License: GPLv3 License
"""

from __future__ import annotations

import os
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from zipfile import ZipFile

import gdown
import numpy as np
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Small-scale sequences (MoCap ground truth, recorded in the MoCap arena) vs. large-scale
# sequences (LiDAR-ICP ground truth, recorded in real indoor architecture) - the two groups were
# calibrated separately (slightly different cam-IMU/stereo extrinsics, see _T_CAM_IMU below), even
# though the physical camera rig and its intrinsics are identical between them.
SMALL_SCALE_SEQUENCES: Final[frozenset[str]] = frozenset({
    "board-slow", "corner-slow", "robot-normal", "robot-fast", "desk-normal", "desk-fast",
    "sofa-normal", "sofa-fast", "mountain-normal", "mountain-fast", "hdr-normal", "hdr-fast",
})

# Native resolution of the regular (grayscale, global-shutter) stereo cameras, at which the
# intrinsics below were calibrated. Confirmed against a real downloaded board-slow frame (1224x1024,
# 8-bit grayscale).
_NATIVE_SIZE: Final[tuple[int, int]] = (1224, 1024)

# Intrinsics are identical for small-scale and large-scale (same physical rig) - from
# left/right_regular_camera_intrinsic_results.yaml on the dataset's calibration page (plumb_bob /
# radtan model, 5th coefficient k3=0, i.e. radtan4).
_CAM_INTRINSICS: Final[dict[str, dict[str, Any]]] = {
    "left": {
        "focal_length": (886.191073, 886.591633),
        "principal_point": (610.578911, 514.59271),
        "distortion_coefficients": [-0.31576, 0.104955, 0.00032, -0.000156],
    },
    "right": {
        "focal_length": (887.804282, 888.04815),
        "principal_point": (616.177573, 514.712952),
        "distortion_coefficients": [-0.311523, 0.09641, 0.000623, -0.000375],
    },
}

# Transformation from IMU to camera (X_cam = T_cam_imu @ X_imu), from
# {small,large}_scale_camera_imu_extrinsic_results1.yaml on the dataset's calibration page - the
# only calibration values that differ between the two sequence groups.
_T_CAM_IMU: Final[dict[str, dict[str, list[list[float]]]]] = {
    "small-scale": {
        "left": [
            [0.017248643674008135, -0.9998037138739959, 0.009747718459772736, 0.07733078169916466],
            [0.012834636469124028, -0.009526963092989282, -0.999872246379971, -0.016637889364465353],
            [0.9997688514842376, 0.017371548520172697, 0.01266779001636642, -0.14481844113148515],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "right": [
            [0.014447870885660574, -0.9997787221387724, -0.015289401081614196, -0.09375829054484337],
            [0.01108843210124949, 0.015450259055501236, -0.9998191517312787, -0.015076213124111792],
            [0.9998341390952836, 0.014275722527461954, 0.011309201824693582, -0.14050229888646543],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "large-scale": {
        "left": [
            [0.017014304328419078, -0.999823414494766, 0.0079783003357361, 0.07138061555049913],
            [0.008227025113892006, -0.007839192351438318, -0.9999354294758499, -0.015324174578544],
            [0.999821398803804, 0.01707884338309873, 0.008092193936149267, -0.14279853029864117],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "right": [
            [0.009776735697359762, -0.9998459603712744, -0.014576383925661074, -0.09932466700287404],
            [0.00622553910421203, 0.014637659992240482, -0.9998734827831038, -0.01319591806009756],
            [0.9999328267948634, 0.009684752923841855, 0.006367688657451687, -0.13869488934651844],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
}

# XSens MTi-30 AHRS noise/random-walk densities (dataset-wide, same unit for every sequence) - from
# imu_intrinsic_results.yaml on the dataset's calibration page. No published saturation-limit
# (a_max/g_max) spec exists for this IMU - kept as a generic MEMS-IMU estimate, unverified (same
# values used by dataset_euroc.py/dataset_hilti2026.py).
_IMU_PARAMS: Final[dict[str, float]] = {
    "sigma_g_c": 0.0007294729852113113,
    "sigma_a_c": 0.0012655720309610252,
    "sigma_gw_c": 6.996094830870257e-06,
    "sigma_aw_c": 5.6386016813618435e-05,
    "update_rate": 200.0,
}


def _group_for(sequence_name: str) -> str:
    return "small-scale" if sequence_name in SMALL_SCALE_SEQUENCES else "large-scale"


class VectorDataset(DatasetVSLAMLAB):
    """VECtor: a versatile event-centric benchmark for multi-sensor SLAM."""

    def __init__(self, dataset_name: str = "vector") -> None:
        super().__init__(dataset_name)

        # Per-sequence, per-stream Google Drive file ids (left_camera/right_camera zip,
        # imu/ground_truth txt) - see dataset_vector.yaml's google_drive_files.
        self.google_drive_files: dict[str, dict[str, str]] = self.cfg["google_drive_files"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        files = self.google_drive_files[sequence_name]
        for stream, filename in (
            ("left_camera", "left_camera.zip"),
            ("right_camera", "right_camera.zip"),
            ("imu", "imu.txt"),
            ("ground_truth", "gt.txt"),
        ):
            target = sequence_path / filename
            if not target.exists():
                gdown.download(id=files[stream], output=str(target), quiet=False)

        marker.touch()

    def _extract_camera_zip(self, sequence_name: str, zip_name: str, raw_name: str) -> Path:
        sequence_path = self.sequence_path(sequence_name)
        raw_path = sequence_path / raw_name
        if not raw_path.exists():
            with ZipFile(sequence_path / zip_name) as zf:
                zf.extractall(raw_path)
        # Each zip contains a single top-level folder (e.g.
        # "board_slow1.synced.left_camera/<timestamp>.png" + "timestamp.txt") - the exact folder
        # name varies per sequence (underscored sequence name + a run suffix), so locate it instead
        # of hardcoding it. Confirmed via a real downloaded board-slow left_camera.zip.
        return next(p for p in raw_path.iterdir() if p.is_dir())

    def create_rgb_folder(self, sequence_name: str) -> None:
        for zip_name, raw_name, final_path in (
            ("left_camera.zip", "left_camera_raw", self.rgb_path(sequence_name)),
            ("right_camera.zip", "right_camera_raw", self.sequence_path(sequence_name) / "rgb_1"),
        ):
            if final_path.exists():
                continue
            image_dir = self._extract_camera_zip(sequence_name, zip_name, raw_name)
            final_path.mkdir(parents=True, exist_ok=True)
            for image_file in sorted(image_dir.glob("*.png")):
                if self.target_resolution is None:
                    shutil.copy2(image_file, final_path / image_file.name)
                else:
                    with Image.open(image_file) as img:
                        target_size = compute_scaled_size(img.size, self.target_resolution)
                        img.resize(target_size, Image.Resampling.LANCZOS).save(final_path / image_file.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # Left/right regular cameras are hardware-synchronized (VECtor's own MCU-triggered sync,
        # confirmed by identical exposure-start timestamps in each stream's timestamp.txt) and
        # every image is named by its exposure-start timestamp in seconds (e.g.
        # "1642521684.836608.png") - sort both folders and zip by index, same as
        # dataset_eth.py/dataset_euroc.py.
        rgb_0_files = sorted(self.rgb_path(sequence_name).glob("*.png"))
        rgb_1_files = sorted((self.sequence_path(sequence_name) / "rgb_1").glob("*.png"))

        n = min(len(rgb_0_files), len(rgb_1_files))
        rows = []
        for rgb_0_file, rgb_1_file in zip(rgb_0_files[:n], rgb_1_files[:n]):
            ts_ns = int(Decimal(rgb_0_file.stem) * Decimal(10**9))
            rows.append([ts_ns, f"rgb_0/{rgb_0_file.name}", ts_ns, f"rgb_1/{rgb_1_file.name}"])

        write_csv_rows(
            self.rgb_csv_path(sequence_name),
            ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"],
            rows,
        )

    def create_imu_csv(self, sequence_name: str) -> None:
        # imu.txt: "# timestamp[s] gx gy gz[rad/s] ax ay az[m/s^2] qx qy qz qw" (qx..qw is the
        # XSens AHRS's own internal orientation estimate - not used here, VSLAM-LAB's imu_0.csv is
        # raw gyro+accel only). Confirmed 11 whitespace-separated fields per data line on a real
        # downloaded board-slow/imu.txt, no malformed/NaN lines in that sample.
        src = self.sequence_path(sequence_name) / "imu.txt"
        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]

        rows = []
        with src.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 11:
                    continue
                ts_s, gx, gy, gz, ax, ay, az = parts[:7]
                ts_ns = int(Decimal(ts_s) * Decimal(10**9))
                rows.append([ts_ns, float(gx), float(gy), float(gz), float(ax), float(ay), float(az)])

        write_csv_rows(self.imu_csv_path(sequence_name), header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        group = _group_for(sequence_name)

        rgb: list[dict[str, Any]] = []
        for cam_key, cam_name, side in (("left", "rgb_0", "left"), ("right", "rgb_1", "right")):
            focal_length, principal_point = scale_intrinsics(
                _CAM_INTRINSICS[cam_key]["focal_length"],
                _CAM_INTRINSICS[cam_key]["principal_point"],
                _NATIVE_SIZE,
                self.target_resolution,
            )
            T_cam_imu = np.array(_T_CAM_IMU[group][side], dtype=float)
            rgb.append({
                "cam_name": cam_name,
                "cam_type": "gray",
                "cam_model": "pinhole",
                "focal_length": focal_length,
                "principal_point": principal_point,
                "distortion_type": "radtan4",
                "distortion_coefficients": _CAM_INTRINSICS[cam_key]["distortion_coefficients"],
                "fps": float(self.rgb_hz),
                "T_BS": np.linalg.inv(T_cam_imu),
            })

        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": _IMU_PARAMS["sigma_g_c"],
            "sigma_a_c": _IMU_PARAMS["sigma_a_c"],
            "sigma_bg": 0.01,
            "sigma_ba": 0.1,
            "sigma_gw_c": _IMU_PARAMS["sigma_gw_c"],
            "sigma_aw_c": _IMU_PARAMS["sigma_aw_c"],
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": _IMU_PARAMS["update_rate"],
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgb, imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # gt.txt: "# timestamp[s] tx ty tz qx qy qz qw" - TUM format. Confirmed 8
        # whitespace-separated fields per data line on a real downloaded board-slow/gt.txt.
        src = self.sequence_path(sequence_name) / "gt.txt"
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        rows = []
        with src.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    continue
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(Decimal(ts_s) * Decimal(10**9))
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])

        write_csv_rows(self.groundtruth_csv_path(sequence_name), header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "left_camera_raw", ignore_errors=True)
            shutil.rmtree(sequence_path / "right_camera_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / "left_camera.zip").unlink(missing_ok=True)
            (sequence_path / "right_camera.zip").unlink(missing_ok=True)
            (sequence_path / "imu.txt").unlink(missing_ok=True)
            (sequence_path / "gt.txt").unlink(missing_ok=True)
