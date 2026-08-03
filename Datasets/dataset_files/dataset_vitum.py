"""
Module: VSLAM-LAB - Datasets - dataset_vitum.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-03
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import yaml
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, decompressFile, downloadFile, scale_intrinsics, write_csv_rows


class VitumDataset(DatasetVSLAMLAB):
    """TUM Visual-Inertial Dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "vitum") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_filename = f"dataset-{sequence_name}_512_16"
        compressed_name = f"{sequence_filename}.tar"
        compressed_file = self.dataset_path / compressed_name
        decompressed_folder = self.dataset_path / sequence_filename

        if not compressed_file.exists() and not decompressed_folder.exists():
            downloadFile(f"{self.url_download_root}/{compressed_name}", str(self.dataset_path))

        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f"dataset-{sequence_name}_512_16"

        for cam, target in (("0", self.rgb_path(sequence_name)), ("1", self.sequence_path(sequence_name) / "rgb_1")):
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)

            src_dir = source_path / "mav0" / f"cam{cam}" / "data"
            for src_image in sorted(src_dir.glob("*.png")):
                # Source images are 16-bit intensity PNGs with a linear photometric response -
                # scale down to 8-bit for the standard rgb_0/rgb_1 layout.
                with Image.open(src_image) as im:
                    img8 = (np.array(im, dtype=np.uint16) >> 8).astype(np.uint8)
                    im8 = Image.fromarray(img8)
                    if self.target_resolution is not None:
                        target_size = compute_scaled_size(im8.size, self.target_resolution)
                        im8 = im8.resize(target_size, Image.Resampling.LANCZOS)
                    im8.save(target / src_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)

        files0 = sorted(p.name for p in self.rgb_path(sequence_name).glob("*.png"))
        files1 = sorted(p.name for p in (sequence_path / "rgb_1").glob("*.png"))

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = []
        for name0, name1 in zip(files0, files1):
            # Filenames are the raw nanosecond capture timestamp (matches dso/camN/times.txt's
            # filename column exactly) - use them directly rather than round-tripping through the
            # times.txt seconds column, which loses precision at this many significant digits.
            ts0 = int(name0.rsplit(".", 1)[0])
            ts1 = int(name1.rsplit(".", 1)[0])
            rows.append([ts0, f"rgb_0/{name0}", ts1, f"rgb_1/{name1}"])
        write_csv_rows(rgb_csv, header, rows)

    def create_imu_csv(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f"dataset-{sequence_name}_512_16" / "dso"
        src = source_path / "imu.txt"
        dst = self.imu_csv_path(sequence_name)

        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        rows = []
        with open(src, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts_s, wx, wy, wz, ax, ay, az = line.split()
                rows.append([int(ts_s), float(wx), float(wy), float(wz), float(ax), float(ay), float(az)])
        write_csv_rows(dst, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f"dataset-{sequence_name}_512_16" / "dso"
        with open(source_path / "camchain.yaml", "r", encoding="utf-8") as f:
            cam_data = yaml.safe_load(f)
        with open(source_path / "imu_config.yaml", "r", encoding="utf-8") as f:
            imu_data = yaml.safe_load(f)

        cam0 = cam_data["cam0"]
        cam1 = cam_data["cam1"]
        T_cam0_imu = np.array(cam0["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1["T_cam_imu"], dtype=float).reshape(4, 4)

        # Kalibr calibration intrinsics are computed at each camera's raw (pre-resize) resolution -
        # rescale to match the actual images create_rgb_folder wrote into rgb_0/rgb_1 (VSLAM-LAB
        # issue #99). A no-op here while target_resolution is unset (vitum's 512x512 source is
        # already below the resize threshold), but kept unconditional so this stays correct if
        # that ever changes.
        focal_length_0, principal_point_0 = scale_intrinsics(
            cam0["intrinsics"][0:2], cam0["intrinsics"][2:4], tuple(cam0["resolution"]), self.target_resolution
        )
        focal_length_1, principal_point_1 = scale_intrinsics(
            cam1["intrinsics"][0:2], cam1["intrinsics"][2:4], tuple(cam1["resolution"]), self.target_resolution
        )

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length_0,
            "principal_point": principal_point_0,
            "distortion_type": "equid4",
            "distortion_coefficients": [float(c) for c in cam0["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(T_cam0_imu),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length_1,
            "principal_point": principal_point_1,
            "distortion_type": "equid4",
            "distortion_coefficients": [float(c) for c in cam1["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        # IMU noise-density spec parsed from this sequence's own imu_config.yaml (Allan-variance
        # derived, inflated 2x/10x by TUM to account for unmodelled effects - see that file's own
        # comments). a_max/g_max aren't part of Kalibr's IMU calibration output and TUM VI doesn't
        # publish a configured full-scale range, so these follow the same generic saturation
        # defaults already used dataset-wide for IMUs without a documented range (dataset_euroc.py,
        # dataset_msd.py).
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": float(imu_data["gyroscope_noise_density"]),
            "sigma_a_c": float(imu_data["accelerometer_noise_density"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": float(imu_data["gyroscope_random_walk"]),
            "sigma_aw_c": float(imu_data["accelerometer_random_walk"]),
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(imu_data["update_rate"]),
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f"dataset-{sequence_name}_512_16" / "dso"
        src = source_path / "gt_imu.csv"
        dst = self.groundtruth_csv_path(sequence_name)

        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        rows = []
        with open(src, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # gt_imu.csv's own column order is ts,tx,ty,tz,qw,qx,qy,qz (w first) - reorder to
                # VSLAM-LAB's qx,qy,qz,qw convention.
                ts_ns, tx, ty, tz, qw, qx, qy, qz = line.split(",")
                rows.append([int(ts_ns), float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])
        write_csv_rows(dst, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f"dataset-{sequence_name}_512_16"

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(source_path, ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"dataset-{sequence_name}_512_16.tar").unlink(missing_ok=True)
