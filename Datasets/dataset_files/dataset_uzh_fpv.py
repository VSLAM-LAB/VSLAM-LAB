"""
Module: VSLAM-LAB - Datasets - dataset_uzh_fpv.py
- Author: Alejandro Fontan
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-17
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, decompressFile, downloadFile, scale_intrinsics, write_csv_rows


class UzhFpvDataset(DatasetVSLAMLAB):
    """UZH-FPV Drone Racing Dataset helper for VSLAM-LAB benchmark."""

    # Sequences whose Leica MS60 ground truth is public - their zips carry a "_with_gt" suffix.
    # The remaining sequences' ground truth was withheld for the (now-closed) UZH-FPV competition,
    # so their groundtruth.csv is header-only.
    _WITH_GT: Final[frozenset[str]] = frozenset(
        {
            "indoor_forward_3",
            "indoor_forward_5",
            "indoor_forward_6",
            "indoor_forward_7",
            "indoor_forward_9",
            "indoor_forward_10",
            "indoor_45_2",
            "indoor_45_4",
            "indoor_45_9",
            "indoor_45_12",
            "indoor_45_13",
            "indoor_45_14",
            "outdoor_forward_1",
            "outdoor_forward_3",
            "outdoor_forward_5",
            "outdoor_45_1",
        }
    )

    def __init__(self, dataset_name: str = "uzh-fpv") -> None:
        super().__init__(dataset_name)

        # Get download urls
        self.url_download_root: str = self.cfg["url_download_root"]
        self.url_download_root_calib: str = self.cfg["url_download_root_calib"]

    def download_sequence_data(self, sequence_name: str) -> None:
        # Per-sequence data zip (no top-level folder inside - extract into its own directory)
        raw_path = self._raw_path(sequence_name)
        marker = raw_path / ".download_complete"
        if not marker.exists():
            zip_name = f"{self._zip_stem(sequence_name)}.zip"
            zip_file = self.dataset_path / zip_name
            if not zip_file.exists():
                downloadFile(f"{self.url_download_root}/{zip_name}", str(self.dataset_path))
            decompressFile(zip_file, raw_path)
            marker.touch()

        # Kalibr calibration zip, shared by every sequence of the same environment group (the zip
        # itself contains a top-level <group>_calib_snapdragon/ folder)
        calib_path = self._calib_path(sequence_name)
        calib_marker = calib_path / ".download_complete"
        if not calib_marker.exists():
            calib_zip_name = f"{calib_path.name}.zip"
            calib_zip = self.dataset_path / calib_zip_name
            if not calib_zip.exists():
                downloadFile(f"{self.url_download_root_calib}/{calib_zip_name}", str(self.dataset_path))
            decompressFile(calib_zip, self.dataset_path)
            calib_marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        raw_img_path = self._raw_path(sequence_name) / "img"
        sequence_path = self.sequence_path(sequence_name)

        for cam, target in ((0, self.rgb_path(sequence_name)), (1, sequence_path / "rgb_1")):
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for src_image in raw_img_path.glob(f"image_{cam}_*.png"):
                if self.target_resolution is None:
                    shutil.copy2(src_image, target / src_image.name)
                else:
                    with Image.open(src_image) as im:
                        target_size = compute_scaled_size(im.size, self.target_resolution)
                        im.resize(target_size, Image.Resampling.LANCZOS).save(target / src_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        raw_path = self._raw_path(sequence_name)
        rows_left = self._parse_image_list(raw_path / "left_images.txt")
        rows_right = self._parse_image_list(raw_path / "right_images.txt")

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = [
            [ts0, f"rgb_0/{name0}", ts1, f"rgb_1/{name1}"]
            for (ts0, name0), (ts1, name1) in zip(rows_left, rows_right)
        ]
        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_imu_csv(self, sequence_name: str) -> None:
        src = self._raw_path(sequence_name) / "imu.txt"
        dst = self.imu_csv_path(sequence_name)

        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        rows = []
        with open(src, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if not parts or parts[0].startswith("#"):
                    continue
                # Columns: id ts(s) wx wy wz ax ay az - drop the leading sample id.
                ts_s, wx, wy, wz, ax, ay, az = parts[1:8]
                rows.append([self._ts_ns(ts_s), float(wx), float(wy), float(wz), float(ax), float(ay), float(az)])
        write_csv_rows(dst, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calib_path = self._calib_path(sequence_name)
        camchain_file = next(calib_path.glob("camchain-imucam-*.yaml"))
        with open(camchain_file, "r", encoding="utf-8") as f:
            cam_data = yaml.safe_load(f)
        with open(calib_path / "imu.yaml", "r", encoding="utf-8") as f:
            imu_data = yaml.safe_load(f)
        if "update_rate" not in imu_data:
            # Kalibr nests the IMU spec under a single top-level key (e.g. imu0:)
            imu_data = next(iter(imu_data.values()))

        rgbs: list[dict[str, Any]] = []
        for idx in (0, 1):
            cam = cam_data[f"cam{idx}"]
            T_cam_imu = np.array(cam["T_cam_imu"], dtype=float).reshape(4, 4)
            # Kalibr intrinsics are at the raw 640x480 resolution - a no-op while
            # target_resolution is unset, kept unconditional so this stays correct if that changes.
            focal_length, principal_point = scale_intrinsics(
                cam["intrinsics"][0:2], cam["intrinsics"][2:4], tuple(cam["resolution"]), self.target_resolution
            )
            rgbs.append(
                {
                    "cam_name": f"rgb_{idx}",
                    "cam_type": "gray",
                    "cam_model": "pinhole",
                    "focal_length": focal_length,
                    "principal_point": principal_point,
                    "distortion_type": "equid4",
                    "distortion_coefficients": [float(c) for c in cam["distortion_coeffs"]],
                    "fps": float(self.rgb_hz),
                    "T_BS": np.linalg.inv(T_cam_imu),
                }
            )

        # Noise-density spec from the group's own Kalibr imu.yaml. The Snapdragon Flight IMU has
        # no documented full-scale range, so a_max/g_max follow the generic saturation defaults
        # already used dataset-wide for IMUs without one (dataset_euroc.py, dataset_vitum.py).
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

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=rgbs, imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        src = self._raw_path(sequence_name) / "groundtruth.txt"
        dst = self.groundtruth_csv_path(sequence_name)

        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        rows = []
        if src.exists():
            with open(src, "r", encoding="utf-8") as fin:
                for line in fin:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith("#"):
                        continue
                    # Columns: ts(s) tx ty tz qx qy qz qw - unlike imu.txt/left_images.txt,
                    # groundtruth.txt has no leading id column.
                    ts_s, tx, ty, tz, qx, qy, qz, qw = parts[:8]
                    rows.append([self._ts_ns(ts_s), float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])
        write_csv_rows(dst, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # The decompressed <group>_calib_snapdragon/ folder is never deleted at any tier - it is
        # re-read by every same-group sequence's create_calibration_yaml (dataset-wide reused).
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self._raw_path(sequence_name), ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"{self._zip_stem(sequence_name)}.zip").unlink(missing_ok=True)
            (self.dataset_path / f"{self._calib_path(sequence_name).name}.zip").unlink(missing_ok=True)

    def _zip_stem(self, sequence_name: str) -> str:
        suffix = "_with_gt" if sequence_name in self._WITH_GT else ""
        return f"{sequence_name}_snapdragon{suffix}"

    def _raw_path(self, sequence_name: str) -> Path:
        return self.dataset_path / self._zip_stem(sequence_name)

    def _calib_path(self, sequence_name: str) -> Path:
        # Sequence names are <environment_group>_<number>; each group shares one calibration.
        group = sequence_name.rsplit("_", 1)[0]
        return self.dataset_path / f"{group}_calib_snapdragon"

    @staticmethod
    def _ts_ns(ts_s: str) -> int:
        return int(Decimal(ts_s) * Decimal(10**9))

    @staticmethod
    def _parse_image_list(list_file: Path) -> list[tuple[int, str]]:
        """Parse a left_images.txt/right_images.txt file into (ts_ns, filename) tuples."""
        rows: list[tuple[int, str]] = []
        with open(list_file, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if not parts or parts[0].startswith("#"):
                    continue
                # Columns: id ts(s) image_path (e.g. img/image_0_0.png) - drop the leading id.
                ts_s, image_path = parts[1], parts[2]
                rows.append((UzhFpvDataset._ts_ns(ts_s), Path(image_path).name))
        return rows
