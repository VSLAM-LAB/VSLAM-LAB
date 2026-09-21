"""
Module: VSLAM-LAB - Datasets - dataset_ariel.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-03
- License: GPLv3 License
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from typing import Any

import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, ensure_hf_sequence_download, hf_token, run_rosbag_frame_extraction, scale_intrinsics,
    write_csv_rows,
)

IMAGE_TOPIC_TEMPLATE = "/alphasense_driver_ros/cam{cam}"
IMU_TOPIC = "/alphasense_driver_ros/imu"

# Calibration is shared by every sequence - a single stereo rig, not re-derived per sequence.
# Only the water-refraction-corrected intrinsics (intrinsics_water/) and the air extrinsics
# (extrinsics_air/, no extrinsics_water/ exists in the repo) are available - confirmed via the
# repo's own file listing (calibrations/cam0_cam1_stereo/{intrinsics_air,intrinsics_water,extrinsics_air}).
INTRINSICS_FILE = "calibrations/cam0_cam1_stereo/intrinsics_water/camchain-stereo-intrinsics-underwater.yaml"
EXTRINSICS_FILE = "calibrations/cam0_cam1_stereo/extrinsics_air/camchain-imucam-stereo-extrinsics-air.yaml"
IMU_NOISE_FILE = "calibrations/cam0_cam1_stereo/extrinsics_air/imu0_alphasense_noise.yaml"


class ArielDataset(DatasetVSLAMLAB):
    """Multi-Camera Underwater Visual-Inertial Dataset (Ariel/NTNU-ARL) helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "ariel") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id: str = self.cfg["hf_repo_id"]

    def _group(self, sequence_name: str) -> str:
        return sequence_name.split("_", 1)[0]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        remote_dir = f"subset-{self._group(sequence_name)}/{sequence_name}"
        ensure_hf_sequence_download(self.hf_repo_id, [remote_dir], sequence_path, token=hf_token())

        for remote_file in (INTRINSICS_FILE, EXTRINSICS_FILE, IMU_NOISE_FILE):
            local_file = self.dataset_path / remote_file
            if not local_file.exists():
                hf_hub_download(repo_id=self.hf_repo_id, filename=remote_file, repo_type="dataset",
                                 token=hf_token(), local_dir=self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        raw_path = sequence_path / "rgb_raw"

        for cam in ("0", "1"):
            image_topic = IMAGE_TOPIC_TEMPLATE.format(cam=cam)
            run_rosbag_frame_extraction("ros1", rosbag_path, raw_path, image_topic, cam)

        for cam, final_path in (("0", self.rgb_path(sequence_name)), ("1", sequence_path / "rgb_1")):
            if final_path.exists():
                continue
            final_path.mkdir(parents=True, exist_ok=True)
            for raw_image in sorted((raw_path / f"rgb_{cam}").glob("*.png")):
                with Image.open(raw_image) as img:
                    target_size = compute_scaled_size(img.size, self.target_resolution)
                    img.resize(target_size, Image.Resampling.LANCZOS).save(final_path / raw_image.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        # extract-rosbag-frames joins cam0's and cam1's rows by position, not timestamp. Verified
        # against fjord_1's real extracted data (SKILL.md step 8): the two cameras' timestamps stay
        # aligned within a stable ~55ms +/- 0.8ms offset for all 6367 paired rows (no drift), so the
        # positional join is valid here - the only defect is a trailing unpaired frame when one
        # camera's stream is one message longer than the other's. Drop any incomplete row rather
        # than assume a fixed misalignment to shift/correct.
        raw_csv = self.sequence_path(sequence_name) / "rgb_raw" / "rgb.csv"
        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = []
        with open(raw_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not all(row[col] for col in header):
                    continue
                rows.append([int(row["ts_rgb_0 (ns)"]), row["path_rgb_0"],
                             int(row["ts_rgb_1 (ns)"]), row["path_rgb_1"]])
        write_csv_rows(self.rgb_csv_path(sequence_name), header, rows)

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        inputs = f"--rosbag_path {rosbag_path} --sequence_path {sequence_path} --imu_topic {IMU_TOPIC}"
        subprocess.run(f"pixi run -e ros1 extract-rosbag-imu {inputs}", shell=True, check=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        with (self.dataset_path / INTRINSICS_FILE).open("r", encoding="utf-8") as f:
            intrinsics = yaml.safe_load(f)
        with (self.dataset_path / EXTRINSICS_FILE).open("r", encoding="utf-8") as f:
            extrinsics = yaml.safe_load(f)
        with (self.dataset_path / IMU_NOISE_FILE).open("r", encoding="utf-8") as f:
            imu_noise = yaml.safe_load(f)

        cam0_int, cam1_int = intrinsics["cam0"], intrinsics["cam1"]
        cam0_ext, cam1_ext = extrinsics["cam0"], extrinsics["cam1"]

        T_cam0_imu = np.array(cam0_ext["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1_ext["T_cam_imu"], dtype=float).reshape(4, 4)

        # Kalibr calibration intrinsics are computed at each camera's raw (pre-resize) resolution -
        # rescale to match the actual images create_rgb_folder wrote into rgb_0/rgb_1 (VSLAM-LAB
        # issue #99). Model: dataset_hilti2022.py/dataset_hilti2026.py.
        focal_length_0, principal_point_0 = scale_intrinsics(
            cam0_int["intrinsics"][0:2], cam0_int["intrinsics"][2:4], tuple(cam0_int["resolution"]),
            self.target_resolution,
        )
        focal_length_1, principal_point_1 = scale_intrinsics(
            cam1_int["intrinsics"][0:2], cam1_int["intrinsics"][2:4], tuple(cam1_int["resolution"]),
            self.target_resolution,
        )

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": focal_length_0,
            "principal_point": principal_point_0,
            "distortion_type": "equid4",
            "distortion_coefficients": [float(v) for v in cam0_int["distortion_coeffs"]],
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
            "distortion_coefficients": [float(v) for v in cam1_int["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        # IMU noise-density values are the real per-unit measurements shipped with this dataset
        # (imu0_alphasense_noise.yaml). a_max/g_max/sigma_bg/sigma_ba/g/g0/a0/s_a have no
        # per-dataset source - taken from the same Alphasense Core hardware's manufacturer example
        # (sevensense-robotics/core_research_manual, files/example_7s_sensors_dont_use.yaml),
        # confirmed as the right hardware reference by its matching /alphasense_driver_ros/*
        # topic names. Model: dataset_hilti2022.py uses the same manufacturer values.
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 150.0,
            "g_max": 7.5,
            "sigma_g_c": float(imu_noise["gyroscope_noise_density"]),
            "sigma_a_c": float(imu_noise["accelerometer_noise_density"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": float(imu_noise["gyroscope_random_walk"]),
            "sigma_aw_c": float(imu_noise["accelerometer_random_walk"]),
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(imu_noise["update_rate"]),
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_tum = sequence_path / f"{sequence_name}_baseline.tum"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        rows = []
        with open(groundtruth_tum, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {line_num} in {groundtruth_tum}: "
                        f"expected 8 columns, got {len(parts)}"
                    )
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(round(float(ts_s) * 1e9))
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])

        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "rgb_raw", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / f"{sequence_name}.bag").unlink(missing_ok=True)
            # calibrations/ itself is dataset-wide and re-read by every sequence's
            # create_calibration_yaml - never delete it.
