"""
Module: VSLAM-LAB - Datasets - dataset_hilti2022.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-02
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import numpy as np
import yaml
from PIL import Image

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import (
    compute_scaled_size, decompressFile, downloadFile, run_rosbag_frame_extraction, scale_intrinsics,
    write_csv_rows,
)

IMAGE_TOPIC_TEMPLATE = "/alphasense/cam{cam}/image_raw"
IMU_TOPIC = "/alphasense/imu"
CALIBRATION_ARCHIVE = "2022322_calibration_files.zip"
CALIBRATION_YAML = "calib_3_cam0-1-camchain-imucam.yaml"

# Ground-truth availability, verified against https://hilti-challenge.com/assets/2022/ground_truth/:
# only these three sequences have a dense, real 6DOF trajectory (an "_imu.txt" file, hundreds of
# rows, genuine quaternions). Every other sequence with a public GT file (the bare "<name>.txt")
# only has a handful of sparse waypoints with an identity-quaternion placeholder (0, 0, 0, 1) - not
# a usable trajectory, but still real position data, so it's still written out.
DENSE_GT_SEQUENCES = {"exp14_basement_2", "exp16_attic_to_upper_gallery_2", "exp18_corridor_lower_gallery_2"}
# No public ground truth file exists at all for this sequence (confirmed via HTTP 404).
NO_GT_SEQUENCES = {"exp10_cupola_2"}


def _gt_name(sequence_name: str) -> str | None:
    if sequence_name in NO_GT_SEQUENCES:
        return None
    if sequence_name in DENSE_GT_SEQUENCES:
        return f"{sequence_name}_imu.txt"
    return f"{sequence_name}.txt"


class Hilti2022Dataset(DatasetVSLAMLAB):
    """Hilti-Oxford SLAM Challenge 2022 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hilti2022") -> None:
        super().__init__(dataset_name)

        # Get download urls
        self.url_download_root: str = self.cfg["url_download_root"]
        self.url_download_root_gt: str = self.cfg["url_download_root_gt"]

        # Sequence nicknames - drop the descriptive suffix, keep just "expNN"
        self.sequence_nicknames = [s.split("_", 1)[0] for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        # Rosbag (one per sequence)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        if not rosbag_path.exists():
            downloadFile(f"{self.url_download_root}/{sequence_name}.bag", str(sequence_path))

        # Calibration (one shared rig for every sequence in the challenge)
        calibration_folder = self.dataset_path / "calibration_files"
        if not calibration_folder.exists():
            compressed_file = self.dataset_path / CALIBRATION_ARCHIVE
            downloadFile(f"{self.url_download_root}/{CALIBRATION_ARCHIVE}", str(self.dataset_path))
            decompressFile(compressed_file, calibration_folder)

        # Ground truth (not every sequence has one - see _gt_name)
        gt_name = _gt_name(sequence_name)
        if gt_name is not None:
            gt_path = sequence_path / gt_name
            if not gt_path.exists():
                downloadFile(f"{self.url_download_root_gt}/{gt_name}", str(sequence_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        raw_path = sequence_path / "rgb_raw"

        # extract-rosbag-frames has no notion of self.target_resolution - it always writes at the
        # bag's native 720x540. Extract into a throwaway raw_path first, then resize into the real
        # rgb_0/rgb_1 below (same rgb_raw-then-resize shape as dataset_pamir.py).
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
        # extract-rosbag-frames already wrote rgb_raw/rgb.csv with paths like "rgb_0/<ts>.png" -
        # identical to the final rgb_0/rgb_1 layout (same filenames, just resized in place above),
        # so the raw csv is valid as-is.
        raw_csv = self.sequence_path(sequence_name) / "rgb_raw" / "rgb.csv"
        shutil.copy2(raw_csv, self.rgb_csv_path(sequence_name))

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_path = sequence_path / f"{sequence_name}.bag"
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        inputs = f"--rosbag_path {rosbag_path} --sequence_path {sequence_path} --imu_topic {IMU_TOPIC}"
        subprocess.run(f"pixi run -e ros1 extract-rosbag-imu {inputs}", shell=True, check=True)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_yaml = self.dataset_path / "calibration_files" / CALIBRATION_YAML
        with calibration_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam0 = data["cam0"]
        cam1 = data["cam1"]
        T_cam0_imu = np.array(cam0["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1["T_cam_imu"], dtype=float).reshape(4, 4)

        # Kalibr calibration intrinsics are computed at each camera's raw (pre-resize) resolution -
        # rescale to match the actual images create_rgb_folder wrote into rgb_0/rgb_1 (VSLAM-LAB
        # issue #99). Model: dataset_hilti2026.py.
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
            "distortion_coefficients": cam0["distortion_coeffs"],
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
            "distortion_coefficients": cam1["distortion_coeffs"],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        # IMU noise-density spec for the real hardware (Sevensense Alphasense Core, confirmed via
        # the bag's own /alphasense/imu topic and hardware_id) - taken from the manufacturer's own
        # example calibration in sevensense-robotics/alphasense_core_manual (files/
        # example_7s_sensors_dont_use.yaml), not a generic/copied-from-another-dataset default.
        # fps is the real measured rate from the downloaded exp14_basement_2.bag (399.2 Hz).
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 150.0,
            "g_max": 7.5,
            "sigma_g_c": 0.019,
            "sigma_a_c": 0.019,
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": 0.000266,
            "sigma_aw_c": 0.0043,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 400.0,
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        gt_name = _gt_name(sequence_name)
        if gt_name is None:
            write_csv_rows(groundtruth_csv, header, [])
            return

        gt_txt = self.sequence_path(sequence_name) / gt_name
        rows = []
        with open(gt_txt, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {line_num} in {gt_txt}: expected 8 columns, got {len(parts)}"
                    )
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(round(float(ts_s) * 1e9))
                rows.append([ts_ns, float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw)])
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "rgb_raw", ignore_errors=True)
            gt_name = _gt_name(sequence_name)
            if gt_name is not None:
                (sequence_path / gt_name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / f"{sequence_name}.bag").unlink(missing_ok=True)
            # calibration_files/ itself is dataset-wide and re-read by every sequence's
            # create_calibration_yaml - never delete it, only the archive that produced it.
            (self.dataset_path / CALIBRATION_ARCHIVE).unlink(missing_ok=True)
