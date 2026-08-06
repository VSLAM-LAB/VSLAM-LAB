"""
Module: VSLAM-LAB - Datasets - dataset_rvp_handheld.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-06
- License: GPLv3 License
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, downloadFile, run_rosbag_frame_extraction, scale_intrinsics, write_csv_rows

IMAGE_TOPIC = {"0": "/camera_left/image_raw", "1": "/camera_right/image_raw"}
IMU_TOPIC = "/imu/data"
CALIBRATION_FILENAME = "vbr_calib.yaml"

# Number of sequentially-numbered "<sequence_name>_NN.bag" chunks each sequence is split into -
# every sequence ships as multiple ~10.7GB rosbag files rather than one. Verified against every
# sequence's own manifest page (http://srrg.diag.uniroma1.it/<sequence_name>.html) on 2026-08-06.
SEQUENCE_CHUNKS: dict[str, int] = {
    "campus_test0": 5, "campus_test1": 4, "campus_train0": 5, "campus_train1": 5,
    "ciampino_test0": 5, "ciampino_test1": 4, "ciampino_train0": 12, "ciampino_train1": 7,
    "colosseo_test0": 7, "colosseo_train0": 12,
    "diag_test0": 6, "diag_train0": 13,
    "pincio_test0": 14, "pincio_train0": 14,
    "spagna_test0": 6, "spagna_train0": 18,
}


def _gt_filename(sequence_name: str) -> str | None:
    # Verified against every sequence's manifest page on 2026-08-06: a real "<name>_gt.txt" link
    # exists iff "train" is in the name - "test" sequences are a held-out benchmark split with no
    # public ground truth at all.
    return f"{sequence_name}_gt.txt" if "train" in sequence_name else None


def _debayer_and_save(raw_image: Path, target_path: Path, target_resolution: tuple[int, int] | None) -> None:
    # The camera_left/camera_right rostopics publish raw Bayer RGGB8 frames (VBR's own
    # calibration_description documents "Bayer RGGB8 format"); cv_bridge's passthrough decoding
    # (extract_rosbag_frames.py) writes that mosaic out unmodified, so it still needs debayering
    # here. ROS's "bayer_rggb8" encoding corresponds to OpenCV's BayerBG code, not BayerRG - ROS
    # and OpenCV name Bayer patterns from opposite corners of the 2x2 tile (a well-known one-pixel
    # offset between the two conventions).
    mosaic = cv2.imread(str(raw_image), cv2.IMREAD_UNCHANGED)
    bgr = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR)
    if target_resolution is not None:
        scaled_w, scaled_h = compute_scaled_size((mosaic.shape[1], mosaic.shape[0]), target_resolution)
        bgr = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(target_path), bgr)


class RvpBagsMixin:
    """Shared VBR (rvp-group.net) rosbag handling for the rvp-handheld/rvp-car datasets.

    Every sequence ships as several chunked "<sequence_name>_NN.bag" files (SEQUENCE_CHUNKS)
    rather than one, and the calibration file is byte-identical across every sequence in both
    datasets (confirmed by diffing vbr_calib.yaml across five different locations/platforms) - it
    is downloaded once per dataset instance and reused for every sequence, never per-sequence.
    """

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)
        self.url_download_root: str = self.cfg["url_download_root"]

    def _location(self, sequence_name: str) -> str:
        return sequence_name.split("_", 1)[0]

    def _sequence_url_root(self, sequence_name: str) -> str:
        return f"{self.url_download_root}/{self._location(sequence_name)}/{sequence_name}"

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)
        marker = sequence_path / ".download_complete"
        if marker.exists():
            return

        url_root = self._sequence_url_root(sequence_name)

        calibration_path = self.dataset_path / CALIBRATION_FILENAME
        if not calibration_path.exists():
            downloadFile(f"{url_root}/{CALIBRATION_FILENAME}", str(self.dataset_path))

        gt_name = _gt_filename(sequence_name)
        if gt_name is not None and not (sequence_path / gt_name).exists():
            downloadFile(f"{url_root}/{gt_name}", str(sequence_path))

        for chunk in range(SEQUENCE_CHUNKS[sequence_name]):
            bag_name = f"{sequence_name}_{chunk:02d}.bag"
            if not (sequence_path / bag_name).exists():
                downloadFile(f"{url_root}/{bag_name}", str(sequence_path))

        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        marker = sequence_path / ".rgb_folder_complete"
        if marker.exists():
            return

        final_paths = {"0": self.rgb_path(sequence_name), "1": sequence_path / "rgb_1"}
        for final_path in final_paths.values():
            final_path.mkdir(parents=True, exist_ok=True)

        for chunk in range(SEQUENCE_CHUNKS[sequence_name]):
            bag_path = sequence_path / f"{sequence_name}_{chunk:02d}.bag"
            chunk_path = sequence_path / f"rgb_raw_{chunk:02d}"
            for cam, topic in IMAGE_TOPIC.items():
                run_rosbag_frame_extraction("ros1", bag_path, chunk_path, topic, cam)

            for cam, final_path in final_paths.items():
                for raw_image in sorted((chunk_path / f"rgb_{cam}").glob("*.png")):
                    _debayer_and_save(raw_image, final_path / raw_image.name, self.target_resolution)

        marker.touch()

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        frames = [
            pd.read_csv(sequence_path / f"rgb_raw_{chunk:02d}" / "rgb.csv")
            for chunk in range(SEQUENCE_CHUNKS[sequence_name])
        ]
        # extract_rosbag_frames.py has no de-duplication: a handful of rostime collisions per
        # chunk (two messages landing on the same nanosecond, confirmed on campus_test1's chunks
        # 01/03) make cv2.imwrite silently overwrite the first PNG with the second, but both
        # still get appended as separate rgb.csv rows - drop_duplicates keeps rgb.csv in sync with
        # the (deduplicated-by-overwrite) files actually on disk in rgb_0/rgb_1.
        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["ts_rgb_0 (ns)"])
            .sort_values("ts_rgb_0 (ns)")
            .reset_index(drop=True)
        )

        rgb_csv = self.rgb_csv_path(sequence_name)
        tmp = rgb_csv.with_suffix(".csv.tmp")
        combined.to_csv(tmp, index=False)
        tmp.replace(rgb_csv)

    def create_imu_csv(self, sequence_name: str) -> None:
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        sequence_path = self.sequence_path(sequence_name)
        frames = []
        for chunk in range(SEQUENCE_CHUNKS[sequence_name]):
            bag_path = sequence_path / f"{sequence_name}_{chunk:02d}.bag"
            chunk_path = sequence_path / f"imu_raw_{chunk:02d}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            inputs = f"--rosbag_path {bag_path} --sequence_path {chunk_path} --imu_topic {IMU_TOPIC}"
            subprocess.run(f"pixi run -e ros1 extract-rosbag-imu {inputs}", shell=True, check=True)
            frames.append(pd.read_csv(chunk_path / "imu_0.csv"))

        combined = pd.concat(frames, ignore_index=True).sort_values("ts (ns)").reset_index(drop=True)
        tmp = imu_csv.with_suffix(".csv.tmp")
        combined.to_csv(tmp, index=False)
        tmp.replace(imu_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_path = self.dataset_path / CALIBRATION_FILENAME
        with calibration_path.open("r", encoding="utf-8") as f:
            calib = yaml.safe_load(f)

        cam_l, cam_r, imu_cfg = calib["cam_l"], calib["cam_r"], calib["imu"]

        focal_length_0, principal_point_0 = scale_intrinsics(
            cam_l["intrinsics"][0:2], cam_l["intrinsics"][2:4], tuple(cam_l["resolution"]), self.target_resolution
        )
        focal_length_1, principal_point_1 = scale_intrinsics(
            cam_r["intrinsics"][0:2], cam_r["intrinsics"][2:4], tuple(cam_r["resolution"]), self.target_resolution
        )

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": focal_length_0,
            "principal_point": principal_point_0,
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(c) for c in cam_l["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            # cam_l's own T_b is already the sensor's pose wrt the rig's base/body frame (VBR's
            # calibration_description: "Transformation matrix representing the sensor's pose
            # relative to the base frame") - unlike Kalibr's T_cam_imu convention, no inversion.
            "T_BS": np.array(cam_l["T_b"], dtype=float),
        }
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": focal_length_1,
            "principal_point": principal_point_1,
            "distortion_type": "radtan4",
            "distortion_coefficients": [float(c) for c in cam_r["distortion_coeffs"]],
            "fps": float(self.rgb_hz),
            "T_BS": np.array(cam_r["T_b"], dtype=float),
        }

        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            # a_max/g_max: typical SBG Ellipse-series MEMS IMU full-scale range (+-16g / +-450
            # deg/s) - VBR's own calibration file only publishes noise-density parameters, not a
            # per-unit saturation spec.
            "a_max": 156.96,
            "g_max": 7.85,
            "sigma_g_c": float(imu_cfg["gyroscope_noise_density"]),
            "sigma_gw_c": float(imu_cfg["gyroscope_random_walk"]),
            "sigma_a_c": float(imu_cfg["accelerometer_noise_density"]),
            "sigma_aw_c": float(imu_cfg["accelerometer_random_walk"]),
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "a0": [0.0, 0.0, 0.0],
            "g0": [0.0, 0.0, 0.0],
            "g": 9.80665,
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(imu_cfg["update_rate"]),
            "T_BS": np.array(imu_cfg["T_b"], dtype=float),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        gt_name = _gt_filename(sequence_name)
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
        num_chunks = SEQUENCE_CHUNKS[sequence_name]

        if BENCHMARK_RETENTION != Retention.FULL:
            for chunk in range(num_chunks):
                shutil.rmtree(sequence_path / f"rgb_raw_{chunk:02d}", ignore_errors=True)
                shutil.rmtree(sequence_path / f"imu_raw_{chunk:02d}", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for chunk in range(num_chunks):
                (sequence_path / f"{sequence_name}_{chunk:02d}.bag").unlink(missing_ok=True)
            gt_name = _gt_filename(sequence_name)
            if gt_name is not None:
                (sequence_path / gt_name).unlink(missing_ok=True)
            # CALIBRATION_FILENAME lives at self.dataset_path (dataset-wide, byte-identical across
            # every sequence, re-read by every create_calibration_yaml call) - never delete it.


class RvpHandheldDataset(RvpBagsMixin, DatasetVSLAMLAB):
    """VBR: A Vision Benchmark in Rome (hand-held sequences) dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "rvp-handheld") -> None:
        super().__init__(dataset_name)
