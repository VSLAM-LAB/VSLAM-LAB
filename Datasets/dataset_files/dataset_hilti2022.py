import csv
import os
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import decompressFile, downloadFile

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class Hilti2022Dataset(DatasetVSLAMLAB):
    """HILTI 2022 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hilti2022") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Create sequence_nicknames
        self.sequence_nicknames = [s.split("_", 1)[0] for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        # Download rosbag
        rosbag = sequence_name + ".bag"
        rosbag_path = sequence_path / rosbag
        if not rosbag_path.exists():
            download_url = f"{self.url_download_root}/{rosbag}"
            downloadFile(download_url, sequence_path)

        # Download calibration files
        decompressed_folder = self.dataset_path / "calibration_files"
        if not decompressed_folder.exists():
            compressed_name_ext = "2022322_calibration_files.zip"
            cal_url = f"{self.url_download_root}/{compressed_name_ext}"
            compressed_file = self.dataset_path / compressed_name_ext

            downloadFile(cal_url, self.dataset_path)
            decompressFile(compressed_file, decompressed_folder)

        # Download gt
        gt_name = self.get_gt_name(sequence_name)
        gt_txt = sequence_path / gt_name
        if not gt_txt.exists():
            gt_url = f"https://hilti-challenge.com/assets/2022/ground_truth/{gt_name}"
            downloadFile(gt_url, sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_name = f"{sequence_name}.bag"
        rosbag = sequence_path / rosbag_name
        for cam in ["0", "1"]:
            image_topic = f"/alphasense/cam{cam}/image_raw"
            rgb_path = sequence_path / f"rgb_{cam}"
            if rgb_path.exists():
                continue
            rgb_path.mkdir(parents=True, exist_ok=True)

            inputs = f"--rosbag_path {rosbag} --sequence_path {sequence_path} --image_topic {image_topic} --cam {cam}"
            command = f"pixi run -e ros1 extract-rosbag-frames {inputs}"
            subprocess.run(command, shell=True)

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_name = f"{sequence_name}.bag"
        rosbag = sequence_path / rosbag_name
        imu_topic = "/alphasense/imu"
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        inputs = f"--rosbag_path {rosbag} --sequence_path {sequence_path} --imu_topic {imu_topic}"
        command = f"pixi run -e ros1 extract-rosbag-imu {inputs}"
        subprocess.run(command, shell=True)

        rgb_csv = self.rgb_csv_path(sequence_name)
        imu_csv = self.imu_csv_path(sequence_name)
        rgb = pd.read_csv(rgb_csv)
        imu = pd.read_csv(imu_csv)

        rgb_0_ts_col = "ts_rgb_0 (ns)"
        rgb_1_ts_col = "ts_rgb_1 (ns)"
        imu_ts_col = "ts (ns)"

        imu[imu_ts_col] = imu[imu_ts_col].astype("int64")
        rgb[rgb_0_ts_col] = rgb[rgb_0_ts_col].astype("int64")
        rgb[rgb_1_ts_col] = rgb[rgb_1_ts_col].astype("int64")

        rgb.to_csv(rgb_csv, index=False)
        imu.to_csv(imu_csv, index=False)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_folder = self.dataset_path / "calibration_files"
        calibration_yaml = calibration_folder / "calib_3_cam0-1-camchain-imucam.yaml"

        with calibration_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam0 = data["cam0"]
        cam1 = data["cam1"]

        T_cam0_imu = np.array(cam0["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1["T_cam_imu"], dtype=float).reshape(4, 4)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam0["intrinsics"][0:2],
            "principal_point": cam0["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam0["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam0_imu),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam1["intrinsics"][0:2],
            "principal_point": cam1["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam1["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": 20.0e-4,
            "sigma_a_c": 20.0e-3,
            "sigma_bg": 0.01,
            "sigma_ba": 0.1,
            "sigma_gw_c": 20.0e-5,
            "sigma_aw_c": 20.0e-3,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.1, 0.04, 0.15],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 200.0,
            "T_BS": np.array(np.eye(4)).reshape((4, 4)),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = self.sequence_path(sequence_name)
        gt_name = self.get_gt_name(sequence_name)
        groundtruth_txt = sequence_path / gt_name
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with (
            open(groundtruth_txt, "r", encoding="utf-8") as fin,
            open(tmp, "w", newline="", encoding="utf-8") as fout,
        ):
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for idx, line in enumerate(fin, start=0):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {idx + 1} in {groundtruth_txt}: expected 8 columns, got {len(parts)}"
                    )

                ts_s, tx, ty, tz, qx, qy, qz, qw = parts
                ts_ns = int(round(float(ts_s) * 1e9))
                w.writerow(
                    [
                        ts_ns,
                        float(tx),
                        float(ty),
                        float(tz),
                        float(qx),
                        float(qy),
                        float(qz),
                        float(qw),
                    ]
                )
        tmp.replace(groundtruth_csv)

    def get_gt_name(self, sequence_name):
        if sequence_name == "exp04_construction_upper_level":
            return "exp01_construction_ground_level.txt"
        if sequence_name == "exp14_basement_2":
            return "exp14_basement_2_imu.txt"
