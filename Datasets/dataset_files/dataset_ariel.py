import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, HfFileSystem, login
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import HUGGINGFACE_TOKEN

INITIAL_TIMESTAMP: int = 1_700_604_776_000_000_000


class ArielDataset(DatasetVSLAMLAB):
    """ARIEL dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "ariel") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.repo_id = cfg["repo_id"]

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_name = f"{sequence_name}.bag"
        rosbag = sequence_path / rosbag_name
        calibration_folder = self.dataset_path / "calibrations"

        if HUGGINGFACE_TOKEN is not None:
            login(token=HUGGINGFACE_TOKEN)
            token = HUGGINGFACE_TOKEN
        else:
            token = os.environ.get("HF_TOKEN")

        api = HfApi(token=token)
        fs = HfFileSystem(token=token)
        disable_progress_bars()

        cache_file = self.dataset_path / "all_files_cache.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                all_files = json.load(f)
        else:
            all_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_files, f, indent=2)
            print(f"Fetched and cached {len(all_files)} files")

        # Download calibration files
        if not calibration_folder.exists():
            remote_folder = "calibrations"
            files = [f for f in all_files if f.startswith(remote_folder + "/")]
            for remote_file in tqdm(files, desc="Downloading calibration files", unit="file"):
                local_file = self.dataset_path / remote_file
                fs.get_file(f"datasets/{self.repo_id}/{remote_file}", str(local_file))

        # Download rosbag
        if rosbag.exists():
            return

        remote_folder = self._remote_folder(sequence_name)
        files = [f for f in all_files if f.startswith(remote_folder + "/")]
        for remote_file in tqdm(files, desc="Downloading rosbag files", unit="file"):
            local_file = sequence_path / Path(remote_file).name
            fs.get_file(f"datasets/{self.repo_id}/{remote_file}", str(local_file))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_name = f"{sequence_name}.bag"
        rosbag = sequence_path / rosbag_name

        for cam in ["0", "1"]:
            image_topic = f"/alphasense_driver_ros/cam{cam}"
            rgb_path = sequence_path / f"rgb_{cam}"
            if rgb_path.exists():
                continue
            rgb_path.mkdir(parents=True, exist_ok=True)

            inputs = f"--rosbag_path {rosbag} --sequence_path {sequence_path} --image_topic {image_topic} --cam {cam}"
            command = f"pixi run -e ros1 extract-rosbag-frames {inputs}"
            subprocess.run(command, shell=True)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb = pd.read_csv(rgb_csv)

        col0 = "path_rgb_0"
        col1 = "path_rgb_1"
        shift_cols = ["path_rgb_0", "ts_rgb_0 (ns)"]

        n0 = rgb[col0].notna().sum()
        n1 = rgb[col1].notna().sum()

        if n0 != n1:
            rgb[shift_cols] = rgb[shift_cols].shift(-1)

            # Drop rows made invalid by the shift
            rgb = rgb.dropna(subset=["path_rgb_0", "ts_rgb_0 (ns)"]).copy()

            # Remove the first N rows
            n_remove = 20
            if n_remove > 0:
                rgb = rgb.iloc[n_remove:].reset_index(drop=True)

            tmp = rgb_csv.with_name(f"{rgb_csv.name}.tmp")
            try:
                rgb.to_csv(tmp, index=False)
                tmp.replace(rgb_csv)
            finally:
                if tmp.exists():
                    tmp.unlink()

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag_name = f"{sequence_name}.bag"
        rosbag = sequence_path / rosbag_name
        imu_topic = "/alphasense_driver_ros/imu"
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

        # Make first IMU timestamp 0 ns, and shift RGB to the same time origin
        imu[imu_ts_col] = imu[imu_ts_col].astype("int64") - INITIAL_TIMESTAMP
        rgb[rgb_0_ts_col] = rgb[rgb_0_ts_col].astype("int64") - INITIAL_TIMESTAMP
        rgb[rgb_1_ts_col] = rgb[rgb_1_ts_col].astype("int64") - INITIAL_TIMESTAMP

        rgb.to_csv(rgb_csv, index=False)
        imu.to_csv(imu_csv, index=False)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_folder = self.dataset_path / "calibrations"
        intrinsics_water_yaml = (
            calibration_folder / "cam0_cam1_stereo" / "intrinsics_water" / "camchain-stereo-intrinsics-underwater.yaml"
        )
        extrinsics_air = (
            calibration_folder / "cam0_cam1_stereo" / "extrinsics_air" / "camchain-imucam-stereo-extrinsics-air.yaml"
        )

        with intrinsics_water_yaml.open("r", encoding="utf-8") as f:
            data_int = yaml.safe_load(f)

        with extrinsics_air.open("r", encoding="utf-8") as f:
            data_ext = yaml.safe_load(f)

        cam0_int = data_int["cam0"]
        cam1_int = data_int["cam1"]
        cam0_ext = data_ext["cam0"]
        cam1_ext = data_ext["cam1"]

        T_cam0_imu = np.array(cam0_ext["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1_ext["T_cam_imu"], dtype=float).reshape(4, 4)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam0_int["intrinsics"][0:2],
            "principal_point": cam0_int["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam0_int["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam0_imu),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam1_int["intrinsics"][0:2],
            "principal_point": cam1_int["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam1_int["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": 5.87e-04,
            "sigma_a_c": 1.86e-02,
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": 2.866e-03,
            "sigma_aw_c": 4.33e-03,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 200.0,
            "T_BS": np.array(np.eye(4)).reshape((4, 4)),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_tum = sequence_path / f"{sequence_name}_baseline.tum"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(groundtruth_tum, "r", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

            for line_num, line in enumerate(fin, start=1):
                line = line.strip()

                # Skip empty lines or comments
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid groundtruth line {line_num} in {groundtruth_tum}: "
                        f"expected 8 columns, got {len(parts)}"
                    )

                ts_s, tx, ty, tz, qx, qy, qz, qw = parts

                # TUM timestamp is usually in seconds -> convert to nanoseconds
                ts_ns = int(round(float(ts_s) * 1e9)) - INITIAL_TIMESTAMP

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

    def _remote_folder(self, sequence_name: str) -> str:
        if "fjord" in sequence_name:
            return f"subset-fjord/{sequence_name}"
        if "mclab" in sequence_name:
            return f"subset-mclab/{sequence_name}"
