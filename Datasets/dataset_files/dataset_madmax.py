"""
Module: VSLAM-LAB - Datasets - dataset_madmax.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-03-07
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile, write_csv_rows


class MadmaxDataset(DatasetVSLAMLAB):
    """MADMAX Mars rover navigation dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "madmax") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]

        # API token
        self.api_token: str = self.cfg.get("api_token", "not_set")

        # Dataset homepage
        self.dataset_homepage: str = self.cfg["about"]["homepage"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        remote_file_urls = self._get_file_url(sequence_name)
        folders = ["rgb_0", "rgb_1", "calibration", "imu_raw.csv", "groundtruth", "depth_0"]
        for url, folder in zip(remote_file_urls, folders):
            download_url = f"{self.url_download_root}/{url}"
            downloaded_file = sequence_path / url
            if folder != "imu_raw.csv":
                compressed_file = sequence_path / f"{url}.zip"
            else:
                compressed_file = sequence_path / folder
            if not compressed_file.exists():
                downloadFile(download_url, sequence_path)
                downloaded_file.rename(compressed_file)

            folder_path = sequence_path / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                decompressFile(compressed_file, folder_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        rgb_0_path = self.rgb_path(sequence_name)
        rgb_0_files = sorted(f for f in os.listdir(rgb_0_path) if (rgb_0_path / f).is_file())

        rgb_1_path = sequence_path / "rgb_1"
        rgb_1_files = sorted(f for f in os.listdir(rgb_1_path) if (rgb_1_path / f).is_file())

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]
        rows = []
        for filename_0, filename_1 in zip(rgb_0_files, rgb_1_files):
            name_0, _ = os.path.splitext(filename_0)
            name_1, _ = os.path.splitext(filename_1)
            name_0 = name_0.replace("img_rect_left_", "")
            name_1 = name_1.replace("img_rect_right_", "")
            ts_ns_0 = int(float(name_0))
            ts_ns_1 = int(float(name_1))
            rows.append([ts_ns_0, f"rgb_0/{filename_0}", ts_ns_1, f"rgb_1/{filename_1}"])
        write_csv_rows(rgb_csv, header, rows)

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        imu_csv = self.imu_csv_path(sequence_name)
        imu_raw_csv = sequence_path / "imu_raw.csv"
        df = pd.read_csv(imu_raw_csv)
        selected_columns = [
            "field.header.stamp",
            "field.angular_velocity.x",
            "field.angular_velocity.y",
            "field.angular_velocity.z",
            "field.linear_acceleration.x",
            "field.linear_acceleration.y",
            "field.linear_acceleration.z",
        ]
        header = [
            "ts (ns)",
            "wx (rad s^-1)",
            "wy (rad s^-1)",
            "wz (rad s^-1)",
            "ax (m s^-2)",
            "ay (m s^-2)",
            "az (m s^-2)",
        ]
        rows = df[selected_columns].astype(object).values.tolist()
        write_csv_rows(imu_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_folder = self.sequence_path(sequence_name) / "calibration" / "calibration"
        intrinsics_0_txt = calibration_folder / "camera_rect_left_info.txt"
        intrinsics_1_txt = calibration_folder / "camera_rect_right_info.txt"

        extrinsics_0_csv = calibration_folder / "tf__imu_to_camera_left.csv"
        extrinsics_0_1_csv = calibration_folder / "tf__camera_left_to_camera_right.csv"

        with open(intrinsics_0_txt, "r", encoding="utf-8") as f:
            data_0 = yaml.safe_load(f)
        P_0 = np.array(data_0["P"], dtype=float).reshape(3, 4).tolist()

        with open(intrinsics_1_txt, "r", encoding="utf-8") as f:
            data_1 = yaml.safe_load(f)
        P_1 = np.array(data_1["P"], dtype=float).reshape(3, 4).tolist()

        T_cam0_imu, data_ext_0_imu = self._load_extrinsics_matrix(extrinsics_0_csv)
        T_cam0_1, data_ext_0_1 = self._load_extrinsics_matrix(extrinsics_0_1_csv)
        T_cam1_imu = np.linalg.inv(T_cam0_1) @ T_cam0_imu

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": [P_0[0][0], P_0[1][1]],
            "principal_point": [P_0[0][2], P_0[1][2]],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam0_imu),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": [P_1[0][0], P_1[1][1]],
            "principal_point": [P_1[0][2], P_1[1][2]],
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
            "a0": [-0.05, 0.09, 0.01],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 100.0,
            "T_BS": np.array(np.eye(4)).reshape((4, 4)),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        gt_6DoF_gnss_and_imu_csv = (
            sequence_path / "groundtruth" / f"{sequence_name}_ground_truth" / "gt_6DoF_gnss_and_imu.csv"
        )
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        df = pd.read_csv(gt_6DoF_gnss_and_imu_csv, skiprows=13)
        selected_columns = [
            "% UNIX time",
            " x-enu(m)",
            " y-enu(m)",
            " z-enu(m)",
            " orientation.x",
            " orientation.y",
            " orientation.z",
            " orientation.w",
        ]
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        df = df[selected_columns].copy()
        df["% UNIX time"] = (df["% UNIX time"] * 1e9).astype("int64")
        rows = df.astype(object).values.tolist()
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(sequence_path / "calibration", ignore_errors=True)
            shutil.rmtree(sequence_path / "groundtruth", ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            for zip_file in sequence_path.rglob("*.zip"):
                zip_file.unlink(missing_ok=True)
            (sequence_path / "imu_raw.csv").unlink(missing_ok=True)

    def get_download_issues(self, _):
        if self.api_token == "not_set":
            return [
                _get_dataset_issue(
                    issue_id="api_token",
                    dataset_name=self.dataset_name,
                    website=self.dataset_homepage,
                    yaml_file=self.yaml_file,
                )
            ]
        return []

    def _load_extrinsics_matrix(self, path):
        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2:
            raise ValueError(f"Extrinsics file {path} does not contain header + data row")

        # Header: allow tabs/spaces/commas, remove leading '#'
        header = [h.lstrip("#") for h in re.split(r"[\t,]\s*|\s+", lines[0].lstrip("#")) if h]

        # Values: allow commas, tabs, or spaces
        values = [float(v) for v in re.split(r"[\t, ]+\s*|,\s*", lines[1]) if v]

        if len(header) != len(values):
            raise ValueError(
                f"Header/value mismatch in {path}: {len(header)} headers vs {len(values)} values\n"
                f"header={header}\nvalues={values}"
            )

        data_ext = dict(zip(header, values))

        t = np.array(
            [
                data_ext["position.x"],
                data_ext["position.y"],
                data_ext["position.z"],
            ],
            dtype=float,
        )

        q = np.array(
            [
                data_ext["orientation.x"],
                data_ext["orientation.y"],
                data_ext["orientation.z"],
                data_ext["orientation.w"],
            ],
            dtype=float,
        )

        T = np.eye(4, dtype=float)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t

        return T, data_ext

    def _get_file_url(self, sequence_name):
        base_url = "file?attachment=true&pid=b1584010878"
        if sequence_name == "A-0":
            ids = [223, 224, 417, 335, 491]
        if sequence_name == "A-1":
            ids = [18, 19, 417, 336, 492]
        if sequence_name == "B-0":
            ids = [62, 63, 425, 371, 483]
        if sequence_name == "C-0":
            ids = [54, 55, 418, 350, 480]
        if sequence_name == "D-0":
            ids = [115, 116, 420, 353, 475]
        if sequence_name == "E-0":
            ids = [145, 146, 421, 358, 472]

        return [f"{base_url}.{id}&access_token={self.api_token}" for id in ids]
