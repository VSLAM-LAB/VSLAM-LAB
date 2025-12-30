from __future__ import annotations

import csv
import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION


class S3LI_dataset(DatasetVSLAMLab):
    """DLR S3LI Etna & Vulcano dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "s3li") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_sequences = cfg["url_download_sequences"]

        # Sequence nicknames
        self.sequence_nicknames = cfg["sequence_nicknames"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        url = self.url_download_sequences[sequence_name]
        zip_path_0 = self.dataset_path / f"{url.rsplit('/', 1)[-1]}"
        zip_path = self.dataset_path / f"{sequence_name}.zip"

        if not zip_path.exists() and not zip_path_0.exists():
            downloadFile(url, str(self.dataset_path))

        if not zip_path.exists() and zip_path_0.exists():
            zip_path_0.rename(zip_path)

        if zip_path.exists() and not sequence_path.exists():
            decompressFile(str(zip_path), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        input_path = sequence_path / "rgb_raw.csv"
        output_path = sequence_path / "rgb.csv"

        if output_path.exists() and not input_path.exists():
            output_path.rename(input_path)

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"]

        with open(input_path, "r", newline="", encoding="utf-8") as fin, \
             open(output_path, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)
            next(reader, None)

            for row in reader:
                ts_ns_0 = int(float(row[0]) * 1e9)
                row[0] = str(ts_ns_0) 
                ts_ns_1 = int(float(row[2]) * 1e9)
                row[2] = str(ts_ns_1) 
                writer.writerow(row)

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        input_path = sequence_path / "imu.csv"
        output_path = sequence_path / "imu_0.csv"

        header = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]

        with open(input_path, "r", newline="", encoding="utf-8") as fin, \
             open(output_path, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)
            next(reader, None)

            for row in reader:
                ts_ns = int(float(row[0]) * 1e9)
                row[0] = str(ts_ns) 
                writer.writerow(row)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_yaml = self.dataset_path / sequence_name / "calibration.yaml"
        fs = cv2.FileStorage(str(calibration_yaml), cv2.FILE_STORAGE_READ)
      
        fx = fs.getNode("Camera0.fx").real()
        fy = fs.getNode("Camera0.fy").real()
        cx = fs.getNode("Camera0.cx").real()
        cy = fs.getNode("Camera0.cy").real()
        fps = fs.getNode("Camera0.fps").real()

        baseline = fs.getNode("Stereo.bf").real()
        T_BS_right = np.array(fs.getNode("IMU.T_b_c1").mat().tolist()).reshape((4, 4))
        T_Right_Left = np.eye(4)
        T_Right_Left[0, 3] = -baseline  # Tx = -B (Move LEFT)
        T_BS_left = T_BS_right @ T_Right_Left

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "fps": fps,
                "T_BS": T_BS_left}
        rgb1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "rgb",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "fps": fps,
                "T_BS": T_BS_right}
        
        imu: dict[str, Any] = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c":  fs.getNode("IMU.NoiseGyro").real(), "sigma_a_c": fs.getNode("IMU.NoiseAcc").real(),
            "sigma_bg":  0.0, "sigma_ba":  0.0,
            "sigma_gw_c":  fs.getNode("IMU.GyroWalk").real(), "sigma_aw_c": fs.getNode("IMU.AccWalk").real(),
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ 0.0, 0.0, 0.0 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": fs.getNode("IMU.Frequency").real(),
            "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        out = sequence_path / "groundtruth.csv"
        tmp = out.with_suffix(".csv.tmp")

        header = ["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"]

        with open(out, "r", newline="", encoding="utf-8") as fin, \
             open(tmp, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)
            next(reader, None)

            for row in reader:
                ts_ns = int(float(row[0]) * 1e9)
                row[0] = str(ts_ns) 
                writer.writerow(row)

        tmp.replace(out)

    def remove_unused_files(self, sequence_name: str) -> None:
        pass
