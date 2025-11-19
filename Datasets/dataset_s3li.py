from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
from contextlib import suppress
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import csv
import yaml
import cv2
import os

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile


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

        out = sequence_path / "rgb.csv"
        tmp = out.with_suffix(".csv.tmp")

        header = ["ts_rgb0 (s)", "path_rgb0", "ts_rgb1 (s)", "path_rgb1"]

        with open(out, "r", newline="", encoding="utf-8") as fin, \
             open(tmp, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)

            for row in reader:
                writer.writerow(row)

        tmp.replace(out)

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        out = sequence_path / "imu.csv"
        tmp = out.with_suffix(".csv.tmp")

        header = ["timestamp [s]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]

        with open(out, "r", newline="", encoding="utf-8") as fin, \
             open(tmp, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)
            next(reader, None)

            for row in reader:
                writer.writerow(row)

        tmp.replace(out)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_yaml = self.dataset_path / sequence_name / "calibration.yaml"
        fs = cv2.FileStorage(str(calibration_yaml), cv2.FILE_STORAGE_READ)
      
        camera0 = {
            "model": "Pinhole", "fx":fs.getNode("Camera0.fx").real(), "fy":fs.getNode("Camera0.fy").real(), 
            "cx": fs.getNode("Camera0.cx").real(), "cy":fs.getNode("Camera0.cy").real()
        }
        camera1 = {
            "model": "Pinhole", "fx":fs.getNode("Camera0.fx").real(), "fy":fs.getNode("Camera0.fy").real(), 
            "cx": fs.getNode("Camera0.cx").real(), "cy":fs.getNode("Camera0.cy").real()
        }

        stereo = {"bf": fs.getNode("Stereo.bf").real()}
        
        imu_out = {
            "transform": fs.getNode("IMU.T_b_c1").mat().tolist() ,
            "gyro_noise": fs.getNode("IMU.NoiseGyro").real(),
            "gyro_bias": fs.getNode("IMU.GyroWalk").real(),
            "accel_noise": fs.getNode("IMU.NoiseAcc").real(),
            "accel_bias": fs.getNode("IMU.AccWalk").real(),
            "frequency": fs.getNode("IMU.Frequency").real(),
        }

        self.write_calibration_yaml(sequence_name, camera0=camera0, camera1=camera1, stereo=stereo, imu=imu_out)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        out = sequence_path / "groundtruth.csv"
        tmp = out.with_suffix(".csv.tmp")

        header = ["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

        with open(out, "r", newline="", encoding="utf-8") as fin, \
             open(tmp, "w", newline="", encoding="utf-8") as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            writer.writerow(header)
            next(reader, None)

            for row in reader:
                writer.writerow(row)

        tmp.replace(out)

    def remove_unused_files(self, sequence_name: str) -> None:
        pass
