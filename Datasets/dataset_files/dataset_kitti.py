
from __future__ import annotations

import os
import csv
import yaml
import shutil
import numpy as np
from typing import  Any
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION, VSLAMLAB_BENCHMARK
from Datasets.DatasetVSLAMLab_issues import _get_dataset_issue


class KITTI_dataset(DatasetVSLAMLab):
    """KITTI dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "kitti") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        self.url_download_root_gt: str = cfg["url_download_root_gt"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name = 'data_odometry_gray'
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = 'dataset'
        download_url = self.url_download_root

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)
            downloadFile(self.url_download_root_gt, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)
            decompressFile(self.dataset_path / 'data_odometry_poses.zip', self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for rgb_i, image in zip(['rgb_0', 'rgb_1'], ['image_0', 'image_1']):    
            rgb_path = sequence_path / rgb_i
            if not rgb_path.exists():
                os.makedirs(rgb_path)

            rgb_path_raw = self.dataset_path / 'dataset' / 'sequences' / sequence_name / image
            if not rgb_path_raw.exists():
                return

            for png_file in os.listdir(rgb_path_raw):
                if png_file.endswith(".png"):
                    shutil.move(rgb_path_raw / png_file, rgb_path / png_file)

            shutil.rmtree(rgb_path_raw)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / 'rgb.csv'

        times_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'times.txt'

        # Read timestamps
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Collect and sort image filenames
        rgb_path = sequence_path / 'rgb_0'
        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        # Write CSV with header
        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1']) 	
            for t, fname in zip(times, rgb_files):  # pairs safely to the shorter list
                t_ns = int(float(t) * 1e9)
                writer.writerow([t_ns, f"rgb_0/{fname}", t_ns, f"rgb_1/{fname}"])
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'calib.txt'

        with open(calibration_txt, 'r') as file:
            calibration_0 = [value for value in file.readline().split()]
            fx_0, fy_0, cx_0, cy_0 = calibration_0[1], calibration_0[6], calibration_0[3], calibration_0[7]
            calibration_1 = [value for value in file.readline().split()]
            fx_1, fy_1, cx_1, cy_1 = calibration_1[1], calibration_1[6], calibration_1[3], calibration_1[7]

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx_0, fy_0], "principal_point": [cx_0, cy_0],
                "fps": self.rgb_hz,
                "T_BS": np.eye(4)}
        
        T_BS_1 = np.eye(4)
        T_BS_1[0, 3] = -float(calibration_1[4]) / float(fx_0)
        rgb1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx_1, fy_1], "principal_point": [cx_1, cy_1],
                "fps": self.rgb_hz,
                "T_BS": T_BS_1}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1])
    
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        out_csv = sequence_path / 'groundtruth.csv'

        # Keep your original guard
        sequence_name_int = int(sequence_name)
        if sequence_name_int > 10:
            return

        # Read timestamps
        times_txt = self.dataset_path / 'dataset' / 'sequences' / sequence_name / 'times.txt'
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Read trajectory and write CSV
        poses_txt = self.dataset_path / 'dataset' / 'poses' / (sequence_name + '.txt')
        with open(poses_txt, 'r') as src, open(out_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'])

            for idx, line in enumerate(src):
                if idx >= len(times):
                    break  # avoid index error if poses has extra lines
                vals = list(map(float, line.strip().split()))
                # row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]
                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                ts = times[idx]
                ts_ns = int(float(ts)*1e9)
                writer.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=23.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)

        if BENCHMARK_RETENTION != Retention.FULL:
            (VSLAMLAB_BENCHMARK / f"dataset").unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (VSLAMLAB_BENCHMARK / f"data_odometry_gray.zip").unlink(missing_ok=True)
            (VSLAMLAB_BENCHMARK / f"data_odometry_poses.zip").unlink(missing_ok=True)