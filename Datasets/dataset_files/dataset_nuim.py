from __future__ import annotations

import os
import csv
import yaml
import numpy as np
from pathlib import Path
from typing import Final, Any

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION, VSLAMLAB_BENCHMARK

CAMERA_PARAMS: Final = [481.20, -480.00, 319.50, 239.50] # Camera intrinsics (fx, fy, cx, cy)


class NUIM_dataset(DatasetVSLAMLab):
    """NUIM dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "nuim") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_frei_png', '') for s in self.sequence_names]
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_nicknames]

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        # Variables
        compressed_name_ext = sequence_name + '.tar.gz'
        decompressed_name = sequence_name
        
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        depth_path = sequence_path / 'depth_0'
        rgb_path_original = sequence_path / 'rgb'
        depth_path_original = sequence_path / 'depth'

        if rgb_path_original.is_dir() and not rgb_path.is_dir():
            rgb_path_original.rename(rgb_path) 
        if depth_path_original.is_dir() and not depth_path.is_dir():    
            depth_path_original.rename(depth_path) 

        for png_file in os.listdir(rgb_path):
            if png_file.endswith(".png"):
                name, ext = os.path.splitext(png_file)
                new_name = f"{int(name):05}{ext}"
                old_file = rgb_path / png_file
                new_file = rgb_path / new_name
                old_file.rename(new_file)

        for png_file in os.listdir(depth_path):
            if png_file.endswith(".png"):
                name, ext = os.path.splitext(png_file)
                new_name = f"{int(name):05}{ext}"
                old_file = depth_path / png_file
                new_file = depth_path / new_name
                old_file.rename(new_file)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        rgb_csv = sequence_path / 'rgb.csv'

        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_depth_0 (ns)', 'path_depth_0'])  # header
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz 
                ts_ns = int(1e10 + ts * 1e9)
                writer.writerow([ts_ns, f"rgb_0/{filename}", ts_ns, f"depth_0/{filename}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:

        fx, fy, cx, cy = CAMERA_PARAMS
        rgbd0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb+depth", "depth_name": "depth_0",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "depth_factor": float(self.depth_factor),
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}        
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = self.dataset_path / sequence_name
        groundtruth_txt = sequence_path / 'groundtruth.txt'
        groundtruth_csv = sequence_path / 'groundtruth.csv'

        freiburg_txt = [file for file in os.listdir(sequence_path) if 'freiburg' in file.lower()]
        with open(sequence_path / freiburg_txt[0], 'r') as source_file:
            with open(groundtruth_txt, 'w') as destination_txt_file, \
                open(groundtruth_csv, 'w', newline='') as destination_csv_file:

                csv_writer = csv.writer(destination_csv_file)
                header = ["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"]
                csv_writer.writerow(header)
                for line in source_file:
                    values = line.strip().split()
                    ts = float(values[0]) / self.rgb_hz 
                    ts_ns = int(1e10 + ts * 1e9)
                    values[0] = str(ts_ns)
                    
                    destination_txt_file.write(" ".join(values) + "\n")
                    csv_writer.writerow(values)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        if BENCHMARK_RETENTION != Retention.FULL:
            for name in ("associations.txt", "groundtruth.txt", "traj0.gt.freiburg"):
                (sequence_path / name).unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (VSLAMLAB_BENCHMARK / f"{sequence_name}.tar.gz").unlink(missing_ok=True)
