import csv
import os
import shutil
from typing import Final, Any

import numpy as np
import yaml

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile

CAMERA_PARAMS: Final = [405.6384738851233, 405.588335378204, 189.9054317917407, 139.9149578253755] # Camera intrinsics (fx, fy, cx, cy)


class CavesDataset(DatasetVSLAMLAB):
    """CAVES dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "caves") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root = cfg['url_download_root']
        self.url_download_timestamps = cfg['url_download_timestamps']
        self.url_download_root_gt = cfg['url_download_root_gt']

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:
        # Variables
        compressed_name = "undistorted_frames"
        compressed_name_ext = compressed_name + '.zip'
        download_url = self.url_download_root
        timestamps_url = self.url_download_timestamps
        groundtruth_url = self.url_download_root_gt
        sequence_path = self.sequence_path(sequence_name)
        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        timestamp_file = self.dataset_path / 'undistorted_frames_timestamps.txt'    
        compressed_dataset_file = self.dataset_path / 'full_dataset.zip'

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)
            os.rename(self.dataset_path / 'undistorted_frames.zip?download=1', compressed_file)

        # Download timestamps file
        if not timestamp_file.exists():
            downloadFile(timestamps_url, self.dataset_path)
            os.rename(self.dataset_path / 'undistorted_frames_timestamps.txt?download=1', timestamp_file)
        
        # Download full dataset to get the odometry file
        if not compressed_dataset_file.exists():
            downloadFile(groundtruth_url, self.dataset_path)
            os.rename(self.dataset_path / 'full_dataset.zip?download=1', self.dataset_path / 'full_dataset.zip')

        rgb_path = self.rgb_path(sequence_name)
        if not sequence_path.exists():
            decompressFile(compressed_file, sequence_path)
            os.rename(sequence_path / 'undistorted_frames', rgb_path)
        
        if not (sequence_path / 'full_dataset').exists():
            decompressFile(self.dataset_path / 'full_dataset.zip', self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        timestamps_txt = self.dataset_path / 'undistorted_frames_timestamps.txt'

        with open(timestamps_txt, "r") as ts_file, open(rgb_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for line in ts_file:
                name, ts = line.strip().split("\t")
                ts_ns = int(float(ts) * 1e9)
                writer.writerow([ts_ns, f"rgb_0/{name}"])
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = CAMERA_PARAMS

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}    
            
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])
        
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        groundtruth_txt_0 = self.dataset_path / 'full_dataset' / 'odometry.txt'

        columns = [
            '%time',
            'field.pose.pose.position.x',
            'field.pose.pose.position.y',
            'field.pose.pose.position.z',
            'field.pose.pose.orientation.x',
            'field.pose.pose.orientation.y',
            'field.pose.pose.orientation.z',
            'field.pose.pose.orientation.w'
        ]

        with open(groundtruth_txt_0, 'r') as infile, open(groundtruth_csv, "w") as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]) 
            for row in reader:
                timestamp = int(row["%time"]) 
                pose = [row[col] for col in columns[1:]]
                writer.writerow([timestamp, *pose])

    def remove_unused_files(self, sequence_name: str) -> None:  
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self.dataset_path / "full_dataset", ignore_errors=True)
            
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(self.dataset_path / "undistorted_frames_timestamps.txt", ignore_errors=True)
            for zip_file in self.dataset_path.rglob("*.zip"):
                zip_file.unlink(missing_ok=True)
           
