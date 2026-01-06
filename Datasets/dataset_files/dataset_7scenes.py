from __future__ import annotations

import os
import csv
import glob
import yaml
import shutil
import numpy as np
from pathlib import Path
from typing import Final, Any
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION

SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
CAMERA_PARAMS: Final = [585.0, 585.0, 320.0, 240.0] # Camera intrinsics (fx, fy, cx, cy)


class SEVENSCENES_dataset(DatasetVSLAMLab):
    """7scenes dataset helper for VSLAM-LAB benchmark."""
    
    def __init__(self, benchmark_path: str | Path, dataset_name: str = "7scenes") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_seq-', ' ') for s in self.sequence_names]

        # Depth factor
        self.depth_factor = cfg["depth_factor"]
        
    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_group = _find_sequence_group(sequence_name)
        compressed_name = sequence_group
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = self.dataset_path / compressed_name_ext
        decompressed_folder = self.dataset_path / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)

        # Variables
        compressed_name = sequence_name.replace(sequence_group + '_', '')
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name

        # Constants
        compressed_file = self.dataset_path / sequence_group / compressed_name_ext
        sequence_path = self.dataset_path / sequence_name
        decompressed_folder = sequence_path

        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)
            os.rename(self.dataset_path / decompressed_name, sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        modes = ['color', 'depth']
        folder = {'color': 'rgb_0', 'depth': 'depth_0'}
        for mode in modes:
            folder_path = sequence_path / f'{folder[mode]}'
            if folder_path.exists():
                continue
            folder_path.mkdir(parents=True, exist_ok=True)
            image_files = glob.glob(str(sequence_path / f'*.{mode}.png'))
            for image_path in image_files:
                image_name = os.path.basename(image_path)
                image_name = image_name.replace("frame-", "")
                image_name = image_name.replace(f"{mode}.", "")
                shutil.copy(image_path, folder_path / image_name)  
            png_files = glob.glob(str(sequence_path / f'*.{mode}.png'))
            for png_file in png_files:
                os.remove(png_file)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return
        tmp = rgb_csv.with_suffix(".csv.tmp")

        modes = ['color', 'depth']
        folder = {'color': 'rgb_0', 'depth': 'depth_0'}
        png_files ={}
        for mode in modes:
            folder_path = sequence_path / f'{folder[mode]}'
            png_files[mode] = [file for file in os.listdir(folder_path) if file.endswith('.png')]
            png_files[mode].sort()

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])  
            for iPNG in range(len(png_files['color'])):
                ts_r0_ns = int(1e12 + float(iPNG / self.rgb_hz) * 1e9)
                path_r0 = f"rgb_0/{png_files['color'][iPNG]}"
                ts_d_ns = int(1e12 + float(iPNG / self.rgb_hz) * 1e9)
                path_d = f"depth_0/{png_files['depth'][iPNG]}"
                w.writerow([ts_r0_ns, path_r0, ts_d_ns, path_d])
        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:

        fx, fy, cx, cy = CAMERA_PARAMS
        rgbd0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb+depth", "depth_name": "depth_0",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "depth_factor": float(self.depth_factor),
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])
        
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / 'groundtruth.csv'
        pose_files = glob.glob(str(sequence_path / '*.pose.txt'))
        tmp = groundtruth_csv.with_suffix(".csv.tmp")
        pose_files.sort()
        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])
            for iRGB, gt0 in enumerate(pose_files, start=0):
                with open(gt0, 'r') as source_file:
                    T = []
                    for line in source_file:
                        row = [float(x) for x in line.split()]
                        T.append(row)
                    tx = T[0][3]
                    ty = T[1][3]
                    tz = T[2][3]
                    rotation_matrix = np.array([[T[0][0], T[0][1], T[0][2]],
                                                [T[1][0], T[1][1], T[1][2]],
                                                [T[2][0], T[2][1], T[2][2]]])
                    rotation = R.from_matrix(rotation_matrix)
                    quaternion = rotation.as_quat()
                    qx = quaternion[0]
                    qy = quaternion[1]
                    qz = quaternion[2]
                    qw = quaternion[3]
                    ts_d_ns = int(1e12 + float(iRGB / self.rgb_hz) * 1e9)
                    w.writerow([ts_d_ns, tx, ty, tz, qx, qy, qz, qw])
                os.remove(gt0)
        tmp.replace(groundtruth_csv)

def _find_sequence_group(sequence_name):
    for scene in SCENES:
         if scene in sequence_name:
            return scene