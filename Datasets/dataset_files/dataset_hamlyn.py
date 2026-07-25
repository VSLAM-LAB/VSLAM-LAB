import csv
import os
import shutil
from typing import Any

import yaml
import numpy as np
from zipfile import ZipFile
from huggingface_hub import login
from huggingface_hub import hf_hub_download

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import HUGGINGFACE_TOKEN, BENCHMARK_RETENTION, Retention


class HamlynDataset(DatasetVSLAMLAB):
    """HAMLYN dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hamlyn") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.repo_id = cfg['repo_id']

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_0_path = sequence_path / 'rgb_0'
        if rgb_0_path.exists():
            return
        
        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        repo_id = self.repo_id

        if HUGGINGFACE_TOKEN is not None:
            login(token=HUGGINGFACE_TOKEN) 
        
        # Download the compressed file
        if not (self.dataset_path / compressed_name_ext).exists():
            file_path = hf_hub_download(repo_id=repo_id, filename=compressed_name_ext, repo_type='dataset')
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)

        # Download instrinsics file
        intrinsics_file = f"intrinsics_{sequence_name}.txt"
        if not (sequence_path / intrinsics_file).exists():
            file_path = hf_hub_download(repo_id=repo_id, filename=intrinsics_file, repo_type='dataset')
            intrinsics_txt = sequence_path / intrinsics_file
            with open(file_path, 'rb') as f_src, open(intrinsics_txt, 'wb') as f_dest:
                f_dest.write(f_src.read())

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for cam, id in zip(['rgb_0', 'rgb_1', 'depth_0', 'depth_1'], ['image01', 'image02', 'depth01', 'depth02']):
            rgb_path = sequence_path / cam
            image_path = sequence_path / id
            if image_path.exists() and not rgb_path.exists():
                image_path.rename(rgb_path)


    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_path = sequence_path / 'rgb'
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb_files = {}
        for cam in ['rgb_0', 'rgb_1', 'depth_0', 'depth_1']:
            rgb_path = sequence_path / cam
            rgb_files[cam] = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
            rgb_files[cam].sort()

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 
                             'ts_depth_0 (ns)', 'path_depth_0',
                             'ts_rgb_1 (ns)', 'path_rgb_1',
                             'ts_depth_1 (ns)', 'path_depth_1'])
            for f0, f1, f2, f3 in zip(rgb_files['rgb_0'], rgb_files['rgb_1'], rgb_files['depth_0'], rgb_files['depth_1']):
                name = {}
                ts_ns = {}
                for f in [f0, f1, f2, f3]:
                    name[f] = os.path.splitext(f)[0]
                    ts = float(name[f]) / self.rgb_hz
                    ts_ns[f] = int(1e10 + ts * 1e9)

                writer.writerow([ts_ns[f0], f"rgb_0/{name[f0]}.jpg", 
                                 ts_ns[f2], f"depth_0/{name[f2]}.png", 
                                 ts_ns[f1], f"rgb_1/{name[f1]}.jpg", 
                                 ts_ns[f3], f"depth_1/{name[f3]}.png"])

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        calibration_file = sequence_path / f"intrinsics_{sequence_name}.txt"

        with open(calibration_file, 'r') as file:
            lines = file.readlines()
        fx, _, cx, _ = map(float, lines[0].split())
        _, fy, cy, _ = map(float, lines[1].split())

        rgbd0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb+depth", "depth_name": "depth_0",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "depth_factor": float(self.depth_factor),
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}
        rgbd1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "rgb+depth", "depth_name": "depth_1",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "depth_factor": float(self.depth_factor),
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}
        
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0, rgbd1])
    
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with  open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])
        tmp.replace(groundtruth_csv)
            
    def remove_unused_files(self, sequence_name: str) -> None:  
        sequence_path = self.sequence_path(sequence_name)
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(sequence_path / f"intrinsics_{sequence_name}.txt", ignore_errors=True)
