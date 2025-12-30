from __future__ import annotations

import csv
import yaml
import os, shutil 
import numpy as np
from pathlib import Path
from typing import Final, Any
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION, VSLAMLAB_BENCHMARK
from Datasets.DatasetVSLAMLab_issues import _get_dataset_issue

CAMERA_PARAMS: Final = [600.0, 600.0, 599.5, 339.5] # Camera intrinsics (fx, fy, cx, cy)


class REPLICA_dataset(DatasetVSLAMLab):
    """REPLICA dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "replica") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name_ext = 'Replica.zip'
        decompressed_name = self.dataset_name.upper()
        
        download_url = self.url_download_root

        # Constants
        compressed_file = VSLAMLAB_BENCHMARK / compressed_name_ext
        decompressed_folder = VSLAMLAB_BENCHMARK / decompressed_name

        # Download the compressed file
        if not compressed_file.exists():
            downloadFile(download_url, VSLAMLAB_BENCHMARK)

        if (not decompressed_folder.is_dir()) or (next(decompressed_folder.iterdir(), None) is None):
            decompressFile(compressed_file, VSLAMLAB_BENCHMARK)
            os.rename(VSLAMLAB_BENCHMARK / 'Replica', decompressed_folder)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        results_path = sequence_path / 'results'
        rgb_path = sequence_path / 'rgb_0'
        depth_path = sequence_path / 'depth_0'

        if not rgb_path.exists():
            results_path.rename(rgb_path)
            depth_path.mkdir(exist_ok=True)
            for filename in os.listdir(rgb_path):
                if 'depth' in filename:
                    old_file = rgb_path / filename
                    new_file = depth_path / filename.replace('depth', '')
                    shutil.move(old_file, new_file)

                if 'frame' in filename:
                    old_file = rgb_path / filename
                    new_file = rgb_path / filename.replace('frame', '')
                    old_file.rename(new_file)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        rgb_csv = sequence_path / 'rgb.csv'

        rgb_files = [f for f in os.listdir(rgb_path) if (rgb_path / f).is_file()]
        rgb_files.sort()

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_depth_0 (ns)', 'path_depth_0']) 
			
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                ts_ns = int(1e10 + ts * 1e9)
                depth_name = name + '.png'
                writer.writerow([ts_ns, f"rgb_0/{filename}", ts_ns, f"depth_0/{depth_name}"])

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
        rgb_csv = sequence_path / 'rgb.csv'
        traj_txt = sequence_path / 'traj.txt'
        groundtruth_csv = sequence_path /'groundtruth.csv'

        # Read RGB timestamps from CSV (skip header)
        rgb_timestamps_ns = []
        with open(rgb_csv, 'r', newline='') as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header if present
            for row in reader:
                if not row:
                    continue
                rgb_timestamps_ns.append(float(row[0]))

        # Write groundtruth.csv with header
        with open(traj_txt, 'r') as src, open(groundtruth_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])

            for idx, line in enumerate(src):
                if idx >= len(rgb_timestamps_ns):
                    break  # avoid index error if traj has extra lines
                vals = list(map(float, line.strip().split()))
                # traj.txt assumed row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]

                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                ts_ns = rgb_timestamps_ns[idx]

                writer.writerow([int(ts_ns), tx, ty, tz, qx, qy, qz, qw])

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        if BENCHMARK_RETENTION != Retention.FULL:
            (sequence_path / "traj.txt").unlink(missing_ok=True)
            (self.dataset_path / "cam_params.json").unlink(missing_ok=True)
            for ply_file in self.dataset_path.glob("*.ply"):
                ply_file.unlink(missing_ok=True)

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=12.4)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (VSLAMLAB_BENCHMARK / f"Replica.zip").unlink(missing_ok=True)
