import os, yaml, shutil 
import numpy as np
import csv
from typing import Final

from scipy.spatial.transform import Rotation as R
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import VSLAMLAB_BENCHMARK
from pathlib import Path
from Datasets.DatasetIssues import _get_dataset_issue

DEFAULT_DEPTH_FACTOR: Final = 6553.5

class REPLICA_dataset(DatasetVSLAMLab):
    """REPLICA dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "replica") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name_ext = 'Replica.zip'
        decompressed_name = self.dataset_name.upper()
        
        download_url = self.url_download_root

        # Constants
        compressed_file = os.path.join(VSLAMLAB_BENCHMARK, compressed_name_ext)
        decompressed_folder = os.path.join(VSLAMLAB_BENCHMARK, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, VSLAMLAB_BENCHMARK)

        if (not os.path.isdir(decompressed_folder)) or (next(os.scandir(decompressed_folder), None) is None):
            decompressFile(compressed_file, VSLAMLAB_BENCHMARK)
            os.rename(os.path.join(VSLAMLAB_BENCHMARK, 'Replica'), decompressed_folder)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        results_path = os.path.join(sequence_path, 'results')
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        depth_path = os.path.join(sequence_path, 'depth_0')

        if not os.path.exists(rgb_path):
            os.rename(results_path, rgb_path)
            os.makedirs(depth_path, exist_ok=True)
            for filename in os.listdir(rgb_path):
                if 'depth' in filename:
                    old_file = os.path.join(rgb_path, filename)
                    new_file = os.path.join(depth_path, filename.replace('depth', ''))
                    shutil.move(old_file, new_file)

                if 'frame' in filename:
                    old_file = os.path.join(rgb_path, filename)
                    new_file = os.path.join(rgb_path, filename.replace('frame', ''))
                    os.rename(old_file, new_file)


    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb0 (s)', 'path_rgb0', 'ts_depth0 (s)', 'path_depth0']) 
			
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                depth_name = name + '.png'
                writer.writerow([f"{ts:.5f}", f"rgb_0/{filename}", f"{ts:.5f}", f"depth_0/{depth_name}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:

        fx, fy, cx, cy = 600.0, 600.0, 599.5, 339.5
        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        rgbd = {"depth0_factor": float(DEFAULT_DEPTH_FACTOR)}

        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd=rgbd)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        traj_txt = os.path.join(sequence_path, 'traj.txt')
        out_csv = os.path.join(sequence_path, 'groundtruth.csv')

        # Read RGB timestamps from CSV (skip header)
        rgb_timestamps = []
        with open(rgb_csv, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header if present
            for row in reader:
                if not row:
                    continue
                rgb_timestamps.append(float(row[0]))

        # Write groundtruth.csv with header
        with open(traj_txt, 'r') as src, open(out_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

            for idx, line in enumerate(src):
                if idx >= len(rgb_timestamps):
                    break  # avoid index error if traj has extra lines
                vals = list(map(float, line.strip().split()))
                # traj.txt assumed row-major 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
                Rm = np.array([[vals[0], vals[1], vals[2]],
                            [vals[4], vals[5], vals[6]],
                            [vals[8], vals[9], vals[10]]], dtype=float)
                tx, ty, tz = vals[3], vals[7], vals[11]

                qx, qy, qz, qw = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
                ts = rgb_timestamps[idx]

                writer.writerow([f"{ts:.5f}", tx, ty, tz, qx, qy, qz, qw])

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=12.4)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)
