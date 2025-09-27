import os
import yaml
import shutil
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

from pathlib import Path


from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.DatasetIssues import _get_dataset_issue

class KITTI_dataset(DatasetVSLAMLab):
    """KITTI dataset helper for VSLAMLab benchmark."""

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
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)
            downloadFile(self.url_download_root_gt, self.dataset_path)

        # Decompress the file
        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, self.dataset_path)
            decompressFile(os.path.join(self.dataset_path, 'data_odometry_poses.zip'), self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        rgb_path_0 = os.path.join(self.dataset_path, 'dataset', 'sequences', sequence_name, 'image_0')
        if not os.path.exists(rgb_path_0):
            return

        for png_file in os.listdir(rgb_path_0):
            if png_file.endswith(".png"):
                shutil.move(os.path.join(rgb_path_0, png_file), os.path.join(rgb_path, png_file))

        shutil.rmtree(rgb_path_0)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        times_txt = os.path.join(self.dataset_path, 'dataset', 'sequences', sequence_name, 'times.txt')

        # Read timestamps
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Collect and sort image filenames
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        # Write CSV with header
        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb0 (s)', 'path_rgb0']) 	
            for t, fname in zip(times, rgb_files):  # pairs safely to the shorter list
                writer.writerow([f"{t:.6f}", f"rgb_0/{fname}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:

        calibration_txt = os.path.join(self.dataset_path, 'dataset', 'sequences', sequence_name, 'calib.txt')
        if not os.path.exists(calibration_txt):
            return

        with open(calibration_txt, 'r') as file:
            calibration = [value for value in file.readline().split()]

        fx, fy, cx, cy = calibration[1], calibration[6], calibration[3], calibration[7]

        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        out_csv = os.path.join(sequence_path, 'groundtruth.csv')

        # Keep your original guard
        sequence_name_int = int(sequence_name)
        if sequence_name_int > 10:
            return

        # Read timestamps
        times_txt = os.path.join(self.dataset_path, 'dataset', 'sequences', sequence_name, 'times.txt')
        times = []
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))

        # Read trajectory and write CSV
        poses_txt = os.path.join(self.dataset_path, 'dataset', 'poses', sequence_name + '.txt')
        with open(poses_txt, 'r') as src, open(out_csv, 'w', newline='') as dst:
            writer = csv.writer(dst)
            writer.writerow(['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])  # header

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

                writer.writerow([f"{ts:.6f}", tx, ty, tz, qx, qy, qz, qw])

    # def remove_unused_files(self, sequence_name: str) -> None:
    #     sequence_folder = os.path.join(self.dataset_path, 'dataset', 'sequences', sequence_name)
    #     if os.path.exists(sequence_folder):
    #         shutil.rmtree(sequence_folder)

    #     sequences_folder = os.path.join(self.dataset_path, 'dataset', 'sequences')
    #     if os.path.exists(sequences_folder):
    #         if not os.listdir(sequences_folder):
    #             shutil.rmtree(sequences_folder)

    #     sequence_name_int = int(sequence_name)
    #     if sequence_name_int < 11:
    #         groundtruth_txt_0 = os.path.join(self.dataset_path, 'dataset', 'poses', sequence_name + '.txt')
    #         if os.path.exists(groundtruth_txt_0):
    #             os.remove(groundtruth_txt_0)

    #     poses_folder = os.path.join(self.dataset_path, 'dataset', 'poses')
    #     if os.path.exists(poses_folder):
    #         if not os.listdir(poses_folder):
    #             shutil.rmtree(poses_folder)

    #     dataset_folder = os.path.join(self.dataset_path, 'dataset')
    #     if os.path.exists(dataset_folder):
    #         if not os.listdir(dataset_folder):
    #             shutil.rmtree(dataset_folder)

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=23.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)