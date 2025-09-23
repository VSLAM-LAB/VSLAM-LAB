import os, yaml
import pandas as pd
import numpy as np
from pathlib import Path

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.dataset_utilities import undistort_rgb_rad_tan, undistort_depth_rad_tan
from path_constants import VSLAMLAB_BENCHMARK_WEIGHT

class RGBDTUM_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'rgbdtum'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('rgbd_dataset_freiburg', 'fr') for s in self.sequence_names]
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('validation', 'v') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('structure', 'st') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('texture', 'tx') for s in self.sequence_nicknames]
        self.sequence_nicknames = [s.replace('walking xyz', 'walk') for s in self.sequence_nicknames]

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Variables
        compressed_name_ext = sequence_name + '.tgz'
        decompressed_name = sequence_name

        if "freiburg1" in sequence_name:
            camera = "freiburg1"
            decompressed_name = decompressed_name.replace('validation', 'secret')
        if "freiburg2" in sequence_name:
            camera = "freiburg2"
            decompressed_name = decompressed_name.replace('validation', 'secret')
        if "freiburg3" in sequence_name:
            camera = "freiburg3"

        download_url = os.path.join(self.url_download_root, camera, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not os.path.exists(sequence_path):
            decompressFile(compressed_file, self.dataset_path)
            os.rename(decompressed_folder, sequence_path)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path_raw = os.path.join(sequence_path, 'rgb')
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        depth_path_raw = os.path.join(sequence_path, 'depth')
        depth_path = os.path.join(sequence_path, 'depth_0')
        if os.path.isdir(rgb_path_raw) and not os.path.isdir(rgb_path): 
            os.replace(rgb_path_raw, rgb_path)
        if os.path.isdir(depth_path_raw) and not os.path.isdir(depth_path): 
            os.replace(depth_path_raw, depth_path)

    def create_rgb_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        depth_txt = os.path.join(sequence_path, 'depth.txt')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        if os.path.exists(rgb_csv):
            return   

        rgb_df = pd.read_csv(rgb_txt, sep=' ', comment='#', header=None, names=['timestamp', 'rgb_filename'])
        depth_df = pd.read_csv(depth_txt, sep=' ', comment='#', header=None, names=['timestamp', 'depth_filename'])

        time_difference_threshold = 0.02 
        def find_closest_depth(rgb_ts):
            diff = abs(depth_df['timestamp'] - rgb_ts)
            min_diff_idx = diff.idxmin()
            if diff[min_diff_idx] <= time_difference_threshold:
                return depth_df.loc[min_diff_idx, 'depth_filename']
            return None

        rgb_df['matched_depth'] = rgb_df['timestamp'].apply(find_closest_depth)
        matched_pairs = rgb_df.dropna(subset=['matched_depth']).copy()
        matched_pairs['depth_timestamp'] = matched_pairs.apply(
            lambda row: depth_df.loc[depth_df['depth_filename'] == row['matched_depth'], 'timestamp'].values[0],
            axis=1
        )

        matched_pairs['timestamp'] = matched_pairs['timestamp'].apply(lambda x: f"{x:.6f}")
        matched_pairs['depth_timestamp'] = matched_pairs['depth_timestamp'].apply(lambda x: f"{x:.6f}")

        out = matched_pairs.rename(columns={
            'timestamp': 'ts_rgb0 (s)',
            'rgb_filename': 'path_rgb0',
            'depth_timestamp': 'ts_depth0 (s)',
            'matched_depth': 'path_depth0',
        })[['ts_rgb0 (s)', 'path_rgb0', 'ts_depth0 (s)', 'path_depth0']]

        out['path_rgb0'] = out['path_rgb0'].astype(str).str.replace(r'^rgb/', 'rgb_0/', regex=True)
        out['path_depth0'] = out['path_depth0'].astype(str).str.replace(r'^depth/', 'depth_0/', regex=True)

        out.to_csv(rgb_csv, index=False, header=True)

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        if "freiburg1" in sequence_name:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                517.306408, 516.469215, 318.643040, 255.313989, 0.262383, -0.953104, -0.005358, 0.002628, 1.163314)
        if "freiburg2" in sequence_name:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = (
                520.908620, 521.007327, 325.1414427, 249.701764, 0.231222, -0.784899, -0.003257, -0.000105, 0.917205)
        if "freiburg3" in sequence_name:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = 535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0, 0.0

        if "freiburg1" in sequence_name or "freiburg2" in sequence_name:
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion_coeffs = np.array([k1, k2, p1, p2, k3])
            fx, fy, cx, cy = undistort_rgb_rad_tan(rgb_csv, sequence_path, camera_matrix, distortion_coeffs)
            undistort_depth_rad_tan(rgb_csv, sequence_path, camera_matrix, distortion_coeffs)

        camera0 = {}
        camera0['model'] = 'Pinhole'
        camera0['fx'], camera0['fy'], camera0['cx'], camera0['cy'] = fx, fy, cx, cy

        rgbd = {}
        rgbd['depth0_factor'] = 5000.0
        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd = rgbd)

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')

        if not os.path.exists(groundtruth_txt):
            return

        with open(groundtruth_txt, 'r') as file:
            lines = file.readlines()

        new_lines = lines[3:]
        with open(groundtruth_txt, 'w') as file:
            file.writelines(new_lines)

        header = "ts,tx,ty,tz,qx,qy,qz,qw\n"
        new_lines.insert(0, header)
        with open(groundtruth_csv, 'w') as file:
            for line in new_lines:
                values = line.split()
                file.write(','.join(values) + '\n')

    def remove_unused_files(self, sequence_name):
        sequence_path = Path(self.dataset_path, sequence_name)
        for name in ("accelerometer.txt", "depth.txt", "groundtruth.txt", "rgb.txt"):
            Path(sequence_path, name).unlink(missing_ok=True)
