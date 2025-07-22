import os, yaml, shutil
import pandas as pd
import numpy as np

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
            rgb_txt = os.path.join(sequence_path, 'rgb.txt')
            os.rename(rgb_txt, rgb_txt.replace('rgb.txt', 'rgb_original.txt'))

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        depth_txt = os.path.join(sequence_path, 'depth.txt')
        rgb_original_txt = os.path.join(sequence_path, 'rgb_original.txt')
        if os.path.exists(rgb_txt):
            return   

        rgb_df = pd.read_csv(rgb_original_txt, sep=' ', comment='#', header=None, names=['timestamp', 'rgb_filename'])
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

        matched_pairs[['timestamp', 'rgb_filename', 'depth_timestamp', 'matched_depth']].to_csv(
            rgb_txt, sep=' ', index=False, header=False
        )



    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
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
            fx, fy, cx, cy = undistort_rgb_rad_tan(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)
            if VSLAMLAB_BENCHMARK_WEIGHT != 'light':
                undistort_depth_rad_tan(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)

        #self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)
        self.write_calibration_rgbd_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name, depth_factor=5000)

    def create_groundtruth_txt(self, sequence_name):
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
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if VSLAMLAB_BENCHMARK_WEIGHT == 'light':
            #shutil.rmtree(os.path.join(sequence_path, 'depth'))
            os.remove(os.path.join(sequence_path, 'depth.txt'))
            os.remove(os.path.join(sequence_path, 'accelerometer.txt'))
            os.remove(os.path.join(sequence_path, 'rgb_original.txt'))
            os.remove(os.path.join(self.dataset_path, f'{sequence_name}.tgz'))
