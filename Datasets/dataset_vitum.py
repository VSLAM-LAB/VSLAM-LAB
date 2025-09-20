import os
import csv
import yaml
import shutil
import numpy as np
import subprocess

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from Datasets.dataset_utilities import undistort_fisheye

from utilities import replace_string_in_files
from path_constants import VSLAM_LAB_DIR

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics

from utilities import ws

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
class VITUM_dataset(DatasetVSLAMLab):

    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('vitum', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = []
        for sequence_name in self.sequence_names:
            sequence_nickname = sequence_name.replace('sequence_', 'seq ')
            self.sequence_nicknames.append(sequence_nickname)

    def download_sequence_data(self, sequence_name):
        # Variables
        sequence_filename = 'dataset-' + sequence_name + '_512_16'
        compressed_name = sequence_filename + '.tar' 
        download_url = os.path.join(self.url_download_root, compressed_name)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name)
        decompressed_folder = os.path.join(self.dataset_path, sequence_filename)

        # Download the sequence data
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if os.path.exists(decompressed_folder):
            shutil.rmtree(decompressed_folder)
        decompressFile(compressed_file, self.dataset_path)
        
        # Delete the compressed file
        if os.path.exists(compressed_file):
           os.remove(compressed_file)
        

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')
        images_path = os.path.join(source_path, 'mav0', 'cam0', 'data')
        rgb_path = os.path.join(sequence_path, 'rgb')

        os.makedirs(rgb_path, exist_ok=True)
        #copy images to rgb folder
        for filename in os.listdir(images_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                source_file = os.path.join(images_path, filename)
                destination_file = os.path.join(rgb_path, filename)
                shutil.copy(source_file, destination_file) 

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        # Create filename -> timestamp mapping
        filename_to_timestamp = {}
        times_txt = os.path.join(source_path, 'dso', 'cam0', 'times.txt')
        with open(times_txt, 'r') as file:
            for line in file:
                if line.startswith('#'):  # Skip header
                    continue
                columns = line.split()
                if len(columns) >= 2:
                    filename = columns[0]  # e.g., "1520621175986840704"
                    timestamp = columns[1]  # e.g., "1520621175.986840704"
                    filename_to_timestamp[filename] = float(timestamp)

        # Write rgb.txt using the mapping
        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
        with open(rgb_txt, 'w') as file:
            for filename in rgb_files:
                base_name = filename.replace('.png', '')  # Remove extension
                if base_name in filename_to_timestamp:
                    timestamp = filename_to_timestamp[base_name]
                    file.write(f'{timestamp:.6f} rgb/{filename}\n')

    def create_imu_csv(self, sequence_name):        
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')

        # Find the IMU CSV file
        imu_csv_path = os.path.join(source_path, 'mav0', 'imu0', 'data.csv')
    
        # Destination path for the renamed file
        imu_destination = os.path.join(sequence_path, 'imu.csv')
    
        # Copy and rename the file
        if os.path.exists(imu_csv_path):
            shutil.copy(imu_csv_path, imu_destination)
        else:
            print(f"Warning: IMU data file not found at {imu_csv_path}")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')
        calibration_file_cam0 = os.path.join(source_path, 'dso', 'camchain.yaml')
        calibration_file_imu0 = os.path.join(source_path, 'dso', 'imu_config.yaml')

        # Load camera calibration from .yaml file
        with open(calibration_file_cam0, 'r') as cam_file:
            cam_data = yaml.safe_load(cam_file)

        # Load IMU calibration from .yaml file
        with open(calibration_file_imu0, 'r') as imu_file:
            imu_data = yaml.safe_load(imu_file)

        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

        if os.path.exists(calibration_yaml):
            return
        
        cam_data = cam_data['cam0']
        intrinsics = cam_data['intrinsics']
        distortion = cam_data['distortion_coeffs']

        gyro_noise = imu_data['gyroscope_noise_density']
        gyro_bias = imu_data['gyroscope_random_walk']
        accel_noise = imu_data['accelerometer_noise_density']
        accel_bias = imu_data['accelerometer_random_walk']

        print(f"{SCRIPT_LABEL}Undistorting images with fisheye model: {rgb_path}")
        camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
        distortion_coeffs = np.array([distortion[0], distortion[1], distortion[2], distortion[3]]) #fisheye model so k1, k2, k3, k4
        fx, fy, cx, cy = undistort_fisheye(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)
        camera_model = 'PINHOLE' # manually specifcy pinhole model after undistortion
        k1 = 0.0
        k2 = 0.0 
        p1 = 0.0 
        p2 = 0.0 

        camera0 = {'model': camera_model,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2,
                }

        imu = {
                'transform': cam_data['T_cam_imu'],  # 4x4 transformation matrix from camera to IMU
                'gyro_noise': gyro_noise,
                'gyro_bias': gyro_bias,
                'accel_noise': accel_noise,
                'accel_bias': accel_bias,
                'frequency': imu_data['update_rate'],
            }
        self.write_calibration_yaml(sequence_name, camera0=camera0, imu=imu)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16', 'dso', 'gt_imu.csv')

        with open(source_path) as source_file:
            with open(groundtruth_txt, 'w') as destination_file:
                csv_reader = csv.reader(source_file)
                header = next(csv_reader, None) # Skip header row if it exists

                for row in csv_reader:
                    line_to_write = " ".join(row) # Join row elements for NaN check and writing
                    if 'NaN' not in line_to_write:
                        destination_file.write(line_to_write + '\n')

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        shutil.rmtree((os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')))