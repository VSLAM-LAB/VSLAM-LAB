import os
import cv2
import yaml
import tqdm
import shutil
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from Datasets.DatasetVSLAMLab_utilities import undistort_fisheye

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
class HILTI2022_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('hilti2022', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.split('_', 1)[0] for s in self.sequence_names]

        # Pick camera from 0-4
        self.camera = data['camera']

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        os.makedirs(sequence_path, exist_ok=True)

        # Download rosbag
        compressed_name_ext = sequence_name + '.bag'
        if not os.path.exists(os.path.join(sequence_path, compressed_name_ext)):
            download_url = os.path.join(self.url_download_root, compressed_name_ext)
            downloadFile(download_url, sequence_path)

        # Download calibration files
        decompressed_folder = os.path.join(self.dataset_path, 'calibration_files')
        if not os.path.exists(decompressed_folder):
            compressed_name_ext = "2022322_calibration_files.zip"
            cal_url = f"{self.url_download_root}/{compressed_name_ext}"
            compressed_file = os.path.join(self.dataset_path, compressed_name_ext)

            downloadFile(cal_url, self.dataset_path)
            decompressFile(compressed_file, decompressed_folder)

        # Download gt
        gt_name = self.get_gt_name(sequence_name)
        gt_txt = os.path.join(sequence_path, gt_name)
        if not os.path.exists(gt_txt):
            gt_url = f"{self.url_download_root}/{gt_name}"
            downloadFile(gt_url, sequence_path)


    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        if os.path.exists(rgb_path):
            return

        os.makedirs(rgb_path, exist_ok=True)
        rosbag_path = f"{os.path.join(sequence_path, sequence_name)}.bag"
        image_topic = f"/alphasense/{self.camera}/image_raw"

        print(f"{SCRIPT_LABEL}Extracting images from {rosbag_path}:{image_topic})")
        command = f"pixi run -e ros-env extract-rosbag-frames {rosbag_path} {rgb_path} {image_topic}"
        subprocess.run(command, shell=True)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = float(name) / 10e8
                file.write(f"{ts:.5f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        if os.path.exists(calibration_yaml):
            return

        print(f"{SCRIPT_LABEL}Undistorting images with fisheye model: {rgb_path}")
        fx, fy, cx, cy, k1, k2, k3, k4 = self.get_intrinsics_from_yaml()
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, k3, k4])
        fx, fy, cx, cy = undistort_fisheye(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        gt_name = self.get_gt_name(sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        groundtruth_txt_0 = os.path.join(sequence_path, gt_name)

        T_cam_imu = self.get_transformation_from_yaml()
        T_imu_cam = np.linalg.inv(T_cam_imu)

        with open(groundtruth_txt_0, 'r') as source_file, open(groundtruth_txt, 'w') as destination_file:
            for idx, line in enumerate(source_file, start=0):
                line = line.strip()
                values = line.split(' ')

                # Extract values
                ts = values[0]
                t_w_imu = np.array([float(values[1]), float(values[2]), float(values[3])])
                qx, qy, qz, qw = map(float, values[4:8])
                R_w_imu = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_w_imu = np.eye(4)
                T_w_imu[:3, :3] = R_w_imu
                T_w_imu[:3, 3] = t_w_imu

                # Compute transformation
                T_w_cam = np.dot(T_w_imu, T_imu_cam)

                R_w_cam = T_w_cam[:3, :3]
                t_w_cam = T_w_cam[:3, 3]

                q_cam = R.from_matrix(R_w_cam).as_quat()  # [qx, qy, qz, qw]
                tx, ty, tz = t_w_cam

                # Write to the output file
                line2 = f"{ts} {tx} {ty} {tz} {q_cam[0]} {q_cam[1]} {q_cam[2]} {q_cam[3]}\n"
                destination_file.write(line2)

    def remove_unused_files(self, sequence_name):
        return

    def get_intrinsics_from_yaml(self):
        if self.camera == "cam0" or self.camera == "cam1":
            calibration_file = os.path.join(self.dataset_path, 'calibration_files',
                                            'calib_3_cam0-1-camchain-imucam.yaml')
        else:
            calibration_file = os.path.join(self.dataset_path, 'calibration_files',
                                            f'calib_3_{self.camera}-camchain-imucam.yaml')
        camera = 'cam0'
        if self.camera == "cam1":
            camera = 'cam1'

        with open(calibration_file, 'r') as file:
            data = yaml.safe_load(file)

        fx, fy, cx, cy = data[camera]['intrinsics']
        k1, k2, k3, k4 = data[camera]['distortion_coeffs']

        return fx, fy, cx, cy, k1, k2, k3, k4

    def get_transformation_from_yaml(self):
        if self.camera == "cam0" or self.camera == "cam1":
            calibration_file = os.path.join(self.dataset_path, 'calibration_files',
                                            'calib_3_cam0-1-camchain-imucam.yaml')
        else:
            calibration_file = os.path.join(self.dataset_path, 'calibration_files',
                                            f'calib_3_{self.camera}-camchain-imucam.yaml')
        camera = 'cam0'
        if self.camera == "cam1":
            camera = 'cam1'

        with open(calibration_file, 'r') as file:
            data = yaml.safe_load(file)

        T_cam_imu = data[camera]['T_cam_imu']

        return np.array(T_cam_imu)

    def get_gt_name(self, sequence_name):
        gt_name = f"{sequence_name}_imu.txt"
        if ('exp_4' in sequence_name) or ('exp_5' in sequence_name):
            gt_name = gt_name.replace("exp", "exp_", 1)
        return gt_name