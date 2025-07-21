import os
import re
import yaml
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

from Datasets.dataset_utilities import resize_rgb_images



class UT_CODA_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):

        # Initialize the dataset
        super().__init__('ut_coda', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [f"seq{s}" for s in self.sequence_names]


        # Get resolution size
        self.resolution_size = data['resolution_size']
        
    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Variables
        compressed_name_ext = sequence_name + '.zip'
        decompressed_name = sequence_name
        
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)
        
        # Decompress the file
        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, sequence_path)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        rgb_path_0 = os.path.join(sequence_path, '2d_rect', 'cam0', sequence_name)
        if not os.path.exists(rgb_path_0):
            return

        for jpg_file in os.listdir(rgb_path_0):
            if jpg_file.endswith(".jpg"):
                shutil.move(os.path.join(rgb_path_0, jpg_file), os.path.join(rgb_path, jpg_file))

        shutil.rmtree(rgb_path_0)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        times_txt = os.path.join(sequence_path, 'timestamps', sequence_name + '.txt')
        times = []
        with open(times_txt, 'r') as file:
            lines = file.readlines()
            for line in lines:
                time = line.strip()
                times.append(float(time))

        def extract_frame_id(filename):
            match = re.search(r'2d_rect_cam0_\d+_(\d+)\.jpg', filename)
            return int(match.group(1)) if match else float('inf')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort(key=extract_frame_id)
        with open(rgb_txt, 'w') as file:
            for idx, time in enumerate(times, start=0):
                file.write(f"{time} rgb/{rgb_files[idx]}\n")
    
    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_file_yaml = os.path.join(sequence_path, 'calibrations', sequence_name, 'calib_cam0_intrinsics.yaml')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        # Load calibration from .yaml file
        with open(calibration_file_yaml, 'r') as file:
            data = yaml.safe_load(file)

        intrinsics = data['projection_matrix']['data']
        fx, fy, cx, cy = intrinsics[0], intrinsics[5], intrinsics[2], intrinsics[6]
        
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        fx, fy, cx, cy = resize_rgb_images(rgb_txt, sequence_path, self.resolution_size[0], self.resolution_size[1], camera_matrix)

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')

        CAM2ENU = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
        ENU2CAM = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])

        groundtruth_txt_0 = os.path.join(sequence_path, 'poses', 'dense_global', sequence_name + '.txt')
        with open(groundtruth_txt_0, 'r') as source_file, open(groundtruth_txt, 'w') as destination_file:
            for idx, line in enumerate(source_file, start=0):
                values = np.array([float(x) for x in line.strip().split()])
                ts = values[0]

                # (ENU LiDAR coordinate system -> cam system)
                SE3_ENU = np.eye(4)
                SE3_ENU[:3, 3] = values[1:4]
                SE3_ENU[:3, :3] = R.from_quat(values[[5, 6, 7, 4]]).as_matrix()

                SE3_CAM = ENU2CAM @ SE3_ENU @ CAM2ENU
                tx, ty, tz = SE3_CAM[0, 3], SE3_CAM[1, 3], SE3_CAM[2, 3]
                quat = R.from_matrix(SE3_CAM[:3, :3]).as_quat()
                qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

                line2 = str(ts) + " " + str(tx) + " " + str(ty) + " " + str(tz) + " " + str(qx) + " " + str(
                    qy) + " " + str(qz) + " " + str(qw) + "\n"
                destination_file.write(line2)

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        image_folder = os.path.join(sequence_path, "2d_rect")
        calibration_folder = os.path.join(sequence_path, "calibrations")
        metadata_folder = os.path.join(sequence_path, "metadata")
        timestamps_folder = os.path.join(sequence_path, "timestamps")

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        if os.path.exists(calibration_folder):
            shutil.rmtree(calibration_folder)
        if os.path.exists(metadata_folder):
            shutil.rmtree(metadata_folder)
        if os.path.exists(timestamps_folder):
            shutil.rmtree(timestamps_folder)
