import os
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
        self.calibration_url = data['url_download_calib_root']
        self.calibration_name = data['calib_filename']
        

        # Create sequence_nicknames
        self.sequence_nicknames = []
        for sequence_name in self.sequence_names:
            sequence_nickname = sequence_name.replace('sequence_', 'seq ')
            self.sequence_nicknames.append(sequence_nickname)

    def download_sequence_data(self, sequence_name):
        # Variables
        compressed_name = 'dataset-' + sequence_name + '_512_16' + '.tar' 
        download_url = os.path.join(self.url_download_root, compressed_name)

        calibration_compressed = self.calibration_name + '.zip'
        full_calibration_url = self.calibration_url + calibration_compressed
        calib_compressed_path = os.path.join(self.dataset_path, calibration_compressed)
        calib_path = os.path.join(self.dataset_path, self.calibration_name)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name)
        decompressed_folder = os.path.join(self.dataset_path, sequence_name)

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

        # download calibration files
        if not os.path.exists(calib_compressed_path):
            downloadFile(full_calibration_url, self.dataset_path)

        # decompress
        if not os.path.exists(calib_path):
            shutil.rmtree(calib_path)
        decompressFile(calibration_compressed, self.dataset_path)

        # delete the compressed file
        if os.path.exists(calibration_compressed):
            os.remove(calibration_compressed)
        

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')

        os.makedirs(rgb_path, exist_ok=True)

        command = f"pixi run -e monodataset undistort {os.path.join(sequence_path, '')} {sequence_path}" # 
        subprocess.run(command, shell=True)

        os.remove(os.path.join(sequence_path, 'images.zip'))

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        times = []
        times_txt = os.path.join(sequence_path, 'times.txt')
        with open(times_txt, 'r') as file:
            for line in file:
                columns = line.split()
                if columns:
                    times.append(columns[1])

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                ts = float(times[iRGB])
                file.write(f'{ts:.5f} rgb/{filename}\n')

    def create_imu_csv(self, sequence_name):        
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Find the IMU CSV file
        imu_csv_path = os.path.join(sequence_path, 'mav0', 'imu0', 'data.csv')
    
        # Destination path for the renamed file
        imu_destination = os.path.join(sequence_path, 'imu.csv')
    
        # Copy and rename the file
        if os.path.exists(imu_csv_path):
            shutil.copy(imu_csv_path, imu_destination)
        else:
            print(f"Warning: IMU data file not found at {imu_csv_path}")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_file_yaml = os.path.join(sequence_path, self.calibration_name, 'camchain-imucam-imucalib.yaml')
        
        # Load calibration from .yaml file
        with open(calibration_file_yaml, 'r') as file:
            data = yaml.safe_load(file)

        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        if os.path.exists(calibration_yaml):
            return    
        cam_data = data['cam0']
        imu_data = data['imu_parameters']
        ##TODO UNDISTORT THESE FROM FISHEYE TO PINHOLE
        intrinsics = cam_data['intrinsics']
        distortion = cam_data['distortion_coeffs']

        print(f"{SCRIPT_LABEL}Undistorting images with fisheye model: {rgb_path}")
        camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
        distortion_coeffs = np.array([distortion[0], distortion[1], distortion[2], distortion[3]]) #fisheye model so k1, k2, k3, k4
        fx, fy, cx, cy = undistort_fisheye(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)
        camera_model = 'PINHOLE' # manually specifcy pinhole model after undistortion
        k1, k2, k3, k4 = distortion # likely don't need these anymore, but keeping for reference

        camera0 = {'model': camera_model,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4,
                }

        imu = {
                'transform': cam_data['T_cam_imu'],  # 4x4 transformation matrix from camera to IMU
                'gyro_noise': imu_data['sigma_g_c'],
                'gyro_bias': imu_data['sigma_bg'],
                'accel_noise': imu_data['sigma_a_c'],
                'accel_bias': imu_data['sigma_ba'],
                'frequency': imu_data['rate_hz'],
            }
        self.write_calibration_yaml(sequence_name, camera0=camera0, imu=imu)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')

        with open(os.path.join(sequence_path, 'groundtruthSync.txt')) as source_file:
            with open(groundtruth_txt, 'w') as destination_file:
                for line in source_file:
                    if 'NaN' not in line:
                        destination_file.write(line)

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        #os.remove(os.path.join(sequence_path, 'calibration.txt'))
        #os.remove(os.path.join(sequence_path, 'camera.txt'))
        #os.remove(os.path.join(sequence_path, 'pcalib.txt'))
        #os.remove(os.path.join(sequence_path, 'statistics.txt'))
        #os.remove(os.path.join(sequence_path, 'times.txt'))
        #os.remove(os.path.join(sequence_path, 'vignette.png'))
        #os.remove(os.path.join(sequence_path, 'groundtruthSync.txt'))

    ############ Couldn't find the IVTUM dataset code
    # def get_vitum_dataset_code(self):

    #     # Clone and compile "https://github.com/tum-vision/mono_dataset_code.git"
    #     self.mono_dataset_code_directory = os.path.join(VSLAM_LAB_DIR, 'Baselines', 'mono_dataset_code')

    #     if not os.path.exists(os.path.join(self.mono_dataset_code_directory, 'bin', 'playbackDataset')):

    #         command = f"pixi run -e monodataset git-clone"
    #         subprocess.run(command, shell=True)

    #         replace_string_in_files(self.mono_dataset_code_directory, 'CV_LOAD_IMAGE_UNCHANGED', 'cv::IMREAD_UNCHANGED')
    #         replace_string_in_files(self.mono_dataset_code_directory, 'CV_LOAD_IMAGE_GRAYSCALE', 'cv::IMREAD_GRAYSCALE')

    #         CMakeLists_txt = os.path.join(self.mono_dataset_code_directory, 'CMakeLists.txt')
    #         CMakeLists_txt_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', 'CMakeLists.txt')
    #         os.remove(CMakeLists_txt)
    #         shutil.copy(CMakeLists_txt_new, CMakeLists_txt)

    #         main_playbackDataset_cpp = os.path.join(self.mono_dataset_code_directory, 'src', 'main_playbackDataset.cpp')
    #         main_playbackDataset_cpp_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles',
    #                                                     'main_playbackDataset.cpp')
    #         os.remove(main_playbackDataset_cpp)
    #         shutil.copy(main_playbackDataset_cpp_new, main_playbackDataset_cpp)

    #         build_sh = os.path.join(self.mono_dataset_code_directory, 'build.sh')
    #         build_sh_new = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', 'build.sh')
    #         shutil.copy(build_sh_new, build_sh)

    #         command = f"pixi run -e monodataset build"
    #         subprocess.run(command, shell=True)

    #     else:
    #         print('[dataset_monotum.py] \'' + self.mono_dataset_code_directory + '\' already built')
