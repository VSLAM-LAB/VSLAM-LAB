import cv2
import subprocess
import numpy as np
from tqdm import tqdm
import os, yaml, shutil

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.DatasetVSLAMLab_utilities import undistort_fisheye

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class NTNU_ARL_UW_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'ntnu_arl_uw'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Rosbag image topic
        self.image_topic = data['image_topic']

    # def download_sequence_data(self, sequence_name):
    #     rosbag_names = self.get_rosbag_names(sequence_name)
    #     rosbag_path = os.path.join(self.dataset_path, f"{rosbag_names[0]}.bag")
    #     gt_file_name = self.gt_file_names[self.seq_name_wo_cam(sequence_name)]
    #     gt_file = os.path.join(self.dataset_path, gt_file_name)

    #     if (not os.path.exists(rosbag_path)) or (not os.path.exists(gt_file)):
    #         message = (
    #             f"[dataset_{self.dataset_name}.py] \n    The \'{self.dataset_name}\' dataset cannot be downloaded automatically. "
    #             f"\n\n    Please manually download the rosbags and ground-truth files from: \'https://github.com/ntnu-arl/underwater-datasets\'"
    #             f"\n    and place them in: \'{self.dataset_path}\'. "
    #             f"\n\n    ros-bags associated with sequence '{sequence_name}' are: {rosbag_names}"
    #             f"\n\n    The ground-truth file associated with sequence '{sequence_name}' is: '{gt_file_name}'"
    #             f"\n\n    WHEN PROVIDING THE ROSBASGS NOTE THAT THE SEQUENCES ARE DIVIDED WITH TEMPORAL ORDER!!!")
    #         print(message)
    #         exit(0)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
            sequence_name_wo_cam = self.seq_name_wo_cam(sequence_name)
            rosbag_path = os.path.join(self.get_data_path(sequence_name), sequence_name_wo_cam, f'{sequence_name_wo_cam}.bag' )
            extract_rosbag_frames_command = f"pixi run --frozen -e ros-env extract-rosbag-frames "
            extract_rosbag_frames_command += f"--rosbag_path {rosbag_path} "
            extract_rosbag_frames_command += f"--sequence_path {sequence_path} "
            extract_rosbag_frames_command += f"--image_topic {self.image_topic}/{self.get_camera(sequence_name)}"
            subprocess.run(extract_rosbag_frames_command, shell=True)

    def get_intrinsics_from_yaml(self, sequence_name):
        calibration_file = os.path.join(self.dataset_path, 'calibrations',
                                        'cam0_cam1_stereo', 'intrinsics_water', 'camchain-stereo-intrinsics-underwater.yaml' )
    
        camera = self.get_camera(sequence_name)
        with open(calibration_file, 'r') as file:
            data = yaml.safe_load(file)

        fx, fy, cx, cy = data[camera]['intrinsics']
        k1, k2, k3, k4 = data[camera]['distortion_coeffs']

        return fx, fy, cx, cy, k1, k2, k3, k4
    
    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        if os.path.exists(calibration_yaml):
            return

        print(f"{SCRIPT_LABEL}Undistorting images with fisheye model: {rgb_path}")
        fx, fy, cx, cy, k1, k2, k3, k4 = self.get_intrinsics_from_yaml(sequence_name)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, k3, k4])
        fx, fy, cx, cy = undistort_fisheye(rgb_txt, sequence_path, camera_matrix, distortion_coeffs)

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        sequence_name_wo_cam = self.seq_name_wo_cam(sequence_name)
        gt_file = os.path.join(self.get_data_path(sequence_name), sequence_name_wo_cam,f'{sequence_name_wo_cam}_baseline.tum' )
        shutil.copy(gt_file, groundtruth_txt)

    def seq_name_wo_cam(self, sequence_name):
        return sequence_name.replace(f"-{self.get_camera(sequence_name)}", "")

    def get_data_path(self, sequence_name):
        if 'fjord' in sequence_name:
            rosbag_path = os.path.join(self.dataset_path, 'subset-fjord')
        if 'mclab' in sequence_name:
            rosbag_path = os.path.join(self.dataset_path, 'subset-mclab')     

        return rosbag_path
        
    def get_camera(self, sequence_name):
        if 'cam0' in sequence_name:
            return 'cam0'
        if 'cam1' in sequence_name:
            return 'cam1'
        raise ValueError(f"Invalid camera (cam0, cam1): '{sequence_name}'")

    # def get_download_issues(self, sequence_names):
        
    #     missing_rosbags = []
    #     for sequence_name in sequence_names: 
    #         sequence_path = os.path.join(self.dataset_path, sequence_name)
    #         rosbag_names = self.get_rosbag_names(sequence_name)
    #         for rosbag_name in rosbag_names:
    #             if not os.path.isfile(os.path.join(self.dataset_path, rosbag_name)):
    #                 missing_rosbags.append(os.path.join(self.dataset_path, rosbag_name))
        
    #     if not missing_rosbags:
    #         return []
        
    #     solution_msg = f'Please manually download the rosbags and ground-truth files from: \'https://github.com/ntnu-arl/underwater-datasets\', '
    #     solution_msg += f'\n         and place them in: \'{self.dataset_path}\'. '
    #     solution_msg += f'Missing rosbags are:'
    #     for missing_rosbag in missing_rosbags:
    #         solution_msg += f'\n             {missing_rosbag}.bag'

    #     issues = []
    #     issue = {}
    #     issue['name'] = 'Manual download'
    #     issue['description'] = f"The \'{self.dataset_name}\' dataset cannot be downloaded automatically."
    #     issue['solution'] = solution_msg
    #     issue['mode'] = '\033[92mmanual download\033[0m'
        # issues.append(issue)
    #     return issues
