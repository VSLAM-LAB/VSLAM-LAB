"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-12
- Updated: 2024-07-12
- License: GPLv3 License

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.

"""

import os, sys, yaml

from utilities import ws
from path_constants import VSLAM_LAB_DIR
from Datasets.dataset_calibration import _get_camera_yaml_section
from Datasets.dataset_calibration import _get_imu_yaml_section
from Datasets.dataset_calibration import _get_rgbd_yaml_section

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class DatasetVSLAMLab:

    def __init__(self, dataset_name, benchmark_path):

        self.dataset_name = dataset_name
        self.dataset_color = "\033[38;2;255;165;0m"
        self.dataset_label = f"{self.dataset_color}{dataset_name}\033[0m"
        self.dataset_folder = dataset_name.upper()
        self.benchmark_path = benchmark_path
        self.dataset_path = os.path.join(self.benchmark_path, self.dataset_folder)

        self.yaml_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_' + self.dataset_name + '.yaml')

        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        self.sequence_names = data['sequence_names']
        self.rgb_hz = data['rgb_hz']
        self.sequence_nicknames = []
        self.modes = data.get('modes', 'mono')

    ####################################################################################################################
    # Download methods
    def download_sequence(self, sequence_name):

        # Check if sequence is already available
        sequence_availability = self.check_sequence_availability(sequence_name)
        if sequence_availability == "available":
            #print(f"{SCRIPT_LABEL}Sequence {self.dataset_color}{sequence_name}:\033[92m downloaded\033[0m")
            return
        if sequence_availability == "corrupted":
            print(f"{ws(8)}Some files in sequence {sequence_name} are corrupted.")
            print(f"{ws(8)}Removing and downloading again sequence {sequence_name} ")
            print(f"{ws(8)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY")
            sys.exit(1)

        # Download process
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)

        self.download_process(sequence_name)

    def download_process(self, sequence_name):
        msg = f"Downloading sequence {self.dataset_color}{sequence_name}\033[0m from dataset {self.dataset_color}{self.dataset_name}\033[0m ..."
        print(SCRIPT_LABEL + msg)
        self.download_sequence_data(sequence_name)
        self.create_rgb_folder(sequence_name)
        self.create_rgb_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.remove_unused_files(sequence_name)

    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_csv(self, sequence_name):
        return

    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_csv(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        return

    def get_download_issues(self, sequence_names):
        return {}

    def write_calibration_yaml(self, sequence_name, camera0=None, camera1=None, imu=None, rgbd=None):
    #Write calibration YAML file with flexible sensor configuration.
    #Args:
    #    sequence_name: Name of the sequence
    #    camera0: Dict with keys: model, fx, fy, cx, cy, k1, k2, p1, p2, k3
    #    camera1: Dict with keys: model, fx, fy, cx, cy, k1, k2, p1, p2, k3 (for stereo)
    #    imu: Dict with keys: transform, accel_noise, gyro_noise, accel_bias, gyro_bias, frequency
    #    rgbd: Dict with keys: depth_factor, depth_scale (optional)       

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        
        yaml_content_lines = ["%YAML:1.0", ""]

        # Camera0 parameters (required)
        if camera0:
            yaml_content_lines.extend(["", "# Camera0 calibration and distortion parameters"])
            yaml_content_lines.extend(_get_camera_yaml_section(self.dataset_path, camera0, sequence_name, self.rgb_hz, "Camera0"))

        # Camera1 parameters (for stereo)
        if camera1:
            yaml_content_lines.extend(["", "# Camera1 calibration and distortion parameters"])
            yaml_content_lines.extend(_get_camera_yaml_section(self.dataset_path, camera1, sequence_name, self.rgb_hz, "Camera1"))

        # IMU parameters
        if imu:
            yaml_content_lines.extend(["", "# IMU parameters"])
            yaml_content_lines.extend(_get_imu_yaml_section(imu))

        # RGBD parameters
        if rgbd:
            yaml_content_lines.extend(["", "# Depth0 map parameters"])
            yaml_content_lines.extend(_get_rgbd_yaml_section(rgbd, "Depth0"))

        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def check_sequence_availability(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if os.path.exists(sequence_path):
            sequence_complete = self.check_sequence_integrity(sequence_name, verbose=True)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

    def check_sequence_integrity(self, sequence_name, verbose):
        
        complete_sequence = True

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if not os.path.exists(sequence_path):
            if verbose:
                print(f"        The folder {sequence_path} doesn't exist !!!!!")
            complete_sequence = False

        rgb_path = os.path.join(sequence_path, 'rgb_0')
        if not os.path.exists(rgb_path):
            if verbose:
                print(f"        The folder {rgb_path} doesn't exist !!!!!")
            complete_sequence = False

        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        if not os.path.exists(rgb_csv):
            if verbose:
                print(f"        The file {rgb_csv} doesn't exist !!!!!")
            complete_sequence = False

        calibration_yaml = os.path.join(sequence_path, "calibration.yaml")
        if not os.path.exists(calibration_yaml):
            if verbose:
                print(f"        The file {calibration_yaml} doesn't exist !!!!!")
            complete_sequence = False

        if 'mono-vi' in self.modes:
            imu_csv = os.path.join(sequence_path, 'imu.csv')
            if not os.path.exists(imu_csv):
                if verbose:
                    print(f"        The file {imu_csv} doesn't exist !!!!!")
                complete_sequence = False

        return complete_sequence

    ####################################################################################################################
    # Utils

    def contains_sequence(self, sequence_name_ref):
        for sequence_name in self.sequence_names:
            if sequence_name == sequence_name_ref:
                return True
        return False

    def print_sequence_names(self):
        print(self.sequence_names)

    def print_sequence_nicknames(self):
        print(self.sequence_nicknames)

    def get_sequence_names(self):
        return self.sequence_names

    def get_sequence_nicknames(self):
        return self.sequence_nicknames

    def get_sequence_nickname(self, sequence_name_ref):
        for i, sequence_name in enumerate(self.sequence_names):
            if sequence_name == sequence_name_ref:
                return self.sequence_nicknames[i]

    def get_sequence_num_rgb(self, sequence_name):
        rgb_txt = os.path.join(self.dataset_path, sequence_name, 'rgb.txt')
        if os.path.exists(rgb_txt):
            with open(rgb_txt, 'r') as file:
                line_count = 0
                for line in file:
                    line_count += 1
            return line_count
        return 0
