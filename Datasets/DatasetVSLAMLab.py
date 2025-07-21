"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-12
- Updated: 2024-07-12
- License: GPLv3 License

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.

"""

import os, sys, cv2, yaml
from utilities import ws, check_sequence_integrity
from path_constants import VSLAM_LAB_DIR, VSLAM_LAB_EVALUATION_FOLDER


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
        self.create_rgb_txt(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_txt(sequence_name)
        self.remove_unused_files(sequence_name)

    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        return

    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_txt(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        return

    def get_download_issues(self, sequence_names):
        return {}

    def get_calibration_yaml(self, camera_model, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        image_0 = cv2.imread(os.path.join(rgb_path, rgb_files[0]))
        h, w, channels = image_0.shape

        yaml_content_lines = [
            "%YAML:1.0",
            "",
            "# Camera calibration and distortion parameters",
            "Camera.model: " + camera_model,
            "",
            "Camera.fx: " + str(fx),
            "Camera.fy: " + str(fy),
            "Camera.cx: " + str(cx),
            "Camera.cy: " + str(cy),
            "",
            "Camera.k1: " + str(k1),
            "Camera.k2: " + str(k2),
            "Camera.p1: " + str(p1),
            "Camera.p2: " + str(p2),
            "Camera.k3: " + str(k3),
            "",
            "Camera.w: " + str(w),
            "Camera.h: " + str(h),
            "",
            "# Camera frames per second",
            "Camera.fps: " + str(self.rgb_hz)
        ]

        return yaml_content_lines

    def write_calibration_yaml(self, camera_model, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name):

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

        yaml_content_lines = self.get_calibration_yaml(camera_model, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def write_calibration_rgbd_yaml(self, camera_model, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name, depth_factor):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

        yaml_content_lines = self.get_calibration_yaml(camera_model, fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)
        yaml_content_lines.extend(["", "# Depth map factor", "depth_factor: " + str(depth_factor)])

        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def check_sequence_availability(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if os.path.exists(sequence_path):
            sequence_complete = check_sequence_integrity(self.dataset_path, sequence_name, True)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

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
