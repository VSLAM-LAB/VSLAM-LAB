import os
import csv
import yaml
import shutil
import subprocess
from typing import Any

import numpy as np

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from PIL import Image


class IMAGEFOLDER_dataset(DatasetVSLAMLAB):
    def __init__(self):
        # Initialize the dataset
        super().__init__('imagefolder')

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.dataset_folder_raw = data['dataset_folder_raw']
        self.fps = data['rgb_hz']
        self.resolution_scale = data['resolution_scale']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
        # Variables
        sequence_path_0 = os.path.join(self.dataset_folder_raw, sequence_name)
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')

        if not os.path.exists(os.path.normpath(sequence_path_0)):
            print(f'The dataset root cannot be found, please correct the root filepath or place the images in the directory: {sequence_path_0}')
            exit(0)

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        for file in os.listdir(sequence_path_0):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                with Image.open(os.path.join(sequence_path_0, file)) as img:
                    scaled_width = int(img.size[0] * self.resolution_scale)

                    # Ensure new_width is even
                    if scaled_width % 2 != 0:
                        scaled_width -= 1
                    scaled_height = int(scaled_width * img.size[1] / img.size[0])
                    
                    # Resize image
                    resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
                    resized_img.save(os.path.join(rgb_path, file))

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        frame_duration = 1.0 / self.fps

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_csv, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for iRGB, filename in enumerate(rgb_files, start=0):
                ts_ns = int(iRGB * frame_duration * 1e9)
                writer.writerow([ts_ns, f"rgb/{filename}"])

    def create_calibration_yaml(self, sequence_name):
        # Create a calibration file with the unknown camera model
        cam_model = "unknown"
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": cam_model,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.fps),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

        # Run glomap to compute calibration parameters
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        output_path = sequence_path
        log_file_path = os.path.join(sequence_path, 'calibration_log_file.txt')
        it = 0
        
        exec_command = [f"sequence_path:{sequence_path}", f"exp_folder:{output_path}", f"exp_id:{it}", "verbose:0"]
        command_str = ' '.join(exec_command)

        ## NB: for this command to work correctly, it relies on building colmap and glomap
        ##     the line in pixi.toml should exist:
        # 329: execute = {cmd = "pixi run -e colmap ./VSLAM-Baselines/glomap/glomap_reconstruction.sh", depends-on = ["build"]}
        full_command = f"pixi run -e glomap execute " + command_str

        print('     Estimating calibration using glomap...')
        print(f'         log file: {log_file_path}')

        with open(log_file_path, 'w') as log_file:
            subprocess.run(full_command, stdout=log_file, stderr=log_file, shell=True)
        
        # Use glomap calibration to rewrite the calibration file
        glomap_calib = os.path.join(sequence_path, 'colmap_00000', 'cameras.txt')

        with open(glomap_calib, 'r') as file:
            lines = file.read().strip().splitlines()

        camera_params = lines[-1].split()
        cam_model = "pinhole"
        fx = float(camera_params[4])
        fy = float(camera_params[4])
        cx = float(camera_params[5])
        cy = float(camera_params[6])

        rgb0 = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": cam_model,
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "fps": float(self.fps),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        glomap_results = os.path.join(sequence_path, 'colmap_00000')
        glomap_keyframe_trajectory = os.path.join(sequence_path, '00000_KeyFrameTrajectory.txt')
        glomap_calibration_log_file = os.path.join(sequence_path, 'calibration_log_file.txt')
        glomap_build_log_file = os.path.join(sequence_path, 'glomap_build_log_file.txt')
        
        if os.path.exists(glomap_results):
            shutil.rmtree(glomap_results)

        if os.path.exists(glomap_keyframe_trajectory):
            os.remove(glomap_keyframe_trajectory)
        if os.path.exists(glomap_calibration_log_file):
            os.remove(glomap_calibration_log_file)
        if os.path.exists(glomap_build_log_file):
            os.remove(glomap_build_log_file)

    def evaluate_trajectory_accuracy(self, trajectory_txt, groundtruth_txt):
        return
