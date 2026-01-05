import os, yaml
import pandas as pd
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.DatasetVSLAMLab_utilities import undistort_rgb_rad_tan, undistort_depth_rad_tan

class OPENLORIS_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('openloris', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names
        
    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Rename color folder to rgb
        color_folder = os.path.join(sequence_path, 'color')
        rgb_folder = os.path.join(sequence_path, 'rgb')
        if not os.path.exists(rgb_folder) and os.path.exists(color_folder):
            os.rename(color_folder, rgb_folder)

      
    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        color_txt = os.path.join(sequence_path, 'color.txt')

        if not os.path.exists(rgb_txt):
            with open(color_txt, 'r') as infile, open(rgb_txt, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        timestamp, path = parts
                        new_path = path.replace("color", "rgb")
                        outfile.write(f"{timestamp} {new_path}\n")


    def create_calibration_yaml(self, sequence_name):
        fx, fy, cx, cy = (6.1145098876953125e+02, 6.1148571777343750e+02, 4.3320397949218750e+02, 2.4947302246093750e+02)
        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        
        with open(groundtruth_txt, 'r') as file:
            lines = file.readlines()

        if lines and lines[0].strip() == "#Time px py pz qx qy qz qw:":
            lines = lines[1:]  
            with open(groundtruth_txt, 'w') as file:
                file.writelines(lines)

    def remove_unused_files(self, sequence_name):
        return
