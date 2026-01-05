import os, yaml
import pandas as pd
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.DatasetVSLAMLab_utilities import undistort_rgb_rad_tan, undistort_depth_rad_tan

class MADMAX_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('madmax', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names


    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        if os.path.exists(rgb_txt):
            return
        
        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                name = name.replace('img_rect_color_', '')
                ts = float(name) / 10e9
                file.write(f"{ts:.5f} rgb/{filename}\n")
               
    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')
        gt_6DoF_gnss_and_imu_csv = os.path.join(sequence_path, 'gt_6DoF_gnss_and_imu.csv')

        if os.path.exists(groundtruth_txt):
            return
        
        df = pd.read_csv(gt_6DoF_gnss_and_imu_csv, skiprows=13)
        selected_columns = ['% UNIX time', ' x-enu(m)', ' y-enu(m)', ' z-enu(m)', ' orientation.x', ' orientation.y', ' orientation.z', ' orientation.w']
        df_selected = df[selected_columns]
        df_selected.to_csv(groundtruth_txt, sep=' ', index=False, header=False, float_format="%.5f")

        exit(0)

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
        return
