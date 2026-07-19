import os
import yaml
import shutil

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile


class DATASET_NAME_TEMPLATE_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('dataset_name_template', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

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
