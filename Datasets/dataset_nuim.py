import os, yaml, shutil
import csv

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

class NUIM_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'nuim'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_frei_png', '') for s in self.sequence_names]
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_nicknames]

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Variables
        compressed_name_ext = sequence_name + '.tar.gz'
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

        for png_file in os.listdir(rgb_path):
            if png_file.endswith(".png"):
                name, ext = os.path.splitext(png_file)
                new_name = f"{int(name):05}{ext}"
                old_file = os.path.join(rgb_path, png_file)
                new_file = os.path.join(rgb_path, new_name)
                os.rename(old_file, new_file)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                file.write(f"{ts:.5f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):

        fx, fy, cx, cy = 481.20, -480.00, 319.50, 239.50
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')

        freiburg_txt = [file for file in os.listdir(sequence_path) if 'freiburg' in file.lower()]
        with open(os.path.join(sequence_path, freiburg_txt[0]), 'r') as source_file:
            with open(groundtruth_txt, 'w') as destination_txt_file, \
                open(groundtruth_csv, 'w', newline='') as destination_csv_file:

                csv_writer = csv.writer(destination_csv_file)
                header = ["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
                csv_writer.writerow(header)
                for line in source_file:
                    values = line.strip().split()
                    values[0] = '{:.8f}'.format(float(values[0]) / self.rgb_hz)
                    
                    destination_txt_file.write(" ".join(values) + "\n")
                    csv_writer.writerow(values)

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        os.remove(os.path.join(sequence_path, 'associations.txt'))
        shutil.rmtree(os.path.join(sequence_path, "depth"))
