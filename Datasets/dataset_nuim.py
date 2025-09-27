from pathlib import Path
import os, yaml, shutil
import csv
from typing import Final

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

DEFAULT_DEPTH_FACTOR: Final = 5000.0

class NUIM_dataset(DatasetVSLAMLab):
    """NUIM dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "nuim") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace('_frei_png', '') for s in self.sequence_names]
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_nicknames]

    def download_sequence_data(self, sequence_name: str) -> None:
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

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        depth_path = sequence_path / 'depth_0'
        rgb_path_original = sequence_path / 'rgb'
        depth_path_original = sequence_path / 'depth'

        if rgb_path_original.is_dir() and not rgb_path.is_dir():
            os.rename(rgb_path_original, rgb_path) 
        if depth_path_original.is_dir() and not depth_path.is_dir():    
            os.rename(depth_path_original, depth_path) 

        for png_file in os.listdir(rgb_path):
            if png_file.endswith(".png"):
                name, ext = os.path.splitext(png_file)
                new_name = f"{int(name):05}{ext}"
                old_file = os.path.join(rgb_path, png_file)
                new_file = os.path.join(rgb_path, new_name)
                os.rename(old_file, new_file)

        for png_file in os.listdir(depth_path):
            if png_file.endswith(".png"):
                name, ext = os.path.splitext(png_file)
                new_name = f"{int(name):05}{ext}"
                old_file = os.path.join(depth_path, png_file)
                new_file = os.path.join(depth_path, new_name)
                os.rename(old_file, new_file)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ts_rgb0 (s)', 'path_rgb0', 'ts_depth0 (s)', 'path_depth0'])  # header
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz 
                writer.writerow([f"{ts:.5f}", f"rgb_0/{filename}", f"{ts:.5f}", f"depth_0/{filename}"])

    def create_calibration_yaml(self, sequence_name: str) -> None:

        fx, fy, cx, cy = 481.20, -480.00, 319.50, 239.50
        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        rgbd = {"depth0_factor": float(DEFAULT_DEPTH_FACTOR)}

        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd=rgbd)

    def create_groundtruth_csv(self, sequence_name):
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