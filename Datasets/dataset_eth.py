import os, yaml, csv
from pathlib import Path

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

class ETH_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'eth'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]
        for i, nickname in enumerate(self.sequence_nicknames):
            if len(nickname) > 15:
                self.sequence_nicknames[i] = nickname[:15]

    def download_sequence_data(self, sequence_name):
        
        for i_mode, mode in enumerate(self.modes):
            # Variables
            compressed_name_ext = sequence_name + f"_{mode}.zip"
            decompressed_name = sequence_name

            download_url = os.path.join(self.url_download_root, 'datasets', compressed_name_ext)

            # Constants
            compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
            decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

            # Download the compressed file
            if not os.path.exists(compressed_file):
                downloadFile(download_url, self.dataset_path)

            # Decompress the file
            if mode == 'mono':
                if not os.path.exists(decompressed_folder):
                    decompressFile(compressed_file, self.dataset_path)
            if mode =='rgbd':
                if not os.path.exists(os.path.join(decompressed_folder, 'depth')) and not os.path.exists(os.path.join(decompressed_folder, 'depth_0')):
                    decompressFile(compressed_file, self.dataset_path)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path_raw = os.path.join(sequence_path, 'rgb')
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        if os.path.isdir(rgb_path_raw) and not os.path.isdir(rgb_path): 
            os.replace(rgb_path_raw, rgb_path)

        depth_path_raw = os.path.join(sequence_path, 'depth')
        depth_path = os.path.join(sequence_path, 'depth_0')
        if os.path.isdir(depth_path_raw) and not os.path.isdir(depth_path): 
            os.replace(depth_path_raw, depth_path)    

    def create_rgb_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        depth_txt = os.path.join(sequence_path, 'depth.txt')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        def iter_entries(txt_path, old_prefix, new_prefix):
            with open(txt_path, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ts, path = s.split(None, 1)  # timestamp + rest of the line
                    if path.startswith(old_prefix):
                        path = new_prefix + path[len(old_prefix):]
                    yield ts, path

        rgb_entries   = list(iter_entries(rgb_txt,   "rgb/",   "rgb_0/"))
        depth_entries = list(iter_entries(depth_txt, "depth/", "depth_0/"))

        with open(rgb_csv, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb0 (s)", "path_rgb0", "ts_depth0 (s)", "path_depth0"])
            for (ts_r, path_r), (ts_d, path_d) in zip(rgb_entries, depth_entries):
                w.writerow([ts_r, path_r, ts_d, path_d])

    def create_calibration_yaml(self, sequence_name):

        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_txt = os.path.join(sequence_path, 'calibration.txt')
        with open(calibration_txt, 'r') as file:
            calibration = [value for value in file.readline().split()]

        camera0 = {}
        camera0['model'] = 'Pinhole'
        camera0['fx'], camera0['fy'], camera0['cx'], camera0['cy'] = calibration[0], calibration[1], calibration[2], calibration[3]

        rgbd = {}
        rgbd['depth0_factor'] = 5000.0

        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd = rgbd)

    def create_groundtruth_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')
        
        with open(groundtruth_txt, 'r') as file:
            lines = file.readlines()

        number_of_grountruth_header_lines = 1
        new_lines = lines[number_of_grountruth_header_lines:]
        with open(groundtruth_txt, 'w') as file:
            file.writelines(new_lines)

        header = "ts,tx,ty,tz,qx,qy,qz,qw\n"
        new_lines.insert(0, header)
        with open(groundtruth_csv, 'w') as file:
            for line in new_lines:
                values = line.split()
                file.write(','.join(values) + '\n')

    def remove_unused_files(self, sequence_name):
        sequence_path = Path(self.dataset_path, sequence_name)
        for name in ("calibration.txt", "groundtruth.txt", "rgb.txt", "depth.txt", "associated.txt"):
            Path(sequence_path, name).unlink(missing_ok=True)
        