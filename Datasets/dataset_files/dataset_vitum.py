import csv
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile


class VITUM_dataset(DatasetVSLAMLab):
    """VITUM dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "vitum") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [name.replace('sequence_', 'seq ') for name in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        # Variables
        sequence_filename = 'dataset-' + sequence_name + '_512_16'
        compressed_name = sequence_filename + '.tar' 
        download_url = os.path.join(self.url_download_root, compressed_name)

        # Constants
        compressed_file = self.dataset_path / compressed_name
        decompressed_folder = self.dataset_path / sequence_filename

        # Download the sequence data
        if not compressed_file.exists():
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not decompressed_folder.exists():
            decompressFile(compressed_file, self.dataset_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        source_path = self.dataset_path / ('dataset-' + sequence_name + '_512_16')
        
        for cam in ['0', '1']:
            images_path = source_path / 'mav0' / f"cam{cam}" / 'data'
            rgb_path = sequence_path / f'rgb_{cam}'

            if rgb_path.exists():
                continue

            rgb_path.mkdir(parents=True, exist_ok=True)

            from PIL import Image    
            for img in images_path.iterdir():
                if img.suffix.lower() == ".png":
                    try:
                        with Image.open(img) as im:

                            img16 = np.array(im, dtype=np.uint16)

                            # scale 16-bit to 8-bit
                            img8 = (img16 / 256).astype(np.uint8)

                            save_path = rgb_path / img.name
                            Image.fromarray(img8).save(save_path)

                    except Exception as e:
                        print(f"Skipping bad image: {img} ({e})")

                        for img in images_path.iterdir():
                            if img.suffix.lower() in {".png", ".jpg"}:
                                shutil.copy(img, rgb_path / img.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        source_path = self.dataset_path / ('dataset-' + sequence_name + '_512_16')
        rgb_csv = self.rgb_csv_path(sequence_name)
        # Build filename -> timestamp mapping from times.txt
        filename_to_timestamp = {}
        rgb_files = {}
        for cam in ['0', '1']:
            filename_to_timestamp[cam] = {}
            times_txt = source_path / 'dso' / f'cam{cam}' / 'times.txt'
            with open(times_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    cols = line.split()
                    if len(cols) >= 2:
                        fname, ts_str = cols[0], cols[1]  # fname like "1520621175986840704"
                        try:
                            filename_to_timestamp[cam][fname] = float(ts_str)
                        except ValueError:
                            continue  # skip malformed rows

            # Collect pngs and write CSV
            rgb_path = sequence_path / f'rgb_{cam}'
            rgb_files[cam] = sorted([f for f in os.listdir(rgb_path) if f.lower().endswith('.png')])

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 	
            writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'])  # header
            for filename_0, filename_1 in zip(rgb_files['0'], rgb_files['1']):
                base_name_0, _ = os.path.splitext(filename_0)
                base_name_1, _ = os.path.splitext(filename_1)
                ts_0 = filename_to_timestamp['0'].get(base_name_0)
                ts_1 = filename_to_timestamp['1'].get(base_name_1)
                ts_0_ns = int(float(ts_0) * 1e9) 
                ts_1_ns = int(float(ts_1) * 1e9) 
                writer.writerow([f'{ts_0_ns}', f'rgb_0/{filename_0}', f'{ts_1_ns}', f'rgb_1/{filename_1}'])
        
    def create_imu_csv(self, sequence_name: str) -> None:
        seq = self.sequence_path(sequence_name)
        source_path = self.dataset_path / ('dataset-' + sequence_name + '_512_16')
        src = source_path / 'mav0'/ 'imu0'/ 'data.csv'
        dst = self.imu_csv_path(sequence_name)
        if not src.exists():
            return

        # Skip if already up-to-date
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        # Read rows, skipping the header line(s) that start with '#'
        # Handle both comma- or whitespace-separated variants.

        cols = ["timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
        df = pd.read_csv(
            src,
            comment="#",
            header=None,
            names=cols,
            sep=r"[\s,]+",
            engine="python",
        )

        if df.empty:
            return

        df.columns = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        tmp = dst.with_suffix(".csv.tmp")
        try:
            df.to_csv(tmp, index=False)
            tmp.replace(dst)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def create_calibration_yaml(self, sequence_name: str) -> None:
        source_path = self.dataset_path / f'dataset-{sequence_name}_512_16'
        calibration_file_cam = source_path / 'dso' / 'camchain.yaml'
        calibration_file_imu0 = source_path / 'dso' / 'imu_config.yaml'

        # Load camera calibration from .yaml file
        with open(calibration_file_cam, 'r') as cam_file:
            cam_data = yaml.safe_load(cam_file)

        # Load IMU calibration from .yaml file
        with open(calibration_file_imu0, 'r') as imu_file:
            imu_data = yaml.safe_load(imu_file)

        cam0_data = cam_data['cam0']
        cam1_data = cam_data['cam1']
        intrinsics_0 = cam0_data['intrinsics']
        intrinsics_1 = cam1_data['intrinsics']
        T_cam0_imu = np.array(cam0_data['T_cam_imu']).reshape((4, 4))
        T_cam1_imu = np.array(cam1_data['T_cam_imu']).reshape((4, 4))

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": intrinsics_0[0:2], "principal_point": intrinsics_0[2:4],
                 "distortion_type": "equid4", "distortion_coefficients": cam0_data['distortion_coeffs'],
                "fps": self.rgb_hz,
                "T_BS": np.linalg.inv(T_cam0_imu)}
        
        rgb1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": intrinsics_1[0:2], "principal_point": intrinsics_1[2:4],
                "distortion_type": "equid4", "distortion_coefficients": cam1_data['distortion_coeffs'],
                "fps": self.rgb_hz,
                "T_BS": np.linalg.inv(T_cam1_imu)}
        
        imu: dict[str, Any] = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c":  imu_data['gyroscope_noise_density'], "sigma_a_c": imu_data['accelerometer_noise_density'],
            "sigma_bg":  0.0, "sigma_ba":  0.0,
            "sigma_gw_c":  imu_data['gyroscope_random_walk'], "sigma_aw_c": imu_data['accelerometer_random_walk'],
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ 0.0, 0.0, 0.0 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": imu_data['update_rate'],
            "T_BS": np.array(np.eye(4)).reshape((4, 4))}
        
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        out_csv = self.groundtruth_csv_path(sequence_name)
        source_path = self.dataset_path / f'dataset-{sequence_name}_512_16' / 'dso' / 'gt_imu.csv'

        with open(source_path, 'r', newline='') as src, open(out_csv, 'w', newline='') as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)

            header = next(reader, None)  # skip/read header
            header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
            if header:
                writer.writerow(header)   # write header to output

            for row in reader:
                if not row:
                    continue
                # Match original behavior: skip any row where any field contains the literal 'NaN'
                if any('NaN' in field for field in row):
                    continue
                writer.writerow(row)

    def remove_unused_files(self, sequence_name: str) -> None:  
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(self.dataset_path / f'dataset-{sequence_name}_512_16', ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            shutil.rmtree(self.dataset_path / f'dataset-{sequence_name}_512_16.tar', ignore_errors=True)