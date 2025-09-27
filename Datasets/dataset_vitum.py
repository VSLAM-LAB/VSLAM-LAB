import os
import csv
import yaml
import shutil
import numpy as np
import subprocess
from pathlib import Path
import pandas as pd

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from Datasets.dataset_utilities import undistort_fisheye

from utilities import replace_string_in_files
from path_constants import VSLAM_LAB_DIR

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics

from utilities import ws

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
class VITUM_dataset(DatasetVSLAMLab):
    """VITUM dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "vitum") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = []
        for sequence_name in self.sequence_names:
            sequence_nickname = sequence_name.replace('sequence_', 'seq ')
            self.sequence_nicknames.append(sequence_nickname)

    def download_sequence_data(self, sequence_name: str) -> None:
        # Variables
        sequence_filename = 'dataset-' + sequence_name + '_512_16'
        compressed_name = sequence_filename + '.tar' 
        download_url = os.path.join(self.url_download_root, compressed_name)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name)
        decompressed_folder = os.path.join(self.dataset_path, sequence_filename)

        # Download the sequence data
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if os.path.exists(decompressed_folder):
            shutil.rmtree(decompressed_folder)
        decompressFile(compressed_file, self.dataset_path)
        
    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')
        images_path = os.path.join(source_path, 'mav0', 'cam0', 'data')
        rgb_path = os.path.join(sequence_path, 'rgb_0')

        os.makedirs(rgb_path, exist_ok=True)
        #copy images to rgb folder
        for filename in os.listdir(images_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                source_file = os.path.join(images_path, filename)
                destination_file = os.path.join(rgb_path, filename)
                shutil.copy(source_file, destination_file) 

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, f'dataset-{sequence_name}_512_16')
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        # Build filename -> timestamp mapping from times.txt
        filename_to_timestamp = {}
        times_txt = os.path.join(source_path, 'dso', 'cam0', 'times.txt')
        with open(times_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cols = line.split()
                if len(cols) >= 2:
                    fname, ts_str = cols[0], cols[1]  # fname like "1520621175986840704"
                    try:
                        filename_to_timestamp[fname] = float(ts_str)
                    except ValueError:
                        continue  # skip malformed rows

        # Collect pngs and write CSV
        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.lower().endswith('.png')])

        with open(rgb_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 	
            writer.writerow(['ts_rgb0 (s)', 'path_rgb0'])  # header
            for filename in rgb_files:
                base_name, _ = os.path.splitext(filename)
                ts = filename_to_timestamp.get(base_name)
                if ts is None:
                    continue  # skip files without a timestamp mapping
                writer.writerow([f'{ts:.6f}', f'rgb_0/{filename}'])

    def create_imu_csv(self, sequence_name: str) -> None:
        """
        Build imu.csv with timestamps in seconds.
        Input:  <seq>/mav0/imu0/data.csv  (EUROC format, #timestamp [ns] ... header)
        Output: <seq>/imu.csv  with columns: ts (s), wx, wy, wz, ax, ay, az
        """
        seq = self.dataset_path / sequence_name
        source_path = self.dataset_path / ('dataset-' + sequence_name + '_512_16')
        src = source_path / 'mav0'/ 'imu0'/ 'data.csv'
        dst = seq / "imu.csv"

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

        # ns → s (float). Keep high precision, then format to 9 decimals for output.
        df["timestamp [s]"] = df["timestamp [ns]"].astype(np.float64) / 1e9
        df["timestamp [s]"] = df["timestamp [s]"].map(lambda x: f"{x:.9f}")

        out = df[["timestamp [s]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]]

        tmp = dst.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(dst)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        source_path = os.path.join(self.dataset_path, 'dataset-' + sequence_name + '_512_16')
        calibration_file_cam0 = os.path.join(source_path, 'dso', 'camchain.yaml')
        calibration_file_imu0 = os.path.join(source_path, 'dso', 'imu_config.yaml')

        # Load camera calibration from .yaml file
        with open(calibration_file_cam0, 'r') as cam_file:
            cam_data = yaml.safe_load(cam_file)

        # Load IMU calibration from .yaml file
        with open(calibration_file_imu0, 'r') as imu_file:
            imu_data = yaml.safe_load(imu_file)

        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

        if os.path.exists(calibration_yaml):
            return
        
        cam_data = cam_data['cam0']
        intrinsics = cam_data['intrinsics']
        distortion = cam_data['distortion_coeffs']

        gyro_noise = imu_data['gyroscope_noise_density']
        gyro_bias = imu_data['gyroscope_random_walk']
        accel_noise = imu_data['accelerometer_noise_density']
        accel_bias = imu_data['accelerometer_random_walk']

        print(f"{SCRIPT_LABEL}Undistorting images with fisheye model: {rgb_path}")
        camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
        distortion_coeffs = np.array([distortion[0], distortion[1], distortion[2], distortion[3]]) #fisheye model so k1, k2, k3, k4
        fx, fy, cx, cy = undistort_fisheye(rgb_csv, sequence_path, camera_matrix, distortion_coeffs)
        camera_model = 'Pinhole' # manually specifcy pinhole model after undistortion
    
        camera0 = {'model': camera_model, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        imu = {
                'transform': cam_data['T_cam_imu'],  # 4x4 transformation matrix from camera to IMU
                'gyro_noise': gyro_noise,
                'gyro_bias': gyro_bias,
                'accel_noise': accel_noise,
                'accel_bias': accel_bias,
                'frequency': imu_data['update_rate']
                }
        self.write_calibration_yaml(sequence_name, camera0=camera0, imu=imu)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        out_csv = os.path.join(sequence_path, 'groundtruth.csv')
        source_path = os.path.join(self.dataset_path, f'dataset-{sequence_name}_512_16', 'dso', 'gt_imu.csv')

        with open(source_path, 'r', newline='') as src, open(out_csv, 'w', newline='') as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)

            header = next(reader, None)  # skip/read header
            if header:
                writer.writerow(header)   # write header to output

            for row in reader:
                if not row:
                    continue
                # Match original behavior: skip any row where any field contains the literal 'NaN'
                if any('NaN' in field for field in row):
                    continue
                writer.writerow(row)
