from __future__ import annotations

import os
import cv2
import yaml
import pandas as pd
from typing import  Any
from pathlib import Path
from huggingface_hub import hf_hub_download

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import decompressFile
from path_constants import Retention, BENCHMARK_RETENTION


class OPENLORIS_dataset(DatasetVSLAMLab):
    """OPENLORIS dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "openloris") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get repo id
        self.repo_id: str = cfg["repo_id"]
        self.dataset_path_raw: Path = self.benchmark_path / "OPENLORIS"

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        if sequence_path.exists():
            return  
        
        compressed_name_ext = sequence_name + '.7z' 
        compressed_file = self.dataset_path_raw / compressed_name_ext
        if not compressed_file.exists():
            file_path = hf_hub_download(repo_id=self.repo_id, filename=_get_compressed_file_name(sequence_name), repo_type='dataset')
            decompressFile(file_path, self.dataset_path_raw)

            if not sequence_name.startswith("corridor1-1"):   
                decompressFile(compressed_file, self.dataset_path_raw)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass
        
    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_calibration_yaml(self, sequence_name):
        pass

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        accel_txt = self.dataset_path_raw / sequence_name / f'{self.camera_name}_accelerometer.txt'
        gyro_txt = self.dataset_path_raw / sequence_name / f'{self.camera_name}_gyroscope.txt'
        imu_csv = sequence_path / "imu_0.csv"

        df_accel = pd.read_csv(accel_txt, sep=r'\s+', comment='#', header=None, names=['Time', 'Ax', 'Ay', 'Az'])
        df_gyro = pd.read_csv(gyro_txt, sep=r'\s+', comment='#', header=None, names=['Time', 'Gx', 'Gy', 'Gz'])

        df_accel = df_accel.sort_values('Time')
        df_gyro = df_gyro.sort_values('Time')

        merged_df = pd.merge_asof(df_gyro, df_accel, on='Time', direction='nearest', tolerance=0.02 )
        merged_df.dropna(inplace=True)
        merged_df['Time'] = (merged_df['Time'] * 1e9).astype('int64')
        merged_df = merged_df.rename(columns={
            'Time': 'ts (ns)',
            'Gx': 'wx (rad s^-1)', 'Gy': 'wy (rad s^-1)', 'Gz': 'wz (rad s^-1)',
            'Ax': 'ax (m s^-2)', 'Ay': 'ay (m s^-2)', 'Az': 'az (m s^-2)'
        })
            
        target_order = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        merged_df = merged_df[target_order]
        merged_df.to_csv(imu_csv, index=False)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_txt = self.dataset_path_raw / sequence_name / 'groundtruth.txt'
        groundtruth_csv = sequence_path / 'groundtruth.csv'

        df = pd.read_csv(groundtruth_txt, sep=r'\s+', header=None, comment='#')
        df.columns = ["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"]
        df['ts (ns)'] = pd.to_datetime(df['ts (ns)'], unit='s').astype('int64')
        df.to_csv(groundtruth_csv, index=False)

    def remove_unused_files(self, sequence_name):
        return

def _get_compressed_file_name(sequence_name: str) -> str:
    if sequence_name.startswith("cafe1"):
        return "package/cafe1-1_2-package.tar"
    if sequence_name.startswith("corridor1") and not sequence_name.startswith("corridor1-1"):
        return "package/corridor1-2_5-package.tar"
    if sequence_name.startswith("office1"):
        return "package/office1-1_7-package.tar"
    if sequence_name.startswith("home1"):
        return "package/home1-1_5-package.tar"
    if sequence_name.startswith("market1"):
        return "package/market1-1_3-package.tar"
    if sequence_name.startswith("corridor1-1"):
        return "package/corridor1-1.7z"
    
class OPENLORIS_d400_dataset(OPENLORIS_dataset):
    """OPENLORIS_d400 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "openloris-d400") -> None:
        super().__init__(Path(benchmark_path), dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Depth factor
        self.depth_factor = cfg["depth_factor"]

        self.camera_name = "d400"

    def create_rgb_folder(self, sequence_name: str) -> None:
        openloris_path = self.dataset_path_raw / sequence_name
        sequence_path = self.dataset_path / sequence_name
        raw_folders = ['color', 'aligned_depth']
        new_folders = ['rgb_0', 'depth_0']
        sequence_path.mkdir(parents=True, exist_ok=True)

        for raw_folder, new_folder in zip(raw_folders, new_folders):
            raw_path = openloris_path / raw_folder
            new_path = sequence_path / new_folder
            if raw_path.is_dir() and not new_path.exists():
               os.symlink(raw_path, new_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        openloris_path = self.dataset_path_raw / sequence_name
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return
        color_txt = openloris_path / 'color.txt'
        depth_txt = openloris_path / 'aligned_depth.txt'
        df_rgb = pd.read_csv(color_txt, sep=r'\s+', names=['ts', 'path'], comment='#', dtype={'ts': str})
        df_depth = pd.read_csv(depth_txt, sep=r'\s+', names=['ts', 'path'], comment='#', dtype={'ts': str})
        ts_rgb_numeric = pd.to_numeric(df_rgb['ts'])
        ts_depth_numeric = pd.to_numeric(df_depth['ts'])        
        df = pd.DataFrame({
            'ts_rgb_0 (ns)': pd.to_datetime(ts_rgb_numeric, unit='s').astype('int64'),
            'path_rgb_0': df_rgb['path'].str.replace('color/', 'rgb_0/', regex=False),
            'ts_depth_0 (ns)': pd.to_datetime(ts_depth_numeric, unit='s').astype('int64'),
            'path_depth_0': df_depth['path'].str.replace('aligned_depth/', 'depth_0/', regex=False)   
        })
        df.to_csv(rgb_csv, index=False)
        
    def create_calibration_yaml(self, sequence_name):
        sensors_yaml = self.dataset_path_raw / sequence_name / 'sensors.yaml'
        trans_yaml = self.dataset_path_raw / sequence_name / 'trans_matrix.yaml'

        fs_sensor = cv2.FileStorage(sensors_yaml, cv2.FILE_STORAGE_READ)
        fs_trans = cv2.FileStorage(trans_yaml, cv2.FILE_STORAGE_READ)
        trans_node = fs_trans.getNode("trans_matrix")

        node = fs_sensor.getNode("d400_color_optical_frame")
        fx, cx, fy, cy = node.getNode("intrinsics").mat().flatten().data

        trans_id = _find_trans_id(trans_node, "base_link", "d400_color_optical_frame")
        T_BS = trans_node.at(trans_id).getNode("matrix").mat()
        
        rgbd0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb+depth", "depth_name": "depth_0",
            "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
            "depth_factor": float(self.depth_factor),
            "fps": float(node.getNode("fps").real()),
            "T_BS": T_BS}

        trans_id = _find_trans_id(trans_node, f"d400_color_optical_frame", "d400_accelerometer")
        T_S_A = trans_node.at(trans_id).getNode("matrix").mat()
        T_BA = T_BS @ T_S_A
    
        fps_gyro, _, _, _, _ = _get_imu_noise_parameters(fs_sensor, self.camera_name)
        imu: dict[str, Any] = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c": 1e-2, "sigma_a_c": 1e-1,
            "sigma_gw_c":  1e-6 , "sigma_aw_c": 1e-4,
            "sigma_bg":  0.0, "sigma_ba":  0.0,
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ 0.0, 0.0, 0.0 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": fps_gyro,
            "T_BS": T_BA}
        
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0], imu =[imu])


class OPENLORIS_t265_dataset(OPENLORIS_dataset):
    """OPENLORIS_t265 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "openloris-t265") -> None:
        super().__init__(Path(benchmark_path), dataset_name)
        self.camera_name = "t265"

    def create_rgb_folder(self, sequence_name: str) -> None:
        openloris_path = self.dataset_path_raw / sequence_name
        sequence_path = self.dataset_path / sequence_name
        raw_folders = ['fisheye1', 'fisheye2']
        new_folders = ['rgb_0', 'rgb_1']
        sequence_path.mkdir(parents=True, exist_ok=True)
        for raw_folder, new_folder in zip(raw_folders, new_folders):
            raw_path = openloris_path / raw_folder
            new_path = sequence_path / new_folder
            if raw_path.is_dir() and not new_path.exists():
               os.symlink(raw_path, new_path)
        
    def create_rgb_csv(self, sequence_name: str) -> None:
        openloris_path = self.dataset_path_raw / sequence_name
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return
        fisheye1_txt = openloris_path / 'fisheye1.txt'
        fisheye2_txt = openloris_path / 'fisheye2.txt'
        df_fisheye1 = pd.read_csv(fisheye1_txt, sep=r'\s+', names=['ts', 'path'], comment='#', dtype={'ts': str})
        df_fisheye2 = pd.read_csv(fisheye2_txt, sep=r'\s+', names=['ts', 'path'], comment='#', dtype={'ts': str})
        ts_fisheye1_numeric = pd.to_numeric(df_fisheye1['ts'])
        ts_fisheye2_numeric = pd.to_numeric(df_fisheye2['ts'])        
        df = pd.DataFrame({
            'ts_rgb_0 (ns)': pd.to_datetime(ts_fisheye1_numeric, unit='s').astype('int64'),
            'path_rgb_0': df_fisheye1['path'].str.replace('fisheye1/', 'rgb_0/', regex=False),
            'ts_rgb_1 (ns)': pd.to_datetime(ts_fisheye2_numeric, unit='s').astype('int64'),
            'path_rgb_1': df_fisheye2['path'].str.replace('fisheye2/', 'rgb_1/', regex=False)   
        })
        df.to_csv(rgb_csv, index=False)
 
    def create_calibration_yaml(self, sequence_name):
        
        sensors_yaml = self.dataset_path_raw / sequence_name / 'sensors.yaml'
        trans_yaml = self.dataset_path_raw / sequence_name / 'trans_matrix.yaml'

        fs_sensor = cv2.FileStorage(sensors_yaml, cv2.FILE_STORAGE_READ)
        fs_trans = cv2.FileStorage(trans_yaml, cv2.FILE_STORAGE_READ)
        trans_node = fs_trans.getNode("trans_matrix")

        node = fs_sensor.getNode(f"{self.camera_name}_fisheye1_optical_frame")
        fx, cx, fy, cy = node.getNode("intrinsics").mat().flatten().data
        k1, k2, k3, k4, _ = node.getNode("distortion_coefficients").mat().flatten().data

        trans_id = _find_trans_id(trans_node, "base_link", "t265_fisheye1_optical_frame")
        T_BS1 = trans_node.at(trans_id).getNode("matrix").mat()

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "distortion_type": "equid4", "distortion_coefficients": [k1, k2, k3, k4],
                "fps": float(node.getNode("fps").real()),
                "T_BS": T_BS1}
        
        node = fs_sensor.getNode(f"{self.camera_name}_fisheye2_optical_frame")
        fx, cx, fy, cy = node.getNode("intrinsics").mat().flatten().data
        k1, k2, k3, k4, _ = node.getNode("distortion_coefficients").mat().flatten().data

        trans_id = _find_trans_id(trans_node, "t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame")
        T_S1S2 = trans_node.at(trans_id).getNode("matrix").mat()
        T_BS2 = T_BS1 @ T_S1S2

        rgb1: dict[str, Any] = {"cam_name": "rgb_1", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "distortion_type": "equid4", "distortion_coefficients": [k1, k2, k3, k4],
                "fps": float(node.getNode("fps").real()),
                "T_BS": T_BS2}

        trans_id = _find_trans_id(trans_node, f"t265_fisheye1_optical_frame", "t265_accelerometer")
        T_S1_A = trans_node.at(trans_id).getNode("matrix").mat()
        T_BA = T_BS1 @ T_S1_A
 
        fps_gyro, sigma_a_c, sigma_aw_c, sigma_g_c, sigma_gw_c = _get_imu_noise_parameters(fs_sensor, self.camera_name)
        
        imu: dict[str, Any] = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c": sigma_g_c, "sigma_a_c": sigma_a_c,
            "sigma_gw_c":  sigma_gw_c, "sigma_aw_c": sigma_aw_c,
            "sigma_bg":  0.0, "sigma_ba":  0.0,
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ 0.0, 0.0, 0.0 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": fps_gyro,
            "T_BS": T_BA}
               
        self.write_calibration_yaml(sequence_name=sequence_name,  rgb=[rgb0, rgb1], imu=[imu])


def _get_imu_noise_parameters(fs_sensor, camera_name):

    fps_accel = fs_sensor.getNode(f"{camera_name}_accelerometer").getNode("fps").real()
    sigma2_accel = fs_sensor.getNode(f"{camera_name}_accelerometer").getNode("noise_variances").mat().flatten().data[0]
    sigma2_bias_accel = fs_sensor.getNode(f"{camera_name}_accelerometer").getNode("bias_variances").mat().flatten().data[0]

    fps_gyro = fs_sensor.getNode(f"{camera_name}_gyroscope").getNode("fps").real()
    sigma2_gyro = fs_sensor.getNode(f"{camera_name}_gyroscope").getNode("noise_variances").mat().flatten().data[0]
    sigma2_bias_gyro = fs_sensor.getNode(f"{camera_name}_gyroscope").getNode("bias_variances").mat().flatten().data[0]

    sigma_a_c = (sigma2_accel  / fps_accel) ** 0.5
    sigma_aw_c = (sigma2_bias_accel / fps_accel) ** 0.5
    sigma_g_c = (sigma2_gyro  / fps_gyro) ** 0.5
    sigma_gw_c = (sigma2_bias_gyro / fps_gyro) ** 0.5

    return fps_gyro, sigma_a_c, sigma_aw_c, sigma_g_c, sigma_gw_c

def _find_trans_id(trans_node, parent_frame, child_frame)-> Any:
    for i in range(trans_node.size()):
        node = trans_node.at(i)
        if node.getNode("child_frame").string() == child_frame and node.getNode("parent_frame").string() == parent_frame:
            return i
    return None
