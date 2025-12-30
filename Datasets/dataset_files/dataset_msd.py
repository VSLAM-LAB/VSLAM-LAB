from __future__ import annotations

import csv
import yaml
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import  Any
from contextlib import suppress
from zipfile import ZipFile
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import Retention, BENCHMARK_RETENTION


class MSD_dataset(DatasetVSLAMLab):
    """MSD dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "msd") -> None:    
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.huggingface_repo_id: str = cfg["huggingface_repo_id"]
        self.huggingface_subfolder: str = cfg["huggingface_subfolder"]
        self.calibration_file: str = cfg["calibration_file"]

        # Sequence nicknames
        self.sequence_nicknames = [s.split("_")[0] for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        if not sequence_path.exists():
            compressed_name_ext = sequence_name + '.zip'
            file_path = hf_hub_download(repo_id=self.huggingface_repo_id, 
                                        subfolder=self.get_huggingface_subfolder(sequence_name, self.huggingface_subfolder),
                                        filename=compressed_name_ext, repo_type='dataset')
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)

            file_path = hf_hub_download(repo_id=self.huggingface_repo_id, subfolder=self.huggingface_subfolder + "/extras", 
                                        filename=self.calibration_file, repo_type='dataset')

            shutil.copy2(file_path, sequence_path / self.calibration_file)

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Create symlinks for each camera folder
        sequence_path = self.dataset_path / sequence_name
        cam_count = len(list((sequence_path / "mav0").glob("cam*/data")))
        
        for cam in range(cam_count):
            target = sequence_path / f"rgb_{cam}"
            if target.exists():
                continue
            src_dir = sequence_path / "mav0" / f"cam{cam}" / "data"
            target.symlink_to(src_dir)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        mav0_dir = sequence_path / "mav0"
        out = pd.DataFrame()
        for cam_dir in mav0_dir.glob("cam*"):
            i = int(cam_dir.name.split("cam")[-1])
            data_csv = cam_dir / "data.csv"
            if not data_csv.exists():
                raise FileNotFoundError(f"Missing {data_csv} in {sequence_path}/mav0")
            df = pd.read_csv(
                data_csv,
                comment="#",
                header=None,
                usecols=[0, 1],
                names=["ts_ns", "name"],
            )
            out[f"ts_rgb_{i} (ns)"] = df["ts_ns"].astype(np.int64)
            out[f"path_rgb_{i}"] = "rgb_" + str(i) + "/" + df["name"].astype(str)

        tmp = rgb_csv.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(rgb_csv)
        finally:
            with suppress(FileNotFoundError):
                tmp.unlink()

    def create_imu_csv(self, sequence_name: str) -> None:
        seq = self.dataset_path / sequence_name
        src = seq / "mav0" / "imu0" / "data.csv"
        dst = seq / "imu_0.csv"

        if not src.exists():
            return

        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        raw_cols = ["timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
        df = pd.read_csv(src, comment="#", header=None, names=raw_cols, sep=r"[\s,]+", engine="python")

        if df.empty:
            return

        new_cols = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        df.columns = new_cols
        out = df[new_cols]

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
        sequence_path = self.dataset_path / sequence_name

        calibration_json = sequence_path / self.calibration_file
        with open(calibration_json, "r") as f:
            json_content = json.load(f)

        intrinsics = json_content["value0"]["intrinsics"]
        T_imu_cam = json_content["value0"]["T_imu_cam"]
        
        cams = []
        for i, (extr, intr) in enumerate(zip(T_imu_cam, intrinsics)):
            params = intr["intrinsics"]
            T_BS = np.eye(4)
            T_BS[:3, :3] = R.from_quat([extr[k] for k in ["qx", "qy", "qz", "qw"]]).as_matrix()
            T_BS[:3, 3] = [extr[k] for k in ["px", "py", "pz"]]

            cam: dict[str, Any] = {
                "cam_name": f"rgb_{i}",
                "cam_type": "gray",
                "cam_model": "pinhole",
                "focal_length": [params["fx"], params["fy"]],
                "principal_point": [params["cx"], params["cy"]],
                "fps": float(self.rgb_hz),
                "T_BS": T_BS,
                "distortion_type": "equid4",
                "distortion_coefficients": [params[f"k{j}"] for j in range(1, 5)]
            }
            cams.append(cam)

        imu_hz = json_content["value0"]["imu_update_rate"]
        imu = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c": json_content["value0"]["gyro_noise_std"][0] / np.sqrt(imu_hz), 
            "sigma_a_c": json_content["value0"]["accel_noise_std"][0] / np.sqrt(imu_hz), 
            "sigma_bg":  json_content["value0"]["calib_gyro_bias"][0], 
            "sigma_ba":  json_content["value0"]["calib_accel_bias"][0], 
            "sigma_gw_c": json_content["value0"]["gyro_bias_std"][0] / np.sqrt(imu_hz), 
            "sigma_aw_c": json_content["value0"]["accel_bias_std"][0] / np.sqrt(imu_hz), 
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ 0.0, 0.0, 0.0 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": json_content["value0"]["imu_update_rate"],
            "T_BS": np.array(np.eye(4)).reshape((4, 4))
            }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=cams, imu=[imu])
    
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        gt_csv = sequence_path / "mav0" / "gt" / "data.csv"
        newgt_csv = sequence_path / "groundtruth.csv"

        if not gt_csv.exists():
            return

        with open(gt_csv, "r", encoding="utf-8") as incsv, \
            open(newgt_csv, "w", encoding="utf-8", newline="") as outcsv:
            
            w = csv.writer(outcsv)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            
            for line in incsv:
                if line.startswith("#") or not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue

                ts_ns = int(float(parts[0]))
                px, py, pz, qw, qx, qy, qz = parts[1:8]
                w.writerow([ts_ns, px, py, pz, qx, qy, qz, qw])

    def get_huggingface_subfolder(self, sequence_name: str, base_path: str) -> str:
        msdmi_subdirs = {
            "MIO": "MIO_others",
            "MIPP": "MIP_playing/MIPP_pistol_whip",
            "MIPB": "MIP_playing/MIPB_beat_saber",
            "MIPT": "MIP_playing/MIPT_thrill_of_the_fight",
        }

        subdir_id = "MIO" if sequence_name.startswith("MIO") else sequence_name[:4]
        subdir = msdmi_subdirs[subdir_id]
        msdmi_path = f"{base_path}/{subdir}"
        return msdmi_path
    
    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (sequence_path / "calibration.json").unlink(missing_ok=True)