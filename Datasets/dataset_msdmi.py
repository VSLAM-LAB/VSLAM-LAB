from __future__ import annotations

from pathlib import Path
from typing import Final
from contextlib import suppress  
from decimal import Decimal      
from zipfile import ZipFile

import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R
from huggingface_hub import hf_hub_download

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.dataset_utilities import undistort_fisheye

# Baseline obtained from substracting camera's `py`s from the main calibration file:
# https://huggingface.co/datasets/collabora/monado-slam-datasets/blob/main/M_monado_datasets/MI_valve_index/extras/calibration.json
STEREO_BASELINE: Final = 0.13378214922445444

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class MSDMI_dataset(DatasetVSLAMLab):
    """MSDMI dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "msdmi") -> None:    
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
        """
        Build rgb.csv with two synchronized camera streams.
        EUROC data.csv has nanoseconds in col 0 and filename in col 1.
        Convert timestamps to seconds with 6-decimal formatting.
        """
        # TODO: 6-decimal looses nano-second precision without a good reason.
        # Handling of timestamps should be through int64_t or equivalent datatypes.

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
            ts = (df["ts_ns"].astype(np.int64) * 1e-9).astype(float)
            out[f"ts_rgb{i} (s)"] = ts.map(lambda x: f"{x:.6f}")
            out[f"path_rgb{i}"] = "rgb_" + str(i) + "/" + df["name"].astype(str)

        tmp = rgb_csv.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(rgb_csv)
        finally:
            with suppress(FileNotFoundError):
                tmp.unlink()
        
    def create_imu_csv(self, sequence_name: str) -> None:
        """
        Build imu.csv with timestamps in seconds.
        Input:  <seq>/mav0/imu0/data.csv  (EUROC format, #timestamp [ns] ... header)
        Output: <seq>/imu.csv  with columns: ts (s), wx, wy, wz, ax, ay, az
        """
        seq = self.dataset_path / sequence_name
        src = seq / "mav0" / "imu0" / "data.csv"
        dst = seq / "imu.csv"

        if not src.exists():
            return

        # Skip if already up-to-date
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        # Read rows, skipping the header line(s) that start with '#'
        # Handle both comma- or whitespace-separated variants.

        cols = [
            "timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ]
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

        out = df[
            [
                "timestamp [s]",
                "w_RS_S_x [rad s^-1]",
                "w_RS_S_y [rad s^-1]",
                "w_RS_S_z [rad s^-1]",
                "a_RS_S_x [m s^-2]",
                "a_RS_S_y [m s^-2]",
                "a_RS_S_z [m s^-2]",
            ]
        ]

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
        # TODO: VSLAM-LAB currently doesn't support:
        # - A way to specify the full relative pose of two or more cameras, only a baseline parameter
        # - Only "pinhole" camera model seems to be supported. Valve Index
        #   prefers kannala-brandt-4, we are using the alternative
        #   calibration provided by MSD with the radial-tangential-8 model
        #   that matches opencv's default since there are some mentions of
        #   this model as "OPENCV" in the code.

        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"

        calibration_json = sequence_path / self.calibration_file
        #if not calibration_json.exists():
            #downloadFile(self.url_calibration, str(calibration_json))

        with open(calibration_json, "r") as f:
            json_content = json.load(f)

        print(json_content)
        intrinsics = json_content["value0"]["intrinsics"]
        T_imu_cam = json_content["value0"]["T_imu_cam"]
        
        cams = []
        cams_extrinsics = []
        for extr, intr in zip(T_imu_cam, intrinsics):
            cam = {
                "model": "OPENCV",
                "fx": intr["intrinsics"]["fx"],
                "fy": intr["intrinsics"]["fy"],
                "cx": intr["intrinsics"]["cx"],
                "cy": intr["intrinsics"]["cy"],
                "k1": intr["intrinsics"]["k1"],
                "k2": intr["intrinsics"]["k2"],
                "k3": intr["intrinsics"]["k3"],
                "k4": intr["intrinsics"]["k4"],
            }
            cams.append(cam)

            px = extr["px"]
            py = extr["py"]
            pz = extr["pz"]
            qx = extr["qx"]
            qy = extr["qy"]
            qz = extr["qz"]
            qw = extr["qw"]
            r = R.from_quat([qx, qy, qz, qw])
            t = np.array([px, py, pz])
            mat = np.eye(4)
            mat[:3, :3] = r.as_matrix()
            mat[:3, 3] = t
            T_SC = mat.flatten().tolist()
            cams_extrinsics.append(T_SC)

        imu = {
            "transform": cams_extrinsics[0],
            "gyro_noise": json_content["value0"]["gyro_noise_std"][0],
            "gyro_bias": json_content["value0"]["gyro_bias_std"][0],
            "accel_noise": json_content["value0"]["accel_noise_std"][0],
            "accel_bias": json_content["value0"]["accel_bias_std"][0],
            "frequency": json_content["value0"]["imu_update_rate"],
        }


        print(f"{SCRIPT_LABEL}Undistorting images with pinhole-radtan8 model: {rgb_csv}")
        fx, fy, cx, cy, k1, k2, k3, k4 = cams[0]["fx"], cams[0]["fy"], cams[0]["cx"], cams[0]["cy"], \
                                        cams[0]["k1"], cams[0]["k2"], cams[0]["k3"], cams[0]["k4"]

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, k3, k4])
        fx, fy, cx, cy = undistort_fisheye(rgb_csv, sequence_path, camera_matrix, distortion_coeffs)
        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}    
        self.write_calibration_yaml(sequence_name, camera0=camera0, imu=imu)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        gt_csv = sequence_path / "mav0" / "gt" / "data.csv"
        newgt_csv = sequence_path / "groundtruth.csv"

        with open(gt_csv, "r", encoding="utf-8") as incsv, open(
            newgt_csv, "w", encoding="utf-8"
        ) as outcsv:
            pattern = re.compile(
                r"(.*), ?(.*), ?(.*), ?(.*), ?(.*), ?(.*), ?(.*), ?(.*)"
            )
            for line in incsv:
                if line.startswith("#"):
                    continue
                match = pattern.match(line)
                assert match, line
                ts, px, py, pz, qw, qx, qy, qz = match.groups()
                ts = f"{Decimal(ts) * Decimal('1e9'):.0f}"
                processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}\n"
                outcsv.write(processed_line)

    def remove_unused_files(self, sequence_name: str) -> None:
        pass  # Nothing to remove really

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