from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
from scipy.spatial.transform import Rotation as R
from contextlib import suppress
from pathlib import Path
from decimal import Decimal
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import csv
import yaml
import json
import re
import cv2
import os

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

# Baseline obtained from substracting camera's `py`s from the main calibration file:
# https://huggingface.co/datasets/collabora/monado-slam-datasets/blob/main/M_monado_datasets/MI_valve_index/extras/calibration.json
STEREO_BASELINE: Final = 0.13378214922445444


class MSDIndex_dataset(DatasetVSLAMLab):
    """MSDIndex dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path) -> None:
        super().__init__("euroc", Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        self.url_calibration: str = cfg["url_calibration"]

        # Sequence nicknames
        self.sequence_nicknames = [
            s.split("_")[0] for s in self.sequence_names
        ]  # "MIPB01"

    def get_sequence_path(self, sequence_name: str, base_path: str) -> str:
        msdmi_subdirs = {
            "MIO": "MIO_others",
            "MIPP": "MIP_playing/MIPP_pistol_whip",
            "MIPB": "MIP_playing/MIPB_beat_saber",
            "MIPT": "MIP_playing/MIPT_thrill_of_the_fight",
        }

        subdir_id = "MIO" if sequence_name.startswith("MIO") else sequence_name[:4]
        subdir = msdmi_subdirs[subdir_id]
        msdmi_path = f"{base_path}/M_monado_datasets/MI_valve_index/{subdir}"
        sequence_path = f"{msdmi_path}/{sequence_name}"
        return sequence_path

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = Path(self.get_sequence_path(sequence_name, self.dataset_path))
        zip_path = sequence_path.with_suffix(".zip")

        if not zip_path.exists():
            url = self._download_url_for(sequence_name)
            downloadFile(url, str(self.dataset_path))

        if sequence_path.exists():
            shutil.rmtree(sequence_path)

        decompressFile(str(zip_path), str(sequence_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = Path(self.get_sequence_path(sequence_name, self.dataset_path))
        cam_count = len(list((sequence_path / "mav0").glob("cam*/data")))
        for cam in range(cam_count):
            target = sequence_path / f"rgb_{cam}"
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
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

        sequence_path = Path(self.get_sequence_path(sequence_name, self.dataset_path))
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
        sequence_path = Path(self.get_sequence_path(sequence_name, self.dataset_path))
        calibration_json = sequence_path / "mav0" / "calibration.json"
        if not calibration_json.exists():
            downloadFile(self.url_calibration, str(calibration_json))

        with open(calibration_json, "r") as f:
            json_content = json.load(f)

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
                "p1": intr["intrinsics"]["p1"],
                "p2": intr["intrinsics"]["p2"],
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
            "gyro_noise": json_content["value0"]["gyro_noise_std"],
            "gyro_bias": json_content["value0"]["gyro_bias_std"],
            "accel_noise": json_content["value0"]["accel_noise_std"],
            "accel_bias": json_content["value0"]["accel_bias_std"],
            "frequency": json_content["value0"]["imu_update_rate"],
        }

        stereo = {"bf": STEREO_BASELINE}

        self.write_calibration_yaml(
            sequence_name, camera0=cams[0], camera1=cams[1], imu=imu, stereo=stereo
        )

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = Path(self.get_sequence_path(sequence_name, self.dataset_path))
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

    def _download_url_for(self, sequence_name: str) -> str:
        base_url = (
            f"{self.url_download_root}/blob/main/M_monado_datasets/MI_valve_index"
        )
        url = f"{self.get_sequence_path(sequence_name, base_url)}.zip"
        return url
