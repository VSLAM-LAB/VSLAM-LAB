from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
from contextlib import suppress
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import csv
import yaml
import cv2
import os

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

STEREO_BASELINE: Final = 0.11

class EUROC_dataset(DatasetVSLAMLab):
    """EUROC MAV dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path) -> None:
        super().__init__("euroc", Path(benchmark_path))
    
        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        
        # Sequence nicknames
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        zip_path = self.dataset_path / f"{sequence_name}.zip"

        if not zip_path.exists():
            url = self._download_url_for(sequence_name)
            downloadFile(url, str(self.dataset_path))
        
        if sequence_path.exists():
            shutil.rmtree(sequence_path)

        decompressFile(str(zip_path), str(sequence_path))

        # Download TUM supplemental ground-truth if needed
        supp_root = self.dataset_path / "supp_v2"
        if not supp_root.exists():
            supp_zip = self.dataset_path / "supp_v2.zip"
            supp_url = "https://cvg.cit.tum.de/mono/supp_v2.zip"
            if not supp_zip.exists():
                downloadFile(supp_url, str(self.dataset_path))

            if supp_root.exists():
                shutil.rmtree(supp_root)

            decompressFile(str(supp_zip), str(supp_root))
            with suppress(FileNotFoundError):
                supp_zip.unlink()

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for cam in (0, 1):
            target = sequence_path / f"rgb_{cam}"
            if target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            src_dir = sequence_path / "mav0" / f"cam{cam}" / "data"
            if not src_dir.is_dir():
                continue
            for png in sorted(src_dir.glob("*.png")):
                shutil.copy(png, target / png.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        """
        Build rgb.csv with two synchronized camera streams.
        EUROC data.csv has nanoseconds in col 0 and filename in col 1.
        Convert timestamps to seconds with 6-decimal formatting.
        """
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        cam0_csv = sequence_path / "mav0" / "cam0" / "data.csv"
        cam1_csv = sequence_path / "mav0" / "cam1" / "data.csv"
        if not (cam0_csv.exists() and cam1_csv.exists()):
            raise FileNotFoundError(f"Missing cam data.csv in {sequence_path}/mav0")

        # Read as strings, then cast/format (some EUROC CSVs have headers)
        df0 = pd.read_csv(cam0_csv, comment="#", header=None, usecols=[0, 1], names=["ts_ns", "name0"])
        df1 = pd.read_csv(cam1_csv, comment="#", header=None, usecols=[0, 1], names=["ts_ns", "name1"])

        # Ensure equal length & alignment by index (EUROC is aligned across cams)
        n = min(len(df0), len(df1))
        df0, df1 = df0.iloc[:n], df1.iloc[:n]

        # Convert ns -> seconds
        ts0 = (df0["ts_ns"].astype(np.int64) * 1e-9).astype(float)
        ts1 = (df1["ts_ns"].astype(np.int64) * 1e-9).astype(float)

        out = pd.DataFrame({
            "ts_rgb0 (s)": ts0.map(lambda x: f"{x:.6f}"),
            "path_rgb0": "rgb_0/" + df0["name0"].astype(str),
            "ts_rgb1 (s)": ts1.map(lambda x: f"{x:.6f}"),
            "path_rgb1": "rgb_1/" + df1["name1"].astype(str),
        })

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
        """
        Build calibration.yaml using cam0/1 YAMLs. Also computes stereo rectification
        to produce R_l, R_r, P_l, P_r.
        """
        seq = self.dataset_path / sequence_name
        cam0_yaml = seq / "mav0" / "cam0" / "sensor.yaml"
        cam1_yaml = seq / "mav0" / "cam1" / "sensor.yaml"
        imu_yaml  = seq / "mav0" / "imu0" / "sensor.yaml"

        with open(cam0_yaml, "r", encoding="utf-8") as f: cam0 = yaml.safe_load(f)
        with open(cam1_yaml, "r", encoding="utf-8") as f: cam1 = yaml.safe_load(f)
        with open(imu_yaml,  "r", encoding="utf-8") as f: imu  = yaml.safe_load(f)

        # Camera dicts (include basic distortion for record-keeping)
        i0, d0 = cam0["intrinsics"], cam0["distortion_coefficients"]
        i1, d1 = cam1["intrinsics"], cam1["distortion_coefficients"]
        
        # Extrinsics T_BS (body->sensor). Build relative cam0->cam1 in body frame.
        T_BS0 = np.array(cam0["T_BS"]["data"], dtype=float).reshape(cam0["T_BS"]["rows"], cam0["T_BS"]["cols"])
        T_BS1 = np.array(cam1["T_BS"]["data"], dtype=float).reshape(cam1["T_BS"]["rows"], cam1["T_BS"]["cols"])
        R_SB0, t_SB0 = T_BS0[:3, :3], T_BS0[:3, 3]
        R_SB1, t_SB1 = T_BS1[:3, :3], T_BS1[:3, 3]
        R_01_B = R_SB1.T @ R_SB0
        t_01_B = R_SB1.T @ (t_SB0 - t_SB1)

        # Stereo rectification inputs
        K_l = np.array([[i0[0], 0.,     i0[2]],
                       [0.,     i0[1], i0[3]],
                       [0.,     0.,     1. ]], dtype=np.float64)
        K_r = np.array([[i1[0], 0.,     i1[2]],
                       [0.,     i1[1], i1[3]],
                       [0.,     0.,     1. ]], dtype=np.float64)
        D_l = np.array([d0[0], d0[1], d0[2], d0[3], 0.0], dtype=np.float64)
        D_r = np.array([d1[0], d1[1], d1[2], d1[3], 0.0], dtype=np.float64)

        w, h = cam0["resolution"]
        size = (int(w), int(h))

        R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, size, R_01_B.astype(np.float64), t_01_B.astype(np.float64),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=size
        )

        ##########################
        map_l = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
        map_r = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

        # Load rgb images
        df = pd.read_csv(seq / "rgb.csv")       
        images_left = df['path_rgb0'].to_list()
        images_right = df['path_rgb1'].to_list()

        for relL, relR in tqdm(zip(images_left, images_right), desc="Rectifying stereo pairs", 
                               total=min(len(images_left), len(images_right))):
            pL = seq / relL
            pR = seq / relR

            imgL_src = cv2.imread(str(pL), cv2.IMREAD_UNCHANGED)
            imgR_src = cv2.imread(str(pR), cv2.IMREAD_UNCHANGED)

            image_l = cv2.remap(imgL_src, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)
            image_r = cv2.remap(imgR_src, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)
        
            cv2.imwrite(str(pL), image_l)
            cv2.imwrite(str(pR), image_r)

        stereo = {"bf": STEREO_BASELINE}
        
        ##########################
        camera0 = {
            "model": "Pinhole",
            "fx": P_l[0,0], "fy": P_l[1,1], "cx": P_l[0,2], "cy": P_l[1,2]
        }
        camera1 = {
            "model": "Pinhole",
            "fx": P_r[0,0], "fy": P_r[1,1], "cx": P_r[0,2], "cy": P_r[1,2]
        }
        ##########################

        imu_out = {
            "transform": cam0["T_BS"]["data"],  # camera->IMU extrinsic as provided
            "gyro_noise": imu["gyroscope_noise_density"],
            "gyro_bias": imu["gyroscope_random_walk"],
            "accel_noise": imu["accelerometer_noise_density"],
            "accel_bias": imu["accelerometer_random_walk"],
            "frequency": imu["rate_hz"],
        }

        self.write_calibration_yaml(sequence_name, camera0=camera0, camera1=camera1, imu=imu_out, stereo=stereo)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        """
        Write groundtruth.csv from TUM 'supp_v2/gtFiles/mav_<sequence>.txt'.
        Input is seconds; we output integer nanoseconds (ts) for consistency with EUROC.
        """
        seq = self.dataset_path / sequence_name
        src = self.dataset_path / "supp_v2" / "gtFiles" / f"mav_{sequence_name}.txt"
        dst = seq / "groundtruth.csv"

        if not src.exists():
            return
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        tmp = dst.with_suffix(".csv.tmp")
        with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

            for line in fin:
                s = line.strip()
                if not s or "NaN" in s:
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 8:
                    continue
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts[:8]
                w.writerow([ts_s, tx, ty, tz, qx, qy, qz, qw])

        tmp.replace(dst)
        with suppress(FileNotFoundError):
            tmp.unlink()

    def remove_unused_files(self, sequence_name: str) -> None:
        """
        Remove bulky raw folders after we’ve produced benchmark-ready files.
        Comment out if you want to keep raw data for debugging.
        """
        seq = self.dataset_path / sequence_name
        for rel in ("mav0", "__MACOSX"):
            with suppress(FileNotFoundError):
                shutil.rmtree(seq / rel)

    def _download_url_for(self, sequence_name: str) -> str:
        """
        EUROC structure on server:
        - machine_hall/MH_xx_xxx/<zip>
        - vicon_room1/V1_xx_xxx/<zip>
        - vicon_room2/V2_xx_xxx/<zip>
        """
        if sequence_name.startswith("MH_"):
            sub = "machine_hall"
        elif sequence_name.startswith("V1_"):
            sub = "vicon_room1"
        elif sequence_name.startswith("V2_"):
            sub = "vicon_room2"
        else:
            raise ValueError(f"Unknown EUROC sequence prefix: {sequence_name}")

        # ensure trailing slash in root before urljoin
        root = self.url_download_root.rstrip("/") + "/"
        return urljoin(root, f"{sub}/{sequence_name}/{sequence_name}.zip")