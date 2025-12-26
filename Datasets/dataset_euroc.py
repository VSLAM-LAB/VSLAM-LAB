from __future__ import annotations

from pathlib import Path
from contextlib import suppress
import shutil

import csv
import yaml
from decimal import Decimal

import numpy as np
import pandas as pd

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION


class EUROC_dataset(DatasetVSLAMLab):
    """EUROC MAV dataset helper for VSLAM-LAB benchmark."""

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
                shutil.move(png , target / png.name)

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
        ts0 = df0["ts_ns"].astype(np.int64)
        ts1 = df1["ts_ns"].astype(np.int64)

        out = pd.DataFrame({
            "ts_rgb_0 (ns)": ts0,
            "path_rgb_0": "rgb_0/" + df0["name0"].astype(str),
            "ts_rgb_1 (ns)": ts1,
            "path_rgb_1": "rgb_1/" + df1["name1"].astype(str),
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
        dst = seq / "imu_0.csv"

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
    
        out = df[["timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]]

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
        cam0_yaml = sequence_path / "mav0" / "cam0" / "sensor.yaml"
        cam1_yaml = sequence_path / "mav0" / "cam1" / "sensor.yaml"
        imu_yaml  = sequence_path / "mav0" / "imu0" / "sensor.yaml"
        with open(cam0_yaml, "r", encoding="utf-8") as f: cam0 = yaml.safe_load(f)
        with open(cam1_yaml, "r", encoding="utf-8") as f: cam1 = yaml.safe_load(f)
        with open(imu_yaml,  "r", encoding="utf-8") as f: imu  = yaml.safe_load(f)

        rgb0 = {"cam_name": "rgb_0", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": cam0["intrinsics"][0:2], "principal_point": cam0["intrinsics"][2:4],
                "distortion_type": "radtan4", "distortion_coefficients": cam0["distortion_coefficients"],
                "fps": cam0["rate_hz"],
                "T_SC": np.array(cam0["T_BS"]['data']).reshape((4, 4))}
        rgb1 = {"cam_name": "rgb_1", "cam_type": "gray",
                "cam_model": "pinhole", "focal_length": cam1["intrinsics"][0:2], "principal_point": cam1["intrinsics"][2:4],
                "distortion_type": "radtan4", "distortion_coefficients": cam1["distortion_coefficients"],
                "fps": cam1["rate_hz"],
                "T_SC": np.array(cam1["T_BS"]['data']).reshape((4, 4))}
        
        imu = {"imu_name": "imu_0",
            "a_max":  176.0, "g_max": 7.8,
            "sigma_g_c":  20.0e-4, "sigma_a_c": 20.0e-3,
            "sigma_bg":  0.01, "sigma_ba":  0.1,
            "sigma_gw_c":  20.0e-5, "sigma_aw_c": 20.0e-3,
            "g":  9.81007, "g0": [ 0.0, 0.0, 0.0 ], "a0": [ -0.05, 0.09, 0.01 ],
            "s_a":  [ 1.0,  1.0, 1.0 ],
            "fps": 200.0,
            "T_SC": np.array(np.eye(4)).reshape((4, 4))}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=imu)
    
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
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])

            for line in fin:
                s = line.strip()
                if not s or "NaN" in s:
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 8:
                    continue
                ts_s, tx, ty, tz, qx, qy, qz, qw = parts[:8]
                ts_ns = int(Decimal(ts_s) * Decimal(10**9))
                w.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        tmp.replace(dst)
        with suppress(FileNotFoundError):
            tmp.unlink()

    def remove_unused_files(self, sequence_name: str) -> None:
        seq = self.dataset_path / sequence_name
        for rel in ("mav0", "__MACOSX"):
            with suppress(FileNotFoundError):
                shutil.rmtree(seq / rel)
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            (self.dataset_path / f"{sequence_name}.zip").unlink(missing_ok=True)

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
        import os
        return os.path.join(root, f"{sub}/{sequence_name}/{sequence_name}.zip")