from __future__ import annotations

import csv
import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import decompressFile, downloadFile


class VITUM_dataset(DatasetVSLAMLab):
    """TUM VI dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "vitum") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        self.url_download_root: str = self.cfg_require("url_download_root")
        self.imu_hz: float = float(self.cfg_get("imu_hz", 200.0))
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:
        archive_name = f"dataset-{sequence_name}_512_16.tar"
        archive_url = urljoin(self.url_download_root.rstrip("/") + "/", archive_name)

        archive_path = self.dataset_path / archive_name
        extracted_root = self._source_root(sequence_name)

        if not archive_path.exists():
            downloadFile(archive_url, str(self.dataset_path))

        if not extracted_root.exists():
            decompressFile(str(archive_path), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        if rgb_path.exists():
            return

        src_dir = self._source_root(sequence_name) / "mav0" / "cam0" / "data"
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing cam0 data folder: {src_dir}")

        rgb_path.mkdir(parents=True, exist_ok=True)
        for img in sorted(src_dir.glob("*.png")):
            shutil.copy2(img, rgb_path / img.name)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        ts_by_stem = self._load_times_map(sequence_name)
        rgb_files = sorted([p.name for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"])

        tmp = rgb_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            for filename in rgb_files:
                stem = Path(filename).stem
                if stem in ts_by_stem:
                    ts_ns = int(ts_by_stem[stem])
                else:
                    # Fallback: filenames are usually nanosecond timestamps.
                    ts_ns = int(stem)
                w.writerow([ts_ns, f"rgb_0/{filename}"])
        tmp.replace(rgb_csv)

    def create_imu_csv(self, sequence_name: str) -> None:
        src = self._source_root(sequence_name) / "mav0" / "imu0" / "data.csv"
        dst = self.dataset_path / sequence_name / "imu_0.csv"
        if not src.exists():
            return
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        cols = [
            "timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ]
        df = pd.read_csv(src, comment="#", header=None, names=cols, sep=r"[\s,]+", engine="python")
        if df.empty:
            return

        out = pd.DataFrame(
            {
                "ts (ns)": df["timestamp [ns]"].astype(np.int64),
                "wx (rad s^-1)": df["w_RS_S_x [rad s^-1]"],
                "wy (rad s^-1)": df["w_RS_S_y [rad s^-1]"],
                "wz (rad s^-1)": df["w_RS_S_z [rad s^-1]"],
                "ax (m s^-2)": df["a_RS_S_x [m s^-2]"],
                "ay (m s^-2)": df["a_RS_S_y [m s^-2]"],
                "az (m s^-2)": df["a_RS_S_z [m s^-2]"],
            }
        )

        tmp = dst.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(dst)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        src_root = self._source_root(sequence_name)

        camchain = src_root / "dso" / "camchain.yaml"
        imu_cfg = src_root / "dso" / "imu_config.yaml"
        cam_sensor = src_root / "mav0" / "cam0" / "sensor.yaml"

        intrinsics = [190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504]
        distortion = [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182]
        T_cam_imu = np.eye(4)

        if camchain.exists():
            with open(camchain, "r", encoding="utf-8") as f:
                cam_data = yaml.safe_load(f) or {}
            cam0 = cam_data.get("cam0", {})
            intrinsics = cam0.get("intrinsics", intrinsics)
            distortion = cam0.get("distortion_coeffs", distortion)
            T_cam_imu = np.array(cam0.get("T_cam_imu", np.eye(4)), dtype=float)
        elif cam_sensor.exists():
            with open(cam_sensor, "r", encoding="utf-8") as f:
                cam_data = yaml.safe_load(f) or {}
            intrinsics = cam_data.get("intrinsics", intrinsics)
            distortion = cam_data.get("distortion_coefficients", distortion)
            T_raw = cam_data.get("T_BS", {}).get("data")
            if T_raw:
                T_cam_imu = np.array(T_raw, dtype=float).reshape((4, 4))

        imu_noise = {
            "gyro_noise_density": 1.7e-4,
            "gyroscope_random_walk": 1.9393e-5,
            "accelerometer_noise_density": 2.0e-3,
            "accelerometer_random_walk": 3.0e-3,
            "update_rate": self.imu_hz,
        }
        if imu_cfg.exists():
            with open(imu_cfg, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            imu_noise.update(loaded)

        fx, fy, cx, cy = [float(x) for x in intrinsics[:4]]
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "equid4",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "distortion_type": "equid4",
            "distortion_coefficients": [float(x) for x in distortion[:4]],
            "fps": float(self.rgb_hz),
            "T_BS": T_cam_imu,
        }

        imu0: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": float(imu_noise["gyro_noise_density"]),
            "sigma_a_c": float(imu_noise["accelerometer_noise_density"]),
            "sigma_bg": float(imu_noise["gyroscope_random_walk"]),
            "sigma_ba": float(imu_noise["accelerometer_random_walk"]),
            "sigma_gw_c": float(imu_noise["gyroscope_random_walk"]),
            "sigma_aw_c": float(imu_noise["accelerometer_random_walk"]),
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(imu_noise.get("update_rate", self.imu_hz)),
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0], imu=[imu0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        src = self._source_root(sequence_name) / "dso" / "gt_imu.csv"
        dst = self.dataset_path / sequence_name / "groundtruth.csv"
        if not src.exists():
            return

        tmp = dst.with_suffix(".csv.tmp")
        with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [p for p in re.split(r"[\s,]+", s) if p]
                if len(parts) < 8:
                    continue
                try:
                    vals = [float(v) for v in parts[:8]]
                except ValueError:
                    continue
                if any(np.isnan(v) for v in vals):
                    continue

                ts = vals[0]
                ts_ns = int(ts * 1e9) if ts < 1e12 else int(ts)
                tx, ty, tz = vals[1], vals[2], vals[3]
                qw, qx, qy, qz = vals[4], vals[5], vals[6], vals[7]
                w.writerow([ts_ns, tx, ty, tz, qx, qy, qz, qw])

        tmp.replace(dst)

    def remove_unused_files(self, sequence_name: str) -> None:
        source_root = self._source_root(sequence_name)
        if BENCHMARK_RETENTION != Retention.FULL:
            shutil.rmtree(source_root, ignore_errors=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            archive = self.dataset_path / f"dataset-{sequence_name}_512_16.tar"
            archive.unlink(missing_ok=True)

    def _source_root(self, sequence_name: str) -> Path:
        return self.dataset_path / f"dataset-{sequence_name}_512_16"

    def _load_times_map(self, sequence_name: str) -> dict[str, int]:
        times_file = self._source_root(sequence_name) / "dso" / "cam0" / "times.txt"
        out: dict[str, int] = {}
        if not times_file.exists():
            return out

        with open(times_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                cols = s.split()
                if len(cols) < 2:
                    continue
                stem = cols[0].split(".")[0]
                try:
                    ts_ns = int(float(cols[1]) * 1e9)
                except ValueError:
                    continue
                out[stem] = ts_ns
        return out
