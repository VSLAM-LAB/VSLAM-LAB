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


class FOURSEASONS_dataset(DatasetVSLAMLab):
    """4Seasons dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "4seasons") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        self.url_download_root: str = self.cfg_require("url_download_root")
        self.recordings: dict[str, str] = self.cfg_require("recordings")
        self.with_reference_poses: list[str] = list(self.cfg_get("with_reference_poses", []))
        self.imu_hz: float = float(self.cfg_get("imu_hz", 200.0))
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        rec = self._recording(sequence_name)
        sequence_path = self.dataset_path / sequence_name
        if sequence_path.exists():
            return

        # Download sequence assets (IMU + undistorted stereo + optional reference poses).
        assets = [
            f"{rec}_imu_gnss.zip",
            f"{rec}_stereo_images_undistorted.zip",
        ]
        if sequence_name in self.with_reference_poses:
            assets.append(f"{rec}_reference_poses.zip")

        for asset in assets:
            url = urljoin(self.url_download_root.rstrip("/") + "/", f"{rec}/{asset}")
            archive = self.dataset_path / asset
            if not archive.exists():
                downloadFile(url, str(self.dataset_path))
            decompressFile(str(archive), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        sequence_path.mkdir(parents=True, exist_ok=True)

        rec = self._recording(sequence_name)
        root = self._find_recording_root(rec)
        cam0_src, cam1_src = self._find_stereo_dirs(root)

        rgb0 = sequence_path / "rgb_0"
        rgb1 = sequence_path / "rgb_1"

        if not rgb0.exists():
            rgb0.symlink_to(cam0_src, target_is_directory=True)
        if not rgb1.exists():
            rgb1.symlink_to(cam1_src, target_is_directory=True)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        rgb0 = sequence_path / "rgb_0"
        rgb1 = sequence_path / "rgb_1"
        files0 = sorted([p.name for p in rgb0.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        files1 = sorted([p.name for p in rgb1.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

        # Use common filenames when available; otherwise zip by index.
        common = sorted(set(files0).intersection(files1))
        rows: list[tuple[int, str, int, str]] = []
        if common:
            for name in common:
                ts_ns = self._timestamp_from_filename(name)
                rows.append((ts_ns, f"rgb_0/{name}", ts_ns, f"rgb_1/{name}"))
        else:
            n = min(len(files0), len(files1))
            for i in range(n):
                ts0 = self._timestamp_from_filename(files0[i])
                ts1 = self._timestamp_from_filename(files1[i])
                rows.append((ts0, f"rgb_0/{files0[i]}", ts1, f"rgb_1/{files1[i]}"))

        tmp = rgb_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"])
            w.writerows(rows)
        tmp.replace(rgb_csv)

    def create_imu_csv(self, sequence_name: str) -> None:
        rec = self._recording(sequence_name)
        root = self._find_recording_root(rec)
        src = self._find_imu_csv(root)
        dst = self.dataset_path / sequence_name / "imu_0.csv"

        if not src.exists():
            return
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return

        # Parse numeric rows robustly from comma/space separated logs.
        df = pd.read_csv(src, comment="#", header=None, sep=r"[\s,]+", engine="python")
        if df.empty or df.shape[1] < 7:
            return
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(subset=[0, 1, 2, 3, 4, 5, 6])
        if num.empty:
            return

        ts = num.iloc[:, 0].astype(np.float64)
        ts_ns = ts.astype(np.int64) if ts.median() > 1e12 else (ts * 1e9).astype(np.int64)
        out = pd.DataFrame(
            {
                "ts (ns)": ts_ns,
                "wx (rad s^-1)": num.iloc[:, 1],
                "wy (rad s^-1)": num.iloc[:, 2],
                "wz (rad s^-1)": num.iloc[:, 3],
                "ax (m s^-2)": num.iloc[:, 4],
                "ay (m s^-2)": num.iloc[:, 5],
                "az (m s^-2)": num.iloc[:, 6],
            }
        )
        tmp = dst.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(dst)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        rec = self._recording(sequence_name)
        root = self._find_recording_root(rec)
        cam0, cam1 = self._load_camera_params(root)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": cam0.get("cam_model", "pinhole"),
            "focal_length": cam0["focal_length"],
            "principal_point": cam0["principal_point"],
            "fps": float(self.rgb_hz),
            "T_BS": cam0.get("T_BS", np.eye(4)),
        }
        if "distortion_type" in cam0:
            rgb0["distortion_type"] = cam0["distortion_type"]
            rgb0["distortion_coefficients"] = cam0.get("distortion_coefficients", [])

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": cam1.get("cam_model", "pinhole"),
            "focal_length": cam1["focal_length"],
            "principal_point": cam1["principal_point"],
            "fps": float(self.rgb_hz),
            "T_BS": cam1.get("T_BS", np.eye(4)),
        }
        if "distortion_type" in cam1:
            rgb1["distortion_type"] = cam1["distortion_type"]
            rgb1["distortion_coefficients"] = cam1.get("distortion_coefficients", [])

        imu0: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": 2.0e-4,
            "sigma_a_c": 2.0e-3,
            "sigma_bg": 1.0e-2,
            "sigma_ba": 1.0e-1,
            "sigma_gw_c": 2.0e-5,
            "sigma_aw_c": 2.0e-3,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(self.imu_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if sequence_name not in self.with_reference_poses:
            return

        rec = self._recording(sequence_name)
        root = self._find_recording_root(rec)
        src = self._find_reference_pose_csv(root)
        if src is None:
            return

        dst = self.dataset_path / sequence_name / "groundtruth.csv"
        tmp = dst.with_suffix(".csv.tmp")

        df = pd.read_csv(src, comment="#")
        # Try named columns first.
        col_map = {c.lower().strip(): c for c in df.columns}

        def col(*names: str) -> str | None:
            for n in names:
                if n in col_map:
                    return col_map[n]
            return None

        ts_c = col("timestamp", "ts", "time", "timestamp_ns", "t")
        tx_c = col("tx", "px", "x")
        ty_c = col("ty", "py", "y")
        tz_c = col("tz", "pz", "z")
        qx_c = col("qx")
        qy_c = col("qy")
        qz_c = col("qz")
        qw_c = col("qw")

        out_rows: list[list[Any]] = []
        if all(v is not None for v in (ts_c, tx_c, ty_c, tz_c, qx_c, qy_c, qz_c, qw_c)):
            for _, r in df.iterrows():
                vals = [r[ts_c], r[tx_c], r[ty_c], r[tz_c], r[qx_c], r[qy_c], r[qz_c], r[qw_c]]
                if any(pd.isna(v) for v in vals):
                    continue
                ts = float(vals[0])
                ts_ns = int(ts if ts > 1e12 else ts * 1e9)
                out_rows.append([ts_ns, float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])])
        else:
            # Fallback numeric parse.
            num = df.apply(pd.to_numeric, errors="coerce")
            num = num.dropna()
            for _, r in num.iterrows():
                if len(r) < 8:
                    continue
                ts = float(r.iloc[0])
                ts_ns = int(ts if ts > 1e12 else ts * 1e9)
                out_rows.append([ts_ns, float(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4]), float(r.iloc[5]), float(r.iloc[6]), float(r.iloc[7])])

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            w.writerows(out_rows)
        tmp.replace(dst)

    def remove_unused_files(self, sequence_name: str) -> None:
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            rec = self._recording(sequence_name)
            for suffix in ("_imu_gnss.zip", "_stereo_images_undistorted.zip", "_reference_poses.zip"):
                (self.dataset_path / f"{rec}{suffix}").unlink(missing_ok=True)

    def _recording(self, sequence_name: str) -> str:
        if sequence_name not in self.recordings:
            raise KeyError(f"Unknown 4seasons sequence '{sequence_name}'")
        return self.recordings[sequence_name]

    def _find_recording_root(self, recording: str) -> Path:
        direct = self.dataset_path / recording
        if direct.exists():
            return direct
        matches = [p for p in self.dataset_path.rglob(recording) if p.is_dir()]
        if matches:
            return matches[0]
        return self.dataset_path

    @staticmethod
    def _timestamp_from_filename(filename: str) -> int:
        stem = Path(filename).stem
        try:
            ts = float(stem)
            return int(ts if ts > 1e12 else ts * 1e9)
        except ValueError:
            # deterministic fallback if filenames are not timestamp-based
            return 0

    @staticmethod
    def _find_stereo_dirs(root: Path) -> tuple[Path, Path]:
        candidates0 = [p for p in root.rglob("*") if p.is_dir() and p.name in {"cam0", "left"}]
        candidates1 = [p for p in root.rglob("*") if p.is_dir() and p.name in {"cam1", "right"}]
        if not candidates0 or not candidates1:
            raise FileNotFoundError(f"Could not locate stereo camera folders under {root}")
        # Prefer folders with image files.
        def score(p: Path) -> int:
            return sum(1 for _ in p.glob("*.png")) + sum(1 for _ in p.glob("*.jpg"))
        cam0 = sorted(candidates0, key=score, reverse=True)[0]
        cam1 = sorted(candidates1, key=score, reverse=True)[0]
        return cam0, cam1

    @staticmethod
    def _find_imu_csv(root: Path) -> Path:
        candidates = []
        for pattern in ("*imu*.csv", "*imu*.txt"):
            candidates.extend([p for p in root.rglob(pattern) if p.is_file()])
        if not candidates:
            raise FileNotFoundError(f"Could not find IMU CSV under {root}")
        # Prefer explicit imu_gnss file names.
        candidates.sort(key=lambda p: (0 if "imu_gnss" in p.name else 1, len(str(p))))
        return candidates[0]

    @staticmethod
    def _find_reference_pose_csv(root: Path) -> Path | None:
        patterns = ("*reference*pose*.csv", "*reference*poses*.txt", "*poses*.csv")
        for pat in patterns:
            matches = [p for p in root.rglob(pat) if p.is_file()]
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _load_camera_params(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        cam0_yaml = next(iter([p for p in root.rglob("cam0/sensor.yaml") if p.is_file()]), None)
        cam1_yaml = next(iter([p for p in root.rglob("cam1/sensor.yaml") if p.is_file()]), None)
        if cam0_yaml and cam1_yaml:
            with open(cam0_yaml, "r", encoding="utf-8") as f:
                c0 = yaml.safe_load(f) or {}
            with open(cam1_yaml, "r", encoding="utf-8") as f:
                c1 = yaml.safe_load(f) or {}
            return FOURSEASONS_dataset._camera_from_sensor_yaml(c0), FOURSEASONS_dataset._camera_from_sensor_yaml(c1)

        camchain = next(iter([p for p in root.rglob("*camchain*.yaml") if p.is_file()]), None)
        if camchain:
            with open(camchain, "r", encoding="utf-8") as f:
                cc = yaml.safe_load(f) or {}
            c0 = cc.get("cam0", {})
            c1 = cc.get("cam1", {})
            return FOURSEASONS_dataset._camera_from_camchain(c0), FOURSEASONS_dataset._camera_from_camchain(c1)

        # Last-resort defaults for undistorted stereo pinhole.
        dflt = {
            "cam_model": "pinhole",
            "focal_length": [320.0, 320.0],
            "principal_point": [320.0, 240.0],
            "T_BS": np.eye(4),
        }
        return dflt, dflt.copy()

    @staticmethod
    def _camera_from_sensor_yaml(cam: dict[str, Any]) -> dict[str, Any]:
        intr = cam.get("intrinsics", [320.0, 320.0, 320.0, 240.0])
        out: dict[str, Any] = {
            "cam_model": "pinhole",
            "focal_length": [float(intr[0]), float(intr[1])],
            "principal_point": [float(intr[2]), float(intr[3])],
            "T_BS": np.eye(4),
        }
        if "distortion_coefficients" in cam:
            out["distortion_type"] = "radtan4"
            out["distortion_coefficients"] = [float(x) for x in cam["distortion_coefficients"][:4]]
        tbs = cam.get("T_BS", {}).get("data")
        if tbs:
            out["T_BS"] = np.array(tbs, dtype=float).reshape((4, 4))
        return out

    @staticmethod
    def _camera_from_camchain(cam: dict[str, Any]) -> dict[str, Any]:
        intr = cam.get("intrinsics", [320.0, 320.0, 320.0, 240.0])
        out: dict[str, Any] = {
            "cam_model": "pinhole",
            "focal_length": [float(intr[0]), float(intr[1])],
            "principal_point": [float(intr[2]), float(intr[3])],
            "T_BS": np.array(cam.get("T_cam_imu", np.eye(4)), dtype=float),
        }
        dist = cam.get("distortion_coeffs")
        if dist:
            out["distortion_type"] = "equid4"
            out["distortion_coefficients"] = [float(x) for x in dist[:4]]
        return out
