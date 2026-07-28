"""
Module: VSLAM-LAB - Datasets - dataset_rover.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2025-09-05
- Updated: 2026-07-28
- License: GPLv3 License
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import BENCHMARK_RETENTION, Retention, VSLAMLAB_BENCHMARK
from utilities import decompressFile, downloadFile, write_csv_rows

TIME_DIFF_THRESH: Final = 0.02  # seconds for RGB/Depth association


class RoverDataset(DatasetVSLAMLAB):
    """ROVER multiseason dataset helper for VSLAM-LAB benchmark."""

    DATES = {
        "campus_small": {
            "autumn": "2023-11-23",
            "winter": "2024-02-19",
            "spring": "2024-04-14",
            "summer": "2023-09-11",
            "day": "2024-05-07",
            "dusk": "2024-05-08_1",
            "night": "2024-05-08_2",
            "night-light": "2024-05-24_1",
        },
        "campus_large": {
            "autumn": "2023-11-07",
            "winter": "2024-01-27",
            "spring": "2024-04-14",
            "summer": "2023-07-20",
            "day": "2024-09-25",
            "dusk": "2024-09-24_2",
            "night": "2024-09-24_3",
            "night-light": "2024-09-24_4",
        },
        "garden_small": {
            "autumn": "2023-09-15",
            "winter": "2024-01-13",
            "spring": "2024-04-11",
            "summer": "2023-08-18",
            "day": "2024-05-29_1",
            "dusk": "2024-05-29_2",
            "night": "2024-05-29_3",
            "night-light": "2024-05-29_4",
        },
        "garden_large": {
            "autumn": "2023-12-21",
            "winter": "2024-01-13",
            "spring": "2024-04-11",
            "summer": "2023-08-18",
            "day": "2024-05-29_1",
            "dusk": "2024-05-29_2",
            "night": "2024-05-30_1",
            "night-light": "2024-05-30_2",
        },
        "park": {
            "autumn": "2023-11-07",
            "spring": "2024-04-14",
            "summer": "2023-07-31",
            "day": "2024-05-08",
            "dusk": "2024-05-13_1",
            "night": "2024-05-13_2",
            "night-light": "2024-05-24_2",
            # Note: no winter data for "park"
        },
    }
    
    SENSOR_NICKNAMES = {"picam": "pi_cam", "d435i": "realsense_D435i", "t265": "realsense_T265", "vn100": "vn100"}

    def __init__(self, dataset_name: str = "rover") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.url_download_root: str = self.cfg["url_download_root"]
        self.master_calibration_path = VSLAMLAB_BENCHMARK / 'ROVER' / "calibration"

    def download_sequence_data(self, sequence_name: str) -> None:
        location, setting, sensor, date = self._sequence_data_from_name(sequence_name)
        resource_name = "_".join([location, date])
        sequence_path = self.sequence_path(sequence_name).resolve()
        sequence_group_path = self._sequence_group_path(sequence_name)
        sequence_subdir = sequence_group_path / self.SENSOR_NICKNAMES[sensor]

        def flatten_subdir():
            # $datapath/ROVER/sequence_id_with_date/date -> $datapath/ROVER/sequence_id_with_date
            sequence_group_path_child = sequence_group_path / date
            if not sequence_group_path_child.is_dir(): return
            for src_item in sequence_group_path_child.iterdir():
                src_item = sequence_group_path_child / src_item.name
                dst_item = sequence_group_path / src_item.name
                shutil.move(src_item, dst_item)
            sequence_group_path_child.rmdir()

        self._ensure_data_exists(
            data_path = sequence_group_path,
            target=resource_name, target_path=sequence_group_path,
            callback=flatten_subdir
        )

        if not sequence_subdir.is_dir():
            raise FileNotFoundError(f"Source directory for symlink not found after decompression: {sequence_subdir}")

        if not sequence_path.exists():
            os.symlink(sequence_subdir, sequence_path)
   
    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        src = sequence_path / "imu" /  "imu.txt"
        dst = self.imu_csv_path(sequence_name)
        if not src.exists():
            return

        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return
        
        temp_cols = ["ts", "ax", "ay", "az", "wx", "wy", "wz"]
        df = pd.read_csv(src, comment="#", header=None, names=temp_cols, sep=r"[\s,]+", engine="python")

        if df.empty:
            return

        df["ts (ns)"] = (df["ts"] * 1e9).astype(int)
        new_cols = ["ts (ns)", "wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)"]
        out = df.rename(columns={
        "wx": "wx (rad s^-1)", "wy": "wy (rad s^-1)", "wz": "wz (rad s^-1)",
        "ax": "ax (m s^-2)", "ay": "ay (m s^-2)", "az": "az (m s^-2)"
        })[new_cols]

        # .astype(object) before .values - the ns timestamp column is int64 and every other
        # column is float64; a plain .values would upcast the whole array to float64, silently
        # losing precision past 2^53 in the timestamp (same bug class fixed for euroc/madmax).
        write_csv_rows(dst, new_cols, out.astype(object).values.tolist())
  
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        gt_src = self._sequence_group_path(sequence_name) / "groundtruth.txt"
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]

        rows = []
        if gt_src.exists():
            with open(gt_src, "r") as src:
                for line in src:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split(" ")
                    if len(parts) < 8:
                        continue

                    ns_timestamp = int(float(parts[0]) * 1e9)
                    parts[0] = str(ns_timestamp)
                    rows.append(parts)

        write_csv_rows(self.groundtruth_csv_path(sequence_name), header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        if BENCHMARK_RETENTION != Retention.FULL:
            # imu/imu.txt lives inside this sequence's own sensor subfolder (not shared with
            # sibling sequences from other registered sensor classes in the same group) - safe
            # to delete once imu_0.csv exists, a pure reformat with no information loss.
            #
            # groundtruth.txt is deliberately NOT deleted here, even though it's also a pure
            # reformat once groundtruth.csv exists: it lives at the shared group root, read by
            # every sibling sensor sequence (rover-t265/d435i/picam) from the same
            # (location, setting, date) group via _sequence_group_path(). There is no way for
            # one sequence's remove_unused_files to know whether those siblings have already
            # run their own create_groundtruth_csv, and the group-level on-demand-redownload
            # fallback doesn't cover this case either (it only re-triggers if the whole group
            # folder and its completion marker are gone, not if a single file inside an
            # otherwise-intact group folder is missing) - deleting it here risks silently
            # leaving an unprocessed sibling with an empty groundtruth.csv instead of real data.
            imu_txt = self.sequence_path(sequence_name) / "imu" / "imu.txt"
            imu_txt.unlink(missing_ok=True)

        if BENCHMARK_RETENTION == Retention.MINIMAL:
            # Both archives are only ever read once, during their own marker-gated extraction
            # in _ensure_data_exists - safe to delete afterward regardless of processing order,
            # and idempotent if multiple sibling sequences sharing the same group each try.
            # The *decompressed* group folder / master_calibration_path are never touched here:
            # every sensor's rgb_0/rgb_1/depth_0 data physically lives inside the group folder
            # (moved or symlinked in place, never copied elsewhere), and every future sequence's
            # create_calibration_yaml re-reads master_calibration_path on every call.
            location, _, _, date = self._sequence_data_from_name(sequence_name)
            resource_name = "_".join([location, date])
            rover_folder = VSLAMLAB_BENCHMARK / 'ROVER'
            (rover_folder / f"{resource_name}.zip").unlink(missing_ok=True)
            (rover_folder / "calibration.zip").unlink(missing_ok=True)


    def _ensure_data_exists(self, data_path: Path, target: str, target_path: Path, callback=None) -> None:
        # A plain data_path.exists() check can't tell "fully extracted" apart from "crashed
        # partway through decompression" - a completion marker (touched only once decompression
        # and callback have both finished) makes retries after a crash actually re-run instead of
        # being silently treated as already done.
        marker = data_path / ".download_complete"
        if marker.exists():
            return

        archive_name = target + ".zip"
        download_url = f"{self.url_download_root}/{archive_name}"
        rover_folder = VSLAMLAB_BENCHMARK / 'ROVER'
        rover_folder.mkdir(parents=True, exist_ok=True)
        archive_path = rover_folder / archive_name
        if not archive_path.exists():
            downloadFile(download_url, rover_folder)
        decompressFile(archive_path, target_path)

        if callback:
            callback()

        data_path.mkdir(parents=True, exist_ok=True)
        marker.touch()

    def _sequence_group_path(self, sequence_name: str) -> Path:
        location, setting, _, date = self._sequence_data_from_name(sequence_name)
        sequence_group_name = "_".join([location, setting, date])
        return (VSLAMLAB_BENCHMARK / 'ROVER' / sequence_group_name).resolve()

    def _sequence_data_from_name(self, sequence_name: str):
        for location_candidate in self.DATES:
            if sequence_name.startswith(location_candidate):
                location = location_candidate
                break
        else:
            raise ValueError(f"Unknown location in sequence name: {sequence_name!r}")

        remainder = sequence_name[len(location) + 1:]
        setting, sensor = remainder.split("_")
        date = self.DATES[location][setting]
        return location, setting, sensor, date

    def _download_master_calibration_archive(self):
        macosx_path = self.dataset_path / "__MACOSX"
        cleanup_macosx = lambda: shutil.rmtree(macosx_path) if macosx_path.exists() else None
        self._ensure_data_exists(
            data_path = self.master_calibration_path,
            target="calibration", target_path = VSLAMLAB_BENCHMARK / 'ROVER',
            callback=cleanup_macosx)
        

class RoverT265Dataset(RoverDataset):    
    """ROVER T265 multiseason dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "rover-t265") -> None:
        super().__init__(dataset_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for src, tgt in (
            (sequence_path / "cam_left", self.rgb_path(sequence_name)),
            (sequence_path / "cam_right", sequence_path / "rgb_1"),
        ):
            if src.is_dir() and not tgt.exists():
                os.symlink(src, tgt)
       
    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        cols = ['ts', 'path']
        df_left = pd.read_csv(sequence_path / "cam_left.txt", sep=' ', header=None, names=cols)
        df_right = pd.read_csv(sequence_path / "cam_right.txt", sep=' ', header=None, names=cols)
        # Rebuilt from each row's filename rather than assuming a "rgb/"/"depth/" raw prefix -
        # cam_right.txt's raw paths don't actually carry a "depth/" prefix (T265 has no depth
        # sensor; that was a copy-paste leftover from the D435i script), so this is robust
        # regardless of whatever prefix (or none) the raw metadata file actually uses.
        df_left['path'] = 'rgb_0/' + df_left['path'].apply(lambda p: Path(p).name)
        df_right['path'] = 'rgb_1/' + df_right['path'].apply(lambda p: Path(p).name)
        df_left['ts'] = (df_left['ts'] * 1e9).round().astype('int64')
        df_right['ts'] = (df_right['ts'] * 1e9).round().astype('int64')
        combined = pd.concat([
            df_left['ts'], df_left['path'],
            df_right['ts'], df_right['path']
        ], axis=1)
        combined.columns = ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1']
        write_csv_rows(rgb_csv, list(combined.columns), combined.values.tolist())
       
    def create_calibration_yaml(self, sequence_name: str) -> None:
        self._download_master_calibration_archive()
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        calibration_file = self.master_calibration_path / f"calib_{sensor}.yaml"
        with open(calibration_file, "r") as file:
            data = yaml.safe_load(file)
            
        cam_data_l = data["CamLeft_Intrinsics"]
        cam_data_r = data["CamRight_Intrinsics"]
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam_data_l["intrinsics"][0:2],
            "principal_point": cam_data_l["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam_data_l["distortion_coeffs"],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(np.array(data["IMU-To-CamLeft"]).reshape((4, 4))),
        }
        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": cam_data_r["intrinsics"][0:2],
            "principal_point": cam_data_r["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam_data_r["distortion_coeffs"],
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(np.array(data["IMU-To-CamRight"]).reshape((4, 4))),
        }

        imu_intrinsics_data = data["IMU_Intrinsics"]
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": imu_intrinsics_data["noise_gyro"],
            "sigma_a_c": imu_intrinsics_data["noise_acc"],
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": imu_intrinsics_data["walk_gyro"],
            "sigma_aw_c": imu_intrinsics_data["walk_acc"],
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 200.0,
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])


class RoverD435iDataset(RoverDataset):    
    """ROVER D435i multiseason dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "rover-d435i") -> None:
        super().__init__(dataset_name)

        # Depth factor
        self.depth_factor = self.cfg["depth_factor"]

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        for src, tgt in (
            (sequence_path / "rgb", self.rgb_path(sequence_name)),
            (sequence_path / "depth", self.depth_path(sequence_name)),
        ):
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        rgb_txt = sequence_path / 'rgb.txt'
        depth_txt = sequence_path / 'depth.txt'
        # Load monotonically sorted timestamps
        rgb = pd.read_csv(rgb_txt, sep=r"\s+", comment="#", header=None, names=["ts", "rgb_path"])
        depth = pd.read_csv(depth_txt, sep=r"\s+", comment="#", header=None, names=["ts", "depth_path"])
        rgb = rgb.sort_values("ts").reset_index(drop=True)
        depth = depth.sort_values("ts").reset_index(drop=True)
        # As-of merge finds nearest earlier match; we do symmetric by duplicating with reversed
        # but here TUM is dense and ordered, so a forward asof then post-check tolerance works well.
        merged = pd.merge_asof(rgb, depth, on="ts", direction="nearest", tolerance=TIME_DIFF_THRESH)
        merged = merged.dropna(subset=["depth_path"]).copy()
        # Format + path prefix fixes
        merged["ts_rgb_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["ts_depth_0 (ns)"] = (merged["ts"] * 1e9).astype(int)
        merged["path_rgb_0"] = merged["rgb_path"].astype(str).str.replace(r"^rgb/", "rgb_0/", regex=True)
        merged["path_depth_0"] = merged["depth_path"].astype(str).str.replace(r"^depth/", "depth_0/", regex=True)
        out = merged[["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"]]
        write_csv_rows(rgb_csv, list(out.columns), out.values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        self._download_master_calibration_archive()
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        calibration_file = self.master_calibration_path / f"calib_{sensor}.yaml"
        with open(calibration_file, "r") as file:
            data = yaml.safe_load(file)
            
        cam_intrinsics_data = data["Cam_Intrinsics"]
        fx, fy, cx, cy = cam_intrinsics_data["intrinsics"]
        k1, k2, p1, p2 = cam_intrinsics_data["distortion_coeffs"]
        rgbd0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": [fx, fy],
            "principal_point": [cx, cy],
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.linalg.inv(np.array(data["IMU-To-Cam"]).reshape((4, 4))),
            "distortion_type": "radtan4",
            "distortion_coefficients": [k1, k2, p1, p2],
        }

        imu_intrinsics_data = data["IMU_Intrinsics"]
        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": imu_intrinsics_data["noise_gyro"],
            "sigma_a_c": imu_intrinsics_data["noise_acc"],
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": imu_intrinsics_data["walk_gyro"],
            "sigma_aw_c": imu_intrinsics_data["walk_acc"],
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 200.0,
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0], imu=[imu])


class RoverPicamDataset(RoverDataset):    
    """ROVER Picam multiseason dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "rover-picam") -> None:
        super().__init__(dataset_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        src = self.sequence_path(sequence_name) / 'rgb'
        tgt = self.rgb_path(sequence_name)
        if src.is_dir() and not tgt.exists():
            os.symlink(src, tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rgb_csv = self.rgb_csv_path(sequence_name)
        cols = ['ts', 'path']
        df = pd.read_csv(sequence_path / "rgb.txt", sep=' ', header=None, names=cols)
        df['path'] = df['path'].str.replace('rgb/', 'rgb_0/', regex=False)
        df['ts'] = (df['ts'] * 1e9).round().astype('int64')
        df.columns = ['ts_rgb_0 (ns)', 'path_rgb_0']
        write_csv_rows(rgb_csv, list(df.columns), df.values.tolist())

    def create_imu_csv(self, sequence_name: str) -> None:
        pass

    def create_calibration_yaml(self, sequence_name: str) -> None:
        self._download_master_calibration_archive()
        _, _, sensor, _ = self._sequence_data_from_name(sequence_name)
        calibration_file = self.master_calibration_path / f"calib_{sensor}.yaml"
        with open(calibration_file, "r") as file:
            data = yaml.safe_load(file)
            
        cam_data = data["Cam_Intrinsics"]
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "pinhole",
            "focal_length": cam_data["intrinsics"][0:2],
            "principal_point": cam_data["intrinsics"][2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam_data["distortion_coeffs"],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])
