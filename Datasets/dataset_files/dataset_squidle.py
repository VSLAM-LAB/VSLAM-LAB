"""
Module: VSLAM-LAB - Datasets - dataset_squidle.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-01-03
- Updated: 2026-07-26
- License: GPLv3 License
"""

from __future__ import annotations

import datetime
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import requests
import utm
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import BENCHMARK_RETENTION, Retention
from utilities import compute_scaled_size, make_printers, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)
CAMPAIGNS: Final = {
    "ssk16": "ssk16-01",
    "ssk17": "ssk17-01",
    "ssk18": "ssk18-01",
}

DEPLOYMENT_IDS: Final = {
    "scottreef15_01": 232,
    "scottreef11_01": 224,
    "scottreef15_02": 238,
    "scottreef11_02": 214,
}

ORIGIN_UTM: Final = {
    "ssk16": (387124.51475913724, 2950359.888579014),
    "ssk17": (387124.51475913724, 2950359.888579014),
    "ssk18": (387124.51475913724, 2950359.888579014),
    "scottreef15_01": (374098.3723509629, 8438570.03130037),
    "scottreef11_01": (374098.3723509629, 8438570.03130037),
    "scottreef15_02": (387124.51475913724, 2950359.888579014),
    "scottreef11_02": (387124.51475913724, 2950359.888579014),
}

ORIGIN_ZONE: Final = {
    "ssk16": (52, 'R'),
    "ssk17": (52, 'R'),
    "ssk18": (52, 'R'),
    "scottreef15_01": (51, 'L'),
    "scottreef11_01": (51, 'L'),
    "scottreef15_02": (52, 'R'),
    "scottreef11_02": (52, 'R'),
}

IMAGE_CROP: Final = {
    "ssk16": [146, 3],
    "ssk17": [6, 13],
    "ssk18": [4, 6],
    "scottreef15_01": [0, 0],
    "scottreef11_01": [0, 0],
    "scottreef15_02": [0, 0],
    "scottreef11_02": [0, 0],
}


class SquidleDataset(DatasetVSLAMLAB):
    """SQUIDLE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "squidle") -> None:
        super().__init__(dataset_name)

        # Get download url
        self.api_url: str = self.cfg["api_url"]
        self.api_token: str = self.cfg.get("api_token", "not_set")
        if len(self.api_token.strip()) == 0:
            self.api_token = "not_set"

        # self.target_resolution is already set by DatasetVSLAMLAB.__init__ (super().__init__()
        # above) directly from this same yaml's target_resolution field - nothing to do here.

    def download_sequence_data(self, sequence_name: str) -> None:
        raw_path: Path = self.sequence_path(sequence_name) / "raw"
        rgb_path: Path = self.rgb_path(sequence_name)
        rgb_csv: Path = self.rgb_csv_path(sequence_name)
        gt_csv: Path = self.groundtruth_csv_path(sequence_name)
        marker = rgb_path / ".download_complete"
        if _mark_complete_if_already_downloaded(marker, rgb_path):
            return
        rgb_path.mkdir(parents=True, exist_ok=True)
        raw_path.mkdir(parents=True, exist_ok=True)

        # Setup initial params
        base_url = self.api_url
        headers = {"auth-token": self.api_token, "Content-type": "application/json", "Accept": "application/json"}
        query_structure = _get_query_structure(sequence_name)

        print_info("Querying for images, this may take a while...")
        page_num = 1
        total_pages = 1
        items = []
        with tqdm(desc="    fetching pages", unit="page") as pbar:
            while page_num <= total_pages:
                params = {
                    "q": json.dumps(query_structure),
                    "results_per_page": 100,
                    "page": page_num
                }

                r = requests.get(base_url + "/api/media", headers=headers, params=params)
                if r.status_code != 200:
                    print_warning(f"Error searching: {r.status_code}. Server response: {r.text}")
                    break

                data = r.json()
                new_objects = data.get("objects", [])
                items.extend(new_objects)

                if "num_pages" in data:
                    total_pages = data["num_pages"]
                elif "num_results" in data:
                    total_pages = math.ceil(data["num_results"] / 100)
                else:
                    total_pages = 1

                pbar.total = total_pages
                pbar.set_postfix(items=len(items))
                pbar.update(1)
                page_num += 1

        print_info(f"Found {len(items)} TOTAL images. Starting download...")
        rgb_header = ["ts_rgb_0 (ns)", "path_rgb_0", "sequence_name"]
        gt_header = ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']
        rgb_rows: list[list] = []
        gt_rows: list[list] = []

        estimated_new_resolution = False
        new_width, new_height = 0, 0
        for item in tqdm(items):
            media_id = item.get("id")
            try:
                detail_r = requests.get(f"{base_url}/api/media/{media_id}", headers=headers)
                if detail_r.status_code == 200:
                    item = detail_r.json()
                else:
                    print_warning(f"[{media_id}] Failed to get details. Skipping.")
                    continue
            except Exception as e:
                print_warning(f"[{media_id}] Connection error: {e}")
                continue

            timestamp = item.get("timestamp_start")
            pose = item.get("pose")
            try:
                ts_ns = _timestamp_to_nanoseconds(timestamp)
                pose_row = _parse_pose_data(pose, origin_utm=ORIGIN_UTM[sequence_name], origin_zone=ORIGIN_ZONE[sequence_name])
            except Exception as e:
                print_warning(f"[{media_id}] Bad timestamp/pose data: {e}")
                continue
            image_url = item.get("path_best")

            if not image_url:
                print_warning(f"Skipping ID {media_id}: No 'path_best' found.")
                continue

            filename = rgb_path / f"{media_id}.jpg"
            raw_filename = raw_path / f"{media_id}.jpg"
            try:
                with requests.get(image_url, stream=True) as stream_r:
                    if stream_r.status_code == 200:
                        with open(raw_filename, 'wb') as f:
                            stream_r.raw.decode_content = True
                            shutil.copyfileobj(stream_r.raw, f)
                        with Image.open(raw_filename) as img:
                            width, height = img.size
                            left = 0
                            top = 0
                            right = width - IMAGE_CROP[sequence_name][0]
                            bottom = height - IMAGE_CROP[sequence_name][1]
                            img_cropped = img.crop((left, top, right, bottom))
                            if self.target_resolution is None:
                                img_cropped.save(filename)
                            else:
                                if not estimated_new_resolution:
                                    estimated_new_resolution = True
                                    new_width, new_height = compute_scaled_size(img_cropped.size, self.target_resolution)
                                img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                img_resized.save(filename)

                        rgb_rows.append([ts_ns, f"rgb_0/{media_id}.jpg", sequence_name])
                        pose_row[0] = ts_ns
                        gt_rows.append(pose_row)
                    else:
                        print_warning(f"[{media_id}] Failed (status {stream_r.status_code})")
            except Exception as e:
                print_warning(f"[{media_id}] Error: {e}")

        write_csv_rows(rgb_csv, rgb_header, rgb_rows)
        write_csv_rows(gt_csv, gt_header, gt_rows)
        marker.touch()

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_calibration_yaml(self, sequence_name: str) -> None:
        # No verified calibration is available for this dataset's cameras - report as unknown
        # rather than publishing untrustworthy focal length/distortion values.
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [0.0, 0.0],
            "principal_point": [0.0, 0.0],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        pass

    def remove_unused_files(self, sequence_name: str) -> None:
        # raw/ holds the original un-cropped/un-resized downloads (the crop/resize source) - not a
        # pure reformat of rgb_0/ (cropped+resized from it), so only removed at MINIMAL, matching
        # HFColmapDatasetMixin's rgb_0_raw/ precedent (dataset_soneva.py).
        if BENCHMARK_RETENTION == Retention.MINIMAL:
            raw_path = self.sequence_path(sequence_name) / "raw"
            shutil.rmtree(raw_path, ignore_errors=True)

    def get_download_issues(self, _):
        if self.api_token != "not_set":
            return []
        return [
            _get_dataset_issue(
                issue_id="api_token",
                dataset_name=self.dataset_name,
                website=self.api_url,
                yaml_file=str(self.yaml_file),
            )
        ]


class SesokoDataset(SquidleDataset):
    """Sesoko dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "sesoko") -> None:
        super().__init__(dataset_name)

        self.subsets = self.cfg.get("subsets", {})
        self.combined = self.cfg.get("combined", {})

    def download_sequence_data(self, sequence_name: str) -> None:
        rgb_path = self.rgb_path(sequence_name)
        if _mark_complete_if_already_downloaded(rgb_path / ".download_complete", rgb_path):
            return

        if sequence_name in self.subsets.keys():
            super().download_sequence_data(self.subsets.get(sequence_name)[0])
            self.download_subsequence(sequence_name)
            return

        if sequence_name in self.combined.keys():
            for subset in self.combined.get(sequence_name):
                self.download_sequence_data(subset)
            self.download_combined_subsequence(sequence_name)
            return

        super().download_sequence_data(sequence_name)

    def download_subsequence(self, sequence_name: str) -> None:
        sequence_path: Path = self.sequence_path(sequence_name)
        rgb_path: Path = self.rgb_path(sequence_name)
        rgb_csv: Path = self.rgb_csv_path(sequence_name)
        gt_csv: Path = self.groundtruth_csv_path(sequence_name)
        marker = rgb_path / ".download_complete"
        if _mark_complete_if_already_downloaded(marker, rgb_path):
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        parent_sequence = self.subsets.get(sequence_name)[0]
        parent_sequence_path: Path = self.dataset_path / parent_sequence

        parent_rgb_csv: Path = parent_sequence_path / "rgb.csv"
        parent_gt_csv: Path = parent_sequence_path / "groundtruth.csv"
        df_rgb = pd.read_csv(parent_rgb_csv)
        df_gt = pd.read_csv(parent_gt_csv)

        target_image_name = self.subsets.get(sequence_name)[1]
        radius = self.subsets.get(sequence_name)[2]
        target_idx = df_rgb.index[df_rgb['path_rgb_0'] == target_image_name].tolist()
        ref_idx = target_idx[0]
        ref_x = df_gt.at[ref_idx, 'tx (m)']
        ref_y = df_gt.at[ref_idx, 'ty (m)']
        ref_z = df_gt.at[ref_idx, 'tz (m)']

        distances = np.sqrt(
            (df_gt['tx (m)'] - ref_x)**2 +
            (df_gt['ty (m)'] - ref_y)**2 +
            (df_gt['tz (m)'] - ref_z)**2
        )
        mask = distances <= radius
        df_rgb_sub = df_rgb[mask].copy().reset_index(drop=True)
        df_gt_sub = df_gt[mask].copy().reset_index(drop=True)

        for _, row in df_rgb_sub.iterrows():
            rel_path = row['path_rgb_0']
            full_src = os.path.abspath(parent_sequence_path / rel_path)
            full_dst = os.path.abspath(sequence_path / rel_path)
            if os.path.exists(full_dst) or os.path.islink(full_dst):
                os.remove(full_dst)
            os.symlink(full_src, full_dst)

        write_csv_rows(rgb_csv, df_rgb_sub.columns.tolist(), df_rgb_sub.astype(object).values.tolist())
        write_csv_rows(gt_csv, df_gt_sub.columns.tolist(), df_gt_sub.astype(object).values.tolist())
        marker.touch()

    def download_combined_subsequence(self, sequence_name: str) -> None:
        sequence_path: Path = self.sequence_path(sequence_name)
        rgb_path: Path = self.rgb_path(sequence_name)
        rgb_csv: Path = self.rgb_csv_path(sequence_name)
        gt_csv: Path = self.groundtruth_csv_path(sequence_name)
        marker = rgb_path / ".download_complete"
        if _mark_complete_if_already_downloaded(marker, rgb_path):
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        dfs_rgb = []
        dfs_pose = []
        for subset in self.combined.get(sequence_name):
            parent_sequence_path = self.dataset_path / subset
            parent_rgb_csv = parent_sequence_path / "rgb.csv"
            parent_gt_csv = parent_sequence_path / "groundtruth.csv"
            dfs_rgb.append(pd.read_csv(parent_rgb_csv))
            dfs_pose.append(pd.read_csv(parent_gt_csv))
            for _, row in dfs_rgb[-1].iterrows():
                rel_path = row['path_rgb_0']
                full_src = os.path.abspath(os.path.join(parent_sequence_path, rel_path))
                full_dst = os.path.abspath(os.path.join(sequence_path, rel_path))
                if os.path.exists(full_dst) or os.path.islink(full_dst):
                    os.remove(full_dst)
                os.symlink(full_src, full_dst)

        df_rgb_all = pd.concat(dfs_rgb, ignore_index=True)
        df_pose_all = pd.concat(dfs_pose, ignore_index=True)
        write_csv_rows(rgb_csv, df_rgb_all.columns.tolist(), df_rgb_all.astype(object).values.tolist())
        write_csv_rows(gt_csv, df_pose_all.columns.tolist(), df_pose_all.astype(object).values.tolist())
        marker.touch()


def _mark_complete_if_already_downloaded(marker: Path, rgb_path: Path) -> bool:
    """True if this sequence should be treated as already downloaded, touching marker as needed.
    Handles two cases: the marker already exists (normal case), or rgb_path has real content from
    a download that predates the .download_complete marker convention - trust it (matching the
    old rgb_path.exists() check this replaced) and touch the marker instead of re-fetching
    everything from scratch. An rgb_path that exists but is empty (e.g. a download that crashed
    right after mkdir) still falls through to a fresh download."""
    if marker.exists():
        return True
    if rgb_path.exists() and any(rgb_path.iterdir()):
        marker.touch()
        return True
    return False


def _timestamp_to_nanoseconds(timestamp_str):
    dt = datetime.datetime.fromisoformat(timestamp_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    seconds = int(dt.timestamp())
    microseconds = dt.microsecond
    nanoseconds = (seconds * 1_000_000_000) + (microseconds * 1_000)
    return nanoseconds


def _get_timestamp_ns(time_str):
    # Parses standard ISO or HTTP formats automatically
    dt = datetime.datetime.strptime(time_str, "%a, %d %b %Y %H:%M:%S GMT")
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _parse_pose_data(pose, origin_utm, origin_zone):
    ts_str = pose.get("timestamp") or pose.get("timestamp_start")
    ts_ns = _get_timestamp_ns(ts_str)
    data_map = {d['name']: d['value'] for d in pose.get('data', [])}
    rot = Rotation.from_euler('zyx', [
        data_map.get('heading', 0),
        data_map.get('pitch', 0),
        data_map.get('roll', 0)
    ])
    qx, qy, qz, qw = rot.as_quat()
    lat = pose.get('lat')
    lon = pose.get('lon')

    zone_num, zone_letter = origin_zone
    easting, northing, _, _ = utm.from_latlon(
        lat, lon,
        force_zone_number=zone_num,
        force_zone_letter=zone_letter
    )

    tx = easting - origin_utm[0]
    ty = northing - origin_utm[1]
    tz = pose.get('dep')
    return [ts_ns, tx, ty, tz, qx, qy, qz, qw]


def _get_query_structure(sequence_name: str) -> dict:
    if sequence_name in ("ssk16", "ssk17", "ssk18"):
        return {
            "filters": [
                {
                    "name": "deployment",
                    "op": "has",
                    "val": {
                        "name": "campaign",
                        "op": "has",
                        "val": {
                            "name": "key",
                            "op": "eq",
                            "val": CAMPAIGNS[sequence_name]
                        }
                    }
                }
            ],
            "limit": 1000000
        }

    if "scottreef" in sequence_name:
        return {
            "filters": [
                {"name": "deployment_id", "op": "eq", "val": DEPLOYMENT_IDS[sequence_name]}
            ],
            "limit": 1000000
        }

    raise ValueError(f"No SQUIDLE query structure defined for sequence_name={sequence_name!r}")


class ScottreefDataset(SesokoDataset):
    """SCOTTREEF dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "scottreef") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.subsets = cfg.get("subsets", {})
        self.combined = cfg.get("combined", {})
