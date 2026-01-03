from __future__ import annotations

import os
import csv
import utm
import math 
import yaml
import json
import shutil
import pandas as pd
import requests
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Final, Any
from scipy.spatial.transform import Rotation 

from utilities import print_msg
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import Retention, BENCHMARK_RETENTION
from Datasets.DatasetVSLAMLab_issues import _get_dataset_issue

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "
CAMPAIGNS: Final = {"ssk16": "ssk16-01", "ssk17": "ssk17-01", "ssk18": "ssk18-01"}
ORIGIN_UTM: Final = {"ssk16": (387124.51475913724, 2950359.888579014),
              "ssk17": (387124.51475913724, 2950359.888579014),
              "ssk18": (387124.51475913724, 2950359.888579014)}
ORIGIN_ZONE: Final = {"ssk16": (52, 'R'),
              "ssk17": (52, 'R'),
              "ssk18": (52, 'R')}
IMAGE_CROP: Final = {"ssk16": [146,3], "ssk17": [6,13], "ssk18": [4,6]}

class SQUIDLE_dataset(DatasetVSLAMLab):
    """SQUIDLE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "squidle") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        self.api_token: str = cfg.get("api_token", "not_set")
        if self.api_token == "not_set":
            logger.error(f"No API token set for SQUIDLE dataset. Register at '{self.url_download_root}' to get an API TOKEN, then set it in '{self.yaml_file}'")
            exit(0)

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Target image resolution
        self.image_resolution = cfg.get("target_resolution", [640, 480])
    
    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path = sequence_path / "rgb.csv"
        gt_csv: Path = sequence_path / "groundtruth.csv"
        test_csv: Path = sequence_path / "test.csv"
        if rgb_path.exists():
            return 
        rgb_path.mkdir(parents=True, exist_ok=True)

        # Setup initial params
        base_url = self.url_download_root
        headers = {"auth-token": self.api_token, "Content-type": "application/json", "Accept": "application/json"}
        query_structure = {
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
            "limit": 10000
        }

    
        print_msg(SCRIPT_LABEL, "Querying for images, this may take a while...")
        page_num = 1
        total_pages = 1 
        items = []
        while page_num <= total_pages:
            params = {
                "q": json.dumps(query_structure),
                "results_per_page": 100, 
                "page": page_num
            }
            
            r = requests.get(base_url + "/api/media", headers=headers, params=params)
            if r.status_code != 200:
                logger.error(f"Error searching: {r.status_code}")
                print(f"Server Response: {r.text}")
                break 

            data = r.json()
            new_objects = data.get("objects", [])
            items.extend(new_objects)
            
            current_page = data.get("page", page_num)
            if "num_pages" in data:
                total_pages = data["num_pages"]
            elif "num_results" in data:
                total_results = data["num_results"]
                total_pages = math.ceil(total_results / 100)
            else:
                total_pages = 1

            total_results = data.get("num_results", "Unknown")
            print(f"\r{SCRIPT_LABEL}Fetched Page {current_page}/{total_pages} ({len(new_objects)} items). Total collected: {len(items)}", end="", flush=True)
            page_num += 1
        print()

        print_msg(SCRIPT_LABEL, f"Found {len(items)} TOTAL images. Starting download...")
        with open(rgb_csv, mode='a', newline='') as f_rgb, open(gt_csv, mode='a', newline='') as f_gt, open(test_csv, mode='a', newline='') as f_test:
            writer_rgb = csv.writer(f_rgb, delimiter=',')
            writer_gt  = csv.writer(f_gt, delimiter=',')     
            writer_rgb.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])     
            writer_gt.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']) 

            estimated_new_resolution = False
            new_height, new_width = 0,0
            for item in tqdm(items):
                media_id = item.get("id")
                try:
                    detail_r = requests.get(f"{base_url}/api/media/{media_id}", headers=headers)
                    if detail_r.status_code == 200:
                        item = detail_r.json()
                    else:
                        print(f"[{media_id}] Failed to get details. Skipping.")
                        continue
                except Exception as e:
                    print(f"[{media_id}] Connection error: {e}")
                    continue

                timestamp = item.get("timestamp_start")
                ts_ns = _timestamp_to_nanoseconds(timestamp)
                pose = item.get("pose")
                pose_row =  _parse_pose_data(pose, origin_utm=ORIGIN_UTM[sequence_name], origin_zone=ORIGIN_ZONE[sequence_name])           
                image_url = item.get("path_best")
                
                if not image_url:
                    tqdm.write(f"   Skipping ID {media_id}: No 'path_best' found.")
                    continue
                
                writer_rgb.writerow([ts_ns, f"rgb_0/{media_id}.jpg"])
                pose_row[0] = ts_ns 
                writer_gt.writerow(pose_row)
                
                # filename = rgb_path / f"{media_id}.jpg"
                # temp_filename = rgb_path / "temp_download.dat"
                # try:
                #     with requests.get(image_url, stream=True) as stream_r:
                #         if stream_r.status_code == 200:                                
                #             # with open(temp_filename, 'wb') as f:
                #             #     stream_r.raw.decode_content = True
                #             #     shutil.copyfileobj(stream_r.raw, f)                         
                #             # with Image.open(temp_filename) as img:
                #             #     width, height = img.size
                #             #     left = 0
                #             #     top = 0
                #             #     right = width - IMAGE_CROP[sequence_name][0]
                #             #     bottom = height - IMAGE_CROP[sequence_name][1]
                #             #     img_cropped = img.crop((left, top, right, bottom))
                #             #     if not estimated_new_resolution:
                #             #         estimated_new_resolution = True
                #             #         new_height = np.sqrt(self.image_resolution[0] * self.image_resolution[1] * img_cropped.size[1] / img_cropped.size[0])
                #             #         new_width = self.image_resolution[0] * self.image_resolution[1] / new_height
                #             #         new_height = int(new_height)
                #             #         new_width = int(new_width)
                #             #         estimated_new_resolution = True
                #             #     img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)      
                #             #     img_resized.save(filename)

                #             writer_rgb.writerow([ts_ns, f"rgb_0/{media_id}.jpg"])
                #             writer_gt.writerow(pose)
                #             writer_test.writerow([ts_ns, test])
                #             os.remove(temp_filename)
                #         else:
                #             print(f"Failed (Status {stream_r.status_code})")
                # except Exception as e:
                #     print(f"Error: {e}")
     
    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb",
                 "cam_model": "unknown", "focal_length": [fx, fy], "principal_point": [cx, cy],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])
        
    def create_groundtruth_csv(self, sequence_name: str) -> None:
        pass

    def remove_unused_files(self, sequence_name: str) -> None:
        pass

    def get_download_issues(self, _):
        if self.api_token != "not_set":
            return {}
        return [_get_dataset_issue(issue_id="api_token", dataset_name=self.dataset_name, website=self.url_download_root, yaml_file=self.yaml_file)]


class SESOKO_dataset(SQUIDLE_dataset):
    """SESOKO dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "sesoko") -> None:
        super().__init__(Path(benchmark_path), dataset_name) 

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.subsets = cfg.get("subsets", {})
        self.combined = cfg.get("combined", {})

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        if rgb_path.exists():
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
            sequence_path: Path = self.dataset_path / sequence_name
            rgb_path: Path = sequence_path / "rgb_0"
            rgb_csv: Path = sequence_path / "rgb.csv"
            gt_csv: Path = sequence_path / "groundtruth.csv"
            if rgb_path.exists():
                    return 
            rgb_path.mkdir(parents=True, exist_ok=True)
    
            parent_sequence = _get_subsequence_name(sequence_name)
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

            df_rgb_sub.to_csv(rgb_csv, index=False, sep=',') 
            df_gt_sub.to_csv(gt_csv, index=False, sep=',')
    
    def download_combined_subsequence(self, sequence_name):
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path = sequence_path / "rgb.csv"
        gt_csv: Path = sequence_path / "groundtruth.csv"
        if rgb_path.exists():
                return 
        rgb_path.mkdir(parents=True, exist_ok=True)

        dfs_rgb = []
        dfs_pose = []
        for subset in self.combined.get(sequence_name):   
            if f"s{sequence_name[-2:]}" not in subset:
                continue; 
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
            df_rgb_all.to_csv(rgb_csv, index=False, sep=',') 
            df_pose_all.to_csv(gt_csv, index=False, sep=',')

def _get_subsequence_name(sequence_name: str):
    if "ssk16" in sequence_name:
        return "ssk16"
    elif "ssk17" in sequence_name:
         return "ssk17"
    elif "ssk18" in sequence_name:
        return "ssk18"
    return

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

def _parse_pose_data(item, origin_utm, origin_zone):
    ts_str = item.get("timestamp") or item.get("timestamp_start")
    ts_ns = _get_timestamp_ns(ts_str)
    data_map = {d['name']: d['value'] for d in item.get('data', [])}
    rot = Rotation.from_euler('zyx', [
        data_map.get('heading', 0),
        data_map.get('pitch', 0),
        data_map.get('roll', 0)
    ])
    qx, qy, qz, qw = rot.as_quat()
    lat = item.get('lat')
    lon = item.get('lon')
    
    zone_num, zone_letter = origin_zone
    easting, northing, _, _ = utm.from_latlon(
        lat, lon, 
        force_zone_number=zone_num, 
        force_zone_letter=zone_letter
    )

    tx = easting - origin_utm[0]
    ty = northing - origin_utm[1]
    tz = item.get('dep')
    return [ts_ns, tx, ty, tz, qx, qy, qz, qw]

