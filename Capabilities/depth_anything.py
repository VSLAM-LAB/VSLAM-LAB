from __future__ import annotations

import argparse
import os, torch
import sys
import numpy as np
import imageio.v2 as imageio
import pandas as pd
import yaml
import re
import math
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Datasets.get_dataset import get_dataset

def parse_args():
    p = argparse.ArgumentParser(prog="depth-inference")
    p.add_argument("dataset_name", help="e.g. eth")
    p.add_argument("sequence_name", help="e.g. table3")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--model_id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="DepthAnything3 model id",
    )
    p.add_argument("--depth_model_name", default="depth_anything")
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def main():
    cam_name = "rgb_0"
    args = parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    dataset = get_dataset(args.dataset_name)
    sequence_path = dataset.sequence_path(args.sequence_name)
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence path does not exist: {sequence_path}")

    rgb_csv = sequence_path / "rgb.csv"
    rgb_csv_raw = sequence_path / "rgb_raw.csv"

    depth_folder = f"{args.depth_model_name}_0"
    depth_path = sequence_path / depth_folder

    # Load calibration.yaml and find the camera section for cam_name
    calibration_yaml = sequence_path / "calibration.yaml"

    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)

    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;
    
    print(f"\nCamera Name: {cam['cam_name']}")
    if 'depth_factor' in cam:
        depth_factor = cam['depth_factor']
    else:
        depth_factor = 5000.0
   
    # Load model
    model = DepthAnything3.from_pretrained(args.model_id).to(device)

    # Clear depth output unless user wants to keep it
    if depth_path.exists():
        shutil.rmtree(depth_path)

    # Keep original csv as rgb_raw.csv once
    if rgb_csv.exists() and (not rgb_csv_raw.exists()):
        rgb_csv.rename(rgb_csv_raw)

    if not rgb_csv_raw.exists():
        raise FileNotFoundError(f"Missing {rgb_csv_raw}. (Expected it to exist or be created from rgb.csv)")

    df = pd.read_csv(rgb_csv_raw)
    if "ts_rgb_0 (ns)" not in df.columns or "path_rgb_0" not in df.columns:
        raise ValueError("rgb_raw.csv must contain columns: 'ts_rgb_0 (ns)' and 'path_rgb_0'")

    df.sort_values(by="ts_rgb_0 (ns)", inplace=True)

    images = [(sequence_path / p).as_posix() for p in df["path_rgb_0"].astype(str).tolist()]

    os.makedirs(depth_path, exist_ok=True)

    B = args.batch_size
    num_batches = math.ceil(len(images) / B)

    with torch.inference_mode():
        for s in tqdm(range(0, len(images), B), total=num_batches, desc="DepthAnything3 batches"):
            batch_paths = images[s : s + B]

            pred = model.inference(batch_paths)
            depth = pred.depth  # [b,h,w] float32

            for i in range(depth.shape[0]):
                in_path = batch_paths[i]
                fname = os.path.basename(in_path)

                rgb = imageio.imread(in_path)
                H0, W0 = rgb.shape[:2]

                d = depth[i]
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

                d_up = cv2.resize(d, (W0, H0), interpolation=cv2.INTER_NEAREST_EXACT)
                d16 = np.clip(d_up * depth_factor, 0, 65535).astype(np.uint16)

                imageio.imwrite(depth_path / fname, d16)

            del pred
            torch.cuda.empty_cache()

    df["ts_depth_0 (ns)"] = df["ts_rgb_0 (ns)"]
    df["path_depth_0"] = df["path_rgb_0"].astype(str).str.replace(r"^rgb_0/", f"{depth_folder}/", regex=True)

    df.to_csv(rgb_csv, index=False)
    print(f"Done.\nSequence: {args.dataset_name}/{args.sequence_name}\nDepth folder: {depth_folder}\nWrote: {rgb_csv}")


if __name__ == "__main__":
    main()
