from pathlib import Path
import re
import pandas as pd
import argparse
import os

# Label for script-specific outputs
SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
    
def downsample_rgb(timestamps, rgb_paths, rows, step, max_count):
    selected_rgb_paths = []
    selected_timestamps = []
    selected_rows = []

    index = 0
    while index < len(rgb_paths):
        index_int = int(index)
        selected_rgb_paths.append(rgb_paths[index_int])
        selected_timestamps.append(timestamps[index_int])
        selected_rows.append(rows[index_int])
        index += step

    # Ensure the number of selected images does not exceed the maximum allowed
    if len(selected_rgb_paths) > max_count:
        selected_rgb_paths = selected_rgb_paths[:int(max_count)]
        selected_timestamps = selected_timestamps[:int(max_count)]
        selected_rows = selected_rows[:int(max_count)]

    return selected_rgb_paths, selected_timestamps, selected_rows

def get_rows(rows_idx, rgb_csv):
    csv_path = Path(rgb_csv)  
    df = pd.read_csv(csv_path)     

    idx = [int(i) for i in rows_idx if isinstance(i, (int,)) or str(i).lstrip("-").isdigit()]
    idx = [i for i in idx if 0 <= i < len(df)]

    return df.iloc[idx].to_dict(orient="records")

def downsample_rgb_frames(rgb_csv, max_rgb_count, min_fps, verbose=False):

    csv_path = Path(rgb_csv)  
    df = pd.read_csv(csv_path)    
    # Reference camera: smallest index present (supports multi-camera CSV)
    path_cols = [c for c in df.columns if c.startswith('path_rgb') and c[len('path_rgb'):len('path_rgb')+1].isdigit()]
    # pick smallest camera index present
    ref_cam = 0
    if len(path_cols) > 0:
        cam_ids = []
        for c in path_cols:
            m = re.match(r'^path_rgb(\d+)$', c)
            if m:
                cam_ids.append(int(m.group(1)))
        if cam_ids:
            ref_cam = min(cam_ids)
    path_col = f'path_rgb{ref_cam}'
    ts_col = f'ts_rgb{ref_cam} (s)'

    rgb_paths = df[path_col].to_list()
    rgb_timestamps = df[ts_col].to_list()
    rows = df.to_dict(orient="records")

    # Determine downsampling parameters
    if verbose:
        print(f"\n{SCRIPT_LABEL} Processing file: {rgb_csv}")
        print("\nDownsampling settings:")
        print(f"  Maximum number of RGB images: {max_rgb_count}")
        print(f"  Minimum FPS: {min_fps:.1f} Hz")

    sequence_duration = rgb_timestamps[-1] - rgb_timestamps[0]
    actual_fps = len(rgb_paths) / sequence_duration
    max_interval = 1.0 / min_fps
    min_interval = sequence_duration / max_rgb_count

    if min_interval < max_interval:
        max_interval = min_interval
        if verbose:
            print(f"  Adjusted FPS to: {1.0 / max_interval:.2f} Hz")

    step_size = max_interval * actual_fps
    if step_size < 1:
        step_size = 1

    if verbose:
        print(f"  Step size: {step_size:.2f}")

    # Downsample RGB images
    if max_rgb_count >= len(rgb_paths):
        downsampled_paths = rgb_paths
        downsampled_timestamps = rgb_timestamps
        downsampled_rows = rows
    else:
        downsampled_paths, downsampled_timestamps, downsampled_rows = downsample_rgb(rgb_timestamps, rgb_paths, rows, step_size, max_rgb_count)


    downsampled_duration = downsampled_timestamps[-1] - downsampled_timestamps[0]
    downsampled_fps = len(downsampled_paths) / downsampled_duration

    if verbose:
        print("\nDownsampling RGB images:")
        print(f"  Sequence duration: {downsampled_duration:.2f} / {sequence_duration:.2f} seconds")
        print(f"  Number of RGB images: {len(downsampled_paths)} / {len(rgb_paths)}")
        print(f"  RGB frequency: {downsampled_fps:.2f} / {actual_fps:.2f} Hz")

    return downsampled_paths, downsampled_timestamps, downsampled_rows
