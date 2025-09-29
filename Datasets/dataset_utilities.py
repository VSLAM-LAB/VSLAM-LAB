import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

def load_rgb_csv(rgb_csv):

    csv_path = Path(rgb_csv)  
    df = pd.read_csv(csv_path)  
    rgb_paths = df['path_rgb0'].to_list()
    rgb_timestamps = df['ts_rgb0 (s)'].to_list()

    if 'ts_depth0 (s)' in df.columns and 'path_depth0' in df.columns:
        depth_paths = df['path_depth0'].to_list()
        depth_timestamps = df['ts_depth0 (s)'].to_list()
        return rgb_paths, rgb_timestamps, depth_paths, depth_timestamps
    
    return rgb_paths, rgb_timestamps


def parse_filename_timestamp(name: str) -> float:
    """Parse a numeric timestamp from filename (without extension), ns→s if large."""
    stem = Path(name).stem
    try:
        ts = float(stem)
        # Heuristic: if it's likely in ns, convert to seconds
        if ts > 1e12:
            ts = ts / 1e9
        return ts
    except Exception:
        return None


def load_rig_yaml(rig_yaml_path: str) -> Dict[str, Any]:
    with open(rig_yaml_path, 'r', encoding='utf-8') as f:
        rig = yaml.safe_load(f) or {}
    # minimal validation
    assert 'cameras' in rig and isinstance(rig['cameras'], list) and len(rig['cameras']) >= 1, "rig.yaml requires a non-empty 'cameras' list"
    assert 'imu' in rig and 'csv' in rig['imu'], "rig.yaml requires 'imu.csv' path"
    return rig


def build_multicam_rgb_csv_rows(dataset_root: Path, rig: Dict[str, Any], selected_cameras: List[int]) -> List[Dict[str, Any]]:
    """Create synchronized rows for the selected cameras using rig pairing.

    Returns a list of dictionaries with columns 'ts_rgbi (s)', 'path_rgbi' per camera id.
    """
    cams = {c['id']: c for c in rig['cameras'] if c['id'] in selected_cameras}
    assert len(cams) == len(selected_cameras), "Selected camera id not found in rig"

    pairing = rig.get('pairing', {})
    ref_cam_id = pairing.get('reference_camera', selected_cameras[0])
    max_dt = float(pairing.get('max_time_offset', 0.01))
    require_all = bool(pairing.get('require_all', True))

    # Load per-camera file lists and timestamps
    cam_files: Dict[int, List[Tuple[float, str]]] = {}
    for cam_id, cam in cams.items():
        data_dir = dataset_root / cam['data_dir']
        glob_pat = cam.get('file_glob', '*.png')
        files = sorted([p.name for p in data_dir.glob(glob_pat)])
        pairs: List[Tuple[float, str]] = []
        for fname in files:
            ts = parse_filename_timestamp(fname)
            if ts is not None:
                pairs.append((ts, f"{cam['data_dir'].rstrip('/')}/{fname}"))
        cam_files[cam_id] = pairs

    # Index other cams for nearest lookup
    import bisect
    indexed_times: Dict[int, List[float]] = {cid: [t for t, _ in pairs] for cid, pairs in cam_files.items()}

    rows: List[Dict[str, Any]] = []
    for t_ref, path_ref in cam_files[ref_cam_id]:
        row: Dict[str, Any] = {}
        ok = True
        for cam_id in selected_cameras:
            times = indexed_times[cam_id]
            pairs = cam_files[cam_id]
            if not times:
                ok = False
                break
            idx = bisect.bisect_left(times, t_ref)
            candidates = []
            if idx < len(times):
                candidates.append((abs(times[idx] - t_ref), idx))
            if idx > 0:
                candidates.append((abs(times[idx-1] - t_ref), idx-1))
            if not candidates:
                ok = False
                break
            _, j = min(candidates)
            t_j, path_j = pairs[j]
            if abs(t_j - t_ref) > max_dt:
                ok = False if require_all else ok
                if require_all:
                    break
            row[f"ts_rgb{cam_id} (s)"] = t_j
            row[f"path_rgb{cam_id}"] = path_j
        if ok:
            rows.append(row)

    return rows

def undistort_rgb_rad_tan(rgb_csv, sequence_path, camera_matrix, distortion_coeffs):

    rgb_paths, *_ = load_rgb_csv(rgb_csv)

    # Estimate undistortion transformation
    rgb_path = os.path.join(sequence_path, rgb_paths[0])
    rgb = cv2.imread(rgb_path)
    h, w = rgb.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h),5)
    x, y, w, h = roi

    # Undistort rgb images
    for rgb_subpath in tqdm(rgb_paths, desc="Undistorting RGB Images"):
        rgb_path = os.path.join(sequence_path, rgb_subpath)
        rgb = cv2.imread(rgb_path)
        undistorted_rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        undistorted_rgb = undistorted_rgb[y:y+h, x:x+w]
        cv2.imwrite(rgb_path, undistorted_rgb)

    fx, fy, cx, cy = (new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                      new_camera_matrix[0, 2], new_camera_matrix[1, 2])
    return fx, fy, cx, cy

def undistort_depth_rad_tan(rgb_csv, sequence_path, camera_matrix, distortion_coeffs):
    
    _, _, depth_paths, _ = load_rgb_csv(rgb_csv)

    # Estimate undistortion transformation
    depth_path = os.path.join(sequence_path, depth_paths[0])
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    h, w = depth.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)
    x, y, w, h = roi

    # Undistort depth images
    for depth_subpath in tqdm(depth_paths, desc="Undistorting Depth Images"):
        depth_path = os.path.join(sequence_path, depth_subpath)
        depth = cv2.imread(depth_path)
        undistorted_depth = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        undistorted_depth = undistorted_depth[y:y+h, x:x+w]
        cv2.imwrite(depth_path, undistorted_depth)

    fx, fy, cx, cy = (new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                      new_camera_matrix[0, 2], new_camera_matrix[1, 2])
    return fx, fy, cx, cy

def undistort_fisheye(rgb_csv, sequence_path, camera_matrix, distortion_coeffs):

    rgb_paths, *_ = load_rgb_csv(rgb_csv)
    first = True
    for image_name in tqdm(rgb_paths):
        image_path = os.path.join(sequence_path, image_name)
        image = cv2.imread(image_path)
        if first:
            first = False
            h, w = image.shape[:2]
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix, distortion_coeffs, (w, h), None)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(image_path, undistorted_image)

    fx, fy, cx, cy = (new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                      new_camera_matrix[0, 2], new_camera_matrix[1, 2])
    return fx, fy, cx, cy


def resize_rgb_images(rgb_csv, sequence_path, target_width, target_height, camera_matrix):
    
    rgb_paths, *_ = load_rgb_csv(rgb_csv)

    # Load the first image to get original dimensions and compute scaling factors
    sample_rgb_path = os.path.join(sequence_path, rgb_paths[0])
    sample_rgb = cv2.imread(sample_rgb_path)
    original_height, original_width = sample_rgb.shape[:2]

    # Scaling factors for adjusting the camera intrinsic parameters
    scale = np.sqrt(target_width * target_height * original_width / original_height) / original_width
    w = int(original_width * scale)
    h = int(original_height * scale)
    scale_x = w / original_width
    scale_y = h / original_height

    # Resize all RGB images
    for rgb_subpath in tqdm(rgb_paths, desc="Resizing RGB Images"):
        rgb_path = os.path.join(sequence_path, rgb_subpath)
        rgb_image = cv2.imread(rgb_path)
        resized_rgb = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(rgb_path, resized_rgb)

    # Adjust the camera matrix for the new resolution
    fx, fy, cx, cy = (camera_matrix[0,0] * scale_x, camera_matrix[1,1] * scale_y,
                      camera_matrix[0,2] * scale_x, camera_matrix[1,2] * scale_y)
    
    return fx, fy, cx, cy
