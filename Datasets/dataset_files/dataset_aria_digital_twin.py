"""
Module: VSLAM-LAB - Datasets - dataset_aria_digital_twin.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.1
- Created: 2026-08-14
- Updated: 2026-08-14
- License: GPLv3 License
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Final

import numpy as np
from utilities import make_printers, write_csv_rows

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.DatasetVSLAMLAB_issues import _get_dataset_issue
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Fixed Aria Gen1 device stream labels (same across every ADT sequence). rgb_0/rgb_1 are the SLAM
# stereo pair (camera-slam-left/camera-slam-right) - the device's only true calibrated stereo
# pair, hardware-synced - not the color camera-rgb stream, which has no stereo partner and is
# unused here. depth_0 is ground-truth depth aligned to camera-slam-left.
_RGB0_LABEL: Final = "camera-slam-left"
_RGB1_LABEL: Final = "camera-slam-right"
_IMU_LABEL: Final = "imu-left"

# Folder/marker convention mirrors Capabilities/mask2former.py's mask2former_<i> output
# (uint8 "L" PNG, 1=static/0=dynamic, one file per rgb_<i> frame) so downstream consumers of
# rgb_exp.csv can't tell the two apart - only the folder name says these came from ADT's own
# ground-truth segmentation rather than a Mask2Former inference pass.
_MASK_FOLDER_BASE: Final = "adt_segmentation"

# aria_dataset_downloader -d indices (run with no -d to list them), paired with the matching key
# aria_dataset_downloader itself writes to <sequence>/.download_status.json (used below to verify
# a download actually succeeded - the CLI exits 0 even when individual data types fail, see
# download_sequence_data). Covers exactly the data groups this file's worker script reads:
# main_vrs (video.vrs: rgb/imu/raw calibration), main_groundtruth (aria_trajectory.csv/
# instances.json/Skeleton_*.json/... for calibration+groundtruth+segmentation labeling),
# segmentation (segmentations[_with_skeleton].vrs), depth (depth_images[_with_skeleton].vrs, by far
# the largest group at several GB/sequence - still needed since rgbd/rgbd-vi modes read ground-truth
# depth from it). Deliberately excludes mps_slam_*/mps_eye_gaze/mps_hand_tracking and synthetic:
# nothing here reads MPS output or the synthetic-rendering video.
_CDN_DATA_TYPES: Final = (
    ("0", "main_vrs"),
    ("6", "main_groundtruth"),
    ("7", "segmentation"),
    ("8", "depth"),
)
_DOWNLOAD_STATUS_FILE: Final = ".download_status.json"
_MAX_DOWNLOAD_ATTEMPTS: Final = 3

# projectaria_tools (VRS parsing, ADT ground-truth API, fisheye-to-pinhole rectification) is only
# installable in the separate "projectaria" pixi environment (Python 3.12) - the main "vslamlab"
# environment this file itself runs under (Python 3.13) never has it. Every hook below that needs
# it shells out via `pixi run -e projectaria python3 <this worker script>`, mirroring
# utilities.run_rosbag_frame_extraction's existing pattern for bridging into the ros1/ros2 pixi
# environments. Filed as a capability gap (add-dataset's Issue exception) rather than worked around
# by hand-rolled VRS parsing: see the "aria-digital-twin: no per-dataset processing environment"
# issue this run filed.
_WORKER_SCRIPT = '''
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider, MotionType,
)


def _open_gt(sequence_path):
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    # get_datapaths(True) ("with skeleton occlusion") does NOT fall back to the plain
    # segmentations.vrs/depth_images.vrs when a sequence has no skeleton ground truth (e.g. the
    # "clean"/non-"_skeleton_" ADT sequences) - it silently leaves segmentation/depth completely
    # unloaded (has_segmentation_images()/has_depth_images() False, every per-frame query invalid),
    # not just missing a few frames. Request the with-skeleton variant only when that file actually
    # exists; otherwise use the base (skeleton-free) ground truth.
    has_skeleton = (Path(sequence_path) / "segmentations_with_skeleton.vrs").exists()
    data_paths = paths_provider.get_datapaths(has_skeleton)
    return AriaDigitalTwinDataProvider(data_paths)


def _dynamic_instance_ids(gt) -> set[int]:
    """Instance ids whose ADT-authored MotionType is DYNAMIC (movable objects and humans/skeletons) -
    ADT's own ground truth, not a class-name heuristic."""
    return {
        iid for iid in gt.get_instance_ids()
        if gt.get_instance_info_by_id(iid).motion_type == MotionType.DYNAMIC
    }


def _static_dynamic_mask(segmentation: np.ndarray, dynamic_ids: set[int]) -> np.ndarray:
    """1 = static pixel, 0 = dynamic pixel, uint8, same convention as mask2former.py. segmentation
    holds per-pixel instance ids (0 = unlabeled/background, treated as static)."""
    mask = np.ones(segmentation.shape, dtype=np.uint8)
    for iid in np.unique(segmentation):
        if iid != 0 and int(iid) in dynamic_ids:
            mask[segmentation == iid] = 0
    return mask


def cmd_rectify(args):
    gt = _open_gt(args.sequence_path)
    raw_provider = gt.raw_data_provider_ptr()
    out_dirs = {"camera-slam-left": Path(args.out_rgb0), "camera-slam-right": Path(args.out_rgb1)}
    out_masks = {"camera-slam-left": Path(args.out_mask0), "camera-slam-right": Path(args.out_mask1)}
    out_depth = Path(args.out_depth0)
    dynamic_ids = _dynamic_instance_ids(gt)

    # ADT's video recording extends past both ends of its ground-truth coverage (~12s each way
    # on the sequences checked), and projectaria's per-timestamp GT queries (pose, depth,
    # segmentation) CLAMP out-of-range timestamps to the nearest sample while still reporting
    # is_valid()=True - so without this trim, every video frame outside the GT range silently
    # gets a byte-identical frozen copy of the first/last depth+segmentation frame (uncorrelated
    # with the moving RGB image; poisons RGB-D runs). Trim ALL streams (rgb included) to the
    # GT-covered range so the per-stream folders stay index-aligned for create_rgb_csv's zip.
    gt_start, gt_end = gt.get_start_time_ns(), gt.get_end_time_ns()
    has_gt_bounds = gt_end > gt_start

    for label, out_dir in out_dirs.items():
        sid = raw_provider.get_stream_id_from_label(label)
        raw_calib = gt.get_aria_camera_calibration(sid)
        focal_length = float(raw_calib.get_projection_params()[0])
        # args.width/height are the final upright (post-rotation) size - the SLAM cameras are
        # mounted physically rotated 90deg, so the pinhole grid used to sample the fisheye image is
        # landscape (swapped) and gets rotated upright below, matching projectaria_tools'
        # rotate_upright_image_and_calibration convention.
        lin_calib = calibration.get_linear_camera_calibration(
            args.height, args.width, focal_length, label, raw_calib.get_transform_device_camera()
        )
        out_mask = out_masks[label]
        write_depth = label == "camera-slam-left"

        for ts in gt.get_aria_device_capture_timestamps_ns(sid):
            if has_gt_bounds and not (gt_start <= ts <= gt_end):
                continue
            out_path = out_dir / f"{ts}.png"
            depth_path = out_depth / f"{ts}.png"
            mask_path = out_mask / f"{ts}.png"
            need_image = not out_path.exists()
            need_depth = write_depth and not depth_path.exists()
            need_mask = not mask_path.exists()
            if not need_image and not need_depth and not need_mask:
                continue

            if need_image:
                image_result = gt.get_aria_image_by_timestamp_ns(ts, sid)
                if image_result.is_valid():
                    raw_image = image_result.data().to_numpy_array()
                    rectified = calibration.distort_by_calibration(
                        raw_image, lin_calib, raw_calib, InterpolationMethod.BILINEAR
                    )
                    Image.fromarray(np.rot90(rectified, k=3)).save(out_path)

            if need_depth:
                depth_result = gt.get_depth_image_by_timestamp_ns(ts, sid)
                if depth_result.is_valid():
                    raw_depth = depth_result.data().to_numpy_array()
                    rectified_depth = calibration.distort_by_calibration(
                        raw_depth, lin_calib, raw_calib, InterpolationMethod.NEAREST_NEIGHBOR
                    )
                    Image.fromarray(np.rot90(rectified_depth, k=3)).save(depth_path)

            if need_mask:
                seg_result = gt.get_segmentation_image_by_timestamp_ns(ts, sid)
                if seg_result.is_valid():
                    # Binarize in the raw fisheye domain first, then rectify with nearest-neighbor:
                    # instance ids are uint64 (unsafe to warp directly), and NEAREST_NEIGHBOR on an
                    # already-binary mask keeps every pixel exactly 0 or 1 (no blended edge values).
                    raw_seg = seg_result.data().to_numpy_array()
                    raw_mask = _static_dynamic_mask(raw_seg, dynamic_ids)
                    rectified_mask = calibration.distort_by_calibration(
                        raw_mask, lin_calib, raw_calib, InterpolationMethod.NEAREST_NEIGHBOR
                    )
                    Image.fromarray(np.rot90(rectified_mask, k=3), mode="L").save(mask_path)


def cmd_calibration(args):
    gt = _open_gt(args.sequence_path)
    raw_provider = gt.raw_data_provider_ptr()
    device_calib = raw_provider.get_device_calibration()

    out = {}
    for key, label in (("rgb_0", "camera-slam-left"), ("rgb_1", "camera-slam-right")):
        sid = raw_provider.get_stream_id_from_label(label)
        raw_calib = gt.get_aria_camera_calibration(sid)
        focal_length = float(raw_calib.get_projection_params()[0])
        # Same landscape-then-rotate-upright pinhole as cmd_rectify, so this calibration matches
        # the images actually written to rgb_0/rgb_1 (and depth_0, which shares rgb_0's calib).
        lin_calib = calibration.get_linear_camera_calibration(
            args.height, args.width, focal_length, label, raw_calib.get_transform_device_camera()
        )
        upright_calib = calibration.rotate_camera_calib_cw90deg(lin_calib)
        fx, fy = upright_calib.get_focal_lengths()
        ppx, ppy = upright_calib.get_principal_point()
        out[key] = {
            "focal_length": [float(fx), float(fy)],
            "principal_point": [float(ppx), float(ppy)],
            "T_device_camera": upright_calib.get_transform_device_camera().to_matrix().tolist(),
        }

    imu_calib = device_calib.get_imu_calib(args.imu_label)
    imu_sid = raw_provider.get_stream_id_from_label(args.imu_label)
    d0 = raw_provider.get_imu_data_by_index(imu_sid, 0)
    d1 = raw_provider.get_imu_data_by_index(imu_sid, 1)
    fps = 1.0e9 / (d1.capture_timestamp_ns - d0.capture_timestamp_ns)
    out["imu_0"] = {
        "T_device_imu": imu_calib.get_transform_device_imu().to_matrix().tolist(),
        "fps": fps,
    }
    # Written to a file, not stdout: projectaria_tools' native C++ logger writes its own
    # info/warning lines directly to real stdout (not just stderr), which would otherwise get
    # mixed in with this JSON and break parsing on the caller's side.
    Path(args.out_json).write_text(json.dumps(out), encoding="utf-8")


def cmd_imu(args):
    provider = data_provider.create_vrs_data_provider(str(Path(args.sequence_path) / "video.vrs"))
    sid = provider.get_stream_id_from_label(args.imu_label)
    rows = []
    for i in range(provider.get_num_data(sid)):
        d = provider.get_imu_data_by_index(sid, i)
        rows.append([d.capture_timestamp_ns, *d.gyro_radsec, *d.accel_msec2])

    out_csv = Path(args.out_csv)
    with open(out_csv.with_suffix(".csv.tmp"), "w", encoding="utf-8") as f:
        f.write("ts (ns),wx (rad s^-1),wy (rad s^-1),wz (rad s^-1),ax (m s^-2),ay (m s^-2),az (m s^-2)\\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\\n")
    out_csv.with_suffix(".csv.tmp").replace(out_csv)


def cmd_groundtruth(args):
    gt = _open_gt(args.sequence_path)
    raw_provider = gt.raw_data_provider_ptr()
    sid = raw_provider.get_stream_id_from_label("camera-slam-left")

    # Same GT-range trim as cmd_rectify: is_valid() alone does NOT reject out-of-range
    # timestamps (the provider clamps and returns the first/last pose as "valid"), which
    # previously produced hundreds of identical constant-pose rows at both ends of the csv.
    gt_start, gt_end = gt.get_start_time_ns(), gt.get_end_time_ns()
    has_gt_bounds = gt_end > gt_start

    rows = []
    for ts in gt.get_aria_device_capture_timestamps_ns(sid):
        if has_gt_bounds and not (gt_start <= ts <= gt_end):
            continue
        pose_result = gt.get_aria_3d_pose_by_timestamp_ns(ts)
        if not pose_result.is_valid():
            continue
        se3 = pose_result.data().transform_scene_device
        tx, ty, tz = se3.translation()[0]
        # to_quat() returns [qw, qx, qy, qz] - reorder to VSLAM-LAB's qx,qy,qz,qw convention.
        qw, qx, qy, qz = se3.rotation().to_quat()[0]
        rows.append([ts, tx, ty, tz, qx, qy, qz, qw])

    out_csv = Path(args.out_csv)
    with open(out_csv.with_suffix(".csv.tmp"), "w", encoding="utf-8") as f:
        f.write("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\\n")
    out_csv.with_suffix(".csv.tmp").replace(out_csv)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rectify")
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--out_rgb0", required=True)
    p.add_argument("--out_rgb1", required=True)
    p.add_argument("--out_depth0", required=True)
    p.add_argument("--out_mask0", required=True)
    p.add_argument("--out_mask1", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.set_defaults(func=cmd_rectify)

    p = sub.add_parser("calibration")
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--imu_label", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser("imu")
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--imu_label", required=True)
    p.add_argument("--out_csv", required=True)
    p.set_defaults(func=cmd_imu)

    p = sub.add_parser("groundtruth")
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--out_csv", required=True)
    p.set_defaults(func=cmd_groundtruth)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
'''


class AriaDigitalTwinDataset(DatasetVSLAMLAB):
    """Aria Digital Twin (ADT) egocentric SLAM benchmark dataset helper for VSLAM-LAB."""

    def __init__(self, dataset_name: str = "aria-digital-twin") -> None:
        super().__init__(dataset_name)

        # All sequences are gated behind ADT's license agreement - there's no anonymous download
        # URL. sequence_location stays 'local' (matching every other manually-fetched dataset)
        # even though download_sequence_data below can now fetch automatically: it still depends
        # on a per-user, ~14-day-expiring CDN links file the user has to obtain and place
        # themselves (see cdn_urls_file).
        self.sequence_location = self.cfg["sequence_location"]
        self.depth_factor = self.cfg["depth_factor"]
        self.cdn_urls_file = self.dataset_path / self.cfg["cdn_urls_file"]
        self.dataset_homepage = self.cfg["about"]["homepage"]

    def get_download_issues(self, _: list[str]) -> list[dict]:
        if self.cdn_urls_file.exists():
            return []
        return [
            _get_dataset_issue(
                issue_id="cdn_links_file",
                dataset_name=self.dataset_name,
                website=self.dataset_homepage,
                target_file=str(self.cdn_urls_file),
            )
        ]

    def download_sequence_data(self, sequence_name: str) -> None:
        if not self.cdn_urls_file.exists():
            print_info(
                f"Sequence '{sequence_name}' is marked as 'local'. Sign ADT's license agreement, "
                f"fetch your per-user CDN links file with the official aria_dataset_downloader "
                f"(projectaria_tools), and place it at {self.cdn_urls_file} (see 'cdn_urls_file' "
                f"in dataset_aria-digital-twin.yaml) so this can download automatically. "
                f"Alternatively, fetch '{sequence_name}' manually and place it at "
                f"{self.sequence_path(sequence_name)}."
            )
            return

        data = json.loads(self.cdn_urls_file.read_text(encoding="utf-8"))
        if sequence_name not in data.get("sequences", {}):
            print_warning(
                f"'{sequence_name}' is not listed in {self.cdn_urls_file.name} - your CDN links "
                f"file may be stale (links expire ~14 days) or predate this sequence being added. "
                f"Re-fetch it from ADT's website and place it at {self.cdn_urls_file}."
            )
            return

        self.dataset_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "pixi", "run", "-e", "projectaria", "aria_dataset_downloader",
            "-c", str(self.cdn_urls_file),
            "-o", str(self.dataset_path),
            "-l", sequence_name,
            "-d", *[index for index, _ in _CDN_DATA_TYPES],
        ]

        # aria_dataset_downloader exits 0 even when individual data types fail (e.g. a transient
        # network timeout on the ~2GB main_vrs file) - it just prints "N of 1 sequences are
        # successfully downloaded" and moves on, so `check=True` alone can't detect this. It writes
        # its own per-data-type status to .download_status.json though, and is resumable (skips
        # data types already marked successful there) - so retry a few times, which is cheap: a
        # retry only re-attempts whatever actually failed, not the multi-GB types that already
        # succeeded.
        missing: list[tuple[str, str]] = list(_CDN_DATA_TYPES)
        for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
            subprocess.run(cmd, check=True, cwd=str(VSLAM_LAB_DIR))
            missing = self._missing_download_types(sequence_name)
            if not missing:
                return
            if attempt < _MAX_DOWNLOAD_ATTEMPTS:
                print_warning(
                    f"'{sequence_name}': data type(s) {[name for _, name in missing]} did not "
                    f"download successfully (likely a transient network error) - retrying "
                    f"({attempt}/{_MAX_DOWNLOAD_ATTEMPTS}) ..."
                )

        raise RuntimeError(
            f"Downloading '{sequence_name}' failed after {_MAX_DOWNLOAD_ATTEMPTS} attempts - data "
            f"type(s) {[name for _, name in missing]} never completed. Check your network "
            f"connection and re-run; aria_dataset_downloader resumes from "
            f"{self.sequence_path(sequence_name) / _DOWNLOAD_STATUS_FILE} and won't re-fetch "
            f"data types that already succeeded."
        )

    def _missing_download_types(self, sequence_name: str) -> list[tuple[str, str]]:
        status_path = self.sequence_path(sequence_name) / _DOWNLOAD_STATUS_FILE
        if not status_path.exists():
            return list(_CDN_DATA_TYPES)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return [(index, name) for index, name in _CDN_DATA_TYPES if not status.get(name, False)]

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb0_path = self.rgb_path(sequence_name)
        rgb1_path = self.sequence_path(sequence_name) / "rgb_1"
        depth0_path = self.depth_path(sequence_name)
        mask0_path = self.sequence_path(sequence_name) / f"{_MASK_FOLDER_BASE}_0"
        mask1_path = self.sequence_path(sequence_name) / f"{_MASK_FOLDER_BASE}_1"
        rgb0_path.mkdir(parents=True, exist_ok=True)
        rgb1_path.mkdir(parents=True, exist_ok=True)
        depth0_path.mkdir(parents=True, exist_ok=True)
        mask0_path.mkdir(parents=True, exist_ok=True)
        mask1_path.mkdir(parents=True, exist_ok=True)

        width, height = self.target_resolution
        self._run_worker(
            "rectify",
            "--sequence_path", str(self.sequence_path(sequence_name)),
            "--out_rgb0", str(rgb0_path),
            "--out_rgb1", str(rgb1_path),
            "--out_depth0", str(depth0_path),
            "--out_mask0", str(mask0_path),
            "--out_mask1", str(mask1_path),
            "--width", str(width),
            "--height", str(height),
        )

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        # camera-slam-left/right are hardware-synced (capture timestamps differ by ~12ns) and every
        # folder is written keyed by its own camera's timestamps in the same chronological order -
        # safe to zip by sorted-filename index. Frame counts can still differ though: cmd_rectify
        # only writes a depth/mask frame when ADT's ground truth is actually valid for that
        # timestamp, and on some sequences (e.g. the "clean" variants, no skeleton) that ground
        # truth coverage ends earlier than the raw video - rgb_0/rgb_1 keep going past that point
        # with no matching depth/mask. zip() below (no strict=True) truncates to the shortest list,
        # i.e. drops those trailing GT-less frames instead of crashing.
        files0 = sorted(self.rgb_path(sequence_name).iterdir(), key=lambda p: int(p.stem))
        files1 = sorted((self.sequence_path(sequence_name) / "rgb_1").iterdir(), key=lambda p: int(p.stem))
        filesd = sorted(self.depth_path(sequence_name).iterdir(), key=lambda p: int(p.stem))
        mask0_path = self.sequence_path(sequence_name) / f"{_MASK_FOLDER_BASE}_0"
        mask1_path = self.sequence_path(sequence_name) / f"{_MASK_FOLDER_BASE}_1"
        filesm0 = sorted(mask0_path.iterdir(), key=lambda p: int(p.stem))
        filesm1 = sorted(mask1_path.iterdir(), key=lambda p: int(p.stem))

        counts = {"rgb_0": len(files0), "rgb_1": len(files1), "depth_0": len(filesd),
                  f"{_MASK_FOLDER_BASE}_0": len(filesm0), f"{_MASK_FOLDER_BASE}_1": len(filesm1)}
        if len(set(counts.values())) > 1:
            print_warning(
                f"{sequence_name}: frame counts differ across streams {counts} - ADT's "
                f"depth/segmentation ground truth doesn't cover the full video for this sequence; "
                f"truncating to the first {min(counts.values())} frames common to all streams."
            )

        header = [
            "ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1",
            "ts_mask_0 (ns)", "path_mask_0", "ts_mask_1 (ns)", "path_mask_1",
            "ts_depth_0 (ns)", "path_depth_0",
        ]
        rows = []
        for f0, f1, fm0, fm1, fd in zip(files0, files1, filesm0, filesm1, filesd):
            rows.append([
                int(f0.stem), f"rgb_0/{f0.name}", int(f1.stem), f"rgb_1/{f1.name}",
                int(fm0.stem), f"{_MASK_FOLDER_BASE}_0/{fm0.name}", int(fm1.stem), f"{_MASK_FOLDER_BASE}_1/{fm1.name}",
                int(fd.stem), f"depth_0/{fd.name}",
            ])
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        out_json = self.sequence_path(sequence_name) / ".adt_calibration.json"
        width, height = self.target_resolution
        try:
            self._run_worker(
                "calibration",
                "--sequence_path", str(self.sequence_path(sequence_name)),
                "--imu_label", _IMU_LABEL,
                "--out_json", str(out_json),
                "--width", str(width),
                "--height", str(height),
            )
            data = json.loads(out_json.read_text(encoding="utf-8"))
        finally:
            out_json.unlink(missing_ok=True)

        rgbd0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": data["rgb_0"]["focal_length"],
            "principal_point": data["rgb_0"]["principal_point"],
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.array(data["rgb_0"]["T_device_camera"]),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": data["rgb_1"]["focal_length"],
            "principal_point": data["rgb_1"]["principal_point"],
            "fps": float(self.rgb_hz),
            "T_BS": np.array(data["rgb_1"]["T_device_camera"]),
        }

        # Aria's factory IMU noise-density spec isn't exposed via projectaria_tools' calibration
        # API - fall back to the same generic saturation/noise defaults already used dataset-wide
        # for IMUs without a documented range (dataset_madmax.py, dataset_euroc.py).
        imu0: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": 20.0e-4,
            "sigma_a_c": 20.0e-3,
            "sigma_bg": 0.01,
            "sigma_ba": 0.1,
            "sigma_gw_c": 20.0e-5,
            "sigma_aw_c": 20.0e-3,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": float(data["imu_0"]["fps"]),
            "T_BS": np.array(data["imu_0"]["T_device_imu"]),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb1], rgbd=[rgbd0], imu=[imu0])

    def create_imu_csv(self, sequence_name: str) -> None:
        self._run_worker(
            "imu",
            "--sequence_path", str(self.sequence_path(sequence_name)),
            "--imu_label", _IMU_LABEL,
            "--out_csv", str(self.imu_csv_path(sequence_name)),
        )

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        self._run_worker(
            "groundtruth",
            "--sequence_path", str(self.sequence_path(sequence_name)),
            "--out_csv", str(self.groundtruth_csv_path(sequence_name)),
        )

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: the raw VRS/MPS files are
        # user-placed local data fetched via ADT's own license-gated downloader - there is no
        # re-download path to recover from deleting them (same rationale as dataset_malaysia_jul2026.py).
        return

    def _run_worker(self, *args: str) -> None:
        script_path = self.dataset_path / ".adt_worker.py"
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(_WORKER_SCRIPT, encoding="utf-8")
        try:
            cmd_parts = ["pixi", "run", "-e", "projectaria", "python3", shlex.quote(str(script_path))]
            cmd_parts += [shlex.quote(a) for a in args]
            subprocess.run(" ".join(cmd_parts), shell=True, check=True, cwd=str(VSLAM_LAB_DIR))
        finally:
            script_path.unlink(missing_ok=True)
