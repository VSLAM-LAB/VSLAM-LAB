"""
Module: VSLAM-LAB - Datasets - dataset_aria_digital_twin.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-14
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

# projectaria_tools (VRS parsing, ADT ground-truth API, fisheye-to-pinhole rectification) is only
# installable in the separate "projectaria" pixi environment (Python 3.12) - the main "vslamlab"
# environment this file itself runs under (Python 3.13) never has it. Every hook below that needs
# it shells out via `pixi run -e projectaria python3 <this worker script>`, mirroring
# utilities.run_rosbag_frame_extraction's existing pattern for bridging into the ros1/ros2 pixi
# environments. Filed as a capability gap (add-dataset's Issue exception) rather than worked around
# by hand-rolled VRS parsing: see the "aria_digital_twin: no per-dataset processing environment"
# issue this run filed.
_WORKER_SCRIPT = '''
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider


def _open_gt(sequence_path):
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    data_paths = paths_provider.get_datapaths(True)
    return AriaDigitalTwinDataProvider(data_paths)


def cmd_rectify(args):
    gt = _open_gt(args.sequence_path)
    raw_provider = gt.raw_data_provider_ptr()
    out_dirs = {"camera-slam-left": Path(args.out_rgb0), "camera-slam-right": Path(args.out_rgb1)}
    out_depth = Path(args.out_depth0)

    for label, out_dir in out_dirs.items():
        sid = raw_provider.get_stream_id_from_label(label)
        raw_calib = gt.get_aria_camera_calibration(sid)
        focal_length = float(raw_calib.get_projection_params()[0])
        lin_calib = calibration.get_linear_camera_calibration(
            args.width, args.height, focal_length, label, raw_calib.get_transform_device_camera()
        )
        write_depth = label == "camera-slam-left"

        for ts in gt.get_aria_device_capture_timestamps_ns(sid):
            out_path = out_dir / f"{ts}.png"
            depth_path = out_depth / f"{ts}.png"
            need_image = not out_path.exists()
            need_depth = write_depth and not depth_path.exists()
            if not need_image and not need_depth:
                continue

            if need_image:
                image_result = gt.get_aria_image_by_timestamp_ns(ts, sid)
                if image_result.is_valid():
                    raw_image = image_result.data().to_numpy_array()
                    rectified = calibration.distort_by_calibration(
                        raw_image, lin_calib, raw_calib, InterpolationMethod.BILINEAR
                    )
                    Image.fromarray(rectified).save(out_path)

            if need_depth:
                depth_result = gt.get_depth_image_by_timestamp_ns(ts, sid)
                if depth_result.is_valid():
                    raw_depth = depth_result.data().to_numpy_array()
                    rectified_depth = calibration.distort_by_calibration(
                        raw_depth, lin_calib, raw_calib, InterpolationMethod.NEAREST_NEIGHBOR
                    )
                    Image.fromarray(rectified_depth).save(depth_path)


def cmd_calibration(args):
    gt = _open_gt(args.sequence_path)
    raw_provider = gt.raw_data_provider_ptr()
    device_calib = raw_provider.get_device_calibration()

    out = {}
    for key, label in (("rgb_0", "camera-slam-left"), ("rgb_1", "camera-slam-right")):
        sid = raw_provider.get_stream_id_from_label(label)
        raw_calib = gt.get_aria_camera_calibration(sid)
        out[key] = {
            "focal_length": float(raw_calib.get_projection_params()[0]),
            "T_device_camera": raw_calib.get_transform_device_camera().to_matrix().tolist(),
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

    rows = []
    for ts in gt.get_aria_device_capture_timestamps_ns(sid):
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
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.set_defaults(func=cmd_rectify)

    p = sub.add_parser("calibration")
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--imu_label", required=True)
    p.add_argument("--out_json", required=True)
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

    def __init__(self, dataset_name: str = "aria_digital_twin") -> None:
        super().__init__(dataset_name)

        # All sequences are user-placed local data - fetched by the user with the official
        # aria_dataset_downloader (projectaria_tools) after signing ADT's license agreement.
        self.sequence_location = self.cfg["sequence_location"]
        self.depth_factor = self.cfg["depth_factor"]

    def download_sequence_data(self, sequence_name: str) -> None:
        print_info(
            f"Sequence '{sequence_name}' is marked as 'local'. Please fetch it with the official "
            f"aria_dataset_downloader (projectaria_tools) and ensure it is available at "
            f"{self.sequence_path(sequence_name)}."
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb0_path = self.rgb_path(sequence_name)
        rgb1_path = self.sequence_path(sequence_name) / "rgb_1"
        depth0_path = self.depth_path(sequence_name)
        rgb0_path.mkdir(parents=True, exist_ok=True)
        rgb1_path.mkdir(parents=True, exist_ok=True)
        depth0_path.mkdir(parents=True, exist_ok=True)

        width, height = self.target_resolution
        self._run_worker(
            "rectify",
            "--sequence_path", str(self.sequence_path(sequence_name)),
            "--out_rgb0", str(rgb0_path),
            "--out_rgb1", str(rgb1_path),
            "--out_depth0", str(depth0_path),
            "--width", str(width),
            "--height", str(height),
        )

    def create_rgb_csv(self, sequence_name: str) -> None:
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        # camera-slam-left/right are hardware-synced (capture timestamps differ by ~12ns) and
        # depth_0 was written keyed by camera-slam-left's own timestamps - all three folders have
        # the same frame count in the same order, safe to zip by sorted-filename index.
        files0 = sorted(self.rgb_path(sequence_name).iterdir(), key=lambda p: int(p.stem))
        files1 = sorted((self.sequence_path(sequence_name) / "rgb_1").iterdir(), key=lambda p: int(p.stem))
        filesd = sorted(self.depth_path(sequence_name).iterdir(), key=lambda p: int(p.stem))

        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1", "ts_depth_0 (ns)", "path_depth_0"]
        rows = []
        for f0, f1, fd in zip(files0, files1, filesd, strict=True):
            rows.append([int(f0.stem), f"rgb_0/{f0.name}", int(f1.stem), f"rgb_1/{f1.name}", int(fd.stem), f"depth_0/{fd.name}"])
        write_csv_rows(rgb_csv, header, rows)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        out_json = self.sequence_path(sequence_name) / ".adt_calibration.json"
        try:
            self._run_worker(
                "calibration",
                "--sequence_path", str(self.sequence_path(sequence_name)),
                "--imu_label", _IMU_LABEL,
                "--out_json", str(out_json),
            )
            data = json.loads(out_json.read_text(encoding="utf-8"))
        finally:
            out_json.unlink(missing_ok=True)

        width, height = self.target_resolution
        principal_point = [width / 2.0, height / 2.0]

        rgbd0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray+depth",
            "depth_name": "depth_0",
            "cam_model": "pinhole",
            "focal_length": [data["rgb_0"]["focal_length"], data["rgb_0"]["focal_length"]],
            "principal_point": principal_point,
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": np.array(data["rgb_0"]["T_device_camera"]),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": [data["rgb_1"]["focal_length"], data["rgb_1"]["focal_length"]],
            "principal_point": principal_point,
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
        # re-download path to recover from deleting them (same rationale as dataset_malaysia.py).
        return

    def _run_worker(self, *args: str) -> None:
        script_path = self.dataset_path / ".adt_worker.py"
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(_WORKER_SCRIPT, encoding="utf-8")
        try:
            cmd = " ".join(["pixi", "run", "-e", "projectaria", "python3", shlex.quote(str(script_path)), *[shlex.quote(a) for a in args]])
            subprocess.run(cmd, shell=True, check=True, cwd=str(VSLAM_LAB_DIR))
        finally:
            script_path.unlink(missing_ok=True)
