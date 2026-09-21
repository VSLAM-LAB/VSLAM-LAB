"""
Module: VSLAM-LAB - Datasets - dataset_blt.py
- Author: Alejandro Fontan
- Assisted by: Claude (Opus 5)
- Version: 1.0
- Created: 2026-09-22
- License: GPLv3 License

Seeded by Riccardo Polvara's own BLT integration (VSLAM-LAB PR #53).
"""

from __future__ import annotations

import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import compute_scaled_size, make_printers, scale_intrinsics, write_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

# Front ZED2 topics (the side-facing ZED2 publishes the same shapes under /side/... and
# /depth_republish_2/, and is deliberately not used - see the yaml's modes comment). Colour is
# rectified and JPEG-compressed; depth is republished through compressed_depth_image_transport.
RGB_TOPIC: Final = "/front/zed_node/rgb/image_rect_color/compressed"
DEPTH_TOPIC: Final = "/depth_republish/compressedDepth"
# The colour stream's CameraInfo, under the names the ZED wrapper has published it as - the first
# one a session's bag carries is the one used.
CAMERA_INFO_TOPICS: Final = (
    "/front/zed_node/rgb/camera_info",
    "/front/zed_node/left/camera_info",
    "/front/zed_node/rgb/image_rect_color/camera_info",
)
# Ground truth: the robot_localization EKF fusing wheel odometry with the Trimble BX992 RTK-GNSS
# (2-3 cm, common datum for every session), as a pose of base_link in the map frame.
GROUNDTRUTH_TOPIC: Final = "/odometry/gps"
TF_STATIC_TOPIC: Final = "/tf_static"

# groundtruth.csv is written in the robot's body frame, so calibration.yaml's T_BS has to carry
# base_link -> camera (see _camera_extrinsics for where each half of that comes from).
BODY_FRAME: Final = "base_link"

# Colour runs faster than depth, so the two streams are associated by nearest timestamp rather
# than zipped by index, and a colour frame keeps whatever depth frame falls within this much of
# it. Pairing every colour frame (rather than every depth frame) keeps mono mode, which reads the
# same rgb.csv, at the full colour rate; the cost is that consecutive rows may reuse one depth
# frame, which create_rgb_csv reports. 100 ms is one period of the ~10 Hz depth stream: measured
# on ktima_2022_09_15, a colour frame's nearest depth frame is 0 ms away for 68% of frames and
# 67 ms away for almost all of the rest, so this keeps 99.7% of them while still refusing a pair
# straddling a depth dropout (the stream's largest gap there was 201 ms).
RGB_DEPTH_TOLERANCE_NS: Final = 100_000_000

# compressed_depth_image_transport's ConfigHeader, prepended to every compressedDepth payload:
# int32 format enum + float32 depthQuantA + float32 depthQuantB.
_DEPTH_HEADER_FORMAT: Final = "<iff"
_DEPTH_HEADER_SIZE: Final = struct.calcsize(_DEPTH_HEADER_FORMAT)

# base_link -> front ZED2, from Table 3 of the BLT paper (translation in m, rotation as
# qx, qy, qz, qw). The bag's own static TF cannot supply this: the ZED wrapper publishes its
# sub-tree unmounted, with base_link -> front_base_link as the identity (verified on
# ktima_2022_09_15), which would put the camera 1.5 cm above the robot's ground-level body origin
# instead of its real 0.763 m. What the bag does carry correctly is everything from the camera
# body inwards, so _camera_extrinsics chains this transform with the bag's own
# MOUNT_FRAME -> <colour lens optical frame> part.
MOUNT_FRAME: Final = "front_camera_center"
_MOUNT_TRANSLATION: Final = (0.345, 0.060, 0.763)
_MOUNT_QUATERNION: Final = (0.000, 0.017, 0.000, 1.000)
# REP-103 camera body frame (x forward, y left, z up) -> optical frame (x right, y down,
# z forward). Only needed if a bag is missing the intra-camera chain, since the ZED wrapper
# publishes exactly this rotation (plus the lens offsets) under MOUNT_FRAME.
_BODY_TO_OPTICAL: Final = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

_SEQUENCE_NAME_RE: Final = re.compile(r"^ktima_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})$")


def _clean_frame(frame: str) -> str:
    """TF frame id without ROS1's optional leading slash."""
    return str(frame or "").strip().lstrip("/")


def _stamp_ns(header: Any) -> int:
    """A ROS message header's own capture stamp, in nanoseconds.

    Header stamps, not the bag's record times: recording lags capture badly here, and unequally
    per topic. On ktima_2022_09_15 a colour frame was written a median 93 ms after its stamp and
    a depth frame - which arrives through a republisher - a median 233 ms after, with both
    exceeding 1.5 s at the 95th percentile, so record times would both mis-time every frame and
    mis-pair the two streams.
    """
    stamp = header.stamp
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    return seconds * 1_000_000_000 + nanoseconds


def _fields(value: Any, names: str) -> tuple[float, ...]:
    """The named components of a ROS vector/quaternion message, or of a plain (x, y, z[, w])
    sequence - both spellings are used here, messages from a bag and constants from the paper."""
    if all(hasattr(value, name) for name in names):
        return tuple(float(getattr(value, name)) for name in names)
    return tuple(float(v) for v in value)


def _matrix(translation: Any, quaternion: Any) -> np.ndarray:
    """4x4 homogeneous transform from a translation and a (qx, qy, qz, qw) quaternion."""
    x, y, z, w = _fields(quaternion, "xyzw")
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Zero-length quaternion in a BLT transform")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = _fields(translation, "xyz")
    return matrix


def _decode_compressed_color(msg: Any) -> np.ndarray:
    """A sensor_msgs/CompressedImage colour frame as a BGR array."""
    image = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode a BLT colour frame (format '{getattr(msg, 'format', '')}')")
    return image


def _decode_compressed_depth(msg: Any) -> np.ndarray:
    """A compressedDepth frame as 16-bit millimetres.

    compressed_depth_image_transport encodes the two depth image types differently, and which one
    a bag carries is stated by the message's own `format` field ("16UC1; compressedDepth png" vs
    "32FC1; compressedDepth png"): a 16UC1 image is PNG-compressed as-is and already holds
    millimetres, while a 32FC1 image (metres) is inverse-quantized into 16 bits first, recovered
    here as depthQuantA / (pixel - depthQuantB) using the parameters in the payload's own header.
    """
    image_format = str(getattr(msg, "format", "") or "").lower()
    payload = bytes(msg.data)
    _, quant_a, quant_b = struct.unpack(_DEPTH_HEADER_FORMAT, payload[:_DEPTH_HEADER_SIZE])
    raw = cv2.imdecode(np.frombuffer(payload[_DEPTH_HEADER_SIZE:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not decode a BLT depth frame (format '{image_format}')")

    if "32fc1" in image_format:
        depth_mm = np.zeros(raw.shape, dtype=np.float32)
        valid = raw > 0
        depth_mm[valid] = 1000.0 * quant_a / (raw[valid].astype(np.float32) - quant_b)
        return np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)

    if "16uc1" not in image_format:
        print_warning(f"Unexpected BLT depth encoding '{image_format}' - reading it as 16-bit millimetres")
    return raw.astype(np.uint16)


def _fresh_dir(path: Path) -> Path:
    """An empty '<path>.tmp' beside path - built into, then renamed onto path once complete, so a
    crash midway can never leave a partial folder that later looks finished."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    return tmp


class BltDataset(DatasetVSLAMLAB):
    """BACCHUS Long-Term (BLT) vineyard dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "blt") -> None:
        super().__init__(dataset_name)

        # Every sequence is local (scalar in the yaml): the session bags are only handed out
        # through the dataset page's request form, so raw_data_path is where the user keeps their
        # own copy - defaulting to a 'raw' folder inside this dataset's benchmark directory.
        self.sequence_location: str = self.cfg["sequence_location"]
        raw_data_path = str(self.cfg.get("raw_data_path") or "")
        self.raw_data_path: Path = Path(raw_data_path).expanduser() if raw_data_path else self.dataset_path / "raw"

        # Depth factor
        self.depth_factor: float = float(self.cfg["depth_factor"])

        # Sequence nicknames - every session is a Ktima one, so only the date distinguishes them
        # (ktima_2022_03_23 -> 2022-03-23).
        self.sequence_nicknames = [s.removeprefix("ktima_").replace("_", "-") for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        bag_link = self._bag_link(sequence_name)
        if bag_link.is_symlink() or bag_link.exists():
            return

        source_bag = self._find_source_bag(sequence_name)
        if source_bag is None:
            print_info(
                f"Sequence '{sequence_name}' is marked as 'local'. No bag for session "
                f"{self._session_date(sequence_name)} was found under {self.raw_data_path} - request the BLT "
                f"data at {self.cfg['about']['homepage']}, then put the session bags there or point "
                f"raw_data_path in dataset_{self.dataset_name}.yaml at your copy."
            )
            return

        self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
        # Absolute target on purpose: the bags are tens of GB and usually live outside the
        # benchmark folder, so a relative link would break if either side were moved.
        os.symlink(source_bag.resolve(), bag_link)

    def create_rgb_folder(self, sequence_name: str) -> None:
        rgb_path, depth_path = self.rgb_path(sequence_name), self.depth_path(sequence_name)
        if rgb_path.is_dir() and depth_path.is_dir():
            return

        # Both streams are extracted in one pass: a session bag is tens of GB, so reading it twice
        # costs far more than the two decoders cost.
        bag_path = self._bag_path(sequence_name)
        rgb_tmp, depth_tmp = _fresh_dir(rgb_path), _fresh_dir(depth_path)
        from rosbags.highlevel import AnyReader

        with AnyReader([bag_path]) as reader:
            connections = self._connections(reader, [RGB_TOPIC, DEPTH_TOPIC], sequence_name)
            messages = reader.messages(connections=connections)
            for connection, _, rawdata in tqdm(messages, total=sum(c.msgcount for c in connections),
                                               desc=f"    extracting {rgb_path.name}/ + {depth_path.name}/"):
                msg = reader.deserialize(rawdata, connection.msgtype)
                name = f"{_stamp_ns(msg.header)}.png"
                if connection.topic == RGB_TOPIC:
                    self._save_color(_decode_compressed_color(msg), rgb_tmp / name)
                else:
                    self._save_depth(_decode_compressed_depth(msg), depth_tmp / name)

        rgb_tmp.rename(rgb_path)
        depth_tmp.rename(depth_path)

    def create_rgb_csv(self, sequence_name: str) -> None:
        """Associate colour and depth by nearest timestamp, within one colour period.

        The two streams are timestamped independently and, per the dataset's sensor table, are not
        even configured at the same rate (15 fps colour against 10 fps depth), so a naive
        index-zip would pair frames from different moments - see dataset_rgbdtum.py, which needs
        the same treatment for TUM's Kinect.
        """
        rgb_csv = self.rgb_csv_path(sequence_name)
        if rgb_csv.exists():
            return

        rgb = self._stream_frame(self.rgb_path(sequence_name), "path_rgb_0")
        depth = self._stream_frame(self.depth_path(sequence_name), "path_depth_0")

        merged = pd.merge_asof(rgb, depth, on="ts", direction="nearest", tolerance=RGB_DEPTH_TOLERANCE_NS)
        merged = merged.dropna(subset=["path_depth_0"]).copy()
        dropped = len(rgb) - len(merged)
        if dropped:
            print_warning(f"{sequence_name}: {dropped} of {len(rgb)} colour frames have no depth frame within "
                          f"{RGB_DEPTH_TOLERANCE_NS / 1e6:.0f} ms - dropped")

        # Back from the matched file name rather than through the merge: a nanosecond stamp needs
        # more precision than the float64 an all-NaN-capable merge column would carry.
        merged["ts_depth_0 (ns)"] = [int(Path(path).stem) for path in merged["path_depth_0"]]
        reused = len(merged) - merged["ts_depth_0 (ns)"].nunique()
        if reused:
            print_info(f"{sequence_name}: {reused} rows reuse a depth frame already paired with an earlier colour "
                       f"frame (depth is recorded at a lower rate than colour)")

        merged = merged.rename(columns={"ts": "ts_rgb_0 (ns)"})
        header = ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"]
        write_csv_rows(rgb_csv, header, merged[header].values.tolist())

    def create_calibration_yaml(self, sequence_name: str) -> None:
        camera_info, transforms = self._read_calibration_messages(sequence_name)

        distortion = [float(v) for v in getattr(camera_info, "d", getattr(camera_info, "D", []))]
        if any(abs(v) > 1e-9 for v in distortion):
            print_warning(f"{sequence_name}: the colour CameraInfo reports non-zero distortion {distortion}, but "
                          f"{RGB_TOPIC} is a rectified stream written as a pinhole camera - check the topic names")

        # K of the rectified stream, given at the bag's native frame size; rgb_0/depth_0 are
        # resized by create_rgb_folder, so rescale to match (VSLAM-LAB issue #99).
        k = [float(v) for v in getattr(camera_info, "k", getattr(camera_info, "K", []))]
        if len(k) != 9:
            raise ValueError(f"{sequence_name}: the colour CameraInfo carries no 3x3 K matrix")
        native_size = (int(camera_info.width), int(camera_info.height))
        self._check_native_resolution(sequence_name, native_size)
        focal_length, principal_point = scale_intrinsics(
            (k[0], k[4]), (k[2], k[5]), native_size, self.target_resolution
        )

        rgbd0: dict[str, Any] = {
            "cam_name": self.rgb_path(sequence_name).name,
            "cam_type": "rgb+depth",
            "depth_name": self.depth_path(sequence_name).name,
            "cam_model": "pinhole",
            "focal_length": focal_length,
            "principal_point": principal_point,
            "depth_factor": float(self.depth_factor),
            "fps": float(self.rgb_hz),
            "T_BS": self._camera_extrinsics(sequence_name, camera_info, transforms),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgbd0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        # /odometry/gps is a nav_msgs/Odometry pose of base_link in the session's map frame (all
        # sessions share one datum), so its poses are written straight through: this is the body
        # frame calibration.yaml's T_BS is relative to. It comes from robot_localization fusing
        # wheel odometry with the RTK-GNSS and is planar - on ktima_2022_09_15 every pose has
        # z = 0 exactly, with roll and pitch left at zero - and the EKF repeats a pose verbatim
        # every few messages (123 of 577 in the scanned window), which _odometry_rows drops.
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        header = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
        if groundtruth_csv.exists():
            return

        from rosbags.highlevel import AnyReader

        rows: list[list[Any]] = []
        child_frames: set[str] = set()
        repeated = 0
        with AnyReader([self._bag_path(sequence_name)]) as reader:
            connections = [c for c in reader.connections if c.topic == GROUNDTRUTH_TOPIC]
            if not connections:
                # Still write the file, with its header and no rows, rather than leaving the
                # sequence without one.
                print_warning(f"{sequence_name}: {GROUNDTRUTH_TOPIC} is not in this bag - writing an empty "
                              f"groundtruth.csv")
                write_csv_rows(groundtruth_csv, header, [])
                return
            messages = reader.messages(connections=connections)
            for connection, _, rawdata in tqdm(messages, total=sum(c.msgcount for c in connections),
                                               desc=f"    reading {GROUNDTRUTH_TOPIC}"):
                msg = reader.deserialize(rawdata, connection.msgtype)
                child_frames.add(_clean_frame(getattr(msg, "child_frame_id", "")))
                position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
                row = [_stamp_ns(msg.header),
                       float(position.x), float(position.y), float(position.z),
                       float(orientation.x), float(orientation.y),
                       float(orientation.z), float(orientation.w)]
                # The EKF republishes a pose under its original stamp; keeping those would leave
                # groundtruth.csv with repeated timestamps, so only the first of each is kept.
                if rows and row[0] <= rows[-1][0]:
                    repeated += 1
                    continue
                rows.append(row)

        if repeated:
            print_info(f"{sequence_name}: dropped {repeated} {GROUNDTRUTH_TOPIC} messages that repeat a timestamp "
                       f"already written")
        unexpected = {frame for frame in child_frames if frame and frame != BODY_FRAME}
        if unexpected:
            print_warning(f"{sequence_name}: {GROUNDTRUTH_TOPIC} reports poses of {sorted(unexpected)} rather than "
                          f"'{BODY_FRAME}' - calibration.yaml's T_BS assumes the latter")
        write_csv_rows(groundtruth_csv, header, rows)

    def remove_unused_files(self, sequence_name: str) -> None:
        # Deliberate no-op at every retention tier, including MINIMAL: the sequence's .bag is a
        # symlink onto the user's own copy of the data (there is no URL to re-download it from),
        # and no other intermediate file is written.
        return

    ####################################################################################################################
    # Auxiliary methods

    def _session_date(self, sequence_name: str) -> str:
        """The recording date a sequence name encodes, as it appears inside the bag's file name:
        'ktima_2022_09_15' -> '2022-09-15'."""
        match = _SEQUENCE_NAME_RE.match(sequence_name)
        if match is None:
            raise ValueError(f"Unknown {self.dataset_name} sequence '{sequence_name}' - "
                             f"expected ktima_<yyyy>_<mm>_<dd>")
        return f"{match['year']}-{match['month']}-{match['day']}"

    def _bag_link(self, sequence_name: str) -> Path:
        """Where download_sequence_data links this session's bag to."""
        return self.sequence_path(sequence_name) / f"{sequence_name}.bag"

    def _bag_path(self, sequence_name: str) -> Path:
        """The linked bag, checked - the link is only created once the user's own copy of the data
        has been found, so every later hook needs it to be there."""
        bag_link = self._bag_link(sequence_name)
        if not bag_link.exists():
            raise FileNotFoundError(
                f"No bag at {bag_link} (sequence marked as 'local'): run download_sequence_data with your copy of "
                f"the BLT session bags in place, and keep it reachable while the sequence is processed."
            )
        return bag_link

    def _find_source_bag(self, sequence_name: str) -> Path | None:
        """This session's bag under raw_data_path, found by the date in its file name (e.g.
        rosbag_compressed_2022-09-15-14-23-20.bag). Searched recursively, so the delivered
        NN_month/ folders and a flat dump of bags both work."""
        if not self.raw_data_path.is_dir():
            return None
        date = self._session_date(sequence_name)
        candidates = sorted(p for p in self.raw_data_path.rglob("*.bag") if date in p.name)
        if len(candidates) > 1:
            print_warning(f"{sequence_name}: {len(candidates)} bags carry the date {date} under {self.raw_data_path} "
                          f"- using {candidates[0]}")
        return candidates[0] if candidates else None

    def _connections(self, reader: Any, topics: list[str], sequence_name: str) -> list[Any]:
        """The reader's connections for the requested topics, failing with the bag's own topic list
        rather than silently extracting nothing."""
        connections = [c for c in reader.connections if c.topic in set(topics)]
        missing = set(topics) - {c.topic for c in connections}
        if missing:
            available = ", ".join(sorted({c.topic for c in reader.connections}))
            raise ValueError(f"{sequence_name}: topics {sorted(missing)} are not in {self._bag_link(sequence_name)}. "
                             f"Available topics: {available}")
        return connections

    def _save_color(self, image: np.ndarray, path: Path) -> None:
        """One decoded BGR frame, resized to target_resolution and written as PNG."""
        frame = Image.fromarray(image[:, :, ::-1])
        if self.target_resolution is not None:
            frame = frame.resize(compute_scaled_size(frame.size, self.target_resolution), Image.Resampling.LANCZOS)
        frame.save(path)

    def _save_depth(self, depth: np.ndarray, path: Path) -> None:
        """One decoded 16-bit millimetre depth map, resized to match rgb_0 and written as a 16-bit
        PNG. Nearest-neighbour only - any interpolating resample blends depth across object
        boundaries and corrupts the metric data - and through cv2, which reads and writes 16-bit
        PNGs natively where PIL's I;16 mode does not resize reliably."""
        if self.target_resolution is not None:
            height, width = depth.shape[:2]
            target_size = compute_scaled_size((width, height), self.target_resolution)
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(path), depth):
            raise RuntimeError(f"Could not write BLT depth frame: {path}")

    @staticmethod
    def _stream_frame(folder: Path, path_column: str) -> pd.DataFrame:
        """One extracted stream as a timestamp-sorted (ts, <path_column>) frame - file names are
        the frames' own header stamps in nanoseconds."""
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing extracted frames: {folder}")
        frames = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
        data = pd.DataFrame({"ts": [int(p.stem) for p in frames],
                             path_column: [f"{folder.name}/{p.name}" for p in frames]})
        return data.sort_values("ts").reset_index(drop=True)

    def _read_calibration_messages(self, sequence_name: str) -> tuple[Any, dict[str, tuple[str, np.ndarray]]]:
        """The first CameraInfo of the colour stream, plus the bag's static TF tree as
        child frame -> (parent frame, T_parent_child). Both are constant over a session, so only
        the messages needed to build calibration.yaml are read."""
        from rosbags.highlevel import AnyReader

        camera_info = None
        transforms: dict[str, tuple[str, np.ndarray]] = {}
        with AnyReader([self._bag_path(sequence_name)]) as reader:
            camera_info_topic = next((t for t in CAMERA_INFO_TOPICS
                                      if any(c.topic == t for c in reader.connections)), None)
            if camera_info_topic is None:
                available = ", ".join(sorted({c.topic for c in reader.connections}))
                raise ValueError(f"{sequence_name}: none of the colour CameraInfo topics {list(CAMERA_INFO_TOPICS)} "
                                 f"is in {self._bag_link(sequence_name)}. Available topics: {available}")

            wanted = {camera_info_topic, TF_STATIC_TOPIC}
            connections = [c for c in reader.connections if c.topic in wanted]
            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic == camera_info_topic:
                    camera_info = camera_info or msg
                    continue
                for transform in msg.transforms:
                    child = _clean_frame(transform.child_frame_id)
                    transforms[child] = (_clean_frame(transform.header.frame_id),
                                         _matrix(transform.transform.translation, transform.transform.rotation))

        if camera_info is None:
            raise ValueError(f"{sequence_name}: {camera_info_topic} carries no message")
        return camera_info, transforms

    def _camera_extrinsics(self, sequence_name: str, camera_info: Any,
                           transforms: dict[str, tuple[str, np.ndarray]]) -> np.ndarray:
        """T_BS for the colour camera: the pose, in the robot's body frame, of the frame its
        CameraInfo names.

        Built from the two sources that are each right about half of it (see MOUNT_FRAME above):
        the paper's published mounting of the front ZED2 on the robot, then the bag's own static
        chain from that camera body to the colour lens' optical frame - which carries the lens
        offsets (60 mm off the camera centre) and the body-to-optical rotation.
        """
        mounting = _matrix(_MOUNT_TRANSLATION, _MOUNT_QUATERNION)
        camera_frame = _clean_frame(getattr(getattr(camera_info, "header", None), "frame_id", ""))
        inside_camera = self._static_transform(transforms, MOUNT_FRAME, camera_frame)
        if inside_camera is None:
            print_warning(f"{sequence_name}: no static TF chain from '{MOUNT_FRAME}' to the camera frame "
                          f"'{camera_frame}' - placing the camera at the mounting published in the BLT paper, "
                          f"without this ZED2's own lens offsets")
            inside_camera = np.eye(4)
            inside_camera[:3, :3] = _BODY_TO_OPTICAL
        return mounting @ inside_camera

    @staticmethod
    def _static_transform(transforms: dict[str, tuple[str, np.ndarray]],
                          source: str, target: str) -> np.ndarray | None:
        """T_source_target through a static TF tree, or None if the two frames have no common
        root. Both frames are walked up to their root and the two chains composed, which covers
        any tree shape without needing source to be an ancestor of target."""
        if not source or not target:
            return None

        def to_root(frame: str) -> tuple[str, np.ndarray]:
            transform, visited = np.eye(4), {frame}
            while frame in transforms:
                parent, T_parent_frame = transforms[frame]
                if parent in visited:
                    break
                transform = T_parent_frame @ transform
                frame = parent
                visited.add(frame)
            return frame, transform  # (root, T_root_frame)

        source_root, T_root_source = to_root(source)
        target_root, T_root_target = to_root(target)
        if source_root != target_root:
            return None
        return np.linalg.inv(T_root_source) @ T_root_target

    def _check_native_resolution(self, sequence_name: str, native_size: tuple[int, int]) -> None:
        """Warn if rgb_0's frames are not the size the CameraInfo resolution scales to - the
        written intrinsics would then describe the wrong image size (#99)."""
        rgb_path = self.rgb_path(sequence_name)
        frames = sorted(p for p in rgb_path.iterdir() if p.suffix.lower() == ".png") if rgb_path.is_dir() else []
        if not frames:
            return
        expected_size = compute_scaled_size(native_size, self.target_resolution)
        with Image.open(frames[0]) as img:
            if img.size != expected_size:
                print_warning(f"{sequence_name}: {rgb_path.name}/{frames[0].name} is {img.size}, expected "
                              f"{expected_size} from a native {native_size} - intrinsics may be scaled for the "
                              f"wrong resolution")
