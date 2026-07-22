from __future__ import annotations

import csv
import os
import re
import shutil
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab


class BLT_dataset(DatasetVSLAMLab):
    """BLT ktima local rosbag dataset helper."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "blt") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "dataset_folder" in cfg:
            self.dataset_folder = str(cfg["dataset_folder"])
            self.dataset_path = self.benchmark_path / self.dataset_folder
        self.sequence_nicknames = [s.replace("_", " ") for s in self.sequence_names]
        modes_override = os.environ.get(cfg.get("modes_env", "BLT_MODES"), "")
        if modes_override:
            self.modes = [
                mode.strip()
                for mode in modes_override.replace(";", ",").split(",")
                if mode.strip()
            ]
        self.source_root_env = cfg.get("source_root_env", "BLT_KTIMA_ROOT")
        source_root_value = os.environ.get(self.source_root_env, cfg.get("source_root_default", ""))
        self.source_root = Path(source_root_value).expanduser() if source_root_value else None
        self.source_bags = dict(cfg["source_bags"])
        self.image_topic = os.environ.get(cfg["image_topic_env"], cfg["image_topic"])
        self.camera_info_topics = list(cfg.get("camera_info_topics", []))
        camera_info_override = os.environ.get(cfg.get("camera_info_topic_env", "BLT_CAMERA_INFO_TOPIC"), "")
        if camera_info_override:
            self.camera_info_topics = [camera_info_override]
        self.depth_topic = os.environ.get(
            cfg.get("depth_topic_env", "BLT_DEPTH_TOPIC"),
            cfg.get("depth_topic", ""),
        )
        self.depth_camera_info_topics = list(cfg.get("depth_camera_info_topics", []))
        depth_camera_info_override = os.environ.get(
            cfg.get("depth_camera_info_topic_env", "BLT_DEPTH_CAMERA_INFO_TOPIC"),
            "",
        )
        if depth_camera_info_override:
            self.depth_camera_info_topics = [depth_camera_info_override]
        self.depth_factor = float(
            os.environ.get(
                cfg.get("depth_factor_env", "BLT_DEPTH_FACTOR"),
                cfg.get("depth_factor", 1000.0),
            )
        )
        self.image_transport = cfg.get("image_transport", "raw")
        self.decompressed_image_topic = cfg.get("decompressed_image_topic", self.image_topic)
        self.groundtruth_topic = os.environ.get(
            cfg["groundtruth_topic_env"], cfg["groundtruth_topic"]
        )
        self.groundtruth_frame_source = cfg.get("groundtruth_frame_source", "tf")
        self.tf_topics = list(cfg.get("tf_topics", ["/tf_static", "/tf"]))
        self.camera_frame = os.environ.get(
            cfg.get("camera_frame_env", "BLT_CAMERA_FRAME"),
            cfg.get("camera_frame", ""),
        )
        self.camera_frame_candidates = list(cfg.get("camera_frame_candidates", []))
        self.max_frames = int(os.environ.get(cfg["max_frames_env"], cfg.get("max_frames", 0)))
        self.max_seconds = float(os.environ.get(cfg.get("max_seconds_env", "BLT_MAX_SECONDS"), cfg.get("max_seconds", 0.0)))
        self.allow_placeholder_calibration = os.environ.get(
            cfg.get("allow_placeholder_calibration_env", "BLT_ALLOW_PLACEHOLDER_CALIBRATION"),
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self.use_camera_info_calibration = os.environ.get(
            cfg.get("use_camera_info_calibration_env", "BLT_USE_CAMERA_INFO_CALIBRATION"),
            str(cfg.get("use_camera_info_calibration", False)),
        ).lower() in {"1", "true", "yes", "on"}
        self.calibration_file_env = cfg.get("calibration_file_env", "BLT_CALIBRATION_FILE")
        self.calibration_sanity_check = os.environ.get(
            cfg.get("calibration_sanity_check_env", "BLT_CALIBRATION_SANITY_CHECK"),
            str(cfg.get("calibration_sanity_check", True)),
        ).lower() in {"1", "true", "yes", "on"}
        self.calibration_sanity_tolerance = float(cfg.get("calibration_sanity_tolerance", 0.10))
        self.experimental_calibration_override = os.environ.get(
            cfg.get("experimental_calibration_override_env", "BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE"),
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self.calibration = self._load_calibration(cfg)
        self._calibration_info_by_sequence: dict[str, Any] = {}
        self._calibration_info_topic_by_sequence: dict[str, str] = {}

    def download_sequence(self, sequence_name: str) -> None:
        if (
            self.check_sequence_availability(sequence_name, verbose=True) == "available"
            and self._sequence_outputs_match_current_fingerprint(sequence_name)
        ):
            return
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.download_process(sequence_name)

    def download_sequence_data(self, sequence_name: str) -> None:
        bag_path = self.get_source_bag_path(sequence_name)
        if not bag_path.is_file():
            raise FileNotFoundError(
                f"Missing BLT source bag for '{sequence_name}': {bag_path}"
            )
        (self.dataset_path / sequence_name).mkdir(parents=True, exist_ok=True)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        existing_images = self._rgb_image_paths(rgb_path)
        fingerprint = self._extraction_fingerprint(sequence_name)
        diagnostics = self._read_diagnostics(self._diagnostics_path(sequence_name))
        if diagnostics.get("extraction_fingerprint") != fingerprint:
            self._clear_generated_sequence_outputs(sequence_name)
            existing_images = []

        needs_rgb = True
        if self.max_frames > 0 and len(existing_images) >= self.max_frames:
            needs_rgb = False
        if self.max_seconds > 0 and self._rgb_images_cover_seconds(existing_images, self.max_seconds):
            needs_rgb = False
        if self.max_frames <= 0 and self.max_seconds <= 0 and existing_images:
            needs_rgb = False

        bag_path = self.get_source_bag_path(sequence_name)
        if needs_rgb:
            rgb_path.mkdir(parents=True, exist_ok=True)
            extraction_info = self._extract_rgb_images(
                bag_path=bag_path,
                image_topics=self.get_image_topic_candidates(),
                output_path=rgb_path,
                max_frames=self.max_frames,
                max_seconds=self.max_seconds,
            )
            self._update_diagnostics(sequence_name, extraction_info)
            self._update_diagnostics(sequence_name, {"extraction_fingerprint": fingerprint})

        if "rgbd" in self.modes:
            depth_path = sequence_path / "depth_0"
            existing_depth_images = self._depth_image_paths(depth_path)
            needs_depth = True
            if self.max_frames > 0 and len(existing_depth_images) >= self.max_frames:
                needs_depth = False
            if self.max_seconds > 0 and self._rgb_images_cover_seconds(existing_depth_images, self.max_seconds):
                needs_depth = False
            if self.max_frames <= 0 and self.max_seconds <= 0 and existing_depth_images:
                needs_depth = False

            if needs_depth:
                if not self.depth_topic:
                    raise ValueError("BLT RGB-D mode requires a configured depth topic")
                depth_path.mkdir(parents=True, exist_ok=True)
                depth_extraction_info = self._extract_depth_images(
                    bag_path=bag_path,
                    depth_topics=[self.depth_topic],
                    output_path=depth_path,
                    max_frames=self.max_frames,
                    max_seconds=self.max_seconds,
                )
                self._update_diagnostics(sequence_name, depth_extraction_info)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        tmp = rgb_csv.with_suffix(".csv.tmp")

        image_paths = self._rgb_image_paths(rgb_path)
        if not image_paths:
            raise FileNotFoundError(f"No BLT RGB images found in {rgb_path}")
        depth_paths = []
        if "rgbd" in self.modes:
            depth_paths = self._depth_image_paths(sequence_path / "depth_0")

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if depth_paths:
                writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])
                for image_path, depth_path in self._nearest_rgb_depth_pairs(image_paths, depth_paths):
                    writer.writerow(
                        [
                            self._timestamp_from_image_name(image_path),
                            f"rgb_0/{image_path.name}",
                            self._timestamp_from_image_name(depth_path),
                            f"depth_0/{depth_path.name}",
                        ]
                    )
            else:
                writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
                for image_path in image_paths:
                    writer.writerow([self._timestamp_from_image_name(image_path), f"rgb_0/{image_path.name}"])

        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        cal = dict(self.calibration[sequence_name])
        camera_info = self._calibration_info_by_sequence.get(sequence_name)
        if camera_info is None and self.use_camera_info_calibration and self.camera_info_topics:
            camera_info_result = self._extract_camera_info(
                bag_path=self.get_source_bag_path(sequence_name),
                camera_info_topics=self.camera_info_topics,
                rgb_time_bounds=self._rgb_time_bounds(sequence_name),
            )
            if camera_info_result is not None:
                camera_info_topic, camera_info = camera_info_result
                self._calibration_info_by_sequence[sequence_name] = camera_info
                self._calibration_info_topic_by_sequence[sequence_name] = camera_info_topic

        calibration_source = (
            "tracked_calibration_yaml"
            if self.calibration_yaml_path != self.yaml_file
            else "dataset_yaml"
        )
        calibration_diagnostics: dict[str, Any] = {}
        if camera_info is not None:
            self._validate_camera_info_dimensions(sequence_name, camera_info)
            cal.update(self._camera_info_to_calibration(camera_info))
            calibration_source = "camera_info"
            calibration_diagnostics = {
                "camera_info_topic": self._calibration_info_topic_by_sequence.get(sequence_name),
                "camera_info_frame": self._clean_frame(getattr(getattr(camera_info, "header", None), "frame_id", "")),
                "camera_info_width": int(getattr(camera_info, "width", 0)),
                "camera_info_height": int(getattr(camera_info, "height", 0)),
            }
        else:
            self._validate_placeholder_calibration(sequence_name, cal)
            calibration_diagnostics.update(
                self._validate_tracked_calibration_against_camera_info(sequence_name, cal)
            )

        rgb0: dict[str, Any] = {
            "cam_name": cal["cam_name"],
            "cam_type": "rgb+depth" if "rgbd" in self.modes else cal["cam_type"],
            "cam_model": cal["cam_model"],
            "focal_length": cal["focal_length"],
            "principal_point": cal["principal_point"],
            "fps": float(cal.get("fps", self.rgb_hz)),
            "T_BS": np.eye(4),
        }
        if "rgbd" in self.modes:
            rgb0["depth_name"] = "depth_0"
            rgb0["depth_factor"] = self.depth_factor
        if "distortion_type" in cal:
            rgb0["distortion_type"] = cal["distortion_type"]
        if "distortion_coefficients" in cal:
            rgb0["distortion_coefficients"] = cal["distortion_coefficients"]
        if "rgbd" in self.modes:
            self.write_calibration_yaml(sequence_name=sequence_name, rgbd=[rgb0])
        else:
            self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])
        self._update_diagnostics(
            sequence_name,
            {
                "calibration_source": calibration_source,
                "camera_info_topics": self.camera_info_topics,
                "focal_length": cal["focal_length"],
                "principal_point": cal["principal_point"],
                "distortion_coefficients": cal.get("distortion_coefficients", []),
                **calibration_diagnostics,
            },
        )

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        groundtruth_csv = self.dataset_path / sequence_name / "groundtruth.csv"
        if groundtruth_csv.is_file():
            return

        rows, diagnostics = self._extract_groundtruth_rows(
            bag_path=self.get_source_bag_path(sequence_name),
            rgb_time_bounds=self._rgb_time_bounds(sequence_name),
        )
        self._write_groundtruth_csv(sequence_name, rows, diagnostics=diagnostics)

    def check_sequence_integrity(self, sequence_name: str, verbose: bool) -> bool:
        complete_sequence = super().check_sequence_integrity(sequence_name, verbose)
        sequence_path = self.dataset_path / sequence_name
        if "rgbd" in self.modes:
            depth_path = sequence_path / "depth_0"
            if not depth_path.is_dir():
                if verbose:
                    from loguru import logger

                    from utilities import ws

                    logger.error(f"\n{ws(4)}Missing Depth folder: {depth_path} !!!!!")
                complete_sequence = False
            rgb_csv = sequence_path / "rgb.csv"
            if rgb_csv.is_file():
                with open(rgb_csv, newline="", encoding="utf-8") as f:
                    fieldnames = csv.DictReader(f).fieldnames or []
                missing_depth_columns = {
                    "ts_depth_0 (ns)",
                    "path_depth_0",
                }.difference(fieldnames)
                if missing_depth_columns:
                    if verbose:
                        from loguru import logger

                        from utilities import ws

                        logger.error(
                            f"\n{ws(4)}Missing RGB-D columns in {rgb_csv}: "
                            f"{sorted(missing_depth_columns)} !!!!!"
                        )
                    complete_sequence = False
        groundtruth_csv = self.dataset_path / sequence_name / "groundtruth.csv"
        if not groundtruth_csv.is_file():
            if verbose:
                from loguru import logger

                from utilities import ws

                logger.error(f"\n{ws(4)}Missing Groundtruth CSV: {groundtruth_csv} !!!!!")
            complete_sequence = False
        return complete_sequence

    def get_source_bag_path(self, sequence_name: str) -> Path:
        if sequence_name not in self.source_bags:
            raise ValueError(f"Unknown BLT sequence: {sequence_name}")
        source_bag = Path(self.source_bags[sequence_name]).expanduser()
        if source_bag.is_absolute():
            return source_bag
        if self.source_root is None:
            raise RuntimeError(
                f"Set {self.source_root_env} to the local BLT ktima root before resolving "
                f"relative source bag for '{sequence_name}': {source_bag}"
            )
        return self.source_root / source_bag

    def _load_calibration(self, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        calibration_file = os.environ.get(self.calibration_file_env, cfg.get("calibration_file", ""))
        if calibration_file:
            calibration_path = Path(calibration_file).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = self.yaml_file.parent / calibration_path
            self.calibration_yaml_path = calibration_path
            with open(self.calibration_yaml_path, "r", encoding="utf-8") as f:
                calibration_cfg = yaml.safe_load(f) or {}
            return dict(calibration_cfg["calibration"])

        self.calibration_yaml_path = self.yaml_file
        return dict(cfg["calibration"])

    def get_image_topic_candidates(self) -> list[str]:
        image_topic = self.image_topic.rstrip("/")
        candidates: list[str] = []

        if self.image_transport == "compressed":
            if image_topic.endswith("/compressed"):
                candidates.append(image_topic)
                candidates.append(image_topic.removesuffix("/compressed"))
            else:
                candidates.append(f"{image_topic}/compressed")
                candidates.append(image_topic)
        else:
            candidates.append(image_topic)

        return list(dict.fromkeys(candidates))

    def _extraction_fingerprint(self, sequence_name: str) -> dict[str, Any]:
        fingerprint = {
            "source_bag": str(self.get_source_bag_path(sequence_name)),
            "image_topic": self.image_topic,
            "image_transport": self.image_transport,
            "camera_info_topics": self.camera_info_topics,
            "calibration_yaml_path": str(self.calibration_yaml_path),
            "calibration_sanity_check": self.calibration_sanity_check,
            "calibration_sanity_tolerance": self.calibration_sanity_tolerance,
            "groundtruth_topic": self.groundtruth_topic,
            "tf_topics": self.tf_topics,
            "camera_frame": self.camera_frame,
            "camera_frame_candidates": self.camera_frame_candidates,
            "use_camera_info_calibration": self.use_camera_info_calibration,
            "max_frames": self.max_frames,
            "max_seconds": self.max_seconds,
            "modes": self.modes,
        }
        if "rgbd" in self.modes:
            fingerprint.update(
                {
                    "depth_topic": self.depth_topic,
                    "depth_camera_info_topics": self.depth_camera_info_topics,
                    "depth_factor": self.depth_factor,
                }
            )
        return fingerprint

    def _clear_generated_sequence_outputs(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for directory_name in ("rgb_0", "depth_0"):
            path = sequence_path / directory_name
            if path.exists():
                shutil.rmtree(path)
        for file_name in (
            "rgb.csv",
            "calibration.yaml",
            "groundtruth.csv",
            "blt_diagnostics.yaml",
        ):
            path = sequence_path / file_name
            if path.exists():
                path.unlink()

    def _sequence_outputs_match_current_fingerprint(self, sequence_name: str) -> bool:
        diagnostics = self._read_diagnostics(self._diagnostics_path(sequence_name))
        return diagnostics.get("extraction_fingerprint") == self._extraction_fingerprint(sequence_name)

    def _rgb_time_bounds(self, sequence_name: str) -> tuple[int, int] | None:
        rgb_csv = self.dataset_path / sequence_name / "rgb.csv"
        if not rgb_csv.is_file():
            return None

        timestamps: list[int] = []
        with open(rgb_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                timestamps.append(int(row["ts_rgb_0 (ns)"]))

        if not timestamps:
            return None
        return min(timestamps), max(timestamps)

    @staticmethod
    def _rgb_images_cover_seconds(image_paths: list[Path], max_seconds: float) -> bool:
        if len(image_paths) < 2:
            return False
        timestamps = [BLT_dataset._timestamp_from_image_name(path) for path in image_paths]
        duration_s = (max(timestamps) - min(timestamps)) / 1e9
        return duration_s >= max_seconds

    def _resolve_groundtruth_target_frame(self, tf_edges: dict[str, list[tuple[str, np.ndarray]]]) -> str:
        available_frames = self._available_tf_frames(tf_edges)
        preferred_frames: list[str] = []
        if self.camera_frame:
            preferred_frames.append(self.camera_frame)
        preferred_frames.extend(self.camera_frame_candidates)
        for frame in preferred_frames:
            clean_frame = self._clean_frame(frame)
            if clean_frame in available_frames:
                return clean_frame

        camera_like = [
            frame for frame in available_frames
            if self._is_camera_like_frame(frame)
        ]
        if camera_like:
            return sorted(camera_like)[0]

        frames = ", ".join(sorted(available_frames)) or "<none>"
        raise ValueError(
            "Could not infer BLT camera frame from TF. "
            f"Set BLT_CAMERA_FRAME. Available frames: {frames}"
        )

    def _extract_groundtruth_rows(
        self,
        bag_path: Path,
        rgb_time_bounds: tuple[int, int] | None = None,
    ) -> tuple[list[list[Any]], dict[str, Any]]:
        odometry_messages: list[tuple[int, Any]] = []
        tf_transforms: list[Any] = []
        tf_timestamps: list[int] = []
        min_rgb_ts = rgb_time_bounds[0] if rgb_time_bounds else None
        max_rgb_ts = rgb_time_bounds[1] if rgb_time_bounds else None
        margin_ns = int(2e9)

        with self._open_fast_ros1_stream(bag_path) as reader:
            groundtruth_connections = [
                c for c in reader.connections if c.topic == self.groundtruth_topic
            ]
            if not groundtruth_connections:
                topics = ", ".join(sorted({c.topic for c in reader.connections}))
                raise ValueError(
                    f"Groundtruth topic '{self.groundtruth_topic}' not found in {bag_path}. "
                    f"Available topics: {topics}"
                )

            tf_connections = [
                c for c in reader.connections if c.topic in set(self.tf_topics)
            ]
            selected_connections = groundtruth_connections + tf_connections
            for connection, timestamp_ns, rawdata in tqdm(
                reader.messages(connections=selected_connections),
                desc=f"Extracting {self.groundtruth_topic} ground truth",
            ):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic in self.tf_topics:
                    transforms = self._transforms_from_tf_message(msg)
                    tf_transforms.extend(transforms)
                    tf_timestamps.extend([timestamp_ns] * len(transforms))
                    continue

                if min_rgb_ts is not None and timestamp_ns < min_rgb_ts - margin_ns:
                    continue
                if max_rgb_ts is not None and timestamp_ns > max_rgb_ts + margin_ns:
                    if odometry_messages:
                        break
                    continue

                if not connection.msgtype.endswith("/Odometry"):
                    raise TypeError(
                        f"Unsupported BLT groundtruth message type on "
                        f"{self.groundtruth_topic}: {connection.msgtype}"
                    )
                odometry_messages.append((timestamp_ns, msg))

        if not odometry_messages:
            raise RuntimeError(
                f"No BLT odometry groundtruth messages extracted from {bag_path}:"
                f"{self.groundtruth_topic}"
            )

        tf_edges = self._tf_edges_from_transforms(tf_transforms, timestamps_ns=tf_timestamps)
        target_frame = self._resolve_groundtruth_target_frame(tf_edges)
        diagnostics: dict[str, Any] = {}
        rows = self._groundtruth_rows_from_odometry_messages(
            odometry_messages,
            tf_edges=tf_edges,
            target_frame=target_frame,
            diagnostics=diagnostics,
        )
        diagnostics.update(
            {
                "groundtruth_topic": self.groundtruth_topic,
                "tf_topics": self.tf_topics,
                "groundtruth_target_frame": target_frame,
                "rgb_time_bounds_ns": list(rgb_time_bounds) if rgb_time_bounds else None,
            }
        )
        return rows, diagnostics

    def _write_groundtruth_csv(
        self,
        sequence_name: str,
        rows: list[list[Any]],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        sequence_path = self.dataset_path / sequence_name
        sequence_path.mkdir(parents=True, exist_ok=True)
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for row in rows:
                writer.writerow([row[0], *[self._format_csv_number(v) for v in row[1:]]])

        tmp.replace(groundtruth_csv)
        if diagnostics is not None:
            diagnostics = dict(diagnostics)
            diagnostics.update(self._groundtruth_row_diagnostics(rows, diagnostics))
            self._update_diagnostics(sequence_name, diagnostics)

    def _extract_camera_info(
        self,
        bag_path: Path,
        camera_info_topics: list[str],
        rgb_time_bounds: tuple[int, int] | None = None,
    ) -> tuple[str, Any] | None:
        min_rgb_ts = rgb_time_bounds[0] if rgb_time_bounds else None
        max_rgb_ts = rgb_time_bounds[1] if rgb_time_bounds else None
        margin_ns = int(2e9)
        with self._open_fast_ros1_stream(bag_path) as reader:
            for camera_info_topic in camera_info_topics:
                connections = [c for c in reader.connections if c.topic == camera_info_topic]
                if not connections:
                    continue
                for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                    if min_rgb_ts is not None and timestamp_ns < min_rgb_ts - margin_ns:
                        continue
                    if max_rgb_ts is not None and timestamp_ns > max_rgb_ts + margin_ns:
                        break
                    return camera_info_topic, reader.deserialize(rawdata, connection.msgtype)
        return None

    @staticmethod
    def _camera_info_to_calibration(camera_info: Any) -> dict[str, Any]:
        distortion_values = getattr(camera_info, "d", getattr(camera_info, "D", []))
        distortion_coefficients = [float(v) for v in list(distortion_values)]
        distortion_model = str(getattr(camera_info, "distortion_model", "") or "").lower()
        if distortion_model in {"plumb_bob", "radtan"}:
            distortion_type = "radtan"
            if len(distortion_coefficients) >= 5:
                cam_model = "radtan5"
            elif len(distortion_coefficients) >= 4:
                cam_model = "radtan4"
            else:
                cam_model = "pinhole"
        else:
            distortion_type = distortion_model or "unknown"
            cam_model = "pinhole"

        k = list(getattr(camera_info, "k", getattr(camera_info, "K", [])))
        if len(k) != 9:
            raise ValueError("BLT CameraInfo must contain a 3x3 K matrix")
        calibration = {
            "cam_model": cam_model,
            "focal_length": [float(k[0]), float(k[4])],
            "principal_point": [float(k[2]), float(k[5])],
        }
        if distortion_coefficients:
            calibration["distortion_type"] = distortion_type
            calibration["distortion_coefficients"] = distortion_coefficients
        return calibration

    def _validate_placeholder_calibration(self, sequence_name: str, calibration: dict[str, Any]) -> None:
        if self.allow_placeholder_calibration:
            return
        sequence_path = self.dataset_path / sequence_name
        image_dimension = self._first_rgb_image_dimension(sequence_path / calibration["cam_name"])
        if image_dimension is None:
            return
        width, height = image_dimension
        focal = [float(v) for v in calibration.get("focal_length", [])]
        principal = [float(v) for v in calibration.get("principal_point", [])]
        looks_placeholder = focal == [525.0, 525.0] and principal == [319.5, 239.5]
        looks_hd = width >= 1280 or height >= 720
        if looks_placeholder and looks_hd:
            raise ValueError(
                "BLT placeholder calibration cannot be used for "
                f"{width}x{height} images. Extract CameraInfo or set "
                "BLT_ALLOW_PLACEHOLDER_CALIBRATION=1 to acknowledge the limitation."
            )

    def _validate_tracked_calibration_against_camera_info(
        self,
        sequence_name: str,
        calibration: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not self.calibration_sanity_check
            or self.use_camera_info_calibration
            or not self.camera_info_topics
        ):
            return {"calibration_sanity_status": "skipped"}

        camera_info_result = self._extract_camera_info(
            bag_path=self.get_source_bag_path(sequence_name),
            camera_info_topics=self.camera_info_topics,
            rgb_time_bounds=self._rgb_time_bounds(sequence_name),
        )
        if camera_info_result is None:
            return {"calibration_sanity_status": "skipped_no_camera_info"}

        camera_info_topic, camera_info = camera_info_result
        self._validate_camera_info_dimensions(sequence_name, camera_info)
        camera_info_calibration = self._camera_info_to_calibration(camera_info)
        tracked_focal = [float(v) for v in calibration.get("focal_length", [])]
        tracked_principal = [float(v) for v in calibration.get("principal_point", [])]
        camera_focal = [float(v) for v in camera_info_calibration["focal_length"]]
        camera_principal = [float(v) for v in camera_info_calibration["principal_point"]]
        image_width = float(getattr(camera_info, "width", 0) or 1)
        image_height = float(getattr(camera_info, "height", 0) or 1)
        focal_delta = [
            abs(tracked_focal[i] - camera_focal[i]) / max(abs(camera_focal[i]), 1.0)
            for i in range(min(len(tracked_focal), len(camera_focal)))
        ]
        principal_delta = [
            abs(tracked_principal[0] - camera_principal[0]) / max(image_width, 1.0),
            abs(tracked_principal[1] - camera_principal[1]) / max(image_height, 1.0),
        ] if len(tracked_principal) >= 2 and len(camera_principal) >= 2 else []
        max_delta = max(focal_delta + principal_delta) if focal_delta or principal_delta else 0.0
        status = "ok" if max_delta <= self.calibration_sanity_tolerance else "mismatch"
        diagnostics = {
            "calibration_sanity_status": status,
            "camera_info_reference_topic": camera_info_topic,
            "camera_info_reference_width": int(getattr(camera_info, "width", 0)),
            "camera_info_reference_height": int(getattr(camera_info, "height", 0)),
            "camera_info_reference_focal_length": camera_focal,
            "camera_info_reference_principal_point": camera_principal,
            "calibration_sanity_max_relative_delta": float(max_delta),
        }
        if status == "mismatch" and self.experimental_calibration_override:
            diagnostics["calibration_sanity_status"] = "override_mismatch"
            return diagnostics
        if status == "mismatch":
            raise ValueError(
                f"BLT tracked calibration for {sequence_name} differs from CameraInfo "
                f"beyond {self.calibration_sanity_tolerance:.0%}: "
                f"max_relative_delta={max_delta:.3f}; tracked_focal={tracked_focal}; "
                f"camera_info_focal={camera_focal}; topic={camera_info_topic}. "
                "Set BLT_EXPERIMENTAL_CALIBRATION_OVERRIDE=1 only for intentional "
                "calibration ablations."
            )
        return diagnostics

    def _validate_camera_info_dimensions(self, sequence_name: str, camera_info: Any) -> None:
        sequence_path = self.dataset_path / sequence_name
        image_dimension = self._first_rgb_image_dimension(sequence_path / "rgb_0")
        if image_dimension is None:
            return
        image_width, image_height = image_dimension
        info_width = int(getattr(camera_info, "width", 0))
        info_height = int(getattr(camera_info, "height", 0))
        if info_width and info_height and (info_width, info_height) != (image_width, image_height):
            raise ValueError(
                f"BLT CameraInfo dimensions {info_width}x{info_height} do not match "
                f"extracted image dimensions {image_width}x{image_height} for {sequence_name}."
            )

    @staticmethod
    def _first_rgb_image_dimension(rgb_path: Path) -> tuple[int, int] | None:
        import cv2

        image_paths = BLT_dataset._rgb_image_paths(rgb_path)
        if not image_paths:
            return None
        image = cv2.imread(str(image_paths[0]))
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height

    def _diagnostics_path(self, sequence_name: str) -> Path:
        return self.dataset_path / sequence_name / "blt_diagnostics.yaml"

    def validate_extraction_gate(self, sequence_names: list[str] | None = None) -> dict[str, Any]:
        if sequence_names is None:
            sequence_names = list(self.sequence_names)
        report: dict[str, Any] = {
            "ready_for_experiments": True,
            "sequences": {},
            "failures": [],
            "later_inspection": [],
        }
        comparable_contract: dict[str, Any] | None = None
        comparable_keys = [
            "image_topic",
            "calibration_source",
            "focal_length",
            "principal_point",
            "groundtruth_topic",
            "groundtruth_target_frame",
        ]

        for sequence_name in sequence_names:
            sequence_path = self.dataset_path / sequence_name
            diagnostics = self._read_diagnostics(self._diagnostics_path(sequence_name))
            issues = self._extraction_gate_sequence_issues(sequence_name, diagnostics)
            status = "failed" if issues else "ok"
            if issues:
                report["ready_for_experiments"] = False
                for issue in issues:
                    report["failures"].append({"sequence": sequence_name, "issue": issue})
                    report["later_inspection"].append(f"{sequence_name}: {issue}")

            summary = {
                "status": status,
                "path": str(sequence_path),
                "availability": self.check_sequence_availability(sequence_name, verbose=False),
                "image_topic": diagnostics.get("image_topic"),
                "camera_info_topic": diagnostics.get("camera_info_topic"),
                "image_count": diagnostics.get("image_count"),
                "rgb_inferred_fps": diagnostics.get("rgb_inferred_fps"),
                "rgb_duration_s": diagnostics.get("rgb_duration_s"),
                "calibration_source": diagnostics.get("calibration_source"),
                "camera_info_width": diagnostics.get("camera_info_width"),
                "camera_info_height": diagnostics.get("camera_info_height"),
                "focal_length": diagnostics.get("focal_length"),
                "principal_point": diagnostics.get("principal_point"),
                "groundtruth_topic": diagnostics.get("groundtruth_topic"),
                "groundtruth_count": diagnostics.get("groundtruth_count"),
                "groundtruth_path_length_m": diagnostics.get("groundtruth_path_length_m"),
                "groundtruth_source_frame": diagnostics.get("groundtruth_source_frame"),
                "groundtruth_target_frame": diagnostics.get("groundtruth_target_frame"),
                "tf_chain": diagnostics.get("tf_chain"),
                "tf_chain_dynamic": diagnostics.get("tf_chain_dynamic"),
                "rgb_groundtruth_overlap_ns": diagnostics.get("rgb_groundtruth_overlap_ns"),
            }
            report["sequences"][sequence_name] = summary

            if not issues:
                contract = {key: diagnostics.get(key) for key in comparable_keys}
                if comparable_contract is None:
                    comparable_contract = contract
                else:
                    for key, expected in comparable_contract.items():
                        actual = contract.get(key)
                        if actual != expected:
                            report["ready_for_experiments"] = False
                            issue = (
                                f"{sequence_name}: {key} differs from extraction gate "
                                f"reference ({actual!r} != {expected!r})"
                            )
                            report["failures"].append({"sequence": sequence_name, "issue": issue})
                            report["later_inspection"].append(issue)

        return report

    def write_extraction_gate_report(
        self,
        output_path: str | Path,
        sequence_names: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self.validate_extraction_gate(sequence_names)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._yaml_safe(report), f, sort_keys=True)
        return report

    def _extraction_gate_sequence_issues(
        self,
        sequence_name: str,
        diagnostics: dict[str, Any],
    ) -> list[str]:
        sequence_path = self.dataset_path / sequence_name
        issues: list[str] = []
        required_paths = [
            sequence_path / "rgb_0",
            sequence_path / "rgb.csv",
            sequence_path / "calibration.yaml",
            sequence_path / "groundtruth.csv",
            sequence_path / "blt_diagnostics.yaml",
        ]
        for path in required_paths:
            if not path.exists():
                issues.append(f"missing required output: {path.name}")

        required_diagnostics = [
            "image_topic",
            "image_count",
            "rgb_inferred_fps",
            "rgb_duration_s",
            "calibration_source",
            "focal_length",
            "principal_point",
            "groundtruth_topic",
            "groundtruth_count",
            "groundtruth_path_length_m",
            "groundtruth_source_frame",
            "groundtruth_target_frame",
            "tf_chain",
            "tf_chain_dynamic",
            "rgb_groundtruth_overlap_ns",
            "extraction_fingerprint",
        ]
        for key in required_diagnostics:
            if diagnostics.get(key) in (None, "", []):
                issues.append(f"missing diagnostic: {key}")

        expected_calibration_source = (
            "camera_info" if self.use_camera_info_calibration else "tracked_calibration_yaml"
        )
        if diagnostics.get("calibration_source") != expected_calibration_source:
            issues.append(
                "calibration_source is "
                f"{diagnostics.get('calibration_source')!r}, expected {expected_calibration_source!r}"
            )
        if diagnostics.get("groundtruth_topic") != self.groundtruth_topic:
            issues.append(
                f"groundtruth_topic is {diagnostics.get('groundtruth_topic')!r}, expected {self.groundtruth_topic!r}"
            )
        if diagnostics.get("rgb_groundtruth_overlap_ns") is None:
            issues.append("missing RGB/groundtruth timestamp overlap")
        if diagnostics.get("extraction_fingerprint") != self._extraction_fingerprint(sequence_name):
            issues.append("extraction fingerprint does not match current settings")
        if int(diagnostics.get("image_count") or 0) <= 0:
            issues.append("image_count must be positive")
        if int(diagnostics.get("groundtruth_count") or 0) <= 0:
            issues.append("groundtruth_count must be positive")
        return issues

    def _update_diagnostics(self, sequence_name: str, values: dict[str, Any]) -> None:
        diagnostics_path = self._diagnostics_path(sequence_name)
        diagnostics = self._read_diagnostics(diagnostics_path)
        diagnostics.update(self._yaml_safe(values))
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = diagnostics_path.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(diagnostics, f, sort_keys=True)
        tmp.replace(diagnostics_path)

    @staticmethod
    def _read_diagnostics(diagnostics_path: Path) -> dict[str, Any]:
        if not diagnostics_path.is_file():
            return {}
        with open(diagnostics_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def _groundtruth_row_diagnostics(
        cls,
        rows: list[list[Any]],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        if not rows:
            return {
                "groundtruth_count": 0,
                "groundtruth_path_length_m": 0.0,
                "groundtruth_time_bounds_ns": None,
                "rgb_groundtruth_overlap_ns": None,
            }
        positions = np.array([[row[1], row[2], row[3]] for row in rows], dtype=float)
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(rows) > 1 else 0.0
        gt_bounds = [int(rows[0][0]), int(rows[-1][0])]
        rgb_bounds = diagnostics.get("rgb_time_bounds_ns")
        overlap = None
        if rgb_bounds and rgb_bounds[0] is not None and rgb_bounds[1] is not None:
            overlap_start = max(int(rgb_bounds[0]), gt_bounds[0])
            overlap_end = min(int(rgb_bounds[1]), gt_bounds[1])
            if overlap_start <= overlap_end:
                overlap = [overlap_start, overlap_end]
        return {
            "groundtruth_count": len(rows),
            "groundtruth_path_length_m": path_length,
            "groundtruth_time_bounds_ns": gt_bounds,
            "rgb_groundtruth_overlap_ns": overlap,
        }

    @classmethod
    def _yaml_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._yaml_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._yaml_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def _rgb_image_paths(rgb_path: Path) -> list[Path]:
        if not rgb_path.is_dir():
            return []
        return sorted(
            (
                p for p in rgb_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ),
            key=BLT_dataset._timestamp_from_image_name,
        )

    @staticmethod
    def _depth_image_paths(depth_path: Path) -> list[Path]:
        if not depth_path.is_dir():
            return []
        return sorted(
            (
                p for p in depth_path.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            ),
            key=BLT_dataset._timestamp_from_image_name,
        )

    @staticmethod
    def _timestamp_from_image_name(image_path: Path) -> int:
        try:
            return int(image_path.stem)
        except ValueError as exc:
            raise ValueError(
                f"BLT RGB image filenames must be nanosecond timestamps: {image_path.name}"
            ) from exc

    @classmethod
    def _nearest_rgb_depth_pairs(
        cls,
        image_paths: list[Path],
        depth_paths: list[Path],
    ) -> list[tuple[Path, Path]]:
        depth_timestamps = [cls._timestamp_from_image_name(path) for path in depth_paths]
        pairs: list[tuple[Path, Path]] = []
        depth_index = 0
        for image_path in image_paths:
            image_timestamp = cls._timestamp_from_image_name(image_path)
            while (
                depth_index + 1 < len(depth_timestamps)
                and abs(depth_timestamps[depth_index + 1] - image_timestamp)
                <= abs(depth_timestamps[depth_index] - image_timestamp)
            ):
                depth_index += 1
            pairs.append((image_path, depth_paths[depth_index]))
        return pairs

    @classmethod
    def _extract_rgb_images(
        cls,
        bag_path: Path,
        image_topics: list[str],
        output_path: Path,
        max_frames: int = 0,
        max_seconds: float = 0.0,
    ) -> dict[str, Any]:
        written = 0
        first_timestamp_ns: int | None = None
        last_timestamp_ns: int | None = None
        selected_topic: str | None = None
        for selected_topic, msgtype, timestamp_ns, msg in cls._iter_image_messages(
            bag_path,
            image_topics,
        ):
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            if max_seconds > 0 and (timestamp_ns - first_timestamp_ns) / 1e9 >= max_seconds:
                break
            cls._write_image_message(
                output_path=output_path,
                timestamp_ns=timestamp_ns,
                msgtype=msgtype,
                msg=msg,
            )
            written += 1
            last_timestamp_ns = timestamp_ns
            if max_frames > 0 and written >= max_frames:
                break

        if written == 0:
            raise RuntimeError(f"No images extracted from {bag_path}:{image_topics}")

        duration_s = 0.0
        if first_timestamp_ns is not None and last_timestamp_ns is not None:
            duration_s = (last_timestamp_ns - first_timestamp_ns) / 1e9
        fps = written / duration_s if duration_s > 0 else 0.0
        return {
            "image_topic": selected_topic,
            "image_count": written,
            "rgb_time_bounds_ns": [first_timestamp_ns, last_timestamp_ns],
            "rgb_duration_s": duration_s,
            "rgb_inferred_fps": fps,
            "max_frames": max_frames,
            "max_seconds": max_seconds,
        }

    @classmethod
    def _extract_depth_images(
        cls,
        bag_path: Path,
        depth_topics: list[str],
        output_path: Path,
        max_frames: int = 0,
        max_seconds: float = 0.0,
    ) -> dict[str, Any]:
        written = 0
        first_timestamp_ns: int | None = None
        last_timestamp_ns: int | None = None
        selected_topic: str | None = None
        for selected_topic, msgtype, timestamp_ns, msg in cls._iter_image_messages(
            bag_path,
            depth_topics,
        ):
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            if max_seconds > 0 and (timestamp_ns - first_timestamp_ns) / 1e9 >= max_seconds:
                break
            cls._write_depth_message(
                output_path=output_path,
                timestamp_ns=timestamp_ns,
                msgtype=msgtype,
                msg=msg,
            )
            written += 1
            last_timestamp_ns = timestamp_ns
            if max_frames > 0 and written >= max_frames:
                break

        if written == 0:
            raise RuntimeError(f"No depth images extracted from {bag_path}:{depth_topics}")

        duration_s = 0.0
        if first_timestamp_ns is not None and last_timestamp_ns is not None:
            duration_s = (last_timestamp_ns - first_timestamp_ns) / 1e9
        fps = written / duration_s if duration_s > 0 else 0.0
        return {
            "depth_topic": selected_topic,
            "depth_count": written,
            "depth_time_bounds_ns": [first_timestamp_ns, last_timestamp_ns],
            "depth_duration_s": duration_s,
            "depth_inferred_fps": fps,
            "max_frames": max_frames,
            "max_seconds": max_seconds,
        }

    @staticmethod
    def _iter_image_messages(
        bag_path: Path,
        image_topics: list[str],
    ):
        with BLT_dataset._open_fast_ros1_stream(bag_path) as reader:
            available_topics = [c.topic for c in reader.connections]
            selected_topic = BLT_dataset._select_first_available_topic(
                image_topics,
                available_topics,
                label="Image",
                bag_path=bag_path,
            )
            connections = [c for c in reader.connections if c.topic == selected_topic]

            messages = reader.messages(connections=connections)
            for connection, timestamp_ns, rawdata in tqdm(messages, desc=f"Extracting {selected_topic}"):
                msg = reader.deserialize(rawdata, connection.msgtype)
                yield selected_topic, connection.msgtype, timestamp_ns, msg

    @staticmethod
    def _select_first_available_topic(
        preferred_topics: list[str],
        available_topics: list[str],
        label: str,
        bag_path: Path,
    ) -> str:
        available = set(available_topics)
        for topic in preferred_topics:
            if topic in available:
                return topic
        topics = ", ".join(sorted(available_topics))
        raise ValueError(
            f"{label} topics {preferred_topics} not found in {bag_path}. "
            f"Available topics: {topics}"
        )

    @staticmethod
    def _open_fast_ros1_stream(bag_path: Path):
        return _FastRosbag1Stream(bag_path)

    @classmethod
    def _groundtruth_rows_from_odometry_messages(
        cls,
        odometry_messages: list[tuple[int, Any]],
        tf_edges: dict[str, list[tuple[str, np.ndarray]]],
        target_frame: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        target_frame = cls._clean_frame(target_frame)
        transform_cache: dict[tuple[str, str], np.ndarray] = {}
        chain_cache: dict[tuple[str, str], tuple[np.ndarray, list[str], bool]] = {}
        dynamic_tf = cls._tf_edges_are_dynamic(tf_edges)
        for timestamp_ns, msg in odometry_messages:
            source_frame = cls._odometry_pose_frame(msg)
            pose_matrix = cls._pose_msg_to_matrix(msg.pose.pose)
            transform_key = (source_frame, target_frame)
            if dynamic_tf:
                if transform_key in transform_cache:
                    source_to_target, tf_chain, tf_chain_dynamic = chain_cache[transform_key]
                else:
                    source_to_target, tf_chain, tf_chain_dynamic = cls._find_tf_chain(
                        tf_edges,
                        source_frame,
                        target_frame,
                        timestamp_ns=timestamp_ns,
                    )
                    if not tf_chain_dynamic:
                        chain_cache[transform_key] = (source_to_target, tf_chain, tf_chain_dynamic)
                        transform_cache[transform_key] = source_to_target
            else:
                if transform_key not in transform_cache:
                    if diagnostics is None:
                        source_to_target = cls._find_tf_chain_matrix(
                            tf_edges,
                            source_frame,
                            target_frame,
                        )
                        chain_cache[transform_key] = (source_to_target, [source_frame, target_frame], False)
                    else:
                        chain_cache[transform_key] = cls._find_tf_chain(
                            tf_edges,
                            source_frame,
                            target_frame,
                            timestamp_ns=timestamp_ns,
                        )
                        source_to_target = chain_cache[transform_key][0]
                    transform_cache[transform_key] = source_to_target
                source_to_target, tf_chain, tf_chain_dynamic = chain_cache[transform_key]

            if diagnostics is not None and "groundtruth_source_frame" not in diagnostics:
                diagnostics.update(
                    {
                        "groundtruth_source_frame": source_frame,
                        "groundtruth_target_frame": target_frame,
                        "tf_chain": tf_chain,
                        "tf_chain_dynamic": tf_chain_dynamic,
                    }
                )
            target_pose = pose_matrix @ source_to_target
            translation = target_pose[:3, 3]
            quaternion = cls._quaternion_from_matrix(target_pose[:3, :3])
            rows.append([
                int(timestamp_ns),
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ])

        return rows

    @classmethod
    def _tf_edges_from_transforms(
        cls,
        transforms: list[Any],
        timestamps_ns: list[int] | None = None,
    ) -> dict[str, list[tuple]]:
        if timestamps_ns is None:
            timestamps_ns = [None] * len(transforms)
        deduped_edges: dict[str, dict[tuple[str, int | None], tuple]] = {}
        seen_static_pairs: set[tuple[str, str]] = set()
        for transform_msg, timestamp_ns in zip(transforms, timestamps_ns):
            parent_frame = cls._clean_frame(transform_msg.header.frame_id)
            child_frame = cls._clean_frame(transform_msg.child_frame_id)
            matrix = cls._transform_msg_to_matrix(transform_msg.transform)
            if timestamp_ns is None:
                pair = (parent_frame, child_frame)
                if pair in seen_static_pairs:
                    continue
                seen_static_pairs.add(pair)
                seen_static_pairs.add((child_frame, parent_frame))
            deduped_edges.setdefault(parent_frame, {})[(child_frame, timestamp_ns)] = (child_frame, matrix, timestamp_ns)
            deduped_edges.setdefault(child_frame, {})[(parent_frame, timestamp_ns)] = (parent_frame, np.linalg.inv(matrix), timestamp_ns)
        return {
            frame: list(edge_map.values())
            for frame, edge_map in deduped_edges.items()
        }

    @staticmethod
    def _transforms_from_tf_message(msg: Any) -> list[Any]:
        if hasattr(msg, "transforms"):
            return list(msg.transforms)
        return [msg]

    @classmethod
    def _find_tf_chain_matrix(
        cls,
        tf_edges: dict[str, list[tuple[str, np.ndarray]]],
        source_frame: str,
        target_frame: str,
    ) -> np.ndarray:
        matrix, _, _ = cls._find_tf_chain(tf_edges, source_frame, target_frame)
        return matrix

    @classmethod
    def _find_tf_chain(
        cls,
        tf_edges: dict[str, list[tuple]],
        source_frame: str,
        target_frame: str,
        timestamp_ns: int | None = None,
    ) -> tuple[np.ndarray, list[str], bool]:
        source_frame = cls._clean_frame(source_frame)
        target_frame = cls._clean_frame(target_frame)
        if source_frame == target_frame:
            return np.eye(4), [source_frame], False

        queue = deque([(source_frame, np.eye(4), [source_frame], False)])
        visited = {source_frame}
        while queue:
            frame, matrix, chain, chain_dynamic = queue.popleft()
            for next_frame, edge_matrix, edge_dynamic in cls._selected_tf_edges(
                tf_edges.get(frame, []),
                timestamp_ns,
            ):
                if next_frame in visited:
                    continue
                next_matrix = matrix @ edge_matrix
                next_chain = [*chain, next_frame]
                next_dynamic = chain_dynamic or edge_dynamic
                if next_frame == target_frame:
                    return next_matrix, next_chain, next_dynamic
                visited.add(next_frame)
                queue.append((next_frame, next_matrix, next_chain, next_dynamic))

        available_frames = ", ".join(sorted(cls._available_tf_frames(tf_edges))) or "<none>"
        raise ValueError(
            f"No TF chain from '{source_frame}' to '{target_frame}'. "
            f"available frames: {available_frames}"
        )

    @classmethod
    def _selected_tf_edges(
        cls,
        edges: list[tuple],
        timestamp_ns: int | None,
    ) -> list[tuple[str, np.ndarray, bool]]:
        grouped: dict[str, list[tuple[np.ndarray, int | None]]] = defaultdict(list)
        for edge in edges:
            next_frame = cls._edge_frame(edge)
            grouped[next_frame].append((cls._edge_matrix(edge), cls._edge_timestamp(edge)))

        selected = []
        for next_frame, candidates in grouped.items():
            dynamic = any(ts is not None for _, ts in candidates) and len(candidates) > 1
            if timestamp_ns is None:
                matrix, _ = candidates[0]
            else:
                matrix, _ = min(
                    candidates,
                    key=lambda item: abs((item[1] if item[1] is not None else timestamp_ns) - timestamp_ns),
                )
            selected.append((next_frame, matrix, dynamic))
        return selected

    @staticmethod
    def _edge_frame(edge: tuple) -> str:
        return edge[0]

    @staticmethod
    def _edge_matrix(edge: tuple) -> np.ndarray:
        return edge[1]

    @staticmethod
    def _edge_timestamp(edge: tuple) -> int | None:
        return edge[2] if len(edge) > 2 else None

    @classmethod
    def _tf_edges_are_dynamic(cls, tf_edges: dict[str, list[tuple]]) -> bool:
        by_pair: dict[tuple[str, str], set[int | None]] = defaultdict(set)
        for frame, edges in tf_edges.items():
            for edge in edges:
                by_pair[(frame, cls._edge_frame(edge))].add(cls._edge_timestamp(edge))
        return any(len(timestamps) > 1 for timestamps in by_pair.values())

    @classmethod
    def _odometry_pose_frame(cls, msg: Any) -> str:
        child_frame = cls._clean_frame(getattr(msg, "child_frame_id", ""))
        if child_frame:
            return child_frame
        return cls._clean_frame(msg.header.frame_id)

    @classmethod
    def _pose_msg_to_matrix(cls, pose: Any) -> np.ndarray:
        translation = cls._vector_to_array(pose.position)
        quaternion = cls._quaternion_to_array(pose.orientation)
        return cls._matrix_from_translation_quaternion(translation, quaternion)

    @classmethod
    def _transform_msg_to_matrix(cls, transform: Any) -> np.ndarray:
        translation = cls._vector_to_array(transform.translation)
        quaternion = cls._quaternion_to_array(transform.rotation)
        return cls._matrix_from_translation_quaternion(translation, quaternion)

    @staticmethod
    def _matrix_from_translation_quaternion(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = BLT_dataset._rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = BLT_dataset._normalize_quaternion(quaternion)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
        trace = np.trace(rotation)
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            idx = int(np.argmax(np.diag(rotation)))
            if idx == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif idx == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        return BLT_dataset._normalize_quaternion(np.array([x, y, z, w]))

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Zero-length quaternion is not valid")
        return quaternion / norm

    @staticmethod
    def _vector_to_array(vector: Any) -> np.ndarray:
        return np.array([float(vector.x), float(vector.y), float(vector.z)])

    @staticmethod
    def _quaternion_to_array(quaternion: Any) -> np.ndarray:
        return np.array([
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        ])

    @staticmethod
    def _clean_frame(frame: str) -> str:
        return str(frame or "").strip().lstrip("/")

    @staticmethod
    def _available_tf_frames(tf_edges: dict[str, list[tuple[str, np.ndarray]]]) -> set[str]:
        frames = set(tf_edges.keys())
        for edges in tf_edges.values():
            frames.update(BLT_dataset._edge_frame(edge) for edge in edges)
        return frames

    @staticmethod
    def _is_camera_like_frame(frame: str) -> bool:
        frame_lower = frame.lower()
        return (
            "camera" in frame_lower
            or "zed" in frame_lower
            or "rgb" in frame_lower
        ) and (
            "front" in frame_lower
            or "zed" in frame_lower
            or "rgb" in frame_lower
            or "camera" in frame_lower
        )

    @staticmethod
    def _format_csv_number(value: float) -> str:
        return format(float(value), ".12g")

    @staticmethod
    def _write_image_message(
        output_path: Path,
        timestamp_ns: int,
        msgtype: str,
        msg: Any,
    ) -> Path:
        image = BLT_dataset._message_to_bgr_image(msgtype, msg)
        image_path = output_path / f"{timestamp_ns}.png"
        import cv2

        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write BLT RGB image: {image_path}")
        return image_path

    @staticmethod
    def _write_depth_message(
        output_path: Path,
        timestamp_ns: int,
        msgtype: str,
        msg: Any,
    ) -> Path:
        depth = BLT_dataset._message_to_depth_image(msgtype, msg)
        image_path = output_path / f"{timestamp_ns}.png"
        import cv2

        if not cv2.imwrite(str(image_path), depth):
            raise RuntimeError(f"Could not write BLT depth image: {image_path}")
        return image_path

    @staticmethod
    def _message_to_depth_image(msgtype: str, msg: Any) -> np.ndarray:
        if msgtype.endswith("/CompressedImage"):
            return BLT_dataset._decode_compressed_depth_image(msg)

        if not msgtype.endswith("/Image"):
            raise TypeError(f"Unsupported BLT depth image message type: {msgtype}")

        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        raw = bytes(msg.data)

        if encoding in {"16uc1", "mono16"}:
            return np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
        if encoding == "32fc1":
            depth_m = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
            return np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)

        raise ValueError(f"Unsupported BLT depth image encoding: {msg.encoding}")

    @staticmethod
    def _decode_compressed_depth_image(msg: Any) -> np.ndarray:
        import cv2

        payload = bytes(msg.data)
        format_hint = str(getattr(msg, "format", "") or "").lower()
        candidates = [payload]
        if "compresseddepth" in format_hint and len(payload) > 12:
            candidates.insert(0, payload[12:])

        for candidate in candidates:
            data = np.frombuffer(candidate, dtype=np.uint8)
            depth = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                return depth

        raise ValueError("Could not decode BLT compressedDepth image message")

    @staticmethod
    def _message_to_bgr_image(msgtype: str, msg: Any) -> np.ndarray:
        import cv2

        if msgtype.endswith("/CompressedImage"):
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode BLT compressed image message")
            return image

        if not msgtype.endswith("/Image"):
            raise TypeError(f"Unsupported BLT image message type: {msgtype}")

        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if encoding in {"rgb8", "bgr8"}:
            image = data.reshape((height, width, 3))
            if encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        if encoding in {"mono8", "8uc1"}:
            return data.reshape((height, width))

        if encoding in {"bgra8", "rgba8"}:
            image = data.reshape((height, width, 4))
            if encoding == "rgba8":
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        raise ValueError(f"Unsupported BLT image encoding: {msg.encoding}")


class _FastRosbag1Stream:
    """ROS1 stream reader that skips the expensive per-message index load."""

    def __init__(self, bag_path: Path) -> None:
        self.bag_path = Path(bag_path)
        self.reader = None
        self.typestore = None
        self.connections = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_args):
        self.close()
        return False

    def open(self) -> None:
        try:
            from rosbags.interfaces import MessageDefinitionFormat
            from rosbags.rosbag1.reader import Header, Reader, ReaderError, RecordType
            from rosbags.typesys import Stores, get_types_from_idl, get_types_from_msg, get_typestore
        except ImportError as exc:
            raise RuntimeError(
                "The BLT dataset requires the 'rosbags' Python package to read source bags."
            ) from exc

        reader = Reader(self.bag_path)
        try:
            reader.bio = self.bag_path.open("rb")
            magic = reader.bio.readline().decode()
            if not magic:
                raise ReaderError(f"File {str(self.bag_path)!r} seems to be empty.")
            matches = re.match(r"#ROSBAG V(\d+).(\d+)\n", magic)
            if not matches:
                raise ReaderError("File magic is invalid.")
            major, minor = matches.groups()
            version = int(major) * 100 + int(minor)
            if version != 200:
                raise ReaderError(f"Bag version {version!r} is not supported.")

            header = Header.read(reader.bio, RecordType.BAGHEADER)
            index_pos = header.get_uint64("index_pos")
            conn_count = header.get_uint32("conn_count")
            chunk_count = header.get_uint32("chunk_count")
            if index_pos == 0:
                raise ReaderError("Bag is not indexed, reindex before reading.")
            if chunk_count == 0:
                self.reader = reader
                self.connections = []
                self.typestore = get_typestore(Stores.EMPTY)
                return

            reader.bio.seek(index_pos)
            reader.connections = [reader.read_connection() for _ in range(conn_count)]
            reader.chunk_infos = [reader.read_chunk_info() for _ in range(chunk_count)]
            self.reader = reader
            self.connections = reader.connections

            typestore = get_typestore(Stores.EMPTY)
            typs = {}
            sep = "=" * 80 + "\n"
            for connection in self.connections:
                if connection.msgdef.format == MessageDefinitionFormat.NONE:
                    continue
                if connection.msgdef.data.startswith(f"{sep}IDL: "):
                    for msgdef in connection.msgdef.data.split(sep)[1:]:
                        hdr, idl = msgdef.split("\n", 1)
                        if hdr.startswith("IDL: "):
                            typs.update(get_types_from_idl(idl))
                else:
                    typs.update(get_types_from_msg(connection.msgdef.data, connection.msgtype))
            typestore.register(typs)
            self.typestore = typestore
        except Exception:
            reader.close()
            raise

    def close(self) -> None:
        if self.reader is not None and self.reader.bio is not None:
            self.reader.close()

    def deserialize(self, rawdata: bytes, msgtype: str) -> object:
        if self.typestore is None:
            raise RuntimeError("BLT fast ROS1 stream is not open")
        return self.typestore.deserialize_ros1(rawdata, msgtype)

    def messages(self, connections):
        if self.reader is None or self.reader.bio is None:
            raise RuntimeError("BLT fast ROS1 stream is not open")
        from rosbags.rosbag1.reader import Header, RecordType, read_bytes, read_uint32

        selected_ids = {connection.id for connection in connections}
        connection_by_id = {connection.id: connection for connection in self.connections}
        for chunk_info in self.reader.chunk_infos:
            if selected_ids and not selected_ids.intersection(chunk_info.connection_counts):
                continue
            self.reader.bio.seek(chunk_info.pos)
            chunk_header = self.reader.read_chunk()
            self.reader.bio.seek(chunk_header.datapos)
            rawbytes = chunk_header.decompressor(
                read_bytes(self.reader.bio, chunk_header.datasize)
            )
            chunk = BytesIO(rawbytes)
            while chunk.tell() < len(rawbytes):
                header = Header.read(chunk)
                op = header.get_uint8("op")
                if op == RecordType.CONNECTION:
                    chunk.seek(read_uint32(chunk), os.SEEK_CUR)
                    continue
                if op != RecordType.MSGDATA:
                    chunk.seek(read_uint32(chunk), os.SEEK_CUR)
                    continue

                data = read_bytes(chunk, read_uint32(chunk))
                conn_id = header.get_uint32("conn")
                if selected_ids and conn_id not in selected_ids:
                    continue
                yield connection_by_id[conn_id], header.get_time("time"), data
