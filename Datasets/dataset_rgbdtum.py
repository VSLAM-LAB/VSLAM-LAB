from __future__ import annotations
from urllib.parse import urljoin
from contextlib import suppress
from pathlib import Path
from typing import Final
import pandas as pd
import numpy as np
import yaml

from Datasets.dataset_utilities import undistort_rgb_rad_tan, undistort_depth_rad_tan
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

DEFAULT_DEPTH_FACTOR: Final = 5000.0
TIME_DIFF_THRESH: Final = 0.02  # seconds for RGB/Depth association

# Camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3)
CAMERA_PARAMS = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989, 0.262383, -0.953104, -0.005358,  0.002628, 1.163314),
    "freiburg2": (520.908620, 521.007327, 325.141443, 249.701764, 0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "freiburg3": (535.4,      539.2,      320.1,      247.6,      0.0,      0.0,       0.0,       0.0,       0.0),
}

class RGBDTUM_dataset(DatasetVSLAMLab):
    """TUM RGB-D dataset helper for VSLAMLab benchmark."""
    
    def __init__(self, benchmark_path: str | Path, dataset_name: str = "rgbdtum") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [self._nickname(s) for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        camera = self._camera_from_sequence(sequence_name)

        # .tgz layout on server: <root>/<camera>/<sequence>.tgz
        compressed_name = f"{sequence_name}.tgz"
        download_url = urljoin(self.url_download_root.rstrip("/") + "/", f"{camera}/{compressed_name}")

        # Some archives unpack into a folder whose name differs (validation → secret for f1/f2)
        decompressed_name = sequence_name.replace("validation", "secret") if camera in ("freiburg1", "freiburg2") else sequence_name
        compressed_file = self.dataset_path / compressed_name
        decompressed_folder = self.dataset_path / decompressed_name

        if not compressed_file.exists():
            downloadFile(download_url, str(self.dataset_path))

        if not sequence_path.exists():
            decompressFile(str(compressed_file), str(self.dataset_path))
            # If archive expands to a different folder name, normalize it to sequence_path
            if decompressed_folder.exists() and decompressed_folder != sequence_path:
                decompressed_folder.replace(sequence_path)

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for raw, dst in (("rgb", "rgb_0"), ("depth", "depth_0")):
            src, tgt = sequence_path / raw, sequence_path / dst
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        """Associate RGB and Depth using nearest timestamp within tolerance."""
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        if rgb_csv.exists():
            return

        rgb_txt = sequence_path / "rgb.txt"
        depth_txt = sequence_path / "depth.txt"
        if not (rgb_txt.exists() and depth_txt.exists()):
            raise FileNotFoundError(f"Missing rgb/depth txt in {sequence_path}")

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
        merged["ts_rgb0 (s)"] = merged["ts"].map(lambda x: f"{x:.6f}")
        merged["ts_depth0 (s)"] = merged["ts"].map(lambda x: f"{x:.6f}")  # after nearest match they’re nearly equal
        merged["path_rgb0"] = merged["rgb_path"].astype(str).str.replace(r"^rgb/", "rgb_0/", regex=True)
        merged["path_depth0"] = merged["depth_path"].astype(str).str.replace(r"^depth/", "depth_0/", regex=True)

        out = merged[["ts_rgb0 (s)", "path_rgb0", "ts_depth0 (s)", "path_depth0"]]
        tmp = rgb_csv.with_suffix(".csv.tmp")
        try:
            out.to_csv(tmp, index=False)
            tmp.replace(rgb_csv)
        finally:
            with suppress(FileNotFoundError):
                tmp.unlink()

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_csv = sequence_path / "rgb.csv"
        camera = self._camera_from_sequence(sequence_name)

        fx, fy, cx, cy, k1, k2, p1, p2, k3 = CAMERA_PARAMS[camera]
        
        # Undistort for f1/f2 (f3 is already zero-distortion by convention here)
        if camera in ("freiburg1", "freiburg2"):
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            D = np.array([k1, k2, p1, p2, k3], dtype=float)
            fx, fy, cx, cy = undistort_rgb_rad_tan(str(rgb_csv), str(sequence_path), K, D)
            undistort_depth_rad_tan(str(rgb_csv), str(sequence_path), K, D)

        camera0 = {"model": "Pinhole", "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
        rgbd = {"depth0_factor": float(DEFAULT_DEPTH_FACTOR)}
        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd=rgbd)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = sequence_path / "groundtruth.csv"

        if not groundtruth_txt.exists():
            return
        
        if groundtruth_csv.exists() and groundtruth_csv.stat().st_mtime >= groundtruth_txt.stat().st_mtime:
            return

        tmp = groundtruth_csv.with_suffix(".csv.tmp")
        with open(groundtruth_txt, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
            # Skip first 3 lines (header/comments), then write CSV header + values
            lines = fin.readlines()
            data_lines = [ln.strip() for ln in lines[3:] if ln.strip() and not ln.lstrip().startswith("#")]
            fout.write("ts,tx,ty,tz,qx,qy,qz,qw\n")
            for ln in data_lines:
                fout.write(",".join(ln.split()) + "\n")
        tmp.replace(groundtruth_csv)
        with suppress(FileNotFoundError):
            tmp.unlink()

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for name in ("accelerometer.txt", "depth.txt", "groundtruth.txt", "rgb.txt"):
            with suppress(FileNotFoundError):
                (sequence_path / name).unlink()

    @staticmethod
    def _nickname(seq: str) -> str:
        s = seq.replace("rgbd_dataset_freiburg", "fr")
        s = s.replace("_", " ")
        s = s.replace("validation", "v").replace("structure", "st").replace("texture", "tx")
        s = s.replace("walking xyz", "walk")
        return s
    
    @staticmethod
    def _camera_from_sequence(name: str) -> str:
        for cam in ("freiburg1", "freiburg2", "freiburg3"):
            if cam in name:
                return cam
        raise ValueError(f"Cannot infer camera from sequence name: {name}")