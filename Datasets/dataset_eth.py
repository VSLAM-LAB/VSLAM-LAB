from __future__ import annotations
from typing import Iterable, Tuple, Final
from urllib.parse import urljoin
from pathlib import Path
import yaml
import csv

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

MAX_NICKNAME_LEN: Final = 15
DEFAULT_DEPTH_FACTOR: Final = 5000.0

class ETH_dataset(DatasetVSLAMLab):
    """ETH dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "eth") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.replace("_", " ")[:MAX_NICKNAME_LEN] for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        for mode in self.modes:
            compressed_name = f"{sequence_name}_{mode}.zip"
            download_url = urljoin(self.url_download_root.rstrip("/") + "/", f"datasets/{compressed_name}")
            
            compressed_file = self.dataset_path / compressed_name
            decompressed_folder = self.dataset_path / sequence_name

            if not compressed_file.exists():
                downloadFile(download_url, str(self.dataset_path))
            
            # Decompress only if needed
            needs_depth = not (decompressed_folder / "depth").exists() and not (decompressed_folder / "depth_0").exists()
            needs_mono  = not decompressed_folder.exists()

            if (mode == "mono" and needs_mono) or (mode == "rgbd" and needs_depth):
                decompressFile(str(compressed_file), str(self.dataset_path))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        for raw, dst in (("rgb", "rgb_0"), ("depth", "depth_0")):
            src = sequence_path / raw
            tgt = sequence_path / dst
            if src.is_dir() and not tgt.exists():
                src.replace(tgt)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_entries   = list(self._iter_entries(sequence_path / "rgb.txt",   "rgb/",   "rgb_0/"))
        depth_entries = list(self._iter_entries(sequence_path / "depth.txt", "depth/", "depth_0/"))

        out = sequence_path / "rgb.csv"
        tmp = out.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb0 (s)", "path_rgb0", "ts_depth0 (s)", "path_depth0"])
            for (ts_r, path_r), (ts_d, path_d) in zip(rgb_entries, depth_entries):
                w.writerow([ts_r, path_r, ts_d, path_d])
        tmp.replace(out)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        calib_txt = sequence_path / "calibration.txt"

        with open(calib_txt, "r", encoding="utf-8") as f:
            first = f.readline().split()
        
        fx, fy, cx, cy = map(float, first[:4])

        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        rgbd = {"depth0_factor": float(DEFAULT_DEPTH_FACTOR)}

        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0, rgbd=rgbd)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        if not groundtruth_txt.exists():
            raise FileNotFoundError(f"Missing groundtruth: {groundtruth_txt}")

        with open(groundtruth_txt, "r", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts","tx","ty","tz","qx","qy","qz","qw"])
            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                w.writerow(s.split())
        tmp.replace(groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        sequence_path = Path(self.dataset_path) / sequence_name
        for name in ("calibration.txt","groundtruth.txt","rgb.txt","depth.txt","associated.txt"):
            (sequence_path / name).unlink(missing_ok=True)

    @staticmethod
    def _iter_entries(txt_path: Path, old_prefix: str, new_prefix: str) -> Iterable[Tuple[str, str]]:
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing file: {txt_path}")
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                ts, path = s.split(None, 1)
                if path.startswith(old_prefix):
                    path = new_prefix + path[len(old_prefix):]
                yield ts, path