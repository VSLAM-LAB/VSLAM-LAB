import os
import yaml
import shutil
from pathlib import Path
from inputimeout import inputimeout, TimeoutOccurred
import csv

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.DatasetIssues import _get_dataset_issue

class TARTANAIR_dataset(DatasetVSLAMLab):
    """TARTANAIR dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "tartanair") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]
        self.url_download_gt_root: str = cfg["url_download_gt_root"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

    def download_sequence_data(self, sequence_name: str) -> None:

        # Variables
        compressed_name = 'tartanair-test-mono-release'
        compressed_name_ext = compressed_name + '.tar.gz'
        decompressed_name = compressed_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, os.path.join(self.dataset_path, compressed_name))

        # Download the gt
        if not os.path.exists(os.path.join(self.dataset_path, 'tartanair_cvpr_gt')):
            compressed_name = '3p1sf0eljfwrz4qgbpc6g95xtn2alyfk'
            compressed_name_ext = compressed_name + '.zip'
            decompressed_name = 'tartanair_cvpr_gt'

            compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
            decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

            download_url = self.url_download_gt_root
            if not os.path.exists(compressed_file):
                downloadFile(download_url, self.dataset_path)

            decompressFile(compressed_file, os.path.join(self.dataset_path, decompressed_name))

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        rgb_path_0 = os.path.join(self.dataset_path, 'tartanair-test-mono-release', 'mono', sequence_name)
        if not os.path.exists(rgb_path_0):
            return

        for png_file in os.listdir(rgb_path_0):
            if png_file.endswith(".png"):
                shutil.move(os.path.join(rgb_path_0, png_file), os.path.join(rgb_path, png_file))

        shutil.rmtree(rgb_path_0)

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb_0')
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        tmp_path = os.path.join(sequence_path, "rgb.csv.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb0 (s)", "path_rgb0"])
            for filename in rgb_files:
                name, _ = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                w.writerow([f"{ts:.5f}", f"rgb_0/{filename}"])

        os.replace(tmp_path, rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
        camera0 = {"model": "Pinhole", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        self.write_calibration_yaml(sequence_name=sequence_name, camera0=camera0)


    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path    = os.path.join(self.dataset_path, sequence_name)
        groundtruth_csv  = os.path.join(sequence_path, "groundtruth.csv")
        groundtruth_txt  = os.path.join(self.dataset_path, "tartanair_cvpr_gt", "mono_gt", f"{sequence_name}.txt")
        tmp_path         = os.path.join(sequence_path, "groundtruth.csv.tmp")

        with open(groundtruth_txt, "r", encoding="utf-8") as fin, \
            open(tmp_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

            frame_idx = 0
            for line in fin:
                line = line.strip()
                parts = line.split()
                ts = frame_idx / float(self.rgb_hz)
                frame_idx += 1
                tx, ty, tz, qx, qy, qz, qw = parts[:7]
                w.writerow([f"{ts:.5f}", tx, ty, tz, qx, qy, qz, qw])

        os.replace(tmp_path, groundtruth_csv)

    def remove_unused_files(self, sequence_name: str) -> None:
        dataset_folder = os.path.join(self.dataset_path, 'tartanair-test-mono-release')
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        gt_folder = os.path.join(self.dataset_path, 'tartanair_cvpr_gt')
        if os.path.exists(gt_folder):
            shutil.rmtree(gt_folder)

    def get_download_issues(self, _):
        return [_get_dataset_issue(issue_id="complete_dataset", dataset_name=self.dataset_name, size_gb=8.2)]

    def download_process(self, _):
        for sequence_name in self.sequence_names:
            super().download_process(sequence_name)