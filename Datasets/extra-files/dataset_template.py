from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final
from urllib.parse import urljoin

import numpy as np
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from path_constants import BENCHMARK_RETENTION, Retention

class DATASET_NAME_TEMPLATE_dataset(DatasetVSLAMLab):
    """DATASET_NAME_TEMPLATE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path):
        super().__init__('dataset_name_template', benchmark_path)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames

    def download_sequence_data(self, sequence_name: str) -> None:
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        return

    def create_rgb_csv(self, sequence_name: str) -> None:
        return

    def create_calibration_yaml(self, sequence_name: str) -> None:
        return

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        return

    def remove_unused_files(self, sequence_name: str) -> None:
        return

    def get_download_issues(self, _):
        return