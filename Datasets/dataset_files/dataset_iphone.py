from __future__ import annotations

import csv
import yaml
import numpy as np
from urllib.parse import urljoin
from typing import Final, Any
from collections.abc import Iterable

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION

MAX_NICKNAME_LEN: Final = 15


class IPHONE_dataset(DatasetVSLAMLab):
    """IPHONE dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "iphone") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names
        

    def download_sequence_data(self, sequence_name: str) -> None:
        print(f"Sequence '{sequence_name}' is marked as 'local'. Please ensure the data is available at {self.sequence_path(sequence_name)}.")
    def create_rgb_folder(self, sequence_name: str) -> None: 
        pass
    def create_rgb_csv(self, sequence_name: str) -> None:
        pass
    def create_calibration_yaml(self, sequence_name: str) -> None:
        pass
    def create_groundtruth_csv(self, sequence_name: str) -> None: 
        pass
    def remove_unused_files(self, sequence_name: str) -> None:
        pass
    