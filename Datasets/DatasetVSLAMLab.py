"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2024-07-12
- Updated: 2025-12-30
- License: GPLv3 License

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.

"""

import sys
import yaml
from loguru import logger
from pathlib import Path
from typing import List, Union
from abc import ABC, abstractmethod

from utilities import ws, print_msg
from path_constants import VSLAM_LAB_DIR
from Datasets.dataset_calibration import (
    _get_rgb_yaml_section,
    _get_imu_yaml_section,
    _get_rgbd_yaml_section
)

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class DatasetVSLAMLab(ABC):
    """Base dataset class for VSLAM-LAB."""
    def __init__(self, dataset_name: str, benchmark_path: Union[str, Path]) -> None:  
        # Basic fields
        self.dataset_name: str = dataset_name
        self.dataset_color: str = "\033[38;2;255;165;0m"
        self.dataset_label: str = f"{self.dataset_color}{dataset_name}\033[0m"
        self.dataset_folder: str = dataset_name.upper()

        # Paths
        self.benchmark_path: Path = Path(benchmark_path)
        self.dataset_path: Path = self.benchmark_path / self.dataset_folder
        self.yaml_file: Path = VSLAM_LAB_DIR / "Datasets" / f"dataset_{self.dataset_name}.yaml"

        # Load YAML config
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.sequence_names: List[str] = cfg["sequence_names"]
        self.rgb_hz: float = float(cfg["rgb_hz"])
        self.modes: List[str] = cfg.get("modes", ["mono"])
        self.sequence_nicknames: List[str] = []

    ####################################################################################################################
    # Download methods
    def download_sequence(self, sequence_name: str) -> None:

        # Check if sequence is already available
        sequence_availability = self.check_sequence_availability(sequence_name, verbose=True)
        if sequence_availability == "available":
            #print(f"{SCRIPT_LABEL}Sequence {self.dataset_color}{sequence_name}:\033[92m downloaded\033[0m")
            return
        if sequence_availability == "corrupted":
            logger.error(f"\n{ws(4)}Files in sequence {sequence_name} are corrupted.\n{ws(4)}Removing and downloading again sequence {sequence_name}.\n{ws(4)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY ")
            sys.exit(1)

        # Download process
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.download_process(sequence_name)

    def download_process(self, sequence_name: str) -> None:
        msg = f"Downloading sequence {self.dataset_color}{sequence_name}\033[0m from dataset {self.dataset_color}{self.dataset_name}\033[0m ..."
        print_msg(SCRIPT_LABEL, msg)
        self.download_sequence_data(sequence_name)
        self.create_rgb_folder(sequence_name)
        self.create_rgb_csv(sequence_name)
        self.create_imu_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.remove_unused_files(sequence_name)
        
    # ---- Abstract hooks that concrete datasets must implement ----
    @abstractmethod
    def download_sequence_data(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_rgb_folder(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_rgb_csv(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_imu_csv(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_calibration_yaml(self, sequence_name: str) -> None: ...
    @abstractmethod
    def create_groundtruth_csv(self, sequence_name: str) -> None: ...
    
    def remove_unused_files(self, sequence_name: str) -> None: 
        pass
    def get_download_issues(self, sequence_names: List[str]) -> dict:
        return {}

    def write_calibration_yaml(self, sequence_name: str, rgb=None, rgbd=None, imu=None) -> None:
        sequence_path = self.dataset_path / sequence_name
        calibration_yaml = sequence_path / 'calibration.yaml'
        
        yaml_content_lines = ["%YAML 1.2", "---",]

        if rgb or rgbd:    
            yaml_content_lines.extend(["cameras:"])
            if rgb:
                for rgb_i in rgb:
                    yaml_content_lines.extend(_get_rgb_yaml_section(rgb_i, sequence_name, self.dataset_path))
            if rgbd:
                for rgbd_i in rgbd:
                    yaml_content_lines.extend(_get_rgbd_yaml_section(rgbd_i, sequence_name, self.dataset_path))

        if imu:
            yaml_content_lines.extend(["\nimus:"])
            for imu_i in imu:
                yaml_content_lines.extend(_get_imu_yaml_section(imu_i))

        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def check_sequence_availability(self, sequence_name: str, verbose: bool = True) -> str:
        sequence_path = self.dataset_path / sequence_name
        if sequence_path.is_dir():
            sequence_complete = self.check_sequence_integrity(sequence_name, verbose=verbose)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

    def check_sequence_integrity(self, sequence_name: str, verbose: bool) -> bool:
        sequence_path = self.dataset_path / sequence_name

        # Define requirements: (Path, Description, is_directory)
        requirements = [
            (sequence_path, "Sequence folder", True),
            (sequence_path / 'rgb_0', "RGB folder", True),
            (sequence_path / 'rgb.csv', "RGB timestamp CSV", False),
            (sequence_path / 'calibration.yaml', "Calibration YAML", False),
        ]
        if 'stereo' in self.modes:
            requirements.append((sequence_path / 'rgb_1', "Right RGB folder", True))
        if 'mono-vi' in self.modes:
            requirements.append((sequence_path / 'imu_0.csv', "IMU CSV", False))

        # Check all requirements
        complete_sequence = True
        for path_obj, desc, should_be_dir in requirements:
            exists = path_obj.is_dir() if should_be_dir else path_obj.is_file()
            if not exists:
                if verbose:
                    logger.error(f"\n{ws(4)}Missing {desc}: {path_obj} !!!!!")
                complete_sequence = False

        return complete_sequence

    ####################################################################################################################
    # Utils

    def contains_sequence(self, sequence_name_ref: str) -> bool:
        return sequence_name_ref in self.sequence_names

    def print_sequence_names(self) -> None:
        print(self.sequence_names)

    def print_sequence_nicknames(self) -> None:
        print(self.sequence_nicknames)

    def get_sequence_names(self) -> list:
        return self.sequence_names

    def get_sequence_nicknames(self) -> list:
        return self.sequence_nicknames

    def get_sequence_nickname(self, sequence_name_ref: str) -> str:
        idx = self.sequence_names.index(sequence_name_ref)
        return self.sequence_nicknames[idx]