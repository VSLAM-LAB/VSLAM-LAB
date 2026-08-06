"""
Module: VSLAM-LAB - Datasets - dataset_rvp_car.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-06
- License: GPLv3 License
"""

from __future__ import annotations

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.dataset_files.dataset_rvp_handheld import RvpBagsMixin


class RvpCarDataset(RvpBagsMixin, DatasetVSLAMLAB):
    """VBR: A Vision Benchmark in Rome (car sequences) dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "rvp-car") -> None:
        super().__init__(dataset_name)
