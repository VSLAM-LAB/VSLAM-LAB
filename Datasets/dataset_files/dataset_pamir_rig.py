"""
Module: VSLAM-LAB - Datasets - dataset_pamir_rig.py
- Author: Alejandro Fontan
- Assisted by: Claude (Sonnet 5)
- Version: 1.0
- Created: 2026-08-01
- License: GPLv3 License
"""

from __future__ import annotations

from typing import ClassVar

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.dataset_files.dataset_pamir import PamirBagsMixin


class PamirRigDataset(PamirBagsMixin, DatasetVSLAMLAB):
    """Pamir shipwreck underwater visual-inertial dataset helper for VSLAM-LAB benchmark (two-camera rig dive)."""

    SEQUENCE_BAGS: ClassVar[dict[str, list[str]]] = {
        "2024_OnRig": ["2024_LeftOnRig", "2024_RightOnRig"],
    }

    def __init__(self, dataset_name: str = "pamir-rig") -> None:
        super().__init__(dataset_name)

        # Get Hugging Face repo id
        self.hf_repo_id: str = self.cfg["hf_repo_id"]
