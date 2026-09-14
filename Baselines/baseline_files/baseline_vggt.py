"""
Module: VSLAM-LAB - Baselines - baseline_vggt.py
- Author: Alejandro Fontan Villacampa
- Version: 1.1
- Created: 2026-01-03
- Updated: 2026-09-15
- License: GPLv3 License

VGGT (Wang et al., CVPR 2025): one feed-forward pass over a window of frames predicting cameras,
depth and points; no tracking, no loop closure. The window is the experiment's rgb csv, so rgb_max /
rgb_step / rgb_placecell decide which frames it covers. Source: https://github.com/VSLAM-LAB/vggt-VSLAM-LAB.
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class VGGT_baseline(BaselineVSLAMLAB):
    """VGGT helper for VSLAM-LAB Baselines (source checkout, run in place)."""

    def __init__(self, baseline_name: str = 'vggt', baseline_folder: str = 'VGGT') -> None:
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'rgb_max': 40,              # frames in the single VGGT window (GPU memory scales with it)
                              'precision': 'autocast'}    # autocast (upstream demo) | bf16 (whole model cast, ~half the memory)

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.150, 0.250)
        self.modes = ['mono']
        self.cam_models = ['pinhole']  # VGGT predicts its own intrinsics; no calibration is consumed
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        # No build step: the entry point runs from the checkout (execute-mono has cwd Baselines/VGGT)
        is_installed = (self.baseline_path / 'vslamlab_vggt.py').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
