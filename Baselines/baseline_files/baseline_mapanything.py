"""
Module: VSLAM-LAB - Baselines - baseline_mapanything.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: GPLv3 License

MapAnything (Keetha et al., 3DV 2026): one feed-forward pass over a window of frames regressing metric
geometry and cameras from any mix of images, intrinsics, depth and poses. Modes:
- mono: images only (`use_calibration: 1` also feeds the calibration.yaml intrinsics),
- rgbd: images + intrinsics + metric depth from the rgb csv (sensor depth or the fastfoundationstereo capability).
The window is the experiment's rgb csv (rgb_max / rgb_step / rgb_placecell); poses are metric.
Source: https://github.com/VSLAM-LAB/map-anything-VSLAM-LAB (fork of facebookresearch/map-anything).
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class MAPANYTHING_baseline(BaselineVSLAMLAB):
    """MapAnything helper for VSLAM-LAB Baselines (editable install of the checkout in the `mapanything` env)."""

    def __init__(self, baseline_name: str = 'mapanything', baseline_folder: str = 'MapAnything') -> None:
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'rgb_max': 100,             # frames in the single window (memory-efficient inference, see wrapper)
                              'model': 'default',         # default (facebook/map-anything, CC BY-NC) | apache | v1 | apache-v1 | a Hugging Face id
                              'use_calibration': 0,       # mono only: 1 feeds calibration.yaml intrinsics (pinhole); rgbd always does
                              'memory_efficient': 1,      # run the dense heads in minibatches (upstream default for inference)
                              'minibatch_size': 0}        # 0 = adaptive from free GPU memory; 1 = smallest footprint

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.950, 0.550, 0.100)
        self.modes = ['mono', 'rgbd']
        self.cam_models = ['pinhole']  # intrinsics are only consumed undistorted; images-only mono tolerates distortion
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        # `pip install -e .` (pixi `install` task) leaves the package's dist-info next to the editable finder in the env
        env = VSLAM_LAB_DIR / '.pixi' / 'envs' / self.baseline_name
        is_installed = any(env.glob('lib/python3.*/site-packages/__editable__.mapanything-*.pth'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
