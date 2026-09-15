"""
Module: VSLAM-LAB - Baselines - baseline_vggt_omega.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: GPLv3 License

VGGT-Omega (Wang et al., CVPR 2026): successor of VGGT, one feed-forward pass over a window of
frames predicting cameras and depth; no tracking, no loop closure. The window is the experiment's
rgb csv, so rgb_max / rgb_step / rgb_placecell decide which frames it covers. Weights are gated on
Hugging Face (facebook/VGGT-Omega): request access there and log in with `huggingface-cli login`
once, or point `checkpoint_dir` in the settings yaml at manually downloaded .pt files.
Source: https://github.com/VSLAM-LAB/vggt-omega-VSLAM-LAB (mirror of facebookresearch/vggt-omega).
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class VGGTOMEGA_baseline(BaselineVSLAMLAB):
    """VGGT-Omega helper for VSLAM-LAB Baselines (source checkout, run in place)."""

    def __init__(self, baseline_name: str = 'vggt-omega', baseline_folder: str = 'VGGT-Omega') -> None:
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'rgb_max': 40,                  # frames in the single window (~13 GB at 100 frames per upstream)
                              'weights': '416-reproduce'}     # 416-reproduce (authors' reference for benchmarks) | 512 (in-the-wild)

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.550, 0.100, 0.350)
        self.modes = ['mono']
        self.cam_models = ['pinhole']  # predicts its own intrinsics; no calibration is consumed
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        # No build step: the entry point runs from the checkout (execute-mono has cwd Baselines/VGGT-Omega)
        is_installed = (self.baseline_path / 'vslamlab_vggt_omega.py').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
