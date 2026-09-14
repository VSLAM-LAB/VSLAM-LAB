"""
Module: VSLAM-LAB - Baselines - baseline_vggtslam.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2026-01-05
- Updated: 2026-09-15
- License: GPLv3 License

VGGT-SLAM 2.0 (Maggio & Carlone, RSS 2026): feed-forward dense monocular SLAM built on VGGT submaps
aligned on the SL(4) manifold, with SALAD retrieval and attention-verified loop closures.
Source: https://github.com/alejandrofontan/VGGT-SLAM-2-VSLAM-LAB (fork of MIT-SPARK/VGGT-SLAM).
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class VGGTSLAM_baseline(BaselineVSLAMLAB):
    """VGGT-SLAM helper for VSLAM-LAB Baselines (conda package `vggtslam-vslamlab`)."""

    def __init__(self, baseline_name: str = 'vggtslam', baseline_folder: str = 'VGGT-SLAM') -> None:
        # Tuning knobs mirror upstream main.py / evals/eval_tum.sh defaults
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'submap_size': 16,        # new frames per submap (VRAM scales with it)
                              'max_loops': 1,           # loop closures per submap, 0 disables
                              'min_disparity': 50,      # optical-flow disparity (px) to accept a keyframe
                              'conf_threshold': 25,     # % of lowest-confidence points dropped per submap
                              'lc_thres': 0.95}         # SALAD retrieval threshold, higher = more loop candidates

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (1.000, 0.050, 0.600)
        self.modes = ['mono']
        self.cam_models = ['pinhole']  # VGGT predicts its own intrinsics; no calibration is consumed
        self.command_style = 'python'


class VGGTSLAM_baseline_dev(VGGTSLAM_baseline):
    """VGGT-SLAM-DEV helper for VSLAM-LAB Baselines (source build of the fork, editable install)."""

    def __init__(self) -> None:
        super().__init__(baseline_name='vggtslam-dev', baseline_folder='VGGT-SLAM-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)

    def is_installed(self) -> tuple[bool, str]:
        # PEP 660 editable installs leave no egg-info in the source tree; the console script from setup.py is the build artifact
        is_installed = (VSLAM_LAB_DIR / '.pixi' / 'envs' / self.baseline_name / 'bin' / 'vslamlab_vggtslam_mono').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
