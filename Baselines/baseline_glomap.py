import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab
from Baselines.baseline_colmap import COLMAP_baseline

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class GLOMAP_baseline(BaselineVSLAMLab):
    """glomap helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'glomap', baseline_folder: str = 'glomap') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono', 
                              'matcher_type': 'exhaustive', 'use_gpu': 1, 'max_rgb': 200}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = tuple(max(c / 2.0, 0.0) for c in (0.800, 0.400, 0.750))
        self.modes = ['mono']
        self.camera_models = ['pinhole', 'radtan4', 'radtan5', 'radtan8', 'equid4']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def git_clone(self) -> None:
        colmap = COLMAP_baseline()
        if not colmap.is_cloned():
            colmap.git_clone()

        super().git_clone()

    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')