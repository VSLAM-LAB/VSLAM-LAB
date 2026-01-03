import os.path
from pathlib import Path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

class VGGT_baseline(BaselineVSLAMLab):
    """vggt helper for VSLAM-LAB Baselines."""
    def __init__(self, baseline_name='vggt', baseline_folder='VGGT'):

        default_parameters = {'verbose': 1, 'mode': 'mono', 'max_rgb': 40}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.150, 0.250)
        self.modes = ['mono']
        self.camera_models = ['pinhole']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'vslamlab_vggt.py'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')