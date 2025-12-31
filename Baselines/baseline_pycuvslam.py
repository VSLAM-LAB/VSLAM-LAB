import os.path
from pathlib import Path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class PYCUVSLAM_baseline(BaselineVSLAMLab):
    """PyCuVSLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'pycuvslam', baseline_folder: str = 'PyCuVSLAM') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.700, 0.300)
        self.modes = ['mono', 'rgbd', 'stereo', 'stereo-vi']
        self.camera_models = ['pinhole', 'radtan4', 'radtan5', 'equid4']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self) -> tuple[bool, str]: 
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'install_pycuvslam.txt'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')