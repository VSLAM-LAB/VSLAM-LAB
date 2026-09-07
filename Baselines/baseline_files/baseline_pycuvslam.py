import os.path
from pathlib import Path
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class PYCUVSLAM_baseline(BaselineVSLAMLAB):
    """PyCuVSLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'pycuvslam', baseline_folder: str = 'PyCuVSLAM') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.700, 0.300)
        self.modes = ['mono', 'rgbd', 'stereo', 'stereo-vi']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5', 'equid4']
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'install_pycuvslam.txt'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')