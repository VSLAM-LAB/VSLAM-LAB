from pathlib import Path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class VGGTSLAM_baseline(BaselineVSLAMLab):
    """VGGT-SLAM helper for VSLAM-LAB Baselines."""
    def __init__(self, baseline_name='vggtslam', baseline_folder='VGGT-SLAM'):

        default_parameters = {'verbose': 1, 'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (1.000, 0.050, 0.600)
        self.modes = ['mono']
        self.camera_models = ['pinhole']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
    
class VGGTSLAM_baseline_dev(VGGTSLAM_baseline):
    """VGGT-SLAM-DEV helper for VSLAM-LAB Baselines."""

    def __init__(self):
        super().__init__(baseline_name = 'vggtslam-dev', baseline_folder =  'VGGT-SLAM-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = (self.baseline_path / 'vggt_slam.egg-info' / 'top_level.txt').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')