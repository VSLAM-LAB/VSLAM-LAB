import os.path
from pathlib import Path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class MONOGS_baseline(BaselineVSLAMLab):
    """MonoGS helper for VSLAM-LAB Baselines."""    
    def __init__(self, baseline_name: str = 'monogs', baseline_folder: str = 'MonoGS') -> None:    

        default_parameters = {'verbose': 1, 'mode': 'mono'}    
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.500, 0.550, 0.600)
        self.modes = ['mono', 'rgbd']       
        self.camera_models = ['pinhole', 'radtan4', 'radtan5']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
        

class MONOGS_baseline_dev(MONOGS_baseline):
    """MonoGS-DEV helper for VSLAM-LAB Baselines."""     

    def __init__(self):
        super().__init__(baseline_name = 'monogs-dev', baseline_folder =  'MonoGS-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = (self.baseline_path / 'submodules' / 'diff-gaussian-rasterization' / 'build' / 'lib.linux-x86_64-cpython-310' / 'diff_gaussian_rasterization' / '_C.cpython-310-x86_64-linux-gnu.so').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')