import os.path
from pathlib import Path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class MONOGS_baseline(BaselineVSLAMLAB):
    """MonoGS helper for VSLAM-LAB Baselines."""    
    def __init__(self, baseline_name: str = 'monogs', baseline_folder: str = 'MonoGS') -> None:    

        default_parameters = {'verbose': 1, 'mode': 'mono'}    
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.500, 0.550, 0.600)
        self.modes = ['mono', 'rgbd']       
        self.cam_models = ['pinhole', 'radtan4', 'radtan5']
        self.command_style = 'python'


class MONOGS_baseline_dev(MONOGS_baseline):
    """MonoGS-DEV helper for VSLAM-LAB Baselines: the only registered MonoGS baseline. The MonoGS licence (Imperial
    College London) does not allow redistributing copies, so there is no conda package; every user builds the fork
    (github.com/VSLAM-LAB/monogs) from source."""

    def __init__(self):
        super().__init__(baseline_name = 'monogs-dev', baseline_folder =  'MonoGS-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def is_installed(self) -> tuple[bool, str]:
        # `install` (pip -e) compiles the CUDA extensions in place and puts the entry points in the clone's own pixi env
        rasterizer = self.baseline_path / 'submodules' / 'diff-gaussian-rasterization' / 'diff_gaussian_rasterization'
        entry_point = self.baseline_path / '.pixi' / 'envs' / 'default' / 'bin' / 'vslamlab_monogs_mono'
        is_installed = any(rasterizer.glob('_C.cpython-*.so')) and entry_point.is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')