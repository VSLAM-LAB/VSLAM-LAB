import os.path
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class DROIDSLAM_baseline(BaselineVSLAMLab):
    """DROID-SLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'droidslam', baseline_folder: str = 'DROID-SLAM') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono', 
                              'upsample': 0, 'weights': f'{VSLAMLAB_BASELINES / baseline_folder / 'droid.pth'}'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.900, 0.600, 0.400) # 'orange'
        self.modes = ['mono', 'rgbd', 'stereo']
        self.camera_models = ['pinhole', 'radtan4', 'radtan5']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
    

class DROIDSLAM_baseline_dev(DROIDSLAM_baseline):
    """DROID-SLAM-DEV helper for VSLAM-LAB Baselines."""

    def __init__(self):
        super().__init__(baseline_name = 'droidslam-dev', baseline_folder =  'DROID-SLAM-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def git_clone(self):
        super().git_clone()
        self.droidslam_download_weights()
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = (self.baseline_path / 'build' / 'lib.linux-x86_64-cpython-311' / 'droid_backends.so').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
        
    def droidslam_download_weights(self): # Download droid.pth
        weights_pth = self.baseline_path / 'droid.pth'
        if not os.path.exists(weights_pth):
            print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {self.baseline_path}/droid.pth",'info')
            file_path = hf_hub_download(repo_id=f'vslamlab/droidslam', filename='droid.pth', repo_type='model',
                                        local_dir=self.baseline_path)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.baseline_path)