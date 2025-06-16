import os.path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class PYSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='pyslam', baseline_folder='PYSLAM'):

        default_parameters = {'verbose': 1, 'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
    
class PYSLAM_baseline_dev(PYSLAM_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'pyslam-dev', baseline_folder =  'PYSLAM-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'thirdparty', 'gtsam', 'install', 'bin', 'gtwrap', 'pybind_wrap.py'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')