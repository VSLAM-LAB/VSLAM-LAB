import os.path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MONOGS_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='monogs', baseline_folder='MonoGS'):

        default_parameters = {'verbose': 1, 'mode': 'mono'}    
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'black'
        self.modes = ['mono', 'rgbd']       

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
        
class MONOGS_baseline_dev(MONOGS_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'monogs-dev', baseline_folder =  'MonoGS-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'build', 'lib.linux-x86_64-cpython-311', 'droid_backends.so'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')