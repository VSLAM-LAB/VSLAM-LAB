import os.path

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class VGGT_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='vggt', baseline_folder='vggt'):

        default_parameters = {'verbose': 1, 'mode': 'mono', 'max_rgb': 20}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self): 
        return True
    
class VGGT_baseline_dev(VGGT_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'vggt-dev', baseline_folder =  'VGGT-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'vslamlab_vggt.py'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')