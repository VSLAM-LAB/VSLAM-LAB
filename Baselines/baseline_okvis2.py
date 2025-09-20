import os.path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class OKVIS2_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='okvis2', baseline_folder='OKVIS2'):

        default_parameters = {'verbose': 1, 'mode': 'mono-vi', 
                              'vocabulary': os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'resources', 'small_voc.yml.gz')}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono-vi']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
    
class OKVIS2_baseline_dev(OKVIS2_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'okvis2-dev', baseline_folder = 'OKVIS2-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_okvis2_mono_vi'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)') 
    
