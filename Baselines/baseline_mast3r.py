import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MAST3R_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='mast3r', baseline_folder='mast3r'):

        default_parameters = {'verbose': 1, 'mode': 'mono', 'max_rgb': 20,
                              'model_name': 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self): 
        return True
    
class MAST3R_baseline_dev(MAST3R_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'mast3r-dev', baseline_folder =  'MASt3R-DEV')

    def is_installed(self):
        is_installed = os.path.isdir(os.path.join(self.baseline_path, 'asmk', 'build')) and os.path.isdir(os.path.join(
            self.baseline_path, 'dust3r', 'croco', 'models', 'curope', 'build'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')