import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class PYCUVSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='pycuvslam', baseline_folder='PyCuVSLAM'):

        default_parameters = {'verbose': 1, 'mode': 'rgbd'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['rgbd']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'install_pycuvslam.txt'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')