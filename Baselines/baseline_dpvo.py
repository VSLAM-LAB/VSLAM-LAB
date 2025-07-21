import os.path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class DPVO_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='dpvo', baseline_folder='DPVO'):
       
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'network': f'{os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'dpvo.pth')}'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'red'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def git_clone(self):
        super().git_clone()
        self.dpvo_download_weights()
    
    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')

    def dpvo_download_weights(self): # Download dpvo.pth
        weights_pth = os.path.join(self.baseline_path, 'dpvo.pth')
        if not os.path.exists(weights_pth):
            print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {self.baseline_path}/dpvo.pth",'info')
            file_path = hf_hub_download(repo_id='vslamlab/dpvo_weights', filename='models.zip', repo_type='model',
                                        local_dir=self.baseline_path)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.baseline_path)

class DPVO_baseline_dev(DPVO_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'dpvo-dev', baseline_folder =  'DPVO-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'build', 'lib.linux-x86_64-cpython-311', 'vslamlab_dpvo_mono.py'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
