import os.path
import tarfile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ORBSLAM2AUTOTUNE_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='orbslam2autotune', baseline_folder='ORB_SLAM2_AUTOTUNE'):
        
        default_parameters = {'verbose': 1, 'mode': 'mono', 
                              'vocabulary': os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'Vocabulary', 'ORBvoc.txt')}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def git_clone(self):
        super().git_clone()
        self.orbslam2_download_vocabulary()

    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')

    def orbslam2_download_vocabulary(self): # Download ORBvoc.txt
        vocabulary_folder = os.path.join(self.baseline_path, 'Vocabulary')
        vocabulary_txt = os.path.join(vocabulary_folder, 'ORBvoc.txt')
        if not os.path.isfile(vocabulary_txt):
            print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {self.baseline_path}/ORBvoc.txt",'info')
            file_path = hf_hub_download(repo_id='vslamlab/orbslam2_vocabulary', filename='ORBvoc.txt.tar.gz', repo_type='model',
                                        local_dir=vocabulary_folder)
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=vocabulary_folder)

class ORBSLAM2AUTOTUNE_baseline_dev(ORBSLAM2AUTOTUNE_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'orbslam2autotune-dev', baseline_folder = 'ORB_SLAM2_AUTOTUNE-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_orbslam2_mono' ))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
    
