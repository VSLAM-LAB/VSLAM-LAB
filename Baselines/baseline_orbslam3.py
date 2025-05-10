import os.path
import tarfile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ORBSLAM3_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='orbslam3', baseline_folder='ORB_SLAM3'):
        default_parameters = {'verbose': 1,
                              'vocabulary': os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'Vocabulary', 'ORBvoc.txt'),
                              'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'blue' #change

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

        # Add the mode argument
        mode = self.default_parameters['mode']
        if 'mode' in exp.parameters:
            mode = exp.parameters['mode']

        # Valid modes that require a specific suffix for the 'execute' part of the command.
        # The pattern is 'execute' -> 'execute_MODE'.
        executable_modes = [
            "mono", "stereo", "rgbd",
            "mono_inertial", "stereo_inertial", "rgbd_inertial"
        ]

        if mode in executable_modes:
            vslamlab_command = vslamlab_command.replace('execute', f'execute_{mode}')
        
        return vslamlab_command

    def is_cloned(self):
        return True
    
    def is_installed(self):
        return True
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: vslamlab_orbslam2_mono")

    def orbslam3_download_vocabulary(self): # Download ORBvoc.txt
        vocabulary_folder = os.path.join(self.baseline_path, 'Vocabulary')
        vocabulary_txt = os.path.join(vocabulary_folder, 'ORBvoc.txt')
        if not os.path.isfile(vocabulary_txt):
            print_msg(SCRIPT_LABEL, "Downloading ORBvoc.txt ...",'info')
            file_path = hf_hub_download(repo_id='vslamlab/orbslam2', filename='ORBvoc.txt.tar.gz', repo_type='model',
                                        local_dir=vocabulary_folder)
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=vocabulary_folder)

        return os.path.isfile(vocabulary_txt)

    def modify_yaml_parameter(self,yaml_file, section_name, parameter_name, new_value):
        with open(yaml_file, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            if f"{section_name}.{parameter_name}" in line:
                line = f"{section_name}.{parameter_name}: {new_value}\n"
            modified_lines.append(line)

        with open(yaml_file, 'w') as file:
            file.writelines(modified_lines)

    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.orbslam3_download_vocabulary() 
        self.download_vslamlab_settings()
        super().execute(command, exp_it, exp_folder, timeout_seconds)
