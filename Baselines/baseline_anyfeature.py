import os.path
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class ANYFEATURE_baseline(BaselineVSLAMLab):
    """AnyFeature-VSLAM helper for VSLAM-LAB Baselines."""    

    def __init__(self, baseline_name: str = 'anyfeature', baseline_folder: str = 'AnyFeature-VSLAM') -> None:    
        
        default_parameters = {'verbose': 1, 'mode': 'mono', 
                              'vocabulary_folder': str(VSLAMLAB_BASELINES / baseline_folder / 'anyfeature_vocabulary'),
                              'feature': 'akaze61',
                              'feature_yaml': str(VSLAMLAB_BASELINES / baseline_folder / 'settings' / 'feature_name_to_fill_settings.yaml')}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.350, 0.300, 0.700)
        self.modes = ['mono']
        self.camera_models = ['pinhole', 'radtan4', 'radtan5']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        command = super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

        # If feature_yaml has not been provided it has to match the feature selected
        import re
        match = re.search(r'feature:(\S+)', command)
        feature_name = match.group(1)
        command = command.replace('feature_name_to_fill', feature_name)

        return command

    def git_clone(self) -> None:
        super().git_clone()
        self.anyfeature_download_vocabulary()

    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')

    def anyfeature_download_vocabulary(self) -> None: 
        REPO_ID = "fontan/anyfeature_vocabulary"
        vocabulary_files = [
            "ORBvoc.txt",
            "Akaze61_DBoW2_voc.txt",
            "Brisk_DBoW2_voc.txt",
            "Surf64_DBoW2_voc.txt",
            "Sift128_DBoW2_voc.txt",
            "Kaze64_DBoW2_voc.txt",
            "R2d2_DBoW2_voc.txt"
        ]

        vocabulary_folder = os.path.join(self.baseline_path, 'anyfeature_vocabulary')
        if not os.path.isdir(vocabulary_folder):
            print_msg(f"\n{SCRIPT_LABEL}", f"Download vocabulary files to: {vocabulary_folder}",'info')
            os.makedirs(vocabulary_folder, exist_ok=True)

        for vocabulary_file in vocabulary_files:

            if os.path.isfile(os.path.join(vocabulary_folder, vocabulary_file)):
                continue

            print_msg(f"{SCRIPT_LABEL}", f"Download vocabulary file: {vocabulary_file}",'info')
            dataset = pd.read_csv(
                hf_hub_download(repo_id=REPO_ID, filename=vocabulary_file, repo_type="dataset")
            )
            dataset.to_csv(os.path.join(vocabulary_folder, vocabulary_file), sep='\t', index=False)


class ANYFEATURE_baseline_dev(ANYFEATURE_baseline):
    """AnyFeature-VSLAM-DEV helper for VSLAM-LAB Baselines."""       

    def __init__(self):
        super().__init__(baseline_name = 'anyfeature-dev', baseline_folder = 'AnyFeature-VSLAM-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = (self.baseline_path / 'bin' / 'vslamlab_anyfeature_mono').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')