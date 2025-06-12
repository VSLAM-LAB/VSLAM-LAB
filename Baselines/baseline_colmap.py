import os.path
from huggingface_hub import hf_hub_download

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class COLMAP_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='colmap', baseline_folder='colmap'):

        default_parameters = {'verbose': 1, 'matcher_type': 'exhaustive', 'use_gpu': 1, 'max_rgb': 200}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))
    
    def git_clone(self):
        super().git_clone()
        self.colmap_download_bag_of_words()

    def is_installed(self): 
        return True

    def info_print(self):
        super().info_print()
        print(f"Default executable: {os.path.join(self.baseline_path, 'colmap_reconstruction.sh')}")

    def colmap_download_bag_of_words(self):
        files = [
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words1M.bin"),
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words32K.bin"),
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words256K.bin")
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not os.path.exists(file):
                print_msg(SCRIPT_LABEL, f"Downloading {file}",'info')
                _ = hf_hub_download(repo_id='vslamlab/colmap_vocabulary', filename=file_name, repo_type='model', local_dir=self.baseline_path)  

        return os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])