import os.path
from pathlib import Path
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class MAST3RSLAM_baseline(BaselineVSLAMLab):
    """MASt3R-SLAM helper for VSLAM-LAB Baselines."""    
    def __init__(self, baseline_name: str = 'mast3rslam', baseline_folder: str = 'MASt3R-SLAM') -> None:
        
        default_parameters = {'verbose': 1, 'mode': 'mono', 
                              'checkpoints_dir': str(VSLAMLAB_BASELINES / baseline_folder / 'checkpoints'), 'use_calib': 1}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.470, 0.862, 0.628)
        self.modes = ['mono']
        self.camera_models = ['pinhole', 'radtan4', 'radtan5']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)        

    def git_clone(self) -> None:
        super().git_clone()
        self.mast3rslam_download_weights()

    def is_installed(self) -> tuple[bool, str]:  
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
   
    def mast3rslam_download_weights(self) -> None: # Download checkpoints
        checkpoints_dir = self.baseline_path / 'checkpoints'
        os.makedirs(checkpoints_dir, exist_ok=True)

        files = [
            checkpoints_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            checkpoints_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            checkpoints_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not file.is_file():
                print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {file}",'info')
                _ = hf_hub_download(repo_id='vslamlab/mast3rslam_weights', filename=file_name, repo_type='model', local_dir=checkpoints_dir) 


class MAST3RSLAM_baseline_dev(MAST3RSLAM_baseline):
    """MASt3R-SLAM-DEV helper for VSLAM-LAB Baselines."""    

    def __init__(self):
        super().__init__(baseline_name = 'mast3rslam-dev', baseline_folder =  'MASt3R-SLAM-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)
        
    def is_installed(self) -> tuple[bool, str]:
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'mast3r_slam_backends.so'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')