import tarfile
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class DPVO_baseline(BaselineVSLAMLAB):
    """DPVO helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'dpvo', baseline_folder: str = 'DPVO') -> None:

        # loop_closure: 0 = DPVO (odometry only), 1 = DPV-SLAM (proximity loop closure),
        #               2 = DPV-SLAM++ (proximity + classic DBoW2 loop closure, uses orb_vocab)
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'network': f"{VSLAMLAB_BASELINES / baseline_folder / 'dpvo.pth'}",
                              'loop_closure': 1,
                              'orb_vocab': f"{VSLAMLAB_BASELINES / baseline_folder / 'ORBvoc.txt'}"}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.862, 0.470, 0.470) # 'red'
        self.modes = ['mono']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5']
        self.command_style = 'python'

    def fetch_source(self) -> None:
        super().fetch_source()
        self.dpvo_download_weights()
    
    def build_execute_command(self, exp_it, exp, dataset, sequence_name) -> str:
        # The ORB vocabulary (145 MB) is only needed by the classic loop closure: fetch it on first use
        if int(self.resolve_parameters(exp)['loop_closure']) == 2:
            self.dpvo_download_vocabulary()
        return super().build_execute_command(exp_it, exp, dataset, sequence_name)

    def dpvo_download_vocabulary(self) -> None: # Download ORBvoc.txt (DBoW2 vocabulary of ORB-SLAM)
        vocab_txt = self.baseline_path / 'ORBvoc.txt'
        if not vocab_txt.is_file():
            print_msg(f"\n{SCRIPT_LABEL}", f"Download ORB vocabulary: {vocab_txt}", 'info')
            file_path = hf_hub_download(repo_id='vslamlab/dpvo_weights', filename='ORBvoc.txt.tar.gz', repo_type='model',
                                        local_dir=self.baseline_path)
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extract('ORBvoc.txt', path=self.baseline_path)

    def dpvo_download_weights(self) -> None: # Download dpvo.pth
        weights_pth = self.baseline_path / 'dpvo.pth'
        if not weights_pth.is_file():
            print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {self.baseline_path}/dpvo.pth",'info')
            file_path = hf_hub_download(repo_id='vslamlab/dpvo_weights', filename='models.zip', repo_type='model',
                                        local_dir=self.baseline_path)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.baseline_path)


class DPVO_baseline_dev(DPVO_baseline):
    """DPVO-DEV helper for VSLAM-LAB Baselines."""   

    def __init__(self):
        super().__init__(baseline_name = 'dpvo-dev', baseline_folder =  'DPVO-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)

    def is_installed(self) -> tuple[bool, str]:
        # `install` (pip -e) compiles the extensions in place and puts the entry point in the clone's own pixi env
        entry_point = self.baseline_path / '.pixi' / 'envs' / 'default' / 'bin' / 'vslamlab_dpvo_mono'
        is_installed = (self.baseline_path / 'cuda_ba.cpython-311-x86_64-linux-gnu.so').is_file() and entry_point.is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
