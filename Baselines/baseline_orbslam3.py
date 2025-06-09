import os.path
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ORBSLAM3_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='orbslam3', baseline_folder='ORB_SLAM3'):

        default_parameters = {'verbose': 1, 'mode': 'mono-vi', 
                              'vocabulary': os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'Vocabulary', 'ORBvoc.txt')}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'blue'
        self.modes = ['mono-vi']

    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')
    
class ORBSLAM3_baseline_dev(ORBSLAM3_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'orbslam3-dev', baseline_folder = 'ORB_SLAM3-DEV')

    def is_installed(self):
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_orbslam3_mono_vi'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)') 
    
