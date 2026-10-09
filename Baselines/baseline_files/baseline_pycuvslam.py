from pathlib import Path
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class PYCUVSLAM_baseline(BaselineVSLAMLAB):
    """PyCuVSLAM helper for VSLAM-LAB Baselines. cuVSLAM comes from NVIDIA's wheel (pycuvslam pixi environment);
    the cloned source only holds the VSLAM-LAB entry scripts and settings."""

    def __init__(self, baseline_name: str = 'pycuvslam', baseline_folder: str = 'PyCuVSLAM') -> None:

        default_parameters = {'verbose': 1, 'mode': 'stereo'}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.700, 0.300)
        # no mono: cuVSLAM v17 monocular odometry is unreliable (eth table_3: 54-116 cm ATE, ORB-SLAM2/3 ~0.6 cm)
        self.modes = ['rgbd', 'stereo', 'stereo-vi']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5', 'equid4']
        self.command_style = 'python'
