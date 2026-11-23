from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class ALLFEATURE_baseline(BaselineVSLAMLAB):
    """ALLFEATURE-VSLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'allfeature', baseline_folder: str = 'AllFeature-VSLAM') -> None:

        # The features, place recognition and everything else are chosen in the settings yaml
        default_parameters = {'verbose': 1, 'mode': 'mono'}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.0, 0.00, 1.000)
        self.modes = ['mono', 'rgbd']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5']
        self.command_style = 'cpp'

    def is_installed(self) -> tuple[bool, str]:
        # The conda package provides the executables; "installed" means the baseline folder holds the settings and
        # the models (resolved relative to the cwd by the executables), placed there by the `install` task. A model
        # still an LFS pointer (the clone's ~130-byte placeholder) was never downloaded.
        required = [self.settings_yaml,
                    self.baseline_path / 'feature_models' / 'lightglue_aliked_fp32.onnx',
                    self.baseline_path / 'megaloc_models' / 'megaloc_322x322.onnx',
                    self.baseline_path / 'segmentation_models' / 'efficientvit-seg-l1-ade20k_512x512.onnx']
        if all(f.is_file() and f.stat().st_size > 1024 for f in required):
            return True, 'is installed'
        return False, 'not installed (conda package; auto install downloads settings and models)'


class ALLFEATURE_baseline_dev(ALLFEATURE_baseline):
    """AllFeature-VSLAM-DEV helper for VSLAM-LAB Baselines."""

    def __init__(self):
        super().__init__(baseline_name = 'allfeature-dev', baseline_folder = 'AllFeature-VSLAM-DEV')
        self.color = tuple(max(c / 1.0, 0.0) for c in self.color)

    def is_installed(self) -> tuple[bool, str]:
        is_installed = (self.baseline_path / 'bin' / 'vslamlab_allfeature_mono').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
