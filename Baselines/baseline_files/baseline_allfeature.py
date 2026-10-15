from pathlib import Path
from typing import TYPE_CHECKING

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

if TYPE_CHECKING:
    from vslamlab_utilities import Experiment


class ALLFEATURE_baseline(BaselineVSLAMLAB):
    """ALLFEATURE-VSLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'allfeature', baseline_folder: str = 'AllFeature-VSLAM') -> None:

        # No Bag-of-Words vocabulary parameter: place recognition goes through the `vpr`
        # backend configured in the settings yaml (MegaLoc / none).
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'feature': 'akaze61',
                              'feature_yaml': str(VSLAMLAB_BASELINES / baseline_folder / 'settings' / 'feature_name_to_fill_settings.yaml')}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.0, 0.00, 1.000)
        self.modes = ['mono', 'rgbd']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5']
        self.command_style = 'cpp'

    def resolve_parameters(self, exp: 'Experiment') -> dict:
        # Unless feature_yaml is given explicitly, it follows the selected feature
        parameters = super().resolve_parameters(exp)
        parameters['feature_yaml'] = parameters['feature_yaml'].replace('feature_name_to_fill', parameters['feature'])
        return parameters

    def build_stream_command(self, mode: str, stream_url: str, calibration_yaml: Path, exp_folder: Path,
                             exp_it: int = 0, overrides: dict | None = None) -> str:
        # Same feature -> feature_yaml derivation as resolve_parameters, on the defaults the stream command is built from
        overrides = dict(overrides or {})
        feature = overrides.get('feature', self.default_parameters['feature'])
        overrides.setdefault('feature_yaml', self.default_parameters['feature_yaml'].replace('feature_name_to_fill', feature))
        return super().build_stream_command(mode, stream_url, calibration_yaml, exp_folder, exp_it, overrides)

    def is_installed(self) -> tuple[bool, str]:
        # The conda package provides the executables; "installed" means the baseline folder holds
        # the settings and the model folders (all resolved relative to the cwd by the executables),
        # placed there by the `allfeature` pixi env's `install` task.
        required = [self.settings_yaml,
                    self.baseline_path / 'lightglue_models' / 'aliked-n16.pt',
                    self.baseline_path / 'superpoint_models' / 'superpoint_v1_fixed.onnx',
                    self.baseline_path / 'megaloc_models' / 'megaloc_322x322.onnx',
                    self.baseline_path / 'segmentation_models' / 'efficientvit-seg-l1-ade20k_512x512.onnx']
        if all(f.is_file() for f in required):
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
