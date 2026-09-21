"""
Module: VSLAM-LAB - Baselines - baseline_da3.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: GPLv3 License

Depth Anything 3 (Lin et al., 2025) as a 3D reconstruction baseline, in two forms sharing one source
checkout (Baselines/Depth-Anything-3, fork of ByteDance-Seed/Depth-Anything-3):
- `da3`: one feed-forward pass over the experiment's frame window (rgb_max / rgb_step / rgb_placecell),
  like vggt / vggt-omega. The nested model returns metric depth and poses.
- `da3-streaming`: upstream's DA3-Streaming (VGGT-Long style): the whole sequence in overlapping chunks,
  Sim(3) alignment between chunks and SALAD loop closure, poses for every frame.
Both predict their own intrinsics: pinhole only, no calibration consumed. Weights come from Hugging
Face (not gated) at first run.
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class DA3_baseline(BaselineVSLAMLAB):
    """Depth Anything 3, single window (editable install of the checkout in the `da3` env)."""

    def __init__(self, baseline_name: str = 'da3', baseline_folder: str = 'Depth-Anything-3') -> None:
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'rgb_max': 30,                       # frames in the single window (nested giant: ~30 fits 20 GB, to be measured)
                              'model': 'nested-giant-large-1.1',   # see MODELS in vslamlab_da3_common.py
                              'use_ray_pose': 0,                   # 1: poses from the ray head (slower, more accurate per upstream)
                              'ref_view_strategy': 'saddle_balanced',
                              'process_res': 504}

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.150, 0.500, 0.750)
        self.modes = ['mono']
        self.cam_models = ['pinhole']
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        # `pip install -e .` (pixi `install` task) leaves the upstream `da3` console script in the env
        is_installed = (VSLAM_LAB_DIR / '.pixi' / 'envs' / self.baseline_name / 'bin' / 'da3').is_file()
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')


class DA3STREAMING_baseline(DA3_baseline):
    """DA3-Streaming: chunked full-sequence reconstruction with loop closure (own env `da3-streaming`, same checkout)."""

    def __init__(self) -> None:
        super().__init__(baseline_name='da3-streaming', baseline_folder='Depth-Anything-3')
        # Upstream defaults (chunk 120 / overlap 60 need ~28 GB at 504x378; halve both on a 20 GB card)
        self.default_parameters = {'verbose': 1, 'mode': 'mono',
                                   'model': 'nested-giant-large-1.1',
                                   'chunk_size': 120,
                                   'overlap': 60,
                                   'loop_enable': 1,
                                   'process_res': 504}
        self.color = (0.100, 0.350, 0.550)
