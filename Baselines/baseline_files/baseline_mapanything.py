"""
Module: VSLAM-LAB - Baselines - baseline_mapanything.py
- Author: Alejandro Fontan Villacampa
- Version: 2.0
- Created: 2026-09-15
- Updated: 2026-09-18
- License: GPLv3 License

MapAnything (Keetha et al., 3DV 2026) and the feed-forward models its model factory hosts, one pass over a window
of frames (the experiment's rgb csv: rgb_max / rgb_step / rgb_placecell). The `module` parameter selects the model:
- mapanything (default): any mix of images, intrinsics, depth and poses -> metric geometry and cameras.
  mono: images only (`use_calibration: 1` also feeds the calibration.yaml intrinsics); rgbd: images + intrinsics +
  metric depth from the rgb csv (sensor depth or the fastfoundationstereo capability).
- vggt (`precision`), vggt-omega (`weights`, gated HF checkpoints), pi3, pi3x, dust3r, mast3r, must3r, pow3r
  (`scene_graph` for the dust3r family): mono, images only, run through MapAnything's wrappers with each model's
  own image preprocessing. Weights land in Baselines/torch_home/hub/checkpoints (VGGT-1B shared with vggtslam).
Source: https://github.com/VSLAM-LAB/map-anything-VSLAM-LAB (fork of facebookresearch/map-anything).
"""

from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB
from path_constants import VSLAM_LAB_DIR

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class MAPANYTHING_baseline(BaselineVSLAMLAB):
    """MapAnything helper for VSLAM-LAB Baselines (editable install of the checkout in the `mapanything` env)."""

    def __init__(self, baseline_name: str = 'mapanything', baseline_folder: str = 'MapAnything') -> None:
        default_parameters = {'verbose': 1, 'mode': 'mono',
                              'module': 'mapanything',        # mapanything | vggt | vggt-omega | pi3 | pi3x | dust3r | mast3r | must3r | pow3r
                              'rgb_max': 20,                 # frames in the single window (mapanything with memory-efficient inference; vggt ~40 at 20 GB, bf16 ~80)
                              'conf_threshold': 10,           # viewer (verbose 1) only: % of lowest-confidence points hidden at start; a slider changes it live
                              # module mapanything
                              'model': 'default',             # default (facebook/map-anything, CC BY-NC) | apache | v1 | apache-v1 | a Hugging Face id
                              'use_calibration': 0,           # mono only: 1 feeds calibration.yaml intrinsics (pinhole); rgbd always does
                              'memory_efficient': 1,          # run the dense heads in minibatches (upstream default for inference)
                              'minibatch_size': 0,            # 0 = adaptive from free GPU memory; 1 = smallest footprint
                              # module vggt
                              'precision': 'autocast',        # autocast (upstream demo) | bf16 (whole model cast, ~half the memory)
                              # module vggt-omega
                              'weights': '416-reproduce',     # 416-reproduce (authors' reference for benchmarks) | 512 (in-the-wild)
                              # modules dust3r / mast3r / pow3r
                              'scene_graph': 'complete'}      # pair graph of the global alignment: complete | swin-<k> | logwin-<k> | oneref-<i>

        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.950, 0.550, 0.100)
        self.modes = ['mono', 'rgbd']  # rgbd: module mapanything only (the entry point exits otherwise)
        self.cam_models = ['pinhole']  # intrinsics are only consumed undistorted; images-only runs tolerate distortion
        self.command_style = 'python'

    def is_installed(self) -> tuple[bool, str]:
        # `pip install -e .` (pixi `install` task) leaves the package's dist-info next to the editable finder in the env;
        # the external modules' packages are installed by the same task, so one check covers them
        env = VSLAM_LAB_DIR / '.pixi' / 'envs' / self.baseline_name
        is_installed = any(env.glob('lib/python3.*/site-packages/__editable__.mapanything-*.pth'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
