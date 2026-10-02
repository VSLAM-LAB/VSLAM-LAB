from pathlib import Path

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class COLMAP_baseline(BaselineVSLAMLAB):
    """colmap helper for VSLAM-LAB Baselines (entry point: Baselines/colmap/vslamlab_colmap.py)."""

    def __init__(self, baseline_name: str = 'colmap', baseline_folder: str = 'colmap') -> None:

        # matcher_type: 'exhaustive' | 'sequential' (sequential with loop detection; the vocabulary
        # tree for the feature type is COLMAP's default, auto-downloaded once into ~/.cache/colmap).
        # matching_type: feature extraction + matching pair, see Baselines/colmap/colmap_matcher.py.
        # use_mask: 1 -> feature extraction honours the rgb csv's path_mask_<i> column when the run
        # pipeline provides one ('segmentation: mask2former', 'refraction: refrax', datasets that
        # ship masks); 0 -> masks ignored (see Baselines/colmap/colmap_matcher.py).
        # optimize_intrinsics: 1 -> bundle adjustment refines the camera intrinsics (focal length and
        # distortion; the principal point stays fixed, as in COLMAP's default); 0 -> the intrinsics
        # from the calibration yaml are kept fixed. Ignored (forced to 1) when the calibration model
        # is 'unknown', since there are no intrinsics to keep (see Baselines/colmap/colmap_mapper.py).
        # dense: 1 -> after the sparse model and trajectory are written, run COLMAP's dense pipeline
        # (image_undistorter -> patch_match_stereo -> stereo_fusion) on the best sub-model, into
        # <exp_folder>/colmap_<id>/dense/ with the fused cloud copied to <exp_folder>/<id>_dense.ply.
        # Needs the CUDA colmap build (linux-64 in pixi.toml) and use_gpu=1; otherwise it is skipped
        # with a warning and the run still succeeds (see Baselines/colmap/colmap_dense.sh).
        # dense_max_image_size: longest image side used for undistortion / patch match / fusion
        # (COLMAP presets: 1000 low, 1600 medium, -1 high = full resolution).
        # mesher: 'none' | 'delaunay' -> mesh the fused cloud, copied to <id>_mesh.ply (COLMAP's
        # poisson_mesher is not offered: its surface trimmer segfaults in the 4.1.1 conda-forge build).
        default_parameters = {'verbose': 1, 'mode': 'mono', 'matcher_type': 'exhaustive',
                             'matching_type': 'sift_bruteforce', 'mapper_type': 'colmap', 'rgb_max': 50000000,
                             'use_mask': 0, 'optimize_intrinsics': 1,
                             'dense': 0, 'dense_max_image_size': 1600, 'mesher': 'none'}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.800, 0.400, 0.750)
        self.modes = ['mono']
        self.cam_models = ['unknown', 'pinhole', 'radtan4', 'radtan5', 'radtan8', 'equid4']
        self.command_style = 'python'
