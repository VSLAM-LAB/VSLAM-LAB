"""
Module: VSLAM-LAB - Capabilities - fastfoundationstereo.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.1
- Created: 2026-08-12
- Updated: 2026-08-25
- License: GPLv3 License

Generates a per-frame metric depth map for every stereo pair (rgb_0/rgb_1) of one or more dataset
sequences using Fast-FoundationStereo (NVlabs, CVPR 2026): each pair is stereo-rectified from the
sequence's calibration.yaml, the model predicts disparity on the rectified pair, disparity is
converted to metric depth (fx * baseline / disparity), and the depth map is warped back into the
original (unrectified) rgb_0 frame so it aligns pixel-wise with path_rgb_0. Depth is stored as
16-bit PNG with depth (m) = pixel_value / DEPTH_FACTOR (default 256, KITTI convention: ~3.9 mm
resolution, 256 m max); 0 means invalid (no disparity, occluded, or outside the rectified FOV).

One depth folder is written per sequence (fastfoundationstereo_0/, one PNG per rgb_0 frame, named
<frame stem>.png). Frames whose depth PNG already exists are skipped, so an interrupted run
resumes where it left off; a sequence whose fastfoundationstereo_0/.fastfoundationstereo_complete
marker exists is skipped entirely unless --overwrite is given (which recomputes from scratch).

Neither rgb.csv nor calibration.yaml of the sequence is ever modified: like mask2former.py's mask
folders, the depth folder is a per-sequence artifact the run pipeline consumes - when an experiment
sets 'depth: fastfoundationstereo', create_rgb_exp_csv (Run/run_functions.py) appends
ts_depth_0 (ns)/path_depth_0 columns to the per-experiment rgb_exp.csv (calling the
'stereo-inference' pixi task first if depth is missing) and registers the depth stream
(depth_name/depth_factor and a '+depth' cam_type on the rgb_0 entry, following the rgbd convention
of DatasetVSLAMLAB_calibration._get_rgbd_yaml_section) in the per-experiment calibration_exp.yaml,
so rgbd baselines can consume the generated depth directly. The depth_factor used is recorded in
the completion marker so the run pipeline registers the same value. The dataset's
Datasets/dataset_files/dataset_<name>.yaml modes list does gain 'rgbd' (and 'rgbd-vi' when
'mono-vi' is present) so rgbd experiments validate.
If a sample_vpr/synch_gt rgb_raw.csv backup exists, depth is generated from that full pre-sampling
frame list, so a downsampled rgb.csv still gets complete depth coverage.

The Fast-FoundationStereo repo is cloned to Baselines/Fast-FoundationStereo by the feature's
'git-clone' pixi task and the default checkpoint is fetched from NVIDIA's HuggingFace repo by
--prefetch (the 'install' pixi task); the paper's alternative checkpoint sizes live on the
authors' Google Drive and can be selected with --checkpoint. Requires a stereo calibration.yaml with two pinhole cameras (rgb_0,
rgb_1) and radtan distortion (or none); fisheye/equidistant models are not supported.

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, read_csv_rows, raw_path,
)

FFS_DIR = REPO_ROOT / "Baselines" / "Fast-FoundationStereo"
# Official NVIDIA checkpoint on HuggingFace (NVIDIA Open Model Agreement) - the default, since the
# authors' Google Drive folder below (which holds the alternative checkpoint sizes from the paper,
# e.g. 23-36-37, 20-26-39) is frequently rate-limited by Drive.
WEIGHTS_HF_REPO = "nvidia/c-fast-foundationstereo"
WEIGHTS_GDRIVE_URL = "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"
DEFAULT_CHECKPOINT = "c-fast-foundationstereo/model_best_bp2_serialize.pth"
DEPTH_FOLDER_BASE = "fastfoundationstereo"
COMPLETE_MARKER = ".fastfoundationstereo_complete"
DEPTH_FACTOR = 256.0
SUPPORTED_DISTORTION = {"radtan4", "radtan", "none"}

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def ensure_weights(checkpoint: Path) -> None:
    """Download the requested checkpoint into Baselines/Fast-FoundationStereo/weights/ if missing:
    the default c-fast-foundationstereo checkpoint from NVIDIA's HuggingFace repo, any other
    (paper-family) checkpoint from the authors' Google Drive folder."""
    if checkpoint.exists():
        return
    if checkpoint.parent.name == "c-fast-foundationstereo":
        from huggingface_hub import snapshot_download
        print_info(f"Checkpoint {checkpoint} missing - downloading {WEIGHTS_HF_REPO} from HuggingFace ...")
        snapshot_download(WEIGHTS_HF_REPO, local_dir=checkpoint.parent)
    else:
        import gdown
        print_info(f"Checkpoint {checkpoint} missing - downloading weights from Google Drive ...")
        gdown.download_folder(WEIGHTS_GDRIVE_URL, output=str(FFS_DIR / "weights"), quiet=False)
    if not checkpoint.exists():
        print_warning(f"Download finished but {checkpoint} still missing - download it manually "
                      f"({WEIGHTS_HF_REPO} on HuggingFace or {WEIGHTS_GDRIVE_URL}) into {FFS_DIR / 'weights'}")
        sys.exit(1)


def load_model(checkpoint: Path, device: torch.device, valid_iters: int, max_disp: int):
    """The checkpoint is a fully serialized model object, so the Fast-FoundationStereo repo must
    be importable (its 'core' package) when unpickling."""
    if not FFS_DIR.exists():
        print_warning(f"{FFS_DIR} not found - run 'pixi run -e fastfoundationstereo install' first")
        sys.exit(1)
    sys.path.insert(0, str(FFS_DIR))
    ensure_weights(checkpoint)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Some checkpoints' serialized args miss keys the forward pass reads (the HF
    # c-fast-foundationstereo one lacks 'normalize'): backfill from the cfg.yaml shipped next to
    # the checkpoint, with normalize defaulting to True as in the repo's scripts/make_plugin_onnx.py.
    cfg_yaml = checkpoint.parent / "cfg.yaml"
    defaults = (yaml.safe_load(cfg_yaml.read_text()) or {}) if cfg_yaml.exists() else {}
    defaults.setdefault("normalize", True)
    missing = {key: value for key, value in defaults.items() if not hasattr(model.args, key)}
    if missing:
        try:
            from omegaconf import open_dict
            with open_dict(model.args):
                for key, value in missing.items():
                    setattr(model.args, key, value)
        except (ImportError, ValueError):  # model.args is a plain namespace, not a DictConfig
            for key, value in missing.items():
                setattr(model.args, key, value)

    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    return model.to(device).eval()


def load_stereo_calibration(dataset_name: str, sequence_name: str) -> dict | None:
    """Parse the sequence's calibration.yaml into stereo-rectification inputs, or None (with a
    warning) if the sequence has no usable rgb_0/rgb_1 pinhole pair."""
    seq_path = sequence_path(dataset_name, sequence_name)
    calib_yaml = seq_path / "calibration.yaml"
    if not calib_yaml.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing {calib_yaml}")
        return None

    cameras = {cam.get("cam_name"): cam for cam in yaml.safe_load(calib_yaml.read_text()).get("cameras", [])}
    if "rgb_0" not in cameras or "rgb_1" not in cameras:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - calibration.yaml has no rgb_0/rgb_1 camera pair")
        return None

    def parse_camera(cam: dict):
        fx, fy = cam["focal_length"]
        cx, cy = cam["principal_point"]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        D = np.array(cam.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        T_BS = np.array(cam["T_BS"], dtype=float).reshape(4, 4)
        return K, D, cam.get("distortion_type", "none"), T_BS, tuple(cam["image_dimension"])

    K0, D0, dist0, T0, size0 = parse_camera(cameras["rgb_0"])
    K1, D1, dist1, T1, size1 = parse_camera(cameras["rgb_1"])
    for dist_type in (dist0, dist1):
        if dist_type not in SUPPORTED_DISTORTION:
            print_warning(f"Skipping {dataset_name}:{sequence_name} - distortion_type '{dist_type}' not supported (only radtan/none)")
            return None
    if size0 != size1:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb_0/rgb_1 image dimensions differ ({size0} vs {size1})")
        return None

    # Relative pose taking points from the rgb_0 frame into the rgb_1 frame, as cv2.stereoRectify expects.
    T_c1c0 = np.linalg.inv(T1) @ T0
    return {"K0": K0, "D0": D0, "K1": K1, "D1": D1, "size": size0,
            "R": T_c1c0[:3, :3], "t": T_c1c0[:3, 3].reshape(3, 1)}


def update_dataset_modes(dataset_name: str) -> None:
    """Once generated depth exists for a sequence, the dataset can run rgbd experiments: add
    'rgbd' to the modes list in Datasets/dataset_files/dataset_<name>.yaml, plus 'rgbd-vi' when
    the dataset already supports 'mono-vi' (imu present). Follows the existing ordering
    convention ('rgbd' after 'mono', 'rgbd-vi' after 'mono-vi'); line-based and idempotent."""
    dataset_yaml = REPO_ROOT / "Datasets" / "dataset_files" / f"dataset_{dataset_name}.yaml"
    if not dataset_yaml.exists():
        print_warning(f"{dataset_yaml} not found; cannot add 'rgbd' to the dataset modes")
        return

    lines = dataset_yaml.read_text().splitlines()
    idx = next((i for i, line in enumerate(lines) if line.startswith("modes:")), None)
    modes = yaml.safe_load(lines[idx].split("modes:", 1)[1]) if idx is not None else None
    if not isinstance(modes, list):
        print_warning(f"no parseable 'modes:' list in {dataset_yaml.name}; cannot add 'rgbd'")
        return

    def insert_after(mode: str, anchor: str) -> None:
        if mode not in new_modes:
            pos = new_modes.index(anchor) + 1 if anchor in new_modes else len(new_modes)
            new_modes.insert(pos, mode)

    new_modes = list(modes)
    insert_after("rgbd", "mono")
    if "mono-vi" in new_modes:
        insert_after("rgbd-vi", "mono-vi")
    if new_modes == modes:
        return  # already up to date

    lines[idx] = "modes: [" + ", ".join(f"'{mode}'" for mode in new_modes) + "]"
    dataset_yaml.write_text("\n".join(lines) + "\n")
    print_info(f"{dataset_name} - modes updated to {new_modes} in {dataset_yaml.name}")


class StereoRectifier:
    """Precomputed per-sequence maps: original pair -> rectified pair for inference, and rectified
    depth -> depth aligned to the original (unrectified) rgb_0 frame."""

    def __init__(self, calib: dict) -> None:
        W, H = calib["size"]
        self.size = (W, H)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            calib["K0"], calib["D0"], calib["K1"], calib["D1"], (W, H), calib["R"], calib["t"],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.fx = P1[0, 0]
        self.baseline = abs(P2[0, 3] / P2[0, 0])
        self.maps0 = cv2.initUndistortRectifyMap(calib["K0"], calib["D0"], R1, P1, (W, H), cv2.CV_32FC1)
        self.maps1 = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], R2, P2, (W, H), cv2.CV_32FC1)

        # Where each original rgb_0 pixel lands in the rectified image, for warping depth back.
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pts = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        rect = cv2.undistortPoints(pts, calib["K0"], calib["D0"], R=R1, P=P1).reshape(H, W, 2)
        self.back_map_x = np.ascontiguousarray(rect[..., 0], dtype=np.float32)
        self.back_map_y = np.ascontiguousarray(rect[..., 1], dtype=np.float32)

        # Depth is z in the rectified camera frame, which is rotated by R1 wrt the original rgb_0
        # frame: z_original = z_rectified * corr(u', v'), with corr the third row of R1^T applied
        # to the rectified viewing ray.
        x_n = (uu - P1[0, 2]) / P1[0, 0]
        y_n = (vv - P1[1, 2]) / P1[1, 1]
        self.z_corr = (R1[0, 2] * x_n + R1[1, 2] * y_n + R1[2, 2]).astype(np.float32)

    def rectify(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (cv2.remap(img0, *self.maps0, cv2.INTER_LINEAR),
                cv2.remap(img1, *self.maps1, cv2.INTER_LINEAR))

    def disparity_to_original_depth(self, disp: np.ndarray, zfar: float) -> np.ndarray:
        """Rectified disparity -> metric depth aligned to the original rgb_0 frame (0 = invalid)."""
        xx = np.arange(disp.shape[1])[None, :]
        valid = (disp > 0) & (xx - disp >= 0)  # xx - disp < 0 falls outside the right image's FOV
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = self.fx * self.baseline / disp[valid]
        depth *= self.z_corr
        depth[depth > zfar] = 0  # near-zero disparity explodes to nonsense range; invalidate it
        return cv2.remap(depth, self.back_map_x, self.back_map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def infer_disparity(model, device: torch.device, left: np.ndarray, right: np.ndarray, valid_iters: int) -> np.ndarray:
    from core.utils.utils import InputPadder  # importable once load_model put FFS_DIR on sys.path

    H, W = left.shape[:2]
    img0 = torch.as_tensor(left).to(device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right).to(device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.inference_mode():
        with torch.amp.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            disp = model.forward(img0, img1, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1")
    return padder.unpad(disp.float()).cpu().numpy().reshape(H, W).clip(0, None)


def read_rgb(image_path: Path) -> np.ndarray:
    """3-channel RGB uint8 (grayscale sources are replicated across channels, as the model expects)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def depth_pair(
    dataset_name: str, sequence_name: str, model_loader: Callable, device: torch.device, *,
    valid_iters: int, depth_factor: float = DEPTH_FACTOR, zfar: float = 100.0,
    depth_folder_base: str = DEPTH_FOLDER_BASE, overwrite: bool = False,
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    # Prefer the pre-sampling backup when one exists (sample_vpr/synch_gt convention): it holds
    # the full frame list, so depth covers every frame even after rgb.csv was downsampled.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    header, rows = read_csv_rows(source_csv)
    if "path_rgb_0" not in header or "path_rgb_1" not in header:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb.csv has no stereo pair (path_rgb_0 and path_rgb_1 required)")
        return

    seq_path = sequence_path(dataset_name, sequence_name)
    depth_dir = seq_path / f"{depth_folder_base}_0"
    marker = depth_dir / COMPLETE_MARKER
    if marker.exists() and not overwrite:
        print_info(f"Skipping {dataset_name}:{sequence_name} - {depth_dir.name} already complete (use --overwrite to recompute)")
        update_dataset_modes(dataset_name)
        return
    if overwrite and depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    calib = load_stereo_calibration(dataset_name, sequence_name)
    if calib is None:
        return
    rectifier = StereoRectifier(calib)
    print_info(f"{dataset_name}:{sequence_name} - rectified fx={rectifier.fx:.1f}, baseline={rectifier.baseline:.4f} m, "
               f"depth = png/{depth_factor:g} m")

    left_idx, right_idx = header.index("path_rgb_0"), header.index("path_rgb_1")
    written = skipped = 0
    for row in tqdm(rows, desc=f"{dataset_name}:{sequence_name} depth"):
        left_path = seq_path / row[left_idx]
        out_png = depth_dir / f"{left_path.stem}.png"
        if out_png.exists():  # resume support: never recompute an existing depth frame
            skipped += 1
            continue
        left, right = rectifier.rectify(read_rgb(left_path), read_rgb(seq_path / row[right_idx]))
        disp = infer_disparity(model_loader(), device, left, right, valid_iters)
        depth = rectifier.disparity_to_original_depth(disp, zfar)
        depth_png = np.clip(np.round(depth * depth_factor), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_png), depth_png)
        written += 1

    # The marker doubles as metadata: Run/run_functions.py reads depth_factor from it when
    # registering the stream in the per-experiment calibration_exp.yaml.
    marker.write_text(f"depth_factor: {float(depth_factor)}\n")
    update_dataset_modes(dataset_name)
    print_info(f"{dataset_name}:{sequence_name} - wrote {written} depth maps to {depth_dir}"
               + (f" ({skipped} already existed)" if skipped else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame metric depth (16-bit PNG) from rgb_0/rgb_1 stereo pairs with Fast-FoundationStereo."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint path relative to Baselines/Fast-FoundationStereo/weights "
                             "(default: the HuggingFace c-fast-foundationstereo checkpoint; paper-family "
                             "checkpoints like 23-36-37/model_best_bp2_serialize.pth come from Google Drive)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid_iters", type=int, default=8, help="Refinement iterations (lower = faster, slightly less accurate)")
    parser.add_argument("--max_disp", type=int, default=192, help="Maximum disparity for volume encoding")
    parser.add_argument("--depth-factor", type=float, default=DEPTH_FACTOR, dest="depth_factor",
                        help="Depth (m) = png_value / depth_factor (default: 256, KITTI convention)")
    parser.add_argument("--zfar", type=float, default=100.0,
                        help="Depth beyond this range (m) is stored as invalid (0) - near-zero disparity is noise")
    parser.add_argument("--depth-folder-base", default=DEPTH_FOLDER_BASE, dest="depth_folder_base",
                        help="Depth folder prefix; depth is written to <base>_0 (default: fastfoundationstereo)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute depth even if it already exists for a sequence")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download/cache the Fast-FoundationStereo checkpoints and exit - no "
                             "sequence targets required (used by the 'install' pixi task)")
    args = parser.parse_args()

    checkpoint = FFS_DIR / "weights" / args.checkpoint
    if args.prefetch:
        if checkpoint.exists():
            print_info(f"Checkpoint {checkpoint} already present")
        else:
            ensure_weights(checkpoint)
            print_info(f"Prefetched Fast-FoundationStereo weights to {FFS_DIR / 'weights'}")
        return

    pairs = resolve_sequence_targets_or_exit(args, parser)

    device = torch.device(args.device)
    # Lazy: sequences that are already complete should not pay the model load.
    model_loader = functools.cache(lambda: load_model(checkpoint, device, args.valid_iters, args.max_disp))

    for dataset_name, sequence_name in pairs:
        depth_pair(
            dataset_name, sequence_name, model_loader, device,
            valid_iters=args.valid_iters, depth_factor=args.depth_factor, zfar=args.zfar,
            depth_folder_base=args.depth_folder_base, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
