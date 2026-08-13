"""
Module: VSLAM-LAB - Datasets - extra-files - mask2former.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Sonnet 5, Fable 5)
- Version: 1.1
- Created: 2026-08-08
- Updated: 2026-08-12
- License: GPLv3 License

Generates a per-frame binary static/dynamic mask for every rgb frame of one or more dataset
sequences: Mask2Former (HuggingFace transformers, COCO-panoptic checkpoint) segments each frame,
and any pixel belonging to a movable "thing" class (person, vehicles, animals - see
DYNAMIC_CLASSES) is flagged dynamic. Mask convention: 1 = static point, 0 = dynamic point. One
mask folder is written per rgb stream found in rgb.csv (path_rgb_0 -> mask2former_0,
path_rgb_1 -> mask2former_1, ...), each mask keeping the same filename as its source frame; a
stream whose mask2former_<i>/.mask2former_complete marker already exists is skipped unless
--overwrite is given.

rgb.csv is never modified: like run_vpr.py's D.npy, the mask folders are a per-sequence artifact
the run pipeline consumes - when an experiment sets 'segmentation: mask2former',
create_rgb_exp_csv (Run/run_functions.py) appends ts_mask_<i> (ns)/path_mask_<i> columns to the
per-experiment rgb_exp.csv (calling the 'mask-inference' pixi task first if the masks are
missing). If a sample_vpr/synch_gt rgb_raw.csv backup exists, masks are generated from that full
pre-sampling frame list, so a downsampled rgb.csv still gets complete mask coverage.

Target arguments follow CLAUDE.md's sequence-target argument convention (see
utilities.add_sequence_target_args / resolve_sequence_targets): a bare <dataset> [<sequence> ...],
or --datasets/--sequences/--exp/--configs for every other shape.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utilities import (
    add_sequence_target_args, resolve_sequence_targets_or_exit, make_printers,
    sequence_path, sequence_rgb_csv, read_csv_rows, raw_path,
)

DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-coco-panoptic"
MASK_FOLDER_BASE = "mask2former"
COMPLETE_MARKER = ".mask2former_complete"

# COCO-panoptic "thing" classes treated as movable/dynamic in a SLAM scene.
DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)


def rgb_stream_indices(header: list[str]) -> list[int]:
    """Stream indices i for every path_rgb_<i> column in an rgb.csv header (mono: [0],
    stereo: [0, 1], ...)."""
    indices = []
    for col in header:
        suffix = col.removeprefix("path_rgb_")
        if col.startswith("path_rgb_") and suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def load_model(model_id: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
    return processor, model


def mask_frame(processor, model, device: torch.device, image_path: Path) -> np.ndarray:
    """1 = static, 0 = dynamic, uint8, same (H, W) as the source frame."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    segmentation = result["segmentation"].cpu().numpy()
    mask = np.ones(segmentation.shape, dtype=np.uint8)
    for segment in result["segments_info"]:
        label = model.config.id2label[segment["label_id"]]
        if label in DYNAMIC_CLASSES:
            mask[segmentation == segment["id"]] = 0
    return mask


def mask_pair(
    dataset_name: str, sequence_name: str, processor, model, device: torch.device, *,
    mask_folder_base: str = MASK_FOLDER_BASE, overwrite: bool = False,
) -> None:
    rgb_csv = sequence_rgb_csv(dataset_name, sequence_name)
    rgb_raw = raw_path(rgb_csv)
    # Prefer the pre-sampling backup when one exists (sample_vpr/synch_gt convention): it holds
    # the full frame list, so masks cover every frame even after rgb.csv was downsampled.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {dataset_name}:{sequence_name} - missing rgb.csv (run 'pixi run download-sequence' first)")
        return

    header, rows = read_csv_rows(source_csv)
    streams = rgb_stream_indices(header)
    if not streams:
        print_warning(f"Skipping {dataset_name}:{sequence_name} - rgb.csv has no 'path_rgb_<i>' columns")
        return

    seq_path = sequence_path(dataset_name, sequence_name)
    for stream in streams:
        mask_dir = seq_path / f"{mask_folder_base}_{stream}"
        marker = mask_dir / COMPLETE_MARKER
        if marker.exists() and not overwrite:
            print_info(f"Skipping {dataset_name}:{sequence_name} rgb_{stream} - {mask_dir.name} already exists (use --overwrite to recompute)")
            continue

        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True)

        path_idx = header.index(f"path_rgb_{stream}")
        image_paths = [seq_path / row[path_idx] for row in rows]
        for image_path in tqdm(image_paths, desc=f"{dataset_name}:{sequence_name} rgb_{stream}"):
            mask = mask_frame(processor, model, device, image_path)
            Image.fromarray(mask, mode="L").save(mask_dir / image_path.name)

        marker.touch()
        print_info(f"{dataset_name}:{sequence_name} - wrote {len(image_paths)} masks to {mask_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame static(1)/dynamic(0) semantic masks with Mask2Former."
    )
    add_sequence_target_args(parser)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HuggingFace Mask2Former checkpoint id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-folder-base", default=MASK_FOLDER_BASE, dest="mask_folder_base",
                         help="Mask folder prefix; each rgb stream i writes to <base>_<i> (default: mask2former)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute masks even if they already exist for a sequence")
    parser.add_argument("--prefetch", action="store_true",
                         help="Download/cache the Mask2Former checkpoint and exit - no sequence "
                              "targets required (used by the 'install' pixi task)")
    args = parser.parse_args()

    if args.prefetch:
        load_model(args.model_id, torch.device(args.device))
        print_info(f"Prefetched {args.model_id} to the local HuggingFace cache")
        return

    pairs = resolve_sequence_targets_or_exit(args, parser)

    device = torch.device(args.device)
    processor, model = load_model(args.model_id, device)

    for dataset_name, sequence_name in pairs:
        mask_pair(
            dataset_name, sequence_name, processor, model, device,
            mask_folder_base=args.mask_folder_base, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
