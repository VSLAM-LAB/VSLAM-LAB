# Run methods

import sys
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path

from Baselines.BaselineVSLAMLAB_utilities import log_run_sequence_time
from path_constants import RGB_BASE_FOLDER, CALIBRATION_EXP_YAML, VSLAM_LAB_DIR, VSLAMLAB_EVALUATION
from Run import ablations
from utilities import print_msg, write_csv_rows

# Datasets/extra-files isn't an importable package (hyphen in the dir name), so it's added to
# sys.path directly - the same trick sample_vpr.py itself uses to import run_vpr.py.
sys.path.insert(0, str(VSLAM_LAB_DIR / "Datasets" / "extra-files"))
from sample_vpr import sweep_thresholds, select_for_target, selected_rows

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

# Mask folders written by Datasets/extra-files/mask2former.py ('pixi run mask-inference'). Kept as
# local constants rather than imported: importing mask2former.py pulls in torch/transformers,
# which the vslamlab environment doesn't ship.
MASK_FOLDER_BASE = "mask2former"
MASK_COMPLETE_MARKER = ".mask2former_complete"

# Depth folder written by Datasets/extra-files/fastfoundationstereo.py ('pixi run stereo-inference').
# Same reasoning as the mask constants above: not imported to avoid torch deps.
DEPTH_FOLDER_BASE = "fastfoundationstereo"
DEPTH_COMPLETE_MARKER = ".fastfoundationstereo_complete"
DEPTH_FACTOR_DEFAULT = 256.0  # the script's default; only used for markers that don't record their depth_factor

def get_rows(rows_idx, rgb_csv):
    df = pd.read_csv(Path(rgb_csv))

    idx = [int(i) for i in rows_idx if isinstance(i, (int,)) or str(i).lstrip("-").isdigit()]
    idx = [i for i in idx if 0 <= i < len(df)]

    return df.iloc[idx].to_dict(orient="records")

#@ray.remote(num_gpus=1)
def run_sequence(exp_it, exp, baseline, dataset, sequence_name, ablation=False):
    # Check baseline is installed
    baseline.check_installation()

    run_time_start = time.time()

    # Create experiment folder
    exp_folder = exp.folder / dataset.dataset_folder / sequence_name
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Per-experiment calibration copy (before create_rgb_exp_csv, which may patch it - e.g. registering generated depth)
    create_calibration_exp_yaml(exp, dataset, sequence_name)

    # Select images
    create_rgb_exp_csv(exp, dataset, sequence_name, baseline.default_parameters)

    # Sava data for evaluation
    get_sequence_data_for_evaluation(exp, dataset, sequence_name)

    # Build execution command
    exec_command = baseline.build_execute_command(exp_it, exp, dataset, sequence_name)

    # Prepare Ablation
    if ablation:
        exec_command = ablations.prepare_ablation(exp_it, exp, baseline, dataset, sequence_name, exec_command)

    # Execute experiment
    print(f"\n{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")
    results = baseline.execute(exec_command, exp_it, exp_folder)

    # Finish Ablation
    if ablation:
        ablations.finish_ablation(exp_it, baseline, dataset, sequence_name)

    # Log iteration duration
    duration_time = time.time() - run_time_start
    log_run_sequence_time(exp_folder, exp_it, duration_time)

    results['duration_time'] = duration_time
    return results

def create_rgb_exp_csv(exp: Any, dataset: Any, sequence_name: str, default_parameters: dict | None = None) -> None:
    """Build the experiment's rgb csv, applying rgb_idx/rgb_step/rgb_max/rgb_vpr filtering and
    appending segmentation mask / generated depth columns if requested. Only the experiment's own
    files (rgb_exp.csv, and calibration_exp.yaml when depth is registered) are ever written - the
    sequence's rgb.csv and calibration.yaml are never modified."""
    sequence_path = dataset.sequence_path(sequence_name)
    exp_folder = exp.folder / dataset.dataset_folder / sequence_name

    # Seed rgb_exp_csv with a full copy of the source csv; if no downsampling/slicing is requested below, this copy is the final result
    if 'rgb_csv' in exp.parameters:
        rgb_csv = sequence_path / exp.parameters['rgb_csv']
    else:
        rgb_csv = dataset.rgb_csv_path(sequence_name)

    rgb_exp_csv = exp_folder / f"{RGB_BASE_FOLDER}_exp.csv"

    if rgb_exp_csv.exists():
        rgb_exp_csv.unlink()
    shutil.copy(rgb_csv, rgb_exp_csv)

    # rgb_idx, rgb_step, rgb_max are independent and stackable: idx slices the frame range first, step then keeps 1 in every n of what's left, and max truncates the remainder to a frame count
    has_default = isinstance(default_parameters, dict)
    has_rgb_idx = 'rgb_idx' in exp.parameters or (has_default and 'rgb_idx' in default_parameters)
    has_rgb_step = 'rgb_step' in exp.parameters or (has_default and 'rgb_step' in default_parameters)
    has_rgb_max = 'rgb_max' in exp.parameters or (has_default and 'rgb_max' in default_parameters)
    has_rgb_vpr = 'rgb_vpr' in exp.parameters or (has_default and 'rgb_vpr' in default_parameters)

    if has_rgb_idx or has_rgb_step or has_rgb_max or has_rgb_vpr:
        filter_info = []

        full_df = pd.read_csv(rgb_csv)
        total_frames = len(full_df)
        full_ts = full_df['ts_rgb_0 (ns)'].to_list()

        # orig_idx tracks each surviving row's position in the source rgb_csv alongside rows, so
        # rgb_vpr (below) can index into D.npy - computed for the full sequence - correctly even
        # after rgb_idx/rgb_step/rgb_max have already trimmed rows down.
        if has_rgb_idx:
            rgb_idx = exp.parameters['rgb_idx'] if 'rgb_idx' in exp.parameters else default_parameters['rgb_idx']
            start, end = rgb_idx[0], rgb_idx[1]
            if 0 <= start <= end < total_frames:
                orig_idx = list(range(start, end + 1))
            else:
                print_msg(SCRIPT_LABEL, f"rgb_idx {rgb_idx} invalid for {sequence_name} (valid range 0-{total_frames - 1}); using the whole sequence instead", flag="error", verb='NONE')
                orig_idx = list(range(total_frames))
                has_rgb_idx = False
        else:
            orig_idx = list(range(total_frames))
        rows = get_rows(orig_idx, rgb_csv)
        filter_info.append(f"rgb_idx={[start, end] if has_rgb_idx else 'all'} -> {len(rows)} frames")

        if has_rgb_step:
            rgb_step = exp.parameters['rgb_step'] if 'rgb_step' in exp.parameters else default_parameters['rgb_step']
            rows = rows[::rgb_step]
            orig_idx = orig_idx[::rgb_step]
            filter_info.append(f"rgb_step={rgb_step} -> {len(rows)} frames")

        if has_rgb_max:
            max_rgb_num = exp.parameters['rgb_max'] if 'rgb_max' in exp.parameters else default_parameters['rgb_max']
            rows = rows[:max_rgb_num]
            orig_idx = orig_idx[:max_rgb_num]
            filter_info.append(f"rgb_max={max_rgb_num} -> {len(rows)} frames")

        if has_rgb_vpr:
            rgb_vpr_num = exp.parameters['rgb_vpr'] if 'rgb_vpr' in exp.parameters else default_parameters['rgb_vpr']
            d_matrix_path = sequence_path / "vpr-lab" / "D.npy"
            if not d_matrix_path.exists():
                print_msg(SCRIPT_LABEL, f"rgb_vpr: {d_matrix_path} not found, running 'pixi run vpr {dataset.dataset_name} {sequence_name}' ...", verb='LOW')
                subprocess.run(["pixi", "run", "-e", "vpr-lab", "vpr", dataset.dataset_name, sequence_name], cwd=VSLAM_LAB_DIR, check=True)

            if rgb_vpr_num >= len(rows):
                filter_info.append(f"rgb_vpr={rgb_vpr_num} -> {len(rows)} frames (already <= target)")
            else:
                D = np.load(d_matrix_path)[np.ix_(orig_idx, orig_idx)]
                sweep = sweep_thresholds(D, max_threshold=0.6, n_thresholds=200, verbose=False)
                chosen_th, chosen_indexes = select_for_target(sweep, rgb_vpr_num)
                rows = selected_rows(rows, chosen_indexes)
                filter_info.append(f"rgb_vpr={rgb_vpr_num} -> {len(rows)} frames (threshold={chosen_th:.4f})")

        header = list(rows[0].keys())
        write_csv_rows(rgb_exp_csv, header, [list(row.values()) for row in rows])

        min_frame, max_frame = min(orig_idx), max(orig_idx)
        num_frames = len(rows)
        total_duration = full_ts[-1] - full_ts[0]
        covered_duration = full_ts[max_frame] - full_ts[min_frame]
        time_pct = (covered_duration / total_duration * 100) if total_duration > 0 else 0.0
        print_msg(SCRIPT_LABEL, f"RGB frame filtering for {sequence_name}: {' | '.join(filter_info)}", verb='LOW')
        print_msg(
            SCRIPT_LABEL,
            f"RGB frame stats for {sequence_name}: min_frame={min_frame}, max_frame={max_frame}, "
            f"num_frames={num_frames}/{total_frames} ({num_frames / total_frames * 100:.1f}% of frames, "
            f"{time_pct:.1f}% of time covered)",
            verb='LOW'
        )

    has_segmentation = 'segmentation' in exp.parameters or (has_default and 'segmentation' in default_parameters)
    if has_segmentation:
        segmentation = exp.parameters['segmentation'] if 'segmentation' in exp.parameters else default_parameters['segmentation']
        if segmentation == 'mask2former':
            append_mask2former_columns(dataset, sequence_name, sequence_path, rgb_exp_csv)
        else:
            print_msg(SCRIPT_LABEL, f"segmentation='{segmentation}' not recognized (only 'mask2former' is supported); ignoring", flag="error", verb='NONE')

    has_depth = 'depth' in exp.parameters or (has_default and 'depth' in default_parameters)
    if has_depth:
        depth = exp.parameters['depth'] if 'depth' in exp.parameters else default_parameters['depth']
        if depth == 'fastfoundationstereo':
            append_stereo_depth_columns(dataset, sequence_name, sequence_path, rgb_exp_csv, exp_folder / CALIBRATION_EXP_YAML)
        else:
            print_msg(SCRIPT_LABEL, f"depth='{depth}' not recognized (only 'fastfoundationstereo' is supported); ignoring", flag="error", verb='NONE')

def create_calibration_exp_yaml(exp: Any, dataset: Any, sequence_name: str) -> Path:
    """Seed the experiment's calibration yaml (<exp_folder>/calibration_exp.yaml) with a fresh copy
    of the sequence's calibration.yaml - the file every baseline is handed as calibration_yaml
    (BaselineVSLAMLAB.build_execute_command_cpp/python). Like rgb_exp.csv, it is rewritten on
    every run so per-experiment edits never leak back into the sequence: later stages patch this
    copy (create_rgb_exp_csv registers generated depth via register_depth_stream; replacing
    intrinsics works the same way) without touching the benchmark data. Must therefore run
    before create_rgb_exp_csv."""
    calibration_yaml = dataset.calibration_yaml_path(sequence_name)
    calibration_exp_yaml = exp.folder / dataset.dataset_folder / sequence_name / CALIBRATION_EXP_YAML

    if not calibration_yaml.exists():
        print_msg(SCRIPT_LABEL, f"{calibration_yaml} not found for {sequence_name} (run 'pixi run download-sequence {dataset.dataset_name} {sequence_name}' first)", flag="error", verb='NONE')
        sys.exit(1)

    if calibration_exp_yaml.exists():
        calibration_exp_yaml.unlink()
    shutil.copy(calibration_yaml, calibration_exp_yaml)
    return calibration_exp_yaml

def append_mask2former_columns(dataset: Any, sequence_name: str, sequence_path: Path, rgb_exp_csv: Path) -> None:
    """Append ts_mask_<i> (ns)/path_mask_<i> columns to the experiment's rgb_exp csv, one pair per
    path_rgb_<i> stream, pointing at the sequence's mask2former_<i> masks. Streams whose
    .mask2former_complete marker is missing trigger 'pixi run mask-inference' first to generate
    them. Only rgb_exp_csv is rewritten - the sequence's rgb.csv is left untouched."""
    df = pd.read_csv(rgb_exp_csv)
    streams = sorted(
        int(col.removeprefix("path_rgb_")) for col in df.columns
        if col.startswith("path_rgb_") and col.removeprefix("path_rgb_").isdigit()
    )
    if not streams:
        print_msg(SCRIPT_LABEL, f"segmentation: no path_rgb_<i> columns in {rgb_exp_csv}; skipping mask columns", flag="error", verb='NONE')
        return

    missing = [i for i in streams if not (sequence_path / f"{MASK_FOLDER_BASE}_{i}" / MASK_COMPLETE_MARKER).exists()]
    if missing:
        print_msg(SCRIPT_LABEL, f"segmentation: masks missing for {sequence_name} (streams {missing}), running 'pixi run mask-inference {dataset.dataset_name} {sequence_name}' ...", verb='LOW')
        subprocess.run(["pixi", "run", "-e", "mask2former", "mask-inference", dataset.dataset_name, sequence_name], cwd=VSLAM_LAB_DIR, check=True)

    for i in streams:
        df[f"ts_mask_{i} (ns)"] = df[f"ts_rgb_{i} (ns)"]
        df[f"path_mask_{i}"] = [f"{MASK_FOLDER_BASE}_{i}/{Path(p).name}" for p in df[f"path_rgb_{i}"]]
    df.to_csv(rgb_exp_csv, index=False)
    print_msg(SCRIPT_LABEL, f"segmentation: appended mask2former columns for streams {streams} to {rgb_exp_csv.name}", verb='LOW')

def append_stereo_depth_columns(dataset: Any, sequence_name: str, sequence_path: Path, rgb_exp_csv: Path, calibration_exp_yaml: Path) -> None:
    """Append ts_depth_0 (ns)/path_depth_0 columns to the experiment's rgb_exp csv, pointing at the
    sequence's fastfoundationstereo_0 depth maps computed from the rgb_0/rgb_1 stereo pair, and
    register that depth stream in the experiment's calibration_exp.yaml (register_depth_stream) so
    rgbd baselines can consume it. If the .fastfoundationstereo_complete marker is missing,
    'pixi run stereo-inference' is triggered first to generate the depth (it resumes per frame,
    never recomputing existing depth PNGs). A sequence that already ships depth columns (a real
    RGBD dataset) is left untouched, as are the sequence's own rgb.csv and calibration.yaml -
    only the experiment's rgb_exp.csv and calibration_exp.yaml are rewritten."""
    df = pd.read_csv(rgb_exp_csv)
    if 'path_depth_0' in df.columns:
        print_msg(SCRIPT_LABEL, f"depth: {rgb_exp_csv.name} already has depth columns; leaving them untouched", verb='LOW')
        return
    if 'path_rgb_1' not in df.columns:
        print_msg(SCRIPT_LABEL, f"depth=fastfoundationstereo requires a stereo sequence (path_rgb_0 and path_rgb_1) but {rgb_exp_csv} has no path_rgb_1; skipping depth columns", flag="error", verb='NONE')
        return

    depth_folder = f"{DEPTH_FOLDER_BASE}_0"
    marker = sequence_path / depth_folder / DEPTH_COMPLETE_MARKER
    if not marker.exists():
        print_msg(SCRIPT_LABEL, f"depth: depth missing for {sequence_name}, running 'pixi run stereo-inference {dataset.dataset_name} {sequence_name}' ...", verb='LOW')
        subprocess.run(["pixi", "run", "-e", "fastfoundationstereo", "stereo-inference", dataset.dataset_name, sequence_name], cwd=VSLAM_LAB_DIR, check=True)

    # The marker records the depth_factor the script encoded the PNGs with (empty markers predate
    # that and were written with the script's default).
    depth_factor = DEPTH_FACTOR_DEFAULT
    for line in marker.read_text().splitlines():
        if line.startswith("depth_factor:"):
            depth_factor = float(line.split(":", 1)[1])

    df["ts_depth_0 (ns)"] = df["ts_rgb_0 (ns)"]
    df["path_depth_0"] = [f"{depth_folder}/{Path(p).stem}.png" for p in df["path_rgb_0"]]
    df.to_csv(rgb_exp_csv, index=False)
    print_msg(SCRIPT_LABEL, f"depth: appended fastfoundationstereo depth columns to {rgb_exp_csv.name}", verb='LOW')

    register_depth_stream(calibration_exp_yaml, depth_folder, depth_factor)

def register_depth_stream(calibration_exp_yaml: Path, depth_folder: str, depth_factor: float) -> None:
    """Declare a generated depth stream on the rgb_0 camera entry of the experiment's
    calibration_exp.yaml: depth_name/depth_factor and a '+depth' cam_type, with the same field
    placement as DatasetVSLAMLAB_calibration._get_rgbd_yaml_section (depth_name after cam_type,
    depth_factor after fps), so rgbd baselines - which read depth_name/depth_factor from the
    calibration yaml - can consume it. The edit is line-based (the file's hand-formatted flow
    style and comments are preserved) and idempotent. An rgb_0 entry that already declares a
    different depth stream (a real RGBD dataset) is left untouched. Only the per-experiment copy
    is edited - never the sequence's calibration.yaml."""
    if not calibration_exp_yaml.exists():
        print_msg(SCRIPT_LABEL, f"depth: {calibration_exp_yaml} missing; cannot register the depth stream", flag="error", verb='NONE')
        return

    lines = calibration_exp_yaml.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "cam_name: rgb_0" in line)
        end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    except StopIteration:
        print_msg(SCRIPT_LABEL, f"depth: no rgb_0 camera entry found in {calibration_exp_yaml}; cannot register the depth stream", flag="error", verb='NONE')
        return

    def find(key: str) -> int | None:
        return next((i for i in range(start, end + 1) if lines[i].lstrip().startswith(key)), None)

    factor_line = f"     depth_factor: {float(depth_factor)},"
    depth_name_idx = find("depth_name:")
    if depth_name_idx is not None:
        if depth_folder not in lines[depth_name_idx]:
            print_msg(SCRIPT_LABEL, f"depth: rgb_0 already declares another depth stream ({lines[depth_name_idx].strip().rstrip(',')}); leaving {calibration_exp_yaml.name} untouched", flag="error", verb='NONE')
            return
        factor_idx = find("depth_factor:")
        if factor_idx is not None and lines[factor_idx] == factor_line:
            return  # already registered with the same depth_factor
        if factor_idx is not None:
            lines[factor_idx] = factor_line
        else:
            lines.insert(depth_name_idx + 1, factor_line)
    else:
        cam_type_idx, fps_idx = find("cam_type:"), find("fps:")
        if cam_type_idx is None or fps_idx is None:
            print_msg(SCRIPT_LABEL, f"depth: rgb_0 entry in {calibration_exp_yaml} has no cam_type/fps line; cannot register the depth stream", flag="error", verb='NONE')
            return
        cam_type = lines[cam_type_idx].split("cam_type:")[1].strip().rstrip(",")
        if "+depth" not in cam_type:
            lines[cam_type_idx] = f"     cam_type: {cam_type}+depth,"
        # insert bottom-up so the earlier index stays valid
        lines.insert(fps_idx + 1, factor_line)
        lines.insert(cam_type_idx + 1, f"     depth_name: {depth_folder},")

    calibration_exp_yaml.write_text("\n".join(lines) + "\n")
    print_msg(SCRIPT_LABEL, f"depth: registered depth stream '{depth_folder}' (depth_factor={depth_factor:g}) in {calibration_exp_yaml.name}", verb='LOW')

def get_sequence_data_for_evaluation(exp: Any, dataset: Any, sequence_name: str) -> None:
    sequence_path = dataset.dataset_path /  sequence_name
    exp_folder = Path(exp.folder) / Path(dataset.dataset_folder) / sequence_name
    groundtruth_csv = sequence_path / 'groundtruth.csv'
    groundtruth_csv_dst = exp_folder / 'groundtruth.csv'
    if not groundtruth_csv_dst.exists():
        shutil.copy(groundtruth_csv, groundtruth_csv_dst)

    rgb_folder = sequence_path / "rgb_0"
    first_image = next(rgb_folder.iterdir())
    thumbnails_folder =  VSLAMLAB_EVALUATION / "thumbnails"
    rgb_thumbnail = thumbnails_folder/ f"rgb_thumbnail_{dataset.dataset_name}_{sequence_name}{first_image.suffix}"
    thumbnails_folder.mkdir(parents=True, exist_ok=True)
    if not rgb_thumbnail.exists():
        shutil.copy(first_image, rgb_thumbnail)