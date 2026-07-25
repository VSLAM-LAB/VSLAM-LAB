import argparse
import csv
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import py7zr
import yaml
from colorama import Fore, Style
from huggingface_hub import login, snapshot_download
from PIL import Image
from scipy.spatial.transform import Rotation

from path_constants import (
    HUGGINGFACE_TOKEN, RGB_BASE_FOLDER, VSLAM_LAB_DIR, VSLAMLAB_BENCHMARK, VSLAMLAB_VERBOSITY, VerbosityManager,
)

##################################################################################################################################################
# Blocks in this file, in order:
#   OTHERS                          - ungrouped misc helpers, some possibly dead (issue #69)
#   Sequence path helpers           - sequence_path / sequence_rgb_csv / raw_path (issue #64)
#   CSV row helpers                 - read_csv_rows / write_csv_rows / ensure_raw_backup /
#                                      overwrite_csv_with_backup / revert_csv_from_backup (issue #65)
#   resolve_sequence_targets        - the sequence-target CLI argument convention (CLAUDE.md),
#                                      plus resolve_sequence_targets_or_exit for main()'s use;
#                                      not yet adopted repo-wide (issue #70)
#   Download / decompress helpers   - downloadFile / decompressFile (issue #66)
#   Hugging Face download helpers   - hf_token / download_hf_snapshot / ensure_hf_sequence_download,
#                                      used by dataset_soneva.py and dataset_sweetcorals.py; other
#                                      HF-based datasets (e.g. dataset_ariel.py) not yet migrated
#   COLMAP helpers                  - read_colmap_cameras / read_colmap_images /
#                                      world_to_camera_to_pose, used by dataset_soneva.py and
#                                      dataset_sweetcorals.py; generic COLMAP binary-format parsing,
#                                      not specific to either dataset
#   Image resize helpers            - compute_scaled_size, used by dataset_soneva.py/
#                                      dataset_sweetcorals.py (via HFColmapDatasetMixin) and
#                                      dataset_eiffel_tower.py; dataset_videos.py's
#                                      estimate_new_resolution (inherited by dataset_strayscanner.py)
#                                      does the same math with a different signature, not yet
#                                      consolidated onto this
#   Printing helpers                - ws / show_time / format_msg / print_msg / make_printers (issue #68)
#   Trajectory / pandas CSV helpers - read_trajectory_csv / read_trajectory_txt /
#                                     save_trajectory_csv / read_csv (issue #67)
##################################################################################################################################################

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

##################################################################################################################################################
# OTHERS
#
# Miscellaneous standalone helpers that don't yet belong to any of the feature-specific blocks
# below (path/CSV/logging/etc.) - grouped here for now rather than left scattered through the
# file. Some appear to have no callers anywhere else in the codebase (check_parameter_for_
# relative_path, filter_inputs, is_image_file, list_image_files_in_folder) - worth confirming
# whether they're genuinely unused (like the activate_env/deactivate_env pair that was removed)
# or just not yet wired up. TODO: audit this block - split each function into (or merge into) an
# appropriate topical block, or remove it if dead - tracked in issue #69.
##################################################################################################################################################
def check_parameter_for_relative_path(parameter_value):
    if "VSLAM-LAB" in parameter_value:
        if ":" in parameter_value:
            return re.sub(r'(?<=:)[^:]*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
        return re.sub(r'^.*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
    return parameter_value

def filter_inputs(args):
    if  not args.run and not args.evaluate and not args.compare:
        args.run = True
        args.evaluate = True
        args.compare = True

def find_files_with_string(folder_path, matching_string):
    matching_files = []
    for file_name in os.listdir(folder_path):
        if matching_string in file_name:
            file_path = os.path.join(folder_path, file_name)
            matching_files.append(file_path)
    matching_files.sort()
    return matching_files


def load_yaml_file(yaml_file: str | Path) -> Any:
    yaml_path = Path(yaml_file)

    # Existence + type
    if not yaml_path.exists():
        print_msg(SCRIPT_LABEL, f"Error: The file '{yaml_file}' does not exist.", flag="error", verb='NONE')
        sys.exit(1)
    if not yaml_path.is_file():
        print_msg(SCRIPT_LABEL, f"Error: The file '{yaml_file}' is not a yaml file.", flag="error", verb='NONE')
        sys.exit(1)
    
    # Extension check
    if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        print_msg(SCRIPT_LABEL, f"The file '{yaml_path}' is not a YAML file.", flag="error", verb='NONE')
        sys.exit(1)

    # Parse YAML
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Re-raise with filename context
        raise yaml.YAMLError(f"Invalid YAML in '{yaml_path}': {e}") from e
    except OSError as e:
        # File I/O issues (permissions, etc.)
        raise OSError(f"Error reading '{yaml_path}': {e}") from e
    
    return yaml_data

def find_common_sequences(experiments):
    num_experiments = len(experiments)
    exp_tmp = {}
    for [_, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                if not (dataset_name in exp_tmp):
                    exp_tmp[dataset_name] = {}
                for sequence_name in sequence_names:
                    if sequence_name in exp_tmp[dataset_name]:
                        exp_tmp[dataset_name][sequence_name] += 1
                    else:
                        exp_tmp[dataset_name][sequence_name] = 1

    dataset_sequences = {}
    for [dataset_name, sequence_names] in exp_tmp.items():
        for [sequence_name, num_sequences] in sequence_names.items():
            if num_experiments == num_sequences:
                if dataset_name not in dataset_sequences:
                    dataset_sequences[dataset_name] = []
                dataset_sequences[dataset_name].append(sequence_name)
    return dataset_sequences


def replace_string_in_files(directory, old_string, new_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h') or file.endswith('.cpp'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = content.replace(old_string, new_string)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)


def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            return True
    except Exception:
        return False


def list_image_files_in_folder(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and is_image_file(file_path):
            image_files.append(filename)
    return image_files
##################################################################################################################################################


##################################################################################################################################################
# Sequence path helpers
#
# General-purpose helpers for building the standard per-sequence paths (and their pre-edit backup
# counterparts) under VSLAMLAB_BENCHMARK. Currently used by the sequence-target CLI machinery
# below and the scripts built on it (Datasets/extra-files/*.py) - the plan is for these to become
# the canonical way any VSLAM-LAB code builds these paths. TODO: migrate other ad-hoc path
# construction across the codebase (e.g. Datasets/DatasetVSLAMLAB.py, vslamlab_utilities.py) to
# use them - tracked in issue #64.
##################################################################################################################################################
def sequence_path(dataset_name: str, sequence_name: str) -> Path:
    """<benchmark>/<DATASET_FOLDER>/<sequence_name>, the standard per-sequence directory."""
    return VSLAMLAB_BENCHMARK / dataset_name.upper() / sequence_name


def sequence_rgb_csv(dataset_name: str, sequence_name: str) -> Path:
    return sequence_path(dataset_name, sequence_name) / f"{RGB_BASE_FOLDER}.csv"


def raw_path(csv_path: str | Path) -> Path:
    """<name>_raw<suffix> next to csv_path - the pre-edit backup convention used by scripts that
    overwrite a sequence's rgb.csv/groundtruth.csv in place (see Datasets/extra-files/synch_gt.py,
    sample_vpr.py)."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_raw{csv_path.suffix}")


##################################################################################################################################################
# CSV row helpers
#
# General-purpose helpers for reading/writing a VSLAM-LAB sequence CSV (rgb.csv/groundtruth.csv)
# as a plain (header, rows) pair, using the atomic write-then-replace pattern used throughout
# Datasets/dataset_files/*.py, plus the backup/revert convention for scripts that edit a
# sequence's CSV in place: ensure_raw_backup/overwrite_csv_with_backup never overwrite an
# existing raw_path() backup, and revert_csv_from_backup restores from it (removing the backup by
# default, so its existence always means "this file has pending edits"). Currently used by
# Datasets/extra-files/synch_gt.py and sample_vpr.py - the plan is for these to become the
# canonical way any VSLAM-LAB code reads/writes/backs up these CSVs. TODO: migrate other ad-hoc
# CSV read/write call sites across the codebase (e.g. Datasets/dataset_files/*.py's own
# csv.reader/csv.writer + atomic-write boilerplate) to use them - tracked in issue #65.
##################################################################################################################################################
def read_csv_rows(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """(header, rows) for a VSLAM-LAB sequence CSV (rgb.csv/groundtruth.csv), rows sorted by the
    first (timestamp) column."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    rows.sort(key=lambda row: int(row[0]))
    return header, rows


def write_csv_rows(path: str | Path, header: list[str], rows: list[list[str]]) -> None:
    """Writes header+rows to path via the atomic write-then-replace pattern used throughout
    Datasets/dataset_files/*.py."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def ensure_raw_backup(csv_path: str | Path) -> Path:
    """Copies csv_path to its raw_path() backup if one doesn't already exist yet. NEVER
    overwrites an existing backup - call this once before any in-place edit of csv_path so the
    original is always recoverable via revert_csv_from_backup(). Returns the raw path either way."""
    csv_path = Path(csv_path)
    raw = raw_path(csv_path)
    if not raw.exists():
        shutil.copy2(csv_path, raw)
    return raw


def overwrite_csv_with_backup(path: str | Path, header: list[str], rows: list[list[str]]) -> Path:
    """ensure_raw_backup(path) followed by write_csv_rows(path, header, rows) - the standard
    "back up once, then overwrite" pattern used by Datasets/extra-files/synch_gt.py and
    sample_vpr.py. Returns the raw path."""
    raw = ensure_raw_backup(path)
    write_csv_rows(path, header, rows)
    return raw


def revert_csv_from_backup(csv_path: str | Path, *, keep_raw: bool = False) -> bool:
    """Restores csv_path from its raw_path() backup, if one exists. By default removes the
    backup afterward, so "a raw backup exists" always means "this file currently has pending
    edits" (the invariant ensure_raw_backup's callers rely on to skip re-processing an
    already-edited file). Pass keep_raw=True to copy instead of restoring-and-removing. Returns
    True if a backup was found and restored, False if there was nothing to revert."""
    csv_path = Path(csv_path)
    raw = raw_path(csv_path)
    if not raw.exists():
        return False
    if keep_raw:
        shutil.copy2(raw, csv_path)
    else:
        raw.replace(csv_path)
    return True


##################################################################################################################################################
# resolve_sequence_targets
#
# Shared CLI-argument resolver for tasks/scripts that operate on one or more dataset sequences
# (downloading, running, evaluating, syncing groundtruth, etc). See CLAUDE.md's "Sequence-target
# argument convention" for the shapes this supports; Datasets/extra-files/synch_gt.py is a
# worked example of a task built on top of it. Only adopted by run_vpr.py/sample_vpr.py/
# synch_gt.py so far - vslamlab_utilities.py's download_sequence/download_sequences/
# download_dataset/download_datasets (and possibly other commands) still reinvent this problem
# with their own narrower, more fragile argument shapes. TODO: migrate them onto this convention
# - tracked in issue #70.
##################################################################################################################################################
##################################################################################################################################################
def sequences_for_dataset(dataset_name: str) -> list[str]:
    """Every sequence with a downloaded rgb.csv under <benchmark>/<DATASET_FOLDER>/."""
    dataset_path = VSLAMLAB_BENCHMARK / dataset_name.upper()
    if not dataset_path.is_dir():
        print_msg(SCRIPT_LABEL, f"Dataset folder not found: {dataset_path}", flag="warning", verb='NONE')
        return []
    return sorted(
        p.name for p in dataset_path.iterdir()
        if p.is_dir() and (p / f"{RGB_BASE_FOLDER}.csv").is_file()
    )


def pairs_from_config_yaml(config_yaml: str | Path) -> list[tuple[str, str]]:
    """dataset: [sequence, ...] format, e.g. configs/config_*.yaml."""
    config = load_yaml_file(config_yaml) or {}
    return [
        (dataset_name, sequence_name)
        for dataset_name, sequence_names in config.items()
        for sequence_name in sequence_names
    ]


def pairs_from_exp_yaml(exp_yaml: str | Path) -> list[tuple[str, str]]:
    """exp_name: {Config: <config.yaml>, ...} format, e.g. configs/exp_*.yaml. Collects every
    pair from each distinct Config file referenced (deduplicated across experiments)."""
    exp_data = load_yaml_file(exp_yaml) or {}
    configs_dir = VSLAM_LAB_DIR / "configs"

    seen_configs: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for exp_name, settings in exp_data.items():
        config_name = (settings or {}).get("Config")
        if not config_name or config_name in seen_configs:
            continue
        seen_configs.add(config_name)

        config_path = configs_dir / config_name
        if not config_path.is_file():
            print_msg(SCRIPT_LABEL, f"Skipping experiment '{exp_name}' - config not found: {config_path}", flag="warning", verb='NONE')
            continue

        for pair in pairs_from_config_yaml(config_path):
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)
    return pairs


def add_sequence_target_args(parser: argparse.ArgumentParser) -> None:
    """Adds the standard sequence-target flags (CLAUDE.md's sequence-target argument convention)
    to an argparse parser. Only the single-dataset case stays positional (it's unambiguous and
    needs no filesystem lookup to parse); every other shape is an explicit flag so that how a
    command gets interpreted never depends on repo state (e.g. a stray file named the same as a
    dataset). Pass the parsed args straight through to resolve_sequence_targets()."""
    parser.add_argument(
        "targets", nargs="*", metavar="DATASET [SEQUENCE ...]",
        help="<dataset> [<sequence> ...] - one dataset, and (optionally) specific sequences of it"
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="DATASET",
        help="Every downloaded sequence of each named dataset"
    )
    parser.add_argument(
        "--sequences", nargs="+", action="append", metavar="DATASET_OR_SEQUENCE",
        help="<dataset> <sequence> [<sequence> ...] - repeatable; explicit sequences of one "
             "dataset per use, e.g. --sequences kitti 05 07 --sequences eth table_3"
    )
    parser.add_argument("--exp", metavar="EXP_YAML", help="Every pair referenced by an experiment yaml's Config file(s)")
    parser.add_argument("--configs", metavar="CONFIG_YAML", help="Every pair listed in a config yaml (dataset: [sequences])")


def resolve_sequence_targets(
    targets: list[str] | None = None,
    datasets: list[str] | None = None,
    sequences: list[list[str]] | None = None,
    exp: str | Path | None = None,
    configs: str | Path | None = None,
) -> list[tuple[str, str]]:
    """
    Resolves the flags added by add_sequence_target_args() into an explicit list of
    (dataset_name, sequence_name) pairs, per CLAUDE.md's sequence-target argument convention.
    Every argument here is additive - pass as many as apply and their results are concatenated.
      targets   - [<dataset>] or [<dataset>, <sequence>, ...]: one dataset, all or specific sequences
      datasets  - every downloaded sequence of each named dataset
      sequences - a list of [<dataset>, <sequence>, ...] groups (one per --sequences use)
      exp       - path to an experiment yaml; every pair from its Config file(s)
      configs   - path to a config yaml (dataset: [sequences])
    """
    pairs: list[tuple[str, str]] = []

    if targets:
        dataset_name, *sequence_names = targets
        if sequence_names:
            pairs.extend((dataset_name, s) for s in sequence_names)
        else:
            pairs.extend((dataset_name, s) for s in sequences_for_dataset(dataset_name))

    for dataset_name in datasets or []:
        pairs.extend((dataset_name, s) for s in sequences_for_dataset(dataset_name))

    for group in sequences or []:
        dataset_name, *sequence_names = group
        if sequence_names:
            pairs.extend((dataset_name, s) for s in sequence_names)
        else:
            pairs.extend((dataset_name, s) for s in sequences_for_dataset(dataset_name))

    if exp:
        pairs.extend(pairs_from_exp_yaml(exp))

    if configs:
        pairs.extend(pairs_from_config_yaml(configs))

    return pairs


def resolve_sequence_targets_or_exit(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """resolve_sequence_targets() from an argparse Namespace produced by add_sequence_target_args(),
    calling parser.error() (which prints usage and exits) if nothing resolved. The standard
    main() one-liner for scripts built on the sequence-target argument convention."""
    pairs = resolve_sequence_targets(
        targets=args.targets, datasets=args.datasets, sequences=args.sequences,
        exp=args.exp, configs=args.configs,
    )
    if not pairs:
        parser.error("No dataset/sequence targets resolved from the given arguments")
    return pairs


##################################################################################################################################################
# Download / decompress helpers
#
# Already the established, widely-adopted way to fetch and unpack a dataset sequence's raw
# archive - used across most of Datasets/dataset_files/*.py (dataset_eth.py, dataset_kitti.py,
# dataset_euroc.py, etc.), not something new that still needs integrating.
##################################################################################################################################################
def downloadFile(url, dest_dir_path, file_size=None):
    file_name = url.split('/')[-1]
    dest_file_path = os.path.join(dest_dir_path, file_name)

    url_object = urllib.request.urlopen(url)

    with open(dest_file_path, 'wb') as outfile:
        meta = url_object.info()
        if sys.version_info[0] == 2:
            header = meta.getheaders("Content-Length")
            if header:
                file_size = int(header[0])
        else:
            length = meta.get("Content-Length")
            if length:
                file_size = int(length)

        if file_size:
            print("    Downloading: %s (size [bytes]: %s)" % (url, file_size))
        else:
            print("    Downloading: %s (size: unknown)" % url)

        file_size_downloaded = 0
        block_size = 8192
        while True:
            buffer = url_object.read(block_size)
            if not buffer:
                break

            file_size_downloaded += len(buffer)
            outfile.write(buffer)

            if file_size:
                sys.stdout.write("        %d / %d  (%3f%%)\r" % (
                    file_size_downloaded, file_size, file_size_downloaded * 100. / file_size))
            else:
                sys.stdout.write("        %d bytes downloaded\r" % file_size_downloaded)
            
            sys.stdout.flush()

    return dest_file_path

def decompressFile(filepath, extract_to=None):
    """
    Decompress a .zip, .tar.gz, .tar, or .7z file and return the extraction directory.
    """
    filepath = str(filepath)
    
    if not extract_to:
        extract_to = os.path.dirname(filepath)

    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            for file in zip_ref.namelist():
                try:
                    zip_ref.extract(file, extract_to)
                except zipfile.BadZipFile:
                    print(f"Skipping corrupted file: {file}")
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz') or filepath.endswith('.tar'):
        mode = 'r:gz' if filepath.endswith('.gz') else 'r'
        with tarfile.open(filepath, mode) as tar_ref:
            tar_ref.extractall(extract_to)
    elif filepath.endswith('.7z'):
        seven_zip_cmd = shutil.which('7zz') or shutil.which('7z')
        if seven_zip_cmd:
            try:
                subprocess.check_call([seven_zip_cmd, "x", filepath, f"-o{extract_to}", "-y"])
            except subprocess.CalledProcessError as e:
                with py7zr.SevenZipFile(filepath, mode='r') as z:
                    z.extractall(path=extract_to)    
        else:
            with py7zr.SevenZipFile(filepath, mode='r') as z:
                z.extractall(path=extract_to)
    else:
        print("Unsupported file format. Please provide a .zip, .tar.gz, .tar, or .7z file.")
        return None

    return extract_to
##################################################################################################################################################


##################################################################################################################################################
# Hugging Face download helpers
#
# Shared by dataset_soneva.py and dataset_sweetcorals.py, which previously each had their own
# byte-identical _hf_token() plus near-identical fetch/flatten/cleanup logic in
# download_sequence_data(). snapshot_download() itself already resumes partial per-file downloads
# and skips files it already has, so it's safe to just call again after a dropped connection - the
# thing that previously defeated that was each dataset's own caller checking "does rgb_0_raw/
# exist" as its idempotency guard, which is true (the dir gets created) even after a download that
# died partway through. ensure_hf_sequence_download() fixes that by only marking a sequence done
# once every remote_dir has finished, via a completion-marker file rather than directory existence.
##################################################################################################################################################
def hf_token() -> str | None:
    """The configured Hugging Face token: HUGGINGFACE_TOKEN from path_constants.py if set (also
    logs in via huggingface_hub so gated repos work), else the HF_TOKEN env var, else None (public
    repos still work; gated ones will fail download with a clear Hugging Face error)."""
    if HUGGINGFACE_TOKEN is not None:
        login(token=HUGGINGFACE_TOKEN)
        return HUGGINGFACE_TOKEN
    return os.environ.get("HF_TOKEN")


def download_hf_snapshot(
    repo_id: str, remote_dir: str, local_dir: str | Path, *,
    pattern: str = "*", token: str | None = None, max_workers: int = 8,
) -> None:
    """Downloads every file matching remote_dir/pattern in a Hugging Face dataset repo, flattened
    directly into local_dir (the repo's remote_dir/ nesting, and any leftover .cache/, are removed
    afterwards). Only flattens/cleans up on success - if snapshot_download() raises partway
    through, local_dir is left in a partial, unflattened state. Callers wanting an idempotency
    guard against that (rather than gating a re-download skip on local_dir merely existing) should
    use ensure_hf_sequence_download() instead of calling this directly."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"{remote_dir}/{pattern}"],
        max_workers=max_workers,
        token=token,
    )

    nested_dir = local_dir / remote_dir
    for file_path in nested_dir.iterdir():
        file_path.rename(local_dir / file_path.name)
    shutil.rmtree(local_dir / remote_dir.split("/")[0], ignore_errors=True)
    shutil.rmtree(local_dir / ".cache", ignore_errors=True)


def ensure_hf_sequence_download(
    repo_id: str, remote_dirs: list[str], local_dir: str | Path, *,
    pattern: str = "*", token: str | None = None, max_workers: int = 8,
) -> bool:
    """Idempotent wrapper around download_hf_snapshot() for a dataset's per-sequence raw download
    step: fetches every remote_dir in remote_dirs into the same flat local_dir (a sequence split
    across multiple remote subfolders, e.g. two capture days), then marks local_dir complete (a
    .download_complete marker file) only once all of them succeed. Returns False without doing
    anything if local_dir is already marked complete. Safe against a dropped connection mid-run: a
    previously-interrupted attempt has no marker, so calling again resumes/retries instead of being
    silently treated as already downloaded."""
    local_dir = Path(local_dir)
    marker = local_dir / ".download_complete"
    if marker.exists():
        return False

    for remote_dir in remote_dirs:
        download_hf_snapshot(repo_id, remote_dir, local_dir, pattern=pattern, token=token, max_workers=max_workers)

    marker.touch()
    return True
##################################################################################################################################################


##################################################################################################################################################
# COLMAP helpers
#
# Parsing for COLMAP's binary reconstruction format (https://colmap.github.io/format.html#binary-file-format),
# used by dataset_soneva.py and dataset_sweetcorals.py to read each sequence's cameras.bin/
# images.bin. Kept as plain functions over a file path rather than dataset methods, since neither
# the binary format nor the world-to-camera pose convention it stores depend on which dataset
# fetched the file - only fetching it (dataset_soneva.py/dataset_sweetcorals.py's own
# _fetch_colmap_file) is dataset-specific.
##################################################################################################################################################

# COLMAP camera models these two datasets' reconstructions are known to use.
# https://colmap.github.io/cameras.html
_COLMAP_MODEL_NUM_PARAMS = {0: 3, 1: 4}  # 0: SIMPLE_PINHOLE (f, cx, cy), 1: PINHOLE (fx, fy, cx, cy)


def read_colmap_cameras(path: str | Path) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    num_cameras = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    cameras: dict[int, tuple[str, int, int, tuple[float, ...]]] = {}
    for _ in range(num_cameras):
        camera_id, model_id = struct.unpack_from("<ii", data, offset)
        offset += 8
        width, height = struct.unpack_from("<QQ", data, offset)
        offset += 16
        n = _COLMAP_MODEL_NUM_PARAMS[model_id]
        params = struct.unpack_from(f"<{n}d", data, offset)
        offset += 8 * n
        model_name = "SIMPLE_PINHOLE" if model_id == 0 else "PINHOLE"
        cameras[camera_id] = (model_name, width, height, params)
    return cameras


def read_colmap_images(path: str | Path) -> dict[str, tuple[int, tuple[float, ...], tuple[float, ...]]]:
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    num_images = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    images: dict[str, tuple[int, tuple[float, ...], tuple[float, ...]]] = {}
    for _ in range(num_images):
        offset += 4  # image_id
        qvec = struct.unpack_from("<4d", data, offset)
        offset += 32
        tvec = struct.unpack_from("<3d", data, offset)
        offset += 24
        camera_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4

        end = data.index(b"\x00", offset)
        name = data[offset:end].decode("utf-8")
        offset = end + 1

        num_points2d = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + num_points2d * 24  # x, y (double) + point3D_id (int64) per point

        images[name] = (camera_id, qvec, tvec)
    return images


def world_to_camera_to_pose(
    qvec: tuple[float, float, float, float], tvec: tuple[float, float, float]
) -> tuple[float, float, float, float, float, float, float]:
    # COLMAP's qvec/tvec transform world -> camera coordinates (X_cam = R * X_world + t).
    # Groundtruth trajectories are stored as the camera pose in the world frame instead.
    qw, qx, qy, qz = qvec
    rot_world_to_cam = Rotation.from_quat([qx, qy, qz, qw])
    rot_cam_to_world = rot_world_to_cam.inv()

    t_world = rot_cam_to_world.apply(-np.array(tvec))
    qx_w, qy_w, qz_w, qw_w = rot_cam_to_world.as_quat()

    return float(t_world[0]), float(t_world[1]), float(t_world[2]), float(qx_w), float(qy_w), float(qz_w), float(qw_w)
##################################################################################################################################################


##################################################################################################################################################
# Image resize helpers
#
# Shared by dataset_soneva.py/dataset_sweetcorals.py (via HFColmapDatasetMixin) and
# dataset_eiffel_tower.py, which each had a byte-identical _compute_scaled_size method before this
# was pulled out. dataset_videos.py's estimate_new_resolution (inherited by
# dataset_strayscanner.py) computes the same thing but takes explicit width/height instead of a
# tuple and tolerates target_resolution=None - not yet migrated onto this, since consolidating it
# means touching dataset_videos.py's/dataset_strayscanner.py's call sites too.
##################################################################################################################################################
def compute_scaled_size(
    original_size: tuple[int, int], target_resolution: tuple[int, int] | None
) -> tuple[int, int]:
    """Scales original_size (width, height) to match target_resolution's pixel area while
    preserving aspect ratio - the downscaling convention used by dataset create_rgb_folder hooks
    when a dataset's source images are bigger than the target (see the add-dataset skill's
    `resize` field). Returns original_size unchanged if target_resolution is None."""
    if target_resolution is None:
        return original_size

    target_w, target_h = target_resolution
    orig_w, orig_h = original_size
    target_area = target_w * target_h

    scaled_h = int(np.sqrt(target_area * orig_h / orig_w))
    scaled_w = int(target_area / scaled_h)
    return scaled_w, scaled_h
##################################################################################################################################################


##################################################################################################################################################
# Printing helpers
#
# The hand-rolled, colorama-based logging story used across most of the codebase (print_msg is
# used in 16+ files) alongside per-file SCRIPT_LABEL prefixes (CLAUDE.md's "Logging uses a
# per-file SCRIPT_LABEL ANSI-colored prefix pattern" convention). Note loguru is already a pixi
# dependency and already used directly (bypassing print_msg) in Datasets/DatasetVSLAMLAB.py and
# dataset_squidle.py - two parallel logging approaches coexist today. TODO: design a single,
# consistent logging story (loguru-based or otherwise) and adopt it across VSLAM-LAB - tracked in
# issue #68.
##################################################################################################################################################
def ws(n):
    white_spaces = ""
    for i in range(0, n):
        white_spaces = white_spaces + " "
    return white_spaces


def show_time(time_s):
    if time_s < 60:
        return f"{time_s:.2f} seconds"
    if time_s < 3600:
        return f"{(time_s / 60):.2f} minutes"
    return f"{(time_s / 3600):.2f} hours"


def format_msg(script_label, msg, flag="info"):
    if flag == "info":
        return f"{script_label}{msg}"
    elif flag == "warning":
        return f"{script_label}{Fore.YELLOW} {msg} {Style.RESET_ALL}"
    elif flag == "error":
        return f"{script_label}{Fore.RED} {msg} {Style.RESET_ALL}"


def print_msg(script_label, msg, flag="info", verb='NONE'):
    if VerbosityManager[verb] <= VerbosityManager[VSLAMLAB_VERBOSITY]:
        print(format_msg(script_label, msg, flag))


def make_printers(script_label: str):
    """Returns (print_info, print_warning) bound to script_label, so callers get the terse
    print_info(msg)/print_warning(msg) call sites without redefining these two wrappers (and
    re-capturing their own SCRIPT_LABEL) in every file - see Datasets/extra-files/*.py."""
    def print_info(msg: str) -> None:
        print_msg(script_label, msg)

    def print_warning(msg: str) -> None:
        print_msg(script_label, msg, flag="warning")

    return print_info, print_warning
##################################################################################################################################################
# Trajectory / pandas CSV helpers
#
# pandas-based readers/writers for trajectory and generic CSV files, distinct from the plain
# csv-module "CSV row helpers" block above (which is sequence rgb.csv/groundtruth.csv specific).
# Known callers today: read_csv -> vslamlab_utilities.py, Evaluate/evaluate_functions.py,
# Evaluate/compare_functions.py, Evaluate/plot_functions.py; read_trajectory_csv/
# read_trajectory_txt/save_trajectory_csv -> Evaluate/evo_functions.py. TODO: audit for any other
# ad-hoc pd.read_csv/to_csv call sites doing the same job across the codebase and make usage
# consistent - tracked in issue #67.
##################################################################################################################################################
def read_trajectory_csv(csv_file):
    try:
        trajectory = pd.read_csv(Path(csv_file))
        if trajectory.empty:
            trajectory = None
    except (pd.errors.EmptyDataError, FileNotFoundError):
        trajectory = None
    return trajectory

def read_trajectory_txt(txt_file):
    try:
        trajectory = pd.read_csv(Path(txt_file), header = None, sep = ' ', lineterminator = '\n')
        if trajectory.empty:
            trajectory = None
    except (pd.errors.EmptyDataError, FileNotFoundError):
        trajectory = None
    return trajectory

def save_trajectory_csv(trajectory_csv, trajectory, header=None, index=False, sep=' ', lineterminator='\n'):
    trajectory.to_csv(trajectory_csv, header=header, index=index, sep=sep, lineterminator=lineterminator)

def read_csv(csv_file):
    if not os.path.exists(csv_file):
        return pd.DataFrame()
    try:
        csv_data = pd.read_csv(csv_file,  dtype={'sequence_name': str})
        if csv_data.empty:
            return pd.DataFrame()
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
    return csv_data
##################################################################################################################################################

