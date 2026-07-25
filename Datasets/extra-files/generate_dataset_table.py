#!/usr/bin/env python3
"""
Module: generate_dataset_table.py
Description: Scans Datasets/dataset_files/*.yaml (and, via get_dataset.py, the
             matching *.py) and writes a Markdown table (dataset name | camera
             models | modes | download source | download issues | maintainer |
             AI-assisted) in the same style as the Datasets table in the
             project README.
Author: Alejandro Fontan Villacampa
Version: 1.2
Created: 2026-07-19
Updated: 2026-07-25
License: GPLv3
List of Known Bugs: None
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

DEFAULT_CAM_MODELS = ["pinhole"]
DEFAULT_MODES = ["mono"]

VSLAM_LAB_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_FILES_DIR = VSLAM_LAB_DIR / "Datasets" / "dataset_files"
DEFAULT_GET_DATASET_PY = VSLAM_LAB_DIR / "Datasets" / "get_dataset.py"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().with_name("dataset_table.md")

_IMPORT_RE = re.compile(r"from Datasets\.dataset_files\.(\S+) import (\w+)")
_SWITCHER_RE = re.compile(r'"([\w\-]+)"\s*:\s*lambda:\s*(\w+)\(')
_CLASS_RE = re.compile(r"^class\s+(\w+)\(([\w.]+)\)\s*:", re.MULTILINE)
_ISSUE_ID_RE = re.compile(r'issue_id\s*=\s*"([^"]+)"')


def _parse_dataset_name_to_class(get_dataset_py: Path) -> dict[str, tuple[str, str]]:
    """Return {dataset_name: (class_name, module_stem)} from get_dataset.py's imports + switcher dict."""
    text = get_dataset_py.read_text(encoding="utf-8")
    class_to_module: dict[str, str] = {}
    for m in _IMPORT_RE.finditer(text):
        module_stem, class_name = m.groups()
        class_to_module[class_name] = module_stem

    mapping: dict[str, tuple[str, str]] = {}
    for dataset_name, class_name in _SWITCHER_RE.findall(text):
        if class_name in class_to_module:
            mapping[dataset_name] = (class_name, class_to_module[class_name])
    return mapping


def _class_blocks(py_source: str) -> dict[str, tuple[str, str]]:
    """Return {class_name: (base_class_name, source_slice)} for every top-level class in the module."""
    matches = list(_CLASS_RE.finditer(py_source))
    blocks: dict[str, tuple[str, str]] = {}
    for i, m in enumerate(matches):
        class_name = m.group(1)
        base_name = m.group(2).rsplit(".", 1)[-1]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(py_source)
        blocks[class_name] = (base_name, py_source[m.start():end])
    return blocks


def _issue_ids_for_class(class_name: str, blocks: dict[str, tuple[str, str]], _seen: set[str] | None = None) -> list[str]:
    """Walk the (single-file) inheritance chain to find the nearest get_download_issues override."""
    _seen = _seen or set()
    if class_name in _seen or class_name not in blocks:
        return []
    _seen.add(class_name)
    base_name, text = blocks[class_name]
    if re.search(r"\bdef\s+get_download_issues\b", text):
        return sorted(set(_ISSUE_ID_RE.findall(text)))
    return _issue_ids_for_class(base_name, blocks, _seen)


def _download_issues_for_dataset(dataset_name: str, dataset_files_dir: Path,
                                  name_to_class: dict[str, tuple[str, str]],
                                  class_blocks_cache: dict[str, dict[str, tuple[str, str]]]) -> list[str]:
    if dataset_name not in name_to_class:
        return []
    class_name, module_stem = name_to_class[dataset_name]
    if module_stem not in class_blocks_cache:
        py_file = dataset_files_dir / f"{module_stem}.py"
        class_blocks_cache[module_stem] = _class_blocks(py_file.read_text(encoding="utf-8"))
    return _issue_ids_for_class(class_name, class_blocks_cache[module_stem])


def _is_google_drive_url(url: object) -> bool:
    """True for any Google Drive host, including pre-resolved direct-download links
    (e.g. drive.usercontent.google.com), not just the drive.google.com share-link host."""
    if not isinstance(url, str):
        return False
    host = urlparse(url).hostname or ""
    return host.endswith("google.com") and "drive" in host


def _download_labels(cfg: dict) -> list[str]:
    """Infer the source pattern(s) (hugging-face/google-drive/website/local/other) from the YAML's download fields."""
    labels: list[str] = []
    if cfg.get("hf_repo_id"):
        labels.append("hugging-face")

    urls = [cfg.get("url_download_root"), cfg.get("url_download_sequences")]
    urls = [u for u in urls if u]
    if any(_is_google_drive_url(u) for u in urls):
        labels.append("google-drive")
    elif urls:
        labels.append("website")

    sequence_location = cfg.get("sequence_location")
    if isinstance(sequence_location, list):
        is_local = "local" in sequence_location
    else:
        is_local = sequence_location == "local"
    if is_local:
        labels.append("local")

    if not labels:
        labels.append("other")
    return labels


def _load_dataset_entries(dataset_files_dir: Path, get_dataset_py: Path) -> list[dict[str, str]]:
    name_to_class = _parse_dataset_name_to_class(get_dataset_py)
    class_blocks_cache: dict[str, dict[str, tuple[str, str]]] = {}

    entries: list[dict[str, str]] = []
    for yaml_file in sorted(dataset_files_dir.glob("dataset_*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        dataset_name = str(cfg.get("dataset_name", yaml_file.stem.removeprefix("dataset_"))).strip()
        cam_models = cfg.get("cam_models", DEFAULT_CAM_MODELS)
        modes = cfg.get("modes", DEFAULT_MODES)
        download_labels = _download_labels(cfg)
        issue_ids = _download_issues_for_dataset(dataset_name, dataset_files_dir, name_to_class, class_blocks_cache)
        maintainer = str(cfg.get("vslamlab_maintainer", {}).get("name", "")).strip()
        assisted_by = str(cfg.get("vslamlab_maintainer", {}).get("assisted_by", "")).strip()

        entries.append(
            {
                "dataset_name": dataset_name,
                "cam_models": " ".join(f"`{m}`" for m in cam_models),
                "modes": " ".join(f"`{m}`" for m in modes),
                "download": " ".join(f"`{d}`" for d in download_labels),
                "issues": " ".join(f"`{i}`" for i in issue_ids),
                "maintainer": maintainer,
                "assisted_by": assisted_by,
            }
        )
    return entries


def _render_markdown_table(entries: list[dict[str, str]]) -> str:
    lines = [
        "| Dataset | Camera Models | Modes | Download | Download Issues | Maintainer | AI-Assisted |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['dataset_name']}` | {entry['cam_models']} | {entry['modes']} | "
            f"{entry['download']} | {entry['issues']} | {entry['maintainer']} | {entry['assisted_by']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-files-dir", type=Path, default=DEFAULT_DATASET_FILES_DIR,
                         help="Directory containing dataset_*.yaml/.py files")
    parser.add_argument("--get-dataset-py", type=Path, default=DEFAULT_GET_DATASET_PY,
                         help="Path to get_dataset.py (used to map dataset name -> implementing class/file)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD,
                         help="Output .md file path")
    args = parser.parse_args()

    entries = _load_dataset_entries(args.dataset_files_dir, args.get_dataset_py)
    markdown = _render_markdown_table(entries)

    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(entries)} dataset rows to {args.output}")


if __name__ == "__main__":
    main()
