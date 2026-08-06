#!/usr/bin/env python3
"""
Module: generate_dataset_table.py
Description: Scans Datasets/dataset_files/*.yaml (and, via get_dataset.py, the
             matching *.py) and writes a Markdown table (features | label |
             modes | camera models | raw format | license | download source |
             download issues | maintainer | AI-assisted). Features and License
             are cross-referenced, not derived from the yaml alone: Features
             comes from README.md's Datasets/Tools tables (including their
             commented-out placeholder rows), License from each dataset's own
             yaml about.license field. The yaml-scanning core is shared (via
             load_dataset_entries) with generate_dataset_xlsx.py (bibliography
             columns from about.publication/year/bibtex_key/bibtex/access) and
             generate_readme_datasets_table.py (regenerates README.md's own
             Datasets table from about.features instead of scraping it).
             Datasets table region in README.md may be Markdown or raw HTML
             (colspan divider rows need HTML) - _readme_features understands
             both, including the HTML table's Label|Features|Summary|...
             column order (Label leads, unlike the Tools Markdown table).
Author: Alejandro Fontan Villacampa
Version: 2.0
Created: 2026-07-19
Updated: 2026-08-04
License: GPLv3
List of Known Bugs: None
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

DEFAULT_CAM_MODELS = ["pinhole"]
DEFAULT_MODES = ["mono"]
DEFAULT_RAW_FORMATS = ["other"]

VSLAM_LAB_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_FILES_DIR = VSLAM_LAB_DIR / "Datasets" / "dataset_files"
DEFAULT_GET_DATASET_PY = VSLAM_LAB_DIR / "Datasets" / "get_dataset.py"
DEFAULT_README_MD = VSLAM_LAB_DIR / "README.md"
DEFAULT_OUTPUT_MD = Path(__file__).resolve().with_name("dataset_table.md")

_IMPORT_RE = re.compile(r"from Datasets\.dataset_files\.(\S+) import (\w+)")
_SWITCHER_RE = re.compile(r'"([\w\-]+)"\s*:\s*lambda:\s*(\w+)\(')
# Base-class list is optional (a mixin can be declared with no parens at all, e.g.
# "class HFColmapDatasetMixin:") and may list more than one base separated by commas (e.g.
# "class SonevaDataset(HFColmapDatasetMixin, DatasetVSLAMLAB):") - both forms are used in
# dataset_soneva.py/dataset_sweetcorals.py, the only multi-inheritance dataset files in the repo.
_CLASS_RE = re.compile(r"^class\s+(\w+)(?:\(([^)]*)\))?\s*:", re.MULTILINE)
_LOCAL_IMPORT_RE = re.compile(r"^from Datasets\.dataset_files\.(\w+) import (.+)$", re.MULTILINE)
_ISSUE_ID_RE = re.compile(r'issue_id\s*=\s*"([^"]+)"')
_HTML_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_HTML_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


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


def _class_blocks(py_source: str) -> dict[str, tuple[list[str], str]]:
    """Return {class_name: (base_class_names, source_slice)} for every top-level class in the module."""
    matches = list(_CLASS_RE.finditer(py_source))
    blocks: dict[str, tuple[list[str], str]] = {}
    for i, m in enumerate(matches):
        class_name = m.group(1)
        bases_raw = m.group(2) or ""
        base_names = [b.strip().rsplit(".", 1)[-1] for b in bases_raw.split(",") if b.strip()]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(py_source)
        blocks[class_name] = (base_names, py_source[m.start():end])
    return blocks


def _local_imports(py_source: str) -> dict[str, str]:
    """Return {imported_class_name: module_stem} for `from Datasets.dataset_files.X import Y[, Z]`
    lines - lets a mixin defined in a sibling dataset file (e.g. HFColmapDatasetMixin, imported by
    dataset_sweetcorals.py from dataset_soneva.py) still be found when walking a class's bases."""
    mapping: dict[str, str] = {}
    for module_stem, names in _LOCAL_IMPORT_RE.findall(py_source):
        for name in names.split(","):
            name = name.strip()
            if name:
                mapping[name] = module_stem
    return mapping


def _issue_ids_for_class(class_name: str, module_stem: str, dataset_files_dir: Path,
                          class_blocks_cache: dict[str, dict[str, tuple[list[str], str]]],
                          imports_cache: dict[str, dict[str, str]],
                          _seen: set[tuple[str, str]] | None = None) -> list[str]:
    """Walk the inheritance chain - following cross-file mixin imports too - to find the nearest
    get_download_issues override."""
    _seen = _seen or set()
    key = (module_stem, class_name)
    if key in _seen:
        return []
    _seen.add(key)

    if module_stem not in class_blocks_cache:
        source = (dataset_files_dir / f"{module_stem}.py").read_text(encoding="utf-8")
        class_blocks_cache[module_stem] = _class_blocks(source)
        imports_cache[module_stem] = _local_imports(source)

    blocks = class_blocks_cache[module_stem]
    if class_name not in blocks:
        return []

    base_names, text = blocks[class_name]
    if re.search(r"\bdef\s+get_download_issues\b", text):
        return sorted(set(_ISSUE_ID_RE.findall(text)))

    for base_name in base_names:
        if base_name in blocks:
            result = _issue_ids_for_class(base_name, module_stem, dataset_files_dir, class_blocks_cache, imports_cache, _seen)
        elif base_name in imports_cache[module_stem]:
            other_module = imports_cache[module_stem][base_name]
            result = _issue_ids_for_class(base_name, other_module, dataset_files_dir, class_blocks_cache, imports_cache, _seen)
        else:
            result = []
        if result:
            return result
    return []


def _download_issues_for_dataset(dataset_name: str, dataset_files_dir: Path,
                                  name_to_class: dict[str, tuple[str, str]],
                                  class_blocks_cache: dict[str, dict[str, tuple[list[str], str]]],
                                  imports_cache: dict[str, dict[str, str]]) -> list[str]:
    if dataset_name not in name_to_class:
        return []
    class_name, module_stem = name_to_class[dataset_name]
    return _issue_ids_for_class(class_name, module_stem, dataset_files_dir, class_blocks_cache, imports_cache)


def _download_labels(cfg: dict) -> list[str]:
    """Infer the source pattern(s) (hugging-face/google-drive/website/api/local/other) from the
    YAML's download fields. google-drive vs. website is decided purely by which field is present,
    not by inspecting the URL's host: google_drive_link is for a real drive.google.com share link
    that needs gdown to navigate Drive's virus-scan interstitial (Model: dataset_drunkards.py,
    dataset_hilti2026.py); url_download_root/url_download_sequences is a website label
    unconditionally, even when the URL happens to be hosted on Google Drive - a pre-resolved
    direct-download link (drive.usercontent.google.com/download?...&confirm=t&...) already bypasses
    the interstitial, so it downloads exactly like any other website URL (Model: dataset_tartanair.py).
    api_url is its own field, not shared with url_download_root, so a paginated-JSON-API dataset
    (Model: dataset_sesoko.yaml) reports api instead of collapsing into website."""
    labels: list[str] = []
    if cfg.get("hf_repo_id"):
        labels.append("hugging-face")

    if cfg.get("google_drive_link"):
        labels.append("google-drive")

    urls = [cfg.get("url_download_root"), cfg.get("url_download_sequences")]
    if any(urls):
        labels.append("website")

    if cfg.get("api_url"):
        labels.append("api")

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


def _expand_label(label_cell: str) -> list[str]:
    """Expand a README Label cell like 'openloris-d400/t265' or 'rover-picam/d435i/t265' - one
    README row covering several dataset_table.md labels that share a prefix - into the individual
    labels it covers. A cell with no '/' is already a single label, returned as-is."""
    parts = [p.strip() for p in label_cell.split("/")]
    if len(parts) == 1:
        return parts
    prefix = parts[0].rsplit("-", 1)[0]
    return [parts[0]] + [f"{prefix}-{p}" for p in parts[1:]]


def _readme_features(readme_path: Path) -> dict[str, str]:
    """Return {label: features_emoji_string}, parsed from every row of README.md's Datasets and
    Tools tables - both the active rows and the commented-out placeholder rows for datasets not
    yet added to the visible table (they already carry curated Features tags, no reason to leave
    them blank). Scoped to the region from the Datasets table header to the next '## ' section
    heading, so the differently-shaped Baselines table above it (no Features column) is never
    touched.

    The Datasets table itself may be either shape: a Markdown pipe-table (`| Datasets | ...`,
    still how the Tools table below it is written) or raw HTML (`<table><tr><th>Label</th>...`,
    used when a table needs a colspan divider row that Markdown's :---: syntax can't express - see
    the "Fix Datasets table dividers in README" commit). Both are scanned in the same pass so this
    keeps working regardless of which shape the Datasets table is in at any given time. The HTML
    table's column order is Label|Features|Summary|Modes|Camera Models (see
    generate_readme_datasets_table.py, which writes it) - Label leads, unlike the Tools Markdown
    table below it, which still has Features|Label."""
    text = readme_path.read_text(encoding="utf-8")
    markers = [text.index(m) for m in ("| Datasets ", "<th>Label</th>", "<th>Datasets</th>") if m in text]
    if not markers:
        raise ValueError(
            f"{readme_path}: could not find a Datasets table (expected a Markdown '| Datasets ' "
            "header or an HTML '<th>Label</th>'/'<th>Datasets</th>' header)"
        )
    start = min(markers)
    next_section = re.search(r"^## ", text[start:], re.MULTILINE)
    region = text[start:start + next_section.start()] if next_section else text[start:]

    mapping: dict[str, str] = {}
    for line in region.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            stripped = stripped[4:-3].strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3 or not cells[2].startswith("`"):
            continue  # header/separator rows, or anything not shaped like a data row
        features, label_cell = cells[1], cells[2].strip("`")
        for label in _expand_label(label_cell):
            mapping[label] = features

    for row in _HTML_TR_RE.findall(region):
        if "colspan" in row:
            continue  # category divider row (e.g. "🌊 Underwater datasets"), not a data row
        cells = _HTML_TD_RE.findall(row)
        if len(cells) < 3:
            continue  # header row (<th>, not <td>) or anything not shaped like a data row
        label_cell = _HTML_TAG_RE.sub("", cells[0]).strip()
        features = _HTML_TAG_RE.sub("", cells[1]).strip()
        for label in _expand_label(label_cell):
            mapping[label] = features

    return mapping


def load_dataset_entries(dataset_files_dir: Path, get_dataset_py: Path, readme_md: Path) -> list[dict]:
    """Return one raw (unformatted) entry per dataset_*.yaml - shared by the Markdown renderer here,
    generate_dataset_xlsx.py, and generate_readme_datasets_table.py. 'Raw' means list-valued fields
    (cam_models, modes, raw_formats, download_labels, issue_ids, authors, features_words) are
    returned as Python lists, not pre-joined/backtick-quoted display strings - each renderer applies
    its own formatting on top. Two distinct Features fields coexist deliberately: 'features' is the
    emoji string scraped from README.md (what generate_dataset_table.py's Markdown table still
    uses), 'features_words' is the word list read directly from this yaml's about.features (what
    generate_readme_datasets_table.py uses to regenerate README.md itself, so it isn't reading back
    the very table it's about to overwrite)."""
    name_to_class = _parse_dataset_name_to_class(get_dataset_py)
    class_blocks_cache: dict[str, dict[str, tuple[list[str], str]]] = {}
    imports_cache: dict[str, dict[str, str]] = {}
    readme_features = _readme_features(readme_md)

    entries: list[dict] = []
    for yaml_file in sorted(dataset_files_dir.glob("dataset_*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        dataset_name = str(cfg.get("dataset_name", yaml_file.stem.removeprefix("dataset_"))).strip()
        about = cfg.get("about", {})
        maintainer_cfg = cfg.get("vslamlab_maintainer", {})

        entries.append(
            {
                "dataset_name": dataset_name,
                "display_name": str(about.get("summary", "")).strip(),
                "homepage": str(about.get("homepage", "")).strip(),
                "features": readme_features.get(dataset_name, ""),
                "features_words": list(about.get("features", []) or []),
                "license": str(about.get("license", "")).strip(),
                "authors": about.get("authors", []) or [],
                "access": str(about.get("access", "")).strip(),
                "publication": str(about.get("publication", "")).strip(),
                "year": str(about.get("year", "")).strip(),
                "bibtex_key": str(about.get("bibtex_key", "")).strip(),
                "bibtex": str(about.get("bibtex", "")).strip(),
                "cam_models": list(cfg.get("cam_models", DEFAULT_CAM_MODELS)),
                "modes": list(cfg.get("modes", DEFAULT_MODES)),
                "raw_formats": list(cfg.get("raw_formats", DEFAULT_RAW_FORMATS)),
                "download_labels": _download_labels(cfg),
                "issue_ids": _download_issues_for_dataset(dataset_name, dataset_files_dir, name_to_class, class_blocks_cache, imports_cache),
                "maintainer": str(maintainer_cfg.get("name", "")).strip(),
                "assisted_by": str(maintainer_cfg.get("assisted_by", "")).strip(),
            }
        )
    return entries


def _render_markdown_table(entries: list[dict]) -> str:
    lines = [
        "| Datasets | Features | Label | Modes | Camera Models | Raw Format | License | Download | Download Issues | Maintainer | AI-Assisted |",
        "|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for entry in entries:
        display_name, homepage = entry["display_name"], entry["homepage"]
        dataset_link = f"[**{display_name}**]({homepage})" if display_name and homepage else display_name
        modes = " ".join(f"`{m}`" for m in entry["modes"])
        cam_models = " ".join(f"`{m}`" for m in entry["cam_models"])
        raw_formats = " ".join(f"`{r}`" for r in entry["raw_formats"])
        download = " ".join(f"`{d}`" for d in entry["download_labels"])
        issues = " ".join(f"`{i}`" for i in entry["issue_ids"])
        lines.append(
            f"| {dataset_link} | {entry['features']} | `{entry['dataset_name']}` | {modes} | {cam_models} | {raw_formats} | "
            f"{entry['license']} | {download} | {issues} | {entry['maintainer']} | {entry['assisted_by']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-files-dir", type=Path, default=DEFAULT_DATASET_FILES_DIR,
                         help="Directory containing dataset_*.yaml/.py files")
    parser.add_argument("--get-dataset-py", type=Path, default=DEFAULT_GET_DATASET_PY,
                         help="Path to get_dataset.py (used to map dataset name -> implementing class/file)")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_MD,
                         help="Path to README.md (used to look up each dataset's Features tags)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD,
                         help="Output .md file path")
    args = parser.parse_args()

    entries = load_dataset_entries(args.dataset_files_dir, args.get_dataset_py, args.readme)
    markdown = _render_markdown_table(entries)

    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(entries)} dataset rows to {args.output}")


if __name__ == "__main__":
    main()
