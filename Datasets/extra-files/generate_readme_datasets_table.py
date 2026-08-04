#!/usr/bin/env python3
"""
Module: generate_readme_datasets_table.py
Description: Regenerates README.md's own Datasets <table>...</table> block (the
             one with category-divider rows like "💻 Synthetic datasets") from
             each dataset_*.yaml's about.features/summary/homepage/modes/
             cam_models, replacing that block in place. This inverts the
             direction generate_dataset_table.py works in: that script reads
             Features *out of* README.md; this one writes README.md's table
             *from* the yaml (about.features), which is now the source of
             truth for a dataset's Real|Synthetic / environment / platform
             tags. Uses load_dataset_entries' features_words field (not the
             README-scraped features string) so it never reads back the very
             table it's about to overwrite.

             Scope: every dataset_*.yaml with about.features set, except
             _SKIP (videos, youtube: no fixed environment to tag; strayscanner:
             lives in README's separate Tools table, not this one).

             Section assignment: a dataset can carry several "special" tags at
             once (e.g. Underwater and Robot); _SECTION_PRIORITY picks one,
             checked in this order: Synthetic, Underwater, Vehicle, UAV,
             Intracorporeal, Planetary-analog Terrain, Construction Site,
             Robot, else the default (no-divider) section. This ordering and
             the display order in _SECTION_ORDER were chosen to match every
             existing placement in README.md as of 2026-08-04.

             Columns are Label | Features | Summary | Modes | Camera Models -
             Label leads (short, fixed-width `code` slug) rather than Summary,
             which is about.summary in full (matching generate_dataset_table.
             py's own convention for that field) truncated to _MAX_SUMMARY_LEN
             chars with '...' (see _truncate) since it's the one variable-
             length, often-long field in the row. There's no "about.title"
             field to hold a short curated display name instead, and no
             reliable rule to derive one from summary (confirmed: truncating
             at the first " - " recovers some prior hand-curated titles, like
             ariel/hilti2022, but is wrong for others, like msd, and doesn't
             apply at all to titles that were independently curated, like
             eiffel-tower/kitti/euroc/rgbdtum - hence truncating by length
             instead, not by content). Multi-source citation cells (e.g.
             Replica's "- iMAP" second link) also collapse to one link, since
             a dataset's about: block has only one homepage.
Author: Alejandro Fontan Villacampa
Version: 1.2
Created: 2026-08-04
Updated: 2026-08-04
License: GPLv3
List of Known Bugs: None
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_dataset_table import (
    DEFAULT_DATASET_FILES_DIR,
    DEFAULT_GET_DATASET_PY,
    DEFAULT_README_MD,
    load_dataset_entries,
)

# Not members of README's Datasets table: videos/youtube have no fixed environment to tag (see
# generate_dataset_table.py's about.features field), strayscanner lives in the separate Tools table.
_SKIP = {"videos", "youtube", "strayscanner"}

# Mirrors README.md's own legend (right below the Datasets table) - the inverse of the mapping used
# to backfill about.features from README in the first place.
_WORD_EMOJI = {
    "Real": "📸",
    "Synthetic": "💻",
    "Indoor": "🏠",
    "Outdoor": "🏞️",
    "Underwater": "🌊",
    "Intracorporeal": "🫀",
    "Handheld": "🤳",
    "Headmounted": "🥽",
    "Vehicle": "🚗",
    "UAV": "🚁",
    "Robot": "🤖",
    "Construction Site": "🏗️",
    "Planetary-analog Terrain": "🪐",
}

# Checked in order; the first tag present decides a dataset's section. Not the same as the display
# order below (e.g. Underwater is checked before Vehicle/UAV/.../Robot, but displayed after them) -
# see this module's docstring for how this order was derived.
_SECTION_PRIORITY = [
    "Synthetic", "Underwater", "Vehicle", "UAV", "Intracorporeal",
    "Planetary-analog Terrain", "Construction Site", "Robot",
]

# (section tag or None for the default section, divider row text). Display order top-to-bottom in
# the generated table. The default section (no special tag matched in _SECTION_PRIORITY) still gets
# its own divider, same as every other section.
_SECTION_ORDER: list[tuple[str | None, str]] = [
    (None, "🏠 Indoor datasets"),
    ("Synthetic", "💻 Synthetic datasets"),
    ("Vehicle", "🚗 Vehicle datasets"),
    ("UAV", "🚁 UAV datasets"),
    ("Robot", "🤖 Robot datasets"),
    ("Construction Site", "🏗️ Construction Site datasets"),
    ("Planetary-analog Terrain", "🪐 Planetary-analog Terrain datasets"),
    ("Underwater", "🌊 Underwater datasets"),
    ("Intracorporeal", "🫀 Intracorporeal datasets"),
]


def _assign_section(features_words: list[str]) -> str | None:
    tags = set(features_words)
    for section in _SECTION_PRIORITY:
        if section in tags:
            return section
    return None


def _compact_modes(modes: list[str]) -> list[str]:
    """Collapse an X/X-vi pair into one 'X(-vi)' cell, matching README's existing convention
    (e.g. dataset_euroc.yaml's ['mono','stereo','mono-vi','stereo-vi'] -> ['mono(-vi)',
    'stereo(-vi)']) instead of listing all four modes separately."""
    modes_set = set(modes)
    ordered_bases: list[str] = []
    for m in modes:
        base = m[:-3] if m.endswith("-vi") else m
        if base not in ordered_bases:
            ordered_bases.append(base)
    result = []
    for base in ordered_bases:
        vi = f"{base}-vi"
        if base in modes_set and vi in modes_set:
            result.append(f"{base}(-vi)")
        elif base in modes_set:
            result.append(base)
        else:
            result.append(vi)  # -vi present without its base (not expected per the yaml convention)
    return result


_MAX_SUMMARY_LEN = 80


def _truncate(text: str, max_len: int = _MAX_SUMMARY_LEN) -> str:
    """Cut text to at most max_len chars, breaking at the last space so words aren't split, and
    append '...'. No-op (returned unchanged) if text already fits."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _render_row(entry: dict) -> str:
    emoji = "".join(_WORD_EMOJI.get(w, "") for w in entry["features_words"])
    modes = " ".join(f"<code>{m}</code>" for m in _compact_modes(entry["modes"]))
    cam_models = " ".join(f"<code>{c}</code>" for c in entry["cam_models"])
    summary = _truncate(entry["display_name"])
    return (
        f'<tr><td><code>{entry["dataset_name"]}</code></td><td>{emoji}</td>'
        f'<td><a href="{entry["homepage"]}"><strong>{summary}</strong></a></td>'
        f"<td>{modes}</td><td>{cam_models}</td></tr>\n"
    )


def build_table(entries: list[dict]) -> str:
    groups: dict[str | None, list[dict]] = {}
    for entry in entries:
        if entry["dataset_name"] in _SKIP or not entry["features_words"]:
            continue
        groups.setdefault(_assign_section(entry["features_words"]), []).append(entry)

    body = []
    for section, divider_text in _SECTION_ORDER:
        members = sorted(groups.get(section, []), key=lambda e: e["dataset_name"])
        if not members:
            continue
        if divider_text is not None:
            body.append(f'<tr><td colspan="5">{divider_text}<hr></td></tr>\n')
        body.extend(_render_row(e) for e in members)

    return (
        "<table>\n<thead>\n"
        "<tr><th>Label</th><th>Features</th><th>Summary</th><th>Modes</th><th>Camera Models</th></tr>\n"
        "</thead>\n<tbody>\n" + "".join(body) + "</tbody>\n</table>\n"
    )


def _replace_readme_table(readme_path: Path, new_table_html: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    # Matches either this script's current header (Label first) or its pre-reorder one (Datasets/
    # summary first, from before this table's columns were reordered/shortened) - so a README still
    # holding output from an older version of this script can still be located and replaced.
    markers = [text.index(m) for m in ("<tr><th>Label</th>", "<tr><th>Datasets</th>") if m in text]
    if not markers:
        raise ValueError(f"{readme_path}: could not find the Datasets table (no recognized header row)")
    anchor = min(markers)
    table_start = text.rindex("<table>", 0, anchor)
    table_end = text.index("</table>", anchor) + len("</table>")
    readme_path.write_text(text[:table_start] + new_table_html.rstrip("\n") + text[table_end:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-files-dir", type=Path, default=DEFAULT_DATASET_FILES_DIR,
                         help="Directory containing dataset_*.yaml/.py files")
    parser.add_argument("--get-dataset-py", type=Path, default=DEFAULT_GET_DATASET_PY,
                         help="Path to get_dataset.py (used to map dataset name -> implementing class/file)")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_MD,
                         help="Path to README.md - both read (to locate/replace its Datasets table) and written")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated table instead of writing README.md")
    args = parser.parse_args()

    entries = load_dataset_entries(args.dataset_files_dir, args.get_dataset_py, args.readme)
    table_html = build_table(entries)

    if args.dry_run:
        print(table_html)
        return

    _replace_readme_table(args.readme, table_html)
    print(f"Regenerated Datasets table in {args.readme}")


if __name__ == "__main__":
    main()
