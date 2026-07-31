---
name: add-youtube-video
description: Add one new YouTube video as a sequence to VSLAM-LAB's youtube dataset (Datasets/dataset_files/dataset_youtube.yaml). Use when the user asks to add a YouTube video/sequence to the youtube dataset, or asks "how do I add a youtube video". Requires a video URL and a sequence name.
---

Usage: `/add-youtube-video <url> <sequence_name>` — e.g. `/add-youtube-video https://youtu.be/abc123 my-drone-clip`. Both are required; parse them from `$ARGUMENTS` (or from however the user phrased the request). If either is missing or ambiguous, stop and ask rather than guessing a sequence name or URL yourself.

**Hard constraint — file scope.** This skill only ever creates or modifies one file: `Datasets/dataset_files/dataset_youtube.yaml`. Nothing else (not `dataset_youtube.py`, not `dataset_videos.py`, not any other dataset's files, not `configs/`, not `README.md`) is in scope — if something else genuinely needs to change to support this specific video, stop and flag it to the user instead of editing it directly. This skill performs no git operations at all — it doesn't stage or commit anything; leave the yaml's changes uncommitted for the user to review and commit themselves.

1. **Gather the two required fields**, in this order — don't skip ahead:
   - `url_download_sequence` — the YouTube URL (`youtu.be/...` or `youtube.com/watch?v=...`). From the prompt/`$ARGUMENTS`; ask if missing.
   - `sequence_name` — the new sequence's slug. From the prompt/`$ARGUMENTS`; ask if missing. Check it doesn't already exist in `dataset_youtube.yaml`'s `sequence_names` — if it does, stop and ask the user how to proceed (a different name, or are they updating an existing sequence rather than adding a new one?).

2. **Always ask about `time_window` and `crop_settings` — even though both are optional, don't silently default them without checking first.** Every existing youtube sequence uses a non-default `time_window`, so silently falling back to "whole video" is very likely wrong for a real clip:
   - `time_window` — `[ti, tf]` in seconds, the clip's start/end within the source video. Ask what portion of the video to use. If the user says to use the whole video, don't add a `time_windows` entry for this sequence at all — an unlisted sequence already defaults to `[0, None]` (the whole video), see `_get_time_window` in `dataset_youtube.py`.
   - `crop_settings` — `[top, bottom, left, right]` in pixels, trimmed from each edge before `target_resolution` resizing (e.g. for a split-screen overlay or watermark). Ask whether this video needs cropping. If not, don't add a `crop_settings` entry — an unlisted sequence defaults to no crop, see `_get_crop`.

3. **Only add `calibration_parameters` if the user proactively supplies real calibration values** — `cam_model`, `focal_length: [fx, fy]`, `principal_point: [cx, cy]`, `distortion_coefficients: [k1, k2, p1, p2]` (only when `cam_model` isn't `unknown` — see `dataset_youtube.py`'s `create_calibration_yaml` for the field shape). Never ask for this proactively, never invent or guess values, and never copy another sequence's calibration onto this one just to fill the field. If nothing real is supplied, don't add a `calibration_parameters` entry at all — an unlisted sequence already correctly defaults to `cam_model: unknown` with zeroed focal_length/principal_point and no distortion fields. Writing invented or borrowed calibration as if it were real is exactly the `cam_model`/distortion-mismatch bug a previous cleanup pass on this file caught and fixed — don't reintroduce it here.

4. **Write the changes into `dataset_youtube.yaml`**, following the file's existing sparse-dict-keyed-by-`sequence_name` shape for each optional field (read the file itself for the exact current shape/formatting before editing):
   - Append `sequence_name: url_download_sequence` to `url_download_sequences`.
   - Append `sequence_name` to `sequence_names` (2-space indented list item, matching the file's existing entries).
   - If a real `time_window` was given (step 2), add `sequence_name: [ti, tf]` to `time_windows`.
   - If real `crop_settings` was given (step 2), add `sequence_name: [top, bottom, left, right]` to `crop_settings`.
   - If real `calibration_parameters` was given (step 3), add the full sub-dict under `sequence_name` to `calibration_parameters`.

5. **Verify.** Confirm the yaml still parses (`python3 -c "import yaml; yaml.safe_load(open('Datasets/dataset_files/dataset_youtube.yaml'))"`), then instantiate `YoutubeDataset()` under `pixi run -e vslamlab python` and confirm the new sequence appears in `sequence_names`, and that `_get_time_window`/`_get_crop`/`_get_calibration_parameters` each return what was just written for it (and the documented default for whatever wasn't provided). If network access allows, `pixi run download-sequence youtube <sequence_name>` is the real end-to-end check — but don't treat a network failure there as a bug in this skill's work; note it and move on if it's clearly a connectivity issue (see the `source_address`/force-IPv4 note in `dataset_youtube.py`'s `download_sequence_data`, added after exactly this kind of failure).

6. **Report back** what was added (the yaml diff) and which optional fields were included vs. skipped, so the user can review before committing themselves.
