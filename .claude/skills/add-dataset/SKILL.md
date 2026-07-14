---
name: add-dataset
description: Add a new dataset to VSLAM-LAB. Use when the user asks to add/integrate a new benchmark dataset, wire up a dataset for download/evaluation, or asks "how do I add a dataset".
---

Adding a dataset means creating a `DatasetVSLAMLab` subclass plus a settings YAML, then registering it in `Datasets/get_dataset.py`.

1. **Copy the template**: start from `Datasets/extra-files/dataset_template.py` and `Datasets/extra-files/dataset_template.yaml`. Save as `Datasets/dataset_files/dataset_<name>.py` and `Datasets/dataset_files/dataset_<name>.yaml`.

2. **Implement the class**: subclass `DatasetVSLAMLab` (`Datasets/DatasetVSLAMLab.py`), name it `<NAME>_dataset`, and implement the required hooks — study a close analogue in `Datasets/dataset_files/` for the same sensor modality (monocular/RGBD/stereo/stereo-VI, see the section comments in `get_dataset.py`) rather than writing from scratch:
   - `download_sequence_data(sequence_name)` — download and decompress raw sequence data.
   - `create_rgb_folder(sequence_name)` — populate the standardized `rgb/` image folder.
   - `create_rgb_txt(sequence_name)` — write the `rgb.txt` frame-timestamp index.
   - `create_calibration_yaml(sequence_name)` — write camera intrinsics/calibration.
   - `create_groundtruth_txt(sequence_name)` — write the groundtruth trajectory file.
   - `remove_unused_files(sequence_name)` — clean up intermediate/raw files after processing.

3. **Fill in the YAML**: `dataset_name`, `url_download_root`, `sequence_names` (list of sequence identifiers), plus any modality-specific fields (e.g. `depth_factor` for RGBD datasets — check a sibling dataset's YAML for the fields its `.py` reads).

4. **Register it** in `Datasets/get_dataset.py`:
   - Add `from Datasets.dataset_files.dataset_<name> import <NAME>_dataset` under the correct modality section comment (Monocular / RGBD / Stereo / Stereo-VI / Development).
   - Add an entry to the `switcher` dict in `get_dataset()`: `"<name>": lambda: <NAME>_dataset(benchmark_path),`.

5. **Verify**: create or reuse a `configs/test_exp_<name>.yaml` (see existing `test_exp_*.yaml` files for the format — `Config:` block listing `<name>:<sequence>` pairs, `NumRuns`, `Parameters`, `Module`) and run `pixi run vslamlab configs/test_exp_<name>.yaml` to confirm the dataset downloads and processes correctly end-to-end.

Full reference docs live on the project's GitHub Wiki if more detail is needed.
