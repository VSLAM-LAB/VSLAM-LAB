import csv
import os
import subprocess
from typing import Any

import gdown
import numpy as np
import pandas as pd
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

ROSBAG_NAME = "rosbag.db3"
IMU_TOPIC = "/imu/data_raw"
IMAGE_TOPIC = "/image_raw/compressed"
CALIBRATION_FILE = "kalibr_imucam_chain.yaml"


class HILTI2026_dataset(DatasetVSLAMLab):
    """HILTI 2026 dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, dataset_name: str = "hilti2026") -> None:
        super().__init__(dataset_name)

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.url_download_root: str = cfg["url_download_root"]

        # Sequence nicknames
        self.sequence_nicknames = [s.split("_", 1)[0] for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        # Download calibration file
        calibration_file = self.dataset_path / CALIBRATION_FILE
        if not calibration_file.exists():
            folder_id = "1kYxgaCAtsVLe1B1MGsc2kR6RnByloHUV"

            file_url = "https://drive.google.com/file/d/1MX_C9kphWyghcQKNN-x70SquzepN588k/view?usp=sharing"
            gdown.download(
                url=file_url,
                output=str(self.dataset_path / "kalibr_imucam_chain.yaml"),
                quiet=False,
                use_cookies=False,
                resume=True,
                fuzzy=True,
            )

        # Download groundtruth file
        gt_url = self._get_gt_url(sequence_name)
        gt_file = sequence_path / "groundtruth.txt"
        if gt_url is not None and not gt_file.exists():
            gdown.download(
                url=gt_url,
                output=str(gt_file),
                quiet=False,
                use_cookies=False,
                resume=True,
                fuzzy=True,
            )

        # Download rosbag
        rosbag = sequence_path / ROSBAG_NAME
        if rosbag.exists():
            return
        folder_id = self._get_folder_id(sequence_name)
        folder_url = f"{self.url_download_root}/{folder_id}"
        gdown.download_folder(
            url=folder_url,
            output=str(sequence_path),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
            resume=True,
        )

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag = sequence_path / ROSBAG_NAME
        for cam in ["0", "1"]:
            image_topic = f"/cam{cam}{IMAGE_TOPIC}"
            rgb_path = sequence_path / f"rgb_{cam}"
            if rgb_path.exists():
                continue
            rgb_path.mkdir(parents=True, exist_ok=True)

            inputs = f"--rosbag_path {rosbag} --sequence_path {sequence_path} --image_topic {image_topic} --cam {cam}"
            command = f"pixi run -e ros2 extract-ros2bag-frames {inputs}"
            subprocess.run(command, shell=True)

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        rosbag = sequence_path / ROSBAG_NAME
        imu_csv = self.imu_csv_path(sequence_name)
        if imu_csv.exists():
            return

        inputs = f"--rosbag_path {rosbag} --sequence_path {sequence_path} --imu_topic {IMU_TOPIC}"
        command = f"pixi run -e ros2 extract-ros2bag-imu {inputs}"
        subprocess.run(command, shell=True)

        rgb_csv = self.rgb_csv_path(sequence_name)
        imu_csv = self.imu_csv_path(sequence_name)
        rgb = pd.read_csv(rgb_csv)
        imu = pd.read_csv(imu_csv)

        rgb_0_ts_col = "ts_rgb_0 (ns)"
        rgb_1_ts_col = "ts_rgb_1 (ns)"
        imu_ts_col = "ts (ns)"

        imu[imu_ts_col] = imu[imu_ts_col].astype("int64")
        rgb[rgb_0_ts_col] = rgb[rgb_0_ts_col].astype("int64")
        rgb[rgb_1_ts_col] = rgb[rgb_1_ts_col].astype("int64")

        rgb.to_csv(rgb_csv, index=False)
        imu.to_csv(imu_csv, index=False)

    def create_calibration_yaml(self, sequence_name: str) -> None:
        calibration_yaml = self.dataset_path / CALIBRATION_FILE

        with calibration_yaml.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if lines and lines[0].lstrip().startswith("%YAML:1.0"):
            lines = lines[1:]

        # Add standard YAML header
        text = "%YAML 1.2\n---\n" + "".join(lines)
        data = yaml.safe_load(text)

        cam0 = data["cam0"]
        cam1 = data["cam1"]
        intrinsics0 = cam0["intrinsics"]
        intrinsics1 = cam1["intrinsics"]
        T_cam0_imu = np.array(cam0["T_cam_imu"], dtype=float).reshape(4, 4)
        T_cam1_imu = np.array(cam1["T_cam_imu"], dtype=float).reshape(4, 4)

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": intrinsics0[0:2],
            "principal_point": intrinsics0[2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam0["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam0_imu),
        }

        rgb1: dict[str, Any] = {
            "cam_name": "rgb_1",
            "cam_type": "gray",
            "cam_model": "pinhole",
            "focal_length": intrinsics1[0:2],
            "principal_point": intrinsics1[2:4],
            "distortion_type": "equid4",
            "distortion_coefficients": cam1["distortion_coeffs"],
            "fps": self.rgb_hz,
            "T_BS": np.linalg.inv(T_cam1_imu),
        }

        imu: dict[str, Any] = {
            "imu_name": "imu_0",
            "a_max": 176.0,
            "g_max": 7.8,
            "sigma_g_c": 0.00047032046,
            "sigma_a_c": 0.00285891188,
            "sigma_bg": 0.0,
            "sigma_ba": 0.0,
            "sigma_gw_c": 0.00036433633,
            "sigma_aw_c": 0.00130039805,
            "g": 9.81007,
            "g0": [0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0],
            "s_a": [1.0, 1.0, 1.0],
            "fps": 1000.0,
            "T_BS": np.array(np.eye(4)).reshape((4, 4)),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0, rgb1], imu=[imu])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.sequence_path(sequence_name)
        gt_txt = sequence_path / "groundtruth.txt"
        groundtruth_csv = self.groundtruth_csv_path(sequence_name)
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        if gt_txt.exists():
            with (
                open(gt_txt, "r", encoding="utf-8") as fin,
                open(tmp, "w", encoding="utf-8", newline="") as fout,
            ):
                # Skip first lines (header/comments), then write CSV header + values
                lines = fin.readlines()
                data_lines = [ln.strip() for ln in lines[1:] if ln.strip() and not ln.lstrip().startswith("#")]
                fout.write("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n")
                for line in data_lines:
                    s = line.strip()
                    parts = s.split()
                    ts_ns = int(float(parts[0]) * 1e9)
                    new_line = f"{ts_ns}," + ",".join(parts[1:]) + "\n"
                    fout.write(new_line)
        else:
            with open(tmp, "w", newline="", encoding="utf-8") as fout:
                w = csv.writer(fout)
                w.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

        tmp.replace(groundtruth_csv)
        tmp.unlink(missing_ok=True)

    def _get_folder_id(self, sequence_name):
        if sequence_name == "floor_1_2025_05_05":
            return "1ZiQHwcoF6NGo8lAW9kmqMgqMYQuQcxKZ"
        if sequence_name == "floor_1_2025_07_07":
            return "1yT3ISWjMLNPoF0LaH7iepgaJ_XKN-Npd"
        if sequence_name == "floor_1_2025_12_02":
            return "1OQ6t5SuEQ0txcAKS2hXJiZc7N3WTuXQi"
        if sequence_name == "floor_2_2025_05_05":
            return "1JWYdtx7sU8YEGfNMQod_UcCc12nMAOWC"
        if sequence_name == "floor_2_2025_10_28_run_1":
            return "1xZbqsjl_ElmD_LHGJLeFCLyZGUOisU_E"
        if sequence_name == "floor_2_2025_10_28_run_2":
            return "1ypN26UYznPYH_eag2cLvu48nzVQAc_ya"
        if sequence_name == "floor_2_2025_12_02":
            return "1kNC02bBxTzqVkKXCGsZcV86uKJEloV53"
        if sequence_name == "floor_2_2025_12_03":
            return "1OfyyCQS5fHSf7DTcMlMqyqZhrr_rJ2e4"
        if sequence_name == "floor_3_2025_05_19":
            return "1S529WppSa7L5u9glioXuKuG6_hNJT99A"
        if sequence_name == "floor_3_2025_12_02":
            return "1gZyCZUcFK2DhuXD-yM6Zz5CqjjpC2Aqx"
        if sequence_name == "floor_4_2025_05_19":
            return "1kCBMvwRPiwDrWPi_z_B8CEZolDrfZvvy"
        if sequence_name == "floor_4_2025_12_02":
            return "1Org1UoNa6h_OFNcmoom3M1vB6fwHKQs4"
        if sequence_name == "floor_5_2025_12_02":
            return "1AqnJ0VwLylxCL5fJpKAidvh21rhaPqlL"
        if sequence_name == "floor_6_2025_06_18":
            return "1k8FfHptZlNAf5aawmK-NJBSEyrZy9zwc"
        if sequence_name == "floor_6_2025_07_07":
            return "1rygrtU47OSFGPBSeyPSmWH9zrh-e3G08"
        if sequence_name == "floor_6_2025_12_02_run_1":
            return "10ngMYxUxKuWrBQzJIorKPyLGvofzpTFX"
        if sequence_name == "floor_6_2025_12_02_run_2":
            return "1CkU7--2PP0k58m1RRBD5PQygKGAiNQTn"
        if sequence_name == "floor_7_2025_12_02_run_1":
            return "1PB-ILVt4pyoFrAjgvzfSosMepdv8af9V"
        if sequence_name == "floor_7_2025_12_02_run_2":
            return "1aCebuVIX9-buMyb_ji4sCvwNo1rFhoLq"
        if sequence_name == "floor_7_2025_12_03":
            return "1MfhlsR5rB_1fVccTFGll3xozAsHgQamw"
        if sequence_name == "floor_EG_2025_10_16":
            return "1JOoZrrLmdpyOfvFsJANgYI66WZ8GGOoJ"
        if sequence_name == "floor_EG_2025_12_02_run_1":
            return "1S2c3u8C1vc9YgXtcIuXZuPLmMzVPHr2r"
        if sequence_name == "floor_EG_2025_12_02_run_2":
            return "159sCP0DbkOiNF8n83BhSFDUMW-jzaZFX"
        if sequence_name == "floor_UG1_2025_05_19":
            return "1E5syGjNmzhQdvJ0F-92nkbWEsVfnxxm9"
        if sequence_name == "floor_UG1_2025_06_18":
            return "1ZWrFeP0BPnXCOsctbRPc4C6oUfEqYYUZ"
        if sequence_name == "floor_UG1_2025_10_16":
            return "1dSqJgKZZlhKdepl2_y1vxuf6_TbRMFHB"
        if sequence_name == "floor_UG1_2025_12_02_run_1":
            return "14U2XwVP7z1mkeEgwpXEmrr1FTTojhFSt"
        if sequence_name == "floor_UG1_2025_12_02_run_2":
            return "1EHU2enVTHdfZ3I7SSahmqs-ad34odDZu"
        if sequence_name == "floor_UG1_2025_12_03":
            return "1aUwNFRRs3u6KQgHxmQQGLbsKe_9xlvSM"
        if sequence_name == "floor_UG2_2025_12_02":
            return "1j_Uzae7lpmStVGUFZYV_ghQC_R5jF6nA"

    def _get_gt_url(self, sequence_name):
        if sequence_name == "floor_1_2025_05_05":
            return "https://drive.google.com/file/d/1vfSWB1MKa2OGWZXxt6TkmI_kvvf5CupX/view?usp=sharing"
        if sequence_name == "floor_2_2025_05_05":
            return "https://drive.google.com/file/d/1Gr7WPdfqZ5xP-lC8rSRnQngHo-UuVgcL/view?usp=sharing"
        if sequence_name == "floor_2_2025_10_28_run_1":
            return "https://drive.google.com/file/d/1lTwd7udC34k_WaJa-_EE1mkcHP_V01NS/view?usp=sharing"
        if sequence_name == "floor_2_2025_10_28_run_2":
            return "https://drive.google.com/file/d/12XG4WFYg3HgeSeECcWRzCXyljiYEsbSH/view?usp=sharing"
        if sequence_name == "floor_UG1_2025_10_16":
            return "https://drive.google.com/file/d/1RZC-DYTN0-rtKNQCxhLjtrcs_CdhP88s/view?usp=sharing"
        return None
