import os
import yaml
import pandas as pd
from tqdm import tqdm

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.dataset_utilities import load_rig_yaml, build_multicam_rgb_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class HMND_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name='hmnd'):
        super().__init__(dataset_name, benchmark_path)

        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name: str) -> None:
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Create convenience symlinks rgb_0, rgb_1 to selected rig cameras (if available)
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rig_yaml = os.path.join(self.dataset_path, 'rig.yaml')
        if not os.path.exists(rig_yaml):
            return
        rig = load_rig_yaml(rig_yaml)
        selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])
        cid2cam = {c['id']: c for c in rig['cameras']}

        os.makedirs(sequence_path, exist_ok=True)

        # Create symlinks for the first two cameras if present
        for idx, cid in enumerate(sorted(selected)[:2]):
            cam = cid2cam[cid]
            target_dir = os.path.join(self.dataset_path, cam['data_dir'])
            link_dir = os.path.join(sequence_path, f'rgb_{idx}')
            try:
                if os.path.islink(link_dir) or os.path.exists(link_dir):
                    continue
                os.symlink(target_dir, link_dir)
            except FileExistsError:
                pass

        # Also mirror the rig data_dir structure inside the sequence so paths in csv resolve
        for cid in sorted(selected):
            cam = cid2cam[cid]
            target_dir = os.path.join(self.dataset_path, cam['data_dir'])
            link_dir = os.path.join(sequence_path, cam['data_dir'])
            parent = os.path.dirname(link_dir)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            try:
                if not (os.path.islink(link_dir) or os.path.exists(link_dir)):
                    os.symlink(target_dir, link_dir)
            except FileExistsError:
                pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rig_yaml = os.path.join(self.dataset_path, 'rig.yaml')
        if not os.path.exists(rig_yaml):
            print(f"{SCRIPT_LABEL}No rig.yaml found at {rig_yaml}. Skipping rgb.csv generation.")
            return

        rig = load_rig_yaml(rig_yaml)
        selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])

        rows = build_multicam_rgb_csv_rows(self.dataset_path, rig, selected)
        if len(rows) == 0:
            print(f"{SCRIPT_LABEL}No synchronized pairs found for selected cameras {selected}")
            return

        # Build columns deterministically sorted by camera id
        cols = []
        for cid in sorted(selected):
            cols.extend([f"ts_rgb{cid} (s)", f"path_rgb{cid}"])

        df = pd.DataFrame(rows)
        df = df[cols]
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        os.makedirs(sequence_path, exist_ok=True)
        df.to_csv(rgb_csv, index=False)

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rig_yaml = os.path.join(self.dataset_path, 'rig.yaml')
        if not os.path.exists(rig_yaml):
            print(f"{SCRIPT_LABEL}No rig.yaml found at {rig_yaml}. Skipping imu.csv link.")
            return
        rig = load_rig_yaml(rig_yaml)
        imu_src = os.path.join(self.dataset_path, rig['imu']['csv'])
        imu_dst = os.path.join(sequence_path, 'imu.csv')
        os.makedirs(sequence_path, exist_ok=True)
        # Create a real imu.csv aligned to camera time if large offset detected
        import pandas as pd
        import numpy as np
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        try:
            df_rgb = pd.read_csv(rgb_csv)
            # pick first available camera ts column
            ts_cols = [c for c in df_rgb.columns if c.startswith('ts_rgb')]
            r0 = float(df_rgb[ts_cols[0]].iloc[0]) if len(ts_cols) else None
        except Exception:
            r0 = None

        try:
            df_imu = pd.read_csv(imu_src)
            i0 = float(df_imu.iloc[0,0])
        except Exception:
            i0 = None

        # If both available and offset is too large, shift IMU timestamps.
        # Make IMU start slightly BEFORE RGB (by 50 ms) to satisfy OKVIS startup.
        if r0 is not None and i0 is not None and abs(i0 - r0) > 1.0:
            delta = i0 - r0  # positive if IMU is newer (lags behind images)
            # We want imu_ts_shifted = i0 - (delta + 0.05) = r0 - 0.05
            shift = delta + 0.05
            df_imu_shift = df_imu.copy()
            df_imu_shift.iloc[:,0] = df_imu_shift.iloc[:,0].astype(np.float64) - shift
            df_imu_shift.to_csv(imu_dst, index=False)
        else:
            # fallback to symlink
            if os.path.abspath(imu_src) != os.path.abspath(imu_dst):
                try:
                    if os.path.islink(imu_dst) or os.path.exists(imu_dst):
                        os.remove(imu_dst)
                except FileNotFoundError:
                    pass
                try:
                    os.symlink(imu_src, imu_dst)
                except FileExistsError:
                    pass

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rig_yaml = os.path.join(self.dataset_path, 'rig.yaml')
        if not os.path.exists(rig_yaml):
            print(f"{SCRIPT_LABEL}No rig.yaml found at {rig_yaml}. Skipping calibration.yaml generation.")
            return

        rig = load_rig_yaml(rig_yaml)
        selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])
        cid2cam = {c['id']: c for c in rig['cameras']}

        # Build camera parameter list in selected id order
        cameras = []
        for cid in sorted(selected):
            cam = cid2cam[cid]
            cam_params = {
                'model': cam['model'],
                'fx': cam['intrinsics'][0], 'fy': cam['intrinsics'][1],
                'cx': cam['intrinsics'][2], 'cy': cam['intrinsics'][3],
                'k1': cam['distortion_coeffs'][0], 'k2': cam['distortion_coeffs'][1],
                'p1': cam['distortion_coeffs'][2], 'p2': cam['distortion_coeffs'][3],
                'k3': cam['distortion_coeffs'][4] if len(cam['distortion_coeffs']) > 4 else 0.0,
                'w': cam['resolution'][0], 'h': cam['resolution'][1],
                'distortion_model': 'radtan'
            }
            cameras.append(cam_params)

        # Build IMU transforms mapping to Camera<i> index order
        imu_transforms = {}
        for idx, cid in enumerate(sorted(selected)):
            cam = cid2cam[cid]
            imu_transforms[f"T_b_c{idx}"] = cam['T_b_c']

        # IMU noise
        imu = {
            'gyro_noise': rig['imu']['noise'].get('gyroscope_noise_density', 0.0),
            'accel_noise': rig['imu']['noise'].get('accelerometer_noise_density', 0.0),
            'gyro_bias': rig['imu']['noise'].get('gyroscope_random_walk', 0.0),
            'accel_bias': rig['imu']['noise'].get('accelerometer_random_walk', 0.0),
            'frequency': rig['imu'].get('frequency', self.rgb_hz)
        }

        # Write standard VSLAM-LAB style
        self.write_calibration_yaml(sequence_name=sequence_name, cameras=cameras, imu=imu, imu_transforms=imu_transforms)

        # Also append camchain-style blocks expected by some OKVIS2 builds
        calib_path = os.path.join(sequence_path, 'calibration.yaml')
        with open(calib_path, 'a') as f:
            f.write("\n")
            for idx, cid in enumerate(sorted(selected)):
                cam = cid2cam[cid]
                fx, fy, cx, cy = cam['intrinsics']
                k1, k2, p1, p2 = cam['distortion_coeffs'][:4]
                w, h = cam['resolution']
                # Build T_BS array
                T = cam['T_b_c']
                if len(T) == 16:
                    flat = T
                else:
                    flat = [item for row in T for item in row]
                f.write(f"cam{idx}:\n")
                f.write(f"  T_BS: [{', '.join(map(str, flat))}]\n")
                f.write(f"  image_dimension: [{w}, {h}]\n")
                f.write(f"  distortion_coefficients: [{k1}, {k2}, {p1}, {p2}]\n")
                f.write(f"  focal_length: [{fx}, {fy}]\n")
                f.write(f"  principal_point: [{cx}, {cy}]\n")
                f.write(f"  distortion_type: radialtangential\n\n")


    def create_groundtruth_csv(self, sequence_name: str) -> None:
        return

    def remove_unused_files(self, sequence_name: str) -> None:
        return



