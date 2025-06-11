"""
Module: VSLAM-LAB - Datasets - DatasetVSLAMLab.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-12
- Updated: 2024-07-12
- License: GPLv3 License

DatasetVSLAMLab: A class to handle Visual SLAM dataset-related operations.

"""

import os, sys, cv2, yaml
from utilities import ws, check_sequence_integrity
from path_constants import VSLAM_LAB_DIR, VSLAM_LAB_EVALUATION_FOLDER


SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class DatasetVSLAMLab:

    def __init__(self, dataset_name, benchmark_path):

        self.dataset_name = dataset_name
        self.dataset_color = "\033[38;2;255;165;0m"
        self.dataset_label = f"{self.dataset_color}{dataset_name}\033[0m"
        self.dataset_folder = dataset_name.upper()
        self.benchmark_path = benchmark_path
        self.dataset_path = os.path.join(self.benchmark_path, self.dataset_folder)

        self.yaml_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', 'dataset_' + self.dataset_name + '.yaml')

        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        self.sequence_names = data['sequence_names']
        self.rgb_hz = data['rgb_hz']
        self.sequence_nicknames = []
        self.modes = data.get('modes', 'mono-vi')

    ####################################################################################################################
    # Download methods
    def download_sequence(self, sequence_name):

        # Check if sequence is already available
        sequence_availability = self.check_sequence_availability(sequence_name)
        if sequence_availability == "available":
            #print(f"{SCRIPT_LABEL}Sequence {self.dataset_color}{sequence_name}:\033[92m downloaded\033[0m")
            return
        if sequence_availability == "corrupted":
            print(f"{ws(8)}Some files in sequence {sequence_name} are corrupted.")
            print(f"{ws(8)}Removing and downloading again sequence {sequence_name} ")
            print(f"{ws(8)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY")
            sys.exit(1)

        # Download process
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)

        self.download_process(sequence_name)

    def download_process(self, sequence_name):
        msg = f"Downloading sequence {self.dataset_color}{sequence_name}\033[0m from dataset {self.dataset_color}{self.dataset_name}\033[0m ..."
        print(SCRIPT_LABEL + msg)
        self.download_sequence_data(sequence_name)
        self.create_rgb_folder(sequence_name)
        self.create_rgb_txt(sequence_name)
        self.create_imu_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)
        self.create_groundtruth_txt(sequence_name)
        self.remove_unused_files(sequence_name)

    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        return
    
    def create_imu_csv(self, sequence_name):
        return

    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_txt(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        return

    def get_download_issues(self, sequence_names):
        return {}

    def write_calibration_yaml(self, sequence_name, camera0=None, camera1=None, imu=None, rgbd=None):
    #Write calibration YAML file with flexible sensor configuration.
    #Args:
    #    sequence_name: Name of the sequence
    #    camera0: Dict with keys: model, fx, fy, cx, cy, k1, k2, p1, p2, k3
    #    camera1: Dict with keys: model, fx, fy, cx, cy, k1, k2, p1, p2, k3 (for stereo)
    #    imu: Dict with keys: transform, accel_noise, gyro_noise, accel_bias, gyro_bias, frequency
    #    rgbd: Dict with keys: depth_factor, depth_scale (optional)
    
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        
        yaml_content_lines = ["%YAML:1.0", ""]
        
        # Camera0 parameters (required)
        if camera0:
            yaml_content_lines.extend(["", "# Camera calibration and distortion parameters"])
            yaml_content_lines.extend(self._get_camera_yaml_section(camera0, sequence_name, "Camera"))
        
        # Camera1 parameters (for stereo)
        if camera1:
            yaml_content_lines.extend(["", "# Camera1 calibration and distortion parameters"])
            yaml_content_lines.extend(self._get_camera_yaml_section(camera1, sequence_name, "Camera1"))
        
        # IMU parameters
        if imu:
            yaml_content_lines.extend(["", "# IMU parameters"])
            yaml_content_lines.extend(self._get_imu_yaml_section(imu))
        
        # RGBD parameters
        if rgbd:
            yaml_content_lines.extend(["", "#Depth map parameters"])
            yaml_content_lines.extend(self._get_rgbd_yaml_section(rgbd))
        
        with open(calibration_yaml, 'w') as file:
            for line in yaml_content_lines:
                file.write(f"{line}\n")

    def _get_camera_yaml_section(self, camera_params, sequence_name, prefix="Camera"):
        """Generate YAML lines for camera parameters."""
        # Get image dimensions
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        
        # Ensure rgb_path exists and has images before trying to read
        if not os.path.exists(rgb_path) or not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(rgb_path)):
            print(f"{SCRIPT_LABEL}Warning: RGB path {rgb_path} not found or no images present. Cannot determine image dimensions.")
            h, w = 0, 0 # Default or raise error
        else:
            rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not rgb_files:
                print(f"{SCRIPT_LABEL}Warning: No image files found in {rgb_path}. Cannot determine image dimensions.")
                h, w = 0,0 # Default or raise error
            else:
                image_0_path = os.path.join(rgb_path, rgb_files[0])
                image_0 = cv2.imread(image_0_path)
                if image_0 is None:
                    print(f"{SCRIPT_LABEL}Warning: Could not read image {image_0_path}. Cannot determine image dimensions.")
                    h,w = 0,0 # Default or raise error
                else:
                    h, w, channels = image_0.shape

        lines = []
        
        # Camera model (Required)
        if 'model' in camera_params:
            lines.append(f"{prefix}.model: {camera_params['model']}")
            lines.append("") # Add a blank line for readability
        
        # Intrinsic parameters (fx, fy, cx, cy - usually all present)
        lines.append(f"{prefix}.fx: {camera_params['fx']}")
        lines.append(f"{prefix}.fy: {camera_params['fy']}")
        lines.append(f"{prefix}.cx: {camera_params['cx']}")
        lines.append(f"{prefix}.cy: {camera_params['cy']}")
        lines.append("")

        # Distortion parameters (conditionally added)
        # Plumb-bob / Rational polynomial / Fisheye parameters
        # Order for ORB_SLAM typically k1, k2, p1, p2, k3 (for plumb-bob)
        # For Fisheye: k1, k2, k3, k4
        # For Rational: k1, k2, p1, p2, k3, k4, k5, k6
        # The key names in camera_params should match what ORB_SLAM expects for these fields.
        
        distortion_keys_ordered = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        added_distortion_param = False
        for key in distortion_keys_ordered:
            if key in camera_params:
                lines.append(f"{prefix}.{key}: {camera_params[key]}")
                added_distortion_param = True
        
        if added_distortion_param:
            lines.append("") # Add a blank line after distortion parameters

        # Image dimensions (w, h)
        lines.append(f"{prefix}.w: {w}")
        lines.append(f"{prefix}.h: {h}")
        lines.append("")
        
        # Camera frames per second
        lines.append("# Camera frames per second")
        lines.append(f"{prefix}.fps: {self.rgb_hz}")
        
        return lines

    def _get_imu_yaml_section(self, imu_params):
        """Generate YAML lines for IMU parameters."""
        lines = []
        
        # IMU transform
        if 'transform' in imu_params:
            transform = imu_params['transform']
            
            # Flatten the transform if it's a nested list
            if isinstance(transform[0], list):  # nested list format [[...], [...], ...]
                flat_transform = [item for row in transform for item in row]
            else:  # already flat list [...]
                flat_transform = transform
            
            lines.extend([
                "# Transformation from camera to IMU",
                "IMU.T_c_i: !!opencv-matrix",
                "  rows: 4",
                "  cols: 4",
                "  dt: f",
                "  data: [" + ", ".join(map(str, flat_transform[:4])) + ",",
                "         " + ", ".join(map(str, flat_transform[4:8])) + ",",
                "         " + ", ".join(map(str, flat_transform[8:12])) + ",",
                "         " + ", ".join(map(str, flat_transform[12:16])) + "]",
                ""
            ])
        lines.append("# IMU noise")
        # Noise parameters
        if 'gyro_noise' in imu_params:
            lines.append(f"IMU.NoiseGyro: {imu_params['gyro_noise']:e}")
        if 'accel_noise' in imu_params:
            lines.append(f"IMU.NoiseAcc: {imu_params['accel_noise']:e}")
        if 'gyro_bias' in imu_params:
            lines.append(f"IMU.GyroWalk: {imu_params['gyro_bias']:e}")
        if 'accel_bias' in imu_params:
            lines.append(f"IMU.AccWalk: {imu_params['accel_bias']:e}")
        if 'frequency' in imu_params:
            lines.append(f"IMU.Frequency: {imu_params['frequency']:e}")
        
        return lines

    def _get_rgbd_yaml_section(self, rgbd_params):
        """Generate YAML lines for RGBD parameters."""
        lines = []
        
        if 'depth_factor' in rgbd_params:
            lines.append(f"depth_factor: {rgbd_params['depth_factor']:e}")
        
        if 'depth_scale' in rgbd_params:
            lines.append(f"depth_scale: {rgbd_params['depth_scale']:e}")
        
        return lines

    def check_sequence_availability(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        if os.path.exists(sequence_path):
            sequence_complete = check_sequence_integrity(self.dataset_path, sequence_name, True)
            if sequence_complete:
                return "available"
            else:
                return "corrupted"
        return "non-available"

    ####################################################################################################################
    # Utils

    def contains_sequence(self, sequence_name_ref):
        for sequence_name in self.sequence_names:
            if sequence_name == sequence_name_ref:
                return True
        return False

    def print_sequence_names(self):
        print(self.sequence_names)

    def print_sequence_nicknames(self):
        print(self.sequence_nicknames)

    def get_sequence_names(self):
        return self.sequence_names

    def get_sequence_nicknames(self):
        return self.sequence_nicknames

    def get_sequence_nickname(self, sequence_name_ref):
        for i, sequence_name in enumerate(self.sequence_names):
            if sequence_name == sequence_name_ref:
                return self.sequence_nicknames[i]

    def get_sequence_num_rgb(self, sequence_name):
        rgb_txt = os.path.join(self.dataset_path, sequence_name, 'rgb.txt')
        if os.path.exists(rgb_txt):
            with open(rgb_txt, 'r') as file:
                line_count = 0
                for line in file:
                    line_count += 1
            return line_count
        return 0
