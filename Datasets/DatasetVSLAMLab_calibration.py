import os, cv2
from pathlib import Path
from typing import List

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "

def _get_rgb_yaml_section(camera_params, sequence_name: str, dataset_path: Path) -> List[str]:
    """Generate YAML lines for rgb parameters."""
    # Get image dimensions
    sequence_path = os.path.join(dataset_path, sequence_name)
    rgb_path = os.path.join(sequence_path, 'rgb_0')
    
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
    lines.append(f"  - {{cam_name: {camera_params['cam_name']},")
    lines.append(f"     cam_type: {camera_params['cam_type']},")
    lines.append(f"     cam_model: {camera_params['cam_model']},")
    if 'distortion_type' in camera_params:
        lines.append(f"     distortion_type: {camera_params['distortion_type']},")
    lines.append(f"     focal_length: {camera_params['focal_length']},")
    lines.append(f"     principal_point: {camera_params['principal_point']},")
    if 'distortion_coefficients' in camera_params:
        lines.append(f"     distortion_coefficients: {camera_params['distortion_coefficients']}, ")

    lines.append(f"     image_dimension: [{w}, {h}],")
    lines.append(f"     fps: {camera_params['fps']},")

    T_BS_data = camera_params['T_BS']
    flat_list_str = [f"{x:.13f}" for x in T_BS_data.flatten()]
    lines.append("     T_BS: [" + ', '.join(flat_list_str) + "] # Sensor extrinsics wrt. the body-frame.")

    lines.append(f"    }}\n")
    return lines

def _get_rgbd_yaml_section(camera_params, sequence_name: str, dataset_path: Path) -> List[str]:
    """Generate YAML lines for rgb parameters."""
    # """Generate YAML lines for RGBD parameters."""
    lines = _get_rgb_yaml_section(camera_params, sequence_name, dataset_path);
    lines.insert(2, f"     depth_name: {camera_params['depth_name']},")
    lines.insert(8, f"     depth_factor: {camera_params['depth_factor']},")
    return lines

def _get_imu_yaml_section(imu_params) -> List[str]:
    """Generate YAML lines for IMU parameters."""
    lines = []
    lines.append(f"  - {{imu_name: {imu_params['imu_name']},")
    lines.append(f"     a_max: {imu_params['a_max']}, # acceleration saturation [m/s^2]")
    lines.append(f"     g_max: {imu_params['g_max']},  # gyro saturation [rad/s]")
    lines.append(f"     sigma_g_c: {imu_params['sigma_g_c']}, # gyro noise density [rad/s/sqrt(Hz)]")
    lines.append(f"     sigma_bg: {imu_params['sigma_bg']}, # gyro bias prior [rad/s]")
    lines.append(f"     sigma_a_c: {imu_params['sigma_a_c']}, # accelerometer noise density [m/s^2/sqrt(Hz)]")
    lines.append(f"     sigma_ba: {imu_params['sigma_ba']}, # accelerometer bias prior [m/s^2]")
    lines.append(f"     sigma_gw_c: {imu_params['sigma_gw_c']}, # gyro drift noise density [rad/s^s/sqrt(Hz)]")
    lines.append(f"     sigma_aw_c: {imu_params['sigma_aw_c']}, # accelerometer drift noise density [m/s^2/sqrt(Hz)]")
    lines.append(f"     a0: {imu_params['a0']}, # initial accelerometer bias [m/s^2]")
    lines.append(f"     g0: {imu_params['g0']}, # initial gyro bias [rad/s]")
    lines.append(f"     g: {imu_params['g']}, # Earth's acceleration due to gravity [m/s^2]")
    lines.append(f"     s_a: {imu_params['s_a']}, # scale factor for accelerometer measurements: a_true = s_a * a_meas + b_a")
    lines.append(f"     fps: {imu_params['fps']},")

    T_BS_data = imu_params['T_BS']
    flat_list_str = [f"{x:.13f}" for x in T_BS_data.flatten()]
    lines.append("     T_BS: [" + ', '.join(flat_list_str) + "]  # Sensor extrinsics wrt. the body-frame.")

    lines.append(f"    }}\n")

    return lines