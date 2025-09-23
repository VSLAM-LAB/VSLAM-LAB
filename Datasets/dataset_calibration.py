import os, cv2

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def _get_camera_yaml_section(dataset_path, camera_params, sequence_name, rgb_hz, prefix="Camera"):
    """Generate YAML lines for camera parameters."""
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
    lines.append(f"{prefix}.fps: {rgb_hz}")
    
    return lines

def _get_imu_yaml_section(imu_params):
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
            "IMU.T_b_c1: !!opencv-matrix",
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


def _get_rgbd_yaml_section(rgbd_params, prefix="Depth"):
    """Generate YAML lines for RGBD parameters."""
    lines = []
    
    if 'depth0_factor' in rgbd_params:
        lines.append(f"{prefix}.factor: {rgbd_params['depth0_factor']:e}")
    
    if 'depth0_scale' in rgbd_params:
        lines.append(f"{prefix}.scale: {rgbd_params['depth0_scale']:e}")
    
    return lines