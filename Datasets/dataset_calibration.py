import os, cv2
import numpy as np

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def _opencv_matrix_yaml(name, M, dtype='f'):
    M = np.asarray(M).reshape(M.shape[0], M.shape[1])
    rows, cols = M.shape
    flat = ", ".join(map(lambda x: f"{x}", M.flatten()))
    return [
        f"{name}: !!opencv-matrix",
        f"  rows: {rows}",
        f"  cols: {cols}",
        f"  dt: {dtype}",
        f"  data: [{flat}]",
        ""
    ]

def _get_camera_yaml_section(dataset_path, camera_params, sequence_name, rgb_hz, prefix="Camera"):
    """Generate YAML lines for camera parameters.

    If camera_params contains explicit 'w' and 'h', use them. Otherwise, fall back
    to inferring from images located at '<sequence>/rgb_0'.
    """
    # Determine image dimensions
    if 'w' in camera_params and 'h' in camera_params:
        w = int(camera_params['w'])
        h = int(camera_params['h'])
    else:
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
    
    # Camera model (Required) - normalize for OKVIS2 expectations
    if 'model' in camera_params:
        model_str = str(camera_params['model'])
        mlow = model_str.lower()
        if mlow == 'pinhole':
            model_str = 'Pinhole'
        elif mlow == 'opencv':
            model_str = 'OPENCV'
        lines.append(f"{prefix}.model: {model_str}")
        lines.append("") # Add a blank line for readability
    
    # Intrinsic parameters (fx, fy, cx, cy - usually all present)
    fx, fy, cx, cy = camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']
    lines.append(f"{prefix}.fx: {fx}")
    lines.append(f"{prefix}.fy: {fy}")
    lines.append(f"{prefix}.cx: {cx}")
    lines.append(f"{prefix}.cy: {cy}")
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
        # Add compact arrays that some consumers expect
        coeffs = [camera_params.get('k1', 0.0), camera_params.get('k2', 0.0), camera_params.get('p1', 0.0), camera_params.get('p2', 0.0)]
        lines.append(f"{prefix}.distortion_coefficients: [{coeffs[0]}, {coeffs[1]}, {coeffs[2]}, {coeffs[3]}]")
        lines.append(f"{prefix}.intrinsics: [{fx}, {fy}, {cx}, {cy}]")
        # Add distortion key (OKVIS2 expects a string)
        distortion = camera_params.get('distortion', None)
        if distortion is None:
            model = str(camera_params.get('distortion_model', '')).lower()
            if model in ('plumb_bob', 'radtan', 'radial-tangential', 'opencv', 'pinhole'):
                distortion = 'radtan'
            elif model in ('equidistant', 'fisheye'):
                distortion = 'equidistant'
            else:
                distortion = 'radtan'
        lines.append(f"{prefix}.distortion: {distortion}")
        lines.append(f"{prefix}.distortion_type: {distortion}")
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
    """Generate YAML lines for IMU parameters.

    Supports either a single 'transform' (legacy) or multiple transforms via
    'transforms': {'T_b_c0': [...16...], 'T_b_c1': [...16...], ...}.
    """
    lines = []
    
    # IMU transforms
    if 'transforms' in imu_params and isinstance(imu_params['transforms'], dict):
        for key, transform in imu_params['transforms'].items():
            # Normalize key to produce IMU.<key>
            label = key if key.startswith('T_b_c') else f"T_b_c{key}"

            # Flatten the transform if it's a nested list
            if isinstance(transform, list) and len(transform) > 0 and isinstance(transform[0], list):
                flat_transform = [item for row in transform for item in row]
            else:
                flat_transform = transform

            lines.extend([
                "# Transformation from camera to IMU",
                f"IMU.{label}: !!opencv-matrix",
                "  rows: 4",
                "  cols: 4",
                "  dt: f",
                "  data: [" + ", ".join(map(str, flat_transform[:4])) + ",",
                "         " + ", ".join(map(str, flat_transform[4:8])) + ",",
                "         " + ", ".join(map(str, flat_transform[8:12])) + ",",
                "         " + ", ".join(map(str, flat_transform[12:16])) + "]",
                ""
            ])
    elif 'transform' in imu_params:
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

def _get_stereo_yaml_section(stereo_params):
    """Generate YAML lines for STEREO parameters."""
    lines = []
    
    if 'bf' in stereo_params:
        lines.append(f"Stereo.bf: {stereo_params['bf']:e}")
    
    return lines