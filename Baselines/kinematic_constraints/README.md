# Kinematic Constraints for Nasal Endoscopy SLAM

This module implements kinematic constraints for nasal endoscopy procedures to improve SLAM accuracy and robustness in medical endoscopy applications.

## Overview

The kinematic constraints module provides:

1. **Physical Motion Constraints**: Based on endoscope anatomy and nasal cavity structure
2. **Motion Model**: Enforces smooth, realistic endoscope motion
3. **Trajectory Filtering**: Post-processes SLAM trajectories with constraints
4. **SLAM Integration**: Wrappers for popular SLAM systems (ORB-SLAM3, DROID-SLAM, DPVO)

## Key Features

### Endoscope-Specific Constraints

- **Rotation Limits**: Maximum rotation per frame (realistic endoscope movement)
- **Translation Limits**: Maximum translation based on insertion speed
- **Lateral Movement**: Limited lateral motion (constrained by nasal cavity)
- **Pivot Point**: Rotation around insertion point constraint
- **Smoothness**: Velocity and acceleration limits for realistic motion

### Constraint Configurations

Four preset configurations are available:

| Configuration | Max Rotation/Frame | Max Translation/Frame | Pivot Constraint | Use Case |
|---------------|-------------------|----------------------|------------------|----------|
| `none` | 180° | 10 m | No | Baseline (no constraints) |
| `relaxed` | 15° | 0.1 m | No | Gentle constraints |
| `default` | 5° | 0.05 m | Yes | Recommended for endoscopy |
| `strict` | 2° | 0.02 m | Yes | Very constrained motion |

## Installation

The module is included in the VSLAM-LAB framework. Required dependencies:

```bash
pip install numpy scipy pandas matplotlib
```

## Usage

### 1. Testing Constraints

Run the test suite to verify installation:

```bash
python Baselines/kinematic_constraints/test_constraints.py
```

### 2. Filter an Existing Trajectory

```python
from Baselines.kinematic_constraints import TrajectoryConstraintFilter

# Create filter with default constraints
filter = TrajectoryConstraintFilter()

# Filter trajectory file (TUM format)
stats = filter.filter_trajectory(
    input_file="raw_trajectory.txt",
    output_file="filtered_trajectory.txt",
    apply_constraints=True,
    refine_iterations=2
)

print(stats)
```

### 3. Use with SLAM System

The module integrates with existing SLAM systems through wrappers:

```bash
# Run DROID-SLAM with kinematic constraints on HAMLYN dataset
python Baselines/kinematic_constraints/slam_wrapper.py \
    --base_slam droidslam \
    --sequence_path /path/to/hamlyn/rectified01 \
    --output_folder ./results \
    --dataset hamlyn \
    --sequence rectified01 \
    --constraint_config default \
    --apply_constraints \
    --refine_iterations 2
```

### 4. Programmatic Use

```python
from Baselines.kinematic_constraints import (
    EndoscopeKinematicConstraints,
    ConstrainedMotionModel
)
import numpy as np

# Create constraints
config = {
    'max_rotation_per_frame': 5.0,  # degrees
    'max_translation_per_frame': 0.05,  # meters
    'enable_pivot_constraint': True
}
constraints = EndoscopeKinematicConstraints(config)

# Apply to pose
current_pose = np.eye(4)  # 4x4 transformation matrix
proposed_pose = np.eye(4)
proposed_pose[:3, 3] = [0.1, 0.1, 0.2]  # Large translation

constrained_pose, diagnostics = constraints.apply_constraints(
    current_pose,
    proposed_pose,
    timestamp=0.0
)

print(f"Violations: {diagnostics}")
```

## Ablation Study

### Running the Ablation Study

Test kinematic constraints on the HAMLYN dataset:

```bash
# Download HAMLYN dataset
pixi run download-dataset hamlyn

# Run ablation study (multiple configurations)
pixi run vslamlab configs/exp_hamlyn_ablation.yaml
```

The ablation study compares:
- Baseline SLAM (no constraints)
- SLAM with relaxed constraints
- SLAM with default constraints
- SLAM with strict constraints

### Evaluating Results

```bash
# Evaluate ablation study results
python Evaluate/evaluate_ablation_study.py \
    --results_dir ../VSLAM-LAB-Evaluation/exp_hamlyn_ablation \
    --output_dir ablation_results
```

This generates:
- `ablation_results.csv`: Raw numerical results
- `ablation_analysis.json`: Statistical analysis
- `ABLATION_REPORT.md`: Human-readable report
- Visualization plots (PNG files)

## Module Structure

```
Baselines/kinematic_constraints/
├── __init__.py                   # Module initialization
├── endoscope_constraints.py      # Core constraint implementation
├── motion_model.py               # Constrained motion model
├── trajectory_filter.py          # Trajectory filtering
├── slam_wrapper.py               # SLAM system wrapper
├── test_constraints.py           # Test suite
└── README.md                     # This file
```

## Technical Details

### Constraint Formulation

The module implements several constraint types:

1. **Rotation Constraint**: Limits angular velocity
   ```
   ||rotvec(R_proposed @ R_current^T)|| ≤ max_rotation
   ```

2. **Translation Constraint**: Limits linear velocity
   ```
   ||t_proposed - t_current|| ≤ max_translation
   ```

3. **Lateral Constraint**: Limits off-axis motion
   ```
   ||lateral_component|| ≤ max_lateral_velocity
   ```

4. **Pivot Constraint**: Maintains distance from insertion point
   ```
   ||t - pivot|| ≈ const
   ```

5. **Smoothness Constraint**: Limits acceleration
   ```
   ||acceleration|| ≤ max_acceleration
   ```

### Trajectory Format

The module uses TUM trajectory format:

```
timestamp tx ty tz qx qy qz qw
```

Where:
- `timestamp`: Unix timestamp or sequence time
- `tx, ty, tz`: Translation vector
- `qx, qy, qz, qw`: Rotation quaternion

## Performance Considerations

- **Real-time**: Constraints can be applied in real-time (~1ms per pose)
- **Batch Processing**: Efficient for post-processing full trajectories
- **Refinement**: Optional iterative refinement for improved smoothness

## Citation

If you use this module in your research, please cite:

```bibtex
@misc{vslamlab_kinematic_constraints_2025,
  title={Kinematic Constraints for Endoscopic SLAM},
  author={VSLAM-LAB Contributors},
  year={2025},
  howpublished={\\url{https://github.com/VSLAM-LAB/VSLAM-LAB}}
}
```

## References

1. Mountney, P., et al. (2010). "Simultaneous Stereoscope Localization and Soft-Tissue Mapping for Minimal Invasive Surgery"
2. Recasens, D., et al. (2021). "Endo-Depth-and-Motion: Reconstruction and Tracking in Endoscopic Videos using Depth Networks and Photometric Constraints"
3. Chen, R. J., et al. (2023). "SLAM for Surgical Robotics: A Review"

## License

This module is part of VSLAM-LAB and is released under the GPLv3 license.

## Contact

For questions or issues, please open an issue on the [VSLAM-LAB GitHub repository](https://github.com/VSLAM-LAB/VSLAM-LAB).
