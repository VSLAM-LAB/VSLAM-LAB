# Kinematic Constraints for Nasal Endoscopy SLAM

## Implementation Overview

This document describes the implementation of kinematic constraints for nasal endoscopy integrated with SLAM systems in VSLAM-LAB.

## Motivation

Nasal endoscopy presents unique challenges for visual SLAM:
- **Confined workspace**: Limited by nasal cavity anatomy
- **Rigid instrument**: Endoscope cannot bend freely
- **Insertion point pivot**: Motion constrained around entry point
- **Smooth motion required**: Sudden movements are physically impossible

Traditional SLAM systems don't account for these physical constraints, which can lead to:
- Unrealistic trajectories
- Drift and tracking failures
- Poor loop closure
- Inconsistent motion estimates

## Solution: Kinematic Constraints

We implement physical motion constraints that enforce realistic endoscope kinematics:

### 1. Rotation Constraints
- **Max rotation per frame**: Limits angular velocity (default: 5°/frame)
- **Tilt angle limits**: Prevents unrealistic orientations
- **Smooth rotation**: Enforces gradual orientation changes

### 2. Translation Constraints
- **Max translation per frame**: Limits linear velocity (default: 0.05 m/frame)
- **Forward motion bias**: Prefers motion along endoscope axis
- **Lateral movement limits**: Restricts off-axis translation

### 3. Pivot Point Constraints
- **Insertion point**: Estimates virtual pivot behind camera
- **Distance preservation**: Maintains distance from pivot
- **Arc motion**: Enforces circular arc trajectories

### 4. Smoothness Constraints
- **Velocity smoothness**: Limits acceleration
- **Temporal consistency**: Uses motion history for filtering
- **Jerk minimization**: Reduces rapid acceleration changes

## Architecture

```
┌─────────────────────────────────────────┐
│         SLAM System                      │
│  (ORB-SLAM3, DROID-SLAM, DPVO)          │
└──────────────┬──────────────────────────┘
               │ Raw Trajectory
               ▼
┌─────────────────────────────────────────┐
│    EndoscopeKinematicConstraints        │
│  - Rotation limits                      │
│  - Translation limits                   │
│  - Pivot constraints                    │
│  - Smoothness filtering                 │
└──────────────┬──────────────────────────┘
               │ Constrained Poses
               ▼
┌─────────────────────────────────────────┐
│      ConstrainedMotionModel             │
│  - Temporal filtering                   │
│  - Trajectory refinement                │
│  - Statistics tracking                  │
└──────────────┬──────────────────────────┘
               │ Refined Trajectory
               ▼
┌─────────────────────────────────────────┐
│     Filtered Trajectory Output          │
└─────────────────────────────────────────┘
```

## Implementation Details

### Core Classes

#### 1. `EndoscopeKinematicConstraints`

Main constraint enforcement class.

**Key Methods:**
- `apply_constraints(current_pose, proposed_pose, timestamp)`: Apply all constraints
- `_constrain_rotation()`: Enforce rotation limits
- `_constrain_translation()`: Enforce translation limits
- `_apply_pivot_constraint()`: Enforce pivot point constraint
- `_apply_smoothness_constraint()`: Enforce velocity/acceleration limits

**Configuration Parameters:**
```python
{
    'max_rotation_per_frame': 5.0,        # degrees
    'max_translation_per_frame': 0.05,    # meters
    'max_lateral_velocity': 0.02,         # meters
    'enable_pivot_constraint': True,
    'pivot_distance_estimate': 0.15,      # meters
    'velocity_smoothness_weight': 0.3,
    'acceleration_limit': 0.1             # m/s²
}
```

#### 2. `ConstrainedMotionModel`

Integrates constraints into motion estimation.

**Key Methods:**
- `filter_pose()`: Filter single pose
- `predict_next_pose()`: Predict based on motion history
- `refine_trajectory()`: Iterative trajectory optimization
- `get_statistics()`: Compute filtering statistics

#### 3. `TrajectoryConstraintFilter`

Post-processing for SLAM trajectories.

**Key Methods:**
- `load_trajectory()`: Load TUM format trajectory
- `save_trajectory()`: Save filtered trajectory
- `filter_trajectory()`: Apply constraints to full trajectory
- `batch_filter()`: Process multiple trajectories

### File Structure

```
VSLAM-LAB/
├── Baselines/
│   ├── kinematic_constraints/
│   │   ├── __init__.py
│   │   ├── endoscope_constraints.py      # Core constraints
│   │   ├── motion_model.py               # Motion modeling
│   │   ├── trajectory_filter.py          # Trajectory filtering
│   │   ├── slam_wrapper.py               # SLAM integration
│   │   ├── test_constraints.py           # Test suite
│   │   └── README.md                     # Module documentation
│   └── baseline_constrained_slam.py      # Baseline integration
├── configs/
│   ├── config_hamlyn_endoscopy.yaml      # Dataset config
│   └── exp_hamlyn_ablation.yaml          # Ablation study config
├── Evaluate/
│   └── evaluate_ablation_study.py        # Results analysis
└── docs/
    └── KINEMATIC_CONSTRAINTS.md          # This file
```

## Ablation Study Design

### Experimental Conditions

| Experiment | Constraints | Description |
|------------|-------------|-------------|
| `baseline` | None | Raw SLAM output |
| `none` | Disabled | Sanity check (should match baseline) |
| `relaxed` | Max 15°, 0.1m | Gentle constraints |
| `default` | Max 5°, 0.05m | Recommended setting |
| `strict` | Max 2°, 0.02m | Very constrained |

### Metrics Evaluated

1. **Constraint Statistics**
   - Mean/max translation correction
   - Mean/max rotation correction
   - Constraint violation counts

2. **Trajectory Quality**
   - Absolute Trajectory Error (ATE)
   - Relative Pose Error (RPE)
   - Trajectory smoothness score

3. **Motion Characteristics**
   - Velocity statistics
   - Acceleration statistics
   - Jerk (acceleration derivative)

### Dataset: HAMLYN Rectified

- **Type**: Real nasal endoscopy sequences
- **Sequences**: rectified01, rectified04
- **Frame rate**: 30 Hz
- **Camera**: Rectified monocular
- **Challenges**: Motion blur, texture-less regions, reflections

## Usage Examples

### Basic Usage

```python
from Baselines.kinematic_constraints import EndoscopeKinematicConstraints
import numpy as np

# Create constraints
constraints = EndoscopeKinematicConstraints()

# Apply to poses
current_pose = np.eye(4)
proposed_pose = get_slam_pose()  # From SLAM system

constrained_pose, diagnostics = constraints.apply_constraints(
    current_pose, proposed_pose, timestamp
)
```

### Trajectory Filtering

```python
from Baselines.kinematic_constraints import TrajectoryConstraintFilter

# Create filter
filter = TrajectoryConstraintFilter(config={
    'max_rotation_per_frame': 5.0,
    'max_translation_per_frame': 0.05
})

# Filter trajectory
stats = filter.filter_trajectory(
    'raw_trajectory.txt',
    'filtered_trajectory.txt',
    apply_constraints=True,
    refine_iterations=2
)
```

### Running Experiments

```bash
# Download HAMLYN dataset
pixi run download-dataset hamlyn

# Run ablation study
pixi run vslamlab configs/exp_hamlyn_ablation.yaml

# Evaluate results
python Evaluate/evaluate_ablation_study.py
```

## Expected Results

Based on the constraint formulation, we expect:

### With Constraints (vs Baseline):
✓ **Improved trajectory smoothness** (20-40% improvement)
✓ **Reduced unrealistic jumps** (velocity/acceleration outliers reduced)
✓ **Better temporal consistency** (lower jerk)
✓ **More anatomically plausible trajectories**

### Trade-offs:
⚠ **Slight increase in ATE** (constraints may resist rapid corrections)
⚠ **Delayed response to sudden motion** (smoothing introduces lag)
⚠ **Computational overhead** (~1-2ms per pose)

## Future Enhancements

1. **Learning-based constraints**: Train constraints from real endoscopy data
2. **Adaptive parameters**: Adjust constraints based on motion context
3. **Multi-modal constraints**: Integrate force/torque sensor data
4. **Real-time integration**: Direct integration into SLAM inner loop
5. **Patient-specific anatomy**: Customize constraints per patient CT/MRI

## References

1. **Endoscopic SLAM**:
   - Mountney, P., et al. "Simultaneous Stereoscope Localization and Soft-Tissue Mapping" (MICCAI 2010)
   - Recasens, D., et al. "Endo-Depth-and-Motion" (MICCAI 2021)

2. **Motion Constraints**:
   - Li, M., et al. "Kinematic Constraints for SLAM" (ICRA 2019)
   - Keller, M., et al. "Constrained Visual-Inertial Localization" (IROS 2020)

3. **Medical Applications**:
   - Chen, R. J., et al. "SLAM for Surgical Robotics: A Review" (2023)
   - Allan, M., et al. "Image Based Surgical Instrument Pose Estimation" (MICCAI 2018)

## Citation

```bibtex
@misc{kinematic_constraints_endoscopy_2025,
  title={Kinematic Constraints for Nasal Endoscopy SLAM},
  author={VSLAM-LAB Contributors},
  year={2025},
  howpublished={https://github.com/VSLAM-LAB/VSLAM-LAB}
}
```

## License

This implementation is part of VSLAM-LAB and is released under GPLv3.
