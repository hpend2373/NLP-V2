# IKEA Data Format Corrections Summary

Based on the verification of IKEA-Manuals-at-Work dataset structure, the following corrections have been implemented:

## Key Corrections Made

### 1. Manual-level vs Frame-level Annotation Separation ✅

**Issue**: The original design incorrectly assumed 6D poses and camera parameters were available at the manual image level.

**Reality**:
- **Manual-level**: Contains `Manual Parts`, `Manual Connections`, `PDF Page`, `Cropped Manual Image` (NO camera/pose)
- **Frame-level**: Contains `6D poses`, `masks`, `camera parameters` (from aligned video frames)

**Fix Applied in `datasets/ikea_dataset_fixed.py`**:
```python
# Lines 289-311: Proper separation of annotation levels
if aligned_frames:
    frame_data = aligned_frames[0]  # Use aligned frame for supervision
    camera = self._parse_camera_params(frame_data.get('camera', {}))
    frame_poses = frame_data.get('poses', {})
    frame_masks = frame_data.get('masks', {})
else:
    # No frame alignment - no camera/pose supervision available
    camera = self._get_default_camera()
    frame_poses = {}
    frame_masks = {}
```

### 2. Manual Connections as Primary Constraint Input ✅

**Issue**: Manual connections were not being utilized as the primary source of assembly constraints.

**Reality**: IKEA dataset provides explicit `Manual Connections` for each step specifying which parts connect and how.

**Fix Applied in `models/constraints/constraint_engine_final.py`**:

1. **Added manual_connections parameter** (lines 147-150):
```python
def get_valid_poses(
    self,
    part_id: str,
    base_assembly: Optional[AssemblyState] = None,
    manual_connections: Optional[List[Tuple[str, str, str]]] = None,  # NEW
    ...
)
```

2. **Prioritized manual connections** (lines 174-182):
```python
# PRIORITY 1: Manual connections from IKEA dataset (highest priority)
if manual_connections:
    manual_poses = self._generate_manual_connection_poses(
        part_id, part_info, base_assembly, manual_connections
    )
    candidates.extend(manual_poses)
```

3. **Added dedicated manual connection handler** (lines 253-342):
```python
def _generate_manual_connection_poses(...):
    # Filter relevant connections for this part
    # Find matching connection points based on type
    # Generate poses with higher scores for manual connections
    # Mark connections as from_manual=True
```

### 3. Connection Metadata Preservation ✅

**Issue**: Connection indices were being lost in the pose generation pipeline.

**Fix Applied**:
- Return type includes metadata: `List[Tuple[np.ndarray, np.ndarray, float, Optional[Dict]]]`
- Metadata preserved through entire pipeline including symmetry variations
- Bidirectional connection occupancy tracking implemented

### 4. Dataset Output Structure Updates ✅

**Fix Applied in `datasets/ikea_dataset_fixed.py`** (lines 354-380):
```python
output = {
    ...
    # CRITICAL: Manual connections for constraint engine
    'manual_connections': sample['manual_connections'],

    # Frame alignment info (for supervision)
    'has_frame_supervision': len(aligned_frames) > 0,
    ...
}
```

## Data Flow Architecture

```
IKEA Dataset Structure:
├── Manual Level (Input)
│   ├── Manual Image (2D visual instruction)
│   ├── Manual Parts (parts to assemble)
│   └── Manual Connections [(part1, part2, type)]
│
├── Frame Level (Supervision)
│   ├── 6D Poses (GT for training)
│   ├── Instance Masks
│   └── Camera Parameters
│
└── MEPNet Adapter Pipeline:
    1. Load manual image + connections
    2. Use connections to constrain pose search
    3. Supervise with frame-level GT when available
    4. Fall back to constraint-only when no frames
```

## Connection Type Mappings

Implemented in `_find_connections_by_type` (lines 365-372):
```python
type_mappings = {
    'dowel': ('dowel', 'dowel_hole'),
    'screw': ('screw', 'screw_hole'),
    'cam_lock': ('cam_lock', 'cam_lock_hole'),
    'surface': ('surface', 'surface'),
    'snap': ('snap', 'snap_slot'),
    'hinge': ('hinge_pin', 'hinge_hole')
}
```

## Training Strategy Updates

### P0: Manual-only Mode (No Camera)
- Use Manual Connections to constrain pose space
- 2D keypoint detection on manual images
- Constraint-based 3D pose inference
- Supervision from aligned frames (transformed to object space)

### P1: Frame-enhanced Mode (With Camera)
- All of P0 plus:
- Differentiable rendering using frame camera params
- Direct mask/silhouette supervision from frames
- Camera-aware pose refinement

## Usage Example

```python
# In training loop
for batch in dataloader:
    manual_img = batch['manual_step_image']
    manual_connections = batch['manual_connections']  # NEW

    # Generate poses with manual connections
    poses = constraint_engine.get_valid_poses(
        part_id=part_id,
        base_assembly=current_assembly,
        manual_connections=manual_connections  # PRIMARY INPUT
    )

    # Supervision from frame-level if available
    if batch['has_frame_supervision']:
        gt_poses = batch['gt_poses']  # From aligned frames
        loss = compute_pose_loss(predicted, gt_poses)
```

## Impact on Other Modules

### Updated Modules
- ✅ `datasets/ikea_dataset_fixed.py`: Complete data loading rewrite
- ✅ `models/constraints/constraint_engine_final.py`: Manual connections support
- ✅ Connection metadata preservation throughout pipeline

### Modules Needing Updates
- `models/mepnet_plus/solver.py`: Should pass manual_connections to constraint engine
- `planner/assembler.py`: Can use manual connections for better planning
- `scripts/train/train_ikea.py`: Update to pass manual_connections in training

## Validation Checklist

- [x] Manual images loaded without camera assumptions
- [x] Frame-level annotations properly separated
- [x] Manual connections parsed and prioritized
- [x] Connection metadata preserved through pipeline
- [x] Bidirectional connection occupancy tracking
- [x] Proper supervision from aligned frames
- [x] Fallback for samples without frame alignment

## References

- IKEA-Manuals-at-Work Paper: [arXiv](https://arxiv.org/abs/2302.01016)
- MEPNet (ECCV'22): [arXiv](https://arxiv.org/abs/2206.08474)
- Dataset Structure: [GitHub](https://github.com/IKEA-Manuals-at-Work/ikea-manuals-at-work)