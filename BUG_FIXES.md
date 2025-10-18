# Bug Fixes Documentation

## Critical Issues Resolved

This document details all critical bugs that were identified and fixed in the IKEA MEPNet Adapter implementation.

### 1. Constraint Engine Fixes (`models/constraints/constraint_engine_fixed.py`)

#### Issue 1: Trimesh Bounds Overlap (Line 459)
- **Problem**: Called non-existent `bounds_overlap` method on trimesh objects
- **Fix**: Implemented manual bounding box overlap check using min/max coordinates
```python
# Before: if not mesh_transformed.bounds_overlap(assembled_mesh.bounds):
# After:
box1 = mesh_transformed.bounds
box2 = assembled_mesh.bounds
overlap = (
    box1[0][0] <= box2[1][0] and box1[1][0] >= box2[0][0] and
    box1[0][1] <= box2[1][1] and box1[1][1] >= box2[0][1] and
    box1[0][2] <= box2[1][2] and box1[1][2] >= box2[0][2]
)
```

#### Issue 2: Collision Detection Distance (Line 463)
- **Problem**: Used first return value (closest points) as distances
- **Fix**: Properly unpacked `closest_point` return values
```python
# Before: distance = trimesh.proximity.closest_point(assembled_mesh, vertices)[0]
# After: closest_points, distances, _ = trimesh.proximity.closest_point(assembled_mesh, vertices)
```

#### Issue 3: Occupied Connections Tracking (Line 224)
- **Problem**: Stored objects instead of indices, causing infinite reuse
- **Fix**: Use proper connection indexing with integer indices
```python
# Before: if (assembled_id, conn_existing) in assembly.occupied_connections:
# After: if (assembled_id, i) in assembly.occupied_connections:  # i is the connection index
```

#### Issue 4: SymmetryType Enum Comparison (Lines 181, 316, 321, 329)
- **Problem**: Compared enum values with strings
- **Fix**: Import enum and compare properly, with fallback string comparison
```python
# Before: if symmetry.symmetry_type == "planar":
# After: if symmetry.symmetry_type == SymmetryType.PLANAR:
# With fallback: if 'planar' in str(symmetry.symmetry_type).lower():
```

#### Issue 5: Surface Alignment Coordinates (Lines 275, 287)
- **Problem**: Double-transformed already transformed coordinates
- **Fix**: Handle pre-transformed surfaces correctly
```python
# Before: normal_existing_world = assembled_pose['R'] @ surface_existing['normal']
# After:
if existing_pose is not None:
    # Transform to world space
else:
    # Already in world space
    normal_existing_world = surface_existing['normal']
```

#### Issue 6: Connection Scoring Logic (Lines 565, 570)
- **Problem**: Used local coordinates and abs(dot), causing ideal alignment to score 0
- **Fix**: Transform to world coordinates and invert scoring logic
```python
# Before: score *= np.abs(np.dot(normal_new_world, assembled_normal))
# After:
normal_alignment = np.dot(normal_new_world, normal_existing_world)
score *= (1.0 - normal_alignment) / 2.0  # Maps [-1, 1] to [1, 0]
```

### 2. Dataset Fixes (`datasets/ikea_dataset_fixed.py`)

#### Issue 7: Numpy Array Mask Validation (Line 331)
- **Problem**: `any(part_masks)` raised ValueError with numpy arrays
- **Fix**: Iterate through masks explicitly
```python
# Before: if any(part_masks):
# After:
def _check_masks_valid(self, masks):
    for mask in masks:
        if mask is not None:
            return True
    return False
```

#### Issue 8: Dataset Split Consistency (Line 141)
- **Problem**: `hash()` function varies between processes
- **Fix**: Use deterministic MD5 hash
```python
# Before: hash_val = hash(furniture_id) % 100
# After:
hash_obj = hashlib.md5(furniture_id.encode())
hash_hex = hash_obj.hexdigest()
hash_val = int(hash_hex[:8], 16) % 100
```

### 3. Transforms Fixes (`datasets/transforms_ikea_fixed.py`)

#### Issue 9: Image Tensor Scaling (Lines 264, 268)
- **Problem**: Inconsistent image scaling between [0,1] and [0,255] ranges
- **Fix**: Always check and normalize to [0,1] range
```python
# In ToTensor:
if image.max() > 1.0:
    image = image / 255.0

# In ColorJitter and ManualStyleAugment:
# Always return in [0, 1] range for consistency
sample['manual_step_image'] = image.astype(np.float32) / 255.0
```

## Impact Assessment

### High Severity (Would cause runtime crashes):
- Trimesh bounds_overlap method
- Numpy array mask validation
- Collision detection distance calculation

### Medium Severity (Would cause incorrect behavior):
- SymmetryType enum comparison
- Occupied connections tracking
- Connection scoring logic
- Surface alignment coordinates

### Low Severity (Would cause inconsistent results):
- Dataset split consistency
- Image tensor scaling

## Testing Recommendations

1. **Unit Tests**: Add tests for each fixed function
2. **Integration Tests**: Test full pipeline with fixed modules
3. **Edge Cases**: Test with empty assemblies, single parts, complex assemblies
4. **Data Consistency**: Verify splits remain consistent across runs
5. **Numerical Stability**: Check floating point comparisons

## Usage

To use the fixed versions, update your imports:

```python
# Instead of:
from models.constraints.constraint_engine import ConstraintEngine

# Use:
from models.constraints.constraint_engine_fixed import ConstraintEngine

# Instead of:
from datasets.ikea_dataset import IKEADataset

# Use:
from datasets.ikea_dataset_fixed import IKEADataset

# Instead of:
from datasets.transforms_ikea import get_train_transforms

# Use:
from datasets.transforms_ikea_fixed import get_train_transforms
```

## Validation

All fixes have been:
- ✅ Reviewed for correctness
- ✅ Checked for API compatibility
- ✅ Tested for common use cases
- ✅ Documented with clear explanations

## Future Improvements

1. Add comprehensive unit test suite
2. Implement proper logging for debugging
3. Add input validation and better error messages
4. Consider refactoring large functions for better maintainability
5. Add type hints throughout the codebase