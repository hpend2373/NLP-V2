# Constraint Engine Final Fixes Summary

## Critical Issues Resolved in `constraint_engine_final.py`

### 1. Connection Metadata Loss (Lines 234-309)

**Problem**:
- Connection indices (`new_conn_idx`, `assembled_part_id`, `assembled_conn_idx`) were generated but then discarded at line 290
- Upper-level logic couldn't determine which connections were used

**Solution**:
```python
# Before (line 290):
return [(R, t, score) for R, t, score, _ in poses]  # Metadata lost!

# After (line 309):
return poses  # Metadata preserved
```

**Changes Made**:
- Modified `get_valid_poses` return type to include optional metadata: `List[Tuple[np.ndarray, np.ndarray, float, Optional[Dict]]]`
- Updated `_generate_connection_poses` to return metadata with each pose
- Preserved metadata through symmetry generation and filtering

### 2. Bidirectional Connection Occupancy Tracking (Lines 623-680)

**Problem**:
- Only new part's connections were marked as occupied
- Assembled part's connection remained available for infinite reuse

**Solution**:
```python
# Added to update_assembly() at lines 665-677:
# CRITICAL FIX: Also mark the assembled part's connection as occupied
if connection_metadata:
    assembled_part_id = connection_metadata.get('assembled_part_id')
    assembled_conn_idx = connection_metadata.get('assembled_conn_idx')
    if assembled_part_id and assembled_conn_idx is not None:
        self.assembly_state.occupied_connections.add(
            (assembled_part_id, assembled_conn_idx)
        )
        # Also update the assembled part's connections_used list
        if assembled_part_id in self.assembly_state.assembled_parts:
            self.assembly_state.assembled_parts[assembled_part_id]['connections_used'].append(
                assembled_conn_idx
            )
```

**Changes Made**:
- Added `connection_metadata` parameter to `update_assembly`
- Mark both new and existing connections as occupied
- Update both parts' `connections_used` lists

## API Changes

### `get_valid_poses` Method

**Before**:
```python
def get_valid_poses(...) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    # Returns: (R, t, score)
```

**After**:
```python
def get_valid_poses(...) -> List[Tuple[np.ndarray, np.ndarray, float, Optional[Dict]]]:
    # Returns: (R, t, score, metadata)
    # metadata = {
    #     'new_conn_idx': int,        # Connection index on new part
    #     'assembled_part_id': str,   # ID of existing part
    #     'assembled_conn_idx': int   # Connection index on existing part
    # }
```

### `update_assembly` Method

**Before**:
```python
def update_assembly(
    part_id: str,
    R: np.ndarray,
    t: np.ndarray,
    connections_used: Optional[List[int]] = None
)
```

**After**:
```python
def update_assembly(
    part_id: str,
    R: np.ndarray,
    t: np.ndarray,
    connections_used: Optional[List[int]] = None,
    connection_metadata: Optional[Dict] = None  # NEW PARAMETER
)
```

## Usage Example

```python
# Get poses with metadata
poses = engine.get_valid_poses("part_b")

# Select a pose
for R, t, score, metadata in poses:
    if metadata:  # Connection-based pose
        print(f"Using connection {metadata['new_conn_idx']} on new part")
        print(f"Connecting to {metadata['assembled_part_id']} connection {metadata['assembled_conn_idx']}")

        # Update assembly with bidirectional tracking
        engine.update_assembly(
            "part_b",
            R, t,
            connections_used=[metadata['new_conn_idx']],
            connection_metadata=metadata  # Pass metadata for bidirectional tracking
        )
        break
```

## Benefits

1. **Complete Information Flow**: Upper-level logic now knows exactly which connections were used
2. **No Connection Reuse**: Both sides of a connection are marked as occupied, preventing reuse
3. **Better Assembly Planning**: Can track connection utilization and make informed decisions
4. **Debugging Support**: Connection usage is explicitly tracked and visible

## Validation

Created `test_constraint_fixes.py` with three comprehensive tests:

1. **test_metadata_preservation**: Verifies metadata flows through the pipeline
2. **test_bidirectional_connection_tracking**: Confirms both connections are marked occupied
3. **test_connection_reuse_prevention**: Ensures occupied connections cannot be reused

## Impact on Other Modules

Modules that call `get_valid_poses` or `update_assembly` should be updated to handle the new return format and parameter:

- `models/mepnet_plus/solver.py`: Should extract metadata from poses and pass to update_assembly
- `planner/assembler.py`: Can use metadata for better assembly sequence planning
- `eval/eval_ikea.py`: May need updates to handle the new tuple format

## Migration Guide

For code using the old API:

```python
# Old code:
R, t, score = poses[0]
engine.update_assembly(part_id, R, t, [conn_idx])

# New code:
R, t, score, metadata = poses[0]
engine.update_assembly(part_id, R, t, [conn_idx], metadata)