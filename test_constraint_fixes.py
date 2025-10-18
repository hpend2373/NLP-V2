"""
Test script to validate constraint engine fixes
Tests metadata preservation and bidirectional connection tracking
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ikea_mepnet_adapter.models.constraints.constraint_engine_final import ConstraintEngine, AssemblyState
from ikea_mepnet_adapter.assets.registry import AssetsRegistry, PartInfo, ConnectionPoint, ConnectionType
import trimesh

def create_test_registry():
    """Create a test registry with sample parts"""
    registry = AssetsRegistry(parts_dir="dummy")

    # Create test part A with connections
    mesh_a = trimesh.creation.box(extents=(0.1, 0.1, 0.02))
    connections_a = [
        ConnectionPoint(
            position=np.array([0.05, 0, 0]),
            normal=np.array([1, 0, 0]),
            connection_type=ConnectionType.DOWEL_HOLE
        ),
        ConnectionPoint(
            position=np.array([-0.05, 0, 0]),
            normal=np.array([-1, 0, 0]),
            connection_type=ConnectionType.DOWEL_HOLE
        )
    ]

    part_a = PartInfo(
        part_id="part_a",
        mesh=mesh_a,
        category="test",
        connection_points=connections_a,
        principal_axes=np.eye(3),
        symmetry=None,
        material_properties={}
    )

    # Create test part B with matching connections
    mesh_b = trimesh.creation.box(extents=(0.1, 0.1, 0.02))
    connections_b = [
        ConnectionPoint(
            position=np.array([0.05, 0, 0]),
            normal=np.array([1, 0, 0]),
            connection_type=ConnectionType.DOWEL
        ),
        ConnectionPoint(
            position=np.array([-0.05, 0, 0]),
            normal=np.array([-1, 0, 0]),
            connection_type=ConnectionType.DOWEL
        )
    ]

    part_b = PartInfo(
        part_id="part_b",
        mesh=mesh_b,
        category="test",
        connection_points=connections_b,
        principal_axes=np.eye(3),
        symmetry=None,
        material_properties={}
    )

    # Manually add parts to registry (bypassing file loading)
    registry.parts = {
        "part_a": part_a,
        "part_b": part_b
    }

    return registry

def test_metadata_preservation():
    """Test that connection metadata is preserved through the pipeline"""
    print("Testing metadata preservation...")

    registry = create_test_registry()
    engine = ConstraintEngine(registry)

    # Add first part to assembly
    R_a = np.eye(3)
    t_a = np.array([0, 0, 0])
    engine.update_assembly("part_a", R_a, t_a, connections_used=[])

    # Get valid poses for second part
    poses = engine.get_valid_poses("part_b", max_candidates=10)

    # Check that poses include metadata
    assert len(poses) > 0, "No valid poses generated"

    for R, t, score, metadata in poses:
        print(f"  Pose score: {score:.3f}")
        if metadata:
            print(f"    Metadata: {metadata}")
            assert 'new_conn_idx' in metadata, "Missing new_conn_idx in metadata"
            assert 'assembled_part_id' in metadata, "Missing assembled_part_id in metadata"
            assert 'assembled_conn_idx' in metadata, "Missing assembled_conn_idx in metadata"
        else:
            print("    No metadata (surface-based pose)")

    print("✓ Metadata preservation test passed")
    return True

def test_bidirectional_connection_tracking():
    """Test that both connections are marked as occupied"""
    print("\nTesting bidirectional connection tracking...")

    registry = create_test_registry()
    engine = ConstraintEngine(registry)

    # Add first part
    R_a = np.eye(3)
    t_a = np.array([0, 0, 0])
    engine.update_assembly("part_a", R_a, t_a, connections_used=[])

    # Check initial state
    print(f"  Initial occupied connections: {engine.assembly_state.occupied_connections}")
    assert len(engine.assembly_state.occupied_connections) == 0, "Should start with no occupied connections"

    # Get valid poses for second part
    poses = engine.get_valid_poses("part_b", max_candidates=10)

    # Use the first connection-based pose
    selected_pose = None
    selected_metadata = None
    for R, t, score, metadata in poses:
        if metadata:  # Found a connection-based pose
            selected_pose = (R, t, score)
            selected_metadata = metadata
            break

    assert selected_pose is not None, "No connection-based poses found"

    R_b, t_b, _ = selected_pose
    new_conn_idx = selected_metadata['new_conn_idx']

    # Update assembly with the second part
    engine.update_assembly(
        "part_b",
        R_b,
        t_b,
        connections_used=[new_conn_idx],
        connection_metadata=selected_metadata
    )

    # Check that both connections are marked as occupied
    print(f"  After assembly occupied connections: {engine.assembly_state.occupied_connections}")

    # Should have both the new part's connection and the assembled part's connection
    assert ("part_b", new_conn_idx) in engine.assembly_state.occupied_connections, \
        f"New part connection ({new_conn_idx}) not marked as occupied"

    assembled_part_id = selected_metadata['assembled_part_id']
    assembled_conn_idx = selected_metadata['assembled_conn_idx']
    assert (assembled_part_id, assembled_conn_idx) in engine.assembly_state.occupied_connections, \
        f"Assembled part connection ({assembled_part_id}, {assembled_conn_idx}) not marked as occupied"

    # Verify that occupied connections are not reused
    poses_third = engine.get_valid_poses("part_b", max_candidates=10)

    for R, t, score, metadata in poses_third:
        if metadata:
            # Check that previously used connections are not suggested again
            assert metadata['assembled_conn_idx'] != assembled_conn_idx or \
                   metadata['assembled_part_id'] != assembled_part_id, \
                   f"Occupied connection ({assembled_part_id}, {assembled_conn_idx}) was reused!"

    print("✓ Bidirectional connection tracking test passed")
    return True

def test_connection_reuse_prevention():
    """Test that occupied connections cannot be reused"""
    print("\nTesting connection reuse prevention...")

    registry = create_test_registry()
    engine = ConstraintEngine(registry)

    # Add first part with two connections
    R_a = np.eye(3)
    t_a = np.array([0, 0, 0])
    engine.update_assembly("part_a", R_a, t_a, connections_used=[])

    # Add second part using first connection
    poses = engine.get_valid_poses("part_b", max_candidates=10)

    # Find and use connection 0 of part_a
    for R, t, score, metadata in poses:
        if metadata and metadata['assembled_conn_idx'] == 0:
            engine.update_assembly(
                "part_b",
                R, t,
                connections_used=[metadata['new_conn_idx']],
                connection_metadata=metadata
            )
            print(f"  Used connection 0 of part_a")
            break

    # Try to add another part_b - it should not suggest connection 0 of part_a
    poses_second = engine.get_valid_poses("part_b", max_candidates=20)

    connection_indices_used = []
    for R, t, score, metadata in poses_second:
        if metadata and metadata['assembled_part_id'] == 'part_a':
            connection_indices_used.append(metadata['assembled_conn_idx'])

    print(f"  Available connections after first assembly: {set(connection_indices_used)}")

    assert 0 not in connection_indices_used, \
        "Connection 0 of part_a was suggested again after being occupied!"

    assert 1 in connection_indices_used, \
        "Connection 1 of part_a should still be available"

    print("✓ Connection reuse prevention test passed")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Constraint Engine Fix Validation Tests")
    print("=" * 60)

    try:
        # Run tests
        test_metadata_preservation()
        test_bidirectional_connection_tracking()
        test_connection_reuse_prevention()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())