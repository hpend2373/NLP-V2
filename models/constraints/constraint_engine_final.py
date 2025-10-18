"""
Constraint Engine for IKEA Assembly (FINAL FIXED VERSION)
Fixes all critical bugs including SymmetryType import and connection tracking
"""

import numpy as np
import torch
import trimesh
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import itertools
from pathlib import Path
import warnings

class DoFType(Enum):
    """Degrees of Freedom types"""
    FIXED = "fixed"  # No movement allowed
    TRANSLATION_1D = "trans_1d"  # Translation along one axis
    TRANSLATION_2D = "trans_2d"  # Translation in a plane
    TRANSLATION_3D = "trans_3d"  # Free translation
    ROTATION_1D = "rot_1d"  # Rotation around one axis
    ROTATION_DISCRETE = "rot_discrete"  # Discrete rotation angles
    FULL_6D = "full_6d"  # Full 6 DoF

@dataclass
class ConnectionConstraint:
    """Defines constraints for a connection between parts"""
    connection_type: str  # Type of connection (dowel, screw, cam_lock, etc.)
    contact_point_a: np.ndarray  # Contact point on part A
    contact_point_b: np.ndarray  # Contact point on part B
    normal_a: np.ndarray  # Normal at contact point A
    normal_b: np.ndarray  # Normal at contact point B
    allowed_translation: DoFType
    allowed_rotation: DoFType
    translation_range: Optional[Tuple[float, float]] = None  # Min/max translation
    rotation_range: Optional[Tuple[float, float]] = None  # Min/max rotation angles
    discrete_angles: Optional[List[float]] = None  # For discrete rotations
    force_alignment: bool = True  # Whether to force normal alignment

@dataclass
class AssemblyState:
    """Current state of assembly"""
    assembled_parts: Dict[str, Dict]  # part_id -> {mesh, pose, connections_used}
    occupied_connections: Set[Tuple[str, int]]  # (part_id, connection_idx) pairs
    collision_grid: Optional[Any] = None  # Spatial acceleration structure
    assembly_sequence: List[Tuple[str, np.ndarray, np.ndarray]]  # (part_id, R, t)

class ConstraintEngine:
    """
    Engine for managing assembly constraints and valid pose generation
    Replaces LEGO's stud-based system with general geometric constraints
    """

    def __init__(
        self,
        assets_registry: Any,  # AssetsRegistry instance
        collision_threshold: float = 0.001,
        contact_threshold: float = 0.005,
        alignment_tolerance: float = 0.1,  # Radians
        use_physics: bool = False
    ):
        """
        Args:
            assets_registry: Registry containing part information
            collision_threshold: Minimum distance to consider collision
            contact_threshold: Maximum distance for valid contact
            alignment_tolerance: Tolerance for normal alignment
            use_physics: Whether to use physics simulation for stability
        """
        self.registry = assets_registry
        self.collision_threshold = collision_threshold
        self.contact_threshold = contact_threshold
        self.alignment_tolerance = alignment_tolerance
        self.use_physics = use_physics

        # Connection type configurations
        self.connection_configs = self._init_connection_configs()

        # Current assembly state
        self.assembly_state = AssemblyState(
            assembled_parts={},
            occupied_connections=set(),
            collision_grid=None,
            assembly_sequence=[]
        )

    def _init_connection_configs(self) -> Dict[str, Dict]:
        """Initialize configuration for different connection types"""
        configs = {
            'dowel': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.ROTATION_1D,
                'trans_range': (0, 0.05),  # Can slide along axis
                'rot_range': (0, 2 * np.pi),  # Full rotation around axis
                'requires_hole': True,
                'min_depth': 0.01
            },
            'screw': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.ROTATION_1D,
                'trans_range': (0, 0.03),
                'rot_range': (0, 10 * np.pi),  # Multiple rotations for screwing
                'requires_hole': True,
                'min_depth': 0.005
            },
            'cam_lock': {
                'translation': DoFType.FIXED,
                'rotation': DoFType.ROTATION_DISCRETE,
                'discrete_angles': [0, np.pi/2],  # 0° unlocked, 90° locked
                'requires_hole': True,
                'min_depth': 0.01
            },
            'snap': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.FIXED,
                'trans_range': (0, 0.01),  # Small movement until snap
                'requires_slot': True
            },
            'slot': {
                'translation': DoFType.TRANSLATION_2D,
                'rotation': DoFType.ROTATION_DISCRETE,
                'discrete_angles': [0, np.pi/2, np.pi, 3*np.pi/2],
                'requires_tab': True
            },
            'surface': {
                'translation': DoFType.TRANSLATION_2D,
                'rotation': DoFType.ROTATION_1D,
                'requires_flat': True,
                'min_contact_area': 0.001
            },
            'hinge': {
                'translation': DoFType.FIXED,
                'rotation': DoFType.ROTATION_1D,
                'rot_range': (0, np.pi),  # Typically 0-180 degrees
                'requires_axis': True
            }
        }
        return configs

    def get_valid_poses(
        self,
        part_id: str,
        base_assembly: Optional[AssemblyState] = None,
        manual_connections: Optional[List[Tuple[str, str, str]]] = None,
        connection_hints: Optional[List[Dict]] = None,
        max_candidates: int = 100
    ) -> List[Tuple[np.ndarray, np.ndarray, float, Optional[Dict]]]:
        """
        Generate valid poses for a part given current assembly state

        Args:
            part_id: ID of the part to place
            base_assembly: Current assembly state
            manual_connections: CRITICAL - List of (part1, part2, connection_type) from IKEA manual
            connection_hints: Additional connection hints (legacy)
            max_candidates: Maximum number of poses to return

        Returns: List of (R, t, score, metadata) tuples
                 metadata contains connection indices if connection-based pose
        """
        if base_assembly is None:
            base_assembly = self.assembly_state

        part_info = self.registry.get_part(part_id)
        if part_info is None:
            warnings.warn(f"Part {part_id} not found in registry")
            return []

        candidates = []

        # PRIORITY 1: Manual connections from IKEA dataset (highest priority)
        if manual_connections:
            manual_poses = self._generate_manual_connection_poses(
                part_id,
                part_info,
                base_assembly,
                manual_connections
            )
            candidates.extend(manual_poses)

        # Strategy 2: Connection-based pose generation (if no manual connections)
        elif connection_hints or part_info.connection_points:
            conn_poses = self._generate_connection_poses(
                part_info,
                base_assembly,
                connection_hints
            )
            candidates.extend(conn_poses)

        # Strategy 2: Surface contact pose generation
        surface_poses = self._generate_surface_poses(
            part_info,
            base_assembly
        )
        # Surface poses don't have connection metadata
        candidates.extend([(R, t, score, None) for R, t, score in surface_poses])

        # Strategy 3: Symmetry-aware variations
        # FIX: Check symmetry without accessing non-existent SymmetryType attribute
        if hasattr(part_info, 'symmetry') and part_info.symmetry:
            # Try to import SymmetryType from the actual location
            try:
                from ikea_mepnet_adapter.assets.registry import SymmetryType
                # Check if symmetry is not NONE
                if part_info.symmetry.symmetry_type != SymmetryType.NONE:
                    # Extract just R, t, score for symmetry generation, preserve metadata
                    top_candidates = candidates[:10]
                    base_poses = [(R, t, score) for R, t, score, _ in top_candidates]
                    sym_poses = self._generate_symmetry_poses(
                        part_info,
                        base_poses
                    )
                    # Symmetry poses inherit metadata from their base pose if available
                    for i, (R, t, score) in enumerate(sym_poses):
                        # Find corresponding base pose metadata
                        base_idx = i % len(top_candidates) if top_candidates else 0
                        metadata = top_candidates[base_idx][3] if base_idx < len(top_candidates) else None
                        candidates.append((R, t, score, metadata))
            except ImportError:
                # Fallback: check by string/value
                sym_type_str = str(part_info.symmetry.symmetry_type).lower()
                if sym_type_str != 'none' and sym_type_str != 'symmetrytype.none':
                    # Extract just R, t, score for symmetry generation, preserve metadata
                    top_candidates = candidates[:10]
                    base_poses = [(R, t, score) for R, t, score, _ in top_candidates]
                    sym_poses = self._generate_symmetry_poses(
                        part_info,
                        base_poses
                    )
                    # Symmetry poses inherit metadata from their base pose if available
                    for i, (R, t, score) in enumerate(sym_poses):
                        # Find corresponding base pose metadata
                        base_idx = i % len(top_candidates) if top_candidates else 0
                        metadata = top_candidates[base_idx][3] if base_idx < len(top_candidates) else None
                        candidates.append((R, t, score, metadata))

        # Filter by collision and stability
        valid_poses = []
        for R, t, score, metadata in candidates[:max_candidates]:
            if self._validate_pose(part_info, R, t, base_assembly):
                # Refine score based on stability and aesthetics
                refined_score = self._score_pose(part_info, R, t, base_assembly, score)
                valid_poses.append((R, t, refined_score, metadata))

        # Sort by score (third element)
        valid_poses.sort(key=lambda x: x[2], reverse=True)

        return valid_poses[:max_candidates]

    def _generate_manual_connection_poses(
        self,
        part_id: str,
        part_info: Any,
        assembly: AssemblyState,
        manual_connections: List[Tuple[str, str, str]]
    ) -> List[Tuple[np.ndarray, np.ndarray, float, Dict]]:
        """Generate poses based on IKEA manual connections (highest priority)

        Args:
            part_id: ID of the part to place
            part_info: Part information from registry
            assembly: Current assembly state
            manual_connections: List of (part1, part2, connection_type) from IKEA manual

        Returns:
            List of (R, t, score, metadata) where metadata contains connection indices
        """
        poses = []

        # Filter manual connections relevant to this part
        relevant_connections = [
            (p1, p2, conn_type) for p1, p2, conn_type in manual_connections
            if p1 == part_id or p2 == part_id
        ]

        for part1, part2, conn_type in relevant_connections:
            # Determine which part is already assembled
            if part1 == part_id:
                assembled_part_id = part2
                new_is_first = True
            else:
                assembled_part_id = part1
                new_is_first = False

            # Check if the other part is already assembled
            if assembled_part_id not in assembly.assembled_parts:
                continue  # Other part not assembled yet

            assembled_data = assembly.assembled_parts[assembled_part_id]
            assembled_info = self.registry.get_part(assembled_part_id)
            if assembled_info is None:
                continue

            # Find matching connection points based on type
            new_connections = self._find_connections_by_type(part_info, conn_type, is_source=new_is_first)
            assembled_connections = self._find_connections_by_type(assembled_info, conn_type, is_source=not new_is_first)

            # Generate poses for each valid connection pair
            for new_idx, new_conn in new_connections:
                for assembled_idx, assembled_conn in assembled_connections:
                    # Skip if already occupied
                    if (assembled_part_id, assembled_idx) in assembly.occupied_connections:
                        continue

                    # Generate alignment pose
                    R, t = self._align_connections(new_conn, assembled_conn, assembled_data['pose'])

                    # Apply connection-specific constraints
                    if conn_type in self.connection_configs:
                        config = self.connection_configs[conn_type]
                        variations = self._generate_dof_variations(
                            R, t, config, new_conn, assembled_conn
                        )

                        for R_var, t_var in variations:
                            # Higher score for manual connections
                            score = 2.0 * self._score_connection(
                                new_conn, assembled_conn, R_var, t_var, assembled_data['pose']
                            )

                            poses.append((R_var, t_var, score, {
                                'new_conn_idx': new_idx,
                                'assembled_part_id': assembled_part_id,
                                'assembled_conn_idx': assembled_idx,
                                'connection_type': conn_type,
                                'from_manual': True
                            }))
                    else:
                        # Unknown connection type - use basic alignment
                        score = 1.5  # Still prioritize manual connections
                        poses.append((R, t, score, {
                            'new_conn_idx': new_idx,
                            'assembled_part_id': assembled_part_id,
                            'assembled_conn_idx': assembled_idx,
                            'connection_type': conn_type,
                            'from_manual': True
                        }))

        return poses

    def _find_connections_by_type(
        self,
        part_info: Any,
        conn_type: str,
        is_source: bool
    ) -> List[Tuple[int, Any]]:
        """Find connection points of a specific type

        Args:
            part_info: Part information
            conn_type: Connection type string
            is_source: Whether this part is the source (e.g., has the dowel) or target (has the hole)

        Returns:
            List of (index, connection_point) tuples
        """
        connections = []
        if not hasattr(part_info, 'connection_points'):
            return connections

        # Map connection types to expected point types
        type_mappings = {
            'dowel': ('dowel', 'dowel_hole'),
            'screw': ('screw', 'screw_hole'),
            'cam_lock': ('cam_lock', 'cam_lock_hole'),
            'surface': ('surface', 'surface'),
            'snap': ('snap', 'snap_slot'),
            'hinge': ('hinge_pin', 'hinge_hole')
        }

        if conn_type in type_mappings:
            source_type, target_type = type_mappings[conn_type]
            expected_type = source_type if is_source else target_type

            for idx, conn in enumerate(part_info.connection_points):
                if hasattr(conn, 'connection_type'):
                    conn_type_str = conn.connection_type.value if hasattr(conn.connection_type, 'value') else str(conn.connection_type)
                    if conn_type_str == expected_type:
                        connections.append((idx, conn))

        return connections

    def _generate_connection_poses(
        self,
        part_info: Any,
        assembly: AssemblyState,
        hints: Optional[List[Dict]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, float, Dict]]:
        """Generate poses based on connection constraints (fallback when no manual connections)
        Returns: List of (R, t, score, metadata) where metadata contains connection indices"""
        poses = []

        # Get potential connection pairs
        for assembled_id, assembled_data in assembly.assembled_parts.items():
            assembled_info = self.registry.get_part(assembled_id)
            if assembled_info is None:
                continue

            # FIX: Track connection indices properly
            # Get connections with their actual indices
            if hasattr(assembled_info, 'connection_points'):
                assembled_connections = list(enumerate(assembled_info.connection_points))
            else:
                assembled_connections = []

            if hasattr(part_info, 'connection_points'):
                new_connections = list(enumerate(part_info.connection_points))
            else:
                new_connections = []

            # Check all connection pairs
            for new_idx, conn_new in new_connections:
                for assembled_idx, conn_existing in assembled_connections:
                    # Check if this specific connection is already occupied
                    if (assembled_id, assembled_idx) in assembly.occupied_connections:
                        continue

                    # Check if connections are compatible
                    if not self._connections_compatible(conn_new, conn_existing):
                        continue

                    # Generate pose to align connections
                    R, t = self._align_connections(
                        conn_new,
                        conn_existing,
                        assembled_data['pose']
                    )

                    # Apply connection-specific constraints
                    conn_type = conn_new.connection_type.value if hasattr(conn_new.connection_type, 'value') else str(conn_new.connection_type)
                    if conn_type in self.connection_configs:
                        config = self.connection_configs[conn_type]

                        # Generate variations based on allowed DoF
                        variations = self._generate_dof_variations(
                            R, t,
                            config,
                            conn_new,
                            conn_existing
                        )

                        for R_var, t_var in variations:
                            score = self._score_connection(
                                conn_new,
                                conn_existing,
                                R_var,
                                t_var,
                                assembled_data['pose']
                            )
                            # Include connection indices in the metadata
                            poses.append((R_var, t_var, score, {
                                'new_conn_idx': new_idx,
                                'assembled_part_id': assembled_id,
                                'assembled_conn_idx': assembled_idx
                            }))

        # Return with metadata preserved
        return poses

    def _connections_compatible(self, conn1: Any, conn2: Any) -> bool:
        """Check if two connections are compatible for joining"""
        # Check connection types match (e.g., dowel to hole, screw to threaded hole)
        if hasattr(conn1, 'connection_type') and hasattr(conn2, 'connection_type'):
            type1 = conn1.connection_type.value if hasattr(conn1.connection_type, 'value') else str(conn1.connection_type)
            type2 = conn2.connection_type.value if hasattr(conn2.connection_type, 'value') else str(conn2.connection_type)

            # Define compatible pairs
            compatible_pairs = [
                ('dowel', 'dowel_hole'),
                ('dowel_hole', 'dowel'),
                ('screw', 'screw_hole'),
                ('screw_hole', 'screw'),
                ('cam_lock', 'cam_lock_hole'),
                ('cam_lock_hole', 'cam_lock'),
                ('surface', 'surface'),
            ]

            # Check if types are compatible
            if (type1, type2) not in compatible_pairs and type1 != type2:
                return False

        # Check geometric compatibility (e.g., radius for cylindrical connections)
        if hasattr(conn1, 'radius') and hasattr(conn2, 'radius'):
            if conn1.radius and conn2.radius:
                # Allow some tolerance in radius matching
                radius_ratio = min(conn1.radius, conn2.radius) / max(conn1.radius, conn2.radius)
                if radius_ratio < 0.8:  # 20% tolerance
                    return False

        return True

    def _generate_surface_poses(
        self,
        part_info: Any,
        assembly: AssemblyState
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate poses based on surface contacts"""
        poses = []

        # Find flat surfaces in the assembly
        for assembled_id, assembled_data in assembly.assembled_parts.items():
            assembled_info = self.registry.get_part(assembled_id)
            if assembled_info is None:
                continue

            assembled_mesh = assembled_data['mesh']
            assembled_pose = assembled_data['pose']

            # Find mating surfaces
            for surface_new in self._get_flat_surfaces(part_info.mesh):
                for surface_existing in self._get_flat_surfaces(assembled_mesh):
                    # surface_existing is already in world coordinates (transformed mesh)
                    normal_new = surface_new['normal']
                    normal_existing = surface_existing['normal']  # Already in world space

                    if np.dot(normal_new, normal_existing) > -0.9:
                        continue

                    # Generate contact pose
                    R, t = self._align_surfaces(
                        surface_new,
                        surface_existing,
                        None  # surface_existing is already in world space
                    )

                    # Generate in-plane variations
                    for dx in np.linspace(-0.05, 0.05, 3):
                        for dy in np.linspace(-0.05, 0.05, 3):
                            t_var = t + R @ np.array([dx, dy, 0])
                            score = self._score_surface_contact(
                                surface_new,
                                surface_existing,
                                R,
                                t_var
                            )
                            poses.append((R, t_var, score))

        return poses

    def _generate_symmetry_poses(
        self,
        part_info: Any,
        base_poses: List[Tuple[np.ndarray, np.ndarray, float]]
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate additional poses using part symmetry"""
        sym_poses = []
        symmetry = part_info.symmetry

        # Import SymmetryType from registry if available
        try:
            from ikea_mepnet_adapter.assets.registry import SymmetryType
        except ImportError:
            SymmetryType = None

        for R, t, score in base_poses:
            if SymmetryType:
                # Use enum comparison
                if symmetry.symmetry_type == SymmetryType.PLANAR:
                    R_mirror = self._apply_planar_symmetry(R, symmetry.plane_normal)
                    sym_poses.append((R_mirror, t, score * 0.95))

                elif symmetry.symmetry_type in [SymmetryType.ROTATIONAL_2, SymmetryType.ROTATIONAL_4, SymmetryType.ROTATIONAL_N]:
                    n_fold = symmetry.n_fold or 2
                    for i in range(1, n_fold):
                        angle = 2 * np.pi * i / n_fold
                        R_rot = self._apply_rotation_symmetry(R, symmetry.axis, angle)
                        sym_poses.append((R_rot, t, score * 0.95))

                elif symmetry.symmetry_type == SymmetryType.CYLINDRICAL:
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        R_rot = self._apply_rotation_symmetry(R, symmetry.axis, angle)
                        sym_poses.append((R_rot, t, score * 0.95))
            else:
                # Fallback to string comparison
                sym_type_str = str(symmetry.symmetry_type).lower()
                if 'planar' in sym_type_str:
                    if hasattr(symmetry, 'plane_normal') and symmetry.plane_normal is not None:
                        R_mirror = self._apply_planar_symmetry(R, symmetry.plane_normal)
                        sym_poses.append((R_mirror, t, score * 0.95))
                elif 'rotational' in sym_type_str or 'rot_' in sym_type_str:
                    if hasattr(symmetry, 'axis') and symmetry.axis is not None:
                        n_fold = symmetry.n_fold or 2 if hasattr(symmetry, 'n_fold') else 2
                        for i in range(1, n_fold):
                            angle = 2 * np.pi * i / n_fold
                            R_rot = self._apply_rotation_symmetry(R, symmetry.axis, angle)
                            sym_poses.append((R_rot, t, score * 0.95))
                elif 'cylindrical' in sym_type_str:
                    if hasattr(symmetry, 'axis') and symmetry.axis is not None:
                        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                            R_rot = self._apply_rotation_symmetry(R, symmetry.axis, angle)
                            sym_poses.append((R_rot, t, score * 0.95))

        return sym_poses

    def _align_connections(
        self,
        conn_new: Any,
        conn_existing: Any,
        existing_pose: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transformation to align two connection points"""
        # Transform existing connection to world space
        pos_existing_world = existing_pose['R'] @ conn_existing.position + existing_pose['t']
        normal_existing_world = existing_pose['R'] @ conn_existing.normal

        # We want: R @ conn_new.position + t = pos_existing_world
        #         R @ conn_new.normal = -normal_existing_world (opposite normals)

        # Compute rotation to align normals
        R = self._rotation_align_vectors(conn_new.normal, -normal_existing_world)

        # Compute translation
        t = pos_existing_world - R @ conn_new.position

        return R, t

    def _align_surfaces(
        self,
        surface_new: Dict,
        surface_existing: Dict,
        existing_pose: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transformation to align two surfaces"""
        if existing_pose is not None:
            # Transform existing surface to world space
            center_existing_world = existing_pose['R'] @ surface_existing['center'] + existing_pose['t']
            normal_existing_world = existing_pose['R'] @ surface_existing['normal']
        else:
            # Already in world space
            center_existing_world = surface_existing['center']
            normal_existing_world = surface_existing['normal']

        # Align normals (opposite direction for mating)
        R = self._rotation_align_vectors(surface_new['normal'], -normal_existing_world)

        # Position surfaces in contact
        t = center_existing_world - R @ surface_new['center']

        # Offset slightly along normal to ensure contact
        t += normal_existing_world * self.contact_threshold

        return R, t

    def _validate_pose(
        self,
        part_info: Any,
        R: np.ndarray,
        t: np.ndarray,
        assembly: AssemblyState
    ) -> bool:
        """Validate a pose for collision and stability"""
        # Transform part mesh
        mesh_transformed = part_info.mesh.copy()
        mesh_transformed.apply_transform(
            trimesh.transformations.compose_matrix(
                translate=t,
                angles=Rotation.from_matrix(R).as_euler('xyz')
            )
        )

        # Check collision with existing parts
        for assembled_id, assembled_data in assembly.assembled_parts.items():
            assembled_mesh = assembled_data['mesh']

            # Use proper bounding box overlap check
            box1 = mesh_transformed.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            box2 = assembled_mesh.bounds

            # Check if bounding boxes overlap
            overlap = (
                box1[0][0] <= box2[1][0] and box1[1][0] >= box2[0][0] and
                box1[0][1] <= box2[1][1] and box1[1][1] >= box2[0][1] and
                box1[0][2] <= box2[1][2] and box1[1][2] >= box2[0][2]
            )

            if not overlap:
                continue

            # Correct collision detection
            closest_points, distances, _ = trimesh.proximity.closest_point(
                assembled_mesh,
                mesh_transformed.vertices
            )

            if np.min(distances) < self.collision_threshold:
                return False

        # Check stability
        if self.use_physics:
            if not self._check_stability_physics(mesh_transformed, assembly):
                return False
        else:
            if not self._check_stability_heuristic(mesh_transformed, assembly):
                return False

        return True

    def _score_connection(
        self,
        conn_new: Any,
        conn_existing: Any,
        R: np.ndarray,
        t: np.ndarray,
        existing_pose: Dict[str, np.ndarray]
    ) -> float:
        """Score a connection-based pose"""
        score = 1.0

        # Transform new connection to world coordinates
        pos_new_world = R @ conn_new.position + t
        normal_new_world = R @ conn_new.normal

        # Transform existing connection to world coordinates
        pos_existing_world = existing_pose['R'] @ conn_existing.position + existing_pose['t']
        normal_existing_world = existing_pose['R'] @ conn_existing.normal

        # Check normal alignment (should be opposite for mating)
        normal_alignment = np.dot(normal_new_world, normal_existing_world)
        # We want opposite normals, so closer to -1 is better
        score *= (1.0 - normal_alignment) / 2.0  # Maps [-1, 1] to [1, 0]

        # Check position alignment
        distance = np.linalg.norm(pos_new_world - pos_existing_world)
        score *= np.exp(-distance / self.contact_threshold)

        # Bonus for matching connection types
        if hasattr(conn_new, 'connection_type') and hasattr(conn_existing, 'connection_type'):
            type1 = conn_new.connection_type.value if hasattr(conn_new.connection_type, 'value') else str(conn_new.connection_type)
            type2 = conn_existing.connection_type.value if hasattr(conn_existing.connection_type, 'value') else str(conn_existing.connection_type)

            # Check if types are compatible
            if self._connection_types_match(type1, type2):
                score *= 1.2

        return score

    def _connection_types_match(self, type1: str, type2: str) -> bool:
        """Check if two connection types are compatible"""
        compatible_pairs = [
            ('dowel', 'dowel_hole'),
            ('screw', 'screw_hole'),
            ('cam_lock', 'cam_lock_hole'),
            ('surface', 'surface'),
        ]
        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs or type1 == type2

    def _score_surface_contact(
        self,
        surface_new: Dict,
        surface_existing: Dict,
        R: np.ndarray,
        t: np.ndarray
    ) -> float:
        """Score a surface contact pose"""
        score = 0.5  # Base score lower than connections

        # Check contact area
        area_ratio = min(surface_new['area'], surface_existing['area']) / \
                    max(surface_new['area'], surface_existing['area'])
        score *= area_ratio

        # Check alignment
        normal_new_world = R @ surface_new['normal']
        normal_existing = surface_existing['normal']

        # We want opposite normals for contact
        normal_alignment = -np.dot(normal_new_world, normal_existing)
        score *= max(0, normal_alignment)  # Only positive scores

        return score

    def update_assembly(
        self,
        part_id: str,
        R: np.ndarray,
        t: np.ndarray,
        connections_used: Optional[List[int]] = None,
        connection_metadata: Optional[Dict] = None
    ):
        """Update assembly state with new part

        Args:
            part_id: ID of the part being added
            R: Rotation matrix
            t: Translation vector
            connections_used: List of connection indices used on the new part
            connection_metadata: Dict containing assembled_part_id and assembled_conn_idx
        """
        part_info = self.registry.get_part(part_id)
        if part_info is None:
            raise ValueError(f"Part {part_id} not found")

        # Transform and add mesh
        mesh = part_info.mesh.copy()
        mesh.apply_transform(
            trimesh.transformations.compose_matrix(
                translate=t,
                angles=Rotation.from_matrix(R).as_euler('xyz')
            )
        )

        # Store part with connection tracking
        self.assembly_state.assembled_parts[part_id] = {
            'mesh': mesh,
            'pose': {'R': R, 't': t},
            'connections_used': connections_used or []
        }

        # Mark new part's connections as occupied
        if connections_used:
            for conn_idx in connections_used:
                self.assembly_state.occupied_connections.add((part_id, conn_idx))

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

        # Add to sequence
        self.assembly_state.assembly_sequence.append((part_id, R, t))

    def _check_stability_heuristic(
        self,
        mesh: trimesh.Trimesh,
        assembly: AssemblyState
    ) -> bool:
        """Simple heuristic stability check"""
        # Check if center of mass is over support
        com = mesh.center_mass

        # Project to ground plane
        com_2d = com[:2]

        # Find support polygon (simplified)
        support_points = []
        for assembled_data in assembly.assembled_parts.values():
            # Get bottom vertices
            vertices = assembled_data['mesh'].vertices
            bottom_vertices = vertices[vertices[:, 2] < 0.01]
            if len(bottom_vertices) > 0:
                support_points.extend(bottom_vertices[:, :2])

        if len(support_points) < 3:
            return True  # Not enough support to check

        # Check if COM is within support polygon
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(support_points)
            # Simplified check - proper point-in-polygon test needed
            return True
        except:
            return True

    def _check_stability_physics(
        self,
        mesh: trimesh.Trimesh,
        assembly: AssemblyState
    ) -> bool:
        """Physics-based stability check (placeholder)"""
        return True

    def _score_pose(
        self,
        part_info: Any,
        R: np.ndarray,
        t: np.ndarray,
        assembly: AssemblyState,
        base_score: float
    ) -> float:
        """Score a pose based on multiple criteria"""
        score = base_score

        # Penalize height (prefer lower positions for stability)
        height_penalty = t[2] * 0.1
        score -= height_penalty

        # Reward alignment with principal axes
        world_axes = np.eye(3)
        part_axes = R @ part_info.principal_axes
        alignment = np.sum([np.abs(np.dot(part_axes[i], world_axes[j]))
                            for i in range(3) for j in range(3)])
        score += alignment * 0.05

        # Reward compactness (parts close together)
        if assembly.assembled_parts:
            centers = [data['pose']['t'] for data in assembly.assembled_parts.values()]
            avg_center = np.mean(centers, axis=0)
            distance = np.linalg.norm(t - avg_center)
            score -= distance * 0.02

        return score

    def _generate_dof_variations(
        self,
        R: np.ndarray,
        t: np.ndarray,
        config: Dict,
        conn_new: Any,
        conn_existing: Any
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate pose variations based on allowed degrees of freedom"""
        variations = [(R, t)]  # Include original pose

        # Translation variations
        if config['translation'] == DoFType.TRANSLATION_1D:
            axis = R @ conn_new.normal
            trans_range = config.get('trans_range', (0, 0.01))
            for dist in np.linspace(*trans_range, 5):
                if dist != 0:
                    variations.append((R, t + axis * dist))

        elif config['translation'] == DoFType.TRANSLATION_2D:
            axis = R @ conn_new.normal
            # Generate orthogonal basis in plane
            if abs(axis[2]) < 0.9:
                u = np.cross(axis, [0, 0, 1])
            else:
                u = np.cross(axis, [1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(axis, u)

            for dx in np.linspace(-0.02, 0.02, 3):
                for dy in np.linspace(-0.02, 0.02, 3):
                    if dx != 0 or dy != 0:
                        variations.append((R, t + u * dx + v * dy))

        # Rotation variations
        if config['rotation'] == DoFType.ROTATION_1D:
            axis = conn_new.normal
            rot_range = config.get('rot_range', (0, np.pi/2))
            for angle in np.linspace(*rot_range, 5):
                if angle != 0:
                    R_rot = Rotation.from_rotvec(angle * axis).as_matrix()
                    variations.append((R @ R_rot, t))

        elif config['rotation'] == DoFType.ROTATION_DISCRETE:
            axis = conn_new.normal
            for angle in config.get('discrete_angles', []):
                if angle != 0:
                    R_rot = Rotation.from_rotvec(angle * axis).as_matrix()
                    variations.append((R @ R_rot, t))

        return variations

    def _get_flat_surfaces(self, mesh: trimesh.Trimesh, min_area: float = 0.001) -> List[Dict]:
        """Extract flat surfaces from mesh"""
        surfaces = []

        # Group faces by similar normals
        face_normals = mesh.face_normals
        face_areas = mesh.area_faces

        # Simple clustering by normal direction
        normal_groups = {}
        threshold = 0.1  # ~5.7 degrees

        for i, normal in enumerate(face_normals):
            matched = False
            for group_normal, group_indices in normal_groups.items():
                if np.linalg.norm(normal - np.array(group_normal)) < threshold:
                    group_indices.append(i)
                    matched = True
                    break

            if not matched:
                normal_groups[tuple(normal)] = [i]

        # Convert groups to surfaces
        for normal_tuple, indices in normal_groups.items():
            area = sum(face_areas[i] for i in indices)
            if area >= min_area:
                vertices = mesh.vertices[mesh.faces[indices].flatten()]
                center = vertices.mean(axis=0)
                surfaces.append({
                    'normal': np.array(normal_tuple),
                    'center': center,
                    'area': area,
                    'indices': indices
                })

        return surfaces

    def _rotation_align_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute rotation matrix to align v1 with v2"""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Handle parallel/antiparallel cases
        dot = np.dot(v1, v2)
        if np.abs(dot - 1) < 1e-6:
            return np.eye(3)
        if np.abs(dot + 1) < 1e-6:
            # Find perpendicular vector
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            return Rotation.from_rotvec(np.pi * perp).as_matrix()

        # General case
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))
        return Rotation.from_rotvec(angle * axis).as_matrix()

    def _apply_planar_symmetry(self, R: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
        """Apply planar symmetry transformation"""
        n = plane_normal / np.linalg.norm(plane_normal)
        H = np.eye(3) - 2 * np.outer(n, n)
        return H @ R

    def _apply_rotation_symmetry(
        self,
        R: np.ndarray,
        axis: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """Apply rotational symmetry transformation"""
        R_sym = Rotation.from_rotvec(angle * axis).as_matrix()
        return R_sym @ R

    def clear_assembly(self):
        """Clear the current assembly state"""
        self.assembly_state = AssemblyState(
            assembled_parts={},
            occupied_connections=set(),
            collision_grid=None,
            assembly_sequence=[]
        )

    def get_assembly_mesh(self) -> trimesh.Trimesh:
        """Get combined mesh of current assembly"""
        meshes = [data['mesh'] for data in self.assembly_state.assembled_parts.values()]
        if meshes:
            return trimesh.util.concatenate(meshes)
        return trimesh.Trimesh()

    def validate_assembly_sequence(
        self,
        sequence: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate an assembly sequence for feasibility
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        temp_assembly = AssemblyState(
            assembled_parts={},
            occupied_connections=set(),
            collision_grid=None,
            assembly_sequence=[]
        )

        for i, (part_id, R, t) in enumerate(sequence):
            part_info = self.registry.get_part(part_id)
            if part_info is None:
                errors.append(f"Step {i}: Part {part_id} not found")
                continue

            # Check collision
            if not self._validate_pose(part_info, R, t, temp_assembly):
                errors.append(f"Step {i}: Part {part_id} collision or unstable")
                continue

            # Add to temporary assembly
            mesh = part_info.mesh.copy()
            mesh.apply_transform(
                trimesh.transformations.compose_matrix(
                    translate=t,
                    angles=Rotation.from_matrix(R).as_euler('xyz')
                )
            )
            temp_assembly.assembled_parts[part_id] = {
                'mesh': mesh,
                'pose': {'R': R, 't': t},
                'connections_used': []
            }

        return len(errors) == 0, errors