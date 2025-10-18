"""
Constraint Engine for IKEA Assembly (COMPLETE FIXED VERSION)
Properly tracks connection metadata and bidirectional occupancy
"""

import numpy as np
import torch
import trimesh
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import itertools
from pathlib import Path
import warnings

class DoFType(Enum):
    """Degrees of Freedom types"""
    FIXED = "fixed"
    TRANSLATION_1D = "trans_1d"
    TRANSLATION_2D = "trans_2d"
    TRANSLATION_3D = "trans_3d"
    ROTATION_1D = "rot_1d"
    ROTATION_DISCRETE = "rot_discrete"
    FULL_6D = "full_6d"

@dataclass
class ConnectionConstraint:
    """Defines constraints for a connection between parts"""
    connection_type: str
    contact_point_a: np.ndarray
    contact_point_b: np.ndarray
    normal_a: np.ndarray
    normal_b: np.ndarray
    allowed_translation: DoFType
    allowed_rotation: DoFType
    translation_range: Optional[Tuple[float, float]] = None
    rotation_range: Optional[Tuple[float, float]] = None
    discrete_angles: Optional[List[float]] = None
    force_alignment: bool = True

@dataclass
class PoseCandidate:
    """Pose candidate with metadata"""
    R: np.ndarray
    t: np.ndarray
    score: float
    metadata: Dict[str, Any] = None  # Stores connection info

@dataclass
class AssemblyState:
    """Current state of assembly"""
    assembled_parts: Dict[str, Dict]  # part_id -> {mesh, pose, connections_used}
    occupied_connections: Set[Tuple[str, int]]  # (part_id, connection_idx) pairs
    collision_grid: Optional[Any] = None
    assembly_sequence: List[Tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)

class ConstraintEngine:
    """
    Engine for managing assembly constraints and valid pose generation
    """

    def __init__(
        self,
        assets_registry: Any,
        collision_threshold: float = 0.001,
        contact_threshold: float = 0.005,
        alignment_tolerance: float = 0.1,
        use_physics: bool = False
    ):
        self.registry = assets_registry
        self.collision_threshold = collision_threshold
        self.contact_threshold = contact_threshold
        self.alignment_tolerance = alignment_tolerance
        self.use_physics = use_physics

        self.connection_configs = self._init_connection_configs()

        self.assembly_state = AssemblyState(
            assembled_parts={},
            occupied_connections=set(),
            collision_grid=None,
            assembly_sequence=[]
        )
        # Cache recent pose candidates for backward compatibility lookups
        self._last_pose_candidates: List[PoseCandidate] = []

    def _init_connection_configs(self) -> Dict[str, Dict]:
        """Initialize configuration for different connection types"""
        configs = {
            'dowel': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.ROTATION_1D,
                'trans_range': (0, 0.05),
                'rot_range': (0, 2 * np.pi),
                'requires_hole': True,
                'min_depth': 0.01
            },
            'screw': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.ROTATION_1D,
                'trans_range': (0, 0.03),
                'rot_range': (0, 10 * np.pi),
                'requires_hole': True,
                'min_depth': 0.005
            },
            'cam_lock': {
                'translation': DoFType.FIXED,
                'rotation': DoFType.ROTATION_DISCRETE,
                'discrete_angles': [0, np.pi/2],
                'requires_hole': True,
                'min_depth': 0.01
            },
            'snap': {
                'translation': DoFType.TRANSLATION_1D,
                'rotation': DoFType.FIXED,
                'trans_range': (0, 0.01),
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
                'rot_range': (0, np.pi),
                'requires_axis': True
            }
        }
        return configs

    def get_valid_poses(
        self,
        part_id: str,
        base_assembly: Optional[AssemblyState] = None,
        connection_hints: Optional[List[Dict]] = None,
        max_candidates: int = 100,
        return_metadata: bool = False
    ) -> Union[List[PoseCandidate], List[Tuple[np.ndarray, np.ndarray, float]]]:
        """
        Generate valid poses for a part given current assembly state
        Returns:
            - List[PoseCandidate] if return_metadata is True
            - List of (R, t, score) tuples otherwise (legacy behaviour)
        """
        if base_assembly is None:
            base_assembly = self.assembly_state

        part_info = self.registry.get_part(part_id)
        if part_info is None:
            warnings.warn(f"Part {part_id} not found in registry")
            return []

        candidates = []

        # Strategy 1: Connection-based pose generation (with metadata)
        if connection_hints or (hasattr(part_info, 'connection_points') and part_info.connection_points):
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
        candidates.extend(surface_poses)

        # Strategy 3: Symmetry-aware variations
        if hasattr(part_info, 'symmetry') and part_info.symmetry:
            if self._has_nontrivial_symmetry(part_info.symmetry):
                sym_poses = self._generate_symmetry_poses(
                    part_info,
                    candidates[:10]
                )
                candidates.extend(sym_poses)

        # Filter by collision and stability
        valid_poses = []
        for candidate in candidates[:max_candidates]:
            if self._validate_pose(part_info, candidate.R, candidate.t, base_assembly):
                refined_score = self._score_pose(
                    part_info, candidate.R, candidate.t,
                    base_assembly, candidate.score
                )
                candidate.score = refined_score
                valid_poses.append(candidate)

        # Sort by score
        valid_poses.sort(key=lambda x: x.score, reverse=True)

        # Cache poses for potential metadata recovery during update_assembly
        self._last_pose_candidates = valid_poses[:max_candidates]

        if return_metadata:
            return valid_poses[:max_candidates]

        return [
            (candidate.R, candidate.t, candidate.score)
            for candidate in valid_poses[:max_candidates]
        ]

    def _has_nontrivial_symmetry(self, symmetry: Any) -> bool:
        """Check if part has non-trivial symmetry"""
        try:
            from assets.registry import SymmetryType
            if hasattr(symmetry, 'symmetry_type'):
                return symmetry.symmetry_type != SymmetryType.NONE
        except ImportError:
            pass

        # Fallback to string comparison
        if hasattr(symmetry, 'symmetry_type'):
            sym_str = str(symmetry.symmetry_type).lower()
            return 'none' not in sym_str
        return False

    def _generate_connection_poses(
        self,
        part_info: Any,
        assembly: AssemblyState,
        hints: Optional[List[Dict]] = None
    ) -> List[PoseCandidate]:
        """Generate poses based on connection constraints with metadata"""
        poses = []

        # Get connections with indices for the new part
        new_connections = []
        if hasattr(part_info, 'connection_points'):
            new_connections = list(enumerate(part_info.connection_points))

        # Check against each assembled part
        for assembled_id, assembled_data in assembly.assembled_parts.items():
            assembled_info = self.registry.get_part(assembled_id)
            if assembled_info is None:
                continue

            # Get connections with indices for the assembled part
            assembled_connections = []
            if hasattr(assembled_info, 'connection_points'):
                assembled_connections = list(enumerate(assembled_info.connection_points))

            # Try all connection pairs
            for new_idx, conn_new in new_connections:
                for assembled_idx, conn_existing in assembled_connections:
                    # Check if this connection is already occupied
                    if (assembled_id, assembled_idx) in assembly.occupied_connections:
                        continue

                    # Check compatibility
                    if not self._connections_compatible(conn_new, conn_existing):
                        continue

                    # Generate pose to align connections
                    R, t = self._align_connections(
                        conn_new,
                        conn_existing,
                        assembled_data['pose']
                    )

                    # Apply connection-specific constraints
                    conn_type = self._get_connection_type(conn_new)
                    if conn_type in self.connection_configs:
                        config = self.connection_configs[conn_type]

                        # Generate variations based on allowed DoF
                        variations = self._generate_dof_variations(
                            R, t, config, conn_new, conn_existing
                        )

                        for R_var, t_var in variations:
                            score = self._score_connection(
                                conn_new, conn_existing,
                                R_var, t_var,
                                assembled_data['pose']
                            )

                            # Create PoseCandidate with connection metadata
                            candidate = PoseCandidate(
                                R=R_var,
                                t=t_var,
                                score=score,
                                metadata={
                                    'type': 'connection',
                                    'new_part_id': part_info.part_id,
                                    'new_conn_idx': new_idx,
                                    'assembled_part_id': assembled_id,
                                    'assembled_conn_idx': assembled_idx
                                }
                            )
                            poses.append(candidate)

        return poses

    def _generate_surface_poses(
        self,
        part_info: Any,
        assembly: AssemblyState
    ) -> List[PoseCandidate]:
        """Generate poses based on surface contacts"""
        poses = []

        for assembled_id, assembled_data in assembly.assembled_parts.items():
            assembled_info = self.registry.get_part(assembled_id)
            if assembled_info is None:
                continue

            assembled_mesh = assembled_data['mesh']

            # Find mating surfaces
            for surface_new in self._get_flat_surfaces(part_info.mesh):
                for surface_existing in self._get_flat_surfaces(assembled_mesh):
                    normal_new = surface_new['normal']
                    normal_existing = surface_existing['normal']

                    if np.dot(normal_new, normal_existing) > -0.9:
                        continue

                    # Generate contact pose
                    R, t = self._align_surfaces(
                        surface_new, surface_existing, None
                    )

                    # Generate in-plane variations
                    for dx in np.linspace(-0.05, 0.05, 3):
                        for dy in np.linspace(-0.05, 0.05, 3):
                            t_var = t + R @ np.array([dx, dy, 0])
                            score = self._score_surface_contact(
                                surface_new, surface_existing, R, t_var
                            )

                            candidate = PoseCandidate(
                                R=R,
                                t=t_var,
                                score=score,
                                metadata={'type': 'surface'}
                            )
                            poses.append(candidate)

        return poses

    def _generate_symmetry_poses(
        self,
        part_info: Any,
        base_poses: List[PoseCandidate]
    ) -> List[PoseCandidate]:
        """Generate additional poses using part symmetry"""
        sym_poses = []
        symmetry = part_info.symmetry

        # Import SymmetryType if available
        try:
            from assets.registry import SymmetryType
            HAS_ENUM = True
        except ImportError:
            HAS_ENUM = False

        for base_candidate in base_poses:
            R, t, score = base_candidate.R, base_candidate.t, base_candidate.score

            if HAS_ENUM:
                if symmetry.symmetry_type == SymmetryType.PLANAR:
                    if hasattr(symmetry, 'plane_normal') and symmetry.plane_normal is not None:
                        R_mirror = self._apply_planar_symmetry(R, symmetry.plane_normal)
                        candidate = PoseCandidate(
                            R=R_mirror, t=t, score=score * 0.95,
                            metadata=base_candidate.metadata
                        )
                        sym_poses.append(candidate)

                elif hasattr(SymmetryType, 'ROTATIONAL_2') and symmetry.symmetry_type in [
                    SymmetryType.ROTATIONAL_2, SymmetryType.ROTATIONAL_4, SymmetryType.ROTATIONAL_N
                ]:
                    if hasattr(symmetry, 'axis') and symmetry.axis is not None:
                        n_fold = getattr(symmetry, 'n_fold', 2)
                        for i in range(1, n_fold):
                            angle = 2 * np.pi * i / n_fold
                            R_rot = self._apply_rotation_symmetry(R, symmetry.axis, angle)
                            candidate = PoseCandidate(
                                R=R_rot, t=t, score=score * 0.95,
                                metadata=base_candidate.metadata
                            )
                            sym_poses.append(candidate)
            else:
                # String-based fallback
                sym_type_str = str(symmetry.symmetry_type).lower()
                if 'planar' in sym_type_str:
                    if hasattr(symmetry, 'plane_normal') and symmetry.plane_normal is not None:
                        R_mirror = self._apply_planar_symmetry(R, symmetry.plane_normal)
                        candidate = PoseCandidate(
                            R=R_mirror, t=t, score=score * 0.95,
                            metadata=base_candidate.metadata
                        )
                        sym_poses.append(candidate)

        return sym_poses

    def update_assembly(
        self,
        part_id: str,
        pose_or_R: Union[PoseCandidate, np.ndarray],
        t: Optional[np.ndarray] = None,
        connections_used: Optional[List[int]] = None,
        connection_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update assembly state with new part.

        Supports both PoseCandidate-based invocations and the legacy
        (part_id, R, t, connections_used) signature.
        """
        part_info = self.registry.get_part(part_id)
        if part_info is None:
            raise ValueError(f"Part {part_id} not found")

        metadata: Optional[Dict[str, Any]] = None
        pose_R: np.ndarray
        pose_t: np.ndarray

        if isinstance(pose_or_R, PoseCandidate):
            candidate = pose_or_R
            pose_R = candidate.R
            pose_t = candidate.t
            metadata = candidate.metadata if isinstance(candidate.metadata, dict) else None
            used_indices = list(connections_used) if connections_used else []
        else:
            if t is None:
                raise ValueError("update_assembly requires translation vector when PoseCandidate is not provided")
            pose_R = pose_or_R
            pose_t = t
            metadata = connection_metadata if isinstance(connection_metadata, dict) else None
            if metadata is None:
                metadata = self._find_candidate_metadata(part_id, pose_R, pose_t)
            used_indices = list(connections_used) if connections_used else []

        used_index_set = set(used_indices)

        if metadata and metadata.get('type') == 'connection':
            new_conn_idx = metadata.get('new_conn_idx')
            if new_conn_idx is not None:
                used_index_set.add(new_conn_idx)

            assembled_part_id = metadata.get('assembled_part_id')
            assembled_conn_idx = metadata.get('assembled_conn_idx')
            if assembled_part_id and assembled_conn_idx is not None:
                self.assembly_state.occupied_connections.add(
                    (assembled_part_id, assembled_conn_idx)
                )

        # Transform and add mesh
        mesh = part_info.mesh.copy()
        mesh.apply_transform(
            trimesh.transformations.compose_matrix(
                translate=pose_t,
                angles=Rotation.from_matrix(pose_R).as_euler('xyz')
            )
        )

        for conn_idx in used_index_set:
            self.assembly_state.occupied_connections.add((part_id, conn_idx))

        connections_used_list = sorted(used_index_set)
        connected_to = metadata.get('assembled_part_id') if metadata else None

        # Store part with all metadata
        self.assembly_state.assembled_parts[part_id] = {
            'mesh': mesh,
            'pose': {'R': pose_R, 't': pose_t},
            'connections_used': connections_used_list,
            'connected_to': connected_to
        }

        # Add to sequence
        self.assembly_state.assembly_sequence.append((part_id, pose_R, pose_t))

    def _find_candidate_metadata(
        self,
        part_id: str,
        R: np.ndarray,
        t: np.ndarray,
        atol: float = 1e-5
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover metadata for a previously generated pose."""
        for candidate in self._last_pose_candidates:
            if candidate.metadata and candidate.metadata.get('new_part_id') != part_id:
                continue
            if np.allclose(candidate.R, R, atol=atol) and np.allclose(candidate.t, t, atol=atol):
                return candidate.metadata if isinstance(candidate.metadata, dict) else None
        return None

    def _get_connection_type(self, conn: Any) -> str:
        """Extract connection type as string"""
        if hasattr(conn, 'connection_type'):
            if hasattr(conn.connection_type, 'value'):
                return conn.connection_type.value
            return str(conn.connection_type)
        return 'unknown'

    def _connections_compatible(self, conn1: Any, conn2: Any) -> bool:
        """Check if two connections are compatible"""
        type1 = self._get_connection_type(conn1)
        type2 = self._get_connection_type(conn2)

        compatible_pairs = [
            ('dowel', 'dowel_hole'), ('dowel_hole', 'dowel'),
            ('screw', 'screw_hole'), ('screw_hole', 'screw'),
            ('cam_lock', 'cam_lock_hole'), ('cam_lock_hole', 'cam_lock'),
            ('surface', 'surface')
        ]

        if (type1, type2) in compatible_pairs or type1 == type2:
            # Check geometric compatibility
            if hasattr(conn1, 'radius') and hasattr(conn2, 'radius'):
                if conn1.radius and conn2.radius:
                    radius_ratio = min(conn1.radius, conn2.radius) / max(conn1.radius, conn2.radius)
                    return radius_ratio >= 0.8
            return True

        return False

    def _align_connections(
        self,
        conn_new: Any,
        conn_existing: Any,
        existing_pose: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transformation to align two connection points"""
        pos_existing_world = existing_pose['R'] @ conn_existing.position + existing_pose['t']
        normal_existing_world = existing_pose['R'] @ conn_existing.normal

        R = self._rotation_align_vectors(conn_new.normal, -normal_existing_world)
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
            center_existing_world = existing_pose['R'] @ surface_existing['center'] + existing_pose['t']
            normal_existing_world = existing_pose['R'] @ surface_existing['normal']
        else:
            center_existing_world = surface_existing['center']
            normal_existing_world = surface_existing['normal']

        R = self._rotation_align_vectors(surface_new['normal'], -normal_existing_world)
        t = center_existing_world - R @ surface_new['center']
        t += normal_existing_world * self.contact_threshold

        return R, t

    def _generate_dof_variations(
        self,
        R: np.ndarray,
        t: np.ndarray,
        config: Dict,
        conn_new: Any,
        conn_existing: Any
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate pose variations based on allowed degrees of freedom"""
        variations = [(R, t)]

        if config['translation'] == DoFType.TRANSLATION_1D:
            axis = R @ conn_new.normal
            trans_range = config.get('trans_range', (0, 0.01))
            for dist in np.linspace(*trans_range, 5):
                if dist != 0:
                    variations.append((R, t + axis * dist))

        elif config['translation'] == DoFType.TRANSLATION_2D:
            axis = R @ conn_new.normal
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

    def _validate_pose(
        self,
        part_info: Any,
        R: np.ndarray,
        t: np.ndarray,
        assembly: AssemblyState
    ) -> bool:
        """Validate a pose for collision and stability"""
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

            box1 = mesh_transformed.bounds
            box2 = assembled_mesh.bounds

            # Check bounding box overlap
            overlap = (
                box1[0][0] <= box2[1][0] and box1[1][0] >= box2[0][0] and
                box1[0][1] <= box2[1][1] and box1[1][1] >= box2[0][1] and
                box1[0][2] <= box2[1][2] and box1[1][2] >= box2[0][2]
            )

            if not overlap:
                continue

            # Detailed collision check
            closest_points, distances, _ = trimesh.proximity.closest_point(
                assembled_mesh,
                mesh_transformed.vertices
            )

            if np.min(distances) < self.collision_threshold:
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

        pos_new_world = R @ conn_new.position + t
        normal_new_world = R @ conn_new.normal

        pos_existing_world = existing_pose['R'] @ conn_existing.position + existing_pose['t']
        normal_existing_world = existing_pose['R'] @ conn_existing.normal

        # Normal alignment (opposite is better)
        normal_alignment = np.dot(normal_new_world, normal_existing_world)
        score *= (1.0 - normal_alignment) / 2.0

        # Position alignment
        distance = np.linalg.norm(pos_new_world - pos_existing_world)
        score *= np.exp(-distance / self.contact_threshold)

        return score

    def _score_surface_contact(
        self,
        surface_new: Dict,
        surface_existing: Dict,
        R: np.ndarray,
        t: np.ndarray
    ) -> float:
        """Score a surface contact pose"""
        score = 0.5

        area_ratio = min(surface_new['area'], surface_existing['area']) / \
                    max(surface_new['area'], surface_existing['area'])
        score *= area_ratio

        normal_new_world = R @ surface_new['normal']
        normal_alignment = -np.dot(normal_new_world, surface_existing['normal'])
        score *= max(0, normal_alignment)

        return score

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

        # Height penalty
        score -= t[2] * 0.1

        # Principal axes alignment
        if hasattr(part_info, 'principal_axes'):
            world_axes = np.eye(3)
            part_axes = R @ part_info.principal_axes
            alignment = np.sum([np.abs(np.dot(part_axes[i], world_axes[j]))
                              for i in range(3) for j in range(3)])
            score += alignment * 0.05

        # Compactness
        if assembly.assembled_parts:
            centers = [data['pose']['t'] for data in assembly.assembled_parts.values()]
            avg_center = np.mean(centers, axis=0)
            distance = np.linalg.norm(t - avg_center)
            score -= distance * 0.02

        return score

    def _get_flat_surfaces(self, mesh: trimesh.Trimesh, min_area: float = 0.001) -> List[Dict]:
        """Extract flat surfaces from mesh"""
        surfaces = []
        face_normals = mesh.face_normals
        face_areas = mesh.area_faces

        normal_groups = {}
        threshold = 0.1

        for i, normal in enumerate(face_normals):
            matched = False
            for group_normal, group_indices in normal_groups.items():
                if np.linalg.norm(normal - np.array(group_normal)) < threshold:
                    group_indices.append(i)
                    matched = True
                    break
            if not matched:
                normal_groups[tuple(normal)] = [i]

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

        dot = np.dot(v1, v2)
        if np.abs(dot - 1) < 1e-6:
            return np.eye(3)
        if np.abs(dot + 1) < 1e-6:
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            return Rotation.from_rotvec(np.pi * perp).as_matrix()

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
        """Validate an assembly sequence for feasibility"""
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

            if not self._validate_pose(part_info, R, t, temp_assembly):
                errors.append(f"Step {i}: Part {part_id} collision or unstable")
                continue

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
