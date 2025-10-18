"""
Assets Registry for IKEA parts
Manages 3D meshes, symmetries, and connection points
"""

import os
import json
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import pickle
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import warnings
from dataclasses import dataclass, asdict
from enum import Enum

class SymmetryType(Enum):
    """Types of symmetry for parts"""
    NONE = "none"
    PLANAR = "planar"  # Mirror symmetry
    ROTATIONAL_2 = "rot_2"  # 180° rotational
    ROTATIONAL_4 = "rot_4"  # 90° rotational
    ROTATIONAL_N = "rot_n"  # N-fold rotational
    CYLINDRICAL = "cylindrical"  # Continuous rotation around axis
    SPHERICAL = "spherical"  # All rotations

class ConnectionType(Enum):
    """Types of connections in IKEA furniture"""
    DOWEL = "dowel"  # Wooden pin connections
    SCREW = "screw"  # Screw connections
    CAM_LOCK = "cam_lock"  # Cam and bolt system
    SNAP = "snap"  # Snap-fit connections
    SLOT = "slot"  # Slot and tab connections
    SURFACE = "surface"  # Surface contact (glue, friction)
    HINGE = "hinge"  # Rotating connections

@dataclass
class ConnectionPoint:
    """Represents a connection point on a part"""
    position: np.ndarray  # 3D position in part coordinates
    normal: np.ndarray  # Normal direction
    connection_type: ConnectionType
    radius: Optional[float] = None  # For cylindrical connections
    allowed_rotations: Optional[List[float]] = None  # Allowed rotation angles
    paired_with: Optional[str] = None  # ID of paired connection point

@dataclass
class PartSymmetry:
    """Symmetry information for a part"""
    symmetry_type: SymmetryType
    axis: Optional[np.ndarray] = None  # Symmetry axis (for rotational/cylindrical)
    plane_normal: Optional[np.ndarray] = None  # Plane normal (for planar)
    n_fold: Optional[int] = None  # N for N-fold symmetry
    continuous: bool = False  # True for continuous symmetries

@dataclass
class PartInfo:
    """Complete information about a part"""
    part_id: str
    mesh: trimesh.Trimesh
    symmetry: PartSymmetry
    connection_points: List[ConnectionPoint]
    bbox: np.ndarray  # Bounding box [min, max]
    centroid: np.ndarray
    principal_axes: np.ndarray  # PCA axes
    volume: float
    surface_area: float
    category: Optional[str] = None  # Part category (e.g., "leg", "panel", "connector")

class AssetsRegistry:
    """
    Central registry for IKEA part assets
    Handles mesh loading, symmetry detection, and connection point extraction
    """

    def __init__(
        self,
        parts_dir: str,
        cache_dir: Optional[str] = None,
        auto_detect: bool = True,
        load_cache: bool = True
    ):
        """
        Args:
            parts_dir: Directory containing part meshes
            cache_dir: Directory for caching processed data
            auto_detect: Whether to auto-detect symmetries and connections
            load_cache: Whether to load from cache if available
        """
        self.parts_dir = Path(parts_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.parts_dir / ".cache"
        self.auto_detect = auto_detect
        self.load_cache = load_cache

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Registry storage
        self.parts: Dict[str, PartInfo] = {}
        self.symmetry_groups: Dict[str, Set[str]] = {}  # Group parts with same symmetry
        self.connection_graph: Dict[str, List[str]] = {}  # Compatible connections

        # Load or build registry
        if load_cache and self._load_cache():
            print(f"Loaded {len(self.parts)} parts from cache")
        else:
            self._build_registry()
            if cache_dir:
                self._save_cache()

    def _load_cache(self) -> bool:
        """Load cached registry data"""
        cache_file = self.cache_dir / "registry.pkl"
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.parts = data['parts']
                self.symmetry_groups = data['symmetry_groups']
                self.connection_graph = data['connection_graph']
                return True
        except Exception as e:
            warnings.warn(f"Failed to load cache: {e}")
            return False

    def _save_cache(self):
        """Save registry data to cache"""
        cache_file = self.cache_dir / "registry.pkl"
        data = {
            'parts': self.parts,
            'symmetry_groups': self.symmetry_groups,
            'connection_graph': self.connection_graph
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _build_registry(self):
        """Build the registry by processing all part meshes"""
        print(f"Building registry from {self.parts_dir}")

        # Find all mesh files
        mesh_files = list(self.parts_dir.glob("*.obj"))
        mesh_files.extend(self.parts_dir.glob("*.stl"))
        mesh_files.extend(self.parts_dir.glob("*.ply"))

        for mesh_file in mesh_files:
            part_id = mesh_file.stem
            try:
                part_info = self._process_part(part_id, mesh_file)
                self.parts[part_id] = part_info

                # Add to symmetry groups
                sym_key = self._get_symmetry_key(part_info.symmetry)
                if sym_key not in self.symmetry_groups:
                    self.symmetry_groups[sym_key] = set()
                self.symmetry_groups[sym_key].add(part_id)

            except Exception as e:
                warnings.warn(f"Failed to process {part_id}: {e}")

        print(f"Processed {len(self.parts)} parts")

    def _process_part(self, part_id: str, mesh_path: Path) -> PartInfo:
        """Process a single part mesh"""
        # Load mesh
        mesh = trimesh.load(str(mesh_path), force='mesh')

        # Normalize mesh
        mesh.vertices -= mesh.centroid
        scale = 1.0 / np.max(np.abs(mesh.vertices))
        mesh.apply_scale(scale)

        # Compute basic properties
        bbox = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        centroid = mesh.centroid
        volume = mesh.volume
        surface_area = mesh.area

        # PCA for principal axes
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        principal_axes = pca.components_

        # Detect symmetry
        symmetry = self._detect_symmetry(mesh) if self.auto_detect else PartSymmetry(SymmetryType.NONE)

        # Extract connection points
        connection_points = self._extract_connection_points(mesh) if self.auto_detect else []

        # Infer category from shape
        category = self._infer_category(mesh, principal_axes)

        return PartInfo(
            part_id=part_id,
            mesh=mesh,
            symmetry=symmetry,
            connection_points=connection_points,
            bbox=bbox,
            centroid=centroid,
            principal_axes=principal_axes,
            volume=volume,
            surface_area=surface_area,
            category=category
        )

    def _detect_symmetry(self, mesh: trimesh.Trimesh) -> PartSymmetry:
        """
        Detect symmetry properties of a mesh
        Uses geometric heuristics and sampling
        """
        # Get principal axes from PCA
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        axes = pca.components_

        # Check for planar symmetry along principal planes
        for axis in axes:
            if self._check_planar_symmetry(mesh, axis):
                return PartSymmetry(
                    symmetry_type=SymmetryType.PLANAR,
                    plane_normal=axis
                )

        # Check for rotational symmetry around principal axes
        for axis in axes:
            n_fold = self._check_rotational_symmetry(mesh, axis)
            if n_fold == -1:  # Continuous
                return PartSymmetry(
                    symmetry_type=SymmetryType.CYLINDRICAL,
                    axis=axis,
                    continuous=True
                )
            elif n_fold == 4:
                return PartSymmetry(
                    symmetry_type=SymmetryType.ROTATIONAL_4,
                    axis=axis,
                    n_fold=4
                )
            elif n_fold == 2:
                return PartSymmetry(
                    symmetry_type=SymmetryType.ROTATIONAL_2,
                    axis=axis,
                    n_fold=2
                )

        return PartSymmetry(symmetry_type=SymmetryType.NONE)

    def _check_planar_symmetry(self, mesh: trimesh.Trimesh, normal: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if mesh has planar symmetry across a plane"""
        vertices = mesh.vertices

        # Mirror vertices across plane
        d = -np.dot(vertices, normal)
        mirrored = vertices - 2 * np.outer(d, normal)

        # Check if mirrored vertices match original
        # Use KDTree for efficient nearest neighbor search
        tree = trimesh.proximity.ProximityQuery(mesh)
        distances, _ = tree.vertex(mirrored)

        return np.mean(distances) < threshold

    def _check_rotational_symmetry(self, mesh: trimesh.Trimesh, axis: np.ndarray, threshold: float = 0.01) -> int:
        """
        Check rotational symmetry around an axis
        Returns: n_fold (2, 3, 4, 6, etc.) or -1 for continuous
        """
        vertices = mesh.vertices

        # Test common n-fold symmetries
        for n in [2, 3, 4, 6, 8]:
            angle = 360 / n
            is_symmetric = True

            for i in range(1, n):
                # Rotate vertices
                rotation = Rotation.from_rotvec(np.radians(angle * i) * axis)
                rotated = rotation.apply(vertices)

                # Check match
                tree = trimesh.proximity.ProximityQuery(mesh)
                distances, _ = tree.vertex(rotated)

                if np.mean(distances) > threshold:
                    is_symmetric = False
                    break

            if is_symmetric:
                return n

        # Check for continuous symmetry (cylinder-like)
        # Test with random angles
        for _ in range(5):
            angle = np.random.uniform(0, 360)
            rotation = Rotation.from_rotvec(np.radians(angle) * axis)
            rotated = rotation.apply(vertices)

            tree = trimesh.proximity.ProximityQuery(mesh)
            distances, _ = tree.vertex(rotated)

            if np.mean(distances) > threshold:
                return 0  # No symmetry

        return -1  # Continuous symmetry

    def _extract_connection_points(self, mesh: trimesh.Trimesh) -> List[ConnectionPoint]:
        """
        Extract potential connection points from mesh geometry
        Looks for holes, cylindrical features, flat surfaces, etc.
        """
        connection_points = []

        # Find cylindrical holes (dowels, screws)
        holes = self._find_cylindrical_holes(mesh)
        for hole in holes:
            connection_points.append(ConnectionPoint(
                position=hole['center'],
                normal=hole['axis'],
                connection_type=ConnectionType.DOWEL if hole['radius'] > 0.003 else ConnectionType.SCREW,
                radius=hole['radius']
            ))

        # Find flat contact surfaces
        surfaces = self._find_flat_surfaces(mesh)
        for surface in surfaces:
            # Sample points on large flat surfaces
            if surface['area'] > 0.01:  # Threshold for significant surfaces
                connection_points.append(ConnectionPoint(
                    position=surface['center'],
                    normal=surface['normal'],
                    connection_type=ConnectionType.SURFACE
                ))

        # Find slots and tabs
        slots = self._find_slots(mesh)
        for slot in slots:
            connection_points.append(ConnectionPoint(
                position=slot['center'],
                normal=slot['normal'],
                connection_type=ConnectionType.SLOT
            ))

        return connection_points

    def _find_cylindrical_holes(self, mesh: trimesh.Trimesh, min_radius: float = 0.002, max_radius: float = 0.02) -> List[Dict]:
        """Find cylindrical holes in the mesh"""
        holes = []

        # Use mesh edges and face normals to detect cylindrical features
        # This is a simplified approach - more sophisticated methods exist

        # Group faces by similar normals (potential cylinder sides)
        face_normals = mesh.face_normals
        edges = mesh.edges_unique

        # Find edge loops that might form holes
        # (This is a placeholder - actual implementation would be more complex)

        return holes

    def _find_flat_surfaces(self, mesh: trimesh.Trimesh, angle_threshold: float = 5.0) -> List[Dict]:
        """Find flat surfaces in the mesh"""
        surfaces = []

        # Group faces by similar normals
        face_normals = mesh.face_normals
        face_areas = mesh.area_faces

        # Cluster faces with similar normals
        unique_normals = []
        normal_groups = []

        for i, normal in enumerate(face_normals):
            found = False
            for j, unique_normal in enumerate(unique_normals):
                angle = np.degrees(np.arccos(np.clip(np.dot(normal, unique_normal), -1, 1)))
                if angle < angle_threshold:
                    normal_groups[j].append(i)
                    found = True
                    break

            if not found:
                unique_normals.append(normal)
                normal_groups.append([i])

        # Process each group
        for normal, group in zip(unique_normals, normal_groups):
            if len(group) > 10:  # Minimum number of faces
                area = sum(face_areas[i] for i in group)
                vertices_in_group = mesh.vertices[mesh.faces[group].flatten()]
                center = vertices_in_group.mean(axis=0)

                surfaces.append({
                    'normal': normal,
                    'center': center,
                    'area': area,
                    'face_indices': group
                })

        return surfaces

    def _find_slots(self, mesh: trimesh.Trimesh) -> List[Dict]:
        """Find slot features in the mesh"""
        slots = []
        # Placeholder for slot detection logic
        # Would involve finding rectangular openings, edge analysis, etc.
        return slots

    def _infer_category(self, mesh: trimesh.Trimesh, principal_axes: np.ndarray) -> str:
        """Infer part category from shape characteristics"""
        # Get aspect ratios along principal axes
        bbox_sizes = mesh.bounding_box.extents
        aspect_ratios = bbox_sizes / bbox_sizes.max()

        # Simple heuristics
        if aspect_ratios[0] > 0.8 and aspect_ratios[1] > 0.8 and aspect_ratios[2] < 0.2:
            return "panel"  # Flat panel
        elif aspect_ratios[0] < 0.2 and aspect_ratios[1] < 0.2 and aspect_ratios[2] > 0.8:
            return "rod"  # Long thin rod
        elif np.min(aspect_ratios) > 0.7:
            return "block"  # Roughly cubic
        elif aspect_ratios[2] > 0.7 and max(aspect_ratios[:2]) < 0.3:
            return "leg"  # Furniture leg (tall and thin)
        else:
            return "generic"

    def _get_symmetry_key(self, symmetry: PartSymmetry) -> str:
        """Generate a key for grouping parts with similar symmetry"""
        if symmetry.symmetry_type == SymmetryType.NONE:
            return "none"
        elif symmetry.symmetry_type == SymmetryType.PLANAR:
            # Round plane normal to avoid float precision issues
            normal_rounded = np.round(symmetry.plane_normal, 2)
            return f"planar_{normal_rounded[0]:.2f}_{normal_rounded[1]:.2f}_{normal_rounded[2]:.2f}"
        elif symmetry.symmetry_type in [SymmetryType.ROTATIONAL_2, SymmetryType.ROTATIONAL_4]:
            axis_rounded = np.round(symmetry.axis, 2)
            return f"{symmetry.symmetry_type.value}_{axis_rounded[0]:.2f}_{axis_rounded[1]:.2f}_{axis_rounded[2]:.2f}"
        else:
            return symmetry.symmetry_type.value

    def get_part(self, part_id: str) -> Optional[PartInfo]:
        """Get part information by ID"""
        return self.parts.get(part_id)

    def get_similar_parts(self, part_id: str, by: str = "symmetry") -> List[str]:
        """Get parts similar to the given part"""
        if part_id not in self.parts:
            return []

        part = self.parts[part_id]

        if by == "symmetry":
            sym_key = self._get_symmetry_key(part.symmetry)
            return list(self.symmetry_groups.get(sym_key, set()) - {part_id})
        elif by == "category":
            return [pid for pid, p in self.parts.items()
                   if p.category == part.category and pid != part_id]
        elif by == "volume":
            # Find parts with similar volume (within 10%)
            target_volume = part.volume
            return [pid for pid, p in self.parts.items()
                   if abs(p.volume - target_volume) / target_volume < 0.1 and pid != part_id]
        else:
            return []

    def get_connection_candidates(
        self,
        part_a: str,
        part_b: str,
        connection_type: Optional[ConnectionType] = None
    ) -> List[Tuple[ConnectionPoint, ConnectionPoint]]:
        """
        Get possible connections between two parts
        Returns list of (connection_point_a, connection_point_b) pairs
        """
        if part_a not in self.parts or part_b not in self.parts:
            return []

        connections_a = self.parts[part_a].connection_points
        connections_b = self.parts[part_b].connection_points

        candidates = []

        for conn_a in connections_a:
            for conn_b in connections_b:
                # Check if connection types are compatible
                if connection_type and conn_a.connection_type != connection_type:
                    continue

                # Check geometric compatibility
                # Normals should be opposite for mating
                if np.dot(conn_a.normal, conn_b.normal) < -0.7:  # ~45 degree tolerance
                    # Check radius compatibility for cylindrical connections
                    if conn_a.radius and conn_b.radius:
                        if abs(conn_a.radius - conn_b.radius) / max(conn_a.radius, conn_b.radius) < 0.1:
                            candidates.append((conn_a, conn_b))
                    else:
                        candidates.append((conn_a, conn_b))

        return candidates

    def export_part_info(self, part_id: str, output_path: str):
        """Export part information to JSON"""
        if part_id not in self.parts:
            raise ValueError(f"Part {part_id} not found")

        part = self.parts[part_id]

        # Convert to serializable format
        info = {
            'part_id': part.part_id,
            'category': part.category,
            'volume': float(part.volume),
            'surface_area': float(part.surface_area),
            'bbox': part.bbox.tolist(),
            'centroid': part.centroid.tolist(),
            'principal_axes': part.principal_axes.tolist(),
            'symmetry': {
                'type': part.symmetry.symmetry_type.value,
                'axis': part.symmetry.axis.tolist() if part.symmetry.axis is not None else None,
                'plane_normal': part.symmetry.plane_normal.tolist() if part.symmetry.plane_normal is not None else None,
                'n_fold': part.symmetry.n_fold,
                'continuous': part.symmetry.continuous
            },
            'connection_points': [
                {
                    'position': conn.position.tolist(),
                    'normal': conn.normal.tolist(),
                    'type': conn.connection_type.value,
                    'radius': conn.radius
                }
                for conn in part.connection_points
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)

    def visualize_part(self, part_id: str, show_connections: bool = True, show_axes: bool = True):
        """Visualize part with connection points and symmetry axes"""
        if part_id not in self.parts:
            raise ValueError(f"Part {part_id} not found")

        part = self.parts[part_id]
        scene = trimesh.Scene()

        # Add mesh
        scene.add_geometry(part.mesh)

        # Add connection points
        if show_connections:
            for conn in part.connection_points:
                # Add sphere at connection point
                sphere = trimesh.creation.uv_sphere(radius=0.005)
                sphere.apply_translation(conn.position)

                # Color by connection type
                colors = {
                    ConnectionType.DOWEL: [1, 0, 0, 1],  # Red
                    ConnectionType.SCREW: [0, 1, 0, 1],  # Green
                    ConnectionType.SURFACE: [0, 0, 1, 1],  # Blue
                    ConnectionType.SLOT: [1, 1, 0, 1],  # Yellow
                }
                sphere.visual.vertex_colors = colors.get(conn.connection_type, [0.5, 0.5, 0.5, 1])
                scene.add_geometry(sphere)

                # Add normal arrow
                arrow = trimesh.creation.cylinder(radius=0.002, height=0.02)
                direction = trimesh.transformations.align_vectors([0, 0, 1], conn.normal)
                arrow.apply_transform(direction)
                arrow.apply_translation(conn.position + conn.normal * 0.01)
                scene.add_geometry(arrow)

        # Add principal axes
        if show_axes:
            for i, axis in enumerate(part.principal_axes):
                color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][i]
                cylinder = trimesh.creation.cylinder(radius=0.003, height=0.1)
                direction = trimesh.transformations.align_vectors([0, 0, 1], axis)
                cylinder.apply_transform(direction)
                cylinder.visual.vertex_colors = color + [1]
                scene.add_geometry(cylinder)

        return scene