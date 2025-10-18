"""
Plan Assembler for IKEA Assembly
Converts per-step predictions into complete assembly plans
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import networkx as nx
from pathlib import Path
import warnings
from collections import defaultdict
import trimesh


class AssemblyActionType(Enum):
    """Types of assembly actions"""
    ADD_PART = "add_part"
    JOIN_PARTS = "join_parts"
    ROTATE = "rotate"
    FLIP = "flip"
    ALIGN = "align"
    SECURE = "secure"  # Tighten screws, lock cam locks, etc.
    SUBASSEMBLY_COMPLETE = "subassembly_complete"


@dataclass
class AssemblyAction:
    """Single assembly action"""
    action_type: AssemblyActionType
    part_id: str
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3D translation
    step_idx: int
    confidence: float = 1.0
    subassembly_id: Optional[str] = None
    connection_points: List[Dict] = field(default_factory=list)
    tool_required: Optional[str] = None  # e.g., "screwdriver", "hammer"
    notes: Optional[str] = None


@dataclass
class AssemblyPlan:
    """Complete assembly plan"""
    furniture_id: str
    category: str
    actions: List[AssemblyAction]
    subassemblies: Dict[str, List[str]]  # subassembly_id -> part_ids
    dependency_graph: nx.DiGraph
    total_steps: int
    estimated_time_minutes: float = 0.0
    difficulty_level: str = "medium"  # easy, medium, hard
    tools_required: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanAssembler:
    """
    Assembles complete plans from per-step predictions
    Handles subassemblies, dependencies, and optimization
    """

    def __init__(
        self,
        constraint_engine: Any,
        assets_registry: Any,
        validate_physics: bool = True,
        optimize_sequence: bool = True
    ):
        """
        Args:
            constraint_engine: Constraint engine instance
            assets_registry: Assets registry instance
            validate_physics: Whether to validate physical feasibility
            optimize_sequence: Whether to optimize assembly sequence
        """
        self.constraint_engine = constraint_engine
        self.registry = assets_registry
        self.validate_physics = validate_physics
        self.optimize_sequence = optimize_sequence

        # Statistics for time estimation
        self.action_time_estimates = {
            AssemblyActionType.ADD_PART: 30,  # seconds
            AssemblyActionType.JOIN_PARTS: 45,
            AssemblyActionType.ROTATE: 10,
            AssemblyActionType.FLIP: 15,
            AssemblyActionType.ALIGN: 20,
            AssemblyActionType.SECURE: 60,
            AssemblyActionType.SUBASSEMBLY_COMPLETE: 5
        }

    def assemble_plan(
        self,
        step_predictions: List[List[Dict[str, Any]]],
        furniture_id: str,
        category: str = "unknown",
        manual_metadata: Optional[Dict] = None
    ) -> AssemblyPlan:
        """
        Assemble complete plan from per-step predictions

        Args:
            step_predictions: List of predictions for each step
            furniture_id: Furniture identifier
            category: Furniture category
            manual_metadata: Additional metadata from manual

        Returns:
            Complete assembly plan
        """
        # Convert predictions to actions
        actions = self._predictions_to_actions(step_predictions)

        # Detect subassemblies
        subassemblies = self._detect_subassemblies(actions)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(actions, subassemblies)

        # Validate and potentially reorder
        if self.validate_physics:
            actions = self._validate_sequence(actions, dependency_graph)

        if self.optimize_sequence:
            actions = self._optimize_sequence(actions, dependency_graph)

        # Estimate time and difficulty
        estimated_time = self._estimate_assembly_time(actions)
        difficulty = self._assess_difficulty(actions, subassemblies)

        # Collect required tools
        tools = self._collect_tools(actions)

        # Create plan
        plan = AssemblyPlan(
            furniture_id=furniture_id,
            category=category,
            actions=actions,
            subassemblies=subassemblies,
            dependency_graph=dependency_graph,
            total_steps=len(actions),
            estimated_time_minutes=estimated_time / 60,
            difficulty_level=difficulty,
            tools_required=tools,
            metadata=manual_metadata or {}
        )

        return plan

    def _predictions_to_actions(
        self,
        step_predictions: List[List[Dict[str, Any]]]
    ) -> List[AssemblyAction]:
        """Convert raw predictions to assembly actions"""
        actions = []

        for step_idx, step_preds in enumerate(step_predictions):
            for pred in step_preds:
                # Determine action type
                action_type = self._infer_action_type(pred, step_idx, step_predictions)

                # Create action
                action = AssemblyAction(
                    action_type=action_type,
                    part_id=pred.get('part_id', f"part_{step_idx}"),
                    R=pred.get('R', np.eye(3)),
                    t=pred.get('t', np.zeros(3)),
                    step_idx=step_idx,
                    confidence=pred.get('final_score', pred.get('score', 1.0)),
                    subassembly_id=pred.get('subassembly_id'),
                    connection_points=pred.get('connections', []),
                    tool_required=self._infer_tool(pred, action_type),
                    notes=pred.get('notes')
                )

                actions.append(action)

        return actions

    def _infer_action_type(
        self,
        pred: Dict,
        step_idx: int,
        all_predictions: List[List[Dict]]
    ) -> AssemblyActionType:
        """Infer action type from prediction context"""

        # Check if it's a joining action
        if pred.get('connections'):
            for conn in pred['connections']:
                if conn.get('type') == 'screw':
                    return AssemblyActionType.SECURE
                elif conn.get('type') in ['dowel', 'cam_lock']:
                    return AssemblyActionType.JOIN_PARTS

        # Check if it's a rotation/flip
        if step_idx > 0:
            # Look for same part in previous step
            prev_preds = all_predictions[step_idx - 1]
            for prev in prev_preds:
                if prev.get('part_id') == pred.get('part_id'):
                    # Check if orientation changed significantly
                    if 'R' in prev and 'R' in pred:
                        R_diff = np.linalg.norm(pred['R'] - prev['R'], 'fro')
                        if R_diff > 0.5:
                            return AssemblyActionType.ROTATE

        # Check if completing subassembly
        if pred.get('subassembly_complete'):
            return AssemblyActionType.SUBASSEMBLY_COMPLETE

        # Default to adding part
        return AssemblyActionType.ADD_PART

    def _infer_tool(
        self,
        pred: Dict,
        action_type: AssemblyActionType
    ) -> Optional[str]:
        """Infer required tool from prediction"""

        if action_type == AssemblyActionType.SECURE:
            # Check connection type
            for conn in pred.get('connections', []):
                if conn.get('type') == 'screw':
                    return 'screwdriver'
                elif conn.get('type') == 'cam_lock':
                    return 'cam_lock_tool'

        # Check part metadata
        part_info = self.registry.get_part(pred.get('part_id'))
        if part_info and part_info.category == 'fastener':
            if 'screw' in pred.get('part_id', '').lower():
                return 'screwdriver'
            elif 'nail' in pred.get('part_id', '').lower():
                return 'hammer'

        return None

    def _detect_subassemblies(
        self,
        actions: List[AssemblyAction]
    ) -> Dict[str, List[str]]:
        """Detect subassemblies from action sequence"""
        subassemblies = defaultdict(list)
        current_subassembly = None
        subassembly_parts = []

        for action in actions:
            if action.subassembly_id:
                current_subassembly = action.subassembly_id
                subassembly_parts.append(action.part_id)

            elif action.action_type == AssemblyActionType.SUBASSEMBLY_COMPLETE:
                if current_subassembly and subassembly_parts:
                    subassemblies[current_subassembly] = subassembly_parts.copy()
                    current_subassembly = None
                    subassembly_parts = []

            elif current_subassembly:
                subassembly_parts.append(action.part_id)

        # Handle unclosed subassembly
        if current_subassembly and subassembly_parts:
            subassemblies[current_subassembly] = subassembly_parts

        # Auto-detect based on spatial clustering
        if not subassemblies:
            subassemblies = self._cluster_subassemblies(actions)

        return dict(subassemblies)

    def _cluster_subassemblies(
        self,
        actions: List[AssemblyAction]
    ) -> Dict[str, List[str]]:
        """Cluster parts into subassemblies based on spatial proximity"""
        from sklearn.cluster import DBSCAN

        if len(actions) < 3:
            return {}

        # Get part positions
        positions = []
        part_ids = []

        for action in actions:
            if action.action_type == AssemblyActionType.ADD_PART:
                positions.append(action.t)
                part_ids.append(action.part_id)

        if len(positions) < 3:
            return {}

        positions = np.array(positions)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(positions)

        # Group by cluster
        subassemblies = defaultdict(list)
        for part_id, label in zip(part_ids, clustering.labels_):
            if label >= 0:  # -1 means noise/outlier
                subassemblies[f"subassembly_{label}"].append(part_id)

        return dict(subassemblies)

    def _build_dependency_graph(
        self,
        actions: List[AssemblyAction],
        subassemblies: Dict[str, List[str]]
    ) -> nx.DiGraph:
        """Build dependency graph for assembly sequence"""
        graph = nx.DiGraph()

        # Add nodes for each action
        for i, action in enumerate(actions):
            graph.add_node(i, action=action)

        # Add edges based on dependencies
        for i, action in enumerate(actions):
            # Physical dependencies (part must exist before joining)
            if action.action_type in [AssemblyActionType.JOIN_PARTS, AssemblyActionType.SECURE]:
                # Find actions that add connected parts
                for j in range(i):
                    other_action = actions[j]
                    if other_action.action_type == AssemblyActionType.ADD_PART:
                        # Check if parts are connected
                        if self._parts_connected(action, other_action):
                            graph.add_edge(j, i, type='physical')

            # Subassembly dependencies
            if action.subassembly_id:
                # Find other actions in same subassembly
                for j in range(i):
                    other_action = actions[j]
                    if other_action.subassembly_id == action.subassembly_id:
                        if not graph.has_edge(j, i):
                            graph.add_edge(j, i, type='subassembly')

            # Spatial dependencies (bottom-up assembly)
            if action.action_type == AssemblyActionType.ADD_PART:
                for j in range(i):
                    other_action = actions[j]
                    if other_action.action_type == AssemblyActionType.ADD_PART:
                        # Check if action is above other_action
                        if action.t[2] > other_action.t[2] + 0.05:
                            if not graph.has_edge(j, i):
                                graph.add_edge(j, i, type='spatial')

        return graph

    def _parts_connected(
        self,
        action1: AssemblyAction,
        action2: AssemblyAction
    ) -> bool:
        """Check if two actions involve connected parts"""
        # Check connection points
        for conn1 in action1.connection_points:
            for conn2 in action2.connection_points:
                if conn1.get('connects_to') == action2.part_id or \
                   conn2.get('connects_to') == action1.part_id:
                    return True

        # Check spatial proximity
        dist = np.linalg.norm(action1.t - action2.t)
        if dist < 0.1:  # Within 10cm
            return True

        return False

    def _validate_sequence(
        self,
        actions: List[AssemblyAction],
        dependency_graph: nx.DiGraph
    ) -> List[AssemblyAction]:
        """Validate and fix assembly sequence"""
        validated = []
        assembly_state = self.constraint_engine.assembly_state

        # Clear assembly state
        self.constraint_engine.clear_assembly()

        for action in actions:
            # Check if action is physically feasible
            part_info = self.registry.get_part(action.part_id)
            if part_info:
                # Validate pose
                is_valid = self.constraint_engine._validate_pose(
                    part_info,
                    action.R,
                    action.t,
                    assembly_state
                )

                if not is_valid:
                    warnings.warn(f"Action {action.step_idx} for {action.part_id} may cause collision")
                    # Try to fix by adjusting position
                    action = self._fix_collision(action, assembly_state)

            # Update assembly state
            if part_info:
                self.constraint_engine.update_assembly(
                    action.part_id,
                    action.R,
                    action.t
                )

            validated.append(action)

        return validated

    def _fix_collision(
        self,
        action: AssemblyAction,
        assembly_state: Any
    ) -> AssemblyAction:
        """Try to fix collision by adjusting pose"""
        part_info = self.registry.get_part(action.part_id)
        if not part_info:
            return action

        # Get alternative poses
        alternative_poses = self.constraint_engine.get_valid_poses(
            action.part_id,
            assembly_state,
            max_candidates=10
        )

        # Find closest valid pose
        min_dist = float('inf')
        best_pose = None

        for R, t, score in alternative_poses:
            dist = np.linalg.norm(t - action.t) + np.linalg.norm(R - action.R, 'fro')
            if dist < min_dist:
                min_dist = dist
                best_pose = (R, t)

        if best_pose:
            action.R = best_pose[0]
            action.t = best_pose[1]
            action.notes = "Position adjusted to avoid collision"

        return action

    def _optimize_sequence(
        self,
        actions: List[AssemblyAction],
        dependency_graph: nx.DiGraph
    ) -> List[AssemblyAction]:
        """Optimize assembly sequence for efficiency"""

        # Topological sort respecting dependencies
        try:
            topo_order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXError:
            warnings.warn("Circular dependencies detected, keeping original order")
            return actions

        # Reorder actions
        optimized = [actions[i] for i in topo_order]

        # Group similar operations
        optimized = self._group_similar_operations(optimized)

        # Minimize tool changes
        optimized = self._minimize_tool_changes(optimized)

        return optimized

    def _group_similar_operations(
        self,
        actions: List[AssemblyAction]
    ) -> List[AssemblyAction]:
        """Group similar operations together"""
        # Group by action type
        grouped = []
        action_groups = defaultdict(list)

        for action in actions:
            action_groups[action.action_type].append(action)

        # Interleave groups while respecting dependencies
        # Simple heuristic: ADD_PART -> JOIN_PARTS -> SECURE
        priority = [
            AssemblyActionType.ADD_PART,
            AssemblyActionType.ALIGN,
            AssemblyActionType.JOIN_PARTS,
            AssemblyActionType.SECURE,
            AssemblyActionType.ROTATE,
            AssemblyActionType.FLIP,
            AssemblyActionType.SUBASSEMBLY_COMPLETE
        ]

        for action_type in priority:
            if action_type in action_groups:
                grouped.extend(action_groups[action_type])

        return grouped if grouped else actions

    def _minimize_tool_changes(
        self,
        actions: List[AssemblyAction]
    ) -> List[AssemblyAction]:
        """Reorder to minimize tool changes"""
        optimized = []
        remaining = actions.copy()
        current_tool = None

        while remaining:
            # Find next action with same tool or no tool
            same_tool_idx = None
            no_tool_idx = None

            for i, action in enumerate(remaining):
                if action.tool_required == current_tool:
                    same_tool_idx = i
                    break
                elif action.tool_required is None:
                    no_tool_idx = i

            # Pick action
            if same_tool_idx is not None:
                next_idx = same_tool_idx
            elif no_tool_idx is not None:
                next_idx = no_tool_idx
            else:
                next_idx = 0
                current_tool = remaining[0].tool_required

            optimized.append(remaining.pop(next_idx))

        return optimized

    def _estimate_assembly_time(self, actions: List[AssemblyAction]) -> float:
        """Estimate total assembly time in seconds"""
        total_time = 0.0

        for action in actions:
            base_time = self.action_time_estimates.get(action.action_type, 30)

            # Adjust based on confidence
            if action.confidence < 0.8:
                base_time *= 1.2  # Less confident = more time

            # Add tool setup time
            if action.tool_required:
                base_time += 10  # Tool pickup/setup

            total_time += base_time

        return total_time

    def _assess_difficulty(
        self,
        actions: List[AssemblyAction],
        subassemblies: Dict[str, List[str]]
    ) -> str:
        """Assess assembly difficulty"""
        # Factors affecting difficulty
        num_parts = len(set(a.part_id for a in actions))
        num_tools = len(set(a.tool_required for a in actions if a.tool_required))
        num_subassemblies = len(subassemblies)
        avg_confidence = np.mean([a.confidence for a in actions])

        # Simple scoring
        difficulty_score = 0

        if num_parts > 20:
            difficulty_score += 2
        elif num_parts > 10:
            difficulty_score += 1

        if num_tools > 3:
            difficulty_score += 2
        elif num_tools > 1:
            difficulty_score += 1

        if num_subassemblies > 3:
            difficulty_score += 1

        if avg_confidence < 0.7:
            difficulty_score += 1

        # Map to difficulty level
        if difficulty_score <= 2:
            return "easy"
        elif difficulty_score <= 4:
            return "medium"
        else:
            return "hard"

    def _collect_tools(self, actions: List[AssemblyAction]) -> Set[str]:
        """Collect all required tools"""
        tools = set()

        for action in actions:
            if action.tool_required:
                tools.add(action.tool_required)

        # Add basic tools always needed
        tools.add("manual")  # Assembly manual
        tools.add("workspace")  # Clear workspace

        return tools

    def export_plan(self, plan: AssemblyPlan, output_path: str, format: str = 'json'):
        """
        Export assembly plan to file

        Args:
            plan: Assembly plan to export
            output_path: Output file path
            format: Export format ('json', 'pickle', 'readable')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            self._export_json(plan, output_path)
        elif format == 'pickle':
            self._export_pickle(plan, output_path)
        elif format == 'readable':
            self._export_readable(plan, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_json(self, plan: AssemblyPlan, output_path: Path):
        """Export plan as JSON"""
        # Convert to JSON-serializable format
        data = {
            'furniture_id': plan.furniture_id,
            'category': plan.category,
            'total_steps': plan.total_steps,
            'estimated_time_minutes': plan.estimated_time_minutes,
            'difficulty_level': plan.difficulty_level,
            'tools_required': list(plan.tools_required),
            'actions': [],
            'subassemblies': plan.subassemblies,
            'metadata': plan.metadata
        }

        for action in plan.actions:
            action_dict = {
                'step': action.step_idx,
                'type': action.action_type.value,
                'part_id': action.part_id,
                'rotation': action.R.tolist(),
                'translation': action.t.tolist(),
                'confidence': action.confidence,
                'tool': action.tool_required,
                'notes': action.notes
            }
            data['actions'].append(action_dict)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_pickle(self, plan: AssemblyPlan, output_path: Path):
        """Export plan as pickle"""
        with open(output_path, 'wb') as f:
            pickle.dump(plan, f)

    def _export_readable(self, plan: AssemblyPlan, output_path: Path):
        """Export human-readable instructions"""
        lines = []

        # Header
        lines.append(f"# Assembly Instructions: {plan.furniture_id}")
        lines.append(f"Category: {plan.category}")
        lines.append(f"Difficulty: {plan.difficulty_level}")
        lines.append(f"Estimated Time: {plan.estimated_time_minutes:.0f} minutes")
        lines.append("")

        # Tools
        lines.append("## Required Tools:")
        for tool in sorted(plan.tools_required):
            lines.append(f"- {tool}")
        lines.append("")

        # Steps
        lines.append("## Assembly Steps:")
        for i, action in enumerate(plan.actions, 1):
            lines.append(f"\n### Step {i}: {action.action_type.value.replace('_', ' ').title()}")
            lines.append(f"Part: {action.part_id}")

            if action.tool_required:
                lines.append(f"Tool: {action.tool_required}")

            if action.notes:
                lines.append(f"Note: {action.notes}")

            lines.append(f"Confidence: {action.confidence:.1%}")

        # Subassemblies
        if plan.subassemblies:
            lines.append("\n## Subassemblies:")
            for sub_id, parts in plan.subassemblies.items():
                lines.append(f"\n### {sub_id}:")
                for part in parts:
                    lines.append(f"- {part}")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def visualize_plan(self, plan: AssemblyPlan) -> Any:
        """
        Visualize assembly plan
        Returns visualization object (e.g., matplotlib figure or 3D scene)
        """
        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Assembly sequence timeline
        ax = axes[0, 0]
        ax.set_title("Assembly Sequence")
        action_types = [a.action_type.value for a in plan.actions]
        action_counts = {t: action_types.count(t) for t in set(action_types)}
        ax.bar(action_counts.keys(), action_counts.values())
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Dependency graph
        ax = axes[0, 1]
        ax.set_title("Dependencies")
        if plan.dependency_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(plan.dependency_graph)
            nx.draw(plan.dependency_graph, pos, ax=ax, with_labels=True,
                   node_size=300, font_size=8, arrows=True)

        # 3. Confidence over steps
        ax = axes[1, 0]
        ax.set_title("Confidence Scores")
        confidences = [a.confidence for a in plan.actions]
        ax.plot(confidences, 'o-')
        ax.set_xlabel("Step")
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # 4. Time breakdown
        ax = axes[1, 1]
        ax.set_title("Time Breakdown")
        time_by_type = defaultdict(float)
        for action in plan.actions:
            time = self.action_time_estimates.get(action.action_type, 30)
            time_by_type[action.action_type.value] += time

        ax.pie(time_by_type.values(), labels=time_by_type.keys(), autopct='%1.1f%%')

        plt.tight_layout()

        return fig