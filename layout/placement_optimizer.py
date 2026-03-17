"""
Graph-based optimization placement system.

Replaces the unstable backtracking approach with a more robust optimization-based method:
1. Build a constraint graph from placement relationships
2. Initialize positions using topological ordering
3. Iteratively optimize positions to minimize constraint violations
4. Use soft constraints with weights for graceful degradation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
import networkx as nx


class PlacementOptimizer:
    """Optimizes object placement using graph-based constraint satisfaction."""

    # Room layout element IDs (fixed positions)
    ROOM_ELEMENTS = {"south_wall", "north_wall", "east_wall", "west_wall", "ceiling", "middle of the room"}

    # Constraint weights (higher = more important)
    WEIGHTS = {
        "collision": 100.0,      # Objects must not overlap
        "adjacency": 10.0,       # Adjacent constraint satisfaction
        "direction": 5.0,        # Directional relationship (left of, in front, etc.)
        "room_bounds": 50.0,     # Stay within room
        "connection": 20.0,      # Connected equipment should be close
    }

    def __init__(self, scene_graph: Dict, room_dimensions: List[float], room_priors: List[Dict], preserve_existing: bool = True):
        """Initialize optimizer.

        Args:
            scene_graph: Scene graph with objects_in_room
            room_dimensions: [length, width, height] of room
            room_priors: Room layout elements (walls, floor, etc.)
            preserve_existing: If True, keep existing positions and only place new objects
        """
        self.room_dimensions = room_dimensions
        self.room_priors = room_priors
        self.preserve_existing = preserve_existing

        # Extract objects (exclude room layout elements)
        if isinstance(scene_graph, dict):
            all_objects = scene_graph.get("objects_in_room", [])
        else:
            all_objects = scene_graph

        self.objects = [obj for obj in all_objects if obj.get("object_id") not in self.ROOM_ELEMENTS]
        self.object_map = {obj["object_id"]: obj for obj in self.objects}

        # Track which objects already have positions (from baseline)
        self.fixed_objects = set()
        if preserve_existing:
            for obj in self.objects:
                if obj.get("position") and self._is_valid_position(obj.get("position")):
                    self.fixed_objects.add(obj["object_id"])

        # Build room element positions
        self.room_element_positions = self._build_room_element_positions()

        # Build constraint graph
        self.constraint_graph = self._build_constraint_graph()

    def _is_valid_position(self, pos: Dict) -> bool:
        """Check if a position dict has valid coordinates."""
        if not pos:
            return False
        x, y, z = pos.get("x"), pos.get("y"), pos.get("z")
        if x is None or y is None:
            return False
        # Check if within reasonable bounds (allow some margin outside room)
        length, width, height = self.room_dimensions
        margin = 2.0
        if x < -margin or x > length + margin:
            return False
        if y < -margin or y > width + margin:
            return False
        return True

    def _build_room_element_positions(self) -> Dict[str, Dict[str, float]]:
        """Build position map for room layout elements."""
        length, width, height = self.room_dimensions
        return {
            "south_wall": {"x": length / 2, "y": 0, "z": 0},
            "north_wall": {"x": length / 2, "y": width, "z": 0},
            "west_wall": {"x": 0, "y": width / 2, "z": 0},
            "east_wall": {"x": length, "y": width / 2, "z": 0},
            "middle of the room": {"x": length / 2, "y": width / 2, "z": 0},
            "ceiling": {"x": length / 2, "y": width / 2, "z": height},
        }

    def _build_constraint_graph(self) -> nx.DiGraph:
        """Build directed graph of placement constraints."""
        G = nx.DiGraph()

        for obj in self.objects:
            obj_id = obj["object_id"]
            G.add_node(obj_id, object=obj)

            placement = obj.get("placement", {})
            room_constraints = placement.get("room_layout_elements", [])
            object_constraints = placement.get("objects_in_room", [])

            # Add edges for room layout constraints
            for constraint in room_constraints:
                ref_id = constraint.get("layout_element_id")
                if ref_id:
                    prep = constraint.get("preposition", "on")
                    adjacent = constraint.get("is_adjacent", True)
                    G.add_edge(ref_id, obj_id, preposition=prep, is_adjacent=adjacent, is_room=True)

            # Add edges for object-to-object constraints
            for constraint in object_constraints:
                ref_id = constraint.get("object_id")
                if ref_id and ref_id in self.object_map:
                    prep = constraint.get("preposition", "on")
                    adjacent = constraint.get("is_adjacent", True)
                    G.add_edge(ref_id, obj_id, preposition=prep, is_adjacent=adjacent, is_room=False)

        return G

    def _get_object_size(self, obj: Dict) -> Tuple[float, float, float]:
        """Get object dimensions."""
        size = obj.get("size_in_meters", {})
        return (
            size.get("length", 1.0),
            size.get("width", 1.0),
            size.get("height", 1.0)
        )

    def _get_position(self, obj_id: str) -> Optional[Dict[str, float]]:
        """Get position of object or room element."""
        if obj_id in self.room_element_positions:
            return self.room_element_positions[obj_id]
        if obj_id in self.object_map:
            return self.object_map[obj_id].get("position")
        return None

    def _compute_initial_position(self, obj_id: str) -> Dict[str, float]:
        """Compute initial position for an object based on constraints."""
        obj = self.object_map[obj_id]
        size = self._get_object_size(obj)
        length, width, height = self.room_dimensions

        # Collect all constraint targets
        positions = []
        weights = []

        for pred_id in self.constraint_graph.predecessors(obj_id):
            edge_data = self.constraint_graph.edges[pred_id, obj_id]
            ref_pos = self._get_position(pred_id)

            if ref_pos is None:
                continue

            prep = edge_data.get("preposition", "on")
            adjacent = edge_data.get("is_adjacent", True)

            # Compute target position based on preposition
            target = self._compute_target_from_preposition(
                ref_pos, prep, adjacent, size, pred_id
            )
            if target:
                positions.append(target)
                # Room constraints get higher weight
                weight = 2.0 if edge_data.get("is_room", False) else 1.0
                weights.append(weight)

        if not positions:
            # No constraints - find clear space instead of center
            return self._find_clear_position(obj_id, size)

        # Weighted average of target positions
        total_weight = sum(weights)
        x = sum(p["x"] * w for p, w in zip(positions, weights)) / total_weight
        y = sum(p["y"] * w for p, w in zip(positions, weights)) / total_weight
        z = sum(p["z"] * w for p, w in zip(positions, weights)) / total_weight

        # Check if this position overlaps with existing objects
        candidate_pos = {"x": x, "y": y, "z": z}
        if self._position_has_collision(candidate_pos, size, obj_id):
            # Find nearby clear position
            clear_pos = self._find_clear_position_near(obj_id, size, candidate_pos)
            if clear_pos:
                return clear_pos

        # Clamp to room bounds
        margin = 0.5
        x = max(margin, min(length - margin, x))
        y = max(margin, min(width - margin, y))
        z = max(0, min(height - size[2], z))

        return {"x": x, "y": y, "z": z}

    def _compute_target_from_preposition(
        self,
        ref_pos: Dict[str, float],
        prep: str,
        adjacent: bool,
        obj_size: Tuple[float, float, float],
        ref_id: str
    ) -> Optional[Dict[str, float]]:
        """Compute target position based on preposition relative to reference."""
        # Get reference object size
        if ref_id in self.object_map:
            ref_size = self._get_object_size(self.object_map[ref_id])
        else:
            ref_size = (1.0, 1.0, 1.0)  # Room elements

        offset = 0.1 if adjacent else 1.5  # Gap between objects
        obj_len, obj_width, obj_height = obj_size
        ref_len, ref_width, ref_height = ref_size

        x, y, z = ref_pos["x"], ref_pos["y"], ref_pos["z"]

        if prep == "left of":
            return {"x": x - ref_len/2 - obj_len/2 - offset, "y": y, "z": z}
        elif prep == "right of":
            return {"x": x + ref_len/2 + obj_len/2 + offset, "y": y, "z": z}
        elif prep == "in front":
            return {"x": x, "y": y - ref_width/2 - obj_width/2 - offset, "z": z}
        elif prep == "behind":
            return {"x": x, "y": y + ref_width/2 + obj_width/2 + offset, "z": z}
        elif prep == "on":
            # On top of or at same location
            if ref_id == "middle of the room":
                return {"x": x, "y": y, "z": 0}
            return {"x": x, "y": y, "z": z + ref_height}
        elif prep == "under":
            return {"x": x, "y": y, "z": max(0, z - obj_height)}
        elif prep == "through":
            # Pass through - same position
            return {"x": x, "y": y, "z": z}
        elif prep == "in the corner":
            # Corner placement - adjust based on which wall
            return {"x": x, "y": y, "z": 0}
        else:
            # Unknown preposition - return reference position
            return {"x": x, "y": y, "z": 0}

    def _compute_constraint_cost(self, obj_id: str) -> float:
        """Compute total constraint violation cost for an object."""
        obj = self.object_map[obj_id]
        pos = obj.get("position")
        if not pos:
            return float('inf')

        total_cost = 0.0
        size = self._get_object_size(obj)
        length, width, height = self.room_dimensions

        # Room bounds cost
        margin = 0.3
        if pos["x"] < margin or pos["x"] > length - margin:
            total_cost += self.WEIGHTS["room_bounds"]
        if pos["y"] < margin or pos["y"] > width - margin:
            total_cost += self.WEIGHTS["room_bounds"]

        # Constraint satisfaction cost
        for pred_id in self.constraint_graph.predecessors(obj_id):
            edge_data = self.constraint_graph.edges[pred_id, obj_id]
            ref_pos = self._get_position(pred_id)

            if ref_pos is None:
                continue

            prep = edge_data.get("preposition", "on")
            adjacent = edge_data.get("is_adjacent", True)

            # Compute expected position
            target = self._compute_target_from_preposition(ref_pos, prep, adjacent, size, pred_id)
            if target:
                # Distance from target
                dist = np.sqrt(
                    (pos["x"] - target["x"])**2 +
                    (pos["y"] - target["y"])**2 +
                    (pos["z"] - target["z"])**2
                )
                # Adjacency violations cost more
                if adjacent and dist > 1.0:
                    total_cost += self.WEIGHTS["adjacency"] * dist
                elif not adjacent and dist < 0.5:
                    total_cost += self.WEIGHTS["adjacency"]
                else:
                    total_cost += self.WEIGHTS["direction"] * min(dist, 5.0)

        # Collision cost with other objects
        for other_id, other_obj in self.object_map.items():
            if other_id == obj_id:
                continue
            other_pos = other_obj.get("position")
            if not other_pos:
                continue

            other_size = self._get_object_size(other_obj)
            if self._objects_overlap(pos, size, other_pos, other_size):
                total_cost += self.WEIGHTS["collision"]

        return total_cost

    def _objects_overlap(
        self,
        pos1: Dict[str, float],
        size1: Tuple[float, float, float],
        pos2: Dict[str, float],
        size2: Tuple[float, float, float],
        margin: float = 0.1
    ) -> bool:
        """Check if two objects overlap."""
        # Check each axis
        for i, (p1, s1, p2, s2) in enumerate([
            (pos1["x"], size1[0], pos2["x"], size2[0]),
            (pos1["y"], size1[1], pos2["y"], size2[1]),
            (pos1["z"], size1[2], pos2["z"], size2[2]),
        ]):
            half1, half2 = s1 / 2, s2 / 2
            if abs(p1 - p2) >= half1 + half2 + margin:
                return False  # Separated on this axis
        return True  # Overlap on all axes

    def _position_has_collision(
        self,
        pos: Dict[str, float],
        size: Tuple[float, float, float],
        exclude_obj_id: str = None
    ) -> bool:
        """Check if a position would cause collision with existing objects.

        Args:
            pos: Position to check
            size: Size of the object being placed
            exclude_obj_id: Object ID to exclude from collision check (the object being placed)

        Returns:
            True if collision exists, False if position is clear
        """
        for other_id, other_obj in self.object_map.items():
            if other_id == exclude_obj_id:
                continue
            other_pos = other_obj.get("position")
            if not other_pos or not self._is_valid_position(other_pos):
                continue
            other_size = self._get_object_size(other_obj)

            if self._objects_overlap(pos, size, other_pos, other_size, margin=0.2):
                return True
        return False

    def _find_clear_position_near(
        self,
        obj_id: str,
        size: Tuple[float, float, float],
        target_pos: Dict[str, float],
        max_search_radius: float = 10.0
    ) -> Optional[Dict[str, float]]:
        """Find a clear position near a target position.

        Uses spiral search pattern to find nearest collision-free position.

        Args:
            obj_id: ID of object being placed
            size: Size of object
            target_pos: Desired position to search near
            max_search_radius: Maximum distance to search from target

        Returns:
            Clear position dict or None if none found
        """
        length, width, height = self.room_dimensions
        step = 0.5  # Search step size

        # Spiral search pattern
        for radius in np.arange(step, max_search_radius, step):
            # Check positions in a circle at this radius
            num_points = max(8, int(2 * np.pi * radius / step))
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                candidate = {
                    "x": target_pos["x"] + radius * np.cos(angle),
                    "y": target_pos["y"] + radius * np.sin(angle),
                    "z": target_pos.get("z", 0)
                }

                # Check room bounds
                margin = size[0] / 2 + 0.1
                if candidate["x"] < margin or candidate["x"] > length - margin:
                    continue
                margin = size[1] / 2 + 0.1
                if candidate["y"] < margin or candidate["y"] > width - margin:
                    continue

                # Check collision
                if not self._position_has_collision(candidate, size, obj_id):
                    return candidate

        # No clear position found within search radius - try expanding room
        return self._find_clear_position_expanding_room(obj_id, size, target_pos)

    def _find_clear_position_expanding_room(
        self,
        obj_id: str,
        size: Tuple[float, float, float],
        target_pos: Dict[str, float]
    ) -> Dict[str, float]:
        """Find clear position by expanding room dimensions if needed.

        Args:
            obj_id: Object being placed
            size: Size of object
            target_pos: Original target position

        Returns:
            Position (room may have been expanded)
        """
        length, width, height = self.room_dimensions
        obj_len, obj_width, obj_height = size

        # Try to place at edges of current room first
        edge_positions = [
            {"x": length - obj_len/2 - 0.5, "y": target_pos["y"], "z": 0},  # Near east wall
            {"x": obj_len/2 + 0.5, "y": target_pos["y"], "z": 0},  # Near west wall
            {"x": target_pos["x"], "y": width - obj_width/2 - 0.5, "z": 0},  # Near north wall
            {"x": target_pos["x"], "y": obj_width/2 + 0.5, "z": 0},  # Near south wall
        ]

        for pos in edge_positions:
            # Clamp to valid range
            pos["x"] = max(obj_len/2 + 0.1, min(length - obj_len/2 - 0.1, pos["x"]))
            pos["y"] = max(obj_width/2 + 0.1, min(width - obj_width/2 - 0.1, pos["y"]))

            if not self._position_has_collision(pos, size, obj_id):
                return pos

        # Need to expand room - add space at the edge furthest from center
        expansion = max(obj_len, obj_width) + 1.0

        # Determine which direction to expand based on target position
        center_x, center_y = length / 2, width / 2
        dx = target_pos["x"] - center_x
        dy = target_pos["y"] - center_y

        if abs(dx) > abs(dy):
            # Expand in X direction
            if dx > 0:
                # Expand east
                new_length = length + expansion
                new_pos = {"x": length + expansion/2, "y": target_pos["y"], "z": 0}
            else:
                # Expand west - shift everything
                new_length = length + expansion
                new_pos = {"x": expansion/2, "y": target_pos["y"], "z": 0}
                self._shift_all_objects(expansion, 0)

            self.room_dimensions[0] = new_length
            self._update_room_element_positions()
        else:
            # Expand in Y direction
            if dy > 0:
                # Expand north
                new_width = width + expansion
                new_pos = {"x": target_pos["x"], "y": width + expansion/2, "z": 0}
            else:
                # Expand south - shift everything
                new_width = width + expansion
                new_pos = {"x": target_pos["x"], "y": expansion/2, "z": 0}
                self._shift_all_objects(0, expansion)

            self.room_dimensions[1] = new_width
            self._update_room_element_positions()

        return new_pos

    def _shift_all_objects(self, dx: float, dy: float):
        """Shift all existing object positions by given delta."""
        for obj in self.objects:
            pos = obj.get("position")
            if pos and self._is_valid_position(pos):
                pos["x"] += dx
                pos["y"] += dy

    def _update_room_element_positions(self):
        """Update room element positions after room dimension change."""
        self.room_element_positions = self._build_room_element_positions()

    def _expand_room_to_fit_objects(self, verbose: bool = False):
        """Expand room dimensions to fit all placed objects with margin.

        This is called after placement and collision resolution to ensure
        no objects are outside the room bounds.
        """
        margin = 1.0  # Space between objects and walls
        length, width, height = self.room_dimensions

        max_x = length
        max_y = width

        for obj in self.objects:
            pos = obj.get("position")
            if not pos:
                continue
            size = self._get_object_size(obj)

            # Calculate required room size to contain this object
            required_x = pos["x"] + size[0] / 2 + margin
            required_y = pos["y"] + size[1] / 2 + margin

            max_x = max(max_x, required_x)
            max_y = max(max_y, required_y)

        # Update room dimensions if needed
        if max_x > length or max_y > width:
            old_dims = self.room_dimensions[:2]
            self.room_dimensions[0] = max_x
            self.room_dimensions[1] = max_y
            self._update_room_element_positions()

            if verbose:
                print(f"📐 Room expanded: {old_dims[0]:.1f}x{old_dims[1]:.1f} -> {max_x:.1f}x{max_y:.1f}")

    def _find_clear_position(
        self,
        obj_id: str,
        size: Tuple[float, float, float]
    ) -> Dict[str, float]:
        """Find a clear position for an object with no constraints.

        Searches the room grid for collision-free space.

        Args:
            obj_id: Object being placed
            size: Size of object

        Returns:
            Clear position
        """
        length, width, height = self.room_dimensions
        obj_len, obj_width, obj_height = size

        # Grid search for clear position
        step = 1.0
        margin = 0.5

        # Start from center and spiral outward
        center_x, center_y = length / 2, width / 2

        # Try center first
        center_pos = {"x": center_x, "y": center_y, "z": 0}
        if not self._position_has_collision(center_pos, size, obj_id):
            return center_pos

        # Spiral search from center
        for radius in np.arange(step, max(length, width), step):
            num_points = max(8, int(2 * np.pi * radius / step))
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                candidate = {
                    "x": center_x + radius * np.cos(angle),
                    "y": center_y + radius * np.sin(angle),
                    "z": 0
                }

                # Check room bounds
                if candidate["x"] < obj_len/2 + margin:
                    continue
                if candidate["x"] > length - obj_len/2 - margin:
                    continue
                if candidate["y"] < obj_width/2 + margin:
                    continue
                if candidate["y"] > width - obj_width/2 - margin:
                    continue

                if not self._position_has_collision(candidate, size, obj_id):
                    return candidate

        # No clear position in current room - expand
        return self._find_clear_position_expanding_room(obj_id, size, center_pos)

    def _optimize_position(self, obj_id: str, iterations: int = 20) -> bool:
        """Optimize position of single object using gradient-free search."""
        obj = self.object_map[obj_id]
        pos = obj.get("position")
        if not pos:
            return False

        best_pos = deepcopy(pos)
        best_cost = self._compute_constraint_cost(obj_id)

        # Search directions
        step_sizes = [1.0, 0.5, 0.2, 0.1]
        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (-1, -1, 0),
            (1, -1, 0), (-1, 1, 0),
        ]

        for _ in range(iterations):
            improved = False
            for step in step_sizes:
                for dx, dy, dz in directions:
                    new_pos = {
                        "x": pos["x"] + dx * step,
                        "y": pos["y"] + dy * step,
                        "z": max(0, pos["z"] + dz * step)
                    }
                    obj["position"] = new_pos
                    cost = self._compute_constraint_cost(obj_id)

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = deepcopy(new_pos)
                        improved = True

            if not improved:
                break

            pos = best_pos
            obj["position"] = best_pos

        obj["position"] = best_pos
        return best_cost < float('inf')

    def optimize(self, max_iterations: int = 50, verbose: bool = False) -> Dict[str, Any]:
        """Run optimization to place all objects.

        Args:
            max_iterations: Maximum optimization iterations
            verbose: Print progress

        Returns:
            Dictionary with placement results
        """
        # Get topological ordering
        try:
            # Filter to only include nodes that are objects (not room elements)
            object_nodes = [n for n in self.constraint_graph.nodes() if n in self.object_map]
            # Create subgraph with only object-to-object edges for ordering
            topo_order = list(nx.topological_sort(self.constraint_graph))
            topo_order = [n for n in topo_order if n in self.object_map]
        except nx.NetworkXUnfeasible:
            # Cycle detected - use arbitrary order
            if verbose:
                print("⚠️ Cycle detected in constraint graph, using arbitrary order")
            topo_order = list(self.object_map.keys())

        # Separate fixed (baseline) objects from new objects that need placement
        new_objects = [obj_id for obj_id in topo_order if obj_id not in self.fixed_objects]

        if verbose:
            print(f"📊 Total objects: {len(topo_order)}, Fixed: {len(self.fixed_objects)}, New: {len(new_objects)}")

        # Phase 1: Initialize positions for NEW objects only
        if verbose:
            print("Phase 1: Initial placement for new objects...")

        for obj_id in new_objects:
            obj = self.object_map[obj_id]
            if not obj.get("position") or not self._is_valid_position(obj.get("position")):
                initial_pos = self._compute_initial_position(obj_id)
                obj["position"] = initial_pos
                if verbose:
                    print(f"  Initialized {obj_id} at ({initial_pos['x']:.2f}, {initial_pos['y']:.2f})")

        # Phase 2: Iterative optimization for NEW objects only
        if new_objects:
            if verbose:
                print("Phase 2: Optimizing new object positions...")

            for iteration in range(max_iterations):
                total_cost_before = sum(self._compute_constraint_cost(obj_id) for obj_id in new_objects)

                # Optimize only new objects (keep baseline fixed)
                for obj_id in new_objects:
                    self._optimize_position(obj_id, iterations=10)

                total_cost_after = sum(self._compute_constraint_cost(obj_id) for obj_id in new_objects)

                if verbose and iteration % 10 == 0:
                    print(f"  Iteration {iteration}: cost {total_cost_before:.1f} -> {total_cost_after:.1f}")

                # Check convergence
                if abs(total_cost_before - total_cost_after) < 0.1:
                    if verbose:
                        print(f"  Converged at iteration {iteration}")
                    break

        # Phase 3: Resolve collisions (only move new objects, not fixed ones)
        if verbose:
            print("Phase 3: Resolving collisions...")

        self._resolve_collisions(verbose)

        # Phase 4: Expand room to fit all placed objects
        self._expand_room_to_fit_objects(verbose)

        # Collect results
        placed = []
        unplaced = []
        for obj_id in topo_order:
            obj = self.object_map[obj_id]
            if obj.get("position"):
                placed.append(obj_id)
            else:
                unplaced.append(obj_id)

        if verbose:
            print(f"✅ Placed {len(placed)} objects")
            if unplaced:
                print(f"⚠️ Could not place: {unplaced}")

        return {
            "placed": placed,
            "unplaced": unplaced,
            "objects": self.objects,
            "room_dimensions": self.room_dimensions  # Return updated dimensions in case room was expanded
        }

    def _resolve_collisions(self, verbose: bool = False):
        """Push apart overlapping objects. Only moves new objects, not fixed baseline objects."""
        max_attempts = 100
        for attempt in range(max_attempts):
            collision_found = False

            for obj_id, obj in self.object_map.items():
                # Skip fixed objects - they shouldn't be moved
                if obj_id in self.fixed_objects:
                    continue

                pos = obj.get("position")
                if not pos:
                    continue
                size = self._get_object_size(obj)

                for other_id, other_obj in self.object_map.items():
                    if other_id == obj_id:
                        continue
                    other_pos = other_obj.get("position")
                    if not other_pos:
                        continue
                    other_size = self._get_object_size(other_obj)

                    if self._objects_overlap(pos, size, other_pos, other_size):
                        collision_found = True
                        # Push objects apart
                        dx = pos["x"] - other_pos["x"]
                        dy = pos["y"] - other_pos["y"]
                        dist = max(0.1, np.sqrt(dx**2 + dy**2))

                        # Normalize and scale push - only move new object, not fixed ones
                        push = 0.5
                        obj["position"]["x"] += (dx / dist) * push
                        obj["position"]["y"] += (dy / dist) * push

                        # Only push the other object if it's not fixed
                        if other_id not in self.fixed_objects:
                            other_obj["position"]["x"] -= (dx / dist) * push
                            other_obj["position"]["y"] -= (dy / dist) * push

                        if verbose and attempt < 5:
                            print(f"  Pushing {obj_id} away from {other_id}")

            if not collision_found:
                break

    def get_scene_graph(self) -> Dict[str, List]:
        """Get updated scene graph with optimized positions."""
        return {"objects_in_room": self.objects}


def optimize_placement(
    scene_graph: Dict,
    room_dimensions: List[float],
    room_priors: List[Dict],
    verbose: bool = False,
    preserve_existing: bool = True
) -> Tuple[Dict, Dict[str, Any]]:
    """Convenience function to optimize placement.

    Args:
        scene_graph: Scene graph with objects_in_room
        room_dimensions: [length, width, height]
        room_priors: Room layout elements
        verbose: Print progress
        preserve_existing: If True, keep existing positions and only place new objects

    Returns:
        Tuple of (updated_scene_graph, results_dict)
    """
    optimizer = PlacementOptimizer(scene_graph, room_dimensions, room_priors, preserve_existing=preserve_existing)
    results = optimizer.optimize(verbose=verbose)

    # Merge back with room priors
    updated_graph = optimizer.get_scene_graph()
    updated_graph["objects_in_room"].extend(room_priors)

    return updated_graph, results
