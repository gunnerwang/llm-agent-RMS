"""
Structure-Aware Reconfiguration System

This module learns the structural patterns from a baseline manufacturing layout
and applies modifications while preserving the core structure.

Key concepts:
1. Loop Structure: Conveyor loops form the backbone of material flow
2. Edge Segments: Each side of the loop has conveyors and stations
3. Inline Stations: Processing stations (Line_06, Line_07) sit on conveyors
4. Support Equipment: Workstations (Line_02), lifts (Line_01) are positioned relative to conveyors

Reconfiguration operations:
- extend_edge: Add conveyors/stations to an edge
- add_parallel_path: Create a parallel processing branch
- insert_station: Add a station inline on existing conveyor

Production-focused operations:
- define_material_flow: Create physical passage with loading/unloading points
  (removes corner, conveyor, and inline stations to open the loop)
- add_tbranch: Add T-branch with lift transfer for side material handling
- add_inline_inspection: Add Line_06 inspection station inline on conveyor
- add_assembly_cell: Add Line_07 robotic cell inline with roller bridges
- add_material_staging: Add carts, pallet mover, and roller bridge near equipment
"""

import json
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math


class LoopStructure:
    """Represents the learned structure of a conveyor loop system."""

    ROOM_ELEMENTS = {'south_wall', 'north_wall', 'east_wall', 'west_wall', 'ceiling', 'middle of the room'}

    def __init__(self, scene_graph: List[Dict]):
        """Learn structure from a baseline scene graph."""
        self.objects = [o for o in scene_graph if o.get('object_id') not in self.ROOM_ELEMENTS]
        self.obj_map = {o['object_id']: o for o in self.objects}

        # Extract structural information
        self._extract_connections()
        self._identify_corners()
        self._identify_edges()
        self._identify_inline_stations()
        self._trace_loop()

    def _extract_connections(self):
        """Build connection graph from scene objects."""
        self.connections = defaultdict(list)
        self.reverse_connections = defaultdict(list)

        for obj in self.objects:
            obj_id = obj.get('object_id', '')
            for conn in obj.get('connections', []):
                target = conn.get('object_id', '')
                if target:
                    self.connections[obj_id].append({
                        'target': target,
                        'src_endpoint': conn.get('source_endpoint', ''),
                        'tgt_endpoint': conn.get('target_endpoint', '')
                    })
                    self.reverse_connections[target].append({
                        'source': obj_id,
                        'src_endpoint': conn.get('source_endpoint', ''),
                        'tgt_endpoint': conn.get('target_endpoint', '')
                    })

    def _identify_corners(self):
        """Find corner pieces (Line_03) and their positions."""
        self.corners = {}
        for oid, obj in self.obj_map.items():
            if 'Line_03' in oid:
                pos = obj.get('position', {})
                self.corners[oid] = {
                    'x': pos.get('x', 0),
                    'y': pos.get('y', 0),
                    'connections': [c['target'] for c in self.connections.get(oid, [])]
                }

        # Determine corner positions (SW, SE, NW, NE)
        if self.corners:
            xs = [c['x'] for c in self.corners.values()]
            ys = [c['y'] for c in self.corners.values()]
            self.bounds = {
                'min_x': min(xs), 'max_x': max(xs),
                'min_y': min(ys), 'max_y': max(ys)
            }

            # Label corners by position
            self.corner_positions = {}
            for oid, corner in self.corners.items():
                x, y = corner['x'], corner['y']
                is_west = abs(x - self.bounds['min_x']) < abs(x - self.bounds['max_x'])
                is_south = abs(y - self.bounds['min_y']) < abs(y - self.bounds['max_y'])
                pos_name = ('S' if is_south else 'N') + ('W' if is_west else 'E')
                self.corner_positions[pos_name] = oid

    def _identify_edges(self):
        """Identify which conveyors and equipment belong to each edge."""
        self.edges = {
            'south': {'conveyors': [], 'stations': [], 'support': []},
            'north': {'conveyors': [], 'stations': [], 'support': []},
            'west': {'conveyors': [], 'stations': [], 'support': []},
            'east': {'conveyors': [], 'stations': [], 'support': []},
            'center': {'conveyors': [], 'stations': [], 'support': []}
        }

        tolerance = 2.0  # meters from edge to be considered "on" that edge

        for oid, obj in self.obj_map.items():
            if 'Line_03' in oid:
                continue  # Skip corners

            pos = obj.get('position', {})
            x, y = pos.get('x', 0), pos.get('y', 0)

            # Determine edge
            edge = self._get_edge(x, y, tolerance)

            # Categorize by type
            if 'Line_04' in oid or 'Line_05' in oid:
                self.edges[edge]['conveyors'].append(oid)
            elif 'Line_06' in oid or 'Line_07' in oid:
                self.edges[edge]['stations'].append(oid)
            else:
                self.edges[edge]['support'].append(oid)

    def _get_edge(self, x: float, y: float, tolerance: float = 2.0) -> str:
        """Determine which edge a position belongs to."""
        if not hasattr(self, 'bounds'):
            return 'center'

        dist_south = abs(y - self.bounds['min_y'])
        dist_north = abs(y - self.bounds['max_y'])
        dist_west = abs(x - self.bounds['min_x'])
        dist_east = abs(x - self.bounds['max_x'])

        min_dist = min(dist_south, dist_north, dist_west, dist_east)

        if min_dist > tolerance:
            return 'center'
        elif min_dist == dist_south:
            return 'south'
        elif min_dist == dist_north:
            return 'north'
        elif min_dist == dist_west:
            return 'west'
        else:
            return 'east'

    def _identify_inline_stations(self):
        """Find stations that are positioned inline on conveyors (pass-through)."""
        self.inline_stations = {}

        station_types = ['Line_06', 'Line_07']
        conveyor_types = ['Line_04', 'Line_05']

        for oid, obj in self.obj_map.items():
            if not any(st in oid for st in station_types):
                continue

            pos = obj.get('position', {})
            sx, sy = pos.get('x', 0), pos.get('y', 0)

            # Find conveyor at same position
            for cid, cobj in self.obj_map.items():
                if not any(ct in cid for ct in conveyor_types):
                    continue

                cpos = cobj.get('position', {})
                cx, cy = cpos.get('x', 0), cpos.get('y', 0)

                # Check if positions overlap (inline)
                if abs(sx - cx) < 2.0 and abs(sy - cy) < 2.0:
                    self.inline_stations[oid] = cid
                    break

    def _trace_loop(self):
        """Trace the main conveyor loop to understand flow direction."""
        self.loop_path = []

        if not self.corner_positions:
            return

        # Start from SW corner
        start = self.corner_positions.get('SW')
        if not start:
            start = list(self.corners.keys())[0]

        visited = set()
        current = start

        for _ in range(50):  # Max iterations
            if current in visited and current == start and len(visited) > 3:
                self.loop_path.append(current)
                break

            visited.add(current)
            self.loop_path.append(current)

            # Find next in flow
            next_node = None
            for conn in self.connections.get(current, []):
                target = conn['target']
                if target not in visited or (target == start and len(visited) > 3):
                    next_node = target
                    break

            if not next_node:
                break
            current = next_node

    def get_edge_sequence(self, edge: str) -> List[str]:
        """Get conveyors on an edge in flow order."""
        edge_conveyors = set(self.edges[edge]['conveyors'])
        sequence = []

        for node in self.loop_path:
            if node in edge_conveyors:
                sequence.append(node)

        return sequence

    def get_summary(self) -> Dict:
        """Get a summary of the learned structure."""
        return {
            'corners': list(self.corners.keys()),
            'corner_positions': self.corner_positions,
            'bounds': self.bounds if hasattr(self, 'bounds') else None,
            'edges': {edge: {k: len(v) for k, v in data.items()}
                     for edge, data in self.edges.items()},
            'loop_length': len(self.loop_path),
            'inline_stations': len(self.inline_stations)
        }


class StructureAwareReconfiguration:
    """Applies structure-preserving modifications to a manufacturing layout."""

    ROOM_ELEMENTS = {'south_wall', 'north_wall', 'east_wall', 'west_wall', 'ceiling', 'middle of the room'}

    def __init__(self, baseline_scene: List[Dict], use_as_template: bool = False):
        """Initialize with baseline scene graph.

        Args:
            baseline_scene: The baseline scene graph to use
            use_as_template: If True, use baseline as structural template only (learn organization pattern)
                            but start with an empty scene (only room elements) instead of preserving equipment
        """
        self.baseline = deepcopy(baseline_scene)
        self.structure = LoopStructure(baseline_scene)
        self.use_as_template = use_as_template

        if use_as_template:
            # Template mode: start with empty scene (only room elements)
            # The structure is learned from baseline but equipment is not preserved
            room_elements = [o for o in baseline_scene if o.get('object_id') in self.ROOM_ELEMENTS]
            self.modified_scene = deepcopy(room_elements)
            print("   Template mode: starting with empty scene, using baseline structure as pattern")
        else:
            # Normal mode: preserve existing equipment and modify
            self.modified_scene = deepcopy(baseline_scene)

        # Track next available IDs
        self._init_id_counters()

    def _init_id_counters(self):
        """Initialize counters for generating new object IDs."""
        self.id_counters = defaultdict(int)

        for obj in self.baseline:
            obj_id = obj.get('object_id', '')
            parts = obj_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
                num = int(parts[1])
                self.id_counters[base] = max(self.id_counters[base], num)

    def _get_next_id(self, base_type: str) -> str:
        """Get next available ID for an equipment type."""
        self.id_counters[base_type] += 1
        return f"{base_type}_{self.id_counters[base_type]}"

    def _get_equipment_template(self, equipment_type: str) -> Dict:
        """Get a template for new equipment based on catalog or existing equipment."""
        # First try to get from equipment catalog for accurate sizes
        try:
            from catalog.equipment_catalog import EQUIPMENT_CATALOG
            if equipment_type in EQUIPMENT_CATALOG:
                cat_entry = EQUIPMENT_CATALOG[equipment_type]
                size = cat_entry.get('approximate_size', {})
                return {
                    'object_id': None,
                    'position': {'x': 0, 'y': 0, 'z': 0},
                    'size_in_meters': {
                        'length': size.get('length', 1.0),
                        'width': size.get('width', 1.0),
                        'height': size.get('height', 1.0)
                    },
                    'style': cat_entry.get('style', 'Industrial'),
                    'material': cat_entry.get('material', 'Metal'),
                    'is_on_the_floor': True,
                    'rotation': {'z_angle': 0},
                    'connections': [],
                    'placement': {'room_layout_elements': [], 'objects_in_room': []}
                }
        except ImportError:
            pass

        # Fallback: look for existing equipment in baseline
        for obj in self.baseline:
            obj_id = obj.get('object_id', '')
            if obj_id.startswith(equipment_type + '_'):
                template = deepcopy(obj)
                # Clear instance-specific data
                template['object_id'] = None
                template['position'] = None
                template['connections'] = []
                return template

        # Return minimal template if no existing equipment found
        return {
            'object_id': None,
            'position': {'x': 0, 'y': 0, 'z': 0},
            'size_in_meters': {'length': 1.0, 'width': 1.0, 'height': 1.0},
            'rotation': {'z_angle': 0},
            'connections': [],
            'placement': {'room_layout_elements': [], 'objects_in_room': []}
        }

    def build_conveyor_system(self, layout: str = "L-shape",
                               conveyors: List[Dict] = None,
                               inline_stations: List[Dict] = None) -> Dict:
        """
        Build a flexible conveyor system with configurable layout and equipment.

        This creates conveyor infrastructure centered in the workspace,
        allowing various layouts (not just rectangular loops).

        Args:
            layout: Layout type - "L-shape", "U-shape", "straight", "loop", or "custom"
            conveyors: List of conveyor configs for custom layout:
                       [{"type": "Line_04", "start": [x,y], "end": [x,y]}, ...]
            inline_stations: List of inline station configs:
                            [{"type": "Line_06", "on_conveyor": 0, "position": 0.5}, ...]
                            Types can be any equipment from catalog.

        Returns:
            Summary of changes made
        """
        changes = {'added': []}

        if not hasattr(self.structure, 'bounds'):
            return {'error': 'No structure bounds learned from template'}

        bounds = self.structure.bounds
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']
        center_x = (bounds['min_x'] + bounds['max_x']) / 2
        center_y = (bounds['min_y'] + bounds['max_y']) / 2

        # Get standard conveyor sizes from catalog
        corner_size = 0.77  # Line_03 size
        conv_width = 0.6    # Conveyor width

        # Calculate layout-specific positions centered in workspace
        if layout == "custom" and conveyors:
            conveyor_configs = conveyors
        elif layout == "L-shape":
            # L-shape: entry from west, turn north at center
            # Centered in workspace with proper spacing
            h_length = width * 0.4  # Horizontal segment
            v_length = height * 0.4  # Vertical segment
            corner_x = center_x
            corner_y = center_y

            conveyor_configs = [
                {"type": "Line_04", "pos": [corner_x - h_length/2 - corner_size/2, corner_y],
                 "length": h_length, "rotation": 90},
                {"type": "Line_03", "pos": [corner_x, corner_y], "rotation": 0},
                {"type": "Line_05", "pos": [corner_x, corner_y + v_length/2 + corner_size/2],
                 "length": v_length, "rotation": 0},
            ]
        elif layout == "U-shape":
            # U-shape: west side up, across top, east side down
            margin = 1.5
            left_x = bounds['min_x'] + margin
            right_x = bounds['max_x'] - margin
            top_y = bounds['max_y'] - margin
            v_length = height * 0.5
            h_length = right_x - left_x - corner_size

            conveyor_configs = [
                {"type": "Line_05", "pos": [left_x, center_y], "length": v_length, "rotation": 0},
                {"type": "Line_03", "pos": [left_x, top_y], "rotation": 270},
                {"type": "Line_04", "pos": [center_x, top_y], "length": h_length, "rotation": 90},
                {"type": "Line_03", "pos": [right_x, top_y], "rotation": 180},
                {"type": "Line_05", "pos": [right_x, center_y], "length": v_length, "rotation": 0},
            ]
        elif layout == "straight":
            # Simple straight line across center
            conveyor_configs = [
                {"type": "Line_04", "pos": [center_x, center_y], "length": width * 0.7, "rotation": 90},
            ]
        else:  # "loop"
            conveyor_configs = self._get_loop_config(bounds)

        # Create conveyors
        created_conveyors = []
        prev_obj = None

        for i, config in enumerate(conveyor_configs):
            conv_type = config.get('type', 'Line_04')
            pos = config.get('pos', [center_x, center_y])
            rotation = config.get('rotation', 0)
            length = config.get('length')

            conv = self._create_conveyor_piece(conv_type,
                {'x': pos[0], 'y': pos[1], 'z': 0},
                rotation, length=length)

            # Add placement relationship
            if prev_obj:
                conv['placement'] = {
                    'room_layout_elements': [],
                    'objects_in_room': [
                        {'object_id': prev_obj['object_id'], 'preposition': 'connected to', 'is_adjacent': True}
                    ]
                }
                # Add connection
                prev_obj['connections'].append({
                    'object_id': conv['object_id'],
                    'connection_type': 'connected',
                    'source_endpoint': 'primary_forward_positive',
                    'target_endpoint': 'primary_forward_negative'
                })

            created_conveyors.append(conv)
            changes['added'].append(conv['object_id'])
            prev_obj = conv

        # Add inline stations (placed adjacent to conveyors, not overlapping)
        if inline_stations:
            for station_config in inline_stations:
                station_type = station_config.get('type', 'Line_06')
                conv_idx = station_config.get('on_conveyor', 0)
                position = station_config.get('position', 0.5)
                side = station_config.get('side', 'outer')  # 'outer' or 'inner'

                if conv_idx < len(created_conveyors):
                    conv = created_conveyors[conv_idx]
                    station = self._create_adjacent_station(conv, station_type, position, side)
                    changes['added'].append(station['object_id'])

        # Store for reference and update structure bounds for workstation placement
        self._created_conveyors = created_conveyors
        self._update_bounds_from_conveyors(created_conveyors)

        print(f"   Built conveyor system ({layout}) with {len(changes['added'])} objects")
        return changes

    def _update_bounds_from_conveyors(self, conveyors: List[Dict]):
        """Update structure bounds based on created conveyors for proper workstation placement."""
        if not conveyors:
            return

        min_x = min(c['position']['x'] for c in conveyors)
        max_x = max(c['position']['x'] for c in conveyors)
        min_y = min(c['position']['y'] for c in conveyors)
        max_y = max(c['position']['y'] for c in conveyors)

        # Add conveyor sizes to bounds
        for c in conveyors:
            size = c.get('size_in_meters', {})
            length = size.get('length', 2.0)
            width = size.get('width', 0.6)
            rot = c.get('rotation', {}).get('z_angle', 0)

            if rot in [0, 180]:  # Vertical
                min_y = min(min_y, c['position']['y'] - length/2)
                max_y = max(max_y, c['position']['y'] + length/2)
            else:  # Horizontal
                min_x = min(min_x, c['position']['x'] - length/2)
                max_x = max(max_x, c['position']['x'] + length/2)

        # Update structure bounds
        self.structure.bounds = {
            'min_x': min_x - 0.5,
            'max_x': max_x + 0.5,
            'min_y': min_y - 0.5,
            'max_y': max_y + 0.5
        }

    def _create_adjacent_station(self, conveyor: Dict, station_type: str,
                                  position: float = 0.5, side: str = 'outer') -> Dict:
        """Create a station positioned adjacent to (beside) a conveyor."""
        conv_pos = conveyor['position']
        conv_rot = conveyor.get('rotation', {}).get('z_angle', 0)
        conv_size = conveyor.get('size_in_meters', {})
        conv_length = conv_size.get('length', 2.0)

        # Get station size
        station_template = self._get_equipment_template(station_type)
        station_size = station_template.get('size_in_meters', {})
        station_width = max(station_size.get('length', 1.0), station_size.get('width', 1.0))

        # Offset from conveyor center (perpendicular to flow)
        offset = 0.6 + station_width / 2  # conveyor half-width + gap + station half-width
        if side == 'inner':
            offset = -offset

        # Calculate position based on conveyor orientation
        if conv_rot in [0, 180]:  # Vertical conveyor
            station_pos = {
                'x': conv_pos['x'] + offset,
                'y': conv_pos['y'] + (position - 0.5) * conv_length,
                'z': 0
            }
            station_rot = 90 if side == 'outer' else 270
        else:  # Horizontal conveyor
            station_pos = {
                'x': conv_pos['x'] + (position - 0.5) * conv_length,
                'y': conv_pos['y'] + offset,
                'z': 0
            }
            station_rot = 0 if side == 'outer' else 180

        return self._create_support_equipment(station_type, station_pos, station_rot)

    def _get_loop_config(self, bounds: Dict) -> List[Dict]:
        """Generate configuration for a rectangular loop layout."""
        margin = 1.0
        left = bounds['min_x'] + margin
        right = bounds['max_x'] - margin
        bottom = bounds['min_y'] + margin
        top = bounds['max_y'] - margin
        h_length = (right - left) * 0.8
        v_length = (top - bottom) * 0.8
        center_x = (left + right) / 2
        center_y = (bottom + top) / 2

        return [
            {"type": "Line_03", "pos": [left, bottom], "rotation": 0},
            {"type": "Line_04", "pos": [center_x, bottom], "length": h_length, "rotation": 90},
            {"type": "Line_03", "pos": [right, bottom], "rotation": 90},
            {"type": "Line_05", "pos": [right, center_y], "length": v_length, "rotation": 0},
            {"type": "Line_03", "pos": [right, top], "rotation": 180},
            {"type": "Line_04", "pos": [center_x, top], "length": h_length, "rotation": 90},
            {"type": "Line_03", "pos": [left, top], "rotation": 270},
            {"type": "Line_05", "pos": [left, center_y], "length": v_length, "rotation": 0},
        ]

    def _create_conveyor_piece(self, conv_type: str, position: Dict, rotation: float, length: float = None) -> Dict:
        """Create a single conveyor piece.

        The conveyor size strictly adheres to the equipment catalog.
        The length parameter is ignored to maintain catalog compliance.
        """
        template = self._get_equipment_template(conv_type)
        conv = deepcopy(template)
        conv['object_id'] = self._get_next_id(conv_type)
        conv['position'] = deepcopy(position)
        conv['rotation'] = {'z_angle': rotation}
        conv['connections'] = []
        conv['placement'] = {'room_layout_elements': [], 'objects_in_room': []}

        # Do NOT modify size_in_meters - always use catalog size as-is
        # The length parameter is kept for API compatibility but ignored

        self.modified_scene.append(conv)
        return conv

    def build_new_loop(self, num_conveyors_per_edge: int = 2,
                        add_lift: bool = True, add_inspection: bool = True,
                        add_assembly: bool = True) -> Dict:
        """
        Build a rectangular conveyor loop (legacy method - calls build_conveyor_system).

        For more flexibility, use build_conveyor_system directly.
        """
        inline_stations = []
        if add_lift:
            inline_stations.append({"type": "Line_01", "on_conveyor": 1, "position": 0.3})
        if add_inspection:
            inline_stations.append({"type": "Line_06", "on_conveyor": 3, "position": 0.5})
        if add_assembly:
            inline_stations.append({"type": "Line_07", "on_conveyor": 5, "position": 0.5})

        return self.build_conveyor_system(layout="loop", inline_stations=inline_stations)

    def build_base_loop(self) -> Dict:
        """
        DEPRECATED: Use build_new_loop() instead for template mode.

        This method copies equipment from baseline which defeats the purpose of template mode.
        Kept for backwards compatibility but will emit a warning.
        """
        print("   WARNING: build_base_loop copies from baseline. Use build_new_loop for fresh layouts.")
        return self.build_new_loop()

    def extend_edge(self, edge: str, extension_distance: float = 3.0,
                    add_stations: List[str] = None) -> Dict:
        """
        Extend the loop outward on one edge.

        When extending, the original corners on that edge are removed because the path
        no longer turns there - it continues straight outward. New corners ARE created
        at the extended positions because the path turns there (from side conveyors to
        the main extended conveyor).

        Args:
            edge: Which edge to extend ('south', 'north', 'east', 'west')
            extension_distance: How far to extend in meters
            add_stations: Station types to add on the extended section

        Returns:
            Summary of changes made
        """
        if edge not in ['south', 'north', 'east', 'west']:
            raise ValueError(f"Invalid edge: {edge}")

        changes = {'added': [], 'modified': [], 'removed': []}

        # Find the corners that bound this edge
        if edge == 'south':
            corner1, corner2 = 'SW', 'SE'
            extend_dir = (0, -extension_distance)  # Extend south (decrease y)
        elif edge == 'north':
            corner1, corner2 = 'NW', 'NE'
            extend_dir = (0, extension_distance)   # Extend north (increase y)
        elif edge == 'west':
            corner1, corner2 = 'SW', 'NW'
            extend_dir = (-extension_distance, 0)  # Extend west (decrease x)
        else:  # east
            corner1, corner2 = 'SE', 'NE'
            extend_dir = (extension_distance, 0)   # Extend east (increase x)

        # Get corner objects
        c1_id = self.structure.corner_positions.get(corner1)
        c2_id = self.structure.corner_positions.get(corner2)

        if not c1_id or not c2_id:
            return {'error': f'Could not find corners for {edge} edge'}

        c1_obj = self._find_object(c1_id)
        c2_obj = self._find_object(c2_id)

        # Store original corner positions before removal
        c1_pos = deepcopy(c1_obj['position'])
        c2_pos = deepcopy(c2_obj['position'])

        # IMPORTANT: Create extension conveyors BEFORE removing corners
        # because _remove_object also removes connections to the corner,
        # and we need those connections to find which conveyors to modify
        # The extension conveyors determine where the corners should be placed
        conv1, new_c1_pos = self._create_extension_conveyor(c1_id, corner1, c1_pos, edge, changes)
        conv2, new_c2_pos = self._create_extension_conveyor(c2_id, corner2, c2_pos, edge, changes)

        # Now remove the original corners - the path no longer turns there
        self._remove_object(c1_id)
        self._remove_object(c2_id)
        changes['removed'].extend([c1_id, c2_id])

        # Create new corners at positions determined by conveyor chain
        new_corner1 = self._create_corner(new_c1_pos, c1_obj)
        new_corner2 = self._create_corner(new_c2_pos, c2_obj)
        changes['added'].extend([new_corner1['object_id'], new_corner2['object_id']])

        # Create main conveyor chain along the extended edge (between the two new corners)
        # Adjust start/end positions to account for corner sizes - conveyors should start
        # at the edge of corners, not at corner centers
        corner_template = self._get_equipment_template('Line_03')
        corner_half_size = corner_template.get('size_in_meters', {}).get('length', 0.769) / 2

        # Calculate adjusted positions (from corner edge to corner edge)
        if edge in ['west', 'east']:
            # Vertical main edge - adjust y positions
            chain_start = {
                'x': new_c1_pos['x'],
                'y': new_c1_pos['y'] + corner_half_size,  # Start from top edge of bottom corner
                'z': new_c1_pos.get('z', 0)
            }
            chain_end = {
                'x': new_c2_pos['x'],
                'y': new_c2_pos['y'] - corner_half_size,  # End at bottom edge of top corner
                'z': new_c2_pos.get('z', 0)
            }
        else:
            # Horizontal main edge - adjust x positions
            chain_start = {
                'x': new_c1_pos['x'] + corner_half_size,
                'y': new_c1_pos['y'],
                'z': new_c1_pos.get('z', 0)
            }
            chain_end = {
                'x': new_c2_pos['x'] - corner_half_size,
                'y': new_c2_pos['y'],
                'z': new_c2_pos.get('z', 0)
            }

        main_conv_chain = self._create_conveyor_chain(chain_start, chain_end, 'Line_04')
        changes['added'].extend([c['object_id'] for c in main_conv_chain])

        # For connection updates, use the first/last conveyors
        main_conv = main_conv_chain[0] if main_conv_chain else None

        # Add stations if requested
        # Separate inline stations (Line_06, Line_07) from side equipment (Line_02)
        if add_stations:
            inline_types = ['Line_06', 'Line_07']
            side_types = ['Line_02', 'Line_01', 'Line_08']

            inline_stations = [s for s in add_stations if any(t in s for t in inline_types)]
            side_equipment = [s for s in add_stations if any(t in s for t in side_types)]

            # Get conveyor properties - use new corner positions for accurate length
            conv_pos = main_conv['position']

            # Calculate actual conveyor span from new corner positions
            c1_y = new_c1_pos['y']
            c2_y = new_c2_pos['y']
            c1_x = new_c1_pos['x']
            c2_x = new_c2_pos['x']

            # Determine if conveyor is horizontal or vertical based on endpoint positions
            is_vertical = abs(c1_x - c2_x) < abs(c1_y - c2_y)

            if is_vertical:
                # Vertical conveyor (north-south)
                conveyor_span = abs(c2_y - c1_y)
                min_coord = min(c1_y, c2_y)
            else:
                # Horizontal conveyor (east-west)
                conveyor_span = abs(c2_x - c1_x)
                min_coord = min(c1_x, c2_x)

            # Place inline stations with proper spacing along the conveyor
            if inline_stations:
                num_inline = len(inline_stations)
                # Get station size to calculate proper spacing
                station_size = 3.0  # Typical station size (Line_07 is about 2.5m)

                # Calculate spacing between stations
                # Leave margin at ends to avoid overlap with connecting conveyors
                end_margin = 2.5  # Keep stations away from conveyor ends
                max_coord = min_coord + conveyor_span

                # Calculate usable range for placing stations
                usable_start = min_coord + end_margin
                usable_end = max_coord - end_margin
                usable_length = usable_end - usable_start

                # Distribute stations evenly in the usable range
                if num_inline == 1:
                    # Single station goes in the middle
                    positions = [(usable_start + usable_end) / 2]
                else:
                    # Multiple stations: space them evenly
                    # Ensure minimum spacing between stations
                    min_spacing = station_size + 0.5  # Station size plus small gap
                    required_length = (num_inline - 1) * min_spacing

                    if required_length > usable_length:
                        # Not enough space, compress spacing
                        actual_spacing = usable_length / (num_inline - 1) if num_inline > 1 else 0
                    else:
                        # Plenty of space, distribute evenly
                        actual_spacing = usable_length / (num_inline - 1) if num_inline > 1 else 0

                    positions = [usable_start + actual_spacing * i for i in range(num_inline)]

                for i, station_type in enumerate(inline_stations):
                    pos_along_conveyor = positions[i]

                    if is_vertical:
                        station_pos = {
                            'x': conv_pos['x'],
                            'y': pos_along_conveyor,
                            'z': conv_pos.get('z', 0)
                        }
                    else:
                        station_pos = {
                            'x': pos_along_conveyor,
                            'y': conv_pos['y'],
                            'z': conv_pos.get('z', 0)
                        }

                    station = self._create_inline_station_at(main_conv, station_type, station_pos)
                    if station:
                        changes['added'].append(station['object_id'])

            # Place side equipment (Line_02) adjacent to the conveyor (side-attached)
            # Position them in BETWEEN inline stations to avoid overlap
            if side_equipment:
                # Get conveyor width to calculate adjacent position
                conv_width = main_conv.get('size_in_meters', {}).get('width', 0.6)

                # Calculate positions along conveyor for side equipment
                # Place them between inline stations or in gaps
                num_side = len(side_equipment)

                # If we have inline stations, place side equipment between them
                # Otherwise, place at the middle of the conveyor
                if inline_stations and len(inline_stations) >= 2:
                    # Place between first and second inline station
                    side_positions_along = [(positions[0] + positions[1]) / 2]
                    # If more side equipment, place after last inline station
                    for j in range(1, num_side):
                        side_positions_along.append(positions[-1] + station_size)
                elif inline_stations and len(inline_stations) == 1:
                    # Single inline station - place side equipment before it
                    side_positions_along = [positions[0] - station_size]
                else:
                    # No inline stations - place at center
                    side_positions_along = [(usable_start + usable_end) / 2]

                for i, equip_type in enumerate(side_equipment):
                    # Get equipment size for proper adjacent placement
                    equip_template = self._get_equipment_template(equip_type)
                    equip_width = equip_template.get('size_in_meters', {}).get('width', 0.7)
                    equip_length = equip_template.get('size_in_meters', {}).get('length', 0.7)

                    # Place adjacent (touching) the conveyor
                    # Side offset = half conveyor width + half equipment width + small gap
                    side_offset = (conv_width / 2) + (equip_width / 2) + 0.1

                    # Get position along the conveyor
                    along_offset = side_positions_along[i] if i < len(side_positions_along) else side_positions_along[-1]

                    if is_vertical:
                        # Vertical conveyor (north-south) on west or east edge
                        if edge == 'west':
                            side_x = conv_pos['x'] - side_offset
                        else:  # east
                            side_x = conv_pos['x'] + side_offset
                        side_y = along_offset
                    else:
                        # Horizontal conveyor (east-west) on north or south edge
                        side_x = along_offset
                        if edge == 'south':
                            side_y = conv_pos['y'] - side_offset
                        else:  # north
                            side_y = conv_pos['y'] + side_offset

                    side_pos = {'x': side_x, 'y': side_y, 'z': conv_pos.get('z', 0)}

                    equip = self._create_side_equipment(main_conv, equip_type, side_pos, edge)
                    if equip:
                        changes['added'].append(equip['object_id'])

        # Update connections to incorporate the new extended edge into the loop
        # 1. Extended side conveyors (conv1, conv2) connect to new corners
        # 2. New corners connect to main conveyor chain
        if conv1 and conv2 and main_conv_chain:
            main_first = main_conv_chain[0]
            main_last = main_conv_chain[-1]

            # Side conveyor 1 connects to new corner 1
            conv1['connections'].append({
                'object_id': new_corner1['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # Side conveyor 2 connects to new corner 2
            conv2['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # New corner 1 connects to main conveyor chain (first conveyor)
            new_corner1['connections'].append({
                'object_id': main_first['object_id'],
                'connection_type': 'connected_corner',
                'source_endpoint': 'forward_positive',
                'target_endpoint': 'primary_forward_negative'
            })

            # Main conveyor chain (last conveyor) connects to new corner 2
            main_last['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_positive'
            })

        return changes

    def add_inline_station(self, conveyor_id: str, station_type: str) -> Dict:
        """
        Add a processing station inline on an existing conveyor.

        Args:
            conveyor_id: ID of conveyor to place station on
            station_type: Type of station ('Line_06', 'Line_07')

        Returns:
            The new station object
        """
        conveyor = self._find_object(conveyor_id)
        if not conveyor:
            return {'error': f'Conveyor {conveyor_id} not found'}

        station = self._create_inline_station(conveyor, station_type)
        return station

    def _find_object(self, obj_id: str) -> Optional[Dict]:
        """Find object by ID in modified scene."""
        for obj in self.modified_scene:
            if obj.get('object_id') == obj_id:
                return obj
        return None

    def _create_corner(self, position: Dict, template_corner: Dict) -> Dict:
        """Create a new corner piece."""
        new_id = self._get_next_id('Line_03')

        corner = deepcopy(template_corner)
        corner['object_id'] = new_id
        corner['position'] = position
        corner['connections'] = []

        # Update placement
        corner['placement'] = {
            'room_layout_elements': [],
            'objects_in_room': []
        }

        self.modified_scene.append(corner)
        return corner

    def _create_conveyor_between(self, obj1: Dict, obj2: Dict, conv_type: str) -> Dict:
        """Create a conveyor connecting two objects."""
        p1 = obj1['position']
        p2 = obj2['position']
        return self._create_conveyor_at_positions(p1, p2, conv_type, ref_object_id=obj1.get('object_id'))

    def _create_extension_conveyor(self, old_corner_id: str, corner_name: str,
                                     old_corner_pos: Dict, extend_edge: str,
                                     changes: Dict) -> tuple:
        """Extend by bridging with conveyors first, then determine corner position.

        Smart approach:
        1. Find the original short conveyor (Line_05) connected to the removed corner
        2. Replace it with a long conveyor (Line_04) - keeps one end connected, extends the other
        3. Add a short conveyor (Line_05) at the extended end
        4. Return the position where the corner should be placed (at the end of Line_05)

        Args:
            old_corner_id: ID of the corner that was removed
            corner_name: Name of corner ('SW', 'SE', 'NW', 'NE')
            old_corner_pos: Original position of the removed corner
            extend_edge: Which edge is being extended ('south', 'north', 'east', 'west')
            changes: Changes dict to update with modifications

        Returns:
            Tuple of (last_conveyor, corner_position) where corner should be placed
        """
        # Determine which adjacent edge to look at based on corner and extend edge
        adjacent_edge_map = {
            ('west', 'SW'): 'south',
            ('west', 'NW'): 'north',
            ('east', 'SE'): 'south',
            ('east', 'NE'): 'north',
            ('south', 'SW'): 'west',
            ('south', 'SE'): 'east',
            ('north', 'NW'): 'west',
            ('north', 'NE'): 'east',
        }

        adjacent_edge = adjacent_edge_map.get((extend_edge, corner_name))
        if not adjacent_edge:
            return None

        # Find the conveyor that was connected to the removed corner
        edge_conveyors = self.structure.edges.get(adjacent_edge, {}).get('conveyors', [])
        original_conv = None

        for conv_id in edge_conveyors:
            conv = self._find_object(conv_id)
            if not conv:
                continue
            was_connected = any(
                conn.get('object_id') == old_corner_id
                for conn in conv.get('connections', [])
            )
            if was_connected:
                original_conv = conv
                break

        if not original_conv:
            return None

        # Get equipment sizes from catalog
        line04_template = self._get_equipment_template('Line_04')
        line05_template = self._get_equipment_template('Line_05')
        line03_template = self._get_equipment_template('Line_03')

        line04_length = line04_template.get('size_in_meters', {}).get('length', 4.65)
        line05_length = line05_template.get('size_in_meters', {}).get('length', 2.247)
        corner_size = line03_template.get('size_in_meters', {}).get('length', 0.769)
        half_corner = corner_size / 2

        # Get original conveyor info
        original_pos = original_conv['position']
        original_rot = original_conv.get('rotation', {}).get('z_angle', 0)
        original_id = original_conv['object_id']

        # Keep connections that are NOT to the removed corner
        other_connections = [
            conn for conn in original_conv.get('connections', [])
            if conn.get('object_id') != old_corner_id
        ]

        # Step 1: Position Line_04 starting from where original Line_05 was
        # Keep the connection end at the same place, extend in the direction of the edge
        new_long_id = self._get_next_id('Line_04')
        half_line04 = line04_length / 2
        half_line05 = line05_length / 2

        # Original Line_05 had one end connected to another conveyor, other end to corner
        # We keep the "connected" end in place and extend the "corner" end
        if extend_edge == 'west':
            # Original's east end connected to another conveyor - keep it there
            original_east_end = original_pos['x'] + half_line05
            new_center_x = original_east_end - half_line04
            new_center_y = original_pos['y']
            # Line_04's west end position
            line04_extend_end = new_center_x - half_line04
        elif extend_edge == 'east':
            original_west_end = original_pos['x'] - half_line05
            new_center_x = original_west_end + half_line04
            new_center_y = original_pos['y']
            line04_extend_end = new_center_x + half_line04
        elif extend_edge == 'south':
            original_north_end = original_pos['y'] + half_line05
            new_center_x = original_pos['x']
            new_center_y = original_north_end - half_line04
            line04_extend_end = new_center_y - half_line04
        else:  # north
            original_south_end = original_pos['y'] - half_line05
            new_center_x = original_pos['x']
            new_center_y = original_south_end + half_line04
            line04_extend_end = new_center_y + half_line04

        # Update the original conveyor to become Line_04
        original_conv['object_id'] = new_long_id
        original_conv['type'] = 'Line_04'
        original_conv['size_in_meters'] = deepcopy(line04_template.get('size_in_meters', {}))
        original_conv['position'] = {
            'x': new_center_x,
            'y': new_center_y,
            'z': original_pos.get('z', 0)
        }
        original_conv['connections'] = other_connections

        changes['removed'].append(original_id)
        changes['added'].append(new_long_id)

        # Step 2: Add a short conveyor (Line_05) connecting to Line_04's extended end
        # Position it so one end touches Line_04, other end determines where corner will be
        new_short_id = self._get_next_id('Line_05')

        if extend_edge == 'west':
            # Line_05's east end touches Line_04's west end
            short_center_x = line04_extend_end - half_line05
            short_center_y = original_pos['y']
            short_rotation = 90
            # Corner will be at Line_05's west end
            corner_pos_x = short_center_x - half_line05 - half_corner
            corner_pos_y = short_center_y
        elif extend_edge == 'east':
            # Line_05's west end touches Line_04's east end
            short_center_x = line04_extend_end + half_line05
            short_center_y = original_pos['y']
            short_rotation = 90
            corner_pos_x = short_center_x + half_line05 + half_corner
            corner_pos_y = short_center_y
        elif extend_edge == 'south':
            # Line_05's north end touches Line_04's south end
            short_center_x = original_pos['x']
            short_center_y = line04_extend_end - half_line05
            short_rotation = 0
            corner_pos_x = short_center_x
            corner_pos_y = short_center_y - half_line05 - half_corner
        else:  # north
            # Line_05's south end touches Line_04's north end
            short_center_x = original_pos['x']
            short_center_y = line04_extend_end + half_line05
            short_rotation = 0
            corner_pos_x = short_center_x
            corner_pos_y = short_center_y + half_line05 + half_corner

        short_conv = {
            'object_id': new_short_id,
            'type': 'Line_05',
            'position': {
                'x': short_center_x,
                'y': short_center_y,
                'z': original_pos.get('z', 0)
            },
            'rotation': {'z_angle': short_rotation},
            'size_in_meters': deepcopy(line05_template.get('size_in_meters', {})),
            'connections': [],
            'placement': {
                'room_layout_elements': [],
                'objects_in_room': []
            }
        }

        self.modified_scene.append(short_conv)
        changes['added'].append(new_short_id)

        # Connect Line_04 to Line_05
        original_conv['connections'].append({
            'object_id': new_short_id,
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_positive',
            'target_endpoint': 'primary_forward_negative'
        })

        # Update connections from other conveyors that pointed to the original
        # They should now point to the new Line_04
        for obj in self.modified_scene:
            if obj.get('object_id') == new_long_id:
                continue  # Skip the conveyor we just modified
            connections = obj.get('connections')
            if connections:
                for conn in connections:
                    if conn.get('object_id') == original_id:
                        conn['object_id'] = new_long_id

        # Return the last conveyor and the calculated corner position
        corner_pos = {
            'x': corner_pos_x,
            'y': corner_pos_y,
            'z': original_pos.get('z', 0)
        }
        return short_conv, corner_pos

    def _create_conveyor_chain(self, start_pos: Dict, end_pos: Dict, conv_type: str) -> List[Dict]:
        """Create a chain of conveyors to span from start to end position.

        Uses catalog-sized conveyors placed sequentially. The number of conveyors
        is calculated based on the distance and catalog conveyor length.

        Args:
            start_pos: Starting position dict with x, y, z
            end_pos: Ending position dict with x, y, z
            conv_type: Conveyor type (e.g., 'Line_04', 'Line_05')

        Returns:
            List of created conveyor objects in order from start to end
        """
        # Get catalog size for this conveyor type
        template = self._get_equipment_template(conv_type)
        catalog_length = template.get('size_in_meters', {}).get('length', 2.0)

        # Calculate distance and direction
        dx = end_pos['x'] - start_pos['x']
        dy = end_pos['y'] - start_pos['y']
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            # Positions are essentially the same, no conveyor needed
            return []

        # Determine orientation
        is_horizontal = abs(dx) > abs(dy)
        rotation = 90 if is_horizontal else 0

        # Calculate number of conveyors needed
        num_conveyors = max(1, round(distance / catalog_length))

        # Calculate step size to evenly distribute conveyors
        if is_horizontal:
            step_x = dx / num_conveyors
            step_y = 0
        else:
            step_x = 0
            step_y = dy / num_conveyors

        conveyors = []
        prev_conv = None

        for i in range(num_conveyors):
            # Position at center of this segment
            conv_pos = {
                'x': start_pos['x'] + step_x * (i + 0.5),
                'y': start_pos['y'] + step_y * (i + 0.5),
                'z': 0
            }

            conv = self._create_single_conveyor(conv_type, conv_pos, rotation)
            conveyors.append(conv)

            # Connect to previous conveyor in chain
            if prev_conv:
                prev_conv['connections'].append({
                    'object_id': conv['object_id'],
                    'connection_type': 'connected',
                    'source_endpoint': 'primary_forward_positive',
                    'target_endpoint': 'primary_forward_negative'
                })

            prev_conv = conv

        return conveyors

    def _create_single_conveyor(self, conv_type: str, position: Dict, rotation: float) -> Dict:
        """Create a single conveyor with catalog size at specified position."""
        template = self._get_equipment_template(conv_type)
        conv = deepcopy(template)
        conv['object_id'] = self._get_next_id(conv_type)
        conv['position'] = deepcopy(position)
        conv['rotation'] = {'z_angle': rotation}
        conv['connections'] = []
        conv['placement'] = {'room_layout_elements': [], 'objects_in_room': []}

        self.modified_scene.append(conv)
        return conv

    def _create_conveyor_at_positions(self, pos1: Dict, pos2: Dict, conv_type: str,
                                       ref_object_id: str = None) -> Dict:
        """Create a conveyor connecting two positions.

        The conveyor length is taken from the equipment catalog, NOT calculated
        from the distance between positions. The conveyor is placed at the midpoint.

        Args:
            pos1: Start position dict with x, y, z
            pos2: End position dict with x, y, z
            conv_type: Conveyor type (e.g., 'Line_04', 'Line_05')
            ref_object_id: Optional reference object ID for placement relationship

        Returns:
            The created conveyor object
        """
        new_id = self._get_next_id(conv_type)

        # Get template - size comes from catalog, not calculated
        template = self._get_equipment_template(conv_type)

        # Calculate position (midpoint)
        mid_x = (pos1['x'] + pos2['x']) / 2
        mid_y = (pos1['y'] + pos2['y']) / 2

        conveyor = deepcopy(template)
        conveyor['object_id'] = new_id
        conveyor['position'] = {'x': mid_x, 'y': mid_y, 'z': 0}
        conveyor['connections'] = []

        # Determine orientation based on direction
        dx = pos2['x'] - pos1['x']
        dy = pos2['y'] - pos1['y']

        if abs(dx) > abs(dy):
            # Horizontal conveyor
            conveyor['rotation'] = {'z_angle': 90}
        else:
            # Vertical conveyor
            conveyor['rotation'] = {'z_angle': 0}

        # Do NOT modify size_in_meters - use catalog size as-is

        # Update placement
        placement = {'room_layout_elements': [], 'objects_in_room': []}
        if ref_object_id:
            placement['objects_in_room'].append(
                {'object_id': ref_object_id, 'preposition': 'right of', 'is_adjacent': True}
            )
        conveyor['placement'] = placement

        self.modified_scene.append(conveyor)
        return conveyor

    def _create_inline_station(self, conveyor: Dict, station_type: str) -> Dict:
        """Create a station positioned inline on a conveyor (at conveyor center)."""
        conv_pos = conveyor['position']
        return self._create_inline_station_at(conveyor, station_type, conv_pos)

    def _create_inline_station_at(self, conveyor: Dict, station_type: str, position: Dict) -> Dict:
        """Create a station positioned inline on a conveyor at specific position."""
        new_id = self._get_next_id(station_type)

        template = self._get_equipment_template(station_type)

        station = deepcopy(template)
        station['object_id'] = new_id
        station['position'] = deepcopy(position)
        station['connections'] = []

        # Match conveyor orientation
        if 'rotation' in conveyor:
            station['rotation'] = deepcopy(conveyor['rotation'])

        # Set placement to be "on" the conveyor
        station['placement'] = {
            'room_layout_elements': [],
            'objects_in_room': [
                {'object_id': conveyor['object_id'], 'preposition': 'on', 'is_adjacent': True}
            ]
        }

        self.modified_scene.append(station)
        return station

    def _create_side_equipment(self, conveyor: Dict, equip_type: str, position: Dict, edge: str) -> Dict:
        """Create equipment positioned beside a conveyor (e.g., side-loader Line_02).

        Args:
            conveyor: The conveyor this equipment serves
            equip_type: Type of equipment (e.g., 'Line_02')
            position: Position beside the conveyor
            edge: Which edge this is on (affects orientation)

        Returns:
            The created equipment object
        """
        new_id = self._get_next_id(equip_type)

        template = self._get_equipment_template(equip_type)

        equip = deepcopy(template)
        equip['object_id'] = new_id
        equip['position'] = deepcopy(position)
        equip['connections'] = []

        # Orient to face the conveyor
        # Side-loaders should face the conveyor they serve
        if edge == 'west':
            equip['rotation'] = {'z_angle': 90}   # Face east toward conveyor
        elif edge == 'east':
            equip['rotation'] = {'z_angle': -90}  # Face west toward conveyor
        elif edge == 'south':
            equip['rotation'] = {'z_angle': 0}    # Face north toward conveyor
        else:  # north
            equip['rotation'] = {'z_angle': 180}  # Face south toward conveyor

        # Set placement to be "beside" the conveyor
        equip['placement'] = {
            'room_layout_elements': [],
            'objects_in_room': [
                {'object_id': conveyor['object_id'], 'preposition': 'left of', 'is_adjacent': True}
            ]
        }

        self.modified_scene.append(equip)
        return equip

    def _update_extended_edge_connections_chain(self, conv1_chain: List[Dict], conv2_chain: List[Dict],
                                                  main_conv_chain: List[Dict],
                                                  new_corner1: Dict, new_corner2: Dict, edge: str):
        """Update connections for extended edge with conveyor chains.

        When extending an edge:
        - Original corners are removed (path no longer turns there)
        - New corners are created at extended positions (path turns there)
        - Side conveyor chains connect existing loop to new corners
        - New corners connect to main conveyor chain

        Args:
            conv1_chain: First side conveyor chain (from original corner position to new corner)
            conv2_chain: Second side conveyor chain (from original corner position to new corner)
            main_conv_chain: Main conveyor chain along the extended edge
            new_corner1: First new corner at extended position
            new_corner2: Second new corner at extended position
            edge: Which edge was extended ('south', 'north', 'east', 'west')
        """
        if not conv1_chain or not conv2_chain or not main_conv_chain:
            return

        # Last conveyor in side chains connects to new corners
        conv1_last = conv1_chain[-1]
        conv2_last = conv2_chain[-1]

        # First and last conveyor in main chain
        main_first = main_conv_chain[0]
        main_last = main_conv_chain[-1]

        if new_corner1 and new_corner2:
            # Side chain 1 (last conveyor) connects to new corner 1
            conv1_last['connections'].append({
                'object_id': new_corner1['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # Side chain 2 (last conveyor) connects to new corner 2
            conv2_last['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # New corner 1 connects to main conveyor chain (first conveyor)
            new_corner1['connections'].append({
                'object_id': main_first['object_id'],
                'connection_type': 'connected_corner',
                'source_endpoint': 'forward_positive',
                'target_endpoint': 'primary_forward_negative'
            })

            # Main conveyor chain (last conveyor) connects to new corner 2
            main_last['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_positive'
            })

    def _update_extended_edge_connections(self, conv1: Dict, conv2: Dict, main_conv: Dict,
                                           new_corner1: Dict = None, new_corner2: Dict = None):
        """Update connections for extended edge conveyors (single conveyor version).

        DEPRECATED: Use _update_extended_edge_connections_chain for chain support.

        When extending an edge:
        - Original corners are removed (path no longer turns there)
        - New corners are created at extended positions (path turns there)
        - Side conveyors connect to new corners
        - New corners connect to main conveyor

        Args:
            conv1: First side conveyor (from original corner position to new corner)
            conv2: Second side conveyor (from original corner position to new corner)
            main_conv: Main conveyor along the extended edge
            new_corner1: First new corner at extended position
            new_corner2: Second new corner at extended position
        """
        if new_corner1 and new_corner2:
            # Side conveyor 1 connects to new corner 1
            conv1['connections'].append({
                'object_id': new_corner1['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # Side conveyor 2 connects to new corner 2
            conv2['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_negative'
            })

            # Main conveyor connects to both new corners
            main_conv['connections'].append({
                'object_id': new_corner1['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_negative',
                'target_endpoint': 'forward_positive'
            })
            main_conv['connections'].append({
                'object_id': new_corner2['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'forward_positive'
            })
        else:
            # Fallback: direct connection without corners (shouldn't happen normally)
            conv1['connections'].append({
                'object_id': main_conv['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'primary_forward_negative'
            })
            conv2['connections'].append({
                'object_id': main_conv['object_id'],
                'connection_type': 'connected',
                'source_endpoint': 'primary_forward_positive',
                'target_endpoint': 'primary_forward_positive'
            })

    def _update_loop_connections(self, old_c1_id: str, old_c2_id: str,
                                  new_c1: Dict, new_c2: Dict,
                                  conv1: Dict, conv2: Dict, main_conv: Dict,
                                  edge: str):
        """Update connections to incorporate new edge into the loop.

        DEPRECATED: This method assumes corners exist at the connection points.
        For extended edges without corners, use _update_extended_edge_connections instead.
        """
        # This is a simplified connection update
        # A full implementation would trace the loop and insert properly

        # Connect old corner 1 to new conveyor 1
        conv1['connections'].append({
            'object_id': old_c1_id,
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_negative',
            'target_endpoint': 'forward_positive'
        })
        conv1['connections'].append({
            'object_id': new_c1['object_id'],
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_positive',
            'target_endpoint': 'forward_negative'
        })

        # Connect old corner 2 to new conveyor 2
        conv2['connections'].append({
            'object_id': old_c2_id,
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_negative',
            'target_endpoint': 'forward_positive'
        })
        conv2['connections'].append({
            'object_id': new_c2['object_id'],
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_positive',
            'target_endpoint': 'forward_negative'
        })

        # Connect new corners via main conveyor
        main_conv['connections'].append({
            'object_id': new_c1['object_id'],
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_negative',
            'target_endpoint': 'forward_positive'
        })
        main_conv['connections'].append({
            'object_id': new_c2['object_id'],
            'connection_type': 'connected',
            'source_endpoint': 'primary_forward_positive',
            'target_endpoint': 'forward_negative'
        })

    # ==================== Production-Focused Operations ====================

    def define_material_flow(self, loading_edge: str, loading_position: float,
                              unloading_edge: str, unloading_position: float,
                              flow_direction: str = 'clockwise') -> Dict:
        """
        Define material flow entry and exit points by creating a physical passage.

        This physically opens the conveyor loop by removing conveyors in the passage
        segment (between unloading and loading, going against flow direction).
        The result is an open production line with clear entry/exit points.

        Args:
            loading_edge: Edge where materials enter ('south', 'north', 'east', 'west')
            loading_position: Position along loading edge (0.0 to 1.0)
            unloading_edge: Edge where products exit
            unloading_position: Position along unloading edge (0.0 to 1.0)
            flow_direction: 'clockwise' or 'counterclockwise'

        Returns:
            Summary including removed conveyors and flow metadata
        """
        changes = {'added': [], 'modified': [], 'removed': []}

        # Calculate passage segment info
        passage_info = self._calculate_passage_segment(
            loading_edge, loading_position,
            unloading_edge, unloading_position,
            flow_direction
        )

        # Physically remove conveyors in the passage segment
        removed_conveyors = self._remove_passage_conveyors(
            loading_edge, loading_position, None,
            unloading_edge, unloading_position, None,
            flow_direction
        )
        changes['removed'].extend(removed_conveyors)

        # Find the two endpoints of the physical gap
        # These are where Loading and Unloading points should be marked
        loading_world_pos, unloading_world_pos = self._find_gap_endpoints(
            removed_conveyors, flow_direction
        )

        # Create flow metadata object to store in scene
        flow_metadata = {
            'object_id': 'MaterialFlow_1',
            'type': 'material_flow_definition',
            'flow_direction': flow_direction,
            'loading_point': {
                'edge': loading_edge,
                'position': loading_position,
                'world_position': loading_world_pos
            },
            'unloading_point': {
                'edge': unloading_edge,
                'position': unloading_position,
                'world_position': unloading_world_pos
            },
            'passage_segment': passage_info,
            'removed_conveyors': removed_conveyors
        }

        self.modified_scene.append(flow_metadata)
        changes['added'].append(flow_metadata['object_id'])

        changes['material_flow'] = flow_metadata

        return changes

    def _find_gap_endpoints(self, removed_conveyors: List[str], flow_direction: str) -> Tuple[Dict, Dict]:
        """Find the two endpoints of the physical gap created by removing conveyors.

        Returns:
            (loading_pos, unloading_pos) - world positions at the two ends of the gap
            Loading is where material enters (after the gap in flow direction)
            Unloading is where material exits (before the gap in flow direction)
        """
        # Find which corner was removed to determine gap location
        removed_corner = None
        for conv_id in removed_conveyors:
            if 'Line_03' in conv_id:
                removed_corner = conv_id
                break

        if not removed_corner:
            # Fallback to center
            bounds = self.structure.bounds
            center = {
                'x': (bounds['min_x'] + bounds['max_x']) / 2,
                'y': (bounds['min_y'] + bounds['max_y']) / 2,
                'z': 0
            }
            return center, center

        # Determine which corner was removed based on original position
        corner_name = None
        for name, cid in self.structure.corner_positions.items():
            if cid == removed_corner:
                corner_name = name
                break

        bounds = self.structure.bounds

        # Gap endpoints depend on which corner was removed
        # For clockwise flow:
        # - SW corner removed: Loading at bottom of west edge, Unloading at left of south edge
        # - SE corner removed: Loading at right of south edge, Unloading at bottom of east edge
        # - NE corner removed: Loading at top of east edge, Unloading at right of north edge
        # - NW corner removed: Loading at left of north edge, Unloading at top of west edge

        if corner_name == 'SW':
            # Gap is at SW corner
            # Loading: bottom of west edge (where material enters the loop)
            # Unloading: left of south edge (where material exits before the gap)
            loading_pos = self._find_edge_endpoint('west', find_min=True)   # Bottom of west
            unloading_pos = self._find_edge_endpoint('south', find_min=True)  # Left of south
        elif corner_name == 'SE':
            loading_pos = self._find_edge_endpoint('south', find_min=False)   # Right of south
            unloading_pos = self._find_edge_endpoint('east', find_min=True)   # Bottom of east
        elif corner_name == 'NE':
            loading_pos = self._find_edge_endpoint('east', find_min=False)    # Top of east
            unloading_pos = self._find_edge_endpoint('north', find_min=False) # Right of north
        elif corner_name == 'NW':
            loading_pos = self._find_edge_endpoint('north', find_min=True)    # Left of north
            unloading_pos = self._find_edge_endpoint('west', find_min=False)  # Top of west
        else:
            center = {'x': (bounds['min_x'] + bounds['max_x']) / 2, 'y': bounds['min_y'], 'z': 0}
            return center, center

        return loading_pos, unloading_pos

    def _find_edge_endpoint(self, edge: str, find_min: bool) -> Dict:
        """Find the endpoint of remaining conveyors on an edge.

        Args:
            edge: Which edge to search
            find_min: If True, find the min coordinate end; if False, find max
        """
        edge_conveyors = self.structure.edges.get(edge, {}).get('conveyors', [])

        # Find remaining outer conveyors
        remaining = []
        for conv_id in edge_conveyors:
            conv = self._find_object(conv_id)
            if conv:
                pos = conv.get('position', {})
                size = conv.get('size_in_meters', {})
                x, y = pos.get('x', 0), pos.get('y', 0)
                length = size.get('length', 2.0)

                is_outer = False
                if edge == 'south':
                    is_outer = abs(y - self.structure.bounds['min_y']) < 1.0
                elif edge == 'north':
                    is_outer = abs(y - self.structure.bounds['max_y']) < 1.0
                elif edge == 'west':
                    is_outer = abs(x - self.structure.bounds['min_x']) < 1.0
                elif edge == 'east':
                    is_outer = abs(x - self.structure.bounds['max_x']) < 1.0

                if is_outer:
                    remaining.append((conv_id, x, y, length, conv))

        if not remaining:
            bounds = self.structure.bounds
            if edge == 'south':
                return {'x': bounds['min_x'], 'y': bounds['min_y'], 'z': 0}
            elif edge == 'north':
                return {'x': bounds['min_x'], 'y': bounds['max_y'], 'z': 0}
            elif edge == 'west':
                return {'x': bounds['min_x'], 'y': bounds['min_y'], 'z': 0}
            else:
                return {'x': bounds['max_x'], 'y': bounds['min_y'], 'z': 0}

        # Sort and find endpoint
        if edge in ['south', 'north']:
            # Horizontal edge - sort by x
            remaining.sort(key=lambda c: c[1])
            if find_min:
                conv_id, x, y, length, conv = remaining[0]
                return {'x': x - length / 2, 'y': y, 'z': 0}  # Left edge
            else:
                conv_id, x, y, length, conv = remaining[-1]
                return {'x': x + length / 2, 'y': y, 'z': 0}  # Right edge
        else:
            # Vertical edge - sort by y
            remaining.sort(key=lambda c: c[2])
            if find_min:
                conv_id, x, y, length, conv = remaining[0]
                return {'x': x, 'y': y - length / 2, 'z': 0}  # Bottom edge
            else:
                conv_id, x, y, length, conv = remaining[-1]
                return {'x': x, 'y': y + length / 2, 'z': 0}  # Top edge

    def _remove_passage_conveyors(self, loading_edge: str, loading_pos: float, loading_conv: Dict,
                                   unloading_edge: str, unloading_pos: float, unloading_conv: Dict,
                                   flow_direction: str) -> List[str]:
        """Remove conveyors in the passage segment to create physical gap."""
        removed = []

        # Determine which edges are in the passage (going against flow from unloading to loading)
        edges_cw = ['south', 'west', 'north', 'east']
        edges_ccw = ['south', 'east', 'north', 'west']
        edges = edges_cw if flow_direction == 'clockwise' else edges_ccw

        try:
            load_idx = edges.index(loading_edge)
            unload_idx = edges.index(unloading_edge)
        except ValueError:
            return removed

        # Find passage edges (going backwards from unloading to loading)
        passage_edges = []
        idx = unload_idx
        while True:
            idx = (idx - 1) % 4  # Go backwards (against flow)
            if idx == load_idx:
                break
            passage_edges.append(edges[idx])
            if len(passage_edges) > 4:
                break

        # For both same-edge and different-edge cases, we create a SINGLE passage
        # by removing ONE corner and its adjacent conveyor segment
        # The passage is always at the corner between loading and unloading edges
        # (or at one end of the edge for same-edge case)

        if loading_edge == unloading_edge:
            # Same edge: remove corner and conveyor at one end
            self._remove_conveyors_between_positions(
                loading_edge, loading_pos, unloading_pos, flow_direction, removed
            )
        else:
            # Different edges: find the corner that connects them and create passage there
            # For clockwise flow with loading on west and unloading on south,
            # the passage should be at SW corner (where west meets south)
            corner_to_remove = self._find_corner_between_edges(loading_edge, unloading_edge)
            if corner_to_remove:
                corner_id = self.structure.corner_positions.get(corner_to_remove)
                if corner_id and self._find_object(corner_id):
                    self._remove_object(corner_id)
                    removed.append(corner_id)

                # Remove the conveyor segment adjacent to this corner on the unloading edge
                self._remove_conveyor_near_corner(unloading_edge, corner_to_remove, removed)

        return removed

    def _find_corner_between_edges(self, edge1: str, edge2: str) -> Optional[str]:
        """Find the corner that connects two edges."""
        corner_map = {
            ('south', 'west'): 'SW',
            ('west', 'south'): 'SW',
            ('south', 'east'): 'SE',
            ('east', 'south'): 'SE',
            ('north', 'west'): 'NW',
            ('west', 'north'): 'NW',
            ('north', 'east'): 'NE',
            ('east', 'north'): 'NE',
        }
        return corner_map.get((edge1, edge2))

    def _remove_conveyor_near_corner(self, edge: str, corner: str, removed: List[str]):
        """Remove the conveyor segment nearest to a corner on the given edge."""
        edge_conveyors = self.structure.edges.get(edge, {}).get('conveyors', [])
        outer_conveyors = self._find_outer_conveyors(edge, edge_conveyors)

        if not outer_conveyors:
            return

        outer_conveyors.sort(key=lambda x: x[1])  # Sort by coordinate

        # Determine which end to remove based on corner
        if corner in ['SW', 'NW']:
            # Remove from start (lower coordinate)
            conv_id, _, conv = outer_conveyors[0]
        else:
            # Remove from end (higher coordinate)
            conv_id, _, conv = outer_conveyors[-1]

        if self._find_object(conv_id):
            # Remove any inline stations on this conveyor first
            conv_pos = conv.get('position', {})
            self._remove_inline_stations_near(conv_pos.get('x', 0), conv_pos.get('y', 0), removed)

            self._remove_object(conv_id)
            removed.append(conv_id)

    def _remove_conveyors_between_positions(self, edge: str, loading_pos: float, unloading_pos: float,
                                             flow_direction: str, removed: List[str]):
        """Remove conveyors/corners to create passage between loading and unloading on same edge.

        For same-edge loading/unloading, the passage is created by removing the corner
        and conveyor segment on the side where the passage should be (between unloading
        and loading going against the flow direction).

        Also removes any inline stations (Line_06, Line_07) that were on the removed conveyors.
        """
        # For clockwise flow on south edge with loading at 0.3 and unloading at 0.7:
        # - Active flow goes left to right (west to east): 0.3 -> 0.7
        # - Passage is on the LEFT (west) side: remove SW corner and leftmost conveyor

        # For counterclockwise flow, it would be the opposite

        # Determine which corner to remove based on edge and flow direction
        corner_to_remove = None

        if edge == 'south':
            if flow_direction == 'clockwise':
                # Passage on west side (left)
                if loading_pos < unloading_pos:
                    corner_to_remove = 'SW'
                else:
                    corner_to_remove = 'SE'
            else:
                if loading_pos < unloading_pos:
                    corner_to_remove = 'SE'
                else:
                    corner_to_remove = 'SW'
        elif edge == 'north':
            if flow_direction == 'clockwise':
                if loading_pos < unloading_pos:
                    corner_to_remove = 'NE'
                else:
                    corner_to_remove = 'NW'
            else:
                if loading_pos < unloading_pos:
                    corner_to_remove = 'NW'
                else:
                    corner_to_remove = 'NE'
        elif edge == 'west':
            if flow_direction == 'clockwise':
                if loading_pos < unloading_pos:
                    corner_to_remove = 'NW'
                else:
                    corner_to_remove = 'SW'
            else:
                if loading_pos < unloading_pos:
                    corner_to_remove = 'SW'
                else:
                    corner_to_remove = 'NW'
        elif edge == 'east':
            if flow_direction == 'clockwise':
                if loading_pos < unloading_pos:
                    corner_to_remove = 'SE'
                else:
                    corner_to_remove = 'NE'
            else:
                if loading_pos < unloading_pos:
                    corner_to_remove = 'NE'
                else:
                    corner_to_remove = 'SE'

        # Remove the corner
        if corner_to_remove:
            corner_id = self.structure.corner_positions.get(corner_to_remove)
            if corner_id and self._find_object(corner_id):
                self._remove_object(corner_id)
                removed.append(corner_id)

        # Remove the conveyor segment closest to the removed corner
        edge_conveyors = self.structure.edges.get(edge, {}).get('conveyors', [])
        outer_conveyors = self._find_outer_conveyors(edge, edge_conveyors)

        if outer_conveyors:
            outer_conveyors.sort(key=lambda x: x[1])  # Sort by coordinate

            # Determine which end of the edge to remove from
            if corner_to_remove in ['SW', 'NW']:
                # Remove from the start (lower coordinate)
                conv_id, _, conv = outer_conveyors[0]
            else:
                # Remove from the end (higher coordinate)
                conv_id, _, conv = outer_conveyors[-1]

            if self._find_object(conv_id):
                # Before removing conveyor, find and remove any inline stations on it
                conv_pos = conv.get('position', {})
                conv_x, conv_y = conv_pos.get('x', 0), conv_pos.get('y', 0)
                self._remove_inline_stations_near(conv_x, conv_y, removed)

                self._remove_object(conv_id)
                removed.append(conv_id)

    def _remove_inline_stations_near(self, x: float, y: float, removed: List[str]):
        """Remove inline stations (Line_06, Line_07) near a given position."""
        # Find and remove any Line_06 or Line_07 stations close to this position
        stations_to_remove = []
        for obj in self.modified_scene:
            obj_id = obj.get('object_id', '')
            if 'Line_06' in obj_id or 'Line_07' in obj_id:
                pos = obj.get('position', {})
                sx, sy = pos.get('x', 0), pos.get('y', 0)
                # Check if station is close to the conveyor position (within 1.5m)
                if abs(sx - x) < 1.5 and abs(sy - y) < 1.5:
                    stations_to_remove.append(obj_id)

        for station_id in stations_to_remove:
            self._remove_object(station_id)
            removed.append(station_id)

    def _remove_edge_conveyors(self, edge: str, removed: List[str]):
        """Remove all outer conveyors on an edge."""
        edge_conveyors = self.structure.edges.get(edge, {}).get('conveyors', [])
        outer_conveyors = self._find_outer_conveyors(edge, edge_conveyors)

        for conv_id, _, _ in outer_conveyors:
            self._remove_object(conv_id)
            removed.append(conv_id)

        # Also remove corners on this edge
        corner_map = {
            'south': ['SW', 'SE'],
            'north': ['NW', 'NE'],
            'west': ['SW', 'NW'],
            'east': ['SE', 'NE']
        }
        for corner_pos in corner_map.get(edge, []):
            corner_id = self.structure.corner_positions.get(corner_pos)
            if corner_id and self._find_object(corner_id):
                self._remove_object(corner_id)
                if corner_id not in removed:
                    removed.append(corner_id)

    def _remove_passage_corners(self, unloading_edge: str, loading_edge: str,
                                 passage_edges: List[str], removed: List[str]):
        """Remove corners that connect passage edges."""
        # Corner naming: SW, SE, NW, NE
        # Each corner connects two edges
        corner_edges = {
            'SW': ('south', 'west'),
            'SE': ('south', 'east'),
            'NW': ('north', 'west'),
            'NE': ('north', 'east')
        }

        all_passage_edges = [unloading_edge] + passage_edges + [loading_edge]

        for corner_name, (edge1, edge2) in corner_edges.items():
            # Check if this corner connects two consecutive passage edges
            for i in range(len(all_passage_edges) - 1):
                e1, e2 = all_passage_edges[i], all_passage_edges[i + 1]
                if (edge1 == e1 and edge2 == e2) or (edge1 == e2 and edge2 == e1):
                    corner_id = self.structure.corner_positions.get(corner_name)
                    if corner_id and self._find_object(corner_id) and corner_id not in removed:
                        self._remove_object(corner_id)
                        removed.append(corner_id)

    def _find_position_on_edge(self, edge: str, position: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find the conveyor and world position for a point on an edge."""
        edge_conveyors = self.structure.edges.get(edge, {}).get('conveyors', [])
        if not edge_conveyors:
            return None, None

        suitable = self._find_outer_conveyors(edge, edge_conveyors)
        if not suitable:
            return None, None

        suitable.sort(key=lambda x: x[1])

        if len(suitable) == 1:
            conv_id, _, conv = suitable[0]
        else:
            idx = int(position * (len(suitable) - 1))
            idx = max(0, min(idx, len(suitable) - 1))
            conv_id, _, conv = suitable[idx]

        # Calculate world position on this conveyor
        conv_pos = conv.get('position', {})
        conv_size = conv.get('size_in_meters', {})
        conv_length = conv_size.get('length', 2.0)

        # Offset along conveyor based on position fraction
        along_offset = (position - 0.5) * conv_length * 0.8

        if edge in ['south', 'north']:
            world_pos = {
                'x': conv_pos.get('x', 0) + along_offset,
                'y': conv_pos.get('y', 0),
                'z': 0
            }
        else:
            world_pos = {
                'x': conv_pos.get('x', 0),
                'y': conv_pos.get('y', 0) + along_offset,
                'z': 0
            }

        return conv, world_pos

    def _calculate_passage_segment(self, loading_edge: str, loading_pos: float,
                                    unloading_edge: str, unloading_pos: float,
                                    flow_direction: str) -> Dict:
        """Calculate the passage segment (gap between unloading and loading)."""
        # The passage is the segment going AGAINST the flow direction
        # from unloading back to loading
        edges_cw = ['south', 'west', 'north', 'east']  # clockwise order
        edges_ccw = ['south', 'east', 'north', 'west']  # counterclockwise order

        edges = edges_cw if flow_direction == 'clockwise' else edges_ccw

        # Find indices
        try:
            load_idx = edges.index(loading_edge)
            unload_idx = edges.index(unloading_edge)
        except ValueError:
            return {'description': 'Unable to calculate passage'}

        # Passage goes from unloading back to loading (against flow)
        # This is the segment NOT used by production flow
        if loading_edge == unloading_edge:
            # Same edge - passage is the portion between unloading and loading
            if loading_pos < unloading_pos:
                passage_desc = f"Segment on {loading_edge} edge from {int(unloading_pos*100)}% to {int(loading_pos*100)}% (wrapping around)"
            else:
                passage_desc = f"Segment on {loading_edge} edge between unloading ({int(unloading_pos*100)}%) and loading ({int(loading_pos*100)}%)"
        else:
            # Different edges - passage spans edges between them
            passage_edges = []
            idx = unload_idx
            while True:
                idx = (idx - 1) % 4  # Go backwards (against flow)
                if edges[idx] == loading_edge:
                    break
                passage_edges.append(edges[idx])
                if len(passage_edges) > 4:
                    break

            if passage_edges:
                passage_desc = f"Passage spans: {unloading_edge} -> {' -> '.join(passage_edges)} -> {loading_edge}"
            else:
                passage_desc = f"Direct passage from {unloading_edge} to {loading_edge}"

        return {
            'description': passage_desc,
            'from_edge': unloading_edge,
            'from_position': unloading_pos,
            'to_edge': loading_edge,
            'to_position': loading_pos
        }

    def _find_outer_conveyors(self, edge: str, edge_conveyors: List[str]) -> List[Tuple]:
        """Find conveyors on the outer perimeter of the specified edge."""
        suitable = []

        for conv_id in edge_conveyors:
            conv = self._find_object(conv_id)
            if not conv:
                continue

            pos = conv.get('position', {})
            x, y = pos.get('x', 0), pos.get('y', 0)

            # Check if this is an outer perimeter conveyor
            is_outer = False
            if edge == 'south':
                is_outer = abs(y - self.structure.bounds['min_y']) < 1.0
                coord = x
            elif edge == 'north':
                is_outer = abs(y - self.structure.bounds['max_y']) < 1.0
                coord = x
            elif edge == 'west':
                is_outer = abs(x - self.structure.bounds['min_x']) < 1.0
                coord = y
            else:  # east
                is_outer = abs(x - self.structure.bounds['max_x']) < 1.0
                coord = y

            if is_outer:
                suitable.append((conv_id, coord, conv))

        return suitable

    def _calculate_side_position(self, conveyor: Dict, edge: str, outer: bool = True) -> Dict:
        """Calculate position for a side station relative to a conveyor."""
        conv_pos = conveyor.get('position', {})
        x, y = conv_pos.get('x', 0), conv_pos.get('y', 0)

        # Offset for side station (1.5m from conveyor center)
        offset = 1.5 if outer else -1.5

        if edge == 'south':
            return {'x': x, 'y': y - offset, 'z': 0}
        elif edge == 'north':
            return {'x': x, 'y': y + offset, 'z': 0}
        elif edge == 'west':
            return {'x': x - offset, 'y': y, 'z': 0}
        else:  # east
            return {'x': x + offset, 'y': y, 'z': 0}

    def _get_adjacent_position(self, position: Dict, edge: str, outer: bool = True) -> Dict:
        """Calculate an adjacent position based on edge and offset direction."""
        x, y = position.get('x', 0), position.get('y', 0)

        # Offset for adjacent position (1.5m from center)
        offset = 1.5 if outer else -1.5

        if edge == 'south':
            return {'x': x, 'y': y - offset, 'z': 0}
        elif edge == 'north':
            return {'x': x, 'y': y + offset, 'z': 0}
        elif edge == 'west':
            return {'x': x - offset, 'y': y, 'z': 0}
        else:  # east
            return {'x': x + offset, 'y': y, 'z': 0}

    def _connect_side_station(self, station: Dict, conveyor: Dict):
        """Connect a side-loading station to its adjacent conveyor."""
        station['connections'].append({
            'object_id': conveyor.get('object_id', ''),
            'connection_type': 'side_load',
            'source_endpoint': 'transfer',
            'target_endpoint': 'side_input'
        })

    # ==================== Extended Equipment Operations ====================
    # These operations support equipment from the legacy catalog for more
    # comprehensive manufacturing cell configurations.
    #
    # Layout Strategy:
    # - Equipment is organized into dedicated ZONES outside the conveyor loop
    # - Each zone has a specific purpose and systematic internal arrangement
    # - Zones are placed on specific edges with proper clearances

    def _get_zone_base_position(self, edge: str, position: float, offset: float = 4.0) -> Dict:
        """
        Calculate base position for a zone on a specific edge.

        Args:
            edge: Edge to place zone on ("north", "south", "east", "west")
            position: Position along edge (0.0 to 1.0)
            offset: Distance from the conveyor loop bounds

        Returns:
            Base position dict with x, y, z coordinates
        """
        bounds = self.structure.bounds

        if edge == 'north':
            return {
                'x': bounds['min_x'] + position * (bounds['max_x'] - bounds['min_x']),
                'y': bounds['max_y'] + offset,
                'z': 0
            }
        elif edge == 'south':
            return {
                'x': bounds['min_x'] + position * (bounds['max_x'] - bounds['min_x']),
                'y': bounds['min_y'] - offset,
                'z': 0
            }
        elif edge == 'east':
            return {
                'x': bounds['max_x'] + offset,
                'y': bounds['min_y'] + position * (bounds['max_y'] - bounds['min_y']),
                'z': 0
            }
        else:  # west
            return {
                'x': bounds['min_x'] - offset,
                'y': bounds['min_y'] + position * (bounds['max_y'] - bounds['min_y']),
                'z': 0
            }

    def _get_zone_layout_vectors(self, edge: str) -> tuple:
        """
        Get layout vectors for arranging equipment in a zone.

        Returns:
            (primary_dir, secondary_dir) - unit vectors for row/column arrangement
        """
        if edge in ['north', 'south']:
            # Horizontal zone: primary along X, secondary along Y (away from loop)
            primary = (1, 0)
            secondary = (0, 1) if edge == 'north' else (0, -1)
        else:
            # Vertical zone: primary along Y, secondary along X (away from loop)
            primary = (0, 1)
            secondary = (1, 0) if edge == 'east' else (-1, 0)
        return primary, secondary

    def add_pallet_staging(self, near_equipment_id: str, num_pallets: int = 4,
                           add_pallet_jack: bool = True,
                           layout: str = "grid") -> Dict:
        """
        Add a pallet staging area adjacent to specified equipment.
        """
        changes = {'added': [], 'modified': []}

        ref_equipment = self._find_object(near_equipment_id)
        if not ref_equipment:
            return {'error': f'Reference equipment {near_equipment_id} not found'}

        ref_pos = ref_equipment['position']
        edge = self.structure._get_edge(ref_pos['x'], ref_pos['y'])

        # Position pallet area adjacent to reference equipment (outer side)
        offset = 3.0
        if edge == 'north':
            base_pos = {'x': ref_pos['x'], 'y': ref_pos['y'] + offset, 'z': 0.005}
            primary, secondary = (1, 0), (0, 1)
        elif edge == 'south':
            base_pos = {'x': ref_pos['x'], 'y': ref_pos['y'] - offset, 'z': 0.005}
            primary, secondary = (1, 0), (0, -1)
        elif edge == 'east':
            base_pos = {'x': ref_pos['x'] + offset, 'y': ref_pos['y'], 'z': 0.005}
            primary, secondary = (0, 1), (1, 0)
        else:  # west
            base_pos = {'x': ref_pos['x'] - offset, 'y': ref_pos['y'], 'z': 0.005}
            primary, secondary = (0, 1), (-1, 0)

        pallet_spacing = 1.4  # Pallet width + access gap

        # Create pallets in organized layout
        if layout == "grid":
            cols = 2
            for i in range(num_pallets):
                row = i // cols
                col = i % cols
                pallet_pos = {
                    'x': base_pos['x'] + col * pallet_spacing * primary[0] + row * pallet_spacing * secondary[0],
                    'y': base_pos['y'] + col * pallet_spacing * primary[1] + row * pallet_spacing * secondary[1],
                    'z': 0.005
                }
                pallet = self._create_support_equipment('Pallet', pallet_pos, 0)
                changes['added'].append(pallet['object_id'])

        elif layout == "linear":
            for i in range(num_pallets):
                pallet_pos = {
                    'x': base_pos['x'] + i * pallet_spacing * primary[0],
                    'y': base_pos['y'] + i * pallet_spacing * primary[1],
                    'z': 0.005
                }
                pallet = self._create_support_equipment('Pallet', pallet_pos, 0)
                changes['added'].append(pallet['object_id'])

        else:  # L-shaped
            half = num_pallets // 2
            for i in range(half):
                pallet_pos = {
                    'x': base_pos['x'] + i * pallet_spacing * primary[0],
                    'y': base_pos['y'] + i * pallet_spacing * primary[1],
                    'z': 0.005
                }
                pallet = self._create_support_equipment('Pallet', pallet_pos, 0)
                changes['added'].append(pallet['object_id'])
            for i in range(num_pallets - half):
                pallet_pos = {
                    'x': base_pos['x'] + (i + 1) * pallet_spacing * secondary[0],
                    'y': base_pos['y'] + (i + 1) * pallet_spacing * secondary[1],
                    'z': 0.005
                }
                pallet = self._create_support_equipment('Pallet', pallet_pos, 0)
                changes['added'].append(pallet['object_id'])

        # Add pallet mover at the entry of the pallet zone
        if add_pallet_jack:
            mover_pos = {
                'x': base_pos['x'] - 2.0 * secondary[0],
                'y': base_pos['y'] - 2.0 * secondary[1],
                'z': 0.005
            }
            pallet_mover = self._create_support_equipment('PalletMover', mover_pos, 0)
            changes['added'].append(pallet_mover['object_id'])

        return changes

    def add_machining_cell(self, edge: str = "north", position: float = 0.5,
                            add_scissor_lift: bool = True,
                            add_power_cutter: bool = False) -> Dict:
        """
        Add a large machining cell in a dedicated zone on the specified edge.
        """
        changes = {'added': [], 'modified': []}

        # Get zone position with large offset for machining equipment
        zone_base = self._get_zone_base_position(edge, position, offset=8.0)
        primary, secondary = self._get_zone_layout_vectors(edge)

        # Determine rotation based on edge (face the conveyor loop)
        rotations = {'north': 180, 'south': 0, 'east': -90, 'west': 90}
        cell_rotation = rotations.get(edge, 0)

        # Create machining cell
        cell = self._create_large_equipment('MFG_Equip_30ftx7ft_w_Exhaust', zone_base, cell_rotation)
        changes['added'].append(cell['object_id'])

        # Add scissor lift beside the cell (along primary direction)
        if add_scissor_lift:
            lift_pos = {
                'x': zone_base['x'] + 6.0 * primary[0],
                'y': zone_base['y'] + 6.0 * primary[1],
                'z': 0
            }
            scissor_lift = self._create_support_equipment('ScissorLift', lift_pos)
            changes['added'].append(scissor_lift['object_id'])

        # Add power cutter on the other side
        if add_power_cutter:
            cutter_pos = {
                'x': zone_base['x'] - 4.0 * primary[0],
                'y': zone_base['y'] - 4.0 * primary[1],
                'z': 0
            }
            power_cutter = self._create_support_equipment('Power_Cutter', cutter_pos)
            changes['added'].append(power_cutter['object_id'])

        return changes

    def add_robot_workstation(self, near_conveyor_id: str = None, robot_type: str = "ur10e",
                               add_work_table: bool = True, add_shelving: bool = True,
                               add_camera_stand: bool = False, edge: str = None,
                               position: float = 0.5) -> Dict:
        """
        Add a robot workstation with robot mounted on table (reference style).

        Layout: Table with robot on top, shelving behind, camera beside.
        Positioned relative to the actual conveyor location, or by edge/position in template mode.

        Args:
            near_conveyor_id: Conveyor to position near (if available)
            robot_type: "ur10e" (collaborative) or "abb_irb2600" (industrial)
            add_work_table: Whether to add Table_6ft under the robot
            add_shelving: Whether to add ShelvingRack behind the station
            add_camera_stand: Whether to add Camera_Stand_7ft
            edge: Edge to place on ("north", "south", "east", "west") - used in template mode
            position: Position along edge 0.0-1.0 - used in template mode
        """
        changes = {'added': [], 'modified': []}

        # Try to find conveyor, or use edge-based positioning
        conveyor = self._find_object(near_conveyor_id) if near_conveyor_id else None

        if conveyor:
            conv_pos = conveyor['position']
            edge = self.structure._get_edge(conv_pos['x'], conv_pos['y'])
        elif edge and hasattr(self.structure, 'bounds'):
            # Template mode: use structure bounds to calculate position
            bounds = self.structure.bounds
            if edge == 'north':
                conv_pos = {'x': bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * position,
                           'y': bounds['max_y'], 'z': 0}
            elif edge == 'south':
                conv_pos = {'x': bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * position,
                           'y': bounds['min_y'], 'z': 0}
            elif edge == 'east':
                conv_pos = {'x': bounds['max_x'],
                           'y': bounds['min_y'] + (bounds['max_y'] - bounds['min_y']) * position, 'z': 0}
            else:  # west
                conv_pos = {'x': bounds['min_x'],
                           'y': bounds['min_y'] + (bounds['max_y'] - bounds['min_y']) * position, 'z': 0}
        else:
            return {'error': f'Conveyor {near_conveyor_id} not found and no edge specified'}

        # Position workstation adjacent to the conveyor (outer side of loop)
        # Offset perpendicular to conveyor direction
        offset = 2.5  # Distance from conveyor center
        if edge == 'north':
            base_pos = {'x': conv_pos['x'], 'y': conv_pos['y'] + offset, 'z': 0.005}
            facing_rotation = 180  # Face south toward conveyor
            primary, secondary = (1, 0), (0, 1)
        elif edge == 'south':
            base_pos = {'x': conv_pos['x'], 'y': conv_pos['y'] - offset, 'z': 0.005}
            facing_rotation = 0  # Face north toward conveyor
            primary, secondary = (1, 0), (0, -1)
        elif edge == 'east':
            base_pos = {'x': conv_pos['x'] + offset, 'y': conv_pos['y'], 'z': 0.005}
            facing_rotation = 270  # Face west toward conveyor
            primary, secondary = (0, 1), (1, 0)
        else:  # west
            base_pos = {'x': conv_pos['x'] - offset, 'y': conv_pos['y'], 'z': 0.005}
            facing_rotation = 90  # Face east toward conveyor
            primary, secondary = (0, 1), (-1, 0)

        robot_name = "ur10e_robot" if robot_type == "ur10e" else "abb_irb2600_12_165"
        table_id = None

        # Create work table first (robot will be placed on it)
        if add_work_table:
            table = self._create_support_equipment('Table_6ft', base_pos, facing_rotation)
            table_id = table['object_id']
            changes['added'].append(table_id)

            # Robot on top of table
            robot = self._create_support_equipment(
                robot_name, base_pos, facing_rotation, on_equipment=table_id
            )
            changes['added'].append(robot['object_id'])
        else:
            # Robot directly on floor
            robot = self._create_support_equipment(robot_name, base_pos, facing_rotation)
            changes['added'].append(robot['object_id'])

        # Shelving behind the workstation (away from conveyor)
        if add_shelving:
            shelf_pos = {
                'x': base_pos['x'] + 2.5 * secondary[0],
                'y': base_pos['y'] + 2.5 * secondary[1],
                'z': 0.005
            }
            shelf_rotation = facing_rotation + 90
            shelving = self._create_support_equipment('ShelvingRack', shelf_pos, shelf_rotation)
            changes['added'].append(shelving['object_id'])

        # Camera stand beside the workstation
        if add_camera_stand:
            camera_pos = {
                'x': base_pos['x'] + 2.0 * primary[0],
                'y': base_pos['y'] + 2.0 * primary[1],
                'z': 0
            }
            camera = self._create_support_equipment('Camera_Stand_7ft', camera_pos, facing_rotation)
            changes['added'].append(camera['object_id'])

        return changes

    def add_quality_station(self, near_conveyor_id: str = None, add_camera: bool = True,
                             add_table: bool = True, add_ventilation: bool = False,
                             edge: str = None, position: float = 0.5) -> Dict:
        """
        Add a quality inspection station adjacent to a conveyor or on specified edge.

        Args:
            near_conveyor_id: Conveyor to position near (if available)
            add_camera: Whether to add Camera_Stand_7ft
            add_table: Whether to add Table_6ft
            add_ventilation: Whether to add VentilatorFan_Straight
            edge: Edge to place on - used in template mode
            position: Position along edge 0.0-1.0 - used in template mode
        """
        changes = {'added': [], 'modified': []}

        # Try to find conveyor, or use edge-based positioning
        conveyor = self._find_object(near_conveyor_id) if near_conveyor_id else None

        if conveyor:
            conv_pos = conveyor['position']
            edge = self.structure._get_edge(conv_pos['x'], conv_pos['y'])
        elif edge and hasattr(self.structure, 'bounds'):
            # Template mode: use structure bounds
            bounds = self.structure.bounds
            if edge == 'north':
                conv_pos = {'x': bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * position,
                           'y': bounds['max_y'], 'z': 0}
            elif edge == 'south':
                conv_pos = {'x': bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * position,
                           'y': bounds['min_y'], 'z': 0}
            elif edge == 'east':
                conv_pos = {'x': bounds['max_x'],
                           'y': bounds['min_y'] + (bounds['max_y'] - bounds['min_y']) * position, 'z': 0}
            else:  # west
                conv_pos = {'x': bounds['min_x'],
                           'y': bounds['min_y'] + (bounds['max_y'] - bounds['min_y']) * position, 'z': 0}
        else:
            return {'error': f'Conveyor {near_conveyor_id} not found and no edge specified'}

        # Position adjacent to conveyor (outer side of loop)
        offset = 2.0
        if edge == 'north':
            base_pos = {'x': conv_pos['x'], 'y': conv_pos['y'] + offset, 'z': 0.005}
            facing_rotation = 180
            primary, secondary = (1, 0), (0, 1)
        elif edge == 'south':
            base_pos = {'x': conv_pos['x'], 'y': conv_pos['y'] - offset, 'z': 0.005}
            facing_rotation = 0
            primary, secondary = (1, 0), (0, -1)
        elif edge == 'east':
            base_pos = {'x': conv_pos['x'] + offset, 'y': conv_pos['y'], 'z': 0.005}
            facing_rotation = 270
            primary, secondary = (0, 1), (1, 0)
        else:  # west
            base_pos = {'x': conv_pos['x'] - offset, 'y': conv_pos['y'], 'z': 0.005}
            facing_rotation = 90
            primary, secondary = (0, 1), (-1, 0)

        # Table first (main inspection surface)
        if add_table:
            table = self._create_support_equipment('Table_6ft', base_pos, facing_rotation)
            changes['added'].append(table['object_id'])

        # Camera stand beside the table
        if add_camera:
            camera_pos = {
                'x': base_pos['x'] + 1.5 * primary[0],
                'y': base_pos['y'] + 1.5 * primary[1],
                'z': 0
            }
            camera = self._create_support_equipment('Camera_Stand_7ft', camera_pos, facing_rotation)
            changes['added'].append(camera['object_id'])

        # Ventilation behind the station
        if add_ventilation:
            vent_pos = {
                'x': base_pos['x'] + 1.5 * secondary[0],
                'y': base_pos['y'] + 1.5 * secondary[1],
                'z': 2.5
            }
            vent = self._create_support_equipment('VentilatorFan_Straight', vent_pos, facing_rotation)
            changes['added'].append(vent['object_id'])

        return changes

    def add_safety_perimeter(self, around_equipment_id: str,
                              num_sections: int = 4,
                              include_gate: bool = True) -> Dict:
        """
        Add safety railing perimeter around equipment (reference style).

        Uses longer 16ft railings for sides, 8ft for ends. Arranged in a
        rectangular pattern similar to the reference scene.
        """
        changes = {'added': [], 'modified': []}

        equipment = self._find_object(around_equipment_id)
        if not equipment:
            return {'error': f'Equipment {around_equipment_id} not found'}

        eq_pos = equipment['position']
        eq_size = equipment.get('size_in_meters', {'length': 2.0, 'width': 2.0})

        # Calculate enclosure dimensions with proper clearance
        clearance = 1.5
        half_l = eq_size.get('length', 2.0) / 2 + clearance
        half_w = eq_size.get('width', 2.0) / 2 + clearance

        # Use 16ft railings for longer sides, 8ft for shorter sides
        # Following reference pattern: railings along Y axis (north-south oriented)
        rail_sections = []

        # East side (16ft railing, facing north)
        rail_sections.append({
            'type': 'SafetyRailing_16ft' if num_sections >= 4 else 'SafetyRailing_8ft',
            'pos': {'x': eq_pos['x'] + half_l, 'y': eq_pos['y'], 'z': 0.01},
            'rotation': 0  # Aligned with Y axis
        })

        # West side (16ft railing, facing north)
        rail_sections.append({
            'type': 'SafetyRailing_16ft' if num_sections >= 4 else 'SafetyRailing_8ft',
            'pos': {'x': eq_pos['x'] - half_l, 'y': eq_pos['y'], 'z': 0.01},
            'rotation': 0
        })

        # North end (8ft railing, facing east)
        if num_sections >= 3:
            rail_sections.append({
                'type': 'SafetyRailing_8ft',
                'pos': {'x': eq_pos['x'], 'y': eq_pos['y'] + half_w, 'z': 0.01},
                'rotation': 90
            })

        # South end (8ft railing, facing east) - gate location
        if num_sections >= 4 and not include_gate:
            rail_sections.append({
                'type': 'SafetyRailing_8ft',
                'pos': {'x': eq_pos['x'], 'y': eq_pos['y'] - half_w, 'z': 0.01},
                'rotation': 90
            })

        # Add additional sections for larger perimeters
        if num_sections > 4:
            # Extended east side
            rail_sections.append({
                'type': 'SafetyRailing_16ft',
                'pos': {'x': eq_pos['x'] + half_l, 'y': eq_pos['y'] + half_w + 2.5, 'z': 0.01},
                'rotation': 0
            })
            # Extended west side
            rail_sections.append({
                'type': 'SafetyRailing_16ft',
                'pos': {'x': eq_pos['x'] - half_l, 'y': eq_pos['y'] + half_w + 2.5, 'z': 0.01},
                'rotation': 0
            })

        # Create railing sections
        for section in rail_sections[:num_sections]:
            rail = self._create_support_equipment(
                section['type'], section['pos'], section['rotation']
            )
            changes['added'].append(rail['object_id'])

        return changes

    def add_workbench_area(self, edge: str = "south", position: float = 0.5,
                           num_tables: int = 2, add_shelving: bool = True) -> Dict:
        """
        Add a manual workbench area along the specified edge of the loop.
        """
        changes = {'added': [], 'modified': []}

        bounds = self.structure.bounds
        offset = 3.0  # Distance from loop edge

        # Calculate base position on specified edge
        if edge == 'north':
            base_x = bounds['min_x'] + position * (bounds['max_x'] - bounds['min_x'])
            base_y = bounds['max_y'] + offset
            facing_rotation = 180
            primary, secondary = (1, 0), (0, 1)
        elif edge == 'south':
            base_x = bounds['min_x'] + position * (bounds['max_x'] - bounds['min_x'])
            base_y = bounds['min_y'] - offset
            facing_rotation = 0
            primary, secondary = (1, 0), (0, -1)
        elif edge == 'east':
            base_x = bounds['max_x'] + offset
            base_y = bounds['min_y'] + position * (bounds['max_y'] - bounds['min_y'])
            facing_rotation = 270
            primary, secondary = (0, 1), (1, 0)
        else:  # west
            base_x = bounds['min_x'] - offset
            base_y = bounds['min_y'] + position * (bounds['max_y'] - bounds['min_y'])
            facing_rotation = 90
            primary, secondary = (0, 1), (-1, 0)

        table_spacing = 2.2  # Table width + working space

        # Add tables in a row along the edge
        for i in range(num_tables):
            offset_i = (i - (num_tables - 1) / 2)  # Center the row
            table_pos = {
                'x': base_x + offset_i * table_spacing * primary[0],
                'y': base_y + offset_i * table_spacing * primary[1],
                'z': 0.005
            }
            table = self._create_support_equipment('Table_6ft', table_pos, facing_rotation)
            changes['added'].append(table['object_id'])

        # Shelving behind the workbenches (away from loop)
        if add_shelving:
            shelf_pos = {
                'x': base_x + 2.5 * secondary[0],
                'y': base_y + 2.5 * secondary[1],
                'z': 0.005
            }
            shelf_rotation = facing_rotation + 90
            shelving = self._create_support_equipment('ShelvingRack', shelf_pos, shelf_rotation)
            changes['added'].append(shelving['object_id'])

        return changes

    def add_ventilation(self, near_equipment_id: str,
                        exhaust_direction: str = "up") -> Dict:
        """
        Add ventilation system positioned appropriately relative to equipment.
        """
        changes = {'added': [], 'modified': []}

        equipment = self._find_object(near_equipment_id)
        if not equipment:
            return {'error': f'Equipment {near_equipment_id} not found'}

        eq_pos = equipment['position']
        eq_size = equipment.get('size_in_meters', {'length': 2.0, 'width': 2.0})

        # Position based on exhaust direction
        if exhaust_direction == "up":
            # Directly above the equipment
            vent_pos = {
                'x': eq_pos['x'],
                'y': eq_pos['y'],
                'z': eq_size.get('height', 2.0) + 1.0
            }
            rotation = 0
        elif exhaust_direction == "out":
            # To the side, pointing outward
            vent_pos = {
                'x': eq_pos['x'] + eq_size.get('length', 2.0) / 2 + 1.0,
                'y': eq_pos['y'],
                'z': 2.0
            }
            rotation = 90
        else:  # filtered
            # Behind the equipment
            vent_pos = {
                'x': eq_pos['x'],
                'y': eq_pos['y'] + eq_size.get('width', 2.0) / 2 + 1.0,
                'z': 1.5
            }
            rotation = 0

        vent = self._create_support_equipment('VentilatorFan_Straight', vent_pos, rotation)
        changes['added'].append(vent['object_id'])

        return changes

    def _get_facing_from_rotation(self, rotation: float) -> str:
        """Convert rotation angle to facing direction."""
        rotation = rotation % 360
        if rotation < 45 or rotation >= 315:
            return "north_wall"
        elif rotation < 135:
            return "east_wall"
        elif rotation < 225:
            return "south_wall"
        else:
            return "west_wall"

    def _create_support_equipment(self, equipment_type: str, position: Dict,
                                   rotation: float = 0, on_equipment: str = None) -> Dict:
        """
        Create support equipment with full schema matching reference style.

        Args:
            equipment_type: Type of equipment
            position: Position dict with x, y, z
            rotation: Z rotation angle
            on_equipment: If set, this equipment is placed on top of another
        """
        new_id = self._get_next_id(equipment_type)
        template = self._get_equipment_template(equipment_type)

        equip = deepcopy(template) if template else {}
        equip['object_id'] = new_id
        equip['style'] = equip.get('style', 'Industrial')
        equip['material'] = equip.get('material', 'Metal')

        # Handle placement on other equipment (e.g., robot on table)
        if on_equipment:
            base_obj = self._find_object(on_equipment)
            if base_obj:
                base_height = base_obj.get('size_in_meters', {}).get('height', 0.9)
                position = deepcopy(position)
                position['z'] = base_height + 0.007  # Small offset above base
                equip['is_on_the_floor'] = False
                equip['placement'] = {
                    'objects_in_room': [{
                        'object_id': on_equipment,
                        'preposition': 'on top of',
                        'is_adjacent': True
                    }]
                }
            else:
                equip['is_on_the_floor'] = True
                equip['placement'] = {'objects_in_room': []}
        else:
            equip['is_on_the_floor'] = True
            equip['placement'] = {'objects_in_room': []}

        equip['facing'] = self._get_facing_from_rotation(rotation)
        equip['position'] = deepcopy(position)
        equip['rotation'] = {'z_angle': float(rotation)}
        equip['cluster'] = {
            'constraint_area': {'x_neg': 0.0, 'x_pos': 0.0, 'y_neg': 0.0, 'y_pos': 0.0}
        }
        equip['connections'] = []

        self.modified_scene.append(equip)
        return equip

    def _create_large_equipment(self, equipment_type: str, position: Dict,
                                 rotation: float = 0) -> Dict:
        """Create large equipment with full schema."""
        new_id = self._get_next_id(equipment_type)
        template = self._get_equipment_template(equipment_type)

        if template:
            equip = deepcopy(template)
        else:
            equip = {
                'size_in_meters': {'length': 3.064, 'width': 9.027, 'height': 7.727},
            }

        equip['object_id'] = new_id
        equip['style'] = 'Industrial'
        equip['material'] = 'Metal'
        equip['is_on_the_floor'] = True
        equip['facing'] = self._get_facing_from_rotation(rotation)
        equip['position'] = deepcopy(position)
        equip['rotation'] = {'z_angle': float(rotation)}
        equip['cluster'] = {
            'constraint_area': {'x_neg': 0.0, 'x_pos': 0.0, 'y_neg': 0.0, 'y_pos': 0.0}
        }
        equip['connections'] = []
        equip['placement'] = {'objects_in_room': []}

        self.modified_scene.append(equip)
        return equip

    def _remove_object(self, object_id: str):
        """Remove an object from the modified scene."""
        self.modified_scene = [obj for obj in self.modified_scene
                               if obj.get('object_id') != object_id]

        # Also remove connections to this object from other objects
        for obj in self.modified_scene:
            connections = obj.get('connections')
            if connections is not None:
                obj['connections'] = [conn for conn in connections
                                      if conn.get('object_id') != object_id]

    def get_modified_scene(self) -> List[Dict]:
        """Get the modified scene graph."""
        return self.modified_scene

    def get_changes_summary(self) -> Dict:
        """Get summary of all changes made."""
        baseline_ids = {o['object_id'] for o in self.baseline}
        modified_ids = {o['object_id'] for o in self.modified_scene}

        added = modified_ids - baseline_ids
        removed = baseline_ids - modified_ids

        return {
            'added': list(added),
            'removed': list(removed),
            'total_baseline': len(baseline_ids),
            'total_modified': len(modified_ids)
        }


def reconfigure_extend_loop(baseline_path: str,
                            edge: str = 'east',
                            extension_distance: float = 4.0,
                            add_stations: List[str] = None) -> List[Dict]:
    """
    Convenience function to extend a conveyor loop.

    Args:
        baseline_path: Path to baseline scene JSON
        edge: Which edge to extend
        extension_distance: How far to extend
        add_stations: Station types to add

    Returns:
        Modified scene graph
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    if isinstance(baseline, dict):
        baseline = baseline.get('objects_in_room', [])

    reconfig = StructureAwareReconfiguration(baseline)

    print(f"Baseline structure summary:")
    summary = reconfig.structure.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nExtending {edge} edge by {extension_distance}m...")
    changes = reconfig.extend_edge(edge, extension_distance, add_stations)
    print(f"Changes: {changes}")

    return reconfig.get_modified_scene()


def reconfigure_add_parallel_branch(baseline_path: str,
                                     branch_from_edge: str = 'east',
                                     station_type: str = 'Line_07') -> List[Dict]:
    """
    Add a parallel processing branch for increased throughput.

    Args:
        baseline_path: Path to baseline scene JSON
        branch_from_edge: Which edge to branch from
        station_type: Type of station for the parallel path

    Returns:
        Modified scene graph
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    if isinstance(baseline, dict):
        baseline = baseline.get('objects_in_room', [])

    reconfig = StructureAwareReconfiguration(baseline)

    # Get conveyors on the specified edge
    edge_conveyors = reconfig.structure.edges[branch_from_edge]['conveyors']
    if not edge_conveyors:
        print(f"No conveyors found on {branch_from_edge} edge")
        return baseline

    # Pick the middle conveyor to branch from
    branch_conveyor_id = edge_conveyors[len(edge_conveyors) // 2]
    branch_conveyor = reconfig._find_object(branch_conveyor_id)

    if not branch_conveyor:
        return baseline

    print(f"Creating parallel branch from {branch_conveyor_id}")

    # Create branch conveyor
    branch_pos = branch_conveyor['position']
    offset = 3.0  # Offset for parallel path

    # Determine offset direction based on edge
    if branch_from_edge in ['east', 'west']:
        new_pos = {'x': branch_pos['x'], 'y': branch_pos['y'] + offset, 'z': 0}
    else:
        new_pos = {'x': branch_pos['x'] + offset, 'y': branch_pos['y'], 'z': 0}

    # Create parallel conveyor
    parallel_conv_id = reconfig._get_next_id('Line_04')
    parallel_conv = deepcopy(branch_conveyor)
    parallel_conv['object_id'] = parallel_conv_id
    parallel_conv['position'] = new_pos
    parallel_conv['connections'] = []
    parallel_conv['placement'] = {
        'room_layout_elements': [],
        'objects_in_room': [
            {'object_id': branch_conveyor_id, 'preposition': 'behind', 'is_adjacent': False}
        ]
    }
    reconfig.modified_scene.append(parallel_conv)

    # Create station on parallel conveyor
    station = reconfig._create_inline_station(parallel_conv, station_type)

    print(f"Added: {parallel_conv_id}, {station['object_id']}")

    return reconfig.get_modified_scene()


def parse_reconfiguration_request(request: str, baseline_path: str) -> List[Dict]:
    """
    Parse a natural language reconfiguration request and apply it.

    Args:
        request: Natural language description of desired changes
        baseline_path: Path to baseline scene

    Returns:
        Modified scene graph
    """
    request_lower = request.lower()

    # Detect extension requests
    if 'extend' in request_lower or 'expand' in request_lower:
        # Determine direction
        edge = 'east'  # default
        if 'west' in request_lower:
            edge = 'west'
        elif 'north' in request_lower:
            edge = 'north'
        elif 'south' in request_lower:
            edge = 'south'

        # Determine stations to add
        stations = []
        if 'assembly' in request_lower or 'line_07' in request_lower:
            stations.append('Line_07')
        if 'inspection' in request_lower or 'line_06' in request_lower:
            stations.append('Line_06')
        if 'workstation' in request_lower or 'line_02' in request_lower:
            stations.append('Line_02')

        if not stations:
            stations = ['Line_07', 'Line_06']  # Default

        return reconfigure_extend_loop(
            baseline_path,
            edge=edge,
            extension_distance=4.0,
            add_stations=stations
        )

    # Detect parallel/throughput requests
    elif 'parallel' in request_lower or 'throughput' in request_lower:
        station = 'Line_07'  # default bottleneck
        if 'inspection' in request_lower:
            station = 'Line_06'

        return reconfigure_add_parallel_branch(
            baseline_path,
            branch_from_edge='east',
            station_type=station
        )

    else:
        print(f"Could not parse request: {request}")
        print("Supported operations: extend/expand, parallel/throughput")
        with open(baseline_path) as f:
            return json.load(f)


if __name__ == '__main__':
    # Test with cell03 baseline
    import sys

    baseline_path = 'scenes/scene_graph_cell03.json'

    # Test extend loop
    print("=== Testing Extend Loop ===")
    modified = reconfigure_extend_loop(
        baseline_path,
        edge='east',
        extension_distance=4.0,
        add_stations=['Line_07', 'Line_06']
    )

    # Save result
    output_path = 'scenes/scene_graph_cell03_structure_extended.json'
    with open(output_path, 'w') as f:
        json.dump(modified, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Total objects: {len(modified)}")
