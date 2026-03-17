import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import cv2
from copy import copy, deepcopy
import random
import math

from core.constraint_functions import get_above_constraint, get_behind_constraint, get_in_corner_constraint, get_in_front_constraint, get_left_of_constraint, get_right_of_constraint, get_on_constraint, get_under_contraint, get_through_constraint

ROOM_LAYOUT_ELEMENTS = ["south_wall", "north_wall", "west_wall", "east_wall", "ceiling", "middle of the room"]


def safe_topological_sort(G):
    """
    Safely get topological ordering of a graph.
    If the graph contains cycles (e.g., conveyor loops), returns nodes in arbitrary order.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of nodes in topological order, or all nodes if graph has cycles
    """
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Graph contains cycles - return nodes in arbitrary order
        return list(G.nodes())

def get_room_priors(room_dimensions):
    x_mid = room_dimensions[0] / 2
    y_mid = room_dimensions[1] / 2
    z_mid = room_dimensions[2] / 2

    room_priors = [
        {"object_id": "south_wall", "itemType": "wall", "position": {"x": x_mid, "y": 0, "z": z_mid}, "size_in_meters": {"length": room_dimensions[0], "width": 0.0, "height": room_dimensions[2]}, "rotation": {"z_angle": 0.0}},
        {"object_id": "north_wall", "itemType": "wall", "position": {"x": x_mid, "y": room_dimensions[1], "z": z_mid}, "size_in_meters": {"length": room_dimensions[0], "width": 0.0, "height": room_dimensions[2]}, "rotation": {"z_angle": 180.0}},
        {"object_id": "east_wall", "itemType": "wall", "position": {"x": room_dimensions[0], "y": y_mid, "z": z_mid}, "size_in_meters": {"length": room_dimensions[1], "width": 0.0, "height": room_dimensions[2]}, "rotation": {"z_angle": 270.0}},
        {"object_id": "west_wall", "itemType": "wall", "position": {"x": 0, "y": y_mid, "z": z_mid}, "size_in_meters": {"length": room_dimensions[1], "width": 0.0, "height": room_dimensions[2]}, "rotation": {"z_angle": 90.0}},
        {"object_id": "middle of the room", "itemType": "floor", "position": {"x": x_mid, "y": y_mid, "z": 0}, "size_in_meters": {"length": room_dimensions[0], "width": room_dimensions[1], "height": 0.0}, "rotation": {"z_angle": 0.0}},
        {"object_id": "ceiling", "itemType": "ceiling", "position": {"x": x_mid, "y": y_mid, "z": room_dimensions[2]}, "size_in_meters": {"length": room_dimensions[0], "width": room_dimensions[1], "height": 0.0}, "rotation": {"z_angle": 0.0}}
    ]

    return room_priors

def extract_list_from_json(input_json):
    for value in input_json.values(): 
        if isinstance(value, list):
            return value
        
def is_thin_object(obj):
    """
    Returns True if the object is thin
    """
    size = obj["size_in_meters"]
    return min(size.values()) > 0.0 and max(size.values()) / min(size.values()) >= 40.0

def is_point_bbox(position):
    """
    Returns whether the plausible bounding box is a point
    """
    return np.isclose(position[0], position[1]) and np.isclose(position[2], position[3]) and np.isclose(position[4], position[5])

def get_rotation(obj_A, scene_graph, _visited=None):
    """
    Get the rotation of an object in the scene graph.
    
    Args:
        obj_A: The object to get rotation for
        scene_graph: The scene graph containing all objects
        _visited: Internal set to track visited objects and prevent infinite recursion
                  in cyclic graphs (e.g., conveyor loops)
    """
    # Initialize visited set on first call
    if _visited is None:
        _visited = set()
    
    # Check for cycles
    obj_id = obj_A.get("object_id", id(obj_A))
    if obj_id in _visited:
        # Cycle detected - return default rotation
        return 0.0
    _visited.add(obj_id)
    
    layout_rot = {
        "west_wall" : 270.0,
        "east_wall" : 90.0,
        "north_wall" : 0.0,
        "south_wall" : 180.0,
        "middle of the room" : 0.0,
        "ceiling" : 0.0
    }

    if "rotation" in obj_A.keys():
        rot = obj_A["rotation"]["z_angle"]
    elif "facing" in obj_A.keys() and obj_A["facing"] in layout_rot.keys():
        rot = layout_rot[obj_A["facing"]]
    elif obj_A["object_id"] in layout_rot.keys():
        rot = layout_rot[obj_A["object_id"]]
    else: 
        parents = []
        for x in obj_A["placement"]["objects_in_room"]:
            p = next((element for element in scene_graph if element.get("object_id") == x["object_id"]), None)
            if p is None:
                # skip references to objects that haven't been added yet
                continue
            parents.append(p)
        if len(parents) > 0:
            parent = parents[0]
            rot = get_rotation(parent, scene_graph, _visited)
        else:
            rot = 0.0
    return rot

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
        
def get_conflicts(G, scene_graph):
    conflicts_wall = check_wall_relationship_impossibilities(G, scene_graph)
    conflicts_corner = check_corner_relationship_impossibilities(G, scene_graph)
    conflicts_room_layout = find_room_layout_conflicts(G, scene_graph)
    conflicts_one_parent = check_corner_relationships(G, scene_graph)
    conflicts_impossible_relationships = check_impossible_relationships(G, scene_graph)
    return conflicts_corner + conflicts_room_layout + conflicts_one_parent + conflicts_impossible_relationships + conflicts_wall

def get_size_conflicts(G, scene_graph, user_input, room_priors, verbose=False):
    conflicts_size = check_size_conflicts(G, scene_graph, user_input, room_priors, verbose)
    return conflicts_size

def preprocess_scene_graph(scene_graph):
    # Correct the preposition for objects in the middle of the room
    name_lookup = {}
    for obj in scene_graph:
        obj_id = obj.get("object_id")
        if obj_id:
            name_lookup[obj_id] = obj_id
            name_lookup[obj_id.lower()] = obj_id

    for obj in scene_graph:
        placement = obj.get("placement")
        if placement is None:
            placement = {}
            obj["placement"] = placement
        room_layout_elements = placement.get("room_layout_elements")
        if room_layout_elements is None:
            room_layout_elements = []
            placement["room_layout_elements"] = room_layout_elements
        objects_in_room = placement.get("objects_in_room")
        if objects_in_room is None:
            objects_in_room = []
            placement["objects_in_room"] = objects_in_room
        connections = obj.get("connections")
        if connections is None:
            obj["connections"] = []
        if not obj["is_on_the_floor"] and "middle of the room" in [x["layout_element_id"] for x in room_layout_elements]:
            #Delete that relationship
            placement["room_layout_elements"] = [x for x in room_layout_elements if x["layout_element_id"] != "middle of the room"]
            room_layout_elements = placement["room_layout_elements"]
        # If the object has explicit object relationships, drop the default
        # "middle of the room" anchor to avoid over-constraining placement.
        if objects_in_room and any(elem.get("preposition") for elem in objects_in_room):
            placement["room_layout_elements"] = [
                x for x in room_layout_elements if x.get("layout_element_id") != "middle of the room"
            ]
            room_layout_elements = placement["room_layout_elements"]
        for elem in room_layout_elements:
            if elem["preposition"] == "in the corner" and elem["layout_element_id"] in ["middle of the room", "ceiling"]:
                elem["preposition"] = "on"
        cleaned_relations = []
        for elem in objects_in_room:
            target_id = elem.get("object_id")
            if target_id == "middle of the room":
                continue
            # Normalize Line_07 relationships to conveyors as pass-through
            if obj.get("object_id", "").lower().startswith("line_07"):
                if str(target_id).lower().startswith(("line_01", "line_02", "line_03", "line_04", "line_05", "conveyor")):
                    elem = elem.copy()
                    elem["preposition"] = "through"
            resolved_id = None
            if target_id in name_lookup:
                resolved_id = target_id
            else:
                normalized = name_lookup.get(str(target_id).lower())
                if normalized:
                    resolved_id = normalized
                else:
                    closest_id = next(
                        (
                            x["object_id"]
                            for x in scene_graph
                            if target_id and str(target_id).lower() in x["object_id"].lower()
                        ),
                        None,
                    )
                    if closest_id is not None:
                        resolved_id = closest_id
            if resolved_id is not None:
                elem["object_id"] = resolved_id
                cleaned_relations.append(elem)
            else:
                print(f"⚠️ Reference '{target_id}' not found in scene graph; dropping this relationship.")
        placement["objects_in_room"] = cleaned_relations
    
    return scene_graph

def strip_room_layout_elements(scene_graph):
    """Return a deep-copied scene graph without placement.room_layout_elements and with defaults."""
    sanitized_graph = deepcopy(scene_graph)

    def _normalize_obj(obj):
        if not isinstance(obj, dict):
            return
        placement = obj.get("placement")
        if not isinstance(placement, dict):
            placement = {}
        # Remove room layout references
        if "room_layout_elements" in placement:
            placement = {k: v for k, v in placement.items() if k != "room_layout_elements"}
        objects_in_room = placement.get("objects_in_room")
        if not isinstance(objects_in_room, list):
            objects_in_room = []
        placement["objects_in_room"] = objects_in_room
        obj["placement"] = placement
        if obj.get("connections") is None:
            obj["connections"] = []

    if isinstance(sanitized_graph, dict):
        objects = sanitized_graph.get("objects_in_room", [])
        for obj in objects:
            _normalize_obj(obj)
        return sanitized_graph

    for obj in sanitized_graph:
        _normalize_obj(obj)
    return sanitized_graph


def populate_conveyor_connections(scene_graph):
    """Infer conveyor connections based on object bounding boxes and rotation."""
    graph = deepcopy(scene_graph)

    # Equipment types that form the conveyor/transport system
    # Line_01: Lift Transfer Unit
    # Line_02: Roller Conveyor Bridge  
    # Line_03: Curved Belt Conveyor (corner)
    # Line_04: Long Straight Belt Conveyor
    # Line_05: Short Straight Belt Conveyor
    TRANSPORT_PREFIXES = ("line_01", "line_02", "line_03", "line_04", "line_05", "conveyor")
    CORNER_PREFIXES = ("line_03", "conveyorcorner")

    def is_conveyor(obj_dict):
        obj_id = (obj_dict or {}).get("object_id", "")
        if not isinstance(obj_id, str):
            return False
        obj_id_lower = obj_id.lower()
        return any(obj_id_lower.startswith(prefix) for prefix in TRANSPORT_PREFIXES)

    def is_corner(obj_dict):
        obj_id = (obj_dict or {}).get("object_id", "")
        if not isinstance(obj_id, str):
            return False
        obj_id_lower = obj_id.lower()
        return any(obj_id_lower.startswith(prefix) for prefix in CORNER_PREFIXES)

    def ensure_connections(obj_dict):
        connections = obj_dict.get("connections")
        if connections is None:
            obj_dict["connections"] = []
        return obj_dict["connections"]

    def get_objects_list(graph_obj):
        if isinstance(graph_obj, dict):
            return graph_obj.get("objects_in_room", [])
        return graph_obj

    objects = get_objects_list(graph)
    conveyors = [obj for obj in objects if isinstance(obj, dict) and is_conveyor(obj)]
    if not conveyors:
        return graph

    endpoints = []
    for obj in conveyors:
        endpoints.extend(_create_conveyor_endpoints(obj))

    if len(endpoints) < 2:
        return graph

    best_pairs = {}
    for i in range(len(endpoints)):
        ep_a = endpoints[i]
        obj_a = ep_a["owner"]
        for j in range(i + 1, len(endpoints)):
            ep_b = endpoints[j]
            obj_b = ep_b["owner"]
            if obj_a is obj_b:
                continue
            key = tuple(sorted((obj_a["object_id"], obj_b["object_id"])))
            involves_corner = is_corner(obj_a) or is_corner(obj_b)
            planar_distance = _planar_distance(ep_a["position"], ep_b["position"])
            height_difference = abs(ep_a["position"][2] - ep_b["position"][2])
            distance_limit = 0.9 if involves_corner else 0.45
            height_limit = 0.2 if involves_corner else 0.12
            if planar_distance > distance_limit or height_difference > height_limit:
                continue
            candidate = {
                "a": ep_a,
                "b": ep_b,
                "planar": planar_distance,
                "height": height_difference,
                "corner": involves_corner,
            }
            if key not in best_pairs or planar_distance < best_pairs[key]["planar"]:
                best_pairs[key] = candidate

    if not best_pairs:
        return graph

    # Validate existing connections and remove only those pointing to non-existent objects
    # We do NOT blindly clear existing transport connections, to preserve baseline/manual edits.
    valid_ids = {o.get("object_id") for o in objects if isinstance(o, dict)}
    
    for obj in conveyors:
        connections = ensure_connections(obj)
        # Filter out connections to objects that don't exist in the scene
        # But KEEP existing transport connections if they are valid
        valid_connections = []
        for conn in connections:
            target_id = conn.get("object_id")
            if target_id in valid_ids:
                valid_connections.append(conn)
        obj["connections"] = valid_connections

    for candidate in best_pairs.values():
        ep_a = candidate["a"]
        ep_b = candidate["b"]
        obj_a = ep_a["owner"]
        obj_b = ep_b["owner"]
        connection_type = "connected_corner" if candidate["corner"] else "connected"
        _add_connection_entry(obj_a, ep_a["label"], ep_b["label"], obj_b["object_id"], connection_type)
        _add_connection_entry(obj_b, ep_b["label"], ep_a["label"], obj_a["object_id"], connection_type)

    # Validate T-branch terminations only if we have actual connections
    if best_pairs:
        _validate_t_branch_terminations(objects)

    return graph


def _validate_t_branch_terminations(objects):
    """
    Validate that T-branch spur lines terminate with Line_01 (Lift Transfer Unit).
    
    T-branch rules:
    1. Line_04/Line_05 have side connections (right_negative endpoint)
    2. When a side connection is used, follow the chain of connected conveyors
    3. The chain must end with a Line_01
    4. Branch conveyors must be oriented perpendicular to the main conveyor
       (their primary_forward direction should align with the main conveyor's side direction)
    
    This function logs warnings for invalid configurations.
    """
    # Build object lookup and adjacency map
    obj_lookup = {}
    adjacency = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_id = obj.get("object_id", "")
        obj_lookup[obj_id] = obj
        connections = obj.get("connections") or []
        adjacency[obj_id] = []
        for conn in connections:
            target = conn.get("object_id")
            source_ep = conn.get("source_endpoint", "")
            if target:
                adjacency[obj_id].append({
                    "target": target,
                    "source_endpoint": source_ep,
                    "target_endpoint": conn.get("target_endpoint", "")
                })
    
    # Find T-branch origins (Line_04/Line_05 with right_negative connections)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_id = obj.get("object_id", "").lower()
        full_obj_id = obj.get("object_id", "")
        if not (obj_id.startswith("line_04") or obj_id.startswith("line_05")):
            continue
        
        connections = obj.get("connections") or []
        for conn in connections:
            source_ep = conn.get("source_endpoint", "")
            if source_ep == "right_negative":
                # This is a potential T-branch origin, validate the branch
                target = conn.get("object_id")
                if target:
                    target_lower = target.lower()
                    
                    # Skip Line_02 (Roller Bridge) - these are side-mounted alignment conveyors, NOT T-branches
                    # Per equipment catalog: "Line_02 is placed on the OUTER SIDE of straight conveyors for alignment. It is NOT part of T-branch."
                    if target_lower.startswith("line_02"):
                        continue
                    
                    # Skip Line_03 (Corner Module) - these are corner pieces connecting main loop segments, NOT T-branches
                    # Corner modules are used to change direction in the main loop, not for branching
                    if target_lower.startswith("line_03"):
                        continue
                    
                    # Check 1: Branch terminates with Line_01
                    terminates_correctly = _trace_spur_to_line01(target, adjacency, set())
                    if not terminates_correctly:
                        print(f"⚠️ T-branch from {full_obj_id} -> {target} does not terminate with Line_01")
                    
                    # Check 2: Branch conveyor is oriented perpendicular to main
                    _validate_branch_orientation(obj, target, obj_lookup)


def _validate_branch_orientation(main_obj, branch_obj_id, obj_lookup):
    """
    Validate that a T-branch conveyor is oriented perpendicular to the main conveyor.
    
    The branch conveyor's primary_forward direction should be aligned with the 
    main conveyor's side (right) direction, meaning the branch extends outward
    from the main line, not parallel to it.
    
    Expected rotation difference: 90° or 270° (perpendicular)
    """
    branch_obj = obj_lookup.get(branch_obj_id)
    if not branch_obj:
        return
    
    branch_id_lower = branch_obj_id.lower()
    # Only check orientation for Line_04/Line_05 branch conveyors
    if not (branch_id_lower.startswith("line_04") or branch_id_lower.startswith("line_05")):
        return
    
    main_rotation = (main_obj.get("rotation") or {}).get("z_angle", 0.0)
    branch_rotation = (branch_obj.get("rotation") or {}).get("z_angle", 0.0)
    
    # Calculate rotation difference (normalized to 0-360)
    diff = abs(branch_rotation - main_rotation) % 360
    # Perpendicular means 90° or 270° difference
    is_perpendicular = (85 <= diff <= 95) or (265 <= diff <= 275)
    
    if not is_perpendicular:
        print(f"⚠️ T-branch conveyor {branch_obj_id} is not perpendicular to main conveyor {main_obj.get('object_id')}. "
              f"Main rotation: {main_rotation}°, Branch rotation: {branch_rotation}° (diff: {diff}°). "
              f"Branch conveyors should extend perpendicular to the main line.")


def _trace_spur_to_line01(current_id, adjacency, visited):
    """
    Trace a spur line to check if it terminates with Line_01.
    Returns True if the chain ends with Line_01.
    """
    if current_id in visited:
        return False  # Cycle detected
    visited.add(current_id)
    
    current_lower = current_id.lower()
    
    # Check if current is Line_01 (terminal)
    if current_lower.startswith("line_01"):
        return True
    
    # Check if current is a conveyor that can continue the spur
    is_spur_conveyor = (
        current_lower.startswith("line_04") or 
        current_lower.startswith("line_05") or
        current_lower.startswith("line_02")
    )
    
    if not is_spur_conveyor:
        return False  # Non-conveyor in spur line
    
    # Follow primary forward connections
    neighbors = adjacency.get(current_id, [])
    for neighbor in neighbors:
        source_ep = neighbor.get("source_endpoint", "")
        # Only follow primary forward connections (not side connections)
        if "primary_forward" in source_ep:
            if _trace_spur_to_line01(neighbor["target"], adjacency, visited):
                return True
    
    # If no forward connections, check if we're at the end
    # A spur line without forward connections should be Line_01
    return False


def _planar_distance(pos_a, pos_b):
    ax, ay = pos_a[0], pos_a[1]
    bx, by = pos_b[0], pos_b[1]
    return math.hypot(ax - bx, ay - by)


def _create_conveyor_endpoints(obj):
    """
    Create connection endpoints for a conveyor/transport object.
    
    Endpoint types:
    - primary_forward_positive/negative: Main flow direction endpoints (front/back)
    - right_positive/negative: Side connection endpoints for T-branches (Line_04/Line_05 only)
    - forward_positive/negative, right_positive/negative: Corner piece endpoints (Line_03)
    """
    position = obj.get("position") or {}
    size = obj.get("size_in_meters") or {}
    rotation = obj.get("rotation") or {}
    x = float(position.get("x", 0.0))
    y = float(position.get("y", 0.0))
    z = float(position.get("z", 0.0))
    height = float(size.get("height", 0.0))
    length = float(size.get("length", 0.0))
    width = float(size.get("width", 0.0))
    rot_deg = float(rotation.get("z_angle", 0.0))
    angle = math.radians(rot_deg)
    forward = (math.cos(angle), math.sin(angle))
    right = (-math.sin(angle), math.cos(angle))
    endpoints = []

    def add_axis_endpoints(base_label, axis_vector, axis_length):
        if axis_length <= 0.01:
            return
        half = axis_length / 2.0
        px = x + axis_vector[0] * half
        py = y + axis_vector[1] * half
        nx = x - axis_vector[0] * half
        ny = y - axis_vector[1] * half
        top_z = z + height
        endpoints.append(
            {"owner": obj, "label": f"{base_label}_positive", "position": (px, py, top_z)}
        )
        endpoints.append(
            {"owner": obj, "label": f"{base_label}_negative", "position": (nx, ny, top_z)}
        )

    obj_id = obj.get("object_id", "").lower()
    
    # Line_03 is a curved belt conveyor (corner piece)
    is_corner_piece = "corner" in obj_id or obj_id.startswith("line_03")
    # Line_04 and Line_05 have side connections for T-branches
    has_side_connections = obj_id.startswith("line_04") or obj_id.startswith("line_05")
    
    if is_corner_piece:
        # Corner pieces have two perpendicular connection axes
        add_axis_endpoints("forward", forward, max(width, 0.0))
        add_axis_endpoints("right", right, max(length, 0.0))
    elif has_side_connections:
        # Straight conveyors with T-branch capability
        # Primary axis: along the length (main flow direction)
        add_axis_endpoints("primary_forward", forward, length)
        # Side axis: perpendicular for T-branch connections (right side only, single point)
        # Side connection is at the center of the conveyor, perpendicular to flow
        side_offset = width / 2.0
        side_x = x + right[0] * side_offset
        side_y = y + right[1] * side_offset
        top_z = z + height
        endpoints.append(
            {"owner": obj, "label": "right_negative", "position": (side_x, side_y, top_z)}
        )
    else:
        # Standard conveyor or other transport (Line_01, Line_02)
        if width >= length:
            add_axis_endpoints("primary_forward", forward, max(width, length))
        else:
            add_axis_endpoints("primary_forward", forward, max(length, width))
    return endpoints


def _add_connection_entry(obj, source_label, target_label, target_id, connection_type):
    if not target_id:
        return
    connections = obj.get("connections")
    if connections is None:
        connections = []
        obj["connections"] = connections
    for conn in connections:
        if (
            conn.get("object_id") == target_id
            and conn.get("source_endpoint") == source_label
            and conn.get("target_endpoint") == target_label
        ):
            return
    connections.append(
        {
            "object_id": target_id,
            "connection_type": connection_type,
            "source_endpoint": source_label,
            "target_endpoint": target_label,
        }
    )

def build_graph(scene_graph):
    G = nx.DiGraph()
    # Create graph
    for obj in scene_graph:
        if obj["object_id"] not in G.nodes():
            G.add_node(obj["object_id"])
        obj_scene_graph = obj["placement"]
        for constraint in obj_scene_graph["room_layout_elements"]:
            if constraint["layout_element_id"] not in G.nodes():
                G.add_node(constraint["layout_element_id"])
            G.add_edge(constraint["layout_element_id"], obj["object_id"], weight={"preposition" : constraint["preposition"], "adjacency" : True})
        for constraint in obj_scene_graph["objects_in_room"]:
            if constraint["object_id"] not in G.nodes():
                G.add_node(constraint["object_id"])
            G.add_edge(constraint["object_id"], obj["object_id"], weight={"preposition" : constraint["preposition"], "adjacency" : constraint["is_adjacent"]})
    return G

def find_room_layout_conflicts(G, scene_graph):
    conflicts = []

    topological_order = safe_topological_sort(G)
    node_layout = dict(G.nodes(data=True))
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents = list(G.predecessors(node))
            parents_room_layout = []
            for p in parents:
                layout_value = node_layout.get(p)
                if layout_value is not None:
                    parents_room_layout.append(layout_value)
            if not parents_room_layout:
                node_layout[node] = None
                continue
            different_parent_room_layout = False
            for p in parents_room_layout[1:]:
                if isinstance(p, list):
                    if isinstance(parents_room_layout[0], list):
                        different_parent_room_layout = True if p != parents_room_layout[0] else different_parent_room_layout
                    else:
                        different_parent_room_layout = True if parents_room_layout[0] not in p else different_parent_room_layout
                elif isinstance(p, str):
                    if isinstance(parents_room_layout[0], list):
                        different_parent_room_layout = True if p not in parents_room_layout[0] else different_parent_room_layout
                    else:
                        different_parent_room_layout = True if p != parents_room_layout[0] else different_parent_room_layout
                elif isinstance(p, dict):
                    if isinstance(parents_room_layout[0], list):
                        different_parent_room_layout = True if p not in parents_room_layout[0] else different_parent_room_layout
                    else:
                        different_parent_room_layout = True if p != parents_room_layout[0] else different_parent_room_layout
            if len(parents_room_layout) > 0 and different_parent_room_layout:
                # Filter out "middle of the room" - it's just a default that shouldn't cause conflicts
                non_middle_parents = [p for p in parents if p != "middle of the room"]
                
                # For manufacturing equipment: allow multiple object relationships
                # A conveyor can be "in front of" one object AND "behind" another - this is a linear chain, not a conflict
                # Only flag conflict if ALL parents are room layout elements with contradictory positions (e.g., on south_wall AND north_wall)
                room_layout_only_parents = [p for p in non_middle_parents if p in ROOM_LAYOUT_ELEMENTS]
                object_parents = [p for p in non_middle_parents if p not in ROOM_LAYOUT_ELEMENTS]
                
                # If there are object parents, we allow the placement (let geometry solver handle it)
                # Only check for room layout conflicts if ALL non-middle parents are room layout elements
                has_real_conflict = len(object_parents) == 0 and len(room_layout_only_parents) > 1
                
                # This should be a spatial conflict only for room layout elements, if the relationship isn't 'corner'
                if has_real_conflict and not all([G[p][node]["weight"]["preposition"] == "in the corner" for p in parents if p in G]) and not any([p == "ceiling" for p in parents]):
                    conflict_string = f"The object {node} cannot have the parents {parents} at the same time! Eliminate one."
                    conflict_string += "\nObject to reposition: " + str(get_object_from_scene_graph(node, scene_graph))
                    conflicts.append(conflict_string)
                else:
                    node_layout[node] = {}
            else:
                node_layout[node] = parents_room_layout[0] if parents_room_layout else {}

        if node in ROOM_LAYOUT_ELEMENTS:
            node_layout[node] = node
    return conflicts

def remove_unnecessary_edges(G):
    """
    Remove non-corner relationships if the object has a corner relationship.
    
    Note: This function handles graphs with cycles (e.g., conveyor loops) by
    iterating over all nodes instead of using topological sort.
    """
    # Check if graph has cycles - if so, we can't use topological sort
    try:
        topological_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Graph contains cycles (e.g., conveyor loop) - iterate over all nodes instead
        topological_order = list(G.nodes())
    
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents = list(G.predecessors(node))
            if any([G[p][node]["weight"]["preposition"] == "in the corner" for p in parents]):
                if len(parents) > 2:
                    # Remove the non-corner relationships
                    for p in parents:
                        if G[p][node]["weight"]["preposition"] != "in the corner":
                            print(f"Removing edge {p} -> {node} with preposition {G[p][node]['weight']['preposition']}")
                            G.remove_edge(p, node)
    return G

def handle_under_prepositions(G, scene_graph):
    """
    For objects that are under another object, remove the object if it isn't a thin object
    """
    nodes = G.nodes()
    nodes_to_remove = []
    for node in nodes:
        incoming_e = list(G.in_edges(node, data=True))
        outgoing_e = list(G.out_edges(node, data=True))
        under_obj = any([e[2]["weight"]["preposition"] == "under" for e in incoming_e])
        if under_obj:
            obj = get_object_from_scene_graph(node, scene_graph)
            if not is_thin_object(obj):
                nodes_to_remove.append(node)
                for e in outgoing_e:
                    nodes_to_remove.append(e[1])
    for node in nodes_to_remove:
        print("Removing node: ", node)
        scene_graph = [x for x in scene_graph if x["object_id"] != node]
        if node in G.nodes():
            G.remove_node(node)
    return G, scene_graph

def check_corner_relationships(G, scene_graph):
    def find_corner_vacancy():
        # Find the corner that is not occupied
        corners = [("south_wall", "west_wall"), ("south_wall", "east_wall"), ("north_wall", "west_wall"), ("north_wall", "east_wall")]
        occupied_corners = []
        for wall_1, wall_2 in corners:
            for node in topological_order:
                if node not in ROOM_LAYOUT_ELEMENTS:
                    parents = list(G.predecessors(node))
                    if wall_1 in parents and wall_2 in parents:
                        occupied_corners.append((wall_1, wall_2))
        vacant_corners = list(set(corners) - set(occupied_corners))
        return vacant_corners
    
    def find_corner_occupancy():
        # Find whether corners are occupied by more than one object 
        corners = [("south_wall", "west_wall"), ("south_wall", "east_wall"), ("north_wall", "west_wall"), ("north_wall", "east_wall")]
        occupied_corners = {k : [] for k in corners}
        for wall_1, wall_2 in corners:
            for node in topological_order:
                if node not in ROOM_LAYOUT_ELEMENTS:
                    parents = list(G.predecessors(node))
                    if wall_1 in parents and wall_2 in parents:
                        occupied_corners[(wall_1, wall_2)].append(node)
        return occupied_corners

    topological_order = safe_topological_sort(G)
    conflicts = []

    corner_occupancy = find_corner_occupancy()
    for key, value in corner_occupancy.items():
        if len(value) > 1:
            conflict_string = f"The corner {key[0].split('_')[0]}-{key[1].split('_')[0]} is occupied by more than one object: {value}. Move one of them to another vacant corner."
            conflict_string += "\nVacant corners: " + str(find_corner_vacancy())
            conflicts.append(conflict_string)
        

    # Check whether objects with "corner" relationships have two corresponding walls
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents = list(G.predecessors(node))
            if any([G[p][node]["weight"]["preposition"] == "in the corner" for p in parents]):
                if len(parents) == 1:
                    vacant_corners = find_corner_vacancy()
                    vacant_corners = [f"{c[0].split('_')[0]}-{c[1].split('_')[0]} corner" for c in vacant_corners]
                    conflict_string = f"Corner relationship for {node} has {len(parents)} parent, add another wall to the relationship. \n Current vacant corners: {vacant_corners}"
                    conflict_string += "\nObject to reposition: " + str(get_object_from_scene_graph(node, scene_graph))
                    conflicts.append(conflict_string)
    return conflicts

directional_preps = ["in front", "left of", "behind", "right of"]

def check_corner_relationship_impossibilities(G, scene_graph):
    conflicts = []
    # Check for impossible relationships in corners
    wall_impossible_preps = {
        "south_wall" : "behind",
        "north_wall" : "in front",
        "west_wall" : "left of",
        "east_wall" : "right of"
    }

    topological_order = safe_topological_sort(G)
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents_raw = list(G.predecessors(node))
            parents = list(filter(lambda x : x not in ROOM_LAYOUT_ELEMENTS, parents_raw))
            parents_rot = [get_rotation(next((x for x in scene_graph if x["object_id"] == p), None), scene_graph) for p in parents]
            # Check whether the parent object is in the corner and if this object is located spatially correctly
            for p, r in zip(parents, parents_rot):
                p_parent = list(G.predecessors(p))
                corners = [p_p for p_p in p_parent if G[p_p][p]["weight"]["preposition"] == "in the corner"]
                impossible_preps = []
                if len(corners) != 2:
                    continue
                for p_p in corners:
                    corner_name = corners[0].split('_')[0] + "-" + corners[1].split('_')[0] + " corner"
                    impossible_prep = wall_impossible_preps[p_p]
                    idx = directional_preps.index(impossible_prep)
                    rotated_idx = int((idx + (r // 90)) % len(directional_preps))
                    impossible_prep = directional_preps[rotated_idx]
                    impossible_preps.append(impossible_prep)
                    # print(f"Impossible prep for {p} with rotation {r}: {impossible_prep}")
                if G[p][node]["weight"]["preposition"] in impossible_preps:
                    # print(f"Impossible relationship between {node} and {p} with rotation {r} and relationship {G[p][node]['weight']}")
                    # print(f"Parent '{p}' in edges: {G.out_edges(p, data=True)}")
                    conflict_string = [
                        f"The object {node} cannot be {G[p][node]['weight']['preposition']} the object {p} as it would be placed out of bounds. ",
                        f"The {impossible_preps[0]} and {impossible_preps[1]} the object are out of bounds. Find another relationship for {node} either with {p}, on the {corners[0]} or on the {corners[1]}!",
                        f"This relationship has to be exclusive, you cannot have two objects with the same relative positioning. IMPORTANT: you can only have one relationship in the new scene graph!!!",
                    ]
                    conflict_string = "\n".join(conflict_string)
                    conflict_string += f"The object {p} is on the {corner_name}. "
                    conflict_string += " ".join([f"{p} has the object {edge[1]} {edge[2]['weight']['preposition']} it. " for edge in G.out_edges(p, data=True) if edge[1] != node and edge[2]["weight"]["adjacency"]])
                    conflict_string += "\n Object to reposition: " + str(get_object_from_scene_graph(node, scene_graph))
                    conflicts.append(conflict_string)
    return conflicts

def check_wall_relationship_impossibilities(G, scene_graph):
    conflicts = []
    # Check for impossible relationships in corners
    wall_impossible_preps = {
        "south_wall" : "behind",
        "north_wall" : "in front",
        "west_wall" : "left of",
        "east_wall" : "right of"
    }

    topological_order = safe_topological_sort(G)
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents_raw = list(G.predecessors(node))
            parents = list(filter(lambda x : x not in ROOM_LAYOUT_ELEMENTS, parents_raw))
            parents_rot = [get_rotation(next((x for x in scene_graph if x["object_id"] == p), None), scene_graph) for p in parents]
            # Check whether the parent object is in the corner and if this object is located spatially correctly
            for p, r in zip(parents, parents_rot): 
                p_parent_raw = list(G.predecessors(p))
                p_parent = list(filter(lambda x : x in wall_impossible_preps.keys(), p_parent_raw))
                walls = [p_p for p_p in p_parent if G[p_p][p]["weight"]["preposition"] == "on"]
                for p_p in walls:
                    impossible_prep = wall_impossible_preps[p_p]
                    idx = directional_preps.index(impossible_prep)
                    rotated_idx = int((idx + (r // 90)) % len(directional_preps))
                    impossible_prep = directional_preps[rotated_idx]
                    if G[p][node]["weight"]["preposition"] == impossible_prep:
                        conflict_string =[
                            f"The object {node} cannot be {G[p][node]['weight']['preposition']} the object {p} as it would be placed out of bounds. ",
                            f"The {impossible_prep} the object is out of bounds. Find another relationship for {node} either with {p}, on the {p_p}!",
                            f"This relationship has to be exclusive, you cannot have two objects with the same relative positioning. IMPORTANT: you can only have one relationship in the new scene graph!!!",
                        ]
                        conflict_string = "\n".join(conflict_string)
                        conflict_string += f"The object {p} is on the {p_p}. "
                        conflict_string += " ".join([f"{p} has the object {edge[1]} {edge[2]['weight']['preposition']} it. " for edge in G.out_edges(p, data=True) if edge[1] != node and edge[2]["weight"]["adjacency"]])
                        conflict_string += "\n Object to reposition: " + str(get_object_from_scene_graph(node, scene_graph))
                        conflicts.append(conflict_string)
    return conflicts


def check_impossible_relationships(G, scene_graph):
    conflicts = []
    topological_order = safe_topological_sort(G)
    # Check for impossible relationships between objects
    for node in topological_order:
        if node not in ROOM_LAYOUT_ELEMENTS:
            parents_raw = list(G.predecessors(node))
            parents = list(filter(lambda x : x not in ROOM_LAYOUT_ELEMENTS, parents_raw))
            children = list(G.successors(node))
            node_rot = get_rotation(next((x for x in scene_graph if x["object_id"] == node), None), scene_graph) 
            # Adjacent child exclusivity
            for p in parents:
                prep = G[p][node]["weight"]["preposition"]
                adj = G[p][node]["weight"]["adjacency"]
                if prep in directional_preps and adj:
                    idx = directional_preps.index(prep)
                    rotated_idx = int((idx + (node_rot // 90)) % len(directional_preps))
                    impossible_prep = directional_preps[(rotated_idx + 2) % len(directional_preps)] 
                    for c in children:
                        # Skip if p == c: this is a bidirectional relationship (A "in front of" B, B "behind" A)
                        # which is valid for manufacturing equipment in a linear chain
                        if p == c:
                            continue
                        if G[node][c]["weight"]["preposition"] == impossible_prep and G[node][c]["weight"]["adjacency"]:
                            # print(f"Impossible relationship between {node} and {c} with rotation {node_rot} and relationship {G[node][c]['weight']['preposition']}")
                            conflict_string = f"The object {c} cannot be {G[node][c]['weight']['preposition']} of the object {node} since the {p} object is there. Find another relationship for {c} with {node}!"
                            conflict_string += "\n Object to reposition: " + str(get_object_from_scene_graph(c, scene_graph))
                            conflicts.append(conflict_string)
    return conflicts

def get_cluster_size(node, G, scene_graph, _visited=None): 
    """
    Get the size of the cluster of objects.
    
    Args:
        node: The node to get cluster size for
        G: The graph
        scene_graph: The scene graph
        _visited: Internal set to track visited nodes and prevent infinite recursion
                  in cyclic graphs (e.g., conveyor loops)
    """
    # Initialize visited set on first call
    if _visited is None:
        _visited = set()
    
    # Check for cycles
    if node in _visited:
        # Cycle detected - return empty size constraint
        return {"left of": 0.0, "right of": 0.0, "behind": 0.0, "in front": 0.0}, set()
    _visited.add(node)
    
    # Get the size of the cluster of objects
    node_obj = get_object_from_scene_graph(node, scene_graph)
    try:
        node_obj_rot = get_rotation(node_obj, scene_graph)
    except:
        print(f"Node: {node}")
        # Return default instead of raising to handle cycles gracefully
        return {"left of": 0.0, "right of": 0.0, "behind": 0.0, "in front": 0.0}, set()
    # Get the outgoing edges
    outgoing_e = list(G.out_edges(node, data=True))
    outgoing_nodes = [edge[1] for edge in outgoing_e]
    # Get the topological order of the outgoing nodes (handles cycles gracefully)
    topological_order_reversed = list(reversed(safe_topological_sort(G)))
    topological_outgoing_nodes = [node for node in topological_order_reversed if node in outgoing_nodes]
    outgoing_e_sorted = sorted(outgoing_e, key=lambda x : topological_outgoing_nodes.index(x[1]) if x[1] in topological_outgoing_nodes else len(topological_outgoing_nodes))
    size_constraint = {"left of" : 0.0, "right of" : 0.0, "behind" : 0.0, "in front" : 0.0}
    children_objs = set()
    if len(outgoing_e_sorted) != 0:
        for edge in outgoing_e_sorted:
            # Check if the child object is already in the children objects
            if edge[1] in children_objs:
                continue
            # Check if the preposition is a directional preposition
            if edge[2]["weight"]["preposition"] not in directional_preps:
                continue
            
            edge_obj = get_object_from_scene_graph(edge[1], scene_graph)
            children_objs.add(edge[1])
            edge_obj_rot = get_rotation(edge_obj, scene_graph)
            rot_diff = abs(node_obj_rot - edge_obj_rot)
            prep = edge[2]["weight"]["preposition"]
            adj = edge[2]["weight"]["adjacency"]

            # Find the side of the child object to add to the size constraint
            direction_check = lambda diff, prep: (diff % 180 == 0 and prep in ["left of", "right of"]) or (diff % 90 == 0 and prep in ["in front", "behind"])
            size_constraint_key = "length" if direction_check(rot_diff, prep) else "width"
            side_to_add = ("left of", "right of") if size_constraint_key == "length" else ("in front", "behind")
            size_constraint_value = edge_obj["size_in_meters"][size_constraint_key]

            # Retrieve the size of the cluster and the additional descendants of the child object
            edge_cluster_size, edge_children = get_cluster_size(edge[1], G, scene_graph, _visited.copy())
            children_objs = children_objs.union(edge_children)

            # Adjust the size constraint based on the preposition 
            constraints = ["left of", "right of", "in front", "behind"]
            value_to_add = size_constraint_value + edge_cluster_size[side_to_add[0]] + edge_cluster_size[side_to_add[1]]
            if prep in constraints:
                if adj:
                    size_constraint[prep] = max(size_constraint[prep], value_to_add)
                else:
                    size_constraint[prep] += value_to_add         
    return size_constraint, children_objs

def check_size_conflicts(G, scene_graph, user_input, room_priors, verbose=False):
    conflicts = []
    topological_order_reversed = list(reversed(safe_topological_sort(G)))

    if verbose:
        for node in topological_order_reversed:
            if node not in ROOM_LAYOUT_ELEMENTS:
                clstr_size, children_objs = get_cluster_size(node, G, scene_graph)
                
    # Find cluster size conflicts
    for node in topological_order_reversed:
        if node not in ROOM_LAYOUT_ELEMENTS:
            node_obj = get_object_from_scene_graph(node, scene_graph)
            node_obj_rot = get_rotation(node_obj, scene_graph)
            outgoing_e = list(G.out_edges(node, data=True))
            size_constraint = {"left of" : 0.0, "right of" : 0.0, "behind" : 0.0, "in front" : 0.0, "on" : [0.0, 0.0]}
            for edge in outgoing_e:
                edge_obj = get_object_from_scene_graph(edge[1], scene_graph)
                edge_obj_rot = get_rotation(edge_obj, scene_graph)
                rot_diff = abs(node_obj_rot - edge_obj_rot)
                prep = edge[2]["weight"]["preposition"]
                adj = edge[2]["weight"]["adjacency"]

                direction_check = lambda diff, prep: (diff % 180 == 0 and prep in ["left of", "right of"]) or (diff % 90 == 0 and prep in ["in front", "behind"])
                size_constraint_key = "width" if direction_check(rot_diff, prep) else "length"

                if prep not in directional_preps and prep != "on":
                    continue

                size_constraint_value = edge_obj["size_in_meters"][size_constraint_key]

                if adj:
                    if prep in ["left of", "right of", "in front", "behind"]:
                        size_constraint[prep] += size_constraint_value
                    elif prep == "on":
                        if rot_diff % 180 == 0:
                            size_constraint["on"][0] += edge_obj["size_in_meters"]["length"]
                            size_constraint["on"][1] += edge_obj["size_in_meters"]["width"]
                        else:
                            size_constraint["on"][0] += edge_obj["size_in_meters"]["width"]
                            size_constraint["on"][1] += edge_obj["size_in_meters"]["length"]
  
            for prep in ["in front", "behind", "left of", "right of"]:
                constraint_key = "length" if prep in ["in front", "behind"] else "width"
                if node_obj["size_in_meters"][constraint_key] < size_constraint[prep]:
                    conflict_str = f"The {constraint_key} of the object {node} is too small to accommodate the following object {prep} of it!"
                    nodes = [edge[1] for edge in outgoing_e if edge[2]["weight"]["preposition"] == prep]
                    conflict_str += "\nDelete one of these nodes depending on which one is the least important for the user preference and the room's functionality: "                
                    conflict_str += ", ".join(nodes)
                    conflict_str += f"\nUser preference: {user_input}"
                    conflicts.append(conflict_str)
            if node_obj["size_in_meters"]["length"] < size_constraint["on"][0] or node_obj["size_in_meters"]["width"] < size_constraint["on"][1]:
                nodes = [edge[1] for edge in outgoing_e if edge[2]["weight"]["preposition"] == "on"]
                conflict_str = f"The area of the {node} is too small to accommodate all of the following objects on it!"
                conflict_str += "\nDelete one of these nodes depending on which one is the least important for the user preference and the room's functionality: "                
                conflict_str += ", ".join(nodes)
                conflict_str += f"\nUser preference: {user_input}"
                conflicts.append(conflict_str)
                
        if node in ROOM_LAYOUT_ELEMENTS:   
            node_obj = get_object_from_scene_graph(node, room_priors)
            node_obj_rot = get_rotation(node_obj, scene_graph)
            outgoing_e = list(G.out_edges(node, data=True))
            outgoing_nodes = [edge[1] for edge in outgoing_e]
            topological_outgoing_nodes = [node for node in topological_order_reversed if node in outgoing_nodes]
            outgoing_e_sorted = sorted(outgoing_e, key=lambda x : topological_outgoing_nodes.index(x[1]))

            outgoing_set = set()
            size_constraint = 0.0 if node != "middle of the room" else (0.0, 0.0)
            for edge in outgoing_e_sorted:
                if edge[1] in outgoing_set:
                    continue
                edge_obj = get_object_from_scene_graph(edge[1], scene_graph)
                if not edge_obj["is_on_the_floor"]:
                    continue
                edge_obj_rot = get_rotation(edge_obj, scene_graph)
                cluster_size, e_children = get_cluster_size(edge[1], G, scene_graph)
                print(f"Cluster size for {edge[1]}: {cluster_size}")
                rot_diff = abs(node_obj_rot - edge_obj_rot)
                constraint_key = ("length", "width") if rot_diff % 180 == 0 else ("width", "length")
                side_to_add = (("left of", "right of"),("in front", "behind"))  if constraint_key[0] == "length" else (("in front", "behind"), ("left of", "right of"))

                outgoing_set.add(edge[1])
                outgoing_set = outgoing_set.union(e_children)
                if node == "middle of the room":
                    x = edge_obj["size_in_meters"][constraint_key[0]] + cluster_size[side_to_add[0][0]] + cluster_size[side_to_add[0][1]]
                    constraint_x = max(size_constraint[0], x)
                    y = edge_obj["size_in_meters"][constraint_key[1]] + cluster_size[side_to_add[1][0]] + cluster_size[side_to_add[1][1]]
                    constraint_y = max(size_constraint[1], y)
                    size_constraint = (constraint_x, constraint_y)
                else:
                    size_constraint += edge_obj["size_in_meters"][constraint_key[0]] + cluster_size[side_to_add[0][0]] + cluster_size[side_to_add[0][1]]

            if verbose:
                print(f"Size constraint for {node}: {size_constraint}!")
                print(f"Outgoing Set: {outgoing_set}")
                print("\n")

            if node != "middle of the room":
                if node_obj["size_in_meters"]["length"] < size_constraint:
                    conflict_str = f"The length of the {node} is too small to accommodate all of the following objects on it: "
                    conflict_str += "\nDelete one of these nodes depending on which one is the least important for the user preference and the room's functionality: "
                    conflict_str += ", ".join(outgoing_set)
                    conflict_str += f"\nUser preference: {user_input}"
                    conflicts.append(conflict_str)
            else:
                if node_obj["size_in_meters"]["length"] < size_constraint[0]:
                    conflict_str = f"The length of the {node} is too small to accommodate all of the following objects on it: "
                    conflict_str += "\nDelete one of these nodes depending on which one is the least important for the user preference and the room's functionality: "
                    conflict_str += ", ".join(outgoing_set)
                    conflict_str += f"\nUser preference: {user_input}"
                    conflicts.append(conflict_str)
                if node_obj["size_in_meters"]["width"] < size_constraint[1]:
                    conflict_str = f"The width of the {node} is too small to accommodate all of the following objects on it: "
                    conflict_str += "\nDelete one of these nodes depending on which one is the least important for the user preference and the room's functionality: "
                    conflict_str += ", ".join(outgoing_set)
                    conflict_str += f"\nUser preference: {user_input}"
                    conflicts.append(conflict_str)
    return conflicts

def get_cluster_objects(scene_graph):
    object_ids_by_scene_graph = {}

    for obj in scene_graph:
        # Don't add thin objects to the cluster
        if is_thin_object(obj):
            continue
        placement = obj.get("placement")
        if placement:
            edges = placement["objects_in_room"] + placement["room_layout_elements"]
            scene_graph_set = frozenset([tuple(sorted(x.items())) for x in edges])
            if scene_graph_set in object_ids_by_scene_graph:
                object_ids_by_scene_graph[scene_graph_set].append(obj["object_id"])
            else:
                object_ids_by_scene_graph[scene_graph_set] = [obj["object_id"]]

    # Filter out groups with only one object
    object_ids_groups = {k: v for k, v in object_ids_by_scene_graph.items() if len(v) > 1 and len(k) > 0}

    return object_ids_groups

def get_object_from_scene_graph(obj_id, scene_graph):
    """
    Get the object from the scene graph by its id
    """
    return next((x for x in scene_graph if x["object_id"] == obj_id), None)

def has_one_parent_and_one_child(tree):
        for node in tree.nodes():
            if tree.in_degree(node) > 1 or tree.out_degree(node) > 1:
                return False
        return True

def find_edges_to_flip(tree):
        edges_to_flip = []
        for node in tree.nodes():
            if tree.in_degree(node) > 1 or tree.out_degree(node) > 1:
                # If a node has more than one parent or child, find the edges to flip
                for parent in list(tree.predecessors(node)):
                    if tree.in_degree(node) > 1:
                        edges_to_flip.append((parent, node))
                for child in list(tree.successors(node)):
                    if tree.out_degree(node) > 1:
                        edges_to_flip.append((node, child))
        return edges_to_flip

def _graph_violation_score(tree):
        """Count how many extra parents/children exist beyond a binary tree."""
        score = 0
        for node in tree.nodes():
            score += max(0, tree.in_degree(node) - 1)
            score += max(0, tree.out_degree(node) - 1)
        return score

def flip_edges(tree, root_node, verbose=False):
    flipped_edges = {}
    max_iterations = max(10, len(tree.edges()) * 4)
    iterations = 0
    while not has_one_parent_and_one_child(tree) and iterations < max_iterations:
        current_score = _graph_violation_score(tree)
        edges_to_flip = find_edges_to_flip(tree)
        if not edges_to_flip:
            break  # No more edges to flip

        # Try edges and pick the one that minimizes violation score
        best_edge = None
        best_score = current_score
        for edge in edges_to_flip:
            tree.remove_edge(*edge)
            tree.add_edge(edge[1], edge[0])
            score = _graph_violation_score(tree)
            tree.remove_edge(edge[1], edge[0])
            tree.add_edge(*edge)
            if score < best_score:
                best_score = score
                best_edge = edge

        if best_edge is None:
            # No edge flip improves the score; avoid infinite loop
            if verbose:
                print("No improving edge flips found; stopping.")
            break

        tree.remove_edge(*best_edge)
        tree.add_edge(best_edge[1], best_edge[0])
        flipped_edges[best_edge] = True
        iterations += 1
        if verbose:
            print(f"Flipped edge {best_edge} (score {current_score} -> {best_score})")

    if iterations >= max_iterations and verbose:
        print(f"Stopped flipping after {iterations} iterations (limit reached).")
    
    while len(list(nx.simple_cycles(tree))) > 0:
        cycles = list(nx.simple_cycles(tree))
        tree.remove_edge(cycles[0][-1], cycles[0][0])
    
    # Populate the dictionary for the remaining edges
    for edge in tree.edges():
        if edge not in flipped_edges:
            flipped_edges[edge] = False

    return tree, flipped_edges

def flip_edges_to_binary_tree(graph, root_node, verbose):
    tree = nx.DiGraph(graph)
    flipped_edges = {}

    if verbose:
        print("Root Node: ", root_node)
    # Ensure that the graph is weakly connected
    if not nx.is_weakly_connected(tree):
        print("The input graph is not weakly connected.")
        return None

    # Perform edge flips until a binary tree is obtained
    while not is_binary_tree(tree, root_node):
        non_tree_edges = find_non_tree_edges(tree, root_node)
        if verbose:
            print("Non tree edges: ", non_tree_edges)
        if not non_tree_edges:
            break  # No more edges to flip

        edge_to_flip = non_tree_edges[0]
        tree.remove_edge(*edge_to_flip)
        tree.add_edge(edge_to_flip[1], edge_to_flip[0])

        if (edge_to_flip[1], edge_to_flip[0]) not in find_non_tree_edges(tree, root_node):
            # Update the dictionary to indicate that the edge has been flipped
            flipped_edges[edge_to_flip] = True
        else:
            # If the edge was flipped, but the graph is still not a binary tree, delete the edge
            tree.remove_edge(edge_to_flip[1], edge_to_flip[0])

    # Populate the dictionary for the remaining edges
    for edge in tree.edges():
        if edge not in flipped_edges:
            flipped_edges[edge] = False

    return tree, flipped_edges

def is_binary_tree(tree, root_node):
    # Check if the graph is a tree (acyclic and connected)
    if not nx.is_tree(tree):
        return False

    # Check if the in-degree of every node is at most 1 (binary tree condition)
    for node in tree.nodes():
        in_degree = tree.in_degree(node)
        if node != root_node and in_degree > 1:
            return False

    return True

def remove_edges_with_connectivity(dag, verbose):
    # Iteratively remove the edges that have weight 0
    edge_to_remove = None
    for edge in dag.edges(data=True):
        if edge[2]["weight"] == 0:
            temp_dag = dag.copy()  # Make a copy of the original DAG
            temp_dag.remove_edge(edge[0], edge[1])  # Remove the edge
            undirected = temp_dag.to_undirected()
            if nx.is_connected(undirected):
                edge_to_remove = (edge[0], edge[1])
                break
    if verbose:
        print("Edge to remove: ", edge_to_remove)
    if edge_to_remove:
        dag.remove_edge(*edge_to_remove)
        return remove_edges_with_connectivity(dag, verbose)
    
    return dag

def find_non_tree_edges(graph, root_node):
    non_tree_edges = []
    for edge in graph.edges():
        temp_graph = nx.DiGraph(graph)
        temp_graph.remove_edge(*edge)
        if not nx.is_weakly_connected(temp_graph) or not nx.is_tree(temp_graph) or not nx.has_path(G=temp_graph, source=edge[0], target=root_node):
            non_tree_edges.append(edge)
    return non_tree_edges
        
def clean_and_extract_edges(relationships, parent_id, verbose):
    # Build the graph
    dag = nx.DiGraph()

    for obj in relationships["children_objects"]:
        if obj["name_id"] != parent_id:
            dag.add_node(obj["name_id"])
    for obj in relationships["children_objects"]:
        if obj["name_id"] != parent_id:
            for rel in obj["placement"]["children_objects"]:
                if rel["name_id"] != parent_id:
                    dag.add_edge(obj["name_id"], rel["name_id"], weight=int(rel["is_adjacent"]))
        

    # Find cycles and remove them from the DAG
    if verbose:
        print("Simple cycles: ", list(nx.simple_cycles(dag)))
    while len(list(nx.simple_cycles(dag))) > 0:
        cycles = list(nx.simple_cycles(dag))
        dag.remove_edge(cycles[0][-1], cycles[0][0])

    if verbose:
        plt.subplot(121)
        pos_original = nx.spring_layout(dag)
        nx.draw(dag, pos_original, with_labels=True, font_weight='bold', node_size=700, arrowsize=20)
        plt.title("Original Graph")
        plt.show()

    dag = remove_edges_with_connectivity(dag, verbose)

    if dag.number_of_nodes() == 0:
        if verbose:
            print("No child nodes remain after cleaning; skipping edge extraction.")
        return [], {}

    print("Edges remaining: ", dag.edges(data=True))

    node_list = list(dag.nodes())
    if not node_list:
        if verbose:
            print("No nodes available to build binary tree; returning empty relationship set.")
        return [], {}

    try:
        binary_tree, flipped_edges = flip_edges(dag, node_list[0], verbose)
    except IndexError as err:
        if verbose:
            print(f"Binary tree extraction failed due to insufficient nodes: {err}")
        return [], {}
    if binary_tree and verbose:
        # Visualize the original graph and the obtained binary tree
        pos_original = nx.spring_layout(dag)
        pos_binary_tree = nx.spring_layout(binary_tree)

        plt.subplot(121)
        nx.draw(dag, pos_original, with_labels=True, font_weight='bold', node_size=700, arrowsize=20)
        plt.title("Original Graph")

        plt.subplot(122)
        nx.draw(binary_tree, pos_binary_tree, with_labels=True, font_weight='bold', node_size=700, arrowsize=20)
        plt.title("Binary Tree")

        plt.show()

    return binary_tree.edges(), flipped_edges

def create_empty_image_with_boxes(image_size, boxes, output_path="visualization.png", material_flow=None):
    """Create a top-down visualization scaled to fit the room footprint with colored boxes and legend.

    Args:
        image_size: Tuple of (height, width) for the output image
        boxes: List of box tuples (x, y, length, width, rotation, object_id)
        output_path: Path to save the visualization image
        material_flow: Optional material flow metadata dict with loading/unloading points
    """
    img_height, img_width = image_size

    # Reserve space for legend on the right
    legend_width = 200
    main_img_width = img_width - legend_width
    img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)  # White background

    if not boxes:
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to: {output_path}")
        return

    def effective_dimensions(length, width, rotation, is_wall=False):
        """Get X and Y extents based on rotation.
        
        For equipment (Unity objects default to facing +Y/north):
        - At 0°/180°: length along Y, width along X
        - At 90°/270°: length along X, width along Y
        
        For walls (extend along their length direction):
        - At 0°/180° (south/north wall): extend along X → X=length, Y=width
        - At 90°/270° (east/west wall): extend along Y → X=width, Y=length
        
        Returns (x_extent, y_extent).
        """
        if is_wall:
            # Walls: length is the wall's extent direction
            if np.isclose(rotation % 180, 90.0):
                return width, length  # east/west walls: vertical
            return length, width  # south/north walls: horizontal
        else:
            # Equipment: default facing +Y
            if np.isclose(rotation % 180, 90.0):
                return length, width  # length is now X-extent
            return width, length  # width is X-extent

    # Separate room elements from equipment
    room_elements = {"middle of the room", "ceiling", "south_wall", "north_wall", "west_wall", "east_wall"}
    equipment_boxes = [(x, y, w, h, r, label) for x, y, w, h, r, label in boxes if label not in room_elements]
    wall_boxes = [(x, y, w, h, r, label) for x, y, w, h, r, label in boxes if "wall" in label]
    
    # Extract object types from object IDs (remove numeric suffixes)
    def get_object_type(object_id):
        import re
        # Remove numeric suffixes like "_1", "_2", etc.
        return re.sub(r'_\d+$', '', object_id)
    
    # Group by object type 
    equipment_types = [get_object_type(label) for _, _, _, _, _, label in equipment_boxes]
    unique_types = list(set(equipment_types))
    
    # Sort object types alphabetically to group similar names together
    unique_types.sort()
    
    # Generate fixed, well-separated colors for each unique equipment type
    colors = {}
    import hashlib
    import colorsys
    
    # Predefined well-separated hues (in degrees)
    base_hues = [0, 30, 60, 120, 180, 210, 240, 270, 300, 330]  # Red, Orange, Yellow, Green, Cyan, Blue, Purple, etc.
    
    for i, obj_type in enumerate(unique_types):
        if i < len(base_hues):
            # Use predefined well-separated hues for first objects
            hue = base_hues[i] / 360.0
        else:
            # For additional objects, use hash-based approach with better spacing
            hash_value = int(hashlib.md5(obj_type.encode()).hexdigest()[:8], 16)
            hue = ((hash_value % 72) * 5) / 360.0  # 72 * 5 = 360, ensures 5-degree minimum spacing
        
        # Vary saturation and brightness for better distinction
        saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
        brightness = 0.8 + (i % 2) * 0.15  # 0.8, 0.95
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        colors[obj_type] = tuple(int(c * 255) for c in rgb)

    # Compute world bounds (in meters) for scaling
    world_min_x = float("inf")
    world_max_x = float("-inf")
    world_min_y = float("inf")
    world_max_y = float("-inf")
    min_dim = 0.05  # Minimal footprint to avoid zero-size boxes

    for x_center, y_center, w, h, r, label in boxes:
        is_wall = "wall" in label
        eff_w, eff_h = effective_dimensions(max(w, min_dim), max(h, min_dim), r, is_wall=is_wall)
        half_w = eff_w / 2.0
        half_h = eff_h / 2.0
        world_min_x = min(world_min_x, x_center - half_w)
        world_max_x = max(world_max_x, x_center + half_w)
        world_min_y = min(world_min_y, y_center - half_h)
        world_max_y = max(world_max_y, y_center + half_h)

    if not np.isfinite(world_min_x) or not np.isfinite(world_min_y):
        world_min_x, world_max_x, world_min_y, world_max_y = 0.0, 1.0, 0.0, 1.0

    world_width = max(world_max_x - world_min_x, 1e-3)
    world_height = max(world_max_y - world_min_y, 1e-3)

    # Add 5% padding around the layout (reduced from 10%)
    padding_x = max(world_width * 0.05, 0.3)
    padding_y = max(world_height * 0.05, 0.3)
    world_min_x -= padding_x
    world_max_x += padding_x
    world_min_y -= padding_y
    world_max_y += padding_y
    world_width = world_max_x - world_min_x
    world_height = world_max_y - world_min_y

    # Calculate optimal image dimensions based on room aspect ratio
    room_aspect_ratio = world_width / world_height
    
    # Determine if we should adjust the main image width to better fit the room
    if room_aspect_ratio < 1.0:  # Room is taller than it is wide
        # Reduce main image width to better match room proportions
        optimal_main_width = min(main_img_width, img_height * room_aspect_ratio)
        main_img_width = int(optimal_main_width)
        # Update legend position accordingly
        legend_width = img_width - main_img_width
    
    pixel_margin = 10
    usable_width = main_img_width - 2 * pixel_margin
    usable_height = img_height - 2 * pixel_margin
    scale = min(usable_width / world_width, usable_height / world_height)

    def world_to_pixel(x, y):
        px = int(round((x - world_min_x) * scale + pixel_margin))
        # Flip Y coordinate to make north at top, south at bottom
        py = int(round((world_max_y - y) * scale + pixel_margin))
        return px, py

    # Draw walls as gray outlines (room boundaries)
    for x_center, y_center, w, h, r, label in wall_boxes:
        eff_w, eff_h = effective_dimensions(max(w, min_dim), max(h, min_dim), r, is_wall=True)

        x_px, y_px = world_to_pixel(x_center, y_center)
        half_w_px = eff_w * scale / 2.0
        half_h_px = eff_h * scale / 2.0
        width_px, height_px = half_w_px * 2, half_h_px * 2

        x1 = int(round(x_px - width_px / 2.0))
        y1 = int(round(y_px - height_px / 2.0))
        x2 = int(round(x_px + width_px / 2.0))
        y2 = int(round(y_px + height_px / 2.0))
        
        # Draw walls as dark gray outlines
        cv2.rectangle(img, (x1, y1), (x2, y2), (64, 64, 64), 2)

    # Draw equipment with colors
    for x_center, y_center, w, h, r, label in equipment_boxes:
        eff_w, eff_h = effective_dimensions(max(w, min_dim), max(h, min_dim), r, is_wall=False)

        x_px, y_px = world_to_pixel(x_center, y_center)
        half_w_px = eff_w * scale / 2.0
        half_h_px = eff_h * scale / 2.0
        width_px, height_px = half_w_px * 2, half_h_px * 2

        x1 = int(round(x_px - width_px / 2.0))
        y1 = int(round(y_px - height_px / 2.0))
        x2 = int(round(x_px + width_px / 2.0))
        y2 = int(round(y_px + height_px / 2.0))
        
        # Use the color for this object type
        obj_type = get_object_type(label)
        color = colors[obj_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)  # Black border

    # Draw material flow points if present
    if material_flow:
        # Loading point - green circle with "L" label
        loading = material_flow.get('loading_point', {})
        loading_pos = loading.get('world_position', {})
        if loading_pos:
            lx = loading_pos.get('x', 0)
            ly = loading_pos.get('y', 0)
            lx_px, ly_px = world_to_pixel(lx, ly)
            # Draw green filled circle
            cv2.circle(img, (lx_px, ly_px), 12, (0, 200, 0), -1)  # Green fill
            cv2.circle(img, (lx_px, ly_px), 12, (0, 100, 0), 2)   # Dark green border
            # Draw "L" label
            cv2.putText(img, "L", (lx_px - 5, ly_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Unloading point - red circle with "U" label
        unloading = material_flow.get('unloading_point', {})
        unloading_pos = unloading.get('world_position', {})
        if unloading_pos:
            ux = unloading_pos.get('x', 0)
            uy = unloading_pos.get('y', 0)
            ux_px, uy_px = world_to_pixel(ux, uy)
            # Draw red filled circle
            cv2.circle(img, (ux_px, uy_px), 12, (0, 0, 200), -1)  # Red fill
            cv2.circle(img, (ux_px, uy_px), 12, (0, 0, 100), 2)   # Dark red border
            # Draw "U" label
            cv2.putText(img, "U", (ux_px - 6, uy_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw passage indicator - dashed line between unloading and loading
        if loading_pos and unloading_pos:
            # Draw a curved arrow or dashed line to indicate passage
            lx_px, ly_px = world_to_pixel(loading_pos.get('x', 0), loading_pos.get('y', 0))
            ux_px, uy_px = world_to_pixel(unloading_pos.get('x', 0), unloading_pos.get('y', 0))

            # Draw dashed line for passage (orange color)
            passage_color = (0, 165, 255)  # Orange in BGR
            # Create dashed line effect
            dist = int(np.sqrt((ux_px - lx_px)**2 + (uy_px - ly_px)**2))
            if dist > 0:
                dash_length = 8
                for i in range(0, dist, dash_length * 2):
                    start_ratio = i / dist
                    end_ratio = min((i + dash_length) / dist, 1.0)
                    start_x = int(ux_px + (lx_px - ux_px) * start_ratio)
                    start_y = int(uy_px + (ly_px - uy_px) * start_ratio)
                    end_x = int(ux_px + (lx_px - ux_px) * end_ratio)
                    end_y = int(uy_px + (ly_px - uy_px) * end_ratio)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), passage_color, 2)

        # Add material flow to legend
        colors['Loading (L)'] = (0, 200, 0)
        colors['Unloading (U)'] = (0, 0, 200)
        colors['Passage'] = (0, 165, 255)

    # Position legend optimally - try to use available vertical space efficiently
    # Use Times New Roman style font (closest available in OpenCV)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 22
    
    # Calculate optimal legend positioning
    total_legend_height = len(colors) * line_height
    available_height = img_height - 40  # Leave some margin
    
    if total_legend_height > available_height:
        # If too many items, make legend more compact
        line_height = max(20, available_height // len(colors))
        font_scale = 0.55
    
    # Center legend vertically and position close to main image
    legend_y_start = max(20, (img_height - total_legend_height) // 2)
    legend_x = main_img_width + 5
    
    for i, (obj_type, color) in enumerate(colors.items()):
        y_pos = legend_y_start + i * line_height
        
        # Draw color rectangle
        cv2.rectangle(img, (legend_x, y_pos - 8), (legend_x + 15, y_pos + 5), color, -1)
        cv2.rectangle(img, (legend_x, y_pos - 8), (legend_x + 15, y_pos + 5), (0, 0, 0), 1)
        
        # Draw text label
        cv2.putText(img, obj_type, (legend_x + 20, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")


def create_compact_visualization(boxes, output_path="visualization.pdf", material_flow=None,
                                  figsize=None, dpi=300, show_legend=True, legend_position='bottom',
                                  font_size=8, show_labels=False, title=None):
    """Create a compact, publication-ready visualization using matplotlib.

    Designed for academic papers with:
    - Vector output support (PDF, SVG, EPS)
    - Compact layout with legend below or to the side
    - Clean, professional styling
    - Automatic sizing based on room aspect ratio

    Args:
        boxes: List of box tuples (x, y, length, width, rotation, object_id)
        output_path: Path to save (supports .pdf, .svg, .eps, .png)
        material_flow: Optional material flow metadata dict
        figsize: Figure size in inches (width, height). Auto-calculated if None.
        dpi: DPI for raster output (default 300 for print quality)
        show_legend: Whether to show the legend
        legend_position: 'bottom', 'right', or 'none'
        font_size: Base font size for labels
        show_labels: Whether to show object ID labels on boxes
        title: Optional title for the figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
    from matplotlib.lines import Line2D
    import re
    import hashlib
    import colorsys

    # Use a clean style suitable for papers
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': font_size,
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

    if not boxes:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, 'No objects to visualize', ha='center', va='center')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return

    def effective_dimensions(length, width, rotation, is_wall=False):
        """Get X and Y extents based on rotation."""
        if is_wall:
            if np.isclose(rotation % 180, 90.0):
                return width, length
            return length, width
        else:
            if np.isclose(rotation % 180, 90.0):
                return length, width
            return width, length

    def get_object_type(object_id):
        return re.sub(r'_\d+$', '', object_id)

    # Separate room elements from equipment
    room_elements = {"middle of the room", "ceiling", "south_wall", "north_wall", "west_wall", "east_wall"}
    equipment_boxes = [(x, y, w, h, r, label) for x, y, w, h, r, label in boxes if label not in room_elements]
    wall_boxes = [(x, y, w, h, r, label) for x, y, w, h, r, label in boxes if "wall" in label]

    # Get unique equipment types and assign colors
    equipment_types = [get_object_type(label) for _, _, _, _, _, label in equipment_boxes]
    unique_types = sorted(set(equipment_types))

    # Fixed color mapping for consistent colors across all figures
    # Colors chosen for good visual distinction and print compatibility
    FIXED_COLORS = {
        # Conveyors - blues/cyans
        'Line_01': (0.95, 0.55, 0.25),   # Orange - entry/exit lifts
        'Line_02': (0.75, 0.75, 0.20),   # Olive/Yellow - workstations
        'Line_03': (0.45, 0.78, 0.45),   # Green - corner/turn units
        'Line_04': (0.30, 0.75, 0.85),   # Cyan - main conveyors
        'Line_05': (0.40, 0.60, 0.95),   # Blue - buffer conveyors
        'Line_06': (0.45, 0.45, 0.75),   # Indigo - inspection stations
        'Line_07': (0.70, 0.45, 0.75),   # Purple - assembly cells
        'Line_08': (0.85, 0.45, 0.65),   # Pink - auxiliary stations
        # Support equipment
        'Cart': (0.85, 0.35, 0.35),      # Red - carts
        'PalletMover': (0.55, 0.35, 0.20),  # Brown - pallet movers
        'RollerBridge': (0.60, 0.60, 0.60),  # Gray - roller bridges
        'ControlCabinet': (0.40, 0.40, 0.40),  # Dark gray - control cabinets
        'SafetyFence': (0.90, 0.70, 0.20),  # Gold - safety fencing
        'Robot': (0.20, 0.60, 0.35),     # Dark green - robots
    }

    # Fallback colors for unknown types (hash-based)
    FALLBACK_HUES = [0, 30, 60, 120, 180, 210, 240, 270, 300, 330]

    colors = {}
    fallback_idx = 0
    for obj_type in unique_types:
        if obj_type in FIXED_COLORS:
            colors[obj_type] = FIXED_COLORS[obj_type]
        else:
            # Hash-based fallback for unknown types
            hash_value = int(hashlib.md5(obj_type.encode()).hexdigest()[:8], 16)
            hue = ((hash_value % 72) * 5) / 360.0
            saturation = 0.6
            brightness = 0.85
            rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
            colors[obj_type] = rgb
            fallback_idx += 1

    # Compute world bounds
    min_dim = 0.05
    world_min_x = float("inf")
    world_max_x = float("-inf")
    world_min_y = float("inf")
    world_max_y = float("-inf")

    for x_center, y_center, w, h, r, label in boxes:
        is_wall = "wall" in label
        eff_w, eff_h = effective_dimensions(max(w, min_dim), max(h, min_dim), r, is_wall=is_wall)
        half_w, half_h = eff_w / 2.0, eff_h / 2.0
        world_min_x = min(world_min_x, x_center - half_w)
        world_max_x = max(world_max_x, x_center + half_w)
        world_min_y = min(world_min_y, y_center - half_h)
        world_max_y = max(world_max_y, y_center + half_h)

    if not np.isfinite(world_min_x):
        world_min_x, world_max_x, world_min_y, world_max_y = 0.0, 1.0, 0.0, 1.0

    # Use room bounds from walls if available (axes as walls)
    room_min_x, room_max_x = None, None
    room_min_y, room_max_y = None, None
    for x_center, y_center, w, h, r, label in wall_boxes:
        if label == "west_wall":
            room_min_x = x_center
        elif label == "east_wall":
            room_max_x = x_center
        elif label == "south_wall":
            room_min_y = y_center
        elif label == "north_wall":
            room_max_y = y_center

    # Use room bounds if all walls found, with axes as walls (start from 0)
    if all(v is not None for v in [room_min_x, room_max_x, room_min_y, room_max_y]):
        world_min_x, world_max_x = room_min_x, room_max_x
        world_min_y, world_max_y = room_min_y, room_max_y

    world_width = max(world_max_x - world_min_x, 1e-3)
    world_height = max(world_max_y - world_min_y, 1e-3)

    # Calculate figure size based on aspect ratio
    if figsize is None:
        aspect = world_width / world_height
        base_size = 3.5  # Single column width in inches for most journals
        if aspect >= 1:
            figsize = (base_size, base_size / aspect)
        else:
            figsize = (base_size * aspect, base_size)
        # Ensure minimum dimensions
        figsize = (max(figsize[0], 2.5), max(figsize[1], 2.0))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw room outline using axes as walls (no separate wall rectangles)
    # The axes spines will serve as the room walls
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#404040')

    # Draw equipment with layering: Line_04/Line_05 at bottom, then others, then Line_06/Line_07 on top
    for x_center, y_center, w, h, r, label in equipment_boxes:
        eff_w, eff_h = effective_dimensions(max(w, min_dim), max(h, min_dim), r, is_wall=False)
        obj_type = get_object_type(label)
        color = colors[obj_type]

        # Layer ordering: conveyors and Line_02 at bottom, workstations on top
        if obj_type in ['Line_04', 'Line_05', 'Line_02']:
            z = 1.5  # Straight conveyors and roller bridges at lowest layer
        elif obj_type in ['Line_06', 'Line_07']:
            z = 2.5  # Workstations on top
        else:
            z = 2  # Other equipment in middle

        rect = FancyBboxPatch(
            (x_center - eff_w/2, y_center - eff_h/2),
            eff_w, eff_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.5, edgecolor='black', facecolor=color,
            zorder=z
        )
        ax.add_patch(rect)

        # Optional: show labels on boxes
        if show_labels:
            ax.text(x_center, y_center, label, fontsize=font_size-2,
                   ha='center', va='center', zorder=3)

    # Draw material flow if present
    legend_handles = []
    if material_flow:
        loading = material_flow.get('loading_point', {})
        loading_pos = loading.get('world_position', {})
        unloading = material_flow.get('unloading_point', {})
        unloading_pos = unloading.get('world_position', {})

        if loading_pos:
            lx = loading_pos.get('x', 0)
            ly = loading_pos.get('y', 0)
            circle = Circle((lx, ly), 0.35, color='#00C853', ec='#00701A', lw=1, zorder=4)
            ax.add_patch(circle)
            ax.text(lx, ly, 'L', fontsize=font_size-2, ha='center', va='center',
                   color='white', fontweight='bold', zorder=5)

        if unloading_pos:
            ux = unloading_pos.get('x', 0)
            uy = unloading_pos.get('y', 0)
            circle = Circle((ux, uy), 0.35, color='#D50000', ec='#9B0000', lw=1, zorder=4)
            ax.add_patch(circle)
            ax.text(ux, uy, 'U', fontsize=font_size-2, ha='center', va='center',
                   color='white', fontweight='bold', zorder=5)

        # Draw passage line
        if loading_pos and unloading_pos:
            lx = loading_pos.get('x', 0)
            ly = loading_pos.get('y', 0)
            ux = unloading_pos.get('x', 0)
            uy = unloading_pos.get('y', 0)
            ax.plot([ux, lx], [uy, ly], 'o--', color='#FF6D00', lw=1.5,
                   markersize=0, zorder=3, dashes=(4, 2))

        # Add flow legend entries
        legend_handles.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00C853',
                  markersize=8, label='Loading (L)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#D50000',
                  markersize=8, label='Unloading (U)'),
            Line2D([0], [0], color='#FF6D00', linestyle='--', lw=1.5, label='Passage'),
        ])

    # Set axis properties - use exact room bounds (axes as walls, starting from 0)
    ax.set_xlim(0, world_max_x)
    ax.set_ylim(0, world_max_y)
    ax.set_aspect('equal')
    # No axis titles for cleaner look
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Use integer ticks only
    from matplotlib.ticker import MaxNLocator, FixedLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Remove 0 from both axes, add single 0 at diagonal corner
    fig.canvas.draw()
    xticks = [t for t in ax.get_xticks() if t != 0]
    yticks = [t for t in ax.get_yticks() if t != 0]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    # Add 0 label at diagonal corner position
    ax.text(-0.5, -0.5, '0', fontsize=font_size, ha='center', va='center',
            transform=ax.transData, clip_on=False)
    # Disable autoscaling to keep exact bounds
    ax.autoscale(False)

    if title:
        ax.set_title(title, fontsize=font_size + 1, fontweight='bold')

    # Create legend
    if show_legend and (colors or legend_handles):
        # Equipment type patches
        for obj_type, color in colors.items():
            legend_handles.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                                  linewidth=0.5, label=obj_type))

        if legend_position == 'bottom':
            # Horizontal legend below the plot
            ncol = min(len(legend_handles), 4)
            legend = ax.legend(handles=legend_handles, loc='upper center',
                              bbox_to_anchor=(0.5, -0.08), ncol=ncol,
                              fontsize=font_size-1, frameon=True,
                              fancybox=False, edgecolor='#CCCCCC',
                              handlelength=1.2, handletextpad=0.4,
                              columnspacing=1.0)
        elif legend_position == 'right':
            legend = ax.legend(handles=legend_handles, loc='center left',
                              bbox_to_anchor=(1.02, 0.5),
                              fontsize=font_size-1, frameon=True,
                              fancybox=False, edgecolor='#CCCCCC',
                              handlelength=1.2, handletextpad=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Compact visualization saved to: {output_path}")


def get_visualization_compact(scene_graph, output_path="visualization.pdf", **kwargs):
    """Generate a compact, publication-ready visualization from a scene graph.

    This is a wrapper around create_compact_visualization that extracts
    the necessary data from a scene graph structure.

    Args:
        scene_graph: List of scene graph objects
        output_path: Path to save visualization (supports .pdf, .svg, .eps, .png)
        **kwargs: Additional arguments passed to create_compact_visualization
            - figsize: Figure size in inches (width, height)
            - dpi: DPI for raster output (default 300)
            - show_legend: Whether to show legend (default True)
            - legend_position: 'bottom', 'right', or 'none'
            - font_size: Base font size (default 8)
            - show_labels: Show object labels on boxes (default False)
            - title: Optional title for the figure
    """
    visual_scene_graph = [
        (
            item["position"]["x"],
            item["position"]["y"],
            item["size_in_meters"]["length"],
            item["size_in_meters"]["width"],
            item["rotation"]["z_angle"],
            item["object_id"]
        )
        for item in scene_graph if "position" in item.keys()
    ]

    # Extract material flow metadata if present
    material_flow = None
    for item in scene_graph:
        if item.get("type") == "material_flow_definition":
            material_flow = item
            break

    create_compact_visualization(visual_scene_graph, output_path,
                                  material_flow=material_flow, **kwargs)


def get_visualization(scene_graph, room_priors=None, output_path="visualization.png"):
    visual_scene_graph = [
        (
            item["position"]["x"],
            item["position"]["y"],
            item["size_in_meters"]["length"],
            item["size_in_meters"]["width"],
            item["rotation"]["z_angle"],
            item["object_id"]
        )
        for item in scene_graph if "position" in item.keys()
    ]

    # Extract material flow metadata if present
    material_flow = None
    for item in scene_graph:
        if item.get("type") == "material_flow_definition":
            material_flow = item
            break

    #TODO : Adjust visualization window size according to the room size
    create_empty_image_with_boxes((800, 800), visual_scene_graph, output_path, material_flow=material_flow)

def calculate_overlap(box1, box2):
    if box1 is None or box2 is None:
        return None
    
    x_min = max(box1[0], box2[0])
    x_max = min(box1[1], box2[1])
    y_min = max(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    z_min = max(box1[4], box2[4])
    z_max = min(box1[5], box2[5])
    
    # Check if the boxes overlap with a small tolerance
    if x_min <= x_max + 1e-03 and y_min <= y_max + 1e-03 and z_min <= z_max + 1e-03:
        return (x_min, x_max, y_min, y_max, z_min, z_max)
    else:
        return None

def is_collision_3d(obj1, obj2, bbox_instead = False):
    pos1, rot1, size1 = copy(obj1['position']), copy(obj1["rotation"]["z_angle"]), copy(obj1['size_in_meters'])
    # We won't check for collisions for objects with very thin surfaces
    if is_thin_object(obj1):
        return False
    if not bbox_instead:
        pos2, rot2, size2 = copy(obj2['position']), copy(obj2["rotation"]["z_angle"]), copy(obj2['size_in_meters'])
        # We won't check for collisions for objects with very thin surfaces
        try:
            if is_thin_object(obj2):
                return False
        except:
            print(obj2)
            raise Exception
    else:
        pos2, rot2, size2 = {"x" : (obj2[1] + obj2[0]) / 2 , "y" : (obj2[3] + obj2[2]) / 2, "z" : (obj2[5] + obj2[4]) / 2}, 0.0, {"length" : (obj2[1] - obj2[0]), "width" : (obj2[3] - obj2[2]), "height" : (obj2[5] - obj2[4])}


    def swap_dimensions_if_rotated(size, rotation):
        if np.isclose(rotation, 90.0) or np.isclose(rotation, 270.0):
            size["length"], size["width"] = size["width"], size["length"]

    def get_bounds(pos, size):
        x_max = pos['x'] + size['length'] / 2
        x_min = pos['x'] - size['length'] / 2
        y_max = pos['y'] + size['width'] / 2
        y_min = pos['y'] - size['width'] / 2
        z_max = pos['z'] + size['height'] / 2
        z_min = pos['z'] - size['height'] / 2
        return x_max, x_min, y_max, y_min, z_max, z_min

    def check_overlap(min1, max1, min2, max2):
        return min1 < max2 and max1 > min2 and abs(min1 - max2) > 1e-3 and abs(max1 - min2) > 1e-3

    # Swap dimensions if needed
    swap_dimensions_if_rotated(size1, rot1)
    swap_dimensions_if_rotated(size2, rot2)

    # Get bounds for both objects
    obj1_bounds = get_bounds(pos1, size1)
    obj2_bounds = get_bounds(pos2, size2)

    # Unpack bounds
    (obj1_x_max, obj1_x_min, obj1_y_max, obj1_y_min, obj1_z_max, obj1_z_min) = obj1_bounds
    (obj2_x_max, obj2_x_min, obj2_y_max, obj2_y_min, obj2_z_max, obj2_z_min) = obj2_bounds

    # Check for overlap in each dimension
    x_check = check_overlap(obj1_x_min, obj1_x_max, obj2_x_min, obj2_x_max)
    y_check = check_overlap(obj1_y_min, obj1_y_max, obj2_y_min, obj2_y_max)
    z_check = check_overlap(obj1_z_min, obj1_z_max, obj2_z_min, obj2_z_max)

    return x_check and y_check and z_check

def get_depth(scene_graph):
    G = nx.DiGraph()
    # Create graph
    for obj in scene_graph:
        if obj["object_id"] not in G.nodes():
            G.add_node(obj["object_id"])
        obj_scene_graph = obj["placement"]
        for constraint in obj_scene_graph["room_layout_elements"]:
            if constraint["layout_element_id"] not in G.nodes():
                G.add_node(constraint["layout_element_id"])
            G.add_edge(constraint["layout_element_id"], obj["object_id"])
        for constraint in obj_scene_graph["objects_in_room"]:
            if constraint["object_id"] not in G.nodes():
                G.add_node(constraint["object_id"])
            G.add_edge(constraint["object_id"], obj["object_id"])

    # DFS Algo
    visited = set()
    prior_ids = ["south_wall", "north_wall", "east_wall", "west_wall", "middle of the room", "ceiling"]
    start_nodes = [node for node in G.nodes() if node in prior_ids]
    all_nodes_depth = {}

    def dfs(node, depth):
        visited.add(node)
        all_nodes_depth[node] = depth
        for successor in G.successors(node):
            if successor not in visited:
                dfs(successor, depth + 1)
            elif successor in all_nodes_depth and all_nodes_depth[successor] < depth + 1:
                # Skip already visited nodes with smaller depth to break out of cycles
                continue
            else:
                all_nodes_depth[successor] = depth + 1

    for start_node in start_nodes:
        dfs(start_node, 0)

    # Ensure every scene object receives a depth even if it was not connected
    for obj in scene_graph:
        obj_id = obj.get("object_id")
        if not obj_id:
            continue
        if obj_id not in all_nodes_depth:
            all_nodes_depth[obj_id] = 1

    all_nodes_depth = {k: v for k, v in all_nodes_depth.items() if k not in prior_ids}
    return all_nodes_depth

def get_possible_positions(object_id, scene_graph, room_dimensions):
    obj = next((element for element in scene_graph if element.get("object_id") == object_id), None)
    if obj is None:
        raise ValueError(f"Object '{object_id}' not found in scene graph.")
    obj_scene_graph = obj["placement"]
    rot = get_rotation(obj, scene_graph)
    obj["rotation"] = {"z_angle" : rot}

    func_map = {
        "on" : get_on_constraint,
        "under" : get_under_contraint,
        "left of" : get_left_of_constraint,
        "right of" : get_right_of_constraint,
        "in front" : get_in_front_constraint,
        "behind" : get_behind_constraint,
        "above" : get_above_constraint,
        "in the corner" : get_in_corner_constraint,
        "in the middle of" : get_on_constraint,
        "through" : get_through_constraint
    }

    constraints = obj_scene_graph["room_layout_elements"] + obj_scene_graph["objects_in_room"]

    # Check if there's a "through" constraint - it's dominant and should be the only constraint used
    through_constraint = None
    for constraint in constraints:
        if constraint.get("preposition") == "through":
            through_constraint = constraint
            break
    
    # If there's a through constraint, use ONLY that (ignore other constraints like "middle of the room")
    if through_constraint is not None:
        key = "layout_element_id" if "layout_element_id" in through_constraint.keys() else "object_id"
        obj_B = next((element for element in scene_graph if element.get("object_id") == through_constraint[key]), None)
        # If the reference object doesn't have a position yet, return empty list (defer placement)
        if obj_B is None or "position" not in obj_B.keys():
            return []
        adjacency = through_constraint.get("is_adjacent", True)
        is_on_floor = obj["is_on_the_floor"]
        return [func_map["through"](obj, obj_B, adjacency, is_on_floor, room_dimensions)]
    
    possible_positions = []
    for constraint in constraints:
        prep = constraint["preposition"]
        adjacency = constraint["is_adjacent"] if "is_adjacent" in constraint.keys() else True
        is_on_floor = obj["is_on_the_floor"]
        obj_A = obj
        key = "layout_element_id" if "layout_element_id" in constraint.keys() else "object_id"
        obj_B = next((element for element in scene_graph if element.get("object_id") == constraint[key]), None)
        if obj_B is None:
            continue
        if "position" in obj_B.keys():
            # Handle unknown prepositions gracefully
            if prep not in func_map:
                # Try to normalize common variations
                prep_normalized = prep.replace(" of", "").strip()  # "in front of" -> "in front"
                if prep_normalized in func_map:
                    prep = prep_normalized
                else:
                    print(f"⚠️ Unknown preposition '{prep}' for {object_id}, skipping constraint")
                    continue
            try:
                possible_positions.append(func_map[prep](obj_A, obj_B, adjacency, is_on_floor, room_dimensions))
            except Exception as e:
                print(f"⚠️ Error computing position for {object_id} with prep '{prep}': {e}")
                continue

    return possible_positions

def get_topological_ordering(scene_graph):
    G = nx.DiGraph()
    # Create graph
    for obj in scene_graph:
        if "placement" in obj.keys():
            if obj["object_id"] not in G.nodes():
                G.add_node(obj["object_id"])
            obj_scene_graph = obj.get("placement") or {}
            room_layout_elements = obj_scene_graph.get("room_layout_elements") or []
            objects_in_room = obj_scene_graph.get("objects_in_room") or []
            for constraint in room_layout_elements:
                if constraint["layout_element_id"] not in G.nodes():
                    G.add_node(constraint["layout_element_id"])
                G.add_edge(constraint["layout_element_id"], obj["object_id"])
            for constraint in objects_in_room:
                if constraint["object_id"] not in G.nodes():
                    G.add_node(constraint["object_id"])
                G.add_edge(constraint["object_id"], obj["object_id"])
    
    # Topological ordering (handles cycles gracefully)
    return safe_topological_sort(G)

def get_no_overlap_reason(obj, positions, valid_constraints, cluster_constraint=None, errors={}):
    overlaps = []
    candidate_positions = positions
    scene_graph_edges = valid_constraints.copy()
    if cluster_constraint is not None:
        candidate_positions = candidate_positions + [cluster_constraint]
        scene_graph_edges = scene_graph_edges + ["cluster"]
    for i, pos1 in enumerate(candidate_positions):
        for j, pos2 in enumerate(candidate_positions[i+1:]):
            if pos1 == pos2:
                continue
            overlap = calculate_overlap(pos1, pos2)
            if overlap is None:
                overlaps.append((i, i + 1 + j))
    for i, j in overlaps:
        print("No Overlap between: ", i, " ", j)
        print("Object: ", obj["object_id"])
        if scene_graph_edges[i] == "cluster":
            key_j = "layout_element_id" if "layout_element_id" in scene_graph_edges[j].keys() else "object_id"
            key = ("no_overlap", obj["object_id"], scene_graph_edges[j][key_j], scene_graph_edges[j]["preposition"], "cluster")
            errors[key] = 1 + errors.get(key, 0)
        elif scene_graph_edges[j] == "cluster":
            key_i = "layout_element_id" if "layout_element_id" in scene_graph_edges[i].keys() else "object_id"
            key = ("no_overlap", obj["object_id"], scene_graph_edges[i][key_i], scene_graph_edges[i]["preposition"], "cluster")
            errors[key] = 1 + errors.get(key, 0)
        else:
            key_i = "layout_element_id" if "layout_element_id" in scene_graph_edges[i].keys() else "object_id"
            key_j = "layout_element_id" if "layout_element_id" in scene_graph_edges[j].keys() else "object_id"
            key = ("no_overlap", obj["object_id"], scene_graph_edges[i][key_i], scene_graph_edges[i]["preposition"], scene_graph_edges[j][key_j], scene_graph_edges[j]["preposition"])
            errors[key] = 1 + errors.get(key, 0)
    return errors

CONVEYOR_PREFIXES = (
    "conveyor_",
    "conveyortable_",
    "conveyorcorner_",
    # Manufacturing line conveyors
    "line_03",
    "line_04",
    "line_05",
)


def _is_conveyor_object(object_id):
    return object_id.lower().startswith(CONVEYOR_PREFIXES)


def _conveyor_clamp(value, axis_min, axis_max):
    if axis_min > axis_max:
        axis_min, axis_max = axis_max, axis_min
    return min(max(value, axis_min), axis_max)


def _conveyor_boundary(reference, axis_min, axis_max):
    if axis_min > axis_max:
        axis_min, axis_max = axis_max, axis_min
    if axis_min == axis_max:
        return axis_min
    if reference <= axis_min:
        return axis_min
    if reference >= axis_max:
        return axis_max
    return axis_min if abs(reference - axis_min) <= abs(reference - axis_max) else axis_max


def _conveyor_dedupe(points):
    seen = set()
    ordered = []
    for point in points:
        if point is None:
            continue
        key = tuple(round(coord, 4) for coord in point)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(point)
    return ordered


def get_conveyor_candidates(overlap, constraints_with_positions, is_on_floor):
    x_min, x_max, y_min, y_max, z_min, z_max = overlap
    candidates = []
    aligned_pairs = []
    default_z = z_min if is_on_floor or np.isclose(z_min, z_max) else (z_min + z_max) / 2.0

    for constraint, obj_B in constraints_with_positions:
        parent_pos = obj_B["position"]
        prep = constraint["preposition"]

        aligned_x = _conveyor_clamp(parent_pos["x"], x_min, x_max)
        aligned_y = _conveyor_clamp(parent_pos["y"], y_min, y_max)
        aligned_pairs.append((aligned_x, aligned_y))

        if np.isclose(z_min, z_max):
            aligned_z = default_z
        else:
            aligned_z = _conveyor_boundary(parent_pos["z"], z_min, z_max)

        if prep in ["left of", "right of"]:
            x_target = _conveyor_boundary(parent_pos["x"], x_min, x_max)
            candidates.append((x_target, aligned_y, aligned_z))
        elif prep in ["in front", "behind"]:
            y_target = _conveyor_boundary(parent_pos["y"], y_min, y_max)
            candidates.append((aligned_x, y_target, aligned_z))
        else:
            candidates.append((aligned_x, aligned_y, aligned_z))

    if aligned_pairs:
        mean_x = sum(pair[0] for pair in aligned_pairs) / len(aligned_pairs)
        mean_y = sum(pair[1] for pair in aligned_pairs) / len(aligned_pairs)
        candidates.insert(
            0,
            (
                _conveyor_clamp(mean_x, x_min, x_max),
                _conveyor_clamp(mean_y, y_min, y_max),
                default_z,
            ),
        )

    span_x = abs(x_max - x_min)
    span_y = abs(y_max - y_min)
    if span_x >= span_y:
        candidates.append((x_min, _conveyor_clamp((y_min + y_max) / 2.0, y_min, y_max), default_z))
        candidates.append((x_max, _conveyor_clamp((y_min + y_max) / 2.0, y_min, y_max), default_z))
    else:
        candidates.append((_conveyor_clamp((x_min + x_max) / 2.0, x_min, x_max), y_min, default_z))
        candidates.append((_conveyor_clamp((x_min + x_max) / 2.0, x_min, x_max), y_max, default_z))

    return _conveyor_dedupe(candidates)


def place_object(obj, scene_graph, room_dimensions, errors={}, verbose=False, _placing_stack=None, _attempt_counts=None):
    # Track objects currently being placed to prevent infinite recursion from circular dependencies
    if _placing_stack is None:
        _placing_stack = set()
    if _attempt_counts is None:
        _attempt_counts = {}
    
    obj_id = obj.get("object_id")
    
    # Track attempt count for this object
    _attempt_counts[obj_id] = _attempt_counts.get(obj_id, 0) + 1
    
    # Skip if too many attempts (limit to 3 attempts per object)
    if _attempt_counts[obj_id] > 3:
        if verbose:
            print(f"Skipping {obj_id} - exceeded max attempts ({_attempt_counts[obj_id]})")
        key = ("max_attempts_exceeded", obj_id)
        errors[key] = 1 + errors.get(key, 0)
        return errors
    
    if obj_id in _placing_stack:
        if verbose:
            print(f"Skipping {obj_id} - already in placement stack (circular dependency)")
        return errors
    
    _placing_stack.add(obj_id)
    
    try:
        if verbose:
            print(f"Placing {obj_id} (stack depth: {len(_placing_stack)})")
        # Visualization is now only done at the beginning and end of backtrack
        # if verbose:
        #     get_visualization(scene_graph)
        if not any(d.get("object_id") == obj["object_id"] for d in scene_graph):
            return errors
        positions = get_possible_positions(obj["object_id"], scene_graph, room_dimensions)
        print(f"Object: {obj['object_id']}")
        print("Possible positions: ", positions)
        abs_length, abs_width = deepcopy(obj["size_in_meters"]["length"]), deepcopy(obj["size_in_meters"]["width"])
        x_neg, x_pos, y_neg, y_pos = obj["cluster"]["constraint_area"]["x_neg"], obj["cluster"]["constraint_area"]["x_pos"], obj["cluster"]["constraint_area"]["y_neg"], obj["cluster"]["constraint_area"]["y_pos"]
        
        # Unity convention: at rotation 0°, length is along Y, width is along X
        # Swap dimensions based on rotation to get correct X/Y extents
        rot = obj["rotation"]["z_angle"]
        if rot in [90.0, 270.0]:
            # Rotated: length along X, width along Y
            x_extent = abs_length
            y_extent = abs_width
        else:
            # Not rotated: width along X, length along Y
            x_extent = abs_width
            y_extent = abs_length
        
        # Calculate cluster constraint in world coordinates
        # cluster x_neg/x_pos are space needed in -X/+X directions
        # cluster y_neg/y_pos are space needed in -Y/+Y directions
        cluster_constraint = (
            x_neg + x_extent / 2,                          # x_min: object center must be this far from west wall
            room_dimensions[0] - x_pos - x_extent / 2,     # x_max: object center must be this far from east wall
            y_neg + y_extent / 2,                          # y_min: object center must be this far from south wall
            room_dimensions[1] - y_pos - y_extent / 2,     # y_max: object center must be this far from north wall
            0.0,
            room_dimensions[2] 
        )
        if verbose:
            print("Cluster constraint: ", cluster_constraint)
        if len(positions) == 0:
            # Create the error
            key = ("no_positions_found", obj["object_id"])
            errors[key] = 1 + errors.get(key, 0)
            return errors 
        obj_id = obj.get("object_id")
        children = []
        for element in scene_graph:
            placement = element.get("placement") or {}
            objects_in_room = placement.get("objects_in_room") or []
            if not isinstance(objects_in_room, list):
                continue
            # Check if this element references obj_id, but exclude "through" relationships
            # "through" objects should be placed independently, not as children
            for child_rel in objects_in_room:
                if isinstance(child_rel, dict) and child_rel.get("object_id") == obj_id:
                    # Skip if it's a "through" relationship
                    if child_rel.get("preposition") == "through":
                        continue
                    children.append(element)
                    break
        topological_sorted = get_topological_ordering(scene_graph)

        # Check condition to skip placing object
        if "position" in obj.keys():
            # Get objects that have "through" relationship with this object
            through_objects = set()
            placement = obj.get("placement", {}) or {}
            if isinstance(placement, dict):
                for rel in placement.get("objects_in_room", []) or []:
                    if isinstance(rel, dict) and rel.get("preposition") == "through":
                        through_objects.add(rel.get("object_id"))
    
            for other_obj in scene_graph:
                other_placement = other_obj.get("placement", {}) or {}
                if not isinstance(other_placement, dict):
                    continue
                for rel in other_placement.get("objects_in_room", []) or []:
                    if not isinstance(rel, dict):
                        continue
                    if rel.get("preposition") == "through" and rel.get("object_id") == obj["object_id"]:
                        through_objects.add(other_obj.get("object_id"))
            
            current_collisions = 0
            for obj_B in scene_graph:
                if obj_B == obj or "position" not in obj_B.keys():
                    continue
                # Skip collision check for "through" relationships
                if obj_B["object_id"] in through_objects:
                    continue
                if is_collision_3d(obj, obj_B):
                    current_collisions += 1
            overlap = calculate_overlap(cluster_constraint, positions[0])
            for pos in positions[1:]:
                overlap = calculate_overlap(overlap, pos)
            check_preposition = is_collision_3d(obj, overlap, bbox_instead=True) if overlap is not None else False
            check_children = any([is_collision_3d(child, item) for child in children if "position" in child.keys() for item in scene_graph if item["object_id"] != child["object_id"] and "position" in item.keys()])
            if current_collisions == 0 and check_preposition and (not check_children or len(children) == 0):
                if verbose:
                    print("Object already placed: ", obj["object_id"])
                    print("Preposition: ", check_preposition)
                return errors
        # Place object
        if len(positions) == 1:
            overlap = calculate_overlap(cluster_constraint, positions[0])            
        else:
            overlap = calculate_overlap(cluster_constraint, positions[0])
            for pos in positions[1:]:
                overlap = calculate_overlap(overlap, pos)
        
        # Find what causes the no overlap
        if overlap is None:
            # Try relaxing cluster first: intersect only the placement constraints
            overlap_constraints_only = positions[0]
            for pos in positions[1:]:
                overlap_constraints_only = calculate_overlap(overlap_constraints_only, pos)
            if overlap_constraints_only is not None:
                overlap = overlap_constraints_only

            # If still none, let conveyors fall back to constraint overlap (even if loose)
            if overlap is None and _is_conveyor_object(obj["object_id"]):
                overlap = positions[0]
                for pos in positions[1:]:
                    overlap = calculate_overlap(overlap, pos)

            if overlap is None:
                if verbose:
                    print("No overlap found for object: ", obj["object_id"])
                
                # Reconstruct valid constraints that correspond to positions
                # Apply "through is dominant" logic here too
                valid_constraints = []
                constraints = obj["placement"]["room_layout_elements"] + obj["placement"]["objects_in_room"]
                
                # Check for through constraint first
                through_constraint = None
                for constraint in constraints:
                    if constraint.get("preposition") == "through":
                        through_constraint = constraint
                        break
                
                if through_constraint is not None:
                    # Only use through constraint
                    key = "layout_element_id" if "layout_element_id" in through_constraint.keys() else "object_id"
                    obj_B = next((element for element in scene_graph if element.get("object_id") == through_constraint[key]), None)
                    if obj_B and "position" in obj_B.keys():
                        valid_constraints = [through_constraint]
                else:
                    # No through constraint, use all valid constraints
                    for constraint in constraints:
                        key = "layout_element_id" if "layout_element_id" in constraint.keys() else "object_id"
                        obj_B = next((element for element in scene_graph if element.get("object_id") == constraint[key]), None)
                        if obj_B and "position" in obj_B.keys():
                            valid_constraints.append(constraint)

                errors = get_no_overlap_reason(obj, positions, valid_constraints, cluster_constraint, errors)
                return errors
        
        conveyor_candidates = []
        if _is_conveyor_object(obj["object_id"]):
            constraints_with_positions = []
            for constraint in obj["placement"]["room_layout_elements"] + obj["placement"]["objects_in_room"]:
                key = "layout_element_id" if "layout_element_id" in constraint.keys() else "object_id"
                obj_B = next((element for element in scene_graph if element.get("object_id") == constraint[key]), None)
                if obj_B and "position" in obj_B.keys():
                    constraints_with_positions.append((constraint, obj_B))
            if constraints_with_positions:
                conveyor_candidates = get_conveyor_candidates(overlap, constraints_with_positions, obj["is_on_the_floor"])

        counter = 0
        candidate_queue = conveyor_candidates.copy()
        tried_positions = set()
        while True:
            counter += 1
            if counter > 50:
                if verbose:
                    print("No positions found for object: ", obj["object_id"])
                    print(overlap)
                obj.pop("position", None)  # Safely remove position if exists
                # If there wasn't any errors, it means that the object was colliding with other objects
                if not errors:
                    key = ("no_positions_found", obj["object_id"])
                    errors[key] = 1 + errors.get(key, 0)
                return errors
            # Check if this object has a "through" placement constraint
            has_through_constraint = False
            obj_placement = obj.get("placement", {}) or {}
            if isinstance(obj_placement, dict):
                for rel in obj_placement.get("objects_in_room", []) or []:
                    if isinstance(rel, dict) and rel.get("preposition") == "through":
                        has_through_constraint = True
                        break
            
            if candidate_queue:
                x, y, z = candidate_queue.pop(0)
            else:
                if is_point_bbox(overlap):
                    # For "through" constraints, accept the point position immediately
                    if has_through_constraint:
                        x, y, z = overlap[0], overlap[2], overlap[4]
                        obj["position"] = {"x": x, "y": y, "z": z}
                        if verbose:
                            print(f"Placed {obj['object_id']} at through position: {obj['position']}")
                        return errors  # Accept this position without collision checks
                    counter = 50
                    x, y, z = overlap[0], overlap[2], overlap[4]
                else:
                    x = random.uniform(overlap[0], overlap[1])
                    y = random.uniform(overlap[2], overlap[3])
                    z = random.uniform(overlap[4], overlap[5])

            candidate_key = (round(x, 4), round(y, 4), round(z, 4))
            if candidate_key in tried_positions:
                continue
            tried_positions.add(candidate_key)
            obj["position"] = {
                "x" : x,
                "y" : y,
                "z" : z
            }
            if verbose:
                print("Assigned position: ", obj["position"], " to object: ", obj["object_id"])
            flag = False
            # Get objects that this object passes "through" - these should be allowed to overlap
            through_objects = set()
            placement = obj.get("placement", {}) or {}
            if isinstance(placement, dict):
                for rel in placement.get("objects_in_room", []) or []:
                    if isinstance(rel, dict) and rel.get("preposition") == "through":
                        through_objects.add(rel.get("object_id"))

            # Also check if obj_B is "through" this object
            for obj_B in scene_graph:
                if obj_B == obj or "position" not in obj_B.keys():
                    continue
                obj_B_placement = obj_B.get("placement", {}) or {}
                if not isinstance(obj_B_placement, dict):
                    continue
                for rel in obj_B_placement.get("objects_in_room", []) or []:
                    if not isinstance(rel, dict):
                        continue
                    if rel.get("preposition") == "through" and rel.get("object_id") == obj["object_id"]:
                        through_objects.add(obj_B.get("object_id"))
            
            for obj_B in scene_graph:
                if obj_B == obj or "position" not in obj_B.keys():
                    continue
                # Skip collision check for "through" relationships
                if obj_B["object_id"] in through_objects:
                    continue
                if is_collision_3d(obj, obj_B):
                    flag = True
                    break
            if flag:
                continue
            
            child_flag = False
            # Topologically sort children
            children = [x for topo in topological_sorted for x in children if topo == x["object_id"]]
            # print("Sorted children: ", [x["object_id"] for x in children])
            for child in children:
                if verbose:
                    print(obj["object_id"], " placing child: ", child["object_id"])
                errors_child = place_object(child, scene_graph, room_dimensions, errors={}, _placing_stack=_placing_stack, _attempt_counts=_attempt_counts)
                if verbose:
                    print("Errors child: ", errors_child)
                if errors_child:
                    child_flag = True
                    # Add the errors to the main errors
                    for key in errors_child.keys():
                        if key in errors.keys():
                            errors[key] += errors_child[key]
                        else:
                            errors[key] = errors_child[key]
                    break
            if verbose:
                print("Child flag: ", child_flag, " for object: ", obj["object_id"])
            if child_flag:
                # Delete the position key in children
                for child in children:
                    if "position" in child.keys():
                        del child["position"]
                continue
            if verbose:
                print("Object placed: ", obj["object_id"])
            errors = {}
            break 
        return errors
    finally:
        _placing_stack.discard(obj_id)
