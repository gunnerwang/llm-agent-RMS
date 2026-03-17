from copy import copy

def get_on_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is on obj_B
    """
    size_A = copy(obj_A["size_in_meters"])
    
    pos_B = obj_B["position"]
    size_B = copy(obj_B["size_in_meters"])

    # Room layout elements already encode their orientation in their position;
    # swapping length/width for their rotation (e.g., east_wall rotated 270°)
    # zeroes out the meaningful dimension because their width is 0.
    is_layout_element = obj_B.get("object_id") in ["south_wall", "north_wall", "east_wall", "west_wall", "ceiling", "middle of the room"]

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]
    if obj_B["rotation"]["z_angle"] in [90.0, 270.0] and not is_layout_element:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    # Extents in world axes after applying rotation swap above
    extent_x = size_A["width"]
    extent_y = size_A["length"]

    if obj_B["object_id"] not in ["south_wall", "north_wall", "east_wall", "west_wall", "ceiling"]:
        z_min = pos_B["z"] + size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2 
        z_max = pos_B["z"] + size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2 
        x_min = pos_B["x"] - size_B["length"] / 2 + size_A["length"] / 2
        x_max = pos_B["x"] + size_B["length"] / 2 - size_A["length"] / 2
        y_min = pos_B["y"] - size_B["width"] / 2 + size_A["width"] / 2
        y_max = pos_B["y"] + size_B["width"] / 2 - size_A["width"] / 2
    elif obj_B["object_id"] == "ceiling":
        z_min = pos_B["z"] - size_B["height"] / 2 - size_A["height"] / 2
        z_max = pos_B["z"] - size_B["height"] / 2 - size_A["height"] / 2
        x_min = pos_B["x"] - size_B["length"] / 2 + size_A["length"] / 2
        x_max = pos_B["x"] + size_B["length"] / 2 - size_A["length"] / 2
        y_min = pos_B["y"] - size_B["width"] / 2 + size_A["width"] / 2
        y_max = pos_B["y"] + size_B["width"] / 2 - size_A["width"] / 2
    elif obj_B["object_id"] == "middle of the room":
        z_min = pos_B["z"] + size_B["height"] / 2 + size_A["height"] / 2
        z_max = pos_B["z"] + size_B["height"] / 2 + size_A["height"] / 2
        x_min = pos_B["x"] - size_B["length"] / 2 + size_A["length"] / 2
        x_max = pos_B["x"] + size_B["length"] / 2 - size_A["length"] / 2
        y_min = pos_B["y"] - size_B["width"] / 2 + size_A["width"] / 2
        y_max = pos_B["y"] + size_B["width"] / 2 - size_A["width"] / 2
    else:
        # Walls: place object flush to wall, using the extent normal to that wall.
        z_min = size_A["height"] / 2 if is_on_floor else pos_B["z"] - size_B["height"] / 2 + size_A["height"] / 2
        z_max = size_A["height"] / 2 if is_on_floor else pos_B["z"] + size_B["height"] / 2 - size_A["height"] / 2

        if obj_B["object_id"] == "south_wall":
            y_min = y_max = extent_y / 2
            x_min = extent_x / 2
            x_max = room_dimensions[0] - extent_x / 2
        elif obj_B["object_id"] == "north_wall":
            y_min = y_max = room_dimensions[1] - extent_y / 2
            x_min = extent_x / 2
            x_max = room_dimensions[0] - extent_x / 2
        elif obj_B["object_id"] == "west_wall":
            x_min = x_max = extent_x / 2
            y_min = extent_y / 2
            y_max = room_dimensions[1] - extent_y / 2
        elif obj_B["object_id"] == "east_wall":
            x_min = x_max = room_dimensions[0] - extent_x / 2
            y_min = extent_y / 2
            y_max = room_dimensions[1] - extent_y / 2

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    x_max = max(size_A["length"] / 2, min(x_max, room_dimensions[0] - size_A["length"] / 2))
    x_min = max(x_min, 0.0 + size_A["length"] / 2)
    y_max = max(size_A["width"] / 2, min(y_max, room_dimensions[1] - size_A["width"] / 2))
    y_min = max(y_min, 0.0 + size_A["width"] / 2)
    z_max = max(size_A["height"] / 2, min(z_max, room_dimensions[2] - size_A["height"] / 2))
    z_min = max(z_min, 0.0 + size_A["height"] / 2)

    return (x_min, x_max, y_min, y_max, z_min, z_max)

def get_under_contraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is under obj_B
    """

    size_A = copy(obj_A["size_in_meters"])
    
    pos_B = obj_B["position"]
    size_B = copy(obj_B["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]
    if obj_B["rotation"]["z_angle"] in [90.0, 270.0]:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    z_min = size_A["height"] / 2
    z_max = pos_B["z"] - size_B["height"] / 2 - size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    x_min = pos_B["x"] - size_B["length"] / 2 - size_A["length"] / 2
    x_max = pos_B["x"] + size_B["length"] / 2 + size_A["length"] / 2
    y_min = pos_B["y"] - size_B["width"] / 2 - size_A["width"] / 2
    y_max = pos_B["y"] + size_B["width"] / 2 + size_A["width"] / 2
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    
    x_max = max(size_A["length"] / 2, min(x_max, room_dimensions[0] - size_A["length"] / 2))
    x_min = max(x_min, 0.0 + size_A["length"] / 2)
    y_max = max(size_A["width"] / 2, min(y_max, room_dimensions[1] - size_A["width"] / 2))
    y_min = max(y_min, 0.0 + size_A["width"] / 2)
    z_max = max(size_A["height"] / 2, min(z_max, room_dimensions[2] - size_A["height"] / 2))
    z_min = max(z_min, 0.0 + size_A["height"] / 2)
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def get_left_of_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is left of obj_B.
    
    Unity convention: "left" is relative to obj_B's facing direction.
    - If obj_B faces north (0°), left is -X (west)
    - If obj_B faces east (90°), left is +Y (north)
    - If obj_B faces south (180°), left is +X (east)
    - If obj_B faces west (270°), left is -Y (south)
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]

    if obj_B["rotation"]["z_angle"] in [90.0, 270.0]:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    z_min = obj_B["position"]["z"] - size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    z_max = room_dimensions[2] - size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2

    if obj_B["rotation"]["z_angle"] == 0.0:
        # Facing north: left is -X (west)
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["width"] / 2 if is_adjacent else size_A["width"] / 2
        x_max = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["width"] / 2
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 90.0:
        # Facing east: left is +Y (north)
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["length"] / 2 
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["length"] / 2 if is_adjacent else room_dimensions[1] - size_A["length"] / 2
    elif obj_B["rotation"]["z_angle"] == 180.0:
        # Facing south: left is +X (east)
        x_min = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["width"] / 2 
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["width"] / 2 if is_adjacent else room_dimensions[0] - size_A["width"] / 2
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 270.0:
        # Facing west: left is -Y (south)
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["length"] / 2 if is_adjacent else size_A["length"] / 2
        y_max = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["length"] / 2 
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    x_max = max(size_A["width"] / 2, min(x_max, room_dimensions[0] - size_A["width"] / 2))
    x_min = max(x_min, 0.0 + size_A["width"] / 2)
    y_max = max(size_A["length"] / 2, min(y_max, room_dimensions[1] - size_A["length"] / 2))
    y_min = max(y_min, 0.0 + size_A["length"] / 2)

    
    return (x_min, x_max, y_min, y_max, z_min, z_max)           


def get_right_of_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is right of obj_B.
    
    Unity convention: "right" is relative to obj_B's facing direction.
    - If obj_B faces north (0°), right is +X (east)
    - If obj_B faces east (90°), right is -Y (south)
    - If obj_B faces south (180°), right is -X (west)
    - If obj_B faces west (270°), right is +Y (north)
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]

    if obj_B["rotation"]["z_angle"] in [90.0, 270.0]:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    z_min = obj_B["position"]["z"] - size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    z_max = room_dimensions[2] - size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2

    if obj_B["rotation"]["z_angle"] == 0.0:
        # Facing north: right is +X (east)
        x_min = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["width"] / 2
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["width"] / 2 if is_adjacent else room_dimensions[0] - size_A["width"] / 2
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 90.0:
        # Facing east: right is -Y (south)
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["length"] / 2 if is_adjacent else size_A["length"] / 2
        y_max = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["length"] / 2 
    elif obj_B["rotation"]["z_angle"] == 180.0:
        # Facing south: right is -X (west)
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["width"] / 2 if is_adjacent else size_A["width"] / 2
        x_max = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["width"] / 2 
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 270.0:
        # Facing west: right is +Y (north)
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["length"] / 2 
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["length"] / 2 if is_adjacent else room_dimensions[1] - size_A["length"] / 2
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    x_max = max(size_A["width"] / 2, min(x_max, room_dimensions[0] - size_A["width"] / 2))
    x_min = max(x_min, 0.0 + size_A["width"] / 2)
    y_max = max(size_A["length"] / 2, min(y_max, room_dimensions[1] - size_A["length"] / 2))
    y_min = max(y_min, 0.0 + size_A["length"] / 2)
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)

def get_in_front_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is in front of obj_B.
    
    Unity convention: objects default to facing +Y (north).
    - rotation 0°: facing north (+Y), length along Y, width along X
    - rotation 90°: facing east (+X), length along X, width along Y
    - rotation 180°: facing south (-Y), length along Y, width along X
    - rotation 270°: facing west (-X), length along X, width along Y
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    # Swap obj_A dimensions if rotated 90/270 (so size_A always represents X/Y extents correctly)
    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]

    # Swap obj_B dimensions based on its rotation for correct X/Y extents
    if obj_B["rotation"]["z_angle"] in [90.0, 270.0]:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    z_min = obj_B["position"]["z"] - size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    z_max = room_dimensions[2] - size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2

    # After swapping, size_B["width"] is X-extent, size_B["length"] is Y-extent for 0°/180°
    # For 90°/270°, after swap: size_B["length"] is X-extent, size_B["width"] is Y-extent
    
    if obj_B["rotation"]["z_angle"] == 0.0:
        # Facing north: "in front" means +Y direction
        # X range: centered on obj_B, extent is obj_B's width (X-extent at 0°)
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        # Y: place in front (+Y) of obj_B
        y_min = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["length"] / 2 
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["length"] / 2 if is_adjacent else room_dimensions[1] - size_A["length"] / 2
    elif obj_B["rotation"]["z_angle"] == 90.0:
        # Facing east: "in front" means +X direction
        x_min = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["width"] / 2 
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["width"] / 2 if is_adjacent else room_dimensions[0] - size_A["width"] / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 180.0:
        # Facing south: "in front" means -Y direction
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["length"] / 2 if is_adjacent else size_A["length"] / 2
        y_max = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["length"] / 2 
    elif obj_B["rotation"]["z_angle"] == 270.0:
        # Facing west: "in front" means -X direction
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["width"] / 2 if is_adjacent else size_A["width"] / 2
        x_max = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["width"] / 2 
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    # Clamp to room boundaries using obj_A's effective dimensions
    x_max = max(size_A["width"] / 2, min(x_max, room_dimensions[0] - size_A["width"] / 2))
    x_min = max(x_min, 0.0 + size_A["width"] / 2)
    y_max = max(size_A["length"] / 2, min(y_max, room_dimensions[1] - size_A["length"] / 2))
    y_min = max(y_min, 0.0 + size_A["length"] / 2)
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)
    
def get_behind_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is behind obj_B.
    
    Unity convention: objects default to facing +Y (north).
    "behind" is the opposite direction of "in front".
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    # Swap obj_A dimensions if rotated 90/270
    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]

    # Swap obj_B dimensions based on its rotation
    if obj_B["rotation"]["z_angle"] in [90.0, 270.0]:
        size_B["length"], size_B["width"] = size_B["width"], size_B["length"]

    z_min = obj_B["position"]["z"] - size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    z_max = room_dimensions[2] - size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2

    if obj_B["rotation"]["z_angle"] == 0.0:
        # Facing north: "behind" means -Y direction
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["length"] / 2 if is_adjacent else size_A["length"] / 2
        y_max = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["length"] / 2 
    elif obj_B["rotation"]["z_angle"] == 90.0:
        # Facing east: "behind" means -X direction
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["width"] / 2 if is_adjacent else size_A["width"] / 2
        x_max = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["width"] / 2 
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    elif obj_B["rotation"]["z_angle"] == 180.0:
        # Facing south: "behind" means +Y direction
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 + ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 - ((is_adjacent * size_A["width"]) - (not is_adjacent * size_A["width"])) / 2
        y_min = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["length"] / 2 
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["length"] / 2 if is_adjacent else room_dimensions[1] - size_A["length"] / 2
    elif obj_B["rotation"]["z_angle"] == 270.0:
        # Facing west: "behind" means +X direction
        x_min = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["width"] / 2 
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["width"] / 2 if is_adjacent else room_dimensions[0] - size_A["width"] / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 + ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 - ((is_adjacent * size_A["length"]) - (not is_adjacent * size_A["length"])) / 2
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    # Clamp to room boundaries
    x_max = max(size_A["width"] / 2, min(x_max, room_dimensions[0] - size_A["width"] / 2))
    x_min = max(x_min, 0.0 + size_A["width"] / 2)
    y_max = max(size_A["length"] / 2, min(y_max, room_dimensions[1] - size_A["length"] / 2))
    y_min = max(y_min, 0.0 + size_A["length"] / 2)

    return (x_min, x_max, y_min, y_max, z_min, z_max)

def get_above_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is above obj_B
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]


    z_min = obj_B["position"]["z"] + size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2
    z_max = room_dimensions[2] if not is_on_floor else size_A["height"] / 2

    if obj_B["rotation"]["z_angle"] == 0.0:
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["length"] / 2
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["length"] / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["width"] / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["width"] / 2
    elif obj_B["rotation"]["z_angle"] == 90.0:
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["length"] / 2 
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["length"] / 2 
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["width"] / 2 
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["width"] / 2
    elif obj_B["rotation"]["z_angle"] == 180.0:
        x_min = obj_B["position"]["x"] - size_B["length"] / 2 - size_A["length"] / 2 
        x_max = obj_B["position"]["x"] + size_B["length"] / 2 + size_A["length"] / 2
        y_min = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["width"] / 2
        y_max = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["width"] / 2
    elif obj_B["rotation"]["z_angle"] == 270.0:
        x_min = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["length"] / 2 
        x_max = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["length"] / 2 
        y_min = obj_B["position"]["y"] - size_B["length"] / 2 - size_A["width"] / 2 
        y_max = obj_B["position"]["y"] + size_B["length"] / 2 + size_A["width"] / 2
    
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    x_max = max(size_A["length"] / 2, min(x_max, room_dimensions[0] - size_A["length"] / 2))
    x_min = max(x_min, 0.0 + size_A["length"] / 2)
    y_max = max(size_A["width"] / 2, min(y_max, room_dimensions[1] - size_A["width"] / 2))
    y_min = max(y_min, 0.0 + size_A["width"] / 2)
    z_min = max(size_A["height"] / 2, min(z_min, room_dimensions[2] - size_A["height"] / 2))
    z_min = max(z_min, 0.0 + size_A["height"] / 2)

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def get_through_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A passes THROUGH obj_B (e.g., robotic cell centered on conveyor, or vice versa).
    obj_A is centered exactly on obj_B's position (point constraint).
    """
    size_A = copy(obj_A["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]

    # obj_A is centered exactly on obj_B
    pos_B = obj_B["position"]
    
    # Z position: on floor for floor objects
    z_val = size_A["height"] / 2 if is_on_floor else pos_B["z"]
    
    # X, Y: exactly at obj_B's center (point constraint - no tolerance needed)
    # This ensures the object is placed precisely at the passthrough location
    x_val = pos_B["x"]
    y_val = pos_B["y"]

    # Return as point constraint (min == max)
    return (x_val, x_val, y_val, y_val, z_val, z_val)


def get_in_corner_constraint(obj_A, obj_B, is_adjacent, is_on_floor, room_dimensions):
    """
    obj_A is in the corner of obj_B
    """
    size_A = copy(obj_A["size_in_meters"])
    size_B = copy(obj_B["size_in_meters"])

    if obj_A["rotation"]["z_angle"] in [90.0, 270.0]:
        size_A["length"], size_A["width"] = size_A["width"], size_A["length"]


    z_min = obj_B["position"]["z"] - size_B["height"] / 2 + size_A["height"] / 2 if not is_on_floor else size_A["height"] / 2

    if obj_B["rotation"]["z_angle"] == 0.0:
        x_1 = obj_B["position"]["x"] - size_B["length"] / 2 + size_A["length"] / 2 
        x_2 = obj_B["position"]["x"] + size_B["length"] / 2 - size_A["length"] / 2 
        y_1 = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["width"] / 2 
        y_2 = obj_B["position"]["y"] + size_B["width"] / 2 + size_A["width"] / 2 
    elif obj_B["rotation"]["z_angle"] == 90.0:
        x_1 = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["length"] / 2 
        x_2 = obj_B["position"]["x"] + size_B["width"] / 2 + size_A["length"] / 2 
        y_1 = obj_B["position"]["y"] - size_B["length"] / 2 + size_A["width"] / 2 
        y_2 = obj_B["position"]["y"] + size_B["length"] / 2 - size_A["width"] / 2
    elif obj_B["rotation"]["z_angle"] == 180.0:
        x_1 = obj_B["position"]["x"] - size_B["length"] / 2 + size_A["length"] / 2 
        x_2 = obj_B["position"]["x"] + size_B["length"] / 2 - size_A["length"] / 2 
        y_1 = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["width"] / 2 
        y_2 = obj_B["position"]["y"] - size_B["width"] / 2 - size_A["width"] / 2
    elif obj_B["rotation"]["z_angle"] == 270.0:
        x_1 = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["length"] / 2 
        x_2 = obj_B["position"]["x"] - size_B["width"] / 2 - size_A["length"] / 2 
        y_1 = obj_B["position"]["y"] - size_B["length"] / 2 + size_A["width"] / 2 
        y_2 = obj_B["position"]["y"] + size_B["length"] / 2 - size_A["width"] / 2
    
    return (x_1, x_2, y_1, y_2, z_min, z_min)
