import sys
import os

# Fix NumPy compatibility issue for older Blender versions BEFORE importing bpy
try:
    import numpy as np
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'int'):
        np.int = int  
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'str'):
        np.str = str
except ImportError:
    pass

import bpy
import json
import math

object_name = 'Cube'
object_to_delete = bpy.data.objects.get(object_name)

# Check if the object exists before trying to delete it
if object_to_delete is not None:
    bpy.data.objects.remove(object_to_delete, do_unlink=True)

def import_glb(file_path, object_name):
    bpy.ops.import_scene.gltf(filepath=file_path)
    imported_object = bpy.context.view_layer.objects.active
    if imported_object is not None:
        imported_object.name = object_name

def create_room(width, depth, height):
    # Create floor
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(width/2, depth/2, 0))
    floor = bpy.context.active_object
    floor.scale = (width/2, depth/2, 1)
    floor.name = "Floor"
    
    # Walls are commented out for better visibility
    # Uncomment any wall below if needed
    
    # # Back wall
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(width/2, depth, height/2))
    # back_wall = bpy.context.active_object
    # back_wall.scale = (width/2, height/2, 1)
    # back_wall.rotation_euler = (math.pi/2, 0, 0)
    # back_wall.name = "BackWall"
    
    # # Left wall
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, depth/2, height/2))
    # left_wall = bpy.context.active_object
    # left_wall.scale = (depth/2, height/2, 1)
    # left_wall.rotation_euler = (math.pi/2, 0, math.pi/2)
    # left_wall.name = "LeftWall"
    
    # # Right wall
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(width, depth/2, height/2))
    # right_wall = bpy.context.active_object
    # right_wall.scale = (depth/2, height/2, 1)
    # right_wall.rotation_euler = (math.pi/2, 0, math.pi/2)
    # right_wall.name = "RightWall"
    
    # # Front wall
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(width/2, 0, height/2))
    # front_wall = bpy.context.active_object
    # front_wall.scale = (width/2, height/2, 1)
    # front_wall.rotation_euler = (math.pi/2, 0, 0)
    # front_wall.name = "FrontWall"
    
    # # Ceiling
    # bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(width/2, depth/2, height))
    # ceiling = bpy.context.active_object
    # ceiling.scale = (width/2, depth/2, 1)
    # ceiling.name = "Ceiling"

def find_glb_files(directory):
    glb_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".glb"):
                key = file.split(".")[0]
                if key not in glb_files:
                    glb_files[key] = os.path.join(root, file)
    return glb_files

def get_highest_parent_objects():
    highest_parent_objects = []

    for obj in bpy.data.objects:
        # Check if the object has no parent
        if obj.parent is None:
            highest_parent_objects.append(obj)
    return highest_parent_objects

def delete_empty_objects():
    # Iterate through all objects in the scene
    for obj in bpy.context.scene.objects:
        # Check if the object is empty (has no geometry)
        print(obj.name, obj.type)
        if obj.type == 'EMPTY':
            bpy.context.view_layer.objects.active = obj
            bpy.data.objects.remove(obj)

def select_meshes_under_empty(empty_object_name):
    # Get the empty object
    empty_object = bpy.data.objects.get(empty_object_name)
    print(empty_object is not None)
    if empty_object is not None and empty_object.type == 'EMPTY':
        # Iterate through the children of the empty object
        for child in empty_object.children:
            # Check if the child is a mesh
            if child.type == 'MESH':
                # Select the mesh
                child.select_set(True)
                bpy.context.view_layer.objects.active = child
            else:
                select_meshes_under_empty(child.name)

def rescale_object(obj, scale):
    # Ensure the object has a mesh data
    if obj.type == 'MESH':
        bbox_dimensions = obj.dimensions
        scale_factors = (
                         scale["length"] / bbox_dimensions.x, 
                         scale["width"] / bbox_dimensions.y, 
                         scale["height"] / bbox_dimensions.z
                        )
        obj.scale = scale_factors


objects_in_room = {}
file_path = "scenes/scene_graph.json"
with open(file_path, 'r') as file:
    data = json.load(file)
    for item in data:
        if item["object_id"] not in ["south_wall", "north_wall", "east_wall", "west_wall", "middle of the room", "ceiling"]:
            objects_in_room[item["object_id"]] = item

directory_path = os.path.join(os.getcwd(), "Assets")
glb_file_paths = find_glb_files(directory_path)

for item_id, object_in_room in objects_in_room.items():
    glb_file_path = os.path.join(directory_path, glb_file_paths[item_id])
    import_glb(glb_file_path, item_id)

parents = get_highest_parent_objects()
empty_parents = [parent for parent in parents if parent.type == "EMPTY"]
print(empty_parents)

for empty_parent in empty_parents:
    bpy.ops.object.select_all(action='DESELECT')
    select_meshes_under_empty(empty_parent.name)
    
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    joined_object = bpy.context.view_layer.objects.active
    if joined_object is not None:
        joined_object.name = empty_parent.name + "-joined"

bpy.context.view_layer.objects.active = None

MSH_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
for OBJS in MSH_OBJS:
    bpy.context.view_layer.objects.active = OBJS
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    OBJS.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.objects.active = OBJS
    OBJS.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

MSH_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
for OBJS in MSH_OBJS:
    item = objects_in_room[OBJS.name.split("-")[0]]
    object_position = (item["position"]["x"], item["position"]["y"], item["position"]["z"])  # X, Y, and Z coordinates
    object_rotation_z = (item["rotation"]["z_angle"] / 180.0) * math.pi + math.pi # Rotation angles in radians around the X, Y, and Z axes
    
    # Set position directly
    OBJS.location = object_position
    # Set rotation directly (more reliable in background mode than using operators)
    OBJS.rotation_euler[2] = object_rotation_z  # Z-axis rotation
    rescale_object(OBJS, item["size_in_meters"])

bpy.ops.object.select_all(action='DESELECT')
delete_empty_objects()

# TODO: Generate the room with the room dimensions
create_room(4.0, 4.0, 2.5)

# Save the scene
output_file = "scene_output.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_file)
print(f"Scene saved to: {output_file}")