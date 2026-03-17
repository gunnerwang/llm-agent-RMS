"""
Function tools for agents to manipulate manufacturing layouts.
These tools allow agents to build layouts incrementally instead of outputting large JSON.
"""

import json
from typing import Dict, List, Optional, Any
from catalog.equipment_catalog import get_equipment_info, get_equipment_list

class LayoutBuilder:
    """Manages incremental layout construction via function tools"""

    def __init__(self, existing_layout=None):
        """Initialize with optional existing layout"""
        if existing_layout:
            if isinstance(existing_layout, dict):
                self.objects = existing_layout.get("objects_in_room", [])
            elif isinstance(existing_layout, list):
                self.objects = existing_layout
            else:
                self.objects = []
        else:
            self.objects = []

        # Track object IDs for validation - initialize from existing objects
        self._object_id_counter = {}
        self._initialize_counters_from_existing()

    def _initialize_counters_from_existing(self):
        """Initialize ID counters based on existing objects to avoid conflicts"""
        import re
        for obj in self.objects:
            obj_id = obj.get("object_id", "")
            if not obj_id:
                continue
            # Parse object_id like "Line_04_2" -> base="Line_04", num=2
            match = re.match(r"^(.+?)_(\d+)$", obj_id)
            if match:
                base_name = match.group(1)
                instance_num = int(match.group(2))
                # Track the highest instance number for each base name
                if base_name not in self._object_id_counter:
                    self._object_id_counter[base_name] = instance_num
                else:
                    self._object_id_counter[base_name] = max(
                        self._object_id_counter[base_name], instance_num
                    )

    def _generate_object_id(self, equipment_name: str) -> str:
        """Generate unique object ID with instance suffix

        Ensures no conflicts with existing object IDs by using the next
        available instance number.
        """
        if equipment_name not in self._object_id_counter:
            self._object_id_counter[equipment_name] = 0
        self._object_id_counter[equipment_name] += 1
        return f"{equipment_name}_{self._object_id_counter[equipment_name]}"

    def get_current_layout(self) -> Dict[str, Any]:
        """Get summary of current layout"""
        summary = {
            "total_objects": len(self.objects),
            "equipment_list": [obj.get("object_id") for obj in self.objects],
            "equipment_by_type": {}
        }

        # Group by equipment type
        for obj in self.objects:
            obj_id = obj.get("object_id", "")
            # Extract base name (before instance number)
            base_name = obj_id.rsplit("_", 1)[0] if "_" in obj_id else obj_id
            if base_name not in summary["equipment_by_type"]:
                summary["equipment_by_type"][base_name] = []
            summary["equipment_by_type"][base_name].append(obj_id)

        return summary

    def get_equipment_details(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get details of specific equipment"""
        for obj in self.objects:
            if obj.get("object_id") == object_id:
                return obj
        return None

    def list_available_equipment(self) -> List[str]:
        """List all equipment available in catalog"""
        return get_equipment_list()

    def add_equipment(
        self,
        equipment_name: str,
        facing: str = "south_wall",
        placement_room_layout: Optional[List[Dict]] = None,
        placement_objects: Optional[List[Dict]] = None,
        process_stage: Optional[str] = None,
        primary_function: Optional[str] = None,
        flow_requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add new equipment to layout

        Args:
            equipment_name: Name from equipment catalog (e.g., "Line_04")
            facing: Direction equipment faces (e.g., "south_wall", "north_wall")
            placement_room_layout: List of room layout relationships
                e.g., [{"layout_element_id": "south_wall", "preposition": "in the corner"}]
            placement_objects: List of object relationships
                e.g., [{"object_id": "Line_04_1", "preposition": "in front", "is_adjacent": true}]
            process_stage: Process stage (optional)
            primary_function: Primary function description (optional)
            flow_requirements: Flow requirements (optional)

        Returns:
            The created equipment object with generated object_id
        """
        # Validate equipment exists in catalog
        if equipment_name not in get_equipment_list():
            raise ValueError(f"Equipment '{equipment_name}' not found in catalog. Use list_available_equipment() to see available options.")

        # Get equipment info from catalog
        catalog_info = get_equipment_info(equipment_name)
        if not catalog_info:
            raise ValueError(f"Could not retrieve catalog info for '{equipment_name}'")

        # Generate unique object ID
        object_id = self._generate_object_id(equipment_name)

        # Build equipment object
        equipment_obj = {
            "object_id": object_id,
            "style": catalog_info.get("style", "industrial"),
            "material": catalog_info.get("material", "metal"),
            "size_in_meters": catalog_info.get("approximate_size", {
                "length": 1.0,
                "width": 1.0,
                "height": 1.0
            }),
            "is_on_the_floor": True,
            "facing": facing,
            "placement": {
                # Only default to "middle of the room" if NO placement constraints specified
                # If there are object relationships, don't add conflicting room constraints
                "room_layout_elements": placement_room_layout if placement_room_layout else (
                    [] if placement_objects else [{"layout_element_id": "middle of the room", "preposition": "on"}]
                ),
                "objects_in_room": placement_objects or []
            },
            "rotation": {"z_angle": 0},
            "cluster": {"constraint_area": {"x_neg": 0, "x_pos": 0, "y_neg": 0, "y_pos": 0}},
            "connections": []
        }

        # Add optional metadata
        if process_stage:
            equipment_obj["process_stage"] = process_stage
        if primary_function:
            equipment_obj["primary_function"] = primary_function
        if flow_requirements:
            equipment_obj["flow_requirements"] = flow_requirements

        self.objects.append(equipment_obj)
        return equipment_obj

    def remove_equipment(self, object_id: str) -> bool:
        """Remove equipment from layout

        Args:
            object_id: ID of object to remove

        Returns:
            True if removed, False if not found
        """
        for i, obj in enumerate(self.objects):
            if obj.get("object_id") == object_id:
                self.objects.pop(i)
                # Also remove any references to this object in other objects' placements
                for other_obj in self.objects:
                    placement = other_obj.get("placement", {})
                    obj_rels = placement.get("objects_in_room", [])
                    placement["objects_in_room"] = [
                        rel for rel in obj_rels if rel.get("object_id") != object_id
                    ]
                    # Remove from connections
                    connections = other_obj.get("connections", [])
                    other_obj["connections"] = [
                        conn for conn in connections if conn.get("object_id") != object_id
                    ]
                return True
        return False

    def update_equipment_placement(
        self,
        object_id: str,
        placement_room_layout: Optional[List[Dict]] = None,
        placement_objects: Optional[List[Dict]] = None
    ) -> bool:
        """Update equipment placement relationships

        Args:
            object_id: ID of object to update
            placement_room_layout: New room layout relationships (replaces existing)
            placement_objects: New object relationships (replaces existing)

        Returns:
            True if updated, False if object not found
        """
        obj = self.get_equipment_details(object_id)
        if not obj:
            return False

        if placement_room_layout is not None:
            obj["placement"]["room_layout_elements"] = placement_room_layout

        if placement_objects is not None:
            obj["placement"]["objects_in_room"] = placement_objects

        # CRITICAL: Clear existing position so backtracking will recalculate it
        # based on the new placement relationships
        if "position" in obj:
            del obj["position"]

        return True

    def add_connection(
        self,
        source_id: str,
        target_id: str,
        source_endpoint: str = "primary_forward_positive",
        target_endpoint: str = "primary_forward_negative",
        connection_type: str = "material_flow"
    ) -> bool:
        """Add connection between two equipment items

        Args:
            source_id: Source equipment object_id
            target_id: Target equipment object_id
            source_endpoint: Source connection point
            target_endpoint: Target connection point
            connection_type: Type of connection (default: "material_flow")

        Returns:
            True if added, False if source object not found
        """
        source_obj = self.get_equipment_details(source_id)
        if not source_obj:
            return False

        # Verify target exists
        target_obj = self.get_equipment_details(target_id)
        if not target_obj:
            return False

        # Add connection
        connection = {
            "object_id": target_id,
            "connection_type": connection_type,
            "source_endpoint": source_endpoint,
            "target_endpoint": target_endpoint
        }

        if "connections" not in source_obj:
            source_obj["connections"] = []

        # Check if connection already exists
        existing = any(
            conn.get("object_id") == target_id and
            conn.get("source_endpoint") == source_endpoint
            for conn in source_obj["connections"]
        )

        if not existing:
            source_obj["connections"].append(connection)

        return True

    def set_equipment_facing(self, object_id: str, facing: str) -> bool:
        """Set equipment facing direction

        Args:
            object_id: ID of object to update
            facing: Direction (e.g., "south_wall", "north_wall", "east_wall", "west_wall")

        Returns:
            True if updated, False if object not found
        """
        obj = self.get_equipment_details(object_id)
        if not obj:
            return False

        obj["facing"] = facing
        return True

    def update_equipment_position(
        self,
        object_id: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None
    ) -> bool:
        """Update equipment position (only if you have explicit coordinates)

        Args:
            object_id: ID of object to update
            x, y, z: Position coordinates (optional, only provide if known)

        Returns:
            True if updated, False if object not found
        """
        obj = self.get_equipment_details(object_id)
        if not obj:
            return False

        if "position" not in obj:
            obj["position"] = {}

        if x is not None:
            obj["position"]["x"] = x
        if y is not None:
            obj["position"]["y"] = y
        if z is not None:
            obj["position"]["z"] = z

        return True

    def validate_current_layout(self) -> Dict[str, Any]:
        """Validate the current layout

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check for duplicate object IDs
        object_ids = [obj.get("object_id") for obj in self.objects]
        duplicates = set([oid for oid in object_ids if object_ids.count(oid) > 1])
        if duplicates:
            errors.append(f"Duplicate object IDs found: {duplicates}")

        # Check for invalid references in placements
        valid_ids = set(object_ids)
        for obj in self.objects:
            obj_id = obj.get("object_id")
            placement = obj.get("placement", {})
            for rel in placement.get("objects_in_room", []):
                ref_id = rel.get("object_id")
                if ref_id not in valid_ids:
                    errors.append(f"{obj_id} references non-existent object: {ref_id}")

        # Check for invalid references in connections
        for obj in self.objects:
            obj_id = obj.get("object_id")
            for conn in obj.get("connections", []):
                ref_id = conn.get("object_id")
                if ref_id not in valid_ids:
                    errors.append(f"{obj_id} has connection to non-existent object: {ref_id}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "object_count": len(self.objects)
        }

    def get_layout_json(self) -> Dict[str, List]:
        """Get the final layout as JSON object

        Returns:
            Dictionary with objects_in_room key
        """
        return {"objects_in_room": self.objects}

    def clear_layout(self):
        """Clear all objects from layout"""
        self.objects = []
        self._object_id_counter = {}


def create_layout_tool_functions(layout_builder: LayoutBuilder) -> List[Dict]:
    """Create function tool definitions for autogen agents

    Args:
        layout_builder: LayoutBuilder instance to operate on

    Returns:
        List of tool function definitions
    """

    def get_current_layout_func() -> str:
        """Get summary of current manufacturing layout"""
        return json.dumps(layout_builder.get_current_layout(), indent=2)

    def get_equipment_details_func(object_id: str) -> str:
        """Get details of specific equipment

        Args:
            object_id: The ID of the equipment object
        """
        details = layout_builder.get_equipment_details(object_id)
        if details:
            return json.dumps(details, indent=2)
        return f"Equipment '{object_id}' not found"

    def list_available_equipment_func() -> str:
        """List all available equipment from catalog"""
        return json.dumps(layout_builder.list_available_equipment(), indent=2)

    def add_equipment_func(
        equipment_name: str,
        facing: str = "south_wall",
        placement_room_layout: str = "[]",
        placement_objects: str = "[]",
        process_stage: str = "",
        primary_function: str = "",
        flow_requirements: str = ""
    ) -> str:
        """Add equipment to manufacturing layout

        Args:
            equipment_name: Name from equipment catalog
            facing: Direction equipment faces
            placement_room_layout: JSON array of room layout relationships
            placement_objects: JSON array of object relationships
            process_stage: Process stage (optional)
            primary_function: Primary function (optional)
            flow_requirements: Flow requirements (optional)
        """
        try:
            room_layout = json.loads(placement_room_layout) if placement_room_layout else []
            obj_layout = json.loads(placement_objects) if placement_objects else []

            result = layout_builder.add_equipment(
                equipment_name=equipment_name,
                facing=facing,
                placement_room_layout=room_layout,
                placement_objects=obj_layout,
                process_stage=process_stage if process_stage else None,
                primary_function=primary_function if primary_function else None,
                flow_requirements=flow_requirements if flow_requirements else None
            )
            return f"Successfully added {result['object_id']}"
        except Exception as e:
            return f"Error adding equipment: {str(e)}"

    def remove_equipment_func(object_id: str) -> str:
        """Remove equipment from layout

        Args:
            object_id: ID of equipment to remove
        """
        if layout_builder.remove_equipment(object_id):
            return f"Successfully removed {object_id}"
        return f"Equipment '{object_id}' not found"

    def update_placement_func(
        object_id: str,
        placement_room_layout: str = "",
        placement_objects: str = ""
    ) -> str:
        """Update equipment placement

        Args:
            object_id: ID of equipment to update
            placement_room_layout: JSON array of room layout relationships
            placement_objects: JSON array of object relationships
        """
        try:
            room_layout = json.loads(placement_room_layout) if placement_room_layout else None
            obj_layout = json.loads(placement_objects) if placement_objects else None

            if layout_builder.update_equipment_placement(object_id, room_layout, obj_layout):
                return f"Successfully updated placement for {object_id}"
            return f"Equipment '{object_id}' not found"
        except Exception as e:
            return f"Error updating placement: {str(e)}"

    def add_connection_func(
        source_id: str,
        target_id: str,
        source_endpoint: str = "primary_forward_positive",
        target_endpoint: str = "primary_forward_negative"
    ) -> str:
        """Add connection between equipment

        Args:
            source_id: Source equipment ID
            target_id: Target equipment ID
            source_endpoint: Source connection point
            target_endpoint: Target connection point
        """
        if layout_builder.add_connection(source_id, target_id, source_endpoint, target_endpoint):
            return f"Successfully added connection: {source_id} -> {target_id}"
        return "Failed to add connection (equipment not found)"

    def set_facing_func(object_id: str, facing: str) -> str:
        """Set equipment facing direction

        Args:
            object_id: ID of equipment
            facing: Direction (south_wall, north_wall, east_wall, west_wall)
        """
        if layout_builder.set_equipment_facing(object_id, facing):
            return f"Successfully set facing for {object_id} to {facing}"
        return f"Equipment '{object_id}' not found"

    def validate_layout_func() -> str:
        """Validate current layout for errors"""
        validation = layout_builder.validate_current_layout()
        return json.dumps(validation, indent=2)

    # Return function definitions in autogen format
    return [
        {
            "name": "get_current_layout",
            "description": "Get summary of current manufacturing layout including all equipment",
            "function": get_current_layout_func
        },
        {
            "name": "get_equipment_details",
            "description": "Get detailed information about a specific equipment object",
            "function": get_equipment_details_func
        },
        {
            "name": "list_available_equipment",
            "description": "List all available equipment from the catalog",
            "function": list_available_equipment_func
        },
        {
            "name": "add_equipment",
            "description": "Add new equipment to the manufacturing layout",
            "function": add_equipment_func
        },
        {
            "name": "remove_equipment",
            "description": "Remove equipment from the layout",
            "function": remove_equipment_func
        },
        {
            "name": "update_placement",
            "description": "Update equipment placement relationships",
            "function": update_placement_func
        },
        {
            "name": "add_connection",
            "description": "Add material flow connection between two equipment items",
            "function": add_connection_func
        },
        {
            "name": "set_facing",
            "description": "Set equipment facing direction",
            "function": set_facing_func
        },
        {
            "name": "validate_layout",
            "description": "Validate the current layout for errors and inconsistencies",
            "function": validate_layout_func
        }
    ]
