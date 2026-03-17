"""
Command-based layout construction.
Agents output simple commands instead of large JSON structures.
More stable and easier to parse than full JSON.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from layout.layout_tools import LayoutBuilder


class LayoutCommandExecutor:
    """Executes layout construction commands from agent responses"""

    VALID_FACING_VALUES = {"north_wall", "south_wall", "east_wall", "west_wall"}

    def __init__(self, layout_builder: LayoutBuilder):
        self.builder = layout_builder
        self.execution_log = []

    def _normalize_facing(self, facing: str) -> str:
        """Normalize facing value to valid wall direction

        Handles common variations and invalid values by attempting to infer
        the intended direction.
        """
        facing_lower = facing.lower().strip()

        # Direct matches
        if facing_lower in self.VALID_FACING_VALUES:
            return facing_lower

        # Common variations
        variations = {
            "north": "north_wall",
            "south": "south_wall",
            "east": "east_wall",
            "west": "west_wall",
            "facing north": "north_wall",
            "facing south": "south_wall",
            "facing east": "east_wall",
            "facing west": "west_wall",
        }
        if facing_lower in variations:
            return variations[facing_lower]

        # Try to extract direction from phrases like "parallel to existing Line_04_5"
        for direction in ["north", "south", "east", "west"]:
            if direction in facing_lower:
                return f"{direction}_wall"

        # Default to south_wall if we can't determine
        return "south_wall"

    def parse_and_execute_commands(self, agent_response: str) -> Tuple[bool, str]:
        """Parse commands from agent response and execute them

        Commands are in format:
        ADD_EQUIPMENT: equipment_name | facing | placement_description | metadata

        Args:
            agent_response: Text response from agent containing commands

        Returns:
            (success, message) tuple
        """
        commands = self._extract_commands(agent_response)

        if not commands:
            return False, "No valid commands found in response"

        results = []
        for cmd in commands:
            try:
                result = self._execute_command(cmd)
                results.append(result)
                self.execution_log.append({
                    "command": cmd,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                error_msg = f"Error executing command: {str(e)}"
                results.append(error_msg)
                self.execution_log.append({
                    "command": cmd,
                    "error": error_msg,
                    "success": False
                })

        success = all("Successfully" in r or "added" in r.lower() for r in results)
        return success, "\n".join(results)

    def _extract_commands(self, text: str) -> List[Dict[str, str]]:
        """Extract structured commands from text

        Looks for patterns like:
        ADD_EQUIPMENT: Line_04 | south_wall | near south wall | material_flow | conveyor transport
        UPDATE_PLACEMENT: Line_04_1 | in front of Line_01_1, adjacent
        ADD_CONNECTION: Line_04_1 -> Line_01_1 | primary_forward_positive -> primary_forward_negative
        """
        commands = []

        # Pattern for ADD_EQUIPMENT
        add_pattern = r'ADD_EQUIPMENT:\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|]+)(?:\s*\|\s*([^\|]+))?(?:\s*\|\s*([^\|\n]+))?'
        for match in re.finditer(add_pattern, text, re.MULTILINE):
            commands.append({
                "type": "ADD_EQUIPMENT",
                "equipment_name": match.group(1).strip(),
                "facing": match.group(2).strip(),
                "placement": match.group(3).strip(),
                "process_stage": match.group(4).strip() if match.group(4) else "",
                "primary_function": match.group(5).strip() if match.group(5) else ""
            })

        # Pattern for UPDATE_PLACEMENT
        update_pattern = r'UPDATE_PLACEMENT:\s*([^\|]+)\s*\|\s*([^\|\n]+)'
        for match in re.finditer(update_pattern, text, re.MULTILINE):
            commands.append({
                "type": "UPDATE_PLACEMENT",
                "object_id": match.group(1).strip(),
                "placement": match.group(2).strip()
            })

        # Pattern for ADD_CONNECTION
        conn_pattern = r'ADD_CONNECTION:\s*([^\s]+)\s*->\s*([^\s\|]+)(?:\s*\|\s*([^\s]+)\s*->\s*([^\s\n]+))?'
        for match in re.finditer(conn_pattern, text, re.MULTILINE):
            commands.append({
                "type": "ADD_CONNECTION",
                "source_id": match.group(1).strip(),
                "target_id": match.group(2).strip(),
                "source_endpoint": match.group(3).strip() if match.group(3) else "primary_forward_positive",
                "target_endpoint": match.group(4).strip() if match.group(4) else "primary_forward_negative"
            })

        # Pattern for REMOVE_EQUIPMENT
        remove_pattern = r'REMOVE_EQUIPMENT:\s*([^\n]+)'
        for match in re.finditer(remove_pattern, text, re.MULTILINE):
            commands.append({
                "type": "REMOVE_EQUIPMENT",
                "object_id": match.group(1).strip()
            })

        return commands

    def _execute_command(self, cmd: Dict[str, str]) -> str:
        """Execute a single command"""
        cmd_type = cmd.get("type")

        if cmd_type == "ADD_EQUIPMENT":
            return self._execute_add_equipment(cmd)
        elif cmd_type == "UPDATE_PLACEMENT":
            return self._execute_update_placement(cmd)
        elif cmd_type == "ADD_CONNECTION":
            return self._execute_add_connection(cmd)
        elif cmd_type == "REMOVE_EQUIPMENT":
            return self._execute_remove_equipment(cmd)
        else:
            return f"Unknown command type: {cmd_type}"

    def _normalize_equipment_name(self, name: str) -> str:
        """Normalize equipment name to base catalog name

        Agents may output names with instance numbers like 'Line_04_7' but
        the catalog expects base names like 'Line_04'. This strips the
        trailing instance number if present.
        """
        import re
        # Pattern: base_name followed by _number at the end
        # e.g., "Line_04_7" -> "Line_04", "Line_07_3" -> "Line_07"
        # But NOT "Line_04" -> "Line" (we need to keep the type number)

        # First check if the name as-is is in the catalog
        from catalog.equipment_catalog import get_equipment_list
        if name in get_equipment_list():
            return name

        # Try stripping trailing _number
        match = re.match(r'^(.+?)_(\d+)$', name)
        if match:
            base = match.group(1)
            if base in get_equipment_list():
                return base

        # Return as-is if no transformation works
        return name

    def _execute_add_equipment(self, cmd: Dict[str, str]) -> str:
        """Execute ADD_EQUIPMENT command"""
        raw_name = cmd["equipment_name"]
        equipment_name = self._normalize_equipment_name(raw_name)  # Normalize to catalog name
        facing = self._normalize_facing(cmd["facing"])  # Normalize facing value
        placement_desc = cmd["placement"]

        # Parse placement description into structured format
        placement_room, placement_objs = self._parse_placement_description(placement_desc)

        # Add equipment
        result = self.builder.add_equipment(
            equipment_name=equipment_name,
            facing=facing,
            placement_room_layout=placement_room,
            placement_objects=placement_objs,
            process_stage=cmd.get("process_stage") or None,
            primary_function=cmd.get("primary_function") or None,
            flow_requirements=cmd.get("flow_requirements") or None
        )

        return f"Successfully added {result['object_id']} (facing: {facing})"

    def _execute_update_placement(self, cmd: Dict[str, str]) -> str:
        """Execute UPDATE_PLACEMENT command"""
        object_id = cmd["object_id"]
        placement_desc = cmd["placement"]

        # Parse placement
        placement_room, placement_objs = self._parse_placement_description(placement_desc)

        if self.builder.update_equipment_placement(object_id, placement_room, placement_objs):
            return f"Successfully updated placement for {object_id}"
        return f"Failed to update placement for {object_id} (not found)"

    def _execute_add_connection(self, cmd: Dict[str, str]) -> str:
        """Execute ADD_CONNECTION command"""
        source_id = cmd["source_id"]
        target_id = cmd["target_id"]
        source_endpoint = cmd.get("source_endpoint", "primary_forward_positive")
        target_endpoint = cmd.get("target_endpoint", "primary_forward_negative")

        if self.builder.add_connection(source_id, target_id, source_endpoint, target_endpoint):
            return f"Successfully added connection: {source_id} -> {target_id}"
        return f"Failed to add connection (equipment not found)"

    def _execute_remove_equipment(self, cmd: Dict[str, str]) -> str:
        """Execute REMOVE_EQUIPMENT command"""
        object_id = cmd["object_id"]

        if self.builder.remove_equipment(object_id):
            return f"Successfully removed {object_id}"
        return f"Failed to remove {object_id} (not found)"

    def _parse_placement_description(self, desc: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse natural language placement description into structured format

        Examples:
        - "near south wall" -> room_layout: [{"layout_element_id": "south_wall", "preposition": "on"}]
        - "in front of Line_04_1, adjacent" -> objects: [{"object_id": "Line_04_1", "preposition": "in front", "is_adjacent": true}]
        - "middle of room" -> room_layout: [{"layout_element_id": "middle of the room", "preposition": "on"}]
        """
        placement_room = []
        placement_objs = []

        desc_lower = desc.lower()

        # Check for wall references
        walls = ["south_wall", "north_wall", "east_wall", "west_wall"]
        for wall in walls:
            if wall.replace("_", " ") in desc_lower or wall in desc_lower:
                preposition = "in the corner" if "corner" in desc_lower else "on"
                placement_room.append({
                    "layout_element_id": wall,
                    "preposition": preposition
                })
                break

        # Check for middle of room
        if "middle" in desc_lower or "center" in desc_lower:
            placement_room.append({
                "layout_element_id": "middle of the room",
                "preposition": "on"
            })

        # Check for object references (pattern: "preposition object_id")
        # Map from input patterns to normalized preposition values expected by utils.py
        preposition_patterns = [
            ("in front of", "in front"),  # "in front of X" -> preposition "in front"
            ("behind", "behind"),
            ("left of", "left of"),
            ("right of", "right of"),
            ("on", "on"),
            ("under", "under"),
            ("through", "through"),
        ]
        wall_ids = ["south_wall", "north_wall", "east_wall", "west_wall"]

        for input_prep, normalized_prep in preposition_patterns:
            pattern = f"{input_prep}\\s+([A-Za-z0-9_]+)"
            match = re.search(pattern, desc_lower)
            if match:
                object_id = match.group(1)

                # Skip if this is a wall reference (already handled in room_layout section)
                if object_id in wall_ids or object_id.replace("_", " ") in [w.replace("_", " ") for w in wall_ids]:
                    continue

                # Preserve original case from description for object_id
                # Try to find the original case version
                original_match = re.search(f"{input_prep}\\s+([A-Za-z0-9_]+)", desc, re.IGNORECASE)
                if original_match:
                    object_id = original_match.group(1)

                is_adjacent = "adjacent" in desc_lower or "touching" in desc_lower
                placement_objs.append({
                    "object_id": object_id,
                    "preposition": normalized_prep,  # Use normalized preposition
                    "is_adjacent": is_adjacent
                })

        # Default to middle of room if nothing specified
        if not placement_room and not placement_objs:
            placement_room.append({
                "layout_element_id": "middle of the room",
                "preposition": "on"
            })

        return placement_room, placement_objs

    def get_final_layout(self) -> Dict[str, Any]:
        """Get the final constructed layout"""
        return self.builder.get_layout_json()

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of command execution"""
        total = len(self.execution_log)
        successful = sum(1 for entry in self.execution_log if entry.get("success", False))

        return {
            "total_commands": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "log": self.execution_log
        }
