import re

"""
Streamlined equipment catalog grouped by manufacturing process stage.
Only representative prefabs are exposed to the LLM to keep plans focused.
"""

# Representative Unity prefabs mapped to high-level process stages
EQUIPMENT_CATALOG = {
    "Line_01": {
        "category": "logistics_equipment",
        "description": "Vertical Lift Conveyor (Line_01) for lifting and positioning products from the main line to other devices (e.g. upper conveyors, loading machines, robot stations). Also serves as a buffer.",
        "approximate_size": {"length": 0.9, "width": 0.93, "height": 1.257},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_04", "Line_05", "Line_02"],
        "process_stage": "MaterialFlow"
    },
    "Line_02": {
        "category": "logistics_equipment",
        "description": "Fixed Roller Bridging/Transition Module (Line_02) placed on the outer side of straight conveyors (Line_04/Line_05) for material alignment and centering. NOT part of T-branch.",
        "approximate_size": {"length": 0.691, "width": 0.691, "height": 0.89},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_01", "Line_04", "Line_05"],
        "placement": "outer_side_of_conveyor",  # Placed alongside straight conveyors, not in T-branch
        "process_stage": "MaterialFlow"
    },
    "Line_03": {
        "category": "conveyance_systems",
        "description": "90-degree Curved Belt Conveyor (Line_03) for changing transport direction while maintaining item orientation.",
        "approximate_size": {"length": 0.769, "width": 0.768, "height": 0.902},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_04", "Line_05"],
        "process_stage": "MaterialFlow"
    },
    "Line_04": {
        "category": "conveyance_systems",
        "description": "Long Straight Belt Conveyor (Line_04) for standard linear transport, providing a stable platform for movement and robot picking.",
        "approximate_size": {"length": 4.65, "width": 0.6, "height": 0.85},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_03", "Line_02", "Line_01", "Line_06", "Line_04", "Line_05"],
        "has_side_connections": True,
        "side_connection_terminates_with": "Line_01",  # T-branch must end with Line_01
        "process_stage": "MaterialFlow"
    },
    "Line_05": {
        "category": "conveyance_systems",
        "description": "Short Straight Belt Conveyor (Line_05) for standard linear transport in tighter spaces.",
        "approximate_size": {"length": 2.247, "width": 0.6, "height": 0.85},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_03", "Line_02", "Line_01", "Line_04", "Line_05"],
        "has_side_connections": True,
        "side_connection_terminates_with": "Line_01",  # T-branch must end with Line_01
        "process_stage": "MaterialFlow"
    },
    "Line_06": {
        "category": "workstations_processing",
        "description": "Alignment & Positioning Station (Line_06) for precise calibration of workpiece position before processing (assembly, inspection, etc.).",
        "approximate_size": {"length": 1.355, "width": 1.16, "height": 2.182},
        "style": "Industrial",
        "material": "Metal",
        "conveyor_passes_through": True,  # Straight conveyors must run through this workstation
        "process_stage": "Assembly"
    },
    "Line_07": {
        "category": "workstations_processing",
        "description": "Robotic Handling & Assembly Cell (Line_07) with a mechanical arm for pick-and-place, assembly, or testing tasks. Straight conveyors (Line_04/Line_05) MUST pass through this cell.",
        "approximate_size": {"length": 2.859, "width": 3.371, "height": 3.215},
        "style": "Industrial",
        "material": "Metal",
        "conveyor_passes_through": True,  # Straight conveyors must run through this cell
        "process_stage": "Assembly"
    },
    "Line_08": {
        "category": "infrastructure",
        "description": "Electrical Control Cabinet (Line_08) housing PLC, drives, and power supplies for the production line.",
        "approximate_size": {"length": 1.1, "width": 0.71, "height": 1.732},
        "style": "Industrial",
        "material": "Metal",
        "process_stage": "Utilities"
    },
    "Cart": {
        "category": "logistics_equipment",
        "description": "Hand Platform Trolley (Cart) for manual transport of small batches, tools, or temporary storage.",
        "approximate_size": {"length": 0.561, "width": 0.941, "height": 1.014},
        "style": "Industrial",
        "material": "Metal",
        "process_stage": "MaterialFlow"
    },
    "PalletMover": {
        "category": "logistics_equipment",
        "description": "Motorized pallet mover for automated material transport and positioning",
        "approximate_size": {"length": 1.2, "width": 0.8, "height": 0.6},
        "style": "Industrial",
        "material": "Metal",
        "process_stage": "MaterialFlow"
    }
}

# High-level process groupings to guide the planner
EQUIPMENT_CATEGORIES = {
    "logistics_equipment": [
        "Line_01",
        "Line_02",
        "Cart",
        "PalletMover"
    ],
    "conveyance_systems": [
        "Line_03",
        "Line_04",
        "Line_05"
    ],
    "workstations_processing": [
        "Line_06",
        "Line_07"
    ],
    "infrastructure": [
        "Line_08"
    ]
}

EQUIPMENT_ALIAS_OVERRIDES = {
    "lift_transfer_unit": "Line_01",
    "lift_transfer": "Line_01",
    "roller_conveyor_bridge": "Line_02",
    "roller_bridge": "Line_02",
    "curved_belt_conveyor": "Line_03",
    "curve_conveyor": "Line_03",
    "straight_belt_conveyor_long": "Line_04",
    "long_conveyor": "Line_04",
    "straight_belt_conveyor_short": "Line_05",
    "short_conveyor": "Line_05",
    "alignment_positioning_station": "Line_06",
    "alignment_station": "Line_06",
    "robotic_assembly_cell": "Line_07",
    "robot_cell": "Line_07",
    "electrical_control_cabinet": "Line_08",
    "control_cabinet": "Line_08",
    "trolley": "Cart",
    "pallet_mover": "PalletMover",
    "motorized_pallet_mover": "PalletMover",
    "line01": "Line_01",
    "line02": "Line_02",
    "line03": "Line_03",
    "line04": "Line_04",
    "line05": "Line_05",
    "line06": "Line_06",
    "line07": "Line_07",
    "line08": "Line_08"
}


def get_equipment_list():
    """Return list of all available equipment names."""
    return list(EQUIPMENT_CATALOG.keys())


def get_equipment_by_category(category):
    """Return equipment names filtered by process category."""
    return EQUIPMENT_CATEGORIES.get(category, [])


def get_equipment_info(equipment_name):
    """Return detailed information for a specific equipment item."""
    return EQUIPMENT_CATALOG.get(equipment_name)


def get_equipment_aliases():
    """Return mapping of lowercase aliases to canonical equipment names."""
    alias_map = {name.lower(): name for name in EQUIPMENT_CATALOG.keys()}
    for name in EQUIPMENT_CATALOG.keys():
        snake_like = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        snake_like = snake_like.replace("__", "_")
        alias_map.setdefault(snake_like, name)
    alias_map.update(EQUIPMENT_ALIAS_OVERRIDES)
    return alias_map


def format_equipment_for_llm():
    """Format the catalog into a process-oriented prompt."""
    formatted = "Available Equipment (grouped by process stage):\n\n"
    for category, items in EQUIPMENT_CATEGORIES.items():
        if not items:
            continue
        formatted += f"## {category.replace('_', ' ').title()}:\n"
        for item in items:
            info = EQUIPMENT_CATALOG[item]
            size = info["approximate_size"]
            formatted += (
                f"- {item}: {info['description']} "
                f"(Size: {size['length']}x{size['width']}x{size['height']}m, "
                f"Style: {info['style']}, Material: {info['material']})"
            )
            if "compatible_with" in info and info["compatible_with"]:
                formatted += f" | Works with: {', '.join(info['compatible_with'])}"
            formatted += "\n"
        formatted += "\n"
    return formatted


def generate_equipment_constraint_prompt():
    """Generate constraint prompt for agents."""
    return f"""
IMPORTANT CONSTRAINT: Use ONLY equipment from the catalog provided below.
Do NOT invent new assets. Select the item that best matches each process stage.

{format_equipment_for_llm()}

When proposing equipment, cite the exact catalog name and honor the representative size, style, and material.

CONNECTION RULES:
1. Main Loop: Use Line_04/Line_05 (straight) + Line_03 (90° curve) to form continuous conveyor loops.
2. Line_02 Placement: Line_02 (Roller Bridge) is placed on the OUTER SIDE of straight conveyors (Line_04/Line_05) for alignment. It is NOT part of T-branch.
3. T-BRANCH Structure:
   - T-branch connects directly from the side of Line_04/Line_05.
   - Branch uses Line_04/Line_05 conveyors, all rotated 90° from main line direction.
   - Every T-branch MUST terminate with a Line_01 (Lift Transfer Unit).
   - Structure: MainConveyor --[side]--> Conveyor(s, ⊥) --> Line_01
4. Line_07 (Robotic Cell): Straight conveyors (Line_04/Line_05) MUST pass THROUGH Line_07. The conveyor runs inside the cell for robot access.

INTEGRATION RULES (CRITICAL for layout modifications):
1. NO ISOLATED EQUIPMENT: Every new conveyor/transport module MUST connect to the existing material flow network. Orphan equipment chains are INVALID.
2. INSERTION POINTS: When adding new conveyors, identify WHERE they connect to the existing loop:
   - Insert INTO an existing segment (break a connection and insert new equipment between)
   - Branch OFF from an existing Line_04/Line_05 using its side connection (right_negative endpoint)
   - Extend FROM a terminal point (e.g., after a Line_01 or Line_03)
3. CONNECTION CONTINUITY: After adding equipment, verify the material flow path remains continuous from start to end.
4. REMOVAL RULES: When removing equipment, reconnect the upstream and downstream neighbors to maintain flow continuity.

Ensure transport modules are connected logically to form continuous material flow paths.
"""


if __name__ == "__main__":
    print("Equipment Catalog Summary")
    print(f"Total equipment items: {len(EQUIPMENT_CATALOG)}")
    print(f"Categories: {list(EQUIPMENT_CATEGORIES.keys())}")
    print()
    print(format_equipment_for_llm())
