import re

"""
Streamlined equipment catalog grouped by manufacturing process stage.
Only representative prefabs are exposed to the LLM to keep plans focused.
"""

# Representative Unity prefabs mapped to high-level process stages
EQUIPMENT_CATALOG = {
    # === Material Flow (Conveyors, Lifts, Transport) ===
    "Line_01": {
        "category": "material_flow",
        "description": "Vertical Lift Conveyor (Line_01) for lifting and positioning products from the main line to other devices (e.g. upper conveyors, loading machines, robot stations). Also serves as a buffer.",
        "approximate_size": {"length": 0.9, "width": 0.93, "height": 1.257},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_04", "Line_05", "Line_02"]
    },
    "Line_02": {
        "category": "material_flow",
        "description": "Fixed Roller Bridging/Transition Module (Line_02) placed on the outer side of straight conveyors (Line_04/Line_05) for material alignment and centering. NOT part of T-branch.",
        "approximate_size": {"length": 0.691, "width": 0.691, "height": 0.89},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_01", "Line_04", "Line_05"],
        "placement": "outer_side_of_conveyor"
    },
    "Line_03": {
        "category": "material_flow",
        "description": "90-degree Curved Belt Conveyor (Line_03) for changing transport direction while maintaining item orientation.",
        "approximate_size": {"length": 0.769, "width": 0.768, "height": 0.902},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_04", "Line_05"]
    },
    "Line_04": {
        "category": "material_flow",
        "description": "Long Straight Belt Conveyor (Line_04) for standard linear transport, providing a stable platform for movement and robot picking.",
        "approximate_size": {"length": 4.65, "width": 0.6, "height": 0.85},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_03", "Line_02", "Line_01", "Line_06", "Line_04", "Line_05"],
        "has_side_connections": True,
        "side_connection_terminates_with": "Line_01"
    },
    "Line_05": {
        "category": "material_flow",
        "description": "Short Straight Belt Conveyor (Line_05) for standard linear transport in tighter spaces.",
        "approximate_size": {"length": 2.247, "width": 0.6, "height": 0.85},
        "style": "Industrial",
        "material": "Metal",
        "compatible_with": ["Line_03", "Line_02", "Line_01", "Line_04", "Line_05"],
        "has_side_connections": True,
        "side_connection_terminates_with": "Line_01"
    },
    "Cart": {
        "category": "material_flow",
        "description": "Hand Platform Trolley (Cart) for manual transport of small batches, tools, or temporary storage.",
        "approximate_size": {"length": 0.561, "width": 0.941, "height": 1.014},
        "style": "Industrial",
        "material": "Metal"
    },
    "PalletMover": {
        "category": "material_flow",
        "description": "Motorized pallet mover for automated material transport and positioning.",
        "approximate_size": {"length": 1.2, "width": 0.8, "height": 0.6},
        "style": "Industrial",
        "material": "Metal"
    },
    "Pallet": {
        "category": "material_flow",
        "description": "Standard wooden pallet for staging inbound or outbound goods.",
        "approximate_size": {"length": 1.131, "width": 1.029, "height": 0.118},
        "style": "Industrial",
        "material": "Wood"
    },
    "Pallet_Jack": {
        "category": "material_flow",
        "description": "Manual pallet jack for repositioning pallets between workzones.",
        "approximate_size": {"length": 0.551, "width": 1.262, "height": 1.142},
        "style": "Industrial",
        "material": "Metal"
    },
    # === Assembly & Processing (Workstations, Robots) ===
    "Line_06": {
        "category": "assembly_processing",
        "description": "Alignment & Positioning Station (Line_06) for precise calibration of workpiece position before processing (assembly, inspection, etc.).",
        "approximate_size": {"length": 1.355, "width": 1.16, "height": 2.182},
        "style": "Industrial",
        "material": "Metal",
        "conveyor_passes_through": True
    },
    "Line_07": {
        "category": "assembly_processing",
        "description": "Robotic Handling & Assembly Cell (Line_07) with a mechanical arm for pick-and-place, assembly, or testing tasks. Straight conveyors (Line_04/Line_05) MUST pass through this cell.",
        "approximate_size": {"length": 2.859, "width": 3.371, "height": 3.215},
        "style": "Industrial",
        "material": "Metal",
        "conveyor_passes_through": True
    },
    "abb_irb2600_12_165": {
        "category": "assembly_processing",
        "description": "ABB IRB2600 industrial robot arm on a fixed pedestal for machine tending or heavy handling. Requires safety fencing.",
        "approximate_size": {"length": 0.0, "width": 1.03, "height": 1.26},
        "style": "Industrial",
        "material": "Metal",
        "reach_envelope": 1.65,
        "payload_kg": 12
    },
    "ur10e_robot": {
        "category": "assembly_processing",
        "description": "UR10e collaborative robot arm for shared human-robot assembly tasks. Safe for human interaction without fencing.",
        "approximate_size": {"length": 0.386, "width": 1.326, "height": 0.275},
        "style": "Industrial",
        "material": "Metal",
        "reach_envelope": 1.3,
        "payload_kg": 10,
        "collaborative": True
    },
    # === Machining & Fabrication ===
    "MFG_Equip_30ftx7ft_w_Exhaust": {
        "category": "machining_fabrication",
        "description": "Large enclosed machining or fabrication cell with integrated exhaust for CNC, milling, or other machining operations.",
        "approximate_size": {"length": 3.064, "width": 9.027, "height": 7.727},
        "style": "Industrial",
        "material": "Metal"
    },
    "Power_Cutter": {
        "category": "machining_fabrication",
        "description": "Stationary power cutter or saw for metal or composite prep.",
        "approximate_size": {"length": 0.956, "width": 0.364, "height": 0.494},
        "style": "Industrial",
        "material": "Metal"
    },
    "ScissorLift": {
        "category": "machining_fabrication",
        "description": "Mobile scissor lift platform for servicing tall equipment.",
        "approximate_size": {"length": 2.437, "width": 1.312, "height": 2.487},
        "style": "Industrial",
        "material": "Metal"
    },
    # === Support Equipment (Storage, Tables, Inspection) ===
    "ShelvingRack": {
        "category": "support_equipment",
        "description": "Heavy-duty shelving rack for kitting totes or part storage near the line.",
        "approximate_size": {"length": 1.232, "width": 3.658, "height": 3.661},
        "style": "Industrial",
        "material": "Metal"
    },
    "Table_6ft": {
        "category": "support_equipment",
        "description": "Six-foot work table for assembly, staging, or inspection tasks.",
        "approximate_size": {"length": 1.016, "width": 1.829, "height": 0.916},
        "style": "Industrial",
        "material": "Metal"
    },
    "Camera_Stand_7ft": {
        "category": "support_equipment",
        "description": "Adjustable camera or sensor stand for inline inspection and QA.",
        "approximate_size": {"length": 0.403, "width": 0.349, "height": 2.133},
        "style": "Industrial",
        "material": "Metal"
    },
    # === Infrastructure (Controls, Ventilation, Safety) ===
    "Line_08": {
        "category": "infrastructure",
        "description": "Electrical Control Cabinet (Line_08) housing PLC, drives, and power supplies for the production line.",
        "approximate_size": {"length": 1.1, "width": 0.71, "height": 1.732},
        "style": "Industrial",
        "material": "Metal"
    },
    "VentilatorFan_Straight": {
        "category": "infrastructure",
        "description": "Straight duct ventilation fan for paint, fumes, or curing booths.",
        "approximate_size": {"length": 2.493, "width": 5.813, "height": 0.803},
        "style": "Industrial",
        "material": "Metal"
    },
    "SafetyRailing_8ft": {
        "category": "infrastructure",
        "description": "Eight-foot safety railing section to segregate walkways from equipment.",
        "approximate_size": {"length": 0.115, "width": 2.5, "height": 1.055},
        "style": "Industrial",
        "material": "Metal"
    },
    "SafetyRailing_16ft": {
        "category": "infrastructure",
        "description": "Sixteen-foot safety railing section for longer perimeter runs.",
        "approximate_size": {"length": 0.115, "width": 4.953, "height": 1.055},
        "style": "Industrial",
        "material": "Metal"
    }
}

# High-level process groupings to guide the planner
EQUIPMENT_CATEGORIES = {
    "material_flow": [
        "Line_01", "Line_02", "Line_03", "Line_04", "Line_05",
        "Cart", "PalletMover", "Pallet", "Pallet_Jack"
    ],
    "assembly_processing": [
        "Line_06", "Line_07",
        "abb_irb2600_12_165", "ur10e_robot"
    ],
    "machining_fabrication": [
        "MFG_Equip_30ftx7ft_w_Exhaust", "Power_Cutter", "ScissorLift"
    ],
    "support_equipment": [
        "ShelvingRack", "Table_6ft", "Camera_Stand_7ft"
    ],
    "infrastructure": [
        "Line_08", "VentilatorFan_Straight", "SafetyRailing_8ft", "SafetyRailing_16ft"
    ]
}

EQUIPMENT_ALIAS_OVERRIDES = {
    # Line equipment aliases
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
    "line08": "Line_08",
    # Legacy equipment aliases
    "pallet": "Pallet",
    "pallet_staging": "Pallet",
    "pallet jack": "Pallet_Jack",
    "pallet_jack": "Pallet_Jack",
    "cnc_bank": "MFG_Equip_30ftx7ft_w_Exhaust",
    "cnc_station": "MFG_Equip_30ftx7ft_w_Exhaust",
    "cnc_machine": "MFG_Equip_30ftx7ft_w_Exhaust",
    "cnc": "MFG_Equip_30ftx7ft_w_Exhaust",
    "machining_cell": "MFG_Equip_30ftx7ft_w_Exhaust",
    "fabrication_cell": "MFG_Equip_30ftx7ft_w_Exhaust",
    "power_cutter": "Power_Cutter",
    "saw": "Power_Cutter",
    "cutter": "Power_Cutter",
    "scissor_lift": "ScissorLift",
    "lift_platform": "ScissorLift",
    "abb_irb2600": "abb_irb2600_12_165",
    "abb_robot": "abb_irb2600_12_165",
    "irb2600": "abb_irb2600_12_165",
    "industrial_robot": "abb_irb2600_12_165",
    "ur10e": "ur10e_robot",
    "ur10": "ur10e_robot",
    "cobot": "ur10e_robot",
    "collaborative_robot": "ur10e_robot",
    "shelving_rack": "ShelvingRack",
    "shelving": "ShelvingRack",
    "rack": "ShelvingRack",
    "storage_rack": "ShelvingRack",
    "table_6ft": "Table_6ft",
    "work_table": "Table_6ft",
    "worktable": "Table_6ft",
    "inspection_table": "Table_6ft",
    "assembly_table": "Table_6ft",
    "camera_stand": "Camera_Stand_7ft",
    "camera_stand_7ft": "Camera_Stand_7ft",
    "inspection_camera": "Camera_Stand_7ft",
    "vision_stand": "Camera_Stand_7ft",
    "ventilator": "VentilatorFan_Straight",
    "ventilator_fan": "VentilatorFan_Straight",
    "exhaust_fan": "VentilatorFan_Straight",
    "ventilation": "VentilatorFan_Straight",
    "safety_railing": "SafetyRailing_8ft",
    "safety_railing_8ft": "SafetyRailing_8ft",
    "safety_railing_16ft": "SafetyRailing_16ft",
    "long_railing": "SafetyRailing_16ft",
    "railing": "SafetyRailing_8ft",
    "safety_fence": "SafetyRailing_8ft",
    "guard_rail": "SafetyRailing_8ft"
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
