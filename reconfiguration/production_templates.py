"""
Production-focused reconfiguration templates.

These templates generate natural language requests for the hybrid reconfiguration system
to handle common production layout changes like worker access, buffering, and safety.
"""


def define_material_flow(
    loading_edge: str = "south",
    loading_position: float = 0.3,
    unloading_edge: str = "south",
    unloading_position: float = 0.7,
    flow_direction: str = "clockwise"
) -> str:
    """Generate request for defining material flow by creating a physical passage.

    This physically opens the conveyor loop by removing the corner, conveyor segment,
    and any inline stations at the passage location. The result is an open production
    line with clear loading (entry) and unloading (exit) points.

    Args:
        loading_edge: Edge for material entry ("north", "south", "east", "west")
        loading_position: Position along loading edge (0.0-1.0)
        unloading_edge: Edge for product exit
        unloading_position: Position along unloading edge (0.0-1.0)
        flow_direction: Direction of loop flow ("clockwise" or "counterclockwise")

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    return f"""Define material flow by creating a physical passage in the conveyor loop.

This physically opens the closed loop to create:
- A LOADING point (L) where materials enter the production cell
- An UNLOADING point (U) where finished products exit
- A PHYSICAL PASSAGE (gap) between U and L for worker access

LOADING POINT (Flow Start):
- Edge: {loading_edge}
- Position: {int(loading_position * 100)}% along the edge
- Material enters here and flows {flow_direction} through the cell

UNLOADING POINT (Flow End):
- Edge: {unloading_edge}
- Position: {int(unloading_position * 100)}% along the edge
- Finished products exit here before the passage gap

PHYSICAL PASSAGE:
- Created by removing the corner, conveyor segment, and any inline stations
- Located between the unloading and loading points (against flow direction)
- Provides worker access to the cell interior for operations and maintenance"""


def extend_conveyor_loop(
    edge: str = "west",
    stations: list = None,
    extension_distance: float = 4.0,
    purpose: str = "expand production capacity"
) -> str:
    """Generate request for extending the conveyor loop outward.

    Args:
        edge: Which edge to extend ("north", "south", "east", "west")
        stations: List of station types to add (e.g., ["Line_07", "Line_06", "Line_02"])
        extension_distance: How far to extend in meters
        purpose: Why extending the loop

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    if stations is None:
        stations = ["Line_07", "Line_06"]

    stations_text = ", ".join(stations)

    return f"""Extend the conveyor loop {edge}ward to {purpose}.

SPECIFICATIONS:
- Extension direction: {edge}
- Extension distance: approximately {extension_distance} meters
- Add the following stations on the new section: {stations_text}

STATION PLACEMENT:
- Line_07 (assembly) and Line_06 (inspection) are placed INLINE on the conveyor
- Line_02 (workstation) is placed as a SIDE-LOADER adjacent to the conveyor

The extension should maintain the loop structure with proper corners and connections."""


# ==================== Extended Equipment Templates (Legacy Catalog) ====================

def add_pallet_staging(
    near_equipment: str,
    num_pallets: int = 4,
    add_pallet_jack: bool = True,
    layout: str = "grid"
) -> str:
    """Generate request for adding a pallet staging/buffer area.

    Args:
        near_equipment: Equipment ID to place staging near (e.g., "Line_01_1")
        num_pallets: Number of Pallet positions to create
        add_pallet_jack: Whether to add Pallet_Jack for material handling
        layout: Arrangement pattern ("grid", "linear", or "L-shaped")

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    jack_text = "Add a Pallet_Jack nearby for material handling." if add_pallet_jack else ""

    return f"""Add a pallet staging area near {near_equipment}.

SPECIFICATIONS:
- Reference equipment: {near_equipment}
- Number of pallet positions: {num_pallets}
- Layout pattern: {layout}

PALLET ARRANGEMENT:
- Grid: 2 columns, pallets arranged in rows
- Linear: Single row of pallets
- L-shaped: Pallets arranged in an L pattern for corner access

{jack_text}

PLACEMENT RULES:
- Keep 2.0m minimum clearance from conveyors for forklift access
- Pallets should be accessible from the main aisle
- Maintain clear pathways between pallet positions"""


def add_machining_cell(
    edge: str = "north",
    position: float = 0.5,
    add_scissor_lift: bool = True,
    add_power_cutter: bool = False
) -> str:
    """Generate request for adding a machining cell outside the conveyor loop.

    Args:
        edge: Which edge to place the cell on ("north", "south", "east", "west")
        position: Position along edge (0.0-1.0)
        add_scissor_lift: Whether to add ScissorLift for height adjustment
        add_power_cutter: Whether to add Power_Cutter for cutting operations

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    extras = []
    if add_scissor_lift:
        extras.append("ScissorLift for height-adjustable material positioning")
    if add_power_cutter:
        extras.append("Power_Cutter for cutting operations")

    extras_text = "\n- ".join(extras) if extras else "No additional equipment"

    return f"""Add a MFG_Equip_30ftx7ft_w_Exhaust machining cell on the {edge} edge.

SPECIFICATIONS:
- Edge: {edge}
- Position: {int(position * 100)}% along the edge
- Cell type: Heavy machining with exhaust ventilation

ADDITIONAL EQUIPMENT:
- {extras_text}

PLACEMENT RULES:
- Position outside the conveyor loop with 2.0m clearance
- Exhaust system should vent away from the main production area
- Maintain clear access paths for material loading"""


def add_robot_workstation(
    near_conveyor: str,
    robot_type: str = "ur10e",
    add_work_table: bool = True,
    add_shelving: bool = True,
    add_camera_stand: bool = False
) -> str:
    """Generate request for adding a standalone robot workstation.

    Args:
        near_conveyor: Conveyor ID to position the workstation near
        robot_type: Robot type ("ur10e" for collaborative, "abb_irb2600" for industrial)
        add_work_table: Whether to add Table_6ft for work surface
        add_shelving: Whether to add ShelvingRack for parts storage
        add_camera_stand: Whether to add Camera_Stand_7ft for vision system

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    robot_name = "ur10e_robot" if robot_type == "ur10e" else "abb_irb2600_12_165"
    robot_desc = "collaborative robot (safe for human interaction)" if robot_type == "ur10e" else "industrial robot (requires safety perimeter)"

    extras = []
    if add_work_table:
        extras.append("Table_6ft work surface for part processing")
    if add_shelving:
        extras.append("ShelvingRack for parts and tool storage")
    if add_camera_stand:
        extras.append("Camera_Stand_7ft for vision-guided operations")

    extras_text = "\n- ".join(extras) if extras else "No additional equipment"

    return f"""Add a {robot_name} workstation near {near_conveyor}.

SPECIFICATIONS:
- Reference conveyor: {near_conveyor}
- Robot type: {robot_name} ({robot_desc})

WORKSTATION EQUIPMENT:
- {extras_text}

PLACEMENT RULES:
- Position robot within reach of the conveyor for pick/place operations
- Work table should be within robot reach envelope
- Shelving positioned for easy restocking access
- {"Maintain 1.5m safety perimeter around industrial robot" if robot_type == "abb_irb2600" else "Collaborative robot can work alongside humans"}"""


def add_quality_station(
    near_conveyor: str,
    add_camera: bool = True,
    add_table: bool = True,
    add_ventilation: bool = False
) -> str:
    """Generate request for adding a quality inspection station.

    Args:
        near_conveyor: Conveyor ID to position the station near
        add_camera: Whether to add Camera_Stand_7ft for visual inspection
        add_table: Whether to add Table_6ft for inspection surface
        add_ventilation: Whether to add VentilatorFan_Straight for fume extraction

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    extras = []
    if add_camera:
        extras.append("Camera_Stand_7ft for automated visual inspection")
    if add_table:
        extras.append("Table_6ft inspection surface with good lighting access")
    if add_ventilation:
        extras.append("VentilatorFan_Straight for fume/dust extraction")

    extras_text = "\n- ".join(extras) if extras else "No inspection equipment"

    return f"""Add a quality inspection station near {near_conveyor}.

SPECIFICATIONS:
- Reference conveyor: {near_conveyor}
- Purpose: Quality control and inspection

STATION EQUIPMENT:
- {extras_text}

PLACEMENT RULES:
- Position for easy access from the conveyor
- Ensure adequate lighting for visual inspection
- Camera should have clear view of inspection area
- Table height suitable for standing inspection work"""


def add_safety_perimeter(
    around_equipment: str,
    num_sections: int = 4,
    include_gate: bool = True
) -> str:
    """Generate request for adding safety railing around hazardous equipment.

    Args:
        around_equipment: Equipment ID to fence (e.g., robot cell, machining center)
        num_sections: Number of SafetyRailing_8ft sections to use
        include_gate: Whether to leave a gap for access gate

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    gate_text = "Leave one section open for access gate with safety interlock." if include_gate else "Complete enclosure with no gaps."

    return f"""Add SafetyRailing_8ft perimeter around {around_equipment}.

SPECIFICATIONS:
- Equipment to fence: {around_equipment}
- Number of railing sections: {num_sections}
- Gate access: {"Yes" if include_gate else "No"}

PERIMETER LAYOUT:
- Arrange railing sections to form a rectangular enclosure
- {gate_text}
- Maintain 1.0m minimum clearance from equipment to railing

SAFETY REQUIREMENTS:
- Railing height must meet safety standards
- Gate should have safety interlock if present
- Clear warning signage on all sides"""


def add_workbench_area(
    edge: str = "south",
    position: float = 0.5,
    num_tables: int = 2,
    add_shelving: bool = True
) -> str:
    """Generate request for adding a manual workbench area.

    Args:
        edge: Which edge to place the area on ("north", "south", "east", "west")
        position: Position along edge (0.0-1.0)
        num_tables: Number of Table_6ft workbenches
        add_shelving: Whether to add ShelvingRack for tool/part storage

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    shelving_text = "Add ShelvingRack behind the workbenches for tool and part storage." if add_shelving else ""

    return f"""Add a manual workbench area on the {edge} edge.

SPECIFICATIONS:
- Edge: {edge}
- Position: {int(position * 100)}% along the edge
- Number of workbenches: {num_tables} Table_6ft units

WORKBENCH LAYOUT:
- Arrange tables side by side for collaborative work
- {shelving_text}

PLACEMENT RULES:
- Position outside the conveyor loop
- Ensure good lighting and electrical access
- Maintain clear floor space for worker movement
- Tables should face the production area for visibility"""


def add_ventilation(
    near_equipment: str,
    exhaust_direction: str = "up"
) -> str:
    """Generate request for adding ventilation near equipment.

    Args:
        near_equipment: Equipment ID to ventilate (e.g., welding station, paint booth)
        exhaust_direction: Direction of exhaust ("up", "out", or "filtered")

    Returns:
        Natural language request for the hybrid reconfiguration system
    """
    direction_desc = {
        "up": "vertical exhaust through ceiling duct",
        "out": "horizontal exhaust through wall",
        "filtered": "filtered recirculation back to room"
    }

    return f"""Add VentilatorFan_Straight ventilation near {near_equipment}.

SPECIFICATIONS:
- Reference equipment: {near_equipment}
- Exhaust direction: {exhaust_direction} ({direction_desc.get(exhaust_direction, 'standard exhaust')})

VENTILATION SETUP:
- Position fan to capture fumes/dust at source
- Direct airflow away from worker breathing zones
- Ensure adequate air replacement for exhaust volume

PLACEMENT RULES:
- Mount fan within 1.0m of emission source
- Do not obstruct equipment access or operation
- Connect to building exhaust system if available"""
