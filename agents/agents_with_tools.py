"""
Updated agent creation using LayoutBuilder function tools.
This approach is more stable than having agents output large JSON structures.
"""

import autogen
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from copy import deepcopy

# Apply Gemini patch to fix max_output_tokens issue
try:
    from integrations import gemini_patch
except ImportError:
    print("⚠️ Warning: gemini_patch.py not found - Gemini models may have truncated outputs")

from layout.layout_tools import LayoutBuilder, create_layout_tool_functions
from catalog.equipment_catalog import generate_equipment_constraint_prompt
from agents.agents import get_model_config, get_rationale_config, is_termination_msg


def create_agents_with_tools(no_of_objects: int, layout_builder: LayoutBuilder, model_name="gpt-4o", reasoning_effort="none"):
    """Create agents that use function tools to build layouts incrementally

    Args:
        no_of_objects: Number of objects in the layout
        layout_builder: LayoutBuilder instance to operate on
        model_name: Name of the LLM model to use
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".

    Returns:
        Tuple of (user_proxy, process_planner, layout_engineer)
    """

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="A human admin.",
        is_termination_msg=is_termination_msg,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    # Get base config
    base_config = get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort)

    # Create tool functions from layout builder
    tool_functions = create_layout_tool_functions(layout_builder)

    # Register tools with autogen
    # For autogen, we need to format tools as function specs
    tools = []
    for tool_def in tool_functions:
        tools.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                # Parameters would be extracted from function signature
                # For simplicity, we'll document them in system message
            }
        })

    process_planner = autogen.AssistantAgent(
        name="Process_planner",
        llm_config=base_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message=f"""Manufacturing Process Planner. Design an end-to-end production workflow for the user's requested manufacturing objective.

You have access to FUNCTION TOOLS to build the layout incrementally. DO NOT output raw JSON - use these tools instead:

AVAILABLE TOOLS:
1. list_available_equipment() - See all available equipment from catalog
2. add_equipment(equipment_name, facing, placement_room_layout, placement_objects, process_stage, primary_function, flow_requirements)
   - equipment_name: MUST match catalog exactly (e.g., "Line_04")
   - facing: Direction (e.g., "south_wall", "north_wall")
   - placement_room_layout: JSON string like '[{{"layout_element_id": "south_wall", "preposition": "in the corner"}}]'
   - placement_objects: JSON string like '[{{"object_id": "Line_04_1", "preposition": "in front", "is_adjacent": true}}]'
   - process_stage: One of ["material_flow", "machining_fabrication", "assembly_collaboration", "quality_finishing"]
   - primary_function: Why this equipment is needed
   - flow_requirements: Adjacency, safety, or handling considerations
3. get_current_layout() - Check what's been added so far
4. validate_layout() - Validate current layout

{generate_equipment_constraint_prompt()}

WORKFLOW:
1. First, call list_available_equipment() to see available options
2. Select {no_of_objects} essential equipment items
3. For each item, call add_equipment() with appropriate parameters
4. Explain your reasoning for each selection
5. When done, call validate_layout() to check for errors

IMPORTANT:
- Call add_equipment() once for each piece of equipment (quantity=1 for each)
- Use exact equipment names from the catalog
- Process stage MUST be one of the four canonical identifiers listed above
- Build a connected workflow with clear upstream → downstream relationships

RESPONSE FORMAT:
Provide structured reasoning explaining your workflow design, then use the function tools to add equipment.
Do NOT output JSON directly - the tools will build it for you.
"""
    )

    layout_engineer = autogen.AssistantAgent(
        name="Layout_engineer",
        llm_config=base_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message=f"""Manufacturing Layout Engineer. Translate the proposed workflow into spatial layout using FUNCTION TOOLS.

You have access to these FUNCTION TOOLS to configure placement and connections:

AVAILABLE TOOLS:
1. get_current_layout() - See current equipment list
2. get_equipment_details(object_id) - Get details of specific equipment
3. update_placement(object_id, placement_room_layout, placement_objects)
   - Updates spatial relationships for equipment
   - placement_room_layout: JSON string for wall/room relationships
   - placement_objects: JSON string for object-to-object relationships
4. add_connection(source_id, target_id, source_endpoint, target_endpoint)
   - Creates material flow connection
   - Endpoints: "primary_forward_positive" (front), "primary_forward_negative" (back), "right_negative" (side)
5. set_facing(object_id, facing)
   - Set facing direction (south_wall, north_wall, east_wall, west_wall)
6. validate_layout() - Check for errors

CRITICAL PLACEMENT RULES:
- If equipment A connects to equipment B, their placement MUST reference each other
- Example: if Line_04_1 connects to Line_01_1, use update_placement to set:
  placement_objects='[{{"object_id": "Line_01_1", "preposition": "in front", "is_adjacent": true}}]'
- Connected equipment must be physically adjacent

PREPOSITION RULES:
- For objects: "on", "left of", "right of", "in front", "behind", "under", "through"
- For room layout: "on" or "in the corner"
- Use "through" when conveyor passes through workstation

PROXIMITY:
- Adjacent: Equipment is physically touching or very close
- Not Adjacent: Equipment is separated with space

CONNECTION ENDPOINTS:
- Straight flow: source="primary_forward_positive", target="primary_forward_negative"
- T-branch from side: source="right_negative"
- Corners: use "forward_positive"/"forward_negative"

WORKFLOW:
1. Call get_current_layout() to see all equipment
2. For each equipment, call update_placement() to set spatial relationships
3. For connected equipment, call add_connection() to establish material flow
4. Ensure placement matches connections (adjacent placement for connected equipment)
5. Call validate_layout() when done

RESPONSE FORMAT:
First explain your spatial layout strategy, then use the function tools to configure placement and connections.
Do NOT output JSON directly - use the tools.
"""
    )

    # Register the actual function implementations
    # This allows agents to call these functions during conversation
    for tool_def in tool_functions:
        user_proxy.register_function(
            function_map={
                tool_def["name"]: tool_def["function"]
            }
        )

    return user_proxy, process_planner, layout_engineer


def create_reconfiguration_agent_with_tools(layout_builder: LayoutBuilder, model_name="gpt-4o", reasoning_effort="none"):
    """Create reconfiguration agent that uses function tools

    Args:
        layout_builder: LayoutBuilder initialized with existing layout
        model_name: Name of the LLM model to use
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".

    Returns:
        Tuple of (user_proxy, reconfiguration_agent)
    """

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="A human admin.",
        is_termination_msg=is_termination_msg,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    base_config = get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort)

    # Create tool functions
    tool_functions = create_layout_tool_functions(layout_builder)

    reconfiguration_agent = autogen.AssistantAgent(
        name="Reconfiguration_Agent",
        llm_config=base_config,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message="""You are an expert Manufacturing Layout Engineer specialized in reconfiguration.

You have access to FUNCTION TOOLS to modify an existing layout:

AVAILABLE TOOLS:
1. get_current_layout() - See current equipment in baseline layout
2. get_equipment_details(object_id) - Get details of specific equipment
3. add_equipment(...) - Add new equipment
4. remove_equipment(object_id) - Remove equipment from layout
5. update_placement(object_id, ...) - Change equipment placement
6. add_connection(source_id, target_id, ...) - Add material flow connection
7. set_facing(object_id, facing) - Change facing direction
8. list_available_equipment() - See catalog options
9. validate_layout() - Check for errors

RECONFIGURATION STRATEGY:
1. First, call get_current_layout() to understand baseline
2. Identify what needs to change based on user requirements
3. Use remove_equipment() to delete unnecessary equipment
4. Use add_equipment() to add new equipment
5. Use update_placement() to relocate equipment (change position via placement relationships)
6. Use add_connection() to update material flow
7. Call validate_layout() to verify changes

IMPORTANT:
- This is a PHYSICAL layout - to reconfigure:
  * To create space/passageway → use remove_equipment()
  * To add capability → use add_equipment()
  * To relocate → use update_placement() with new spatial relationships
- Minimize unnecessary changes - only modify what's needed
- After modifying equipment, update connections to match new layout

RESPONSE FORMAT:
1. Explain your reconfiguration strategy
2. Use the function tools to make changes
3. Do NOT output JSON directly - the tools handle that

When you're finished reconfiguring, call validate_layout() to verify.
"""
    )

    # Register functions
    for tool_def in tool_functions:
        user_proxy.register_function(
            function_map={
                tool_def["name"]: tool_def["function"]
            }
        )

    return user_proxy, reconfiguration_agent
