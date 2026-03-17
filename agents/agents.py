import autogen
from autogen.agentchat.agent import Agent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.assistant_agent import AssistantAgent
import json
from jsonschema import validate
from copy import deepcopy
import ast

# Apply Gemini patch to fix max_output_tokens issue
try:
    from integrations import gemini_patch
except ImportError:
    print("⚠️ Warning: gemini_patch.py not found - Gemini models may have truncated outputs")

from core.schemas import initial_schema, interior_architect_schema, interior_designer_schema, engineer_schema
from catalog.equipment_catalog import generate_equipment_constraint_prompt

# OAI_CONFIG_LIST.json is needed! Check the Autogen repo for more info!
def get_config_list(model_name="gpt-4o"):
    """Get configuration list for specified model"""
    supported_models = [
        "gpt-4o",
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5-nano",
        "deepseek-chat",
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")

    return autogen.config_list_from_json(
        "OAI_CONFIG_LIST.json",
        filter_dict={
            "model": [model_name],
        },
    )

def get_model_config(model_name="gpt-4o", reasoning_effort="none"):
    """Get model configuration for specified model

    Args:
        model_name: Name of the model to use
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none").
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".
    """
    config_list = get_config_list(model_name)

    base_config = {
        "cache_seed": 42,
        "top_p": 1.0,
        "config_list": config_list,
        "timeout": 600,
    }

    # GPT-5-mini and GPT-5-nano only support temperature=1 (default)
    # Other models can use custom temperature
    if model_name not in ["gpt-5-mini", "gpt-5-nano"]:
        base_config["temperature"] = 0.7

    # GPT-5 models use max_completion_tokens instead of max_tokens
    if model_name.startswith("gpt-5"):
        base_config["max_completion_tokens"] = 8192
    elif model_name.startswith("gemini"):
        # Gemini models use max_tokens (AG2 maps it to max_output_tokens internally)
        # Gemini 2.0/2.5 supports up to 8192 tokens output (doubled for large responses)
        base_config["max_tokens"] = 8192
        # Also set in config_list for direct API parameter passing
        base_config["config_list"][0]["max_tokens"] = 8192
    else:
        base_config["max_tokens"] = 8192  # Increased for large layout revisions

    # Add thinking_level for Gemini models (native Google API parameter)
    # Note: reasoning_effort is for OpenAI-compatible API and gets mapped to thinking_level
    if reasoning_effort and model_name.startswith("gemini"):
        if reasoning_effort not in ["low", "medium", "high", "none"]:
            raise ValueError(f"reasoning_effort must be 'low', 'medium', 'high', or 'none', got: {reasoning_effort}")
        # Only add thinking_level if it's not "none"
        if reasoning_effort != "none":
            base_config["config_list"][0]["thinking_level"] = reasoning_effort

    return base_config

def get_json_config(model_name="gpt-4o", temperature=0.7, force_json=True, reasoning_effort="none"):
    """Get JSON configuration for specified model

    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        force_json: Whether to enforce JSON output format
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".
    """
    base_config = get_model_config(model_name, reasoning_effort=reasoning_effort)
    json_config = deepcopy(base_config)

    # GPT-5-mini and GPT-5-nano only support temperature=1 (default)
    # Don't override temperature for these models
    if model_name not in ["gpt-5-mini", "gpt-5-nano"]:
        json_config["temperature"] = temperature

    # Only enforce JSON format if explicitly requested
    if force_json:
        json_config["config_list"][0]["response_format"] = { "type": "json_object" }

    return json_config

def get_rationale_config(model_name="gpt-4o", temperature=0.7, reasoning_effort="none"):
    """Get configuration that allows rationale (no forced JSON)
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature parameter for generation
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".
    """
    return get_json_config(model_name, temperature, force_json=False, reasoning_effort=reasoning_effort)

def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and content["name"] == "Json_schema_debugger" and "SUCCESS" in content["content"]:
        return True
    return False


def build_initial_plan_prompt(user_requirements, dimension_sentence):
    return f"""
            {dimension_sentence}
            User manufacturing objectives, constraints, or preferences (in triple backquotes):
            ```
            {user_requirements}    
            ```
            Workspace layout elements available for reference (in triple backquotes):
            ```
            ['south_wall', 'north_wall', 'west_wall', 'east_wall', 'middle of the room', 'ceiling']
            ```
            
            Please provide a HIGH-LEVEL PLAN for this manufacturing setup including:
            1. Process Planner: Equipment selection and workflow reasoning
            2. Layout Engineer: Spatial arrangement strategy and safety considerations
            
            This is a PLANNING phase - do NOT generate detailed JSON yet. Focus on the conceptual approach.
            """


def build_simple_reconfiguration_prompt(user_requirements, layout_summary):
    """Build reconfiguration prompt with layout summary instead of full JSON

    Args:
        user_requirements: User's reconfiguration requirements
        layout_summary: Dictionary with layout summary (not full JSON)
    """
    equipment_prompt = generate_equipment_constraint_prompt()

    # Format the summary nicely
    equipment_list = layout_summary.get("equipment_list", [])
    equipment_by_type = layout_summary.get("equipment_by_type", {})

    summary_text = f"Total equipment: {layout_summary.get('total_objects', 0)}\n"
    summary_text += f"Equipment IDs: {', '.join(equipment_list)}\n\n"
    summary_text += "Equipment by type:\n"
    for base_name, instances in equipment_by_type.items():
        summary_text += f"  - {base_name}: {', '.join(instances)}\n"

    return f"""
{user_requirements}

{equipment_prompt}

BASELINE LAYOUT SUMMARY:
```
{summary_text}
```
"""

class JSONSchemaAgent(UserProxyAgent):
    def __init__(self, name : str, is_termination_msg):
        super().__init__(name, is_termination_msg=is_termination_msg, code_execution_config=False)

    def _ensure_schema_defaults(self, payload):
        """Fill missing placement fields so schema validation has required keys."""
        if not isinstance(payload, dict):
            return payload
        objects = payload.get("objects_in_room")
        if not isinstance(objects, list):
            return payload
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            placement = obj.get("placement")
            if not isinstance(placement, dict):
                placement = {}
                obj["placement"] = placement
            room_layout = placement.get("room_layout_elements")
            if not isinstance(room_layout, list):
                room_layout = []
            placement["room_layout_elements"] = room_layout
            object_rels = placement.get("objects_in_room")
            if not isinstance(object_rels, list):
                object_rels = []
            placement["objects_in_room"] = object_rels
        return payload

    def get_human_input(self, prompt: str) -> str:
        message = self.last_message()
        preps_layout = ['in front', 'on', 'in the corner', 'in the middle of']
        preps_objs = ['on', 'left of', 'right of', 'in front', 'behind', 'under', 'above', 'through']

        def parse_candidate(candidate: str):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                    return json.loads(json.dumps(parsed))
                except (ValueError, SyntaxError, TypeError) as err:
                    raise ValueError from err

        # Extract JSON from message that may contain rationale text
        try:
            # First try direct JSON parsing
            json_obj_new = parse_candidate(message["content"])
        except (json.JSONDecodeError, ValueError, SyntaxError):
            # If direct parsing fails, extract JSON from mixed content
            import re
            # First try to find JSON in ```json blocks
            json_block_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_block_pattern, message["content"], re.DOTALL)
            if match:
                try:
                    json_obj_new = parse_candidate(match.group(1))
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    pass
            else:
                # Look for any JSON-like structure (nested braces)
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, message["content"], re.DOTALL)
                for match in matches:
                    try:
                        json_obj_new = parse_candidate(match)
                        break
                    except (json.JSONDecodeError, ValueError, SyntaxError):
                        continue
                else:
                    raise ValueError(f"No valid JSON found in message: {message['content'][:200]}...")
        json_obj_new = self._ensure_schema_defaults(json_obj_new)
        try:
            json_obj_new_ids = [item["object_id"] for item in json_obj_new["objects_in_room"]]
        except Exception as parse_err:
            return f"Invalid JSON structure: ensure 'objects_in_room' is a list and each item has an 'object_id' field. Error: {parse_err}"

        is_success  = False
        try:
            validate(instance=json_obj_new, schema=initial_schema)
            is_success = True
        except Exception as e:
            feedback = str(e.message)
            if e.validator == "enum":
                if e.instance in json_obj_new_ids:
                    feedback += f" Put the {e.instance} object under 'objects_in_room' instead of 'room_layout_elements' and delete the {e.instance} object under 'room_layout_elements'"
                elif str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_objs}"
                elif str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_layout}"

        if is_success:
            return "SUCCESS"
        return feedback

def create_agents(no_of_objects : int, model_name="gpt-4o", reasoning_effort="none"):
    """Create agents for the manufacturing layout design system
    
    Args:
        no_of_objects: Number of objects in the layout
        model_name: Name of the LLM model to use
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".
    """
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message = "A human admin.",
        is_termination_msg = is_termination_msg,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    json_schema_debugger = JSONSchemaAgent(
        name = "Json_schema_debugger",
        is_termination_msg = is_termination_msg,
    )
    process_planner = autogen.AssistantAgent(
        name = "Process_planner",
        llm_config = get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort),
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Manufacturing Process Planner. Design an end-to-end production workflow for the user's requested manufacturing objective. Break the goal into concrete process stages and select {no_of_objects} essential equipment items from the catalog that collectively deliver the workflow within the available workspace.
        
        {generate_equipment_constraint_prompt()}

        FLOW DESIGN REQUIREMENTS:
        - Build a connected sequence of stages with explicit upstream → downstream relationships. Describe how material, totes, or parts travel from one equipment item to the next.
        - Whenever a conveyor or transport system is required, specify the complementary straight and corner modules so the path is continuous (e.g., pair Line_04/Line_05 straight conveyors with Line_03 curved conveyors to form loops or U-turns, use Line_01 lift transfer units and Line_02 roller bridges for transitions).
        - T-BRANCH RULE: Line_04 and Line_05 have side connections for T-branches. When creating a spur line from the main conveyor:
          1. The spur can include multiple conveyors but MUST terminate with a Line_01 (Lift Transfer Unit) at the end.
          2. Branch conveyors MUST be oriented perpendicular to the main conveyor (rotated 90° from main line direction).
        - Call out buffer/queue locations and shared work areas that multiple stages rely on; these will later be turned into aligned spatial clusters.
        
        For each process stage, capture the following (assume exactly ONE unit per catalog item for now—quantity should always be 1). IMPORTANT: `process_stage` must be one of these canonical snake_case identifiers so downstream simulators can read it: ["material_flow", "machining_fabrication", "assembly_collaboration", "quality_finishing"].

        1. `equipment_name` — MUST match a catalog entry exactly.
        2. `process_stage` — choose one canonical category from the list above that best represents the production step this equipment supports.
        3. `primary_function` — why this equipment is needed for that stage.
        4. `flow_requirements` — adjacency, safety, or handling considerations (e.g., needs staging buffer, requires forklift access).
        5. `quantity` — always set to 1 for every selected item in this simplified workflow.
        6. `supporting_notes` — any constraints downstream agents must respect (clearance, utilities, operators).

        IMPORTANT:
        - ONLY select equipment from the predefined catalog above. Use exact names, styles, materials, and sizes.
        - Do NOT invent new process stage labels; reuse the four canonical identifiers exactly as written.
        - Sequence the stages to optimize takt time, ergonomics, and safety (e.g., avoid cross-traffic, keep hazardous zones isolated) and make sure consecutive stages can sit on the same flow lane without breaks.
        - Highlight when flexible space (buffers, inspection tables, AGV paths) is mandatory to keep flow continuous.
        - Treat this as a manufacturing system design problem, not interior decoration.

        RESPONSE FORMAT:
        First, provide structured reasoning that explains the proposed workflow, throughput logic, and safety considerations.
        Then, provide the JSON output following this schema:
        {interior_designer_schema}

        """
    )


    layout_engineer = autogen.AssistantAgent(
        name = "Layout_engineer",
        llm_config = get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort),
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Manufacturing Layout Engineer. Translate the proposed workflow into a spatial layout that keeps materials, operators, and equipment moving safely and efficiently within the workspace dimensions.
        Analyse process dependencies, travel distances, and utilities to position every item suggested by the Process Planner.
        Assume there is exactly ONE instance for every equipment item (quantity is fixed at 1 for now) and give explicit answers for EACH object on the following aspects:

        Placement: 
        Identify a location that respects process sequence, flow lines, and safety envelopes (e.g., staging area near incoming dock, inspection cell downstream of assembly).
        Group consecutive stages on shared axes to form uninterrupted lanes; conveyors and transfer points must touch so that parts can flow without gaps.
        
        CRITICAL: PLACEMENT MUST MATCH CONNECTIONS!
        - If equipment A connects to equipment B in the material flow, then A's placement MUST reference B (or vice versa).
        - Example: if Line_04_1 connects to Line_01_1, then Line_04_1's placement should be "in front of Line_01_1" (adjacent).
        - This ensures connected equipment is physically positioned next to each other.
        - Do NOT place connected equipment relative to walls/room only - always reference the connected upstream/downstream neighbor.
        
        CONNECTION ENDPOINTS (REQUIRED for all connections):
        - Each connection MUST specify source_endpoint and target_endpoint.
        - Endpoint values: "primary_forward_positive" (front), "primary_forward_negative" (back), 
          "right_negative" (side for T-branch), "forward_positive"/"forward_negative" (for corners).
        - Straight-line flow: source="primary_forward_positive", target="primary_forward_negative".
        - T-branches from side: source="right_negative".
        - Corners (Line_03): use "forward_positive"/"forward_negative" with "right_negative".
        
        For relative placement with other equipment use ONLY the prepositions "on", "left of", "right of", "in front", "behind", "under", "through".
        Use "through" when a conveyor passes THROUGH a workstation (e.g., Line_04 through Line_07 robotic cell).
        For relationships to workspace layout elements (walls, floor center, ceiling) use ONLY "on" or "in the corner".
        Explicitly state the placement for each instance. When two elements belong to the same lane, reference the immediate upstream/downstream neighbor.

        Proximity: 
        Specify whether the object must stay adjacent or separated to maintain throughput, safety buffers, or maintenance access:
        1. Adjacent: The object is physically contacting the other object or it is supported by the other object or they are touching or they are close to each other.
        2. Not Adjacent: The object is not physically contacting the other object and it is distant from the other object.


        Facing:
        Indicate the wall (west/east/north/south_wall) or layout element the equipment should face to align operator entry, conveyors, or service access (ex. one is facing the south_wall). Keep shared lanes facing the same direction to signal a single connected workflow.

        RESPONSE FORMAT:
        First, explain your reasoning for the spatial layout decisions, highlighting material flow, safety clearances, maintenance access, and visibility.
        Then, provide the JSON output following this schema:
        {interior_architect_schema}

        JSON
        """
    )

    engineer = autogen.AssistantAgent(
        name = "Engineer",
        llm_config = get_json_config(model_name, temperature=0.0, reasoning_effort=reasoning_effort),
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Engineer. You listen to the input by the Admin and create a JSON file.
        Every time when the Admin outputs objects to be in the room you will save ALL of them in the given schema!
        For the scene graph, you can use the ids for the objects that are already in the room, but only output the objects to be placed!
        If the Json_schema_debugger reports a validation error about the JSON schema, solve the error in a way that preserves the manufacturing workflow and spatial logic!

        IMPORTANT: The inputted "Placement" key should be used for the "placement" key in the JSON object. Follow exactly the allowed prepositions,
        and do not use the information in the "Facing" key to encode room layout relationships!

        IMPORTANT: Assume quantity is always one for every catalog item in this simplified workflow—only encode single instances.
        
        CONNECTION ENDPOINTS (REQUIRED):
        - Every connection MUST include source_endpoint and target_endpoint.
        - Values: "primary_forward_positive" (front), "primary_forward_negative" (back), "right_negative" (side T-branch).
        - Straight flow: source="primary_forward_positive", target="primary_forward_negative".

        Output ONLY valid JSON following this schema (no additional text):
        {engineer_schema}

        """
    )

    return user_proxy, json_schema_debugger, process_planner, layout_engineer, engineer

def create_reconfiguration_agent(model_name="gpt-4o", reasoning_effort="none"):
    """Create a single agent for layout reconfiguration using command-based approach

    This agent outputs simple commands instead of large JSON structures for better stability.
    """

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message = "A human admin.",
        is_termination_msg = is_termination_msg,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    # Use rationale config instead of JSON config - agent will output commands, not JSON
    reconfiguration_agent = autogen.AssistantAgent(
        name = "Reconfiguration_Agent",
        llm_config = get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort),
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" You are an expert Manufacturing Layout Engineer specialized in reconfiguration.

        Your task is to reconfigure an existing manufacturing layout based on user requirements.

        You will be provided with:
        1. User requirements/objectives
        2. A catalog of available equipment
        3. A SUMMARY of the existing baseline layout (NOT the full JSON)

        IMPORTANT: Instead of outputting large JSON structures, you will use SIMPLE COMMANDS to modify the layout.

        AVAILABLE COMMANDS:

        1. ADD_EQUIPMENT: equipment_name | facing | placement_description | process_stage | primary_function
           Example: ADD_EQUIPMENT: Line_05 | south_wall | near south wall | material_flow | Conveyor transport

        2. REMOVE_EQUIPMENT: object_id
           Example: REMOVE_EQUIPMENT: Line_04_2

        3. UPDATE_PLACEMENT: object_id | placement_description
           Example: UPDATE_PLACEMENT: Line_07_1 | left of Line_05_1, adjacent

        4. ADD_CONNECTION: source_id -> target_id | source_endpoint -> target_endpoint
           Example: ADD_CONNECTION: Line_04_1 -> Line_05_1 | primary_forward_positive -> primary_forward_negative

        PLACEMENT DESCRIPTIONS:
        - "near south_wall" - Places near wall
        - "in the corner south_wall" - Places in corner
        - "middle of room" - Places in center
        - "in front of Line_04_1, adjacent" - Places in front of object, touching
        - "left of Line_05_1" - Places to left of object, not touching

        CRITICAL CONSTRAINTS - MUST FOLLOW:

        1. MATERIAL FLOW CONTINUITY:
           - Conveyor loops and material flow paths are CRITICAL INFRASTRUCTURE
           - Before removing ANY Line_04, Line_05, or conveyor equipment, check if it has connections
           - Equipment with connections is part of the material flow loop - DO NOT REMOVE
           - Instead, RELOCATE connected equipment if it blocks required space

        2. UNDERSTANDING "OPEN PASSAGE":
           - "Open passage" = empty WALKWAY SPACE for human workers to access equipment
           - NOT the same as material flow path (conveyors)
           - To create passage: move blocking equipment to perimeter, DO NOT delete flow-critical conveyors
           - Passage width: typically 1-2 meters for worker access

        3. EQUIPMENT TYPES:
           - Line_04, Line_05: Conveyors (usually part of flow loops) - RELOCATE, don't remove
           - Line_02: Workstations/stands - can be relocated for passages
           - Line_06, Line_08: Support equipment - can be relocated
           - Line_07: Large robotic cells - harder to move, plan around them

        RECONFIGURATION STRATEGY:
        1. Analyze baseline layout summary - identify equipment positions and connections
        2. For "open passage" requests:
           a. Determine passage location (e.g., north-south axis, center aisle)
           b. Identify equipment occupying passage space
           c. Check if blocking equipment has connections (is it in the flow loop?)
           d. If connected: USE UPDATE_PLACEMENT to move to perimeter, maintain connections
           e. If not connected: Can use REMOVE_EQUIPMENT or UPDATE_PLACEMENT
           f. Ensure 1-2m clear width for worker access
        3. Maintain all material flow connections
        4. Update connections if equipment was relocated

        RESPONSE FORMAT:
        1. First, analyze the baseline layout and explain your strategy
        2. Explicitly state which equipment you will RELOCATE vs REMOVE and why
        3. Output commands (one per line)
        4. Do NOT output JSON directly

        Example response for "open passage for workers":
        ```
        ANALYSIS: The baseline has a conveyor loop formed by Line_04 (horizontal) and Line_05 (vertical segments).
        Line_04_2 and Line_04_4 run through the center, blocking a north-south passage for workers.
        However, these conveyors have connections - they're part of the critical flow loop.

        STRATEGY: Instead of removing these conveyors, I will:
        1. Keep the conveyor loop intact
        2. Relocate some Line_02 workstations that block passage space
        3. This creates a clear 1.5m wide aisle while maintaining material flow

        COMMANDS:
        UPDATE_PLACEMENT: Line_02_1 | near east_wall
        UPDATE_PLACEMENT: Line_02_4 | near east_wall
        UPDATE_PLACEMENT: Line_02_5 | near west_wall
        ```

        WRONG example (DO NOT DO THIS):
        ```
        To create passage, I'll remove the inner conveyors:  ❌ WRONG
        REMOVE_EQUIPMENT: Line_04_2  ❌ Breaks flow loop!
        REMOVE_EQUIPMENT: Line_04_4  ❌ Breaks flow loop!
        ```
        """
    )

    return user_proxy, reconfiguration_agent
