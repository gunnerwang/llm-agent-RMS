"""
Multi-agent reconfiguration workflow using function tools.

This provides a more sophisticated reconfiguration approach compared to the single-agent version:
- Process_planner: Analyzes requirements and decides what equipment to add/remove/modify
- Layout_engineer: Determines spatial placement and connections
- Engineer: Executes the commands against the LayoutBuilder

More collaborative reasoning while maintaining stability through command-based execution.
"""

import autogen
from autogen import GroupChat, GroupChatManager
from copy import deepcopy

# Apply Gemini patch to fix max_output_tokens issue
try:
    from integrations import gemini_patch
except ImportError:
    print("⚠️ Warning: gemini_patch.py not found - Gemini models may have truncated outputs")

from layout.layout_tools import LayoutBuilder
from catalog.equipment_catalog import generate_equipment_constraint_prompt
from agents.agents import get_model_config, get_rationale_config, is_termination_msg


def create_multiagent_reconfiguration_agents(
    layout_builder: LayoutBuilder,
    model_name="gpt-4o",
    reasoning_effort="none"
):
    """Create multi-agent team for sophisticated reconfiguration using tools

    Args:
        layout_builder: LayoutBuilder initialized with existing layout
        model_name: Name of the LLM model to use
        reasoning_effort: For Gemini models, controls thinking depth ("low", "medium", "high", "none"). 
                         Internally mapped to thinking_level for native Gemini API. Defaults to "none".

    Returns:
        Tuple of (user_proxy, process_planner, layout_engineer, reconfiguration_engineer)
    """

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="A human admin.",
        is_termination_msg=is_termination_msg,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    # Get current layout summary for context
    layout_summary = layout_builder.get_current_layout()
    equipment_prompt = generate_equipment_constraint_prompt()

    # Build equipment type info
    equipment_by_type = layout_summary.get('equipment_by_type', {})
    equipment_type_summary = "\n".join([
        f"  - {base}: {', '.join(instances)}"
        for base, instances in equipment_by_type.items()
    ])

    # Process Planner - analyzes requirements and decides on equipment changes
    process_planner = autogen.AssistantAgent(
        name="Reconfiguration_Planner",
        llm_config=get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort),
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message=f"""Manufacturing Reconfiguration Planner. You are part of a multi-agent team reconfiguring an existing manufacturing layout.

Your role: Analyze the user's reconfiguration requirements and determine what equipment changes are needed.

BASELINE LAYOUT SUMMARY:
- Total equipment: {layout_summary['total_objects']}
- Equipment by type:
{equipment_type_summary}

{equipment_prompt}

VALID FACING VALUES (MUST use one of these exactly):
- north_wall
- south_wall
- east_wall
- west_wall

CRITICAL CONSTRAINTS - MUST FOLLOW:

1. MATERIAL FLOW CONTINUITY:
   - Conveyor loops (Line_04, Line_05) are CRITICAL INFRASTRUCTURE
   - Before removing conveyors, verify they don't have connections
   - Equipment with connections is part of the material flow loop - DO NOT REMOVE
   - Instead, RELOCATE connected equipment if it blocks required space

2. UNDERSTANDING "OPEN PASSAGE":
   - "Open passage" = empty WALKWAY SPACE for human workers to access equipment
   - NOT the same as material flow path (conveyors)
   - To create passage: relocate blocking equipment to perimeter, preserve flow loops
   - Passage width: 1-2 meters for worker access

3. UNDERSTANDING "INCREASE THROUGHPUT" / "PARALLEL PROCESSING":
   - Identify the BOTTLENECK equipment (often Line_07 assembly cells)
   - Add a PARALLEL processing cell of the same type
   - Create a T-branch BEFORE the bottleneck with new conveyors
   - Merge back AFTER the parallel cells
   - NAMING: If Line_07_1, Line_07_2 exist, the new one is Line_07_3
   - NAMING: If Line_04_1 through Line_04_6 exist, new branch is Line_04_7

4. UNDERSTANDING "EXTEND CONVEYOR LOOP":
   - Expand the OUTER perimeter of the existing rectangular loop
   - Add new conveyors (Line_04/Line_05) OUTSIDE the current loop boundary
   - Place new stations (Line_06, Line_07) INLINE on the new conveyors
   - INLINE means the station sits ON TOP of the conveyor (pass-through design)
   - Reconnect the extended path back to the existing loop at corners (Line_03)
   - The extended loop should form a larger rectangle, not a floating branch

5. EQUIPMENT RELOCATION PRIORITY (for "open passage"):
   - High priority to relocate: Line_02 (workstations), Line_06/Line_08 (support)
   - Low priority to relocate: Line_07 (large robots), Line_04/Line_05 (if connected)
   - DO NOT remove: Any equipment with material flow connections

YOUR TASK:
1. Analyze the user's requirements carefully
2. Review the baseline layout - note EXISTING equipment IDs
3. Decide what equipment should be:
   - RELOCATED (UPDATE_PLACEMENT): Equipment blocking passage space
   - ADDED: New equipment with UNIQUE IDs (check existing IDs!)
   - REMOVED: Only if truly redundant AND has no connections
4. Explain your reasoning for each decision

OUTPUT FORMAT:
Provide structured reasoning, then list commands:
- UPDATE_PLACEMENT: object_id | placement_description
- REMOVE_EQUIPMENT: object_id
- ADD_EQUIPMENT: equipment_name | facing | placement | process_stage | function
  (facing MUST be: north_wall, south_wall, east_wall, or west_wall)

Example for "increase throughput at assembly bottleneck":
```
ANALYSIS: Assembly stage (Line_07) is the bottleneck.
Existing assembly cells: Line_07_1, Line_07_2
Existing conveyors: Line_04_1 through Line_04_6, Line_05_1 through Line_05_6

STRATEGY:
1. Add Line_07_3 as parallel assembly cell (next available ID)
2. Add Line_04_7 as branch conveyor to feed Line_07_3
3. Add Line_05_7 as outfeed from Line_07_3 back to main flow

COMMANDS:
ADD_EQUIPMENT: Line_07 | east_wall | left of Line_07_1, not adjacent | assembly | parallel assembly cell
ADD_EQUIPMENT: Line_04 | east_wall | in front of Line_04_3, not adjacent | conveyance | branch conveyor
ADD_EQUIPMENT: Line_05 | east_wall | behind Line_07_3, adjacent | conveyance | outfeed conveyor
ADD_CONNECTION: Line_04_3 -> Line_04_7 | right_negative -> primary_forward_negative
ADD_CONNECTION: Line_04_7 -> Line_07_3 | primary_forward_positive -> primary_forward_negative
ADD_CONNECTION: Line_07_3 -> Line_05_7 | primary_forward_positive -> primary_forward_negative
```

IMPORTANT: Always use UNIQUE object IDs. Check the baseline equipment list above before adding!
"""
    )

    # Layout Engineer - handles spatial placement and connections
    layout_engineer = autogen.AssistantAgent(
        name="Reconfiguration_Layout_Engineer",
        llm_config=get_rationale_config(model_name, temperature=0.7, reasoning_effort=reasoning_effort),
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message=f"""Manufacturing Layout Engineer. You work with the Reconfiguration Planner to optimize spatial layout.

Your role: Take the Planner's equipment decisions and add detailed spatial placement and connection specifications.

BASELINE LAYOUT SUMMARY:
- Equipment by type:
{equipment_type_summary}

VALID FACING VALUES (MUST use exactly one of these):
- north_wall
- south_wall
- east_wall
- west_wall

PLACEMENT RULES:
- For objects: "on", "left of", "right of", "in front", "behind", "under", "through"
- For room layout: "on" or "in the corner"
- Use "through" or "on" when placing stations (Line_06, Line_07) INLINE with conveyors
- Connected equipment MUST be placed adjacent to each other

INLINE STATION PLACEMENT (CRITICAL for extending loops):
- Processing stations (Line_06, Line_07) must be placed ON conveyors, not floating in space
- The conveyor belt physically runs THROUGH the station (pass-through design)
- Correct: "Line_07_3 | on Line_04_7" or "Line_07_3 | through Line_04_7"
- WRONG: "Line_07_3 | in front of Line_05_6" (this creates a floating disconnected station)

PROXIMITY:
- Adjacent: Equipment is physically touching or very close
- Not Adjacent: Equipment is separated with space

CONNECTION ENDPOINTS (REQUIRED):
- Straight flow: source="primary_forward_positive", target="primary_forward_negative"
- T-branch from side: source="right_negative" -> target="primary_forward_negative"
- Corners (Line_03): use "forward_positive"/"forward_negative"

YOUR TASK:
1. Review the Planner's equipment change recommendations
2. VALIDATE facing values - must be north_wall, south_wall, east_wall, or west_wall
3. Add detailed placement specifications for:
   - New equipment: Specify exact placement relative to walls or existing equipment
   - Modified equipment: Specify new placement
4. Add connection specifications between equipment
5. Ensure spatial consistency (connected equipment must be adjacent)

OUTPUT FORMAT:
First explain your spatial layout strategy, then output commands:

- ADD_EQUIPMENT: equipment_name | facing | placement | process_stage | function
  (facing MUST be: north_wall, south_wall, east_wall, or west_wall - NOT "parallel to X")

- UPDATE_PLACEMENT: object_id | detailed_placement_description
  Example: UPDATE_PLACEMENT: Line_07_1 | in front of Line_05_1, adjacent

- ADD_CONNECTION: source_id -> target_id | source_endpoint -> target_endpoint
  Example: ADD_CONNECTION: Line_05_1 -> Line_07_1 | primary_forward_positive -> primary_forward_negative

Include all of the Planner's commands (with corrected facing values) plus your additions.
"""
    )

    # Reconfiguration Engineer - executes commands
    reconfiguration_engineer = autogen.AssistantAgent(
        name="Reconfiguration_Engineer",
        llm_config=get_rationale_config(model_name, temperature=0.0, reasoning_effort=reasoning_effort),
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        system_message=f"""Reconfiguration Engineer. You execute the reconfiguration commands from the team.

Your role: Collect all commands from the Planner and Layout Engineer and output the FINAL COMPLETE command list.

COMMAND TYPES:
1. REMOVE_EQUIPMENT: object_id
2. ADD_EQUIPMENT: equipment_name | facing | placement | process_stage | function
3. UPDATE_PLACEMENT: object_id | placement_description
4. ADD_CONNECTION: source_id -> target_id | source_endpoint -> target_endpoint

YOUR TASK:
1. Review all commands from Planner and Layout Engineer
2. Consolidate into a single, complete command list
3. Remove any duplicates
4. Ensure logical order:
   - REMOVE commands first
   - ADD commands second
   - UPDATE_PLACEMENT commands third
   - ADD_CONNECTION commands last
5. Output ONLY the commands, no additional explanation

OUTPUT FORMAT (commands only):
```
REMOVE_EQUIPMENT: Line_04_2
ADD_EQUIPMENT: Line_05 | south_wall | near south wall | material_flow | Wide conveyor
UPDATE_PLACEMENT: Line_07_1 | left of Line_05_1, adjacent
ADD_CONNECTION: Line_05_1 -> Line_07_1 | primary_forward_positive -> primary_forward_negative
```

CRITICAL: Output ONLY commands in the format above. No explanatory text, just the command list.
"""
    )

    return user_proxy, process_planner, layout_engineer, reconfiguration_engineer


def build_multiagent_reconfiguration_prompt(user_requirements, layout_summary):
    """Build prompt for multi-agent reconfiguration workflow

    Args:
        user_requirements: User's reconfiguration requirements
        layout_summary: Dictionary with layout summary

    Returns:
        Formatted prompt string
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


class MultiAgentReconfigurationChat(GroupChat):
    """Custom group chat for multi-agent reconfiguration workflow"""

    def select_speaker(self, last_speaker, selector):
        """Control speaking order: Planner → Layout Engineer → Engineer"""

        if last_speaker.name == "Admin":
            return self.agent_by_name("Reconfiguration_Planner")
        elif last_speaker.name == "Reconfiguration_Planner":
            return self.agent_by_name("Reconfiguration_Layout_Engineer")
        elif last_speaker.name == "Reconfiguration_Layout_Engineer":
            return self.agent_by_name("Reconfiguration_Engineer")
        else:
            # Reconfiguration_Engineer has spoken, we're done
            return None
