"""
Hybrid Reconfiguration System

Combines LLM agent intelligence for understanding and planning with
structure-aware execution for reliable implementation.

Architecture:
1. LLM Planner: Understands user requests, analyzes baseline, proposes operations
2. Structure-Aware Engineer: Executes operations with proper placement & connectivity

This gives us:
- Natural language understanding from LLM
- Reliable, collision-free placement from structure-aware system
- Predictable results with proper spacing and connectivity
"""

import json
from typing import Dict, List, Optional, Any
import autogen
from reconfiguration.structure_aware_reconfig import StructureAwareReconfiguration, LoopStructure


# Define the operations that the structure-aware system can execute
AVAILABLE_OPERATIONS = """
Available reconfiguration operations:

=== Template Mode Operations ===

0. build_conveyor_system(layout, conveyors, inline_stations)
   - RECOMMENDED FIRST in template mode: creates flexible conveyor infrastructure
   - layout: "L-shape", "U-shape", "straight", "loop", or "custom"
   - conveyors: for custom layout, list of {"type": "Line_04", "start": [x,y], "end": [x,y]}
   - inline_stations: list of {"type": "<any_equipment>", "on_conveyor": 0, "position": 0.5}
     - type can be ANY catalog equipment: Line_01, Line_06, Line_07, Camera_Stand_7ft,
       Table_6ft, ur10e_robot, abb_irb2600_12_165, ScissorLift, etc.
     - on_conveyor: index of conveyor segment to place on (0-based)
     - position: 0.0-1.0 along conveyor

0b. build_new_loop(num_conveyors_per_edge, add_lift, add_inspection, add_assembly)
   - Legacy: creates rectangular loop (internally calls build_conveyor_system)

=== Layout Extension Operations ===

1. extend_edge(edge, stations)
   - Extends the conveyor loop outward on one edge
   - edge: "north", "south", "east", or "west"
   - stations: list of equipment types to add, e.g. ["Line_07", "Line_06", "Line_02"]
   - Line_07/Line_06 are placed inline on the new conveyor
   - Line_02 is placed as a side-loader adjacent to the conveyor

2. add_inline_station(conveyor_id, station_type)
   - Adds a processing station inline on an existing conveyor
   - conveyor_id: ID of target conveyor (e.g., "Line_04_3")
   - station_type: "Line_06" (inspection) or "Line_07" (assembly)

3. add_parallel_branch(from_conveyor, station_type)
   - Creates a parallel processing path for increased throughput
   - from_conveyor: ID of conveyor to branch from
   - station_type: type of station for the parallel path

=== Material Flow Definition ===

4. define_material_flow(loading_edge, loading_position, unloading_edge, unloading_position, flow_direction)
   - Defines material flow by creating a PHYSICAL passage (gap) in the conveyor loop
   - loading_edge: edge where materials enter ("north", "south", "east", "west")
   - loading_position: position along loading edge (0.0 to 1.0)
   - unloading_edge: edge where finished products exit
   - unloading_position: position along unloading edge (0.0 to 1.0)
   - flow_direction: "clockwise" or "counterclockwise"
   - Physically removes corner, conveyor segment, and any inline stations at the passage
   - Creates an open production line with clear loading (L) and unloading (U) points
   - The gap between U and L is the worker passage for accessing the cell interior

=== Extended Equipment Operations (Legacy Catalog) ===

5. add_pallet_staging(near_equipment_id, num_pallets, add_pallet_jack, layout)
    - Adds a pallet staging/buffer area near specified equipment
    - near_equipment_id: equipment to place staging near
    - num_pallets: number of Pallet positions (default 4)
    - add_pallet_jack: whether to add Pallet_Jack (default true)
    - layout: "grid", "linear", or "L-shaped" (default "grid")

6. add_machining_cell(edge, position, add_scissor_lift, add_power_cutter)
    - Adds MFG_Equip_30ftx7ft_w_Exhaust machining cell outside the loop
    - edge: which edge to place on ("north", "south", "east", "west")
    - position: position along edge (0.0 to 1.0)
    - add_scissor_lift: whether to add ScissorLift (default true)
    - add_power_cutter: whether to add Power_Cutter (default false)

7. add_robot_workstation(near_conveyor_id, robot_type, add_work_table, add_shelving, add_camera_stand, edge, position)
    - Adds standalone robot workstation near a conveyor or on specified edge
    - near_conveyor_id: conveyor to position near (optional in template mode)
    - robot_type: "ur10e" (collaborative) or "abb_irb2600" (industrial)
    - add_work_table: whether to add Table_6ft (default true)
    - add_shelving: whether to add ShelvingRack (default true)
    - add_camera_stand: whether to add Camera_Stand_7ft (default false)
    - edge: "north", "south", "east", "west" (for template mode positioning)
    - position: 0.0 to 1.0 position along edge (for template mode)

8. add_quality_station(near_conveyor_id, add_camera, add_table, add_ventilation, edge, position)
    - Adds quality inspection station near a conveyor or on specified edge
    - near_conveyor_id: conveyor to position near (optional in template mode)
    - add_camera: whether to add Camera_Stand_7ft (default true)
    - add_table: whether to add Table_6ft (default true)
    - add_ventilation: whether to add VentilatorFan_Straight (default false)
    - edge: "north", "south", "east", "west" (for template mode positioning)
    - position: 0.0 to 1.0 position along edge (for template mode)

9. add_safety_perimeter(around_equipment_id, num_sections, include_gate)
    - Adds SafetyRailing_8ft sections around hazardous equipment
    - around_equipment_id: equipment to fence
    - num_sections: number of railing sections (default 4)
    - include_gate: whether to leave gap for access (default true)

10. add_workbench_area(edge, position, num_tables, add_shelving)
    - Adds manual workbench area on specified edge
    - edge: which edge ("north", "south", "east", "west")
    - position: position along edge (0.0 to 1.0)
    - num_tables: number of Table_6ft tables (default 2)
    - add_shelving: whether to add ShelvingRack (default true)

11. add_ventilation(near_equipment_id, exhaust_direction)
    - Adds VentilatorFan_Straight near specified equipment
    - near_equipment_id: equipment to ventilate
    - exhaust_direction: "up", "out", or "filtered" (default "up")

"""


def create_reconfig_planning_prompt(user_request: str, baseline_summary: Dict,
                                     structure_summary: Dict, use_as_template: bool = False) -> str:
    """Create a prompt for the LLM to plan reconfiguration operations."""

    template_mode_note = ""
    if use_as_template:
        template_mode_note = """
IMPORTANT - TEMPLATE MODE:
You are in TEMPLATE MODE. The baseline layout provides STRUCTURAL PATTERNS (dimensions, edge organization)
but you will BUILD A COMPLETELY NEW LAYOUT from scratch - not copy any equipment from the baseline.

TEMPLATE MODE WORKFLOW:
1. FIRST: Call build_new_loop to create the base conveyor infrastructure
   - This creates NEW corners, conveyors, and basic stations based on the template's dimensions
   - Parameters: num_conveyors_per_edge (default 2), add_lift, add_inspection, add_assembly
2. THEN: Add additional equipment using operations like add_robot_workstation, add_workbench_area, etc.
3. Use edge and position parameters for placement (e.g., edge="east", position=0.5)

REQUIRED FIRST OPERATION IN TEMPLATE MODE:
- build_new_loop (creates fresh conveyor loop based on template dimensions)

AVAILABLE OPERATIONS AFTER build_new_loop:
- add_robot_workstation, add_quality_station, add_workbench_area, add_machining_cell
- add_pallet_staging, add_safety_perimeter, add_ventilation
- extend_edge (to expand the new loop further)
"""

    prompt = f"""You are a manufacturing layout reconfiguration planner.

USER REQUEST:
{user_request}
{template_mode_note}
TEMPLATE LAYOUT SUMMARY (for reference):
- Total equipment in template: {baseline_summary.get('total_objects', 0)} objects
- Room dimensions: {baseline_summary.get('room_dimensions', 'unknown')}

STRUCTURE ANALYSIS (learned from template):
- Loop corners: {structure_summary.get('corners', [])}
- Corner positions: {structure_summary.get('corner_positions', {})}
- Loop bounds: {structure_summary.get('bounds', {})}
- Edge contents:
{_format_edges(structure_summary.get('edges', {}))}

{AVAILABLE_OPERATIONS}

TASK:
Analyze the user's request and create a reconfiguration plan using the available operations.
Consider:
1. Which edge(s) to modify based on the request
2. What equipment to add and where
3. How to maintain proper flow and connectivity

OUTPUT FORMAT:
Return a JSON object with your plan:
{{
    "understanding": "Brief summary of what the user wants",
    "analysis": "Your analysis of how to achieve it",
    "operations": [
        {{
            "operation": "add_robot_workstation",
            "params": {{
                "near_conveyor_id": "Line_04_2",
                "robot_type": "ur10e"
            }},
            "rationale": "Add robot workstation on east side"
        }},
        {{
            "operation": "add_workbench_area",
            "params": {{
                "edge": "west",
                "position": 0.5,
                "num_tables": 2
            }},
            "rationale": "Add workbench area on west side"
        }}
    ]
}}

Return ONLY the JSON object, no other text.
"""
    return prompt


def _format_edges(edges: Dict) -> str:
    """Format edge information for the prompt."""
    lines = []
    for edge, data in edges.items():
        convs = data.get('conveyors', 0)
        stats = data.get('stations', 0)
        support = data.get('support', 0)
        lines.append(f"  - {edge}: {convs} conveyors, {stats} stations, {support} support equipment")
    return "\n".join(lines)


class HybridReconfiguration:
    """
    Hybrid reconfiguration system combining LLM planning with structure-aware execution.
    """

    def __init__(self, baseline_scene: List[Dict], model_name: str = "gpt-4o", reasoning_effort: str = "none", use_as_template: bool = False):
        """
        Initialize hybrid reconfiguration.

        Args:
            baseline_scene: The baseline scene graph
            model_name: Model to use for planning (uses OAI_CONFIG_LIST.json)
            reasoning_effort: For Gemini models, controls thinking depth
            use_as_template: If True, use baseline as structural template only (learn organization pattern)
                            but build a new layout from scratch instead of preserving the original equipment
        """
        self.baseline = baseline_scene
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.use_as_template = use_as_template

        # Initialize structure-aware system
        self.structure_aware = StructureAwareReconfiguration(baseline_scene, use_as_template=use_as_template)
        self.structure = self.structure_aware.structure

        # Track operations performed
        self.operations_log = []

        # Initialize LLM planner agent using autogen config (same as agents.py)
        self._init_planner_agent()

    def _init_planner_agent(self):
        """Initialize the LLM planner agent using autogen configuration."""
        try:
            # Import config functions from agents.py
            from agents.agents import get_config_list, get_rationale_config

            # Get LLM config using the same approach as agents.py
            llm_config = get_rationale_config(
                self.model_name,
                temperature=0.3,  # Lower temperature for more consistent planning
                reasoning_effort=self.reasoning_effort
            )

            # Create planner agent
            self.planner_agent = autogen.AssistantAgent(
                name="Reconfiguration_Planner",
                llm_config=llm_config,
                human_input_mode="NEVER",
                system_message=self._get_planner_system_message()
            )

            # Create user proxy for single-turn interaction
            self.user_proxy = autogen.UserProxyAgent(
                name="User",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,  # Single turn only
                code_execution_config=False
            )

            self.llm_available = True
            print(f"   LLM planner initialized with {self.model_name}")

        except Exception as e:
            print(f"   Note: Could not initialize LLM planner ({e}), using rule-based planning")
            self.llm_available = False
            self.planner_agent = None
            self.user_proxy = None

    def _get_planner_system_message(self) -> str:
        """Get system message for the planner agent."""
        return f"""You are a Manufacturing Layout Reconfiguration Planner.

Your task is to analyze user requests and create a structured plan for reconfiguring a manufacturing layout.

{AVAILABLE_OPERATIONS}

PLANNING GUIDELINES:
1. Analyze the user's request to understand the intent
2. Consider the current layout structure (corners, edges, equipment)
3. Choose appropriate operations that achieve the goal
4. Ensure operations maintain connectivity and proper spacing

OUTPUT FORMAT:
You MUST respond with a JSON object containing your plan:
{{
    "understanding": "Brief summary of what the user wants",
    "analysis": "Your analysis of how to achieve it",
    "operations": [
        {{
            "operation": "extend_edge",
            "params": {{
                "edge": "west",
                "stations": ["Line_07", "Line_06", "Line_02"]
            }},
            "rationale": "Why this operation"
        }}
    ]
}}

IMPORTANT: Return ONLY the JSON object, no additional text or markdown formatting."""

    def get_baseline_summary(self) -> Dict:
        """Get summary of baseline layout."""
        room_elements = {'south_wall', 'north_wall', 'east_wall', 'west_wall', 'ceiling', 'middle of the room'}
        equipment = [o for o in self.baseline if o.get('object_id') not in room_elements]

        return {
            'total_objects': len(equipment),
            'equipment_types': self._count_equipment_types(equipment),
            'room_dimensions': self._get_room_dimensions()
        }

    def _count_equipment_types(self, equipment: List[Dict]) -> Dict[str, int]:
        """Count equipment by type."""
        counts = {}
        for obj in equipment:
            obj_id = obj.get('object_id', '')
            # Extract base type (e.g., "Line_04" from "Line_04_1")
            parts = obj_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_type = parts[0]
            else:
                base_type = obj_id
            counts[base_type] = counts.get(base_type, 0) + 1
        return counts

    def _get_room_dimensions(self) -> str:
        """Extract room dimensions from baseline."""
        for obj in self.baseline:
            if obj.get('object_id') == 'south_wall':
                size = obj.get('size_in_meters', {})
                return f"{size.get('length', 0):.1f}m x {size.get('width', 0):.1f}m"
        return "unknown"

    def plan_reconfiguration(self, user_request: str) -> Dict:
        """
        Use LLM to plan reconfiguration operations.

        Args:
            user_request: Natural language description of desired changes

        Returns:
            Plan with operations to execute
        """
        baseline_summary = self.get_baseline_summary()
        structure_summary = self.structure.get_summary()

        prompt = create_reconfig_planning_prompt(
            user_request, baseline_summary, structure_summary,
            use_as_template=self.use_as_template
        )

        # Call LLM for planning using autogen agent
        if not self.llm_available or not self.planner_agent:
            raise RuntimeError("LLM planner not available. Please check your model configuration.")

        response = self._call_llm_agent(prompt)
        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            plan = self._extract_json(response)

        return plan

    def _call_llm_agent(self, prompt: str) -> str:
        """Call LLM using autogen agent."""
        # Use initiate_chat for single-turn interaction
        self.user_proxy.initiate_chat(
            self.planner_agent,
            message=prompt,
            max_turns=1,
            silent=True  # Suppress autogen's verbose output
        )

        # Get the last message from the planner agent
        last_message = self.planner_agent.last_message()
        if last_message and 'content' in last_message:
            return last_message['content']
        return "{}"

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response text."""
        import re
        # Try to find JSON in the response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Could not parse LLM response", "raw": text}

    def execute_plan(self, plan: Dict) -> Dict:
        """
        Execute a reconfiguration plan using the structure-aware system.

        Args:
            plan: Plan with operations from plan_reconfiguration()

        Returns:
            Summary of changes made
        """
        all_changes = {'added': [], 'modified': [], 'errors': []}

        operations = plan.get('operations', [])

        for op in operations:
            op_name = op.get('operation', '')
            params = op.get('params', {})
            rationale = op.get('rationale', '')

            print(f"   Executing: {op_name}")
            print(f"   Params: {params}")
            print(f"   Rationale: {rationale}")

            try:
                if op_name == 'build_conveyor_system':
                    changes = self.structure_aware.build_conveyor_system(
                        layout=params.get('layout', 'L-shape'),
                        conveyors=params.get('conveyors'),
                        inline_stations=params.get('inline_stations')
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'build_new_loop':
                    changes = self.structure_aware.build_new_loop(
                        num_conveyors_per_edge=params.get('num_conveyors_per_edge', 2),
                        add_lift=params.get('add_lift', True),
                        add_inspection=params.get('add_inspection', True),
                        add_assembly=params.get('add_assembly', True)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'extend_edge':
                    changes = self.structure_aware.extend_edge(
                        edge=params.get('edge', 'west'),
                        extension_distance=params.get('distance', 4.0),
                        add_stations=params.get('stations', [])
                    )
                    all_changes['added'].extend(changes.get('added', []))
                    all_changes['modified'].extend(changes.get('modified', []))

                elif op_name == 'add_inline_station':
                    station = self.structure_aware.add_inline_station(
                        conveyor_id=params.get('conveyor_id'),
                        station_type=params.get('station_type')
                    )
                    if 'error' not in station:
                        all_changes['added'].append(station.get('object_id'))
                    else:
                        all_changes['errors'].append(station['error'])

                elif op_name == 'add_parallel_branch':
                    # Use the parallel branch function
                    changes = self._add_parallel_branch(params)
                    all_changes['added'].extend(changes.get('added', []))

                # === Material Flow Definition ===

                elif op_name == 'define_material_flow':
                    changes = self.structure_aware.define_material_flow(
                        loading_edge=params.get('loading_edge', 'south'),
                        loading_position=params.get('loading_position', 0.3),
                        unloading_edge=params.get('unloading_edge', 'south'),
                        unloading_position=params.get('unloading_position', 0.7),
                        flow_direction=params.get('flow_direction', 'clockwise')
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                        # Store flow metadata for reference
                        all_changes['material_flow'] = changes.get('material_flow')
                    else:
                        all_changes['errors'].append(changes['error'])

                # === Extended Equipment Operations (Legacy Catalog) ===

                elif op_name == 'add_pallet_staging':
                    changes = self.structure_aware.add_pallet_staging(
                        near_equipment_id=params.get('near_equipment_id'),
                        num_pallets=params.get('num_pallets', 4),
                        add_pallet_jack=params.get('add_pallet_jack', True),
                        layout=params.get('layout', 'grid')
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_machining_cell':
                    changes = self.structure_aware.add_machining_cell(
                        edge=params.get('edge', 'north'),
                        position=params.get('position', 0.5),
                        add_scissor_lift=params.get('add_scissor_lift', True),
                        add_power_cutter=params.get('add_power_cutter', False)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_robot_workstation':
                    changes = self.structure_aware.add_robot_workstation(
                        near_conveyor_id=params.get('near_conveyor_id'),
                        robot_type=params.get('robot_type', 'ur10e'),
                        add_work_table=params.get('add_work_table', True),
                        add_shelving=params.get('add_shelving', True),
                        add_camera_stand=params.get('add_camera_stand', False),
                        edge=params.get('edge'),
                        position=params.get('position', 0.5)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_quality_station':
                    changes = self.structure_aware.add_quality_station(
                        near_conveyor_id=params.get('near_conveyor_id'),
                        add_camera=params.get('add_camera', True),
                        add_table=params.get('add_table', True),
                        add_ventilation=params.get('add_ventilation', False),
                        edge=params.get('edge'),
                        position=params.get('position', 0.5)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_safety_perimeter':
                    changes = self.structure_aware.add_safety_perimeter(
                        around_equipment_id=params.get('around_equipment_id'),
                        num_sections=params.get('num_sections', 4),
                        include_gate=params.get('include_gate', True)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_workbench_area':
                    changes = self.structure_aware.add_workbench_area(
                        edge=params.get('edge', 'south'),
                        position=params.get('position', 0.5),
                        num_tables=params.get('num_tables', 2),
                        add_shelving=params.get('add_shelving', True)
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                elif op_name == 'add_ventilation':
                    changes = self.structure_aware.add_ventilation(
                        near_equipment_id=params.get('near_equipment_id'),
                        exhaust_direction=params.get('exhaust_direction', 'up')
                    )
                    if 'error' not in changes:
                        all_changes['added'].extend(changes.get('added', []))
                    else:
                        all_changes['errors'].append(changes['error'])

                else:
                    all_changes['errors'].append(f"Unknown operation: {op_name}")

            except Exception as e:
                all_changes['errors'].append(f"Error in {op_name}: {str(e)}")

            self.operations_log.append({
                'operation': op_name,
                'params': params,
                'rationale': rationale,
                'success': op_name not in str(all_changes['errors'])
            })

        return all_changes

    def _add_parallel_branch(self, params: Dict) -> Dict:
        """Add a parallel processing branch."""
        # Find a suitable conveyor to branch from
        station_type = params.get('station_type', 'Line_07')

        # Get conveyors from an edge with existing stations
        for edge in ['east', 'west', 'north', 'south']:
            conveyors = self.structure.edges[edge]['conveyors']
            if conveyors:
                # Branch from the first conveyor on this edge
                conveyor_id = conveyors[0]
                conveyor = self.structure_aware._find_object(conveyor_id)
                if conveyor:
                    station = self.structure_aware._create_inline_station(
                        conveyor, station_type
                    )
                    return {'added': [station['object_id']]}

        return {'added': [], 'error': 'No suitable conveyor found'}

    def reconfigure(self, user_request: str) -> List[Dict]:
        """
        Full reconfiguration: plan and execute.

        Args:
            user_request: Natural language description of desired changes

        Returns:
            Modified scene graph
        """
        print(f"\n🤖 Planning reconfiguration...")
        plan = self.plan_reconfiguration(user_request)

        print(f"\n📋 Plan:")
        print(f"   Understanding: {plan.get('understanding', 'N/A')}")
        print(f"   Analysis: {plan.get('analysis', 'N/A')}")
        print(f"   Operations: {len(plan.get('operations', []))}")

        print(f"\n🔧 Executing plan...")
        changes = self.execute_plan(plan)

        print(f"\n✅ Changes:")
        print(f"   Added: {changes.get('added', [])}")
        if changes.get('errors'):
            print(f"   Errors: {changes.get('errors', [])}")

        return self.structure_aware.get_modified_scene()

    def get_modified_scene(self) -> List[Dict]:
        """Get the modified scene graph."""
        return self.structure_aware.get_modified_scene()

    def get_operations_log(self) -> List[Dict]:
        """Get log of all operations performed."""
        return self.operations_log


def hybrid_reconfigure(baseline_path: str, user_request: str,
                       model_name: str = "gpt-4o", reasoning_effort: str = "none") -> List[Dict]:
    """
    Convenience function for hybrid reconfiguration.

    Args:
        baseline_path: Path to baseline scene JSON
        user_request: Natural language reconfiguration request
        model_name: Model name for planning (uses OAI_CONFIG_LIST.json)
        reasoning_effort: For Gemini models, controls thinking depth

    Returns:
        Modified scene graph
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    if isinstance(baseline, dict):
        baseline = baseline.get('objects_in_room', baseline)

    hybrid = HybridReconfiguration(baseline, model_name, reasoning_effort)
    return hybrid.reconfigure(user_request)


if __name__ == '__main__':
    # Test without LLM (uses rule-based planning)
    print("=== Testing Hybrid Reconfiguration (Rule-Based) ===\n")

    result = hybrid_reconfigure(
        'scenes/scene_graph_cell03.json',
        'Extend the conveyor loop westward with additional assembly and inspection stations'
    )

    print(f"\nResult: {len(result)} objects in modified scene")
