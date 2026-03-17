from autogen import GroupChatManager
import json
import re
import networkx as nx
from datetime import datetime
import ast
import math
from copy import deepcopy
from pathlib import Path

from agents.agents import create_agents
from agents.agents import (
    is_termination_msg,
    get_model_config,
    build_initial_plan_prompt,
    create_reconfiguration_agent,
    build_simple_reconfiguration_prompt,
)
from agents.corrector_agents import get_corrector_agents
from agents.refiner_agents import get_refiner_agents

from integrations.chats import GroupChat, ChatWithEngineer, LayoutCorrectorGroupChat, ObjectDeletionGroupChat, LayoutRefinerGroupChat 

from core.utils import get_room_priors, extract_list_from_json
from core.utils import preprocess_scene_graph, build_graph, remove_unnecessary_edges, handle_under_prepositions, get_conflicts, get_size_conflicts, get_object_from_scene_graph
from core.utils import get_object_from_scene_graph, get_rotation, get_cluster_objects, clean_and_extract_edges
from core.utils import get_cluster_size
from core.utils import get_possible_positions, is_point_bbox, calculate_overlap, get_topological_ordering, place_object, get_depth, get_visualization, strip_room_layout_elements, populate_conveyor_connections
from catalog.equipment_catalog import get_equipment_list, get_equipment_aliases, get_equipment_info
from evaluation.simulation import ProcessFlowSimulator
from evaluation.hybrid_simulation import HybridPerformanceEvaluator, HybridSimulationReport
from reconfiguration.structure_aware_reconfig import (
    LoopStructure,
    StructureAwareReconfiguration,
    reconfigure_extend_loop,
    parse_reconfiguration_request
)
from reconfiguration.hybrid_reconfig import HybridReconfiguration
from layout.placement_optimizer import PlacementOptimizer, optimize_placement

class LLM4RMS:
    ROOM_LAYOUT_IDS = {
        "south_wall",
        "north_wall",
        "west_wall",
        "east_wall",
        "ceiling",
        "middle of the room",
    }
    ROOM_LAYOUT_IDS_LOWER = {item.lower() for item in ROOM_LAYOUT_IDS}

    def __init__(self, no_of_objects, user_input, room_dimensions, model_name="gpt-4o", initial_design=None, multiagent_reconfig=False, reasoning_effort="none", structure_aware_reconfig=False, hybrid_reconfig=False, use_as_template=False):
        """Initialize LLM4RMS

        Args:
            no_of_objects: Number of objects in the layout
            user_input: User requirements/objectives
            room_dimensions: Room dimensions [length, width, height] or None for auto-size
            model_name: LLM model name
            initial_design: Path to existing layout file or dict/list (enables reconfiguration)
            multiagent_reconfig: If True, use multi-agent reconfiguration workflow (more sophisticated)
                               If False, use single-agent reconfiguration (faster, simpler)
            reasoning_effort: For Gemini models, controls thinking depth ("none", "low", "medium", "high")
            structure_aware_reconfig: If True, use structure-aware reconfiguration (best for production systems)
                                     Learns loop structure from baseline and applies targeted modifications
            hybrid_reconfig: If True, use hybrid reconfiguration (LLM planning + structure-aware execution)
                            Combines natural language understanding with reliable placement
            use_as_template: If True, use initial_design as a structural template only (learn organization pattern)
                            but build a new layout from scratch instead of preserving the original equipment
        """
        self.no_of_objects = no_of_objects
        self.user_input = user_input
        self.auto_room_dimensions = (
            room_dimensions is None
            or (isinstance(room_dimensions, (list, tuple)) and len(room_dimensions) < 3)
        )
        self._auto_room_dimensions_result = None
        if self.auto_room_dimensions:
            # Start with a conservative compact default; will be resized after equipment selection
            self.room_dimensions = [8.0, 6.0, 3.5]
        else:
            self.room_dimensions = list(room_dimensions)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.model_config = get_model_config(model_name, reasoning_effort=reasoning_effort)
        self.room_priors = get_room_priors(self.room_dimensions)
        self.scene_graph = None
        self.allowed_equipment_names = set(get_equipment_list())
        self._equipment_aliases = get_equipment_aliases()
        self._initial_design_loaded = False
        self._initial_design_source = None
        self._initial_design_snapshot = None
        self._initial_object_lookup = {}
        self._initial_design_was_list = False
        self._was_reconfigured = False  # Track if layout was reconfigured (skip refiner)
        self.multiagent_reconfig = multiagent_reconfig  # Choose reconfiguration workflow
        self.structure_aware_reconfig = structure_aware_reconfig  # Use structure-aware approach
        self.hybrid_reconfig = hybrid_reconfig  # Use hybrid (LLM + structure-aware) approach
        self.use_as_template = use_as_template  # Use initial design as template only (don't preserve equipment)

        if initial_design is not None:
            self.load_initial_design(initial_design)

    def _get_latest_agent_message(self, messages, agent_name):
        """Return the most recent message authored by the specified agent."""
        for msg in reversed(messages):
            if msg.get("name") == agent_name:
                return msg
        return None

    def _parse_json_like(self, candidate):
        """Parse JSON or Python literal structures into standard JSON-compatible dict/list."""
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                return json.loads(json.dumps(parsed))
            except (ValueError, SyntaxError, TypeError) as err:
                raise ValueError from err

    def extract_json_from_response(self, content):
        """Extract JSON from a response that may contain rationale text"""
        try:
            # First try to parse as direct JSON
            return self._parse_json_like(content)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            # Look for JSON in the response
            # First try to find JSON in ```json blocks
            json_block_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_block_pattern, content, re.DOTALL)
            if match:
                try:
                    return self._parse_json_like(match.group(1))
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    pass

            # Look for any JSON-like structure (nested braces)
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return self._parse_json_like(match)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue

            raise ValueError(f"No valid JSON found in response: {content[:200]}...")

    def _auto_size_room(self, objects):
        """Estimate compact room dimensions based on selected equipment."""
        if not objects:
            return [8.0, 6.0, 3.5]

        side_clearance = 1.0  # operational aisle allowance
        packing_buffer = 0.5  # per-object buffer to limit overlap

        lengths, widths, heights = [], [], []
        footprint_area = 0.0

        for obj in objects:
            size = obj.get("size_in_meters", {})
            length = max(float(size.get("length", 1.0)), 0.2)
            width = max(float(size.get("width", 1.0)), 0.2)
            height = max(float(size.get("height", 1.0)), 0.2)

            lengths.append(length)
            widths.append(width)
            heights.append(height)

            footprint_area += (length + packing_buffer) * (width + packing_buffer)

        footprint_area = max(footprint_area, 9.0)  # never smaller than 3m x 3m
        max_length = max(lengths) + side_clearance
        max_width = max(widths) + side_clearance

        # Estimate a linear flow extent (sum of lengths) to avoid overly short rooms
        linear_extent = sum(lengths) + packing_buffer * max(len(lengths) - 1, 0) + side_clearance

        target_area = max(footprint_area * 1.25, max_length * max_width)

        base_length = max(max_length, math.sqrt(target_area))
        base_width = max(max_width, target_area / base_length)

        # Ensure the longer axis can accommodate the linear extent
        if base_length >= base_width:
            base_length = max(base_length, linear_extent)
        else:
            base_width = max(base_width, linear_extent)

        # Add a small circulation margin
        base_length += side_clearance * 0.5
        base_width += side_clearance * 0.5

        ceiling_height = max(max(heights) + 1.0, 3.5)

        return [
            round(base_length, 2),
            round(base_width, 2),
            round(ceiling_height, 2),
        ]

    def create_initial_plan(self):
        """Create initial design plan for user review"""
        user_proxy, json_schema_debugger, process_planner, layout_engineer, engineer = create_agents(self.no_of_objects, self.model_name, reasoning_effort=self.reasoning_effort)
        
        groupchat = GroupChat(
            agents=[user_proxy, process_planner, layout_engineer],
            messages=[],
            max_round=3
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=self.model_config, is_termination_msg=is_termination_msg)
        if self.auto_room_dimensions:
            dimension_sentence = (
                "Workspace dimensions will be auto-sized to stay compact around the selected equipment. "
                "Prioritize an efficient footprint without unnecessary empty space."
            )
        else:
            dimension_sentence = (
                f"The manufacturing workspace has dimensions "
                f"{self.room_dimensions[0]}m x {self.room_dimensions[1]}m x {self.room_dimensions[2]}m."
            )
        
        plan_prompt = build_initial_plan_prompt(self.user_input, dimension_sentence)
        user_proxy.initiate_chat(manager, message=plan_prompt)
        
        # Store the planning conversation
        self._planning_conversation = [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in groupchat.messages]
        
        return self._planning_conversation

    def display_plan_summary(self):
        """Display a formatted summary of the plan for user review"""
        if not hasattr(self, '_planning_conversation'):
            return "No plan has been created yet. Please run create_initial_plan() first."
        
        print("=" * 80)
        print("🎯 INITIAL DESIGN PLAN SUMMARY")
        print("=" * 80)
        
        print(f"📋 Project: {self.user_input[:100]}...")
        if self.auto_room_dimensions:
            print(f"📐 Workspace: Auto-sized (will be determined after equipment selection)")
        else:
            print(f"📐 Workspace: {self.room_dimensions[0]}m × {self.room_dimensions[1]}m × {self.room_dimensions[2]}m")
        print(f"🔧 Target Objects: {self.no_of_objects}")
        print(f"🤖 Model: {self.model_name}")
        print()
        
        # Extract plan content from conversation
        for msg in self._planning_conversation:
            if msg.get('name') in ['Process_planner', 'Layout_engineer']:
                agent_name = "🔧 Process Planner" if msg['name'] == 'Process_planner' else "📐 Layout Engineer"
                print(f"{agent_name}")
                print("-" * 40)
                content = msg['content']
                # Clean up the content for display
                if len(content) > 1000:
                    content = content[:1000] + "..."
                print(content)
                print()
        
        print("=" * 80)

    def get_user_feedback(self):
        """Interactive method to collect user feedback on the plan"""
        print("\n🤔 PLAN REVIEW & FEEDBACK")
        print("=" * 80)
        print("Please review the plan above and provide your feedback.")
        print("You can:")
        print("1. Approve the plan (type 'approve' or 'yes')")
        print("2. Request modifications (describe what you'd like to change)")
        print("3. Add additional requirements (describe new constraints)")
        print("=" * 80)
        
        feedback = input("\nYour feedback: ").strip()
        
        if feedback.lower() in ['approve', 'yes', 'ok', 'good', 'approved']:
            return {"approved": True, "feedback": "Plan approved by user"}
        else:
            return {"approved": False, "feedback": feedback}

    def create_initial_design(self, user_feedback=None, force_regenerate=False):
        """
        Create the detailed initial design, optionally incorporating user feedback.
        When a user-provided scene graph was loaded, this step can be skipped unless
        force_regenerate=True.
        """
        user_proxy, json_schema_debugger, process_planner, layout_engineer, engineer = create_agents(self.no_of_objects, self.model_name, reasoning_effort=self.reasoning_effort)

        if self.scene_graph is not None and self._initial_design_loaded and not force_regenerate:
            print("🔧 Revising user-provided initial design based on new requirements.")
            # Choose reconfiguration workflow based on flags
            if getattr(self, 'hybrid_reconfig', False):
                print("   Using hybrid reconfiguration (LLM planning + structure-aware execution)")
                return self._reconfigure_hybrid(user_feedback)
            elif getattr(self, 'structure_aware_reconfig', False):
                print("   Using structure-aware reconfiguration (recommended for production systems)")
                return self._reconfigure_structure_aware(user_feedback)
            elif self.multiagent_reconfig:
                print("   Using multi-agent collaborative workflow")
                return self._reconfigure_multiagent(user_feedback)
            else:
                print("   Using single-agent workflow")
                return self._reconfigure_initial_design(user_feedback)
        
        groupchat = GroupChat(
            agents=[user_proxy, process_planner, layout_engineer],
            messages=[],
            max_round=3
        )

        chat_with_engineer = ChatWithEngineer(
            agents  =[user_proxy, engineer, json_schema_debugger],
            messages=[],
            max_round=15
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=self.model_config, is_termination_msg=is_termination_msg)
        if self.auto_room_dimensions:
            dimension_sentence = (
                "Workspace dimensions will be auto-sized to stay compact around the selected equipment. "
                "Prioritize an efficient footprint without unnecessary empty space."
            )
        else:
            dimension_sentence = (
                f"The manufacturing workspace has dimensions "
                f"{self.room_dimensions[0]}m x {self.room_dimensions[1]}m x {self.room_dimensions[2]}m."
            )
        
        # Prepare user input with feedback if provided
        user_requirements = self.user_input
        if user_feedback and not user_feedback.get("approved", True):
            user_requirements += f"\n\nADDITIONAL USER FEEDBACK:\n{user_feedback['feedback']}"
        
        plan_prompt = build_initial_plan_prompt(user_requirements, dimension_sentence)
        user_proxy.initiate_chat(manager, message=plan_prompt)

        # Get responses from planner and layout engineer
        # These may be text-only (rationale) or contain JSON
        planner_content = groupchat.messages[-2]["content"] if len(groupchat.messages) >= 2 else ""
        layout_content = groupchat.messages[-1]["content"] if len(groupchat.messages) >= 1 else ""
        
        # Try to extract JSON blocks, fall back to using raw text as planning context
        try:
            designer_response = self.extract_json_from_response(planner_content)
            blocks_designer = extract_list_from_json(designer_response)
        except (ValueError, TypeError):
            # No JSON found - use the text as planning context
            blocks_designer = None
            
        try:
            architect_response = self.extract_json_from_response(layout_content)
            blocks_architect = extract_list_from_json(architect_response)
        except (ValueError, TypeError):
            # No JSON found - use the text as planning context
            blocks_architect = None

        json_data = None
        
        # If we have structured JSON blocks, use them
        if blocks_designer and blocks_architect:
            if len(blocks_designer) != len(blocks_architect):
                min_blocks = min(len(blocks_designer), len(blocks_architect))
                print(f"⚠️ Designer/Architect block mismatch ({len(blocks_designer)} vs {len(blocks_architect)}). Truncating to {min_blocks}.")
                blocks_designer = blocks_designer[:min_blocks]
                blocks_architect = blocks_architect[:min_blocks]
                if min_blocks == 0:
                    raise ValueError("No usable blocks were returned by designer/architect agents.")

            for d_block, a_block in zip(blocks_designer, blocks_architect):
                engineer.reset(), json_schema_debugger.reset()
                prompt = json.dumps(d_block) + "\n" + json.dumps(a_block)

                object_ids = [item["object_id"] for item in json_data["objects_in_room"]] if json_data is not None else []

                manager = GroupChatManager(groupchat=chat_with_engineer, 
                                           llm_config=self.model_config, 
                                           human_input_mode="NEVER", 
                                           is_termination_msg=is_termination_msg)
                user_proxy.initiate_chat(
                    manager,
                    message=f"""
                    Workspace layout elements for reference (in triple backquotes):
                    ```
                    ['south_wall', 'north_wall', 'west_wall', 'east_wall', 'middle of the floor', 'ceiling']
                    ```
                    Array of objects in the room (in triple backquotes):
                    ```
                    {object_ids}
                    ```
                    Objects to be placed in the room (in triple backquotes):
                    ```
                    {prompt}
                    ```
                    json
                    """,
                )
        else:
            # Fallback: Use raw planning text as context for engineer
            print("📝 Using text-based planning context (no structured JSON from planners)")
            engineer.reset(), json_schema_debugger.reset()
            
            # Combine planner and layout engineer text as context
            planning_context = f"""
Process Planner's Analysis:
{planner_content}

Layout Engineer's Analysis:
{layout_content}
"""
            manager = GroupChatManager(groupchat=chat_with_engineer, 
                                       llm_config=self.model_config, 
                                       human_input_mode="NEVER", 
                                       is_termination_msg=is_termination_msg)
            user_proxy.initiate_chat(
                manager,
                message=f"""
                {dimension_sentence}
                
                Workspace layout elements for reference:
                ['south_wall', 'north_wall', 'west_wall', 'east_wall', 'middle of the room', 'ceiling']
                
                User requirements:
                {user_requirements}
                
                Planning context from Process Planner and Layout Engineer:
                {planning_context}
                
                Based on the above planning, generate the complete scene graph JSON with all equipment placements.
                json
                """,
            )
            engineer_msg = self._get_latest_agent_message(chat_with_engineer.messages, "Engineer")
            if engineer_msg is None:
                raise ValueError("Engineer did not produce a JSON payload for the layout generation.")

            engineer_payload = self.extract_json_from_response(engineer_msg["content"])
            if not isinstance(engineer_payload, dict) or "objects_in_room" not in engineer_payload:
                raise ValueError("Engineer response missing required 'objects_in_room' field.")

            if json_data is None:
                json_data = engineer_payload
            else:
                json_data["objects_in_room"] += engineer_payload.get("objects_in_room", [])
            
        if self.auto_room_dimensions and json_data:
            new_dimensions = self._auto_size_room(json_data.get("objects_in_room", []))
            self.room_dimensions = new_dimensions
            self.room_priors = get_room_priors(self.room_dimensions)
            self._auto_room_dimensions_result = new_dimensions
            print(f"📐 Auto-sized workspace dimensions: {new_dimensions[0]}m x {new_dimensions[1]}m x {new_dimensions[2]}m")
        
        self.scene_graph = json_data
        self._normalize_scene_objects()
        self._enforce_equipment_catalog_names()
        
        # Save conversation history for initial design
        self._initial_conversation = {
            "main_conversation": [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in groupchat.messages],
            "engineer_conversations": []
        }
        
        # Also save each engineer conversation
        if blocks_designer and blocks_architect:
            for i, (d_block, a_block) in enumerate(zip(blocks_designer, blocks_architect)):
                engineer_msgs = [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in chat_with_engineer.messages]
                self._initial_conversation["engineer_conversations"].append({
                    "block_index": i,
                    "designer_block": str(d_block),
                    "architect_block": str(a_block), 
                    "conversation": engineer_msgs
                })
        else:
            # Fallback mode - save the single engineer conversation
            engineer_msgs = [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in chat_with_engineer.messages]
            self._initial_conversation["engineer_conversations"].append({
                "block_index": 0,
                "designer_block": "text-based planning",
                "architect_block": "text-based planning", 
                "conversation": engineer_msgs
            })

        # Newly generated design supersedes any previously loaded layout
        self._initial_design_loaded = False
        self._initial_design_source = None
        self._was_reconfigured = False  # This is a new design, not reconfiguration

    def _canonicalize_equipment_base(self, base_name):
        normalized = base_name.replace("-", "_").strip()
        candidates = {normalized, normalized.replace(" ", "_")}
        for cand in list(candidates):
            if cand in self.allowed_equipment_names:
                return cand
        for cand in candidates:
            lower = cand.lower()
            if lower in self._equipment_aliases:
                return self._equipment_aliases[lower]
        return self._equipment_aliases.get(normalized.lower())

    def _normalize_scene_objects(self):
        if not isinstance(self.scene_graph, dict):
            return
        objects = self.scene_graph.get("objects_in_room")
        if not isinstance(objects, list):
            return
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            placement = obj.get("placement")
            if not isinstance(placement, dict):
                placement = {}
            room_layout = placement.get("room_layout_elements")
            if not isinstance(room_layout, list):
                room_layout = [] if room_layout is None else []
            placement["room_layout_elements"] = room_layout
            obj_rel = placement.get("objects_in_room")
            if not isinstance(obj_rel, list):
                obj_rel = [] if obj_rel is None else []
            placement["objects_in_room"] = obj_rel
            if not room_layout:
                placement["room_layout_elements"] = [
                    {
                        "layout_element_id": "middle of the room",
                        "preposition": "on",
                    }
                ]
            obj["placement"] = placement
            if obj.get("connections") is None:
                obj["connections"] = []

    def _ensure_placement_defaults_in_graph(self, graph_obj):
        """Ensure every object has placement/objects_in_room arrays before export."""
        if graph_obj is None:
            return

        def _normalize_obj(obj):
            if not isinstance(obj, dict):
                return
            placement = obj.get("placement")
            if not isinstance(placement, dict):
                placement = {}
                obj["placement"] = placement
            room_layout = placement.get("room_layout_elements")
            if not isinstance(room_layout, list):
                placement["room_layout_elements"] = []
            else:
                placement["room_layout_elements"] = room_layout
            obj_rels = placement.get("objects_in_room")
            if not isinstance(obj_rels, list):
                placement["objects_in_room"] = []
            else:
                placement["objects_in_room"] = obj_rels

        if isinstance(graph_obj, dict):
            objects = graph_obj.get("objects_in_room", [])
            for obj in objects:
                _normalize_obj(obj)
        elif isinstance(graph_obj, list):
            for obj in graph_obj:
                _normalize_obj(obj)

    def _split_object_id(self, object_id):
        match = re.match(r"(.+?)(?:_(\d+))?$", object_id)
        if match:
            return match.group(1), match.group(2)
        return object_id, None

    def _extract_equipment_base(self, object_id):
        """
        Extract the equipment base name from an object ID that may contain
        additional descriptive suffixes (e.g., 'line_05_t_branch' -> 'Line_05').
        
        Tries progressively shorter prefixes to find a catalog match.
        """
        from catalog.equipment_catalog import EQUIPMENT_CATALOG, EQUIPMENT_ALIAS_OVERRIDES
        
        # Normalize to lowercase for matching
        obj_lower = object_id.lower().replace("-", "_")
        
        # Try to match Line_XX pattern directly first (handles line1, line_1, line01, line_01, etc.)
        line_match = re.match(r"line[_-]?0?(\d+)", obj_lower, re.IGNORECASE)
        if line_match:
            num = int(line_match.group(1))
            # Normalize to Line_XX format with zero-padding (Line_01, Line_02, etc.)
            normalized = f"Line_{num:02d}"
            if normalized in EQUIPMENT_CATALOG:
                return normalized
        
        # Try to match Cart
        if obj_lower.startswith("cart"):
            return "Cart"
        
        # Check aliases
        for alias, canonical in EQUIPMENT_ALIAS_OVERRIDES.items():
            if obj_lower.startswith(alias.lower()):
                return canonical
        
        return None

    def _next_available_id(self, base_name, used_ids):
        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if candidate not in used_ids:
                return candidate
            counter += 1

    def _canonicalize_equipment_id(self, object_id):
        # First try to match the full identifier (some catalog names contain digits/underscores)
        direct_match = self._canonicalize_equipment_base(object_id)
        if direct_match is not None:
            return direct_match

        base, suffix = self._split_object_id(object_id)
        canonical_base = self._canonicalize_equipment_base(base)
        
        # If standard split didn't work, try extracting equipment base from complex names
        # like "line_05_t_branch" -> "Line_05"
        if canonical_base is None:
            canonical_base = self._extract_equipment_base(object_id)
        
        if canonical_base is None:
            raise ValueError(
                f"Object '{object_id}' is not available in the equipment catalog. "
                "Please regenerate the design with catalog-compliant items."
            )
        if suffix is None:
            return canonical_base
        return f"{canonical_base}_{suffix}"

    def _enforce_equipment_catalog_names(self):
        if not isinstance(self.scene_graph, dict):
            return
        objects = self.scene_graph.get("objects_in_room")
        if not objects:
            return
        name_mapping = {}
        used_ids = set()
        for obj in objects:
            original_id = obj.get("object_id", "")
            canonical_id = self._canonicalize_equipment_id(original_id)
            
            # Check if the canonical_id itself is a catalog item (e.g. "Line_04")
            # If so, treat the whole string as the base name to ensure we add an instance suffix
            if canonical_id in self.allowed_equipment_names:
                base = canonical_id
                suffix = None
            else:
                base, suffix = self._split_object_id(canonical_id)
            
            # Always ensure instance suffix (e.g., Line_04 -> Line_04_1)
            if suffix is None:
                # No suffix in canonical_id, assign next available
                resolved_id = self._next_available_id(base, used_ids)
            elif canonical_id in used_ids:
                # Suffix exists but ID already used, get next available
                resolved_id = self._next_available_id(base, used_ids)
            else:
                resolved_id = canonical_id
            
            used_ids.add(resolved_id)
            name_mapping[original_id] = resolved_id
            obj["object_id"] = resolved_id

        for obj in objects:
            placement = obj.get("placement", {})
            for rel in placement.get("objects_in_room", []):
                ref_id = rel.get("object_id")
                if ref_id in name_mapping:
                    rel["object_id"] = name_mapping[ref_id]
            
            # Also update connections references
            connections = obj.get("connections", [])
            for conn in connections:
                ref_id = conn.get("object_id")
                if ref_id in name_mapping:
                    conn["object_id"] = name_mapping[ref_id]

        # Enforce catalog metadata (style, material, size) after renaming
        for obj in objects:
            canonical_id = obj.get("object_id")
            if not canonical_id:
                continue
            base_name, _ = self._split_object_id(canonical_id)
            catalog_info = get_equipment_info(base_name)
            if not catalog_info:
                continue
            obj["style"] = catalog_info.get("style", obj.get("style"))
            obj["material"] = catalog_info.get("material", obj.get("material"))
            size_info = catalog_info.get("approximate_size")
            if size_info:
                obj["size_in_meters"] = {
                    "length": float(size_info.get("length", obj.get("size_in_meters", {}).get("length", 1.0))),
                    "width": float(size_info.get("width", obj.get("size_in_meters", {}).get("width", 1.0))),
                    "height": float(size_info.get("height", obj.get("size_in_meters", {}).get("height", 1.0))),
                }

    def _reconfigure_initial_design(self, user_feedback=None):
        """Reconfigure the existing layout using command-based approach for better stability.

        Instead of passing the full layout JSON to the agent, we:
        1. Load the existing layout into a LayoutBuilder
        2. Provide only a summary to the agent
        3. Agent outputs simple commands (not full JSON)
        4. Commands are executed against the LayoutBuilder
        """
        from layout.layout_tools import LayoutBuilder
        from layout.layout_command_executor import LayoutCommandExecutor

        # Create agents
        user_proxy, reconfiguration_agent = create_reconfiguration_agent(self.model_name, reasoning_effort=self.reasoning_effort)

        # Initialize LayoutBuilder with existing layout
        builder = LayoutBuilder(self.scene_graph)
        executor = LayoutCommandExecutor(builder)

        # Get layout summary (not full JSON - much smaller!)
        layout_summary = builder.get_current_layout()

        # Prepare user requirements
        user_requirements = self.user_input
        if user_feedback and not user_feedback.get("approved", True):
            user_requirements += f"\n\nADDITIONAL USER FEEDBACK:\n{user_feedback['feedback']}"

        # Build prompt with summary instead of full JSON
        prompt = build_simple_reconfiguration_prompt(user_requirements, layout_summary)

        print(f"🔄 Reconfiguring layout with {layout_summary['total_objects']} objects...")

        # Get agent response with commands
        user_proxy.initiate_chat(reconfiguration_agent, message=prompt, max_turns=1)

        agent_msg = self._get_latest_agent_message(user_proxy.chat_messages[reconfiguration_agent], "Reconfiguration_Agent")
        if agent_msg is None:
            raise ValueError("Reconfiguration Agent did not return a response.")

        # Parse and execute commands from agent response
        print("📝 Parsing and executing reconfiguration commands...")
        success, execution_message = executor.parse_and_execute_commands(agent_msg["content"])

        # Get execution summary
        exec_summary = executor.get_execution_summary()
        print(f"✅ Executed {exec_summary['successful']}/{exec_summary['total_commands']} commands successfully")

        if exec_summary['failed'] > 0:
            print(f"⚠️ {exec_summary['failed']} commands failed:")
            for entry in exec_summary['log']:
                if not entry.get('success', True):
                    print(f"   - {entry.get('error', 'Unknown error')}")

        # Get the reconfigured layout from builder
        json_data = executor.get_final_layout()

        if not isinstance(json_data, dict) or "objects_in_room" not in json_data:
            raise ValueError("Reconfiguration failed to produce valid layout.")

        # Update scene graph
        self.scene_graph = json_data
        self._normalize_scene_objects()
        self._enforce_equipment_catalog_names()

        # DO NOT reuse initial positions - we want to recalculate based on new placement relationships
        # (removed _reuse_initial_positions() call that was copying old positions)

        self._initial_design_loaded = False
        self._initial_design_source = None

        # Now calculate actual positions based on updated placement relationships
        print("🔄 Optimizing positions using graph-based solver...")

        # Use new optimizer instead of backtrack
        updated_graph, results = optimize_placement(
            scene_graph=self.scene_graph,
            room_dimensions=self.room_dimensions,
            room_priors=self.room_priors,
            verbose=True,
            preserve_existing=True
        )
        self.scene_graph = updated_graph["objects_in_room"]

        # Update room dimensions if optimizer expanded the room
        if "room_dimensions" in results:
            new_dims = results["room_dimensions"]
            if new_dims[0] > self.room_dimensions[0] or new_dims[1] > self.room_dimensions[1]:
                print(f"📐 Room expanded from {self.room_dimensions[:2]} to {new_dims[:2]}")
                self.room_dimensions = list(new_dims)  # Ensure it's a mutable list
                # Rebuild room priors with new dimensions
                self.room_priors = get_room_priors(self.room_dimensions)

        # Always expand room to fit all objects after reconfiguration (even for loaded designs)
        self._expand_room_to_fit_objects_unconditional(margin=1.0)

        get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

        # Mark that this was a reconfiguration so we can skip refiner
        self._was_reconfigured = True

        self._initial_conversation = {
            "source": "reconfigured_initial_design",
            "conversation": [
                {"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]}
                for msg in user_proxy.chat_messages[reconfiguration_agent]
            ],
            "command_execution_summary": exec_summary,
        }

        print(f"✨ Reconfiguration complete! New layout has {len(json_data['objects_in_room'])} objects")

        return self.scene_graph

    def _reconfigure_structure_aware(self, user_feedback=None):
        """Reconfigure using structure-aware approach.

        This learns the loop structure from the baseline design and applies
        targeted modifications while preserving the physical connectivity.
        Best for production systems where maintaining conveyor loop integrity is critical.
        """
        print("📐 Analyzing baseline structure...")

        # Get baseline as list
        if isinstance(self.scene_graph, dict):
            baseline_objects = self.scene_graph.get('objects_in_room', [])
        else:
            baseline_objects = self.scene_graph

        # Learn structure
        structure = LoopStructure(baseline_objects)
        summary = structure.get_summary()

        print(f"   Detected loop with {len(summary['corners'])} corners")
        print(f"   Bounds: {summary['bounds']}")
        print(f"   Edges: {summary['edges']}")

        # Parse the user request to determine operation
        request = self.user_input
        if user_feedback and not user_feedback.get("approved", True):
            request += f"\n{user_feedback['feedback']}"

        print(f"\n🔧 Processing request: {request[:100]}...")

        # Create reconfiguration handler
        reconfig = StructureAwareReconfiguration(baseline_objects)

        # Parse request and determine operation
        request_lower = request.lower()

        if 'extend' in request_lower or 'expand' in request_lower:
            # Determine direction
            edge = 'east'  # default
            if 'west' in request_lower:
                edge = 'west'
            elif 'north' in request_lower:
                edge = 'north'
            elif 'south' in request_lower:
                edge = 'south'

            # Determine stations to add
            stations = []
            if 'assembly' in request_lower or 'line_07' in request_lower:
                stations.append('Line_07')
            if 'inspection' in request_lower or 'line_06' in request_lower:
                stations.append('Line_06')
            if 'workstation' in request_lower or 'line_02' in request_lower:
                stations.append('Line_02')

            if not stations:
                stations = ['Line_07', 'Line_06']  # Default

            print(f"   Operation: Extend {edge} edge")
            print(f"   Adding stations: {stations}")

            changes = reconfig.extend_edge(edge, extension_distance=4.0, add_stations=stations)
            print(f"   Changes: {changes}")

        elif 'parallel' in request_lower or 'throughput' in request_lower:
            print("   Operation: Add parallel processing branch")
            # This would call reconfig.add_parallel_branch()
            # For now, extend east as fallback
            changes = reconfig.extend_edge('east', extension_distance=4.0, add_stations=['Line_07'])

        else:
            # Default: extend east with assembly and inspection
            print("   Operation: Default extension (east edge)")
            changes = reconfig.extend_edge('east', extension_distance=4.0, add_stations=['Line_07', 'Line_06'])

        # Get modified scene
        modified_scene = reconfig.get_modified_scene()

        # Update room dimensions to fit new layout
        self._update_scene_and_room(modified_scene)

        # Generate visualization
        get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

        # Mark as reconfigured
        self._was_reconfigured = True

        self._initial_conversation = {
            "source": "structure_aware_reconfiguration",
            "changes": changes,
        }

        change_summary = reconfig.get_changes_summary()
        print(f"\n✨ Structure-aware reconfiguration complete!")
        print(f"   Added: {len(change_summary['added'])} objects")
        print(f"   Total: {change_summary['total_modified']} objects")

        return self.scene_graph

    def _reconfigure_hybrid(self, user_feedback=None):
        """Reconfigure using hybrid approach: LLM planning + structure-aware execution.

        This combines:
        - LLM agent for understanding natural language and creating a plan
        - Structure-aware system for reliable, collision-free execution

        Best for complex reconfiguration requests that need natural language understanding
        but also require predictable, physically-correct results.
        """
        use_as_template = getattr(self, 'use_as_template', False)
        if use_as_template:
            print("🤖 Initializing hybrid reconfiguration (template mode - building new layout)...")
        else:
            print("🤖 Initializing hybrid reconfiguration...")

        # Get baseline as list
        if isinstance(self.scene_graph, dict):
            baseline_objects = self.scene_graph.get('objects_in_room', [])
        else:
            baseline_objects = self.scene_graph

        # Build user request
        request = self.user_input
        if user_feedback and not user_feedback.get("approved", True):
            request += f"\n{user_feedback['feedback']}"

        # Create hybrid reconfiguration handler
        # Uses autogen config from OAI_CONFIG_LIST.json (same as agents.py)
        hybrid = HybridReconfiguration(
            baseline_objects,
            model_name=self.model_name,
            reasoning_effort=getattr(self, 'reasoning_effort', 'none'),
            use_as_template=use_as_template
        )

        # Run hybrid reconfiguration (plan + execute)
        modified_scene = hybrid.reconfigure(request)

        # Update room dimensions to fit new layout
        self._update_scene_and_room(modified_scene)

        # Generate visualization
        get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

        # Mark as reconfigured
        self._was_reconfigured = True

        self._initial_conversation = {
            "source": "hybrid_reconfiguration",
            "operations_log": hybrid.get_operations_log(),
        }

        change_summary = hybrid.structure_aware.get_changes_summary()
        print(f"\n✨ Hybrid reconfiguration complete!")
        print(f"   Added: {len(change_summary['added'])} objects")
        print(f"   Total: {change_summary['total_modified']} objects")

        return self.scene_graph

    def _update_scene_and_room(self, modified_scene: list):
        """Update scene graph and expand room to fit all objects.

        Handles cases where equipment has negative coordinates by shifting
        everything to ensure all objects are within positive coordinate space.
        """
        ROOM_ELEMENTS = {'south_wall', 'north_wall', 'east_wall', 'west_wall', 'ceiling', 'middle of the room'}

        # Filter out old room elements
        equipment_only = [obj for obj in modified_scene if obj.get('object_id') not in ROOM_ELEMENTS]

        # Calculate actual bounds (including negative coordinates)
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for obj in equipment_only:
            pos = obj.get('position', {})
            size = obj.get('size_in_meters', {})
            x, y = pos.get('x', 0), pos.get('y', 0)
            l, w = size.get('length', 1), size.get('width', 1)

            min_x = min(min_x, x - w/2)
            max_x = max(max_x, x + w/2)
            min_y = min(min_y, y - l/2)
            max_y = max(max_y, y + l/2)

        # If objects have negative coordinates, shift everything to positive space
        shift_x = 0
        shift_y = 0
        margin = 1.0

        if min_x < margin:
            shift_x = margin - min_x  # Shift right so min becomes margin
        if min_y < margin:
            shift_y = margin - min_y  # Shift up so min becomes margin

        if shift_x > 0 or shift_y > 0:
            print(f"📐 Shifting objects by ({shift_x:.1f}, {shift_y:.1f}) to fit in positive coordinate space")
            for obj in equipment_only:
                pos = obj.get('position', {})
                if pos:
                    pos['x'] = pos.get('x', 0) + shift_x
                    pos['y'] = pos.get('y', 0) + shift_y

            # Recalculate bounds after shift
            max_x += shift_x
            max_y += shift_y

        # Calculate required room size
        new_dims = [max_x + margin, max_y + margin, self.room_dimensions[2]]

        if new_dims[0] > self.room_dimensions[0] or new_dims[1] > self.room_dimensions[1]:
            print(f"📐 Expanding room: {self.room_dimensions[0]:.1f}x{self.room_dimensions[1]:.1f} -> {new_dims[0]:.1f}x{new_dims[1]:.1f}")
            self.room_dimensions = new_dims

        # Regenerate room priors
        self.room_priors = get_room_priors(self.room_dimensions)

        # Update scene graph
        self.scene_graph = equipment_only + self.room_priors

    def _reconfigure_multiagent(self, user_feedback=None):
        """Reconfigure using multi-agent collaborative workflow with function tools.

        This is a more sophisticated alternative to _reconfigure_initial_design():
        - Uses full 3-agent pipeline (Planner → Layout Engineer → Engineer)
        - More collaborative reasoning about equipment changes
        - Still uses command-based approach for stability
        - Better for complex reconfiguration scenarios

        Args:
            user_feedback: Optional user feedback to incorporate

        Returns:
            Updated scene graph
        """
        from layout.layout_tools import LayoutBuilder
        from layout.layout_command_executor import LayoutCommandExecutor
        from agents.agents_multiagent_reconfig import (
            create_multiagent_reconfiguration_agents,
            build_multiagent_reconfiguration_prompt,
            MultiAgentReconfigurationChat
        )
        from autogen import GroupChatManager

        print("🔄 Starting multi-agent reconfiguration workflow...")

        # Initialize LayoutBuilder with existing layout
        builder = LayoutBuilder(self.scene_graph)
        executor = LayoutCommandExecutor(builder)

        # Get layout summary (not full JSON)
        layout_summary = builder.get_current_layout()
        print(f"   Baseline: {layout_summary['total_objects']} objects")

        # Create multi-agent team
        user_proxy, planner, layout_engineer, engineer = create_multiagent_reconfiguration_agents(
            layout_builder=builder,
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort
        )

        # Prepare user requirements
        user_requirements = self.user_input
        if user_feedback and not user_feedback.get("approved", True):
            user_requirements += f"\n\nADDITIONAL USER FEEDBACK:\n{user_feedback['feedback']}"

        # Create group chat with controlled speaking order
        groupchat = MultiAgentReconfigurationChat(
            agents=[user_proxy, planner, layout_engineer, engineer],
            messages=[],
            max_round=4  # Admin → Planner → Layout Engineer → Engineer
        )

        # Build prompt
        prompt = build_multiagent_reconfiguration_prompt(user_requirements, layout_summary)

        # Run multi-agent conversation
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.model_config,
            is_termination_msg=is_termination_msg
        )

        print("💬 Multi-agent team collaborating on reconfiguration...")
        user_proxy.initiate_chat(manager, message=prompt)

        # Get final commands from engineer
        engineer_msg = self._get_latest_agent_message(groupchat.messages, "Reconfiguration_Engineer")
        if engineer_msg is None:
            raise ValueError("Reconfiguration Engineer did not produce commands.")

        # Parse and execute commands
        print("📝 Executing reconfiguration commands from multi-agent team...")
        success, execution_message = executor.parse_and_execute_commands(engineer_msg["content"])

        # Get execution summary
        exec_summary = executor.get_execution_summary()
        print(f"✅ Executed {exec_summary['successful']}/{exec_summary['total_commands']} commands successfully")

        if exec_summary['failed'] > 0:
            print(f"⚠️ {exec_summary['failed']} commands failed:")
            for entry in exec_summary['log']:
                if not entry.get('success', True):
                    print(f"   - {entry.get('error', 'Unknown error')}")

        # Get reconfigured layout
        json_data = executor.get_final_layout()

        if not isinstance(json_data, dict) or "objects_in_room" not in json_data:
            raise ValueError("Multi-agent reconfiguration failed to produce valid layout.")

        # Update scene graph
        self.scene_graph = json_data
        self._normalize_scene_objects()
        self._enforce_equipment_catalog_names()

        # DO NOT reuse initial positions - we want to recalculate based on new placement relationships
        self._initial_design_loaded = False
        self._initial_design_source = None

        # Now calculate actual positions based on updated placement relationships
        print("🔄 Optimizing positions using graph-based solver...")

        # Use new optimizer instead of backtrack
        updated_graph, results = optimize_placement(
            scene_graph=self.scene_graph,
            room_dimensions=self.room_dimensions,
            room_priors=self.room_priors,
            verbose=True,
            preserve_existing=True
        )
        self.scene_graph = updated_graph["objects_in_room"]

        # Update room dimensions if optimizer expanded the room
        if "room_dimensions" in results:
            new_dims = results["room_dimensions"]
            if new_dims[0] > self.room_dimensions[0] or new_dims[1] > self.room_dimensions[1]:
                print(f"📐 Room expanded from {self.room_dimensions[:2]} to {new_dims[:2]}")
                self.room_dimensions = list(new_dims)  # Ensure it's a mutable list
                # Rebuild room priors with new dimensions
                self.room_priors = get_room_priors(self.room_dimensions)

        # Always expand room to fit all objects after reconfiguration (even for loaded designs)
        self._expand_room_to_fit_objects_unconditional(margin=1.0)

        get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

        # Mark that this was a reconfiguration so we can skip refiner
        self._was_reconfigured = True

        # Save conversation
        self._initial_conversation = {
            "source": "multiagent_reconfiguration",
            "conversation": [
                {"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]}
                for msg in groupchat.messages
            ],
            "command_execution_summary": exec_summary,
        }

        print(f"✨ Multi-agent reconfiguration complete! New layout has {len(json_data['objects_in_room'])} objects")

        return self.scene_graph

    def _split_scene_and_room_layout(self, objects):
        if not isinstance(objects, list):
            return [], []
        scene_objects = []
        layout_objects = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_copy = deepcopy(obj)
            obj_id = str(obj_copy.get("object_id", "")).lower()
            target = (
                layout_objects
                if obj_id in self.ROOM_LAYOUT_IDS_LOWER
                else scene_objects
            )
            target.append(obj_copy)
        return scene_objects, layout_objects

    def _infer_room_dimensions_from_layout(self, layout_objects):
        if not layout_objects:
            return None
        length = width = height = None
        for obj in layout_objects:
            obj_id = str(obj.get("object_id", "")).lower()
            size = obj.get("size_in_meters") or {}
            try:
                if obj_id in ("south_wall", "north_wall"):
                    if size.get("length") is not None:
                        length = float(size["length"])
                    if size.get("height") is not None:
                        height = float(size["height"])
                elif obj_id in ("east_wall", "west_wall"):
                    if size.get("length") is not None:
                        width = float(size["length"])
                    if size.get("height") is not None:
                        height = float(size["height"])
            except (TypeError, ValueError):
                continue
        if not all(value and value > 0 for value in (length, width, height)):
            return None
        return [round(length, 3), round(width, 3), round(height, 3)]

    def load_initial_design(self, design_source, infer_room_dimensions=True):
        """
        Load an externally provided scene graph and use it as the starting point.

        Args:
            design_source: Path-like, JSON-serializable dict, or list representing the scene graph.
            infer_room_dimensions: When True, attempt to set room dimensions from wall geometry.
        """
        if isinstance(design_source, (str, Path)):
            path = Path(design_source)
            with path.open() as file:
                payload = json.load(file)
            source_label = str(path)
        else:
            payload = deepcopy(design_source)
            source_label = "in-memory scene graph"

        if isinstance(payload, dict):
            objects = payload.get("objects_in_room")
            self._initial_design_was_list = False
        elif isinstance(payload, list):
            objects = payload
            self._initial_design_was_list = True
        else:
            raise ValueError("Initial design must be a path, dict with 'objects_in_room', or list of objects.")

        if not isinstance(objects, list):
            raise ValueError("Unable to interpret the provided initial design format.")

        scene_objects, layout_objects = self._split_scene_and_room_layout(objects)
        self._initial_design_snapshot = deepcopy(scene_objects)
        self._initial_object_lookup = {
            obj.get("object_id"): deepcopy(obj)
            for obj in scene_objects
            if isinstance(obj, dict) and obj.get("object_id")
        }
        self.scene_graph = {"objects_in_room": scene_objects}
        self._normalize_scene_objects()
        self._enforce_equipment_catalog_names()

        if layout_objects:
            self.room_priors = layout_objects
        else:
            self.room_priors = get_room_priors(self.room_dimensions)

        if infer_room_dimensions:
            inferred_dims = self._infer_room_dimensions_from_layout(layout_objects)
            if inferred_dims:
                self.room_dimensions = inferred_dims
                self.auto_room_dimensions = False
                self._auto_room_dimensions_result = inferred_dims

        self._initial_design_loaded = True
        self._initial_design_source = source_label
        self._initial_conversation = {
            "source": "user_provided_initial_design",
            "path": source_label,
            "notes": f"Loaded {len(scene_objects)} objects prior to corrections/refinements.",
        }
        print(f"📥 Loaded initial design from {source_label} with {len(scene_objects)} objects.")
        if layout_objects:
            print(f"↔️  Preserved {len(layout_objects)} room layout priors from the supplied file.")
        if infer_room_dimensions and self._auto_room_dimensions_result:
            dims = self._auto_room_dimensions_result
            print(f"📐 Using room dimensions inferred from initial design: {dims[0]}m x {dims[1]}m x {dims[2]}m")
        return self.scene_graph

    def correct_design(self, verbose=False, auto_prune=True, allow_object_deletions=False):
        # Skip correction for reconfigured layouts (already optimized)
        if getattr(self, '_was_reconfigured', False):
            if verbose:
                print("⏭️  Skipping correction for reconfigured layout (already optimized by reconfiguration workflow)")
            return

        # Correct Spatial Conflicts
        scene_graph = preprocess_scene_graph(self.scene_graph["objects_in_room"])
        G = build_graph(scene_graph)
        G = remove_unnecessary_edges(G)
        G, scene_graph = handle_under_prepositions(G, scene_graph)

        conflicts = get_conflicts(G, scene_graph)

        if verbose:
            print("-------------------CONFLICTS-------------------")
            for conflict in conflicts:
                print(conflict)
                print("\n\n")

        user_proxy, spatial_corrector_agent, json_schema_debugger, object_deletion_agent = get_corrector_agents(self.model_name, reasoning_effort=self.reasoning_effort)

        # Initialize correction conversations storage
        self._correction_conversations = {
            "spatial_corrections": [],
            "object_deletions": [],
            "pending_size_conflicts": []
        }

        while len(conflicts) > 0:
            spatial_corrector_agent.reset(), json_schema_debugger.reset()
            groupchat = LayoutCorrectorGroupChat(
                agents  =[user_proxy, spatial_corrector_agent, json_schema_debugger],
                messages=[],
                max_round=15
            )
            manager = GroupChatManager(groupchat=groupchat, llm_config=self.model_config, is_termination_msg=is_termination_msg)
            user_proxy.initiate_chat(
                manager,
                message=f"""
                {conflicts[0]}
                """,
            )
            correction = groupchat.messages[-2]
            # Extract JSON from correction response
            correction_json = self.extract_json_from_response(correction["content"])
            corrected_payload = correction_json.get("corrected_object")
            if isinstance(corrected_payload, dict) and "properties" in corrected_payload and isinstance(corrected_payload["properties"], dict):
                # Some agents respond with a JSON-schema-style wrapper; extract the actual object values.
                flattened = {}
                for key, value in corrected_payload["properties"].items():
                    if isinstance(value, dict):
                        if "default" in value:
                            flattened[key] = value["default"]
                        elif "const" in value:
                            flattened[key] = value["const"]
                        else:
                            flattened[key] = value
                    else:
                        flattened[key] = value
                correction_json["corrected_object"] = flattened
                corrected_payload = flattened
            corr_obj = get_object_from_scene_graph(correction_json["corrected_object"]["object_id"], scene_graph)
            corr_obj["is_on_the_floor"] = correction_json["corrected_object"]["is_on_the_floor"]
            corr_obj["facing"] = correction_json["corrected_object"]["facing"]
            corr_obj["placement"] = correction_json["corrected_object"]["placement"]
            scene_graph = preprocess_scene_graph(scene_graph)
            G = build_graph(scene_graph)
            conflicts = get_conflicts(G, scene_graph)
            
            # Save this spatial correction conversation
            spatial_correction_data = {
                "conflict": conflicts[0] if conflicts else "Resolved",
                "correction": correction_json,
                "conversation": [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in groupchat.messages]
            }
            self._correction_conversations["spatial_corrections"].append(spatial_correction_data)

        if auto_prune:
            size_conflicts = get_size_conflicts(G, scene_graph, self.user_input, self.room_priors, verbose)

            if verbose:
                print("-------------------SIZE CONFLICTS-------------------")
                for conflict in size_conflicts:
                    print(conflict)
                    print("\n\n")

            while len(size_conflicts) > 0:
                if not allow_object_deletions:
                    self._correction_conversations["pending_size_conflicts"].extend(size_conflicts)
                    if verbose:
                        print("⚠️ Size conflicts remain but object deletions are disabled.")
                    break
                object_deletion_agent.reset()
                groupchat = ObjectDeletionGroupChat(
                    agents  =[user_proxy, object_deletion_agent],
                    messages=[],
                    max_round=2
                )
                manager = GroupChatManager(groupchat=groupchat, llm_config=self.model_config, is_termination_msg=is_termination_msg)
                user_proxy.initiate_chat(
                    manager,
                    message=f"""
                    {size_conflicts[0]}
                    """,
                )
                correction = groupchat.messages[-1]
                correction_json = self.extract_json_from_response(correction["content"])
                object_to_delete = correction_json["object_to_delete"]
                descendants = nx.descendants(G, object_to_delete)
                objs_to_delete = descendants.union({object_to_delete})
                print("Objs to Delete: ", objs_to_delete)
                scene_graph = [x for x in scene_graph if x["object_id"] not in objs_to_delete]
                for obj in objs_to_delete:
                    G.remove_node(obj)

                size_conflicts = get_size_conflicts(G, scene_graph, self.user_input, self.room_priors, verbose)
                
                # Save this object deletion conversation
                deletion_data = {
                    "size_conflict": size_conflicts[0] if size_conflicts else "Resolved",
                    "deleted_object": object_to_delete,
                    "deleted_descendants": list(descendants),
                    "conversation": [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in groupchat.messages]
                }
                self._correction_conversations["object_deletions"].append(deletion_data)
                
        self.scene_graph["objects_in_room"] = scene_graph

    def refine_design(self, verbose=False):
        # Skip refinement for reconfigured layouts (agent already optimized placement)
        if getattr(self, '_was_reconfigured', False):
            if verbose:
                print("⏭️  Skipping refinement for reconfigured layout (already optimized by reconfiguration agent)")
            return

        cluster_dict = get_cluster_objects(self.scene_graph["objects_in_room"])

        # Initialize refinement conversations storage
        self._refinement_conversations = []

        inputs = []
        for key, value in cluster_dict.items():
            key = list(key)
            if len(key[0]) == 2:
                parent_id = key[0][0][1]
                prep = key[0][1][1]
            elif len(key[0]) == 3:
                parent_id = key[0][1][1]
                prep = key[0][2][1]
            objs = value

            inputs.append((parent_id, prep, objs))

        if verbose:
            if inputs == []:
                print("No clusters found")
            for parent_id, prep, objs in inputs:
                print(f"Parent Object : {parent_id}")
                print(f"Children Objects : {objs}")
                print(f"The children objects are '{prep}' the parent object")
                print("\n")


        for parent_id, prep, obj_names in inputs:
            objs = [get_object_from_scene_graph(obj, self.scene_graph["objects_in_room"]) for obj in obj_names]
            objs_rot = [get_rotation(obj, self.scene_graph["objects_in_room"]) for obj in objs]

            parent_obj = get_object_from_scene_graph(parent_id, self.scene_graph["objects_in_room"])
            if parent_obj is None:
                parent_obj = next((prior for prior in self.room_priors if prior.get("object_id") == parent_id), None)
            if parent_obj is None:
                if verbose:
                    print(f"⚠️ Skipping refinement for '{parent_id}' because it is missing from the scene graph.")
                continue
            parent_obj_rot = get_rotation(parent_obj, self.scene_graph["objects_in_room"])

            rot_diffs = [obj_rot - parent_obj_rot for obj_rot in objs_rot]
            direction_check = lambda diff, prep: (diff % 180 == 0 and prep in ["left of", "right of"]) or (diff % 180 != 0 and prep in ["in front", "behind"]) or (diff % 180 != 0 and prep == "on")
            possibilities_str = "Constraints:\n" + '\n'.join(["\t" + f"Place objects {'`behind` or `in front`' if direction_check(diff, prep) else '`left of` or `right of`'} of {name}!" for name, diff in zip(obj_names, rot_diffs)])

            user_proxy, layout_refiner, json_schema_debugger = get_refiner_agents(self.model_name, reasoning_effort=self.reasoning_effort)

            layout_refiner.reset(), json_schema_debugger.reset()
            groupchat = LayoutRefinerGroupChat(
                agents  =[user_proxy, layout_refiner, json_schema_debugger],
                messages=[],
                max_round=15
            )
            manager = GroupChatManager(groupchat=groupchat, llm_config=self.model_config, is_termination_msg=is_termination_msg)
            user_proxy.initiate_chat(
                manager,
                message=f"""
                Parent Object : {parent_id}
                Children Objects : {obj_names}

                {possibilities_str}

                The children objects are '{prep}' the parent object
                """,
            )

            new_relationships = self.extract_json_from_response(groupchat.messages[-2]["content"])
            if "items" in new_relationships["children_objects"]:
                new_relationships = {"children_objects" : new_relationships["children_objects"]["items"]}
            # Check whether the relationships are valid
            invalid_name_ids = set()
            for child in new_relationships["children_objects"]:
                child_obj = get_object_from_scene_graph(child["name_id"], self.scene_graph["objects_in_room"])
                if child_obj is None:
                    continue
                child_rot = get_rotation(child_obj, self.scene_graph["objects_in_room"])
                aligns_with_flow = direction_check(child_rot - parent_obj_rot, prep)
                allowed_preps = ["in front", "behind"] if aligns_with_flow else ["left of", "right of"]

                for other_child in child.get("placement", {}).get("children_objects", []):
                    if other_child["preposition"] not in allowed_preps:
                        invalid_name_ids.add(child["name_id"])
                        break

            if verbose:
                print("Invalid name IDs: ", list(invalid_name_ids))
            new_relationships["children_objects"] = [
                child
                for child in new_relationships["children_objects"]
                if child["name_id"] not in invalid_name_ids
            ]         
            
            if len(new_relationships["children_objects"]) == 0:
                continue

            edges, edges_to_flip = clean_and_extract_edges(new_relationships, parent_id, verbose=verbose)

            prep_correspondences ={
                "left of" : "right of",
                "right of" : "left of",
                "in front" : "behind",
                "behind" : "in front",
            }


            for obj in new_relationships["children_objects"]:
                name_id = obj["name_id"]
                rel = obj["placement"]["children_objects"]
                for r in rel:
                    if (name_id, r["name_id"]) in edges:
                        to_flip = edges_to_flip[(name_id, r["name_id"])]
                        if to_flip:
                            corr_obj = get_object_from_scene_graph(r["name_id"], self.scene_graph["objects_in_room"])
                            corr_prep = prep_correspondences[r["preposition"]]
                            corr_obj["placement"]["objects_in_room"].append({"object_id" : name_id, "preposition" : corr_prep, "is_adjacent" : r["is_adjacent"]})
                        else:
                            corr_obj = get_object_from_scene_graph(name_id, self.scene_graph["objects_in_room"])
                            corr_obj["placement"]["objects_in_room"].append({"object_id" : r["name_id"], "preposition" : r["preposition"], "is_adjacent" : r["is_adjacent"]})
            
            # Save this refinement conversation
            refinement_data = {
                "parent_object": parent_id,
                "children_objects": obj_names,
                "preposition": prep,
                "new_relationships": new_relationships,
                "conversation": [{"role": msg["role"], "name": msg.get("name", ""), "content": msg["content"]} for msg in groupchat.messages]
            }
            self._refinement_conversations.append(refinement_data)

    def create_object_clusters(self, verbose=False, relaxation_factor=1.2):
        """
        Create object clusters with constraint areas.

        Args:
            verbose: Print detailed information
            relaxation_factor: Multiplier to add slack to cluster constraints (default 1.2 = 20% larger)
                             Higher values give more flexibility, lower values are tighter
        """
        # Skip clustering for reconfigured layouts (already done in reconfiguration workflow)
        if getattr(self, '_was_reconfigured', False):
            if verbose:
                print("⏭️  Skipping clustering for reconfigured layout (already done in reconfiguration workflow)")
            return

        # Assign the rotations
        for obj in self.scene_graph["objects_in_room"]:
            rot = get_rotation(obj, self.scene_graph["objects_in_room"])
            obj["rotation"] = {"z_angle" : rot}

        ROOM_LAYOUT_ELEMENTS = ["south_wall", "north_wall", "west_wall", "east_wall", "ceiling", "middle of the room"]

        G = build_graph(self.scene_graph["objects_in_room"])
        nodes = G.nodes()

        # Create clusters with relaxation
        for node in nodes:
            if node not in ROOM_LAYOUT_ELEMENTS:
                cluster_size, children_objs = get_cluster_size(node, G, self.scene_graph["objects_in_room"])

                # Apply relaxation factor to give more space
                relaxed_cluster_size = {
                    "left of": cluster_size["left of"] * relaxation_factor,
                    "right of": cluster_size["right of"] * relaxation_factor,
                    "behind": cluster_size["behind"] * relaxation_factor,
                    "in front": cluster_size["in front"] * relaxation_factor
                }

                if verbose:
                    print("Node: ", node)
                    print("Original cluster size: ", cluster_size)
                    print("Relaxed cluster size: ", relaxed_cluster_size)
                    print("Children: ", children_objs)
                    print("\n")

                node_obj = get_object_from_scene_graph(node, self.scene_graph["objects_in_room"])
                cluster_size_mapped = {
                    "x_neg": relaxed_cluster_size["left of"],
                    "x_pos": relaxed_cluster_size["right of"],
                    "y_neg": relaxed_cluster_size["behind"],
                    "y_pos": relaxed_cluster_size["in front"]
                }
                node_obj["cluster"] = {"constraint_area": cluster_size_mapped}

    def _resize_room_based_on_clusters(self):
        """
        Resize the room if the object clusters are too large for the current dimensions.
        """
        if not isinstance(self.scene_graph, dict) or "objects_in_room" not in self.scene_graph:
            return

        # Determine max required dimensions
        max_req_x = 0.0
        max_req_y = 0.0
        
        for obj in self.scene_graph["objects_in_room"]:
            if "cluster" not in obj:
                continue
                
            cluster = obj["cluster"]["constraint_area"]
            size = obj["size_in_meters"]
            rot = obj.get("rotation", {}).get("z_angle", 0.0)
            
            # Unity convention: objects default to facing +Y (north)
            # - At rotation 0°/180°: length along Y, width along X
            # - At rotation 90°/270°: length along X, width along Y
            
            # For cluster constraints (x_neg/x_pos affect X, y_neg/y_pos affect Y):
            # At rotation 0°: width contributes to X, length contributes to Y
            if abs(rot % 180 - 90) < 1.0:
                # Rotated 90/270: length along X, width along Y
                req_global_x = cluster["x_neg"] + size["length"] + cluster["x_pos"]
                req_global_y = cluster["y_neg"] + size["width"] + cluster["y_pos"]
            else:
                # Rotation 0/180: width along X, length along Y
                req_global_x = cluster["x_neg"] + size["width"] + cluster["x_pos"]
                req_global_y = cluster["y_neg"] + size["length"] + cluster["y_pos"]
                
            max_req_x = max(max_req_x, req_global_x)
            max_req_y = max(max_req_y, req_global_y)
            
        # Add margin (2.0 meters for walls/circulation)
        margin = 2.0
        req_room_x = max_req_x + margin
        req_room_y = max_req_y + margin
        
        # Check if we need to resize
        updated = False
        if self.auto_room_dimensions:
            if req_room_x > self.room_dimensions[0]:
                self.room_dimensions[0] = round(req_room_x, 2)
                updated = True
            if req_room_y > self.room_dimensions[1]:
                self.room_dimensions[1] = round(req_room_y, 2)
                updated = True
        else:
             # Just warn if fixed dimensions are too small
             if req_room_x > self.room_dimensions[0] or req_room_y > self.room_dimensions[1]:
                 print(f"⚠️ Warning: Room dimensions ({self.room_dimensions}) may be too small for object clusters (required: {req_room_x:.1f}x{req_room_y:.1f})")

        if updated:
            print(f"📐 Auto-resized workspace to fit clusters: {self.room_dimensions[0]}x{self.room_dimensions[1]}x{self.room_dimensions[2]}m")
            # Update priors since they depend on room dimensions
            self.room_priors = get_room_priors(self.room_dimensions)

    def _expand_room_to_fit_layout(self, margin=1.0):
        """
        Expand the auto-sized room after placement so all positioned objects fit inside.
        """
        if not self.auto_room_dimensions or not isinstance(self.scene_graph, list):
            return

        layout_ids = self.ROOM_LAYOUT_IDS
        placed_objects = [
            obj for obj in self.scene_graph
            if obj.get("object_id") not in layout_ids and obj.get("position") and obj.get("size_in_meters")
        ]
        if not placed_objects:
            return

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for obj in placed_objects:
            pos = obj["position"]
            size = obj["size_in_meters"]
            rot = obj.get("rotation", {}).get("z_angle", 0.0)

            length = max(float(size.get("length", 0.0)), 0.05)
            width = max(float(size.get("width", 0.0)), 0.05)

            # Unity convention: at 0/180 length along Y, width along X; at 90/270 swap
            if abs(rot % 180 - 90) < 1.0:
                eff_x, eff_y = length, width
            else:
                eff_x, eff_y = width, length

            half_x, half_y = eff_x / 2.0, eff_y / 2.0

            min_x = min(min_x, pos["x"] - half_x)
            max_x = max(max_x, pos["x"] + half_x)
            min_y = min(min_y, pos["y"] - half_y)
            max_y = max(max_y, pos["y"] + half_y)

        if not all(math.isfinite(v) for v in [min_x, max_x, min_y, max_y]):
            return

        req_length = max_x + margin  # south-west corner at (0,0), so only extend positive axis
        req_width = max_y + margin

        updated = False
        if req_length > self.room_dimensions[0]:
            self.room_dimensions[0] = round(req_length, 2)
            updated = True
        if req_width > self.room_dimensions[1]:
            self.room_dimensions[1] = round(req_width, 2)
            updated = True

        if updated:
            print(f"📐 Expanded workspace after placement: {self.room_dimensions[0]}x{self.room_dimensions[1]}x{self.room_dimensions[2]}m")
            self.room_priors = get_room_priors(self.room_dimensions)
            equipment_only = [obj for obj in self.scene_graph if obj.get("object_id") not in layout_ids]
            self.scene_graph = equipment_only + self.room_priors

    def _expand_room_to_fit_objects_unconditional(self, margin=1.0):
        """
        Expand room to fit all positioned objects, regardless of auto_room_dimensions setting.
        Used after reconfiguration to ensure newly added equipment fits in the room.
        """
        if not isinstance(self.scene_graph, list):
            return

        layout_ids = self.ROOM_LAYOUT_IDS
        placed_objects = [
            obj for obj in self.scene_graph
            if obj.get("object_id") not in layout_ids and obj.get("position") and obj.get("size_in_meters")
        ]
        if not placed_objects:
            return

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for obj in placed_objects:
            pos = obj["position"]
            size = obj["size_in_meters"]
            rot = obj.get("rotation", {}).get("z_angle", 0.0)

            length = max(float(size.get("length", 0.0)), 0.05)
            width = max(float(size.get("width", 0.0)), 0.05)

            # Unity convention: at 0/180 length along Y, width along X; at 90/270 swap
            if abs(rot % 180 - 90) < 1.0:
                eff_x, eff_y = length, width
            else:
                eff_x, eff_y = width, length

            half_x, half_y = eff_x / 2.0, eff_y / 2.0

            min_x = min(min_x, pos["x"] - half_x)
            max_x = max(max_x, pos["x"] + half_x)
            min_y = min(min_y, pos["y"] - half_y)
            max_y = max(max_y, pos["y"] + half_y)

        if not all(math.isfinite(v) for v in [min_x, max_x, min_y, max_y]):
            return

        req_length = max_x + margin  # south-west corner at (0,0), so only extend positive axis
        req_width = max_y + margin

        updated = False
        if req_length > self.room_dimensions[0]:
            self.room_dimensions[0] = round(req_length, 2)
            updated = True
        if req_width > self.room_dimensions[1]:
            self.room_dimensions[1] = round(req_width, 2)
            updated = True

        if updated:
            print(f"📐 Expanded workspace to fit equipment: {self.room_dimensions[0]}x{self.room_dimensions[1]}x{self.room_dimensions[2]}m")
            self.room_priors = get_room_priors(self.room_dimensions)
            equipment_only = [obj for obj in self.scene_graph if obj.get("object_id") not in layout_ids]
            self.scene_graph = equipment_only + self.room_priors

    def _relax_adjacent_constraints(self, max_adjacent_per_object=2):
        """
        Automatically relax excessive is_adjacent constraints.

        Args:
            max_adjacent_per_object: Maximum number of is_adjacent constraints per object
        """
        if not isinstance(self.scene_graph, dict) or "objects_in_room" not in self.scene_graph:
            return

        relaxed_count = 0
        for obj in self.scene_graph["objects_in_room"]:
            placement = obj.get("placement", {})
            obj_refs = placement.get("objects_in_room", [])

            # Count is_adjacent constraints
            adjacent_refs = [ref for ref in obj_refs if ref.get("is_adjacent")]

            if len(adjacent_refs) > max_adjacent_per_object:
                # Keep only the first max_adjacent_per_object, relax the rest
                for i, ref in enumerate(obj_refs):
                    if ref.get("is_adjacent") and i >= max_adjacent_per_object:
                        ref["is_adjacent"] = False
                        relaxed_count += 1

        if relaxed_count > 0:
            print(f"🔧 Auto-relaxed {relaxed_count} excessive is_adjacent constraints")

    def backtrack(self, verbose=False, auto_relax=True):
        """
        Place objects in 3D space using constraint satisfaction.

        Args:
            verbose: Print detailed progress
            auto_relax: Automatically relax is_adjacent constraints (recommended)
        """
        # Skip backtracking for reconfigured layouts (already done in reconfiguration workflow)
        if getattr(self, '_was_reconfigured', False):
            if verbose:
                print("⏭️  Skipping backtracking for reconfigured layout (already done in reconfiguration workflow)")
            return

        # Auto-relax constraints before placement
        if auto_relax:
            self._relax_adjacent_constraints(max_adjacent_per_object=2)

        # Resize room if needed before flattening the graph
        self._resize_room_based_on_clusters()

        self.scene_graph = self.scene_graph["objects_in_room"] + self.room_priors
        prior_ids = ["south_wall", "north_wall", "east_wall", "west_wall", "ceiling", "middle of the room"]
        
        point_bbox = dict.fromkeys([item["object_id"] for item in self.scene_graph], False)

        # Place the objects that already have coordinates or can be trivially inferred
        for item in self.scene_graph:
            obj_id = item.get("object_id")
            if obj_id in prior_ids:
                continue

            if item.get("position"):
                point_bbox[obj_id] = True
                continue

            try:
                possible_pos = get_possible_positions(obj_id, self.scene_graph, self.room_dimensions)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error getting positions for {obj_id}: {e}")
                possible_pos = []

            # Determine the overlap based on the possible positions
            overlap = None
            if len(possible_pos) == 1:
                overlap = possible_pos[0]
            elif len(possible_pos) > 1:
                overlap = possible_pos[0]
                for pos in possible_pos[1:]:
                    overlap = calculate_overlap(overlap, pos)
            # If the overlap is a point bbox, assign the position
            if overlap is not None and is_point_bbox(overlap) and len(possible_pos) > 0:
                item["position"] = {"x" : overlap[0], "y" : overlap[2], "z" : overlap[4]}
                point_bbox[obj_id] = True
        
        scene_graph_wo_layout = [item for item in self.scene_graph if item["object_id"] not in prior_ids]
        if not scene_graph_wo_layout:
            if verbose:
                print("⚠️ No placeable objects found during backtrack step.")
            return
        object_ids = [item["object_id"] for item in scene_graph_wo_layout]
        # Get depths
        depth_scene_graph = get_depth(scene_graph_wo_layout)
        max_depth = max(depth_scene_graph.values())
        
        if verbose:
            print("Max depth: ", max_depth)
            print("Depth scene graph: ", depth_scene_graph)
            print("Point BBox: ", [key for key, value in point_bbox.items() if value])
            get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_initial.png")
            for obj in scene_graph_wo_layout:
                if "position" in obj.keys():
                    print(obj["object_id"], obj["position"])
        
        topological_order = get_topological_ordering(scene_graph_wo_layout)
        topological_order = [item for item in topological_order if item not in prior_ids]
        if verbose:
            print("Topological order: ", topological_order)
        
        d = 1
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        stuck_counter = 0  # Track how many times we're stuck at same depth
        last_depth = 0
        
        while d <= max_depth and iteration_count < max_iterations:   
            iteration_count += 1
            
            # Check if we're stuck (repeating same depth too many times)
            if d == last_depth:
                stuck_counter += 1
            else:
                stuck_counter = 0
                last_depth = d
                
            if stuck_counter > 50:  # If stuck at same depth for too long
                if verbose:
                    print(f"⚠️ Stuck at depth {d} for too long. Unable to find valid placement.")
                    unplaced_objects = []
                    for node in topological_order:
                        if depth_scene_graph[node] >= d and not point_bbox[node]:
                            obj = next(item for item in scene_graph_wo_layout if item["object_id"] == node)
                            if "position" not in obj:
                                unplaced_objects.append(node)
                    if unplaced_objects:
                        print(f"❌ Failed to place objects: {unplaced_objects}")
                        print("💡 Consider reducing object count or increasing room size")
                break
            
            if verbose:
                print(f"Depth: {d} (iteration {iteration_count})")
            error_flag = False
            
            # Get nodes at the current depth
            nodes = [node for node in topological_order if depth_scene_graph[node] == d]
            if verbose:
                print(f"Nodes at depth {d}:", nodes)
            
            errors = {}
            for node in nodes:
                if point_bbox[node]:
                    continue
                
                # Find the object corresponding to the current node
                obj = next(item for item in scene_graph_wo_layout if item["object_id"] == node)
                errors = place_object(obj, self.scene_graph, self.room_dimensions, errors={}, verbose=verbose)
                if verbose:
                    if errors:
                        print(f"❌ Errors for {obj['object_id']}:", errors)
                    else:
                        print(f"✓ {obj['object_id']} placed successfully")

                if errors:
                    if d > 1:
                        d -= 1
                        if verbose:
                            print(f"🔄 Backtracking to depth: {d}")
                    else:
                        # If we can't go back further, this object cannot be placed
                        if verbose:
                            print(f"❌ Cannot place {node} - no valid position found at depth 1")
                            print("💡 Consider reducing constraints or increasing workspace size")
                        # Don't set position, let the algorithm continue with what it has
                    
                    error_flag = True
                    # Delete positions for objects at or beyond the current depth
                    for del_item in scene_graph_wo_layout:
                        if depth_scene_graph[del_item["object_id"]] >= d:
                            if "position" in del_item.keys() and not point_bbox[del_item["object_id"]]:
                                if verbose:
                                    print(f"🗑️ Deleting position for: {del_item['object_id']}")
                                del del_item["position"]
                    errors = {}
                    break
                            
            if not error_flag:
                d += 1

        # Report final status
        placed_objects = []
        unplaced_objects = []
        for item in scene_graph_wo_layout:
            if "position" in item:
                placed_objects.append(item["object_id"])
            else:
                unplaced_objects.append(item["object_id"])
        
        if iteration_count >= max_iterations:
            if verbose:
                print(f"⚠️ Reached maximum iterations ({max_iterations}), stopping backtrack")
                print(f"📊 Placement summary: {len(placed_objects)} placed, {len(unplaced_objects)} failed")
                if unplaced_objects:
                    print(f"❌ Unplaced objects: {unplaced_objects}")
        elif verbose:
            print(f"✅ Backtracking completed in {iteration_count} iterations")
            print(f"📊 Successfully placed {len(placed_objects)} objects")
            if unplaced_objects:
                print(f"⚠️ Could not place {len(unplaced_objects)} objects: {unplaced_objects}")

        if self.auto_room_dimensions:
            self._expand_room_to_fit_layout(margin=1.0)

        if verbose:
            get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

    def optimize_positions(self, verbose: bool = False):
        """Optimize object positions using graph-based constraint satisfaction.

        This is a more robust alternative to backtrack() that:
        - Uses soft constraints with weights instead of hard failures
        - Handles conflicting constraints gracefully
        - Iteratively optimizes to minimize total constraint violation
        - Resolves collisions automatically

        Args:
            verbose: Print progress information
        """
        # Skip if already done via reconfiguration
        if getattr(self, '_was_reconfigured', False):
            if verbose:
                print("⏭️  Skipping optimization for reconfigured layout (already done)")
            return

        if verbose:
            print("🔧 Starting graph-based position optimization...")

        # Ensure scene_graph is in dict format
        if isinstance(self.scene_graph, list):
            self.scene_graph = {"objects_in_room": self.scene_graph}

        # Run optimization
        updated_graph, results = optimize_placement(
            scene_graph=self.scene_graph,
            room_dimensions=self.room_dimensions,
            room_priors=self.room_priors,
            verbose=verbose,
            preserve_existing=True
        )

        # Update scene graph (convert to flat list format for compatibility)
        self.scene_graph = updated_graph["objects_in_room"]

        # Update room dimensions if optimizer expanded the room
        if "room_dimensions" in results:
            new_dims = results["room_dimensions"]
            if new_dims[0] > self.room_dimensions[0] or new_dims[1] > self.room_dimensions[1]:
                if verbose:
                    print(f"📐 Room expanded from {self.room_dimensions[:2]} to {new_dims[:2]}")
                self.room_dimensions = new_dims
                # Rebuild room priors with new dimensions
                self.room_priors = get_room_priors(self.room_dimensions)

        if self.auto_room_dimensions:
            self._expand_room_to_fit_layout(margin=1.0)

        if verbose:
            placed_count = len(results.get("placed", []))
            unplaced = results.get("unplaced", [])
            print(f"✅ Optimization complete: {placed_count} objects placed")
            if unplaced:
                print(f"⚠️ Could not place: {unplaced}")
            get_visualization(self.scene_graph, self.room_priors, "scenes/visualization_final.png")

    def to_json(self, filename="scenes/scene_graph.json"):
        # Save the scene graph to a json file
        graph_with_connections = populate_conveyor_connections(self.scene_graph)
        output_graph = deepcopy(graph_with_connections)

        self._ensure_placement_defaults_in_graph(output_graph)
        
        # If the input was a list, the output should also be a list to maintain schema consistency
        if getattr(self, "_initial_design_was_list", False) and isinstance(output_graph, dict):
            output_data = output_graph.get("objects_in_room", [])
        else:
            output_data = output_graph

        with open(filename, "w") as file:
            json.dump(output_data, file, indent=4)
    
    def extract_rationale_from_content(self, content):
        """Extract rationale text from agent response, excluding JSON"""
        try:
            # Find JSON block and remove it
            import re
            json_block_pattern = r'```json\s*\{.*?\}\s*```'
            content_without_json = re.sub(json_block_pattern, '', content, flags=re.DOTALL)
            
            # Also remove standalone JSON at the end
            lines = content_without_json.strip().split('\n')
            filtered_lines = []
            in_json = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('{') and not in_json:
                    in_json = True
                    continue
                elif in_json and stripped.endswith('}'):
                    in_json = False
                    continue
                elif in_json:
                    continue
                else:
                    filtered_lines.append(line)
            
            rationale = '\n'.join(filtered_lines).strip()
            return rationale if len(rationale) > 10 else None  # Only save meaningful rationale
        except:
            return None

    def save_rationale_summary(self, filename=None):
        """Save only the rationale/reasoning parts in markdown format for easy reading"""
        if filename is None:
            filename = "scenes/scene_graph_rationale.md"
        
        # Ensure markdown extension
        if filename.endswith('.json'):
            filename = filename.replace('.json', '.md')
        elif not filename.endswith('.md'):
            filename += '.md'
        
        # Prepare markdown content
        markdown_content = []
        
        # Header with metadata
        markdown_content.append("# Design Rationale Report")
        markdown_content.append("")
        markdown_content.append("## Project Metadata")
        markdown_content.append(f"- **Timestamp**: {datetime.now().isoformat()}")
        markdown_content.append(f"- **Model**: {self.model_name}")
        markdown_content.append(f"- **Objects**: {self.no_of_objects}")
        markdown_content.append(f"- **User Input**: {self.user_input}")
        markdown_content.append(f"- **Room Dimensions**: {self.room_dimensions[0]}m × {self.room_dimensions[1]}m × {self.room_dimensions[2]}m")
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

        def build_structured_summary(agent_identifier, content):
            try:
                parsed = self.extract_json_from_response(content)
            except ValueError:
                return None

            def normalize(value):
                if isinstance(value, (list, tuple)):
                    return ", ".join(str(v) for v in value)
                if isinstance(value, dict):
                    try:
                        return json.dumps(value, ensure_ascii=False)
                    except Exception:
                        return str(value)
                return value

            blocks = []
            if isinstance(parsed, list):
                blocks = parsed
            elif isinstance(parsed, dict):
                extracted = extract_list_from_json(parsed)
                if isinstance(extracted, list):
                    blocks = extracted

            if not blocks:
                return None

            lines = []
            if agent_identifier == "Process_planner":
                lines.append("Equipment Selection Summary:")
                for item in blocks:
                    if not isinstance(item, dict):
                        continue
                    equipment = item.get("equipment_name") or item.get("name") or item.get("object_id") or "Equipment"
                    stage = normalize(item.get("process_stage"))
                    primary = normalize(item.get("primary_function"))
                    flow = normalize(item.get("flow_requirements"))
                    notes = normalize(item.get("supporting_notes"))

                    details = [equipment]
                    if stage:
                        details.append(f"stage: {stage}")
                    if primary:
                        details.append(f"function: {primary}")
                    if flow:
                        details.append(f"flow: {flow}")
                    if notes:
                        details.append(f"notes: {notes}")
                    lines.append(f"- {' | '.join(details)}")
            else:
                lines.append("Placement Summary:")
                for item in blocks:
                    if not isinstance(item, dict):
                        continue
                    name = (
                        item.get("equipment_name")
                        or item.get("object_id")
                        or item.get("object_name")
                        or item.get("object_id")
                        or item.get("name_id")
                        or "Object"
                    )
                    placement_info = item.get("placement")
                    proximity = normalize(item.get("proximity"))
                    facing = normalize(item.get("facing"))

                    details = []
                    if isinstance(placement_info, dict):
                        room_rels = placement_info.get("room_layout_elements")
                        obj_rels = placement_info.get("objects_in_room")
                        if isinstance(room_rels, list) and room_rels:
                            room_text = []
                            for rel in room_rels:
                                if isinstance(rel, dict):
                                    prep = rel.get("preposition", "")
                                    elem = rel.get("layout_element_id", "")
                                    combo = " ".join(filter(None, [prep, elem])).strip()
                                    if combo:
                                        room_text.append(combo)
                            if room_text:
                                details.append(f"layout: {', '.join(room_text)}")
                        if isinstance(obj_rels, list) and obj_rels:
                            obj_text = []
                            for rel in obj_rels:
                                if isinstance(rel, dict):
                                    prep = rel.get("preposition", "")
                                    obj = rel.get("object_id", "")
                                    combo = " ".join(filter(None, [prep, obj])).strip()
                                    if combo:
                                        obj_text.append(combo)
                            if obj_text:
                                details.append(f"objects: {', '.join(obj_text)}")
                    if proximity:
                        details.append(f"proximity: {proximity}")
                    if facing:
                        details.append(f"facing: {facing}")
                    details_text = " | ".join(details) if details else "no spatial details provided"
                    lines.append(f"- {name}: {details_text}")

            return "\n".join(lines) if len(lines) > 1 else None

        # Extract rationale from initial design, if available
        initial_conv = getattr(self, "_initial_conversation", None)
        if isinstance(initial_conv, dict):
            main_conv = initial_conv.get("main_conversation")
            if isinstance(main_conv, list) and main_conv:
                markdown_content.append("## Initial Design Rationale")
                markdown_content.append("")
                
                for msg in main_conv:
                    if msg.get("name") in ['Process_planner', 'Layout_engineer']:
                        rationale = self.extract_rationale_from_content(msg.get('content', ''))
                        if not rationale:
                            rationale = build_structured_summary(msg.get('name', ''), msg.get('content', ''))
                        if rationale:
                            agent_name = "Process Planner" if msg.get('name') == 'Process_planner' else "Layout Engineer"
                            markdown_content.append(f"### {agent_name}")
                            markdown_content.append("")
                            markdown_content.append(rationale)
                            markdown_content.append("")

        # Extract rationale from corrections
        if hasattr(self, '_correction_conversations'):
            corrections = self._correction_conversations
            
            spatial_corrections = []
            object_deletions = []
            
            for correction in corrections.get("spatial_corrections", []):
                for msg in correction.get("conversation", []):
                    if msg['name'] == 'Spatial_corrector_agent':
                        rationale = self.extract_rationale_from_content(msg['content'])
                        if rationale:
                            spatial_corrections.append({
                                "conflict": correction.get("conflict", ""),
                                "rationale": rationale
                            })
            
            for deletion in corrections.get("object_deletions", []):
                for msg in deletion.get("conversation", []):
                    if msg['name'] == 'Object_deletion_agent':
                        rationale = self.extract_rationale_from_content(msg['content'])
                        if rationale:
                            object_deletions.append({
                                "conflict": deletion.get("size_conflict", ""),
                                "deleted_object": deletion.get("deleted_object", ""),
                                "rationale": rationale
                            })
            
            if spatial_corrections or object_deletions:
                markdown_content.append("## Design Corrections")
                markdown_content.append("")
                
                if spatial_corrections:
                    markdown_content.append("### Spatial Conflict Resolutions")
                    markdown_content.append("")
                    for i, correction in enumerate(spatial_corrections, 1):
                        markdown_content.append(f"#### Correction {i}")
                        if correction["conflict"]:
                            markdown_content.append(f"**Conflict**: {correction['conflict']}")
                            markdown_content.append("")
                        markdown_content.append(correction["rationale"])
                        markdown_content.append("")
                
                if object_deletions:
                    markdown_content.append("### Object Deletions")
                    markdown_content.append("")
                    for i, deletion in enumerate(object_deletions, 1):
                        markdown_content.append(f"#### Deletion {i}")
                        if deletion["deleted_object"]:
                            markdown_content.append(f"**Deleted Object**: {deletion['deleted_object']}")
                        if deletion["conflict"]:
                            markdown_content.append(f"**Conflict**: {deletion['conflict']}")
                        markdown_content.append("")
                        markdown_content.append(deletion["rationale"])
                        markdown_content.append("")

        # Extract rationale from refinements
        if hasattr(self, '_refinement_conversations'):
            refinements = []
            
            for refinement in self._refinement_conversations:
                for msg in refinement.get("conversation", []):
                    if msg['name'] == 'Layout_refiner':
                        rationale = self.extract_rationale_from_content(msg['content'])
                        if rationale:
                            refinements.append({
                                "parent_object": refinement.get("parent_object", ""),
                                "children_objects": refinement.get("children_objects", []),
                                "rationale": rationale
                            })
            
            if refinements:
                markdown_content.append("## Design Refinements")
                markdown_content.append("")
                
                for i, refinement in enumerate(refinements, 1):
                    markdown_content.append(f"### Refinement {i}")
                    if refinement["parent_object"]:
                        markdown_content.append(f"**Parent Object**: {refinement['parent_object']}")
                    if refinement["children_objects"]:
                        children_list = ", ".join(refinement["children_objects"])
                        markdown_content.append(f"**Children Objects**: {children_list}")
                    markdown_content.append("")
                    markdown_content.append(refinement["rationale"])
                    markdown_content.append("")
        
        # Write markdown file
        with open(filename, "w", encoding='utf-8') as file:
            file.write("\n".join(markdown_content))
        
        return filename

    def simulate_process_flow(
        self,
        trials=200,
        process_sequence=None,
        max_flow_gap=6.0,
        seed=None,
        verbose=False,
    ):
        """
        Run a lightweight discrete simulation to verify process flow correctness.

        Args:
            trials: Number of Monte Carlo trials to evaluate success rate.
            process_sequence: Optional explicit stage order (e.g., ["material_flow", "assembly_collaboration"]).
            max_flow_gap: Maximum allowed distance (meters) between consecutive stages before
                they are treated as disconnected.
            seed: Optional RNG seed for reproducibility.
            verbose: When True, print a short simulation summary table.

        Returns:
            SimulationReport emitted by ProcessFlowSimulator.
        """
        if self.scene_graph is None:
            raise ValueError("No scene graph available. Generate a design before simulating.")

        simulator = ProcessFlowSimulator(
            scene_graph=self.scene_graph,
            room_dimensions=self.room_dimensions,
            process_sequence=process_sequence,
            max_flow_gap=max_flow_gap,
            seed=seed,
        )
        report = simulator.run(trials=trials, verbose=verbose)
        self._last_simulation_report = report.to_dict()
        return report

    def evaluate_hybrid_performance(
        self,
        max_iterations=50,
        simulation_time=10000.0,
        convergence_threshold=0.01,
        seed=None,
        verbose=False,
    ) -> HybridSimulationReport:
        """
        Run hybrid performance evaluation using decomposition method.

        Based on Mastrangelo & Tolio (2024) hybrid approach combining:
        - Analytical models (Markov Chain) for simple subsystems
        - Discrete event simulation for complex subsystems
        - Remote Models for blocking/starvation propagation

        Args:
            max_iterations: Maximum iterations for convergence algorithm.
            simulation_time: Time units for each simulation evaluation.
            convergence_threshold: Relative throughput difference for convergence.
            seed: Optional RNG seed for reproducibility.
            verbose: Print progress information.

        Returns:
            HybridSimulationReport with comprehensive performance metrics including:
            - throughput (with confidence interval)
            - work-in-progress (WIP)
            - blocking/starvation probabilities
            - per-stage breakdown
            - bottleneck identification
        """
        if self.scene_graph is None:
            raise ValueError("No scene graph available. Generate a design before evaluating.")

        evaluator = HybridPerformanceEvaluator(
            scene_graph=self.scene_graph,
            room_dimensions=self.room_dimensions,
            seed=seed,
        )
        report = evaluator.run(
            max_iterations=max_iterations,
            simulation_time=simulation_time,
            convergence_threshold=convergence_threshold,
            verbose=verbose,
        )
        self._last_hybrid_report = report.to_dict()
        return report

    def interactive_design_workflow(self):
        """
        Complete interactive design workflow with user confirmation steps.
        Similar to deep research pattern - plan, review, confirm, execute.
        """
        print("🚀 Starting Interactive Design Workflow")
        print("=" * 80)
        
        # Step 1: Create initial plan
        print("📋 Step 1: Creating Initial Plan...")
        try:
            self.create_initial_plan()
            print("✅ Initial plan created successfully!")
        except Exception as e:
            print(f"❌ Error creating plan: {str(e)}")
            return False
        
        # Step 2: Display plan for user review
        print("\n📖 Step 2: Displaying Plan for Review...")
        self.display_plan_summary()
        
        # Step 3: Get user feedback
        feedback = self.get_user_feedback()
        
        # Step 4: Handle feedback and proceed
        if feedback["approved"]:
            print("\n✅ Plan approved! Proceeding with detailed design generation...")
        else:
            print(f"\n📝 Feedback received: {feedback['feedback']}")
            print("🔄 Incorporating feedback into design...")
        
        # Step 5: Create detailed design with feedback
        print("\n🏗️ Step 3: Generating Detailed Design...")
        try:
            self.create_initial_design(user_feedback=feedback)
            print("✅ Detailed design created successfully!")
        except Exception as e:
            print(f"❌ Error creating detailed design: {str(e)}")
            return False
        
        # Step 6: Corrections and refinements
        print("\n🔧 Step 4: Applying Design Corrections...")
        try:
            self.correct_design()
            print("✅ Design corrections completed!")
        except Exception as e:
            print(f"❌ Error in design corrections: {str(e)}")
            return False
        
        print("\n🎨 Step 5: Refining Design...")
        try:
            self.refine_design(verbose=True)  # Enable verbose to generate initial visualization
            print("✅ Design refinements completed!")
        except Exception as e:
            print(f"❌ Error in design refinements: {str(e)}")
            return False
        
        print("\n🔗 Step 6: Creating Object Clusters...")
        try:
            self.create_object_clusters(verbose=False)
            print("✅ Object clusters created!")
        except Exception as e:
            print(f"❌ Error creating object clusters: {str(e)}")
            return False
        
        print("\n🔄 Step 7: Final Layout Optimization...")
        try:
            self.backtrack(verbose=True)  # Enable verbose to generate final visualization
            print("✅ Final optimization completed!")
        except Exception as e:
            print(f"❌ Error in final optimization: {str(e)}")
            return False
        
        # Display final results
        if getattr(self, "auto_room_dimensions", False):
            dims = self.room_dimensions
            print(f"\n📐 Final workspace dimensions: {dims[0]}m × {dims[1]}m × {dims[2]}m")
        
        print("\n🎉 Interactive Design Workflow Completed Successfully!")
        print("=" * 80)
        
        return True

    def save_interactive_design(self, scenario_name="interactive_design"):
        """Save the results of an interactive design session"""
        if not self.scene_graph:
            print("❌ No design to save. Please complete the design workflow first.")
            return False
        
        try:
            # Save scene graph
            output_filename = f"scenes/scene_graph_{scenario_name.lower().replace(' ', '_')}.json"
            self.to_json(output_filename)
            
            # Save rationale
            rationale_filename = self.save_rationale_summary(f"scenes/scene_graph_{scenario_name.lower().replace(' ', '_')}_rationale.md")
            
            print(f"✅ Design saved successfully!")
            print(f"📄 Scene graph: {output_filename}")
            print(f"🧠 Rationale: {rationale_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving design: {str(e)}")
            return False
