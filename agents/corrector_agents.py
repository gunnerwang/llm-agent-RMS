import autogen
from autogen.agentchat.groupchat import GroupChat
from autogen.agentchat.agent import Agent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.assistant_agent import AssistantAgent
from copy import deepcopy
from jsonschema import validate
import json
import re
import ast

from core.schemas import layout_corrector_schema, deletion_schema
from agents.agents import is_termination_msg, get_json_config, get_model_config, get_rationale_config

class JSONSchemaAgent(UserProxyAgent):
    def __init__(self, name : str, is_termination_msg):
        super().__init__(name, is_termination_msg=is_termination_msg, code_execution_config=False)

    def get_human_input(self, prompt: str) -> str:
        message = self.last_message()
        preps_layout = ["left-side", "right-side", "in the middle"]
        preps_objs = ['on', 'left of', 'right of', 'in front', 'behind', 'under', 'above', 'through']

        pattern = r'```json\s*([^`]+)\s*```' # Match the json object
        match_result = re.search(pattern, message["content"], re.DOTALL)
        if not match_result:
            # Try to find any JSON-like structure
            pattern_fallback = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match_result = re.search(pattern_fallback, message["content"], re.DOTALL)
            if not match_result:
                return "ERROR: No JSON found in response. Please provide a valid JSON object wrapped in ```json ``` code block."
        match = match_result.group(1) if match_result.lastindex else match_result.group(0)

        def parse_candidate(candidate: str):
            # First, try to fix common Python-to-JSON issues
            fixed = candidate.strip()
            # Replace Python booleans/None with JSON equivalents
            fixed = re.sub(r'\bTrue\b', 'true', fixed)
            fixed = re.sub(r'\bFalse\b', 'false', fixed)
            fixed = re.sub(r'\bNone\b', 'null', fixed)
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            
            # Try parsing with ast.literal_eval first (more robust for single quotes)
            try:
                parsed = ast.literal_eval(fixed)
                return json.loads(json.dumps(parsed))
            except (ValueError, SyntaxError, TypeError):
                # Fallback to direct JSON parsing if ast fails
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError as err:
                    # Try original candidate with ast as last resort
                    try:
                        parsed = ast.literal_eval(candidate)
                        return json.loads(json.dumps(parsed))
                    except (ValueError, SyntaxError, TypeError):
                        raise ValueError(f"Failed to parse JSON: {err}") from err

        json_obj_new = parse_candidate(match)

        is_success  = False
        try:
            validate(instance=json_obj_new, schema=layout_corrector_schema)
            is_success = True
        except Exception as e:
            feedback = str(e.message)
            if e.validator == "enum":
                if str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_objs}"
                elif str(preps_layout) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_layout}"
        if is_success:
            return "SUCCESS"
        return feedback

def get_corrector_agents(model_name="gpt-4o", reasoning_effort="none"):
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message = "A human admin.",
        is_termination_msg = is_termination_msg,
        human_input_mode = "NEVER",
        code_execution_config=False
    )

    json_schema_debugger = JSONSchemaAgent(
        name = "Json_schema_debugger",
        is_termination_msg = is_termination_msg,
    )

    spatial_corrector_agent = AssistantAgent(
        name="Spatial_corrector_agent",
        llm_config=get_rationale_config(model_name, reasoning_effort=reasoning_effort),
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",
        system_message=f"""
        Spatial Corrector Agent. Whenever a user provides an object that don't fit the room for various spatial conflicts,
        You are going to make changes to its "scene_graph" and "facing_object" keys so that these conflicts are removed. 
        You are going to use the JSON Schema to validate the JSON object that the user provides.

        For relative placement with other objects in the room use the prepositions "on", "left of", "right of", "in front", "behind", "under".
        For relative placement with the room layout elements (walls, the middle of the room, ceiling) use the prepositions "on", "in the corner".

        RESPONSE FORMAT:
        First, explain your analysis of the spatial conflict and your reasoning for the correction strategy.
        Then, provide the corrected configuration in a ```json code block following this schema:
        {layout_corrector_schema}
        """
    )

    object_deletion_agent = AssistantAgent(
        name="Object_deletion_agent",
        llm_config=get_rationale_config(model_name, temperature=0.0, reasoning_effort=reasoning_effort),
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",
        system_message=f"""
        Object Deletion Agent. When a user provides a list of objects that doesn't fit the room, select one object to delete that would be less essential for the room.

        RESPONSE FORMAT:
        First, explain your analysis of the size conflicts and reasoning for which object to delete.
        Then, provide the JSON output following this schema:
        {deletion_schema}
        """
    )
    return user_proxy, json_schema_debugger, spatial_corrector_agent, object_deletion_agent
