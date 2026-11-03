"""Requirement analyst and architect agents, with a check-and-repair loop."""

import json
from dataclasses import dataclass, field
from typing import List, Optional

from .catalog import catalog_summary
from .llm import ChatFn, extract_json
from .scene_graph import SceneGraph

ANALYST_PROMPT = """You are the requirement analyst in a team that designs layouts for
reconfigurable manufacturing systems. Split the production requirement into
process stages in the order the parts go through them, and choose the
equipment for each stage.

Use only these equipment types:
{catalog}

Reply with one JSON object:
{{"stages": [{{"name": "<stage>", "equipment": [{{"type": "<equipment type>", "count": <integer>}}]}}],
  "rationale": "<two or three sentences>"}}"""

ARCHITECT_PROMPT = """You are the architect in a team that designs layouts for reconfigurable
manufacturing systems. Turn the process plan you receive into a layout scene
graph for a room of {length} m x {width} m.

Give every piece of equipment a unique id. Add a "flow" edge [from_id, to_id]
for each step the material takes from one piece of equipment to the next.
Add an "adjacent" pair [id, id] for equipment that should stand next to other
equipment without being part of the material flow, such as a tool cabinet
beside a workstation.

Use only these equipment types:
{catalog}

Reply with one JSON object:
{{"room": {{"length": {length}, "width": {width}}},
  "objects": [{{"id": "<id>", "type": "<equipment type>", "stage": "<stage>"}}],
  "flow": [["<from_id>", "<to_id>"]],
  "adjacent": [["<id>", "<id>"]]}}"""

REPAIR_PROMPT = """The scene graph has these problems:
{errors}
Reply with the corrected scene graph as one JSON object in the same format."""


@dataclass
class DesignResult:
    plan: dict
    graph: Optional[SceneGraph]
    errors: List[str] = field(default_factory=list)  # problems left after the last repair
    repairs: int = 0


def analyze_requirements(requirement: str, chat: ChatFn) -> dict:
    """Ask the requirement analyst for process stages and equipment."""
    messages = [
        {"role": "system", "content": ANALYST_PROMPT.format(catalog=catalog_summary())},
        {"role": "user", "content": requirement},
    ]
    return extract_json(chat(messages))


def design_scene_graph(plan: dict, room_length: float, room_width: float, chat: ChatFn,
                       max_repairs: int = 2) -> DesignResult:
    """Ask the architect for a scene graph and send any problems back for repair."""
    system = ARCHITECT_PROMPT.format(catalog=catalog_summary(), length=room_length, width=room_width)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(plan, indent=2)},
    ]
    repairs = 0
    while True:
        reply = chat(messages)
        graph, errors = _parse_scene_graph(reply, room_length, room_width)
        if not errors or repairs >= max_repairs:
            return DesignResult(plan, graph, errors, repairs)
        problems = "\n".join(f"- {e}" for e in errors)
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": REPAIR_PROMPT.format(errors=problems)},
        ]
        repairs += 1


def generate_layout(requirement: str, room_length: float, room_width: float, chat: ChatFn,
                    max_repairs: int = 2) -> DesignResult:
    """Run the requirement analyst and the architect on a natural-language requirement."""
    plan = analyze_requirements(requirement, chat)
    return design_scene_graph(plan, room_length, room_width, chat, max_repairs)


def _parse_scene_graph(reply: str, room_length: float, room_width: float):
    try:
        graph = SceneGraph.from_dict(extract_json(reply))
    except ValueError as exc:  # also covers json.JSONDecodeError
        return None, [str(exc)]
    graph.room_length, graph.room_width = room_length, room_width  # the room is given, not designed
    return graph, graph.validate()
