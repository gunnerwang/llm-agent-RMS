import json

from llm4rms.agents import generate_layout
from llm4rms.llm import extract_json

PLAN = {"stages": [{"name": "transport", "equipment": [{"type": "conveyor", "count": 2}]}],
        "rationale": "Two conveyor segments."}
BROKEN = {"objects": [{"id": "c1", "type": "conveyor"}, {"id": "c2", "type": "belt"}],
          "flow": [["c1", "c2"]]}
FIXED = {"objects": [{"id": "c1", "type": "conveyor"}, {"id": "c2", "type": "conveyor"}],
         "flow": [["c1", "c2"]]}


class ScriptedChat:
    """Returns prepared replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def test_repairs_an_invalid_scene_graph():
    chat = ScriptedChat([json.dumps(PLAN), json.dumps(BROKEN), "```json\n" + json.dumps(FIXED) + "\n```"])
    result = generate_layout("Move parts between two stations.", 12, 8, chat)
    assert result.errors == []
    assert result.repairs == 1
    assert [o.type for o in result.graph.objects] == ["conveyor", "conveyor"]
    assert (result.graph.room_length, result.graph.room_width) == (12, 8)
    repair_request = chat.calls[-1][-1]["content"]
    assert "unknown equipment type 'belt'" in repair_request


def test_stops_after_max_repairs():
    chat = ScriptedChat([json.dumps(PLAN)] + [json.dumps(BROKEN)] * 3)
    result = generate_layout("Move parts between two stations.", 12, 8, chat, max_repairs=2)
    assert result.repairs == 2
    assert result.errors


def test_unparseable_reply_is_sent_back_for_repair():
    chat = ScriptedChat([json.dumps(PLAN), "Sorry, here is the layout.", json.dumps(FIXED)])
    result = generate_layout("Move parts between two stations.", 12, 8, chat)
    assert result.errors == [] and result.repairs == 1


def test_extract_json_handles_fences_and_nesting():
    reply = 'Here it is:\n```json\n{"room": {"length": 5}, "objects": []}\n```\nDone.'
    assert extract_json(reply) == {"room": {"length": 5}, "objects": []}
