import json
from pathlib import Path

import pytest

from llm4rms.scene_graph import SceneGraph

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "assembly_cell.json"


def example_dict():
    return json.loads(EXAMPLE.read_text())


def test_example_is_valid():
    assert SceneGraph.from_dict(example_dict()).validate() == []


def test_round_trip():
    graph = SceneGraph.from_dict(example_dict())
    assert SceneGraph.from_dict(graph.to_dict()) == graph


def test_reports_unknown_type_and_missing_ids():
    data = example_dict()
    data["objects"][0]["type"] = "laser_cutter"
    data["flow"].append(["packing", "ghost"])
    errors = SceneGraph.from_dict(data).validate()
    assert any("unknown equipment type 'laser_cutter'" in e for e in errors)
    assert any("unknown object 'ghost'" in e for e in errors)


def test_reports_duplicate_ids():
    data = example_dict()
    data["objects"][1]["id"] = "raw_rack"
    errors = SceneGraph.from_dict(data).validate()
    assert any("duplicate object ids: raw_rack" in e for e in errors)


def test_accepts_from_to_edges():
    data = example_dict()
    data["flow"] = [{"from": a, "to": b} for a, b in data["flow"]]
    graph = SceneGraph.from_dict(data)
    assert graph.flow[0] == ("raw_rack", "loading_robot")
    assert graph.validate() == []


def test_rejects_object_without_type():
    data = example_dict()
    del data["objects"][2]["type"]
    with pytest.raises(ValueError, match="needs an 'id' and a 'type'"):
        SceneGraph.from_dict(data)
