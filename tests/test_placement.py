import json
from pathlib import Path

from llm4rms.placement import check_layout, flow_distance, flow_order, place
from llm4rms.scene_graph import SceneGraph, SceneObject

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "assembly_cell.json"


def example_graph():
    return SceneGraph.from_dict(json.loads(EXAMPLE.read_text()))


def test_order_follows_flow_and_keeps_companions_next_to_anchor():
    order = flow_order(example_graph())
    assert order == [
        "raw_rack", "loading_robot", "cnc_1", "cnc_2", "buffer", "assembly",
        "cobot", "tools", "inspection", "conveyor", "packing", "outbound",
    ]


def test_order_survives_a_loop():
    graph = SceneGraph(10, 10, [SceneObject("a", "conveyor"), SceneObject("b", "conveyor")],
                       flow=[("a", "b"), ("b", "a")])
    assert sorted(flow_order(graph)) == ["a", "b"]


def test_example_fits_without_overlaps():
    graph = example_graph()
    layout = place(graph)
    assert layout.unplaced == []
    assert len(layout.placements) == len(graph.objects)
    assert check_layout(graph, layout) == []
    assert flow_distance(graph, layout) > 0


def test_rows_alternate_direction():
    graph = example_graph()
    pos = place(graph).by_id()
    first_row_y = pos["raw_rack"].y
    second_row = [p for p in pos.values() if p.y > first_row_y + 1]
    assert second_row, "the example should need a second row"
    xs = [pos[i].x for i in ["inspection", "conveyor", "packing", "outbound"]]
    assert xs == sorted(xs, reverse=True)  # the second row runs right to left


def test_small_room_reports_unplaced_objects():
    graph = example_graph()
    graph.room_length, graph.room_width = 10, 6
    layout = place(graph)
    assert layout.unplaced
    issues = check_layout(graph, layout)
    assert any("no space left" in i for i in issues)
    assert not any("overlaps" in i or "outside" in i for i in issues)
