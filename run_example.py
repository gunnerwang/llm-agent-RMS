"""Generate a layout from a production requirement, or place an existing scene graph."""

import argparse
import json
import sys
from pathlib import Path

from llm4rms.agents import generate_layout
from llm4rms.placement import check_layout, flow_distance, place
from llm4rms.scene_graph import SceneGraph
from llm4rms.visualize import plot_layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--requirement", help="production requirement in natural language (calls the LLM)")
    source.add_argument("--scene", type=Path, help="existing scene graph JSON (no LLM call)")
    parser.add_argument("--length", type=float, default=24.0, help="room length in m, with --requirement (default: 24)")
    parser.add_argument("--width", type=float, default=14.0, help="room width in m, with --requirement (default: 14)")
    parser.add_argument("--model", default="gpt-4o", help="chat model name (default: gpt-4o)")
    parser.add_argument("--out", type=Path, default=Path("output/layout"),
                        help="output path without extension (default: output/layout)")
    args = parser.parse_args()

    if args.scene:
        graph = SceneGraph.from_dict(json.loads(args.scene.read_text()))
        errors = graph.validate()
        plan = None
    else:
        from llm4rms.llm import openai_chat

        result = generate_layout(args.requirement, args.length, args.width, openai_chat(args.model))
        graph, errors, plan = result.graph, result.errors, result.plan
        print(f"Requirement analysis: {plan.get('rationale', '')}")
        print(f"Repair rounds: {result.repairs}")
    if errors:
        print("The scene graph is not usable:", *errors, sep="\n  ")
        return 1

    layout = place(graph)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "plan": plan,
        "scene_graph": graph.to_dict(),
        "placements": [vars(p) for p in layout.placements],
        "unplaced": layout.unplaced,
    }
    json_path, png_path = args.out.with_suffix(".json"), args.out.with_suffix(".png")
    json_path.write_text(json.dumps(output, indent=2))
    plot_layout(graph, layout, png_path)

    print(f"Placed {len(layout.placements)} of {len(graph.objects)} objects; "
          f"flow distance {flow_distance(graph, layout):.1f} m")
    for issue in check_layout(graph, layout):
        print(f"  {issue}")
    print(f"Wrote {json_path} and {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
