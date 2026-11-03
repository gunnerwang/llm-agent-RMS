"""2D plot of a placed layout."""

from pathlib import Path
from typing import Optional

from .catalog import CATALOG
from .placement import Layout
from .scene_graph import SceneGraph


def plot_layout(graph: SceneGraph, layout: Layout, path: Path, title: Optional[str] = None) -> None:
    from matplotlib.figure import Figure  # no pyplot, so no GUI backend is needed
    from matplotlib.patches import Rectangle

    scale = 10.0 / max(graph.room_length, graph.room_width)
    fig = Figure(figsize=(graph.room_length * scale + 2.5, graph.room_width * scale + 1))
    ax = fig.subplots()
    ax.add_patch(Rectangle((0, 0), graph.room_length, graph.room_width, fill=False, linewidth=1.5))

    categories = sorted({CATALOG[p.type].category for p in layout.placements})
    colours = {c: f"C{i % 10}" for i, c in enumerate(categories)}
    for p in layout.placements:
        x0, y0, _, _ = p.bounds
        category = CATALOG[p.type].category
        ax.add_patch(Rectangle((x0, y0), p.length, p.width, facecolor=colours[category],
                               alpha=0.35, edgecolor=colours[category], linewidth=1.2, zorder=1))
        ax.text(p.x, p.y, p.id, ha="center", va="center", fontsize=6, zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    pos = layout.by_id()
    for a, b in graph.flow:
        if a in pos and b in pos:
            ax.annotate("", xy=(pos[b].x, pos[b].y), xytext=(pos[a].x, pos[a].y), zorder=2,
                        arrowprops=dict(arrowstyle="->", color="0.25", linewidth=0.9,
                                        shrinkA=10, shrinkB=10))

    handles = [Rectangle((0, 0), 1, 1, facecolor=colours[c], alpha=0.35, edgecolor=colours[c])
               for c in categories]
    ax.legend(handles, [c.replace("_", " ") for c in categories], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    ax.set_xlim(-0.5, graph.room_length + 0.5)
    ax.set_ylim(-0.5, graph.room_width + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
