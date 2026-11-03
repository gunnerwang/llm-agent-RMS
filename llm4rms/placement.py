"""Row-based placement of a scene graph along the material flow.

Objects are ordered along the material flow and placed in rows parallel to
the x axis. Consecutive rows run in opposite directions, so the flow
continues where the previous row ended. Objects outside the material flow
are placed right after the object they are adjacent to. Every object keeps
its catalog orientation.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .catalog import CATALOG
from .scene_graph import SceneGraph


@dataclass
class Placement:
    id: str
    type: str
    x: float  # centre, metres
    y: float  # centre, metres
    length: float
    width: float

    @property
    def bounds(self):
        return (self.x - self.length / 2, self.y - self.width / 2,
                self.x + self.length / 2, self.y + self.width / 2)


@dataclass
class Layout:
    placements: List[Placement] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)

    def by_id(self) -> Dict[str, Placement]:
        return {p.id: p for p in self.placements}


def flow_order(graph: SceneGraph) -> List[str]:
    """Order object ids along the material flow.

    Flow objects are sorted topologically, keeping the order in which they
    appear in the scene graph among equals. If the flow contains a loop, the
    remaining objects follow in their original order. Objects outside the
    flow come right after the object they are adjacent to, or at the end.
    """
    ids = [o.id for o in graph.objects]
    rank = {oid: i for i, oid in enumerate(ids)}
    in_flow = {oid for edge in graph.flow for oid in edge}

    indegree = {oid: 0 for oid in in_flow}
    successors: Dict[str, List[str]] = {oid: [] for oid in in_flow}
    for a, b in graph.flow:
        successors[a].append(b)
        indegree[b] += 1

    order: List[str] = []
    ready = sorted((oid for oid in in_flow if indegree[oid] == 0), key=rank.get)
    while ready:
        current = ready.pop(0)
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=rank.get)
    order += sorted((oid for oid in in_flow if oid not in order), key=rank.get)

    neighbours: Dict[str, List[str]] = {}
    for a, b in graph.adjacent:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    for oid in ids:
        if oid in in_flow:
            continue
        anchor = next((n for n in neighbours.get(oid, []) if n in order), None)
        if anchor is None:
            order.append(oid)
            continue
        position = order.index(anchor) + 1
        while position < len(order) and order[position] not in in_flow:
            position += 1  # keep earlier companions of the same anchor first
        order.insert(position, oid)
    return order


def place(graph: SceneGraph, clearance: float = 1.0, aisle: float = 2.0,
          margin: float = 1.0) -> Layout:
    """Place the objects of a valid scene graph in serpentine rows."""
    types = {o.id: o.type for o in graph.objects}
    x_min, x_max = margin, graph.room_length - margin
    y_max = graph.room_width - margin

    layout = Layout()
    row_y = margin        # lower edge of the current row
    row_depth = 0.0       # widest object in the current row
    direction = 1         # +1: left to right, -1: right to left
    cursor = x_min        # next free x coordinate in the current direction

    def fits(length: float) -> bool:
        return cursor + length <= x_max if direction == 1 else cursor - length >= x_min

    for oid in flow_order(graph):
        equipment = CATALOG[types[oid]]
        length, width = equipment.length, equipment.width
        if not fits(length) and row_depth > 0:
            row_y += row_depth + aisle
            row_depth = 0.0
            direction = -direction
            cursor = x_min if direction == 1 else x_max
        if not fits(length) or row_y + width > y_max:
            layout.unplaced.append(oid)
            continue
        x = cursor + direction * length / 2
        layout.placements.append(Placement(oid, types[oid], x, row_y + width / 2, length, width))
        cursor += direction * (length + clearance)
        row_depth = max(row_depth, width)
    return layout


def check_layout(graph: SceneGraph, layout: Layout) -> List[str]:
    """Report unplaced objects, objects outside the room, and overlaps."""
    issues = [f"{oid}: no space left in the room" for oid in layout.unplaced]
    for p in layout.placements:
        x0, y0, x1, y1 = p.bounds
        if x0 < 0 or y0 < 0 or x1 > graph.room_length or y1 > graph.room_width:
            issues.append(f"{p.id}: outside the room")
    placed = layout.placements
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            ax0, ay0, ax1, ay1 = a.bounds
            bx0, by0, bx1, by1 = b.bounds
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                issues.append(f"{a.id} overlaps {b.id}")
    return issues


def flow_distance(graph: SceneGraph, layout: Layout) -> float:
    """Sum of rectilinear centre-to-centre distances over the flow edges that were placed."""
    pos = layout.by_id()
    return sum(
        abs(pos[a].x - pos[b].x) + abs(pos[a].y - pos[b].y)
        for a, b in graph.flow
        if a in pos and b in pos
    )
