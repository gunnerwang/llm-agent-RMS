"""Scene graph written by the layout engineer agent, and its validation.

A scene graph lists equipment instances, material flow edges between them,
and pairs of instances that should stand next to each other without being
part of the material flow (for example, a tool cabinet beside a workstation).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .catalog import CATALOG

Edge = Tuple[str, str]


@dataclass
class SceneObject:
    id: str
    type: str
    stage: str = ""


@dataclass
class SceneGraph:
    room_length: float
    room_width: float
    objects: List[SceneObject]
    flow: List[Edge] = field(default_factory=list)
    adjacent: List[Edge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        """Build a scene graph from parsed JSON. Raises ValueError on malformed input."""
        if not isinstance(data, dict):
            raise ValueError("scene graph must be a JSON object")
        room = data.get("room") or {}
        objects = []
        for i, obj in enumerate(data.get("objects") or []):
            if not isinstance(obj, dict) or "id" not in obj or "type" not in obj:
                raise ValueError(f"object #{i + 1} needs an 'id' and a 'type'")
            objects.append(SceneObject(str(obj["id"]), str(obj["type"]), str(obj.get("stage", ""))))
        try:
            length = float(room.get("length", 0))
            width = float(room.get("width", 0))
        except (TypeError, ValueError):
            raise ValueError("room length and width must be numbers") from None
        return cls(
            room_length=length,
            room_width=width,
            objects=objects,
            flow=_edges(data.get("flow"), "flow"),
            adjacent=_edges(data.get("adjacent"), "adjacent"),
        )

    def to_dict(self) -> dict:
        return {
            "room": {"length": self.room_length, "width": self.room_width},
            "objects": [{"id": o.id, "type": o.type, "stage": o.stage} for o in self.objects],
            "flow": [list(e) for e in self.flow],
            "adjacent": [list(e) for e in self.adjacent],
        }

    def validate(self) -> List[str]:
        """Return a list of problems; an empty list means the scene graph is usable."""
        errors = []
        if self.room_length <= 0 or self.room_width <= 0:
            errors.append("room length and width must be positive")
        if not self.objects:
            errors.append("the scene graph has no objects")
        ids = [o.id for o in self.objects]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            errors.append("duplicate object ids: " + ", ".join(duplicates))
        for obj in self.objects:
            if obj.type not in CATALOG:
                errors.append(f"{obj.id}: unknown equipment type '{obj.type}'")
        known = set(ids)
        for kind, edges in (("flow", self.flow), ("adjacent", self.adjacent)):
            for a, b in edges:
                if a == b:
                    errors.append(f"{kind} edge [{a}, {b}] connects an object to itself")
                for end in (a, b):
                    if end not in known:
                        errors.append(f"{kind} edge [{a}, {b}] refers to unknown object '{end}'")
        return errors


def _edges(raw, kind: str) -> List[Edge]:
    """Accept edges as [a, b] pairs or as {"from": a, "to": b} objects."""
    edges = []
    for item in raw or []:
        if isinstance(item, dict) and "from" in item and "to" in item:
            edges.append((str(item["from"]), str(item["to"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            edges.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"{kind} edge {item!r} must be a pair of object ids")
    return edges
