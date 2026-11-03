"""Generic equipment catalog.

Footprints are rounded, representative values in metres (length along x,
width along y). They do not describe specific commercial models.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EquipmentType:
    name: str
    category: str
    length: float
    width: float
    description: str


_TYPES = [
    EquipmentType("storage_rack", "storage", 2.0, 1.0, "Shelving for raw material or finished parts"),
    EquipmentType("pallet_station", "storage", 1.2, 1.0, "Floor position for an incoming or outgoing pallet"),
    EquipmentType("conveyor", "material_handling", 3.0, 0.8, "Straight belt conveyor segment"),
    EquipmentType("buffer_table", "material_handling", 1.2, 0.8, "Small buffer between two process steps"),
    EquipmentType("agv_dock", "material_handling", 1.5, 1.0, "Docking point for an automated guided vehicle"),
    EquipmentType("industrial_robot", "robot", 1.5, 1.5, "Floor-mounted articulated robot on a pedestal"),
    EquipmentType("collaborative_robot", "robot", 1.0, 1.0, "Collaborative robot on a mobile table"),
    EquipmentType("cnc_machine", "machining", 3.0, 2.0, "CNC machining centre"),
    EquipmentType("assembly_workstation", "assembly", 2.0, 1.2, "Manual assembly bench"),
    EquipmentType("inspection_station", "quality", 1.5, 1.0, "Measurement or vision inspection station"),
    EquipmentType("packaging_station", "packaging", 2.0, 1.5, "Packing and labelling station"),
    EquipmentType("tool_cabinet", "support", 1.0, 0.6, "Cabinet for tools and fixtures"),
]

CATALOG: Dict[str, EquipmentType] = {t.name: t for t in _TYPES}


def catalog_summary() -> str:
    """Describe the catalog in one line per equipment type, for agent prompts."""
    return "\n".join(
        f"- {t.name} ({t.category}, {t.length} x {t.width} m): {t.description}"
        for t in CATALOG.values()
    )
