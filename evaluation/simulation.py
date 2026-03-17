from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from catalog.equipment_catalog_cell import EQUIPMENT_CATALOG, get_equipment_aliases

ROOM_LAYOUT_ELEMENTS = {
    "south_wall",
    "north_wall",
    "west_wall",
    "east_wall",
    "ceiling",
    "middle of the room",
}

DEFAULT_PROCESS_SEQUENCE: Tuple[str, ...] = (
    "MaterialFlow",
    "Assembly",
    "Utilities",
)

DEFAULT_STAGE_BEHAVIOR: Dict[str, Dict[str, float]] = {
    "MaterialFlow": {
        "base_time": 4.5,
        "volume_factor": 0.35,
        "failure_rate": 0.04,
        "transfer_speed": 1.1,
        "default_transfer": 2.0,
    },
    "Assembly": {
        "base_time": 6.5,
        "volume_factor": 0.30,
        "failure_rate": 0.06,
        "transfer_speed": 1.0,
        "default_transfer": 2.5,
    },
    "Utilities": {
        "base_time": 2.0,
        "volume_factor": 0.10,
        "failure_rate": 0.02,
        "transfer_speed": 0.5,
        "default_transfer": 1.0,
    },
}

FALLBACK_STAGE_BEHAVIOR = {
    "base_time": 5.0,
    "volume_factor": 0.30,
    "failure_rate": 0.05,
    "transfer_speed": 1.0,
    "default_transfer": 2.5,
}


@dataclass(frozen=True)
class EquipmentNode:
    """Immutable wrapper around a scene graph entry."""

    object_id: str
    canonical_name: str
    stage: Optional[str]
    raw: Dict
    position: Optional[Tuple[float, float, float]]

    def footprint(self) -> float:
        size = self.raw.get("size_in_meters") or {}
        length = float(size.get("length", 1.0) or 1.0)
        width = float(size.get("width", 1.0) or 1.0)
        return max(length, 0.2) * max(width, 0.2)

    def volume(self) -> float:
        size = self.raw.get("size_in_meters") or {}
        height = float(size.get("height", 1.0) or 1.0)
        return self.footprint() * max(height, 0.2)


@dataclass
class SimulationReport:
    trials: int
    success_rate: float
    average_cycle_time: Optional[float]
    cycle_time_std: Optional[float]
    throughput_per_hour: Optional[float]
    efficiency_index: Optional[float]
    bottleneck_stage: Optional[str]
    stage_breakdown: Dict[str, Dict[str, Optional[float]]]
    missing_stages: List[str]
    disconnected_pairs: List[Tuple[str, str]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "trials": self.trials,
            "success_rate": self.success_rate,
            "average_cycle_time": self.average_cycle_time,
            "cycle_time_std": self.cycle_time_std,
            "throughput_per_hour": self.throughput_per_hour,
            "efficiency_index": self.efficiency_index,
            "bottleneck_stage": self.bottleneck_stage,
            "stage_breakdown": self.stage_breakdown,
            "missing_stages": self.missing_stages,
            "disconnected_pairs": self.disconnected_pairs,
            "notes": self.notes,
        }


class ProcessFlowSimulator:
    """Light-weight Monte Carlo simulator for validating process flows."""

    def __init__(
        self,
        scene_graph,
        room_dimensions: Optional[Sequence[float]] = None,
        process_sequence: Optional[Iterable[str]] = None,
        max_flow_gap: float = 6.0,
        seed: Optional[int] = None,
    ):
        self.room_dimensions = room_dimensions
        self.alias_map = get_equipment_aliases()
        self.max_flow_gap = max_flow_gap
        self.rng = random.Random(seed)
        self._notes: List[str] = []

        self.nodes = self._build_nodes(scene_graph)
        self.stage_objects = self._group_nodes_by_stage(self.nodes)
        self.sequence = self._resolve_sequence(process_sequence)

        self._warn_if_missing_positions()

    def run(self, trials: int = 200, verbose: bool = False) -> SimulationReport:
        if not self.sequence:
            raise ValueError(
                "No process sequence available for simulation. Pass process_sequence or ensure catalog items exist."
            )

        validation = self.validate_process_flow()
        if validation["missing_stages"]:
            self._notes.append(
                f"Missing stages from sequence: {', '.join(validation['missing_stages'])}"
            )
        if validation["disconnected_pairs"]:
            pairs = [f"{a}->{b}" for a, b in validation["disconnected_pairs"]]
            self._notes.append(f"Disconnected stage pairs: {', '.join(pairs)}")

        success_count = 0
        success_times: List[float] = []
        stage_totals = {stage: 0.0 for stage in self.sequence}

        for _ in range(trials):
            success, total_time, stage_times = self._simulate_trial()
            if success:
                success_count += 1
                success_times.append(total_time)
                for stage, spent in stage_times.items():
                    stage_totals[stage] += spent

        success_rate = success_count / trials if trials else 0.0
        average_cycle = (
            sum(success_times) / len(success_times) if success_times else None
        )
        cycle_std = (
            math.sqrt(
                sum((t - average_cycle) ** 2 for t in success_times)
                / len(success_times)
            )
            if success_times and average_cycle is not None
            else None
        )
        throughput = 3600.0 / average_cycle if average_cycle else None

        reference_cycle = sum(
            self._stage_behavior(stage)["base_time"] for stage in self.sequence
        )
        efficiency_index = (
            success_rate * reference_cycle / average_cycle
            if average_cycle
            else None
        )

        total_stage_time = sum(stage_totals.values())
        stage_breakdown = {}
        bottleneck_stage = None
        bottleneck_value = -1.0

        for stage in self.sequence:
            avg_stage_time = (
                stage_totals[stage] / len(success_times) if success_times else None
            )
            utilization = (
                stage_totals[stage] / total_stage_time if total_stage_time > 0 else None
            )
            stage_breakdown[stage] = {
                "average_time": avg_stage_time,
                "utilization": utilization,
                "equipment_count": len(self.stage_objects.get(stage, [])),
            }
            if utilization is not None and utilization > bottleneck_value:
                bottleneck_value = utilization
                bottleneck_stage = stage

        if verbose:
            self._log_summary(success_rate, average_cycle, throughput, stage_breakdown)

        return SimulationReport(
            trials=trials,
            success_rate=success_rate,
            average_cycle_time=average_cycle,
            cycle_time_std=cycle_std,
            throughput_per_hour=throughput,
            efficiency_index=efficiency_index,
            bottleneck_stage=bottleneck_stage,
            stage_breakdown=stage_breakdown,
            missing_stages=validation["missing_stages"],
            disconnected_pairs=validation["disconnected_pairs"],
            notes=self._notes.copy(),
        )

    def validate_process_flow(self) -> Dict[str, List]:
        missing = [
            stage
            for stage in self.sequence
            if not self.stage_objects.get(stage)
        ]
        disconnected = self._find_disconnected_pairs()
        return {"missing_stages": missing, "disconnected_pairs": disconnected}

    # Internal helpers -------------------------------------------------

    def _build_nodes(self, scene_graph) -> List[EquipmentNode]:
        if isinstance(scene_graph, dict):
            objects = scene_graph.get("objects_in_room", [])
        elif isinstance(scene_graph, list):
            objects = scene_graph
        else:
            raise TypeError("scene_graph must be a dict or list.")

        nodes: List[EquipmentNode] = []
        for obj in objects:
            object_id = obj.get("object_id")
            if not object_id or object_id in ROOM_LAYOUT_ELEMENTS:
                continue
            if obj.get("itemType") in {"wall", "floor", "ceiling"}:
                continue
            canonical = self._canonicalize(object_id)
            stage = (
                EQUIPMENT_CATALOG.get(canonical, {}).get("process_stage")
                if canonical
                else None
            )
            position = self._extract_position(obj)
            nodes.append(
                EquipmentNode(
                    object_id=object_id,
                    canonical_name=canonical or object_id,
                    stage=stage,
                    raw=obj,
                    position=position,
                )
            )
        return nodes

    def _group_nodes_by_stage(
        self, nodes: List[EquipmentNode]
    ) -> Dict[str, List[EquipmentNode]]:
        stage_map: Dict[str, List[EquipmentNode]] = {}
        for node in nodes:
            if not node.stage:
                continue
            stage_map.setdefault(node.stage, []).append(node)
        return stage_map

    def _resolve_sequence(
        self, process_sequence: Optional[Iterable[str]]
    ) -> List[str]:
        if process_sequence:
            normalized = []
            seen = set()
            for stage in process_sequence:
                canonical = self._normalize_stage_name(stage)
                if canonical and canonical not in seen:
                    normalized.append(canonical)
                    seen.add(canonical)
            return normalized

        derived = [
            stage for stage in DEFAULT_PROCESS_SEQUENCE if stage in self.stage_objects
        ]
        remaining = [
            stage
            for stage in self.stage_objects.keys()
            if stage not in derived
        ]
        return derived + remaining

    def _warn_if_missing_positions(self) -> None:
        missing = [
            node.object_id for node in self.nodes if node.position is None
        ]
        if missing:
            self._notes.append(
                f"Missing absolute positions for {len(missing)} object(s): {', '.join(missing[:5])}"
            )

    def _simulate_trial(self) -> Tuple[bool, float, Dict[str, float]]:
        stage_times = {stage: 0.0 for stage in self.sequence}
        total_time = 0.0
        previous_node: Optional[EquipmentNode] = None

        for stage in self.sequence:
            node = self._select_node(stage, previous_node)
            if not node:
                return False, total_time, stage_times

            behavior = self._stage_behavior(stage)
            process_time = self._estimate_process_time(node, behavior)

            if self.rng.random() < behavior["failure_rate"]:
                total_time += process_time
                stage_times[stage] += process_time
                return False, total_time, stage_times

            transfer_time = (
                self._estimate_transfer_time(previous_node, node, behavior)
                if previous_node
                else 0.0
            )

            total_time += process_time + transfer_time
            stage_times[stage] += process_time
            previous_node = node

        return True, total_time, stage_times

    def _select_node(
        self, stage: str, previous_node: Optional[EquipmentNode]
    ) -> Optional[EquipmentNode]:
        candidates = self.stage_objects.get(stage, [])
        if not candidates:
            return None
        if not previous_node or previous_node.position is None:
            return candidates[0]

        def distance_to_prev(node: EquipmentNode) -> float:
            dist = self._distance(previous_node, node)
            return dist if dist is not None else float("inf")

        return min(candidates, key=distance_to_prev)

    def _estimate_process_time(
        self, node: EquipmentNode, behavior: Dict[str, float]
    ) -> float:
        volume = node.volume()
        return behavior["base_time"] + behavior["volume_factor"] * volume

    def _estimate_transfer_time(
        self,
        previous: EquipmentNode,
        current: EquipmentNode,
        behavior: Dict[str, float],
    ) -> float:
        distance = self._distance(previous, current)
        if distance is None:
            return behavior["default_transfer"]
        return distance / max(behavior["transfer_speed"], 0.2)

    def _distance(
        self, node_a: EquipmentNode, node_b: EquipmentNode
    ) -> Optional[float]:
        if node_a.position is None or node_b.position is None:
            return None
        ax, ay, _ = node_a.position
        bx, by, _ = node_b.position
        return math.dist((ax, ay), (bx, by))

    def _find_disconnected_pairs(self) -> List[Tuple[str, str]]:
        disconnected: List[Tuple[str, str]] = []
        for idx in range(len(self.sequence) - 1):
            current_stage = self.sequence[idx]
            next_stage = self.sequence[idx + 1]
            current_nodes = self.stage_objects.get(current_stage, [])
            next_nodes = self.stage_objects.get(next_stage, [])
            if not current_nodes or not next_nodes:
                continue

            if not self._has_plausible_connection(current_nodes, next_nodes):
                disconnected.append((current_stage, next_stage))
        return disconnected

    def _has_plausible_connection(
        self,
        current_nodes: List[EquipmentNode],
        next_nodes: List[EquipmentNode],
    ) -> bool:
        for src in current_nodes:
            for dst in next_nodes:
                distance = self._distance(src, dst)
                if distance is not None and distance <= self.max_flow_gap:
                    return True
                if self._are_objects_related(src, dst):
                    return True
        return False

    def _are_objects_related(
        self, src: EquipmentNode, dst: EquipmentNode
    ) -> bool:
        placement = src.raw.get("placement", {})
        relations = placement.get("objects_in_room", []) or []
        for rel in relations:
            if rel.get("object_id") == dst.object_id:
                return True
        placement_dst = dst.raw.get("placement", {})
        relations_dst = placement_dst.get("objects_in_room", []) or []
        for rel in relations_dst:
            if rel.get("object_id") == src.object_id:
                return True
        return False

    def _stage_behavior(self, stage: str) -> Dict[str, float]:
        return DEFAULT_STAGE_BEHAVIOR.get(stage, FALLBACK_STAGE_BEHAVIOR)

    def _canonicalize(self, object_id: str) -> Optional[str]:
        base = object_id.rsplit("_", 1)[0]
        if base in EQUIPMENT_CATALOG:
            return base
        lower = base.lower()
        if lower in self.alias_map:
            return self.alias_map[lower]
        for name in EQUIPMENT_CATALOG:
            if name.lower() == lower:
                return name
        return None

    def _normalize_stage_name(self, stage: str) -> Optional[str]:
        normalized = stage.strip().lower().replace(" ", "_")
        return normalized or None

    def _extract_position(self, obj: Dict) -> Optional[Tuple[float, float, float]]:
        pos = obj.get("position")
        if not isinstance(pos, dict):
            return None
        try:
            return (
                float(pos["x"]),
                float(pos["y"]),
                float(pos.get("z", 0.0)),
            )
        except (TypeError, ValueError, KeyError):
            return None

    def _log_summary(
        self,
        success_rate: float,
        average_cycle: Optional[float],
        throughput: Optional[float],
        stage_breakdown: Dict[str, Dict[str, Optional[float]]],
    ) -> None:
        print("=== Process Flow Simulation Summary ===")
        print(f"Success rate: {success_rate:.2%}")
        if average_cycle is not None:
            print(f"Average cycle time: {average_cycle:.2f}s")
        if throughput is not None:
            print(f"Estimated throughput: {throughput:.1f} units/hour")
        for stage, stats in stage_breakdown.items():
            avg = stats.get("average_time")
            util = stats.get("utilization")
            count = stats.get("equipment_count")
            avg_text = f"{avg:.2f}s" if avg is not None else "n/a"
            util_text = f"{util:.2%}" if util is not None else "n/a"
            print(f"- {stage}: avg={avg_text} util={util_text} equipment={count}")


def _load_scene_graph(path: Path):
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        # Support files that only contain objects list
        return {"objects_in_room": data}
    return data


def _print_cli_report(report: SimulationReport) -> None:
    print("\n=== Simulation Report ===")
    print(f"Trials: {report.trials}")
    print(f"Success rate: {report.success_rate:.2%}")
    if report.average_cycle_time:
        print(f"Average cycle time: {report.average_cycle_time:.2f}s")
    if report.throughput_per_hour:
        print(f"Throughput: {report.throughput_per_hour:.1f} units/hour")
    if report.bottleneck_stage:
        print(f"Bottleneck stage: {report.bottleneck_stage}")

    if report.stage_breakdown:
        print("\nStage breakdown:")
        for stage, stats in report.stage_breakdown.items():
            avg = stats.get("average_time")
            util = stats.get("utilization")
            avg_text = f"{avg:.2f}s" if avg is not None else "n/a"
            util_text = f"{util:.1%}" if util is not None else "n/a"
            equip = stats.get("equipment_count", 0)
            print(f"- {stage}: avg={avg_text}, util={util_text}, equipment={equip}")

    if report.missing_stages:
        print(f"\nMissing stages: {', '.join(report.missing_stages)}")
    if report.disconnected_pairs:
        formatted = ", ".join(f"{a}->{b}" for a, b in report.disconnected_pairs)
        print(f"Disconnected stage pairs: {formatted}")
    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"- {note}")
    print("=====================================")


def _parse_room_dims(values: Optional[List[str]]) -> Optional[List[float]]:
    if not values:
        return None
    if len(values) != 3:
        raise ValueError("Room dimensions must provide exactly three numbers (length width height).")
    return [float(v) for v in values]


def main():
    parser = argparse.ArgumentParser(description="Simulate process flow for a generated scene graph.")
    parser.add_argument("scene_graph", type=Path, help="Path to scene_graph.json generated by LLM4RMS.")
    parser.add_argument(
        "--room-dims",
        nargs=3,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        help="Override room dimensions in meters (default: use stored dimensions or auto).",
    )
    parser.add_argument(
        "--process-sequence",
        nargs="+",
        help="Optional explicit process stage sequence (e.g., material_flow assembly_collaboration).",
    )
    parser.add_argument("--trials", type=int, default=200, help="Number of simulation trials (default: 200).")
    parser.add_argument(
        "--max-flow-gap",
        type=float,
        default=6.0,
        help="Max distance between consecutive stages before they are considered disconnected (default: 6.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed iteration summary in addition to the final report.",
    )

    args = parser.parse_args()
    scene_graph = _load_scene_graph(args.scene_graph)
    room_dims = _parse_room_dims(args.room_dims)

    simulator = ProcessFlowSimulator(
        scene_graph=scene_graph,
        room_dimensions=room_dims,
        process_sequence=args.process_sequence,
        max_flow_gap=args.max_flow_gap,
        seed=args.seed,
    )
    report = simulator.run(trials=args.trials, verbose=args.verbose)
    _print_cli_report(report)


if __name__ == "__main__":
    main()
