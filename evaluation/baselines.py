"""
Baseline Layout Generation Methods for Experimental Comparison

This module implements several baseline methods for manufacturing layout generation:
1. Random Layout: Random placement within room constraints
2. Grid Layout: Regular grid-based placement
3. Genetic Algorithm (GA): Evolutionary optimization
4. Simulated Annealing (SA): Temperature-based optimization
5. SLP-inspired: Systematic Layout Planning heuristic

These baselines can be compared against the hybrid LLM+structure-aware method.
"""

import json
import random
import math
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catalog.equipment_catalog import EQUIPMENT_CATALOG, get_equipment_info
from core.utils import get_room_priors


class BaselineMethod(Enum):
    """Enumeration of available baseline methods"""
    RANDOM = "random"
    GRID = "grid"
    GENETIC_ALGORITHM = "ga"
    SIMULATED_ANNEALING = "sa"
    SLP = "slp"  # Systematic Layout Planning
    # Hybrid methods: LLM4RMS + optimization
    LLM4RMS_GA = "llm4rms+ga"  # LLM4RMS layout refined by GA
    LLM4RMS_SA = "llm4rms+sa"  # LLM4RMS layout refined by SA


@dataclass
class LayoutConfig:
    """Configuration for layout generation"""
    room_dimensions: Tuple[float, float, float]  # (length, width, height)
    equipment_list: List[str]  # List of equipment names to place
    min_clearance: float = 1.0  # Minimum clearance between equipment (meters)
    wall_margin: float = 1.5  # Minimum distance from walls (meters)
    seed: Optional[int] = None  # Random seed for reproducibility

    # GA parameters
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8

    # SA parameters
    sa_initial_temp: float = 1000.0
    sa_cooling_rate: float = 0.995
    sa_min_temp: float = 1.0
    sa_iterations_per_temp: int = 50


@dataclass
class PlacedEquipment:
    """Represents a placed piece of equipment"""
    object_id: str
    equipment_type: str
    x: float
    y: float
    z: float = 0.0
    rotation: float = 0.0  # z_angle in degrees
    size: Dict[str, float] = field(default_factory=dict)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x_min, x_max, y_min, y_max) considering rotation"""
        length = self.size.get("length", 1.0)
        width = self.size.get("width", 1.0)

        # Swap dimensions for 90/270 degree rotations
        if self.rotation in [90.0, 270.0]:
            length, width = width, length

        half_l = length / 2
        half_w = width / 2

        return (self.x - half_l, self.x + half_l, self.y - half_w, self.y + half_w)


class BaselineLayoutGenerator:
    """Base class for layout generation methods"""

    def __init__(self, config: LayoutConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # Pre-compute equipment sizes
        self.equipment_sizes = {}
        for equip_name in config.equipment_list:
            info = get_equipment_info(equip_name.split("_")[0] if "_" in equip_name and equip_name.split("_")[-1].isdigit() else equip_name)
            if info:
                self.equipment_sizes[equip_name] = info["approximate_size"]
            else:
                # Fallback for equipment with instance numbers
                base_name = "_".join(equip_name.split("_")[:-1]) if equip_name.split("_")[-1].isdigit() else equip_name
                info = get_equipment_info(base_name)
                if info:
                    self.equipment_sizes[equip_name] = info["approximate_size"]
                else:
                    self.equipment_sizes[equip_name] = {"length": 1.0, "width": 1.0, "height": 1.0}

    def generate(self) -> List[PlacedEquipment]:
        """Generate layout - to be implemented by subclasses"""
        raise NotImplementedError

    def to_scene_graph(self, placements: List[PlacedEquipment]) -> List[Dict]:
        """Convert placements to scene graph format with connections"""
        room_dims = self.config.room_dimensions
        scene_graph = get_room_priors(list(room_dims))

        # Build placement objects first
        placement_objects = []
        for p in placements:
            base_name = "_".join(p.equipment_type.split("_")[:-1]) if p.equipment_type.split("_")[-1].isdigit() else p.equipment_type
            info = get_equipment_info(base_name) or get_equipment_info(p.equipment_type) or {}

            obj = {
                "object_id": p.object_id,
                "style": info.get("style", "Industrial"),
                "material": info.get("material", "Metal"),
                "size_in_meters": p.size,
                "is_on_the_floor": True,
                "facing": self._rotation_to_facing(p.rotation),
                "placement": {
                    "room_layout_elements": [],
                    "objects_in_room": []
                },
                "rotation": {"z_angle": p.rotation},
                "cluster": {
                    "constraint_area": {
                        "x_neg": 0.0, "x_pos": 0.0,
                        "y_neg": 0.0, "y_pos": 0.0
                    }
                },
                "position": {"x": p.x, "y": p.y, "z": p.z},
                "connections": [],
                "itemType": None,
                "size": None,
                "_placement": p,  # Temporary reference for connection generation
            }
            placement_objects.append(obj)

        # Generate connections based on proximity and equipment type
        self._generate_connections(placement_objects)

        # Remove temporary placement reference and add to scene graph
        for obj in placement_objects:
            del obj["_placement"]
            scene_graph.append(obj)

        return scene_graph

    def _generate_connections(self, objects: List[Dict]):
        """Generate material flow connections based on proximity and equipment types"""
        # Define which equipment types can connect
        connectable_types = {
            "Line_01", "Line_02", "Line_03", "Line_04", "Line_05",
            "Line_06", "Line_07", "Line_08"
        }

        # Sort objects by position (create a flow path)
        # Use a simple heuristic: sort by x then y to create a flow
        sorted_objects = sorted(
            [o for o in objects if any(o["object_id"].startswith(t) for t in connectable_types)],
            key=lambda o: (o["position"]["x"], o["position"]["y"])
        )

        # Create chain connections
        for i in range(len(sorted_objects) - 1):
            current = sorted_objects[i]
            next_obj = sorted_objects[i + 1]

            # Calculate distance
            dist = math.sqrt(
                (current["position"]["x"] - next_obj["position"]["x"]) ** 2 +
                (current["position"]["y"] - next_obj["position"]["y"]) ** 2
            )

            # Only connect if reasonably close (within 8 meters)
            if dist < 8.0:
                current["connections"].append({
                    "object_id": next_obj["object_id"],
                    "connection_type": "connected",
                    "source_endpoint": "primary_forward_positive",
                    "target_endpoint": "primary_forward_negative"
                })

    def _rotation_to_facing(self, rotation: float) -> str:
        """Convert rotation angle to facing direction"""
        rotation = rotation % 360
        if rotation < 45 or rotation >= 315:
            return "north_wall"
        elif rotation < 135:
            return "east_wall"
        elif rotation < 225:
            return "south_wall"
        else:
            return "west_wall"

    def check_collision(self, p1: PlacedEquipment, p2: PlacedEquipment) -> bool:
        """Check if two placements collide (with clearance)"""
        b1 = p1.get_bounds()
        b2 = p2.get_bounds()
        clearance = self.config.min_clearance

        # Check overlap with clearance
        return not (b1[1] + clearance < b2[0] or
                   b2[1] + clearance < b1[0] or
                   b1[3] + clearance < b2[2] or
                   b2[3] + clearance < b1[2])

    def check_in_bounds(self, p: PlacedEquipment) -> bool:
        """Check if placement is within room bounds"""
        bounds = p.get_bounds()
        margin = self.config.wall_margin
        room = self.config.room_dimensions

        return (bounds[0] >= margin and
                bounds[1] <= room[0] - margin and
                bounds[2] >= margin and
                bounds[3] <= room[1] - margin)

    def evaluate_layout(self, placements: List[PlacedEquipment]) -> float:
        """
        Evaluate layout quality (higher is better)

        Metrics:
        - Collision penalty
        - Out of bounds penalty
        - Flow distance (simplified)
        - Space utilization
        """
        score = 1000.0  # Base score

        # Collision penalty
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                if self.check_collision(p1, p2):
                    score -= 100.0

        # Out of bounds penalty
        for p in placements:
            if not self.check_in_bounds(p):
                score -= 50.0

        # Flow distance penalty (sum of distances between consecutive equipment)
        if len(placements) > 1:
            total_dist = 0.0
            for i in range(len(placements) - 1):
                p1, p2 = placements[i], placements[i+1]
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                total_dist += dist
            # Normalize and penalize long distances
            avg_dist = total_dist / (len(placements) - 1)
            score -= avg_dist * 2.0

        # Space utilization bonus
        room_area = self.config.room_dimensions[0] * self.config.room_dimensions[1]
        used_area = sum(p.size.get("length", 1.0) * p.size.get("width", 1.0) for p in placements)
        utilization = used_area / room_area
        score += utilization * 50.0

        return score


class RandomLayoutGenerator(BaselineLayoutGenerator):
    """Random placement within room constraints"""

    def generate(self) -> List[PlacedEquipment]:
        placements = []
        room = self.config.room_dimensions
        margin = self.config.wall_margin

        instance_counts = {}

        for equip_name in self.config.equipment_list:
            # Generate instance ID
            base_name = equip_name.split("_")[0] if "_" in equip_name else equip_name
            # Check if already has instance number
            if equip_name.split("_")[-1].isdigit():
                object_id = equip_name
            else:
                instance_counts[base_name] = instance_counts.get(base_name, 0) + 1
                object_id = f"{equip_name}_{instance_counts[base_name]}"

            size = self.equipment_sizes.get(equip_name, {"length": 1.0, "width": 1.0, "height": 1.0})

            # Try random placements until valid one found
            max_attempts = 100
            for attempt in range(max_attempts):
                x = random.uniform(margin + size["length"]/2, room[0] - margin - size["length"]/2)
                y = random.uniform(margin + size["width"]/2, room[1] - margin - size["width"]/2)
                rotation = random.choice([0.0, 90.0, 180.0, 270.0])

                new_placement = PlacedEquipment(
                    object_id=object_id,
                    equipment_type=equip_name,
                    x=x, y=y, z=0.0,
                    rotation=rotation,
                    size=size
                )

                # Check collisions with existing placements
                valid = True
                for existing in placements:
                    if self.check_collision(new_placement, existing):
                        valid = False
                        break

                if valid and self.check_in_bounds(new_placement):
                    placements.append(new_placement)
                    break
            else:
                # If no valid position found, place anyway (will be penalized in evaluation)
                placements.append(PlacedEquipment(
                    object_id=object_id,
                    equipment_type=equip_name,
                    x=room[0]/2, y=room[1]/2, z=0.0,
                    rotation=0.0,
                    size=size
                ))

        return placements


class GridLayoutGenerator(BaselineLayoutGenerator):
    """Grid-based regular placement"""

    def generate(self) -> List[PlacedEquipment]:
        placements = []
        room = self.config.room_dimensions
        margin = self.config.wall_margin
        clearance = self.config.min_clearance

        # Calculate grid dimensions
        n_items = len(self.config.equipment_list)
        if n_items == 0:
            return placements

        # Estimate cell size based on largest equipment
        max_length = max(s.get("length", 1.0) for s in self.equipment_sizes.values())
        max_width = max(s.get("width", 1.0) for s in self.equipment_sizes.values())
        cell_size = max(max_length, max_width) + clearance

        # Calculate grid dimensions
        available_x = room[0] - 2 * margin
        available_y = room[1] - 2 * margin

        cols = max(1, int(available_x / cell_size))
        rows = max(1, int(math.ceil(n_items / cols)))

        # Adjust cell size to fit
        cell_x = available_x / cols
        cell_y = available_y / max(rows, 1)

        instance_counts = {}

        for idx, equip_name in enumerate(self.config.equipment_list):
            # Generate instance ID
            base_name = equip_name.split("_")[0] if "_" in equip_name else equip_name
            if equip_name.split("_")[-1].isdigit():
                object_id = equip_name
            else:
                instance_counts[base_name] = instance_counts.get(base_name, 0) + 1
                object_id = f"{equip_name}_{instance_counts[base_name]}"

            size = self.equipment_sizes.get(equip_name, {"length": 1.0, "width": 1.0, "height": 1.0})

            # Calculate grid position
            col = idx % cols
            row = idx // cols

            x = margin + cell_x * (col + 0.5)
            y = margin + cell_y * (row + 0.5)

            # Alternate rotation for variety
            rotation = [0.0, 90.0, 180.0, 270.0][idx % 4]

            placements.append(PlacedEquipment(
                object_id=object_id,
                equipment_type=equip_name,
                x=x, y=y, z=0.0,
                rotation=rotation,
                size=size
            ))

        return placements


class GeneticAlgorithmLayoutGenerator(BaselineLayoutGenerator):
    """Genetic Algorithm for layout optimization"""

    def generate(self) -> List[PlacedEquipment]:
        # Initialize population with random layouts
        population = []
        for _ in range(self.config.ga_population_size):
            random_gen = RandomLayoutGenerator(self.config)
            layout = random_gen.generate()
            population.append(layout)

        # Evolution loop
        for gen in range(self.config.ga_generations):
            # Evaluate fitness
            fitness = [(self.evaluate_layout(layout), layout) for layout in population]
            fitness.sort(key=lambda x: x[0], reverse=True)

            # Select top performers
            elite_count = max(2, self.config.ga_population_size // 5)
            elites = [f[1] for f in fitness[:elite_count]]

            # Generate new population
            new_population = elites.copy()

            while len(new_population) < self.config.ga_population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_select(fitness)
                parent2 = self._tournament_select(fitness)

                # Crossover
                if random.random() < self.config.ga_crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))

                # Mutation
                if random.random() < self.config.ga_mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        # Return best layout
        best_layout = max(population, key=lambda x: self.evaluate_layout(x))
        return best_layout

    def _tournament_select(self, fitness: List[Tuple[float, List]], tournament_size: int = 3) -> List[PlacedEquipment]:
        """Tournament selection"""
        tournament = random.sample(fitness, min(tournament_size, len(fitness)))
        winner = max(tournament, key=lambda x: x[0])
        return winner[1]

    def _crossover(self, parent1: List[PlacedEquipment], parent2: List[PlacedEquipment]) -> List[PlacedEquipment]:
        """Single-point crossover"""
        if len(parent1) <= 1:
            return copy.deepcopy(parent1)

        crossover_point = random.randint(1, len(parent1) - 1)
        child = []

        for i in range(len(parent1)):
            if i < crossover_point:
                child.append(copy.deepcopy(parent1[i]))
            else:
                child.append(copy.deepcopy(parent2[i]))

        return child

    def _mutate(self, layout: List[PlacedEquipment]) -> List[PlacedEquipment]:
        """Mutate a layout by randomly moving equipment"""
        if not layout:
            return layout

        mutated = copy.deepcopy(layout)
        idx = random.randint(0, len(mutated) - 1)
        room = self.config.room_dimensions
        margin = self.config.wall_margin

        # Random perturbation
        mutated[idx].x += random.uniform(-2.0, 2.0)
        mutated[idx].y += random.uniform(-2.0, 2.0)
        mutated[idx].rotation = random.choice([0.0, 90.0, 180.0, 270.0])

        # Clamp to room bounds
        size = mutated[idx].size
        mutated[idx].x = max(margin + size["length"]/2,
                            min(room[0] - margin - size["length"]/2, mutated[idx].x))
        mutated[idx].y = max(margin + size["width"]/2,
                            min(room[1] - margin - size["width"]/2, mutated[idx].y))

        return mutated


class SimulatedAnnealingLayoutGenerator(BaselineLayoutGenerator):
    """Simulated Annealing for layout optimization"""

    def generate(self) -> List[PlacedEquipment]:
        # Initialize with random layout
        random_gen = RandomLayoutGenerator(self.config)
        current = random_gen.generate()
        current_energy = -self.evaluate_layout(current)  # Minimize energy

        best = copy.deepcopy(current)
        best_energy = current_energy

        temperature = self.config.sa_initial_temp

        while temperature > self.config.sa_min_temp:
            for _ in range(self.config.sa_iterations_per_temp):
                # Generate neighbor
                neighbor = self._generate_neighbor(current)
                neighbor_energy = -self.evaluate_layout(neighbor)

                # Acceptance probability
                delta_energy = neighbor_energy - current_energy

                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                    current = neighbor
                    current_energy = neighbor_energy

                    if current_energy < best_energy:
                        best = copy.deepcopy(current)
                        best_energy = current_energy

            temperature *= self.config.sa_cooling_rate

        return best

    def _generate_neighbor(self, layout: List[PlacedEquipment]) -> List[PlacedEquipment]:
        """Generate a neighboring solution"""
        if not layout:
            return layout

        neighbor = copy.deepcopy(layout)

        # Choose move type
        move_type = random.choice(["shift", "rotate", "swap"])

        if move_type == "shift":
            # Shift random equipment
            idx = random.randint(0, len(neighbor) - 1)
            room = self.config.room_dimensions
            margin = self.config.wall_margin
            size = neighbor[idx].size

            neighbor[idx].x += random.uniform(-1.5, 1.5)
            neighbor[idx].y += random.uniform(-1.5, 1.5)

            # Clamp to bounds
            neighbor[idx].x = max(margin + size["length"]/2,
                                min(room[0] - margin - size["length"]/2, neighbor[idx].x))
            neighbor[idx].y = max(margin + size["width"]/2,
                                min(room[1] - margin - size["width"]/2, neighbor[idx].y))

        elif move_type == "rotate":
            # Rotate random equipment
            idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx].rotation = random.choice([0.0, 90.0, 180.0, 270.0])

        elif move_type == "swap" and len(neighbor) >= 2:
            # Swap positions of two equipment
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1].x, neighbor[idx2].x = neighbor[idx2].x, neighbor[idx1].x
            neighbor[idx1].y, neighbor[idx2].y = neighbor[idx2].y, neighbor[idx1].y

        return neighbor


class SLPLayoutGenerator(BaselineLayoutGenerator):
    """
    Systematic Layout Planning (SLP) inspired heuristic

    Based on Muther's SLP methodology:
    1. Analyze material flow relationships
    2. Create relationship diagram
    3. Place equipment based on relationship closeness
    """

    def __init__(self, config: LayoutConfig, relationship_matrix: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__(config)
        # Relationship matrix: equipment pairs -> closeness rating (higher = should be closer)
        self.relationship_matrix = relationship_matrix or self._generate_default_relationships()

    def _generate_default_relationships(self) -> Dict[str, Dict[str, float]]:
        """Generate default relationship matrix based on equipment categories"""
        relationships = {}

        # Define category-based closeness
        category_closeness = {
            ("material_flow", "material_flow"): 0.8,
            ("material_flow", "assembly_processing"): 0.9,
            ("assembly_processing", "assembly_processing"): 0.7,
            ("assembly_processing", "support_equipment"): 0.6,
            ("material_flow", "support_equipment"): 0.4,
            ("support_equipment", "support_equipment"): 0.3,
            ("infrastructure", "assembly_processing"): 0.5,
            ("infrastructure", "material_flow"): 0.4,
        }

        for equip1 in self.config.equipment_list:
            relationships[equip1] = {}
            cat1 = self._get_category(equip1)

            for equip2 in self.config.equipment_list:
                if equip1 == equip2:
                    relationships[equip1][equip2] = 0.0
                else:
                    cat2 = self._get_category(equip2)
                    key = (cat1, cat2) if (cat1, cat2) in category_closeness else (cat2, cat1)
                    relationships[equip1][equip2] = category_closeness.get(key, 0.5)

        return relationships

    def _get_category(self, equip_name: str) -> str:
        """Get equipment category"""
        base_name = equip_name.split("_")[0] if "_" in equip_name else equip_name
        # Handle instance numbers
        if base_name[-1].isdigit():
            base_name = "_".join(equip_name.split("_")[:-1])

        info = get_equipment_info(base_name)
        if info:
            return info.get("category", "support_equipment")
        return "support_equipment"

    def generate(self) -> List[PlacedEquipment]:
        placements = []
        room = self.config.room_dimensions
        margin = self.config.wall_margin

        # Sort equipment by total relationship strength (most connected first)
        equipment_priority = []
        for equip in self.config.equipment_list:
            total_rel = sum(self.relationship_matrix.get(equip, {}).values())
            equipment_priority.append((total_rel, equip))
        equipment_priority.sort(reverse=True)

        instance_counts = {}

        for _, equip_name in equipment_priority:
            # Generate instance ID
            base_name = equip_name.split("_")[0] if "_" in equip_name else equip_name
            if equip_name.split("_")[-1].isdigit():
                object_id = equip_name
            else:
                instance_counts[base_name] = instance_counts.get(base_name, 0) + 1
                object_id = f"{equip_name}_{instance_counts[base_name]}"

            size = self.equipment_sizes.get(equip_name, {"length": 1.0, "width": 1.0, "height": 1.0})

            if not placements:
                # Place first equipment in center
                x = room[0] / 2
                y = room[1] / 2
            else:
                # Find best position based on relationships with placed equipment
                best_pos = self._find_best_position(equip_name, placements, size)
                x, y = best_pos

            rotation = self._choose_rotation(equip_name, placements)

            placements.append(PlacedEquipment(
                object_id=object_id,
                equipment_type=equip_name,
                x=x, y=y, z=0.0,
                rotation=rotation,
                size=size
            ))

        return placements

    def _find_best_position(self, equip_name: str, placements: List[PlacedEquipment],
                           size: Dict[str, float]) -> Tuple[float, float]:
        """Find optimal position based on relationships"""
        room = self.config.room_dimensions
        margin = self.config.wall_margin

        # Calculate weighted center based on relationships
        wx, wy = 0.0, 0.0
        total_weight = 0.0

        for p in placements:
            rel = self.relationship_matrix.get(equip_name, {}).get(p.equipment_type, 0.5)
            wx += p.x * rel
            wy += p.y * rel
            total_weight += rel

        if total_weight > 0:
            target_x = wx / total_weight
            target_y = wy / total_weight
        else:
            target_x = room[0] / 2
            target_y = room[1] / 2

        # Try positions around the target
        best_pos = (target_x, target_y)
        best_score = float('-inf')

        for angle in range(0, 360, 30):
            for radius in [0, 2.0, 4.0, 6.0]:
                test_x = target_x + radius * math.cos(math.radians(angle))
                test_y = target_y + radius * math.sin(math.radians(angle))

                # Clamp to bounds
                test_x = max(margin + size["length"]/2, min(room[0] - margin - size["length"]/2, test_x))
                test_y = max(margin + size["width"]/2, min(room[1] - margin - size["width"]/2, test_y))

                # Check validity
                test_placement = PlacedEquipment(
                    object_id="test", equipment_type=equip_name,
                    x=test_x, y=test_y, rotation=0.0, size=size
                )

                valid = True
                for p in placements:
                    if self.check_collision(test_placement, p):
                        valid = False
                        break

                if valid:
                    # Score based on relationship distances
                    score = 0.0
                    for p in placements:
                        rel = self.relationship_matrix.get(equip_name, {}).get(p.equipment_type, 0.5)
                        dist = math.sqrt((test_x - p.x)**2 + (test_y - p.y)**2)
                        # Higher relationship = prefer closer
                        score -= rel * dist

                    if score > best_score:
                        best_score = score
                        best_pos = (test_x, test_y)

        return best_pos

    def _choose_rotation(self, equip_name: str, placements: List[PlacedEquipment]) -> float:
        """Choose optimal rotation based on nearby equipment"""
        if not placements:
            return 0.0

        # Find closest placed equipment
        closest = min(placements, key=lambda p:
                     self.relationship_matrix.get(equip_name, {}).get(p.equipment_type, 0.0),
                     default=None)

        if closest:
            # Face towards closest related equipment
            return closest.rotation

        return 0.0


class LLM4RMSGAOptimizer(GeneticAlgorithmLayoutGenerator):
    """
    Hybrid method: Use LLM4RMS layout as initial solution, then optimize with GA.

    This tests whether GA can further improve LLM4RMS's layout.
    Key: Preserves original LLM4RMS connections (material flow topology).
    """

    def __init__(self, config: LayoutConfig, initial_scene_graph: List[Dict]):
        super().__init__(config)
        self.initial_scene_graph = initial_scene_graph
        self.initial_placements = self._extract_placements(initial_scene_graph)
        # Store original connections to preserve material flow topology
        self.original_connections = self._extract_connections(initial_scene_graph)

    def _extract_placements(self, scene_graph: List[Dict]) -> List[PlacedEquipment]:
        """Extract placements from an existing scene graph"""
        placements = []
        for obj in scene_graph:
            obj_id = obj.get("object_id", "")
            item_type = obj.get("itemType")

            # Skip room layout elements
            if item_type in ["wall", "floor", "ceiling"]:
                continue
            if obj_id in ["south_wall", "north_wall", "east_wall", "west_wall",
                          "ceiling", "middle of the room"]:
                continue

            pos = obj.get("position", {})
            rot = obj.get("rotation", {})
            size = obj.get("size_in_meters", {"length": 1.0, "width": 1.0, "height": 1.0})

            placements.append(PlacedEquipment(
                object_id=obj_id,
                equipment_type=obj_id,
                x=pos.get("x", 0),
                y=pos.get("y", 0),
                z=pos.get("z", 0),
                rotation=rot.get("z_angle", 0),
                size=size,
            ))

        return placements

    def _extract_connections(self, scene_graph: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract original connections from scene graph"""
        connections = {}
        for obj in scene_graph:
            obj_id = obj.get("object_id", "")
            obj_connections = obj.get("connections", [])
            if obj_connections:
                connections[obj_id] = obj_connections
        return connections

    def to_scene_graph(self, placements: List[PlacedEquipment]) -> List[Dict]:
        """Override to preserve original LLM4RMS connections instead of regenerating"""
        room_dims = self.config.room_dimensions
        scene_graph = get_room_priors(list(room_dims))

        for p in placements:
            # Get original connections for this object
            original_conn = self.original_connections.get(p.object_id, [])

            base_name = "_".join(p.equipment_type.split("_")[:-1]) if p.equipment_type.split("_")[-1].isdigit() else p.equipment_type
            info = get_equipment_info(base_name) or get_equipment_info(p.equipment_type) or {}

            obj = {
                "object_id": p.object_id,
                "style": info.get("style", "Industrial"),
                "material": info.get("material", "Metal"),
                "size_in_meters": p.size,
                "is_on_the_floor": True,
                "facing": self._rotation_to_facing(p.rotation),
                "placement": {
                    "room_layout_elements": [],
                    "objects_in_room": []
                },
                "rotation": {"z_angle": p.rotation},
                "cluster": {
                    "constraint_area": {
                        "x_neg": 0.0, "x_pos": 0.0,
                        "y_neg": 0.0, "y_pos": 0.0
                    }
                },
                "position": {"x": p.x, "y": p.y, "z": p.z},
                "connections": original_conn,  # Preserve original connections!
                "itemType": None,
                "size": None,
            }
            scene_graph.append(obj)

        return scene_graph

    def evaluate_layout(self, placements: List[PlacedEquipment]) -> float:
        """
        Enhanced evaluation that considers material flow connections.
        Optimizes positions to minimize connected equipment distances.
        """
        score = 1000.0

        # Build position lookup
        pos_lookup = {p.object_id: (p.x, p.y) for p in placements}

        # Collision penalty
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                if self.check_collision(p1, p2):
                    score -= 100.0

        # Out of bounds penalty
        for p in placements:
            if not self.check_in_bounds(p):
                score -= 50.0

        # **Key: Material flow distance optimization**
        # Minimize distance between connected equipment
        total_connection_dist = 0.0
        connection_count = 0
        for obj_id, connections in self.original_connections.items():
            if obj_id not in pos_lookup:
                continue
            src_pos = pos_lookup[obj_id]
            for conn in connections:
                target_id = conn.get("object_id", "")
                if target_id in pos_lookup:
                    tgt_pos = pos_lookup[target_id]
                    dist = math.sqrt((src_pos[0] - tgt_pos[0])**2 + (src_pos[1] - tgt_pos[1])**2)
                    total_connection_dist += dist
                    connection_count += 1

        # Reward shorter connection distances (higher weight for material flow)
        if connection_count > 0:
            avg_connection_dist = total_connection_dist / connection_count
            # Optimal distance is around 2-4 meters for material flow
            optimal_dist = 3.0
            dist_penalty = abs(avg_connection_dist - optimal_dist) * 10.0
            score -= dist_penalty
            # Bonus for very short connections
            if avg_connection_dist < 5.0:
                score += (5.0 - avg_connection_dist) * 20.0

        # Space utilization bonus
        room_area = self.config.room_dimensions[0] * self.config.room_dimensions[1]
        used_area = sum(p.size.get("length", 1.0) * p.size.get("width", 1.0) for p in placements)
        utilization = used_area / room_area
        score += utilization * 30.0

        return score

    def _focused_mutate(self, placements: List[PlacedEquipment]) -> List[PlacedEquipment]:
        """
        Focused mutation that moves connected equipment closer together.
        More aggressive than standard mutation for material flow optimization.
        """
        result = copy.deepcopy(placements)
        if not result:
            return result

        pos_lookup = {p.object_id: p for p in result}

        # Find equipment with connections and move them closer
        for obj_id, connections in self.original_connections.items():
            if obj_id not in pos_lookup or not connections:
                continue

            src = pos_lookup[obj_id]
            for conn in connections:
                target_id = conn.get("object_id", "")
                if target_id not in pos_lookup:
                    continue

                tgt = pos_lookup[target_id]

                # Move target closer to source (or vice versa)
                dx = tgt.x - src.x
                dy = tgt.y - src.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > 4.0:  # If too far, move closer
                    # Move target 10-30% closer to source
                    move_ratio = random.uniform(0.1, 0.3)
                    tgt.x -= dx * move_ratio
                    tgt.y -= dy * move_ratio

                    # Ensure in bounds
                    margin = self.config.wall_margin
                    room = self.config.room_dimensions
                    tgt.x = max(margin + 1, min(room[0] - margin - 1, tgt.x))
                    tgt.y = max(margin + 1, min(room[1] - margin - 1, tgt.y))

        return result

    def generate(self) -> List[PlacedEquipment]:
        """Generate layout starting from LLM4RMS solution with enhanced optimization"""
        # Use larger population and more generations for thorough optimization
        pop_size = max(self.config.ga_population_size, 80)
        generations = max(self.config.ga_generations, 150)

        # Initialize population with LLM4RMS layout + focused mutations
        population = []

        # Add original LLM4RMS layout (preserve as elite)
        population.append(copy.deepcopy(self.initial_placements))

        # Add focused mutations that move connected equipment closer
        for _ in range((pop_size - 1) // 2):
            mutated = self._focused_mutate(self.initial_placements)
            population.append(mutated)

        # Add random mutations for diversity
        for _ in range(pop_size - len(population)):
            mutated = copy.deepcopy(self.initial_placements)
            for _ in range(random.randint(2, 5)):
                mutated = self._mutate(mutated)
            population.append(mutated)

        # Evolution loop with enhanced parameters
        best_ever = copy.deepcopy(self.initial_placements)
        best_ever_fitness = self.evaluate_layout(best_ever)

        for gen in range(generations):
            fitness = [(self.evaluate_layout(layout), layout) for layout in population]
            fitness.sort(key=lambda x: x[0], reverse=True)

            # Track best ever
            if fitness[0][0] > best_ever_fitness:
                best_ever = copy.deepcopy(fitness[0][1])
                best_ever_fitness = fitness[0][0]

            elite_count = max(4, pop_size // 4)  # Keep more elites
            elites = [f[1] for f in fitness[:elite_count]]

            new_population = [copy.deepcopy(e) for e in elites]

            while len(new_population) < pop_size:
                parent1 = self._tournament_select(fitness)
                parent2 = self._tournament_select(fitness)

                if random.random() < 0.85:  # Higher crossover rate
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))

                # Use focused mutation more often
                if random.random() < 0.3:
                    child = self._focused_mutate(child)
                elif random.random() < 0.15:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        return best_ever


class LLM4RMSSAOptimizer(SimulatedAnnealingLayoutGenerator):
    """
    Hybrid method: Use LLM4RMS layout as initial solution, then optimize with SA.

    This tests whether SA can further improve LLM4RMS's layout.
    Key: Preserves original LLM4RMS connections (material flow topology).
    """

    def __init__(self, config: LayoutConfig, initial_scene_graph: List[Dict]):
        super().__init__(config)
        self.initial_scene_graph = initial_scene_graph
        self.initial_placements = self._extract_placements(initial_scene_graph)
        # Store original connections to preserve material flow topology
        self.original_connections = self._extract_connections(initial_scene_graph)

    def _extract_placements(self, scene_graph: List[Dict]) -> List[PlacedEquipment]:
        """Extract placements from an existing scene graph"""
        placements = []
        for obj in scene_graph:
            obj_id = obj.get("object_id", "")
            item_type = obj.get("itemType")

            if item_type in ["wall", "floor", "ceiling"]:
                continue
            if obj_id in ["south_wall", "north_wall", "east_wall", "west_wall",
                          "ceiling", "middle of the room"]:
                continue

            pos = obj.get("position", {})
            rot = obj.get("rotation", {})
            size = obj.get("size_in_meters", {"length": 1.0, "width": 1.0, "height": 1.0})

            placements.append(PlacedEquipment(
                object_id=obj_id,
                equipment_type=obj_id,
                x=pos.get("x", 0),
                y=pos.get("y", 0),
                z=pos.get("z", 0),
                rotation=rot.get("z_angle", 0),
                size=size,
            ))

        return placements

    def _extract_connections(self, scene_graph: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract original connections from scene graph"""
        connections = {}
        for obj in scene_graph:
            obj_id = obj.get("object_id", "")
            obj_connections = obj.get("connections", [])
            if obj_connections:
                connections[obj_id] = obj_connections
        return connections

    def to_scene_graph(self, placements: List[PlacedEquipment]) -> List[Dict]:
        """Override to preserve original LLM4RMS connections instead of regenerating"""
        room_dims = self.config.room_dimensions
        scene_graph = get_room_priors(list(room_dims))

        for p in placements:
            # Get original connections for this object
            original_conn = self.original_connections.get(p.object_id, [])

            base_name = "_".join(p.equipment_type.split("_")[:-1]) if p.equipment_type.split("_")[-1].isdigit() else p.equipment_type
            info = get_equipment_info(base_name) or get_equipment_info(p.equipment_type) or {}

            obj = {
                "object_id": p.object_id,
                "style": info.get("style", "Industrial"),
                "material": info.get("material", "Metal"),
                "size_in_meters": p.size,
                "is_on_the_floor": True,
                "facing": self._rotation_to_facing(p.rotation),
                "placement": {
                    "room_layout_elements": [],
                    "objects_in_room": []
                },
                "rotation": {"z_angle": p.rotation},
                "cluster": {
                    "constraint_area": {
                        "x_neg": 0.0, "x_pos": 0.0,
                        "y_neg": 0.0, "y_pos": 0.0
                    }
                },
                "position": {"x": p.x, "y": p.y, "z": p.z},
                "connections": original_conn,  # Preserve original connections!
                "itemType": None,
                "size": None,
            }
            scene_graph.append(obj)

        return scene_graph

    def evaluate_layout(self, placements: List[PlacedEquipment]) -> float:
        """
        Enhanced evaluation that considers material flow connections.
        Optimizes positions to minimize connected equipment distances.
        """
        score = 1000.0

        # Build position lookup
        pos_lookup = {p.object_id: (p.x, p.y) for p in placements}

        # Collision penalty
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                if self.check_collision(p1, p2):
                    score -= 100.0

        # Out of bounds penalty
        for p in placements:
            if not self.check_in_bounds(p):
                score -= 50.0

        # **Key: Material flow distance optimization**
        # Minimize distance between connected equipment
        total_connection_dist = 0.0
        connection_count = 0
        for obj_id, connections in self.original_connections.items():
            if obj_id not in pos_lookup:
                continue
            src_pos = pos_lookup[obj_id]
            for conn in connections:
                target_id = conn.get("object_id", "")
                if target_id in pos_lookup:
                    tgt_pos = pos_lookup[target_id]
                    dist = math.sqrt((src_pos[0] - tgt_pos[0])**2 + (src_pos[1] - tgt_pos[1])**2)
                    total_connection_dist += dist
                    connection_count += 1

        # Reward shorter connection distances (higher weight for material flow)
        if connection_count > 0:
            avg_connection_dist = total_connection_dist / connection_count
            # Optimal distance is around 2-4 meters for material flow
            optimal_dist = 3.0
            dist_penalty = abs(avg_connection_dist - optimal_dist) * 10.0
            score -= dist_penalty
            # Bonus for very short connections
            if avg_connection_dist < 5.0:
                score += (5.0 - avg_connection_dist) * 20.0

        # Space utilization bonus
        room_area = self.config.room_dimensions[0] * self.config.room_dimensions[1]
        used_area = sum(p.size.get("length", 1.0) * p.size.get("width", 1.0) for p in placements)
        utilization = used_area / room_area
        score += utilization * 30.0

        return score

    def _focused_neighbor(self, placements: List[PlacedEquipment]) -> List[PlacedEquipment]:
        """
        Generate neighbor that moves connected equipment closer.
        More targeted than random perturbation.
        """
        result = copy.deepcopy(placements)
        if not result:
            return result

        pos_lookup = {p.object_id: p for p in result}

        # Find a random connected pair and move them closer
        connected_pairs = []
        for obj_id, connections in self.original_connections.items():
            if obj_id not in pos_lookup:
                continue
            for conn in connections:
                target_id = conn.get("object_id", "")
                if target_id in pos_lookup:
                    connected_pairs.append((obj_id, target_id))

        if connected_pairs:
            src_id, tgt_id = random.choice(connected_pairs)
            src = pos_lookup[src_id]
            tgt = pos_lookup[tgt_id]

            dx = tgt.x - src.x
            dy = tgt.y - src.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 3.0:
                # Move target closer
                move_ratio = random.uniform(0.05, 0.2)
                tgt.x -= dx * move_ratio
                tgt.y -= dy * move_ratio

                # Ensure in bounds
                margin = self.config.wall_margin
                room = self.config.room_dimensions
                tgt.x = max(margin + 1, min(room[0] - margin - 1, tgt.x))
                tgt.y = max(margin + 1, min(room[1] - margin - 1, tgt.y))

        return result

    def generate(self) -> List[PlacedEquipment]:
        """Generate layout starting from LLM4RMS solution with enhanced SA"""
        # Start from LLM4RMS layout
        current = copy.deepcopy(self.initial_placements)
        current_energy = -self.evaluate_layout(current)

        best = copy.deepcopy(current)
        best_energy = current_energy

        # Use higher temperature and more iterations for thorough search
        temperature = max(self.config.sa_initial_temp, 2000.0)
        min_temp = 0.5
        cooling_rate = 0.992
        iterations_per_temp = 80

        while temperature > min_temp:
            for _ in range(iterations_per_temp):
                # Use focused neighbor more often
                if random.random() < 0.4:
                    neighbor = self._focused_neighbor(current)
                else:
                    neighbor = self._generate_neighbor(current)

                neighbor_energy = -self.evaluate_layout(neighbor)
                delta_energy = neighbor_energy - current_energy

                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                    current = neighbor
                    current_energy = neighbor_energy

                    if current_energy < best_energy:
                        best = copy.deepcopy(current)
                        best_energy = current_energy

            temperature *= cooling_rate

        return best


def create_baseline_generator(method: BaselineMethod, config: LayoutConfig,
                             **kwargs) -> BaselineLayoutGenerator:
    """Factory function to create baseline generators"""
    generators = {
        BaselineMethod.RANDOM: RandomLayoutGenerator,
        BaselineMethod.GRID: GridLayoutGenerator,
        BaselineMethod.GENETIC_ALGORITHM: GeneticAlgorithmLayoutGenerator,
        BaselineMethod.SIMULATED_ANNEALING: SimulatedAnnealingLayoutGenerator,
        BaselineMethod.SLP: SLPLayoutGenerator,
    }

    generator_class = generators.get(method)

    # Handle hybrid methods (LLM4RMS + optimization)
    if method == BaselineMethod.LLM4RMS_GA:
        initial_scene_graph = kwargs.get("initial_scene_graph")
        if initial_scene_graph is None:
            raise ValueError("LLM4RMS_GA requires 'initial_scene_graph' parameter")
        return LLM4RMSGAOptimizer(config, initial_scene_graph)

    if method == BaselineMethod.LLM4RMS_SA:
        initial_scene_graph = kwargs.get("initial_scene_graph")
        if initial_scene_graph is None:
            raise ValueError("LLM4RMS_SA requires 'initial_scene_graph' parameter")
        return LLM4RMSSAOptimizer(config, initial_scene_graph)

    if generator_class is None:
        raise ValueError(f"Unknown baseline method: {method}")

    # Only pass kwargs that the generator accepts (SLP takes relationship_matrix)
    if method == BaselineMethod.SLP and "relationship_matrix" in kwargs:
        return generator_class(config, relationship_matrix=kwargs["relationship_matrix"])
    else:
        return generator_class(config)


def generate_baseline_scene_graph(
    method: BaselineMethod,
    equipment_list: List[str],
    room_dimensions: Tuple[float, float, float] = (14.0, 14.0, 4.5),
    seed: Optional[int] = None,
    **kwargs
) -> List[Dict]:
    """
    Convenience function to generate a scene graph using a baseline method

    Args:
        method: Baseline method to use
        equipment_list: List of equipment names
        room_dimensions: Room dimensions (length, width, height)
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for the generator

    Returns:
        Scene graph as list of dictionaries
    """
    config = LayoutConfig(
        room_dimensions=room_dimensions,
        equipment_list=equipment_list,
        seed=seed,
        **{k: v for k, v in kwargs.items() if hasattr(LayoutConfig, k)}
    )

    generator = create_baseline_generator(method, config, **kwargs)
    placements = generator.generate()
    return generator.to_scene_graph(placements)


def extract_equipment_from_scene_graph(scene_graph: List[Dict]) -> List[str]:
    """Extract equipment list from an existing scene graph"""
    equipment = []
    for obj in scene_graph:
        obj_id = obj.get("object_id", "")
        item_type = obj.get("itemType")

        # Skip room layout elements
        if item_type in ["wall", "floor", "ceiling"]:
            continue
        if obj_id in ["south_wall", "north_wall", "east_wall", "west_wall",
                      "ceiling", "middle of the room"]:
            continue

        equipment.append(obj_id)

    return equipment


if __name__ == "__main__":
    # Example usage
    equipment = [
        "Line_04", "Line_04", "Line_04", "Line_04",
        "Line_05", "Line_05",
        "Line_03", "Line_03", "Line_03", "Line_03",
        "Line_06", "Line_07",
        "Line_01", "Line_01",
        "Cart"
    ]

    print("Testing baseline methods...")

    for method in BaselineMethod:
        print(f"\n{method.value}:")
        scene_graph = generate_baseline_scene_graph(
            method=method,
            equipment_list=equipment,
            room_dimensions=(14.0, 14.0, 4.5),
            seed=42
        )

        # Count equipment (exclude room elements)
        equip_count = len([o for o in scene_graph if o.get("itemType") is None])
        print(f"  Generated {equip_count} equipment placements")
