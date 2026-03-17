"""
Hybrid Performance Evaluation for Reconfigurable Manufacturing Systems

Based on: Mastrangelo & Tolio (2024) "A hybrid method combining analytical
and simulation models for performance evaluation of reconfigurable
manufacturing systems", Journal of Manufacturing Systems.

This module implements a decomposition-based hybrid approach that combines:
- Analytical models (Markov Chain based) for serial lines
- Discrete event simulation for complex subsystems
- Remote Models for synthesizing blocking/starvation dynamics

Key Features:
- System decomposition into System Views
- Blocking and starvation propagation via Remote Models
- Iterative algorithm with flow conservation convergence
- Support for unreliable machines with multiple failure modes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import statistics

from catalog.equipment_catalog_cell import EQUIPMENT_CATALOG, get_equipment_aliases


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class MachineState(Enum):
    """Machine operational states"""
    UP = "up"           # Operational
    DOWN = "down"       # Failed
    BLOCKED = "blocked" # Blocked by downstream
    STARVED = "starved" # Starved by upstream


class LimitationType(Enum):
    """Types of material flow limitations"""
    DETERMINISTIC = "deterministic"  # Caused by workload imbalance
    STOCHASTIC = "stochastic"        # Caused by random failures


ROOM_LAYOUT_ELEMENTS = {
    "south_wall", "north_wall", "west_wall", "east_wall",
    "ceiling", "middle of the room",
}

# Default machine parameters by process stage
DEFAULT_MACHINE_PARAMS: Dict[str, Dict[str, float]] = {
    "MaterialFlow": {
        "processing_rate": 1.0,      # parts per time unit
        "efficiency": 0.95,          # availability
        "failure_rate": 0.02,        # failures per time unit when operating
        "repair_rate": 0.10,         # repairs per time unit when down
    },
    "Assembly": {
        "processing_rate": 0.8,
        "efficiency": 0.92,
        "failure_rate": 0.03,
        "repair_rate": 0.08,
    },
    "Utilities": {
        "processing_rate": 1.2,
        "efficiency": 0.97,
        "failure_rate": 0.01,
        "repair_rate": 0.15,
    },
}

FALLBACK_MACHINE_PARAMS = {
    "processing_rate": 1.0,
    "efficiency": 0.93,
    "failure_rate": 0.025,
    "repair_rate": 0.10,
}

# Default buffer parameters
DEFAULT_BUFFER_CAPACITY = 5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Machine:
    """
    Represents a machine in the manufacturing system.

    Attributes:
        id: Unique identifier
        processing_rate: μ - parts processed per time unit when operational
        failure_rate: p - rate of failure occurrence (operation-dependent)
        repair_rate: r - rate of repair completion
        position: Optional (x, y, z) coordinates
        stage: Process stage this machine belongs to
    """
    id: str
    processing_rate: float
    failure_rate: float = 0.02
    repair_rate: float = 0.10
    position: Optional[Tuple[float, float, float]] = None
    stage: Optional[str] = None

    @property
    def processing_time(self) -> float:
        """T^P - time to process one part"""
        return 1.0 / self.processing_rate if self.processing_rate > 0 else float('inf')

    @property
    def efficiency(self) -> float:
        """Machine availability/efficiency"""
        if self.failure_rate + self.repair_rate == 0:
            return 1.0
        return self.repair_rate / (self.failure_rate + self.repair_rate)

    @property
    def isolated_throughput(self) -> float:
        """Throughput if machine operated in isolation"""
        return self.processing_rate * self.efficiency


@dataclass
class Buffer:
    """
    Represents a finite-capacity buffer between machines.

    Attributes:
        id: Unique identifier
        capacity: N - maximum number of parts the buffer can hold
        upstream_machine: Machine feeding into this buffer
        downstream_machine: Machine drawing from this buffer
    """
    id: str
    capacity: int = DEFAULT_BUFFER_CAPACITY
    upstream_machine_id: Optional[str] = None
    downstream_machine_id: Optional[str] = None


@dataclass
class TransitionTimeDistribution:
    """
    Distribution of transition times between states.

    Supports both deterministic and stochastic components as per the paper.
    For stochastic parts, stores non-central moments for distribution fitting.
    """
    deterministic_values: List[float] = field(default_factory=list)
    stochastic_samples: List[float] = field(default_factory=list)

    @property
    def has_deterministic(self) -> bool:
        return len(self.deterministic_values) > 0

    @property
    def has_stochastic(self) -> bool:
        return len(self.stochastic_samples) > 0

    @property
    def deterministic_weight(self) -> float:
        """Fraction of transitions that are deterministic"""
        total = len(self.deterministic_values) + len(self.stochastic_samples)
        if total == 0:
            return 0.0
        return len(self.deterministic_values) / total

    def moment(self, order: int = 1) -> Optional[float]:
        """Compute non-central moment of stochastic component"""
        if not self.stochastic_samples:
            return None
        return sum(x ** order for x in self.stochastic_samples) / len(self.stochastic_samples)

    @property
    def mean(self) -> Optional[float]:
        """Expected value of the distribution"""
        if not self.stochastic_samples and not self.deterministic_values:
            return None

        total_weight = len(self.deterministic_values) + len(self.stochastic_samples)
        if total_weight == 0:
            return None

        det_contrib = sum(self.deterministic_values) if self.deterministic_values else 0
        stoch_contrib = sum(self.stochastic_samples) if self.stochastic_samples else 0

        return (det_contrib + stoch_contrib) / total_weight


@dataclass
class RemoteModel:
    """
    Remote Model RM[n] - Synthesizes material flow dynamics between System Views.

    Contains state-based representation with:
    - Operational state U: material flows freely
    - Starvation states S: upstream disruptions prevent flow
    - Blocking states B: downstream disruptions prevent flow

    Each limiting state has deterministic and stochastic components.
    """
    id: str

    # Starvation dynamics (from upstream System View)
    starvation_entry_dist: TransitionTimeDistribution = field(
        default_factory=TransitionTimeDistribution
    )
    starvation_exit_dist: TransitionTimeDistribution = field(
        default_factory=TransitionTimeDistribution
    )
    starvation_probability: float = 0.0

    # Blocking dynamics (from downstream System View)
    blocking_entry_dist: TransitionTimeDistribution = field(
        default_factory=TransitionTimeDistribution
    )
    blocking_exit_dist: TransitionTimeDistribution = field(
        default_factory=TransitionTimeDistribution
    )
    blocking_probability: float = 0.0

    @property
    def operational_probability(self) -> float:
        """Probability of being in operational state"""
        return max(0.0, 1.0 - self.starvation_probability - self.blocking_probability)


@dataclass
class SystemViewResult:
    """Results from evaluating a System View"""
    throughput: float
    average_buffer_levels: Dict[str, float]
    machine_state_probabilities: Dict[str, Dict[MachineState, float]]
    starvation_transitions: List[Tuple[float, float]]  # (entry_time, exit_time) pairs
    blocking_transitions: List[Tuple[float, float]]


@dataclass
class HybridSimulationReport:
    """
    Comprehensive performance evaluation report.

    Main metrics as defined in the paper:
    - TH: System throughput
    - WIP: Total work-in-progress (sum of average buffer levels)
    - Π_S: Average starvation probability
    - Π_B: Average blocking probability
    """
    iterations: int
    converged: bool

    # Primary metrics
    throughput: float
    throughput_std: float
    throughput_ci: Tuple[float, float]  # 95% confidence interval

    work_in_progress: float
    wip_std: float

    # State probabilities
    avg_starvation_prob: float
    avg_blocking_prob: float

    # Per-stage breakdown
    stage_metrics: Dict[str, Dict[str, float]]

    # Buffer levels
    buffer_levels: Dict[str, float]

    # Bottleneck identification
    bottleneck_stage: Optional[str]
    bottleneck_machine: Optional[str]

    # Efficiency metrics
    efficiency_index: Optional[float]

    # Validation
    missing_stages: List[str]
    disconnected_pairs: List[Tuple[str, str]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "throughput": self.throughput,
            "throughput_std": self.throughput_std,
            "throughput_ci": self.throughput_ci,
            "work_in_progress": self.work_in_progress,
            "wip_std": self.wip_std,
            "avg_starvation_prob": self.avg_starvation_prob,
            "avg_blocking_prob": self.avg_blocking_prob,
            "stage_metrics": self.stage_metrics,
            "buffer_levels": self.buffer_levels,
            "bottleneck_stage": self.bottleneck_stage,
            "bottleneck_machine": self.bottleneck_machine,
            "efficiency_index": self.efficiency_index,
            "missing_stages": self.missing_stages,
            "disconnected_pairs": self.disconnected_pairs,
            "notes": self.notes,
        }


# =============================================================================
# SYSTEM VIEW BASE CLASS
# =============================================================================

class SystemView(ABC):
    """
    Abstract base class for System Views.

    A System View SV(n) contains:
    - Detailed model of sub-system n (machines + buffers)
    - Pseudo-machines derived from Remote Models for boundary effects

    Can be implemented as:
    - Analytical (Markov Chain based)
    - Simulated (Discrete Event Simulation)
    """

    def __init__(
        self,
        view_id: str,
        machines: List[Machine],
        buffers: List[Buffer],
    ):
        self.view_id = view_id
        self.machines = {m.id: m for m in machines}
        self.buffers = {b.id: b for b in buffers}

        # Build machine order based on buffer connections (topological sort)
        self.machine_order = self._build_machine_order(machines, buffers)

        # Pseudo-machines from Remote Models
        self.upstream_remote: Optional[RemoteModel] = None
        self.downstream_remote: Optional[RemoteModel] = None

    def _build_machine_order(
        self, machines: List[Machine], buffers: List[Buffer]
    ) -> List[str]:
        """
        Build machine order based on buffer connections using topological sort.

        For network topologies (loops), picks an arbitrary ordering that respects
        local precedence constraints where possible.
        """
        machine_ids = {m.id for m in machines}
        if not machine_ids:
            return []

        # Build adjacency from buffers
        outgoing: Dict[str, List[str]] = {m.id: [] for m in machines}
        incoming: Dict[str, List[str]] = {m.id: [] for m in machines}

        for buf in buffers:
            up = buf.upstream_machine_id
            down = buf.downstream_machine_id
            if up in machine_ids and down in machine_ids:
                outgoing[up].append(down)
                incoming[down].append(up)

        # Find entry points (machines with no incoming from within the stage)
        entry_machines = [m_id for m_id in machine_ids if not incoming[m_id]]

        if not entry_machines:
            # Cyclic network - start with first machine
            entry_machines = [machines[0].id] if machines else []

        # BFS-based ordering
        ordered = []
        visited = set()

        queue = list(entry_machines)
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ordered.append(current)

            # Add successors
            for succ in outgoing.get(current, []):
                if succ not in visited:
                    queue.append(succ)

        # Add any remaining machines not reachable from entry points
        for m in machines:
            if m.id not in visited:
                ordered.append(m.id)

        return ordered

    def set_upstream_remote(self, remote: RemoteModel):
        """Set Remote Model for starvation from upstream"""
        self.upstream_remote = remote

    def set_downstream_remote(self, remote: RemoteModel):
        """Set Remote Model for blocking from downstream"""
        self.downstream_remote = remote

    @abstractmethod
    def evaluate(self, simulation_time: float = 10000.0) -> SystemViewResult:
        """
        Evaluate the System View and return performance metrics.

        Args:
            simulation_time: Duration for simulation-based views

        Returns:
            SystemViewResult with throughput, buffer levels, state probabilities
        """
        pass

    @property
    def first_machine(self) -> Optional[Machine]:
        """Machine at material entry point"""
        if self.machine_order:
            return self.machines.get(self.machine_order[0])
        return None

    @property
    def last_machine(self) -> Optional[Machine]:
        """Machine at material exit point"""
        if self.machine_order:
            return self.machines.get(self.machine_order[-1])
        return None


# =============================================================================
# SIMULATED SYSTEM VIEW (Discrete Event Simulation)
# =============================================================================

class SimulatedSystemView(SystemView):
    """
    Discrete Event Simulation based System View.

    Simulates part flow through machines with:
    - Operation-dependent failures
    - Finite buffer blocking/starvation
    - Pseudo-machine effects from Remote Models

    Supports both serial lines and network topologies:
    - Serial: Parts flow linearly through machine_order
    - Network: Parts follow buffer connections
    """

    def __init__(
        self,
        view_id: str,
        machines: List[Machine],
        buffers: List[Buffer],
        seed: Optional[int] = None,
    ):
        super().__init__(view_id, machines, buffers)
        self.rng = random.Random(seed)
        self.seed = seed

        # Build buffer adjacency for network simulation
        self._upstream_buffers: Dict[str, List[str]] = {m_id: [] for m_id in self.machines}
        self._downstream_buffers: Dict[str, List[str]] = {m_id: [] for m_id in self.machines}

        # Track buffers that have external (cross-stage) upstream sources
        self._external_input_buffers: List[str] = []

        for b_id, buf in self.buffers.items():
            if buf.upstream_machine_id in self.machines:
                self._downstream_buffers[buf.upstream_machine_id].append(b_id)
            else:
                # Buffer has external upstream (from previous stage)
                self._external_input_buffers.append(b_id)

            if buf.downstream_machine_id in self.machines:
                self._upstream_buffers[buf.downstream_machine_id].append(b_id)

        # Identify entry and exit machines
        # Entry machines: have no internal upstream buffers (may have external input buffers)
        self._entry_machines = [
            m_id for m_id in self.machines
            if not self._upstream_buffers[m_id]
        ]
        self._exit_machines = [
            m_id for m_id in self.machines
            if not self._downstream_buffers[m_id]
        ]

        # If no clear entry points, use first in order
        if not self._entry_machines and self.machine_order:
            self._entry_machines = [self.machine_order[0]]
        if not self._exit_machines and self.machine_order:
            self._exit_machines = [self.machine_order[-1]]

    def evaluate(self, simulation_time: float = 10000.0) -> SystemViewResult:
        """Run discrete event simulation"""

        # Initialize state tracking
        machine_states = {m_id: MachineState.UP for m_id in self.machines}
        buffer_levels = {b_id: 0 for b_id in self.buffers}

        # Initialize buffers with some parts for cyclic systems
        # If a buffer's upstream is internal (cyclic) and has no external feed, seed it
        for b_id, buf in self.buffers.items():
            if b_id not in self._external_input_buffers:
                # Check if this is part of an internal cycle (no true entry point)
                up_m = buf.upstream_machine_id
                if up_m in self.machines:
                    # If upstream machine also has no external input, seed the buffer
                    up_has_external = any(
                        b in self._external_input_buffers
                        for b in self._upstream_buffers.get(up_m, [])
                    )
                    if not up_has_external and up_m not in self._entry_machines:
                        # This is part of an internal cycle - initialize with some parts
                        buffer_levels[b_id] = buf.capacity // 2

        # Time tracking for state probabilities
        state_times = {
            m_id: {state: 0.0 for state in MachineState}
            for m_id in self.machines
        }
        buffer_level_times = {b_id: [] for b_id in self.buffers}

        # Transition tracking for Remote Model updates
        starvation_transitions: List[Tuple[float, float]] = []
        blocking_transitions: List[Tuple[float, float]] = []

        # Event-driven simulation
        current_time = 0.0
        parts_completed = 0

        # Track inter-departure times for throughput
        departure_times: List[float] = []

        # Initialize failure times for each machine
        next_failure = {}
        for m_id, machine in self.machines.items():
            if machine.failure_rate > 0:
                ttf = self.rng.expovariate(machine.failure_rate)
                next_failure[m_id] = ttf
            else:
                next_failure[m_id] = float('inf')

        # Simplified simulation loop
        time_step = 0.1  # Small time step for discrete approximation
        last_state_update = 0.0

        # Track blocking/starvation episodes
        starvation_start = None
        blocking_start = None

        while current_time < simulation_time:
            # Update state times
            dt = current_time - last_state_update
            for m_id in self.machines:
                state_times[m_id][machine_states[m_id]] += dt
            last_state_update = current_time

            # Fill external input buffers (from upstream stage via Remote Model)
            # These buffers receive parts from the previous stage
            for b_id in self._external_input_buffers:
                buf = self.buffers[b_id]
                # If upstream remote model is not starving, add parts to buffer
                if self.upstream_remote is None or self.rng.random() > self.upstream_remote.starvation_probability:
                    # Add part if buffer has space (simulates upstream production)
                    if buffer_levels[b_id] < buf.capacity:
                        # Use a base arrival rate (can be tuned)
                        arrival_rate = 0.8  # parts per time unit
                        if self.rng.random() < arrival_rate * time_step:
                            buffer_levels[b_id] += 1

            # Process each machine
            for i, m_id in enumerate(self.machine_order):
                machine = self.machines[m_id]

                # Check for failure
                if next_failure[m_id] <= current_time and machine_states[m_id] == MachineState.UP:
                    machine_states[m_id] = MachineState.DOWN
                    # Schedule repair
                    repair_time = current_time + self.rng.expovariate(machine.repair_rate) if machine.repair_rate > 0 else float('inf')
                    next_failure[m_id] = repair_time
                elif machine_states[m_id] == MachineState.DOWN and next_failure[m_id] <= current_time:
                    # Repair completed
                    machine_states[m_id] = MachineState.UP
                    # Schedule next failure
                    ttf = self.rng.expovariate(machine.failure_rate) if machine.failure_rate > 0 else float('inf')
                    next_failure[m_id] = current_time + ttf

                is_entry = m_id in self._entry_machines
                is_exit = m_id in self._exit_machines
                upstream_bufs = self._upstream_buffers.get(m_id, [])
                downstream_bufs = self._downstream_buffers.get(m_id, [])

                # Check starvation (entry machines affected by upstream Remote Model)
                if is_entry:
                    if self.upstream_remote and self.upstream_remote.starvation_probability > 0:
                        if self.rng.random() < self.upstream_remote.starvation_probability * time_step:
                            if machine_states[m_id] == MachineState.UP:
                                machine_states[m_id] = MachineState.STARVED
                                if starvation_start is None:
                                    starvation_start = current_time
                        elif machine_states[m_id] == MachineState.STARVED:
                            machine_states[m_id] = MachineState.UP
                            if starvation_start is not None:
                                starvation_transitions.append((starvation_start, current_time))
                                starvation_start = None
                    # If no upstream remote model, treat as infinite supply (no starvation)

                # Check blocking (exit machines affected by downstream Remote Model)
                if is_exit:
                    if self.downstream_remote and self.downstream_remote.blocking_probability > 0:
                        if self.rng.random() < self.downstream_remote.blocking_probability * time_step:
                            if machine_states[m_id] == MachineState.UP:
                                machine_states[m_id] = MachineState.BLOCKED
                                if blocking_start is None:
                                    blocking_start = current_time
                        elif machine_states[m_id] == MachineState.BLOCKED:
                            machine_states[m_id] = MachineState.UP
                            if blocking_start is not None:
                                blocking_transitions.append((blocking_start, current_time))
                                blocking_start = None
                    # If no downstream remote model, treat as infinite sink (no blocking)

                # Check buffer-based starvation (for non-entry machines)
                if upstream_bufs and not is_entry:
                    # Check if any upstream buffer has parts
                    has_parts = any(buffer_levels[b_id] > 0 for b_id in upstream_bufs)
                    if not has_parts and machine_states[m_id] == MachineState.UP:
                        machine_states[m_id] = MachineState.STARVED
                    elif has_parts and machine_states[m_id] == MachineState.STARVED:
                        # Recover from starvation when buffer has parts
                        machine_states[m_id] = MachineState.UP

                # Check buffer-based blocking (for non-exit machines)
                if downstream_bufs and not is_exit:
                    # Check if all downstream buffers are full
                    all_full = all(
                        buffer_levels[b_id] >= self.buffers[b_id].capacity
                        for b_id in downstream_bufs
                    )
                    if all_full and machine_states[m_id] == MachineState.UP:
                        machine_states[m_id] = MachineState.BLOCKED
                    elif not all_full and machine_states[m_id] == MachineState.BLOCKED:
                        # Recover from blocking when buffer has space
                        machine_states[m_id] = MachineState.UP

            # Process parts through the network
            for m_id in self.machine_order:
                machine = self.machines[m_id]

                if machine_states[m_id] == MachineState.UP:
                    # Process a part with probability based on processing rate
                    if self.rng.random() < machine.processing_rate * time_step:
                        is_entry = m_id in self._entry_machines
                        is_exit = m_id in self._exit_machines
                        upstream_bufs = self._upstream_buffers.get(m_id, [])
                        downstream_bufs = self._downstream_buffers.get(m_id, [])

                        # Can process if entry machine (infinite supply) or upstream buffer has parts
                        if is_entry:
                            can_process = True
                        elif upstream_bufs:
                            # Find a buffer with parts
                            can_process = any(buffer_levels[b_id] > 0 for b_id in upstream_bufs)
                        else:
                            can_process = True  # No upstream buffers, assume supply available

                        # Can output if exit machine (infinite sink) or downstream buffer has space
                        if is_exit:
                            can_output = True
                        elif downstream_bufs:
                            # Find a buffer with space
                            can_output = any(
                                buffer_levels[b_id] < self.buffers[b_id].capacity
                                for b_id in downstream_bufs
                            )
                        else:
                            can_output = True  # No downstream buffers, assume sink available

                        if can_process and can_output:
                            # Remove from first available upstream buffer
                            for b_id in upstream_bufs:
                                if buffer_levels[b_id] > 0:
                                    buffer_levels[b_id] -= 1
                                    break

                            # Add to first available downstream buffer
                            for b_id in downstream_bufs:
                                if buffer_levels[b_id] < self.buffers[b_id].capacity:
                                    buffer_levels[b_id] += 1
                                    break

                            # If exit machine, record completion
                            if is_exit:
                                parts_completed += 1
                                departure_times.append(current_time)

            # Record buffer levels
            for b_id in self.buffers:
                buffer_level_times[b_id].append((current_time, buffer_levels[b_id]))

            current_time += time_step

        # Compute results
        total_time = simulation_time

        # Throughput from inter-departure times
        if len(departure_times) > 1:
            inter_departures = [departure_times[i+1] - departure_times[i]
                               for i in range(len(departure_times) - 1)]
            avg_inter_departure = sum(inter_departures) / len(inter_departures)
            throughput = 1.0 / avg_inter_departure if avg_inter_departure > 0 else 0.0
        else:
            throughput = parts_completed / total_time if total_time > 0 else 0.0

        # Average buffer levels
        avg_buffer_levels = {}
        for b_id, level_history in buffer_level_times.items():
            if level_history:
                avg_buffer_levels[b_id] = sum(l for _, l in level_history) / len(level_history)
            else:
                avg_buffer_levels[b_id] = 0.0

        # State probabilities
        machine_state_probs = {}
        for m_id, times in state_times.items():
            total = sum(times.values())
            if total > 0:
                machine_state_probs[m_id] = {
                    state: t / total for state, t in times.items()
                }
            else:
                machine_state_probs[m_id] = {state: 0.0 for state in MachineState}

        return SystemViewResult(
            throughput=throughput,
            average_buffer_levels=avg_buffer_levels,
            machine_state_probabilities=machine_state_probs,
            starvation_transitions=starvation_transitions,
            blocking_transitions=blocking_transitions,
        )


# =============================================================================
# ANALYTICAL SYSTEM VIEW (Markov Chain based)
# =============================================================================

class AnalyticalSystemView(SystemView):
    """
    Analytical System View based on continuous-time Markov Chains.

    Implements the two-machine-one-buffer building block with:
    - Continuous flow approximation
    - Threshold-based control for blocking/starvation
    - Phase-type distribution fitting for transitions

    Based on the methodology in Magnanini & Tolio (2023).
    """

    def __init__(
        self,
        view_id: str,
        machines: List[Machine],
        buffers: List[Buffer],
    ):
        super().__init__(view_id, machines, buffers)

    def evaluate(self, simulation_time: float = 10000.0) -> SystemViewResult:
        """
        Analytical evaluation using decomposition equations.

        For a two-machine line with buffer B{k}:
        - Throughput: th = μ{k+1} * P(M{k+1} operational)
        - Buffer level: weighted average over state space
        """

        if len(self.machines) == 0:
            return SystemViewResult(
                throughput=0.0,
                average_buffer_levels={},
                machine_state_probabilities={},
                starvation_transitions=[],
                blocking_transitions=[],
            )

        # For simplicity, use approximation formulas
        # In full implementation, solve the Markov Chain system

        machine_list = [self.machines[m_id] for m_id in self.machine_order]

        # Find bottleneck machine (lowest isolated throughput)
        bottleneck_throughput = min(m.isolated_throughput for m in machine_list)

        # Approximate system throughput (accounting for blocking/starvation)
        # Using the decomposition approximation from the paper
        if len(machine_list) == 1:
            m = machine_list[0]
            throughput = m.isolated_throughput

            # Apply Remote Model effects
            if self.upstream_remote:
                throughput *= (1 - self.upstream_remote.starvation_probability)
            if self.downstream_remote:
                throughput *= (1 - self.downstream_remote.blocking_probability)
        else:
            # Multi-machine approximation
            # Use product-form approximation with correction factors
            throughput = bottleneck_throughput

            # Correction for finite buffers
            buffer_correction = 1.0
            for b_id, buf in self.buffers.items():
                # Approximate correction based on buffer capacity
                buffer_correction *= (1 - 1.0 / (buf.capacity + 2))

            throughput *= buffer_correction

            # Apply Remote Model effects
            if self.upstream_remote:
                throughput *= (1 - self.upstream_remote.starvation_probability)
            if self.downstream_remote:
                throughput *= (1 - self.downstream_remote.blocking_probability)

        # Compute state probabilities
        machine_state_probs = {}
        for m_id, machine in self.machines.items():
            # Approximate state probabilities
            p_up = machine.efficiency
            p_down = 1 - machine.efficiency

            # Estimate blocking/starvation based on position
            idx = self.machine_order.index(m_id)
            p_starved = 0.0
            p_blocked = 0.0

            if idx == 0 and self.upstream_remote:
                p_starved = self.upstream_remote.starvation_probability * p_up
                p_up -= p_starved

            if idx == len(self.machine_order) - 1 and self.downstream_remote:
                p_blocked = self.downstream_remote.blocking_probability * p_up
                p_up -= p_blocked

            machine_state_probs[m_id] = {
                MachineState.UP: max(0, p_up),
                MachineState.DOWN: p_down,
                MachineState.STARVED: p_starved,
                MachineState.BLOCKED: p_blocked,
            }

        # Approximate buffer levels using Little's law approximation
        avg_buffer_levels = {}
        for b_id, buf in self.buffers.items():
            # Find upstream and downstream machines
            upstream_m = self.machines.get(buf.upstream_machine_id)
            downstream_m = self.machines.get(buf.downstream_machine_id)

            if upstream_m and downstream_m:
                # Buffer tends to fill if upstream is faster
                rate_diff = upstream_m.processing_rate - downstream_m.processing_rate
                if rate_diff > 0:
                    # Buffer tends to fill
                    avg_level = buf.capacity * 0.6
                elif rate_diff < 0:
                    # Buffer tends to empty
                    avg_level = buf.capacity * 0.4
                else:
                    avg_level = buf.capacity * 0.5
            else:
                avg_level = buf.capacity * 0.5

            avg_buffer_levels[b_id] = avg_level

        # Generate synthetic transition data for Remote Model updates
        starvation_transitions = []
        blocking_transitions = []

        # Estimate deterministic cycle times
        if machine_list:
            slowest = max(machine_list, key=lambda m: m.processing_time)
            cycle_time = slowest.processing_time

            # Generate synthetic transitions based on cycle behavior
            first_machine = machine_list[0]
            last_machine = machine_list[-1]

            # Starvation-operational cycle for first machine
            if first_machine.processing_rate > slowest.processing_rate:
                wait_time = cycle_time - first_machine.processing_time
                starvation_transitions.append((cycle_time, wait_time))

            # Blocking-operational cycle for last machine
            if last_machine.processing_rate > slowest.processing_rate:
                wait_time = cycle_time - last_machine.processing_time
                blocking_transitions.append((cycle_time, wait_time))

        return SystemViewResult(
            throughput=throughput,
            average_buffer_levels=avg_buffer_levels,
            machine_state_probabilities=machine_state_probs,
            starvation_transitions=starvation_transitions,
            blocking_transitions=blocking_transitions,
        )


# =============================================================================
# HYBRID PERFORMANCE EVALUATOR
# =============================================================================

class HybridPerformanceEvaluator:
    """
    Main class for hybrid performance evaluation of manufacturing systems.

    Implements the decomposition algorithm from the paper:
    1. Decompose system into System Views
    2. Initialize Remote Models
    3. Iteratively evaluate SVs and update RMs until convergence
    4. Compute final performance metrics
    """

    def __init__(
        self,
        scene_graph,
        room_dimensions: Optional[Sequence[float]] = None,
        max_flow_gap: float = 6.0,
        seed: Optional[int] = None,
    ):
        self.room_dimensions = room_dimensions
        self.max_flow_gap = max_flow_gap
        self.seed = seed
        self.rng = random.Random(seed)
        self.alias_map = get_equipment_aliases()

        self._notes: List[str] = []

        # Parse scene graph and build system model
        self.machines, self.buffers = self._build_system_model(scene_graph)
        self.system_views: List[SystemView] = []
        self.remote_models: List[RemoteModel] = []

        # Decompose system
        self._decompose_system()

    def _build_system_model(
        self, scene_graph
    ) -> Tuple[Dict[str, Machine], Dict[str, Buffer]]:
        """Build machines and buffers from scene graph"""

        if isinstance(scene_graph, dict):
            objects = scene_graph.get("objects_in_room", [])
        elif isinstance(scene_graph, list):
            objects = scene_graph
        else:
            raise TypeError("scene_graph must be a dict or list")

        machines = {}
        connections = {}  # Track equipment connections for buffer inference

        for obj in objects:
            object_id = obj.get("object_id")
            if not object_id or object_id in ROOM_LAYOUT_ELEMENTS:
                continue
            if obj.get("itemType") in {"wall", "floor", "ceiling"}:
                continue

            canonical = self._canonicalize(object_id)
            if not canonical:
                continue

            catalog_entry = EQUIPMENT_CATALOG.get(canonical, {})
            stage = catalog_entry.get("process_stage")

            # Get machine parameters
            params = DEFAULT_MACHINE_PARAMS.get(stage, FALLBACK_MACHINE_PARAMS)

            position = self._extract_position(obj)

            machine = Machine(
                id=object_id,
                processing_rate=params["processing_rate"],
                failure_rate=params["failure_rate"],
                repair_rate=params["repair_rate"],
                position=position,
                stage=stage,
            )
            machines[object_id] = machine

            # Track connections
            conns = obj.get("connections", [])
            if conns:
                connections[object_id] = [c.get("object_id") for c in conns if c.get("object_id")]

        # Infer buffers from connections
        buffers = {}
        buffer_count = 0

        for upstream_id, downstream_ids in connections.items():
            for downstream_id in downstream_ids:
                if downstream_id in machines:
                    buffer_id = f"B_{buffer_count}"
                    buffers[buffer_id] = Buffer(
                        id=buffer_id,
                        capacity=DEFAULT_BUFFER_CAPACITY,
                        upstream_machine_id=upstream_id,
                        downstream_machine_id=downstream_id,
                    )
                    buffer_count += 1

        # Create inter-stage buffers if stages are not connected
        # Group machines by stage
        stages: Dict[str, List[str]] = {}
        for m_id, machine in machines.items():
            stage = machine.stage or "Unknown"
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(m_id)

        # Check if there are cross-stage connections
        stage_connections: Dict[str, set] = {s: set() for s in stages}
        for buf in buffers.values():
            up_stage = machines[buf.upstream_machine_id].stage if buf.upstream_machine_id in machines else None
            down_stage = machines[buf.downstream_machine_id].stage if buf.downstream_machine_id in machines else None
            if up_stage and down_stage and up_stage != down_stage:
                stage_connections[up_stage].add(down_stage)

        # Create virtual inter-stage buffers if needed
        stage_order = ["MaterialFlow", "Assembly", "Utilities"]
        for i in range(len(stage_order) - 1):
            from_stage = stage_order[i]
            to_stage = stage_order[i + 1]
            if from_stage not in stages or to_stage not in stages:
                continue
            if to_stage not in stage_connections.get(from_stage, set()):
                # No direct connection - create virtual inter-stage buffers
                # Connect last machines of from_stage to first machines of to_stage
                from_machines = stages[from_stage]
                to_machines = stages[to_stage]
                # Create one buffer per destination machine (simplified)
                for to_m in to_machines[:3]:  # Limit to first 3 machines
                    from_m = from_machines[0] if from_machines else None
                    if from_m:
                        buffer_id = f"B_stage_{buffer_count}"
                        buffers[buffer_id] = Buffer(
                            id=buffer_id,
                            capacity=DEFAULT_BUFFER_CAPACITY * 2,  # Larger inter-stage buffer
                            upstream_machine_id=from_m,
                            downstream_machine_id=to_m,
                        )
                        buffer_count += 1

        return machines, buffers

    def _decompose_system(self):
        """
        Decompose the system into System Views based on process stages.

        Creates one System View per stage, with Remote Models at interfaces.
        Buffers are assigned to the stage of their downstream machine (pull model).
        """

        # Group machines by stage
        stages: Dict[str, List[Machine]] = {}
        for machine in self.machines.values():
            stage = machine.stage or "Unknown"
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(machine)

        # Determine stage order based on connections/positions
        stage_order = self._determine_stage_order(stages)

        # Build machine-to-stage mapping
        machine_to_stage = {m.id: m.stage or "Unknown" for m in self.machines.values()}

        # Create System Views
        for i, stage in enumerate(stage_order):
            machines = stages.get(stage, [])
            if not machines:
                continue

            machine_ids = {m.id for m in machines}

            # Find buffers for this stage:
            # 1. Buffers entirely within this stage (both ends in same stage)
            # 2. Buffers where downstream is in this stage (incoming buffers)
            # This follows a "pull" model where each stage owns its input buffers
            stage_buffers = []
            for b in self.buffers.values():
                up_in_stage = b.upstream_machine_id in machine_ids
                down_in_stage = b.downstream_machine_id in machine_ids

                if up_in_stage and down_in_stage:
                    # Internal buffer
                    stage_buffers.append(b)
                elif down_in_stage and not up_in_stage:
                    # Incoming buffer from previous stage - include it
                    stage_buffers.append(b)

            # Choose simulation or analytical based on complexity
            # Use analytical for simple serial lines, simulation for complex layouts
            if len(machines) <= 3 and len(stage_buffers) <= 2:
                sv = AnalyticalSystemView(
                    view_id=f"SV_{stage}",
                    machines=machines,
                    buffers=stage_buffers,
                )
            else:
                sv = SimulatedSystemView(
                    view_id=f"SV_{stage}",
                    machines=machines,
                    buffers=stage_buffers,
                    seed=self.seed,
                )

            self.system_views.append(sv)

        # Create Remote Models between consecutive System Views
        for i in range(len(self.system_views) - 1):
            rm = RemoteModel(id=f"RM_{i}")
            self.remote_models.append(rm)

        # Link System Views with Remote Models
        for i, sv in enumerate(self.system_views):
            if i > 0:
                sv.set_upstream_remote(self.remote_models[i - 1])
            if i < len(self.remote_models):
                sv.set_downstream_remote(self.remote_models[i])

    def _determine_stage_order(self, stages: Dict[str, List[Machine]]) -> List[str]:
        """Determine the order of process stages based on typical flow"""

        # Default order
        default_order = ["MaterialFlow", "Assembly", "Utilities"]

        ordered = []
        for stage in default_order:
            if stage in stages:
                ordered.append(stage)

        # Add any remaining stages
        for stage in stages:
            if stage not in ordered:
                ordered.append(stage)

        return ordered

    def run(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 0.01,
        convergent_iterations_required: int = 5,
        simulation_time: float = 10000.0,
        verbose: bool = False,
    ) -> HybridSimulationReport:
        """
        Run the hybrid evaluation algorithm.

        Algorithm (from paper):
        1. Forward iteration: evaluate SVs, update starvation in RMs
        2. Backward iteration: evaluate SVs, update blocking in RMs
        3. Repeat until flow conservation achieved

        Args:
            max_iterations: Maximum algorithm iterations
            convergence_threshold: Relative throughput difference for convergence
            convergent_iterations_required: Number of convergent iterations before stopping
            simulation_time: Time for each SV simulation
            verbose: Print progress information

        Returns:
            HybridSimulationReport with comprehensive metrics
        """

        if not self.system_views:
            return self._create_empty_report()

        throughput_history: List[List[float]] = []
        convergent_count = 0
        converged = False

        for iteration in range(max_iterations):
            iteration_throughputs = []

            # Forward iteration: update starvation
            for i, sv in enumerate(self.system_views):
                result = sv.evaluate(simulation_time)
                iteration_throughputs.append(result.throughput)

                # Update starvation in downstream Remote Model
                if i < len(self.remote_models):
                    self._update_remote_starvation(
                        self.remote_models[i],
                        result,
                        sv.last_machine,
                    )

            # Backward iteration: update blocking
            for i in range(len(self.system_views) - 1, -1, -1):
                sv = self.system_views[i]
                result = sv.evaluate(simulation_time)

                # Update blocking in upstream Remote Model
                if i > 0:
                    self._update_remote_blocking(
                        self.remote_models[i - 1],
                        result,
                        sv.first_machine,
                    )

            throughput_history.append(iteration_throughputs)

            # Check convergence (flow conservation)
            if self._check_convergence(iteration_throughputs, convergence_threshold):
                convergent_count += 1
                if convergent_count >= convergent_iterations_required:
                    converged = True
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
            else:
                convergent_count = 0

            if verbose and iteration % 10 == 0:
                avg_th = sum(iteration_throughputs) / len(iteration_throughputs)
                print(f"Iteration {iteration + 1}: avg throughput = {avg_th:.4f}")

        # Compute final metrics from convergent iterations
        return self._compute_final_report(
            throughput_history,
            converged,
            simulation_time,
        )

    def _update_remote_starvation(
        self,
        remote: RemoteModel,
        result: SystemViewResult,
        exit_machine: Optional[Machine],
    ):
        """Update starvation dynamics in Remote Model from upstream SV"""

        if not exit_machine:
            return

        # Extract starvation probability from machine state
        state_probs = result.machine_state_probabilities.get(exit_machine.id, {})
        starved_prob = state_probs.get(MachineState.STARVED, 0.0)

        # Update Remote Model
        remote.starvation_probability = starved_prob

        # Update transition time distribution
        if result.starvation_transitions:
            for entry_time, exit_time in result.starvation_transitions:
                remote.starvation_entry_dist.stochastic_samples.append(entry_time)
                remote.starvation_exit_dist.stochastic_samples.append(exit_time)

    def _update_remote_blocking(
        self,
        remote: RemoteModel,
        result: SystemViewResult,
        entry_machine: Optional[Machine],
    ):
        """Update blocking dynamics in Remote Model from downstream SV"""

        if not entry_machine:
            return

        # Extract blocking probability from machine state
        state_probs = result.machine_state_probabilities.get(entry_machine.id, {})
        blocked_prob = state_probs.get(MachineState.BLOCKED, 0.0)

        # Update Remote Model
        remote.blocking_probability = blocked_prob

        # Update transition time distribution
        if result.blocking_transitions:
            for entry_time, exit_time in result.blocking_transitions:
                remote.blocking_entry_dist.stochastic_samples.append(entry_time)
                remote.blocking_exit_dist.stochastic_samples.append(exit_time)

    def _check_convergence(
        self,
        throughputs: List[float],
        threshold: float,
    ) -> bool:
        """Check if flow is conserved among System Views"""

        if len(throughputs) < 2:
            return True

        max_th = max(throughputs)
        min_th = min(throughputs)

        if max_th == 0:
            return True

        relative_diff = (max_th - min_th) / max_th
        return relative_diff < threshold

    def _compute_final_report(
        self,
        throughput_history: List[List[float]],
        converged: bool,
        simulation_time: float,
    ) -> HybridSimulationReport:
        """Compute final performance metrics from convergent iterations"""

        # Use last few iterations for estimates
        num_samples = min(5, len(throughput_history))
        recent_throughputs = [
            sum(th) / len(th) for th in throughput_history[-num_samples:]
        ]

        if recent_throughputs:
            throughput = statistics.mean(recent_throughputs)
            throughput_std = statistics.stdev(recent_throughputs) if len(recent_throughputs) > 1 else 0.0

            # 95% confidence interval
            if len(recent_throughputs) > 1:
                se = throughput_std / math.sqrt(len(recent_throughputs))
                t_value = 2.776  # t-distribution for 4 df, 95%
                ci = (throughput - t_value * se, throughput + t_value * se)
            else:
                ci = (throughput, throughput)
        else:
            throughput = 0.0
            throughput_std = 0.0
            ci = (0.0, 0.0)

        # Evaluate final state for detailed metrics
        buffer_levels = {}
        stage_metrics = {}
        all_starvation_probs = []
        all_blocking_probs = []

        for sv in self.system_views:
            result = sv.evaluate(simulation_time)

            # Buffer levels
            buffer_levels.update(result.average_buffer_levels)

            # Stage metrics with theoretical capacity
            stage = sv.view_id.replace("SV_", "")

            # Compute theoretical max throughput for this stage
            # (sum of isolated throughputs of all machines)
            theoretical_capacity = sum(
                m.isolated_throughput for m in sv.machines.values()
            )

            stage_metrics[stage] = {
                "throughput": result.throughput,
                "equipment_count": len(sv.machines),
                "theoretical_capacity": theoretical_capacity,
                "utilization": result.throughput / theoretical_capacity if theoretical_capacity > 0 else 0.0,
            }

            # State probabilities
            for m_id, probs in result.machine_state_probabilities.items():
                all_starvation_probs.append(probs.get(MachineState.STARVED, 0.0))
                all_blocking_probs.append(probs.get(MachineState.BLOCKED, 0.0))

        # Work in progress
        wip = sum(buffer_levels.values())
        wip_std = 0.0  # Would need multiple samples

        # Average state probabilities
        avg_starv = sum(all_starvation_probs) / len(all_starvation_probs) if all_starvation_probs else 0.0
        avg_block = sum(all_blocking_probs) / len(all_blocking_probs) if all_blocking_probs else 0.0

        # Identify bottleneck based on utilization rate
        # The bottleneck is the stage with highest utilization (most constrained)
        bottleneck_stage = None
        max_utilization = -1.0
        for stage, metrics in stage_metrics.items():
            if metrics["utilization"] > max_utilization:
                max_utilization = metrics["utilization"]
                bottleneck_stage = stage

        # Efficiency index
        if throughput > 0:
            # Compute reference throughput (isolated machines)
            ref_throughput = min(
                m.isolated_throughput for m in self.machines.values()
            ) if self.machines else throughput
            efficiency_index = throughput / ref_throughput if ref_throughput > 0 else None
        else:
            efficiency_index = None

        return HybridSimulationReport(
            iterations=len(throughput_history),
            converged=converged,
            throughput=throughput,
            throughput_std=throughput_std,
            throughput_ci=ci,
            work_in_progress=wip,
            wip_std=wip_std,
            avg_starvation_prob=avg_starv,
            avg_blocking_prob=avg_block,
            stage_metrics=stage_metrics,
            buffer_levels=buffer_levels,
            bottleneck_stage=bottleneck_stage,
            bottleneck_machine=None,
            efficiency_index=efficiency_index,
            missing_stages=[],
            disconnected_pairs=[],
            notes=self._notes.copy(),
        )

    def _create_empty_report(self) -> HybridSimulationReport:
        """Create empty report when no system to evaluate"""
        return HybridSimulationReport(
            iterations=0,
            converged=False,
            throughput=0.0,
            throughput_std=0.0,
            throughput_ci=(0.0, 0.0),
            work_in_progress=0.0,
            wip_std=0.0,
            avg_starvation_prob=0.0,
            avg_blocking_prob=0.0,
            stage_metrics={},
            buffer_levels={},
            bottleneck_stage=None,
            bottleneck_machine=None,
            efficiency_index=None,
            missing_stages=[],
            disconnected_pairs=[],
            notes=["No system views to evaluate"],
        )

    def _canonicalize(self, object_id: str) -> Optional[str]:
        """Convert object_id to canonical equipment name"""
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

    def _extract_position(self, obj: Dict) -> Optional[Tuple[float, float, float]]:
        """Extract position from scene graph object"""
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


# =============================================================================
# CLI AND UTILITIES
# =============================================================================

def _load_scene_graph(path: Path):
    """Load scene graph from JSON file"""
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"objects_in_room": data}
    return data


def _save_results(
    report: HybridSimulationReport,
    scene_graph_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save evaluation results to JSON file.

    Args:
        report: The evaluation report to save
        scene_graph_path: Path to the input scene graph (for naming)
        output_dir: Output directory (default: evaluation/results)

    Returns:
        Path to the saved results file
    """
    # Default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    scene_name = scene_graph_path.stem  # e.g., "scene_graph_cell03"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scene_name}_{timestamp}.json"
    output_path = output_dir / filename

    # Build output data
    result_data = {
        "metadata": {
            "scene_graph": str(scene_graph_path),
            "timestamp": datetime.now().isoformat(),
            "scene_name": scene_name,
        },
        "report": report.to_dict(),
    }

    # Save to JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return output_path


def _save_csv_metrics(
    report: HybridSimulationReport,
    scene_graph_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save key metrics to CSV file for paper tables.

    Metrics format based on Mastrangelo & Tolio (2024) Table 2:
    - TH: Throughput (parts/time unit)
    - WIP: Work-in-Progress (total parts in buffers)
    - Pi_S: Average starvation probability
    - Pi_B: Average blocking probability

    Args:
        report: The evaluation report
        scene_graph_path: Path to the input scene graph
        output_dir: Output directory (default: evaluation/results)

    Returns:
        Path to the CSV file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV file for aggregated metrics across runs
    csv_path = output_dir / "metrics_summary.csv"

    # Check if file exists to determine if we need to write header
    file_exists = csv_path.exists()

    # Prepare row data matching paper Table 2 format
    scene_name = scene_graph_path.stem
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate total buffer capacity for error normalization
    total_buffer_capacity = sum(report.buffer_levels.values()) if report.buffer_levels else 0
    num_buffers = len(report.buffer_levels) if report.buffer_levels else 0

    row_data = {
        "timestamp": timestamp,
        "scene": scene_name,
        "iterations": report.iterations,
        "converged": report.converged,
        # Primary metrics (Table 2)
        "TH": report.throughput,
        "TH_std": report.throughput_std,
        "TH_ci_low": report.throughput_ci[0],
        "TH_ci_high": report.throughput_ci[1],
        "WIP": report.work_in_progress,
        "WIP_std": report.wip_std,
        "Pi_S": report.avg_starvation_prob,
        "Pi_B": report.avg_blocking_prob,
        # Additional info
        "num_buffers": num_buffers,
        "bottleneck_stage": report.bottleneck_stage or "",
        "efficiency_index": report.efficiency_index if report.efficiency_index else "",
    }

    # Column order matching paper format
    fieldnames = [
        "timestamp", "scene", "iterations", "converged",
        "TH", "TH_std", "TH_ci_low", "TH_ci_high",
        "WIP", "WIP_std",
        "Pi_S", "Pi_B",
        "num_buffers", "bottleneck_stage", "efficiency_index"
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Also save per-buffer levels to a separate CSV for detailed analysis
    if report.buffer_levels:
        buffer_csv_path = output_dir / "buffer_levels.csv"
        buffer_exists = buffer_csv_path.exists()

        buffer_fieldnames = ["timestamp", "scene", "buffer_id", "avg_level"]

        with buffer_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=buffer_fieldnames)
            if not buffer_exists:
                writer.writeheader()
            for buf_id, level in report.buffer_levels.items():
                writer.writerow({
                    "timestamp": timestamp,
                    "scene": scene_name,
                    "buffer_id": buf_id,
                    "avg_level": level,
                })

    return csv_path


def _print_report(report: HybridSimulationReport) -> None:
    """Print formatted simulation report"""
    print("\n" + "=" * 60)
    print("HYBRID PERFORMANCE EVALUATION REPORT")
    print("=" * 60)

    print(f"\nIterations: {report.iterations}")
    print(f"Converged: {'Yes' if report.converged else 'No'}")

    print(f"\n--- PRIMARY METRICS ---")
    print(f"Throughput: {report.throughput:.4f} parts/time unit")
    print(f"  Std Dev: {report.throughput_std:.4f}")
    print(f"  95% CI: [{report.throughput_ci[0]:.4f}, {report.throughput_ci[1]:.4f}]")

    print(f"\nWork-in-Progress (WIP): {report.work_in_progress:.2f} parts")

    print(f"\n--- STATE PROBABILITIES ---")
    print(f"Avg Starvation: {report.avg_starvation_prob:.2%}")
    print(f"Avg Blocking: {report.avg_blocking_prob:.2%}")

    if report.stage_metrics:
        print(f"\n--- STAGE BREAKDOWN ---")
        for stage, metrics in report.stage_metrics.items():
            print(f"  {stage}:")
            print(f"    Throughput: {metrics.get('throughput', 0):.4f}")
            print(f"    Equipment: {metrics.get('equipment_count', 0)}")
            print(f"    Theoretical Capacity: {metrics.get('theoretical_capacity', 0):.4f}")
            print(f"    Utilization: {metrics.get('utilization', 0):.2%}")

    if report.buffer_levels:
        print(f"\n--- BUFFER LEVELS ---")
        for buf_id, level in report.buffer_levels.items():
            print(f"  {buf_id}: {level:.2f}")

    if report.bottleneck_stage:
        print(f"\n--- BOTTLENECK ---")
        print(f"Stage: {report.bottleneck_stage}")

    if report.efficiency_index is not None:
        print(f"\n--- EFFICIENCY ---")
        print(f"Efficiency Index: {report.efficiency_index:.2%}")

    if report.notes:
        print(f"\n--- NOTES ---")
        for note in report.notes:
            print(f"  - {note}")

    print("\n" + "=" * 60)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid performance evaluation for manufacturing systems"
    )
    parser.add_argument(
        "scene_graph",
        type=Path,
        help="Path to scene_graph.json"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum algorithm iterations (default: 50)"
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=10000.0,
        help="Simulation time per iteration (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: evaluation/results)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file"
    )

    args = parser.parse_args()

    scene_graph = _load_scene_graph(args.scene_graph)

    evaluator = HybridPerformanceEvaluator(
        scene_graph=scene_graph,
        seed=args.seed,
    )

    report = evaluator.run(
        max_iterations=args.max_iterations,
        simulation_time=args.simulation_time,
        verbose=args.verbose,
    )

    _print_report(report)

    # Save results unless --no-save is specified
    if not args.no_save:
        output_path = _save_results(report, args.scene_graph, args.output_dir)
        print(f"\nResults saved to: {output_path}")

        # Also save to CSV for paper tables
        csv_path = _save_csv_metrics(report, args.scene_graph, args.output_dir)
        print(f"CSV metrics appended to: {csv_path}")


if __name__ == "__main__":
    main()
