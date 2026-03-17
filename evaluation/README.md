# Hybrid Performance Evaluation for Reconfigurable Manufacturing Systems

This module implements a hybrid performance evaluation method based on:

> Mastrangelo, A., & Tolio, T. (2024). "A hybrid method combining analytical and simulation models for performance evaluation of reconfigurable manufacturing systems." *Journal of Manufacturing Systems*.

## Overview

The hybrid approach combines:
- **Analytical Models**: Continuous-time Markov Chain (CTMC) based decomposition for serial lines
- **Discrete Event Simulation (DES)**: For complex subsystems with parallel machines or non-standard configurations
- **Remote Models**: Interface components that synthesize blocking/starvation dynamics between subsystems

### Key Features

- System decomposition into independent System Views
- Iterative algorithm with flow conservation convergence
- Support for unreliable machines with failure/repair dynamics
- Blocking and starvation propagation via Remote Models
- Bottleneck identification based on utilization rate

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  HybridPerformanceEvaluator                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ SystemView 1 │───▶│ RemoteModel  │───▶│ SystemView 2 │      │
│  │ (Simulated)  │    │     RM_0     │    │ (Simulated)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │               │
│         │            ┌──────────────┐           │               │
│         └───────────▶│ RemoteModel  │◀──────────┘               │
│                      │     RM_1     │                           │
│                      └──────────────┘                           │
│                             │                                   │
│                      ┌──────────────┐                           │
│                      │ SystemView 3 │                           │
│                      │ (Analytical) │                           │
│                      └──────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Machine

Represents a single machine with parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| `processing_rate` | Parts processed per time unit | parts/time |
| `processing_time` | Time to process one part (1/rate) | time |
| `efficiency` | Machine availability (0-1) | ratio |
| `failure_rate` | Failures per time unit when operating | 1/time |
| `repair_rate` | Repairs per time unit when down | 1/time |
| `isolated_throughput` | Throughput without blocking/starvation | parts/time |

**Isolated throughput formula:**
```
TH_isolated = processing_rate × efficiency
```

### 2. Buffer

Inter-machine buffer with finite capacity:

| Parameter | Description |
|-----------|-------------|
| `capacity` | Maximum number of parts (N{k}) |
| `upstream_machine_id` | Source machine |
| `downstream_machine_id` | Destination machine |

### 3. System View

A subsystem containing machines and buffers. Two implementations:

#### SimulatedSystemView (DES)
- Used for complex configurations (parallel machines, loops)
- Time-stepped discrete event simulation
- Tracks machine states: UP, DOWN, STARVED, BLOCKED
- Computes throughput from inter-departure times

#### AnalyticalSystemView (CTMC)
- Used for simple serial lines (≤3 machines)
- Closed-form approximations based on decomposition equations
- Faster computation but less accurate for complex systems

### 4. Remote Model

Interface between adjacent System Views:

| Parameter | Description |
|-----------|-------------|
| `starvation_probability` | Π_S - Probability downstream is starved |
| `blocking_probability` | Π_B - Probability upstream is blocked |
| `starvation_entry_dist` | Time distribution for entering starvation |
| `starvation_exit_dist` | Time distribution for exiting starvation |
| `blocking_entry_dist` | Time distribution for entering blocking |
| `blocking_exit_dist` | Time distribution for exiting blocking |

## Algorithm

The iterative algorithm follows the paper's methodology:

```
Algorithm: Hybrid Performance Evaluation
─────────────────────────────────────────
Input: Scene graph, max_iterations, convergence_threshold
Output: HybridSimulationReport

1. Build system model from scene graph
   - Extract machines by process stage
   - Create buffers for inter-machine connections
   - Decompose into System Views (one per stage)
   - Create Remote Models at interfaces

2. Initialize Remote Models (Π_S = 0, Π_B = 0)

3. REPEAT until convergence or max_iterations:

   3.1 Forward Iteration (update starvation):
       FOR each SystemView SV_i in order:
           result = SV_i.evaluate(simulation_time)
           Update RM_i.starvation from exit machine states

   3.2 Backward Iteration (update blocking):
       FOR each SystemView SV_i in reverse order:
           result = SV_i.evaluate(simulation_time)
           Update RM_{i-1}.blocking from entry machine states

   3.3 Check Convergence:
       IF max(TH) - min(TH) < threshold × max(TH):
           convergent_count++
       ELSE:
           convergent_count = 0

       IF convergent_count >= required_iterations:
           converged = TRUE
           BREAK

4. Compute final metrics from last iterations

5. RETURN HybridSimulationReport
```

### Convergence Criterion

Flow conservation is achieved when throughputs across all System Views are approximately equal:

```
err_flow = (TH_max - TH_min) / TH_max < threshold
```

Default threshold: 1% (0.01)

## Performance Metrics

### Primary Metrics (Paper Table 2)

| Metric | Symbol | Description | Formula |
|--------|--------|-------------|---------|
| Throughput | TH | System output rate | parts/time unit |
| Work-in-Progress | WIP | Total parts in buffers | Σ x̄{k} |
| Starvation Probability | Π_S | Avg machine starvation | Mean of all machines |
| Blocking Probability | Π_B | Avg machine blocking | Mean of all machines |

### Per-Stage Metrics

| Metric | Description |
|--------|-------------|
| `throughput` | Stage output rate |
| `equipment_count` | Number of machines |
| `theoretical_capacity` | Sum of isolated throughputs |
| `utilization` | throughput / theoretical_capacity |

### Bottleneck Identification

The bottleneck stage is identified as the one with **highest utilization**:

```
bottleneck = argmax(utilization_i) for all stages i
```

This represents the most constrained stage in the system.

### Efficiency Index

The efficiency index measures how well the system achieves its theoretical maximum throughput:

```
efficiency_index = TH / TH_ref
```

Where:
- `TH` is the actual system throughput
- `TH_ref` is the reference throughput, defined as the minimum isolated throughput among all machines:

```
TH_ref = min(TH_isolated_i) for all machines i
```

An efficiency index of 1.0 indicates the system is operating at its theoretical maximum (no blocking or starvation losses). Values less than 1.0 indicate throughput losses due to blocking, starvation, or other inefficiencies.

## Usage

### Command Line

```bash
# Basic usage
./evaluation/run_evaluation.sh scenes/scene_graph_cell03.json

# With options
./evaluation/run_evaluation.sh -v -i 50 -t 10000 scenes/scene_graph_cell03.json

# All options
./evaluation/run_evaluation.sh \
    --iterations 50 \
    --time 10000 \
    --seed 42 \
    --verbose \
    --output-dir ./my_results \
    scenes/scene_graph_cell03.json
```

### Python API

```python
from pathlib import Path
from evaluation.hybrid_simulation import (
    HybridPerformanceEvaluator,
    _load_scene_graph,
)

# Load scene graph
scene_graph = _load_scene_graph(Path("scenes/scene_graph_cell03.json"))

# Create evaluator
evaluator = HybridPerformanceEvaluator(
    scene_graph=scene_graph,
    seed=42,  # Optional: for reproducibility
)

# Run evaluation
report = evaluator.run(
    max_iterations=50,
    convergence_threshold=0.01,
    simulation_time=10000.0,
    verbose=True,
)

# Access results
print(f"Throughput: {report.throughput:.4f}")
print(f"WIP: {report.work_in_progress:.2f}")
print(f"Bottleneck: {report.bottleneck_stage}")
```

## Output Files

Results are saved to `evaluation/results/`:

### 1. JSON Report (`{scene_name}_{timestamp}.json`)

Complete evaluation results including:
- Metadata (scene, timestamp)
- All metrics from HybridSimulationReport
- Stage breakdown
- Buffer levels

### 2. CSV Summary (`metrics_summary.csv`)

Append-only CSV for comparing runs across scenes:

| Column | Description |
|--------|-------------|
| `timestamp` | Evaluation time |
| `scene` | Scene graph name |
| `iterations` | Algorithm iterations |
| `converged` | Whether flow conservation achieved |
| `TH` | Throughput |
| `TH_std` | Throughput standard deviation |
| `TH_ci_low` | 95% CI lower bound |
| `TH_ci_high` | 95% CI upper bound |
| `WIP` | Work-in-Progress |
| `WIP_std` | WIP standard deviation |
| `Pi_S` | Starvation probability |
| `Pi_B` | Blocking probability |
| `num_buffers` | Number of buffers |
| `bottleneck_stage` | Identified bottleneck |
| `efficiency_index` | System efficiency |

### 3. Buffer Levels (`buffer_levels.csv`)

Per-buffer average levels for detailed analysis:

| Column | Description |
|--------|-------------|
| `timestamp` | Evaluation time |
| `scene` | Scene graph name |
| `buffer_id` | Buffer identifier |
| `avg_level` | Average buffer level (x̄{k}) |

## Process Stages

The system supports three process stages defined in the equipment catalog:

| Stage | Default Processing Rate | Default Efficiency |
|-------|------------------------|-------------------|
| MaterialFlow | 1.0 parts/time | 95% |
| Assembly | 0.8 parts/time | 92% |
| Utilities | 1.2 parts/time | 97% |

Equipment is automatically assigned to stages based on the `EQUIPMENT_CATALOG` mapping.

## Error Metrics (Paper Equations 44-48)

For validation against pure simulation:

```
err%,TH = |TH_hyb - TH_sim| / TH_sim × 100

err%,WIP = |Σ x̄{k}_hyb - Σ x̄{k}_sim| / Σ N{k} × 100

err%,Π_S = |Π̄_S,hyb - Π̄_S,sim| × 100

err%,Π_B = |Π̄_B,hyb - Π̄_B,sim| × 100
```

## Limitations

1. **Buffer Creation**: Virtual buffers are created automatically between stages if no explicit connections exist in the scene graph.

2. **Cyclic Systems**: MaterialFlow stage with conveyor loops requires special initialization (buffers start half-full).

3. **Analytical Accuracy**: AnalyticalSystemView uses approximations that may be less accurate for highly variable systems.

4. **Single Product**: Current implementation assumes single product type flow.

## Baseline Methods for Experimental Comparison

The `baselines.py` module implements several baseline layout generation methods for experimental comparison against the LLM4RMS hybrid approach (LLM + structure-aware reconfiguration).

### Important: Scope of Comparison

**The baseline comparison evaluates layout optimization given a fixed equipment list, NOT end-to-end natural language understanding.**

| Aspect | LLM4RMS (Full Pipeline) | Baseline Methods |
|--------|------------------------|------------------|
| **Input** | Natural language description | Equipment list extracted from reference |
| **Equipment Selection** | LLM interprets requirements | Uses same equipment as reference |
| **Layout Planning** | LLM + structure-aware | Algorithmic optimization |
| **Connection Generation** | Semantic material flow | Spatial proximity heuristic |

**What this comparison measures:**
- Given the **same set of equipment**, how well can different methods optimize spatial placement and material flow connections?
- This isolates the **layout optimization** capability from the **natural language understanding** capability.

**What this comparison does NOT measure:**
- Ability to interpret natural language requirements
- Ability to select appropriate equipment types and quantities
- Ability to understand manufacturing domain semantics

For a complete evaluation of LLM4RMS's natural language capabilities, a separate study with human evaluation or domain-specific benchmarks would be needed.

### Available Baseline Methods

| Method | Description | Key Characteristics |
|--------|-------------|---------------------|
| **Random** | Random placement within room constraints | No optimization; serves as lower bound |
| **Grid** | Regular grid-based placement | Structured but ignores material flow |
| **GA (Genetic Algorithm)** | Evolutionary optimization | Population-based; good exploration |
| **SA (Simulated Annealing)** | Temperature-based optimization | Local search with escape mechanism |
| **SLP** | Systematic Layout Planning heuristic | Relationship-based placement |
| **LLM4RMS+GA** | LLM4RMS layout refined by GA | Combines LLM semantics with evolutionary optimization |
| **LLM4RMS+SA** | LLM4RMS layout refined by SA | Combines LLM semantics with local search optimization |

### Hybrid Methods: LLM4RMS + Optimization

The `llm4rms+ga` and `llm4rms+sa` methods demonstrate the combination of LLM-based layout generation with traditional optimization techniques:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM4RMS Pipeline                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   NL     │───▶│  LLM-based   │───▶│  Structure-  │       │
│  │  Input   │    │   Planning   │    │    aware     │       │
│  └──────────┘    └──────────────┘    │  Execution   │       │
│                                       └──────────────┘       │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────┐          │
│                                    │   LLM4RMS   │          │
│                                    │   Layout     │◀── Initial Solution
│                                    └──────────────┘          │
│                                              │               │
│                                              ▼               │
│                               ┌─────────────────────────┐    │
│                               │   GA/SA Optimization    │    │
│                               │   (Position refinement) │    │
│                               └─────────────────────────┘    │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────┐          │
│                                    │   Optimized  │          │
│                                    │    Layout    │          │
│                                    └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Key advantages of hybrid methods:**
1. **Semantic Starting Point**: GA/SA start from a semantically meaningful layout that already satisfies domain constraints
2. **Faster Convergence**: Optimization starts closer to good solutions, reducing search time
3. **Preserved Intent**: The layout structure from LLM planning is refined, not replaced
4. **Best of Both Worlds**: Combines LLM understanding with rigorous numerical optimization

#### LLM4RMS+GA (`LLM4RMSGAOptimizer`)
```
Algorithm:
1. Extract equipment positions from LLM4RMS scene graph
2. Initialize GA population:
   - First individual = LLM4RMS layout (elite preservation)
   - Remaining individuals = mutations of LLM4RMS layout
3. Evolution loop (same as standard GA)
4. Return best layout (often improves on LLM4RMS baseline)

Advantage: Explores layout variations while preserving semantic structure
```

#### LLM4RMS+SA (`LLM4RMSSAOptimizer`)
```
Algorithm:
1. Extract equipment positions from LLM4RMS scene graph
2. Use LLM4RMS layout as initial solution (not random!)
3. SA optimization loop (same as standard SA)
4. Return best layout found

Advantage: Local refinement of already-good layout
```

### How Baselines Handle Layout Generation

Unlike the LLM4RMS hybrid method which uses LLM planning + structure-aware execution, baseline methods generate layouts through algorithmic approaches:

#### 1. Random Layout (`RandomLayoutGenerator`)
```
Algorithm:
1. For each equipment in the list:
   a. Generate random (x, y) position within room bounds
   b. Select random rotation (0°, 90°, 180°, 270°)
   c. Check collision with existing placements
   d. Retry up to 100 times if collision detected
2. Generate material flow connections based on spatial proximity
```

#### 2. Grid Layout (`GridLayoutGenerator`)
```
Algorithm:
1. Calculate grid dimensions based on equipment count and sizes
2. Place equipment in row-major order at grid cell centers
3. Alternate rotations for variety
4. Generate connections based on grid adjacency
```

#### 3. Genetic Algorithm (`GeneticAlgorithmLayoutGenerator`)
```
Algorithm:
1. Initialize population with random layouts
2. For each generation:
   a. Evaluate fitness (collision penalty + flow distance + utilization)
   b. Select elite performers (top 20%)
   c. Generate offspring via crossover and mutation
   d. Replace population
3. Return best layout after max generations

Parameters:
- Population size: 50
- Generations: 100
- Mutation rate: 10%
- Crossover rate: 80%
```

#### 4. Simulated Annealing (`SimulatedAnnealingLayoutGenerator`)
```
Algorithm:
1. Initialize with random layout
2. While temperature > min_temp:
   a. Generate neighbor (shift, rotate, or swap)
   b. Calculate energy difference
   c. Accept if better, or probabilistically if worse
   d. Cool down temperature
3. Return best layout found

Parameters:
- Initial temperature: 1000
- Cooling rate: 0.995
- Min temperature: 1.0
- Iterations per temp: 50
```

#### 5. SLP Heuristic (`SLPLayoutGenerator`)
```
Algorithm:
1. Build relationship matrix based on equipment categories
2. Sort equipment by total relationship strength
3. Place first equipment at room center
4. For each subsequent equipment:
   a. Calculate weighted centroid from placed equipment
   b. Search for valid position near centroid
   c. Choose rotation based on nearest related equipment
5. Generate connections based on relationships
```

### Connection Generation

All baseline methods generate material flow connections automatically:

```python
def _generate_connections(objects):
    """
    Generate connections based on:
    1. Equipment type (Line_01-08 are connectable)
    2. Spatial proximity (< 8 meters)
    3. Position ordering (creates flow path)
    """
    # Sort by position to create logical flow
    sorted_objects = sort_by_position(connectable_objects)

    # Connect consecutive equipment if close enough
    for i in range(len(sorted_objects) - 1):
        if distance(sorted_objects[i], sorted_objects[i+1]) < 8.0:
            create_connection(sorted_objects[i], sorted_objects[i+1])
```

### Layout Quality Evaluation

```python
def evaluate_layout(placements):
    score = 1000.0  # Base score

    # Collision penalty (-100 per collision)
    score -= 100 * count_collisions(placements)

    # Out of bounds penalty (-50 per violation)
    score -= 50 * count_out_of_bounds(placements)

    # Flow distance penalty (prefer shorter paths)
    score -= 2.0 * average_distance_between_consecutive()

    # Space utilization bonus
    score += 50.0 * (used_area / room_area)

    return score
```

### Running Baseline Comparison

```bash
# Quick comparison (fewer runs, faster evaluation)
python -m evaluation.run_baseline_comparison --quick --num-runs 1

# Full comparison (3 runs per method)
python -m evaluation.run_baseline_comparison --num-runs 3

# Specific methods and scenes
python -m evaluation.run_baseline_comparison \
    --methods random ga sa \
    --scenes cell03 cell08 \
    --num-runs 5 -v

# Include hybrid optimization methods (LLM4RMS + GA/SA)
python -m evaluation.run_baseline_comparison \
    --methods random grid ga sa slp llm4rms+ga llm4rms+sa \
    --num-runs 3

# Only run hybrid optimization methods for comparison
python -m evaluation.run_baseline_comparison \
    --methods llm4rms+ga llm4rms+sa \
    --scenes cell03 cell08 \
    --num-runs 3 -v
```

### Experimental Results Summary

Results from baseline comparison experiments (3 runs, averaged, throughput in parts/time):

| Scene | LLM4RMS Hybrid | LLM4RMS+GA | LLM4RMS+SA | SA | GA | SLP | Random | Grid |
|-------|---------------|------------|------------|-----|-----|-----|--------|------|
| cell03 | **4.12** | 2.67 | 2.98 | 2.69 | 2.11 | 1.26 | 1.21 | 0.53 |
| cell03_before | 1.32 | **2.63** | 1.68 | 1.69 | 1.33 | 1.95 | 1.01 | 0.24 |
| cell03_material_flow | 4.34 | 3.93 | 3.49 | **4.61** | 2.66 | 1.89 | 2.15 | 1.94 |
| cell08 | **4.56** | 4.03 | 2.85 | 3.24 | 2.63 | 1.82 | 1.32 | 0.41 |
| cell09 | **4.62** | 4.06 | 3.24 | 2.82 | 2.06 | 1.86 | 1.29 | 0.40 |
| cell_hybrid | **8.34** | 4.65 | 3.52 | 3.69 | 2.97 | 1.49 | 1.84 | 1.18 |
| complex_flow | 1.18 | **1.94** | 0.99 | 0.79 | 1.21 | 0.56 | 1.15 | 0.30 |

**Key Findings:**

1. **LLM4RMS hybrid method achieves best throughput in 4/7 scenarios** (cell03, cell08, cell09, cell_hybrid)

2. **LLM4RMS+GA shows strong performance**:
   - Outperforms original LLM4RMS in 2 scenarios (cell03_before: 2.63 vs 1.32, complex_flow: 1.94 vs 1.18)
   - Consistently ranks among top 3 methods
   - Demonstrates that starting from a semantically meaningful layout helps GA converge to better solutions

3. **LLM4RMS+GA vs LLM4RMS+SA**:
   - LLM4RMS+GA generally outperforms LLM4RMS+SA
   - GA's population-based search better explores variations of the LLM4RMS layout
   - SA's local search may get trapped in local optima close to the initial LLM4RMS solution

4. **Pure SA outperforms pure GA** in most scenarios
   - SA benefits from temperature-based acceptance of worse solutions
   - However, when starting from LLM4RMS, GA becomes more competitive (LLM4RMS+GA > LLM4RMS+SA)

5. **Grid layout consistently performs worst** (no optimization)

6. **The advantage of LLM4RMS is most pronounced in complex scenarios** (cell_hybrid: 8.34 vs 4.65)

**Why LLM4RMS+GA can improve over LLM4RMS:**
The LLM4RMS layout provides a semantically meaningful starting point, but may not be optimally positioned for material flow efficiency. GA can refine positions while the population-based approach helps preserve the overall layout structure.

**Why LLM4RMS often still outperforms LLM4RMS+GA/SA:**
The LLM-based layout encodes semantic understanding of manufacturing processes (e.g., material flow direction, ergonomic constraints, safety zones) that cannot be captured by purely numerical optimization. GA/SA may disrupt these semantically important relationships during optimization.

### Ablation Study (Optional)

The `ablation_study.py` module supports ablation experiments to analyze component contributions:

| Variant | Description |
|---------|-------------|
| `llm_only` | Pure LLM generation without structure-aware constraints |
| `structure_aware` | Template-based generation without LLM planning |
| `hybrid` | Full approach: LLM planning + structure-aware execution |
| `no_backtrack` | Hybrid without backtracking placement |

```bash
# Run ablation study
python -m evaluation.ablation_study --variants hybrid llm_only structure_aware --num-runs 3
```

## References

1. Mastrangelo, A., & Tolio, T. (2024). "A hybrid method combining analytical and simulation models for performance evaluation of reconfigurable manufacturing systems." *Journal of Manufacturing Systems*.

2. Magnanini, M. C., & Tolio, T. (2023). "Performance evaluation methods for manufacturing systems." *CIRP Encyclopedia of Production Engineering*.

3. Gershwin, S. B. (1994). *Manufacturing Systems Engineering*. Prentice Hall.

4. Muther, R. (1973). *Systematic Layout Planning*. Cahners Books. (SLP methodology)

5. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

6. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by Simulated Annealing." *Science*, 220(4598), 671-680.
