#!/usr/bin/env python3
"""
Baseline Comparison Script

Compares baseline layout methods against LLM4RMS hybrid method.
LLM4RMS results are loaded from existing evaluation results (metrics_summary.csv).
Baseline methods generate new layouts and evaluate them.
"""

import argparse
import json
import os
import sys
import time
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.baselines import (
    BaselineMethod,
    LayoutConfig,
    generate_baseline_scene_graph,
    extract_equipment_from_scene_graph,
)
from evaluation.hybrid_simulation import HybridPerformanceEvaluator


def is_hybrid_method(method: BaselineMethod) -> bool:
    """Check if method requires LLM4RMS scene graph as initial solution"""
    return method in (BaselineMethod.LLM4RMS_GA, BaselineMethod.LLM4RMS_SA)


@dataclass
class ComparisonResult:
    """Result from a comparison experiment"""
    scene_name: str
    method: str
    throughput: float
    throughput_std: float
    wip: float
    starvation: float
    blocking: float
    efficiency: float
    generation_time: float
    num_runs: int


def load_llm4rms_results(results_dir: str = "evaluation/results") -> Dict[str, Dict]:
    """Load existing LLM4RMS evaluation results from CSV"""
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    results = {}

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return results

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle the actual CSV format: scene column contains scene_graph_XXX
            scene_full = row.get("scene", "")
            if not scene_full:
                continue

            # Extract scene name (remove scene_graph_ prefix)
            scene_name = scene_full.replace("scene_graph_", "")

            results[scene_name] = {
                "throughput": float(row.get("TH", 0)),
                "throughput_std": float(row.get("TH_std", 0)),
                "wip": float(row.get("WIP", 0)),
                "starvation": float(row.get("Pi_S", 0)),
                "blocking": float(row.get("Pi_B", 0)),
                "efficiency": float(row.get("efficiency_index", 0)),
            }

    return results


def run_baseline_experiment(
    scene_name: str,
    scene_graph_path: str,
    method: BaselineMethod,
    num_runs: int = 3,
    seed: int = 42,
    max_iterations: int = 5,
    simulation_time: float = 1000.0,
    verbose: bool = False,
) -> Optional[ComparisonResult]:
    """Run a single baseline experiment"""

    # Load reference scene graph
    if not os.path.exists(scene_graph_path):
        print(f"  Scene not found: {scene_graph_path}")
        return None

    with open(scene_graph_path, "r") as f:
        ref_scene = json.load(f)

    # Extract equipment and room dimensions
    equipment = extract_equipment_from_scene_graph(ref_scene)

    # Get room dimensions from scene
    room_dims = (14.0, 14.0, 4.5)  # default
    for obj in ref_scene:
        if obj.get("object_id") == "middle of the room":
            size = obj.get("size_in_meters", {})
            room_dims = (size.get("length", 14.0), size.get("width", 14.0), 4.5)
            break

    throughputs = []
    wips = []
    starvations = []
    blockings = []
    efficiencies = []
    gen_times = []

    for run_id in range(num_runs):
        run_seed = seed + run_id

        if verbose:
            print(f"    Run {run_id + 1}/{num_runs}...", end=" ", flush=True)

        try:
            # Generate layout
            start_time = time.time()

            # For hybrid methods (LLM4RMS+GA, LLM4RMS+SA), pass the LLM4RMS scene graph
            if is_hybrid_method(method):
                scene_graph = generate_baseline_scene_graph(
                    method=method,
                    equipment_list=equipment,
                    room_dimensions=room_dims,
                    seed=run_seed,
                    initial_scene_graph=ref_scene,  # Use LLM4RMS layout as starting point
                )
            else:
                scene_graph = generate_baseline_scene_graph(
                    method=method,
                    equipment_list=equipment,
                    room_dimensions=room_dims,
                    seed=run_seed,
                )
            gen_time = time.time() - start_time
            gen_times.append(gen_time)

            # Evaluate
            evaluator = HybridPerformanceEvaluator(scene_graph=scene_graph, seed=run_seed)
            report = evaluator.run(
                max_iterations=max_iterations,
                simulation_time=simulation_time,
                verbose=False,
            )

            throughputs.append(report.throughput)
            wips.append(report.work_in_progress)
            starvations.append(report.avg_starvation_prob)
            blockings.append(report.avg_blocking_prob)
            efficiencies.append(report.efficiency_index or 0.0)

            if verbose:
                print(f"TH={report.throughput:.2f}, Eff={report.efficiency_index:.0%}")

        except Exception as e:
            if verbose:
                print(f"Error: {e}")

    if not throughputs:
        return None

    return ComparisonResult(
        scene_name=scene_name,
        method=method.value,
        throughput=statistics.mean(throughputs),
        throughput_std=statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
        wip=statistics.mean(wips),
        starvation=statistics.mean(starvations),
        blocking=statistics.mean(blockings),
        efficiency=statistics.mean(efficiencies),
        generation_time=statistics.mean(gen_times),
        num_runs=len(throughputs),
    )


def print_comparison_table(results: List[ComparisonResult], llm4rms_results: Dict[str, Dict]):
    """Print formatted comparison table"""
    print("\n" + "=" * 110)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 110)

    header = f"{'Scene':<25} {'Method':<12} {'Throughput':>14} {'WIP':>10} {'Starv%':>8} {'Block%':>8} {'Efficiency':>12}"
    print(header)
    print("-" * 110)

    # Group by scene
    by_scene = {}
    for r in results:
        if r.scene_name not in by_scene:
            by_scene[r.scene_name] = []
        by_scene[r.scene_name].append(r)

    for scene_name in sorted(by_scene.keys()):
        scene_results = by_scene[scene_name]

        # Print LLM4RMS result first (if available)
        if scene_name in llm4rms_results:
            ir = llm4rms_results[scene_name]
            row = (
                f"{scene_name:<25} "
                f"{'hybrid':<12} "
                f"{ir['throughput']:>10.4f}±{ir['throughput_std']:<3.2f} "
                f"{ir['wip']:>10.2f} "
                f"{ir['starvation']*100:>7.2f}% "
                f"{ir['blocking']*100:>7.2f}% "
                f"{ir['efficiency']*100:>10.2f}%"
            )
            print(row)

        # Print baseline results
        for r in sorted(scene_results, key=lambda x: -x.throughput):
            row = (
                f"{'':<25} "
                f"{r.method:<12} "
                f"{r.throughput:>10.4f}±{r.throughput_std:<3.2f} "
                f"{r.wip:>10.2f} "
                f"{r.starvation*100:>7.2f}% "
                f"{r.blocking*100:>7.2f}% "
                f"{r.efficiency*100:>10.2f}%"
            )
            print(row)

        print("-" * 110)

    print("=" * 110)


def save_results(results: List[ComparisonResult], llm4rms_results: Dict[str, Dict], output_dir: str):
    """Save comparison results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(output_dir, f"baseline_comparison_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["scene_name", "method", "throughput", "throughput_std", "wip",
                     "starvation", "blocking", "efficiency", "generation_time", "num_runs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write LLM4RMS results
        for scene_name, ir in llm4rms_results.items():
            writer.writerow({
                "scene_name": scene_name,
                "method": "hybrid",
                "throughput": ir["throughput"],
                "throughput_std": ir["throughput_std"],
                "wip": ir["wip"],
                "starvation": ir["starvation"],
                "blocking": ir["blocking"],
                "efficiency": ir["efficiency"],
                "generation_time": 0,
                "num_runs": 1,
            })

        # Write baseline results
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiments")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output-dir", default="evaluation/experiment_results",
                       help="Output directory")
    parser.add_argument("-m", "--methods", nargs="+",
                       choices=["random", "grid", "ga", "sa", "slp", "llm4rms+ga", "llm4rms+sa"],
                       default=["random", "grid", "ga", "sa", "slp"],
                       help="Baseline methods to run (llm4rms+ga/sa use LLM4RMS layout as initial solution)")
    parser.add_argument("-s", "--scenes", nargs="+",
                       help="Scenes to test (default: all)")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of runs per method")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (fewer iterations)")

    args = parser.parse_args()

    # Define scenes
    all_scenes = [
        ("cell03", "scenes/scene_graph_cell03.json"),
        ("cell03_before", "scenes/scene_graph_cell03_before.json"),
        ("cell03_material_flow", "scenes/scene_graph_cell03_material_flow.json"),
        ("cell08", "scenes/scene_graph_cell08.json"),
        ("cell09", "scenes/scene_graph_cell09.json"),
        ("cell_hybrid", "scenes/scene_graph_cell_hybrid.json"),
        ("complex_flow_layout", "scenes/scene_graph_complex_flow_layout.json"),
    ]

    # Filter scenes if specified
    if args.scenes:
        all_scenes = [(n, p) for n, p in all_scenes if any(s in n for s in args.scenes)]

    # Map method names to enums
    method_map = {
        "random": BaselineMethod.RANDOM,
        "grid": BaselineMethod.GRID,
        "ga": BaselineMethod.GENETIC_ALGORITHM,
        "sa": BaselineMethod.SIMULATED_ANNEALING,
        "slp": BaselineMethod.SLP,
        "llm4rms+ga": BaselineMethod.LLM4RMS_GA,
        "llm4rms+sa": BaselineMethod.LLM4RMS_SA,
    }
    methods = [method_map[m] for m in args.methods]

    # Evaluation parameters
    max_iter = 3 if args.quick else 5
    sim_time = 500.0 if args.quick else 1000.0

    # Load LLM4RMS results
    print("Loading LLM4RMS results...")
    llm4rms_results = load_llm4rms_results()
    print(f"  Found {len(llm4rms_results)} existing evaluations")

    # Run baseline experiments
    results = []
    total_experiments = len(all_scenes) * len(methods)
    current = 0

    print(f"\nRunning {total_experiments} baseline experiments...")
    print("=" * 60)

    for scene_name, scene_path in all_scenes:
        print(f"\n{scene_name}:")

        for method in methods:
            current += 1
            print(f"  [{current}/{total_experiments}] {method.value}...")

            result = run_baseline_experiment(
                scene_name=scene_name,
                scene_graph_path=scene_path,
                method=method,
                num_runs=args.num_runs,
                seed=args.seed,
                max_iterations=max_iter,
                simulation_time=sim_time,
                verbose=args.verbose,
            )

            if result:
                results.append(result)
                if not args.verbose:
                    print(f"    TH={result.throughput:.2f}, Eff={result.efficiency:.0%}")

    # Print and save results
    print_comparison_table(results, llm4rms_results)
    save_results(results, llm4rms_results, args.output_dir)


if __name__ == "__main__":
    main()
