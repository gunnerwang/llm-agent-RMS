from LLM4RMS import LLM4RMS
import json
import os
import shutil
import time
import traceback

# Configuration
INTERACTIVE_MODE = False  # Set to True to enable interactive workflow by default
                         # When True: scenarios will pause for user feedback during planning
                         # When False: scenarios will run automatically without interruption

REASONING_EFFORT = "none"  # For Gemini models: "none", "low", "medium", "high"

SIMULATE_PROCESS_FLOW = True
SIMULATION_TRIALS = 200
SIMULATION_MAX_FLOW_GAP = 6.0
SIMULATION_VERBOSE = False

# Hybrid Performance Evaluation
USE_HYBRID_EVALUATION = True
HYBRID_MAX_ITERATIONS = 30
HYBRID_SIMULATION_TIME = 5000.0
HYBRID_VERBOSE = False


def _print_simulation_report(report):
    print("\n--- Process Flow Simulation ---")
    print(f"Trials: {report.trials}")
    print(f"Success rate: {report.success_rate:.1%}")
    if report.average_cycle_time:
        print(f"Average cycle time: {report.average_cycle_time:.2f} s")
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
            print(f"  {stage}: avg={avg_text}, util={util_text}, equipment={equip}")

    if report.missing_stages:
        print(f"\nMissing stages: {', '.join(report.missing_stages)}")
    if report.disconnected_pairs:
        formatted = ", ".join([f"{a}->{b}" for a, b in report.disconnected_pairs])
        print(f"\nDisconnected stage pairs: {formatted}")
    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  {note}")
    print("-" * 60)


def _print_hybrid_report(report):
    """Print hybrid performance evaluation report."""
    print("\n--- Hybrid Performance Evaluation ---")
    print(f"Iterations: {report.iterations}")
    print(f"Converged: {'Yes' if report.converged else 'No'}")
    print(f"Throughput: {report.throughput:.4f} parts/time unit")
    if report.throughput_std > 0:
        print(f"  Std Dev: {report.throughput_std:.4f}")
        print(f"  95% CI: [{report.throughput_ci[0]:.4f}, {report.throughput_ci[1]:.4f}]")
    print(f"Work-in-Progress: {report.work_in_progress:.2f} parts")
    print(f"Avg Starvation: {report.avg_starvation_prob:.2%}")
    print(f"Avg Blocking: {report.avg_blocking_prob:.2%}")

    if report.stage_metrics:
        print("\nStage Breakdown:")
        for stage, metrics in report.stage_metrics.items():
            th = metrics.get("throughput", 0)
            equip = metrics.get("equipment_count", 0)
            print(f"  {stage}: throughput={th:.4f}, equipment={equip}")

    if report.bottleneck_stage:
        print(f"\nBottleneck: {report.bottleneck_stage}")
    if report.efficiency_index is not None:
        print(f"Efficiency Index: {report.efficiency_index:.2%}")

    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  {note}")
    print("-" * 60)


def _format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def run_test_scenario(scenario_name, no_of_objects, user_input, room_dimensions,
                      model_name="gpt-4o", verbose=True, initial_design_path=None,
                      structure_aware=False, hybrid_reconfig=False, use_as_template=False):
    """Run a single manufacturing-layout scenario and save results."""
    scenario_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Running Test Scenario: {scenario_name}")
    print(f"{'='*60}")
    print(f"Objects: {no_of_objects}")
    if room_dimensions and len(room_dimensions) == 3:
        print(f"Workspace: {room_dimensions[0]}x{room_dimensions[1]}x{room_dimensions[2]}m")
    else:
        print("Workspace: auto-sized")
    print(f"Model: {model_name}")
    if REASONING_EFFORT != "none":
        print(f"Reasoning effort: {REASONING_EFFORT}")
    if hybrid_reconfig:
        if use_as_template:
            print(f"Reconfiguration: hybrid (template mode)")
        else:
            print(f"Reconfiguration: hybrid (LLM + structure-aware)")
    elif structure_aware:
        print(f"Reconfiguration: structure-aware")
    print(f"Input: {user_input}")
    print(f"{'='*60}")

    scenario_ready = False

    try:
        llm4rms = LLM4RMS(
            no_of_objects=no_of_objects,
            user_input=user_input,
            room_dimensions=room_dimensions,
            model_name=model_name,
            initial_design=initial_design_path,
            multiagent_reconfig=True,
            reasoning_effort=REASONING_EFFORT,
            structure_aware_reconfig=structure_aware,
            hybrid_reconfig=hybrid_reconfig,
            use_as_template=use_as_template,
        )

        if INTERACTIVE_MODE:
            success = llm4rms.interactive_design_workflow()
            if not success:
                print(f"Interactive workflow failed for '{scenario_name}'")
                return False
        else:
            if initial_design_path:
                print(f"Starting from initial design: {initial_design_path}")
            llm4rms.create_initial_design()
            llm4rms.correct_design()
            llm4rms.refine_design(verbose=verbose)
            llm4rms.create_object_clusters(verbose=False, relaxation_factor=1.3)
            llm4rms.backtrack(verbose=verbose, auto_relax=True)

        if getattr(llm4rms, "auto_room_dimensions", False):
            dims = llm4rms.room_dimensions
            print(f"Auto-sized workspace to: {dims[0]}x{dims[1]}x{dims[2]}m")

        output_filename = f"scenes/scene_graph_{scenario_name.lower().replace(' ', '_')}.json"
        llm4rms.to_json(output_filename)

        base_name = f"scene_graph_{scenario_name.lower().replace(' ', '_')}"
        rationale_filename = llm4rms.save_rationale_summary(f"scenes/{base_name}_rationale.md")

        design_duration = time.time() - scenario_start_time
        print(f"Scenario '{scenario_name}' completed successfully!")
        print(f"Design time: {_format_duration(design_duration)}")
        print(f"Scene graph saved to: {output_filename}")
        print(f"Design rationale saved to: {rationale_filename}")
        scenario_ready = True

    except Exception as e:
        traceback.print_exc()
        print(f"Error in scenario '{scenario_name}': {str(e)}")
        design_duration = time.time() - scenario_start_time
        print(f"Time elapsed before error: {_format_duration(design_duration)}")
        return False

    if SIMULATE_PROCESS_FLOW and scenario_ready:
        try:
            report = llm4rms.simulate_process_flow(
                trials=SIMULATION_TRIALS,
                max_flow_gap=SIMULATION_MAX_FLOW_GAP,
                verbose=SIMULATION_VERBOSE,
            )
            _print_simulation_report(report)
        except Exception as sim_err:
            traceback.print_exc()
            print(f"Simulation failed: {sim_err}")

    if USE_HYBRID_EVALUATION and scenario_ready:
        try:
            hybrid_report = llm4rms.evaluate_hybrid_performance(
                max_iterations=HYBRID_MAX_ITERATIONS,
                simulation_time=HYBRID_SIMULATION_TIME,
                verbose=HYBRID_VERBOSE,
            )
            _print_hybrid_report(hybrid_report)
        except Exception as hybrid_err:
            traceback.print_exc()
            print(f"Hybrid evaluation failed: {hybrid_err}")

    total_duration = time.time() - scenario_start_time
    print(f"\nTotal scenario time: {_format_duration(total_duration)}")

    return True


def display_scenarios():
    """Define available test scenarios."""
    scenarios = [
        {
            "name": "Simple Assembly Line",
            "objects": 8,
            "input": (
                "Design a straight-line production flow:\n"
                "- Entry conveyor at the south wall\n"
                "- Two buffer conveyors in sequence\n"
                "- Two processing conveyors between buffers\n"
                "- An inspection station inline on the conveyor\n"
                "Keep everything on one straight lane flowing south to north."
            ),
            "dimensions": [],
        },
        {
            "name": "Collaborative Robot Assembly Cell",
            "objects": 12,
            "input": (
                "Design a collaborative robot assembly cell:\n"
                "- A UR10e robot in the center for pick-and-place\n"
                "- Two conveyor belts: one inbound, one outbound\n"
                "- A workbench for manual inspection\n"
                "- Safety barriers around the robot workspace\n"
                "- Shelving for parts storage near the workbench\n"
                "Ensure safe clearance between robot workspace and operator areas."
            ),
            "dimensions": [10.0, 8.0, 3.0],
        },
        {
            "name": "Multi-Robot Production Cell",
            "objects": 16,
            "input": (
                "Design a production cell with two robot stations:\n"
                "- ABB IRB2600 robot for material handling on the north side\n"
                "- UR10e robot for assembly on the south side\n"
                "- Conveyor loop connecting both stations\n"
                "- Buffer conveyors between stations for work-in-progress\n"
                "- Inspection camera at the outbound conveyor\n"
                "- Safety perimeter around the ABB robot area\n"
                "Material flows clockwise through the cell."
            ),
            "dimensions": [14.0, 10.0, 4.0],
        },
        {
            "name": "Quality Inspection Station",
            "objects": 10,
            "input": (
                "Design a quality inspection workstation:\n"
                "- Inbound conveyor from the west wall\n"
                "- Camera inspection station inline on the conveyor\n"
                "- Manual inspection workbench adjacent to the conveyor\n"
                "- Outbound conveyor to the east wall\n"
                "- Reject bin and shelving for test equipment\n"
                "- Good lighting around inspection areas."
            ),
            "dimensions": [8.0, 6.0, 3.0],
        },
        {
            "name": "Warehouse Pick-and-Pack",
            "objects": 15,
            "input": (
                "Design a warehouse pick-and-pack area:\n"
                "- Three rows of industrial shelving for storage\n"
                "- A central conveyor belt running north to south\n"
                "- Packing workbench at the south end\n"
                "- Pallet staging area near the loading dock (north wall)\n"
                "- A scissor lift for accessing high shelves\n"
                "- Clear aisles (min 2m) between shelving rows for forklift access."
            ),
            "dimensions": [16.0, 12.0, 5.0],
        },
    ]

    return scenarios


def select_model():
    """Model selection interface."""
    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "deepseek-chat",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    print("\nModel Selection")
    print(f"{'='*60}")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    print(f"{'='*60}")

    try:
        choice = int(input(f"Select model (1-{len(models)}, default=1): ") or "1")
        if 1 <= choice <= len(models):
            return models[choice - 1]
        else:
            print("Invalid choice. Using default gpt-4o.")
            return "gpt-4o"
    except ValueError:
        print("Invalid input. Using default gpt-4o.")
        return "gpt-4o"


def main():
    """Interactive test scenario selection."""

    scenarios = display_scenarios()

    print("LLM4RMS Test Scenarios")

    # Model selection
    selected_model = select_model()
    print(f"Selected model: {selected_model}")

    print("\nSelect a scenario to test:")
    print(f"{'='*60}")

    for i, scenario in enumerate(scenarios, 1):
        dims = scenario["dimensions"]
        if dims and len(dims) == 3:
            workspace_text = f"{dims[0]}x{dims[1]}x{dims[2]}m"
        else:
            workspace_text = "auto-sized"
        print(f"{i:2d}. {scenario['name']}")
        print(f"    Objects: {scenario['objects']}, Workspace: {workspace_text}")
        print(f"    {scenario['input'][:80]}...")
        print()

    print(f"{len(scenarios)+1:2d}. Run ALL scenarios (batch test)")
    print(f"{'='*60}")
    if INTERACTIVE_MODE:
        print("Interactive Mode: ENABLED (scenarios will include user feedback)")
    else:
        print("Standard Mode: ENABLED (scenarios will run automatically)")

    try:
        choice = int(input("Enter your choice (1-{0}): ".format(len(scenarios)+1)))

        if choice == len(scenarios) + 1:
            # Run all scenarios
            print("\nRunning all scenarios...")
            batch_start_time = time.time()
            successful_scenarios = 0
            scenario_times = []

            for i, scenario in enumerate(scenarios, 1):
                print(f"\nScenario {i}/{len(scenarios)}")
                scenario_start = time.time()
                success = run_test_scenario(
                    scenario_name=scenario["name"],
                    no_of_objects=scenario["objects"],
                    user_input=scenario["input"],
                    room_dimensions=scenario["dimensions"],
                    model_name=selected_model,
                    verbose=False,
                )
                scenario_duration = time.time() - scenario_start
                scenario_times.append((scenario["name"], scenario_duration, success))
                if success:
                    successful_scenarios += 1

            batch_duration = time.time() - batch_start_time

            print(f"\n{'='*60}")
            print(f"BATCH TEST SUMMARY")
            print(f"{'='*60}")
            print(f"Successful: {successful_scenarios}/{len(scenarios)}")
            print(f"Success rate: {(successful_scenarios/len(scenarios))*100:.1f}%")
            print(f"\nTiming Summary:")
            for name, duration, success in scenario_times:
                status = "OK" if success else "FAIL"
                print(f"  [{status}] {name}: {_format_duration(duration)}")
            print(f"\n  Total batch time: {_format_duration(batch_duration)}")

        elif 1 <= choice <= len(scenarios):
            selected_scenario = scenarios[choice - 1]
            print(f"\nRunning: {selected_scenario['name']}")

            success = run_test_scenario(
                scenario_name=selected_scenario["name"],
                no_of_objects=selected_scenario["objects"],
                user_input=selected_scenario["input"],
                room_dimensions=selected_scenario["dimensions"],
                model_name=selected_model,
                verbose=True,
            )

            if success:
                print("\nTest completed successfully!")
            else:
                print("\nTest failed. Check error messages above.")

        else:
            print("Invalid choice. Please run the script again.")

    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        traceback.print_exc()
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
