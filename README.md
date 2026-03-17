# LLM4RMS - LLM Agents-Driven Layout Generation for Reconfigurable Manufacturing Systems

This is the official repository for **LLM4RMS**: An AI-powered industrial layout design system that creates optimized workspace configurations using equipment catalogs, robot dimensions, and intelligent placement algorithms.

## 🚀 Features

- **Equipment-Constrained Design**: Uses predefined equipment catalogs to ensure realistic industrial layouts
- **URDF-Based Robot Dimensions**: Automatically calculates robot workspace dimensions from URDF files
- **Multi-Agent Design System**: Interior Designer, Architect, and Engineer agents collaborate on layout creation
- **Interactive Design Workflow**: Deep research-style planning with user confirmation and feedback loops
- **Intelligent Rationale Tracking**: Automated extraction and markdown export of design reasoning
- **Auto-Sizing Workspace**: Intelligent workspace dimension calculation based on selected equipment
- **Intelligent Placement Algorithm**: Backtracking algorithm with spatial reasoning for optimal object placement
- **Unity Integration**: Complete Unity C# scripts for 3D scene instantiation
- **Blender Visualization**: Automated 3D scene generation and rendering
- **Real-time Visualization**: PNG visualizations for design progress tracking
- **User-Provided Layout Seeds**: Start from an existing JSON scene graph and have the agents revise it for new requirements


## 📁 Project Structure

```
LLM4RMS/
├── 📄 LLM4RMS.py                      # Main framework class
├── 📄 test.py                         # Testing and demo scenarios
├── 📄 interactive_example.py          # Interactive workflow example
│
├── 📦 agents/                         # Agent-related modules
│   ├── agents.py                      # Core agent creation and configuration
│   ├── agents_with_tools.py           # Agents using function tools
│   ├── agents_multiagent_reconfig.py  # Multi-agent reconfiguration
│   ├── corrector_agents.py            # Layout correction agents
│   └── refiner_agents.py              # Layout refinement agents
│
├── 📦 reconfiguration/                # Reconfiguration logic
│   ├── structure_aware_reconfig.py    # Structure-aware reconfiguration
│   ├── hybrid_reconfig.py             # Hybrid reconfiguration approach
│   └── reconfiguration_templates.py   # Predefined reconfiguration templates
│
├── 📦 layout/                         # Layout and placement
│   ├── layout_tools.py                # Layout builder and tools
│   ├── layout_command_executor.py     # Command execution for layout
│   └── placement_optimizer.py         # Optimization algorithms for placement
│
├── 📦 visualization/                  # Visualization tools
│   ├── visualize_scene.py             # Scene visualization
│   └── place_in_blender.py            # Blender integration
│
├── 📦 core/                           # Core utilities
│   ├── utils.py                       # General utility functions
│   ├── schemas.py                     # JSON schemas for validation
│   └── constraint_functions.py        # Spatial constraint functions
│
├── 📦 catalog/                        # Equipment catalog and retrieval
│   ├── equipment_catalog.py           # Equipment definitions and constraints
│   └── retrieve.py                    # Equipment retrieval using OpenShape
│
├── 📦 evaluation/                     # Simulation and evaluation
│   ├── simulation.py                  # Process flow simulation
│   └── gpt_v_as_evaluator.py          # Vision-based evaluation
│
├── 📦 integrations/                   # External integrations
│   ├── chats.py                       # Chat system and group chat logic
│   └── gemini_patch.py                # Gemini API patches and fixes
│
├── 📁 Unity/                          # Unity Integration
│   ├── PlaceInUnity.cs                # Unity C# scene placement script
│   ├── Equipment/                     # Unity equipment prefabs
│   ├── Robots/                        # Unity robot prefabs (ABB, KUKA, UR)
│   ├── urdf_dimension_parser.py       # URDF robot dimension calculator
│   ├── update_robot_dimensions.py     # Equipment catalog updater
│   └── robot_dimensions_calculated.txt # Calculated robot dimensions
│
└── 📁 scenes/                         # Scene Outputs
    ├── scene_graph.json               # Generated scene layout
    ├── scene_output.blend             # Blender 3D scene
    ├── visualization_initial.png      # Initial layout visualization
    └── visualization_final.png        # Final layout visualization
```

### Module Descriptions

**Main Programs:**
- **LLM4RMS.py**: Core framework class that orchestrates the entire design workflow
- **test.py**: Main testing script with various scenario demonstrations
- **interactive_example.py**: Example showing interactive user workflow with confirmation steps

**agents/** - Multi-agent system using AutoGen
- **agents.py**: Core functions for creating and configuring agents, model configurations
- **agents_with_tools.py**: Agents that use function tools to build layouts incrementally
- **agents_multiagent_reconfig.py**: Multi-agent collaboration for reconfiguration tasks
- **corrector_agents.py**: Specialized agents for detecting and correcting layout issues
- **refiner_agents.py**: Agents for refining and optimizing existing layouts

**reconfiguration/** - Layout reconfiguration strategies
- **structure_aware_reconfig.py**: Reconfiguration that understands and preserves structural patterns (loops, flows)
- **hybrid_reconfig.py**: Hybrid approach combining multiple reconfiguration strategies
- **reconfiguration_templates.py**: Predefined templates for common reconfiguration tasks

**layout/** - Spatial layout tools
- **layout_tools.py**: LayoutBuilder class and utility functions for layout construction
- **layout_command_executor.py**: Executes high-level commands against the layout builder
- **placement_optimizer.py**: Optimization algorithms for equipment placement

**visualization/** - Scene visualization
- **visualize_scene.py**: Generates 2D visualizations of scene graphs
- **place_in_blender.py**: Exports and renders scenes in Blender (3D)

**core/** - Core utilities and data structures
- **utils.py**: General utility functions for graph operations, spatial calculations, conflict detection
- **schemas.py**: JSON schemas for validating agent outputs
- **constraint_functions.py**: Functions for spatial constraints (on, left of, above, etc.)

**catalog/** - Equipment catalog and retrieval
- **equipment_catalog.py**: Comprehensive catalog of industrial equipment with specifications
- **retrieve.py**: Retrieval system using OpenShape embeddings for finding similar equipment

**evaluation/** - Simulation and evaluation
- **simulation.py**: Process flow simulation to validate throughput and cycle times
- **gpt_v_as_evaluator.py**: Vision-based evaluation using GPT-4V or similar models

**integrations/** - External API integrations
- **chats.py**: Custom group chat implementations for agent coordination
- **gemini_patch.py**: Patches for Gemini API compatibility issues

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Unity 2022.3+ (for Unity integration)
- Blender 3.0+ (for 3D visualization)

### Environment Setup

```bash
# Create conda environment
conda create -n llm4rms python=3.9
conda activate llm4rms

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install MinkowskiEngine for 3D processing
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine

# Install DGL for graph neural networks
conda install -c dglteam/label/cu113 dgl
```

### API Configuration

Copy the template and fill in your API keys:

```bash
cp OAI_CONFIG_LIST.json.template OAI_CONFIG_LIST.json
# Edit OAI_CONFIG_LIST.json and replace YOUR_OPENAI_API_KEY with your actual keys
```

## 🎯 Usage

### 1. Interactive Design Workflow (Recommended)

```python
from LLM4RMS import LLM4RMS

# Initialize the design system
llm4rms = LLM4RMS(
    no_of_objects=6, 
    user_input="Design a collaborative assembly cell where operators and robots work together", 
    room_dimensions=[]  # Auto-sized workspace
)

# Interactive workflow with user confirmation
success = llm4rms.interactive_design_workflow()

if success:
    # Save design with rationale
    llm4rms.save_interactive_design("collaborative_assembly_cell")
```

**Interactive Workflow Steps:**
1. **Initial Planning**: AI generates high-level equipment selection and layout strategy
2. **User Review**: Formatted display of the plan for user evaluation  
3. **Feedback Collection**: User can approve, modify, or add requirements
4. **Detailed Design**: Generates complete layout incorporating user feedback
5. **Automatic Optimization**: Corrections, refinements, and final placement

**User Feedback Examples:**
- **Approve**: `approve`, `yes`, `ok`
- **Modify**: `"Add more safety barriers around the robot workspace"`
- **Constraints**: `"Ensure all equipment is accessible from the main entrance"`

### 2. Traditional Non-Interactive Workflow

```python
from LLM4RMS import LLM4RMS

# Initialize the design system
llm4rms = LLM4RMS(
    no_of_objects=15, 
    user_input="An efficient manufacturing workspace with robots and conveyors", 
    room_dimensions=[10.0, 8.0, 3.0]  # length, width, height in meters
)

# Multi-agent design process
llm4rms.create_initial_design()    # Designer agent creates initial layout
llm4rms.correct_design()           # Corrector agent fixes constraints
llm4rms.refine_design()            # Refiner agent optimizes placement

# Spatial reasoning and placement
llm4rms.create_object_clusters(verbose=False)
llm4rms.backtrack(verbose=True)    # Intelligent placement algorithm

# Export scene graph and rationale
llm4rms.to_json()  # Saves to scenes/scene_graph.json
rationale_file = llm4rms.save_rationale_summary("scenes/design_rationale.md")
```

### 2b. Continue From an Existing JSON Layout

You can seed the system with a previously generated layout stored as a JSON scene graph (e.g., `scenes/scene_graph_gentest.json`). When you call `create_initial_design()`, the usual Process Planner + Layout Engineer agents first discuss how to modify the baseline, and then the Engineer agent emits an updated `objects_in_room` JSON that can keep, delete, or add equipment as needed. If you want to keep the baseline exactly as-is, just skip the call to `create_initial_design()` and move straight to corrections/refinements.

```python
from LLM4RMS import LLM4RMS

llm4rms = LLM4RMS(
    no_of_objects=0,  # not used when an initial design is provided
    user_input="Tighten safety buffers and re-sequence conveyors",
    room_dimensions=[],  # will be inferred from the JSON if walls are included
    initial_design="LLM4RMS/scenes/scene_graph_gentest.json",
)

# Ask the agent team to revise the baseline layout using the new requirements
llm4rms.create_initial_design()
llm4rms.correct_design(verbose=True)
llm4rms.refine_design()
llm4rms.create_object_clusters(verbose=False)
llm4rms.backtrack(verbose=True)
llm4rms.to_json("scenes/scene_graph_gentest_refined.json")
```

Alternatively, call `llm4rms.load_initial_design("path/to/scene_graph.json")` after instantiation. Skip `create_initial_design()` entirely if you want to keep the baseline untouched; or pass `force_regenerate=True` to `create_initial_design()` if you ever want to discard the loaded layout and rebuild from scratch.

### 3. Test Scenarios with Interactive Mode

```bash
python test.py
# Choose from:
# 1-10: Predefined scenarios (non-interactive)
# 11: Run ALL scenarios (batch test)
# 12: Interactive mode (single scenario with user feedback)
```

Scenarios may optionally specify an `initial_design` path in `test.py`, or you can supply one directly when calling `run_test_scenario(..., initial_design_path="path/to/scene_graph.json")` to reuse and optimize an existing layout.

### 4. Custom Interactive Session

```bash
python interactive_example.py
# Choose:
# 1: Run example scenario
# 2: Create custom scenario with user-defined parameters
```

### 5. Design Rationale and Reasoning

LLM4RMS automatically extracts and saves design reasoning in human-readable markdown format:

```python
# The system automatically generates rationale files showing:
# - Process Planner reasoning for equipment selection
# - Layout Engineer spatial arrangement strategies  
# - Correction decisions and conflict resolutions
# - Refinement optimizations and object relationships

# Rationale is saved as: scenes/[scenario_name]_rationale.md
```

**Rationale Content Structure:**
- **Project Metadata**: Timestamp, model, objects, dimensions, user input
- **Initial Design Rationale**: Equipment selection and placement reasoning
- **Design Corrections**: Spatial conflict resolutions and object deletions
- **Design Refinements**: Object relationship optimizations

**Use Cases:**
- Design review and approval processes
- Understanding AI decision-making logic
- Error analysis and debugging
- Quality improvement and prompt optimization
- Client presentations and explanations

### 6. Auto-Sizing Workspace

Set `room_dimensions=[]` for intelligent workspace sizing:

```python
llm4rms = LLM4RMS(
    no_of_objects=5,
    user_input="Compact robot assembly cell",
    room_dimensions=[]  # Automatically sized based on equipment
)
```

The system calculates optimal dimensions considering:
- Equipment footprints and clearances
- Operational aisle requirements
- Safety buffer zones
- Workflow efficiency

### 7. Equipment Catalog Management

The system includes a comprehensive equipment catalog with accurate dimensions:

```python
from catalog.equipment_catalog import get_equipment_list, get_equipment_by_category

# View available equipment
equipment_list = get_equipment_list()
print(f"Available equipment: {len(equipment_list)} items")

# Get equipment by category
robots = get_equipment_by_category("robots")
storage = get_equipment_by_category("storage") 
manufacturing = get_equipment_by_category("manufacturing")
```

**Import Examples:**

```python
# Importing from agents module
from agents.agents import create_agents, get_model_config

# Importing from reconfiguration module
from reconfiguration.structure_aware_reconfig import StructureAwareReconfiguration

# Importing from layout module
from layout.placement_optimizer import PlacementOptimizer

# Importing from core utilities
from core.utils import get_visualization, build_graph
from core.schemas import initial_schema

# Importing from catalog
from catalog.equipment_catalog import get_equipment_list

# Importing from evaluation
from evaluation.simulation import ProcessFlowSimulator

# Importing from integrations
from integrations.chats import GroupChat
```

### 8. Robot Dimension Calculation

Calculate accurate robot dimensions from URDF files:

```bash
cd Unity/
python urdf_dimension_parser.py
```

This scans all robot URDF files and calculates:
- Working envelope dimensions
- Base footprint requirements  
- Maximum reach distances
- Workspace clearances

### 9. 3D Asset Retrieval

```bash
# Clone OpenShape support (if not already done)
git clone https://huggingface.co/OpenShape/openshape-demo-support

# Retrieve 3D models based on scene graph
python -m catalog.retrieve
```

### 10. Blender Visualization

```bash
# Generate 3D scene in Blender
python -m visualization.place_in_blender
```

This creates:
- 3D positioned objects with accurate scales
- Material assignments
- Scene lighting and camera setup
- Exports to `scenes/scene_output.blend`

### 11. Unity Integration

1. Open Unity 2022.3+
2. Import the `Unity/` folder as Assets
3. Add `PlaceInUnity.cs` to a GameObject
4. Configure scene graph JSON path
5. Run to instantiate the scene with equipment prefabs

## 🤖 Supported Equipment

### Industrial Robots (43 models)
All robots are configured for realistic **pick-and-place** and **visual inspection** tasks:

- **ABB**: IRB series (120, 1200, 1600, 2400, 2600, 4400, 4600, 5400, 6600, 6640, 6650, 6700, 7600), CRB collaborative robots
- **KUKA**: KR series (3, 5, 6, 10, 16, 120, 150, 210), LBR iiwa collaborative robots  
- **Universal Robots**: UR5, UR10, UR10e
- **Robotiq**: 2F-85 adaptive grippers

**Robot Capabilities:**
- Material handling and part transfer
- Pick-and-place operations between stations
- Visual inspection assistance
- Collaborative human-robot workflows
- Assembly component positioning

### Manufacturing Equipment
- Conveyor systems (straight, corner, tables)
- Air compressors and power tools
- Scissor lifts and handling equipment
- Manufacturing workstations

### Storage Systems
- Industrial shelving and racks
- Pallets and containers
- Tool storage solutions
- Various box sizes and containers

### Safety Equipment
- Safety railings and barriers
- Emergency equipment
- Lighting systems

## 📊 Evaluation

Evaluate generated layouts using GPT-V:

```bash
python -m evaluation.gpt_v_as_evaluator
```

This provides automated scoring for:
- Layout efficiency
- Safety compliance  
- Workflow optimization
- Space utilization

## 🔧 Configuration

### Room Setup
- Modify room dimensions in the `LLM4RMS` constructor
- Adjust wall configurations in `core/utils.py`
- Customize floor layouts and constraints

### Equipment Constraints  
- Edit `catalog/equipment_catalog.py` to add new equipment
- Update size specifications and categories
- Modify material and style properties

### Placement Parameters
- Adjust backtracking algorithm parameters in `LLM4RMS.py`
- Modify clustering algorithms for object grouping
- Customize spatial relationship constraints

## 📖 Documentation

### User Documentation
- **README.md** (this file): Installation, usage examples, and quick start guide
- **Interactive Examples**: See `interactive_example.py` for workflow demonstrations
- **Test Scenarios**: See `test.py` for comprehensive usage examples

### Code Documentation
All modules include inline documentation. See the Project Structure section above for module organization.

## 📈 Advanced Features

### Multi-Objective Optimization
- Minimize material handling distances
- Maximize workflow efficiency
- Ensure safety clearances
- Optimize equipment utilization

### Constraint Satisfaction
- Equipment compatibility checking
- Spatial interference detection
- Safety regulation compliance
- Power and utility routing

### Export Formats
- Unity scene files (.unity)
- Blender scenes (.blend)
- CAD-compatible formats
- 2D layout drawings (PNG/PDF)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏭 Industrial Applications

- **Manufacturing**: Assembly line design, pick-and-place robot cells, material handling workflows
- **Warehousing**: Storage optimization, robotic picking systems, automated material flow
- **Quality Control**: Visual inspection stations, collaborative QA workflows  
- **Automation**: Robotic workcell configuration for part transfer and positioning
- **Collaborative Workspaces**: Human-robot collaboration zones with safety considerations
- **Safety**: Emergency egress planning, safety zone definition, robot workspace barriers
- **Research**: Industrial layout optimization studies, human-robot interaction design

## Citation

If you use LLM4RMS in your research, please cite:

```bibtex
@article{llm4rms2026,
  title={LLM Agents-Driven Layout Generation for Reconfigurable Manufacturing Systems},
  author={...},
  journal={CIRP Annals},
  year={2026}
}
```

## Links & Resources

- [Equipment Catalog Reference](catalog/equipment_catalog.py)
- [Evaluation Guide](evaluation/README.md)
