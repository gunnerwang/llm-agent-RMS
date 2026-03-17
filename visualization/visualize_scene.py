#!/usr/bin/env python3
"""
Script to generate visualization from an existing scene graph JSON file.

Supports two visualization modes:
- Standard: Original OpenCV-based visualization (800x800 PNG)
- Compact: Publication-ready matplotlib visualization (PDF/SVG/PNG)
"""

import json
import sys
import argparse
from core.utils import get_visualization, get_visualization_compact

def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization from scene graph JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard visualization (PNG)
  python visualize_scene.py scene.json

  # Compact PDF for papers (default 3.5" width)
  python visualize_scene.py scene.json -o figure.pdf --compact

  # Compact with custom size and legend on right
  python visualize_scene.py scene.json -o figure.pdf --compact --figsize 4.5 3.0 --legend right

  # Compact with no legend and title
  python visualize_scene.py scene.json -o figure.svg --compact --no-legend --title "Layout A"

  # Show object labels on the boxes
  python visualize_scene.py scene.json -o figure.pdf --compact --labels
        """
    )
    parser.add_argument('input_file', help='Path to the scene graph JSON file')
    parser.add_argument('-o', '--output', default=None,
                       help='Output image path (default: visualization_output.png or .pdf for compact)')

    # Compact mode options
    parser.add_argument('--compact', action='store_true',
                       help='Use compact publication-ready format (matplotlib-based)')
    parser.add_argument('--figsize', nargs=2, type=float, metavar=('W', 'H'),
                       help='Figure size in inches (width height). Default: auto-calculated')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster output (default: 300)')
    parser.add_argument('--legend', choices=['bottom', 'right', 'none'], default='bottom',
                       help='Legend position (default: bottom)')
    parser.add_argument('--no-legend', action='store_true',
                       help='Hide the legend')
    parser.add_argument('--font-size', type=int, default=8,
                       help='Base font size (default: 8)')
    parser.add_argument('--labels', action='store_true',
                       help='Show object ID labels on boxes')
    parser.add_argument('--title', type=str, default=None,
                       help='Optional figure title')

    args = parser.parse_args()

    # Determine default output path
    if args.output is None:
        if args.compact:
            args.output = 'visualization_output.pdf'
        else:
            args.output = 'visualization_output.png'

    try:
        # Load the scene graph JSON
        with open(args.input_file, 'r') as file:
            scene_graph = json.load(file)

        print(f"Loaded scene graph with {len(scene_graph)} objects")

        if args.compact:
            # Use compact visualization
            kwargs = {
                'dpi': args.dpi,
                'show_legend': not args.no_legend,
                'legend_position': args.legend if not args.no_legend else 'none',
                'font_size': args.font_size,
                'show_labels': args.labels,
                'title': args.title,
            }
            if args.figsize:
                kwargs['figsize'] = tuple(args.figsize)

            get_visualization_compact(scene_graph, output_path=args.output, **kwargs)
        else:
            # Use standard visualization
            get_visualization(scene_graph, output_path=args.output)

        print(f"Visualization saved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
