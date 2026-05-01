"""
Standalone visualization script for BubbleFence results.

Usage:
    python visualize_bubblefence.py --output_dir test_output           # Standalone: heatmap + scatter
    python visualize_bubblefence.py --all-trajectories --output_dir test_output  # + trajectory curves
    python visualize_bubblefence.py --all-trajectories --no-points --output_dir test_output  # no scatter
    python visualize_bubblefence.py --all-trajectories --thumbnails --output_dir test_output  # + thumbnails
    python visualize_bubblefence.py --radius-circles --no-heatmaps --output_dir test_output  # radius circles only
    python visualize_bubblefence.py --detailed --output_dir test_output   # 4-panel view
    python visualize_bubblefence.py --trajectory frames --output_dir test_output  # single-folder temporal
    python visualize_bubblefence.py --3d --output_dir test_output        # 3D standalone plot
    python visualize_bubblefence.py --3d --all-trajectories --output_dir test_output  # 3D + trajectories
    python visualize_bubblefence.py --3d --clean --output_dir test_output             # 3D without grid/labels
"""

import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add bubblefence to path
sys.path.insert(0, str(Path(__file__).parent))

from bubblefence.visualization import (
    visualize_from_persistence,
    visualize_standalone_from_persistence,
    visualize_standalone_3d_from_persistence,
    visualize_trajectory_from_persistence,
    visualize_all_trajectories_from_persistence,
    get_available_runs,
    print_dataset_stats,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize BubbleFence Results")

    # Mode selection
    parser.add_argument('--detailed', action='store_true',
                       help='Show the 4-panel view (default: standalone single panel)')
    parser.add_argument('--summary', action='store_true',
                       help='Show the 2-panel view (splits + heatmap only)')
    parser.add_argument('--stats', action='store_true',
                       help='Print dataset statistics to stdout (no plot)')
    parser.add_argument('--trajectory', metavar='FOLDER',
                       help='Plot temporal color-gradient for a single folder '
                            '(e.g. --trajectory frames)')

    # Layer toggles
    parser.add_argument('--all-trajectories', action='store_true',
                       help='Show smooth trajectory curves for all run_ids')
    parser.add_argument('--heatmaps', action='store_true', default=True,
                       help='Show KDE heatmap of bubble interiors (default)')
    parser.add_argument('--no-heatmaps', action='store_false', dest='heatmaps',
                       help='Disable heatmaps, show convex hull outlines instead (4-panel)')
    parser.add_argument('--no-points', action='store_true',
                       help='Hide scatter points (standalone mode only)')
    parser.add_argument('--thumbnails', action='store_true',
                       help='Overlay image thumbnails along trajectories '
                            '(requires --all-trajectories)')
    parser.add_argument('--thumbnail-interval', type=float, default=0.6,
                       help='Fraction interval for thumbnail sampling (default: 0.6)')
    parser.add_argument('--radius-circles', action='store_true',
                       help='Draw average-radius circles around each anchor')
    parser.add_argument('--anchor-thumbnails', type=int, nargs='?', const=0, default=None,
                       help='Show anchor-based thumbnails. Pass a number to auto-select '
                            'that many spread-out anchors (e.g. --anchor-thumbnails 5). '
                            'Without a number, shows all anchors.')
    parser.add_argument('--anchor-thumbs-per-bubble', type=int, default=1,
                       help='Number of representative thumbnails per anchor (default: 1)')
    parser.add_argument('--anchor-thumb-pixels', type=int, default=80,
                       help='Size of anchor thumbnails in pixels (default: 80)')
    parser.add_argument('--point-alpha', type=float, default=0.3,
                       help='Opacity for scatter points (default: 0.3)')

    # 3D options
    parser.add_argument('--3d', action='store_true', dest='three_d',
                       help='Use interactive 3D plotly visualization')
    parser.add_argument('--clean', action='store_true',
                       help='Strip gridlines, axis labels, ticks from 3D plot')

    # Common options
    parser.add_argument('--run', help='Specific run ID to visualize (default: all runs)')
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive plot with hover (can be laggy)')
    parser.add_argument('--method', default='PCA', choices=['PCA', 'TSNE'],
                       help='Dimensionality reduction method')
    parser.add_argument('--save', default='bubblefence_plot.png',
                       help='Output filename for the plot')
    parser.add_argument('--smoothing', type=float, default=0.1,
                       help='Smoothing factor for trajectory splines (default: 0.1)')
    parser.add_argument('--output_dir', default='.',
                       help='Directory containing BubbleFence outputs: anchors_state.pkl, '
                            'embeddings/, full_dataset.csv (default: current directory)')
    parser.add_argument('--data-dir', default='test_data',
                       help='Directory containing image folders for thumbnails '
                            '(default: test_data)')

    # Legacy single-trajectory options
    parser.add_argument('--show-connections', action='store_true',
                       help='Draw dotted arrows between points (--trajectory mode)')
    parser.add_argument('--smooth-path', action='store_true',
                       help='Overlay smoothed spline curve (--trajectory mode)')

    args = parser.parse_args()
    output_dir = args.output_dir

    try:
        # --- Single-folder trajectory mode ---
        if args.trajectory:
            save_path = args.save if args.save != 'bubblefence_plot.png' \
                else f'trajectory_{args.trajectory}.png'
            save_path = str(Path(output_dir) / save_path)
            print(f"Creating trajectory visualization for folder '{args.trajectory}'...")
            fig = visualize_trajectory_from_persistence(
                data_root=output_dir,
                folder_name=args.trajectory,
                reduction_method=args.method,
                figsize=(10, 8),
                save_path=save_path,
                show_connections=args.show_connections,
                smooth_path=args.smooth_path
            )
            plt.show()
            print(f"Plot saved as '{save_path}'")
            return

        # --- Stats-only mode ---
        if args.stats:
            print_dataset_stats(output_dir)
            return

        # --- 4-panel or 2-panel detailed mode ---
        if args.detailed or args.summary:
            num_panels = 2 if args.summary else 4
            print(f"Creating {num_panels}-panel visualization...")
            runs = get_available_runs(output_dir)
            print(f"Available runs: {runs}")

            if args.run and args.run not in runs:
                print(f"Error: Run '{args.run}' not found. Available runs: {runs}")
                return

            save_path = str(Path(output_dir) / args.save)
            fig_size = (20, 8) if args.summary else (14, 10)
            fig = visualize_from_persistence(
                data_root=output_dir,
                run_filter=args.run,
                interactive=args.interactive,
                reduction_method=args.method,
                show_heatmap=args.heatmaps,
                figsize=fig_size,
                save_path=save_path,
                show_all_trajectories=args.all_trajectories,
                smoothing=args.smoothing,
                num_panels=num_panels
            )
            plt.show()
            print(f"Plot saved as '{save_path}'")
            return

        # --- Standalone mode (default) ---
        if args.three_d:
            # Default to .html for interactive plotly output
            save_name = args.save
            if save_name == 'bubblefence_plot.png':
                save_name = 'bubblefence_3d.html'
            save_path = str(Path(output_dir) / save_name)

            print("Creating interactive 3D visualization (plotly)...")
            fig = visualize_standalone_3d_from_persistence(
                data_root=output_dir,
                reduction_method=args.method,
                save_path=save_path,
                show_heatmap=args.heatmaps,
                show_trajectories=args.all_trajectories,
                show_points=not args.no_points,
                smoothing=args.smoothing,
                clean=args.clean,
            )
            print(f"Plot saved as '{save_path}'")
            return

        print("Creating standalone visualization...")
        save_path = str(Path(output_dir) / args.save)
        fig = visualize_standalone_from_persistence(
            data_root=output_dir,
            reduction_method=args.method,
            figsize=(14, 10),
            save_path=save_path,
            show_heatmap=args.heatmaps,
            show_trajectories=args.all_trajectories,
            show_points=not args.no_points,
            show_thumbnails=args.thumbnails,
            show_radius_circles=args.radius_circles,
            show_anchor_thumbnails=args.anchor_thumbnails is not None,
            num_anchor_thumbnails=args.anchor_thumbnails if args.anchor_thumbnails else None,
            anchor_thumbs_per_bubble=args.anchor_thumbs_per_bubble,
            anchor_thumb_pixels=args.anchor_thumb_pixels,
            thumbnail_interval=args.thumbnail_interval,
            smoothing=args.smoothing,
            interactive=args.interactive,
            data_dir=args.data_dir,
            point_alpha=args.point_alpha
        )
        plt.show()
        print(f"Plot saved as '{save_path}'")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
