"""
Unified CLI for BubbleFence.

Single folder:
    python run_bubblefence.py --folder path/to/images --output_dir results
    python run_bubblefence.py --folder path/to/images --output_dir results --bf-config my_config.yaml

Batch (multiple folders from YAML):
    python run_bubblefence.py --batch config/batch.yaml
    python run_bubblefence.py --batch config/batch.yaml --skip-bf
    python run_bubblefence.py --batch config/batch.yaml --skip-vis
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bubblefence import load_config
from bubblefence.pipeline_runner import (
    run_folder,
    run_batch,
    setup_file_logging,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="BubbleFence - Semantic Fencing of Image Streams",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--folder',
        help='Run BubbleFence on a single image folder',
    )
    mode.add_argument(
        '--batch',
        metavar='YAML',
        help='Run BubbleFence on multiple folders defined in a YAML config',
    )

    # --folder mode options
    parser.add_argument(
        '--output_dir',
        help='Output directory for results (required with --folder)',
    )
    parser.add_argument(
        '--bf-config',
        default=str(Path(__file__).parent / 'config' / 'bubblefence_config.yaml'),
        help='BubbleFence YAML config file (default: config/bubblefence_config.yaml)',
    )

    # --batch mode options
    parser.add_argument('--skip-bf', action='store_true',
                        help='Skip BubbleFence processing (vis only, --batch mode)')
    parser.add_argument('--skip-vis', action='store_true',
                        help='Skip visualization generation (--batch mode)')

    args = parser.parse_args()

    # --- Batch mode ---
    if args.batch:
        if not os.path.exists(args.batch):
            logger.error(f"Batch config not found: {args.batch}")
            sys.exit(1)
        run_batch(args.batch, skip_bf=args.skip_bf, skip_vis=args.skip_vis)
        return

    # --- Single folder mode ---
    if not args.output_dir:
        parser.error("--output_dir is required when using --folder")

    folder = args.folder
    output_dir = args.output_dir

    if not os.path.isdir(folder):
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    config = load_config(args.bf_config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, config)

    config.streaming.anchor_persistence_path = os.path.join(
        output_dir, "anchors_state.pkl")

    try:
        run_folder(folder, output_dir, config)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
