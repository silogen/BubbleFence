"""
BubbleFence pipeline runner -- importable orchestration functions.

Provides:
    ingest_data()      - core BubbleFence ingestion on a DataFrame
    load_annotations() - load annotations.csv or scan for image files
    setup_file_logging() - add a file handler to the root logger
    run_folder()       - run BubbleFence on a single folder end-to-end
    run_batch()        - run BubbleFence on multiple folders from a YAML config
    run_visualizations() - generate configured visualization plots
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import BubbleFenceConfig
from .bubble_fence import BubbleFencePipeline
from .visualization import (
    visualize_standalone_from_persistence,
    visualize_standalone_3d_from_persistence,
    visualize_from_persistence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def load_annotations(folder_path: str) -> pd.DataFrame:
    """Load annotations.csv from *folder_path*, or scan for image files.

    Returns a DataFrame with at least a ``filename`` column.
    """
    annotations_csv = os.path.join(folder_path, "annotations.csv")
    if os.path.exists(annotations_csv):
        df = pd.read_csv(annotations_csv)
        logger.info(f"Loaded {len(df)} annotations from {annotations_csv}")
        return df

    logger.warning(
        f"Annotations not found: {annotations_csv} -- "
        "scanning for image files (no labels)"
    )
    image_files = sorted(
        f.name for f in Path(folder_path).iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    logger.info(f"Found {len(image_files)} image files (no labels)")
    return pd.DataFrame({'filename': image_files})


def setup_file_logging(output_dir: str, config: BubbleFenceConfig) -> None:
    """Add a file handler to the root logger inside *output_dir*."""
    log_file = os.path.join(output_dir, config.logging.log_file)
    handler = logging.FileHandler(log_file)
    handler.setLevel(getattr(logging, config.logging.level, logging.INFO))
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(handler)
    logger.info(f"Logging to file: {log_file}")


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def ingest_data(
    annotations_df: pd.DataFrame,
    data_dir: str,
    config: BubbleFenceConfig,
    full_dataset_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run BubbleFence ingestion on a DataFrame of annotations.

    Args:
        annotations_df: DataFrame with at least a ``filename`` column.
        data_dir: Base path to the image directory.
        config: Loaded BubbleFence configuration.
        full_dataset_csv_path: Path to cumulative CSV for cross-batch
            collision avoidance (optional).

    Returns:
        The enriched DataFrame with ``dataset_split``, ``train_split``,
        ``anchor_id``, and ``similar_images`` columns added.
    """
    logger.info("Starting BubbleFence semantic data ingestion")
    logger.info(f"Processing {len(annotations_df)} images from {data_dir}")

    pipeline = BubbleFencePipeline(config, full_dataset_csv_path=full_dataset_csv_path)

    result_df = pipeline.process_dataframe(
        annotations_df,
        image_path_column='filename',
        base_path=data_dir,
    )

    # Map dataset splits to match original format
    split_mapping = {
        'train': 'train',
        'validation': 'val',
        'test': 'test',
        'unassigned': 'train',
    }
    if 'dataset_split' in result_df.columns:
        result_df['train_split'] = result_df['dataset_split'].map(split_mapping)

    # Add similar_images column for compatibility
    if 'anchor_id' in result_df.columns:
        result_df['similar_images'] = ''
        for anchor_id in result_df['anchor_id'].dropna().unique():
            mask = result_df['anchor_id'] == anchor_id
            indices = result_df[mask].index.tolist()
            if len(indices) > 1:
                for idx in indices:
                    similar = [i for i in indices if i != idx]
                    if similar:
                        result_df.loc[idx, 'similar_images'] = str(similar)

    stats = pipeline.get_pipeline_stats()
    logger.info("BubbleFence processing completed successfully")
    logger.info(f"Pipeline stats: {stats}")
    return result_df


# ---------------------------------------------------------------------------
# Single-folder runner
# ---------------------------------------------------------------------------

def run_folder(
    folder_path: str,
    output_dir: str,
    config: BubbleFenceConfig,
    full_dataset_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run BubbleFence on a single folder end-to-end.

    1. Loads annotations (CSV or image scan).
    2. Calls :func:`ingest_data`.
    3. Appends assigned rows to the cumulative CSV with ``run_id`` and
       ``folder_path`` columns.
    4. Logs a split summary.

    Args:
        folder_path: Directory containing images (and optionally
            ``annotations.csv``).
        output_dir: Where to write results (``full_dataset.csv``,
            ``anchors_state.pkl``, embeddings, etc.).
        config: Loaded BubbleFence configuration.  The caller is responsible
            for setting ``config.streaming.anchor_persistence_path`` before
            calling this function.
        full_dataset_csv: Explicit path for the cumulative CSV.  Defaults to
            ``<output_dir>/full_dataset.csv``.

    Returns:
        The result DataFrame from :func:`ingest_data`.
    """
    if full_dataset_csv is None:
        full_dataset_csv = os.path.join(output_dir, "full_dataset.csv")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    annotations_df = load_annotations(folder_path)

    result_df = ingest_data(
        annotations_df,
        folder_path,
        config=config,
        full_dataset_csv_path=full_dataset_csv,
    )

    # Filter to assigned rows and tag with run_id + folder_path
    new_assigned = result_df[result_df['train_split'].notna()].copy()
    folder_name = Path(folder_path).name
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    run_id = f"{folder_name}_{timestamp}"
    new_assigned.insert(0, 'run_id', run_id)
    new_assigned.insert(1, 'folder_path', folder_path)
    logger.info(f"Run ID: {run_id}, assigned: {len(new_assigned)}")

    # Append to cumulative CSV
    if os.path.exists(full_dataset_csv):
        existing = pd.read_csv(full_dataset_csv)
        full_dataset = pd.concat([existing, new_assigned], ignore_index=True)
        logger.info(
            f"Updated full_dataset.csv: {len(existing)} existing + "
            f"{len(new_assigned)} new = {len(full_dataset)} total"
        )
    else:
        full_dataset = new_assigned
        logger.info(f"Created full_dataset.csv with {len(full_dataset)} data points")

    full_dataset.to_csv(full_dataset_csv, index=False)

    # Print split summary
    total = len(new_assigned)
    if total > 0 and 'train_split' in new_assigned.columns:
        for split, count in new_assigned['train_split'].value_counts().items():
            logger.info(f"  {split}: {count} ({count / total * 100:.1f}%)")

    return result_df


# ---------------------------------------------------------------------------
# Visualization runner
# ---------------------------------------------------------------------------

def run_visualizations(output_dir: str, vis_configs: list,
                       show: bool = False) -> None:
    """Generate all configured visualizations.

    Args:
        output_dir: Directory containing BubbleFence outputs.
        vis_configs: List of dicts, each with a ``type`` key
            (``standalone`` or ``detailed``) and a ``save`` key for the
            output filename.  All remaining keys are forwarded as kwargs
            to the corresponding visualization function.
        show: If True, keep figures open for display (e.g. in notebooks).
            If False (default), close figures after saving to free memory.
    """
    for i, vis in enumerate(vis_configs):
        vis = vis.copy()
        vis_type = vis.pop("type", "standalone")

        if vis_type == "stats":
            save_name = vis.pop("save", "stats.txt")
        elif vis_type == "3d":
            save_name = vis.pop("save", "bubblefence_3d.html")
        else:
            save_name = vis.pop("save", f"vis_{i}.png")
        save_path = os.path.join(output_dir, save_name)

        logger.info(
            f"=== Visualization [{i + 1}/{len(vis_configs)}]: "
            f"{save_name} ({vis_type}) ==="
        )

        if vis_type == "stats":
            from bubblefence.visualization import print_dataset_stats
            print_dataset_stats(output_dir, save_path=save_path)
            continue
        elif vis_type == "standalone":
            fig = visualize_standalone_from_persistence(
                data_root=output_dir, save_path=save_path, **vis)
        elif vis_type == "detailed":
            fig = visualize_from_persistence(
                data_root=output_dir, save_path=save_path, **vis)
        elif vis_type == "summary":
            vis.setdefault("figsize", (20, 8))
            fig = visualize_from_persistence(
                data_root=output_dir, save_path=save_path,
                num_panels=2, **vis)
        elif vis_type == "3d":
            visualize_standalone_3d_from_persistence(
                data_root=output_dir, save_path=save_path, **vis)
            logger.info(f"Saved: {save_path}")
            continue
        else:
            logger.warning(f"Unknown vis type: {vis_type}, skipping")
            continue

        if show:
            plt.show()
        else:
            plt.close('all')
        logger.info(f"Saved: {save_path}")

    logger.info(f"All {len(vis_configs)} visualizations saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _load_batch_config(config_path: str) -> dict:
    """Load batch config from YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return {
        "folders": cfg["folders"],
        "output_dir": cfg["output_dir"],
        "bf_config": cfg.get("bf_config"),
        "visualizations": cfg.get("visualizations", []),
    }


def run_batch(
    batch_config_path: str,
    skip_bf: bool = False,
    skip_vis: bool = False,
) -> None:
    """Run BubbleFence on multiple folders then generate visualizations.

    Reads a YAML config with ``folders``, ``output_dir``, ``bf_config``,
    and ``visualizations`` keys.

    Args:
        batch_config_path: Path to the batch YAML config file.
        skip_bf: If True, skip BubbleFence processing (visualizations only).
        skip_vis: If True, skip visualization generation.
    """
    cfg = _load_batch_config(batch_config_path)
    output_dir = cfg["output_dir"]
    logger.info(f"Loaded batch config from {batch_config_path}")

    if not skip_bf:
        folders = cfg["folders"]
        bf_config_path = cfg["bf_config"]

        if not bf_config_path or not os.path.exists(bf_config_path):
            logger.error(f"BubbleFence YAML config not found: {bf_config_path}")
            sys.exit(1)

        from . import load_config
        config = load_config(bf_config_path)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_file_logging(output_dir, config)

        config.streaming.anchor_persistence_path = os.path.join(
            output_dir, "anchors_state.pkl")

        full_dataset_csv = os.path.join(output_dir, "full_dataset.csv")

        for i, folder in enumerate(folders):
            logger.info(
                f"=== Processing folder [{i + 1}/{len(folders)}]: {folder} ==="
            )
            try:
                run_folder(folder, output_dir, config,
                           full_dataset_csv=full_dataset_csv)
            except FileNotFoundError as e:
                logger.error(str(e))
                continue
            except Exception as e:
                logger.error(f"Failed on {folder}: {e}")
                continue

        logger.info(
            f"All {len(folders)} folders processed. "
            f"Results in {full_dataset_csv}"
        )

    if not skip_vis:
        vis_configs = cfg.get("visualizations", [])
        if vis_configs:
            run_visualizations(output_dir, vis_configs)

    logger.info("Done.")
