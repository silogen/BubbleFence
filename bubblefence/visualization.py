"""
Visualization tools for BubbleFence semantic data splitting.

This module provides utilities to visualize embeddings, anchor placement,
hypersphere boundaries, and dataset assignments in 2D PCA space.

Supports both single-run and multi-run persistence modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle
import torch
import os

from .data_structures import (
    EmbeddingTrajectory, HypersphereAnchor, AnchorRegistry,
    DatasetAssignmentResult, DatasetSplit, EmbeddingPoint
)
from .bubble_fence import BubbleFencePipeline


logger = logging.getLogger(__name__)


class BubbleFenceVisualizer:
    """
    Visualizer for BubbleFence data splits, anchors, and embeddings.

    Supports both single-run mode (with pipeline) and multi-run mode (from persistence files).
    """

    def __init__(self, pipeline: Optional[BubbleFencePipeline] = None,
                 data_root: Optional[str] = None):
        """
        Initialize visualizer with either a BubbleFence pipeline or data root for persistence mode.

        Args:
            pipeline: Trained BubbleFence pipeline with anchor registry (single-run mode)
            data_root: Root directory containing full_dataset.csv, anchors_state.pkl,
                      and embeddings/ folder (multi-run mode)
        """
        if pipeline is not None:
            # Single-run mode
            self.pipeline = pipeline
            self.config = pipeline.config
            self.mode = "single_run"
            self.data_root = None
        elif data_root is not None:
            # Multi-run mode
            self.pipeline = None
            self.config = None
            self.mode = "multi_run"
            self.data_root = Path(data_root)
            self._validate_persistence_files()
        else:
            raise ValueError("Either pipeline or data_root must be provided")

        # Color scheme for dataset splits
        self.colors = {
            DatasetSplit.TRAIN: '#1f77b4',      # Blue
            DatasetSplit.VALIDATION: '#ff7f0e', # Orange
            DatasetSplit.TEST: '#2ca02c',        # Green
            DatasetSplit.UNASSIGNED: '#d62728'   # Red
        }

        self.split_names = {
            DatasetSplit.TRAIN: 'Train',
            DatasetSplit.VALIDATION: 'Validation',
            DatasetSplit.TEST: 'Test',
            DatasetSplit.UNASSIGNED: 'Unassigned'
        }

    def _validate_persistence_files(self):
        """Validate that required persistence files exist."""
        required_files = [
            self.data_root / "full_dataset.csv",
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required persistence file not found: {file_path}")

        # Check if embeddings directory exists
        embeddings_dir = self.data_root / "embeddings"
        if not embeddings_dir.exists():
            logger.warning(f"Embeddings directory not found: {embeddings_dir}. "
                         "Visualization will fail without real embeddings.")

        # Check if anchor state file exists
        anchor_file = self.data_root / "anchors_state.pkl"
        if not anchor_file.exists():
            logger.warning(f"Anchor state file not found: {anchor_file}. "
                         "Will not show anchor information.")

    def _load_full_dataset(self) -> pd.DataFrame:
        """Load the full dataset CSV."""
        if self.mode == "single_run":
            raise RuntimeError("Cannot load full dataset in single-run mode")

        csv_path = self.data_root / "full_dataset.csv"
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} data points from {csv_path}")
        return df

    def _load_all_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Load and align embeddings for all rows in the dataframe.

        Each .pt file stores embeddings for one run. The .pt metadata contains
        full file paths (e.g. 'zod_data/drives/000000/camera_front_blur/img.jpg')
        from which we extract the folder path. The CSV has a 'folder_path' column
        with the same value. We match .pt files to CSV run_id groups by
        folder_path, keeping chronological order for multiple runs from the same
        folder. CSV rows within a run are already aligned with .pt embedding rows.

        Args:
            df: The full dataset dataframe (must have folder_path, run_id columns)

        Returns:
            numpy array of shape (len(df), emb_dim) aligned to df rows
        """
        embeddings_dir = self.data_root / "embeddings"
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        # Load .pt files in chronological order (sorted by filename which
        # contains unix timestamp). Group by folder_path extracted from
        # the stored filenames, keeping a list per folder_path.
        from collections import defaultdict
        folder_path_to_pt_list = defaultdict(list)  # folder_path -> [emb_array, ...]
        emb_dim = None

        for pt_file in sorted(embeddings_dir.glob("*.pt")):
            try:
                data = torch.load(pt_file, map_location='cpu')

                if not isinstance(data, dict) or 'embeddings' not in data:
                    logger.warning(f"Skipping {pt_file.name}: unexpected format")
                    continue

                embeddings = data['embeddings']
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()

                meta = data.get('metadata', {})
                filenames = meta.get('filenames', [])

                if not filenames:
                    logger.warning(f"Skipping {pt_file.name}: no filenames in metadata")
                    continue

                emb_dim = embeddings.shape[1]

                # Extract folder path from the first filename's parent dir
                first_file = filenames[0].replace('\\', '/')
                parts = first_file.rsplit('/', 1)
                if len(parts) == 2:
                    folder_path = parts[0]
                else:
                    folder_path = ''

                folder_path_to_pt_list[folder_path].append(embeddings)
                logger.info(f"Loaded {embeddings.shape[0]} embeddings for "
                           f"'{folder_path}' from {pt_file.name}")

            except Exception as e:
                logger.warning(f"Failed to load {pt_file.name}: {e}")

        if not folder_path_to_pt_list or emb_dim is None:
            raise RuntimeError("No embeddings could be loaded from any .pt file")

        # Get unique run_ids in CSV order, grouped by their folder_path.
        # For each folder_path, pop .pt files in order to match runs.
        # Track which .pt array maps to which run_id.
        folder_path_pop_idx = defaultdict(int)  # folder_path -> next .pt index
        run_id_to_embeddings = {}

        seen = set()
        for _, row in df.iterrows():
            run_id = row['run_id']
            if run_id in seen:
                continue
            seen.add(run_id)

            folder_path = str(row['folder_path']).replace('\\', '/')

            pt_list = folder_path_to_pt_list.get(folder_path, [])
            pop_idx = folder_path_pop_idx[folder_path]

            if pop_idx < len(pt_list):
                run_id_to_embeddings[run_id] = pt_list[pop_idx]
                folder_path_pop_idx[folder_path] = pop_idx + 1
                logger.info(f"Mapped run '{run_id}' -> "
                           f"'{folder_path}' .pt #{pop_idx} "
                           f"({pt_list[pop_idx].shape[0]} embeddings)")
            else:
                logger.warning(f"No .pt file for run '{run_id}' "
                             f"(folder_path='{folder_path}')")

        logger.info(f"Embedding index: "
                    f"{', '.join(f'{k}({v.shape[0]})' for k, v in run_id_to_embeddings.items())}")

        # Assemble the full embedding matrix.
        # CSV rows for each run_id are in order and aligned with .pt rows.
        all_embeddings = np.zeros((len(df), emb_dim))
        matched = 0
        run_row_counter = {}  # run_id -> next within-run index

        for i, (idx, row) in enumerate(df.iterrows()):
            run_id = row['run_id']

            if run_id not in run_id_to_embeddings:
                continue

            embeddings = run_id_to_embeddings[run_id]
            within_run_idx = run_row_counter.get(run_id, 0)

            if within_run_idx < len(embeddings):
                all_embeddings[i] = embeddings[within_run_idx]
                matched += 1
                run_row_counter[run_id] = within_run_idx + 1
            else:
                logger.warning(f"Run '{run_id}' has more CSV rows than "
                             f"embeddings ({len(embeddings)})")

        logger.info(f"Matched {matched}/{len(df)} embeddings")

        if matched < len(df):
            logger.warning(f"{len(df) - matched} points have no embeddings")

        return all_embeddings

    def _load_anchor_registry(self) -> Optional[AnchorRegistry]:
        """Load anchor registry from persistence file."""
        if self.mode == "single_run":
            return self.pipeline.anchor_registry

        anchor_file = self.data_root / "anchors_state.pkl"
        if not anchor_file.exists():
            return None

        try:
            with open(anchor_file, 'rb') as f:
                anchor_data = pickle.load(f)
            logger.info(f"Loaded anchor registry from {anchor_file}")

            # Handle different possible formats of anchor data
            if isinstance(anchor_data, AnchorRegistry):
                return anchor_data
            elif isinstance(anchor_data, dict):
                # If it's a dict, check if it has an anchor registry
                if 'anchor_registry' in anchor_data:
                    registry = anchor_data['anchor_registry']
                    logger.info(f"Extracted anchor registry with {getattr(registry, 'num_anchors', 0)} anchors")
                    return registry
                else:
                    # Create a simple mock anchor registry for compatibility
                    logger.warning("Anchor data is in dict format without AnchorRegistry. Creating mock registry.")
                    return None
            else:
                logger.warning(f"Unknown anchor data format: {type(anchor_data)}")
                return None

        except Exception as e:
            logger.warning(f"Failed to load anchor registry: {e}")
            return None

    def _create_trajectory_from_dataframe(self, df: pd.DataFrame,
                                        embeddings: np.ndarray) -> EmbeddingTrajectory:
        """Create an EmbeddingTrajectory from DataFrame data and real embeddings."""
        trajectory = EmbeddingTrajectory("multi_run_visualization")

        split_mapping = {
            'train': DatasetSplit.TRAIN,
            'val': DatasetSplit.VALIDATION,
            'validation': DatasetSplit.VALIDATION,
            'test': DatasetSplit.TEST
        }

        logger.info(f"Building trajectory from {len(df)} rows with embeddings shape {embeddings.shape}")

        # Load anchor registry to compute distances for training points
        anchor_registry = self._load_anchor_registry()
        anchor_centers = None
        if anchor_registry and anchor_registry.num_anchors > 0:
            anchor_centers = np.array([anchor.center for anchor in anchor_registry.anchors.values()])
            logger.info(f"Loaded {len(anchor_centers)} anchor centers for distance computation")

        points = []
        for i, (idx, row) in enumerate(df.iterrows()):
            dataset_split = split_mapping.get(row['dataset_split'], DatasetSplit.UNASSIGNED)

            # Determine full image path from run_id and filename
            run_id = row['run_id']
            # Use rsplit to handle folder names with underscores (e.g. deus_1_TIMESTAMP)
            parts = run_id.rsplit('_', 1)
            folder_name = parts[0] if len(parts) == 2 else run_id
            full_path = f"{folder_name}/{row['filename']}"

            point = EmbeddingPoint(
                original_index=int(row.get('original_index', idx)),
                file_path=full_path,
                metadata=row.to_dict(),
                dataset_split=dataset_split
            )

            # Add anchor info if available from CSV
            if 'anchor_id' in row and pd.notna(row['anchor_id']):
                point.anchor_id = int(row['anchor_id'])
            if 'distance_to_anchor' in row and pd.notna(row['distance_to_anchor']):
                # Clamp negative floating point artifacts to 0
                point.distance_to_anchor = max(0.0, float(row['distance_to_anchor']))

            # For training points (no anchor assigned), compute distance to closest anchor
            if dataset_split == DatasetSplit.TRAIN and anchor_centers is not None:
                embedding = embeddings[i]
                # Compute cosine distances to all anchors
                embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                anchor_norms = anchor_centers / (np.linalg.norm(anchor_centers, axis=1, keepdims=True) + 1e-8)
                cosine_similarities = np.dot(anchor_norms, embedding_norm)
                cosine_distances = 1 - cosine_similarities
                min_distance = max(0.0, float(np.min(cosine_distances)))
                closest_anchor_idx = np.argmin(cosine_distances)

                point.distance_to_anchor = min_distance
                point.closest_anchor_id = list(anchor_registry.anchors.keys())[closest_anchor_idx]

            points.append(point)

        # Set embeddings and points in bulk
        embeddings_tensor = torch.from_numpy(embeddings).float()
        trajectory.set_embeddings(embeddings_tensor, points)

        return trajectory

    def _create_assignment_result_from_dataframe(self, df: pd.DataFrame) -> DatasetAssignmentResult:
        """Create a DatasetAssignmentResult from DataFrame data."""
        # Map indices by split
        train_indices = []
        validation_indices = []
        test_indices = []
        unassigned_indices = []

        for i, split in enumerate(df['dataset_split']):
            if split == 'train':
                train_indices.append(i)
            elif split in ['val', 'validation']:
                validation_indices.append(i)
            elif split == 'test':
                test_indices.append(i)
            else:
                unassigned_indices.append(i)

        # Compute coverage: points inside at least one anchor hypersphere
        covered_count = 0
        if 'num_containing_anchors' in df.columns:
            covered_count = int((df['num_containing_anchors'].fillna(0) > 0).sum())
        elif 'anchor_id' in df.columns:
            covered_count = int(df['anchor_id'].notna().sum())

        # Count unique anchors used
        num_anchors_used = 0
        anchor_registry = self._load_anchor_registry()
        if anchor_registry:
            num_anchors_used = anchor_registry.num_anchors
        elif 'anchor_id' in df.columns:
            num_anchors_used = int(df['anchor_id'].dropna().nunique())

        statistics = {
            'total_points': len(df),
            'train_count': len(train_indices),
            'validation_count': len(validation_indices),
            'test_count': len(test_indices),
            'unassigned_count': len(unassigned_indices),
            'train_ratio': len(train_indices) / len(df) if len(df) > 0 else 0,
            'validation_ratio': len(validation_indices) / len(df) if len(df) > 0 else 0,
            'test_ratio': len(test_indices) / len(df) if len(df) > 0 else 0,
            'eval_ratio': (len(validation_indices) + len(test_indices)) / len(df) if len(df) > 0 else 0,
            'coverage_ratio': covered_count / len(df) if len(df) > 0 else 0,
            'num_anchors_used': num_anchors_used,
        }

        return DatasetAssignmentResult(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            unassigned_indices=unassigned_indices,
            assignment_details=[],
            statistics=statistics
        )

    def _generate_all_boundary_points(self, anchor_registry, n_samples: int = 200):
        """
        Generate surface sample points for all anchor hyperspheres.

        Returns:
            boundary_points_hd: numpy array of shape (total_boundary_pts, emb_dim)
            boundary_anchor_map: list of (anchor_index, 'outer'|'shell_i') for each point
        """
        boundary_points_hd = []
        boundary_anchor_map = []

        for anchor_idx, anchor in enumerate(anchor_registry.anchors.values()):
            # Outer boundary
            surface = self._sample_cosine_sphere_surface(
                anchor.center, anchor.total_radius, n_samples=n_samples)
            for pt in surface:
                boundary_points_hd.append(pt)
                boundary_anchor_map.append((anchor_idx, 'outer'))

            # Inner shells
            if anchor.nested_shells:
                for shell_idx, shell in enumerate(anchor.nested_shells):
                    shell_surface = self._sample_cosine_sphere_surface(
                        anchor.center, shell.outer_radius, n_samples=n_samples)
                    for pt in shell_surface:
                        boundary_points_hd.append(pt)
                        boundary_anchor_map.append((anchor_idx, f'shell_{shell_idx}'))

        if boundary_points_hd:
            boundary_points_hd = np.array(boundary_points_hd)
        else:
            boundary_points_hd = np.empty((0, 512))

        logger.info(f"Generated {len(boundary_points_hd)} boundary sample points "
                   f"for {anchor_registry.num_anchors} anchors")
        return boundary_points_hd, boundary_anchor_map

    def visualize_data_splits(self, trajectory: Optional[EmbeddingTrajectory] = None,
                            assignment_result: Optional[DatasetAssignmentResult] = None,
                            reduction_method: str = "PCA",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None,
                            show_anchors: bool = True,
                            show_hyperspheres: bool = True,
                            show_density: bool = False,
                            show_heatmap: bool = True,
                            run_filter: Optional[str] = None,
                            show_all_trajectories: bool = False,
                            smoothing: float = 0.1,
                            num_panels: int = 4) -> plt.Figure:
        """
        Create a comprehensive visualization of data splits and anchors.

        Args:
            trajectory: Embedding trajectory with data points (optional in multi-run mode)
            assignment_result: Dataset assignment results (optional in multi-run mode)
            reduction_method: "PCA" or "TSNE" for dimensionality reduction
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            show_anchors: Whether to show anchor points
            show_hyperspheres: Whether to show hypersphere boundaries
            show_density: Whether to show density contours
            show_heatmap: Whether to show KDE heatmap of bubble interiors
                instead of convex hull outlines (default True). Mutually
                exclusive with show_hyperspheres -- heatmap takes priority.
            run_filter: Optional run ID to filter data (multi-run mode only)
            show_all_trajectories: Whether to overlay smooth trajectory curves
                for each run_id on the heatmap / convex-hull panel (ax2) and
                the data-splits panel (ax1). Each run gets a distinct colour.

        Returns:
            matplotlib Figure object
        """
        # Heatmap and hypersphere outlines are mutually exclusive; heatmap wins
        if show_heatmap:
            show_hyperspheres = False

        # Load data based on mode
        df = None
        if self.mode == "multi_run":
            # Load data from persistence files
            df = self._load_full_dataset()

            # Filter by run if specified
            if run_filter:
                df = df[df['run_id'] == run_filter]
                logger.info(f"Filtered to {len(df)} points for run {run_filter}")

            # Load real embeddings for all rows
            embeddings = self._load_all_embeddings(df)

            # Create trajectory and assignment result from dataframe
            trajectory = self._create_trajectory_from_dataframe(df, embeddings)
            assignment_result = self._create_assignment_result_from_dataframe(df)

        # Get anchor registry
        anchor_registry = self._load_anchor_registry()
        num_anchors = anchor_registry.num_anchors if anchor_registry else 0

        logger.info(f"Creating visualization with {trajectory.num_points} points and "
                   f"{num_anchors} anchors")

        # Get embeddings
        embeddings = trajectory.embeddings_matrix
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()

        n_real = len(embeddings)

        # Generate boundary points if needed (before dim reduction so we can
        # include them in the t-SNE fit)
        boundary_points_hd = None
        boundary_2d = None
        boundary_anchor_map = None
        volume_points_hd = None
        volume_2d = None
        volume_anchor_map = None

        if anchor_registry and anchor_registry.num_anchors > 0:
            # Convex hulls (show_hyperspheres) now use real eval points in 2D,
            # so no synthetic boundary points are needed for that mode.

            if show_heatmap and reduction_method.upper() != "TSNE":
                # For PCA: generate synthetic points for sigma computation
                # (they can be projected independently without affecting the fit).
                # For t-SNE: skip synth points to avoid distorting the layout;
                # sigmas are computed from real assigned points only.
                boundary_points_hd, boundary_anchor_map = self._generate_all_boundary_points(
                    anchor_registry, n_samples=50)
                if len(boundary_points_hd) == 0:
                    boundary_points_hd = None
                    boundary_anchor_map = None

                volume_points_hd, volume_anchor_map = self._generate_all_volume_points(
                    anchor_registry, n_samples=150)
                if len(volume_points_hd) == 0:
                    volume_points_hd = None
                    volume_anchor_map = None

        # Collect all extra high-dim points that need projection
        extra_hd_parts = []
        boundary_slice = None
        volume_slice = None
        if boundary_points_hd is not None:
            boundary_slice = (len(extra_hd_parts),
                              len(extra_hd_parts) + len(boundary_points_hd))
            extra_hd_parts.append(boundary_points_hd)
        if volume_points_hd is not None:
            offset = sum(len(p) for p in extra_hd_parts)
            volume_slice = (offset, offset + len(volume_points_hd))
            extra_hd_parts.append(volume_points_hd)

        extra_hd = np.concatenate(extra_hd_parts, axis=0) if extra_hd_parts else None

        # Fit dimensionality reduction
        if reduction_method.upper() == "PCA":
            # PCA: fit on real data only, then transform extra points separately
            embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)
            if extra_hd is not None:
                extra_2d = reducer.transform(extra_hd)
                logger.info(f"Projected {len(extra_hd)} extra points through PCA transform")
        elif reduction_method.upper() == "TSNE":
            if extra_hd is not None:
                # t-SNE: fit on combined real + extra points, then split back
                combined = np.concatenate([embeddings, extra_hd], axis=0)
                logger.info(f"Fitting t-SNE on {n_real} real + {len(extra_hd)} "
                           f"extra = {len(combined)} total points")
                combined_2d, reducer = self._reduce_dimensions(combined, reduction_method)
                embeddings_2d = combined_2d[:n_real]
                extra_2d = combined_2d[n_real:]
            else:
                # No extra points, just fit on real data
                embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)
        else:
            embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)

        # Unpack projected extra points
        if extra_hd is not None:
            if boundary_slice is not None:
                boundary_2d = extra_2d[boundary_slice[0]:boundary_slice[1]]
            if volume_slice is not None:
                volume_2d = extra_2d[volume_slice[0]:volume_slice[1]]

        # Pre-compute per-run trajectory curves if requested
        run_trajectories = None
        if show_all_trajectories and df is not None:
            run_trajectories = self._compute_run_trajectories(df, embeddings_2d)
            logger.info(f"Computed trajectory curves for {len(run_trajectories)} runs")

        # Compute anchor point indices for star markers
        anchor_point_indices = self._get_anchor_point_indices(trajectory, anchor_registry)

        # Create figure with subplots
        if num_panels == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('BubbleFence Data Splitting Visualization', fontsize=16, fontweight='bold')

        # Plot 1: All data points colored by split assignment (NO boundary points)
        self._plot_data_splits(ax1, embeddings_2d, assignment_result, trajectory,
                              anchor_point_indices=anchor_point_indices)
        # Overlay trajectory lines on ax1 if enabled
        if run_trajectories is not None:
            self._draw_trajectory_splines(ax1, run_trajectories, linewidth=1.5,
                                          alpha=0.6, add_legend=False,
                                          smoothing=smoothing)
        ax1.set_title('Dataset Split Assignments')

        # Plot 2: Anchors and hyperspheres (WITH boundary convex hulls or heatmap)
        if show_heatmap:
            self._plot_bubble_heatmap(
                ax2, embeddings_2d, volume_2d, volume_anchor_map,
                reducer, anchor_registry, trajectory,
                boundary_2d=boundary_2d, boundary_anchor_map=boundary_anchor_map,
                run_trajectories=run_trajectories, smoothing=smoothing)
            ax2.set_title('Anchor Bubble Heatmap' +
                          (' + Trajectories' if run_trajectories else ''))
            # Sync axis limits with the splits plot (ax1)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
        elif show_anchors or show_hyperspheres:
            self._plot_anchors_and_hyperspheres(
                ax2, embeddings_2d, reducer,
                show_anchors, show_hyperspheres, anchor_registry, trajectory,
                boundary_2d=boundary_2d, boundary_anchor_map=boundary_anchor_map,
                run_trajectories=run_trajectories, smoothing=smoothing)
            ax2.set_title('Anchors and Hypersphere Boundaries' +
                          (' + Trajectories' if run_trajectories else ''))
        else:
            ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20, c='gray')
            ax2.set_title('Anchors and Hypersphere Boundaries')

        # Bottom row (4-panel only)
        if num_panels == 4:
            # Plot 3: Density visualization (if enabled)
            if show_density:
                self._plot_density_contours(ax3, embeddings_2d)
            else:
                # Show distribution by split with histograms
                self._plot_split_distributions(ax3, embeddings_2d, assignment_result)
            ax3.set_title('Density Distribution' if show_density else 'Split Distribution')

            # Plot 4: Statistics and summary
            self._plot_statistics(ax4, assignment_result, trajectory, anchor_registry)
            ax4.set_title('Assignment Statistics')

        # Set common labels for embedding plots
        method_name = reduction_method.upper()
        embed_axes = [ax1, ax2] if num_panels == 2 else [ax1, ax2, ax3]
        for ax in embed_axes:
            ax.set_xlabel(f'{method_name} Component 1')
            ax.set_ylabel(f'{method_name} Component 2')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        return fig

    def _reduce_dimensions(self, embeddings,
                          method: str = "PCA",
                          n_components: int = 2) -> Tuple[np.ndarray, Any]:
        """Reduce embeddings to 2D (or 3D) for visualization."""
        # Convert torch tensor to numpy if needed
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
        logger.debug(f"Reducing {embeddings.shape} embeddings to {n_components}D using {method}")

        if method.upper() == "PCA":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            logger.debug(f"PCA explained variance: {reducer.explained_variance_ratio_}")
        elif method.upper() == "TSNE":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
            reduced = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return reduced, reducer

    def _get_anchor_point_indices(self, trajectory: EmbeddingTrajectory,
                                  anchor_registry: Optional[AnchorRegistry] = None,
                                  threshold: float = 1e-4) -> set:
        """
        Find data point indices that correspond to anchor centers.

        First checks CSV metadata (distance_to_anchor ~ 0).  For any anchor
        whose center point was assigned to a different anchor in the CSV
        (can happen with LID-weighted snap), falls back to finding the
        closest embedding in the trajectory.

        Args:
            trajectory: EmbeddingTrajectory with points containing metadata
            anchor_registry: Optional anchor registry for fallback matching
            threshold: Cosine distance threshold for matching (default 1e-4)

        Returns:
            Set of data point indices that are anchor centers
        """
        anchor_indices = set()
        matched_anchor_ids = set()
        for i, point in enumerate(trajectory.points):
            dist = getattr(point, 'distance_to_anchor', None)
            anchor_id = getattr(point, 'anchor_id', None)
            if anchor_id is not None and dist is not None and dist < threshold:
                anchor_indices.add(i)
                matched_anchor_ids.add(anchor_id)

        # Fallback: find center points for anchors that had no CSV match
        if anchor_registry is not None:
            missing = set(anchor_registry.anchors.keys()) - matched_anchor_ids
            if missing:
                embeddings = trajectory.embeddings_matrix
                if hasattr(embeddings, 'cpu'):
                    emb_np = embeddings.cpu().numpy()
                else:
                    emb_np = np.asarray(embeddings)
                for aid in missing:
                    center = anchor_registry.anchors[aid].center
                    if hasattr(center, 'cpu'):
                        center = center.cpu().numpy()
                    dists = np.linalg.norm(emb_np - center.reshape(1, -1), axis=1)
                    closest = int(np.argmin(dists))
                    if dists[closest] < threshold:
                        anchor_indices.add(closest)
                        logger.debug(f"Anchor {aid} center matched to point "
                                     f"{closest} via embedding fallback")

        logger.info(f"Found {len(anchor_indices)} anchor center data points")
        return anchor_indices

    def _plot_data_splits(self, ax: plt.Axes, embeddings_2d: np.ndarray,
                         assignment_result: DatasetAssignmentResult,
                         trajectory: EmbeddingTrajectory,
                         anchor_point_indices: Optional[set] = None):
        """Plot data points colored by dataset split assignment.

        Anchor data points are rendered with star markers instead of circles.
        """
        if anchor_point_indices is None:
            anchor_point_indices = set()

        # Create scatter plot - split each group into anchor vs non-anchor
        for split in DatasetSplit:
            split_indices = [i for i, point in enumerate(trajectory.points)
                           if point.dataset_split == split]
            if not split_indices:
                continue

            non_anchor = [i for i in split_indices if i not in anchor_point_indices]
            anchor = [i for i in split_indices if i in anchor_point_indices]

            # Plot non-anchor points as circles
            if non_anchor:
                ax.scatter(embeddings_2d[non_anchor, 0], embeddings_2d[non_anchor, 1],
                          c=self.colors[split], label=self.split_names[split],
                          alpha=0.7, s=30, edgecolors='black', linewidths=0.3)

            # Plot anchor points as stars (same color, star marker)
            if anchor:
                label = self.split_names[split] if not non_anchor else None
                ax.scatter(embeddings_2d[anchor, 0], embeddings_2d[anchor, 1],
                          c=self.colors[split], label=label,
                          alpha=0.9, s=80, marker='*',
                          edgecolors='black', linewidths=0.5)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _sample_cosine_sphere_surface(self, center: np.ndarray,
                                      cosine_radius: float,
                                      n_samples: int = 200) -> np.ndarray:
        """
        Sample points on the surface of a cosine-distance hypersphere in high-dim space.

        Cosine similarity between two vectors is defined as:
            cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
        which equals cos(theta) where theta is the angle between the vectors.

        Cosine distance = 1 - cosine_similarity, so the surface at cosine distance r
        from center c is the set of vectors v where:
            1 - cos(theta) = r  =>  cos(theta) = 1 - r  =>  theta = arccos(1 - r)

        We sample by constructing vectors at exactly angle theta from the center:
            v = cos(theta) * c_hat + sin(theta) * orth_hat
        where c_hat is the unit center direction and orth_hat is a random unit vector
        orthogonal to c_hat.

        Args:
            center: Anchor center vector (high-dimensional, e.g. 512-dim)
            cosine_radius: Cosine distance radius (typically 0.02 - 0.2)
            n_samples: Number of points to sample on the surface

        Returns:
            Array of shape (n_samples, dim) with points on the hypersphere surface
        """
        # Convert torch tensor to numpy if needed
        if hasattr(center, 'cpu'):
            center = center.cpu().numpy()
        center = np.asarray(center, dtype=np.float64)

        dim = len(center)
        target_cos_sim = np.clip(1.0 - cosine_radius, -1.0, 1.0)
        theta = np.arccos(target_cos_sim)

        center_norm = center / (np.linalg.norm(center) + 1e-8)
        center_mag = np.linalg.norm(center)

        rng = np.random.RandomState(42)
        surface_points = []

        for _ in range(n_samples):
            # Generate random vector, project out the center component to get orthogonal
            rand_vec = rng.randn(dim)
            rand_vec -= np.dot(rand_vec, center_norm) * center_norm
            orth_norm = np.linalg.norm(rand_vec)
            if orth_norm < 1e-10:
                continue
            rand_vec /= orth_norm

            # Point on surface at angle theta from center direction
            surface_point = np.cos(theta) * center_norm + np.sin(theta) * rand_vec
            # Scale to same magnitude as center so PCA projection is consistent
            surface_point *= center_mag
            surface_points.append(surface_point)

        surface_points = np.array(surface_points)

        # Re-center: the mean of surface points is cos(theta)*center, not center.
        # Shift them so their centroid matches the actual center vector.
        # This is exact under linear projections like PCA.
        if len(surface_points) > 0:
            centroid = surface_points.mean(axis=0)
            surface_points += (center - centroid)

        return surface_points

    def _sample_cosine_sphere_volume(self, center: np.ndarray,
                                     cosine_radius: float,
                                     n_samples: int = 50) -> np.ndarray:
        """
        Sample points uniformly within a cosine-distance hypersphere.

        Similar to _sample_cosine_sphere_surface but samples at random distances
        from 0 to cosine_radius, producing volume-filling points instead of
        surface-only points. Used for heatmap visualization.

        Args:
            center: Anchor center vector (high-dimensional)
            cosine_radius: Cosine distance radius
            n_samples: Number of points to sample inside the volume

        Returns:
            Array of shape (n_samples, dim) with points inside the hypersphere
        """
        if hasattr(center, 'cpu'):
            center = center.cpu().numpy()
        center = np.asarray(center, dtype=np.float64)

        dim = len(center)
        center_norm = center / (np.linalg.norm(center) + 1e-8)
        center_mag = np.linalg.norm(center)

        rng = np.random.RandomState(42)
        volume_points = []

        for _ in range(n_samples):
            # Random cosine distance from 0 to cosine_radius
            # Use uniform in [0, 1] then scale, so points spread evenly in
            # cosine-distance space
            frac = rng.uniform(0.0, 1.0)
            r = cosine_radius * frac

            target_cos_sim = np.clip(1.0 - r, -1.0, 1.0)
            theta = np.arccos(target_cos_sim)

            # Random orthogonal direction
            rand_vec = rng.randn(dim)
            rand_vec -= np.dot(rand_vec, center_norm) * center_norm
            orth_norm = np.linalg.norm(rand_vec)
            if orth_norm < 1e-10:
                continue
            rand_vec /= orth_norm

            point = np.cos(theta) * center_norm + np.sin(theta) * rand_vec
            point *= center_mag
            volume_points.append(point)

        return np.array(volume_points) if volume_points else np.empty((0, dim))

    def _generate_all_volume_points(self, anchor_registry, n_samples: int = 50):
        """
        Generate volume-filling sample points for all anchor hyperspheres.

        Returns:
            volume_points_hd: numpy array of shape (total_pts, emb_dim)
            volume_anchor_map: list of anchor_index for each point
        """
        volume_points_hd = []
        volume_anchor_map = []

        for anchor_idx, anchor in enumerate(anchor_registry.anchors.values()):
            pts = self._sample_cosine_sphere_volume(
                anchor.center, anchor.total_radius, n_samples=n_samples)
            for pt in pts:
                volume_points_hd.append(pt)
                volume_anchor_map.append(anchor_idx)

        if volume_points_hd:
            volume_points_hd = np.array(volume_points_hd)
        else:
            volume_points_hd = np.empty((0, 512))

        logger.info(f"Generated {len(volume_points_hd)} volume sample points "
                   f"for {anchor_registry.num_anchors} anchors")
        return volume_points_hd, volume_anchor_map

    def _project_anchor_centers(self, anchor_registry, reducer,
                                embeddings_2d: np.ndarray,
                                original_embeddings=None):
        """Project anchor centers to 2D using the fitted reducer.

        Returns (anchor_centers_hd, anchor_centers_2d, anchor_radii).
        """
        anchor_centers_hd = []
        anchor_radii = []
        for anchor in anchor_registry.anchors.values():
            anchor_centers_hd.append(anchor.center)
            anchor_radii.append(anchor.total_radius)
        anchor_centers_hd = np.array(anchor_centers_hd)
        anchor_radii = np.array(anchor_radii)

        can_transform = hasattr(reducer, 'transform')
        if can_transform:
            anchor_centers_2d = reducer.transform(anchor_centers_hd)
        else:
            anchor_centers_2d = []
            for center in anchor_centers_hd:
                if original_embeddings is not None:
                    dists = np.linalg.norm(
                        original_embeddings - center.reshape(1, -1), axis=1)
                else:
                    dists = np.linalg.norm(
                        embeddings_2d - center.reshape(1, -1), axis=1)
                anchor_centers_2d.append(embeddings_2d[np.argmin(dists)])
            anchor_centers_2d = np.array(anchor_centers_2d)

        return anchor_centers_hd, anchor_centers_2d, anchor_radii

    def _render_heatmap_contours(self, ax: plt.Axes,
                                  embeddings_2d: np.ndarray,
                                  anchor_centers_2d: np.ndarray,
                                  anchor_registry,
                                  trajectory=None,
                                  boundary_2d: Optional[np.ndarray] = None,
                                  boundary_anchor_map: Optional[list] = None):
        """Render Gaussian heatmap contours on the given axes.

        Draws only the contourf layer (no scatter, no anchor markers).
        """
        pad = 0.15
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= pad * x_range
        x_max += pad * x_range
        y_min -= pad * y_range
        y_max += pad * y_range

        n_grid = 300
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                             np.linspace(y_min, y_max, n_grid))

        K = 50
        data_spread = max(x_range, y_range)
        num_anchors = len(anchor_centers_2d)
        sigmas = np.full(num_anchors, 0.05 * data_spread)

        low_point_anchors = []
        anchor_ids = list(anchor_registry.anchors.keys())
        for anchor_idx, aid in enumerate(anchor_ids):
            real_indices = []
            if trajectory is not None:
                real_indices = [
                    i for i, pt in enumerate(trajectory.points)
                    if getattr(pt, 'anchor_id', None) == aid
                ]
            real_2d = embeddings_2d[real_indices] if real_indices else np.empty((0, 2))

            n_synth = max(0, K - len(real_indices))
            synth_2d = np.empty((0, 2))
            if n_synth > 0 and boundary_2d is not None and boundary_anchor_map is not None:
                surf_indices = [
                    i for i, (a_idx, btype) in enumerate(boundary_anchor_map)
                    if a_idx == anchor_idx and btype == 'outer'
                ]
                if surf_indices:
                    chosen = surf_indices[:n_synth]
                    synth_2d = boundary_2d[chosen]

            combined = np.concatenate([real_2d, synth_2d], axis=0)
            if len(combined) >= 2:
                dists = np.linalg.norm(
                    combined - anchor_centers_2d[anchor_idx], axis=1)
                sigmas[anchor_idx] = np.mean(dists)

            if len(combined) < 5:
                low_point_anchors.append((aid, len(combined)))

        if low_point_anchors:
            logger.warning(
                f"Heatmap: {len(low_point_anchors)} anchor(s) have very few "
                f"points for sigma estimation (no synth fallback in t-SNE mode): "
                f"{low_point_anchors}")

        density = np.zeros(xx.shape)
        for center_2d, sigma in zip(anchor_centers_2d, sigmas):
            dist_sq = (xx - center_2d[0])**2 + (yy - center_2d[1])**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            density = np.maximum(density, blob)

        threshold = 0.05
        density_masked = np.where(density >= threshold, density, np.nan)
        ax.contourf(xx, yy, density_masked, levels=15,
                    cmap='OrRd', alpha=0.7, zorder=2)

    def _render_radius_circles(self, ax: plt.Axes,
                                anchor_centers_2d: np.ndarray,
                                anchor_registry,
                                trajectory=None,
                                embeddings_2d: Optional[np.ndarray] = None,
                                boundary_2d: Optional[np.ndarray] = None,
                                boundary_anchor_map: Optional[list] = None):
        """Draw circles around each anchor at the average projected radius."""
        from matplotlib.patches import Circle

        anchor_ids = list(anchor_registry.anchors.keys())
        circles = []  # collect (center, radius) for union shading

        for anchor_idx, aid in enumerate(anchor_ids):
            center = anchor_centers_2d[anchor_idx]
            radius_2d = None

            # Try boundary points first (PCA mode)
            if boundary_2d is not None and boundary_anchor_map is not None:
                outer_mask = [
                    i for i, (a_idx, btype) in enumerate(boundary_anchor_map)
                    if a_idx == anchor_idx and btype == 'outer'
                ]
                if len(outer_mask) >= 2:
                    pts = boundary_2d[outer_mask]
                    dists = np.linalg.norm(pts - center, axis=1)
                    radius_2d = np.mean(dists)

            # Fallback: use real eval points assigned to this anchor
            if radius_2d is None and trajectory is not None and embeddings_2d is not None:
                eval_indices = [
                    i for i, pt in enumerate(trajectory.points)
                    if getattr(pt, 'anchor_id', None) == aid
                ]
                if len(eval_indices) >= 2:
                    pts = embeddings_2d[eval_indices]
                    dists = np.linalg.norm(pts - center, axis=1)
                    radius_2d = np.mean(dists)

            if radius_2d is None or radius_2d < 1e-8:
                continue

            circles.append((center, radius_2d))

            circle_patch = Circle(center, radius_2d, fill=False,
                                  edgecolor='red', linewidth=2.5, alpha=0.85,
                                  linestyle='-', zorder=6)
            ax.add_patch(circle_patch)

        # Shade the union of all circles (no deepening on overlap)
        if circles:
            all_centers = np.array([c for c, _ in circles])
            all_radii = np.array([r for _, r in circles])
            x_min = (all_centers[:, 0] - all_radii).min()
            x_max = (all_centers[:, 0] + all_radii).max()
            y_min = (all_centers[:, 1] - all_radii).min()
            y_max = (all_centers[:, 1] + all_radii).max()
            pad = max(x_max - x_min, y_max - y_min) * 0.05
            gx = np.linspace(x_min - pad, x_max + pad, 300)
            gy = np.linspace(y_min - pad, y_max + pad, 300)
            xx, yy = np.meshgrid(gx, gy)
            inside = np.zeros_like(xx, dtype=bool)
            for center, radius_2d in circles:
                dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
                inside |= (dist <= radius_2d)
            shade = inside.astype(float)
            ax.contourf(xx, yy, shade, levels=[0.5, 1.5],
                        colors=['red'], alpha=0.08, zorder=5.5)

        return circles

    def _render_anchor_markers(self, ax: plt.Axes,
                                anchor_centers_hd: np.ndarray,
                                anchor_centers_2d: np.ndarray,
                                original_embeddings=None,
                                embeddings_2d: Optional[np.ndarray] = None,
                                marker_zorder: int = 8):
        """Render anchor center star/X markers on the given axes."""
        ref_embeddings = original_embeddings if original_embeddings is not None else embeddings_2d
        if ref_embeddings is None:
            return
        anchor_types = self._classify_anchors(anchor_centers_hd, ref_embeddings)
        for i, (c2d, atype) in enumerate(zip(anchor_centers_2d, anchor_types)):
            if atype == 'data_point':
                ax.scatter(c2d[0], c2d[1], c='red', s=160, marker='*',
                          alpha=0.85, edgecolors='black', linewidths=1.0,
                          label='Anchor (Data Point)' if i == 0 else '',
                          zorder=marker_zorder)
            else:
                ax.scatter(c2d[0], c2d[1], c='darkorange', s=120, marker='X',
                          alpha=0.85, edgecolors='black', linewidths=1.0,
                          label='Anchor (Synthetic)' if i == 0 else '',
                          zorder=marker_zorder)

    def _draw_trajectory_thumbnails(self, ax: plt.Axes,
                                     df: 'pd.DataFrame',
                                     embeddings_2d: np.ndarray,
                                     run_trajectories: Dict[str, np.ndarray],
                                     data_dir: str,
                                     interval: float = 0.1,
                                     thumb_pixels: int = 80):
        """Overlay small image thumbnails at regular intervals along trajectories.

        Uses a two-pass approach: first collects all thumbnails, then resolves
        overlaps by displacing colliding thumbnails and drawing connector lines
        back to the original data points.

        Args:
            ax: Matplotlib axes to draw on
            df: Full dataset dataframe with run_id, filename, index columns
            embeddings_2d: 2D projected embeddings aligned to df rows
            run_trajectories: Dict of run_id -> (N, 2) sorted arrays
            data_dir: Base directory containing image folders
            interval: Fraction of trajectory to space thumbnails (default 0.1 = every 10%)
            thumb_pixels: Thumbnail size in pixels (default 40)
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from PIL import Image
        from pathlib import Path

        data_path = Path(data_dir)

        # Estimate thumbnail bounding box size in data coordinates
        x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
        y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
        fig_w = ax.figure.get_figwidth()
        fig_h = ax.figure.get_figheight()
        dpi = ax.figure.dpi
        box_w = thumb_pixels / (fig_w * dpi) * x_range * 1.5
        box_h = thumb_pixels / (fig_h * dpi) * y_range * 1.5

        # Plot bounds with margin for thumbnails (half a box inset from axes)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bound_x_min = x_min + box_w * 0.5
        bound_x_max = x_max - box_w * 0.5
        bound_y_min = y_min + box_h * 0.5
        bound_y_max = y_max - box_h * 0.5

        def in_bounds(pos):
            return (bound_x_min <= pos[0] <= bound_x_max and
                    bound_y_min <= pos[1] <= bound_y_max)

        def overlaps_any(pos, placed_positions):
            for p in placed_positions:
                if abs(pos[0] - p[0]) < box_w and abs(pos[1] - p[1]) < box_h:
                    return True
            return False

        # Compute data centroid for directional displacement
        centroid_x = embeddings_2d[:, 0].mean()
        centroid_y = embeddings_2d[:, 1].mean()

        def find_clear_position(xy, placed_positions):
            """Find non-overlapping position within plot bounds. Returns None if none found."""
            if in_bounds(xy) and not overlaps_any(xy, placed_positions):
                return (xy[0], xy[1])

            # Primary direction: away from data centroid
            dx = xy[0] - centroid_x
            dy = xy[1] - centroid_y
            outward_angle = np.degrees(np.arctan2(dy, dx))

            # 24 angles: outward first, then fan out in 15-degree increments
            offsets = [0]
            for delta in range(15, 181, 15):
                offsets.extend([delta, -delta])
            angles = [(outward_angle + off) % 360 for off in offsets]

            step = max(box_w, box_h)
            for dist_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
                for angle_deg in angles:
                    rad = np.radians(angle_deg)
                    candidate = (xy[0] + dist_mult * step * np.cos(rad),
                                 xy[1] + dist_mult * step * np.sin(rad))
                    if in_bounds(candidate) and not overlaps_any(candidate, placed_positions):
                        return candidate
            return None  # no good spot, skip this thumbnail

        # --- Pass 1: collect all thumbnail data ---
        thumb_entries = []  # list of (data_xy, img_array)

        for run_id, pts_2d in run_trajectories.items():
            n = len(pts_2d)
            if n == 0:
                continue

            mask = df['run_id'] == run_id
            sub_df = df.loc[mask].copy()
            if 'index' in sub_df.columns:
                sub_df = sub_df.sort_values('index')

            folder_name = run_id.rsplit('_', 1)[0] if '_' in run_id else run_id

            sample_fracs = np.arange(0, 1.0 + interval / 2, interval)
            sample_indices = np.clip(
                (sample_fracs * (n - 1)).astype(int), 0, n - 1)
            sample_indices = list(dict.fromkeys(sample_indices))

            for idx in sample_indices:
                if idx >= len(sub_df):
                    continue

                row = sub_df.iloc[idx]
                filename = row.get('filename', '')
                if not filename:
                    continue

                # Use folder_path from CSV if available (handles nested dirs),
                # otherwise fall back to deriving folder from run_id
                row_folder_path = row.get('folder_path', '')
                if row_folder_path:
                    img_path = Path(row_folder_path) / filename
                else:
                    img_path = data_path / folder_name / filename
                if not img_path.exists():
                    logger.debug(f"Thumbnail not found: {img_path}")
                    continue

                try:
                    img = Image.open(img_path)
                    img.thumbnail((thumb_pixels, thumb_pixels), Image.LANCZOS)
                    img_array = np.array(img)
                    data_xy = pts_2d[idx]
                    thumb_entries.append((data_xy, img_array))
                except Exception as e:
                    logger.debug(f"Failed to load thumbnail {img_path}: {e}")

        # --- Pass 2: resolve overlaps and draw ---
        placed = []  # display positions of already-placed thumbnails

        skipped = 0
        for data_xy, img_array in thumb_entries:
            display_xy = find_clear_position(data_xy, placed)
            if display_xy is None:
                skipped += 1
                continue
            placed.append(display_xy)

            imagebox = OffsetImage(img_array, zoom=1.0)
            imagebox.image.axes = ax

            is_displaced = (abs(display_xy[0] - data_xy[0]) > 1e-6 or
                            abs(display_xy[1] - data_xy[1]) > 1e-6)
            arrow = dict(arrowstyle='-', color='black', lw=1.0,
                         linestyle='dashed', shrinkA=0, shrinkB=0) if is_displaced else None

            ab = AnnotationBbox(imagebox, (data_xy[0], data_xy[1]),
                                xybox=display_xy,
                                xycoords='data', boxcoords='data',
                                frameon=True,
                                pad=0.1,
                                arrowprops=arrow,
                                bboxprops=dict(edgecolor='black',
                                              linewidth=0.5,
                                              alpha=0.9),
                                zorder=10)
            ax.add_artist(ab)

        if skipped > 0:
            logger.info(f"Placed {len(placed)} thumbnails, skipped {skipped} (no clear position)")

    @staticmethod
    def _farthest_point_sample(points_2d: np.ndarray, n: int) -> List[int]:
        """Select n indices from points_2d that are maximally spread out.

        Uses greedy farthest-point sampling: starts with the point farthest
        from the centroid, then iteratively picks the point farthest from
        all already-selected points.
        """
        if n >= len(points_2d):
            return list(range(len(points_2d)))

        centroid = points_2d.mean(axis=0)
        dists_to_centroid = np.linalg.norm(points_2d - centroid, axis=1)
        selected = [int(np.argmax(dists_to_centroid))]

        for _ in range(n - 1):
            # Min distance from each candidate to any selected point
            min_dists = np.full(len(points_2d), np.inf)
            for s in selected:
                d = np.linalg.norm(points_2d - points_2d[s], axis=1)
                min_dists = np.minimum(min_dists, d)
            # Zero out already selected
            for s in selected:
                min_dists[s] = -1
            selected.append(int(np.argmax(min_dists)))

        return selected

    def _draw_anchor_thumbnails(self, ax: plt.Axes,
                                 df: 'pd.DataFrame',
                                 embeddings_hd: np.ndarray,
                                 embeddings_2d: np.ndarray,
                                 anchor_registry,
                                 anchor_centers_2d: np.ndarray,
                                 data_dir: str,
                                 thumbs_per_bubble: int = 1,
                                 num_anchors: Optional[int] = None,
                                 thumb_pixels: int = 80,
                                 traj_splines: Optional[List[np.ndarray]] = None,
                                 radius_circles: Optional[List] = None):
        """Show representative image thumbnails for each anchor bubble.

        For each anchor, finds the images closest to the anchor center in
        high-dimensional embedding space and renders them offset from the
        anchor's 2D position so they don't cover circles/markers or
        trajectory lines. A dashed arrow connector points back to the
        anchor center.

        If num_anchors is set, auto-selects the most spread-out anchors
        using farthest-point sampling in 2D space, then drops any whose
        thumbnails would overlap.

        Args:
            ax: Matplotlib axes to draw on
            df: Full dataset dataframe with anchor_id, filename columns
            embeddings_hd: High-dimensional embeddings aligned to df rows
            embeddings_2d: 2D projected embeddings aligned to df rows
            anchor_registry: Registry of anchors (possibly filtered)
            anchor_centers_2d: Projected anchor center positions (N_anchors, 2)
            data_dir: Fallback directory for image files
            thumbs_per_bubble: Number of representative images per anchor
            num_anchors: Auto-select this many spread-out anchors (None = all)
            thumb_pixels: Thumbnail size in pixels
            traj_splines: Optional list of (M, 2) arrays of spline points
                from rendered trajectories, used to avoid overlap
            radius_circles: Optional list of (center_2d, radius_2d) tuples
                from rendered radius circles, used to avoid overlap
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from PIL import Image
        from pathlib import Path

        if 'anchor_id' not in df.columns:
            logger.warning("No anchor_id column in dataframe, cannot draw anchor thumbnails")
            return

        # Auto-select spread-out anchors if num_anchors is specified
        anchor_ids_list = list(anchor_registry.anchors.keys())
        if num_anchors is not None and num_anchors < len(anchor_ids_list):
            selected_indices = self._farthest_point_sample(
                anchor_centers_2d, num_anchors)
            anchor_ids_list = [anchor_ids_list[i] for i in selected_indices]
            anchor_centers_2d_filtered = anchor_centers_2d[selected_indices]
            logger.info(
                f"Auto-selected {len(anchor_ids_list)} spread-out anchors "
                f"from {anchor_registry.num_anchors} total")
        else:
            anchor_centers_2d_filtered = anchor_centers_2d

        # Estimate thumbnail bounding box in data coordinates
        x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
        y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
        fig_w = ax.figure.get_figwidth()
        fig_h = ax.figure.get_figheight()
        dpi = ax.figure.dpi
        box_w = thumb_pixels / (fig_w * dpi) * x_range * 1.5
        box_h = thumb_pixels / (fig_h * dpi) * y_range * 1.5

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bound_x_min = x_min + box_w * 0.5
        bound_x_max = x_max - box_w * 0.5
        bound_y_min = y_min + box_h * 0.5
        bound_y_max = y_max - box_h * 0.5

        centroid_x = embeddings_2d[:, 0].mean()
        centroid_y = embeddings_2d[:, 1].mean()

        # Subsample trajectory spline points for fast overlap checks
        traj_pts = None
        if traj_splines:
            all_pts = []
            for spline in traj_splines:
                # Every 4th point is enough for overlap detection
                all_pts.append(spline[::4])
            traj_pts = np.concatenate(all_pts, axis=0)

        def in_bounds(pos):
            return (bound_x_min <= pos[0] <= bound_x_max and
                    bound_y_min <= pos[1] <= bound_y_max)

        def overlaps_any(pos, placed_positions):
            for p in placed_positions:
                if abs(pos[0] - p[0]) < box_w and abs(pos[1] - p[1]) < box_h:
                    return True
            return False

        def traj_overlap_count(pos):
            """Count how many trajectory points fall inside the thumbnail box."""
            if traj_pts is None:
                return 0
            inside = ((np.abs(traj_pts[:, 0] - pos[0]) < box_w * 0.5) &
                      (np.abs(traj_pts[:, 1] - pos[1]) < box_h * 0.5))
            return int(inside.sum())

        def circle_overlap(pos):
            """Check if thumbnail box overlaps any radius circle."""
            if not radius_circles:
                return 0
            half_w, half_h = box_w * 0.5, box_h * 0.5
            count = 0
            for c_center, c_radius in radius_circles:
                # Closest point on the box to the circle center
                closest_x = np.clip(c_center[0], pos[0] - half_w, pos[0] + half_w)
                closest_y = np.clip(c_center[1], pos[1] - half_h, pos[1] + half_h)
                dist = np.sqrt((closest_x - c_center[0])**2 +
                               (closest_y - c_center[1])**2)
                if dist <= c_radius:
                    count += 1
            return count

        def find_clear_position(xy, placed_positions):
            """Find a position offset from xy with least overlap."""
            dx = xy[0] - centroid_x
            dy = xy[1] - centroid_y
            outward_angle = np.degrees(np.arctan2(dy, dx))
            offsets = [0]
            for delta in range(15, 181, 15):
                offsets.extend([delta, -delta])
            angles = [(outward_angle + off) % 360 for off in offsets]
            step = max(box_w, box_h)
            best = None
            best_score = float('inf')
            for dist_mult in [1.5, 2.0, 2.5, 3.0]:
                for angle_deg in angles:
                    rad = np.radians(angle_deg)
                    candidate = (xy[0] + dist_mult * step * np.cos(rad),
                                 xy[1] + dist_mult * step * np.sin(rad))
                    if in_bounds(candidate) and not overlaps_any(candidate, placed_positions):
                        # Penalize both trajectory overlap and circle overlap
                        score = traj_overlap_count(candidate) + circle_overlap(candidate) * 50
                        if score == 0:
                            return candidate
                        if score < best_score:
                            best_score = score
                            best = candidate
            return best

        placed = []
        total_placed = 0

        for anchor_idx, aid in enumerate(anchor_ids_list):
            anchor = anchor_registry.anchors[aid]

            # Find dataframe rows assigned to this anchor
            mask = df['anchor_id'] == aid
            if not mask.any():
                # Try float comparison (anchor_id may be stored as float)
                mask = df['anchor_id'].apply(
                    lambda x: pd.notna(x) and int(x) == aid)
            if not mask.any():
                continue

            assigned_indices = df.index[mask].tolist()
            assigned_embeddings = embeddings_hd[mask.values]

            # Rank by distance to anchor center
            center = np.asarray(anchor.center).reshape(1, -1)
            dists = np.linalg.norm(assigned_embeddings - center, axis=1)
            sorted_local = np.argsort(dists)

            # Pick the top N closest
            n_pick = min(thumbs_per_bubble, len(sorted_local))
            selected_local_indices = sorted_local[:n_pick]

            for local_idx in selected_local_indices:
                df_idx = assigned_indices[local_idx]
                row = df.loc[df_idx]
                filename = row.get('filename', '')
                if not filename:
                    continue

                # Resolve image path
                row_folder_path = row.get('folder_path', '')
                if row_folder_path and pd.notna(row_folder_path):
                    img_path = Path(str(row_folder_path)) / filename
                else:
                    img_path = Path(data_dir) / filename
                if not img_path.exists():
                    logger.debug(f"Anchor thumbnail not found: {img_path}")
                    continue

                try:
                    img = Image.open(img_path)
                    img.thumbnail((thumb_pixels, thumb_pixels), Image.LANCZOS)
                    img_array = np.array(img)
                except Exception as e:
                    logger.debug(f"Failed to load anchor thumbnail {img_path}: {e}")
                    continue

                # Place thumbnail offset from the anchor center so it
                # doesn't cover circles/markers
                anchor_2d = anchor_centers_2d_filtered[anchor_idx]
                display_xy = find_clear_position(anchor_2d, placed)
                if display_xy is None:
                    # Could not place without overlap -- skip this one
                    continue

                placed.append(display_xy)

                imagebox = OffsetImage(img_array, zoom=1.0)
                imagebox.image.axes = ax

                # Dashed arrow from thumbnail to anchor center
                arrow = dict(arrowstyle='->', color='black', lw=1.0,
                             linestyle='dashed', shrinkA=5, shrinkB=3)

                ab = AnnotationBbox(imagebox, (anchor_2d[0], anchor_2d[1]),
                                    xybox=display_xy,
                                    xycoords='data', boxcoords='data',
                                    frameon=True,
                                    pad=0.1,
                                    arrowprops=arrow,
                                    bboxprops=dict(edgecolor='black',
                                                  linewidth=0.5,
                                                  alpha=0.9),
                                    zorder=10)
                ax.add_artist(ab)
                total_placed += 1

        logger.info(f"Placed {total_placed} anchor thumbnails for "
                   f"{len(anchor_ids_list)} anchors")

    def _plot_bubble_heatmap(self, ax: plt.Axes, embeddings_2d: np.ndarray,
                             volume_2d: np.ndarray, volume_anchor_map: list,
                             reducer: Any,
                             anchor_registry=None,
                             trajectory=None,
                             boundary_2d: Optional[np.ndarray] = None,
                             boundary_anchor_map: Optional[list] = None,
                             run_trajectories: Optional[Dict[str, np.ndarray]] = None,
                             smoothing: float = 0.1):
        """
        Plot a Gaussian heatmap around each anchor center in 2D PCA space.

        Places a 2D Gaussian blob at each anchor's projected center, sized
        proportionally to its radius. Uses element-wise max across anchors
        to avoid constructive interference between nearby bubbles.

        If run_trajectories is provided, trajectory spline curves are drawn
        on top of the heatmap instead of the default scatter points.
        """
        if run_trajectories is not None:
            # Draw trajectory lines instead of scatter points
            self._draw_trajectory_splines(ax, run_trajectories, linewidth=1.5,
                                          alpha=0.7, add_legend=True,
                                          smoothing=smoothing)
        else:
            # Plot all real data points in background
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.6, s=20, c='steelblue', edgecolors='black',
                      linewidths=0.2, zorder=3)

        if anchor_registry is None:
            anchor_registry = self.pipeline.anchor_registry if self.pipeline else None
        if not anchor_registry or anchor_registry.num_anchors == 0:
            ax.text(0.5, 0.5, 'No anchors placed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            return

        try:
            original_embeddings = None
            if trajectory is not None:
                original_embeddings = trajectory.embeddings_matrix
                if hasattr(original_embeddings, 'cpu'):
                    original_embeddings = original_embeddings.cpu().numpy()

            anchor_centers_hd, anchor_centers_2d, _radii = \
                self._project_anchor_centers(
                    anchor_registry, reducer, embeddings_2d, original_embeddings)

            self._render_heatmap_contours(
                ax, embeddings_2d, anchor_centers_2d, anchor_registry,
                trajectory=trajectory,
                boundary_2d=boundary_2d,
                boundary_anchor_map=boundary_anchor_map)

        except Exception as e:
            logger.warning(f"Failed to create bubble heatmap: {e}")
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.3, s=20, c='gray', zorder=1)
            return

        self._render_anchor_markers(
            ax, anchor_centers_hd, anchor_centers_2d,
            original_embeddings=original_embeddings,
            embeddings_2d=embeddings_2d, marker_zorder=4)

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_anchors_and_hyperspheres(self, ax: plt.Axes, embeddings_2d: np.ndarray,
                                     reducer: Any, show_anchors: bool,
                                     show_hyperspheres: bool,
                                     anchor_registry: Optional[AnchorRegistry] = None,
                                     trajectory: Optional[EmbeddingTrajectory] = None,
                                     boundary_2d: Optional[np.ndarray] = None,
                                     boundary_anchor_map: Optional[List] = None,
                                     run_trajectories: Optional[Dict[str, np.ndarray]] = None,
                                     smoothing: float = 0.1):
        """Plot anchor points and convex hulls around eval points per anchor.

        Draws convex hulls around the actual eval (val + test) points assigned
        to each anchor in the 2D projected space, rather than using synthetic
        boundary points from high-dimensional hyperspheres.

        If run_trajectories is provided, trajectory spline curves are drawn
        instead of the default scatter points.
        """
        from scipy.spatial import ConvexHull

        if run_trajectories is not None:
            # Draw trajectory lines instead of scatter points
            self._draw_trajectory_splines(ax, run_trajectories, linewidth=1.5,
                                          alpha=0.7, add_legend=True,
                                          smoothing=smoothing)
        else:
            # Plot all real data points in background
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.3, s=20, c='lightgray', zorder=1)

        if anchor_registry is None:
            anchor_registry = self.pipeline.anchor_registry if self.pipeline else None

        if not anchor_registry or anchor_registry.num_anchors == 0:
            ax.text(0.5, 0.5, 'No anchors placed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            return

        # Get original high-dim embeddings for anchor classification
        original_embeddings = None
        if trajectory is not None:
            original_embeddings = trajectory.embeddings_matrix
            if hasattr(original_embeddings, 'cpu'):
                original_embeddings = original_embeddings.cpu().numpy()

        # Collect anchor center data
        anchor_centers_hd = []
        for anchor in anchor_registry.anchors.values():
            anchor_centers_hd.append(anchor.center)
        anchor_centers_hd = np.array(anchor_centers_hd)

        # Project anchor centers to 2D
        can_transform = hasattr(reducer, 'transform')
        if can_transform:
            anchor_centers_2d = reducer.transform(anchor_centers_hd)
        else:
            # t-SNE: find nearest data point in high-dim and use its 2D position
            anchor_centers_2d = []
            for center in anchor_centers_hd:
                if original_embeddings is not None:
                    dists = np.linalg.norm(original_embeddings - center.reshape(1, -1), axis=1)
                else:
                    dists = np.linalg.norm(embeddings_2d - center.reshape(1, -1), axis=1)
                anchor_centers_2d.append(embeddings_2d[np.argmin(dists)])
            anchor_centers_2d = np.array(anchor_centers_2d)

        # Plot convex hulls around eval points belonging to each anchor
        if show_hyperspheres and trajectory is not None:
            eval_splits = {DatasetSplit.VALIDATION, DatasetSplit.TEST}
            anchor_ids = list(anchor_registry.anchors.keys())

            for anchor_idx, aid in enumerate(anchor_ids):
                # Collect all eval point indices assigned to this anchor
                eval_indices = [
                    i for i, pt in enumerate(trajectory.points)
                    if getattr(pt, 'anchor_id', None) == aid
                    and pt.dataset_split in eval_splits
                ]
                if len(eval_indices) > 2:
                    eval_pts = embeddings_2d[eval_indices]
                    try:
                        hull = ConvexHull(eval_pts)
                        verts = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(eval_pts[verts, 0], eval_pts[verts, 1],
                               'r-', linewidth=2, alpha=0.7, zorder=3)
                        ax.fill(eval_pts[hull.vertices, 0],
                                eval_pts[hull.vertices, 1],
                                alpha=0.08, color='red', zorder=2)
                    except Exception:
                        pass

                # Optionally draw per-shell hulls (val vs test) if nested shells exist
                anchor = anchor_registry.anchors[aid]
                if anchor.nested_shells:
                    for split in [DatasetSplit.VALIDATION, DatasetSplit.TEST]:
                        shell_indices = [
                            i for i, pt in enumerate(trajectory.points)
                            if getattr(pt, 'anchor_id', None) == aid
                            and pt.dataset_split == split
                        ]
                        if len(shell_indices) > 2:
                            shell_pts = embeddings_2d[shell_indices]
                            try:
                                hull = ConvexHull(shell_pts)
                                verts = np.append(hull.vertices, hull.vertices[0])
                                shell_color = self.colors[split]
                                ax.plot(shell_pts[verts, 0], shell_pts[verts, 1],
                                       color=shell_color, linestyle='--',
                                       linewidth=1, alpha=0.5, zorder=2)
                            except Exception:
                                pass

        # Plot anchor center markers
        if show_anchors:
            anchor_types = self._classify_anchors(
                anchor_centers_hd,
                original_embeddings if original_embeddings is not None else embeddings_2d)

            for i, (center_2d, anchor_type) in enumerate(zip(anchor_centers_2d, anchor_types)):
                if anchor_type == 'data_point':
                    ax.scatter(center_2d[0], center_2d[1],
                             c='red', s=200, marker='*',
                             edgecolors='black', linewidths=2,
                             label='Anchor (Data Point)' if i == 0 else "", zorder=4)
                else:
                    ax.scatter(center_2d[0], center_2d[1],
                             c='orange', s=150, marker='X',
                             edgecolors='black', linewidths=2,
                             label='Anchor (Synthetic)' if i == 0 else "", zorder=4)

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _classify_anchors(self, anchor_centers: np.ndarray,
                         embeddings, threshold: float = 1e-6) -> List[str]:
        """
        Classify anchors as either actual data points or synthetic points.

        Returns:
            List of 'data_point' or 'synthetic' for each anchor
        """
        # Convert torch tensor to numpy if needed
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()

        anchor_types = []

        for anchor_center in anchor_centers:
            # Check if anchor center is very close to any data point
            distances = np.linalg.norm(embeddings - anchor_center.reshape(1, -1), axis=1)
            min_distance = np.min(distances)

            if min_distance < threshold:
                anchor_types.append('data_point')
            else:
                anchor_types.append('synthetic')

        return anchor_types

    def _plot_density_contours(self, ax: plt.Axes, embeddings_2d: np.ndarray):
        """Plot density contours of the embedding distribution."""
        from scipy.stats import gaussian_kde

        # Create density estimation
        try:
            kde = gaussian_kde(embeddings_2d.T)

            # Create grid for contour plot
            x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
            y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()

            xx, yy = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]
            grid_coords = np.array([xx.ravel(), yy.ravel()])
            density = kde(grid_coords).reshape(xx.shape)

            # Plot contours
            contour = ax.contour(xx, yy, density, levels=8, alpha=0.6)
            ax.clabel(contour, inline=True, fontsize=8)

            # Overlay scatter plot
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.5, s=20, c='black')

        except Exception as e:
            logger.warning(f"Failed to create density plot: {e}")
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
            ax.text(0.5, 0.5, 'Density plot failed', transform=ax.transAxes,
                   ha='center', va='center')

    def _plot_split_distributions(self, ax: plt.Axes, embeddings_2d: np.ndarray,
                                assignment_result: DatasetAssignmentResult):
        """Plot histograms showing distribution of splits across dimensions."""

        # Get split indices
        splits = {
            'Train': assignment_result.train_indices,
            'Validation': assignment_result.validation_indices,
            'Test': assignment_result.test_indices,
            'Unassigned': assignment_result.unassigned_indices
        }

        # Plot distribution along first component
        for split_name, indices in splits.items():
            if indices:
                values = embeddings_2d[indices, 0]
                ax.hist(values, alpha=0.6, bins=20, label=f'{split_name} (PC1)',
                       density=True)

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('Density')
        ax.legend()

    def _plot_statistics(self, ax: plt.Axes, assignment_result: DatasetAssignmentResult,
                        trajectory: EmbeddingTrajectory,
                        anchor_registry: Optional[AnchorRegistry] = None):
        """Plot assignment statistics and summary information."""

        # Clear axis and remove ticks
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        # Prepare statistics text
        stats = assignment_result.statistics
        total_points = stats['total_points']

        stats_text = [
            f"Total Data Points: {total_points:,}",
            f"Anchors Placed: {stats.get('num_anchors_used', 0)}",
            "",
            "Dataset Split Summary:",
            f"  Train: {stats['train_count']:,} ({stats['train_ratio']:.1%})",
            f"  Validation: {stats['validation_count']:,} ({stats['validation_ratio']:.1%})",
            f"  Test: {stats['test_count']:,} ({stats['test_ratio']:.1%})",
            f"  Unassigned: {stats['unassigned_count']:,}",
            "",
            f"Eval Ratio: {stats['eval_ratio']:.1%}",
        ]

        # Add anchor statistics if available
        if anchor_registry is None:
            anchor_registry = self.pipeline.anchor_registry if self.pipeline else None

        if anchor_registry and anchor_registry.num_anchors > 0:
            try:
                anchor_stats = anchor_registry.get_coverage_stats()
                stats_text.extend([
                    "",
                    "Anchor Statistics:",
                    f"  Mean Radius: {anchor_stats.get('mean_radius', 0):.4f}",
                    f"  Std Radius: {anchor_stats.get('std_radius', 0):.4f}",
                    f"  Min/Max Radius: {anchor_stats.get('min_radius', 0):.4f} / {anchor_stats.get('max_radius', 0):.4f}"
                ])
            except Exception as e:
                stats_text.extend([
                    "",
                    f"Anchor Statistics: Error loading ({e})"
                ])

        # Display text
        text_content = "\n".join(stats_text)
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes,
               fontfamily='monospace', fontsize=10,
               verticalalignment='top', horizontalalignment='left')

        # Add a border
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False,
                                     transform=ax.transAxes, linewidth=1))

    def create_anchor_analysis(self, embeddings: np.ndarray,
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed analysis of anchor placement and coverage.

        Args:
            embeddings: Full embedding matrix
            figsize: Figure size
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        # Get anchor registry
        anchor_registry = self._load_anchor_registry()
        if not anchor_registry or anchor_registry.num_anchors == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No anchors to analyze', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16)
            return fig

        # Reduce embeddings to 2D
        embeddings_2d, reducer = self._reduce_dimensions(embeddings, "PCA")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Detailed Anchor Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Anchor coverage heatmap
        self._plot_coverage_heatmap(ax1, embeddings_2d)
        ax1.set_title('Coverage Density')

        # Plot 2: Anchor radius distribution
        self._plot_radius_distribution(ax2)
        ax2.set_title('Anchor Radius Distribution')

        # Plot 3: Distance to nearest anchor
        self._plot_distance_to_anchors(ax3, embeddings_2d, reducer)
        ax3.set_title('Distance to Nearest Anchor')

        # Plot 4: Anchor placement quality metrics
        self._plot_placement_metrics(ax4, embeddings)
        ax4.set_title('Placement Quality Metrics')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anchor analysis saved to {save_path}")

        return fig

    def _plot_coverage_heatmap(self, ax: plt.Axes, embeddings_2d: np.ndarray):
        """Plot heatmap showing anchor coverage."""
        # This is a simplified version - could be enhanced with actual coverage calculation
        ax.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, cmap='Blues')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')

    def _plot_radius_distribution(self, ax: plt.Axes):
        """Plot distribution of anchor radii."""
        anchor_registry = self._load_anchor_registry()
        if not anchor_registry:
            ax.text(0.5, 0.5, 'No anchor data available', transform=ax.transAxes,
                   ha='center', va='center')
            return

        radii = [anchor.base_radius for anchor in anchor_registry.anchors.values()]
        total_radii = [anchor.total_radius for anchor in anchor_registry.anchors.values()]

        ax.hist(radii, bins=20, alpha=0.7, label='Base Radius', density=True)
        if any(r != br for r, br in zip(total_radii, radii)):
            ax.hist(total_radii, bins=20, alpha=0.7, label='Total Radius', density=True)

        ax.set_xlabel('Radius')
        ax.set_ylabel('Density')
        ax.legend()

    def _plot_distance_to_anchors(self, ax: plt.Axes, embeddings_2d: np.ndarray, reducer: Any):
        """Plot distance from each point to nearest anchor."""
        # Simplified visualization
        distances = np.random.random(len(embeddings_2d))  # Placeholder
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=distances, cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Distance to Nearest Anchor')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')

    def _plot_placement_metrics(self, ax: plt.Axes, embeddings: np.ndarray):
        """Plot metrics about anchor placement quality."""
        anchor_registry = self._load_anchor_registry()
        if not anchor_registry:
            ax.text(0.5, 0.5, 'No anchor data available', transform=ax.transAxes,
                   ha='center', va='center')
            return

        try:
            anchor_stats = anchor_registry.get_coverage_stats()
        except Exception:
            anchor_stats = {'num_anchors': 0, 'mean_radius': 0, 'std_radius': 0}

        metrics = {
            'Number of Anchors': anchor_stats.get('num_anchors', 0),
            'Mean Radius': anchor_stats.get('mean_radius', 0),
            'Std Radius': anchor_stats.get('std_radius', 0),
            'Coverage': 0.75  # Placeholder
        }

        # Create bar plot
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())

        bars = ax.bar(metrics_names, metrics_values)
        ax.set_ylabel('Value')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')

    def visualize_standalone(self,
                             reduction_method: str = "PCA",
                             figsize: Tuple[int, int] = (14, 10),
                             save_path: Optional[str] = None,
                             show_heatmap: bool = True,
                             show_trajectories: bool = False,
                             show_points: bool = True,
                             point_alpha: float = 0.3,
                             show_thumbnails: bool = False,
                             show_radius_circles: bool = False,
                             show_anchor_thumbnails: bool = False,
                             num_anchor_thumbnails: Optional[int] = None,
                             anchor_thumbs_per_bubble: int = 1,
                             anchor_thumb_pixels: int = 80,
                             thumbnail_interval: float = 0.6,
                             smoothing: float = 0.1,
                             interactive: bool = False,
                             data_dir: Optional[str] = None,
                             anchor_ids: Optional[List] = None) -> plt.Figure:
        """
        Create a standalone single-panel plot with toggleable layers.

        Layers (bottom to top):
        - Heatmap contours (if show_heatmap)
        - Radius circles around anchors (if show_radius_circles)
        - Scatter points colored by run_id (if show_points)
        - Trajectory spline curves per run (if show_trajectories)
        - Anchor center star markers (always)
        - Image thumbnails along trajectories (if show_thumbnails)
        - Anchor-based thumbnails near anchor centers (if show_anchor_thumbnails)

        Args:
            reduction_method: "PCA" or "TSNE"
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            show_heatmap: Whether to show Gaussian heatmap contours
            show_trajectories: Whether to show per-run trajectory curves
            show_points: Whether to show scatter points
            show_thumbnails: Whether to overlay image thumbnails along trajectories
            show_radius_circles: Whether to draw average-radius circles around anchors
            show_anchor_thumbnails: Whether to show representative thumbnails
                per anchor bubble (one image closest to each anchor center)
            num_anchor_thumbnails: Auto-select this many spread-out anchors
                for thumbnails using farthest-point sampling. If None, shows
                all anchors (or those in anchor_ids).
            anchor_thumbs_per_bubble: Number of representative thumbnails per
                anchor bubble (default 1)
            anchor_thumb_pixels: Size of anchor thumbnails in pixels (default 80)
            thumbnail_interval: Fraction interval for thumbnail sampling (default 0.1)
            smoothing: Spline smoothing factor
            interactive: Whether to add hover annotations on points
            data_dir: Directory containing image folders (for thumbnails)
            anchor_ids: Optional list of anchor IDs to render. If None,
                all anchors are rendered. Use this to cherry-pick specific
                bubbles for visualization.

        Returns:
            matplotlib Figure object
        """
        if self.mode != "multi_run" or self.data_root is None:
            raise RuntimeError("visualize_standalone requires multi_run mode")

        # --- Load data ---
        df = self._load_full_dataset()
        embeddings = self._load_all_embeddings(df)
        anchor_registry = self._load_anchor_registry()

        # --- Filter anchors if cherry-picking ---
        if anchor_ids is not None and anchor_registry is not None:
            filtered_reg = AnchorRegistry()
            for aid in anchor_ids:
                if aid in anchor_registry.anchors:
                    filtered_reg.anchors[aid] = anchor_registry.anchors[aid]
            logger.info(
                f"Cherry-picked {filtered_reg.num_anchors}/"
                f"{anchor_registry.num_anchors} anchors: {anchor_ids}")
            anchor_registry = filtered_reg

        # --- Generate boundary points for heatmap sigma computation ---
        # For t-SNE: skip synth points to avoid distorting the layout;
        # sigmas are computed from real assigned points only.
        boundary_points_hd = None
        boundary_2d = None
        boundary_anchor_map = None

        if (show_heatmap or show_radius_circles) and reduction_method.upper() != "TSNE" \
                and anchor_registry and anchor_registry.num_anchors > 0:
            boundary_points_hd, boundary_anchor_map = \
                self._generate_all_boundary_points(anchor_registry, n_samples=50)
            if len(boundary_points_hd) == 0:
                boundary_points_hd = None
                boundary_anchor_map = None

        # --- Dimensionality reduction ---
        if reduction_method.upper() == "PCA":
            embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)
            if boundary_points_hd is not None:
                boundary_2d = reducer.transform(boundary_points_hd)
        else:
            embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)

        # --- Build trajectory if needed ---
        trajectory = self._create_trajectory_from_dataframe(df, embeddings)

        # --- Pre-compute run trajectories ---
        run_trajectories = None
        if show_trajectories or show_thumbnails:
            run_trajectories = self._compute_run_trajectories(df, embeddings_2d)

        # --- Build color map and label map for runs ---
        # Skip red indices (3 in tab10, 6-7 in tab20) so they don't
        # clash with the red radius-circle outlines.
        palette = plt.cm.get_cmap('tab10')
        red_indices = {3}
        run_ids = df['run_id'].unique()
        if len(run_ids) > 10:
            palette = plt.cm.get_cmap('tab20')
            red_indices = {6, 7}
        run_colors = {}
        color_idx = 0
        for rid in run_ids:
            while color_idx % palette.N in red_indices:
                color_idx += 1
            run_colors[rid] = palette(color_idx % palette.N)
            color_idx += 1

        # Build run_id -> legend label. Default to folder name from run_id
        # (strip timestamp). If those aren't unique, use folder_path to
        # trim shared prefix/suffix and produce distinguishing labels.
        run_labels = {}
        default_labels = {}
        for rid in run_ids:
            parts = rid.rsplit('_', 1)
            default_labels[rid] = parts[0] if len(parts) == 2 else rid

        if len(set(default_labels.values())) == len(run_ids):
            run_labels = default_labels
        elif 'folder_path' in df.columns:
            folder_paths = []
            for rid in run_ids:
                fp = df.loc[df['run_id'] == rid, 'folder_path'].iloc[0]
                folder_paths.append(str(fp).replace('\\', '/') if pd.notna(fp) else rid)
            split = [p.split('/') for p in folder_paths]
            prefix_len = 0
            for parts in zip(*split):
                if len(set(parts)) == 1:
                    prefix_len += 1
                else:
                    break
            suffix_len = 0
            for parts in zip(*[s[::-1] for s in split]):
                if len(set(parts)) == 1:
                    suffix_len += 1
                else:
                    break
            end = len(split[0]) - suffix_len if suffix_len else len(split[0])
            for rid, s in zip(run_ids, split):
                label = '/'.join(s[prefix_len:end])
                run_labels[rid] = label if label else default_labels[rid]
        else:
            run_labels = default_labels

        # --- Project anchor centers once (shared by heatmap, markers, anchor thumbs) ---
        anchor_centers_hd = None
        anchor_centers_2d = None
        if anchor_registry and anchor_registry.num_anchors > 0:
            try:
                anchor_centers_hd, anchor_centers_2d, _ = \
                    self._project_anchor_centers(
                        anchor_registry, reducer, embeddings_2d,
                        embeddings)
            except Exception as e:
                logger.warning(f"Failed to project anchor centers: {e}")

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=figsize)

        # Layer 1: Heatmap contours
        if show_heatmap and anchor_centers_2d is not None:
            try:
                self._render_heatmap_contours(
                    ax, embeddings_2d, anchor_centers_2d, anchor_registry,
                    trajectory=trajectory,
                    boundary_2d=boundary_2d,
                    boundary_anchor_map=boundary_anchor_map)
            except Exception as e:
                logger.warning(f"Failed to render heatmap: {e}")

        # Layer 1.5: Radius circles around anchors
        radius_circles = []
        if show_radius_circles and anchor_registry and anchor_registry.num_anchors > 0:
            try:
                # Re-use anchor_centers_2d if heatmap already computed them,
                # otherwise project now.
                if not (show_heatmap and anchor_registry):
                    anchor_centers_hd, anchor_centers_2d, _ = \
                        self._project_anchor_centers(
                            anchor_registry, reducer, embeddings_2d,
                            embeddings)
                radius_circles = self._render_radius_circles(
                    ax, anchor_centers_2d, anchor_registry,
                    trajectory=trajectory,
                    embeddings_2d=embeddings_2d,
                    boundary_2d=boundary_2d,
                    boundary_anchor_map=boundary_anchor_map)
            except Exception as e:
                logger.warning(f"Failed to render radius circles: {e}")

        # Layer 2: Scatter points (batched per run for legend)
        if show_points:
            # Add legend entries from scatter only if trajectories won't add them
            add_scatter_legend = not show_trajectories
            for rid in run_ids:
                mask = df['run_id'] == rid
                pts = embeddings_2d[mask.values]
                label = run_labels[rid]
                ax.scatter(pts[:, 0], pts[:, 1],
                          c=[run_colors[rid]], s=20, alpha=point_alpha,
                          edgecolors='none', zorder=3,
                          label=label if add_scatter_legend else None)

        # Layer 3: Trajectory splines
        traj_splines = None
        if show_trajectories and run_trajectories is not None:
            traj_splines = self._draw_trajectory_splines(
                ax, run_trajectories, colors=run_colors,
                labels=run_labels, smoothing=smoothing)

        # Layer 4: Anchor markers (always)
        if anchor_centers_2d is not None:
            try:
                self._render_anchor_markers(
                    ax, anchor_centers_hd, anchor_centers_2d,
                    original_embeddings=embeddings,
                    embeddings_2d=embeddings_2d)
            except Exception as e:
                logger.warning(f"Failed to render anchor markers: {e}")

        # Layer 5: Thumbnails (data_dir is optional fallback; folder_path from CSV is primary)
        if show_thumbnails and run_trajectories is not None:
            self._draw_trajectory_thumbnails(
                ax, df, embeddings_2d, run_trajectories, data_dir or ".",
                interval=thumbnail_interval)

        # Pad axis limits to create margin space for thumbnail placement
        if show_anchor_thumbnails:
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_pad = (x_hi - x_lo) * 0.15
            y_pad = (y_hi - y_lo) * 0.15
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Layer 6: Anchor-based thumbnails (representative images per bubble)
        if show_anchor_thumbnails and anchor_centers_2d is not None:
            try:
                self._draw_anchor_thumbnails(
                    ax, df, embeddings, embeddings_2d,
                    anchor_registry, anchor_centers_2d,
                    data_dir or ".",
                    thumbs_per_bubble=anchor_thumbs_per_bubble,
                    num_anchors=num_anchor_thumbnails,
                    thumb_pixels=anchor_thumb_pixels,
                    traj_splines=traj_splines,
                    radius_circles=radius_circles)
            except Exception as e:
                logger.warning(f"Failed to render anchor thumbnails: {e}")

        # Interactive hover
        if interactive and show_points:
            point_data = []
            points_xy = []
            for i, (idx, row) in enumerate(df.iterrows()):
                points_xy.append([embeddings_2d[i, 0], embeddings_2d[i, 1]])
                point_data.append({
                    'x': embeddings_2d[i, 0],
                    'y': embeddings_2d[i, 1],
                    'filename': row.get('filename', ''),
                    'run_id': row['run_id'],
                    'split': row.get('dataset_split', ''),
                    'index': int(row.get('index', i)),
                })
            points_xy = np.array(points_xy)

            annotation = ax.annotate(
                '', xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"))
            annotation.set_visible(False)

            def on_hover(event):
                if event.inaxes != ax:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                threshold = 0.03 * max(x_range, y_range)
                cursor = np.array([event.xdata, event.ydata])
                dists = np.linalg.norm(points_xy - cursor, axis=1)
                ci = int(np.argmin(dists))
                if dists[ci] < threshold:
                    p = point_data[ci]
                    text = (f"File: {p['filename']}\n"
                            f"Run: {p['run_id']}\n"
                            f"Split: {p['split']}\n"
                            f"Index: {p['index']}")
                    annotation.xy = (p['x'], p['y'])
                    annotation.set_text(text)
                    annotation.set_visible(True)
                else:
                    annotation.set_visible(False)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            fig.text(0.01, 0.01, "Hover over points to see details",
                     fontsize=8, alpha=0.6)

        # Labels and legend
        method_name = reduction_method.upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')

        parts = ['BubbleFence Visualization']
        if show_heatmap:
            parts.append('Heatmap')
        if show_radius_circles:
            parts.append('Radius Circles')
        if show_trajectories:
            parts.append('Trajectories')
        ax.set_title(' + '.join(parts) + f' ({method_name})')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Standalone plot saved to {save_path}")

        return fig

    def visualize_standalone_3d(self,
                                 reduction_method: str = "PCA",
                                 save_path: Optional[str] = None,
                                 show_heatmap: bool = True,
                                 show_trajectories: bool = False,
                                 show_points: bool = True,
                                 smoothing: float = 0.1,
                                 clean: bool = False,
                                 **kwargs):
        """
        Create an interactive 3D plot using plotly.

        Layers:
        - Heatmap scatter clouds (if show_heatmap)
        - Scatter points colored by run_id (if show_points)
        - Trajectory spline curves per run (if show_trajectories)
        - Anchor center markers (always)

        Args:
            reduction_method: "PCA" or "TSNE"
            save_path: Path to save as .html (interactive) or .png (static)
            show_heatmap: Whether to show heatmap scatter clouds
            show_trajectories: Whether to show per-run trajectory curves
            show_points: Whether to show scatter points
            smoothing: Spline smoothing factor

        Returns:
            plotly Figure object
        """
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors

        if self.mode != "multi_run" or self.data_root is None:
            raise RuntimeError("visualize_standalone_3d requires multi_run mode")

        # --- Load data ---
        df = self._load_full_dataset()
        embeddings = self._load_all_embeddings(df)
        anchor_registry = self._load_anchor_registry()

        # --- Dimensionality reduction to 3D ---
        embeddings_3d, reducer = self._reduce_dimensions(
            embeddings, reduction_method, n_components=3)

        # --- Build trajectory if needed for heatmap sigma ---
        trajectory = self._create_trajectory_from_dataframe(df, embeddings)

        # --- Build color map and label map for runs ---
        palette = plt.cm.get_cmap('tab10')
        run_ids = df['run_id'].unique()
        if len(run_ids) > 10:
            palette = plt.cm.get_cmap('tab20')
        # Skip tab10 index 3 (red) to avoid clashing with anchor/bubble colors
        run_colors = {}
        skip = {3}  # tab10 red
        color_idx = 0
        for rid in run_ids:
            while color_idx in skip:
                color_idx += 1
            rgba = palette(color_idx % palette.N)
            run_colors[rid] = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
            color_idx += 1

        run_labels = {}
        default_labels = {}
        for rid in run_ids:
            parts = rid.rsplit('_', 1)
            default_labels[rid] = parts[0] if len(parts) == 2 else rid

        if len(set(default_labels.values())) == len(run_ids):
            run_labels = default_labels
        elif 'folder_path' in df.columns:
            folder_paths = []
            for rid in run_ids:
                fp = df.loc[df['run_id'] == rid, 'folder_path'].iloc[0]
                folder_paths.append(str(fp).replace('\\', '/') if pd.notna(fp) else rid)
            split_paths = [p.split('/') for p in folder_paths]
            prefix_len = 0
            for parts in zip(*split_paths):
                if len(set(parts)) == 1:
                    prefix_len += 1
                else:
                    break
            suffix_len = 0
            for parts in zip(*[s[::-1] for s in split_paths]):
                if len(set(parts)) == 1:
                    suffix_len += 1
                else:
                    break
            end = len(split_paths[0]) - suffix_len if suffix_len else len(split_paths[0])
            for rid, s in zip(run_ids, split_paths):
                label = '/'.join(s[prefix_len:end])
                run_labels[rid] = label if label else default_labels[rid]
        else:
            run_labels = default_labels

        fig = go.Figure()

        # Layer 1: Heatmap scatter clouds
        if show_heatmap and anchor_registry and anchor_registry.num_anchors > 0:
            try:
                anchor_centers_hd, anchor_centers_3d, _ = \
                    self._project_anchor_centers(
                        anchor_registry, reducer, embeddings_3d, embeddings)
                anchor_ids = list(anchor_registry.anchors.keys())
                data_spread = np.ptp(embeddings_3d, axis=0).max()
                rng = np.random.RandomState(42)

                all_cloud_x, all_cloud_y, all_cloud_z = [], [], []
                for anchor_idx, aid in enumerate(anchor_ids):
                    real_indices = [
                        i for i, pt in enumerate(trajectory.points)
                        if getattr(pt, 'anchor_id', None) == aid
                    ] if trajectory else []
                    if len(real_indices) >= 2:
                        real_3d = embeddings_3d[real_indices]
                        dists = np.linalg.norm(
                            real_3d - anchor_centers_3d[anchor_idx], axis=1)
                        sigma = np.mean(dists)
                    else:
                        sigma = 0.05 * data_spread
                    center = anchor_centers_3d[anchor_idx]
                    cloud = rng.normal(loc=center, scale=sigma, size=(200, 3))
                    all_cloud_x.extend(cloud[:, 0])
                    all_cloud_y.extend(cloud[:, 1])
                    all_cloud_z.extend(cloud[:, 2])

                fig.add_trace(go.Scatter3d(
                    x=all_cloud_x, y=all_cloud_y, z=all_cloud_z,
                    mode='markers',
                    marker=dict(size=3, color='red', opacity=0.04),
                    name='Heatmap', showlegend=True,
                    hoverinfo='skip'
                ))
            except Exception as e:
                logger.warning(f"Failed to render 3D heatmap: {e}")

        # Layer 2: Scatter points
        if show_points:
            for rid in run_ids:
                mask = df['run_id'] == rid
                pts = embeddings_3d[mask.values]
                sub_df = df.loc[mask]
                hover_text = [
                    f"{row.get('filename', '')}<br>split: {row.get('dataset_split', '')}"
                    f"<br>anchor: {row.get('anchor_id', '')}"
                    for _, row in sub_df.iterrows()
                ]
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=run_colors[rid], opacity=0.4),
                    name=run_labels[rid],
                    text=hover_text, hoverinfo='text',
                    showlegend=not show_trajectories
                ))

        # Layer 3: Trajectory splines
        if show_trajectories:
            from scipy.interpolate import splprep, splev
            run_trajectories = self._compute_run_trajectories(df, embeddings_3d)
            for run_id, pts_3d in run_trajectories.items():
                color = run_colors[run_id]
                label = run_labels[run_id]
                n = len(pts_3d)

                if n < 4:
                    sx, sy, sz = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
                else:
                    x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
                    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
                    arc = np.concatenate([[0], np.cumsum(
                        np.sqrt(dx**2 + dy**2 + dz**2))])
                    if arc[-1] < 1e-12:
                        continue
                    min_smooth = 2.0 if reduction_method.upper() == "TSNE" else 0.5
                    s_param = arc[-1] * max(smoothing, min_smooth)
                    try:
                        tck, _u = splprep([x, y, z], u=arc, s=s_param, k=3)
                        u_fine = np.linspace(arc[0], arc[-1], 1500)
                        sx, sy, sz = splev(u_fine, tck)
                    except Exception:
                        sx, sy, sz = x, y, z

                # Arrowhead via cone at the end
                fig.add_trace(go.Scatter3d(
                    x=np.asarray(sx), y=np.asarray(sy), z=np.asarray(sz),
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=label, showlegend=True,
                    hoverinfo='name'
                ))
                # Arrow cone at trajectory end
                if len(sx) > 20:
                    dx_a = float(sx[-1] - sx[-20])
                    dy_a = float(sy[-1] - sy[-20])
                    dz_a = float(sz[-1] - sz[-20])
                    fig.add_trace(go.Cone(
                        x=[float(sx[-1])], y=[float(sy[-1])], z=[float(sz[-1])],
                        u=[dx_a], v=[dy_a], w=[dz_a],
                        sizemode='absolute', sizeref=0.06 * np.ptp(embeddings_3d, axis=0).max(),
                        colorscale=[[0, color], [1, color]],
                        showscale=False, showlegend=False,
                        hoverinfo='skip'
                    ))

        # Layer 4: Anchor markers
        if anchor_registry and anchor_registry.num_anchors > 0:
            try:
                anchor_centers_hd, anchor_centers_3d, _ = \
                    self._project_anchor_centers(
                        anchor_registry, reducer, embeddings_3d, embeddings)
                # Anchor center markers (small points)
                fig.add_trace(go.Scatter3d(
                    x=anchor_centers_3d[:, 0],
                    y=anchor_centers_3d[:, 1],
                    z=anchor_centers_3d[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='red', opacity=0.9),
                    name='Anchors', showlegend=True,
                    hoverinfo='name',
                ))

                # Wireframe spheres showing average reach of each anchor
                # in the projected 3D space
                anchor_ids = list(anchor_registry.anchors.keys())
                u_s = np.linspace(0, 2 * np.pi, 24)
                v_s = np.linspace(0, np.pi, 16)
                unit_x = np.outer(np.cos(u_s), np.sin(v_s))
                unit_y = np.outer(np.sin(u_s), np.sin(v_s))
                unit_z = np.outer(np.ones_like(u_s), np.cos(v_s))

                for anchor_idx, aid in enumerate(anchor_ids):
                    # Compute radius from real assigned points in 3D space
                    assigned = [
                        i for i, pt in enumerate(trajectory.points)
                        if getattr(pt, 'anchor_id', None) == aid
                    ] if trajectory else []
                    if len(assigned) < 2:
                        continue
                    pts_a = embeddings_3d[assigned]
                    center = anchor_centers_3d[anchor_idx]
                    radius = np.mean(np.linalg.norm(pts_a - center, axis=1))

                    # Draw wireframe rings (longitude + latitude lines)
                    # All share legendgroup so one toggle controls all bubbles
                    is_first_bubble = (anchor_idx == 0)
                    for row in range(unit_x.shape[0]):
                        show = is_first_bubble and row == 0
                        fig.add_trace(go.Scatter3d(
                            x=center[0] + radius * unit_x[row, :],
                            y=center[1] + radius * unit_y[row, :],
                            z=center[2] + radius * unit_z[row, :],
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0.35)', width=2),
                            legendgroup='bubbles',
                            showlegend=show,
                            name='Bubble radius' if show else '',
                            hoverinfo='skip',
                        ))
                    for col in range(unit_x.shape[1]):
                        fig.add_trace(go.Scatter3d(
                            x=center[0] + radius * unit_x[:, col],
                            y=center[1] + radius * unit_y[:, col],
                            z=center[2] + radius * unit_z[:, col],
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0.35)', width=2),
                            legendgroup='bubbles',
                            showlegend=False,
                            hoverinfo='skip',
                        ))
            except Exception as e:
                logger.warning(f"Failed to render 3D anchor markers: {e}")

        method_name = reduction_method.upper()
        title_parts = ['BubbleFence 3D']
        if show_heatmap:
            title_parts.append('Heatmap')
        if show_trajectories:
            title_parts.append('Trajectories')

        # --- Opacity sliders ---
        # Categorize traces by their role for slider targeting
        point_indices = []
        traj_indices = []
        cone_indices = []
        sphere_indices = []
        heatmap_indices = []
        for i, trace in enumerate(fig.data):
            name = getattr(trace, 'name', '') or ''
            lg = getattr(trace, 'legendgroup', '') or ''
            if name == 'Heatmap':
                heatmap_indices.append(i)
            elif lg == 'bubbles' or name == 'Bubble radius':
                sphere_indices.append(i)
            elif name == 'Anchors':
                pass  # anchor dots, no slider
            elif isinstance(trace, go.Cone):
                cone_indices.append(i)
            elif isinstance(trace, go.Scatter3d):
                if getattr(trace, 'mode', '') == 'lines':
                    traj_indices.append(i)
                elif getattr(trace, 'mode', '') == 'markers':
                    point_indices.append(i)

        sliders = []
        slider_y_start = -0.08
        opacity_steps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

        def _make_slider(label, trace_idxs, default_op, y_pos):
            if not trace_idxs:
                return None
            steps = []
            for op in opacity_steps:
                steps.append(dict(
                    method='restyle',
                    args=[{'opacity': [op] * len(trace_idxs)}, trace_idxs],
                    label=str(op),
                ))
            active = min(range(len(opacity_steps)),
                         key=lambda j: abs(opacity_steps[j] - default_op))
            return dict(
                active=active,
                currentvalue=dict(prefix=f'{label}: ', font=dict(size=12)),
                pad=dict(t=10),
                y=y_pos, yanchor='top',
                x=0.0, xanchor='left', len=0.4,
                steps=steps,
            )

        # Build sliders for each group
        if point_indices:
            s = _make_slider('Points', point_indices, 0.4, slider_y_start)
            if s:
                sliders.append(s)

        if sphere_indices:
            s = _make_slider('Bubbles', sphere_indices, 0.15,
                             slider_y_start - 0.08)
            if s:
                sliders.append(s)

        if traj_indices or cone_indices:
            all_traj = traj_indices + cone_indices
            s = _make_slider('Trajectories', all_traj, 1.0,
                             slider_y_start - 0.16)
            if s:
                sliders.append(s)

        if heatmap_indices:
            s = _make_slider('Heatmap', heatmap_indices, 0.04,
                             slider_y_start - 0.24)
            if s:
                sliders.append(s)

        layout_kwargs = dict(
            width=1800, height=1200,
            legend=dict(x=1.02, y=1, font=dict(size=10)),
        )
        if sliders:
            layout_kwargs['sliders'] = sliders
            # Add bottom margin for sliders
            layout_kwargs['margin'] = dict(b=max(120, 60 + 40 * len(sliders)))

        if clean:
            hidden_axis = dict(
                showgrid=False, showticklabels=False,
                title='', showline=False, zeroline=False,
                showbackground=False,
            )
            layout_kwargs['title'] = ''
            layout_kwargs['scene'] = dict(
                xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis)
        else:
            layout_kwargs['title'] = ' + '.join(title_parts) + f' ({method_name})'
            layout_kwargs['scene'] = dict(
                xaxis_title=f'{method_name} 1',
                yaxis_title=f'{method_name} 2',
                zaxis_title=f'{method_name} 3',
            )

        fig.update_layout(**layout_kwargs)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                # Save both static image and interactive HTML
                html_path = save_path.rsplit('.', 1)[0] + '.html'
                fig.write_html(html_path)
                try:
                    fig.write_image(save_path)
                except Exception:
                    logger.info(f"Static image export requires kaleido; "
                                f"saved interactive HTML to {html_path}")
            logger.info(f"3D plot saved to {save_path}")

        return fig

    def _compute_run_trajectories(self, df: pd.DataFrame,
                                   embeddings_2d: np.ndarray
                                   ) -> Dict[str, np.ndarray]:
        """
        Group 2D embeddings by run_id and return ordered arrays per run.

        Each run's points are sorted by the within-run 'index' column so the
        spline follows the original temporal order of the images.

        Args:
            df: Full dataset dataframe (must contain 'run_id' and 'index' cols)
            embeddings_2d: 2D projected embeddings aligned to df rows

        Returns:
            Dict mapping run_id -> ndarray of shape (n_points_in_run, 2)
        """
        run_groups: Dict[str, np.ndarray] = {}
        for run_id in df['run_id'].unique():
            mask = df['run_id'] == run_id
            sub_df = df.loc[mask].copy()
            sub_emb = embeddings_2d[mask.values]
            # Sort by within-run index to preserve temporal order
            if 'index' in sub_df.columns:
                order = sub_df['index'].astype(int).argsort()
                sub_emb = sub_emb[order]
            run_groups[run_id] = sub_emb
        return run_groups

    def _draw_trajectory_splines(self, ax: plt.Axes,
                                  run_trajectories: Dict[str, np.ndarray],
                                  colors: Optional[Dict[str, str]] = None,
                                  labels: Optional[Dict[str, str]] = None,
                                  linewidth: float = 2.0,
                                  alpha: float = 0.75,
                                  add_arrows: bool = True,
                                  add_legend: bool = True,
                                  smoothing: float = 0.1) -> List[np.ndarray]:
        """
        Draw smooth spline curves for each run on the given axes.

        Args:
            ax: Matplotlib axes to draw on
            run_trajectories: Dict of run_id -> (N, 2) arrays from
                              _compute_run_trajectories()
            colors: Optional dict mapping run_id -> color string.
                    If None, auto-assigns from a qualitative palette.
            labels: Optional dict mapping run_id -> legend label string.
                    If None, derives label from run_id by stripping timestamp.
            linewidth: Width of the trajectory lines
            alpha: Opacity of the trajectory lines
            add_arrows: Whether to add an arrowhead at the end of each curve
            add_legend: Whether to add a legend entry for each run
            smoothing: Multiplier on arc length for the spline smoothing factor.
                       Higher = smoother/simpler curves, lower = more detailed.
                       0 = interpolate through every point. Default 0.5.

        Returns:
            List of (M, 2) arrays of rendered spline points per run.
        """
        from scipy.interpolate import splprep, splev
        all_splines: List[np.ndarray] = []

        if colors is None:
            # Use a qualitative colormap with enough distinct colours
            palette = plt.cm.get_cmap('tab10')
            n_runs = len(run_trajectories)
            if n_runs > 10:
                palette = plt.cm.get_cmap('tab20')
            colors = {}
            for i, run_id in enumerate(run_trajectories):
                colors[run_id] = palette(i % palette.N)

        for run_id, pts_2d in run_trajectories.items():
            color = colors[run_id]
            n = len(pts_2d)

            # Use provided label or derive from run_id
            if labels and run_id in labels:
                label = labels[run_id]
            else:
                parts = run_id.rsplit('_', 1)
                label = parts[0] if len(parts) == 2 else run_id

            if n < 4:
                # Too few points for cubic spline; just draw straight segments
                ax.plot(pts_2d[:, 0], pts_2d[:, 1], color=color, lw=linewidth,
                        alpha=alpha, label=label if add_legend else None,
                        solid_capstyle='round', zorder=5)
                continue

            x, y = pts_2d[:, 0], pts_2d[:, 1]
            dx, dy = np.diff(x), np.diff(y)
            arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])

            if arc[-1] < 1e-12:
                continue  # degenerate (all points overlap)

            s = arc[-1] * smoothing
            try:
                tck, _u = splprep([x, y], u=arc, s=s, k=3)
            except Exception:
                # Fallback: just connect the dots
                ax.plot(x, y, color=color, lw=linewidth, alpha=alpha,
                        label=label if add_legend else None,
                        solid_capstyle='round', zorder=5)
                continue

            u_fine = np.linspace(arc[0], arc[-1], 800)
            sx, sy = splev(u_fine, tck)

            ax.plot(sx, sy, color=color, lw=linewidth, alpha=alpha,
                    label=label if add_legend else None,
                    solid_capstyle='round', zorder=5)

            # Collect rendered spline points for downstream overlap checks
            all_splines.append(np.column_stack([sx, sy]))

            # Arrowhead only at the spline tip, oriented along the curve.
            if add_arrows and len(sx) > 5:
                # Direction from the last few samples
                dx = sx[-1] - sx[-5]
                dy = sy[-1] - sy[-5]
                angle = np.arctan2(dy, dx)
                from matplotlib.markers import MarkerStyle
                import matplotlib.transforms as mtransforms
                marker = MarkerStyle('>')
                marker._transform = (marker._transform
                                     + mtransforms.Affine2D().rotate(angle))
                ax.scatter(sx[-1], sy[-1], marker=marker, s=70,
                           color=color, zorder=6, edgecolors='none')

        return all_splines

    def visualize_all_trajectories(self,
                                    reduction_method: str = "PCA",
                                    figsize: Tuple[int, int] = (12, 9),
                                    save_path: Optional[str] = None,
                                    interactive: bool = False,
                                    smoothing: float = 0.1) -> plt.Figure:
        """
        Plot smooth trajectory curves for every run_id on a single plot.

        Each run_id gets its own distinct colour so overlaps and differences
        between ingestion runs are clearly visible.

        Args:
            reduction_method: "PCA" or "TSNE"
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            interactive: If True, also scatter all points with hover
                annotations showing filename, run_id, and split info.

        Returns:
            matplotlib Figure object
        """
        if self.mode != "multi_run" or self.data_root is None:
            raise RuntimeError("visualize_all_trajectories requires multi_run mode")

        df = self._load_full_dataset()
        embeddings = self._load_all_embeddings(df)
        embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)

        run_trajectories = self._compute_run_trajectories(df, embeddings_2d)
        logger.info(f"Plotting trajectories for {len(run_trajectories)} runs")

        # Build per-run colour map so points and lines share colours
        palette = plt.cm.get_cmap('tab10')
        n_runs = len(run_trajectories)
        if n_runs > 10:
            palette = plt.cm.get_cmap('tab20')
        run_colors = {}
        for i, run_id in enumerate(run_trajectories):
            run_colors[run_id] = palette(i % palette.N)

        fig, ax = plt.subplots(figsize=figsize)

        # Draw trajectory splines
        self._draw_trajectory_splines(ax, run_trajectories, colors=run_colors,
                                      smoothing=smoothing)

        # Plot anchor centers as star markers
        anchor_registry = self._load_anchor_registry()
        if anchor_registry and anchor_registry.num_anchors > 0:
            anchor_centers_hd = np.array([
                a.center for a in anchor_registry.anchors.values()])
            can_transform = hasattr(reducer, 'transform')
            if can_transform:
                anchor_centers_2d = reducer.transform(anchor_centers_hd)
            else:
                # t-SNE fallback: find nearest data point
                anchor_centers_2d = []
                for center in anchor_centers_hd:
                    dists = np.linalg.norm(embeddings - center.reshape(1, -1), axis=1)
                    anchor_centers_2d.append(embeddings_2d[np.argmin(dists)])
                anchor_centers_2d = np.array(anchor_centers_2d)

            anchor_types = self._classify_anchors(anchor_centers_hd, embeddings)
            for i, (c2d, atype) in enumerate(zip(anchor_centers_2d, anchor_types)):
                if atype == 'data_point':
                    ax.scatter(c2d[0], c2d[1], c='red', s=160, marker='*',
                              alpha=0.85, edgecolors='black', linewidths=1.0,
                              label='Anchor (Data Point)' if i == 0 else '',
                              zorder=8)
                else:
                    ax.scatter(c2d[0], c2d[1], c='darkorange', s=120, marker='X',
                              alpha=0.85, edgecolors='black', linewidths=1.0,
                              label='Anchor (Synthetic)' if i == 0 else '',
                              zorder=8)

        # Always scatter all points at low alpha so curves can be seen in context
        point_data = []
        for i, (idx, row) in enumerate(df.iterrows()):
            run_id = row['run_id']
            color = run_colors.get(run_id, 'gray')
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                      c=[color], s=20, alpha=0.3,
                      edgecolors='none', zorder=3)
            if interactive:
                point_data.append({
                    'x': embeddings_2d[i, 0],
                    'y': embeddings_2d[i, 1],
                    'filename': row.get('filename', ''),
                    'run_id': run_id,
                    'split': row.get('dataset_split', ''),
                    'index': int(row.get('index', i)),
                })

        if interactive:
            # Hover annotation
            points_xy = np.array([[p['x'], p['y']] for p in point_data])
            annotation = ax.annotate(
                '', xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1")
            )
            annotation.set_visible(False)

            def on_hover(event):
                if event.inaxes != ax:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                threshold = 0.03 * max(x_range, y_range)
                cursor = np.array([event.xdata, event.ydata])
                dists = np.linalg.norm(points_xy - cursor, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < threshold:
                    p = point_data[idx]
                    text = (f"File: {p['filename']}\n"
                            f"Run: {p['run_id']}\n"
                            f"Split: {p['split']}\n"
                            f"Index: {p['index']}")
                    annotation.xy = (p['x'], p['y'])
                    annotation.set_text(text)
                    annotation.set_visible(True)
                else:
                    annotation.set_visible(False)
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_hover)
            fig.text(0.01, 0.01, "Hover over points to see details",
                     fontsize=8, alpha=0.6)

        method_name = reduction_method.upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'All Run Trajectories ({len(run_trajectories)} runs, {method_name})')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"All-trajectories plot saved to {save_path}")

        return fig

    def visualize_trajectory(self,
                             folder_name: str,
                             reduction_method: str = "TSNE",
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: Optional[str] = None,
                             show_connections: bool = False,
                             smooth_path: bool = False) -> plt.Figure:
        """
        Plot embeddings for a single folder/trajectory with a temporal color gradient.

        Points are colored from the first image (dark purple) to the last image
        (yellow) using the 'plasma' colormap, so you can see how the visual
        content evolves over the sequence.

        Args:
            folder_name: Name of the data folder (e.g. "data", "frames", "deus_1").
                         Must match a folder whose embeddings exist in the
                         embeddings/ directory under data_root.
            reduction_method: "PCA" or "TSNE" for dimensionality reduction.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure object.
        """
        if self.mode != "multi_run" or self.data_root is None:
            raise RuntimeError("visualize_trajectory requires multi_run mode (data_root)")

        embeddings_dir = self.data_root / "embeddings"
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        # Find the .pt file whose metadata filenames belong to this folder
        target_embeddings = None
        target_filenames = None

        for pt_file in sorted(embeddings_dir.glob("*.pt")):
            try:
                data = torch.load(pt_file, map_location='cpu')
                if not isinstance(data, dict) or 'embeddings' not in data:
                    continue
                meta = data.get('metadata', {})
                filenames = meta.get('filenames', [])
                if not filenames:
                    continue
                first_file = filenames[0].replace('\\', '/')
                parts = first_file.rsplit('/', 1)
                if len(parts) == 2:
                    dir_part = parts[0]
                    file_folder = dir_part.rsplit('/', 1)[-1]
                else:
                    file_folder = meta.get('trajectory_id', first_file)
                if file_folder == folder_name:
                    embs = data['embeddings']
                    if hasattr(embs, 'numpy'):
                        embs = embs.numpy()
                    target_embeddings = embs
                    target_filenames = [f.replace('\\', '/').split('/')[-1]
                                        for f in filenames]
                    logger.info(f"Loaded {len(target_embeddings)} embeddings for "
                                f"folder '{folder_name}' from {pt_file.name}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load {pt_file.name}: {e}")

        if target_embeddings is None:
            raise RuntimeError(
                f"No embeddings found for folder '{folder_name}'. "
                f"Available folders: check embeddings/ directory."
            )

        n_points = len(target_embeddings)

        # Reduce to 2D
        embeddings_2d, _ = self._reduce_dimensions(target_embeddings, reduction_method)

        # Temporal color values: 0 = first image, 1 = last image
        color_values = np.arange(n_points) / max(n_points - 1, 1)

        fig, ax = plt.subplots(figsize=figsize)

        sc = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=color_values,
            cmap='plasma_r',
            s=40,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.3
        )

        if show_connections:
            cmap_obj = plt.get_cmap('plasma_r')
            for i in range(n_points - 1):
                x0, y0 = embeddings_2d[i]
                x1, y1 = embeddings_2d[i + 1]
                color = cmap_obj(color_values[i])
                ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle='->', color=color,
                        linestyle='dotted', lw=1.0, alpha=0.6,
                        connectionstyle='arc3,rad=0.0'
                    )
                )

        if smooth_path and n_points >= 4:
            from scipy.interpolate import splprep, splev
            x = embeddings_2d[:, 0]
            y = embeddings_2d[:, 1]
            # Parameterise by cumulative arc length so the spline follows
            # the actual point-to-point path order
            dx = np.diff(x)
            dy = np.diff(y)
            arc = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
            # smoothing factor s: larger = smoother; scale by arc length
            s = arc[-1] * 0.5
            tck, u = splprep([x, y], u=arc, s=s, k=3)
            u_fine = np.linspace(arc[0], arc[-1], 800)
            sx, sy = splev(u_fine, tck)
            # Draw the smooth curve
            ax.plot(sx, sy, color='black', lw=2.5, alpha=0.75,
                    zorder=2, solid_capstyle='round')
            # Arrowhead at the end
            ax.annotate(
                '', xy=(sx[-1], sy[-1]),
                xytext=(sx[-20], sy[-20]),
                arrowprops=dict(
                    arrowstyle='->', color='black', lw=2.5
                ),
                zorder=3
            )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Temporal position (0 = first, 1 = last)')
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([
            f'0 ({target_filenames[0] if target_filenames else "start"})',
            f'{int(0.25 * (n_points-1))}',
            f'{int(0.5 * (n_points-1))}',
            f'{int(0.75 * (n_points-1))}',
            f'{n_points-1} ({target_filenames[-1] if target_filenames else "end"})',
        ])

        method_name = reduction_method.upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f"Trajectory: '{folder_name}'  ({n_points} images, {method_name})")
        ax.grid(True, alpha=0.3)

        # Hover annotation
        annotation = ax.annotate(
            '', xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1")
        )
        annotation.set_visible(False)

        points_xy = embeddings_2d  # (N, 2)

        def on_hover(event):
            if event.inaxes != ax:
                annotation.set_visible(False)
                fig.canvas.draw_idle()
                return
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            threshold = 0.03 * max(x_range, y_range)
            cursor = np.array([event.xdata, event.ydata])
            dists = np.linalg.norm(points_xy - cursor, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < threshold:
                fname = target_filenames[idx] if target_filenames else str(idx)
                annotation.xy = (points_xy[idx, 0], points_xy[idx, 1])
                annotation.set_text(f"[{idx}] {fname}")
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.text(0.01, 0.01, "Hover over points to see filenames",
                 fontsize=8, alpha=0.6)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory visualization saved to {save_path}")

        return fig

    def save_interactive_plot(self, trajectory: EmbeddingTrajectory,
                            assignment_result: DatasetAssignmentResult,
                            output_path: str,
                            include_metadata: bool = True):
        """Save an interactive HTML plot using plotly (if available)."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not available. Install with: pip install plotly")
            return

        # Reduce dimensions
        embeddings_2d, _ = self._reduce_dimensions(trajectory.embeddings_matrix, "PCA")

        # Create DataFrame for plotly
        plot_data = []
        for i, point in enumerate(trajectory.points):
            row = {
                'x': embeddings_2d[i, 0],
                'y': embeddings_2d[i, 1],
                'split': self.split_names[point.dataset_split],
                'index': i,
                'original_index': point.original_index,
                'file_path': point.file_path or f'point_{i}',
                'anchor_id': getattr(point, 'anchor_id', None),
                'distance_to_anchor': getattr(point, 'distance_to_anchor', None)
            }

            if include_metadata and point.metadata:
                row.update(point.metadata)

            plot_data.append(row)

        df = pd.DataFrame(plot_data)

        # Create interactive scatter plot
        fig = px.scatter(df, x='x', y='y', color='split',
                        hover_data=['file_path', 'anchor_id', 'distance_to_anchor'],
                        title='BubbleFence Data Split Visualization (Interactive)')

        fig.update_layout(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            width=1000,
            height=700
        )

        # Save interactive plot
        fig.write_html(output_path)
        logger.info(f"Interactive plot saved to {output_path}")

    def create_interactive_matplotlib_plot(self, trajectory: EmbeddingTrajectory,
                                         assignment_result: DatasetAssignmentResult,
                                         reduction_method: str = "TSNE",
                                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create an interactive matplotlib plot with cursor hover to show filenames.

        Args:
            trajectory: Embedding trajectory with data points
            assignment_result: Dataset assignment results
            reduction_method: "PCA" or "TSNE" for dimensionality reduction
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object with hover functionality
        """
        logger.info(f"Creating interactive matplotlib plot with {trajectory.num_points} points")

        # Get embeddings and reduce dimensionality
        embeddings = trajectory.embeddings_matrix
        embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)

        # Compute anchor point indices for star markers
        anchor_registry = self._load_anchor_registry()
        anchor_point_indices = self._get_anchor_point_indices(trajectory, anchor_registry)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data points colored by split
        scatter_plots = {}
        point_data = []  # Store point information for hover

        for split in DatasetSplit:
            split_indices = [i for i, point in enumerate(trajectory.points)
                           if point.dataset_split == split]
            if split_indices:
                non_anchor = [i for i in split_indices if i not in anchor_point_indices]
                anchor = [i for i in split_indices if i in anchor_point_indices]

                # Plot non-anchor points as circles
                if non_anchor:
                    scatter = ax.scatter(
                        embeddings_2d[non_anchor, 0],
                        embeddings_2d[non_anchor, 1],
                        c=self.colors[split],
                        label=self.split_names[split],
                        alpha=0.7,
                        s=50,
                        edgecolors='black',
                        linewidths=0.3
                    )
                    scatter_plots[split] = scatter

                # Plot anchor points as stars
                if anchor:
                    label = self.split_names[split] if not non_anchor else None
                    ax.scatter(
                        embeddings_2d[anchor, 0],
                        embeddings_2d[anchor, 1],
                        c=self.colors[split],
                        label=label,
                        alpha=0.9,
                        s=120,
                        marker='*',
                        edgecolors='black',
                        linewidths=0.5
                    )

                # Store point data for hover functionality
                for idx in split_indices:
                    point = trajectory.points[idx]
                    filename = Path(point.file_path).name if point.file_path else f"Point_{idx}"
                    # Get anchor info - either assigned anchor or closest anchor for training points
                    anchor_id = getattr(point, 'anchor_id', None)
                    distance = getattr(point, 'distance_to_anchor', None)

                    # For training points, show closest anchor info
                    if split == DatasetSplit.TRAIN:
                        closest_anchor_id = getattr(point, 'closest_anchor_id', None)
                        if closest_anchor_id is not None:
                            anchor_id = f"closest: {closest_anchor_id}"

                    point_data.append({
                        'x': embeddings_2d[idx, 0],
                        'y': embeddings_2d[idx, 1],
                        'filename': filename,
                        'split': self.split_names[split],
                        'original_index': point.original_index,
                        'anchor_id': anchor_id,
                        'distance_to_anchor': distance
                    })

        # Set up plot
        method_name = reduction_method.upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'BubbleFence Semantic Fence Visualization ({method_name})\n'
                    'Hover over points to see filenames')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Create annotation for hover text
        annotation = ax.annotate(
            '', xy=(0, 0), xytext=(20, 20), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1")
        )
        annotation.set_visible(False)

        # Convert point_data to numpy arrays for efficient distance calculation
        points_array = np.array([[p['x'], p['y']] for p in point_data])

        def on_hover(event):
            """Handle mouse hover events to show filename."""
            if event.inaxes != ax:
                annotation.set_visible(False)
                fig.canvas.draw_idle()
                return

            # Calculate distances to all points
            distances = np.sqrt((points_array[:, 0] - event.xdata)**2 +
                              (points_array[:, 1] - event.ydata)**2)

            # Find closest point
            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]

            # Show annotation if close enough to a point
            # Use adaptive threshold based on plot size
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            threshold = 0.02 * max(x_range, y_range)  # 2% of the larger axis range

            if min_distance < threshold:
                point_info = point_data[closest_idx]

                # Create hover text
                hover_text = f"File: {point_info['filename']}\n"
                hover_text += f"Split: {point_info['split']}\n"
                hover_text += f"Index: {point_info['original_index']}"

                if point_info['anchor_id'] is not None:
                    hover_text += f"\nAnchor: {point_info['anchor_id']}"
                if point_info['distance_to_anchor'] is not None:
                    hover_text += f"\nDistance: {point_info['distance_to_anchor']:.4f}"

                # Position annotation
                annotation.xy = (point_info['x'], point_info['y'])
                annotation.set_text(hover_text)
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)

            fig.canvas.draw_idle()

        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        # Add instructions
        fig.text(0.02, 0.02, "* Hover over points to see filenames and details",
                fontsize=10, style='italic', alpha=0.7)

        plt.tight_layout()

        return fig


def create_interactive_semantic_fence_plot(csv_path: str,
                                         data_dir: str = None,
                                         config_path: str = None,
                                         reduction_method: str = "tsne",
                                         figsize: Tuple[int, int] = (14, 10),
                                         save_plot: str = None,
                                         show_duplicates: bool = False) -> plt.Figure:
    """
    Create an interactive matplotlib plot directly from CSV results with hover functionality.

    Args:
        csv_path: Path to ingested_data.csv from BubbleFence
        data_dir: Optional data directory to regenerate embeddings
        config_path: Optional BubbleFence config path
        reduction_method: "PCA" or "TSNE"
        figsize: Figure size
        save_plot: Optional path to save the plot
        show_duplicates: Whether to show removed duplicate points (default: False)

    Returns:
        matplotlib Figure object with hover functionality
    """
    import pandas as pd
    from .foundation_models import FoundationModelProcessor
    from .data_structures import EmbeddingTrajectory, EmbeddingPoint, DatasetSplit
    from . import load_config

    logger.info(f"Creating interactive plot from {csv_path}")

    # Read CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} images from CSV")

    # Filter duplicates if not showing them
    if not show_duplicates:
        # Keep only rows with valid dataset_split values
        original_len = len(df)
        df = df[df['dataset_split'].notna() & (df['dataset_split'] != '')]
        filtered_len = len(df)
        logger.info(f"Filtered out {original_len - filtered_len} duplicate/removed points (show_duplicates={show_duplicates})")

    # Count splits
    split_counts = df['dataset_split'].value_counts(dropna=False)
    logger.info("Dataset distribution:")
    for split, count in split_counts.items():
        split_name = split if pd.notna(split) and split != '' else 'removed_duplicates'
        logger.info(f"  {split_name}: {count}")

    # Generate embeddings for filtered images
    embeddings = None
    if config_path and data_dir and Path(config_path).exists():
        logger.info(f"Regenerating embeddings for {len(df)} filtered images...")
        try:
            config = load_config(config_path)
            foundation_processor = FoundationModelProcessor(config)

            # Get image paths for filtered data
            image_paths = [Path(data_dir) / filename for filename in df['filename']]

            # Generate embeddings for filtered images
            embedding_points = foundation_processor.embed_images(
                image_paths,
                original_indices=df.index.tolist()
            )
            embeddings = np.array([point.embedding for point in embedding_points])
            logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")

        except Exception as e:
            logger.warning(f"Failed to regenerate embeddings: {e}")
            embeddings = None

    # Require real embeddings
    if embeddings is None:
        raise RuntimeError(
            "No embeddings available. Provide config_path and data_dir to generate embeddings, "
            "or use visualize_from_persistence() which loads embeddings from .pt files."
        )

    # Create trajectory from filtered images
    trajectory = EmbeddingTrajectory("csv_visualization")

    split_mapping = {
        'train': DatasetSplit.TRAIN,
        'val': DatasetSplit.VALIDATION,
        'validation': DatasetSplit.VALIDATION,
        'test': DatasetSplit.TEST
    }

    # Get anchor embeddings for distance calculation
    anchor_embeddings = []
    if 'anchor_id' in df.columns:
        # Find unique anchors and their representative points
        unique_anchors = df[df['anchor_id'].notna()]['anchor_id'].unique()
        for anchor_id in unique_anchors:
            # Get a representative embedding for this anchor (use first point with this anchor)
            anchor_rows = df[df['anchor_id'] == anchor_id]
            if len(anchor_rows) > 0:
                first_row_idx = anchor_rows.index[0]
                anchor_embeddings.append(embeddings[first_row_idx])

        if anchor_embeddings:
            anchor_embeddings = np.array(anchor_embeddings)
            logger.info(f"Found {len(anchor_embeddings)} anchor embeddings for distance calculation")

    points = []
    for i, (idx, row) in enumerate(df.iterrows()):
        dataset_split = split_mapping.get(row['dataset_split'], DatasetSplit.UNASSIGNED)

        point = EmbeddingPoint(
            original_index=idx,  # Use original CSV index
            file_path=row['filename'],
            metadata=row.to_dict(),
            dataset_split=dataset_split
        )

        # Add anchor info if available
        if 'anchor_id' in row and pd.notna(row['anchor_id']):
            point.anchor_id = int(row['anchor_id'])

        # Calculate distance to anchor
        if 'distance_to_anchor' in row and pd.notna(row['distance_to_anchor']):
            point.distance_to_anchor = max(0.0, float(row['distance_to_anchor']))
        elif len(anchor_embeddings) > 0:
            # Calculate minimum cosine distance to any anchor
            current_embedding = embeddings[i]
            # Normalize embeddings for cosine distance
            current_norm = current_embedding / (np.linalg.norm(current_embedding) + 1e-8)
            anchor_norms = anchor_embeddings / (np.linalg.norm(anchor_embeddings, axis=1, keepdims=True) + 1e-8)

            # Cosine distance = 1 - cosine similarity
            cosine_similarities = np.dot(anchor_norms, current_norm)
            cosine_distances = 1 - cosine_similarities
            min_distance = max(0.0, float(np.min(cosine_distances)))
            point.distance_to_anchor = min_distance

        points.append(point)

    # Set embeddings and points in bulk
    import torch
    embeddings_tensor = torch.from_numpy(embeddings).float()
    trajectory.set_embeddings(embeddings_tensor, points)

    # Create dummy assignment result
    train_indices = [i for i, p in enumerate(trajectory.points) if p.dataset_split == DatasetSplit.TRAIN]
    val_indices = [i for i, p in enumerate(trajectory.points) if p.dataset_split == DatasetSplit.VALIDATION]
    test_indices = [i for i, p in enumerate(trajectory.points) if p.dataset_split == DatasetSplit.TEST]

    from .data_structures import DatasetAssignmentResult
    assignment_result = DatasetAssignmentResult(
        train_indices=train_indices,
        validation_indices=val_indices,
        test_indices=test_indices,
        unassigned_indices=[],
        assignment_details=[],
        statistics={
            'total_points': len(trajectory.points),
            'train_count': len(train_indices),
            'validation_count': len(val_indices),
            'test_count': len(test_indices),
            'unassigned_count': 0
        }
    )

    # Create a simple visualizer without full pipeline
    class SimpleVisualizer:
        def __init__(self):
            self.colors = {
                DatasetSplit.TRAIN: '#1f77b4',      # Blue
                DatasetSplit.VALIDATION: '#ff7f0e', # Orange
                DatasetSplit.TEST: '#2ca02c',        # Green
                DatasetSplit.UNASSIGNED: '#888888'   # Gray for duplicates/removed
            }
            self.split_names = {
                DatasetSplit.TRAIN: 'Train',
                DatasetSplit.VALIDATION: 'Validation',
                DatasetSplit.TEST: 'Test',
                DatasetSplit.UNASSIGNED: 'Removed Duplicates'
            }

        def _reduce_dimensions(self, embeddings, method):
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            # Convert torch tensor to numpy if needed
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()

            if method.upper() == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            elif method.upper() == "TSNE":
                perplexity = min(30, len(embeddings) - 1)
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embeddings_2d = reducer.fit_transform(embeddings)
            else:
                raise ValueError(f"Unknown method: {method}")

            return embeddings_2d, reducer

        def create_interactive_matplotlib_plot(self, trajectory, assignment_result,
                                               reduction_method, figsize,
                                               anchor_point_indices=None):
            if anchor_point_indices is None:
                anchor_point_indices = set()

            # Get embeddings and reduce dimensionality
            embeddings = trajectory.embeddings_matrix
            embeddings_2d, reducer = self._reduce_dimensions(embeddings, reduction_method)

            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize)

            # Plot data points colored by split
            point_data = []  # Store point information for hover

            for split in DatasetSplit:
                split_indices = [i for i, point in enumerate(trajectory.points)
                               if point.dataset_split == split]
                if split_indices:
                    non_anchor = [i for i in split_indices if i not in anchor_point_indices]
                    anchor = [i for i in split_indices if i in anchor_point_indices]

                    # Plot non-anchor points as circles
                    if non_anchor:
                        ax.scatter(
                            embeddings_2d[non_anchor, 0],
                            embeddings_2d[non_anchor, 1],
                            c=self.colors[split],
                            label=self.split_names[split],
                            alpha=0.7,
                            s=50,
                            edgecolors='black',
                            linewidths=0.3
                        )

                    # Plot anchor points as stars
                    if anchor:
                        label = self.split_names[split] if not non_anchor else None
                        ax.scatter(
                            embeddings_2d[anchor, 0],
                            embeddings_2d[anchor, 1],
                            c=self.colors[split],
                            label=label,
                            alpha=0.9,
                            s=120,
                            marker='*',
                            edgecolors='black',
                            linewidths=0.5
                        )

                    # Store point data for hover functionality
                    for idx in split_indices:
                        point = trajectory.points[idx]
                        filename = Path(point.file_path).name if point.file_path else f"Point_{idx}"

                        # Get anchor info - either assigned anchor or closest anchor for training points
                        anchor_id = getattr(point, 'anchor_id', None)
                        distance = getattr(point, 'distance_to_anchor', None)

                        # For training points, show closest anchor info
                        if split == DatasetSplit.TRAIN:
                            closest_anchor_id = getattr(point, 'closest_anchor_id', None)
                            if closest_anchor_id is not None:
                                anchor_id = f"closest: {closest_anchor_id}"

                        point_data.append({
                            'x': embeddings_2d[idx, 0],
                            'y': embeddings_2d[idx, 1],
                            'filename': filename,
                            'split': self.split_names[split],
                            'original_index': point.original_index,
                            'anchor_id': anchor_id,
                            'distance_to_anchor': distance
                        })

            # Set up plot
            method_name = reduction_method.upper()
            ax.set_xlabel(f'{method_name} Component 1')
            ax.set_ylabel(f'{method_name} Component 2')
            ax.set_title(f'BubbleFence Semantic Fence Visualization ({method_name})\n'
                        'Hover over points to see filenames')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Create annotation for hover text
            annotation = ax.annotate(
                '', xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1")
            )
            annotation.set_visible(False)

            # Convert point_data to numpy arrays for efficient distance calculation
            points_array = np.array([[p['x'], p['y']] for p in point_data])

            def on_hover(event):
                """Handle mouse hover events to show filename."""
                if event.inaxes != ax:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()
                    return

                # Calculate distances to all points
                distances = np.sqrt((points_array[:, 0] - event.xdata)**2 +
                                  (points_array[:, 1] - event.ydata)**2)

                # Find closest point
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]

                # Show annotation if close enough to a point
                # Use adaptive threshold based on plot size
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                threshold = 0.02 * max(x_range, y_range)  # 2% of the larger axis range

                if min_distance < threshold:
                    point_info = point_data[closest_idx]

                    # Create hover text
                    hover_text = f"File: {point_info['filename']}\n"
                    hover_text += f"Split: {point_info['split']}\n"
                    hover_text += f"Index: {point_info['original_index']}"

                    if point_info['anchor_id'] is not None:
                        hover_text += f"\nAnchor: {point_info['anchor_id']}"
                    if point_info['distance_to_anchor'] is not None:
                        hover_text += f"\nDistance: {point_info['distance_to_anchor']:.4f}"

                    # Position annotation
                    annotation.xy = (point_info['x'], point_info['y'])
                    annotation.set_text(hover_text)
                    annotation.set_visible(True)
                else:
                    annotation.set_visible(False)

                fig.canvas.draw_idle()

            # Connect the hover event
            fig.canvas.mpl_connect('motion_notify_event', on_hover)

            # Add instructions
            fig.text(0.02, 0.02, "* Hover over points to see filenames and details",
                    fontsize=10, style='italic', alpha=0.7)

            plt.tight_layout()
            return fig

    # Identify anchor point indices from CSV data
    # Anchor centers are data points with distance_to_anchor == 0 (or very near 0)
    anchor_point_indices = set()
    if 'distance_to_anchor' in df.columns:
        for i, (idx, row) in enumerate(df.iterrows()):
            if pd.notna(row.get('distance_to_anchor')) and float(row['distance_to_anchor']) < 1e-6:
                anchor_point_indices.add(i)
        if anchor_point_indices:
            logger.info(f"Found {len(anchor_point_indices)} anchor data points for star markers")

    # Create visualizer and interactive plot
    visualizer = SimpleVisualizer()
    fig = visualizer.create_interactive_matplotlib_plot(
        trajectory, assignment_result, reduction_method, figsize,
        anchor_point_indices=anchor_point_indices
    )

    # Save if requested
    if save_plot:
        fig.savefig(save_plot, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_plot}")

    return fig


# Convenience function for quick visualization
def visualize_bubblefence_results(pipeline: BubbleFencePipeline,
                                trajectory: EmbeddingTrajectory,
                                assignment_result: DatasetAssignmentResult,
                                output_dir: str = "visualizations",
                                reduction_method: str = "PCA") -> Dict[str, str]:
    """
    Create all visualizations for BubbleFence results.

    Args:
        pipeline: Trained BubbleFence pipeline
        trajectory: Embedding trajectory
        assignment_result: Dataset assignment results
        output_dir: Directory to save visualizations
        reduction_method: "PCA" or "TSNE"

    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    visualizer = BubbleFenceVisualizer(pipeline)

    saved_files = {}

    # Main visualization
    main_fig = visualizer.visualize_data_splits(
        trajectory, assignment_result, reduction_method=reduction_method,
        save_path=str(output_path / f"bubblefence_splits_{reduction_method.lower()}.png")
    )
    saved_files['main_visualization'] = str(output_path / f"bubblefence_splits_{reduction_method.lower()}.png")

    # Anchor analysis
    anchor_fig = visualizer.create_anchor_analysis(
        trajectory.embeddings_matrix,
        save_path=str(output_path / f"anchor_analysis_{reduction_method.lower()}.png")
    )
    saved_files['anchor_analysis'] = str(output_path / f"anchor_analysis_{reduction_method.lower()}.png")

    # Interactive plot
    try:
        visualizer.save_interactive_plot(
            trajectory, assignment_result,
            str(output_path / "interactive_plot.html")
        )
        saved_files['interactive_plot'] = str(output_path / "interactive_plot.html")
    except Exception as e:
        logger.warning(f"Could not create interactive plot: {e}")

    # Close figures to save memory
    plt.close(main_fig)
    plt.close(anchor_fig)

    logger.info(f"All visualizations saved to {output_dir}")
    return saved_files


def visualize_standalone_from_persistence(
        data_root: str,
        reduction_method: str = "PCA",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show_heatmap: bool = True,
        show_trajectories: bool = False,
        show_points: bool = True,
        point_alpha: float = 0.3,
        show_thumbnails: bool = False,
        show_radius_circles: bool = False,
        show_anchor_thumbnails: bool = False,
        num_anchor_thumbnails: Optional[int] = None,
        anchor_thumbs_per_bubble: int = 1,
        anchor_thumb_pixels: int = 80,
        thumbnail_interval: float = 0.6,
        smoothing: float = 0.1,
        interactive: bool = False,
        data_dir: Optional[str] = None,
        anchor_ids: Optional[List] = None) -> plt.Figure:
    """
    Create a standalone single-panel visualization from persistence files.

    Args:
        data_root: Root directory containing full_dataset.csv, anchors_state.pkl,
                  and embeddings/ folder
        reduction_method: "PCA" or "TSNE"
        figsize: Figure size
        save_path: Optional path to save the figure
        show_heatmap: Show Gaussian heatmap contours
        show_trajectories: Show per-run trajectory curves
        show_points: Show scatter points
        point_alpha: Opacity of scatter points (default 0.3)
        show_thumbnails: Show image thumbnails along trajectories
        show_radius_circles: Draw average-radius circles around anchors
        show_anchor_thumbnails: Show representative thumbnails per anchor bubble
        num_anchor_thumbnails: Auto-select this many spread-out anchors for
            thumbnails (None = all anchors or those in anchor_ids)
        anchor_thumbs_per_bubble: Number of thumbnails per anchor (default 1)
        anchor_thumb_pixels: Size of anchor thumbnails in pixels (default 80)
        thumbnail_interval: Fraction interval for thumbnails (default 0.1)
        smoothing: Spline smoothing factor
        interactive: Add hover annotations (can be laggy with large datasets)
        data_dir: Directory containing image folders for thumbnails
        anchor_ids: Optional list of anchor IDs to cherry-pick for rendering

    Returns:
        matplotlib Figure object
    """
    visualizer = BubbleFenceVisualizer(data_root=data_root)
    return visualizer.visualize_standalone(
        reduction_method=reduction_method,
        figsize=figsize,
        save_path=save_path,
        show_heatmap=show_heatmap,
        show_trajectories=show_trajectories,
        show_points=show_points,
        point_alpha=point_alpha,
        show_thumbnails=show_thumbnails,
        show_radius_circles=show_radius_circles,
        show_anchor_thumbnails=show_anchor_thumbnails,
        num_anchor_thumbnails=num_anchor_thumbnails,
        anchor_thumbs_per_bubble=anchor_thumbs_per_bubble,
        anchor_thumb_pixels=anchor_thumb_pixels,
        thumbnail_interval=thumbnail_interval,
        smoothing=smoothing,
        interactive=interactive,
        data_dir=data_dir,
        anchor_ids=anchor_ids,
    )


def visualize_standalone_3d_from_persistence(
        data_root: str,
        reduction_method: str = "PCA",
        save_path: Optional[str] = None,
        show_heatmap: bool = True,
        show_trajectories: bool = False,
        show_points: bool = True,
        smoothing: float = 0.1,
        clean: bool = False):
    """
    Create an interactive 3D visualization from persistence files (plotly).

    Args:
        data_root: Root directory containing full_dataset.csv, anchors_state.pkl,
                  and embeddings/ folder
        reduction_method: "PCA" or "TSNE"
        save_path: Path to save (.html for interactive, .png for static)
        show_heatmap: Show heatmap scatter clouds around anchors
        show_trajectories: Show per-run trajectory curves
        show_points: Show scatter points
        smoothing: Spline smoothing factor
        clean: Strip gridlines, axis labels, and ticks

    Returns:
        plotly Figure object
    """
    visualizer = BubbleFenceVisualizer(data_root=data_root)
    return visualizer.visualize_standalone_3d(
        reduction_method=reduction_method,
        save_path=save_path,
        show_heatmap=show_heatmap,
        show_trajectories=show_trajectories,
        show_points=show_points,
        smoothing=smoothing,
        clean=clean,
    )


def visualize_from_persistence(data_root: str,
                              reduction_method: str = "PCA",
                              figsize: Tuple[int, int] = (14, 10),
                              save_path: Optional[str] = None,
                              run_filter: Optional[str] = None,
                              show_anchors: bool = True,
                              show_hyperspheres: bool = True,
                              show_heatmap: bool = True,
                              interactive: bool = False,
                              show_all_trajectories: bool = False,
                              smoothing: float = 0.1,
                              num_panels: int = 4) -> plt.Figure:
    """
    Create visualization directly from BubbleFence persistence files.

    This function provides a simple interface for visualizing multi-run BubbleFence
    results without needing to reconstruct the full pipeline.

    Args:
        data_root: Root directory containing full_dataset.csv, anchors_state.pkl,
                  and embeddings/ folder
        reduction_method: "PCA" or "TSNE" for dimensionality reduction
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        run_filter: Optional run ID to filter data (None for all runs)
        show_anchors: Whether to show anchor points
        show_hyperspheres: Whether to show hypersphere boundaries
        show_heatmap: Whether to show KDE heatmap of bubble interiors (default
            True). Mutually exclusive with show_hyperspheres -- heatmap wins.
        interactive: Whether to create interactive matplotlib plot with hover
        show_all_trajectories: Whether to overlay smooth trajectory curves for
            each run_id on the heatmap / convex-hull panel.

    Returns:
        matplotlib Figure object

    Example:
        # Visualize all runs
        fig = visualize_from_persistence("./")

        # Visualize specific run
        fig = visualize_from_persistence("./", run_filter="data_260216181258")

        # Create interactive plot
        fig = visualize_from_persistence("./", interactive=True)

        # Overlay per-run trajectory curves on the 4-panel plot
        fig = visualize_from_persistence("./", show_all_trajectories=True)
    """
    logger.info(f"Creating visualization from persistence files in {data_root}")

    # Create visualizer in multi-run mode
    visualizer = BubbleFenceVisualizer(data_root=data_root)

    if interactive:
        # Load data for interactive plot
        df = visualizer._load_full_dataset()

        # Filter by run if specified
        if run_filter:
            df = df[df['run_id'] == run_filter]
            logger.info(f"Filtered to {len(df)} points for run {run_filter}")

        # Load real embeddings for all rows
        embeddings = visualizer._load_all_embeddings(df)

        # Create trajectory
        trajectory = visualizer._create_trajectory_from_dataframe(df, embeddings)
        assignment_result = visualizer._create_assignment_result_from_dataframe(df)

        # Create interactive plot
        fig = visualizer.create_interactive_matplotlib_plot(
            trajectory, assignment_result, reduction_method, figsize
        )
    else:
        # Create standard visualization
        fig = visualizer.visualize_data_splits(
            reduction_method=reduction_method,
            figsize=figsize,
            run_filter=run_filter,
            show_anchors=show_anchors,
            show_hyperspheres=show_hyperspheres,
            show_heatmap=show_heatmap,
            show_all_trajectories=show_all_trajectories,
            smoothing=smoothing,
            num_panels=num_panels
        )

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    return fig


def print_dataset_stats(data_root: str, save_path: Optional[str] = None):
    """Print dataset statistics from persistence files to stdout and optionally to a file."""
    import pandas as pd
    import pickle

    csv_path = Path(data_root) / "full_dataset.csv"
    if not csv_path.exists():
        print(f"full_dataset.csv not found in {data_root}")
        return

    df = pd.read_csv(csv_path)
    total = len(df)
    lines = []

    runs = df['run_id'].unique()
    lines.append(f"Total data points: {total:,}")
    lines.append(f"Runs: {len(runs)}")

    # Split breakdown
    if 'dataset_split' in df.columns:
        lines.append("")
        splits = df['dataset_split'].value_counts()
        for split_name in ['train', 'validation', 'test']:
            count = splits.get(split_name, 0)
            lines.append(f"{split_name.capitalize():>12}: {count:>6,} ({count/total:.1%})")
        unassigned = splits.get('unassigned', 0) + splits.get('', 0)
        if unassigned:
            lines.append(f"{'Unassigned':>12}: {unassigned:>6,} ({unassigned/total:.1%})")

    # Anchor stats
    pkl_path = Path(data_root) / "anchors_state.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            state = pickle.load(f)
        registry = state if hasattr(state, 'anchors') else state.get('anchor_registry', None)
        if registry and hasattr(registry, 'anchors'):
            anchors = registry.anchors
            lines.append(f"\nAnchors: {len(anchors)}")
            radii = [a.base_radius for a in anchors.values()]
            if radii:
                radii = np.array(radii)
                lines.append(f"  Mean radius: {radii.mean():.4f}")
                lines.append(f"  Std radius:  {radii.std():.4f}")
                lines.append(f"  Min/Max:     {radii.min():.4f} / {radii.max():.4f}")

            # Mean points per anchor
            if 'anchor_id' in df.columns:
                counts = []
                for aid in anchors:
                    mask = df['anchor_id'] == aid
                    if not mask.any():
                        mask = df['anchor_id'].apply(
                            lambda x: pd.notna(x) and int(x) == aid)
                    counts.append(mask.sum())
                counts = np.array(counts)
                lines.append(f"  Mean points/anchor: {counts.mean():.1f}")

    output = "\n".join(lines)
    print(output)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(output + "\n")
        logger.info(f"Stats saved to {save_path}")


def get_available_runs(data_root: str) -> List[str]:
    """
    Get list of available run IDs from full_dataset.csv.

    Args:
        data_root: Root directory containing full_dataset.csv

    Returns:
        List of unique run IDs
    """
    import pandas as pd

    csv_path = Path(data_root) / "full_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"full_dataset.csv not found in {data_root}")

    df = pd.read_csv(csv_path)
    runs = df['run_id'].unique().tolist()
    logger.info(f"Found {len(runs)} runs: {runs}")
    return runs


def visualize_trajectory_from_persistence(data_root: str,
                                         folder_name: str,
                                         reduction_method: str = "TSNE",
                                         figsize: Tuple[int, int] = (10, 8),
                                         save_path: Optional[str] = None,
                                         show_connections: bool = False,
                                         smooth_path: bool = False) -> plt.Figure:
    """
    Plot a temporal color-gradient trajectory for a single folder.

    Thin wrapper around BubbleFenceVisualizer.visualize_trajectory() for
    use from command-line scripts.

    Args:
        data_root: Root directory containing the embeddings/ folder.
        folder_name: Name of the data folder (e.g. "data", "frames", "deus_1").
        reduction_method: "PCA" or "TSNE".
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.
        show_connections: Draw dotted arrows between consecutive points.

    Returns:
        matplotlib Figure object.
    """
    visualizer = BubbleFenceVisualizer(data_root=data_root)
    return visualizer.visualize_trajectory(
        folder_name=folder_name,
        reduction_method=reduction_method,
        figsize=figsize,
        save_path=save_path,
        show_connections=show_connections,
        smooth_path=smooth_path
    )


def visualize_all_trajectories_from_persistence(
        data_root: str,
        reduction_method: str = "PCA",
        figsize: Tuple[int, int] = (12, 9),
        save_path: Optional[str] = None,
        interactive: bool = False,
        smoothing: float = 0.1) -> plt.Figure:
    """
    Plot smooth trajectory curves for every run_id on a single standalone plot.

    Each run_id gets a distinct solid colour. Useful for comparing the
    embedding-space paths of different ingestion runs side by side.

    Args:
        data_root: Root directory containing full_dataset.csv and embeddings/.
        reduction_method: "PCA" or "TSNE".
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.
        interactive: Whether to show hoverable scatter points alongside the
            trajectory curves (default False).

    Returns:
        matplotlib Figure object.
    """
    visualizer = BubbleFenceVisualizer(data_root=data_root)
    return visualizer.visualize_all_trajectories(
        reduction_method=reduction_method,
        figsize=figsize,
        save_path=save_path,
        interactive=interactive,
        smoothing=smoothing
    )


def summarize_dataset(data_root: str) -> Dict[str, Any]:
    """
    Get summary statistics about the multi-run dataset.

    Args:
        data_root: Root directory containing persistence files

    Returns:
        Dictionary with summary statistics
    """
    import pandas as pd

    csv_path = Path(data_root) / "full_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"full_dataset.csv not found in {data_root}")

    df = pd.read_csv(csv_path)

    # Overall statistics
    total_points = len(df)
    unique_runs = df['run_id'].nunique()

    # Split statistics
    split_counts = df['dataset_split'].value_counts()

    # Run-wise statistics
    run_stats = df.groupby('run_id').agg({
        'filename': 'count',
        'dataset_split': lambda x: x.value_counts().to_dict()
    }).rename(columns={'filename': 'count'})

    # Anchor statistics
    anchor_info = {}
    if 'anchor_id' in df.columns:
        anchor_counts = df['anchor_id'].value_counts()
        anchor_info = {
            'total_anchors': len(anchor_counts),
            'points_per_anchor': anchor_counts.describe().to_dict()
        }

    summary = {
        'total_data_points': total_points,
        'unique_runs': unique_runs,
        'runs': df['run_id'].unique().tolist(),
        'split_distribution': split_counts.to_dict(),
        'run_statistics': run_stats.to_dict(),
        'anchor_info': anchor_info
    }

    return summary