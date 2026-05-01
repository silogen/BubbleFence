"""
Anchor placement algorithms for BubbleFence semantic data splitting.

GPU-accelerated: all distance computations use torch tensors on the pipeline's device.
"""

import torch
import numpy as np
import math
import logging
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any

from scipy.stats import qmc

from .config import BubbleFenceConfig
from .data_structures import EmbeddingTrajectory, NestedShell, DatasetSplit
from .device_utils import (
    to_tensor, to_numpy,
    pairwise_distance_matrix, pairwise_distance_cross
)


logger = logging.getLogger(__name__)


class AnchorPlacer:
    """
    Handles anchor placement for hypersphere bubbles using various sampling strategies.
    All distance computations run on GPU via torch.
    """

    def __init__(self, config: BubbleFenceConfig, foundation_processor=None,
                 device: Optional[torch.device] = None,
                 density_analyzer=None):
        self.config = config
        self.anchor_config = config.anchor_placement
        self.dataset_config = config.dataset_splits
        self.nested_config = config.nested_shells
        self.distance_metric = config.distance.metric
        self.foundation_processor = foundation_processor
        self.device = device or torch.device('cpu')
        self.density_analyzer = density_analyzer

        # Set random seeds if configured
        self._setup_random_seeds()

    def _setup_random_seeds(self):
        """Set up random seeds for reproducible results."""
        if self.config.random_seed.random_seed is not None:
            from . import set_random_seeds
            set_random_seeds(
                self.config.random_seed.random_seed,
                self.config.random_seed.set_global_seeds
            )
            logger.info(f"Random seeds set to {self.config.random_seed.random_seed}")

    def place_anchors(self, trajectory: EmbeddingTrajectory) -> List[Tuple[torch.Tensor, float, List[NestedShell]]]:
        """
        Place anchors in the embedding space using the configured method.

        Args:
            trajectory: Trajectory containing embedding points

        Returns:
            List of (center_tensor, base_radius, nested_shells) tuples
        """
        if trajectory.num_points == 0:
            logger.warning("Empty trajectory, no anchors placed")
            return []

        embeddings = trajectory.embeddings_matrix  # (N, D) GPU tensor
        logger.info(f"Placing anchors for trajectory with {trajectory.num_points} points "
                   f"using method: {self.anchor_config.method}")

        if self.anchor_config.method == "QMC":
            anchor_centers = self._place_anchors_qmc(embeddings)
        elif self.anchor_config.method == "random":
            anchor_centers = self._place_anchors_random(embeddings)
        elif self.anchor_config.method == "kmeans":
            anchor_centers = self._place_anchors_kmeans(embeddings)
        else:
            raise ValueError(f"Unknown anchor placement method: {self.anchor_config.method}")

        # Compute base radius from current trajectory embeddings
        base_radius = self._compute_base_radius(embeddings)

        # Create nested shells for each anchor
        anchors_with_shells = []
        for center in anchor_centers:
            nested_shells = self._create_nested_shells(base_radius)
            anchors_with_shells.append((center, base_radius, nested_shells))

        logger.info(f"Placed {len(anchors_with_shells)} anchors")
        return anchors_with_shells

    def _place_anchors_qmc(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Place anchors using Quasi-Monte Carlo sampling. Returns GPU tensors."""
        logger.debug("Using QMC sampling for anchor placement")

        num_anchors = self._estimate_num_anchors(embeddings)

        # QMC requires numpy for scipy - move to CPU briefly
        min_vals = embeddings.min(dim=0).values.cpu().numpy()
        max_vals = embeddings.max(dim=0).values.cpu().numpy()

        # Create QMC sampler with seed for reproducibility
        dim = embeddings.shape[1]
        seed = self.config.random_seed.random_seed
        if self.anchor_config.qmc_sequence == "sobol":
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        elif self.anchor_config.qmc_sequence == "halton":
            sampler = qmc.Halton(d=dim, scramble=True, seed=seed)
        else:
            raise ValueError(f"Unknown QMC sequence: {self.anchor_config.qmc_sequence}")

        # Generate samples in unit hypercube and scale
        samples = sampler.random(num_anchors)
        anchor_centers_np = qmc.scale(samples, min_vals, max_vals)

        # Convert to GPU tensors
        anchor_centers = to_tensor(anchor_centers_np, self.device)  # (num_anchors, D)

        # Filter anchors that are too close to each other (GPU)
        anchor_centers = self._filter_close_anchors(anchor_centers)

        # Snap anchors to nearby embedding points (GPU)
        anchor_centers = self._snap_to_embeddings(anchor_centers, embeddings)

        logger.debug(f"QMC method placed {len(anchor_centers)} anchors")

        return anchor_centers

    def _place_anchors_random(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Place anchors using random sampling. Returns GPU tensors."""
        logger.debug("Using random sampling for anchor placement")

        num_anchors = self._estimate_num_anchors(embeddings)

        min_vals = embeddings.min(dim=0).values
        max_vals = embeddings.max(dim=0).values

        # Generate random samples on GPU
        anchor_centers = torch.rand(num_anchors * 2, embeddings.shape[1], device=self.device)
        anchor_centers = anchor_centers * (max_vals - min_vals) + min_vals

        # Filter close anchors
        anchor_centers = self._filter_close_anchors(anchor_centers)

        # Limit to target number
        anchor_centers = anchor_centers[:num_anchors]

        return [anchor_centers[i] for i in range(len(anchor_centers))]

    def _place_anchors_kmeans(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Place anchors using K-means clustering centers. Returns GPU tensors."""
        logger.debug("Using K-means clustering for anchor placement")
        from sklearn.cluster import KMeans

        num_anchors = self._estimate_num_anchors(embeddings)

        # KMeans requires numpy
        embeddings_np = to_numpy(embeddings)
        kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
        kmeans.fit(embeddings_np)

        # Convert back to GPU tensors
        centers = to_tensor(kmeans.cluster_centers_, self.device)

        # Filter close anchors
        centers = self._filter_close_anchors(centers)

        return [centers[i] for i in range(len(centers))]

    def _estimate_num_anchors(self, embeddings: torch.Tensor) -> int:
        """Estimate the number of anchors needed based on evaluation ratio target."""
        num_points = embeddings.shape[0]
        eval_ratio = self.dataset_config.eval_ratio

        points_per_anchor = max(5, min(15, num_points // 10))
        target_eval_points = int(num_points * eval_ratio)
        estimated_anchors = max(1, target_eval_points // points_per_anchor)

        min_anchors = max(1, int(np.sqrt(num_points) / 5))
        max_anchors = max(min_anchors, int(num_points / 3))
        estimated_anchors = max(min_anchors, min(max_anchors, estimated_anchors))

        logger.debug(f"Estimated {estimated_anchors} anchors for {num_points} points "
                    f"(target eval ratio: {eval_ratio:.2%}, ~{points_per_anchor} points/anchor)")

        return estimated_anchors

    def _filter_close_anchors(self, anchor_centers: torch.Tensor) -> List[torch.Tensor]:
        """
        Filter out anchors that are too close to each other.

        Args:
            anchor_centers: (K, D) tensor of candidate anchor centers

        Returns:
            List of kept anchor center tensors
        """
        if isinstance(anchor_centers, list):
            if len(anchor_centers) <= 1:
                return anchor_centers
            anchor_centers = torch.stack(anchor_centers)

        if anchor_centers.shape[0] <= 1:
            return [anchor_centers[0]]

        min_distance = self.anchor_config.min_anchor_distance

        # Compute full pairwise distance matrix on GPU
        dist_matrix = pairwise_distance_matrix(anchor_centers, self.distance_metric)

        # Greedy selection: keep first, skip any too close to already-kept
        kept_indices = [0]
        for i in range(1, len(anchor_centers)):
            # Check distances to all kept anchors
            dists_to_kept = dist_matrix[i, kept_indices]
            if dists_to_kept.min().item() >= min_distance:
                kept_indices.append(i)

        logger.debug(f"Filtered {len(anchor_centers)} -> {len(kept_indices)} anchors "
                    f"(min_distance: {min_distance})")

        return [anchor_centers[i] for i in kept_indices]

    def _snap_to_embeddings(self, anchor_centers: List[torch.Tensor],
                           embeddings: torch.Tensor) -> List[torch.Tensor]:
        """
        Snap anchor centers to nearby embedding points.

        When a density_analyzer is available, uses LID-weighted selection among
        the k-nearest neighbors (1/LID weighting so dense regions attract more
        anchors). Otherwise falls back to pure nearest-neighbor argmin.

        Args:
            anchor_centers: List of (D,) tensors or (K, D) tensor
            embeddings: (N, D) tensor of data embeddings

        Returns:
            List of snapped anchor center tensors
        """
        if embeddings.shape[0] == 0:
            return anchor_centers

        # Stack into (K, D) if list
        if isinstance(anchor_centers, list):
            if len(anchor_centers) == 0:
                return anchor_centers
            centers = torch.stack(anchor_centers)
        else:
            centers = anchor_centers

        # Compute distances from each anchor to all embeddings: (K, N)
        dist_matrix = pairwise_distance_cross(centers, embeddings, self.distance_metric)

        # --- LID-weighted snap ---
        if self.anchor_config.snap_strategy == "lid_weighted" and self.density_analyzer is not None:
            k = min(self.anchor_config.lid_snap_k, embeddings.shape[0])

            # Compute LID once for all embeddings, then discard after use
            lid_values = self.density_analyzer.compute_lid(embeddings)  # (N,)

            # Safety: handle invalid LID values
            if torch.any(~torch.isfinite(lid_values)) or torch.any(lid_values <= 0):
                logger.warning("Invalid LID values detected, falling back to uniform weights")
                weights = torch.ones_like(lid_values)
            else:
                weights = 1.0 / lid_values  # low LID = dense = higher weight

            logger.info(f"LID-weighted snap: k={k}, "
                        f"LID range=[{lid_values.min().item():.3f}, {lid_values.max().item():.3f}]")

            snapped = []
            for i in range(centers.shape[0]):
                # Get k nearest neighbor indices for this anchor
                _, topk_indices = torch.topk(dist_matrix[i], k, largest=False)  # (k,)

                # Gather their weights and sample
                candidate_weights = weights[topk_indices]  # (k,)
                candidate_weights = candidate_weights / candidate_weights.sum()  # normalize

                chosen_local = torch.multinomial(candidate_weights, 1).item()
                chosen_idx = topk_indices[chosen_local].item()

                snapped.append(embeddings[chosen_idx])
                logger.debug(f"Snapped anchor {i} to embedding {chosen_idx} "
                             f"(LID={lid_values[chosen_idx].item():.3f}, "
                             f"dist={dist_matrix[i, chosen_idx].item():.4f})")

            return snapped

        # --- Fallback: pure nearest neighbor ---
        nearest_indices = dist_matrix.argmin(dim=1)  # (K,)
        snapped = embeddings[nearest_indices]  # (K, D)

        for i in range(len(snapped)):
            logger.debug(f"Snapped anchor to embedding point (distance: {dist_matrix[i, nearest_indices[i]].item():.4f})")

        return [snapped[i] for i in range(len(snapped))]

    def _create_nested_shells(self, base_radius: float) -> List[NestedShell]:
        """Create nested shells for validation/test split within a hypersphere."""
        if not self.nested_config.enabled:
            return [NestedShell(0.0, base_radius, DatasetSplit.TEST)]

        validation_ratio = self.nested_config.validation_ratio

        if self.foundation_processor:
            embedding_dim = self.foundation_processor.get_embedding_dimension()
        else:
            embedding_dim = 512
            logger.warning("Foundation processor not available, using default embedding dimension 512")

        configured = self.nested_config.shell_configuration
        if configured == "random":
            actual_config = "inner_test" if random.random() < 0.5 else "inner_val"
        else:
            actual_config = configured

        if actual_config == "inner_val":
            validation_radius = base_radius * (validation_ratio ** (1.0 / embedding_dim)) * 0.7
            shells = [
                NestedShell(0.0, validation_radius, DatasetSplit.VALIDATION),
                NestedShell(validation_radius, base_radius, DatasetSplit.TEST)
            ]
            logger.debug(f"Created nested shells: val_radius={validation_radius:.4f}, "
                         f"test_radius={base_radius:.4f}")
        else:
            test_ratio = 1.0 - validation_ratio
            test_radius = base_radius * (test_ratio ** (1.0 / embedding_dim)) * 0.7
            shells = [
                NestedShell(0.0, test_radius, DatasetSplit.TEST),
                NestedShell(test_radius, base_radius, DatasetSplit.VALIDATION)
            ]
            logger.debug(f"Created nested shells: test_radius={test_radius:.4f}, "
                         f"val_radius={base_radius:.4f}")

        return shells

    def _compute_base_radius(self, embeddings: torch.Tensor) -> float:
        """Compute base radius based on embedding space characteristics (GPU)."""
        hypersphere_config = self.config.hypersphere

        if hypersphere_config.base_radius_mode == "fixed":
            base_radius = hypersphere_config.base_radius_fixed
            logger.debug(f"Using fixed base radius: {base_radius}")
            return base_radius

        # Auto mode: compute base radius from data using GPU
        distance_matrix = pairwise_distance_matrix(embeddings, self.distance_metric)

        # Extract upper triangle (excluding diagonal) on GPU
        N = distance_matrix.shape[0]
        row_idx, col_idx = torch.triu_indices(N, N, offset=1, device=self.device)
        pairwise_distances = distance_matrix[row_idx, col_idx]

        # Compute percentile - use torch.quantile (available since PyTorch 1.7)
        # NOTE: torch.quantile takes fraction [0,1], not percentage [0,100]
        percentile = hypersphere_config.base_radius_percentile
        scale = hypersphere_config.base_radius_scale
        base_radius_raw = torch.quantile(pairwise_distances, percentile / 100.0).item()
        base_radius = base_radius_raw * scale

        # Clamp
        base_radius = max(hypersphere_config.min_radius,
                         min(hypersphere_config.max_radius, base_radius))

        logger.debug(f"Auto base radius: {percentile}th pctl={base_radius_raw:.6f}, "
                     f"scale={scale}, clamped={base_radius:.6f}")

        return base_radius

    def place_additional_anchors(self, existing_anchors: List[torch.Tensor],
                                new_trajectory: EmbeddingTrajectory,
                                target_coverage_increase: float = 0.05) -> List[Tuple[torch.Tensor, float, List[NestedShell]]]:
        """
        Place additional anchors for new data that increases coverage.

        Args:
            existing_anchors: List of existing anchor center tensors
            new_trajectory: New trajectory with additional data points
            target_coverage_increase: Target increase in coverage ratio

        Returns:
            List of new anchors with their shells
        """
        if new_trajectory.num_points == 0:
            return []

        new_embeddings = new_trajectory.embeddings_matrix  # (N, D) GPU tensor
        logger.info(f"Placing additional anchors for {new_trajectory.num_points} new points")

        # Compute base radius from current trajectory embeddings
        base_radius = self._compute_base_radius(new_embeddings)

        # Find uncovered points using GPU batch operation
        uncovered_indices = self._find_uncovered_points(existing_anchors, new_embeddings, base_radius)

        if len(uncovered_indices) == 0:
            logger.info("All new points are already covered by existing anchors")
            return []

        logger.info(f"Found {len(uncovered_indices)} uncovered points")

        # Create sub-trajectory with uncovered points (GPU slice)
        uncovered_trajectory = new_trajectory.subset(uncovered_indices, "uncovered")

        # Place anchors in uncovered regions
        new_anchors = self.place_anchors(uncovered_trajectory)

        # Filter out anchors that are too close to existing ones (GPU batch)
        if not existing_anchors or not new_anchors:
            return new_anchors

        existing_centers = torch.stack(existing_anchors)  # (E, D)
        filtered_anchors = []
        for center, br, nested_shells in new_anchors:
            # Compute distance from this center to all existing anchors
            dists = pairwise_distance_cross(
                center.unsqueeze(0), existing_centers, self.distance_metric
            )  # (1, E)
            if dists.min().item() >= self.anchor_config.min_anchor_distance:
                filtered_anchors.append((center, br, nested_shells))

        logger.info(f"Added {len(filtered_anchors)} new anchors")
        return filtered_anchors

    def _find_uncovered_points(self, existing_anchors: List[torch.Tensor],
                              new_embeddings: torch.Tensor, base_radius: float) -> List[int]:
        """Find points not covered by existing anchors (GPU batch operation)."""
        if len(existing_anchors) == 0:
            return list(range(new_embeddings.shape[0]))

        # Stack existing anchors: (E, D)
        existing_centers = torch.stack(existing_anchors)

        # Compute distances from all new points to all existing anchors: (N, E)
        distances = pairwise_distance_cross(new_embeddings, existing_centers, self.distance_metric)

        # A point is covered if its min distance to any anchor <= base_radius
        min_distances = distances.min(dim=1).values  # (N,)
        uncovered_mask = min_distances > base_radius
        uncovered_indices = uncovered_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()

        # Handle edge case where squeeze removes dimension for single element
        if isinstance(uncovered_indices, int):
            uncovered_indices = [uncovered_indices]

        return uncovered_indices

    # ------------------------------------------------------------------
    # Closed-loop helpers: over-propose candidates, validate one-by-one
    # ------------------------------------------------------------------

    def propose_candidates(self, embeddings: torch.Tensor,
                           existing_anchor_centers: List[torch.Tensor],
                           overpropose_factor: int = 3) -> List[Tuple[torch.Tensor, float]]:
        """
        Over-propose anchor candidates once. The caller iterates through them
        one-by-one in a closed loop, validating and checking the eval deficit
        after each accepted candidate.

        Args:
            embeddings: (N, D) GPU tensor to sample from (typically UNASSIGNED points)
            existing_anchor_centers: centers of already-registered anchors
            overpropose_factor: how many multiples of the estimated need to generate

        Returns:
            List of (center, base_radius) tuples -- raw candidates before
            collision validation. Ordered so callers can iterate sequentially.
        """
        if embeddings.shape[0] == 0:
            return []

        base_radius = self._compute_base_radius(embeddings)

        num_target = self._estimate_num_anchors(embeddings) * overpropose_factor
        num_target = max(num_target, 3)

        logger.info(f"Over-proposing {num_target} anchor candidates "
                    f"(factor={overpropose_factor})")

        # Generate raw centers using configured method
        if self.anchor_config.method == "QMC":
            centers = self._generate_qmc_centers(embeddings, num_target)
        elif self.anchor_config.method == "random":
            centers = self._generate_random_centers(embeddings, num_target)
        elif self.anchor_config.method == "kmeans":
            # kmeans doesn't easily overshoot, fall back to QMC
            centers = self._generate_qmc_centers(embeddings, num_target)
        else:
            centers = self._generate_qmc_centers(embeddings, num_target)

        # Deduplicate candidates among themselves
        centers = self._filter_close_anchors(centers)

        # Remove candidates too close to existing anchors
        if existing_anchor_centers:
            existing_stacked = torch.stack(existing_anchor_centers)
            kept = []
            for c in centers:
                dists = pairwise_distance_cross(
                    c.unsqueeze(0), existing_stacked, self.distance_metric
                )
                if dists.min().item() >= self.anchor_config.min_anchor_distance:
                    kept.append(c)
            centers = kept

        # Snap to nearest real embedding
        centers = self._snap_to_embeddings(centers, embeddings)

        logger.info(f"After filtering: {len(centers)} candidates, "
                    f"base_radius={base_radius:.6f}")
        return [(c, base_radius) for c in centers]

    def propose_candidates_for_class(
            self, class_embeddings: torch.Tensor,
            existing_anchor_centers: List[torch.Tensor],
            base_radius: float,
            num_candidates: int) -> List[Tuple[torch.Tensor, float]]:
        """
        Propose anchor candidates for a single class. Uses pre-computed
        base_radius to avoid redundant pairwise distance computation.

        QMC generates points in the bounding box of this class's embeddings,
        snaps to this class's points only.

        Args:
            class_embeddings: (C, D) GPU tensor of this class's unassigned points
            existing_anchor_centers: centers of already-registered anchors
            base_radius: pre-computed from all unassigned embeddings
            num_candidates: how many candidates to generate

        Returns:
            List of (center, base_radius) tuples
        """
        if class_embeddings.shape[0] == 0:
            return []

        # Degenerate: bounding box collapses when min == max in any dim.
        # Use the points themselves as candidates instead of QMC.
        min_vals = class_embeddings.min(dim=0).values
        max_vals = class_embeddings.max(dim=0).values
        if torch.any(min_vals >= max_vals):
            logger.info(f"Class bounding box degenerate ({class_embeddings.shape[0]} points), "
                        f"using points directly as candidates")
            centers = [class_embeddings[i] for i in range(class_embeddings.shape[0])]
            return [(c, base_radius) for c in centers]

        num_candidates = max(num_candidates, 3)

        logger.info(f"Proposing {num_candidates} candidates for class "
                    f"({class_embeddings.shape[0]} points)")

        # QMC in this class's bounding box
        if self.anchor_config.method == "QMC":
            centers = self._generate_qmc_centers(class_embeddings, num_candidates)
        elif self.anchor_config.method == "random":
            centers = self._generate_random_centers(class_embeddings, num_candidates)
        else:
            centers = self._generate_qmc_centers(class_embeddings, num_candidates)

        # Filter close candidates
        centers = self._filter_close_anchors(centers)

        # Remove candidates too close to existing anchors
        if existing_anchor_centers:
            existing_stacked = torch.stack(existing_anchor_centers)
            kept = []
            for c in centers:
                dists = pairwise_distance_cross(
                    c.unsqueeze(0), existing_stacked, self.distance_metric
                )
                if dists.min().item() >= self.anchor_config.min_anchor_distance:
                    kept.append(c)
            centers = kept

        # Snap to this class's embeddings only
        centers = self._snap_to_embeddings(centers, class_embeddings)

        logger.info(f"After filtering: {len(centers)} class candidates")
        return [(c, base_radius) for c in centers]

    def _generate_qmc_centers(self, embeddings: torch.Tensor,
                              num: int) -> List[torch.Tensor]:
        """Generate QMC sample centers (no filtering / snapping)."""
        min_vals = embeddings.min(dim=0).values.cpu().numpy()
        max_vals = embeddings.max(dim=0).values.cpu().numpy()

        dim = embeddings.shape[1]
        seed = self.config.random_seed.random_seed
        if self.anchor_config.qmc_sequence == "sobol":
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        else:
            sampler = qmc.Halton(d=dim, scramble=True, seed=seed)

        # Suppress "n should be power of 2" warning. We already over-propose
        # anchors and filter down, so rounding up is wasteful. With scramble=True
        # the impact of non-power-of-2 is reduced. Sobol sequences fill coarse
        # regions first then refine, so the first num points still give good
        # global-to-local spatial coverage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            samples = sampler.random(num)
        centers_np = qmc.scale(samples, min_vals, max_vals)
        centers_t = to_tensor(centers_np, self.device)
        return [centers_t[i] for i in range(centers_t.shape[0])]

    def _generate_random_centers(self, embeddings: torch.Tensor,
                                 num: int) -> List[torch.Tensor]:
        """Generate random sample centers (no filtering / snapping)."""
        lo = embeddings.min(dim=0).values
        hi = embeddings.max(dim=0).values
        centers = torch.rand(num, embeddings.shape[1], device=self.device)
        centers = centers * (hi - lo) + lo
        return [centers[i] for i in range(centers.shape[0])]

    def validate_candidate(self, center: torch.Tensor, radius: float,
                           train_embeddings: Optional[torch.Tensor],
                           shrink_attempts: int = 3
                           ) -> Optional[Tuple[torch.Tensor, float]]:
        """
        Validate a single anchor candidate against TRAIN-only embeddings.

        If any TRAIN point falls inside the proposed radius, shrink the radius
        (halve each attempt) down to config min_radius.  If still colliding at
        min_radius, the candidate is discarded.

        Args:
            center: (D,) tensor -- proposed anchor center
            radius: proposed radius
            train_embeddings: (T, D) tensor of TRAIN-only embeddings, or None
            shrink_attempts: max halvings before discard

        Returns:
            (center, validated_radius) if collision-free, else None
        """
        min_radius = self.config.hypersphere.min_radius

        if train_embeddings is None or train_embeddings.shape[0] == 0:
            return (center, radius)

        # Distances from center to every train point (computed once)
        dists = pairwise_distance_cross(
            center.unsqueeze(0), train_embeddings, self.distance_metric
        ).squeeze(0)  # (T,)

        current_radius = radius
        for attempt in range(shrink_attempts + 1):
            collisions = (dists <= current_radius).sum().item()
            if collisions == 0:
                logger.debug(f"Candidate valid at radius={current_radius:.6f} "
                             f"(attempt {attempt})")
                return (center, current_radius)

            new_radius = current_radius * 0.5
            if new_radius < min_radius:
                new_radius = min_radius
            if new_radius == current_radius:
                # Already at floor, still colliding
                logger.debug(f"Candidate DISCARDED: {collisions} train collisions "
                             f"at min_radius={min_radius:.6f}")
                return None

            logger.debug(f"Shrink {current_radius:.6f} -> {new_radius:.6f} "
                         f"({collisions} train collisions)")
            current_radius = new_radius

        # Final check after all shrink attempts
        if (dists <= current_radius).sum().item() == 0:
            return (center, current_radius)

        logger.debug(f"Candidate DISCARDED after {shrink_attempts} shrink attempts")
        return None

    def get_placement_stats(self, anchors: List[Tuple[torch.Tensor, float, List[NestedShell]]],
                           embeddings: torch.Tensor) -> Dict[str, Any]:
        """Get statistics about anchor placement quality (GPU accelerated)."""
        if not anchors:
            return {}

        anchor_centers = torch.stack([a[0] for a in anchors])  # (A, D)
        radii = [a[1] for a in anchors]
        radii_tensor = torch.tensor(radii, device=self.device)

        # Compute distances from all points to all anchors: (N, A)
        distances = pairwise_distance_cross(embeddings, anchor_centers, self.distance_metric)

        # Coverage: point is covered if dist to any anchor <= that anchor's radius
        covered_mask = distances <= radii_tensor.unsqueeze(0)  # (N, A)
        covered_points = covered_mask.any(dim=1).sum().item()
        coverage = covered_points / embeddings.shape[0] if embeddings.shape[0] > 0 else 0

        # Anchor separation
        if len(anchor_centers) > 1:
            anchor_dists = pairwise_distance_matrix(anchor_centers, self.distance_metric)
            # Get upper triangle
            row_idx, col_idx = torch.triu_indices(len(anchor_centers), len(anchor_centers), offset=1, device=self.device)
            if len(row_idx) > 0:
                min_separation = anchor_dists[row_idx, col_idx].min().item()
            else:
                min_separation = 0.0
        else:
            min_separation = 0.0

        return {
            'num_anchors': len(anchors),
            'coverage_ratio': coverage,
            'mean_radius': np.mean(radii),
            'std_radius': np.std(radii),
            'min_radius': np.min(radii),
            'max_radius': np.max(radii),
            'min_anchor_separation': min_separation
        }
