"""
Density analysis and Local Intrinsic Dimensionality (LID) computation for BubbleFence.

GPU-accelerated: kNN uses torch distance matrix + topk instead of sklearn.
All computations stay on the pipeline's device.
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

from .config import BubbleFenceConfig
from .data_structures import EmbeddingTrajectory, TrajectoryStats, EmbeddingPoint
from .device_utils import to_tensor, to_numpy, pairwise_distance_matrix, pairwise_distance_cross


logger = logging.getLogger(__name__)


class DensityAnalyzer:
    """
    Analyzes density patterns in embedding trajectories and computes Local Intrinsic Dimensionality.
    All operations run on GPU via torch.
    """

    def __init__(self, config: BubbleFenceConfig, device: Optional[torch.device] = None):
        self.config = config
        self.density_config = config.density_transformation
        self.distance_metric = config.distance.metric
        self.device = device or torch.device('cpu')

    def _knn_distances(self, embeddings: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k-nearest neighbor distances on GPU.

        Replaces sklearn NearestNeighbors with torch distance matrix + topk.

        Args:
            embeddings: (N, D) tensor on self.device
            k: number of neighbors (excluding self)

        Returns:
            Tuple of (distances, indices) each of shape (N, k)
        """
        # Compute full pairwise distance matrix: (N, N)
        dist_matrix = pairwise_distance_matrix(embeddings, self.distance_metric)

        # Set diagonal to large value so self is not selected as neighbor
        dist_matrix.fill_diagonal_(float('inf'))

        # Get k smallest distances per row
        # NOTE: torch.topk with largest=False returns the k smallest values
        distances, indices = torch.topk(dist_matrix, k, dim=1, largest=False)

        return distances, indices

    def _knn_distances_cross(self, queries: torch.Tensor, references: torch.Tensor,
                            k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k-nearest neighbors of queries in references (cross-set).

        Args:
            queries: (Q, D) tensor
            references: (R, D) tensor
            k: number of neighbors

        Returns:
            Tuple of (distances, indices) each of shape (Q, k)
        """
        dist_matrix = pairwise_distance_cross(queries, references, self.distance_metric)
        k_clamped = min(k, references.shape[0])
        distances, indices = torch.topk(dist_matrix, k_clamped, dim=1, largest=False)
        return distances, indices

    def estimate_local_density(self, embeddings: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Estimate local density around each point using k-nearest neighbors (GPU).

        Args:
            embeddings: (N, D) tensor on self.device
            k: Number of nearest neighbors to consider

        Returns:
            (N,) tensor of local density estimates
        """
        logger.debug(f"Estimating local density for {embeddings.shape[0]} points with k={k}")

        if embeddings.shape[0] <= 1:
            return torch.ones(embeddings.shape[0], device=self.device)

        k = min(k, embeddings.shape[0] - 1)

        distances, _ = self._knn_distances(embeddings, k)  # (N, k)
        mean_distances = distances.mean(dim=1)  # (N,)
        density = 1.0 / (mean_distances + 1e-8)

        return density

    def compute_lid(self, embeddings: torch.Tensor, k: int = 20) -> torch.Tensor:
        """
        Compute Local Intrinsic Dimensionality (LID) for each point (GPU).

        Uses the maximum likelihood estimator based on distances to k nearest neighbors.

        Args:
            embeddings: (N, D) tensor on self.device
            k: Number of nearest neighbors to use for LID estimation

        Returns:
            (N,) tensor of LID values
        """
        logger.debug(f"Computing LID for {embeddings.shape[0]} points with k={k}")

        if embeddings.shape[0] <= 1:
            return torch.ones(embeddings.shape[0], device=self.device)

        k = min(k, embeddings.shape[0] - 1)

        distances, _ = self._knn_distances(embeddings, k)  # (N, k)

        # LID MLE: LID = 1 / mean(log(r_k / r_i)) for i = 1..k-1
        # distances are sorted ascending by topk(largest=False)
        max_dist = distances[:, -1:]  # (N, 1) - distance to k-th neighbor
        inner_dists = distances[:, :-1]  # (N, k-1) - distances to 1..k-1 neighbors

        # Handle zero distances (duplicate points)
        # NOTE: torch.where differs from np.where in that both branches are always evaluated,
        # but the gradient only flows through the selected branch
        safe_inner = inner_dists.clamp(min=1e-10)
        safe_max = max_dist.clamp(min=1e-10)

        log_ratios = torch.log(safe_max / safe_inner)  # (N, k-1)
        mean_log_ratios = log_ratios.mean(dim=1)  # (N,)

        lid_values = 1.0 / (mean_log_ratios + 1e-8)

        # Clamp to reasonable range
        lid_values = lid_values.clamp(0.1, 100.0)

        return lid_values

    def identify_high_density_regions(self, embeddings: torch.Tensor,
                                    density_threshold: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Identify regions with high density in the embedding space.

        Args:
            embeddings: (N, D) tensor
            density_threshold: Threshold multiplier for identifying high-density regions

        Returns:
            List of (start_idx, end_idx) tuples representing high-density regions
        """
        if density_threshold is None:
            density_threshold = self.density_config.density_threshold

        densities = self.estimate_local_density(embeddings)
        mean_density = densities.mean().item()
        threshold = density_threshold * mean_density

        high_density_mask = densities > threshold
        mask_cpu = high_density_mask.cpu().tolist()

        # Find contiguous regions
        high_density_regions = []
        in_region = False
        start_idx = 0

        for i, is_high in enumerate(mask_cpu):
            if is_high and not in_region:
                start_idx = i
                in_region = True
            elif not is_high and in_region:
                high_density_regions.append((start_idx, i - 1))
                in_region = False

        if in_region:
            high_density_regions.append((start_idx, len(mask_cpu) - 1))

        logger.info(f"Identified {len(high_density_regions)} high-density regions")
        return high_density_regions

    def apply_density_transformation(self, trajectory: EmbeddingTrajectory) -> EmbeddingTrajectory:
        """
        Apply density-aware transformation to stretch high-density regions.
        Operates entirely on GPU tensors.

        Args:
            trajectory: Input trajectory to transform

        Returns:
            Transformed trajectory with stretched high-density regions
        """
        if not self.density_config.enabled:
            logger.info("Density transformation disabled, returning original trajectory")
            return trajectory

        logger.info(f"Applying density transformation using method: {self.density_config.method}")

        embeddings = trajectory.embeddings_matrix  # (N, D) GPU tensor
        if embeddings.shape[0] == 0:
            return trajectory

        if self.density_config.method == "LID":
            transformed_embeddings = self._apply_lid_transformation(embeddings)
        elif self.density_config.method == "PCA":
            transformed_embeddings = self._apply_pca_whitening(embeddings)
        else:
            logger.warning(f"Unknown transformation method: {self.density_config.method}")
            return trajectory

        # Create new trajectory with transformed embeddings (stays on GPU)
        transformed_trajectory = EmbeddingTrajectory(
            trajectory.trajectory_id + "_transformed",
            device=self.device
        )

        # Copy metadata points (no embedding data in them)
        new_points = []
        for point in trajectory.points:
            new_point = EmbeddingPoint(
                original_index=point.original_index,
                file_path=point.file_path,
                metadata=point.metadata.copy()
            )
            new_points.append(new_point)

        transformed_trajectory.set_embeddings(transformed_embeddings, new_points)

        return transformed_trajectory

    def _apply_lid_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply LID-based transformation to stretch high-density regions (GPU)."""
        logger.debug("Applying LID-based transformation")

        lid_values = self.compute_lid(embeddings)  # (N,) GPU tensor

        # Use quantile instead of percentile (torch API)
        max_lid = torch.quantile(lid_values, 0.95).item()
        min_lid = torch.quantile(lid_values, 0.05).item()

        normalized_lid = (lid_values - min_lid) / (max_lid - min_lid + 1e-8)
        normalized_lid = normalized_lid.clamp(0, 1)

        scaling_factors = 1.0 + (1.0 - normalized_lid) * 2.0  # Scale between 1 and 3

        # Apply scaling (broadcast: (N, 1) * (N, D))
        transformed_embeddings = embeddings * scaling_factors.unsqueeze(1)

        logger.debug(f"LID range: [{lid_values.min().item():.3f}, {lid_values.max().item():.3f}]")
        logger.debug(f"Scaling factor range: [{scaling_factors.min().item():.3f}, {scaling_factors.max().item():.3f}]")

        return transformed_embeddings

    def _apply_pca_whitening(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply PCA whitening transformation (GPU via torch SVD)."""
        logger.debug("Applying PCA whitening transformation")

        # Center the data
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean

        # SVD-based whitening on GPU
        # NOTE: torch.linalg.svd returns (U, S, Vh) vs numpy's (U, S, V^T)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Whitened = U (already scaled by singular values in the decomposition)
        # For proper whitening: X_white = U * sqrt(N-1) but we just use U for simplicity
        transformed = U * (embeddings.shape[0] - 1) ** 0.5

        logger.debug(f"PCA top 5 singular values: {S[:5].cpu().tolist()}")

        return transformed

    def compute_adaptive_radius(self, center: torch.Tensor, embeddings: torch.Tensor,
                               base_radius: float, method: str = "LID") -> float:
        """
        Compute adaptive radius for a hypersphere based on local density/complexity.

        Args:
            center: (D,) tensor - center point of the hypersphere
            embeddings: (N, D) tensor - all embedding points
            base_radius: Base radius to scale
            method: Method for radius computation

        Returns:
            Adaptive radius value (Python float)
        """
        if method == "LID":
            return self._compute_radius_lid(center, embeddings, base_radius)
        elif method == "local_density":
            return self._compute_radius_density(center, embeddings, base_radius)
        elif method == "knn_distance":
            return self._compute_radius_knn(center, embeddings, base_radius)
        else:
            logger.warning(f"Unknown radius computation method: {method}")
            return base_radius

    def _compute_radius_lid(self, center: torch.Tensor, embeddings: torch.Tensor,
                           base_radius: float) -> float:
        """Compute radius based on local intrinsic dimensionality (GPU)."""
        k = min(20, embeddings.shape[0] - 1)
        if k <= 0:
            return base_radius

        # Find k nearest neighbors to the center using cross-set kNN.
        # Request k+1 neighbors and exclude the self-match (distance ~ 0) if the
        # anchor center was snapped to a data point in embeddings.
        center_2d = center.unsqueeze(0)  # (1, D)
        k_fetch = min(k + 1, embeddings.shape[0])
        knn_dists, knn_indices = self._knn_distances_cross(center_2d, embeddings, k_fetch)
        knn_distances = knn_dists[0]  # (k_fetch,)
        local_indices_raw = knn_indices[0]  # (k_fetch,)

        # Drop self-match: exclude the first neighbor if its distance is near zero
        self_match_threshold = 1e-6
        if knn_distances[0].item() < self_match_threshold and k_fetch > k:
            knn_distances = knn_distances[1:]
            local_indices = local_indices_raw[1:]
        else:
            knn_distances = knn_distances[:k]
            local_indices = local_indices_raw[:k]

        # Compute LID for the local neighborhood
        local_embeddings = embeddings[local_indices]  # (k, D)
        if local_embeddings.shape[0] > 1:
            local_lid = self.compute_lid(local_embeddings, k=min(k, local_embeddings.shape[0] - 1))
            mean_lid = local_lid.mean().item()
        else:
            mean_lid = 1.0

        # Scale radius proportionally with LID/4
        # LID ~1.6-2.6 typical range -> scale ~0.4-0.65
        # Dense (low LID) -> smaller radius, spread (high LID) -> larger radius
        normalized_lid = max(0.5, min(10.0, mean_lid))
        radius_scale = normalized_lid / 4.0

        radius = base_radius * radius_scale * self.config.hypersphere.radius_scale_factor

        clamped_radius = max(self.config.hypersphere.min_radius,
                           min(self.config.hypersphere.max_radius, radius))

        logger.debug(f"LID radius: mean_lid={mean_lid:.4f}, scale={radius_scale:.4f}, "
                     f"radius={clamped_radius:.6f}")

        return clamped_radius

    def _compute_radius_density(self, center: torch.Tensor, embeddings: torch.Tensor,
                               base_radius: float) -> float:
        """Compute radius based on local density (GPU)."""
        center_2d = center.unsqueeze(0)  # (1, D)

        # Estimate local density around center
        k = min(10, embeddings.shape[0] - 1)
        if k <= 0:
            return base_radius

        # Fetch k+1 and exclude self-match if anchor is snapped to a data point
        k_fetch = min(k + 1, embeddings.shape[0])
        center_knn_dists, _ = self._knn_distances_cross(center_2d, embeddings, k_fetch)
        dists = center_knn_dists[0]
        if dists[0].item() < 1e-6 and k_fetch > k:
            dists = dists[1:]
        else:
            dists = dists[:k]
        local_density = 1.0 / (dists.mean().item() + 1e-8)

        # Get overall density statistics
        all_densities = self.estimate_local_density(embeddings)
        mean_density = all_densities.mean().item()

        density_ratio = local_density / (mean_density + 1e-8)
        radius_scale = 1.0 / (density_ratio ** 0.5 + 1e-8)

        radius = base_radius * radius_scale * self.config.hypersphere.radius_scale_factor

        radius = max(self.config.hypersphere.min_radius,
                    min(self.config.hypersphere.max_radius, radius))

        return radius

    def _compute_radius_knn(self, center: torch.Tensor, embeddings: torch.Tensor,
                           base_radius: float) -> float:
        """Compute radius based on k-nearest neighbor distances (GPU)."""
        k = min(10, embeddings.shape[0] - 1)
        if k <= 0:
            return base_radius

        center_2d = center.unsqueeze(0)
        # Fetch k+1 and exclude self-match if anchor is snapped to a data point
        k_fetch = min(k + 1, embeddings.shape[0])
        knn_dists, _ = self._knn_distances_cross(center_2d, embeddings, k_fetch)
        dists = knn_dists[0]
        if dists[0].item() < 1e-6 and k_fetch > k:
            dists = dists[1:]
        else:
            dists = dists[:k]
        mean_distance = dists.mean().item()

        radius = mean_distance * self.config.hypersphere.radius_scale_factor

        radius = max(self.config.hypersphere.min_radius,
                    min(self.config.hypersphere.max_radius, radius))

        return radius

    def get_density_stats(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Get comprehensive density statistics for embeddings (GPU)."""
        densities = self.estimate_local_density(embeddings)
        lid_values = self.compute_lid(embeddings)

        return {
            'num_points': embeddings.shape[0],
            'density_mean': densities.mean().item(),
            'density_std': densities.std().item(),
            'density_min': densities.min().item(),
            'density_max': densities.max().item(),
            'lid_mean': lid_values.mean().item(),
            'lid_std': lid_values.std().item(),
            'lid_min': lid_values.min().item(),
            'lid_max': lid_values.max().item(),
            'density_percentiles': {
                '25': torch.quantile(densities, 0.25).item(),
                '50': torch.quantile(densities, 0.50).item(),
                '75': torch.quantile(densities, 0.75).item(),
                '95': torch.quantile(densities, 0.95).item()
            }
        }
