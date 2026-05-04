"""
Core data structures for BubbleFence semantic data splitting.

GPU-accelerated: embeddings are stored as torch tensors on a shared device.
EmbeddingPoint is metadata-only; the actual embedding data lives as a single
(N, D) tensor on EmbeddingTrajectory to avoid duplication.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import pickle
import time
from pathlib import Path

from .device_utils import to_tensor, to_numpy, pairwise_distance_matrix, pairwise_distance_cross


class DatasetSplit(Enum):
    """Dataset split assignments."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNASSIGNED = "unassigned"


@dataclass
class EmbeddingPoint:
    """
    Metadata for a single data point in the embedding space.

    NOTE: Does NOT store the embedding itself. The embedding lives in the
    parent EmbeddingTrajectory's tensor. Use trajectory.get_embedding(index)
    to access it.
    """
    original_index: int
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_split: DatasetSplit = DatasetSplit.UNASSIGNED
    anchor_id: Optional[int] = None
    distance_to_anchor: Optional[float] = None


@dataclass
class NestedShell:
    """Represents a nested shell within a hypersphere."""
    inner_radius: float
    outer_radius: float
    dataset_split: DatasetSplit

    def contains_point(self, distance_to_center: float) -> bool:
        """Check if a point at given distance falls within this shell."""
        return self.inner_radius <= distance_to_center < self.outer_radius


@dataclass
class HypersphereAnchor:
    """
    Represents a hypersphere bubble anchor in the embedding space.
    Center is stored as a torch tensor on the pipeline's device.
    """
    anchor_id: int
    center: torch.Tensor  # (D,) tensor on device
    base_radius: float
    buffer_radius: float = 0.0
    nested_shells: List[NestedShell] = field(default_factory=list)
    local_density: Optional[float] = None
    lid_value: Optional[float] = None
    created_timestamp: Optional[float] = field(default_factory=lambda: time.time())

    def __post_init__(self):
        """Validate shells."""
        # Validate nested shells are properly ordered
        if len(self.nested_shells) > 1:
            for i in range(len(self.nested_shells) - 1):
                if self.nested_shells[i].outer_radius > self.nested_shells[i + 1].inner_radius:
                    raise ValueError("Nested shells must not overlap")

    @property
    def total_radius(self) -> float:
        """Get the total radius including buffer zone."""
        return self.base_radius + self.buffer_radius

    def get_point_assignment(self, distance: float) -> DatasetSplit:
        """
        Get dataset assignment for a point given its precomputed distance.

        Args:
            distance: Precomputed distance from this anchor's center.

        Returns:
            DatasetSplit assignment.
        """
        if distance > self.total_radius:
            return DatasetSplit.TRAIN

        # Check if in buffer zone
        if distance > self.base_radius:
            return DatasetSplit.UNASSIGNED

        # Check nested shells
        for shell in self.nested_shells:
            if shell.contains_point(distance):
                return shell.dataset_split

        # Default to train if no shells match (shouldn't happen with proper shell setup)
        return DatasetSplit.TRAIN

    def __getstate__(self):
        """Custom pickle: move tensor to CPU for serialization."""
        state = self.__dict__.copy()
        if isinstance(state['center'], torch.Tensor):
            state['center'] = state['center'].cpu()
        return state

    def __setstate__(self, state):
        """Custom unpickle: tensor stays on CPU, will be moved to device later."""
        self.__dict__.update(state)


@dataclass
class TrajectoryStats:
    """Statistics about a trajectory in embedding space."""
    num_points: int
    mean_embedding: torch.Tensor
    std_embedding: torch.Tensor
    density_map: Optional[torch.Tensor] = None
    high_density_regions: List[Tuple[int, int]] = field(default_factory=list)


class EmbeddingTrajectory:
    """
    Represents a trajectory of embeddings in the embedding space.

    Stores embeddings as a single (N, D) GPU tensor to avoid duplication.
    EmbeddingPoint objects hold only metadata.
    """

    def __init__(self, trajectory_id: str, device: Optional[torch.device] = None):
        self.trajectory_id = trajectory_id
        self.points: List[EmbeddingPoint] = []
        self.device = device or torch.device('cpu')
        self.stats: Optional[TrajectoryStats] = None

        # The single source of truth for embedding data
        self._embeddings: Optional[torch.Tensor] = None  # (N, D) on self.device

    def set_embeddings(self, embeddings: torch.Tensor, points: List[EmbeddingPoint]) -> None:
        """
        Set embeddings and points in bulk (preferred over add_point for performance).

        Args:
            embeddings: (N, D) tensor on any device (will be moved to self.device)
            points: List of N EmbeddingPoint metadata objects
        """
        assert len(points) == embeddings.shape[0], \
            f"Mismatch: {len(points)} points but {embeddings.shape[0]} embeddings"
        self._embeddings = embeddings.to(self.device)
        self.points = points

    def add_point_with_embedding(self, point: EmbeddingPoint, embedding: torch.Tensor) -> None:
        """Add a single point with its embedding. Less efficient than set_embeddings for bulk."""
        embedding = embedding.to(self.device)
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            self._embeddings = embedding.unsqueeze(0)
        else:
            self._embeddings = torch.cat([self._embeddings, embedding.unsqueeze(0)], dim=0)
        self.points.append(point)

    def get_embedding(self, index: int) -> torch.Tensor:
        """Get the embedding for point at given index. Returns a view (no copy)."""
        return self._embeddings[index]

    @property
    def embeddings_matrix(self) -> torch.Tensor:
        """Get all embeddings as a (N, D) tensor on self.device."""
        if self._embeddings is None:
            return torch.empty(0, 0, device=self.device)
        return self._embeddings

    @property
    def embeddings_numpy(self) -> np.ndarray:
        """Get all embeddings as numpy array. Only use at I/O boundaries."""
        if self._embeddings is None:
            return np.empty((0, 0))
        return to_numpy(self._embeddings)

    @property
    def num_points(self) -> int:
        """Get number of points in trajectory."""
        return len(self.points)

    def subset(self, indices: Union[List[int], torch.Tensor],
               new_id: Optional[str] = None) -> 'EmbeddingTrajectory':
        """
        Create a new trajectory from a subset of indices. Efficient GPU slice.

        Args:
            indices: List of indices or 1D tensor of indices to keep
            new_id: Optional ID for the new trajectory

        Returns:
            New EmbeddingTrajectory with only the selected points
        """
        if isinstance(indices, list):
            indices_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices_t = indices.to(self.device)

        new_traj = EmbeddingTrajectory(
            new_id or f"{self.trajectory_id}_subset",
            device=self.device
        )

        if len(indices_t) == 0:
            new_traj._embeddings = torch.empty(0, self._embeddings.shape[1], device=self.device) \
                if self._embeddings is not None else None
            return new_traj

        new_traj._embeddings = self._embeddings[indices_t]
        indices_list = indices_t.cpu().tolist() if isinstance(indices_t, torch.Tensor) else indices
        new_traj.points = [self.points[i] for i in indices_list]
        return new_traj

    def compute_stats(self) -> TrajectoryStats:
        """Compute and cache statistics for this trajectory (on GPU)."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            raise ValueError("Cannot compute stats for empty trajectory")

        self.stats = TrajectoryStats(
            num_points=len(self.points),
            mean_embedding=self._embeddings.mean(dim=0),
            std_embedding=self._embeddings.std(dim=0)
        )
        return self.stats


class AnchorRegistry:
    """
    Registry for managing hypersphere anchors.

    Supports batch GPU operations: all anchor centers are stacked into a single
    (A, D) tensor for vectorized distance computation.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.anchors: Dict[int, HypersphereAnchor] = {}
        self._next_id = 0
        self.device = device or torch.device('cpu')

        # Cached stacked tensor for batch operations
        self._centers_tensor: Optional[torch.Tensor] = None  # (A, D) on device
        self._radii_tensor: Optional[torch.Tensor] = None    # (A,) on device
        self._total_radii_tensor: Optional[torch.Tensor] = None  # (A,) on device
        self._anchor_id_list: List[int] = []  # maps tensor row -> anchor_id

    def _invalidate_cache(self):
        """Invalidate cached tensors when anchors change."""
        self._centers_tensor = None
        self._radii_tensor = None
        self._total_radii_tensor = None
        self._anchor_id_list = []

    def _ensure_cache(self):
        """Build cached tensors if needed."""
        if self._centers_tensor is not None:
            return

        if not self.anchors:
            return

        self._anchor_id_list = list(self.anchors.keys())
        centers = []
        radii = []
        total_radii = []

        for aid in self._anchor_id_list:
            anchor = self.anchors[aid]
            center = anchor.center
            if isinstance(center, torch.Tensor):
                centers.append(center.to(self.device))
            else:
                centers.append(torch.tensor(center, dtype=torch.float32, device=self.device))
            radii.append(anchor.base_radius)
            total_radii.append(anchor.total_radius)

        self._centers_tensor = torch.stack(centers)  # (A, D)
        self._radii_tensor = torch.tensor(radii, dtype=torch.float32, device=self.device)
        self._total_radii_tensor = torch.tensor(total_radii, dtype=torch.float32, device=self.device)

    def add_anchor(self, center: torch.Tensor, base_radius: float,
                   buffer_radius: float = 0.0,
                   nested_shells: Optional[List[NestedShell]] = None) -> int:
        """Add a new anchor and return its ID.

        Args:
            center: (D,) tensor on self.device from anchor_placer
            base_radius: Base radius of the hypersphere
            buffer_radius: Additional buffer zone radius
            nested_shells: Shell definitions for val/test split within the hypersphere
        """
        anchor_id = self._next_id
        self._next_id += 1

        anchor = HypersphereAnchor(
            anchor_id=anchor_id,
            center=center,
            base_radius=base_radius,
            buffer_radius=buffer_radius,
            nested_shells=nested_shells or []
        )

        self.anchors[anchor_id] = anchor
        self._invalidate_cache()
        return anchor_id

    def get_anchor(self, anchor_id: int) -> HypersphereAnchor:
        """Get anchor by ID."""
        if anchor_id not in self.anchors:
            raise KeyError(f"Anchor {anchor_id} not found")
        return self.anchors[anchor_id]

    def batch_distances(self, points: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
        """
        Compute distances from all points to all anchors in one GPU operation.

        Args:
            points: (N, D) tensor of embeddings on self.device
            metric: Distance metric to use

        Returns:
            (N, A) tensor of distances, or empty tensor if no anchors
        """
        self._ensure_cache()
        if self._centers_tensor is None:
            return torch.empty(points.shape[0], 0, device=self.device)

        return pairwise_distance_cross(points, self._centers_tensor, metric)

    def batch_find_containing(self, points: torch.Tensor,
                              metric: str = "cosine") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each point, find which anchors contain it (distance <= total_radius).

        Args:
            points: (N, D) tensor
            metric: distance metric

        Returns:
            Tuple of:
                - contained_mask: (N, A) bool tensor
                - distances: (N, A) float tensor
        """
        distances = self.batch_distances(points, metric)  # (N, A)
        if distances.shape[1] == 0:
            return (torch.zeros(points.shape[0], 0, dtype=torch.bool, device=self.device),
                    distances)

        contained_mask = distances <= self._total_radii_tensor.unsqueeze(0)  # broadcast (1, A)
        return contained_mask, distances

    def batch_assign(self, points: torch.Tensor,
                     metric: str = "cosine") -> List[Tuple[Optional[int], DatasetSplit, Optional[float]]]:
        """
        Batch-assign all points to dataset splits using GPU-computed distances.

        For each point, finds the first containing anchor and uses nested shell
        assignment. Points outside all anchors go to TRAIN.

        Args:
            points: (N, D) tensor
            metric: distance metric

        Returns:
            List of (anchor_id, split, distance) tuples for each point
        """
        contained_mask, distances = self.batch_find_containing(points, metric)  # (N, A), (N, A)
        N = points.shape[0]
        results = []

        if contained_mask.shape[1] == 0:
            # No anchors at all
            return [(None, DatasetSplit.TRAIN, None)] * N

        for i in range(N):
            # Find first containing anchor for this point
            containing = contained_mask[i].nonzero(as_tuple=False)  # (num_containing, 1)

            if len(containing) == 0:
                # Not inside any anchor -> train
                results.append((None, DatasetSplit.TRAIN, None))
            else:
                # Use first containing anchor
                anchor_tensor_idx = containing[0].item()
                anchor_id = self._anchor_id_list[anchor_tensor_idx]
                dist = distances[i, anchor_tensor_idx].item()
                anchor = self.anchors[anchor_id]
                split = anchor.get_point_assignment(dist)
                results.append((anchor_id, split, dist))

        return results

    def find_containing_anchors(self, point: torch.Tensor,
                                metric: str = "cosine") -> List[int]:
        """Find all anchors whose hyperspheres contain the given point (single point)."""
        self._ensure_cache()
        if self._centers_tensor is None:
            return []

        # Single point batch operation
        point_2d = point.unsqueeze(0) if point.dim() == 1 else point
        contained_mask, _ = self.batch_find_containing(point_2d, metric)
        containing_indices = contained_mask[0].nonzero(as_tuple=False).squeeze(-1)
        return [self._anchor_id_list[idx.item()] for idx in containing_indices]

    def get_nearest_anchor(self, point: torch.Tensor,
                           metric: str = "cosine") -> Tuple[int, float]:
        """Find the nearest anchor to a point."""
        if not self.anchors:
            raise ValueError("No anchors in registry")

        self._ensure_cache()
        point_2d = point.unsqueeze(0) if point.dim() == 1 else point
        distances = self.batch_distances(point_2d, metric)  # (1, A)
        min_idx = distances[0].argmin().item()
        return self._anchor_id_list[min_idx], distances[0, min_idx].item()

    def move_to_device(self, device: torch.device):
        """Move all anchor centers to a new device."""
        self.device = device
        for anchor in self.anchors.values():
            if isinstance(anchor.center, torch.Tensor):
                anchor.center = anchor.center.to(device)
        self._invalidate_cache()

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save anchor registry state to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'anchors': self.anchors,
                'next_id': self._next_id
            }, f)

    def load_state(self, filepath: Union[str, Path]) -> None:
        """Load anchor registry state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.anchors = state['anchors']
            self._next_id = state['next_id']
        # Move loaded anchors to current device and rebuild cache
        self.move_to_device(self.device)

    @property
    def num_anchors(self) -> int:
        """Get number of anchors in registry."""
        return len(self.anchors)

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about anchor coverage."""
        if not self.anchors:
            return {}

        radii = [anchor.base_radius for anchor in self.anchors.values()]
        total_radii = [anchor.total_radius for anchor in self.anchors.values()]

        return {
            'num_anchors': len(self.anchors),
            'mean_radius': np.mean(radii),
            'std_radius': np.std(radii),
            'min_radius': np.min(radii),
            'max_radius': np.max(radii),
            'mean_total_radius': np.mean(total_radii),
        }


@dataclass
class DatasetAssignmentResult:
    """Result of dataset assignment for a set of points."""
    train_indices: List[int]
    validation_indices: List[int]
    test_indices: List[int]
    unassigned_indices: List[int]

    assignment_details: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_points(self) -> int:
        """Get total number of points processed."""
        return len(self.train_indices) + len(self.validation_indices) + \
               len(self.test_indices) + len(self.unassigned_indices)

    @property
    def train_ratio(self) -> float:
        """Get actual training data ratio."""
        total = self.total_points
        return len(self.train_indices) / total if total > 0 else 0.0

    @property
    def eval_ratio(self) -> float:
        """Get actual evaluation data ratio (val + test)."""
        total = self.total_points
        eval_count = len(self.validation_indices) + len(self.test_indices)
        return eval_count / total if total > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert assignment result to a pandas DataFrame."""
        assignments = []

        for idx in self.train_indices:
            assignments.append({'index': idx, 'split': DatasetSplit.TRAIN.value})
        for idx in self.validation_indices:
            assignments.append({'index': idx, 'split': DatasetSplit.VALIDATION.value})
        for idx in self.test_indices:
            assignments.append({'index': idx, 'split': DatasetSplit.TEST.value})
        for idx in self.unassigned_indices:
            assignments.append({'index': idx, 'split': DatasetSplit.UNASSIGNED.value})

        df = pd.DataFrame(assignments)

        # Add assignment details if available
        if self.assignment_details:
            details_df = pd.DataFrame(self.assignment_details)
            df = df.merge(details_df, on='index', how='left')

        return df
