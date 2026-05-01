"""
Device detection and tensor utility functions for GPU-accelerated BubbleFence.

Provides centralized device management so all modules share a single device
and helper functions for torch/numpy interop at boundaries.
"""

import torch
import numpy as np
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def detect_device(requested: str = "auto") -> torch.device:
    """
    Detect the best available device.

    Args:
        requested: "auto", "cuda", "cpu", or a specific device string.

    Returns:
        torch.device for the selected device.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected, using CPU")
    else:
        device = torch.device(requested)

    logger.info(f"BubbleFence device: {device}")
    return device


def to_tensor(data: Union[np.ndarray, torch.Tensor, list],
              device: torch.device,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert data to a torch tensor on the specified device.

    Avoids unnecessary copies if already a tensor on the right device.
    """
    if isinstance(data, torch.Tensor):
        if data.device == device and data.dtype == dtype:
            return data
        return data.to(device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    return torch.tensor(data, device=device, dtype=dtype)


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a tensor to numpy, handling GPU tensors.

    Only call this at true boundaries (pickle, sklearn, CSV output).
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute full pairwise cosine distance matrix on GPU.

    Args:
        embeddings: (N, D) tensor, assumed already normalized or will be normalized.

    Returns:
        (N, N) cosine distance matrix (1 - cosine_similarity).
    """
    # Normalize for numerical stability
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = embeddings / norms
    sim_matrix = torch.mm(normalized, normalized.t())
    # Clamp to [0, 2] range to avoid tiny negative distances from float precision
    return (1.0 - sim_matrix).clamp(min=0.0)


def cosine_distance_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distances between two sets of embeddings.

    Args:
        a: (N, D) tensor
        b: (M, D) tensor

    Returns:
        (N, M) cosine distance matrix.
    """
    a_norm = a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(a_norm, b_norm.t())
    return (1.0 - sim).clamp(min=0.0)


def euclidean_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute full pairwise Euclidean distance matrix on GPU."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = (embeddings ** 2).sum(dim=1, keepdim=True)
    dist_sq = sq_norms + sq_norms.t() - 2.0 * torch.mm(embeddings, embeddings.t())
    return dist_sq.clamp(min=0.0).sqrt()


def manhattan_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute full pairwise Manhattan distance matrix on GPU."""
    # This is O(N^2 * D) and memory-heavy; for large N consider chunking
    return torch.cdist(embeddings, embeddings, p=1.0)


def pairwise_distance_matrix(embeddings: torch.Tensor, metric: str) -> torch.Tensor:
    """Dispatch to the right distance matrix function based on metric."""
    if metric == "cosine":
        return cosine_distance_matrix(embeddings)
    elif metric == "euclidean":
        return euclidean_distance_matrix(embeddings)
    elif metric == "manhattan":
        return manhattan_distance_matrix(embeddings)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def pairwise_distance_cross(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """Compute cross-set pairwise distances."""
    if metric == "cosine":
        return cosine_distance_cross(a, b)
    elif metric == "euclidean":
        return torch.cdist(a, b, p=2.0)
    elif metric == "manhattan":
        return torch.cdist(a, b, p=1.0)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
