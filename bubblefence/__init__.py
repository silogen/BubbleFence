"""
BubbleFence: Semantic Fencing of Video Streams Using Embedding Bounded Regions

A modular, scalable implementation of the BubbleFence algorithm for semantic data splitting
using hypersphere bubbles in foundation model embedding spaces.

Key Features:
- Foundation model embeddings with multi-encoder consensus support
- Density-adaptive hypersphere placement using LID and QMC sampling
- Streaming data processing with persistent anchor states
- Configurable parameters via YAML configuration
- GPU-accelerated distance computations via PyTorch (CPU fallback supported)

Main Classes:
- BubbleFencePipeline: Main pipeline for end-to-end processing
- BubbleFenceConfig: Configuration management
- HypersphereAnchor: Individual bubble/anchor representation
- EmbeddingTrajectory: Trajectory of embeddings in space

Usage:
    from bubblefence import BubbleFenceConfig, BubbleFencePipeline

    config = BubbleFenceConfig.from_yaml('config.yaml')
    pipeline = BubbleFencePipeline(config)
    result = pipeline.process_image_stream(image_paths)
"""

__version__ = "1.0.0"

# Main API exports
from .config import (
    BubbleFenceConfig,
    FoundationModelConfig,
    EmbeddingConfig,
    HypersphereConfig,
    NestedShellConfig,
    DatasetSplitConfig,
    StreamingConfig,
    PreprocessingConfig
)

from .data_structures import (
    EmbeddingPoint,
    EmbeddingTrajectory,
    HypersphereAnchor,
    NestedShell,
    AnchorRegistry,
    DatasetAssignmentResult,
    DatasetSplit
)

from .bubble_fence import BubbleFencePipeline

from .foundation_models import FoundationModelProcessor
from .density_analysis import DensityAnalyzer
from .anchor_placement import AnchorPlacer
from .image_preprocessing import ImagePreprocessor
from .visualization import BubbleFenceVisualizer, visualize_bubblefence_results

from .device_utils import (
    detect_device,
    to_tensor,
    to_numpy,
    pairwise_distance_matrix,
    pairwise_distance_cross
)

from .pipeline_runner import (
    ingest_data,
    load_annotations,
    run_folder,
    run_batch,
    run_visualizations,
)

# Utility functions
def set_random_seeds(seed: int, set_global: bool = True) -> None:
    """Set random seeds for reproducible results."""
    import random
    import numpy as np

    if set_global:
        # Set Python's built-in random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make PyTorch deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # Set environment variable for Python hash seed
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)

# Convenience functions
def load_config(config_path: str) -> BubbleFenceConfig:
    """Load configuration from YAML file."""
    return BubbleFenceConfig.from_yaml(config_path)

def create_pipeline(config_path: str) -> BubbleFencePipeline:
    """Create a BubbleFence pipeline from configuration file."""
    config = load_config(config_path)
    return BubbleFencePipeline(config)

__all__ = [
    # Main API
    'BubbleFencePipeline',
    'BubbleFenceConfig',

    # Configuration classes
    'FoundationModelConfig',
    'EmbeddingConfig',
    'HypersphereConfig',
    'NestedShellConfig',
    'DatasetSplitConfig',
    'StreamingConfig',
    'PreprocessingConfig',

    # Data structures
    'EmbeddingPoint',
    'EmbeddingTrajectory',
    'HypersphereAnchor',
    'NestedShell',
    'AnchorRegistry',
    'DatasetAssignmentResult',
    'DatasetSplit',

    # Component classes
    'FoundationModelProcessor',
    'DensityAnalyzer',
    'AnchorPlacer',
    'ImagePreprocessor',

    # Visualization
    'BubbleFenceVisualizer',
    'visualize_bubblefence_results',

    # Device utilities
    'detect_device',
    'to_tensor',
    'to_numpy',
    'pairwise_distance_matrix',
    'pairwise_distance_cross',

    # Convenience functions
    'load_config',
    'create_pipeline',

    # Pipeline runner
    'ingest_data',
    'load_annotations',
    'run_folder',
    'run_batch',
    'run_visualizations',
]