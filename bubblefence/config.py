"""
Configuration management for BubbleFence semantic data splitting.

This module handles loading and validation of configuration parameters from YAML files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FoundationModelConfig:
    """Configuration for foundation models."""
    primary_model: str
    multi_encoder_enabled: bool = False
    additional_models: List[str] = field(default_factory=list)
    consensus_method: str = "intersection"
    consensus_threshold: float = 0.8


@dataclass
class EmbeddingConfig:
    """Configuration for embedding computation."""
    device: str = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    io_workers: str = "auto"  # "auto", or integer string like "8"


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    enabled: bool = True
    similarity_threshold: float = 0.98
    method: str = "cosine"


@dataclass
class DensityTransformConfig:
    """Configuration for density transformation."""
    enabled: bool = False  # disable if snap_strategy="lid_weighted" to avoid redundancy; enabling affects visualizations since stored embeddings are not re-normalized
    method: str = "LID"
    stretch_high_density_regions: bool = True
    density_threshold: float = 1.5


@dataclass
class AnchorPlacementConfig:
    """Configuration for anchor placement."""
    method: str = "QMC"
    qmc_sequence: str = "sobol"
    min_anchor_distance: float = 0.1
    snap_strategy: str = "lid_weighted"  # "nearest" or "lid_weighted" (1/LID bias toward dense regions)
    lid_snap_k: int = 5  # k-nearest neighbors for LID-weighted anchor snapping (only used when snap_strategy="lid_weighted")


@dataclass
class BufferZoneConfig:
    """Configuration for buffer zones around hyperspheres."""
    enabled: bool = False
    buffer_radius: float = 0.02


@dataclass
class HypersphereConfig:
    """Configuration for hypersphere bubbles."""
    radius_computation: str = "adaptive"
    adaptive_method: str = "LID"

    # Base radius configuration
    base_radius_mode: str = "auto"  # "auto" or "fixed"
    base_radius_fixed: float = 0.05  # Used when base_radius_mode is "fixed"

    # Auto base radius settings
    base_radius_percentile: float = 50.0  # Percentile of pairwise distances
    base_radius_scale: float = 1.0  # Scale applied to percentile

    radius_scale_factor: float = 1.0
    min_radius: float = 0.01
    max_radius: float = 0.5
    fixed_radius: float = 0.1
    buffer_zones: BufferZoneConfig = field(default_factory=BufferZoneConfig)


@dataclass
class NestedShellConfig:
    """Configuration for nested shells within hyperspheres."""
    enabled: bool = True
    validation_ratio: float = 0.5
    shell_configuration: str = "random"  # "inner_val", "inner_test", or "random"


@dataclass
class RatioMonitoringConfig:
    """Configuration for dataset ratio monitoring."""
    enabled: bool = True
    tolerance: float = 0.05
    rebalancing_strategy: str = "add_anchors"


@dataclass
class DatasetSplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.8
    eval_ratio: float = 0.2
    min_eval_per_batch: float = 0.05  # minimum fraction of each batch allocated to eval
    eval_tolerance: float = 0.03  # cap eval overshoot to stay within +/- this of eval_ratio
    ratio_monitoring: RatioMonitoringConfig = field(default_factory=RatioMonitoringConfig)


@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""
    enabled: bool = True
    persistent_anchors: bool = True
    anchor_persistence_path: str = "anchors_state.pkl"


@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 100


@dataclass
class ANNIndexConfig:
    """Configuration for ANN index."""
    index_type: str = "HNSW"
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    memory_mapping: bool = False
    index_persistence_path: str = "ann_index.bin"


@dataclass
class DistanceConfig:
    """Configuration for distance computation."""
    metric: str = "cosine"
    precision: str = "float32"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    log_file: str = "bubblefence.log"


@dataclass
class OutputConfig:
    """Configuration for output."""
    output_path: str = "bubblefence_results"


@dataclass
class ExperimentalConfig:
    """Configuration for experimental features."""
    dynamic_anchor_adjustment: bool = False
    adaptive_buffer_zones: bool = False
    hierarchical_anchors: bool = False


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing before embedding.

    Controls whether to use the GPU-accelerated fast image processor
    (torchvision backend) or the standard CPU processor.

    Any field left as None inherits the model's default value.
    """
    # Backend selection: "auto" uses fast on GPU, slow on CPU.
    # "always" forces fast (will error if torchvision missing),
    # "never" forces slow.
    use_fast: str = "auto"

    # Resize - int for shortest_edge (e.g. 224), or dict for explicit dims
    # (e.g. {"height": 224, "width": 224})
    do_resize: Optional[bool] = None
    resize_size: Optional[Union[int, Dict[str, int]]] = None

    # Center crop
    do_center_crop: Optional[bool] = None
    crop_size: Optional[int] = None         # square crop side length

    # Rescale (pixel values 0-255 -> 0-1)
    do_rescale: Optional[bool] = None

    # Normalize
    do_normalize: Optional[bool] = None
    image_mean: Optional[List[float]] = None   # e.g. [0.481, 0.458, 0.408]
    image_std: Optional[List[float]] = None    # e.g. [0.269, 0.261, 0.276]

    # RGB conversion
    do_convert_rgb: Optional[bool] = None


@dataclass
class RandomSeedConfig:
    """Configuration for random seed control."""
    random_seed: Optional[int] = 42
    set_global_seeds: bool = True


@dataclass
class BubbleFenceConfig:
    """Main configuration class for BubbleFence."""
    foundation_models: FoundationModelConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    density_transformation: DensityTransformConfig = field(default_factory=DensityTransformConfig)
    anchor_placement: AnchorPlacementConfig = field(default_factory=AnchorPlacementConfig)
    hypersphere: HypersphereConfig = field(default_factory=HypersphereConfig)
    nested_shells: NestedShellConfig = field(default_factory=NestedShellConfig)
    dataset_splits: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    ann_index: ANNIndexConfig = field(default_factory=ANNIndexConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    random_seed: RandomSeedConfig = field(default_factory=RandomSeedConfig)
    class_aware: bool = True  # class-aware dedup, anchor placement, and eval balancing

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate ratios
        if not 0 < self.dataset_splits.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        if not 0 < self.dataset_splits.eval_ratio < 1:
            raise ValueError("eval_ratio must be between 0 and 1")

        if abs(self.dataset_splits.train_ratio + self.dataset_splits.eval_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + eval_ratio must equal 1.0")

        # Validate consensus
        if self.foundation_models.multi_encoder_enabled:
            if not self.foundation_models.additional_models:
                raise ValueError("additional_models required when multi_encoder_enabled is True")

            if not 0 < self.foundation_models.consensus_threshold <= 1:
                raise ValueError("consensus_threshold must be between 0 and 1")

        # Validate hypersphere configuration
        if self.hypersphere.radius_computation == "adaptive":
            if self.hypersphere.min_radius >= self.hypersphere.max_radius:
                raise ValueError("min_radius must be less than max_radius")

        # Validate nested shells
        if self.nested_shells.enabled:
            if not 0 < self.nested_shells.validation_ratio < 1:
                raise ValueError("validation_ratio must be between 0 and 1")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "BubbleFenceConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BubbleFenceConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        foundation_models = FoundationModelConfig(**config_dict["foundation_models"])

        # Create other config objects with defaults
        embedding = EmbeddingConfig(**config_dict.get("embedding", {}))
        deduplication = DeduplicationConfig(**config_dict.get("deduplication", {}))
        density_transformation = DensityTransformConfig(**config_dict.get("density_transformation", {}))
        anchor_placement = AnchorPlacementConfig(**config_dict.get("anchor_placement", {}))

        # Handle nested hypersphere config
        hypersphere_dict = config_dict.get("hypersphere", {})
        buffer_zones_dict = hypersphere_dict.pop("buffer_zones", {})
        buffer_zones = BufferZoneConfig(**buffer_zones_dict)
        hypersphere = HypersphereConfig(buffer_zones=buffer_zones, **hypersphere_dict)

        nested_shells = NestedShellConfig(**config_dict.get("nested_shells", {}))

        # Handle nested dataset split config
        dataset_splits_dict = config_dict.get("dataset_splits", {})
        ratio_monitoring_dict = dataset_splits_dict.pop("ratio_monitoring", {})
        ratio_monitoring = RatioMonitoringConfig(**ratio_monitoring_dict)
        dataset_splits = DatasetSplitConfig(ratio_monitoring=ratio_monitoring, **dataset_splits_dict)

        streaming = StreamingConfig(**config_dict.get("streaming", {}))

        # Handle nested ANN index config
        ann_index_dict = config_dict.get("ann_index", {})
        hnsw_dict = ann_index_dict.pop("hnsw", {})
        hnsw = HNSWConfig(**hnsw_dict)
        ann_index = ANNIndexConfig(hnsw=hnsw, **ann_index_dict)

        distance = DistanceConfig(**config_dict.get("distance", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        output = OutputConfig(**config_dict.get("output", {}))
        experimental = ExperimentalConfig(**config_dict.get("experimental", {}))
        preprocessing = PreprocessingConfig(**config_dict.get("preprocessing", {}))
        random_seed = RandomSeedConfig(**config_dict.get("random_seed", {}))
        class_aware = config_dict.get("class_aware", True)

        config = cls(
            foundation_models=foundation_models,
            embedding=embedding,
            deduplication=deduplication,
            density_transformation=density_transformation,
            anchor_placement=anchor_placement,
            hypersphere=hypersphere,
            nested_shells=nested_shells,
            dataset_splits=dataset_splits,
            streaming=streaming,
            ann_index=ann_index,
            distance=distance,
            logging=logging_config,
            output=output,
            experimental=experimental,
            preprocessing=preprocessing,
            random_seed=random_seed,
            class_aware=class_aware
        )

        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)