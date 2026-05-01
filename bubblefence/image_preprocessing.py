"""
Image preprocessing for BubbleFence foundation model pipelines.

Provides a factory that creates a **single shared** image processor for
the entire pipeline.  Resize, center-crop, and rescale run once through
the fastest available backend (CLIPImageProcessorFast on ROCm/CUDA,
CLIPImageProcessor on CPU).  Normalization is handled **separately** on
a per-model basis so that multi-encoder pipelines (CLIP + SigLIP + DINOv2)
share one preprocessing pass but each model gets its own mean/std.

Model-specific normalization constants
---------------------------------------
- CLIP / DINOv2:  OPENAI_CLIP_MEAN / OPENAI_CLIP_STD
- SigLIP:         IMAGENET_STANDARD_MEAN / IMAGENET_STANDARD_STD  (0.5, 0.5, 0.5)
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from PIL import Image

from .config import PreprocessingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Per-model normalization constants
# ---------------------------------------------------------------
# Source: transformers.image_utils
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

# Map model family -> (mean, std)
_MODEL_NORM_MAP: Dict[str, tuple] = {
    "clip":   (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    "dinov2": (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    "dino":   (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    "siglip": (IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD),
}


def _detect_model_family(model_name: str) -> str:
    """Return a canonical family key from a HuggingFace model id."""
    name = model_name.lower()
    if "siglip" in name:
        return "siglip"
    if "dinov2" in name:
        return "dinov2"
    if "dino" in name:
        return "dino"
    if "clip" in name:
        return "clip"
    return "clip"  # default fallback


def get_normalization_constants(model_name: str):
    """
    Return ``(mean, std)`` lists for a given model name.

    >>> mean, std = get_normalization_constants("openai/clip-vit-base-patch32")
    """
    family = _detect_model_family(model_name)
    return _MODEL_NORM_MAP.get(family, (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD))


def normalize_pixel_values(
    pixel_values: torch.Tensor,
    model_name: str,
) -> torch.Tensor:
    """
    Normalize a ``(B, C, H, W)`` tensor in-place using the correct
    mean/std for *model_name*.  Input is expected to be already rescaled
    to [0, 1] but **not** normalized.

    Args:
        pixel_values: (B, 3, H, W) float tensor on any device.
        model_name: HuggingFace model id.

    Returns:
        Normalized tensor (same object, modified in-place for speed).
    """
    mean_list, std_list = get_normalization_constants(model_name)
    device = pixel_values.device
    dtype = pixel_values.dtype

    mean = torch.tensor(mean_list, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=device, dtype=dtype).view(1, 3, 1, 1)

    pixel_values.sub_(mean).div_(std)
    return pixel_values


class ImagePreprocessor:
    """
    Shared image preprocessor for the BubbleFence pipeline.

    Handles resize, center-crop, rescale, and (optionally) normalization
    through the fastest available HuggingFace backend.

    In **multi-encoder mode** the preprocessor is created with
    ``do_normalize=False`` so that all models share one resize/crop pass
    and normalization is applied per-model via :func:`normalize_pixel_values`.

    In **single-encoder mode** normalization is included in the
    preprocessing pass for maximum throughput.

    Usage::

        # Single encoder (normalization included):
        preprocessor = ImagePreprocessor.from_config(config, model, device,
                                                      multi_encoder=False)
        inputs = preprocessor(images)
        embeddings = model(**inputs)

        # Multi encoder (normalization deferred):
        preprocessor = ImagePreprocessor.from_config(config, model, device,
                                                      multi_encoder=True)
        inputs = preprocessor(images)         # unnormalized pixel_values
        pv_clip = normalize_pixel_values(inputs["pixel_values"].clone(), "clip")
        pv_siglip = normalize_pixel_values(inputs["pixel_values"].clone(), "siglip")
    """

    def __init__(self, processor, device: torch.device, is_fast: bool):
        self._processor = processor
        self._device = device
        self._is_fast = is_fast

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        preprocess_config: PreprocessingConfig,
        model_name: str,
        device: torch.device,
        multi_encoder: bool = False,
    ) -> "ImagePreprocessor":
        """
        Build the best available image processor.

        Args:
            preprocess_config: ``PreprocessingConfig`` from YAML.
            model_name: HuggingFace model id used to load processor
                        weights / defaults (e.g. ``openai/clip-vit-base-patch32``).
            device: The torch device the pipeline runs on.
            multi_encoder: If True, disables normalization inside the
                           processor so it can be applied per-model later.

        Returns:
            An ``ImagePreprocessor`` ready to use.
        """
        use_gpu = device.type == "cuda"

        # Build kwargs from YAML overrides (None = keep model default)
        proc_kwargs: Dict[str, Any] = {}

        if preprocess_config.do_resize is not None:
            proc_kwargs["do_resize"] = preprocess_config.do_resize
        if preprocess_config.resize_size is not None:
            if isinstance(preprocess_config.resize_size, dict):
                proc_kwargs["size"] = preprocess_config.resize_size
            else:
                proc_kwargs["size"] = {"shortest_edge": preprocess_config.resize_size}
        if preprocess_config.do_center_crop is not None:
            proc_kwargs["do_center_crop"] = preprocess_config.do_center_crop
        if preprocess_config.crop_size is not None:
            proc_kwargs["crop_size"] = {
                "height": preprocess_config.crop_size,
                "width": preprocess_config.crop_size,
            }
        if preprocess_config.do_rescale is not None:
            proc_kwargs["do_rescale"] = preprocess_config.do_rescale
        if preprocess_config.do_convert_rgb is not None:
            proc_kwargs["do_convert_rgb"] = preprocess_config.do_convert_rgb

        # Normalization handling
        if multi_encoder:
            # Disable normalization so we can apply per-model later
            proc_kwargs["do_normalize"] = False
            logger.info(
                "Multi-encoder mode: normalization disabled in preprocessor "
                "(will be applied per-model)"
            )
        else:
            # Single encoder: include normalization in the preprocess pass
            if preprocess_config.do_normalize is not None:
                proc_kwargs["do_normalize"] = preprocess_config.do_normalize
            # Use model-specific mean/std unless overridden in YAML
            if preprocess_config.image_mean is not None:
                proc_kwargs["image_mean"] = preprocess_config.image_mean
            if preprocess_config.image_std is not None:
                proc_kwargs["image_std"] = preprocess_config.image_std

        # Decide backend
        is_fast = False
        if preprocess_config.use_fast == "auto":
            want_fast = use_gpu
        elif preprocess_config.use_fast == "always":
            want_fast = True
        else:  # "never"
            want_fast = False

        if want_fast:
            try:
                from transformers import CLIPImageProcessorFast

                processor = CLIPImageProcessorFast.from_pretrained(
                    model_name, **proc_kwargs
                )
                is_fast = True
                logger.info(
                    "Using CLIPImageProcessorFast (torchvision backend, "
                    "device=%s)",
                    device,
                )
            except Exception as e:
                logger.warning(
                    "CLIPImageProcessorFast unavailable (%s), "
                    "falling back to slow processor",
                    e,
                )
                want_fast = False

        if not is_fast:
            from transformers import CLIPImageProcessor

            processor = CLIPImageProcessor.from_pretrained(
                model_name, **proc_kwargs
            )
            logger.info(
                "Using %s (CPU backend)", type(processor).__name__
            )

        return cls(processor, device, is_fast)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        images: List[Image.Image],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a list of PIL images and return pixel_values on the
        pipeline's device.

        Args:
            images: List of PIL.Image.Image objects.
            return_tensors: Tensor format (always ``"pt"``).

        Returns:
            Dict with at least ``pixel_values`` key, tensor on self._device.
        """
        if self._is_fast:
            outputs = self._processor(
                images=images,
                return_tensors=return_tensors,
                device=str(self._device),
            )
        else:
            outputs = self._processor(
                images=images,
                return_tensors=return_tensors,
            )

        return {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
        }

    @property
    def backend_name(self) -> str:
        return type(self._processor).__name__

    @property
    def is_fast(self) -> bool:
        return self._is_fast
