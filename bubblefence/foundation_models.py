"""
Foundation model processing for BubbleFence semantic data splitting.

This module handles embedding generation using foundation models with optional
multi-encoder consensus mechanisms. Embeddings are kept as GPU tensors throughout
to avoid unnecessary CPU<->GPU transfers.

Image preprocessing is delegated to ImagePreprocessor which selects the
GPU-accelerated fast backend (CLIPImageProcessorFast / torchvision) on ROCm/CUDA
devices and falls back to the standard CPU processor otherwise.
"""

import os
import time
import torch
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from transformers import CLIPModel, AutoModel

from .config import BubbleFenceConfig, FoundationModelConfig
from .data_structures import EmbeddingPoint
from .device_utils import detect_device, to_numpy
from .image_preprocessing import ImagePreprocessor, normalize_pixel_values


logger = logging.getLogger(__name__)


class FoundationModelProcessor:
    """
    Handles embedding generation from foundation models with optional multi-encoder consensus.
    Embeddings stay as torch tensors on self.device until final output.
    """

    def __init__(self, config: BubbleFenceConfig, device: Optional[torch.device] = None):
        self.config = config
        self.fm_config = config.foundation_models
        self.embedding_config = config.embedding

        # Use shared device if provided, otherwise detect
        if device is not None:
            self.device = device
        else:
            self.device = detect_device(self.embedding_config.device)

        logger.info(f"Using device: {self.device}")

        # Cached embedding dimension (lazy, computed once on first call)
        self._embedding_dim: Optional[int] = None

        # Timing breakdown from last embed_images call
        self._last_timing: Dict[str, float] = {}

        # Persistent thread pool for parallel image I/O
        io_workers = self._resolve_io_workers(
            self.embedding_config.io_workers, self.device
        )
        self._io_pool = ThreadPoolExecutor(max_workers=io_workers)
        logger.info(f"I/O thread pool: {io_workers} workers (device={self.device})")

        self._multi_encoder = self.fm_config.multi_encoder_enabled

        # Build ONE shared image preprocessor for all models.
        # - Single encoder: normalization included in the preprocess pass.
        # - Multi encoder: normalization disabled here; applied per-model
        #   via normalize_pixel_values() before each model's forward pass.
        self.shared_preprocessor = ImagePreprocessor.from_config(
            self.config.preprocessing,
            self.fm_config.primary_model,
            self.device,
            multi_encoder=self._multi_encoder,
        )
        logger.info(
            "Shared preprocessor: backend=%s, fast=%s, multi_encoder=%s",
            self.shared_preprocessor.backend_name,
            self.shared_preprocessor.is_fast,
            self._multi_encoder,
        )

        # Load models (no per-model preprocessor - they all share one)
        self.primary_model = self._load_model(self.fm_config.primary_model)
        self.additional_models = {}

        if self._multi_encoder:
            for model_name in self.fm_config.additional_models:
                self.additional_models[model_name] = self._load_model(model_name)
                logger.info(f"Loaded additional model: {model_name}")

        logger.info(f"Foundation model processor initialized with {len(self.additional_models) + 1} models")

    def _load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a foundation model (weights only, no preprocessor).

        The image preprocessor is shared across all models and lives on
        ``self.shared_preprocessor``.
        """
        try:
            if "clip" in model_name.lower():
                model = CLIPModel.from_pretrained(model_name)
                model_type = "clip"
            else:
                model = AutoModel.from_pretrained(model_name)
                model_type = "generic"

            model = model.to(self.device)
            model.eval()

            return {
                'model': model,
                'model_type': model_type,
                'name': model_name
            }

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    @staticmethod
    def _resolve_io_workers(setting: str, device: torch.device) -> int:
        """Compute number of I/O workers from config setting and device."""
        setting = str(setting).strip().lower()
        if setting != "auto":
            try:
                return max(1, int(setting))
            except ValueError:
                logger.warning(f"Invalid io_workers value '{setting}', falling back to auto")

        cpu_count = os.cpu_count() or 4
        if str(device) == "cpu":
            return 2 if cpu_count > 2 else 1
        else:
            return min(16, max(1, int(cpu_count * 0.70)))

    @staticmethod
    def _load_single_image(path: Union[str, Path]) -> Image.Image:
        """Load a single PIL image from disk. Returns black placeholder on failure."""
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_copy = img.copy()
            img.close()
            return img_copy
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return Image.new('RGB', (224, 224), color='black')

    def _load_images(self, image_paths: List[Union[str, Path]]) -> List[Image.Image]:
        """Load PIL images from disk in parallel using thread pool."""
        return list(self._io_pool.map(self._load_single_image, image_paths))

    def _encode_images_single_model(self, image_paths: List[Union[str, Path]],
                                   model_info: Dict[str, Any]) -> torch.Tensor:
        """Encode images using a single model. Returns GPU tensor.

        Uses prefetching: batch N+1's images are loaded from disk in parallel
        while batch N is being preprocessed and run through the model on GPU.

        In single-encoder mode the shared preprocessor already includes
        normalization, so pixel_values go straight to the model.

        In multi-encoder mode the shared preprocessor outputs unnormalized
        pixel_values; this method applies per-model normalization before
        each forward pass.
        """
        embeddings = []
        batch_size = self.embedding_config.batch_size
        batch_slices = [image_paths[i:i + batch_size]
                        for i in range(0, len(image_paths), batch_size)]
        total_batches = len(batch_slices)

        load_time = 0.0
        preprocess_time = 0.0
        inference_time = 0.0

        # Kick off first batch load immediately
        t0 = time.time()
        prefetch_future = self._io_pool.submit(self._load_images, batch_slices[0])

        for batch_idx in range(total_batches):
            # Collect the prefetched images for this batch
            images = prefetch_future.result()
            load_time += time.time() - t0

            # Kick off next batch load while we do preprocess + inference
            if batch_idx + 1 < total_batches:
                t0 = time.time()
                prefetch_future = self._io_pool.submit(
                    self._load_images, batch_slices[batch_idx + 1]
                )

            # 2. Shared preprocess: resize, crop, rescale (+ normalize if single encoder)
            t1 = time.time()
            inputs = self.shared_preprocessor(images)

            # 3. Per-model normalization (multi-encoder only)
            if self._multi_encoder:
                pv = inputs["pixel_values"].clone()
                normalize_pixel_values(pv, model_info["name"])
                inputs["pixel_values"] = pv
            preprocess_time += time.time() - t1

            # 4. Forward pass
            t1 = time.time()
            batch_embeddings = self._forward(inputs, model_info)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time += time.time() - t1
            embeddings.append(batch_embeddings)

            logger.info(f"  Batch {batch_idx+1}/{total_batches}: "
                        f"{len(batch_slices[batch_idx])} images processed")

        # Concatenate all batch tensors (already on self.device)
        embeddings_tensor = torch.cat(embeddings, dim=0)

        # Cache embedding dimension from first real forward pass
        if self._embedding_dim is None:
            self._embedding_dim = embeddings_tensor.shape[1]

        # L2-normalize if requested (stays on GPU)
        if self.embedding_config.normalize_embeddings:
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

        # Store timing breakdown for caller to access
        self._last_timing = {
            'image_load_time': load_time,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
        }
        logger.info(f"  Embedding breakdown: "
                    f"load={load_time:.2f}s, preprocess={preprocess_time:.2f}s, "
                    f"inference={inference_time:.2f}s")

        return embeddings_tensor

    @staticmethod
    def _forward(inputs: Dict[str, torch.Tensor],
                 model_info: Dict[str, Any]) -> torch.Tensor:
        """Run the model forward pass and return the embedding tensor."""
        model = model_info['model']
        model_type = model_info['model_type']

        with torch.no_grad():
            if model_type == "clip":
                return model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state.mean(dim=1)
                else:
                    raise ValueError(
                        f"Could not extract embeddings from model {model_info['name']}"
                    )

    def _apply_consensus(self, all_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply consensus mechanism across multiple encoders. All tensors on GPU."""
        if not self.fm_config.multi_encoder_enabled:
            return all_embeddings[self.fm_config.primary_model]

        method = self.fm_config.consensus_method
        threshold = self.fm_config.consensus_threshold

        if method == "intersection":
            return self._intersection_consensus(all_embeddings, threshold)
        elif method == "union":
            return self._union_consensus(all_embeddings, threshold)
        elif method == "majority":
            return self._majority_consensus(all_embeddings, threshold)
        else:
            logger.warning(f"Unknown consensus method {method}, using primary model")
            return all_embeddings[self.fm_config.primary_model]

    def _intersection_consensus(self, all_embeddings: Dict[str, torch.Tensor],
                              threshold: float) -> torch.Tensor:
        """Intersection consensus: use primary model embeddings where all models agree."""
        primary_embeddings = all_embeddings[self.fm_config.primary_model]

        if len(all_embeddings) == 1:
            return primary_embeddings

        # Compute pairwise similarities between models (vectorized on GPU)
        model_names = list(all_embeddings.keys())
        num_samples = primary_embeddings.shape[0]

        consensus_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                emb1 = all_embeddings[model_names[i]]
                emb2 = all_embeddings[model_names[j]]

                # Cosine similarity (batch, vectorized on GPU)
                similarities = (emb1 * emb2).sum(dim=1) / (
                    emb1.norm(dim=1) * emb2.norm(dim=1) + 1e-8
                )
                consensus_mask &= (similarities >= threshold)

        agreed = consensus_mask.sum().item()
        logger.info(f"Intersection consensus: {agreed}/{num_samples} points agreed upon")

        # For points without consensus, use primary model
        return primary_embeddings

    def _union_consensus(self, all_embeddings: Dict[str, torch.Tensor],
                        threshold: float) -> torch.Tensor:
        """Union consensus: average embeddings where models agree. Vectorized on GPU."""
        primary_embeddings = all_embeddings[self.fm_config.primary_model]

        if len(all_embeddings) == 1:
            return primary_embeddings

        model_names = list(all_embeddings.keys())
        num_samples = primary_embeddings.shape[0]

        # Stack all model embeddings: (num_models, num_samples, dim)
        all_stacked = torch.stack([all_embeddings[name] for name in model_names])

        # Compute cosine similarity of each model with primary
        primary = primary_embeddings.unsqueeze(0)  # (1, N, D)
        sims = (all_stacked * primary).sum(dim=2) / (
            all_stacked.norm(dim=2) * primary.norm(dim=2) + 1e-8
        )  # (num_models, N)

        # Mask: which models agree for each sample
        agree_mask = sims >= threshold  # (num_models, N)

        # Weighted average: sum agreeing embeddings, divide by count
        agree_mask_expanded = agree_mask.unsqueeze(2).float()  # (num_models, N, 1)
        weighted_sum = (all_stacked * agree_mask_expanded).sum(dim=0)  # (N, D)
        counts = agree_mask_expanded.sum(dim=0).clamp(min=1.0)  # (N, 1)
        consensus_embeddings = weighted_sum / counts

        return consensus_embeddings

    def _majority_consensus(self, all_embeddings: Dict[str, torch.Tensor],
                          threshold: float) -> torch.Tensor:
        """Majority consensus: use embedding where majority of models agree."""
        primary_embeddings = all_embeddings[self.fm_config.primary_model]

        if len(all_embeddings) == 1:
            return primary_embeddings

        return self._union_consensus(all_embeddings, threshold)

    def embed_images(self, image_paths: List[Union[str, Path]],
                    original_indices: Optional[List[int]] = None,
                    metadata: Optional[List[Dict[str, Any]]] = None
                    ) -> Tuple[torch.Tensor, List[EmbeddingPoint]]:
        """
        Generate embeddings for a list of images with optional multi-encoder consensus.

        Returns a GPU tensor + metadata-only EmbeddingPoint list (no embedding duplication).

        In single-encoder mode each batch is: load -> preprocess (with norm) -> forward.

        In multi-encoder mode each batch is: load -> preprocess (no norm) -> for each
        model: clone pixel_values, normalize per-model, forward.  This means images are
        loaded from disk and preprocessed exactly **once** regardless of how many models
        are configured.

        Args:
            image_paths: List of paths to image files
            original_indices: Original indices for tracking (defaults to enumerate)
            metadata: Optional metadata for each image

        Returns:
            Tuple of:
                - (N, D) tensor of embeddings on self.device
                - List of EmbeddingPoint metadata objects
        """
        logger.info(f"Generating embeddings for {len(image_paths)} images")

        if original_indices is None:
            original_indices = list(range(len(image_paths)))

        if metadata is None:
            metadata = [{}] * len(image_paths)

        if self._multi_encoder:
            # Multi-encoder: preprocess once, normalize + forward per model
            all_embeddings = self._embed_multi_encoder(image_paths)
        else:
            # Single-encoder: straightforward batch processing
            all_embeddings = {
                self.fm_config.primary_model: self._encode_images_single_model(
                    image_paths, self.primary_model
                )
            }

        # Apply consensus (stays on GPU)
        final_embeddings = self._apply_consensus(all_embeddings)  # (N, D) GPU tensor

        # Create metadata-only EmbeddingPoint objects (no embedding stored)
        embedding_points = []
        for i, (orig_idx, meta) in enumerate(zip(original_indices, metadata)):
            point = EmbeddingPoint(
                original_index=orig_idx,
                file_path=str(image_paths[i]) if i < len(image_paths) else None,
                metadata=meta
            )
            embedding_points.append(point)

        logger.info(f"Generated {len(embedding_points)} embedding points")
        return final_embeddings, embedding_points

    def _embed_multi_encoder(
        self, image_paths: List[Union[str, Path]]
    ) -> Dict[str, torch.Tensor]:
        """Multi-encoder path: load + preprocess once, normalize + forward per model.

        For each batch we:
          1. Load PIL images from disk (once).
          2. Run the shared preprocessor (resize/crop/rescale, NO normalization).
          3. For every model: clone pixel_values, apply model-specific
             normalization, run the forward pass.

        This avoids redundant disk I/O and preprocessing across encoders.
        """
        all_models = {self.fm_config.primary_model: self.primary_model}
        all_models.update(self.additional_models)

        # Accumulate per-model batch embeddings
        per_model_batches: Dict[str, List[torch.Tensor]] = {
            name: [] for name in all_models
        }

        batch_size = self.embedding_config.batch_size

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # 1. Load images (once)
            images = self._load_images(batch_paths)

            # 2. Shared preprocess (no normalization)
            inputs = self.shared_preprocessor(images)
            unnormed_pv = inputs["pixel_values"]  # (B, 3, H, W) on device

            # 3. Per-model: normalize clone -> forward
            for model_name, model_info in all_models.items():
                pv = unnormed_pv.clone()
                normalize_pixel_values(pv, model_name)
                model_inputs = {**inputs, "pixel_values": pv}
                batch_emb = self._forward(model_inputs, model_info)
                per_model_batches[model_name].append(batch_emb)

        # Concatenate and optionally L2-normalize
        all_embeddings: Dict[str, torch.Tensor] = {}
        for model_name, batches in per_model_batches.items():
            emb = torch.cat(batches, dim=0)

            # Cache dim from first model's first result
            if self._embedding_dim is None:
                self._embedding_dim = emb.shape[1]

            if self.embedding_config.normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings[model_name] = emb

        logger.info(
            "Multi-encoder embedding complete: %s",
            ", ".join(f"{n}={e.shape}" for n, e in all_embeddings.items()),
        )
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this processor.

        Result is cached after the first real forward pass.  If no images
        have been processed yet, runs a single dummy image through the
        primary model to discover the dimension.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        # Single dummy forward pass (in-memory, no disk I/O)
        logger.debug("Computing embedding dimension from dummy image")
        dummy = Image.new('RGB', (224, 224), color='black')
        inputs = self.shared_preprocessor([dummy])

        # In multi-encoder mode the preprocessor skips normalization,
        # so apply the primary model's normalization before the forward pass.
        if self._multi_encoder:
            pv = inputs["pixel_values"].clone()
            normalize_pixel_values(pv, self.primary_model["name"])
            inputs["pixel_values"] = pv

        emb = self._forward(inputs, self.primary_model)
        self._embedding_dim = emb.shape[1]
        return self._embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'primary_model': self.fm_config.primary_model,
            'device': str(self.device),
            'multi_encoder_enabled': self._multi_encoder,
            'embedding_dimension': self.get_embedding_dimension(),
            'preprocessor_backend': self.shared_preprocessor.backend_name,
            'preprocessor_fast': self.shared_preprocessor.is_fast,
        }

        if self._multi_encoder:
            info['additional_models'] = self.fm_config.additional_models
            info['consensus_method'] = self.fm_config.consensus_method
            info['consensus_threshold'] = self.fm_config.consensus_threshold

        return info