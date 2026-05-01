"""
Main BubbleFence pipeline for semantic data splitting using hypersphere bubbles.

GPU-accelerated: all embedding, distance, and assignment operations run on a
shared torch device. numpy is only used at I/O boundaries (CSV, pickle, .pt files).
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import time
import pickle
import torch
import hashlib

from .config import BubbleFenceConfig
from .data_structures import (
    EmbeddingTrajectory, EmbeddingPoint, HypersphereAnchor,
    AnchorRegistry, DatasetAssignmentResult, DatasetSplit, NestedShell
)
from .foundation_models import FoundationModelProcessor
from .density_analysis import DensityAnalyzer
from .anchor_placement import AnchorPlacer
from .device_utils import (
    detect_device, to_tensor, to_numpy,
    pairwise_distance_matrix, pairwise_distance_cross
)


logger = logging.getLogger(__name__)


class BubbleFencePipeline:
    """
    Main pipeline for BubbleFence semantic data splitting.

    Integrates foundation model processing, density analysis, anchor placement,
    and hypersphere-based dataset assignment. All tensor operations run on a
    shared device (GPU when available).
    """

    def __init__(self, config: BubbleFenceConfig,
                 full_dataset_csv_path: Optional[str] = None):
        """
        Initialize the BubbleFence pipeline.

        Args:
            config: Configuration object with all pipeline parameters
            full_dataset_csv_path: Optional path to cumulative full_dataset.csv.
                None on the very first run. When provided, the CSV's
                dataset_split column is used to build a boolean mask over
                self.all_embeddings so we can extract TRAIN-only embeddings
                for collision checking (no train point may fall inside a bubble).
        """
        self.config = config

        # Detect and set shared device for the entire pipeline
        self.device = detect_device(config.embedding.device)

        # Initialize components with shared device
        self.anchor_registry = AnchorRegistry(device=self.device)
        self.foundation_processor = FoundationModelProcessor(config, device=self.device)
        self.density_analyzer = DensityAnalyzer(config, device=self.device)
        self.anchor_placer = AnchorPlacer(config, self.foundation_processor,
                                                 device=self.device,
                                                 density_analyzer=self.density_analyzer)

        # Performance monitoring
        self.performance_stats = {
            'total_processed': 0,
            'current_batch_processed': 0,
            'total_time': 0,
            'embedding_time': 0,
            'image_load_time': 0,
            'preprocess_time': 0,
            'inference_time': 0,
            'dedup_time': 0,
            'density_time': 0,
            'anchor_time': 0,
            'assignment_time': 0,
            'storage_time': 0,
        }

        # Continual learning: embeddings storage for cross-batch deduplication
        # Stored as a GPU tensor for fast cross-batch similarity computation
        self.embeddings_storage_dir = None
        self.all_embeddings: Optional[torch.Tensor] = None  # (M, D) on self.device
        self.all_labels: Optional[List[Optional[str]]] = None  # per-point class labels
        if config.streaming.persistent_anchors and config.streaming.enabled:
            # Create embeddings storage directory
            storage_path = Path(config.streaming.anchor_persistence_path).parent / "embeddings"
            storage_path.mkdir(exist_ok=True)
            self.embeddings_storage_dir = storage_path
            logger.info(f"Embeddings storage at {storage_path}")

            # Load all stored embeddings + labels once upfront
            self.all_embeddings, self.all_labels = self._load_all_stored_embeddings()
            if self.all_embeddings is not None and self.all_embeddings.shape[0] > 0:
                logger.info(f"Loaded {self.all_embeddings.shape[0]} stored embeddings from previous runs")

        # Load persistent state if configured
        if config.streaming.persistent_anchors and config.streaming.enabled:
            self._load_persistent_state()

        # ---- Closed-loop collision avoidance ----
        # Boolean mask into self.all_embeddings: True = train, False = eval
        # Use self.all_embeddings[self._train_mask] to get train-only slice
        self._train_mask: Optional[torch.Tensor] = None  # (M,) bool
        self.stored_train_count: int = 0
        self.stored_eval_count: int = 0
        self.stored_class_eval_counts: Dict[str, int] = {}
        self.stored_class_total_counts: Dict[str, int] = {}
        self.full_dataset_csv_path = full_dataset_csv_path

        if full_dataset_csv_path and Path(full_dataset_csv_path).exists():
            self._load_split_mask_from_csv(full_dataset_csv_path)

        logger.info(f"BubbleFence pipeline initialized (device: {self.device})")

    def process_image_stream(self, image_paths: List[Union[str, Path]],
                           trajectory_id: str = "default",
                           metadata: Optional[List[Dict[str, Any]]] = None,
                           original_indices: Optional[List[int]] = None) -> DatasetAssignmentResult:
        """
        Process a stream of images and assign them to train/validation/test sets.

        Closed-loop flow:
          1. Embed + cross-batch dedup
          2. Create trajectory, within-batch dedup, density transform
          3. Preliminary assignment against existing anchors only:
             inside bubble -> val/test, outside -> UNASSIGNED
          4. Compute eval deficit across stored + new points
             (UNASSIGNED count as train for this calculation)
          5. If deficit > 0: over-propose candidates once, then iterate
             per-candidate: validate (no train collisions), register,
             assign captured UNASSIGNED points, update running eval count,
             break as soon as deficit is filled
          6. Remaining UNASSIGNED -> TRAIN
          7. Build final DatasetAssignmentResult
          8. Store embeddings, save persistent state
        """
        logger.info(f"Processing {len(image_paths)} images for trajectory '{trajectory_id}'")
        start_time = time.time()

        # ---- Step 1: Embed ----
        embedding_start = time.time()
        embeddings_tensor, embedding_points = self.foundation_processor.embed_images(
            image_paths, original_indices=original_indices, metadata=metadata
        )
        embedding_elapsed = time.time() - embedding_start
        self.performance_stats['embedding_time'] += embedding_elapsed

        # Pull sub-timers from the foundation processor
        sub_timing = getattr(self.foundation_processor, '_last_timing', {})
        self.performance_stats['image_load_time'] += sub_timing.get('image_load_time', 0)
        self.performance_stats['preprocess_time'] += sub_timing.get('preprocess_time', 0)
        self.performance_stats['inference_time'] += sub_timing.get('inference_time', 0)

        # ---- Step 2: Cross-batch dedup ----
        dedup_start = time.time()
        if self.config.streaming.persistent_anchors and self.config.streaming.enabled and self.embeddings_storage_dir:
            embeddings_tensor, embedding_points = self._deduplicate_against_stored_embeddings(
                embeddings_tensor, embedding_points
            )
            logger.info(f"{len(embedding_points)} embeddings after cross-batch deduplication")

            if len(embedding_points) == 0:
                logger.info("All images were duplicates of previously ingested data.")
                total_time = time.time() - start_time
                self.performance_stats['current_batch_processed'] = len(image_paths)
                self.performance_stats['total_processed'] += len(image_paths)
                self.performance_stats['total_time'] += total_time
                return DatasetAssignmentResult(
                    train_indices=[], validation_indices=[], test_indices=[],
                    unassigned_indices=[], assignment_details=[],
                    statistics={'total_points': 0, 'train_count': 0, 'validation_count': 0,
                                'test_count': 0, 'unassigned_count': 0, 'train_ratio': 0,
                                'validation_ratio': 0, 'test_ratio': 0, 'eval_ratio': 0,
                                'coverage_ratio': 0, 'num_anchors_used': self.anchor_registry.num_anchors}
                )

        # ---- Step 3: Create trajectory, within-batch dedup, density transform ----
        trajectory = EmbeddingTrajectory(trajectory_id, device=self.device)
        trajectory.set_embeddings(embeddings_tensor, embedding_points)

        logger.info(f"Processing {trajectory.num_points} embeddings")
        if self.anchor_registry.num_anchors > 0:
            logger.info(f"Using {self.anchor_registry.num_anchors} existing anchors")

        if self.config.deduplication.enabled:
            trajectory = self._deduplicate_trajectory(trajectory)

        density_start = time.time()
        self.performance_stats['dedup_time'] += density_start - dedup_start

        if self.config.density_transformation.enabled:
            trajectory = self.density_analyzer.apply_density_transformation(trajectory)
        self.performance_stats['density_time'] += time.time() - density_start

        embeddings = trajectory.embeddings_matrix  # (N, D) GPU tensor
        N = trajectory.num_points

        # ---- Step 4: Preliminary assignment against existing anchors ----
        anchor_start = time.time()

        point_splits: List[DatasetSplit] = [DatasetSplit.UNASSIGNED] * N
        point_anchor_ids: List[Optional[int]] = [None] * N
        point_distances: List[Optional[float]] = [None] * N

        if self.anchor_registry.num_anchors > 0:
            prelim = self.anchor_registry.batch_assign(
                embeddings, self.config.distance.metric
            )
            for i, (anchor_id, split, dist) in enumerate(prelim):
                if anchor_id is not None:
                    point_splits[i] = split
                    point_anchor_ids[i] = anchor_id
                    point_distances[i] = dist

        new_eval_count = sum(1 for s in point_splits
                            if s in (DatasetSplit.VALIDATION, DatasetSplit.TEST))
        new_unassigned_indices = [i for i, s in enumerate(point_splits)
                                  if s == DatasetSplit.UNASSIGNED]

        logger.info(f"Preliminary: {new_eval_count} eval, "
                    f"{len(new_unassigned_indices)} unassigned / {N} new points")

        # ---- Step 5: Compute eval deficit ----
        total_dataset_size = self.stored_train_count + self.stored_eval_count + N
        target_eval_count = int(total_dataset_size * self.config.dataset_splits.eval_ratio)
        current_eval_count = self.stored_eval_count + new_eval_count
        eval_deficit = target_eval_count - current_eval_count

        logger.info(f"=== EVAL DEFICIT === "
                    f"stored={self.stored_train_count}T/{self.stored_eval_count}E, "
                    f"batch={N}, target_eval={target_eval_count}, "
                    f"current_eval={current_eval_count}, deficit={eval_deficit}")

        # ---- Step 6: Closed-loop per-candidate bubble placement ----
        # Minimum eval per batch: even when deficit <= 0, if this batch has
        # enough unassigned points and would get 0 eval, force a small anchor
        # placement so every ingested folder gets some eval representation.
        min_eval_per_batch = self.config.dataset_splits.min_eval_per_batch
        batch_min_eval = int(N * min_eval_per_batch)
        force_min_eval = (eval_deficit <= 0
                          and new_eval_count == 0
                          and batch_min_eval >= 1
                          and len(new_unassigned_indices) > 0)

        effective_deficit = eval_deficit
        if force_min_eval:
            effective_deficit = batch_min_eval
            logger.info(f"Min-eval-per-batch: forcing deficit={effective_deficit} "
                        f"({min_eval_per_batch:.0%} of {N} batch points)")

        # Check if class-aware placement should be used
        batch_labels = [
            trajectory.points[i].metadata.get('label') if trajectory.points[i].metadata else None
            for i in range(N)
        ]
        has_labels = self.config.class_aware and any(l is not None for l in batch_labels)

        if effective_deficit > 0 and len(new_unassigned_indices) > 0:
            if has_labels:
                self._class_aware_anchor_placement(
                    trajectory, embeddings, point_splits,
                    point_anchor_ids, point_distances,
                    new_unassigned_indices, effective_deficit,
                    batch_labels, target_eval_count
                )
            else:
                self._closed_loop_anchor_placement(
                    trajectory, embeddings, point_splits,
                    point_anchor_ids, point_distances,
                    new_unassigned_indices, effective_deficit
                )

        self.performance_stats['anchor_time'] += time.time() - anchor_start

        # ---- Step 7: UNASSIGNED -> TRAIN ----
        for i in range(N):
            if point_splits[i] == DatasetSplit.UNASSIGNED:
                point_splits[i] = DatasetSplit.TRAIN

        # ---- Step 8: Build result ----
        assignment_start = time.time()
        assignment_result = self._build_assignment_result(
            trajectory, point_splits, point_anchor_ids, point_distances
        )
        self.performance_stats['assignment_time'] += time.time() - assignment_start

        # ---- Step 9: Store embeddings + save state ----
        storage_start = time.time()
        if self.config.streaming.persistent_anchors and self.config.streaming.enabled and self.embeddings_storage_dir:
            if trajectory.num_points > 0:
                self._store_new_embeddings(trajectory, trajectory_id)
        self.performance_stats['storage_time'] += time.time() - storage_start

        total_time = time.time() - start_time
        self.performance_stats['current_batch_processed'] = len(image_paths)
        self.performance_stats['total_processed'] += len(image_paths)
        self.performance_stats['total_time'] += total_time

        # Always log the timing breakdown
        self._log_performance_stats()

        if self.config.streaming.persistent_anchors and self.config.streaming.enabled:
            self._save_persistent_state()

        logger.info(f"Completed processing trajectory '{trajectory_id}' in {total_time:.2f}s")
        return assignment_result

    def _closed_loop_anchor_placement(
            self,
            trajectory: EmbeddingTrajectory,
            embeddings: torch.Tensor,
            point_splits: List[DatasetSplit],
            point_anchor_ids: List[Optional[int]],
            point_distances: List[Optional[float]],
            unassigned_indices: List[int],
            eval_deficit: int) -> None:
        """
        Over-propose anchor candidates once, then iterate per-candidate:
        validate against train embeddings, register, assign captured
        UNASSIGNED points, and stop as soon as the eval deficit is filled.

        Eval capping: if accepting an anchor would push the remaining deficit
        below -max_overshoot (derived from eval_tolerance), shrink the bubble
        radius via binary search so we stay within the tolerance band.

        Mutates point_splits / point_anchor_ids / point_distances in place.
        """
        logger.info(f"Need {eval_deficit} more eval points -- entering closed loop")

        # Eval overshoot cap: how many eval points beyond the deficit are we
        # willing to accept from a single anchor before shrinking its radius.
        total_dataset_size = self.stored_train_count + self.stored_eval_count + embeddings.shape[0]
        max_overshoot = int(total_dataset_size * self.config.dataset_splits.eval_tolerance)
        max_overshoot = max(max_overshoot, 1)  # always allow at least 1
        logger.info(f"Eval cap: max_overshoot={max_overshoot} "
                    f"({self.config.dataset_splits.eval_tolerance:.0%} of {total_dataset_size})")

        # Train embeddings for collision checking (slice from all_embeddings)
        train_embs: Optional[torch.Tensor] = None
        if (self._train_mask is not None
                and self.all_embeddings is not None
                and self.all_embeddings.shape[0] > 0):
            train_embs = self.all_embeddings[self._train_mask]
            logger.info(f"Train embeddings for collision check: {train_embs.shape[0]}")

        # Pool of UNASSIGNED embeddings to sample candidate positions from
        unassigned_embs = embeddings[unassigned_indices]  # (U, D)

        existing_centers = [a.center for a in self.anchor_registry.anchors.values()]

        # Over-propose ONCE
        candidates = self.anchor_placer.propose_candidates(
            unassigned_embs, existing_centers, overpropose_factor=3
        )
        logger.info(f"Proposed {len(candidates)} anchor candidates")

        # Per-candidate inner loop
        accepted = 0
        remaining_deficit = eval_deficit
        for cand_idx, (center, base_radius) in enumerate(candidates):
            if remaining_deficit <= 0:
                logger.info(f"Eval deficit filled after {accepted} accepted anchors")
                break

            # Adaptive radius
            if self.config.hypersphere.radius_computation == "adaptive":
                radius = self.density_analyzer.compute_adaptive_radius(
                    center, embeddings, base_radius,
                    self.config.hypersphere.adaptive_method
                )
            else:
                radius = base_radius

            # Validate: no train collisions (shrink or discard)
            validated = self.anchor_placer.validate_candidate(
                center, radius, train_embs
            )
            if validated is None:
                continue

            center, final_radius = validated

            # --- Eval capping: shrink bubble if it captures too many eval ---
            final_radius = self._cap_eval_radius(
                center, final_radius, embeddings, point_splits,
                unassigned_indices, remaining_deficit, max_overshoot
            )

            # Build shell config before registering
            nested_shells = self.anchor_placer._create_nested_shells(final_radius)
            buffer_radius = (self.config.hypersphere.buffer_zones.buffer_radius
                             if self.config.hypersphere.buffer_zones.enabled else 0.0)
            total_radius = final_radius + buffer_radius

            # Pre-scan: find which UNASSIGNED points this candidate would capture
            center_2d = center.unsqueeze(0)
            temp_anchor = HypersphereAnchor(
                anchor_id=-1, center=center,
                base_radius=final_radius, buffer_radius=buffer_radius,
                nested_shells=nested_shells
            )

            captured = []  # (ui, split, dist)
            for ui in unassigned_indices:
                if point_splits[ui] != DatasetSplit.UNASSIGNED:
                    continue

                dist = pairwise_distance_cross(
                    embeddings[ui].unsqueeze(0), center_2d,
                    self.config.distance.metric
                ).item()

                if dist <= total_radius:
                    split = temp_anchor.get_point_assignment(dist)
                    captured.append((ui, split, dist))

            eval_added = sum(1 for _, s, _ in captured
                             if s in (DatasetSplit.VALIDATION, DatasetSplit.TEST))

            if eval_added <= 1:
                logger.debug(f"Candidate {cand_idx} skipped: captures {eval_added} eval points "
                             f"(need >1 to exclude center-only anchors)")
                continue

            # Register now that we know it captures eval points
            anchor_id = self.anchor_registry.add_anchor(
                center=center,
                base_radius=final_radius,
                buffer_radius=buffer_radius,
                nested_shells=nested_shells
            )

            # Apply captured assignments
            for ui, split, dist in captured:
                point_splits[ui] = split
                point_anchor_ids[ui] = anchor_id
                point_distances[ui] = dist
                if split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
                    remaining_deficit -= 1

            accepted += 1
            logger.info(f"Anchor {anchor_id}: r={final_radius:.6f}, "
                        f"+{eval_added} eval, deficit={remaining_deficit}")

        logger.info(f"Closed loop done: {accepted} anchors accepted, "
                    f"deficit remaining={remaining_deficit}")

    def _class_aware_anchor_placement(
            self,
            trajectory: EmbeddingTrajectory,
            embeddings: torch.Tensor,
            point_splits: List[DatasetSplit],
            point_anchor_ids: List[Optional[int]],
            point_distances: List[Optional[float]],
            unassigned_indices: List[int],
            eval_deficit: int,
            batch_labels: List[Optional[str]],
            target_eval_count: int) -> None:
        """
        Class-aware anchor placement using priority scheduling.

        Rare classes go first, each filled to 80% of per-class eval target.
        Then a final mop-up pass fills remaining global deficit class-blind.
        Bubbles for dominant classes are shrunk if they would bleed too many
        eval points from already-satisfied rare classes.

        Mutates point_splits / point_anchor_ids / point_distances in place.
        """
        from collections import defaultdict

        logger.info(f"Class-aware anchor placement: deficit={eval_deficit}")

        eval_ratio = self.config.dataset_splits.eval_ratio
        total_dataset_size = self.stored_train_count + self.stored_eval_count + embeddings.shape[0]
        max_overshoot = max(1, int(total_dataset_size * self.config.dataset_splits.eval_tolerance))

        # Train embeddings for collision checking
        train_embs: Optional[torch.Tensor] = None
        if (self._train_mask is not None
                and self.all_embeddings is not None
                and self.all_embeddings.shape[0] > 0):
            train_embs = self.all_embeddings[self._train_mask]

        existing_centers = [a.center for a in self.anchor_registry.anchors.values()]

        # Build per-class counts (batch + stored)
        class_batch_counts: Dict[str, int] = defaultdict(int)
        class_batch_eval: Dict[str, int] = defaultdict(int)
        class_unassigned: Dict[str, List[int]] = defaultdict(list)

        for i in range(embeddings.shape[0]):
            label = batch_labels[i]
            if label is None:
                continue
            label_lower = label.lower()
            class_batch_counts[label_lower] += 1
            if point_splits[i] in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
                class_batch_eval[label_lower] += 1
            if point_splits[i] == DatasetSplit.UNASSIGNED:
                class_unassigned[label_lower].append(i)

        # Merge with stored counts
        class_total: Dict[str, int] = defaultdict(int)
        class_current_eval: Dict[str, int] = defaultdict(int)
        for cls in set(list(class_batch_counts.keys()) + list(self.stored_class_total_counts.keys())):
            class_total[cls] = self.stored_class_total_counts.get(cls, 0) + class_batch_counts.get(cls, 0)
            class_current_eval[cls] = self.stored_class_eval_counts.get(cls, 0) + class_batch_eval.get(cls, 0)

        # Per-class eval targets
        class_targets: Dict[str, int] = {}
        for cls, total in class_total.items():
            class_targets[cls] = max(1, int(total * eval_ratio))

        # Sort by count ascending (rare first)
        classes_sorted = sorted(class_total.keys(), key=lambda c: class_total[c])

        logger.info(f"Class-aware placement: classes={classes_sorted}, "
                    f"targets={dict(class_targets)}, current_eval={dict(class_current_eval)}")

        # Compute base_radius ONCE from all unassigned embeddings
        all_unassigned_embs = embeddings[unassigned_indices]
        if all_unassigned_embs.shape[0] == 0:
            return
        base_radius = self.anchor_placer._compute_base_radius(all_unassigned_embs)

        # Per-class candidate proposal (cache upfront)
        # Skip classes with too few points -- they participate in the mop-up
        # pass instead, where nearby bubbles can capture them if the global
        # deficit still needs filling.  Otherwise they just go to train.
        min_class_size = 5
        class_candidates: Dict[str, List[Tuple[torch.Tensor, float]]] = {}
        for cls in classes_sorted:
            if cls not in class_unassigned or not class_unassigned[cls]:
                continue
            if len(class_unassigned[cls]) < min_class_size:
                logger.info(f"Class '{cls}' has only {len(class_unassigned[cls])} "
                            f"unassigned points (< {min_class_size}), deferring to mop-up")
                continue
            cls_embs = embeddings[class_unassigned[cls]]
            # Scale candidate count by class size relative to total
            num_cands = max(3, int(len(class_unassigned[cls]) * 0.5) * 3)
            cls_cands = self.anchor_placer.propose_candidates_for_class(
                cls_embs, existing_centers, base_radius, num_cands
            )
            class_candidates[cls] = cls_cands

        # Convert to iterators
        class_iters: Dict[str, int] = {cls: 0 for cls in class_candidates}

        # Satisfaction ratio
        def satisfaction(cls: str) -> float:
            target = class_targets.get(cls, 1)
            return class_current_eval.get(cls, 0) / max(1, target)

        # Total eval running count
        total_eval = sum(class_current_eval.values()) + self.stored_eval_count
        # Subtract stored to avoid double counting (class_current_eval already includes stored)
        total_eval = self.stored_eval_count + sum(class_batch_eval.values())

        accepted = 0
        fill_target = 0.8  # fill each class to 80% of target

        # Phase 1: Per-class, rare first, fill to 80%
        for cls in classes_sorted:
            if cls not in class_candidates:
                continue

            cls_fill_target = int(class_targets[cls] * fill_target)
            candidates = class_candidates[cls]
            cand_idx = class_iters[cls]

            # Allow placement past global budget if this class is severely
            # underrepresented (< 50% of its per-class target)
            global_ok = total_eval < target_eval_count
            class_severely_under = satisfaction(cls) < 0.5
            while cand_idx < len(candidates) and (global_ok or class_severely_under):
                if class_current_eval[cls] >= cls_fill_target:
                    break  # this class is at 80%

                center, _ = candidates[cand_idx]
                cand_idx += 1

                # Adaptive radius from ALL embeddings (global density awareness)
                if self.config.hypersphere.radius_computation == "adaptive":
                    radius = self.density_analyzer.compute_adaptive_radius(
                        center, embeddings, base_radius,
                        self.config.hypersphere.adaptive_method
                    )
                else:
                    radius = base_radius

                # Validate: no train collisions
                validated = self.anchor_placer.validate_candidate(center, radius, train_embs)
                if validated is None:
                    continue
                center, final_radius = validated

                # Global eval cap
                remaining_deficit = target_eval_count - total_eval
                final_radius = self._cap_eval_radius(
                    center, final_radius, embeddings, point_splits,
                    unassigned_indices, remaining_deficit, max_overshoot
                )

                # Scan captured points
                nested_shells = self.anchor_placer._create_nested_shells(final_radius)
                buffer_radius = (self.config.hypersphere.buffer_zones.buffer_radius
                                 if self.config.hypersphere.buffer_zones.enabled else 0.0)
                total_radius = final_radius + buffer_radius
                center_2d = center.unsqueeze(0)
                temp_anchor = HypersphereAnchor(
                    anchor_id=-1, center=center,
                    base_radius=final_radius, buffer_radius=buffer_radius,
                    nested_shells=nested_shells
                )

                captured = []
                for ui in unassigned_indices:
                    if point_splits[ui] != DatasetSplit.UNASSIGNED:
                        continue
                    dist = pairwise_distance_cross(
                        embeddings[ui].unsqueeze(0), center_2d,
                        self.config.distance.metric
                    ).item()
                    if dist <= total_radius:
                        split = temp_anchor.get_point_assignment(dist)
                        captured.append((ui, split, dist))

                # Check focus class gained eval
                focus_eval = sum(1 for ui, s, _ in captured
                                 if s in (DatasetSplit.VALIDATION, DatasetSplit.TEST)
                                 and batch_labels[ui] is not None
                                 and batch_labels[ui].lower() == cls)
                total_eval_added = sum(1 for _, s, _ in captured
                                       if s in (DatasetSplit.VALIDATION, DatasetSplit.TEST))

                if total_eval_added <= 1:
                    continue

                # Register anchor
                anchor_id = self.anchor_registry.add_anchor(
                    center=center, base_radius=final_radius,
                    buffer_radius=buffer_radius, nested_shells=nested_shells
                )
                existing_centers.append(center)

                # Apply assignments and update counts
                for ui, split, dist in captured:
                    point_splits[ui] = split
                    point_anchor_ids[ui] = anchor_id
                    point_distances[ui] = dist
                    if split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
                        total_eval += 1
                        pt_label = batch_labels[ui]
                        if pt_label is not None:
                            class_current_eval[pt_label.lower()] += 1

                accepted += 1
                global_ok = total_eval < target_eval_count
                class_severely_under = satisfaction(cls) < 0.5
                logger.info(f"[class={cls}] Anchor {anchor_id}: r={final_radius:.6f}, "
                            f"+{total_eval_added} eval (focus={focus_eval}), "
                            f"satisfaction={satisfaction(cls):.2f}")

            class_iters[cls] = cand_idx

        # Phase 2: Mop-up pass (class-blind) for remaining global deficit
        remaining_global = target_eval_count - total_eval
        remaining_unassigned = [i for i in unassigned_indices
                                if point_splits[i] == DatasetSplit.UNASSIGNED]

        if remaining_global > 0 and remaining_unassigned:
            logger.info(f"Mop-up pass: {remaining_global} remaining global deficit")
            self._closed_loop_anchor_placement(
                trajectory, embeddings, point_splits,
                point_anchor_ids, point_distances,
                remaining_unassigned, remaining_global
            )

        final_total_eval = total_eval + max(0, remaining_global - (target_eval_count - total_eval))
        logger.info(f"Class-aware placement done: {accepted} class-aware anchors, "
                    f"per-class eval={dict(class_current_eval)}")

    def _cap_eval_radius(
            self,
            center: torch.Tensor,
            radius: float,
            embeddings: torch.Tensor,
            point_splits: List[DatasetSplit],
            unassigned_indices: List[int],
            remaining_deficit: int,
            max_overshoot: int,
            shrink_steps: int = 5) -> float:
        """
        Binary-search shrink the bubble radius if it would capture too many
        eval points (pushing deficit beyond -max_overshoot).

        The allowed eval capture is: remaining_deficit + max_overshoot.
        If the bubble at `radius` would exceed that, halve the radius
        repeatedly until it fits or we hit min_radius.

        Returns the (possibly shrunk) radius.
        """
        allowed_eval = remaining_deficit + max_overshoot

        # Quick count of eval captures at current radius
        eval_count = self._count_eval_captures(center, radius, embeddings,
                                                point_splits, unassigned_indices)

        if eval_count <= allowed_eval:
            return radius  # no shrinking needed

        logger.info(f"Eval cap triggered: {eval_count} eval > allowed {allowed_eval}, "
                    f"shrinking from r={radius:.6f}")

        min_radius = self.config.hypersphere.min_radius
        lo, hi = min_radius, radius

        for step in range(shrink_steps):
            mid = (lo + hi) / 2.0
            mid_eval = self._count_eval_captures(center, mid, embeddings,
                                                  point_splits, unassigned_indices)
            if mid_eval <= allowed_eval:
                lo = mid  # can grow a bit more
            else:
                hi = mid  # still too many, shrink further

        # Use the conservative (smaller) end
        final_radius = max(lo, min_radius)
        final_eval = self._count_eval_captures(center, final_radius, embeddings,
                                                point_splits, unassigned_indices)
        logger.info(f"Eval cap: radius {radius:.6f} -> {final_radius:.6f}, "
                    f"eval {eval_count} -> {final_eval}")
        return final_radius

    def _count_eval_captures(
            self,
            center: torch.Tensor,
            radius: float,
            embeddings: torch.Tensor,
            point_splits: List[DatasetSplit],
            unassigned_indices: List[int]) -> int:
        """Count how many unassigned points within radius would become eval."""
        center_2d = center.unsqueeze(0)
        buffer_radius = (self.config.hypersphere.buffer_zones.buffer_radius
                         if self.config.hypersphere.buffer_zones.enabled else 0.0)
        total_radius = radius + buffer_radius

        # Build a temporary anchor for shell assignment
        nested_shells = self.anchor_placer._create_nested_shells(radius)
        temp_anchor = HypersphereAnchor(
            anchor_id=-1, center=center,
            base_radius=radius, buffer_radius=buffer_radius,
            nested_shells=nested_shells
        )

        count = 0
        for ui in unassigned_indices:
            if point_splits[ui] != DatasetSplit.UNASSIGNED:
                continue
            dist = pairwise_distance_cross(
                embeddings[ui].unsqueeze(0), center_2d,
                self.config.distance.metric
            ).item()
            if dist <= total_radius:
                split = temp_anchor.get_point_assignment(dist)
                if split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
                    count += 1
        return count

    def _build_assignment_result(
            self,
            trajectory: EmbeddingTrajectory,
            point_splits: List[DatasetSplit],
            point_anchor_ids: List[Optional[int]],
            point_distances: List[Optional[float]]) -> DatasetAssignmentResult:
        """Build DatasetAssignmentResult from per-point split arrays."""
        N = trajectory.num_points
        train_indices = []
        validation_indices = []
        test_indices = []
        unassigned_indices = []
        assignment_details = []

        for i in range(N):
            point = trajectory.points[i]
            split = point_splits[i]

            point.dataset_split = split
            point.anchor_id = point_anchor_ids[i]
            point.distance_to_anchor = point_distances[i]

            if split == DatasetSplit.TRAIN:
                train_indices.append(i)
            elif split == DatasetSplit.VALIDATION:
                validation_indices.append(i)
            elif split == DatasetSplit.TEST:
                test_indices.append(i)
            else:
                unassigned_indices.append(i)

            assignment_details.append({
                'index': i,
                'original_index': point.original_index,
                'anchor_id': point_anchor_ids[i],
                'distance_to_anchor': point_distances[i],
                'num_containing_anchors': 1 if point_anchor_ids[i] is not None else 0
            })

        statistics = {
            'total_points': N,
            'train_count': len(train_indices),
            'validation_count': len(validation_indices),
            'test_count': len(test_indices),
            'unassigned_count': len(unassigned_indices),
            'train_ratio': len(train_indices) / N if N > 0 else 0,
            'validation_ratio': len(validation_indices) / N if N > 0 else 0,
            'test_ratio': len(test_indices) / N if N > 0 else 0,
            'eval_ratio': (len(validation_indices) + len(test_indices)) / N if N > 0 else 0,
            'coverage_ratio': (N - len(train_indices)) / N if N > 0 else 0,
            'num_anchors_used': self.anchor_registry.num_anchors
        }

        logger.info(f"Final: {statistics['train_count']} train, "
                    f"{statistics['validation_count']} val, "
                    f"{statistics['test_count']} test")

        return DatasetAssignmentResult(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            unassigned_indices=unassigned_indices,
            assignment_details=assignment_details,
            statistics=statistics
        )

    def process_dataframe(self, df: pd.DataFrame,
                         image_path_column: str = 'filename',
                         base_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process images from a DataFrame and add split assignments.

        Args:
            df: DataFrame with image information
            image_path_column: Name of column containing image file paths
            base_path: Optional base path to prepend to image paths

        Returns:
            DataFrame with added 'dataset_split' column
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")

        # Extract image paths
        image_paths = df[image_path_column].tolist()
        if base_path:
            image_paths = [Path(base_path) / path for path in image_paths]

        # Extract metadata and original indices
        metadata = []
        original_indices = []
        for idx, row in df.iterrows():
            meta = row.to_dict()
            metadata.append(meta)
            original_indices.append(idx)

        # Process images with original indices
        assignment_result = self.process_image_stream(
            image_paths, metadata=metadata, original_indices=original_indices
        )

        # Add split assignments to DataFrame
        df_result = df.copy()

        # Create a mapping from ORIGINAL index to split
        split_mapping = {}

        if assignment_result.assignment_details:
            for detail in assignment_result.assignment_details:
                trajectory_idx = detail.get('index')
                original_idx = detail.get('original_index')

                if trajectory_idx in assignment_result.train_indices:
                    split_mapping[original_idx] = DatasetSplit.TRAIN.value
                elif trajectory_idx in assignment_result.validation_indices:
                    split_mapping[original_idx] = DatasetSplit.VALIDATION.value
                elif trajectory_idx in assignment_result.test_indices:
                    split_mapping[original_idx] = DatasetSplit.TEST.value
                elif trajectory_idx in assignment_result.unassigned_indices:
                    split_mapping[original_idx] = DatasetSplit.UNASSIGNED.value
        else:
            for idx in assignment_result.train_indices:
                split_mapping[idx] = DatasetSplit.TRAIN.value
            for idx in assignment_result.validation_indices:
                split_mapping[idx] = DatasetSplit.VALIDATION.value
            for idx in assignment_result.test_indices:
                split_mapping[idx] = DatasetSplit.TEST.value
            for idx in assignment_result.unassigned_indices:
                split_mapping[idx] = DatasetSplit.UNASSIGNED.value

        # Add split column
        df_result['dataset_split'] = df_result.index.map(split_mapping)

        # Add assignment details if available
        if assignment_result.assignment_details:
            details_df = pd.DataFrame(assignment_result.assignment_details)
            df_result = df_result.merge(details_df, left_index=True, right_on='original_index', how='left')

        return df_result

    def _deduplicate_trajectory(self, trajectory: EmbeddingTrajectory) -> EmbeddingTrajectory:
        """Remove duplicate or near-duplicate points from trajectory using GPU ops."""
        if trajectory.num_points <= 1:
            return trajectory

        logger.debug(f"Deduplicating trajectory with {trajectory.num_points} points")

        embeddings = trajectory.embeddings_matrix  # (N, D) GPU tensor
        threshold = self.config.deduplication.similarity_threshold
        method = self.config.deduplication.method

        # Compute pairwise similarity matrix on GPU
        if method == "cosine":
            # Cosine similarity = 1 - cosine_distance
            dist_matrix = pairwise_distance_matrix(embeddings, "cosine")
            similarity_matrix = 1.0 - dist_matrix
        else:
            # For euclidean/manhattan, convert to similarity-like measure
            dist_matrix = pairwise_distance_matrix(embeddings, method)
            similarity_matrix = 1.0 / (1.0 + dist_matrix)

        # Zero out lower triangle and diagonal (only check upper triangle)
        N = similarity_matrix.shape[0]
        mask = torch.triu(torch.ones(N, N, device=self.device, dtype=torch.bool), diagonal=1)
        masked_sims = similarity_matrix * mask.float()

        # Greedy FCFS deduplication on GPU
        # For each kept point, find all points similar to it and mark them for removal
        class_aware = self.config.class_aware
        to_remove = set()
        for i in range(N):
            if i in to_remove:
                continue
            # Find indices where similarity > threshold in this row
            similar_mask = masked_sims[i] > threshold
            similar_indices = similar_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
            if isinstance(similar_indices, int):
                similar_indices = [similar_indices]
            if class_aware:
                label_i = trajectory.points[i].metadata.get('label') if trajectory.points[i].metadata else None
                for j in similar_indices:
                    if j in to_remove:
                        continue
                    label_j = trajectory.points[j].metadata.get('label') if trajectory.points[j].metadata else None
                    if label_i is not None and label_j is not None and label_i != label_j:
                        continue
                    to_remove.add(j)
            else:
                to_remove.update(similar_indices)

        # Build list of kept indices
        kept_indices = [i for i in range(N) if i not in to_remove]

        logger.info(f"Deduplication removed {len(to_remove)} points, "
                    f"{len(kept_indices)} remaining")

        if to_remove:
            return trajectory.subset(kept_indices, trajectory.trajectory_id + "_deduplicated")
        else:
            return trajectory


    def _load_split_mask_from_csv(self, csv_path: str) -> None:
        """
        Build a boolean train mask over self.all_embeddings from the CSV.

        The CSV rows and .pt embedding rows are both appended chronologically
        per batch, so row i in the CSV corresponds to row i in
        self.all_embeddings.

        Sets:
            self._train_mask: (M,) bool tensor (True = train)
            self.stored_train_count, self.stored_eval_count
        """
        if self.all_embeddings is None or self.all_embeddings.shape[0] == 0:
            logger.info("No stored embeddings -- skipping CSV split mask")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Failed to read {csv_path}: {e}")
            return

        num_embeddings = self.all_embeddings.shape[0]
        num_csv_rows = len(df)

        if num_csv_rows != num_embeddings:
            logger.warning(
                f"CSV rows ({num_csv_rows}) != stored embeddings ({num_embeddings}). "
                f"Mask may be misaligned -- using min of the two."
            )

        usable = min(num_csv_rows, num_embeddings)

        # dataset_split column: "train" | "validation" | "test" | "unassigned"
        # train_split column:   "train" | "val"        | "test"
        # Use dataset_split as primary, fall back to train_split
        if 'dataset_split' in df.columns:
            splits = df['dataset_split'].iloc[:usable].fillna('train')
            is_train = splits.str.lower().isin(['train', 'unassigned'])
        elif 'train_split' in df.columns:
            splits = df['train_split'].iloc[:usable].fillna('train')
            is_train = splits.str.lower() == 'train'
        else:
            logger.warning("CSV has neither dataset_split nor train_split column")
            return

        mask = torch.tensor(is_train.values, dtype=torch.bool, device=self.device)

        # If embeddings are longer than CSV, treat the extras as train
        if num_embeddings > usable:
            extra = torch.ones(num_embeddings - usable, dtype=torch.bool,
                               device=self.device)
            mask = torch.cat([mask, extra])

        self._train_mask = mask
        self.stored_train_count = int(mask.sum().item())
        self.stored_eval_count = int((~mask).sum().item())

        # Per-class eval counts for class-aware anchor placement
        self.stored_class_eval_counts: Dict[str, int] = {}
        self.stored_class_total_counts: Dict[str, int] = {}
        if self.config.class_aware and 'label' in df.columns:
            labels = df['label'].iloc[:usable]
            for label_val in labels.dropna().unique():
                label_lower = str(label_val).lower()
                label_mask = labels.str.lower() == label_lower
                class_total = int(label_mask.sum())
                class_eval = int((label_mask & ~is_train).sum())
                self.stored_class_eval_counts[label_lower] = class_eval
                self.stored_class_total_counts[label_lower] = class_total
            logger.info(f"Per-class eval counts from CSV: {self.stored_class_eval_counts}")

        logger.info(f"Split mask from CSV: {self.stored_train_count} train, "
                    f"{self.stored_eval_count} eval out of {num_embeddings} stored")

    def _save_persistent_state(self) -> None:
        """Save anchor registry and pipeline state for persistence."""
        if not self.config.streaming.anchor_persistence_path:
            return

        state_path = Path(self.config.streaming.anchor_persistence_path)

        # Use deterministic hash for config validation
        config_str = str(sorted(self.config.to_dict().items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Clone anchor centers so pickle doesn't serialize entire embedding storages
        # (centers are views into large tensors during processing)
        for anchor in self.anchor_registry.anchors.values():
            anchor.center = anchor.center.clone()

        state = {
            'anchor_registry': self.anchor_registry,
            'performance_stats': self.performance_stats,
            'config_hash': config_hash
        }

        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            logger.debug(f"Saved persistent state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")

    def _load_persistent_state(self) -> None:
        """Load anchor registry and pipeline state from persistence."""
        if not self.config.streaming.anchor_persistence_path:
            return

        state_path = Path(self.config.streaming.anchor_persistence_path)
        if not state_path.exists():
            logger.debug("No persistent state file found")
            return

        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)

            # Basic validation - be lenient with config changes
            config_str = str(sorted(self.config.to_dict().items()))
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            saved_hash = state.get('config_hash')

            if saved_hash != config_hash:
                logger.warning(f"Configuration changed (hash: {saved_hash} -> {config_hash})")
                logger.warning("Loading persistent state anyway for continual learning")

            self.anchor_registry = state['anchor_registry']
            self.performance_stats.update(state.get('performance_stats', {}))

            # Move anchor registry to current device (may have been saved from different device)
            self.anchor_registry.move_to_device(self.device)

            logger.info(f"LOADED PERSISTENT STATE: {self.anchor_registry.num_anchors} existing anchors")
            logger.info(f"Previous performance stats: {self.performance_stats['total_processed']} images processed")

        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")

    def _deduplicate_against_stored_embeddings(
            self, new_embeddings: torch.Tensor,
            new_points: List[EmbeddingPoint]
    ) -> Tuple[torch.Tensor, List[EmbeddingPoint]]:
        """
        Deduplicate new embeddings against all previously stored embeddings (GPU batch op).

        Args:
            new_embeddings: (N, D) GPU tensor of new embeddings
            new_points: List of N EmbeddingPoint metadata objects

        Returns:
            Tuple of (kept_embeddings_tensor, kept_points_list)
        """
        if not self.embeddings_storage_dir or not new_points:
            return new_embeddings, new_points

        logger.info(f"CROSS-BATCH DEDUPLICATION: Checking {len(new_points)} new embeddings against stored ones")

        # Use pre-loaded stored embeddings (GPU tensor)
        if self.all_embeddings is None or self.all_embeddings.shape[0] == 0:
            logger.info("No previously stored embeddings found")
            return new_embeddings, new_points

        stored_embeddings = self.all_embeddings  # (M, D) GPU tensor
        logger.info(f"Comparing against {stored_embeddings.shape[0]} previously stored embeddings")

        threshold = self.config.deduplication.similarity_threshold

        # Compute cross-set similarity on GPU: (N, M)
        if self.config.deduplication.method == "cosine":
            # Cosine similarity = 1 - cosine_distance
            dist_matrix = pairwise_distance_cross(
                new_embeddings, stored_embeddings, "cosine"
            )
            similarity_matrix = 1.0 - dist_matrix
        else:
            dist_matrix = pairwise_distance_cross(
                new_embeddings, stored_embeddings, self.config.deduplication.method
            )
            similarity_matrix = 1.0 / (1.0 + dist_matrix)

        # Class-aware: zero out cross-class similarities before taking max
        if (self.config.class_aware
                and self.all_labels is not None
                and any(l is not None for l in self.all_labels)):
            new_labels = [p.metadata.get('label') if p.metadata else None for p in new_points]
            # Encode labels as ints for vectorized comparison
            # None -> 0 (matches everything), actual labels -> 1, 2, ...
            unique_labels = set(l for l in self.all_labels if l is not None)
            unique_labels.update(l for l in new_labels if l is not None)
            label_to_int = {l: idx + 1 for idx, l in enumerate(sorted(unique_labels))}
            new_encoded = torch.tensor([label_to_int.get(l, 0) for l in new_labels],
                                       device=self.device)
            stored_encoded = torch.tensor([label_to_int.get(l, 0) for l in self.all_labels],
                                          device=self.device)
            new_col = new_encoded.unsqueeze(1)       # (N, 1)
            stored_row = stored_encoded.unsqueeze(0)  # (1, M)
            match_mask = (new_col == 0) | (stored_row == 0) | (new_col == stored_row)
            similarity_matrix = similarity_matrix * match_mask.float()

        # Max similarity for each new embedding across all stored embeddings
        max_similarities = similarity_matrix.max(dim=1).values  # (N,)
        duplicate_mask = max_similarities > threshold  # (N,) bool

        # Keep only non-duplicate embeddings
        keep_mask = ~duplicate_mask
        kept_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)

        # Handle edge cases with squeeze
        if kept_indices.dim() == 0:
            kept_indices = kept_indices.unsqueeze(0)

        removed_count = duplicate_mask.sum().item()

        # Log removed duplicates
        duplicate_indices = duplicate_mask.nonzero(as_tuple=False).squeeze(-1)
        if duplicate_indices.dim() == 0:
            duplicate_indices = duplicate_indices.unsqueeze(0)
        for idx in duplicate_indices.cpu().tolist():
            if isinstance(idx, int) and idx < len(new_points):
                filename = Path(new_points[idx].file_path).name if new_points[idx].file_path else f"Point_{idx}"
                logger.debug(f"DUPLICATE: {filename} (similarity: {max_similarities[idx].item():.4f} > {threshold})")

        logger.info(f"CROSS-BATCH DEDUPLICATION: Removed {removed_count}/{len(new_points)} duplicates")

        if kept_indices.shape[0] == 0:
            logger.info("All embeddings are duplicates")
            return torch.empty(0, new_embeddings.shape[1], device=self.device), []

        kept_embeddings = new_embeddings[kept_indices]
        kept_points = [new_points[i] for i in kept_indices.cpu().tolist()]

        logger.info(f"Keeping {len(kept_points)} unique embeddings")
        return kept_embeddings, kept_points

    def _load_all_stored_embeddings(self) -> Tuple[Optional[torch.Tensor], Optional[List[Optional[str]]]]:
        """Load all previously stored embeddings and labels from .pt files.

        Returns:
            Tuple of (embeddings GPU tensor, list of label strings per point).
            Either both are None (no files) or both are populated.
        """
        if not self.embeddings_storage_dir or not self.embeddings_storage_dir.exists():
            return None, None

        all_embeddings = []
        all_labels: List[Optional[str]] = []
        embedding_files = sorted(self.embeddings_storage_dir.glob("*.pt"))

        if not embedding_files:
            return None, None

        for embedding_file in embedding_files:
            try:
                data = torch.load(embedding_file, map_location='cpu')
                if isinstance(data, dict) and 'embeddings' in data:
                    embeddings = data['embeddings']
                    if isinstance(embeddings, np.ndarray):
                        embeddings = torch.from_numpy(embeddings).float()
                    elif isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.float()

                    # Extract labels (backward compat: old files lack labels)
                    batch_labels = None
                    if 'metadata' in data and isinstance(data['metadata'], dict):
                        batch_labels = data['metadata'].get('labels', None)
                    if batch_labels is not None:
                        all_labels.extend(batch_labels)
                    else:
                        all_labels.extend([None] * embeddings.shape[0])
                elif isinstance(data, torch.Tensor):
                    embeddings = data.float()
                    all_labels.extend([None] * embeddings.shape[0])
                else:
                    embeddings = torch.tensor(data, dtype=torch.float32)
                    all_labels.extend([None] * embeddings.shape[0])

                all_embeddings.append(embeddings)
                logger.debug(f"Loaded {embeddings.shape[0]} embeddings from {embedding_file.name}")

            except Exception as e:
                logger.warning(f"Failed to load embeddings from {embedding_file}: {e}")

        if all_embeddings:
            stacked = torch.cat(all_embeddings, dim=0)
            return stacked.to(self.device), all_labels
        else:
            return None, None

    def _store_new_embeddings(self, trajectory: EmbeddingTrajectory, trajectory_id: str) -> None:
        """Store new embeddings to .pt file for future cross-batch deduplication."""
        if not trajectory.points:
            return

        # Create filename with timestamp for uniqueness
        timestamp = int(time.time())
        filename = f"embeddings_batch_{timestamp}_{trajectory_id}.pt"
        file_path = self.embeddings_storage_dir / filename

        # Get embeddings as CPU tensor for storage (avoid saving GPU tensors)
        embeddings_cpu = trajectory.embeddings_matrix.detach().cpu()

        metadata = {
            'filenames': [point.file_path for point in trajectory.points],
            'original_indices': [point.original_index for point in trajectory.points],
            'labels': [point.metadata.get('label') if point.metadata else None
                       for point in trajectory.points],
            'timestamp': timestamp,
            'trajectory_id': trajectory_id,
            'num_points': trajectory.num_points
        }

        try:
            torch.save({
                'embeddings': embeddings_cpu,
                'metadata': metadata
            }, file_path)

            logger.info(f"Stored {embeddings_cpu.shape[0]} embeddings to {filename}")

            # Update in-memory GPU cache
            embeddings_gpu = trajectory.embeddings_matrix  # already on self.device
            if self.all_embeddings is not None and self.all_embeddings.shape[0] > 0:
                self.all_embeddings = torch.cat([self.all_embeddings, embeddings_gpu], dim=0)
            else:
                self.all_embeddings = embeddings_gpu.clone()

            # Update in-memory label cache
            new_labels = [point.metadata.get('label') if point.metadata else None
                          for point in trajectory.points]
            if self.all_labels is not None:
                self.all_labels.extend(new_labels)
            else:
                self.all_labels = new_labels

        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")

    def _log_performance_stats(self) -> None:
        """Log current performance statistics with full timing breakdown."""
        stats = self.performance_stats
        if stats['total_processed'] > 0 and stats['total_time'] > 0:
            total = stats['total_time']
            throughput = stats['total_processed'] / total

            logger.info(
                f"=== TIMING BREAKDOWN ({stats['total_processed']} images, "
                f"{total:.2f}s total, {throughput:.1f} img/s) ==="
            )
            logger.info(
                f"  Embedding:    {stats['embedding_time']:.2f}s "
                f"({100*stats['embedding_time']/total:.1f}%)"
            )
            if stats['image_load_time'] > 0 or stats['preprocess_time'] > 0 or stats['inference_time'] > 0:
                logger.info(
                    f"    - Image I/O:    {stats['image_load_time']:.2f}s "
                    f"({100*stats['image_load_time']/total:.1f}%)"
                )
                logger.info(
                    f"    - Preprocess:   {stats['preprocess_time']:.2f}s "
                    f"({100*stats['preprocess_time']/total:.1f}%)"
                )
                logger.info(
                    f"    - Inference:    {stats['inference_time']:.2f}s "
                    f"({100*stats['inference_time']/total:.1f}%)"
                )
            logger.info(
                f"  Dedup:        {stats['dedup_time']:.2f}s "
                f"({100*stats['dedup_time']/total:.1f}%)"
            )
            logger.info(
                f"  Density:      {stats['density_time']:.2f}s "
                f"({100*stats['density_time']/total:.1f}%)"
            )
            logger.info(
                f"  Anchors:      {stats['anchor_time']:.2f}s "
                f"({100*stats['anchor_time']/total:.1f}%)"
            )
            logger.info(
                f"  Assignment:   {stats['assignment_time']:.2f}s "
                f"({100*stats['assignment_time']/total:.1f}%)"
            )
            logger.info(
                f"  Storage:      {stats['storage_time']:.2f}s "
                f"({100*stats['storage_time']/total:.1f}%)"
            )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the pipeline."""
        stats = {
            'performance': self.performance_stats.copy(),
            'anchors': self.anchor_registry.get_coverage_stats(),
            'foundation_models': self.foundation_processor.get_model_info(),
            'device': str(self.device)
        }

        if self.anchor_registry.num_anchors > 0:
            all_anchors = list(self.anchor_registry.anchors.values())
            radii = [anchor.base_radius for anchor in all_anchors]
            total_radii = [anchor.total_radius for anchor in all_anchors]

            stats['hyperspheres'] = {
                'num_hyperspheres': len(all_anchors),
                'mean_base_radius': np.mean(radii),
                'std_base_radius': np.std(radii),
                'mean_total_radius': np.mean(total_radii),
                'buffer_zones_enabled': self.config.hypersphere.buffer_zones.enabled
            }

        return stats

    def reset_pipeline(self) -> None:
        """Reset the pipeline state (clear anchors and stats)."""
        logger.info("Resetting pipeline state")
        self.anchor_registry = AnchorRegistry(device=self.device)
        self.all_embeddings = None
        self.performance_stats = {
            'total_processed': 0,
            'current_batch_processed': 0,
            'total_time': 0,
            'embedding_time': 0,
            'anchor_time': 0,
            'assignment_time': 0
        }

        # Remove persistent state file if it exists
        if self.config.streaming.anchor_persistence_path:
            state_path = Path(self.config.streaming.anchor_persistence_path)
            if state_path.exists():
                state_path.unlink()
                logger.debug("Removed persistent state file")
