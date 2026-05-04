# BubbleFence

Semantic train/val/test splitting for image datasets using density-adaptive hypersphere regions in foundation model embedding spaces. BubbleFence embeds images with CLIP, places "bubble" anchors in the embedding space using quasi-random sampling, and assigns each image based on whether it falls inside a bubble (eval) or outside (train). Nested shells within each bubble distinguish validation from test. The system supports streaming operation: anchors and embeddings persist across ingestion rounds so new data integrates without reprocessing old batches.

## How It Works

1. **Embed** images into a shared CLIP embedding space (GPU-accelerated).
2. **Deduplicate** across all prior batches using cosine similarity.
3. **Place anchors** via Sobol quasi-random sequences, snapped to real data points weighted by Local Intrinsic Dimensionality (LID).
4. **Compute adaptive radii** per anchor using LID, so bubbles are smaller in dense regions and larger in sparse ones.
5. **Assign splits**: points inside a bubble get eval, points outside get train. Each bubble's eval region is divided into two concentric shells: an inner shell (closer to the anchor center) and an outer shell (the ring between the inner boundary and the bubble edge). One shell is assigned to validation, the other to test. The `shell_configuration` parameter controls which split goes where: `"inner_val"` puts validation in the inner shell and test in the outer, `"inner_test"` does the reverse, and `"random"` (the default) flips a coin per anchor so that neither split is systematically closer to or farther from anchor centers.
6. **Closed-loop quota**: a feedback loop adds or shrinks anchors until the target eval ratio is met within tolerance.
7. **Persist** anchor state and embeddings to disk for incremental ingestion.

## Installation

CPU:
```bash
pip install -r bubblefence/requirements.txt
```

GPU (ROCm):
```bash
pip install -r bubblefence/requirements.txt --index-url https://download.pytorch.org/whl/rocm7.1
```

Note: Other GPU backends (e.g. CUDA) may also work but have not been tested.

## Quick Start

### Option 1: CLI (quickest)

```bash
# Run BubbleFence on a single folder
python run_bubblefence.py --folder path/to/images/ --output_dir output/

# Run on multiple folders from a batch config (see config/batch.yaml)
python run_bubblefence.py --batch config/batch.yaml

# Visualize existing results
python visualize_bubblefence.py --all-trajectories --radius-circles \
  --anchor-thumbnails 7 --method TSNE --save plot.png --output_dir output/
```

See [Visualization](#visualization) below for all CLI flags and plot types.

### Option 2: `run_folder` (Python API)

```python
from bubblefence import load_config
from bubblefence.pipeline_runner import run_folder

config = load_config("config/bubblefence_config.yaml")
config.streaming.anchor_persistence_path = "output/anchors_state.pkl"

result_df = run_folder("path/to/images/", "output/", config)
print(result_df["dataset_split"].value_counts())
```

Call `run_folder` again with a new folder to add data incrementally -- existing anchors are reused, new anchors are placed as needed to maintain the eval ratio.

### Option 3: `run_batch` (multi-folder YAML)

Define folders and visualizations in a YAML config (see `config/batch.yaml` for a template):

```yaml
# batch_config.yaml
output_dir: "output/"
bf_config: "config/bubblefence_config.yaml"
folders:
  - "data/drive_001"
  - "data/drive_002"
  - "data/drive_003"
visualizations:
  - type: standalone
    save: tsne_traj.png
    show_heatmap: false
    show_trajectories: true
    show_radius_circles: true
    show_anchor_thumbnails: true
    num_anchor_thumbnails: 7
    reduction_method: TSNE
    point_alpha: 0.1
  - type: stats
  - type: summary
    save: summary_2panel.png
    reduction_method: TSNE
```

Then run:

```python
from bubblefence.pipeline_runner import run_batch

run_batch("batch_config.yaml")

# Re-generate visualizations only (skip processing):
run_batch("batch_config.yaml", skip_bf=True)
```

### Option 4: Direct pipeline API

For full programmatic control:

```python
from bubblefence import BubbleFenceConfig, BubbleFencePipeline

config = BubbleFenceConfig.from_yaml("config/bubblefence_config.yaml")
pipeline = BubbleFencePipeline(config)

# From image paths
result = pipeline.process_image_stream(["img1.jpg", "img2.jpg", ...])
print(result.train_indices, result.validation_indices, result.test_indices)

# From a DataFrame
import pandas as pd
df = pd.read_csv("annotations.csv")
result_df = pipeline.process_dataframe(df, image_path_column="filename", base_path="images/")
```

## Configuration

All parameters live in a single YAML file. The most commonly tuned knobs:

```yaml
# Foundation model
foundation_models:
  primary_model: "openai/clip-vit-base-patch32"

# Split ratios
dataset_splits:
  train_ratio: 0.8
  eval_ratio: 0.2
  min_eval_per_batch: 0.05   # minimum eval fraction forced per batch
  eval_tolerance: 0.03       # acceptable overshoot

# Anchor placement
anchor_placement:
  method: "QMC"              # QMC, random, or kmeans
  qmc_sequence: "sobol"      # sobol or halton
  snap_strategy: "lid_weighted"  # nearest or lid_weighted
  min_anchor_distance: 0.1

# Bubble sizing
hypersphere:
  radius_computation: "adaptive"
  adaptive_method: "LID"
  base_radius_mode: "auto"   # auto or fixed
  min_radius: 0.01
  max_radius: 0.5

# Val/test split within each bubble
nested_shells:
  enabled: true
  validation_ratio: 0.5      # fraction of eval going to val
  shell_configuration: "random"   # "random", "inner_val", or "inner_test"

# Cross-batch deduplication
deduplication:
  enabled: true
  similarity_threshold: 0.9999

# Streaming persistence
streaming:
  enabled: true
  persistent_anchors: true
  anchor_persistence_path: "anchors_state.pkl"

# Device
embedding:
  device: "auto"             # auto, cuda, or cpu
  batch_size: 32
```

See `config/bubblefence_config.yaml` for the full reference with all defaults.

## Visualization

### CLI

```bash
# Print dataset statistics (saves to stats.txt)
python visualize_bubblefence.py --stats --output_dir output/

# Standalone plot with trajectories, circles, and thumbnails
python visualize_bubblefence.py \
  --all-trajectories --radius-circles --anchor-thumbnails 7 \
  --method TSNE --point-alpha 0.1 --no-heatmaps \
  --save tsne_plot.png --output_dir output/

# 2-panel summary (splits + heatmap)
python visualize_bubblefence.py --summary --method TSNE --save summary.png --output_dir output/

# 4-panel detailed view
python visualize_bubblefence.py --detailed --method TSNE --save detailed.png --output_dir output/

# Interactive 3D plot (opens in browser as .html)
python visualize_bubblefence.py --3d --all-trajectories --radius-circles --output_dir output/

# Clean 3D plot (no gridlines, axis labels, or ticks)
python visualize_bubblefence.py --3d --clean --output_dir output/

# Re-run batch visualizations without reprocessing
python run_bubblefence.py --batch config/batch.yaml --skip-bf
```

Layer toggles for standalone mode:
- `--all-trajectories` -- smooth spline curves per ingestion run
- `--radius-circles` -- draw bubble radius circles around anchors
- `--anchor-thumbnails [N]` -- show representative image thumbnails (N = number of spread-out anchors to auto-select)
- `--anchor-thumb-pixels N` -- thumbnail size in pixels (default 80)
- `--no-points` -- hide scatter points
- `--no-heatmaps` -- show convex hulls instead of KDE heatmap
- `--point-alpha FLOAT` -- scatter point opacity (default 0.3)
- `--smoothing FLOAT` -- trajectory spline smoothness (default 0.1, works well up to 100+)
- `--3d` -- interactive 3D plotly visualization (saves as `.html`)
- `--clean` -- strip gridlines, axis labels, and ticks from 3D plot

### Programmatic

```python
from bubblefence.pipeline_runner import run_visualizations

vis_configs = [
    {"type": "stats"},
    {"type": "standalone", "save": "traj_plot.png",
     "show_trajectories": True, "show_radius_circles": True,
     "show_anchor_thumbnails": True, "num_anchor_thumbnails": 5,
     "reduction_method": "TSNE", "point_alpha": 0.1},
    {"type": "summary", "save": "summary.png",
     "reduction_method": "TSNE", "show_all_trajectories": True},
    {"type": "detailed", "save": "detailed.png",
     "show_heatmap": True, "reduction_method": "TSNE"},
]

run_visualizations("output/", vis_configs)
```

Vis types:
- `"stats"` -- prints split/anchor statistics to stdout and saves to `stats.txt`
- `"standalone"` -- single-panel scatter with configurable overlays
- `"summary"` -- 2-panel (dataset splits + anchor heatmap)
- `"detailed"` -- 4-panel (splits + heatmap + histogram + stats table)

## Output Files

After running, `output_dir/` contains:

```
output/
  full_dataset.csv          # Cumulative dataset with split assignments
  anchors_state.pkl         # Serialized anchor registry (persistent across runs)
  stats.txt                 # Dataset statistics (generated by stats vis type)
  embeddings/
    embeddings_batch_*.pt   # Per-batch embedding tensors + metadata
  bubblefence.log           # Processing log
```

**`full_dataset.csv` columns:**

| Column | Description |
|--------|-------------|
| `run_id` | Ingestion run identifier (folder name + timestamp) |
| `folder_path` | Source folder path |
| `filename` | Image filename |
| `dataset_split` | `train`, `validation`, `test`, or `unassigned` |
| `train_split` | Legacy column: `train`, `val`, or `test` |
| `anchor_id` | Assigned anchor ID (NaN for train points) |

## Architecture

| Module | Purpose |
|--------|---------|
| `bubble_fence.py` | Main pipeline: closed-loop anchor placement, streaming, assignment |
| `foundation_models.py` | CLIP/DINO/SigLIP embedding with prefetch I/O and multi-encoder consensus |
| `anchor_placement.py` | QMC (Sobol/Halton) anchor candidate generation, LID-weighted snapping |
| `density_analysis.py` | Local Intrinsic Dimensionality, adaptive radius computation |
| `data_structures.py` | `EmbeddingTrajectory`, `HypersphereAnchor`, `AnchorRegistry`, `DatasetAssignmentResult` |
| `config.py` | YAML config loading and validation |
| `device_utils.py` | GPU detection, tensor utilities, pairwise distance computation |
| `image_preprocessing.py` | Shared image processor with model-specific normalization |
| `visualization.py` | All plotting: standalone, detailed, trajectory, statistics |
| `pipeline_runner.py` | High-level API: `run_folder`, `run_batch`, `run_visualizations` |

## Notebooks

- **`BubbleFence_ZOD_Demo.ipynb`** -- step-by-step walkthrough of each pipeline stage (embedding, dedup, density analysis, anchor placement, assignment) with intermediate outputs. Good for understanding the internals.
- **`BubbleFence_ZOD_Blog.ipynb`** -- concise end-to-end demo containing steps from the blog post, using `run_batch` with YAML configs. Runs two rounds of incremental ingestion and generates trajectory plots, stats, and summary visualizations. Good for a quick overview.
