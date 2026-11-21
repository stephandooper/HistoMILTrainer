# HistoMILTrainer container

Build from the repository root with Docker Desktop's Linux engine or Docker on
Linux. The build requires network access and installs the Git dependencies in
`pyproject.toml`; rebuilding may resolve newer upstream commits. The image records
the installed versions in `/opt/venv/requirements-resolved.txt`.

```bash
docker build --platform linux/amd64 -t histomil:local .
```

The dedicated image contains Python 3.10, Torch 2.10.0 / torchvision 0.25.0 with
CUDA 12.8, TorchMetrics 1.9.0, MIL-Lab, smooth-topk, TRIDENT and HistoMILTrainer.
OpenSlide 4.0.0 and libvips 8.16.0 are built from source, including the same
partial-bottom-tile OpenSlide backport used by mimyc. TRIDENT's dependencies may
also install the `openslide-bin` wheel, which the Python OpenSlide binding prefers;
libvips uses the source-built library. Both paths are checked with a synthetic TIFF.
No model weights are downloaded during build validation. Training on precomputed
Mascaret features does not require access to Mascaret weights.

The build runs focused tests, dependency checks, all five CLI help commands and a
CPU smoke test of the model, metrics and native libraries. No GPU is needed to build.

## Run training

For an interactive Slurm/Pyxis session and the four-encoder/four-model CV matrix,
see [the cluster workflow](../docs/slurm_cv.md).

Re-extract `data_splits_converted.zip` into a local directory first. Supply a scalar
model-parameter JSON as described in the repository README. Mount inputs read-only
and keep output and caches writable. This example uses Bash syntax; replace host
paths with absolute paths available to Docker:

```bash
docker run --rm --gpus all \
  --mount type=bind,source=/absolute/path/features,target=/data/features,readonly \
  --mount type=bind,source=/absolute/path/data_splits_converted,target=/data/splits,readonly \
  --mount type=bind,source=/absolute/path/params.json,target=/data/params.json,readonly \
  --mount type=bind,source=/absolute/path/results,target=/results \
  --mount type=volume,source=histomil-cache,target=/cache \
  histomil:local histomil-train \
    --features_path /data/features \
    --csv_path /data/splits/cv/dataset.csv \
    --splits_dir /data/splits/cv \
    --params_path /data/params.json \
    --results_dir /results \
    --feature_extractor mascaret --mil abmil --folds 10 \
    --use_weighted_sampler --no-use_class_weights
```

The mounted results directory must already exist. The converted splits deliberately
reuse validation slides for the final test pass; use the separate external CSVs for
independent prediction afterward. Choose the feature extractor matching the H5 files.

For PowerShell, use one line or replace Bash continuation backslashes with
PowerShell backticks, and use absolute Windows host paths such as
`source=C:\data\features` in each `--mount` argument.

Run `histomil-grid`, `histomil-predict`, `histomil-splits` or `histomil-heatmap` in
place of `histomil-train`; each command accepts `--help`. Starting the image without
a command opens Bash in `/workspace`; add `-it` for an interactive terminal.

## GPU validation

The host needs an NVIDIA driver compatible with CUDA 12.8 and Docker GPU support
(NVIDIA Container Toolkit on Linux, or NVIDIA GPU support in Docker Desktop/WSL2).
Check device access separately after building:

```bash
docker run --rm --gpus all histomil:local python /opt/container/validate.py --gpu
```

This requires a GPU and checks a CUDA allocation. Omitting `--gpu` performs CPU
validation. Dataset files, ZIP archives, checkpoints and credentials are excluded
from the build context by an allowlist. Supply any needed credentials at runtime.
