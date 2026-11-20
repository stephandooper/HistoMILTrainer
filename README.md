# HistoMILTrainer

A library for training Multi-Instance Learning (MIL) architectures from [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) on histology datasets. HistoMILTrainer provides a unified interface to train and evaluate various state-of-the-art MIL models for whole slide image (WSI) analysis. It also supports transfer learning from previously trained MIL checkpoints.

## Overview

HistoMILTrainer offers a streamlined framework to train MIL architectures on histology data. It supports multiple architectures from MIL-Lab, including:

- **ABMIL** - Attention-based Multiple Instance Learning
- **CLAM** - Clustering-constrained Attention Multiple instance learning
- **DSMIL** - Dual-stream Multiple Instance Learning
- **DFTD** - Deep Feature-based Top-Down attention
- **ILRA** - Instance-Level Representation Aggregation
- **RRT** - Residual Regression Transformer
- **Transformer** - Transformer-based MIL
- **TransMIL** - Transductive Multiple Instance Learning
- **WIKG** - Weighted Instance Knowledge Graph

## Features

- **Unified Training Interface**: Train any supported MIL architecture with consistent parameters
- **Flexible Data Loading**: Support for variable number of patches per slide
- **Cross-Validation**: Built-in support for k-fold cross-validation
- **Transfer Learning**: Initialize MIL models from existing checkpoints and control which model components are updated during training
- **Feature Extraction Integration**: Works seamlessly with pre-extracted patch features (e.g., from TRIDENT)
- **Class Weighting and Sampling**: Independent options for handling imbalanced datasets
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **Case-Level Splitting**: Prevents data leakage by splitting at the case level
- **Inference Pipeline**: Run predictions on new slides with trained models
- **Attention Heatmap Visualization**: Generate interpretable attention heatmaps overlaid on WSIs

## Installation

A dedicated GPU-capable Docker image is available; see [container build and run instructions](docker/README.md). It uses Python 3.10 and CUDA 12.8 and runs CPU validation during the build.

HistoMILTrainer requires **Python 3.10**. Choose one of the following installation methods:

### Option 1: Using Conda (Recommended)

```bash
# Clone HistoMILTrainer
git clone https://github.com/digenoma-lab/HistoMILTrainer
cd HistoMILTrainer

# Create conda environment with all dependencies
conda env create -f environment.yml
conda activate histomil

# Install HistoMILTrainer in editable mode
pip install -e .
```

After installation, the CLI commands `histomil-splits`, `histomil-grid`, `histomil-train`, `histomil-predict`, and `histomil-heatmap` will be available in your PATH.

The `environment.yml` file includes:
- Python 3.10
- MIL-Lab (from GitHub)
- smooth-topk (required for CLAM, from GitHub)
- All required dependencies (seaborn, matplotlib, pytest, etc.)

### Option 2: Using pip/Poetry

```bash
# Clone HistoMILTrainer
git clone https://github.com/digenoma-lab/HistoMILTrainer
cd HistoMILTrainer

# Install with pip (MIL-Lab will be installed automatically as a dependency)
pip install -e .
```

After installation, the CLI commands `histomil-splits`, `histomil-grid`, `histomil-train`, `histomil-predict`, and `histomil-heatmap` will be available in your PATH.

**Note**: When installing with pip, dependencies are automatically installed from the `pyproject.toml` configuration. The package will install:
- MIL-Lab (from GitHub: `https://github.com/GabrielCabas/MIL-Lab.git`)
- smooth-topk (from GitHub: `https://github.com/oval-group/smooth-topk.git`, required for CLAM)
- seaborn
- matplotlib
- All other dependencies from MIL-Lab (torch, numpy, pandas, scikit-learn, tqdm, h5py, etc.)

### CLAM Support

The `smooth-topk` dependency (required for CLAM architecture) is automatically installed with HistoMILTrainer. No additional installation steps are required.

**Note**: MIL-Lab is not available on PyPI and is installed directly from GitHub. The installation process handles this automatically through the dependency configuration.

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
features/
  ├── slide1.h5
  ├── slide2.h5
  └── ...
```

Each H5 file should contain:
- `features`: Array of shape `(num_patches, feature_dim)`
- Optionally: `coords`: Array of patch coordinates

For Mascaret, extract patch features with a current [TRIDENT](https://github.com/mahmoodlab/TRIDENT) installation using `--patch_encoder mascaret`. Mascaret access requires approval on its [Hugging Face model page](https://huggingface.co/wearewaiv/mascaret); authenticate with an approved account before extraction. Update TRIDENT using its installation instructions if your version does not include Mascaret. Point HistoMILTrainer at the resulting precomputed H5 files and use `--feature_extractor mascaret`; their per-patch CLS features must have dimension 1536.

### 2. Create Dataset CSV

Create a CSV file with columns:
- `case_id`: Unique identifier for each case (patient)
- `slide_id`: Unique identifier for each slide
- `target`: Target label for classification (or specify custom column name with `--target`)

**Important**: Splits are created at the case level to prevent data leakage. Multiple slides from the same case will always be in the same split.

### 3. Generate Splits

```bash
histomil-splits \
  --csv_path ./data/dataset.csv \
  --splits_dir ./splits/ \
  --output_name my_task \
  --folds 10 \
  --test_frac 0.2 \
  --target target
```

**Note**: The `histomil-splits` command is available after installing the package. The `--target` argument specifies the column name for labels (default: "target"). Splits are stratified at the case level to ensure no data leakage between train/val/test sets.

### 4. Train Models with Grid Search

Use `histomil-grid` to perform hyperparameter optimization and train models across all folds:

```bash
histomil-grid \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4 \
  --grid_params configs/abmil.json
```

**Note**: The `histomil-grid` command performs grid search across parameter combinations, trains models for all folds, selects the best parameters, and evaluates on the test set. The `--csv_path` should point to the `dataset.csv` file generated in the splits directory (e.g., `./splits/my_task/dataset.csv`).

### 5. Train with Fixed Parameters or Transfer Learning

Use `histomil-train` to train and evaluate one fixed parameter configuration across the requested folds. The command accepts model hyperparameters through `--params_path` and supports random initialization or checkpoint-based transfer learning.

The `--params_path` JSON must contain scalar values for one model configuration, such as a `best_params_<feature_extractor>.<mil>.json` file generated by a previous grid search.

Available training modes:

- `scratch`: initializes the complete model randomly and trains all parameters using the fixed configuration.
- `head_only`: loads a compatible checkpoint, freezes the rest of the model, and trains only the classification heads.
- `partial`: loads a compatible checkpoint and applies the architecture-specific configuration from `histomil/configs/req_grid/<mil>.json`.

```bash
histomil-train \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil_transfer/ \
  --params_path ./previous_results/best_params_uni_v2.abmil.json \
  --transfer_mode partial \
  --pretrained_checkpoint ./previous_results/0_best_model.pt
```

`head_only` and `partial` require `--pretrained_checkpoint`. In `scratch`, omit the checkpoint; the parameters JSON is reused, but all model weights are initialized randomly.

### 6. Run Multiple Folds (SLURM)

Use the provided shell scripts for running multiple folds:

```bash
sbatch run_abmil.sh
sbatch run_clam.sh
# ... etc
```

## Usage

### Available CLI Commands

After installation, the following commands are available:

- `histomil-splits`: Generate train/validation/test splits
- `histomil-grid`: Perform grid search with hyperparameter optimization
- `histomil-train`: Train one fixed parameter configuration with `scratch`, `head_only`, or `partial`
- `histomil-predict`: Run inference on new slides with a trained model
- `histomil-heatmap`: Generate attention heatmap visualizations

### Split Generation (`histomil-splits`)

```bash
histomil-splits \
  --csv_path <path>                 # Path to dataset CSV (required)
  --output_name <name>              # Output directory name (required)
  --folds <int>                     # Number of folds (default: 10)
  --splits_dir <path>               # Output directory (default: ./splits)
  --test_frac <float>               # Test set fraction (default: 0.2)
  --target <column_name>            # Target column name (default: target)
```

### Grid Search (`histomil-grid`)

```bash
histomil-grid \
  --features_path <path>            # Path to H5 feature files directory (required)
  --splits_dir <path>               # Directory containing split files (required)
  --csv_path <path>                 # Path to dataset CSV (required)
  --mil <model_name>                # MIL architecture: abmil, clam, dsmil, dftd, etc. (default: abmil)
  --feature_extractor <name>        # Feature extractor: uni_v2, etc. (default: uni_v2)
  --results_dir <path>              # Output directory for results (default: ./temp_dir/)
  --epochs <int>                    # Number of training epochs (default: 10)
  --learning_rate <float>           # Learning rate (default: 4e-4)
  --folds <int>                     # Number of cross-validation folds (default: 10)
  --use_class_weights               # Enable class weighting (default)
  --no-use_class_weights            # Disable class weighting
  --use_weighted_sampler            # Enable weighted training sampling
  --no-use_weighted_sampler         # Disable weighted sampling (default)
  --grid_params <path>              # Grid JSON with parameter lists (default: configs/<mil>.json)
```

`histomil-grid` performs hyperparameter optimization. The grid JSON defines candidate values as lists. The command trains every parameter combination, selects the best configuration using validation AUC, and evaluates the selected fold models on the test set.

### Fixed-Parameter Training (`histomil-train`)

```bash
histomil-train \
  --features_path <path>            # Path to H5 feature files directory (required)
  --splits_dir <path>               # Directory containing split files (required)
  --csv_path <path>                 # Path to dataset CSV (required)
  --params_path <path>              # JSON containing one fixed model configuration (required)
  --mil <model_name>                # MIL architecture (default: abmil)
  --feature_extractor <name>        # Feature extractor (default: uni_v2)
  --results_dir <path>              # Output directory for results (default: ./temp_dir/)
  --epochs <int>                    # Number of training epochs (default: 10)
  --learning_rate <float>           # Learning rate (default: 4e-4)
  --folds <int>                     # Number of folds (default: 1)
  --use_class_weights               # Enable class weighting (default)
  --no-use_class_weights            # Disable class weighting
  --use_weighted_sampler            # Enable weighted training sampling
  --no-use_weighted_sampler         # Disable weighted sampling (default)
  --transfer_mode <mode>            # scratch, head_only, or partial (default: scratch)
  --pretrained_checkpoint <path>    # Required by head_only and partial
```

`histomil-train` trains and evaluates the single configuration supplied through `--params_path`. In `scratch` mode, the model is initialized randomly and no checkpoint is provided. In `head_only` and `partial`, a compatible checkpoint is required.

Both training commands default to weighted sampling off, independently of class weighting. Use `--use_weighted_sampler --no-use_class_weights` for sampling alone. The sampler draws training slides with replacement, using inverse class frequencies; each epoch draws as many samples as there are training slides, so classes are balanced in expectation rather than necessarily in every epoch. Validation and test data are not sampled this way.

### Partial Configuration (`histomil/configs/req_grid/`)

The `histomil/configs/req_grid/` directory contains one JSON file for each supported MIL architecture. When `histomil-train --transfer_mode partial` is selected, HistoMILTrainer automatically loads the file matching `--mil`.

Each JSON uses:

- `full_finetune=0` to freeze the model first and then apply the layer groups defined in `groups`.
- `full_finetune=1` to train the complete checkpoint-loaded model and ignore `groups`.
- `trainable=1` or `trainable=0` to mark configured module prefixes as trainable or frozen.
- `layers` to list PyTorch module names or prefixes.
- `strict=1` to raise an error when a configured layer is not found, or `strict=0` to skip missing layers.

The `_comment` field in each file documents the meaning and use of the available fields. Each file defines the standard trainability configuration for one MIL architecture and can be modified manually for a particular experiment. Full fine-tuning uses `--transfer_mode partial` with `full_finetune=1`; `partial_full` is not a separate CLI mode.

`--params_path` and `req_grid` serve different purposes: `--params_path` defines the fixed model hyperparameters, whereas `req_grid` defines which model components are updated in `partial` mode.

### Prediction (`histomil-predict`)

Run inference on new slides using a trained model:

```bash
histomil-predict \
  --features_folder <path>          # Path to H5 feature files directory (required)
  --weights_path <path>             # Path to trained model weights (.pt file) (required)
  --csv_path <path>                 # Path to CSV with slide_id column (required)
  --params_path <path>              # Path to JSON with model parameters (required)
  --mil <model_name>                # MIL architecture: abmil, clam, etc. (default: abmil)
  --feature_extractor <name>        # Feature extractor: virchow2, uni_v2, etc. (default: virchow2)
  --results_dir <path>              # Output directory for predictions (default: ./)
  --log_level <level>               # Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

**Output:**
- `predictions.csv`: Contains `slide_id`, `prob` (probability), and `pred` (binary prediction) columns
- `attention_scores/`: Directory containing H5 files with attention scores for each slide

### Heatmap Visualization (`histomil-heatmap`)

Generate attention heatmap overlays on whole slide images:

```bash
histomil-heatmap \
  --slide_id <filename>             # Slide filename (required)
  --slide_folder <path>             # Directory containing original WSI files (required)
  --features_folder <path>          # Directory with H5 feature files containing coordinates (required)
  --attn_scores_folder <path>       # Directory with attention scores H5 files (required)
  --results_dir <path>              # Output directory for heatmaps (default: ./)
  --log_level <level>               # Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

**Output:**
- `heatmap_{slide_name}.png`: Attention heatmap visualization overlaid on the WSI thumbnail

**Note:** This module requires [TRIDENT](https://github.com/mahmoodlab/TRIDENT) for WSI handling and visualization.

### Example: Inference and Visualization Workflow

After training, you can run predictions on new data and generate attention heatmaps:

```bash
# Step 1: Run predictions on new slides
histomil-predict \
  --features_folder ./features/ \
  --weights_path ./results/abmil/0-checkpoint.pt \
  --csv_path ./data/new_slides.csv \
  --params_path ./results/abmil/best_params_virchow2.abmil.json \
  --mil abmil \
  --feature_extractor virchow2 \
  --results_dir ./predictions/

# Step 2: Generate heatmap for a specific slide
histomil-heatmap \
  --slide_id slide_001.svs \
  --slide_folder ./slides/ \
  --features_folder ./features/ \
  --attn_scores_folder ./predictions/attention_scores/ \
  --results_dir ./heatmaps/
```

### Supported Models

Use the `--mil` argument to specify the architecture:

- `abmil` - Attention-based MIL
- `clam` - CLAM architecture (requires batch_size=1)
- `dsmil` - Dual-stream MIL
- `dftd` - Deep Feature Top-Down
- `ilra` - Instance-Level Representation Aggregation
- `rrt` - Residual Regression Transformer
- `transformer` - Transformer-based MIL
- `transmil` - Transductive MIL
- `wikg` - Weighted Instance Knowledge Graph

**Note**: CLAM automatically sets batch_size to 1 during training.

## Project Structure

```
HistoMILTrainer/
├── histomil/           # Main library package
│   ├── models.py       # Model import functions
│   ├── datasets.py     # Dataset classes
│   ├── train.py        # Training and evaluation functions
│   ├── splits.py       # Split management
│   ├── grid_search.py  # Hyperparameter grid search
│   ├── fixed_training.py # Fixed-parameter and transfer training
│   ├── predict.py      # Inference/prediction functions
│   ├── heatmap.py      # Attention heatmap visualization
│   ├── transfer.py     # Transfer learning configuration
│   ├── cli.py          # Command-line interface
│   ├── utils.py        # Utility functions
│   └── configs/
│       └── req_grid/   # Standard partial trainability configuration by MIL architecture
├── tests/              # Test suite
│   └── test_import.py  # Import tests
├── environment.yml     # Conda environment configuration
└── pyproject.toml      # Package metadata and dependencies
```

## Output

### Training Output

`histomil-grid` produces grid-search metrics, selected best-parameter JSON files, fold checkpoints, and test predictions. Its `training_results_<feature_extractor>.<mil>.csv` records the best-epoch training/validation metrics for every fold and parameter configuration, including the parameter columns.

`histomil-train` produces:
- `{fold}_best_model.pt`: Best model checkpoint for each fold
- `{fold}-<parameters>-checkpoint.pt`: Internal early-stopping checkpoint
- `training_results_<feature_extractor>.<mil>.csv`: Training and validation metrics
- `test_results_<feature_extractor>.<mil>.csv`: Test metrics
- `predictions_<feature_extractor>.<mil>_<fold>.csv`: Test probabilities and labels
- `best_params_<feature_extractor>.<mil>.json`: Copy of the fixed configuration used for training

Both training workflows use TorchMetrics 1.9.0 to match the mimyc definitions: multiclass accuracy (micro), multiclass ROC AUC (macro), macro accuracy (`balanced_acc`), and binary calibration error (`ece`, 15 bins, L1 norm). Accuracy, AUC and balanced accuracy receive both class softmax probabilities; ECE receives the positive-class probability. Metrics are accumulated over the complete phase, with separate state for each epoch and phase. AUC is reported as `NaN` when only one target class is present. Class weights affect the loss, while metrics are unweighted; training metrics reflect the slides drawn by the sampler.

Existing `train_acc`, `val_acc`, `test_acc` and corresponding `*_auc` fields retain their names. The result CSVs also include `train_balanced_acc`, `val_balanced_acc`, `test_balanced_acc` and corresponding `*_ece` fields. Training and validation metrics are logged each epoch and the selected checkpoint's epoch is saved in the results. Testing runs once per selected fold checkpoint. Existing test F1 and prediction outputs are retained.

### Importing the supplied DLBCL folds

Run `python scripts/convert_data_splits.py data_splits.zip` to generate `data_splits_converted.zip`. Extract it and pass `cv/dataset.csv` as `--csv_path` and `cv/` as `--splits_dir`, using `--folds 10`. Each fold's validation slides are also marked as its test slides, allowing the existing final test pass to run. Those automatic test statistics evaluate the validation cohort again and are not independent test estimates. Original training assignments are preserved.

The eight CSVs in `test_sets/` remain separate for later prediction using saved checkpoints. The joint cohorts overlap with their component cohorts. `histomil-predict` exports predictions and attention scores; calculating external-test metrics still requires joining predictions to labels by `slide_id`.

### Split Generation Output

The `histomil-splits` command generates:
- `dataset.csv`: Processed dataset with case_id, slide_id, and label columns
- `splits_{fold}_bool.csv`: Boolean splits for each fold (train/val/test columns)
- `splits_{fold}_descriptor.csv`: Summary statistics for each split

### Prediction Output

The `histomil-predict` command generates:
- `predictions.csv`: Contains slide_id, probability scores, and binary predictions
- `attention_scores/`: Directory with H5 files containing attention weights per patch for each slide

### Heatmap Output

The `histomil-heatmap` command generates:
- `heatmap_{slide_name}.png`: Attention heatmap visualization overlaid on the WSI thumbnail
- Top 20 patches with highest attention scores are highlighted

### Grid Search

You can perform grid search with hyperparameter optimization using:

```bash
histomil-grid \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4 \
  --grid_params configs/abmil.json
```

The `histomil-grid` command performs:
- Grid search across all parameter combinations
- Cross-validation training for each combination
- Selection of best parameters based on validation AUC
- Testing of best models on test set
- Output of results, predictions, and best parameters


**Note**: MIL-Lab is automatically installed as a dependency when you install HistoMILTrainer. The `import_model` function relies on `src.builder.create_model` from MIL-Lab, which should be available after installation.

## Citation

If you use HistoMILTrainer in your research, please cite the original MIL-Lab paper and the specific architecture papers you use.

## License

See LICENSE file for details.

### Contact

Author: **Gabriel Cabas**  
For questions or suggestions, please open an *issue* or *pull request* in this repository.
