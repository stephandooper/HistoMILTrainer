# Run the 10-fold matrix inside Slurm / Pyxis

The runner trains four encoders × four MIL architectures × ten folds: **160 models**.
It evaluates each fold's selected checkpoint on each of the eight external CSVs:
**1,280 evaluations**. Jobs run sequentially on one allocated GPU.

Checkpoint selection uses the existing trainer: **lowest validation loss**, with
early stopping (currently patience 2, earliest stopping after epoch index 2). It
does not pick a single winning fold. Validation and training run each epoch; the
converted splits also cause one final evaluation on the validation cohort per fold.
External evaluations remain separate and do not affect checkpoint selection.

## 1. Container image on the cluster

Your local Docker tag `histomil:local` is not automatically available to Enroot on
the cluster. Transfer/import it through your normal cluster workflow, or publish
it to a registry you control. For example, on the Docker build machine:

```bash
docker tag histomil:local registry.example.org/your-user/histomil:latest
docker push registry.example.org/your-user/histomil:latest
```

On the cluster, import it once into a shared location:

```bash
enroot import -o /shared/images/histomil.sqsh \
  'docker://registry.example.org#your-user/histomil:latest'
```

Use your actual registry, image tag and shared paths. Registry authentication follows
your site's Enroot configuration. See the official [Enroot import documentation](https://github.com/NVIDIA/enroot/blob/main/doc/cmd/import.md).

The previously built image can run the new scripts with the repository bind-mounted
as below; rebuilding is only needed if you want the scripts baked into the image.

## 2. Enter a standard interactive session

Run this on the cluster login node, replacing the example values. Create writable
host directories for results and cache first. Mount an up-to-date checkout containing
the runner, parameter JSONs and evaluation helper:

```bash
PARTITION=gpu
IMAGE=/shared/images/histomil.sqsh
REPO=/shared/projects/HistoMILTrainer
FEATURES=/shared/data/features
SPLITS=/shared/data/data_splits_converted
RESULTS=/shared/results/histomil
CACHE=/shared/cache/histomil
mkdir -p "$RESULTS" "$CACHE"

srun --partition="$PARTITION" --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
  --container-image="$IMAGE" \
  --container-mounts="$REPO:/workspace/repo:ro,$FEATURES:/data/features:ro,$SPLITS:/data/splits:ro,$RESULTS:/results,$CACHE:/cache" \
  --container-workdir=/workspace/repo \
  --no-container-entrypoint --pty bash
```

Adjust partition, account/QOS, resources and wall time to your site. This is one
interactive allocation; the runner does not submit nested jobs or start distributed
training. Pyxis handles the container and Slurm handles GPU allocation. The mount
and working-directory flags follow the [Pyxis usage documentation](https://github.com/NVIDIA/pyxis#usage).

## 3. Feature and split paths

Mount the **newly extracted** converted ZIP at `/data/splits`, containing `cv/`
and `test_sets/`. The runner expects all eight named test CSVs from that archive.

Default feature paths:

| Encoder | Directory | Features per patch |
|---|---|---:|
| Virchow2 (`virchow2`) | `FEATURES_ROOT/virchow2_0.5mpp_trident` | 2560 |
| UNI2 (`uni_v2`) | `FEATURES_ROOT/uni2_0.5mpp_trident` | 1536 |
| CONCH 1.5 (`conch_v15`) | `FEATURES_ROOT/conch15_0.5mpp_trident` | 768 |
| Mascaret (`mascaret`) | `FEATURES_ROOT/mascaret_0.5mpp_trident` | 1536 |

These directories are TRIDENT job folders. The runner reads H5 embeddings from
`FEATURES_ROOT/MAPPED_FOLDER/20x_224px_0px_overlap/features_ENCODER/`, where
`ENCODER` is the canonical key (for example `features_virchow2` or `features_uni_v2`).
This follows TRIDENT's `job_dir/coords_dir/features_<encoder>/` output layout.
Set `TRIDENT_COORDS_DIR` to override the default `20x_224px_0px_overlap` patch folder.
Each resolved features directory can hold all CV and external H5 files together,
named `<slide_id without extension>.h5`. Alternatively, use `cv/` under that directory
for the CV cohort and `COHORT/` under it for individual test cohorts
(for example `hovon_test/`). When a subset directory exists, it takes precedence
over the flat encoder directory. Joint cohorts need access to all their component
slides; use a flat encoder folder or symlink views if features live in multiple
locations. Paths containing spaces are supported; Pyxis mount specifications must
not contain commas in host paths.

## 4. Run the matrix inside the container

With the single mount
`--container-mounts=/data/pathology/projects:/data/pathology/projects`, configure
the actual cluster paths once per shell session:

```bash
cd /data/pathology/projects/mimyc/HistoMILTrainer
export FEATURES_ROOT=/data/pathology/projects/mimyc/embeddings/trident
export SPLITS_ROOT=/data/pathology/projects/mimyc/HistoMILTrainer/data_splits_converted
export RESULTS_ROOT=/data/pathology/projects/mimyc/HistoMILTrainer/outputs
export EPOCHS=50
```

The encoder folder mapping above applies to both training and external evaluation.
Output folders continue using encoder keys (`virchow2`, `uni_v2`, `conch_v15`,
`mascaret`), not the feature folder names. Until Mascaret features are available,
select the existing three with
`ENCODERS="virchow2 uni_v2 conch_v15" bash scripts/run_cv_matrix.sh`.

First inspect the commands without needing data or a GPU:

```bash
bash scripts/run_cv_matrix.sh --dry-run
```

Then train and evaluate everything:

```bash
EPOCHS=50 LEARNING_RATE=4e-4 bash scripts/run_cv_matrix.sh
```

`EPOCHS` is a maximum, not a promise to train that many epochs: validation early
stopping still applies. If omitted, the runner uses the repository's default of
10 epochs and learning rate `4e-4`. Fixed parameters live in
`scripts/cv_params/{abmil,dsmil,transmil,clam}.json`; edit these or set `PARAMS_ROOT`
to a writable directory of your own JSON files before starting. These are explicit
starting presets, not parameters selected by a hyperparameter search.

Balanced sampling is enabled by default with loss class weights disabled. Set
`BALANCE=class_weights` for class-weighted loss only or `BALANCE=none` for neither.
CLAM uses the repository's automatic batch size of 1 and the SVM instance-loss
setting in its JSON; the other architectures use batch size 16.

Useful subsets and stages:

```bash
# One encoder/model pair, still all ten folds and eight external cohorts.
ENCODERS=mascaret MILS=clam bash scripts/run_cv_matrix.sh

# Train only.
STAGE=train bash scripts/run_cv_matrix.sh

# Evaluate existing fold checkpoints, without training again.
STAGE=test bash scripts/run_cv_matrix.sh

# Use different paths visible INSIDE the container.
FEATURES_ROOT=/my/features SPLITS_ROOT=/my/splits RESULTS_ROOT=/my/results \
  bash scripts/run_cv_matrix.sh
```

Completed training pairs are skipped only when the training settings, parameter
JSON, split files and expected checkpoints match. Partial training is not resumed
within a fold: set `RERUN_PARTIAL=1` to rerun the interrupted pair's ten folds.
Changed training settings require a new `RESULTS_ROOT`. External evaluations rerun
when requested; they do not silently reuse old metric files. A failure stops the
script and leaves its log available. Use `STAGE=test` to repeat evaluation alone.

## Outputs

```text
/results/ENCODER/MIL/
  train.log
  evaluate.log
  cv/
    0_best_model.pt ... 9_best_model.pt
    best_params_ENCODER.MIL.json
    training_results_ENCODER.MIL.csv
  external/fold_0/COHORT/
    predictions.csv
    metrics.csv
/results/external_metrics.csv
```

Each external metrics row includes accuracy, ROC AUC, balanced accuracy, ECE,
existing F1, fold, cohort, encoder, architecture and slide count. The combined CSV
covers only the encoder/model pairs selected in that invocation. Predictions
preserve slide IDs, labels and extra cohort metadata. The evaluator uses the same
`histomil.train.test` implementation as training, without exporting patch attention.
Its exported class prediction follows argmax (a probability tie predicts class 0);
the existing F1 implementation retains its historical `>= 0.5` threshold.

`joint_1_test` and `joint_2_test` overlap their constituent cohorts. Stanford has
only one positive slide. Keep cohort results separate when interpreting metrics.
