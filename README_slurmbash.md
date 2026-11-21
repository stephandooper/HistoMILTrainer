# Run the CV matrix inside your Slurm container

The Bash runner uses these feature folders under `FEATURES_ROOT`:

| Encoder key | Feature folder |
|---|---|
| `virchow2` | `virchow2_0.5mpp_trident` |
| `uni_v2` | `uni2_0.5mpp_trident` |
| `conch_v15` | `conch15_0.5mpp_trident` |
| `mascaret` | `mascaret_0.5mpp_trident` |

These are TRIDENT job folders. H5 embeddings are read from
`FEATURES_ROOT/<feature folder>/20x_224px_0px_overlap/features_<encoder key>/`.
For example, Virchow2 uses
`virchow2_0.5mpp_trident/20x_224px_0px_overlap/features_virchow2/`.
Set `TRIDENT_COORDS_DIR` if your patch configuration folder differs from
`20x_224px_0px_overlap`.

Inside your container, set these once:

```bash
cd /data/pathology/projects/mimyc/HistoMILTrainer

export FEATURES_ROOT=/data/pathology/projects/mimyc/embeddings/trident
export SPLITS_ROOT=/data/pathology/projects/mimyc/HistoMILTrainer/data_splits_converted
export RESULTS_ROOT=/data/pathology/projects/mimyc/HistoMILTrainer/outputs
export EPOCHS=50
```

Then use:

```bash
# Print commands without training.
bash scripts/run_cv_matrix.sh --dry-run

# Run the three currently available encoders with all four classifiers.
ENCODERS="virchow2 uni_v2 conch_v15" bash scripts/run_cv_matrix.sh

# Run everything once Mascaret features exist.
bash scripts/run_cv_matrix.sh

# One encoder/classifier combination—all ten folds and external cohorts.
ENCODERS=virchow2 MILS=clam bash scripts/run_cv_matrix.sh

# Several selected combinations.
ENCODERS="uni_v2 conch_v15" MILS="abmil transmil" bash scripts/run_cv_matrix.sh

# Train only.
STAGE=train ENCODERS="virchow2 uni_v2 conch_v15" bash scripts/run_cv_matrix.sh

# Evaluate saved checkpoints without retraining.
STAGE=test ENCODERS="virchow2 uni_v2 conch_v15" bash scripts/run_cv_matrix.sh

# Use class-weighted loss instead of balanced sampling.
BALANCE=class_weights ENCODERS=virchow2 MILS=abmil bash scripts/run_cv_matrix.sh
```

Any variation accepts `--dry-run`. Outputs remain under
`outputs/<encoder>/<classifier>/`, with combined external metrics in
`outputs/external_metrics.csv`.

For the Slurm/Pyxis launch command and full workflow details, see
[the cluster workflow](docs/slurm_cv.md).
