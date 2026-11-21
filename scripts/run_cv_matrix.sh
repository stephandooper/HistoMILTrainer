#!/usr/bin/env bash
# Run inside an allocated Pyxis container with one GPU, not via nested sbatch.
# Feature layout: FEATURES_ROOT/MAPPED_ENCODER/TRIDENT_COORDS_DIR/features_ENCODER.
# Optional cv or COHORT subdirectories override that folder for each dataset.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FEATURES_ROOT=${FEATURES_ROOT:-/data/features}
TRIDENT_COORDS_DIR=${TRIDENT_COORDS_DIR:-20x_224px_0px_overlap}
SPLITS_ROOT=${SPLITS_ROOT:-/data/splits}
RESULTS_ROOT=${RESULTS_ROOT:-/results}
PARAMS_ROOT=${PARAMS_ROOT:-${SCRIPT_DIR}/cv_params}
EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-4e-4}
ENCODERS=${ENCODERS:-"virchow2 uni_v2 conch_v15 mascaret"}
MILS=${MILS:-"abmil dsmil transmil clam"}
STAGE=${STAGE:-all}
BALANCE=${BALANCE:-weighted_sampler}
RERUN_PARTIAL=${RERUN_PARTIAL:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if (($#)); then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
case "$STAGE" in all|train|test) ;; *) echo "Invalid STAGE: $STAGE" >&2; exit 2;; esac
case "$BALANCE" in
  weighted_sampler) balance_args=(--use_weighted_sampler --no-use_class_weights);;
  class_weights) balance_args=(--no-use_weighted_sampler --use_class_weights);;
  none) balance_args=(--no-use_weighted_sampler --no-use_class_weights);;
  *) echo "Invalid BALANCE: $BALANCE" >&2; exit 2;;
esac
read -r -a encoders <<< "$ENCODERS"
read -r -a mils <<< "$MILS"
((${#encoders[@]} && ${#mils[@]})) || { echo 'ENCODERS and MILS must not be empty' >&2; exit 2; }
cohorts=(stanford_test slide_thickness_test s05_test ood_test joint_2_test joint_1_test internal_test hovon_test)
for encoder in "${encoders[@]}"; do
  case "$encoder" in virchow2|uni_v2|conch_v15|mascaret) ;; *) echo "Invalid encoder: $encoder" >&2; exit 2;; esac
done
for mil in "${mils[@]}"; do
  case "$mil" in abmil|dsmil|transmil|clam) ;; *) echo "Invalid MIL: $mil" >&2; exit 2;; esac
done
feature_dir() {
  local mapped subset="$2" root
  case "$1" in
    virchow2) mapped=virchow2_0.5mpp_trident;;
    uni_v2) mapped=uni2_0.5mpp_trident;;
    conch_v15) mapped=conch15_0.5mpp_trident;;
    mascaret) mapped=mascaret_0.5mpp_trident;;
  esac
  root="${FEATURES_ROOT}/${mapped}/${TRIDENT_COORDS_DIR}/features_$1"
  if [[ -d "$root/$subset" ]]; then printf '%s\n' "$root/$subset"; else printf '%s\n' "$root"; fi
}
run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if (( ! DRY_RUN )); then "$@"; fi
}
if (( ! DRY_RUN )); then
  python -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required"'
  for encoder in "${encoders[@]}"; do
    if [[ $STAGE != test ]]; then
      for mil in "${mils[@]}"; do [[ -f "$PARAMS_ROOT/$mil.json" ]] || { echo "Missing parameters: $PARAMS_ROOT/$mil.json" >&2; exit 1; }; done
      for fold in {0..9}; do for suffix in bool descriptor; do [[ -f "$SPLITS_ROOT/cv/splits_${fold}_${suffix}.csv" ]] || { echo "Missing split: $fold $suffix" >&2; exit 1; }; done; done
      python - "$SPLITS_ROOT/cv/dataset.csv" "$(feature_dir "$encoder" cv)" <<'PY'
import csv, pathlib, sys
csv_path, root = map(pathlib.Path, sys.argv[1:])
with csv_path.open(newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    if not {'slide_id', 'label'} <= set(reader.fieldnames or []): raise ValueError(f'{csv_path}: expected slide_id and label')
    rows = list(reader)
    if not rows: raise ValueError(f'{csv_path}: empty dataset')
    for row in rows:
        slide, label = row['slide_id'].strip(), row['label'].strip()
        if not slide or label not in ('0', '1', '0.0', '1.0'): raise ValueError(f'{csv_path}: invalid slide_id or label')
        path = root / (pathlib.Path(slide).stem + '.h5')
        if not path.is_file(): raise FileNotFoundError(path)
PY
    fi
    if [[ $STAGE != train ]]; then
      for cohort in "${cohorts[@]}"; do
        python - "$SPLITS_ROOT/test_sets/$cohort.csv" "$(feature_dir "$encoder" "$cohort")" <<'PY'
import csv, pathlib, sys
csv_path, root = map(pathlib.Path, sys.argv[1:])
with csv_path.open(newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    if not {'slide_id', 'label'} <= set(reader.fieldnames or []): raise ValueError(f'{csv_path}: expected slide_id and label')
    rows = list(reader)
    if not rows: raise ValueError(f'{csv_path}: empty dataset')
    ids = set()
    for row in rows:
        slide, label = row['slide_id'].strip(), row['label'].strip()
        if not slide or slide in ids or label not in ('0', '1', '0.0', '1.0'): raise ValueError(f'{csv_path}: invalid or duplicate slide_id or invalid label')
        ids.add(slide)
        path = root / (pathlib.Path(slide).stem + '.h5')
        if not path.is_file(): raise FileNotFoundError(path)
PY
      done
    fi
  done
fi
for encoder in "${encoders[@]}"; do
  for mil in "${mils[@]}"; do
    pair="$RESULTS_ROOT/$encoder/$mil"
    cv="$pair/cv"
    params="$PARAMS_ROOT/$mil.json"
    if (( ! DRY_RUN )); then mkdir -p "$pair"; fi
    if [[ $STAGE != test ]]; then
      train=(histomil-train --folds 10 --features_path "$(feature_dir "$encoder" cv)" --splits_dir "$SPLITS_ROOT/cv" --csv_path "$SPLITS_ROOT/cv/dataset.csv" --results_dir "$cv" --feature_extractor "$encoder" --mil "$mil" --params_path "$params" --epochs "$EPOCHS" --learning_rate "$LEARNING_RATE" "${balance_args[@]}")
      if (( DRY_RUN )); then run "${train[@]}"; else
        signature=$( { printf '%s\n' "$encoder" "$mil" "$EPOCHS" "$LEARNING_RATE" "$BALANCE" "$(feature_dir "$encoder" cv)"; sha256sum "$params" "$SPLITS_ROOT/cv/dataset.csv" "$SPLITS_ROOT"/cv/splits_{0..9}_{bool,descriptor}.csv; } | sha256sum | cut -d ' ' -f1 )
        marker="$cv/.matrix_complete"
        if [[ -f $marker ]]; then
          [[ $(cat "$marker") == "$signature" ]] || { echo "Training fingerprint changed for $pair; choose a fresh RESULTS_ROOT" >&2; exit 1; }
          for fold in {0..9}; do [[ -f $cv/${fold}_best_model.pt ]] || { echo "Missing checkpoint in completed run: $cv/${fold}_best_model.pt" >&2; exit 1; }; done
          [[ -f $cv/best_params_${encoder}.${mil}.json ]] || { echo "Missing saved parameters in $cv" >&2; exit 1; }
          echo "Skipping completed training: $encoder $mil"
        else
          if compgen -G "$cv/*_best_model.pt" > /dev/null && [[ $RERUN_PARTIAL != 1 ]]; then echo "Partial training found in $cv; set RERUN_PARTIAL=1 to rerun this pair's ten folds, or choose a fresh RESULTS_ROOT" >&2; exit 1; fi
          run "${train[@]}" 2>&1 | tee -a "$pair/train.log"
          for fold in {0..9}; do [[ -f $cv/${fold}_best_model.pt ]] || { echo "Missing checkpoint: $cv/${fold}_best_model.pt" >&2; exit 1; }; done
          [[ -f $cv/best_params_${encoder}.${mil}.json ]] || { echo "Missing saved parameters in $cv" >&2; exit 1; }
          printf '%s\n' "$signature" > "$marker"
        fi
      fi
    fi
    if [[ $STAGE != train ]]; then
      for fold in {0..9}; do
        for cohort in "${cohorts[@]}"; do
          cmd=(python "$SCRIPT_DIR/evaluate_checkpoint.py" --csv_path "$SPLITS_ROOT/test_sets/$cohort.csv" --features_folder "$(feature_dir "$encoder" "$cohort")" --weights_path "$cv/${fold}_best_model.pt" --feature_extractor "$encoder" --mil "$mil" --params_path "$cv/best_params_${encoder}.${mil}.json" --results_dir "$pair/external/fold_${fold}/$cohort" --fold "$fold")
          if (( DRY_RUN )); then run "${cmd[@]}"; else run "${cmd[@]}" 2>&1 | tee -a "$pair/evaluate.log"; fi
        done
      done
    fi
  done
done
if [[ $STAGE != train ]] && (( ! DRY_RUN )); then
  # Aggregate only the encoder/model pairs selected for this invocation.
  python - "$RESULTS_ROOT" "${#encoders[@]}" "${encoders[@]}" "${mils[@]}" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
encoders, mils = sys.argv[3:3 + count], sys.argv[3 + count:]
paths = [root / encoder / mil / 'external' / f'fold_{fold}' / cohort / 'metrics.csv'
         for encoder in encoders for mil in mils for fold in range(10)
         for cohort in ('stanford_test', 'slide_thickness_test', 's05_test', 'ood_test',
                        'joint_2_test', 'joint_1_test', 'internal_test', 'hovon_test')]
rows, fields = [], None
for path in paths:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if fields is None: fields = reader.fieldnames
        if reader.fieldnames != fields: raise ValueError(f'Inconsistent metrics columns: {path}')
        rows.extend(reader)
with (root / 'external_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
fi
