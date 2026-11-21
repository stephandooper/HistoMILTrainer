"""Evaluate a saved MIL checkpoint on a labeled external cohort."""
import argparse
import json
from pathlib import Path
import sys

# Permit running this file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(args):
    """Write slide predictions and the same metrics as the training test loop."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from histomil.datasets import H5Dataset, variable_patches_collate_fn
    from histomil.models import import_model
    from histomil.train import test

    df = pd.read_csv(args.csv_path, dtype={"slide_id": "string"})
    if not {"slide_id", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain slide_id and label columns")
    if df.empty:
        raise ValueError("CSV must contain at least one slide")
    ids = df["slide_id"].str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("slide_id values must be nonempty and unique")
    df["slide_id"] = ids
    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("labels must be binary (0 or 1)")
    df["label"] = labels.astype(int)
    with open(args.params_path, encoding="utf-8") as file:
        params = json.load(file)
    if not isinstance(params, dict) or not params or any(isinstance(v, (dict, list)) or v is None for v in params.values()):
        raise ValueError("params JSON must be a nonempty object of scalar values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = import_model(args.mil, args.feature_extractor, **params).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True), strict=True)
    loader = DataLoader(H5Dataset(args.features_folder, df), batch_size=1, shuffle=False,
                        collate_fn=variable_patches_collate_fn)
    metrics, probabilities, labels = test(model, loader, model_name=args.mil)
    predictions = df.copy()
    predictions["label"] = labels.astype(int)
    predictions["prob"] = probabilities
    predictions["pred"] = (probabilities > 0.5).astype(int)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    metadata = {"n_slides": len(df), "encoder": args.feature_extractor, "mil": args.mil,
                "cohort": Path(args.csv_path).stem, "checkpoint_path": str(args.weights_path)}
    if getattr(args, "fold", None) is not None:
        metadata["fold"] = args.fold
    pd.DataFrame([{**metrics, **metadata}]).to_csv(results_dir / "metrics.csv", index=False)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--features_folder", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--feature_extractor", required=True)
    parser.add_argument("--mil", required=True, choices=["abmil", "dsmil", "transmil", "clam"])
    parser.add_argument("--params_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--fold", type=int)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
