import json
from argparse import Namespace

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from scripts.evaluate_checkpoint import evaluate


@pytest.fixture
def args(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"slide_id": ["a", "b"], "label": [0, 1], "site": ["x", "y"]}).to_csv(csv, index=False)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"n_classes": 2, "dropout": 0.25}))
    return Namespace(csv_path=str(csv), features_folder=str(features), weights_path=str(tmp_path / "model.pt"),
                     feature_extractor="mascaret", mil="abmil", params_path=str(params),
                     results_dir=str(tmp_path / "results"), fold=3)


def test_real_checkpoint_matches_training_test(args):
    from histomil.datasets import H5Dataset, variable_patches_collate_fn
    from histomil.models import import_model
    from histomil.train import test as run_test, device
    from torch.utils.data import DataLoader

    torch.manual_seed(2)
    for slide in ("a", "b"):
        with h5py.File(f"{args.features_folder}/{slide}.h5", "w") as f:
            f["features"] = torch.randn(12, 1536).numpy()
    model = import_model("abmil", "mascaret", n_classes=2, dropout=0.25).to(device)
    torch.save(model.state_dict(), args.weights_path)
    frame = pd.read_csv(args.csv_path)
    loader = DataLoader(H5Dataset(args.features_folder, frame), batch_size=1, shuffle=False,
                        collate_fn=variable_patches_collate_fn)
    expected, probabilities, labels = run_test(model, loader, model_name="abmil")
    actual, predictions = evaluate(args)
    assert predictions["prob"].tolist() == pytest.approx(probabilities)
    assert predictions["label"].tolist() == list(labels)
    assert predictions["site"].tolist() == ["x", "y"]
    assert actual == pytest.approx(expected, nan_ok=True)
    saved = pd.read_csv(f"{args.results_dir}/metrics.csv")
    assert saved.loc[0, "fold"] == 3
    assert saved.loc[0, "n_slides"] == 2
    assert saved.loc[0, "cohort"] == "cohort"
    assert len(pd.read_csv(f"{args.results_dir}/predictions.csv")) == 2


@pytest.mark.parametrize("frame", [
    {"slide_id": [], "label": []},
    {"slide_id": ["a"]},
    {"slide_id": ["", "b"], "label": [0, 1]},
    {"slide_id": ["a", "a"], "label": [0, 1]},
    {"slide_id": ["a"], "label": [2]},
])
def test_invalid_csv_rejected(args, frame):
    pd.DataFrame(frame).to_csv(args.csv_path, index=False)
    with pytest.raises(ValueError):
        evaluate(args)


@pytest.mark.parametrize("params", [{}, {"n_classes": [2]}, {"n_classes": {"value": 2}}])
def test_invalid_params_rejected(args, params):
    with open(args.params_path, "w") as file:
        json.dump(params, file)
    with pytest.raises(ValueError):
        evaluate(args)
