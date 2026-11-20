"""Balanced slide sampling and training integration."""
import importlib
import math
import sys

import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from histomil.fixed_training import FixedParameterTrainer
from histomil.grid_search import GridSearch


@pytest.mark.parametrize("trainer_type", [FixedParameterTrainer, GridSearch])
def test_sampler_only_affects_training_and_uses_filtered_order(trainer_type):
    runner = trainer_type.__new__(trainer_type)
    runner.features_path = "."
    runner.batch_size = 2
    runner.use_weighted_sampler = True
    runner.use_class_weights = False
    import logging
    runner.logger = logging.getLogger(__name__)
    frame = pd.DataFrame({"slide_id": list("abcde"), "label": [0, 1, 0, 1, 0],
                          "train": [True, False, True, True, True],
                          "val": [False, True, False, False, False],
                          "test": [False, True, False, False, False]}, index=[9, 4, 7, 2, 6])
    loader = runner._create_loader(frame, "train", shuffle=True)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.weights.tolist() == pytest.approx([1/3, 1/3, 1, 1/3])
    assert loader.sampler.replacement and loader.sampler.num_samples == 4
    for split in ("val", "test"):
        assert isinstance(runner._create_loader(frame, split, shuffle=False).sampler, SequentialSampler)
    torch.manual_seed(17)
    first = list(loader.sampler)
    torch.manual_seed(17)
    assert list(loader.sampler) == first
    runner.use_weighted_sampler = False
    runner.use_class_weights = True
    assert isinstance(runner._create_loader(frame, "train", shuffle=True).sampler, RandomSampler)


@pytest.mark.parametrize("flags,expected", [([], False), (["--use_weighted_sampler"], True),
                                         (["--use_weighted_sampler", "--no-use_weighted_sampler"], False)])
@pytest.mark.parametrize("entry,module,name,extra", [
    ("grid_search", "histomil.grid_search", "GridSearch", ["--grid_params", "config.json"]),
    ("train_model", "histomil.fixed_training", "FixedParameterTrainer", ["--params_path", "config.json"]),
])
def test_cli_forwards_sampler_independently(monkeypatch, entry, module, name, extra, flags, expected):
    cli = importlib.import_module("histomil.cli")
    target = importlib.import_module(module)
    received = {}
    class Runner:
        def __init__(self, **kwargs):
            received.update(kwargs)
        def run(self):
            pass
    monkeypatch.setattr(target, name, Runner)
    monkeypatch.setattr(sys, "argv", [entry, "--features_path", ".", "--splits_dir", ".",
                                    "--csv_path", "data.csv", *flags,
                                    "--no-use_class_weights", *extra])
    getattr(cli, entry)()
    assert received["use_weighted_sampler"] is expected
    assert received["use_class_weights"] is False


def test_single_class_training_auc_is_nan(tmp_path, monkeypatch):
    training = importlib.import_module("histomil.train")
    monkeypatch.setattr(training, "device", torch.device("cpu"))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2)
        def forward(self, x):
            return {"logits": self.linear(x.mean(dim=1))}, None
    slide = lambda label: (torch.ones(1, 1), torch.tensor(label))
    _, metrics, _ = training.train(Model(), [[slide(0), slide(0)]],
                                    [[slide(0), slide(1)]], str(tmp_path), 0.001, 0, 1)
    assert math.isnan(metrics["train_auc"])
    assert math.isfinite(metrics["val_auc"])
