import math

import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, BinaryCalibrationError

from histomil.metrics import SlideMetrics


def direct(probs, labels):
    metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=2),
        "auc": AUROC(task="multiclass", num_classes=2),
        "balanced_acc": Accuracy(task="multiclass", num_classes=2, average="macro"),
    })
    values = metrics(probs, labels)
    values["ece"] = BinaryCalibrationError(n_bins=15, norm="l1")(probs[:, 1], labels)
    return {key: value.item() for key, value in values.items()}


def test_matches_direct_metrics_with_uneven_updates_and_ties():
    probs = torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
    labels = torch.tensor([0, 0, 1, 1, 0])
    accumulator = SlideMetrics("cpu")
    accumulator.update(probs[:2], labels[:2])
    accumulator.update(probs[2:], labels[2:])
    assert accumulator.compute() == pytest.approx(direct(probs, labels))
    accumulator.reset()
    accumulator.update(probs[1:], labels[1:])
    assert accumulator.compute() == pytest.approx(direct(probs[1:], labels[1:]))


def test_single_class_auc_and_phase_isolation():
    first, second = SlideMetrics("cpu"), SlideMetrics("cpu")
    first.update(torch.tensor([[0.8, 0.2], [0.6, 0.4]]), torch.tensor([0, 0]))
    second.update(torch.tensor([[0.1, 0.9], [0.9, 0.1]]), torch.tensor([1, 0]))
    assert math.isnan(first.compute()["auc"])
    assert second.compute()["auc"] == pytest.approx(1.0)
    first.reset()
    first.update(torch.tensor([[0.1, 0.9], [0.9, 0.1]]), torch.tensor([1, 0]))
    assert first.compute() == pytest.approx(second.compute())


def test_training_and_test_phase_integration(tmp_path, monkeypatch):
    import importlib

    training = importlib.import_module("histomil.train")
    monkeypatch.setattr(training, "device", torch.device("cpu"))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(1, 2)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, features):
            return {"logits": self.head(features.mean(dim=1))}, None

    def slide(label):
        return torch.tensor([[1.0]]), torch.tensor(label)

    train_batches = [[slide(0), slide(0), slide(0)], [slide(1)]]
    val_batches = [[slide(0)], [slide(1)]]
    model, best, checkpoint = training.train(
        Model(), train_batches, val_batches, str(tmp_path), 0.0, 0, 2,
        patience=5, class_weights=[1, 3],
    )
    assert best["epoch"] == 2
    for phase, labels in (("train", torch.tensor([0, 0, 0, 1])),
                          ("val", torch.tensor([0, 1]))):
        expected = direct(torch.full((len(labels), 2), 0.5), labels)
        for metric, value in expected.items():
            assert best[f"{phase}_{metric}"] == pytest.approx(value)
    scores, outputs, labels = training.test(model, train_batches, class_weights=[1, 3])
    expected = direct(torch.full((4, 2), 0.5), torch.tensor([0, 0, 0, 1]))
    for metric, value in expected.items():
        assert scores[f"test_{metric}"] == pytest.approx(value)
    assert outputs.tolist() == pytest.approx([0.5] * 4)
    assert labels.tolist() == [0, 0, 0, 1]
    assert "f1_macro" in scores
    assert checkpoint
