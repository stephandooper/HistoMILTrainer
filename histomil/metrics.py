"""Slide-level classification metrics shared by all evaluation phases."""

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, BinaryCalibrationError


class SlideMetrics:
    """Accumulate detached probabilities and compute metrics once per phase."""

    def __init__(self, device):
        self.metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=2),
            "auc": AUROC(task="multiclass", num_classes=2),
            "balanced_acc": Accuracy(task="multiclass", num_classes=2, average="macro"),
        }).to(device)
        self.ece = BinaryCalibrationError(n_bins=15, norm="l1").to(device)
        self.classes_seen = set()

    def update(self, probabilities, labels):
        probabilities = probabilities.detach()
        labels = labels.detach().long()
        self.metrics.update(probabilities, labels)
        self.ece.update(probabilities[:, 1], labels)
        self.classes_seen.update(labels.cpu().tolist())

    def compute(self):
        # AUROC is undefined when a phase contains only one target class.
        if len(self.classes_seen) < 2:
            values = {name: metric.compute() for name, metric in self.metrics.items() if name != "auc"}
            values["auc"] = torch.tensor(float("nan"))
        else:
            values = self.metrics.compute()
        values["ece"] = self.ece.compute()
        return {name: value.item() for name, value in values.items()}

    def reset(self):
        self.metrics.reset()
        self.ece.reset()
        self.classes_seen.clear()
