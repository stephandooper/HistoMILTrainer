"""Tests for the MIL-Lab model wrapper and Mascaret features."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


MODEL_PATH = Path(__file__).resolve().parents[1] / "histomil" / "models.py"


def load_models():
    spec = importlib.util.spec_from_file_location("histomil_models_test", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestModelWrapper(unittest.TestCase):
    def test_registers_mascaret_without_overwriting_upstream(self):
        for existing in (None, 2048):
            with self.subTest(existing=existing):
                mapping = {"other": 512}
                if existing is not None:
                    mapping["mascaret"] = existing
                builder = types.ModuleType("src.builder")
                builder.create_model = Mock(side_effect=lambda *args, **kwargs: dict(mapping))
                mappings = types.ModuleType("src._global_mappings")
                mappings.ENCODER_DIM_MAPPING = mapping
                src = types.ModuleType("src")
                src.__path__ = []
                with patch.dict(sys.modules, {"src": src, "src.builder": builder,
                                              "src._global_mappings": mappings}):
                    result = load_models().import_model("abmil", "mascaret", embed_dim=256)
                self.assertEqual(result, {"other": 512, "mascaret": existing or 1536})
                builder.create_model.assert_called_once_with("abmil.base.mascaret", embed_dim=256)

    def test_abmil_forward_cpu(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("src") is None:
            self.skipTest("MIL-Lab and PyTorch are required")
        import torch

        models = load_models()
        for encoder in ("mascaret", "uni_v2"):
            with self.subTest(encoder=encoder):
                model = models.import_model(
                    "abmil", encoder, embed_dim=256, attn_dim=256, dropout=0.25
                )
                model.eval()
                with torch.no_grad():
                    results, _ = model(torch.randn(1, 4, 1536))
                logits = results["logits"]
                self.assertEqual(tuple(logits.shape), (1, 2))
                self.assertTrue(torch.isfinite(logits).all())
