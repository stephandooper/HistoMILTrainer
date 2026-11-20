"""Validate the installed container without downloading weights; --gpu requires CUDA."""

import argparse
import ctypes
import importlib
import importlib.metadata
from pathlib import Path
import sys
import tempfile


def check_environment():
    import openslide
    import pyvips
    import torch
    import torchvision

    assert sys.version_info[:2] == (3, 10), sys.version
    for name in ("histomil", "src.builder", "trident", "topk.svm", "torchmetrics"):
        importlib.import_module(name)
    assert torch.__version__ == "2.10.0+cu128", torch.__version__
    assert torchvision.__version__ == "0.25.0+cu128", torchvision.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert importlib.metadata.version("torchmetrics") == "1.9.0"
    native = ctypes.CDLL("/opt/native/lib/libopenslide.so.1")
    native.openslide_get_version.restype = ctypes.c_char_p
    assert native.openslide_get_version() == b"4.0.0"
    assert tuple(pyvips.version(i) for i in range(3)) == (8, 16, 0)
    assert pyvips.type_find("VipsOperation", "openslideload") != 0
    print(f"PASS: Python, installed dependencies, CUDA wheels and native libraries "
          f"(Python OpenSlide binding uses {openslide.__library_version__})", flush=True)


def check_imaging():
    import numpy as np
    import openslide
    import pyvips

    # Exercise a partial bottom tile, including the backported libtiff fix.
    pixels = np.random.default_rng(42).integers(0, 256, (519, 769, 3), dtype=np.uint8)
    image = pyvips.Image.new_from_memory(pixels.tobytes(), 769, 519, 3, "uchar")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.tif"
        image.tiffsave(str(path), tile=True, pyramid=True, compression="lzw",
                       tile_width=256, tile_height=256, bigtiff=True)
        expected = pixels[-64:, -64:]
        with openslide.OpenSlide(str(path)) as slide:
            actual = np.asarray(slide.read_region((705, 455), 0, (64, 64)).convert("RGB"))
            np.testing.assert_array_equal(actual, expected)
        loaded = pyvips.Image.openslideload(str(path)).crop(705, 455, 64, 64)
        actual = np.frombuffer(loaded.write_to_memory(), dtype=np.uint8).reshape(64, 64, -1)
        np.testing.assert_array_equal(actual[:, :, :3], expected)
    print("PASS: synthetic TIFF read through OpenSlide and libvips", flush=True)


def check_model_and_metrics():
    import torch
    from histomil.metrics import SlideMetrics
    from histomil.models import import_model

    torch.manual_seed(2)
    model = import_model("abmil", "mascaret", embed_dim=256, attn_dim=256, dropout=0.25)
    model.eval()
    with torch.inference_mode():
        results, _ = model(torch.randn(1, 4, 1536))
    assert results["logits"].shape == (1, 2)
    assert torch.isfinite(results["logits"]).all()
    metrics = SlideMetrics("cpu")
    metrics.update(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), torch.tensor([0, 1]))
    scores = metrics.compute()
    for name in ("acc", "auc", "balanced_acc"):
        assert abs(scores[name] - 1.0) < 1e-6, scores
    assert abs(scores["ece"] - 0.15) < 1e-6, scores
    print("PASS: CPU Mascaret-dimension ABMIL forward and four metrics", flush=True)


def check_gpu():
    import torch

    assert torch.cuda.is_available(), "CUDA unavailable; check --gpus all and the host driver"
    values = torch.ones(2, device="cuda")
    assert values.sum().item() == 2
    print(f"PASS: CUDA allocation on {torch.cuda.get_device_name(0)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    check_environment()
    check_imaging()
    check_model_and_metrics()
    if args.gpu:
        check_gpu()


if __name__ == "__main__":
    main()
