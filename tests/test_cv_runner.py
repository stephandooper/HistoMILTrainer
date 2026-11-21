"""Exercise the matrix runner without GPUs or real slide features."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cv_matrix.sh"
BASH = shutil.which("bash") if os.name != "nt" else None
pytestmark = pytest.mark.skipif(BASH is None, reason="requires Linux bash")


def run(env=None, *args):
    return subprocess.run([BASH, str(SCRIPT), *args], text=True, capture_output=True,
                          env={**os.environ, **(env or {})})


def commands(result):
    assert result.returncode == 0, result.stderr
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


def option(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_full_matrix_dry_run():
    cmds = commands(run({}, "--dry-run"))
    trains = [c for c in cmds if c[0] == "histomil-train"]
    evaluations = [c for c in cmds if len(c) > 1 and c[1].endswith("evaluate_checkpoint.py")]
    assert len(trains) == 16
    assert len(evaluations) == 1280
    pairs = {(option(c, "--feature_extractor"), option(c, "--mil")) for c in trains}
    assert len(pairs) == 16
    assert all(option(c, "--folds") == "10" and "--use_weighted_sampler" in c
               and "--no-use_class_weights" in c for c in trains)
    assert {(option(c, "--feature_extractor"), option(c, "--mil")) for c in evaluations} == pairs
    for pair in pairs:
        pair_evals = [c for c in evaluations if (option(c, "--feature_extractor"), option(c, "--mil")) == pair]
        assert len(pair_evals) == 80
        assert {option(c, "--fold") for c in pair_evals} == set(map(str, range(10)))
        assert all(option(c, "--weights_path").endswith(f"/{option(c, '--fold')}_best_model.pt") for c in pair_evals)


@pytest.mark.parametrize("stage, expected", [("all", 81), ("train", 1), ("test", 80)])
def test_selection_and_stages(stage, expected):
    cmds = commands(run({"ENCODERS": "uni_v2", "MILS": "clam", "STAGE": stage,
                         "BALANCE": "class_weights"}, "--dry-run"))
    assert len(cmds) == expected
    assert all(option(c, "--feature_extractor") == "uni_v2" and option(c, "--mil") == "clam" for c in cmds)
    if stage != "test":
        assert "--use_class_weights" in cmds[0] and "--no-use_weighted_sampler" in cmds[0]


@pytest.mark.parametrize("env, message", [({"STAGE": "bad"}, "Invalid STAGE"),
    ({"BALANCE": "bad"}, "Invalid BALANCE"), ({"ENCODERS": "bad"}, "Invalid encoder"),
    ({"MILS": "bad"}, "Invalid MIL")])
def test_invalid_settings(env, message):
    result = run(env, "--dry-run")
    assert result.returncode != 0
    assert message in result.stderr


def test_train_resume_fingerprint_and_explicit_partial_rerun(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text(f"#!/bin/bash\nif [[ $1 == -c ]]; then exit 0; fi\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o755)
    trainer = bindir / "histomil-train"
    trainer.write_text("""#!/bin/bash
set -eu
while (($#)); do
  case "$1" in
    --results_dir) out=$2; shift 2;;
    --feature_extractor) encoder=$2; shift 2;;
    --mil) mil=$2; shift 2;;
    --params_path) params=$2; shift 2;;
    *) shift;;
  esac
done
mkdir -p "$out"
for fold in {0..9}; do touch "$out/${fold}_best_model.pt"; done
cp "$params" "$out/best_params_${encoder}.${mil}.json"
echo trained >> "$out/calls"
""")
    trainer.chmod(0o755)
    cv = tmp_path / "splits" / "cv"
    cv.mkdir(parents=True)
    (cv / "dataset.csv").write_text("slide_id,label\nslide,1\n")
    for fold in range(10):
        for suffix in ("bool", "descriptor"):
            (cv / f"splits_{fold}_{suffix}.csv").write_text("slide_id\nslide\n")
    features = tmp_path / "features" / "uni2_0.5mpp_trident" / "20x_224px_0px_overlap" / "features_uni_v2"
    features.mkdir(parents=True)
    (features / "slide.h5").touch()
    params = tmp_path / "params"
    params.mkdir()
    (params / "clam.json").write_text('{"embed_dim": 512}')
    env = {"PATH": str(bindir) + os.pathsep + os.environ["PATH"], "STAGE": "train",
           "ENCODERS": "uni_v2", "MILS": "clam", "FEATURES_ROOT": str(tmp_path / "features"),
           "SPLITS_ROOT": str(tmp_path / "splits"), "RESULTS_ROOT": str(tmp_path / "results"),
           "PARAMS_ROOT": str(params)}
    out = tmp_path / "results" / "uni_v2" / "clam" / "cv"
    first = run(env)
    assert first.returncode == 0, first.stderr
    second = run(env)
    assert second.returncode == 0 and "Skipping completed training" in second.stdout
    assert (out / "calls").read_text().count("trained") == 1
    (params / "clam.json").write_text('{"embed_dim": 256}')
    changed = run(env)
    assert changed.returncode != 0 and "fingerprint changed" in changed.stderr
    (params / "clam.json").write_text('{"embed_dim": 512}')
    (out / ".matrix_complete").unlink()
    partial = run(env)
    assert partial.returncode != 0 and "Partial training" in partial.stderr
    rerun = run({**env, "RERUN_PARTIAL": "1"})
    assert rerun.returncode == 0, rerun.stderr
    assert (out / "calls").read_text().count("trained") == 2
