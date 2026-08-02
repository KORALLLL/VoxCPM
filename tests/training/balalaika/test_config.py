import json
from pathlib import Path

from voxcpm.training.balalaika import artifacts as artifacts_module
from voxcpm.training.balalaika.artifacts import atomic_json
from voxcpm.training.balalaika.config import BalalaikaConfig


def minimal_config(tmp_path):
    return {
        "data": {"corpus_root": str(tmp_path / "corpus")},
        "output_dir": str(tmp_path / "runs"),
    }


def test_default_curriculum_is_exact(tmp_path):
    """Protect the intended two-stage curriculum defaults from drift."""
    cfg = BalalaikaConfig.model_validate(minimal_config(tmp_path))
    assert cfg.stage1.agreement == "lt"
    assert cfg.stage1.threshold == 0.95
    assert cfg.stage1.epochs == 2
    assert cfg.stage2.agreement == "ge"
    assert cfg.stage2.epochs == 3
    assert cfg.lora.model_dump() == {
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": False,
        "r": 32,
        "alpha": 32,
        "dropout": 0.0,
    }


def test_atomic_json_never_leaves_temporary_file(tmp_path):
    """Protect atomic state writes from leaving a staging artifact behind."""
    target = tmp_path / "state.json"
    atomic_json(target, {"step": 7})
    assert json.loads(target.read_text()) == {"step": 7}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_json_fsyncs_the_containing_directory_after_rename(tmp_path, monkeypatch):
    """Catches a renamed manifest being reported durable before its directory entry is flushed."""
    opened: dict[int, Path] = {}
    synced: list[int] = []
    real_open = artifacts_module.os.open
    real_fsync = artifacts_module.os.fsync

    def observe_open(path, flags, mode=0o777):
        descriptor = real_open(path, flags, mode)
        opened[descriptor] = Path(path).resolve()
        return descriptor

    def observe_fsync(descriptor):
        synced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(artifacts_module.os, "open", observe_open)
    monkeypatch.setattr(artifacts_module.os, "fsync", observe_fsync)
    atomic_json(tmp_path / "state.json", {"step": 8})

    assert any(opened.get(descriptor) == tmp_path.resolve() for descriptor in synced)


def test_config_load_resolves_index_dir_outside_corpus_root(tmp_path):
    """Catches generated index artifacts being resolved into the immutable corpus."""
    config_path = tmp_path / "balalaika.yaml"
    config_path.write_text(
        "data:\n" "  corpus_root: corpus\n" "  index_dir: prepared\n" "output_dir: runs\n",
        encoding="utf-8",
    )

    config = BalalaikaConfig.load(config_path)

    assert config.data.index_dir == tmp_path / "prepared"
