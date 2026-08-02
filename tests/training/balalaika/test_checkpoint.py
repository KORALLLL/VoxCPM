import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from voxcpm.training.balalaika.checkpoint import CheckpointError, CheckpointManager, CheckpointMismatch
from voxcpm.training.balalaika.schedule import TrainingProgress


class _HookHandle:
    def __init__(self, hooks, hook):
        self.hooks = hooks
        self.hook = hook

    def remove(self):
        self.hooks.remove(self.hook)


class FakeModel:
    def __init__(self):
        self.weights = {
            "encoder.weight": torch.tensor([99.0]),
            "lm.layer.lora_A.weight": torch.tensor([1.0, 2.0]),
            "dit.layer.lora_B.weight": torch.tensor([3.0, 4.0]),
        }
        self.loaded_lora = False
        self.loaded_optimizer = False

    def state_dict(self):
        return dict(self.weights)

    def load_state_dict(self, weights, strict=False):
        assert strict is False
        self.loaded_lora = True
        self.weights.update(weights)
        return [], []


class FakeAccelerator:
    is_main_process = True

    def __init__(self, model, *, fail_save=False):
        self.model = model
        self.fail_save = fail_save
        self.save_hooks = []
        self.load_hooks = []
        self.checkpointables = []
        self.load_state_calls = 0

    def register_save_state_pre_hook(self, hook):
        self.save_hooks.append(hook)
        return _HookHandle(self.save_hooks, hook)

    def register_load_state_pre_hook(self, hook):
        self.load_hooks.append(hook)
        return _HookHandle(self.load_hooks, hook)

    def register_for_checkpointing(self, checkpointable):
        if checkpointable not in self.checkpointables:
            self.checkpointables.append(checkpointable)

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def save_state(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = [self.model.state_dict()]
        models = [self.model]
        for hook in tuple(self.save_hooks):
            hook(models, weights, output_dir)
        if self.fail_save:
            raise RuntimeError("simulated interruption")
        assert weights == []
        (output_dir / "optimizer.bin").write_bytes(b"optimizer")
        (output_dir / "scheduler.bin").write_bytes(b"scheduler")
        (output_dir / "random_states_0.pkl").write_bytes(b"rng")
        state = self.checkpointables[-1].state_dict()
        (output_dir / "custom_checkpoint_0.pkl").write_text(json.dumps(state), encoding="utf-8")

    def load_state(self, input_dir):
        self.load_state_calls += 1
        models = [self.model]
        for hook in tuple(self.load_hooks):
            hook(models, input_dir)
        assert models == []
        self.model.loaded_optimizer = True
        state = json.loads((Path(input_dir) / "custom_checkpoint_0.pkl").read_text(encoding="utf-8"))
        self.checkpointables[-1].load_state_dict(state)


def checkpoint_metadata(**updates):
    value = {
        "stage_epochs": 2,
        "world_size": 8,
        "microbatch": 2,
        "accumulation": 4,
        "global_batch_size": 64,
        "rng_state": {"accelerate": "random_states_0.pkl"},
        "base_revision": "base-commit",
        "evaluator_revision": "asr-commit",
        "data_fingerprint": "data-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": "lora-sha",
        "optimization_fingerprint": "optim-sha",
        "wandb_run_id": "run-id",
        "wandb_group": "group-id",
    }
    value.update(updates)
    return value


def save_checkpoint(tmp_path, *, progress=None, metadata=None, fail_save=False):
    model = FakeModel()
    accelerator = FakeAccelerator(model, fail_save=fail_save)
    progress = progress or TrainingProgress(
        stage="stage1",
        epoch=1,
        boundary=8,
        microstep=128,
        optimizer_step=32,
        global_step=32,
        sampler_seed=7,
        sampler_epoch=1,
    )
    manager = CheckpointManager(tmp_path)
    path = manager.save_same_stage(accelerator, model, progress, metadata or checkpoint_metadata())
    return manager, path, accelerator, model, progress


def test_same_stage_checkpoint_is_lora_only_and_atomically_published(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)

    assert path.name == "stage1-epoch-0002-boundary-08"
    assert not list(tmp_path.glob(".*.tmp"))
    assert set(load_file(path / "adapter_model.safetensors")) == {
        "lm.layer.lora_A.weight",
        "dit.layer.lora_B.weight",
    }
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["checkpoint"] == path.name
    assert latest["fingerprint"] == manager.verify(path, {})["checkpoint_fingerprint"]


def test_same_stage_roundtrip_restores_adapter_optimizer_rng_and_progress(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    resumed_model = FakeModel()
    resumed_model.weights["lm.layer.lora_A.weight"] = torch.tensor([-1.0, -1.0])
    accelerator = FakeAccelerator(resumed_model)
    resumed_progress = TrainingProgress(stage="stage1")

    metadata = manager.resume_same_stage(
        accelerator,
        resumed_model,
        resumed_progress,
        path,
        expected=checkpoint_metadata(),
    )

    assert accelerator.load_state_calls == 1
    assert resumed_model.loaded_lora and resumed_model.loaded_optimizer
    assert resumed_model.weights["lm.layer.lora_A.weight"].tolist() == [1.0, 2.0]
    assert resumed_progress.optimizer_step == 32
    assert resumed_progress.boundary == 8
    assert metadata["rng_state"] == {"accelerate": "random_states_0.pkl"}


def test_stage2_load_uses_only_final_stage1_adapter_and_resets_progress(tmp_path):
    manager, path, accelerator, _, _ = save_checkpoint(tmp_path)
    stage2_model = FakeModel()
    stage2_progress = TrainingProgress(stage="stage1", optimizer_step=32, global_step=32)

    manager.load_stage_adapter(
        stage2_model,
        path,
        expected={
            "stage": "stage2",
            "base_revision": "base-commit",
            "data_fingerprint": "data-sha",
            "lora_fingerprint": "lora-sha",
        },
        progress=stage2_progress,
        sampler_seed=91,
    )

    assert stage2_model.loaded_lora
    assert not stage2_model.loaded_optimizer
    assert accelerator.load_state_calls == 0
    assert stage2_progress.stage == "stage2"
    assert stage2_progress.optimizer_step == 0
    assert stage2_progress.global_step == 32


def test_stage_transition_rejects_non_final_or_non_stage1_checkpoint(tmp_path):
    progress = TrainingProgress(stage="stage1", epoch=0, boundary=7, optimizer_step=7, global_step=7)
    manager, path, _, _, _ = save_checkpoint(tmp_path, progress=progress)
    with pytest.raises(CheckpointMismatch, match="final stage1"):
        manager.load_stage_adapter(FakeModel(), path, expected={"stage": "stage2"})


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("world_size", 4),
        ("base_revision", "other-base"),
        ("data_fingerprint", "other-data"),
        ("selection_fingerprint", "other-selection"),
        ("lora_fingerprint", "other-lora"),
        ("microbatch", 9),
        ("accumulation", 9),
        ("optimization_fingerprint", "other-optim"),
    ],
)
def test_same_stage_resume_rejects_immutable_mismatch(tmp_path, field, wrong):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    with pytest.raises(CheckpointMismatch, match=field):
        manager.verify(path, {field: wrong})


def test_missing_or_tampered_checkpoint_piece_fails_before_load(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    (path / "optimizer.bin").write_bytes(b"tampered")

    with pytest.raises(CheckpointError, match="optimizer.bin.*checksum"):
        manager.verify(path, {})


def test_adapter_file_with_non_lora_key_is_rejected_even_with_updated_manifest(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    from safetensors.torch import save_file

    save_file({"encoder.weight": torch.tensor([1.0])}, path / "adapter_model.safetensors")
    manager._write_manifest(path)

    with pytest.raises(CheckpointError, match="non-LoRA"):
        manager.verify(path, {})


def test_interrupted_temporary_checkpoint_is_ignored_and_latest_is_unchanged(tmp_path):
    manager, complete_path, _, _, _ = save_checkpoint(tmp_path)
    latest_before = (tmp_path / "latest.json").read_bytes()
    failed_progress = TrainingProgress(stage="stage1", epoch=1, boundary=7, optimizer_step=33, global_step=33)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        save_checkpoint(tmp_path, progress=failed_progress, fail_save=True)

    assert manager.latest() == complete_path
    assert (tmp_path / "latest.json").read_bytes() == latest_before
    assert all(path.name.startswith(".") for path in tmp_path.iterdir() if "tmp" in path.name)


def test_missing_accelerate_state_category_is_rejected_during_publication(tmp_path):
    class MissingSchedulerAccelerator(FakeAccelerator):
        def save_state(self, output_dir):
            super().save_state(output_dir)
            (Path(output_dir) / "scheduler.bin").unlink()

    model = FakeModel()
    accelerator = MissingSchedulerAccelerator(model)
    progress = TrainingProgress(stage="stage1", boundary=1, optimizer_step=2, global_step=2)

    with pytest.raises(CheckpointError, match="scheduler"):
        CheckpointManager(tmp_path).save_same_stage(accelerator, model, progress, checkpoint_metadata())
