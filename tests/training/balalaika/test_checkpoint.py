import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from voxcpm.training.balalaika.artifacts import fingerprint
from voxcpm.training.balalaika.checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointMismatch,
    CheckpointRestoreError,
)
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


class WrappedModel:
    """Faithful DDP state-key shape: wrapper saves module.*, unwrap loads bare keys."""

    def __init__(self, module):
        self.module = module

    def state_dict(self):
        return {f"module.{key}": value for key, value in self.module.state_dict().items()}


class FakeAccelerator:
    is_main_process = True

    def __init__(
        self,
        model,
        *,
        fail_save=False,
        fail_load=False,
        distributed_type="NO",
        world_size=8,
        optimizer_count=1,
        scheduler_count=1,
        checkpointable_count=1,
    ):
        self.model = model
        self.fail_save = fail_save
        self.fail_load = fail_load
        self.distributed_type = distributed_type
        self.world_size = world_size
        self.optimizer_count = optimizer_count
        self.scheduler_count = scheduler_count
        self.checkpointable_count = checkpointable_count
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
        return getattr(model, "module", model)

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
        for index in range(self.optimizer_count):
            suffix = "" if index == 0 else f"_{index}"
            (output_dir / f"optimizer{suffix}.bin").write_bytes(f"optimizer-{index}".encode())
        for index in range(self.scheduler_count):
            suffix = "" if index == 0 else f"_{index}"
            (output_dir / f"scheduler{suffix}.bin").write_bytes(f"scheduler-{index}".encode())
        for rank in range(self.world_size):
            (output_dir / f"random_states_{rank}.pkl").write_bytes(f"rng-{rank}".encode())
        state = self.checkpointables[-1].state_dict()
        for index in range(self.checkpointable_count):
            torch.save(state, output_dir / f"custom_checkpoint_{index}.pkl")

    def load_state(self, input_dir):
        self.load_state_calls += 1
        models = [self.model]
        for hook in tuple(self.load_hooks):
            hook(models, input_dir)
        assert models == []
        self.unwrap_model(self.model).loaded_optimizer = True
        if self.fail_load:
            raise RuntimeError("simulated late load failure")
        state = torch.load(Path(input_dir) / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=True)
        self.checkpointables[-1].load_state_dict(state)


def checkpoint_metadata(**updates):
    value = {
        "stage_epochs": 2,
        "world_size": 8,
        "microbatch": 2,
        "accumulation": 4,
        "global_batch_size": 64,
        "optimizer_steps_per_epoch": 16,
        "optimizer_count": 1,
        "scheduler_count": 1,
        "checkpointable_count": 1,
        "stage_start_global_step": 0,
        # Manager replaces this declaration with exact per-rank file hashes.
        "rng_state": {"accelerate": "derive"},
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


def same_stage_expected(**updates):
    value = checkpoint_metadata()
    value.pop("rng_state")
    value.update({"stage": "stage1", "sampler_seed": 7})
    value.update(updates)
    return value


def stage2_expected(**updates):
    value = {
        "source_stage": "stage1",
        "source_stage_epochs": 2,
        "source_epoch": 1,
        "source_boundary": 8,
        "base_revision": "base-commit",
        "data_fingerprint": "data-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": "lora-sha",
    }
    value.update(updates)
    return value


def save_checkpoint(tmp_path, *, progress=None, metadata=None, fail_save=False, model=None, accelerator=None):
    model = model or FakeModel()
    accelerator = accelerator or FakeAccelerator(model, fail_save=fail_save)
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
        expected=same_stage_expected(),
    )

    assert accelerator.load_state_calls == 1
    assert resumed_model.loaded_lora and resumed_model.loaded_optimizer
    assert resumed_model.weights["lm.layer.lora_A.weight"].tolist() == [1.0, 2.0]
    assert resumed_progress.optimizer_step == 32
    assert resumed_progress.boundary == 8
    assert set(metadata["rng_state"]["files"]) == {f"random_states_{rank}.pkl" for rank in range(8)}


def test_stage2_load_uses_only_final_stage1_adapter_and_resets_progress(tmp_path):
    manager, path, accelerator, _, _ = save_checkpoint(tmp_path)
    stage2_model = FakeModel()
    stage2_progress = TrainingProgress(stage="stage2")

    manager.load_stage_adapter(
        stage2_model,
        path,
        expected=stage2_expected(),
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
    progress = TrainingProgress(stage="stage1", epoch=0, boundary=7, microstep=56, optimizer_step=14, global_step=14)
    manager, path, _, _, _ = save_checkpoint(tmp_path, progress=progress)
    with pytest.raises(CheckpointMismatch, match="final stage1"):
        manager.load_stage_adapter(FakeModel(), path, expected=stage2_expected())


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
    failed_progress = TrainingProgress(
        stage="stage1",
        epoch=1,
        boundary=7,
        microstep=120,
        optimizer_step=30,
        global_step=30,
        sampler_seed=7,
        sampler_epoch=1,
    )

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
    progress = TrainingProgress(stage="stage1", boundary=1, microstep=8, optimizer_step=2, global_step=2)

    with pytest.raises(CheckpointError, match="scheduler"):
        CheckpointManager(tmp_path).save_same_stage(accelerator, model, progress, checkpoint_metadata())


def test_ddp_prefixed_save_roundtrips_into_complete_unwrapped_lora_target(tmp_path):
    source = FakeModel()
    wrapped_source = WrappedModel(source)
    manager, path, _, _, _ = save_checkpoint(
        tmp_path,
        model=wrapped_source,
        accelerator=FakeAccelerator(wrapped_source),
    )
    assert set(load_file(path / "adapter_model.safetensors")) == {
        "lm.layer.lora_A.weight",
        "dit.layer.lora_B.weight",
    }

    target = FakeModel()
    target.weights["lm.layer.lora_A.weight"] = torch.tensor([-1.0, -1.0])
    wrapped_target = WrappedModel(target)
    manager.resume_same_stage(
        FakeAccelerator(wrapped_target),
        wrapped_target,
        TrainingProgress(stage="stage1"),
        path,
        expected=same_stage_expected(),
    )
    assert target.weights["lm.layer.lora_A.weight"].tolist() == [1.0, 2.0]


def test_real_cpu_ddp_state_keys_roundtrip_without_module_prefix(tmp_path):
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        pytest.skip("requires ownership of a local torch process group")

    class TinyLoRAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1, bias=False)
            self.lora_A = torch.nn.Linear(1, 1, bias=False)
            self.lora_B = torch.nn.Linear(1, 1, bias=False)

    rendezvous = tmp_path / "gloo-rendezvous"
    torch.distributed.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{rendezvous}")
    try:
        source = TinyLoRAModel()
        with torch.no_grad():
            source.lora_A.weight.fill_(5.0)
        wrapped_source = torch.nn.parallel.DistributedDataParallel(source)
        accelerator = FakeAccelerator(wrapped_source, distributed_type="MULTI_CPU", world_size=1)
        metadata = checkpoint_metadata(world_size=1, global_batch_size=8)
        progress = TrainingProgress(
            stage="stage1",
            epoch=1,
            boundary=8,
            microstep=128,
            optimizer_step=32,
            global_step=32,
            sampler_seed=7,
            sampler_epoch=1,
        )
        manager, path, _, _, _ = save_checkpoint(
            tmp_path / "checkpoints",
            model=wrapped_source,
            accelerator=accelerator,
            metadata=metadata,
            progress=progress,
        )
        assert set(load_file(path / "adapter_model.safetensors")) == {
            "lora_A.weight",
            "lora_B.weight",
        }

        target = TinyLoRAModel()
        with torch.no_grad():
            target.lora_A.weight.fill_(-5.0)
        wrapped_target = torch.nn.parallel.DistributedDataParallel(target)
        manager.resume_same_stage(
            FakeAccelerator(wrapped_target, distributed_type="MULTI_CPU", world_size=1),
            wrapped_target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(world_size=1, global_batch_size=8),
        )
        assert target.lora_A.weight.item() == 5.0
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("distributed_type", ["FSDP", "DEEPSPEED"])
def test_unsupported_accelerate_backend_is_rejected_before_state_mutation(tmp_path, distributed_type):
    model = FakeModel()
    accelerator = FakeAccelerator(model, distributed_type=distributed_type)
    progress = TrainingProgress(
        stage="stage1",
        epoch=1,
        boundary=8,
        microstep=128,
        optimizer_step=32,
        global_step=32,
        sampler_seed=7,
        sampler_epoch=1,
    )

    with pytest.raises(CheckpointError, match=f"unsupported.*{distributed_type}"):
        CheckpointManager(tmp_path).save_same_stage(accelerator, model, progress, checkpoint_metadata())

    assert accelerator.save_hooks == []
    assert not any(tmp_path.iterdir())


def test_unsupported_accelerate_backend_is_rejected_before_resume_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    target = FakeModel()
    accelerator = FakeAccelerator(target, distributed_type="FSDP")

    with pytest.raises(CheckpointError, match="unsupported.*FSDP"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(),
        )

    assert not target.loaded_lora
    assert accelerator.load_state_calls == 0


def test_same_stage_resume_rejects_missing_lora_tensor_before_model_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    adapter = load_file(path / "adapter_model.safetensors")
    save_file({"lm.layer.lora_A.weight": adapter["lm.layer.lora_A.weight"]}, path / "adapter_model.safetensors")
    manager._write_manifest(path)
    target = FakeModel()
    accelerator = FakeAccelerator(target)

    with pytest.raises(CheckpointMismatch, match="missing.*dit.layer.lora_B.weight"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(),
        )

    assert not target.loaded_lora
    assert accelerator.load_state_calls == 0


def test_stage2_rejects_missing_lora_tensor_before_model_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    adapter = load_file(path / "adapter_model.safetensors")
    save_file({"lm.layer.lora_A.weight": adapter["lm.layer.lora_A.weight"]}, path / "adapter_model.safetensors")
    manager._write_manifest(path)
    target = FakeModel()

    with pytest.raises(CheckpointMismatch, match="missing.*dit.layer.lora_B.weight"):
        manager.load_stage_adapter(target, path, expected=stage2_expected())

    assert not target.loaded_lora


def test_declared_world_size_requires_every_rank_rng_file_before_load(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    (path / "random_states_7.pkl").unlink()
    manager._write_manifest(path)

    with pytest.raises(CheckpointError, match="random_states_7.pkl"):
        manager.verify(path, {})


def test_exact_accelerate_state_counts_reject_an_extra_optimizer_file(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    (path / "optimizer_1.bin").write_bytes(b"unexpected optimizer")
    manager._write_manifest(path)

    with pytest.raises(CheckpointError, match="optimizer.*file set"):
        manager.verify(path, {})


def test_exact_accelerate_state_counts_reject_an_extra_custom_checkpoint(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    torch.save({"unexpected": True}, path / "custom_checkpoint_1.pkl")
    manager._write_manifest(path)

    with pytest.raises(CheckpointError, match="registered progress.*file set"):
        manager.verify(path, {})


def test_rng_metadata_hash_is_bound_to_each_rank_file(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["rng_state"]["files"]["random_states_3.pkl"] = "0" * 64
    payload = {key: value for key, value in metadata.items() if key != "checkpoint_fingerprint"}
    metadata["checkpoint_fingerprint"] = fingerprint(payload)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manager._write_manifest(path)

    with pytest.raises(CheckpointError, match="RNG metadata checksum.*random_states_3.pkl"):
        manager.verify(path, {})


@pytest.mark.parametrize(
    ("progress", "message"),
    [
        (
            TrainingProgress(stage="stage1", epoch=1, boundary=8, optimizer_step=0, global_step=0, sampler_epoch=1),
            "optimizer_step",
        ),
        (
            TrainingProgress(
                stage="stage1",
                epoch=1,
                boundary=8,
                microstep=127,
                optimizer_step=32,
                global_step=32,
                sampler_epoch=1,
            ),
            "microstep",
        ),
        (
            TrainingProgress(
                stage="stage1",
                epoch=1,
                boundary=8,
                microstep=128,
                optimizer_step=32,
                global_step=32,
                sampler_epoch=0,
            ),
            "sampler_epoch",
        ),
        (
            TrainingProgress(
                stage="stage1",
                epoch=1,
                boundary=8,
                microstep=124,
                optimizer_step=31,
                global_step=31,
                sampler_epoch=1,
            ),
            "boundary",
        ),
        (
            TrainingProgress(
                stage="stage1",
                epoch=1,
                boundary=8,
                microstep=128,
                optimizer_step=32,
                global_step=31,
                sampler_epoch=1,
            ),
            "global_step",
        ),
    ],
)
def test_progress_geometry_invariants_reject_false_boundary_checkpoint(tmp_path, progress, message):
    with pytest.raises(CheckpointError, match=message):
        save_checkpoint(tmp_path, progress=progress)


def test_custom_progress_is_preflighted_against_metadata_before_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    state = torch.load(path / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=True)
    state["optimizer_step"] = 31
    torch.save(state, path / "custom_checkpoint_0.pkl")
    manager._write_manifest(path)
    target = FakeModel()
    accelerator = FakeAccelerator(target)

    with pytest.raises(CheckpointError, match="registered progress.*metadata"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(),
        )

    assert not target.loaded_lora
    assert accelerator.load_state_calls == 0


def test_late_accelerate_load_failure_poisons_manager_until_process_restart(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    target = FakeModel()
    accelerator = FakeAccelerator(target, fail_load=True)

    with pytest.raises(CheckpointRestoreError, match="restart required"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(),
        )
    assert target.loaded_lora

    with pytest.raises(CheckpointRestoreError, match="restart required"):
        manager.load_stage_adapter(FakeModel(), path, expected=stage2_expected())
    with pytest.raises(CheckpointRestoreError, match="restart required"):
        manager.save_same_stage(
            FakeAccelerator(FakeModel()),
            FakeModel(),
            TrainingProgress(
                stage="stage1",
                epoch=1,
                boundary=8,
                microstep=128,
                optimizer_step=32,
                global_step=32,
                sampler_seed=7,
                sampler_epoch=1,
            ),
            checkpoint_metadata(),
            name="after-poison",
        )


def test_resume_requires_complete_expected_identity_before_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    target = FakeModel()
    accelerator = FakeAccelerator(target)

    with pytest.raises(CheckpointMismatch, match="expected identity.*wandb_run_id"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected={key: value for key, value in same_stage_expected().items() if key != "wandb_run_id"},
        )

    assert not target.loaded_lora
    assert accelerator.load_state_calls == 0


def test_resume_identity_mismatch_is_rejected_before_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    target = FakeModel()
    accelerator = FakeAccelerator(target)

    with pytest.raises(CheckpointMismatch, match="base_revision"):
        manager.resume_same_stage(
            accelerator,
            target,
            TrainingProgress(stage="stage1"),
            path,
            expected=same_stage_expected(base_revision="wrong-base"),
        )

    assert not target.loaded_lora
    assert accelerator.load_state_calls == 0


def test_stage2_requires_complete_source_identity_before_mutation(tmp_path):
    manager, path, _, _, _ = save_checkpoint(tmp_path)
    target = FakeModel()

    with pytest.raises(CheckpointMismatch, match="expected source identity.*selection_fingerprint"):
        manager.load_stage_adapter(
            target,
            path,
            expected={key: value for key, value in stage2_expected().items() if key != "selection_fingerprint"},
        )
    assert not target.loaded_lora


def test_recovery_checkpoint_accepts_only_complete_accumulation_geometry(tmp_path):
    model = FakeModel()
    accelerator = FakeAccelerator(model)
    manager = CheckpointManager(tmp_path)
    progress = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=44,
        optimizer_step=11,
        global_step=11,
        sampler_seed=7,
        sampler_epoch=0,
    )

    metadata_input = checkpoint_metadata(optimizer_steps_per_epoch=80)
    path = manager.save_recovery(accelerator, model, progress, metadata_input)
    metadata = manager.verify(path, {"checkpoint_kind": "recovery"})

    assert path.name == "stage1-epoch-0001-recovery-step-0000000011"
    assert metadata["checkpoint_kind"] == "recovery"
    assert metadata["accumulation_microstep"] == 0
    assert metadata["optimizer_step"] == 11
    assert metadata["boundary"] == 1

    incomplete = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=45,
        optimizer_step=11,
        global_step=11,
        sampler_seed=7,
        sampler_epoch=0,
    )
    with pytest.raises(CheckpointError, match="complete accumulation"):
        manager.save_recovery(accelerator, model, incomplete, metadata_input)


def test_boundary_save_still_rejects_non_eighth_optimizer_step(tmp_path):
    progress = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=44,
        optimizer_step=11,
        global_step=11,
        sampler_seed=7,
        sampler_epoch=0,
    )

    with pytest.raises(CheckpointError, match="boundary"):
        save_checkpoint(tmp_path, progress=progress)


def test_same_stage_resume_restores_recovery_before_next_microbatch(tmp_path):
    model = FakeModel()
    manager = CheckpointManager(tmp_path)
    progress = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=44,
        optimizer_step=11,
        global_step=11,
        sampler_seed=7,
        sampler_epoch=0,
    )
    path = manager.save_recovery(
        FakeAccelerator(model),
        model,
        progress,
        checkpoint_metadata(optimizer_steps_per_epoch=80),
    )
    target = FakeModel()
    restored = TrainingProgress(stage="stage1")

    metadata = manager.resume_same_stage(
        FakeAccelerator(target),
        target,
        restored,
        path,
        expected=same_stage_expected(optimizer_steps_per_epoch=80),
    )

    assert metadata["checkpoint_kind"] == "recovery"
    assert restored.state_dict() == progress.state_dict()


def test_stage2_adapter_transition_rejects_recovery_checkpoint(tmp_path):
    model = FakeModel()
    manager = CheckpointManager(tmp_path)
    progress = TrainingProgress(
        stage="stage1",
        epoch=1,
        boundary=8,
        microstep=128,
        optimizer_step=32,
        global_step=32,
        sampler_seed=7,
        sampler_epoch=1,
    )
    path = manager.save_recovery(FakeAccelerator(model), model, progress, checkpoint_metadata())

    with pytest.raises(CheckpointMismatch, match="boundary checkpoint"):
        manager.load_stage_adapter(FakeModel(), path, expected=stage2_expected())
