from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika.checkpoint import CheckpointCollectiveError
from voxcpm.training.balalaika.schedule import TrainingProgress
from voxcpm.training.balalaika.trainer import (
    ApprovalRequired,
    BalalaikaTrainer,
    TrainerConfigurationError,
    TrainingRestartRequired,
    build_model,
)


class TinyAudioVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.sample_rate = 16_000


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor([4.0]))
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.lora_B = torch.nn.Parameter(torch.tensor([2.0]))
        self.audio_vae = TinyAudioVAE()
        self.text_tokenizer = lambda text: [len(text)]
        self.config = SimpleNamespace(max_length=32, patch_size=1, feat_dim=1)
        self.sample_rate = 16_000

    def forward(self, value, *, progress):
        anchor = (self.lora_A + self.lora_B).sum() * 0.0
        return {"loss/diff": anchor + 2.0, "loss/stop": anchor + 3.0}


class FakeRuntime:
    def __init__(
        self,
        events,
        *,
        accumulation=1,
        signal_after_sync=None,
        fail_backward_at=None,
        skipped_sync_attempts=(),
        rank=0,
        world_size=1,
        gathered_values=(),
    ):
        self.events = events
        self.accumulation = accumulation
        self.signal_after_sync = signal_after_sync
        self.fail_backward_at = fail_backward_at
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")
        self.sync_gradients = False
        self.accumulate_calls = 0
        self.backward_calls = 0
        self.prepare_calls = 0
        self.clip_calls = 0
        self.barrier_calls = 0
        self.gather_calls = 0
        self.backward_values = []
        self.accelerator = self
        self.skipped_sync_attempts = set(skipped_sync_attempts)
        self.gathered_values = list(gathered_values)

    @property
    def optimizer_step_was_skipped(self):
        return self.accumulate_calls // self.accumulation in self.skipped_sync_attempts

    def prepare(self, *objects):
        self.prepare_calls += 1
        self.events.append(("prepare",))
        assert self.prepare_calls == 1
        return objects

    @contextmanager
    def accumulate(self, model):
        self.accumulate_calls += 1
        self.sync_gradients = self.accumulate_calls % self.accumulation == 0
        self.events.append(("accumulate", self.accumulate_calls))
        yield

    def backward(self, loss):
        self.backward_calls += 1
        self.backward_values.append(float(loss.detach()))
        if self.backward_calls == self.fail_backward_at:
            raise RuntimeError("synthetic backward collective failure")
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_calls += 1
        self.events.append(("clip", self.clip_calls))
        return torch.nn.utils.clip_grad_norm_(tuple(parameters), max_norm)

    def gather(self, value):
        self.gather_calls += 1
        if self.gathered_values:
            return torch.tensor(self.gathered_values.pop(0), dtype=value.dtype, device=value.device)
        sync_attempt = self.accumulate_calls // self.accumulation
        if self.signal_after_sync == sync_attempt:
            return torch.ones_like(value)
        return value

    def barrier(self):
        self.barrier_calls += 1
        self.events.append(("barrier",))

    def unwrap(self, model):
        return model


class CountingOptimizer(torch.optim.SGD):
    def __init__(self, parameters, *, lr, weight_decay, events):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)
        self.events = events
        self.step_calls = 0
        self.zero_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        self.events.append(("optimizer", self.step_calls))
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zero_calls += 1
        self.events.append(("zero", self.zero_calls))
        return super().zero_grad(set_to_none=set_to_none)


class CountingScheduler:
    def __init__(self, optimizer, *, warmup_steps, total_steps, events):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.events = events
        self.step_calls = 0
        self.last_epoch = -1

    def step(self):
        self.step_calls += 1
        self.last_epoch += 1
        self.events.append(("scheduler", self.step_calls))


class FakeCheckpointManager:
    def __init__(self, root, events):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.events = events
        self.boundaries = []
        self.recoveries = []
        self.recovery_metadata = []
        self.resume_state = None
        self.resume_kind = "boundary"
        self.stage2_expected = None

    def _write(self, name, progress, kind, supplied_metadata):
        path = self.root / name
        path.mkdir()
        metadata = {
            **supplied_metadata,
            "checkpoint_fingerprint": f"{kind}-{progress.optimizer_step}",
            "checkpoint_kind": kind,
            **{key: value for key, value in progress.state_dict().items() if key != "schema_version"},
        }
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def save_same_stage(self, accelerator, model, progress, metadata, *, name=None):
        self.events.append(("checkpoint", progress.optimizer_step))
        self.boundaries.append(progress.state_dict())
        return self._write(name or f"boundary-{progress.optimizer_step:04d}", progress, "boundary", metadata)

    def save_recovery(self, accelerator, model, progress, metadata, *, name=None):
        self.events.append(("recovery", progress.optimizer_step))
        self.recoveries.append(progress.state_dict())
        self.recovery_metadata.append(dict(metadata))
        return self._write(name or f"recovery-{progress.optimizer_step:04d}", progress, "recovery", metadata)

    def resume_same_stage(self, accelerator, model, progress, checkpoint, *, expected):
        self.events.append(("resume",))
        metadata_path = Path(checkpoint) / "metadata.json"
        existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if self.resume_state is None:
            self.resume_kind = existing["checkpoint_kind"]
            self.resume_state = {
                "schema_version": 1,
                **{
                    key: existing[key]
                    for key in (
                        "stage",
                        "epoch",
                        "boundary",
                        "microstep",
                        "optimizer_step",
                        "global_step",
                        "sampler_seed",
                        "sampler_epoch",
                    )
                },
            }
        progress.load_state_dict(self.resume_state)
        metadata = {
            **existing,
            "checkpoint_kind": self.resume_kind,
            "checkpoint_fingerprint": f"{self.resume_kind}-{progress.optimizer_step}",
            **{key: value for key, value in progress.state_dict().items() if key != "schema_version"},
        }
        (Path(checkpoint) / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def verify(self, checkpoint, expected):
        self.events.append(("verify-resume",))
        metadata_path = Path(checkpoint) / "metadata.json"
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        assert self.resume_state is not None
        state = self.resume_state
        return {
            "checkpoint_kind": self.resume_kind,
            "stage_start_global_step": state["global_step"] - state["optimizer_step"],
            **{key: value for key, value in state.items() if key != "schema_version"},
        }

    def load_stage_adapter(self, model, checkpoint, *, expected, progress, sampler_seed):
        self.events.append(("stage2-adapter",))
        self.stage2_expected = dict(expected)
        with torch.no_grad():
            model.lora_A.fill_(9.0)
            model.lora_B.fill_(8.0)
        progress.global_step = 123
        progress.reset_for_stage("stage2", sampler_seed=sampler_seed)
        return {"checkpoint_kind": "boundary", "global_step": 123}


class FakeEvaluator:
    def __init__(self, events, *, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.steps = []

    def run(self, model, audio_vae, checkpoint, boundary):
        assert (Path(checkpoint) / "metadata.json").is_file()
        self.events.append(("evaluate", boundary.global_step))
        self.steps.append(boundary.global_step)
        if boundary.global_step == self.fail_at:
            raise RuntimeError("synthetic evaluator failure")
        return SimpleNamespace(durable=True)


def _config(tmp_path, *, stage1_epochs=2, stage2_epochs=3):
    return SimpleNamespace(
        output_dir=tmp_path / "run",
        stage1=SimpleNamespace(epochs=stage1_epochs, learning_rate=1e-4, batch_size=1),
        stage2=SimpleNamespace(epochs=stage2_epochs, learning_rate=5e-5, batch_size=1),
    )


def _identity():
    return {
        "base_revision": "base-sha",
        "evaluator_revision": "asr-sha",
        "data_fingerprint": "data-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": "lora-sha",
        "optimization_fingerprint": "optim-sha",
        "wandb_run_id": "run-id",
        "wandb_group": "group-id",
    }


def _make_trainer(
    tmp_path,
    *,
    stage1_rows=80,
    stage2_rows=80,
    accumulation=1,
    runtime=None,
    checkpoint_manager=None,
    evaluator=None,
    approval=True,
    resume_checkpoint=None,
    stage1_checkpoint=None,
    microbatch_selector=None,
    batch_processor=None,
    loader_factory_override=None,
    config_override=None,
    use_default_marker=False,
):
    events = [] if runtime is None else runtime.events
    runtime = runtime or FakeRuntime(events, accumulation=accumulation)
    manager = checkpoint_manager or FakeCheckpointManager(tmp_path / "checkpoints", events)
    evaluator = evaluator or FakeEvaluator(events)
    optimizers = []
    schedulers = []
    models = []

    def model_builder(config, stage, adapter_checkpoint=None):
        events.append(("model", stage))
        model = TinyModel()
        audio_vae = model.audio_vae
        del model.audio_vae
        model.base_weight.requires_grad_(False)
        audio_vae.requires_grad_(False)
        models.append(model)
        return model, audio_vae, lambda text: [len(text)]

    def dataset_factory(config, stage, tokenizer):
        row_count = stage1_rows if stage == 1 else stage2_rows
        return [torch.tensor([float(index + 1)]) for index in range(row_count)]

    def loader_factory(dataset, **kwargs):
        if loader_factory_override is not None:
            return loader_factory_override(dataset, **kwargs)
        used = len(dataset) - kwargs.get("dropped_samples", 0)
        return list(dataset[:used])

    def processor_factory(model, audio_vae, runtime):
        if batch_processor is not None:
            return batch_processor
        return lambda batch: {"value": batch}

    def optimizer_factory(parameters, *, lr, weight_decay):
        optimizer = CountingOptimizer(parameters, lr=lr, weight_decay=weight_decay, events=events)
        optimizers.append(optimizer)
        return optimizer

    def scheduler_factory(optimizer, *, warmup_steps, total_steps):
        scheduler = CountingScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            events=events,
        )
        schedulers.append(scheduler)
        return scheduler

    def verifier(path, expected):
        events.append(("approval", Path(path).name, dict(expected)))
        if not approval:
            raise ValueError("approval fingerprint mismatch")
        return SimpleNamespace(approved=True)

    def marker(boundary, checkpoint):
        events.append(("mark", boundary.global_step))
        path = tmp_path / "markers" / f"{Path(checkpoint).name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")
        return path

    trainer = BalalaikaTrainer(
        config_override or _config(tmp_path),
        runtime,
        checkpoint_manager=manager,
        evaluator_factory=lambda boundary, checkpoint: evaluator,
        identity=_identity(),
        approval_verifier=verifier if approval is not None else None,
        approval_path=tmp_path / "approval.json" if approval is not None else None,
        approval_expected={"memorization": "approved"} if approval is not None else None,
        model_builder=model_builder,
        dataset_factory=dataset_factory,
        loader_factory=loader_factory,
        batch_processor_factory=processor_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        microbatch_selector=microbatch_selector,
        boundary_marker=None if use_default_marker else marker,
        accumulation=accumulation,
        warmup_fraction=0.25,
        loss_weights={"loss/diff": 0.5, "loss/stop": 2.0},
        resume_checkpoint=resume_checkpoint,
        stage1_checkpoint=stage1_checkpoint,
        install_signal_handlers=False,
    )
    return SimpleNamespace(
        trainer=trainer,
        runtime=runtime,
        manager=manager,
        evaluator=evaluator,
        events=events,
        optimizers=optimizers,
        schedulers=schedulers,
        models=models,
    )


class FakeVoxCPM2:
    call = None

    @classmethod
    def from_local(cls, path, **kwargs):
        cls.call = (path, kwargs)
        return TinyModel()


class FakeLoRAConfig(SimpleNamespace):
    pass


def test_build_model_uses_verified_local_voxcpm2_and_freezes_exactly_non_lora(tmp_path):
    model_dir = tmp_path / "hub" / "model"
    model_dir.mkdir(parents=True)
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"tiny-local-model")
    pin = {
        "kind": "model",
        "repo_id": "OpenBMB/VoxCPM2",
        "revision": "immutable-sha",
        "local_dir": str(model_dir),
        "files": {"model.safetensors": sha256_file(weights)},
    }
    (tmp_path / "hub" / "hub-pins.json").write_text(json.dumps({"model": pin}), encoding="utf-8")
    config = SimpleNamespace(
        hub=SimpleNamespace(
            local_dir=tmp_path / "hub",
            model_repo_id="OpenBMB/VoxCPM2",
        ),
        lora=SimpleNamespace(
            enable_lm=True,
            enable_dit=True,
            enable_proj=False,
            r=32,
            alpha=32,
            dropout=0.0,
        ),
    )

    model, audio_vae, tokenizer = build_model(
        config,
        1,
        model_cls=FakeVoxCPM2,
        lora_config_cls=FakeLoRAConfig,
    )

    assert FakeVoxCPM2.call[0] == str(model_dir)
    lora = FakeVoxCPM2.call[1]["lora_config"]
    assert (lora.enable_lm, lora.enable_dit, lora.enable_proj, lora.r, lora.alpha, lora.dropout) == (
        True,
        True,
        False,
        32,
        32,
        0.0,
    )
    assert all(("lora_" in name) == parameter.requires_grad for name, parameter in model.named_parameters())
    assert not hasattr(model, "audio_vae")
    assert audio_vae is not None and all(not parameter.requires_grad for parameter in audio_vae.parameters())
    assert tokenizer("тест") == [4]


@pytest.mark.parametrize(("stage", "expected_count"), [(1, 16), (2, 24)])
def test_each_stage_fires_exactly_eight_boundaries_per_epoch_with_no_step_zero(tmp_path, stage, expected_count):
    stage1_checkpoint = tmp_path / "stage1-final" if stage == 2 else None
    if stage1_checkpoint is not None:
        stage1_checkpoint.mkdir()
    fixture = _make_trainer(tmp_path, stage1_checkpoint=stage1_checkpoint)

    fixture.trainer.run_stage(stage)

    assert len(fixture.evaluator.steps) == expected_count
    expected_steps = list(range(10, 161, 10)) if stage == 1 else list(range(133, 364, 10))
    assert fixture.evaluator.steps == expected_steps
    assert 0 not in fixture.evaluator.steps
    ordering = [event[0] for event in fixture.events if event[0] in {"checkpoint", "evaluate", "mark"}]
    assert ordering == ["checkpoint", "evaluate", "mark"] * expected_count


def test_accumulation_clips_steps_and_zeros_only_at_sync_and_scheduler_never_leads_progress(tmp_path):
    fixture = _make_trainer(tmp_path, stage1_rows=17, accumulation=2)

    fixture.trainer.run_stage(1)

    optimizer = fixture.optimizers[0]
    scheduler = fixture.schedulers[0]
    assert fixture.runtime.accumulate_calls == 32
    assert fixture.runtime.backward_calls == 32
    assert fixture.runtime.clip_calls == 16
    assert optimizer.step_calls == optimizer.zero_calls == scheduler.step_calls == 16
    assert fixture.manager.boundaries[-1]["optimizer_step"] == 16
    assert all(value == pytest.approx(7.0) for value in fixture.runtime.backward_values)


def test_skipped_optimizer_attempt_does_not_advance_scheduler_or_progress(tmp_path):
    events = []
    runtime = FakeRuntime(events, accumulation=2, skipped_sync_attempts={3})
    fixture = _make_trainer(tmp_path, stage1_rows=17, accumulation=2, runtime=runtime)

    fixture.trainer.run_stage(1)

    optimizer = fixture.optimizers[0]
    scheduler = fixture.schedulers[0]
    assert optimizer.step_calls == optimizer.zero_calls == 17
    assert scheduler.step_calls == 16
    assert fixture.runtime.clip_calls == 17
    assert fixture.manager.boundaries[-1]["optimizer_step"] == 16


def test_stop_request_survives_skipped_attempt_until_next_real_update(tmp_path):
    events = []
    runtime = FakeRuntime(events, skipped_sync_attempts={1}, signal_after_sync=1)
    fixture = _make_trainer(tmp_path, stage1_rows=8, runtime=runtime)

    result = fixture.trainer.run_stage(1)

    assert result.name == "recovery-0001"
    assert fixture.optimizers[0].step_calls == 2
    assert fixture.schedulers[0].step_calls == 1
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 1


def test_stage2_loads_final_stage1_adapter_before_fresh_optimizer_and_resets_all_stage_progress(tmp_path):
    source = tmp_path / "stage1-final"
    source.mkdir()
    fixture = _make_trainer(tmp_path, stage2_rows=8, stage1_checkpoint=source)

    result = fixture.trainer.run_stage(2)

    assert result.is_dir()
    assert fixture.events.index(("stage2-adapter",)) < fixture.events.index(("prepare",))
    assert fixture.runtime.prepare_calls == 1
    assert fixture.optimizers[0].state == {}
    assert fixture.schedulers[0].total_steps == 24
    assert fixture.schedulers[0].step_calls == 24
    assert fixture.manager.boundaries[0]["optimizer_step"] == 1
    assert fixture.manager.boundaries[0]["global_step"] == 124
    assert fixture.manager.boundaries[-1]["global_step"] == 147
    assert fixture.manager.stage2_expected == {
        "source_stage": "stage1",
        "source_stage_epochs": 2,
        "source_epoch": 1,
        "source_boundary": 8,
        "base_revision": "base-sha",
        "data_fingerprint": "data-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": "lora-sha",
    }


def test_same_stage_boundary_resume_finishes_validation_before_next_update(tmp_path):
    checkpoint = tmp_path / "resume-boundary"
    checkpoint.mkdir()
    fixture = _make_trainer(tmp_path, resume_checkpoint=checkpoint)
    fixture.manager.resume_state = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=10,
        optimizer_step=10,
        global_step=10,
        sampler_seed=0,
        sampler_epoch=0,
    ).state_dict()

    fixture.trainer.run_stage(1)

    names = [event[0] for event in fixture.events]
    assert names.index("resume") < names.index("evaluate") < names.index("mark") < names.index("accumulate")
    assert fixture.evaluator.steps.count(10) == 1


def test_recovery_resume_does_not_falsely_rerun_last_completed_boundary(tmp_path):
    checkpoint = tmp_path / "resume-recovery"
    checkpoint.mkdir()
    events = []
    runtime = FakeRuntime(events, signal_after_sync=1)
    manager = FakeCheckpointManager(tmp_path / "checkpoints", events)
    manager.resume_kind = "recovery"
    manager.resume_state = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=11,
        optimizer_step=11,
        global_step=11,
        sampler_seed=0,
        sampler_epoch=0,
    ).state_dict()
    fixture = _make_trainer(
        tmp_path,
        runtime=runtime,
        checkpoint_manager=manager,
        resume_checkpoint=checkpoint,
    )

    fixture.trainer.run_stage(1)

    assert fixture.evaluator.steps == []
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 12


def test_recovery_resume_at_exact_boundary_finishes_missing_validation_before_update(tmp_path):
    checkpoint = tmp_path / "resume-recovery"
    checkpoint.mkdir()
    events = []
    runtime = FakeRuntime(events, signal_after_sync=1)
    manager = FakeCheckpointManager(tmp_path / "checkpoints", events)
    manager.resume_kind = "recovery"
    manager.resume_state = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=10,
        optimizer_step=10,
        global_step=10,
        sampler_seed=0,
        sampler_epoch=0,
    ).state_dict()
    fixture = _make_trainer(
        tmp_path,
        runtime=runtime,
        checkpoint_manager=manager,
        resume_checkpoint=checkpoint,
        use_default_marker=True,
    )

    fixture.trainer.run_stage(1)

    names = [event[0] for event in fixture.events]
    assert names.index("resume") < names.index("evaluate") < names.index("accumulate")
    assert fixture.evaluator.steps.count(10) == 1


def test_recovery_resume_at_exact_boundary_honors_matching_durable_marker(tmp_path):
    checkpoint = tmp_path / "resume-recovery"
    checkpoint.mkdir()
    events = []
    runtime = FakeRuntime(events, signal_after_sync=1)
    manager = FakeCheckpointManager(tmp_path / "checkpoints", events)
    manager.resume_kind = "recovery"
    progress = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=10,
        optimizer_step=10,
        global_step=10,
        sampler_seed=0,
        sampler_epoch=0,
    )
    manager.resume_state = progress.state_dict()
    fixture = _make_trainer(
        tmp_path,
        runtime=runtime,
        checkpoint_manager=manager,
        resume_checkpoint=checkpoint,
        use_default_marker=True,
    )
    (checkpoint / "metadata.json").write_text(json.dumps({"checkpoint_fingerprint": "recovery-10"}), encoding="utf-8")
    boundary = fixture.trainer._evaluation_boundary(progress, total_steps=160)
    marker_path = fixture.trainer._marker_path(checkpoint)
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps(fixture.trainer._marker_value(checkpoint, boundary)), encoding="utf-8")

    fixture.trainer.run_stage(1)

    assert fixture.evaluator.steps == []
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 11


def test_recovery_resume_at_exact_boundary_replaces_mismatched_marker_before_update(tmp_path):
    checkpoint = tmp_path / "resume-recovery"
    checkpoint.mkdir()
    events = []
    runtime = FakeRuntime(events, signal_after_sync=1)
    manager = FakeCheckpointManager(tmp_path / "checkpoints", events)
    manager.resume_kind = "recovery"
    manager.resume_state = TrainingProgress(
        stage="stage1",
        epoch=0,
        boundary=1,
        microstep=10,
        optimizer_step=10,
        global_step=10,
        sampler_seed=0,
        sampler_epoch=0,
    ).state_dict()
    fixture = _make_trainer(
        tmp_path,
        runtime=runtime,
        checkpoint_manager=manager,
        resume_checkpoint=checkpoint,
        use_default_marker=True,
    )
    marker_path = fixture.trainer._marker_path(checkpoint)
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps({"status": "complete", "checkpoint": "wrong"}), encoding="utf-8")

    fixture.trainer.run_stage(1)

    names = [event[0] for event in fixture.events]
    assert names.index("evaluate") < names.index("accumulate")
    assert fixture.evaluator.steps == [10]
    assert json.loads(marker_path.read_text(encoding="utf-8"))["checkpoint_fingerprint"] == "recovery-10"


def test_exact_boundary_signal_recovery_reuses_production_boundary_marker_on_resume(tmp_path):
    class StopAfterBoundaryEvaluator(FakeEvaluator):
        trainer = None

        def run(self, model, audio_vae, checkpoint, boundary):
            result = super().run(model, audio_vae, checkpoint, boundary)
            assert self.trainer is not None
            self.trainer.request_stop()
            return result

    first_events = []
    manager = FakeCheckpointManager(tmp_path / "checkpoints", first_events)
    first_evaluator = StopAfterBoundaryEvaluator(first_events)
    first = _make_trainer(
        tmp_path,
        runtime=FakeRuntime(first_events),
        checkpoint_manager=manager,
        evaluator=first_evaluator,
        use_default_marker=True,
    )
    first_evaluator.trainer = first.trainer

    recovery = first.trainer.run_stage(1)

    boundary_checkpoint = manager.root / "boundary-0010"
    boundary_marker = first.trainer._marker_path(boundary_checkpoint)
    assert recovery.name == "recovery-0010"
    assert first_evaluator.steps == [10]
    assert boundary_marker.is_file()
    assert not first.trainer._marker_path(recovery).exists()
    assert manager.recovery_metadata[-1]["completed_boundary_proof"] == {
        "version": 1,
        "checkpoint_name": "boundary-0010",
        "checkpoint_fingerprint": "boundary-10",
        "boundary": {
            "stage": "stage1",
            "epoch": 0,
            "boundary": 1,
            "global_step": 10,
            "stage_progress": 0.0625,
        },
    }

    second_events = []
    second_evaluator = FakeEvaluator(second_events)
    second = _make_trainer(
        tmp_path,
        runtime=FakeRuntime(second_events, signal_after_sync=1),
        checkpoint_manager=manager,
        evaluator=second_evaluator,
        resume_checkpoint=recovery,
        use_default_marker=True,
    )
    manager.resume_state = None

    second.trainer.run_stage(1)

    assert second_evaluator.steps == []
    assert manager.recoveries[-1]["optimizer_step"] == 11


def test_same_stage2_recovery_resume_does_not_require_stage1_checkpoint(tmp_path):
    checkpoint = tmp_path / "stage2-recovery"
    checkpoint.mkdir()
    events = []
    runtime = FakeRuntime(events, signal_after_sync=1)
    manager = FakeCheckpointManager(tmp_path / "checkpoints", events)
    manager.resume_kind = "recovery"
    manager.resume_state = TrainingProgress(
        stage="stage2",
        epoch=0,
        boundary=1,
        microstep=1,
        optimizer_step=1,
        global_step=124,
        sampler_seed=0,
        sampler_epoch=0,
    ).state_dict()
    fixture = _make_trainer(
        tmp_path,
        stage2_rows=8,
        runtime=runtime,
        checkpoint_manager=manager,
        resume_checkpoint=checkpoint,
        stage1_checkpoint=None,
    )

    fixture.trainer.run_stage(2)

    assert ("stage2-adapter",) not in fixture.events
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 2
    assert fixture.manager.recoveries[-1]["global_step"] == 125


def test_stage1_requires_an_injected_matching_manual_approval_before_model_setup(tmp_path):
    missing = _make_trainer(tmp_path / "missing", approval=None)
    with pytest.raises(ApprovalRequired, match="manual memorization approval"):
        missing.trainer.run_stage(1)
    assert missing.events == []

    mismatch = _make_trainer(tmp_path / "mismatch", approval=False)
    with pytest.raises(ValueError, match="approval fingerprint mismatch"):
        mismatch.trainer.run_stage(1)
    assert mismatch.events[0][0] == "approval"
    assert mismatch.events[0][2] == {
        "memorization": "approved",
        "base_revision": "base-sha",
        "data_fingerprint": "data-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": "lora-sha",
    }
    assert all(event[0] != "model" for event in mismatch.events)


def test_probe_runs_on_local_unwrapped_model_before_one_joint_prepare(tmp_path):
    observed = []

    def selector(runtime, model, optimizer, stage_config):
        observed.append((runtime.prepare_calls, model, optimizer, stage_config.batch_size))
        return 1

    fixture = _make_trainer(tmp_path, microbatch_selector=selector)

    fixture.trainer.run_stage(1)

    assert observed[0][0] == 0
    assert observed[0][1] is fixture.models[0]
    assert observed[0][2] is fixture.optimizers[0]
    assert fixture.runtime.prepare_calls == 1


def test_remote_signal_saves_non_boundary_recovery_after_current_real_step(tmp_path):
    events = []
    runtime = FakeRuntime(events, signal_after_sync=11)
    fixture = _make_trainer(tmp_path, runtime=runtime)

    result = fixture.trainer.run_stage(1)

    assert result.name == "recovery-0011"
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 11
    assert fixture.manager.recoveries[-1]["microstep"] == 11
    assert fixture.evaluator.steps == [10]
    assert fixture.schedulers[0].step_calls == 11


def test_coordinated_exception_saves_safe_recovery_and_reraises_original(tmp_path):
    calls = 0

    def processor(batch):
        nonlocal calls
        calls += 1
        if calls == 12:
            raise ValueError("synthetic batch failure")
        return {"value": batch}

    fixture = _make_trainer(tmp_path, batch_processor=processor)

    with pytest.raises(ValueError, match="synthetic batch failure"):
        fixture.trainer.run_stage(1)

    assert fixture.manager.recoveries[-1]["optimizer_step"] == 11
    assert fixture.runtime.barrier_calls >= 1


def test_forward_failure_mid_accumulation_group_requires_restart_without_recovery(tmp_path):
    calls = 0

    def processor(batch):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise ValueError("mid-group forward failure")
        return {"value": batch}

    fixture = _make_trainer(tmp_path, stage1_rows=17, accumulation=2, batch_processor=processor)

    with pytest.raises(TrainingRestartRequired, match="last durable checkpoint") as raised:
        fixture.trainer.run_stage(1)

    assert isinstance(raised.value.__cause__, ValueError)
    assert fixture.manager.recoveries == []
    assert fixture.manager.boundaries[-1]["optimizer_step"] == 1


def test_loader_failure_mid_accumulation_group_requires_restart_without_recovery(tmp_path):
    class FailingLoader:
        def __iter__(self):
            for index in range(3):
                yield torch.tensor([float(index + 1)])
            raise OSError("mid-group loader failure")

    fixture = _make_trainer(
        tmp_path,
        stage1_rows=17,
        accumulation=2,
        loader_factory_override=lambda dataset, **kwargs: FailingLoader(),
    )

    with pytest.raises(TrainingRestartRequired, match="last durable checkpoint") as raised:
        fixture.trainer.run_stage(1)

    assert isinstance(raised.value.__cause__, OSError)
    assert fixture.manager.recoveries == []
    assert fixture.manager.boundaries[-1]["optimizer_step"] == 1


def test_replay_reuses_sampler_epoch_and_next_real_epoch_increments_once(tmp_path):
    class EpochLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

        def __iter__(self):
            return iter(self.dataset)

    holder = {}

    def loader_factory(dataset, **kwargs):
        holder["loader"] = EpochLoader(dataset)
        return holder["loader"]

    events = []
    runtime = FakeRuntime(events, skipped_sync_attempts={8})
    fixture = _make_trainer(
        tmp_path,
        stage1_rows=8,
        runtime=runtime,
        loader_factory_override=loader_factory,
    )

    fixture.trainer.run_stage(1)

    assert holder["loader"].epochs[:3] == [0, 0, 1]


def test_main_rank_marker_failure_is_published_before_original_error(tmp_path):
    events = []
    runtime = FakeRuntime(events, rank=0, world_size=2, gathered_values=([0, 1],))
    fixture = _make_trainer(tmp_path, runtime=runtime)
    model = TinyModel()
    audio_vae = model.audio_vae
    checkpoint = tmp_path / "boundary"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"checkpoint_fingerprint": "boundary-1"}), encoding="utf-8")
    boundary = fixture.trainer._evaluation_boundary(
        TrainingProgress(stage="stage1", boundary=1, optimizer_step=1, global_step=1),
        total_steps=16,
    )

    def fail_marker(boundary, checkpoint):
        raise OSError("marker disk failure")

    fixture.trainer.boundary_marker = fail_marker

    with pytest.raises(OSError, match="marker disk failure"):
        fixture.trainer._finish_boundary(model, audio_vae, checkpoint, boundary)

    assert runtime.gather_calls == 1


def test_peer_rank_observes_main_marker_failure_without_writing_marker(tmp_path):
    events = []
    runtime = FakeRuntime(events, rank=1, world_size=2, gathered_values=([0, 1],))
    fixture = _make_trainer(tmp_path, runtime=runtime)
    model = TinyModel()
    audio_vae = model.audio_vae
    checkpoint = tmp_path / "boundary"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"checkpoint_fingerprint": "boundary-1"}), encoding="utf-8")
    boundary = fixture.trainer._evaluation_boundary(
        TrainingProgress(stage="stage1", boundary=1, optimizer_step=1, global_step=1),
        total_steps=16,
    )
    marker_called = False

    def marker(boundary, checkpoint):
        nonlocal marker_called
        marker_called = True
        return tmp_path / "should-not-exist"

    fixture.trainer.boundary_marker = marker

    with pytest.raises(RuntimeError, match="main rank failed.*marker"):
        fixture.trainer._finish_boundary(model, audio_vae, checkpoint, boundary)

    assert marker_called is False
    assert runtime.gather_calls == 1


@pytest.mark.parametrize(
    ("stage", "field", "value"),
    [
        (1, "epochs", 3),
        (1, "learning_rate", 2e-4),
        (2, "epochs", 2),
        (2, "learning_rate", 1e-4),
    ],
)
def test_curriculum_rejects_stage_epoch_or_learning_rate_drift_before_model_setup(tmp_path, stage, field, value):
    config = _config(tmp_path)
    setattr(getattr(config, f"stage{stage}"), field, value)
    fixture = _make_trainer(tmp_path, config_override=config)

    with pytest.raises(TrainerConfigurationError, match="mandated curriculum"):
        fixture.trainer.run_stage(stage)

    assert all(event[0] != "model" for event in fixture.events)


def test_rank_local_dataloader_failure_is_coordinated_before_recovery(tmp_path):
    class FailingLoader:
        def __iter__(self):
            for index in range(11):
                yield torch.tensor([float(index + 1)])
            raise OSError("synthetic worker failure")

    events = []
    runtime = FakeRuntime(events)
    fixture = _make_trainer(
        tmp_path,
        runtime=runtime,
        loader_factory_override=lambda dataset, **kwargs: FailingLoader(),
    )

    with pytest.raises(OSError, match="synthetic worker failure"):
        fixture.trainer.run_stage(1)

    # Eleven successful batches use fetch + forward + optimizer + stop
    # collectives. Boundary-marker publication and the failing fetch each add
    # one collective outcome.
    assert runtime.gather_calls == 46
    assert fixture.manager.recoveries[-1]["optimizer_step"] == 11


def test_broken_backward_collective_does_not_attempt_divergent_recovery_save(tmp_path):
    events = []
    runtime = FakeRuntime(events, fail_backward_at=12)
    fixture = _make_trainer(tmp_path, runtime=runtime)

    with pytest.raises(TrainingRestartRequired, match="last durable checkpoint") as raised:
        fixture.trainer.run_stage(1)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert fixture.manager.recoveries == []
    assert fixture.manager.boundaries[-1]["optimizer_step"] == 10


def test_broken_checkpoint_outcome_collective_does_not_attempt_recovery_save(tmp_path):
    events = []

    class BrokenCheckpointManager(FakeCheckpointManager):
        def save_same_stage(self, accelerator, model, progress, metadata, *, name=None):
            raise CheckpointCollectiveError("checkpoint outcome collective failed; process restart required")

    manager = BrokenCheckpointManager(tmp_path / "checkpoints", events)
    fixture = _make_trainer(
        tmp_path,
        stage1_rows=8,
        runtime=FakeRuntime(events),
        checkpoint_manager=manager,
    )

    with pytest.raises(TrainingRestartRequired, match="last durable checkpoint") as raised:
        fixture.trainer.run_stage(1)

    assert isinstance(raised.value.__cause__, CheckpointCollectiveError)
    assert manager.recoveries == []
