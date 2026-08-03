from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from voxcpm.training.balalaika.artifacts import fingerprint, sha256_file
from voxcpm.training.balalaika.memorization import (
    ApprovalMismatch,
    MemorizationError,
    approve_memorization,
    run_memorization,
    verify_approval,
)
from voxcpm.training.balalaika.tracking import WandbRunManager, memorization_pair_payload
from voxcpm.training.balalaika.trainer import BalalaikaTrainer


class TinyAudioVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class TinyModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.base_weight = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.sample_rate = 16_000
        self.config = SimpleNamespace(max_length=32, patch_size=1, feat_dim=1)

    def forward(self, value, *, progress):
        del progress
        anchor = self.lora_A.sum() * 0.0
        return {"loss/diff": anchor + value.float().mean(), "loss/stop": anchor + 0.5}

    def generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        *,
        seed: int | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_len: int = 2_000,
    ):
        if not hasattr(self, "audio_vae"):
            raise RuntimeError("generation requires the retained AudioVAE")
        len(prompt_wav_path)
        kwargs = {
            "target_text": target_text,
            "prompt_text": prompt_text,
            "prompt_wav_path": prompt_wav_path,
            "seed": seed,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "max_len": max_len,
        }
        self.events.append(("generate", kwargs))
        return np.zeros(160, dtype=np.float32)


class FakeRuntime:
    device = torch.device("cpu")
    sync_gradients = True

    def __init__(self, events, *, rank=0, world_size=1, gathered_values=(), on_gather=None):
        self.events = events
        self.rank = rank
        self.world_size = world_size
        self.accelerator = self
        self.backward_calls = 0
        self.skipped_attempts = set()
        self.gathered_values = list(gathered_values)
        self.gather_calls = 0
        self.on_gather = on_gather

    @property
    def optimizer_step_was_skipped(self):
        return self.backward_calls in self.skipped_attempts

    def prepare(self, *objects):
        self.events.append(("prepare", len(objects)))
        return objects

    @contextmanager
    def accumulate(self, model):
        del model
        yield

    def backward(self, loss):
        self.backward_calls += 1
        self.events.append(("backward", float(loss.detach())))
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(tuple(parameters), max_norm)

    def gather(self, value):
        self.gather_calls += 1
        if self.on_gather is not None:
            self.on_gather(self.gather_calls)
        if self.gathered_values:
            return torch.tensor(self.gathered_values.pop(0), dtype=value.dtype, device=value.device)
        return value

    def barrier(self):
        self.events.append(("barrier",))

    def unwrap(self, model):
        return model


class FakeScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class FakeCheckpointManager:
    instances = []

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.saved_metadata = None
        type(self).instances.append(self)

    def save_recovery(self, accelerator, model, progress, metadata, *, name=None):
        del accelerator, model
        self.saved_metadata = dict(metadata)
        checkpoint = self.root / (name or "memorization-final")
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fake-lora-checkpoint")
        (checkpoint / "metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "checkpoint_kind": "recovery",
                    "checkpoint_fingerprint": "checkpoint-sha",
                    "global_step": progress.global_step,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint


class FakeRunManager:
    def __init__(self):
        self.run_id = "wandb123"
        self._settings = SimpleNamespace(mode="online", group=None)
        self.losses = []
        self.pairs = []
        self.finish_calls = 0
        self.fail_train = False

    def log_train(self, metrics, global_step):
        if self.fail_train:
            raise RuntimeError("injected train log failure")
        self.losses.append((dict(metrics), global_step))

    def log_memorization(self, pairs, global_step):
        self.pairs.append((list(pairs), global_step))
        payload = memorization_pair_payload(
            pairs,
            global_step=global_step,
            run_id=self.run_id,
            mode="online",
        )
        return {
            "run_id": self.run_id,
            "mode": "online",
            "global_step": global_step,
            "pair_count": 4,
            "pair_payload": payload,
            "pair_payload_fingerprint": fingerprint(payload),
        }

    def finish(self):
        self.finish_calls += 1


class FakeASR:
    def __init__(self):
        self.paths = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def transcribe(self, path):
        self.paths.append(Path(path))
        return "намеренно неверная диагностика"


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)


@pytest.fixture
def mem_fixture(tmp_path, monkeypatch):
    events = []
    selection_dir = tmp_path / "selection"
    selected_ids = [f"000000/sample-{number:03d}.wav" for number in range(4)]
    samples = []
    for number, sample_id in enumerate(selected_ids):
        wav_path = selection_dir / "audio" / f"item-{number:02d}.wav"
        _write_wav(wav_path)
        samples.append(
            {
                "source_relative_path": sample_id,
                "agreement": 0.99,
                "stage": 2,
                "text": f"текст {number}",
                "wav_path": str(wav_path),
                "wav_sha256": sha256_file(wav_path),
            }
        )
    selection_fingerprint = "selection-sha"
    (selection_dir / "memorization.json").write_text(
        json.dumps({"schema_version": 1, "fingerprint": selection_fingerprint, "seed": 19, "memorization": samples}),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with sqlite3.connect(index_dir / "balalaika-index.sqlite3") as database:
        database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        database.execute("INSERT INTO metadata VALUES ('fingerprint', 'index-sha')")

    hub_dir = tmp_path / "hub"
    (hub_dir / "model").mkdir(parents=True)
    (hub_dir / "gigaam").mkdir()
    (hub_dir / "hub-pins.json").write_text(
        json.dumps(
            {
                "model": {"kind": "model", "revision": "base-sha", "local_dir": str(hub_dir / "model")},
                "gigaam": {"kind": "model", "revision": "asr-sha", "local_dir": str(hub_dir / "gigaam")},
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        output_dir=tmp_path / "run",
        selection_dir=selection_dir,
        data=SimpleNamespace(index_dir=index_dir),
        hub=SimpleNamespace(local_dir=hub_dir),
        lora=SimpleNamespace(enable_lm=True, enable_dit=True, enable_proj=False, r=32, alpha=32, dropout=0.0),
        memorization=SimpleNamespace(
            updates=8,
            learning_rate=0.01,
            batch_size=1,
            accumulation=1,
            seed=23,
            weight_decay=0.0,
            max_grad_norm=1.0,
            loss_weights={"loss/diff": 1.0, "loss/stop": 1.0},
        ),
        generation=SimpleNamespace(cfg_value=4.25, inference_timesteps=23, max_length=999),
        wandb=SimpleNamespace(project="test", mode="online"),
    )
    runtime = FakeRuntime(events)
    run_manager = FakeRunManager()
    asr = FakeASR()
    models = []

    def model_builder(config, stage, adapter_checkpoint=None):
        del config, adapter_checkpoint
        assert stage == 1
        model = TinyModel(events)
        models.append(model)
        return model, TinyAudioVAE(), lambda text: [len(text)]

    def processor_factory(model, audio_vae, runtime):
        del model, audio_vae, runtime
        return lambda batch: {"value": batch["dataset_ids"].float() + 1.0}

    from voxcpm.training.balalaika import memorization as module

    FakeCheckpointManager.instances.clear()
    monkeypatch.setattr(module, "build_model", model_builder)
    monkeypatch.setattr(module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(module, "create_run_manager", lambda **kwargs: run_manager)
    monkeypatch.setattr(module, "_default_batch_processor_factory", processor_factory)
    monkeypatch.setattr(
        module,
        "_default_optimizer_factory",
        lambda parameters, *, lr, weight_decay: torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay),
    )
    monkeypatch.setattr(
        module,
        "_default_scheduler_factory",
        lambda optimizer, *, warmup_steps, total_steps: FakeScheduler(),
    )
    monkeypatch.setattr(module, "_create_asr", lambda config, runtime: asr)
    return SimpleNamespace(
        config=config,
        runtime=runtime,
        selected_ids=selected_ids,
        samples=samples,
        events=events,
        run_manager=run_manager,
        asr=asr,
        models=models,
    )


def test_memorization_repeats_only_four_selected_items_and_stops(mem_fixture):
    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert set(result.seen_sample_ids) == set(mem_fixture.selected_ids)
    assert len(result.seen_sample_ids) == 8
    assert len(result.reference_audio) == len(result.generated_audio) == 4
    assert result.large_training_started is False
    assert len(mem_fixture.run_manager.losses) == 8
    assert mem_fixture.run_manager.finish_calls == 1
    assert result.result_path.name == "memorization-result.json"
    assert set(result.attempted_sample_ids) == set(mem_fixture.selected_ids)
    assert set(result.contributing_sample_ids) == set(mem_fixture.selected_ids)


def test_memorization_batch_order_is_reproducibly_shuffled_by_seed(mem_fixture):
    first = run_memorization(mem_fixture.config, mem_fixture.runtime)
    mem_fixture.config.output_dir = mem_fixture.config.output_dir.parent / "same-seed-run"
    second = run_memorization(mem_fixture.config, mem_fixture.runtime)
    mem_fixture.config.output_dir = mem_fixture.config.output_dir.parent / "different-seed-run"
    mem_fixture.config.memorization.seed += 1
    third = run_memorization(mem_fixture.config, mem_fixture.runtime)

    sequential = tuple(mem_fixture.selected_ids * 2)
    assert first.attempted_sample_ids == second.attempted_sample_ids
    assert first.attempted_sample_ids != sequential
    assert third.attempted_sample_ids != first.attempted_sample_ids


def test_memorization_replays_repeat_stream_after_skipped_optimizer_attempt(mem_fixture):
    mem_fixture.runtime.skipped_attempts = {1}

    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert result.status == "complete"
    assert mem_fixture.runtime.backward_calls == 9
    assert len(mem_fixture.run_manager.losses) == 8


def test_memorization_generates_without_prompt_and_logs_every_pair(mem_fixture):
    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    generation_calls = [event[1] for event in mem_fixture.events if event[0] == "generate"]
    assert len(generation_calls) == 4
    assert all(call["prompt_text"] == "" and call["prompt_wav_path"] == "" for call in generation_calls)
    assert [call["target_text"] for call in generation_calls] == [sample["text"] for sample in mem_fixture.samples]
    assert all(call["cfg_value"] == 4.25 for call in generation_calls)
    assert all(call["inference_timesteps"] == 23 for call in generation_calls)
    assert all(call["max_len"] == 999 and "max_length" not in call for call in generation_calls)
    assert result.fingerprints["generation_fingerprint"] == fingerprint(
        {"cfg_value": 4.25, "inference_timesteps": 23, "max_length": 999}
    )
    assert len(mem_fixture.run_manager.pairs) == 1
    logged_pairs, logged_step = mem_fixture.run_manager.pairs[0]
    assert logged_step == 8
    assert [pair["sample_id"] for pair in logged_pairs] == mem_fixture.selected_ids
    assert [Path(pair["reference_audio"]) for pair in logged_pairs] == list(result.reference_audio)
    assert [Path(pair["generated_audio"]) for pair in logged_pairs] == list(result.generated_audio)
    assert [pair["asr_hypothesis"] for pair in logged_pairs] == ["намеренно неверная диагностика"] * 4
    assert not hasattr(mem_fixture.models[0], "audio_vae")


def test_gigaam_text_diagnostics_never_gate_result(mem_fixture):
    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert result.status == "complete"
    assert result.diagnostics == ("намеренно неверная диагностика",) * 4


def test_gigaam_failure_is_logged_but_never_gates_result(mem_fixture):
    def unavailable(path):
        raise RuntimeError(f"diagnostic unavailable for {Path(path).name}")

    mem_fixture.asr.transcribe = unavailable

    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert result.status == "complete"
    assert all(value.startswith("ERROR: RuntimeError: diagnostic unavailable") for value in result.diagnostics)
    logged_pairs, _ = mem_fixture.run_manager.pairs[0]
    assert [pair["asr_hypothesis"] for pair in logged_pairs] == list(result.diagnostics)


@pytest.mark.parametrize("failure_point", ["constructor", "context"])
def test_gigaam_setup_failure_is_logged_but_never_gates_result(mem_fixture, monkeypatch, failure_point):
    from voxcpm.training.balalaika import memorization as module

    class BrokenContext:
        def __enter__(self):
            raise RuntimeError("diagnostic context unavailable")

        def __exit__(self, *args):
            return None

    def create_asr(config, runtime):
        del config, runtime
        if failure_point == "constructor":
            raise RuntimeError("diagnostic constructor unavailable")
        return BrokenContext()

    monkeypatch.setattr(module, "_create_asr", create_asr)

    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert result.status == "complete"
    assert len([event for event in mem_fixture.events if event[0] == "generate"]) == 4
    assert all(value.startswith("ERROR: RuntimeError: diagnostic") for value in result.diagnostics)


def test_memorization_requires_four_distinct_published_samples(mem_fixture):
    manifest_path = mem_fixture.config.selection_dir / "memorization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["memorization"][-1] = manifest["memorization"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MemorizationError, match="four distinct"):
        run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert mem_fixture.models == []


def test_memorization_rejects_geometry_that_cannot_cover_all_four_before_setup(mem_fixture):
    mem_fixture.config.memorization.updates = 3

    with pytest.raises(MemorizationError, match="geometry.*four selected"):
        run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert mem_fixture.models == []
    assert mem_fixture.run_manager.losses == []


def test_worker_rank_uses_shared_wandb_identity_and_never_generates_or_logs(mem_fixture, monkeypatch):
    from voxcpm.training.balalaika import memorization as module

    class WorkerRunManager:
        def log_train(self, metrics, global_step):
            return None

        def log_memorization(self, pairs, global_step):
            raise AssertionError("worker must not upload memorization pairs")

        def finish(self):
            return None

    result_dir = mem_fixture.config.output_dir / "memorization"
    result_dir.mkdir(parents=True)
    (result_dir / "wandb-run.json").write_text(
        json.dumps({"version": 1, "job_type": "memorization", "run_id": "wandb123", "config_fingerprint": "cfg"}),
        encoding="utf-8",
    )
    mem_fixture.runtime.rank = 1
    mem_fixture.runtime.world_size = 2

    def publish_main_result(gather_call):
        if gather_call != 16:
            return
        generated = []
        for number in range(4):
            path = result_dir / "generated" / f"item-{number:02d}.wav"
            _write_wav(path)
            generated.append({"path": str(path), "sha256": sha256_file(path)})
        pairs = [
            {
                "sample_id": sample_id,
                "text": sample["text"],
                "asr_hypothesis": "worker-shared",
                "reference_audio": sample["wav_path"],
                "generated_audio": generated[number]["path"],
            }
            for number, (sample_id, sample) in enumerate(
                zip(mem_fixture.selected_ids, mem_fixture.samples, strict=True)
            )
        ]
        pair_payload = memorization_pair_payload(
            pairs,
            global_step=8,
            run_id="wandb123",
            mode="online",
        )
        completion = result_dir / "wandb-complete.json"
        completion.write_text(
            json.dumps(
                {
                    "version": 1,
                    "status": "complete",
                    "job_type": "memorization",
                    "run_id": "wandb123",
                    "mode": "online",
                    "global_step": 8,
                    "pair_count": 4,
                    "pair_payload_fingerprint": fingerprint(pair_payload),
                    "client_log_returned": True,
                    "remote_receipt_verified": False,
                    "verification_scope": (
                        "local W&B client log and finish calls returned; "
                        "W&B provides no server-side receipt in this workflow"
                    ),
                }
            ),
            encoding="utf-8",
        )
        selection = mem_fixture.config.selection_dir / "memorization.json"
        references = [{"path": sample["wav_path"], "sha256": sample["wav_sha256"]} for sample in mem_fixture.samples]
        audit_path = result_dir / "batch-audit" / "rank-00001.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        (result_dir / "memorization-result.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "status": "complete",
                    "large_training_started": False,
                    "updates": 8,
                    "seen_sample_ids": audit["attempted_sample_ids"],
                    "attempted_sample_ids": audit["attempted_sample_ids"],
                    "contributing_sample_ids": audit["contributing_sample_ids"],
                    "batch_audits": [{"path": str(audit_path), "sha256": sha256_file(audit_path)}],
                    "wandb_pair_payload": pair_payload,
                    "reference_audio": references,
                    "generated_audio": generated,
                    "diagnostics": ["worker-shared"] * 4,
                    "losses": [1.0] * 8,
                    "checkpoint": str(result_dir / "checkpoints" / "memorization-final"),
                    "selection_manifest": str(selection),
                    "wandb_completion": str(completion),
                    "fingerprints": {
                        "base_revision": "base-sha",
                        "data_fingerprint": "index-sha",
                        "selection_fingerprint": "selection-sha",
                        "lora_fingerprint": "lora-sha",
                        "checkpoint": "checkpoint-sha",
                        "wandb_run_id": "wandb123",
                        "wandb_completion": sha256_file(completion),
                    },
                }
            ),
            encoding="utf-8",
        )

    mem_fixture.runtime.on_gather = publish_main_result
    monkeypatch.setattr(module, "create_run_manager", lambda **kwargs: WorkerRunManager())

    result = run_memorization(mem_fixture.config, mem_fixture.runtime)

    assert result.wandb_run_id == "wandb123"
    assert [event for event in mem_fixture.events if event[0] == "generate"] == []
    assert mem_fixture.runtime.gather_calls == 16


def test_main_rank_train_log_failure_is_published_and_coordinated(mem_fixture):
    mem_fixture.runtime.world_size = 2
    mem_fixture.run_manager.fail_train = True

    with pytest.raises(
        MemorizationError,
        match="memorization train-log failed on rank zero: RuntimeError: injected train log failure",
    ):
        run_memorization(mem_fixture.config, mem_fixture.runtime)

    failure = json.loads((mem_fixture.config.output_dir / "memorization" / "memorization-failed.json").read_text())
    assert failure == {
        "version": 1,
        "status": "failed",
        "phase": "train-log",
        "error": {"type": "RuntimeError", "message": "injected train log failure"},
    }
    assert mem_fixture.runtime.gather_calls == 3


def test_worker_reconstructs_main_train_log_failure_without_final_barrier(mem_fixture, monkeypatch):
    from voxcpm.training.balalaika import memorization as module

    result_dir = mem_fixture.config.output_dir / "memorization"
    result_dir.mkdir(parents=True)
    (result_dir / "wandb-run.json").write_text(
        json.dumps({"version": 1, "job_type": "memorization", "run_id": "wandb123", "config_fingerprint": "cfg"}),
        encoding="utf-8",
    )
    (result_dir / "memorization-failed.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "failed",
                "phase": "train-log",
                "error": {"type": "RuntimeError", "message": "injected train log failure"},
            }
        ),
        encoding="utf-8",
    )
    runtime = FakeRuntime(mem_fixture.events, rank=1, world_size=2, gathered_values=([1, 1], [1, 1], [0, 1]))
    mem_fixture.runtime = runtime
    monkeypatch.setattr(module, "create_run_manager", lambda **kwargs: SimpleNamespace(log_train=lambda *args: None))

    with pytest.raises(
        MemorizationError,
        match="memorization train-log failed on rank zero: RuntimeError: injected train log failure",
    ):
        run_memorization(mem_fixture.config, runtime)

    assert runtime.gather_calls == 3
    assert [event for event in mem_fixture.events if event[0] == "generate"] == []


def test_main_result_failure_is_published_and_coordinated(mem_fixture, monkeypatch):
    from voxcpm.training.balalaika import memorization as module

    write_json = module.atomic_json
    result_path = (mem_fixture.config.output_dir / "memorization" / "memorization-result.json").resolve()

    def fail_result(path, value):
        if Path(path).resolve() == result_path:
            raise OSError("injected result publication failure")
        return write_json(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_result)

    result_failure = (
        "memorization result-publication failed on rank zero: " "OSError: injected result publication failure"
    )
    with pytest.raises(
        MemorizationError,
        match=result_failure,
    ):
        run_memorization(mem_fixture.config, mem_fixture.runtime)

    failure_path = result_path.parent / "memorization-failed.json"
    failure = json.loads(failure_path.read_text())
    assert failure["phase"] == "result-publication"
    assert failure["error"] == {
        "type": "OSError",
        "message": "injected result publication failure",
    }


def test_worker_reconstructs_main_result_failure(mem_fixture, monkeypatch):
    from voxcpm.training.balalaika import memorization as module

    result_dir = mem_fixture.config.output_dir / "memorization"
    result_dir.mkdir(parents=True)
    (result_dir / "wandb-run.json").write_text(
        json.dumps(
            {
                "version": 1,
                "job_type": "memorization",
                "run_id": "wandb123",
                "config_fingerprint": "cfg",
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "memorization-failed.json").write_text(
        json.dumps(
            {
                "version": 1,
                "status": "failed",
                "phase": "result-publication",
                "error": {
                    "type": "OSError",
                    "message": "injected result publication failure",
                },
            }
        ),
        encoding="utf-8",
    )
    successes = [[1, 1]] * 15
    runtime = FakeRuntime(
        mem_fixture.events,
        rank=1,
        world_size=2,
        gathered_values=(*successes, [0, 1]),
    )
    mem_fixture.runtime = runtime
    worker = SimpleNamespace(log_train=lambda *args: None)
    monkeypatch.setattr(module, "create_run_manager", lambda **kwargs: worker)

    result_failure = (
        "memorization result-publication failed on rank zero: " "OSError: injected result publication failure"
    )
    with pytest.raises(
        MemorizationError,
        match=result_failure,
    ):
        run_memorization(mem_fixture.config, runtime)

    assert runtime.gather_calls == 16


@pytest.fixture
def mem_result(mem_fixture):
    return run_memorization(mem_fixture.config, mem_fixture.runtime)


def test_approval_is_bound_to_checkpoint_wandb_and_approver(mem_result):
    path = approve_memorization(mem_result.dir, "wandb123", "operator")
    record = verify_approval(path, mem_result.fingerprints)

    assert record.wandb_run_id == "wandb123"
    assert record.approver == "operator"
    tampered = {**mem_result.fingerprints, "checkpoint": "different"}
    with pytest.raises(ApprovalMismatch, match="checkpoint"):
        verify_approval(path, tampered)


_STAGE_IDENTITY_FIELDS = [
    "base_revision",
    "data_fingerprint",
    "selection_fingerprint",
    "lora_fingerprint",
]


@pytest.mark.parametrize("field", _STAGE_IDENTITY_FIELDS)
def test_approval_rejects_another_stage1_identity(mem_result, field):
    path = approve_memorization(mem_result.dir, "wandb123", "operator")
    expected = {**mem_result.fingerprints, field: "other"}

    with pytest.raises(ApprovalMismatch, match=field):
        verify_approval(path, expected)


@pytest.mark.parametrize("field", _STAGE_IDENTITY_FIELDS)
def test_stage1_rejects_another_identity_before_model_setup(mem_result, field):
    result_dir = mem_result.dir
    approval_path = approve_memorization(result_dir, "wandb123", "operator")
    model_calls = []
    result_fingerprints = mem_result.fingerprints
    current = {
        "base_revision": result_fingerprints["base_revision"],
        "evaluator_revision": "asr-sha",
        "data_fingerprint": result_fingerprints["data_fingerprint"],
        "selection_fingerprint": result_fingerprints["selection_fingerprint"],
        "lora_fingerprint": result_fingerprints["lora_fingerprint"],
        "optimization_fingerprint": "stage-optimization-sha",
        "wandb_run_id": "stage1-run",
        "wandb_group": None,
    }
    current[field] = "other"
    stage_identity_fields = set(_STAGE_IDENTITY_FIELDS)
    approval_expected = dict(mem_result.fingerprints)
    for key in stage_identity_fields:
        approval_expected.pop(key)

    def model_builder(*args, **kwargs):
        model_calls.append((args, kwargs))
        message = "model setup must not run before approval verification"
        raise AssertionError(message)

    trainer = BalalaikaTrainer(
        SimpleNamespace(
            output_dir=mem_result.dir,
            stage1=SimpleNamespace(epochs=2, learning_rate=1e-4, batch_size=1),
            stage2=SimpleNamespace(epochs=3, learning_rate=5e-5, batch_size=1),
        ),
        SimpleNamespace(),
        checkpoint_manager=SimpleNamespace(),
        evaluator_factory=lambda *args: None,
        identity=current,
        approval_verifier=verify_approval,
        approval_path=approval_path,
        approval_expected=approval_expected,
        model_builder=model_builder,
        install_signal_handlers=False,
    )

    with pytest.raises(ApprovalMismatch, match=field):
        trainer.run_stage(1)
    assert model_calls == []


def test_approval_requires_complete_expected_identity(mem_result):
    path = approve_memorization(mem_result.dir, "wandb123", "operator")

    incomplete_error = "expected fingerprints.*incomplete"
    with pytest.raises(ApprovalMismatch, match=incomplete_error):
        verify_approval(path, {})


def test_missing_wandb_completion_cannot_be_approved(mem_result):
    mem_result.wandb_completion_path.unlink()

    with pytest.raises(ApprovalMismatch, match="W&B completion"):
        approve_memorization(mem_result.dir, "wandb123", "operator")


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_approval_rejects_non_online_wandb_completion(mem_result, mode):
    completion_path = mem_result.wandb_completion_path
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["mode"] = mode
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    result = json.loads(mem_result.result_path.read_text(encoding="utf-8"))
    result["fingerprints"]["wandb_completion"] = sha256_file(completion_path)
    mem_result.result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ApprovalMismatch, match="online W&B"):
        approve_memorization(mem_result.dir, "wandb123", "operator")


def test_approval_rejects_unverifiable_wandb_pair_payload(mem_result):
    completion_path = mem_result.wandb_completion_path
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["pair_payload_fingerprint"] = "0" * 64
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    result = json.loads(mem_result.result_path.read_text(encoding="utf-8"))
    result["fingerprints"]["wandb_completion"] = sha256_file(completion_path)
    mem_result.result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ApprovalMismatch, match="four-pair payload"):
        approve_memorization(mem_result.dir, "wandb123", "operator")


def test_approval_rejects_tampered_observed_identity_coverage(mem_result):
    value = json.loads(mem_result.result_path.read_text(encoding="utf-8"))
    value["contributing_sample_ids"] = [value["contributing_sample_ids"][0]]
    value["seen_sample_ids"] = list(value["contributing_sample_ids"])
    mem_result.result_path.write_text(json.dumps(value), encoding="utf-8")

    coverage_error = "observed contribution coverage.*four selected"
    with pytest.raises(ApprovalMismatch, match=coverage_error):
        approve_memorization(mem_result.dir, "wandb123", "operator")


def test_approval_rejects_wrong_run_and_tampered_artifact(mem_result):
    with pytest.raises(ApprovalMismatch, match="W&B run"):
        approve_memorization(mem_result.dir, "another-run", "operator")

    path = approve_memorization(mem_result.dir, "wandb123", "operator")
    mem_result.generated_audio[0].write_bytes(b"tampered")
    with pytest.raises(ApprovalMismatch, match="artifact"):
        verify_approval(path, mem_result.fingerprints)


def test_approval_record_integrity_binds_approver(mem_result):
    path = approve_memorization(mem_result.dir, "wandb123", "operator")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["approver"] = "intruder"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ApprovalMismatch, match="approval fingerprint"):
        verify_approval(path, mem_result.fingerprints)


def test_wandb_uploads_exactly_four_complete_memorization_pairs(tmp_path):
    class Audio:
        def __init__(self, path, *, caption):
            self.path = path
            self.caption = caption

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    class Run:
        def __init__(self):
            self.dir = str(tmp_path / "wandb-run")
            Path(self.dir).mkdir()
            self.logged = []

        def log(self, payload, *, step):
            self.logged.append((payload, step))

        def finish(self):
            return None

    run = Run()

    def initialize(**kwargs):
        return run

    wandb = SimpleNamespace(Audio=Audio, Table=Table, init=initialize)
    manager = WandbRunManager.start(
        "memorization",
        tmp_path / "run-state.json",
        {
            "output_dir": tmp_path,
            "mode": "online",
            "config_fingerprint": "config-sha",
        },
        wandb_module=wandb,
    )
    pairs = []
    for number in range(4):
        reference = tmp_path / f"reference-{number}.wav"
        generated = tmp_path / f"generated-{number}.wav"
        _write_wav(reference)
        _write_wav(generated)
        pairs.append(
            {
                "sample_id": f"sample-{number}",
                "text": f"text {number}",
                "reference_audio": reference,
                "generated_audio": generated,
                "asr_hypothesis": f"hypothesis {number}",
            }
        )

    receipt = manager.log_memorization(pairs, global_step=8)

    payload, step = run.logged[0]
    assert step == 8
    assert payload["memorization/pairs"].columns == [
        "sample_id",
        "text",
        "asr_hypothesis",
        "reference_audio",
        "generated_audio",
    ]
    assert len(payload["memorization/pairs"].data) == 4
    assert len(payload["memorization/reference_audio"]) == 4
    assert len(payload["memorization/generated_audio"]) == 4
    assert receipt["run_id"] == manager.run_id
    assert receipt["mode"] == "online"
    assert receipt["global_step"] == 8
    assert receipt["pair_count"] == 4
    assert len(receipt["pair_payload_fingerprint"]) == 64


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_wandb_memorization_rejects_non_online_modes(tmp_path, mode):
    def no_op(*args, **kwargs):
        return None

    run = SimpleNamespace(log=no_op, finish=no_op)
    wandb = SimpleNamespace(init=lambda **kwargs: run)
    manager = WandbRunManager.start(
        "memorization",
        tmp_path / f"{mode}-state.json",
        {
            "output_dir": tmp_path,
            "mode": mode,
            "config_fingerprint": "config-sha",
        },
        wandb_module=wandb,
    )

    with pytest.raises(RuntimeError, match="online W&B"):
        manager.log_memorization([], global_step=8)
