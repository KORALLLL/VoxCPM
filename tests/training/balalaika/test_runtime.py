from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from voxcpm.training.balalaika.runtime import AccelerateRuntime


class FakeAccelerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_index = 2
        self.num_processes = 8
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.calls = []

    def prepare(self, *objects):
        self.calls.append(("prepare", objects))
        return objects[0] if len(objects) == 1 else objects

    def accumulate(self, *models):
        self.calls.append(("accumulate", models))
        return SimpleNamespace(__enter__=lambda _: None, __exit__=lambda *_: None)

    def backward(self, loss):
        self.calls.append(("backward", loss))

    def clip_grad_norm_(self, parameters, max_norm):
        parameters = tuple(parameters)
        self.calls.append(("clip", parameters, max_norm))
        return torch.tensor(1.25)

    def gather_for_metrics(self, value):
        self.calls.append(("gather", value))
        return value

    def wait_for_everyone(self):
        self.calls.append(("barrier",))

    def unwrap_model(self, model):
        self.calls.append(("unwrap", model))
        return model

    def save_state(self, output_dir):
        self.calls.append(("save", output_dir))

    def load_state(self, input_dir):
        self.calls.append(("load", input_dir))


def runtime_config(accumulation=4):
    return SimpleNamespace(accumulation=accumulation)


def test_accelerate_runtime_configures_bf16_seedable_unsplit_batches_and_ddp():
    runtime = AccelerateRuntime.create(runtime_config(), accelerator_cls=FakeAccelerator)

    kwargs = runtime.accelerator.kwargs
    assert kwargs["mixed_precision"] == "bf16"
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["dataloader_config"].split_batches is False
    assert kwargs["dataloader_config"].even_batches is False
    assert kwargs["dataloader_config"].use_seedable_sampler is True
    assert len(kwargs["kwargs_handlers"]) == 1
    assert kwargs["kwargs_handlers"][0].find_unused_parameters is False
    assert kwargs["kwargs_handlers"][0].broadcast_buffers is False
    assert runtime.rank == 2
    assert runtime.world_size == 8
    assert runtime.device == torch.device("cpu")


def test_runtime_rejects_non_positive_accumulation():
    with pytest.raises(ValueError, match="accumulation"):
        AccelerateRuntime.create(runtime_config(accumulation=0), accelerator_cls=FakeAccelerator)


def test_prepare_accepts_unsharded_loader_and_rejects_distributed_sampler():
    runtime = AccelerateRuntime.create(runtime_config(), accelerator_cls=FakeAccelerator)
    dataset = TensorDataset(torch.arange(8))
    unsharded = DataLoader(dataset, batch_size=2, shuffle=False)

    assert runtime.prepare(unsharded) is unsharded

    pre_sharded = DataLoader(
        dataset,
        batch_size=2,
        sampler=DistributedSampler(dataset, num_replicas=2, rank=0),
    )
    with pytest.raises(ValueError, match="DistributedSampler"):
        runtime.prepare(pre_sharded)


def test_runtime_forwards_only_public_training_and_state_operations(tmp_path):
    runtime = AccelerateRuntime.create(runtime_config(), accelerator_cls=FakeAccelerator)
    model = torch.nn.Linear(1, 1)
    loss = torch.tensor(3.0)
    parameters = tuple(model.parameters())

    runtime.backward(loss)
    assert runtime.clip_grad_norm_(parameters, 0.5).item() == pytest.approx(1.25)
    assert runtime.gather(torch.tensor([7])).tolist() == [7]
    runtime.barrier()
    assert runtime.unwrap(model) is model
    runtime.save(tmp_path / "state")
    runtime.load(tmp_path / "state")

    assert [call[0] for call in runtime.accelerator.calls] == [
        "backward",
        "clip",
        "gather",
        "barrier",
        "unwrap",
        "save",
        "barrier",
        "load",
    ]


def test_save_waits_until_main_process_finishes_shared_checkpoint_files(tmp_path):
    runtime = AccelerateRuntime.create(runtime_config(), accelerator_cls=FakeAccelerator)

    runtime.save(tmp_path / "state")

    assert [call[0] for call in runtime.accelerator.calls] == ["save", "barrier"]
