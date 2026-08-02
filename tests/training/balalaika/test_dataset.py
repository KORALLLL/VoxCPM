"""Random-access training samples streamed from the Balalaika offset index."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import struct

import pytest
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from voxcpm.training.balalaika.dataset import (
    DatasetIntegrityError,
    IndexedBalalaikaDataset,
    build_unsharded_dataloader,
)
from voxcpm.training.balalaika import dataset as dataset_module
from voxcpm.training.balalaika.index import build_index
from voxcpm.training.data import HFVoxCPMDataset, VoxCPMCollator


@dataclass(frozen=True)
class SyntheticIndex:
    root: object
    path: object
    sidecar: object


class _ToyDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "text_ids": [index],
            "audio_array": [float(index)],
            "audio_sampling_rate": 16_000,
            "dataset_id": 0,
            "is_prompt": False,
        }


def _constant_tokenizer(_: str) -> list[int]:
    return [1]


@pytest.fixture
def synthetic_index(synthetic_corpus) -> SyntheticIndex:
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)
    return SyntheticIndex(synthetic_corpus.root, audit.index_path, synthetic_corpus.combined_sidecar)


def test_indexed_dataset_reads_audio_and_text_without_extraction(synthetic_index):
    """Catches an implementation that extracts tar members or ignores sidecar offsets."""
    ds = IndexedBalalaikaDataset(
        synthetic_index.path,
        synthetic_index.sidecar,
        stage=1,
        tokenizer=lambda text: [len(text)],
        sample_rate=16_000,
    )

    item = ds[0]

    assert item["text_ids"] == [len("тест один")]
    assert item["audio_sampling_rate"] == 16_000
    assert item["audio_array"].ndim == 1
    assert not list(synthetic_index.root.rglob("extracted-*"))


def test_indexed_dataset_rejects_ordinal_bounds(synthetic_index):
    """Catches SQLite negative-index semantics leaking into the public dataset API."""
    ds = IndexedBalalaikaDataset(synthetic_index.path, synthetic_index.sidecar, stage=1, tokenizer=_constant_tokenizer)

    with pytest.raises(IndexError):
        ds[-1]
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_indexed_dataset_rejects_corrupt_audio_byte_range(synthetic_index):
    """Catches partial tar reads being handed to the decoder as if complete."""
    with sqlite3.connect(synthetic_index.path) as database:
        database.execute("UPDATE samples SET audio_offset = 999999999 WHERE stage = 1")

    ds = IndexedBalalaikaDataset(synthetic_index.path, synthetic_index.sidecar, stage=1, tokenizer=lambda text: [1])

    with pytest.raises(DatasetIntegrityError, match="audio byte range"):
        ds[0]


def test_indexed_dataset_rejects_sidecar_identity_mismatch(synthetic_index):
    """Catches tokenizing a valid line that belongs to another sample."""
    sidecar = synthetic_index.sidecar
    sidecar.write_bytes(sidecar.read_bytes().replace(b"000000/a.wav", b"000000/z.wav", 1))
    ds = IndexedBalalaikaDataset(synthetic_index.path, sidecar, stage=1, tokenizer=lambda text: [1])

    with pytest.raises(DatasetIntegrityError, match="sidecar identity"):
        ds[0]


def test_indexed_dataset_normalizes_equivalent_sidecar_identity(synthetic_index):
    """Catches rejecting a valid sidecar path that uses Task 3's non-canonical spelling."""
    sidecar = synthetic_index.sidecar
    sidecar.write_bytes(sidecar.read_bytes().replace(b"000000/a.wav", b"000000\\\\a.wav", 1))
    with sqlite3.connect(synthetic_index.path) as database:
        database.execute("UPDATE samples SET sidecar_size = sidecar_size + 1 WHERE stage = 1")
    ds = IndexedBalalaikaDataset(synthetic_index.path, sidecar, stage=1, tokenizer=lambda text: [len(text)])

    assert ds[0]["text_ids"] == [len("тест один")]


def test_indexed_dataset_rejects_oversized_range_before_pread(synthetic_index, monkeypatch):
    """Catches a corrupt positive size reaching pread before file-size validation."""
    with sqlite3.connect(synthetic_index.path) as database:
        database.execute("UPDATE samples SET audio_size = ? WHERE stage = 1", (2**63 - 1,))

    def fail_if_read(*_args, **_kwargs):
        raise AssertionError("pread must not run for an oversized range")

    monkeypatch.setattr(dataset_module.os, "pread", fail_if_read)
    ds = IndexedBalalaikaDataset(synthetic_index.path, synthetic_index.sidecar, stage=1, tokenizer=lambda text: [1])

    with pytest.raises(DatasetIntegrityError, match="audio byte range"):
        ds[0]


def test_indexed_dataset_downmixes_and_resamples_audio(synthetic_index):
    """Catches returning a source sample rate or non-mono waveform to VoxCPM."""
    with sqlite3.connect(synthetic_index.path) as database:
        tar_path, audio_offset = database.execute(
            "SELECT source_tar_path, audio_offset FROM samples WHERE stage = 1"
        ).fetchone()
    with open(tar_path, "r+b") as source:
        source.seek(audio_offset + 24)
        source.write(struct.pack("<I", 8_000))
        source.seek(audio_offset + 28)
        source.write(struct.pack("<I", 16_000))

    ds = IndexedBalalaikaDataset(synthetic_index.path, synthetic_index.sidecar, stage=1, tokenizer=lambda text: [1])
    item = ds[0]

    assert item["audio_sampling_rate"] == 16_000
    assert item["audio_array"].shape == (320,)


def test_indexed_dataset_reconnects_safely_in_dataloader_workers(synthetic_index):
    """Catches workers inheriting a parent SQLite connection after a fork."""
    ds = IndexedBalalaikaDataset(synthetic_index.path, synthetic_index.sidecar, stage=1, tokenizer=_constant_tokenizer)
    ds[0]
    loader = build_unsharded_dataloader(ds, batch_size=1, workers=1, seed=13, world_size=1, accumulation=1)

    batch = next(iter(loader))

    assert batch["text_tokens"].tolist() == [[1]]


def test_indexed_dataset_evicts_old_tar_descriptors(synthetic_corpus):
    """Catches unbounded file-descriptor growth while accessing many source tars."""
    synthetic_corpus.replace_agreement("000001/c.wav", 0.10)
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)
    ds = IndexedBalalaikaDataset(
        audit.index_path,
        synthetic_corpus.combined_sidecar,
        stage=1,
        tokenizer=lambda text: [1],
        max_open_files=2,
    )

    ds[0]
    ds[1]

    assert len(ds._file_descriptors) == 2
    assert synthetic_corpus.source_tar_paths[0] not in ds._file_descriptors


def test_public_collator_matches_hf_dataset_compatibility_alias():
    """Catches the collator extraction changing established HFVoxCPMDataset batches."""
    batch = [
        {"text_ids": [1], "audio_array": [0.5], "dataset_id": 2, "is_prompt": True},
        {"text_ids": [3, 4], "audio_array": [0.25, 0.75], "dataset_id": 7, "is_prompt": False},
    ]

    new = VoxCPMCollator()(batch)
    old = HFVoxCPMDataset.collate_fn(batch)

    assert new.keys() == old.keys()
    for key in ("text_tokens", "audio_tokens", "task_ids", "dataset_ids"):
        assert torch.equal(new[key], old[key])
    assert new["is_prompts"] == old["is_prompts"]


def test_unsharded_loader_has_no_distributed_sampler():
    """Catches pre-sharding before Accelerate can own distributed data partitioning."""
    dataset = _ToyDataset(130)
    loader = build_unsharded_dataloader(dataset, batch_size=2, workers=0, seed=13, world_size=8, accumulation=4)

    assert not isinstance(loader.sampler, DistributedSampler)
    assert loader.batch_sampler.drop_last is True
    assert len(loader.batch_sampler) % (8 * 4) == 0
    assert len(loader.batch_sampler) == 64
    assert loader.dropped_samples == 2


def test_unsharded_loader_is_deterministic_for_the_same_seed():
    """Catches non-deterministic ordering before Accelerator receives the batch stream."""
    first = build_unsharded_dataloader(_ToyDataset(128), batch_size=2, workers=0, seed=13, world_size=8, accumulation=4)
    second = build_unsharded_dataloader(
        _ToyDataset(128), batch_size=2, workers=0, seed=13, world_size=8, accumulation=4
    )

    assert list(first.batch_sampler) == list(second.batch_sampler)


def test_unsharded_loader_reports_drop_last_remainder():
    """Catches the final incomplete local batch being omitted from drop accounting."""
    loader = build_unsharded_dataloader(_ToyDataset(5), batch_size=2, workers=0, seed=13, world_size=1, accumulation=1)

    assert len(loader.batch_sampler) == 2
    assert loader.dropped_samples == 1
