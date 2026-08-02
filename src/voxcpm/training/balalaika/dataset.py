"""Direct random access to Balalaika audio and text indexed inside tar files."""

from __future__ import annotations

from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Sequence

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

from ..data import VoxCPMCollator


class DatasetIntegrityError(RuntimeError):
    """Raised when index offsets or sidecar rows no longer describe the corpus."""


class IndexedBalalaikaDataset(Dataset[dict[str, Any]]):
    """Read one curriculum stage directly from indexed source tar members."""

    def __init__(
        self,
        index_path: str | Path,
        sidecar_path: str | Path,
        *,
        stage: int,
        tokenizer: Callable[[str], Sequence[int]],
        sample_rate: int = 16_000,
        max_open_files: int = 32,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")

        self.index_path = Path(index_path).resolve()
        self.sidecar_path = Path(sidecar_path).resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if not self.sidecar_path.is_file():
            raise FileNotFoundError(self.sidecar_path)

        self.stage = stage
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_open_files = max_open_files
        self._connection: sqlite3.Connection | None = None
        self._connection_pid: int | None = None
        self._file_descriptors: OrderedDict[Path, int] = OrderedDict()
        self._length: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Do not send parent process SQLite connections or descriptors to workers."""
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_connection_pid"] = None
        state["_file_descriptors"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Release this process's local descriptors and SQLite connection."""
        for descriptor in self._file_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._file_descriptors.clear()
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._connection_pid = None

    def __len__(self) -> int:
        if self._length is None:
            row = (
                self._database()
                .execute("SELECT COUNT(*) FROM stage_ordinals WHERE stage = ?", (self.stage,))
                .fetchone()
            )
            self._length = int(row[0])
        return self._length

    def __getitem__(self, ordinal: int) -> dict[str, Any]:
        if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= len(self):
            raise IndexError(f"stage {self.stage} ordinal out of range: {ordinal}")

        row = (
            self._database()
            .execute(
                """
            SELECT samples.source_relative_path, samples.source_tar_path,
                   samples.audio_offset, samples.audio_size,
                   samples.sidecar_offset, samples.sidecar_size
            FROM stage_ordinals
            JOIN samples USING (sample_id)
            WHERE stage_ordinals.stage = ? AND stage_ordinals.ordinal = ?
            """,
                (self.stage, ordinal),
            )
            .fetchone()
        )
        if row is None:
            raise DatasetIntegrityError(f"missing stage ordinal: stage={self.stage}, ordinal={ordinal}")

        audio_bytes = self._read_range(Path(row["source_tar_path"]), row["audio_offset"], row["audio_size"], "audio")
        text = self._read_text(row["source_relative_path"], row["sidecar_offset"], row["sidecar_size"])
        waveform = self._decode_audio(audio_bytes)
        token_ids = self.tokenizer(text)

        return {
            "text_ids": list(token_ids),
            "audio_array": waveform.numpy(),
            "audio_sampling_rate": self.sample_rate,
            "dataset_id": 0,
            "is_prompt": False,
        }

    def _database(self) -> sqlite3.Connection:
        process_id = os.getpid()
        if self._connection is None or self._connection_pid != process_id:
            self.close()
            uri = self.index_path.as_uri() + "?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only=ON")
            self._connection_pid = process_id
        return self._connection

    def _descriptor(self, path: Path) -> int:
        path = path.resolve()
        descriptor = self._file_descriptors.pop(path, None)
        if descriptor is None:
            try:
                descriptor = os.open(path, os.O_RDONLY)
            except OSError as error:
                raise DatasetIntegrityError(f"cannot open indexed file: {path}") from error
            while len(self._file_descriptors) >= self.max_open_files:
                _, old_descriptor = self._file_descriptors.popitem(last=False)
                os.close(old_descriptor)
        self._file_descriptors[path] = descriptor
        return descriptor

    def _read_range(self, path: Path, offset: int, size: int, label: str) -> bytes:
        if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size <= 0:
            raise DatasetIntegrityError(f"invalid {label} byte range")
        try:
            payload = os.pread(self._descriptor(path), size, offset)
        except OSError as error:
            raise DatasetIntegrityError(f"cannot read {label} byte range") from error
        if len(payload) != size:
            raise DatasetIntegrityError(f"truncated {label} byte range")
        return payload

    def _read_text(self, identity: str, offset: int, size: int) -> str:
        payload = self._read_range(self.sidecar_path, offset, size, "sidecar")
        try:
            row = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetIntegrityError("malformed sidecar byte range") from error
        if not isinstance(row, dict) or row.get("source_relative_path") != identity:
            raise DatasetIntegrityError("sidecar identity does not match indexed sample")
        text = row.get("rover_punctuated_accented")
        if not isinstance(text, str) or not text.strip():
            raise DatasetIntegrityError("sidecar text is missing or empty")
        return text

    def _decode_audio(self, payload: bytes) -> torch.Tensor:
        try:
            waveform, source_rate = torchaudio.load(io.BytesIO(payload))
        except Exception as error:
            raise DatasetIntegrityError("cannot decode indexed audio") from error
        if waveform.ndim != 2 or waveform.size(0) == 0:
            raise DatasetIntegrityError("decoded audio is not a channel-first waveform")
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        return waveform.contiguous()


class TruncatedBatchSampler(Sampler[list[int]]):
    """Drop incomplete batches and align the remaining batch count for Accelerate."""

    def __init__(self, sampler: Sampler[int], *, batch_size: int, world_size: int, accumulation: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if accumulation <= 0:
            raise ValueError("accumulation must be positive")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = True
        self.world_size = world_size
        self.accumulation = accumulation
        full_batches = len(sampler) // batch_size
        alignment = world_size * accumulation
        self._batch_count = (full_batches // alignment) * alignment
        self.dropped_samples = len(sampler) - self._batch_count * batch_size

    def __iter__(self) -> Iterator[list[int]]:
        iterator = iter(self.sampler)
        for _ in range(self._batch_count):
            yield [next(iterator) for _ in range(self.batch_size)]

    def __len__(self) -> int:
        return self._batch_count


def build_unsharded_dataloader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
    world_size: int,
    accumulation: int,
) -> DataLoader:
    """Build the deterministic global stream before ``Accelerator`` shards it."""
    if workers < 0:
        raise ValueError("workers must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    batch_sampler = TruncatedBatchSampler(
        sampler,
        batch_size=batch_size,
        world_size=world_size,
        accumulation=accumulation,
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=workers, collate_fn=VoxCPMCollator())
    loader.dropped_samples = batch_sampler.dropped_samples
    return loader
