"""Real, tiny corpus artifacts used by the Balalaika index tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
import tarfile
import wave

import pytest
import zstandard

from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika.config import DataConfig
from voxcpm.training.balalaika.index import BuildExpectations


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def _tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, BytesIO(payload))


@dataclass
class SyntheticCorpus:
    root: Path
    config: DataConfig
    expectations: BuildExpectations
    source_rows: list[dict[str, object]]
    rover_rows: list[dict[str, object]]
    combined_rows: list[dict[str, object]]

    @property
    def rover_archive(self) -> Path:
        return (
            self.root
            / "punctuation_artifacts"
            / "20260729T135419Z"
            / "balalaika-rover-results-20260729T135419Z.tar.zst"
        )

    @property
    def combined_sidecar(self) -> Path:
        return self.root / "combined_sidecars" / "rover-punctuation-stress-v1" / "rover-punctuation-stress.jsonl"

    @property
    def source_tar_paths(self) -> list[Path]:
        return sorted((self.root / "train").glob("shard_*.tar"))

    def remove_combined_row(self, identity: str) -> None:
        self.combined_rows = [row for row in self.combined_rows if row["source_relative_path"] != identity]
        self._write_combined()
        self._refresh_hashes()
        self.expectations = replace(self.expectations, combined_row_count=len(self.combined_rows))

    def add_duplicate_combined_row(self, identity: str) -> None:
        row = next(row for row in self.combined_rows if row["source_relative_path"] == identity)
        self.combined_rows.append(dict(row))
        self._write_combined()

    def remove_source_audio(self, identity: str) -> None:
        row = next(row for row in self.source_rows if row["source_relative_path"] == identity)
        row["include_audio"] = False
        self._write_source_tars()
        self._refresh_hashes()

    def replace_agreement(self, identity: str, agreement: object) -> None:
        row = next(row for row in self.rover_rows if row["source_relative_path"] == identity)
        row["asr_agreement_mean"] = agreement
        self._write_rover()
        self._refresh_hashes()

    def remove_rover_shard(self, shard: str) -> None:
        self.rover_rows = [
            row for row in self.rover_rows if not str(row["source_relative_path"]).startswith(f"{shard}/")
        ]
        self._write_rover()
        self._refresh_hashes()
        self.expectations = replace(self.expectations, rover_row_count=len(self.rover_rows))

    def replace_text(self, identity: str, text: object) -> None:
        row = next(row for row in self.combined_rows if row["source_relative_path"] == identity)
        row["rover_punctuated_accented"] = text
        self._write_combined()
        self._refresh_hashes()

    def _write_source_tars(self) -> None:
        train = self.root / "train"
        train.mkdir(parents=True, exist_ok=True)
        for path in train.glob("shard_*.tar"):
            path.unlink()
        rows_by_shard: dict[str, list[dict[str, object]]] = {}
        for row in self.source_rows:
            rows_by_shard.setdefault(str(row["shard"]), []).append(row)
        for shard, rows in rows_by_shard.items():
            with tarfile.open(train / f"shard_{shard}.tar", "w") as archive:
                for row in rows:
                    identity = str(row["source_relative_path"])
                    _tar_member(
                        archive,
                        identity.removesuffix(".wav") + ".json",
                        json.dumps(
                            {
                                "source_relative_path": identity,
                                "duration": 0.01,
                            },
                            ensure_ascii=False,
                        ).encode(),
                    )
                    if row.get("include_audio", True):
                        _tar_member(archive, identity, _wav_bytes())

    def _write_rover(self) -> None:
        self.rover_archive.parent.mkdir(parents=True, exist_ok=True)
        payload = BytesIO()
        rows_by_shard: dict[str, list[dict[str, object]]] = {}
        for row in self.rover_rows:
            shard = str(row["source_relative_path"]).split("/", maxsplit=1)[0]
            rows_by_shard.setdefault(shard, []).append(row)
        with tarfile.open(fileobj=payload, mode="w") as archive:
            for shard, rows in sorted(rows_by_shard.items()):
                lines = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
                _tar_member(archive, f"rover/shard_{shard}.jsonl", lines)
        self.rover_archive.write_bytes(zstandard.ZstdCompressor().compress(payload.getvalue()))

    def _write_combined(self) -> None:
        self.combined_sidecar.parent.mkdir(parents=True, exist_ok=True)
        self.combined_sidecar.write_bytes(
            b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in self.combined_rows)
        )

    def _refresh_hashes(self) -> None:
        self.expectations = replace(
            self.expectations,
            rover_archive_sha256=sha256_file(self.rover_archive),
            combined_sidecar_sha256=sha256_file(self.combined_sidecar),
            source_tar_sha256={
                path.relative_to(self.root).as_posix(): sha256_file(path) for path in self.source_tar_paths
            },
        )


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> SyntheticCorpus:
    root = tmp_path / "corpus"
    source_rows = [
        {"source_relative_path": "000000/a.wav", "shard": "000000", "include_audio": True},
        {"source_relative_path": "000000/b.wav", "shard": "000000", "include_audio": True},
        {"source_relative_path": "000001/c.wav", "shard": "000001", "include_audio": True},
    ]
    rover_rows = [
        {"source_relative_path": "000000/a.wav", "asr_agreement_mean": 0.949999},
        {"source_relative_path": "000000/b.wav", "asr_agreement_mean": 0.95},
        {"source_relative_path": "000001/c.wav", "asr_agreement_mean": None},
    ]
    combined_rows = [
        {"source_relative_path": "000000/a.wav", "rover_punctuated_accented": "тест один"},
        {"source_relative_path": "000000/b.wav", "rover_punctuated_accented": "тест два"},
        {"source_relative_path": "000001/c.wav", "rover_punctuated_accented": "тест три"},
    ]
    bootstrap = BuildExpectations(
        source_shard_count=2,
        source_row_count=3,
        rover_row_count=3,
        combined_row_count=3,
        rover_archive_sha256="",
        combined_sidecar_sha256="",
        source_tar_sha256={},
    )
    corpus = SyntheticCorpus(
        root=root,
        config=DataConfig(corpus_root=root, index_dir=tmp_path / "prepared"),
        expectations=bootstrap,
        source_rows=source_rows,
        rover_rows=rover_rows,
        combined_rows=combined_rows,
    )
    corpus._write_source_tars()
    corpus._write_rover()
    corpus._write_combined()
    corpus._refresh_hashes()
    return corpus
