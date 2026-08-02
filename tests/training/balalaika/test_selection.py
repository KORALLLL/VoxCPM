"""Deterministic selection manifests built from a tiny local offset index."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tarfile
import wave

import pytest

from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika import selection as selection_module
from voxcpm.training.balalaika.selection import SelectionError, create_selection_manifests


def _wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


@dataclass(frozen=True)
class SelectionFixture:
    index_path: Path
    benchmark_path: Path
    output_dir: Path
    text_by_identity: dict[str, str]

    @property
    def kwargs(self) -> dict[str, Path]:
        return {
            "index_path": self.index_path,
            "benchmark_path": self.benchmark_path,
            "output_dir": self.output_dir,
        }


def _write_selection_fixture(tmp_path: Path, *, row_count: int = 24) -> SelectionFixture:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    tar_path = corpus_dir / "shard_000000.tar"
    sidecar_path = corpus_dir / "combined.jsonl"
    rows: list[tuple[str, float, int, int, int, int, int]] = []
    text_by_identity: dict[str, str] = {}
    with tarfile.open(tar_path, "w") as archive:
        for number in range(row_count):
            identity = f"000000/sample-{number:03d}.wav"
            payload = _wav_bytes()
            member = tarfile.TarInfo(identity)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
            text_by_identity[identity] = f"текст для {number:03d}"
    with tarfile.open(tar_path) as archive, sidecar_path.open("wb") as sidecar:
        for number in range(row_count):
            identity = f"000000/sample-{number:03d}.wav"
            encoded = (
                json.dumps(
                    {
                        "source_relative_path": identity,
                        "rover_punctuated_accented": text_by_identity[identity],
                        "speaker_id": None,
                        "is_single_speaker": None,
                    }, ensure_ascii=False).encode("utf-8")
                + b"\n"
            )
            member = archive.getmember(identity)
            offset = sidecar.tell()
            sidecar.write(encoded)
            agreement = 0.94 if number < row_count // 2 else 0.95
            stage = 1 if number < row_count // 2 else 2
            rows.append((identity, agreement, stage, member.offset_data, member.size, offset, len(encoded)))

    index_path = tmp_path / "selection.sqlite3"
    with sqlite3.connect(index_path) as database:
        database.executescript("""
            CREATE TABLE samples (
                source_relative_path TEXT PRIMARY KEY,
                source_tar_path TEXT NOT NULL,
                audio_offset INTEGER NOT NULL,
                audio_size INTEGER NOT NULL,
                sidecar_offset INTEGER NOT NULL,
                sidecar_size INTEGER NOT NULL,
                agreement REAL,
                stage INTEGER,
                speaker_id TEXT,
                is_single_speaker INTEGER
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        database.executemany(
            """
            INSERT INTO samples (
                source_relative_path, source_tar_path, audio_offset, audio_size,
                sidecar_offset, sidecar_size, agreement, stage, speaker_id, is_single_speaker
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            [(identity, str(tar_path), audio_offset, audio_size, sidecar_offset, sidecar_size, agreement, stage)
             for identity, agreement, stage, audio_offset, audio_size, sidecar_offset, sidecar_size in rows],
        )
        database.execute("INSERT INTO metadata VALUES ('sidecar_path', ?)", (str(sidecar_path),))
        database.execute("INSERT INTO metadata VALUES ('fingerprint', 'synthetic-index')")

    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        "".join(json.dumps({"id": identifier, "stressed": f"число {identifier}"}) + "\n" for identifier in range(2_000)),
        encoding="utf-8",
    )
    return SelectionFixture(index_path, benchmark_path, tmp_path / "selection-artifacts", text_by_identity)


@pytest.fixture
def selection_fixture(tmp_path: Path) -> SelectionFixture:
    return _write_selection_fixture(tmp_path)


def test_selections_are_fixed_and_sample_without_replacement(selection_fixture):
    """Catches unstable selection, duplicate reservoir entries, or wrong stage-2 gate."""
    first = create_selection_manifests(**selection_fixture.kwargs, seed=29)
    second = create_selection_manifests(**selection_fixture.kwargs, seed=29)

    assert first.model_dump() == second.model_dump()
    assert len({item.source_relative_path for item in first.memorization}) == 4
    assert all(item.agreement >= 0.95 and item.stage == 2 for item in first.memorization)
    assert len({item.source_relative_path for item in first.prompts}) == 20
    assert {item.stage for item in first.prompts} == {1, 2}
    assert len(first.benchmark_prompt_by_id) == 2_000
    assert set(first.benchmark_prompt_by_id.values()) <= {item.prompt_id for item in first.prompts}


def test_selection_stores_matching_text_and_extracted_wav_hashes(selection_fixture):
    """Catches writing a prompt WAV with text or hash from a different indexed row."""
    bundle = create_selection_manifests(**selection_fixture.kwargs, seed=29)

    for prompt in bundle.prompts:
        assert prompt.text == selection_fixture.text_by_identity[prompt.source_relative_path]
        assert prompt.wav_path.is_file()
        assert prompt.wav_sha256 == sha256_file(prompt.wav_path)
        with wave.open(str(prompt.wav_path), "rb") as audio:
            assert audio.getframerate() == 16_000
            assert audio.getnframes() == 160

    written = {path.name for path in selection_fixture.output_dir.iterdir() if path.suffix == ".json"}
    assert written == {"memorization.json", "prompts.json", "benchmark-prompts.json", "audio-log-ids.json"}
    for path in selection_fixture.output_dir.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"] == bundle.fingerprint


def test_selection_uses_null_speaker_metadata_and_changes_with_seed(selection_fixture):
    """Catches speaker metadata filtering or a seed that does not affect sampled artifacts."""
    first = create_selection_manifests(**selection_fixture.kwargs, seed=29)
    second = create_selection_manifests(**selection_fixture.kwargs, seed=31)

    assert all(item.stage in {1, 2} for item in first.prompts)
    assert (
        [item.source_relative_path for item in first.memorization],
        [item.source_relative_path for item in first.prompts],
        first.benchmark_prompt_by_id,
    ) != (
        [item.source_relative_path for item in second.memorization],
        [item.source_relative_path for item in second.prompts],
        second.benchmark_prompt_by_id,
    )


def test_selection_assigns_all_benchmarks_and_keeps_four_fixed_log_ids(selection_fixture):
    """Catches partial benchmark assignment or W&B examples that drift between validations."""
    first = create_selection_manifests(**selection_fixture.kwargs, seed=29)
    second = create_selection_manifests(**selection_fixture.kwargs, seed=29)

    assert list(first.benchmark_prompt_by_id) == list(range(2_000))
    assert len(first.audio_log_ids) == 4
    assert len(set(first.audio_log_ids)) == 4
    assert set(first.audio_log_ids) <= set(first.benchmark_prompt_by_id)
    assert first.audio_log_ids == second.audio_log_ids


def test_selection_fails_when_fewer_than_twenty_usable_prompt_rows(tmp_path: Path):
    """Catches silently publishing a validation set too small for the 2,000-item protocol."""
    fixture = _write_selection_fixture(tmp_path, row_count=19)

    with pytest.raises(SelectionError, match="at least 20 usable prompt rows"):
        create_selection_manifests(**fixture.kwargs, seed=29)


def _published_files(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def _assert_complete_published_bundle(output_dir: Path) -> None:
    manifests = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in output_dir.glob("*.json")
    }
    assert set(manifests) == {"memorization.json", "prompts.json", "benchmark-prompts.json", "audio-log-ids.json"}
    assert len({payload["fingerprint"] for payload in manifests.values()}) == 1
    for kind in ("memorization", "prompts"):
        for sample in manifests[f"{kind}.json"][kind]:
            wav_path = Path(sample["wav_path"])
            assert wav_path.is_file()
            assert sample["wav_sha256"] == sha256_file(wav_path)


def test_failed_publication_restores_the_prior_complete_bundle(selection_fixture, monkeypatch):
    """Catches a partial replacement that leaves old manifests pointing at new selected WAVs."""
    create_selection_manifests(**selection_fixture.kwargs, seed=29)
    before = _published_files(selection_fixture.output_dir)
    _assert_complete_published_bundle(selection_fixture.output_dir)
    replace = selection_module.os.replace
    failed = False

    def fail_before_replacing_prompts(source, destination):
        nonlocal failed
        if not failed and Path(destination) == selection_fixture.output_dir / "prompts.json":
            failed = True
            raise OSError("injected manifest publish failure")
        return replace(source, destination)

    monkeypatch.setattr(selection_module.os, "replace", fail_before_replacing_prompts)

    with pytest.raises(OSError, match="injected manifest publish failure"):
        create_selection_manifests(**selection_fixture.kwargs, seed=31)

    assert _published_files(selection_fixture.output_dir) == before
    _assert_complete_published_bundle(selection_fixture.output_dir)
    assert not [path for path in selection_fixture.output_dir.rglob("*") if path.name.startswith(".")]
    assert not list(selection_fixture.output_dir.parent.glob(f".{selection_fixture.output_dir.name}.build-*"))


def test_selection_rejects_a_concurrent_builder(selection_fixture):
    """Catches two builders interleaving writes beneath one published selection directory."""
    with selection_module._SelectionBuildLock(selection_fixture.output_dir):
        with pytest.raises(SelectionError, match="another selection build"):
            create_selection_manifests(**selection_fixture.kwargs, seed=29)
