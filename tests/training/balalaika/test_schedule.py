import pytest

from voxcpm.training.balalaika.schedule import EpochGeometry, TrainingProgress


def test_fractional_boundaries_are_eight_unique_and_end_at_epoch():
    geometry = EpochGeometry.from_counts(1_003_000, world_size=8, microbatch=2, accumulation=4)

    assert geometry.validation_steps(epoch=0) == (
        1_959,
        3_918,
        5_877,
        7_836,
        9_795,
        11_754,
        13_713,
        15_671,
    )
    assert len(set(geometry.validation_steps(epoch=0))) == 8
    assert 0 not in geometry.validation_steps(epoch=0)


def test_geometry_records_every_sample_dropped_for_batch_and_accumulation_remainders():
    geometry = EpochGeometry.from_counts(1_003_000, world_size=8, microbatch=2, accumulation=4)

    assert geometry.global_microbatch_size == 16
    assert geometry.available_microsteps_per_epoch == 62_687
    assert geometry.microsteps_per_epoch == 62_684
    assert geometry.optimizer_steps_per_epoch == 15_671
    assert geometry.dropped_accumulation_microsteps == 3
    assert geometry.dropped_unbatched_samples == 8
    assert geometry.dropped_samples == 56


def test_boundaries_are_stage_relative_and_epoch_end_appears_once():
    geometry = EpochGeometry.from_counts(80, world_size=1, microbatch=1, accumulation=1)

    assert geometry.validation_steps(epoch=0) == (10, 20, 30, 40, 50, 60, 70, 80)
    assert geometry.validation_steps(epoch=1) == (90, 100, 110, 120, 130, 140, 150, 160)
    assert len(geometry.validation_steps_for_epochs(2)) == 16
    assert len(geometry.validation_steps_for_epochs(3)) == 24


@pytest.mark.parametrize(
    ("rows", "world_size", "microbatch", "accumulation"),
    [(0, 1, 1, 1), (100, 0, 1, 1), (100, 1, 0, 1), (100, 1, 1, 0)],
)
def test_geometry_rejects_non_positive_counts(rows, world_size, microbatch, accumulation):
    with pytest.raises(ValueError, match="positive"):
        EpochGeometry.from_counts(rows, world_size, microbatch, accumulation)


def test_geometry_rejects_an_incomplete_or_too_short_accumulation_schedule():
    with pytest.raises(ValueError, match="eight unique"):
        EpochGeometry.from_counts(31, world_size=1, microbatch=1, accumulation=4)


def test_progress_emits_boundary_only_after_completed_step_and_once_after_resume():
    geometry = EpochGeometry.from_counts(9, world_size=1, microbatch=1, accumulation=1)
    progress = TrainingProgress(stage="stage1", sampler_seed=17)

    assert progress.complete_optimizer_step(geometry) == ()
    assert progress.optimizer_step == 1
    assert progress.complete_optimizer_step(geometry) == (1,)
    assert progress.boundary == 1

    restored = TrainingProgress(stage="stage1")
    restored.load_state_dict(progress.state_dict())
    assert restored.boundaries_due(geometry) == ()
    assert restored.complete_optimizer_step(geometry) == (2,)


def test_progress_roundtrip_and_explicit_epoch_transition_preserve_global_step():
    geometry = EpochGeometry.from_counts(64, world_size=1, microbatch=1, accumulation=1)
    progress = TrainingProgress(stage="stage1", sampler_seed=44)
    for _ in range(64):
        progress.complete_optimizer_step(geometry)

    assert progress.boundary == 8
    progress.start_next_epoch()
    assert progress.state_dict() == {
        "schema_version": 1,
        "stage": "stage1",
        "epoch": 1,
        "boundary": 0,
        "microstep": 64,
        "optimizer_step": 64,
        "global_step": 64,
        "sampler_seed": 44,
        "sampler_epoch": 1,
    }


def test_stage_reset_is_fresh_but_keeps_cross_stage_global_step():
    progress = TrainingProgress(
        stage="stage1",
        epoch=1,
        boundary=8,
        microstep=100,
        optimizer_step=25,
        global_step=25,
        sampler_seed=11,
        sampler_epoch=1,
    )

    progress.reset_for_stage("stage2", sampler_seed=22)

    assert progress.state_dict() == {
        "schema_version": 1,
        "stage": "stage2",
        "epoch": 0,
        "boundary": 0,
        "microstep": 0,
        "optimizer_step": 0,
        "global_step": 25,
        "sampler_seed": 22,
        "sampler_epoch": 0,
    }
