"""Auditable hard-number and full-utterance ASR metrics.

The number reference is not inferred from the ASR output.  A monotonic token
alignment first projects the digit-bearing interval in the benchmark source
onto its normalized spoken gold, then a second alignment projects that fixed
gold interval onto the hypothesis.  Both projected spans are retained on the
item score for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Sequence, TypeVar

_Unit = TypeVar("_Unit")
_STRESS_MARKS = frozenset({"\u0301", "\u0341"})


@dataclass(frozen=True)
class BenchmarkRow:
    """Benchmark fields needed by the pure-Python scorer."""

    id: int
    category: str
    text: str
    normalized_gold: str
    stressed: str


@dataclass(frozen=True)
class _EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    correct: int
    reference_units: int


@dataclass(frozen=True)
class _AlignmentStep:
    """One edge in a monotonic source-to-target edit alignment."""

    kind: str
    source_before: int
    target_before: int
    source_after: int
    target_after: int


@dataclass(frozen=True)
class ItemScore:
    """Item-level spans, SDI counts, denominators, and derived rates."""

    row_id: int
    category: str
    normalized_gold: str
    normalized_hypothesis: str
    gold_number_span: tuple[int, int]
    hypothesis_number_span: tuple[int, int]
    gold_number_text: str
    hypothesis_number_text: str

    num_char_substitutions: int
    num_char_deletions: int
    num_char_insertions: int
    num_char_correct: int
    num_char_reference_units: int
    num_word_substitutions: int
    num_word_deletions: int
    num_word_insertions: int
    num_word_correct: int
    num_word_reference_units: int

    utt_char_substitutions: int
    utt_char_deletions: int
    utt_char_insertions: int
    utt_char_correct: int
    utt_char_reference_units: int
    utt_word_substitutions: int
    utt_word_deletions: int
    utt_word_insertions: int
    utt_word_correct: int
    utt_word_reference_units: int

    @property
    def num_substitutions(self) -> int:
        """Number-span word substitutions (the WER edit unit)."""
        return self.num_word_substitutions

    @property
    def num_deletions(self) -> int:
        """Number-span word deletions (the WER edit unit)."""
        return self.num_word_deletions

    @property
    def num_insertions(self) -> int:
        """Number-span word insertions (the WER edit unit)."""
        return self.num_word_insertions

    @property
    def utt_substitutions(self) -> int:
        """Utterance word substitutions (the WER edit unit)."""
        return self.utt_word_substitutions

    @property
    def utt_deletions(self) -> int:
        """Utterance word deletions (the WER edit unit)."""
        return self.utt_word_deletions

    @property
    def utt_insertions(self) -> int:
        """Utterance word insertions (the WER edit unit)."""
        return self.utt_word_insertions

    @property
    def num_char_errors(self) -> int:
        return self.num_char_substitutions + self.num_char_deletions + self.num_char_insertions

    @property
    def num_word_errors(self) -> int:
        return self.num_word_substitutions + self.num_word_deletions + self.num_word_insertions

    @property
    def utt_char_errors(self) -> int:
        return self.utt_char_substitutions + self.utt_char_deletions + self.utt_char_insertions

    @property
    def utt_word_errors(self) -> int:
        return self.utt_word_substitutions + self.utt_word_deletions + self.utt_word_insertions

    @property
    def num_cer(self) -> float:
        return _rate(self.num_char_errors, self.num_char_reference_units)

    @property
    def num_wer(self) -> float:
        return _rate(self.num_word_errors, self.num_word_reference_units)

    @property
    def utt_cer(self) -> float:
        return _rate(self.utt_char_errors, self.utt_char_reference_units)

    @property
    def utt_wer(self) -> float:
        return _rate(self.utt_word_errors, self.utt_word_reference_units)


@dataclass(frozen=True)
class AggregateScore:
    """Corpus micro-averages computed only from summed counts."""

    item_count: int
    num_char_substitutions: int
    num_char_deletions: int
    num_char_insertions: int
    num_char_correct: int
    num_char_reference_units: int
    num_word_substitutions: int
    num_word_deletions: int
    num_word_insertions: int
    num_word_correct: int
    num_word_reference_units: int
    utt_char_substitutions: int
    utt_char_deletions: int
    utt_char_insertions: int
    utt_char_correct: int
    utt_char_reference_units: int
    utt_word_substitutions: int
    utt_word_deletions: int
    utt_word_insertions: int
    utt_word_correct: int
    utt_word_reference_units: int
    category_metrics: dict[str, "AggregateScore"] = field(default_factory=dict)

    @property
    def num_substitutions(self) -> int:
        return self.num_word_substitutions

    @property
    def num_deletions(self) -> int:
        return self.num_word_deletions

    @property
    def num_insertions(self) -> int:
        return self.num_word_insertions

    @property
    def utt_substitutions(self) -> int:
        return self.utt_word_substitutions

    @property
    def utt_deletions(self) -> int:
        return self.utt_word_deletions

    @property
    def utt_insertions(self) -> int:
        return self.utt_word_insertions

    @property
    def num_char_errors(self) -> int:
        return self.num_char_substitutions + self.num_char_deletions + self.num_char_insertions

    @property
    def num_word_errors(self) -> int:
        return self.num_word_substitutions + self.num_word_deletions + self.num_word_insertions

    @property
    def utt_char_errors(self) -> int:
        return self.utt_char_substitutions + self.utt_char_deletions + self.utt_char_insertions

    @property
    def utt_word_errors(self) -> int:
        return self.utt_word_substitutions + self.utt_word_deletions + self.utt_word_insertions

    @property
    def num_cer(self) -> float:
        return _rate(self.num_char_errors, self.num_char_reference_units)

    @property
    def num_wer(self) -> float:
        return _rate(self.num_word_errors, self.num_word_reference_units)

    @property
    def utt_cer(self) -> float:
        return _rate(self.utt_char_errors, self.utt_char_reference_units)

    @property
    def utt_wer(self) -> float:
        return _rate(self.utt_word_errors, self.utt_word_reference_units)


def normalize_ru(text: str) -> str:
    """Apply the scorer's sole Russian text normalization.

    Case and ``ё`` are folded, acute stress marks are removed without stripping
    meaningful combining marks such as the breve in ``й``, punctuation/symbols
    become token boundaries, and all whitespace is collapsed.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold().replace("ё", "е"))
    without_stress = "".join(character for character in decomposed if character not in _STRESS_MARKS)
    normalized = unicodedata.normalize("NFC", without_stress)
    tokenizable = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character for character in normalized
    )
    return " ".join(tokenizable.split())


def score_utterance(row: BenchmarkRow, hypothesis: str) -> ItemScore:
    """Score one successful ASR result, including an empty hypothesis."""
    source_tokens = normalize_ru(row.text).split()
    gold_text = normalize_ru(row.normalized_gold)
    hypothesis_text = normalize_ru(hypothesis)
    gold_tokens = gold_text.split()
    hypothesis_tokens = hypothesis_text.split()

    digit_positions = [
        index for index, token in enumerate(source_tokens) if any(character.isdigit() for character in token)
    ]
    if digit_positions:
        source_number_span = (digit_positions[0], digit_positions[-1] + 1)
        gold_number_span = _project_interval(
            _alignment(source_tokens, gold_tokens),
            *source_number_span,
            target_length=len(gold_tokens),
        )
        hypothesis_number_span = _project_interval(
            _alignment(gold_tokens, hypothesis_tokens),
            *gold_number_span,
            target_length=len(hypothesis_tokens),
        )
    else:
        gold_number_span = (0, 0)
        hypothesis_number_span = (0, 0)

    gold_number_tokens = gold_tokens[slice(*gold_number_span)]
    hypothesis_number_tokens = hypothesis_tokens[slice(*hypothesis_number_span)]

    num_chars = _edit_counts(tuple("".join(gold_number_tokens)), tuple("".join(hypothesis_number_tokens)))
    num_words = _edit_counts(gold_number_tokens, hypothesis_number_tokens)
    utt_chars = _edit_counts(tuple("".join(gold_tokens)), tuple("".join(hypothesis_tokens)))
    utt_words = _edit_counts(gold_tokens, hypothesis_tokens)

    return ItemScore(
        row_id=row.id,
        category=row.category,
        normalized_gold=gold_text,
        normalized_hypothesis=hypothesis_text,
        gold_number_span=gold_number_span,
        hypothesis_number_span=hypothesis_number_span,
        gold_number_text=" ".join(gold_number_tokens),
        hypothesis_number_text=" ".join(hypothesis_number_tokens),
        **_prefixed_counts("num_char", num_chars),
        **_prefixed_counts("num_word", num_words),
        **_prefixed_counts("utt_char", utt_chars),
        **_prefixed_counts("utt_word", utt_words),
    )


def aggregate_scores(scores: Sequence[ItemScore]) -> AggregateScore:
    """Return exact corpus and category micro-averages from summed SDI counts."""
    materialized = tuple(scores)
    aggregate = _aggregate(materialized)
    categories = sorted({score.category for score in materialized})
    category_metrics = {
        category: _aggregate(tuple(score for score in materialized if score.category == category))
        for category in categories
    }
    return AggregateScore(
        **{name: value for name, value in aggregate.__dict__.items() if name != "category_metrics"},
        category_metrics=category_metrics,
    )


def _aggregate(scores: Sequence[ItemScore]) -> AggregateScore:
    count_fields = (
        "num_char_substitutions",
        "num_char_deletions",
        "num_char_insertions",
        "num_char_correct",
        "num_char_reference_units",
        "num_word_substitutions",
        "num_word_deletions",
        "num_word_insertions",
        "num_word_correct",
        "num_word_reference_units",
        "utt_char_substitutions",
        "utt_char_deletions",
        "utt_char_insertions",
        "utt_char_correct",
        "utt_char_reference_units",
        "utt_word_substitutions",
        "utt_word_deletions",
        "utt_word_insertions",
        "utt_word_correct",
        "utt_word_reference_units",
    )
    return AggregateScore(
        item_count=len(scores),
        **{name: sum(getattr(score, name) for score in scores) for name in count_fields},
    )


def _alignment(source: Sequence[_Unit], target: Sequence[_Unit]) -> tuple[_AlignmentStep, ...]:
    """Return a deterministic minimum-edit alignment, preferring diagonals."""
    rows = len(source) + 1
    columns = len(target) + 1
    distance = [[0] * columns for _ in range(rows)]
    for source_index in range(rows):
        distance[source_index][0] = source_index
    for target_index in range(columns):
        distance[0][target_index] = target_index

    for source_index in range(1, rows):
        for target_index in range(1, columns):
            substitution_cost = source[source_index - 1] != target[target_index - 1]
            distance[source_index][target_index] = min(
                distance[source_index - 1][target_index - 1] + substitution_cost,
                distance[source_index - 1][target_index] + 1,
                distance[source_index][target_index - 1] + 1,
            )

    reversed_steps: list[_AlignmentStep] = []
    source_index = len(source)
    target_index = len(target)
    while source_index or target_index:
        if source_index and target_index:
            substitution_cost = source[source_index - 1] != target[target_index - 1]
            if distance[source_index][target_index] == (
                distance[source_index - 1][target_index - 1] + substitution_cost
            ):
                reversed_steps.append(
                    _AlignmentStep(
                        "substitution" if substitution_cost else "correct",
                        source_index - 1,
                        target_index - 1,
                        source_index,
                        target_index,
                    )
                )
                source_index -= 1
                target_index -= 1
                continue
        if source_index and distance[source_index][target_index] == distance[source_index - 1][target_index] + 1:
            reversed_steps.append(
                _AlignmentStep("deletion", source_index - 1, target_index, source_index, target_index)
            )
            source_index -= 1
            continue
        reversed_steps.append(_AlignmentStep("insertion", source_index, target_index - 1, source_index, target_index))
        target_index -= 1

    return tuple(reversed(reversed_steps))


def _project_interval(
    alignment: Sequence[_AlignmentStep],
    source_start: int,
    source_end: int,
    *,
    target_length: int,
) -> tuple[int, int]:
    """Project a half-open source interval, including insertions at its edges."""
    if source_start == source_end:
        return (0, 0)

    selected_target_indices: list[int] = []
    deletion_boundaries: list[int] = []
    for step in alignment:
        if step.kind == "insertion":
            if source_start <= step.source_before <= source_end:
                selected_target_indices.append(step.target_before)
        elif source_start <= step.source_before < source_end:
            if step.kind == "deletion":
                deletion_boundaries.append(step.target_before)
            else:
                selected_target_indices.append(step.target_before)

    if selected_target_indices:
        return (min(selected_target_indices), min(target_length, max(selected_target_indices) + 1))
    boundary = min(deletion_boundaries, default=target_length)
    return (boundary, boundary)


def _edit_counts(reference: Sequence[_Unit], hypothesis: Sequence[_Unit]) -> _EditCounts:
    substitutions = deletions = insertions = correct = 0
    for step in _alignment(reference, hypothesis):
        if step.kind == "substitution":
            substitutions += 1
        elif step.kind == "deletion":
            deletions += 1
        elif step.kind == "insertion":
            insertions += 1
        else:
            correct += 1
    return _EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        correct=correct,
        reference_units=len(reference),
    )


def _prefixed_counts(prefix: str, counts: _EditCounts) -> dict[str, int]:
    return {
        f"{prefix}_substitutions": counts.substitutions,
        f"{prefix}_deletions": counts.deletions,
        f"{prefix}_insertions": counts.insertions,
        f"{prefix}_correct": counts.correct,
        f"{prefix}_reference_units": counts.reference_units,
    }


def _rate(errors: int, reference_units: int) -> float:
    return errors / reference_units if reference_units else 0.0
