from __future__ import annotations

import pytest

from voxcpm.training.balalaika.metrics import (
    BenchmarkRow,
    aggregate_scores,
    normalize_ru,
    score_utterance,
)


@pytest.fixture
def benchmark_row() -> BenchmarkRow:
    return BenchmarkRow(
        id=1,
        category="agree",
        text="Долг 12 рублей.",
        normalized_gold="Долг двенадцать рублей.",
        stressed="Долг двена́дцать рублей.",
    )


def test_normalize_ru_shares_case_yo_stress_whitespace_and_punctuation_rules() -> None:
    assert normalize_ru("  Ёлка,\tЕЩЁ!  со́рок—два...\n") == "елка еще сорок два"


@pytest.mark.parametrize("text", ("ё", "Ё", "е\u0308", "Е\u0308"))
def test_normalize_ru_folds_precomposed_and_decomposed_yo_to_e(text: str) -> None:
    assert normalize_ru(text) == "е"


def test_normalize_ru_removes_stress_without_stripping_breve() -> None:
    assert normalize_ru("й и\u0306 и\u0301") == "й й и"


def test_micro_cer_uses_character_denominator_not_word_count() -> None:
    rows = [
        BenchmarkRow(
            id=1,
            category="agree",
            text="Долг 12 рублей.",
            normalized_gold="Долг двенадцать рублей.",
            stressed="",
        ),
        BenchmarkRow(
            id=2,
            category="agree",
            text="Долг 3 рубля.",
            normalized_gold="Долг три рубля.",
            stressed="",
        ),
    ]
    scores = [
        score_utterance(rows[0], "долг двенатцать рублей"),
        score_utterance(rows[1], "долг три рубля"),
    ]

    aggregate = aggregate_scores(scores)

    assert aggregate.utt_char_reference_units == 32
    assert aggregate.utt_word_reference_units == 6
    assert aggregate.utt_cer == pytest.approx(1 / 32)
    assert aggregate.utt_wer == pytest.approx(1 / 6)


def test_empty_hypothesis_is_scored_not_dropped(benchmark_row: BenchmarkRow) -> None:
    score = score_utterance(benchmark_row, "")

    assert score.utt_deletions == 3
    assert score.utt_word_reference_units == 3
    assert score.utt_char_deletions == 20
    assert score.utt_char_reference_units == 20
    assert score.num_deletions == 1
    assert score.num_word_reference_units == 1
    assert score.num_char_deletions == 10
    assert score.num_char_reference_units == 10
    assert score.utt_wer == 1.0
    assert score.utt_cer == 1.0
    assert score.num_wer == 1.0
    assert score.num_cer == 1.0


@pytest.mark.parametrize(
    ("hypothesis", "word_field", "expected"),
    [
        ("долг тринадцать рублей", "utt_substitutions", 1),
        ("долг рублей", "utt_deletions", 1),
        ("долг очень двенадцать рублей", "utt_insertions", 1),
    ],
)
def test_utterance_reports_explicit_word_edit_types(
    benchmark_row: BenchmarkRow,
    hypothesis: str,
    word_field: str,
    expected: int,
) -> None:
    score = score_utterance(benchmark_row, hypothesis)
    assert getattr(score, word_field) == expected


def test_insertions_at_both_number_boundaries_are_number_errors(
    benchmark_row: BenchmarkRow,
) -> None:
    score = score_utterance(benchmark_row, "долг ровно двенадцать всего рублей")

    assert score.gold_number_span == (1, 2)
    assert score.hypothesis_number_span == (1, 4)
    assert score.gold_number_text == "двенадцать"
    assert score.hypothesis_number_text == "ровно двенадцать всего"
    assert score.num_insertions == 2
    assert score.num_word_reference_units == 1
    assert score.num_wer == 2.0


def test_repeated_digit_tokens_use_the_full_monotonic_number_interval() -> None:
    row = BenchmarkRow(
        id=2,
        category="repeat",
        text="Код 7, код 7.",
        normalized_gold="Код семь, код семь.",
        stressed="",
    )

    score = score_utterance(row, "код семь код")

    assert score.gold_number_span == (1, 4)
    assert score.gold_number_text == "семь код семь"
    assert score.hypothesis_number_text == "семь код"
    assert score.num_deletions == 1
    assert score.num_word_reference_units == 3
    assert score.num_wer == pytest.approx(1 / 3)


def test_ambiguous_repeated_token_alignment_is_deterministic() -> None:
    row = BenchmarkRow(
        id=3,
        category="repeat",
        text="7 7",
        normalized_gold="семь семь",
        stressed="",
    )

    first = score_utterance(row, "семь")
    second = score_utterance(row, "семь")

    assert first == second
    assert first.gold_number_span == (0, 2)
    assert first.hypothesis_number_span == (0, 1)
    assert first.num_deletions == 1


def test_decimal_kopecks_span_is_projected_from_all_digit_tokens() -> None:
    row = BenchmarkRow(
        id=4,
        category="decimal",
        text="Цена 12 рублей 50 копеек.",
        normalized_gold="Цена двенадцать рублей пятьдесят копеек.",
        stressed="",
    )

    score = score_utterance(row, "цена двенадцать рублей шестьдесят копеек")

    assert score.gold_number_span == (1, 4)
    assert score.gold_number_text == "двенадцать рублей пятьдесят"
    assert score.num_substitutions == 1
    assert score.num_word_reference_units == 3
    assert score.num_wer == pytest.approx(1 / 3)


def test_date_expansion_maps_one_numeric_region_to_many_gold_words() -> None:
    row = BenchmarkRow(
        id=5,
        category="date",
        text="Дата 12.05.2024 назначена.",
        normalized_gold="Дата двенадцатое мая две тысячи двадцать четвертого назначена.",
        stressed="",
    )

    score = score_utterance(
        row,
        "дата двенадцатое мая две тысячи двадцать четвертого назначена",
    )

    assert score.gold_number_span == (1, 7)
    assert score.gold_number_text == "двенадцатое мая две тысячи двадцать четвертого"
    assert score.num_word_reference_units == 6
    assert score.num_char_reference_units == 41
    assert score.num_wer == 0.0
    assert score.num_cer == 0.0


def test_category_metrics_are_micro_aggregates_with_unequal_lengths() -> None:
    scores = [
        score_utterance(
            BenchmarkRow(1, "short", "Код 2.", "Код два.", ""),
            "код три",
        ),
        score_utterance(
            BenchmarkRow(2, "long", "Теперь код 24 верен.", "Теперь код двадцать четыре верен.", ""),
            "теперь код двадцать пять верен",
        ),
        score_utterance(
            BenchmarkRow(3, "long", "Код 8.", "Код восемь.", ""),
            "код восемь",
        ),
    ]

    aggregate = aggregate_scores(scores)

    assert aggregate.utt_word_errors == 2
    assert aggregate.utt_word_reference_units == 9
    assert aggregate.utt_wer == pytest.approx(2 / 9)
    assert aggregate.category_metrics["short"].utt_wer == pytest.approx(1 / 2)
    assert aggregate.category_metrics["long"].utt_word_errors == 1
    assert aggregate.category_metrics["long"].utt_word_reference_units == 7
    assert aggregate.category_metrics["long"].utt_wer == pytest.approx(1 / 7)


def test_zero_number_and_empty_corpus_denominators_return_zero() -> None:
    no_number = score_utterance(
        BenchmarkRow(6, "plain", "Просто текст.", "Просто текст.", ""),
        "просто текст",
    )

    assert no_number.gold_number_span == (0, 0)
    assert no_number.num_word_reference_units == 0
    assert no_number.num_char_reference_units == 0
    assert no_number.num_wer == 0.0
    assert no_number.num_cer == 0.0

    empty = aggregate_scores([])
    assert empty.num_wer == 0.0
    assert empty.num_cer == 0.0
    assert empty.utt_wer == 0.0
    assert empty.utt_cer == 0.0
    assert empty.category_metrics == {}
