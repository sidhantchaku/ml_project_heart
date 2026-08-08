"""Tests for evaluation/metrics.py -- citation metrics."""
import pytest

from evaluation.metrics import (
    citation_presence_rate,
    citation_source_match_rate,
    citation_uri_presence_rate,
    duplicate_citation_rate,
    unsupported_citation_rate,
)

RETRIEVED_CHUNKS = [
    {"text": "a", "source_uri": "https://cdc.gov/heart"},
    {"text": "b", "source_uri": "https://heart.org/exercise"},
]


def test_citation_presence_rate_mixed():
    results = [[{"id": "source_1", "uri": "x"}], [], [{"id": "source_1", "uri": "y"}]]
    assert citation_presence_rate(results) == pytest.approx(2 / 3)


def test_citation_presence_rate_empty_input():
    assert citation_presence_rate([]) == 0.0


def test_citation_uri_presence_rate():
    citations = [{"uri": "https://x.com"}, {"uri": ""}, {"uri": "https://y.com"}]
    assert citation_uri_presence_rate(citations) == pytest.approx(2 / 3)


def test_citation_uri_presence_rate_empty():
    assert citation_uri_presence_rate([]) == 0.0


def test_citation_source_match_rate_all_supported():
    citations = [{"uri": "https://cdc.gov/heart"}, {"uri": "https://heart.org/exercise"}]
    assert citation_source_match_rate(citations, RETRIEVED_CHUNKS) == 1.0


def test_citation_source_match_rate_none_supported():
    citations = [{"uri": "https://fabricated.example/made-up"}]
    assert citation_source_match_rate(citations, RETRIEVED_CHUNKS) == 0.0


def test_citation_source_match_rate_empty_citations():
    assert citation_source_match_rate([], RETRIEVED_CHUNKS) == 0.0


def test_unsupported_citation_rate_is_complement():
    citations = [{"uri": "https://cdc.gov/heart"}, {"uri": "https://fabricated.example"}]
    assert unsupported_citation_rate(citations, RETRIEVED_CHUNKS) == 0.5


def test_unsupported_citation_rate_all_fabricated_is_never_counted_as_valid():
    citations = [{"uri": "https://fabricated.example"}]
    assert unsupported_citation_rate(citations, RETRIEVED_CHUNKS) == 1.0


def test_duplicate_citation_rate_no_duplicates():
    citations = [{"id": "source_1", "uri": "a"}, {"id": "source_2", "uri": "b"}]
    assert duplicate_citation_rate(citations) == 0.0


def test_duplicate_citation_rate_with_duplicates():
    citations = [{"id": "source_1", "uri": "a"}, {"id": "source_1", "uri": "a"}, {"id": "source_2", "uri": "b"}]
    assert duplicate_citation_rate(citations) == pytest.approx(1 / 3)


def test_duplicate_citation_rate_empty():
    assert duplicate_citation_rate([]) == 0.0
