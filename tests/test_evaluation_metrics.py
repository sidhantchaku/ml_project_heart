"""Tests for evaluation/metrics.py -- retrieval metrics."""
import pytest

from evaluation.metrics import (
    empty_retrieval_rate,
    expected_source_retrieval_rate,
    hit_rate_at_k,
    is_relevant_chunk,
    keyword_coverage,
    mean_retrieval_score,
    normalize_text,
    precision_at_k,
)

CASE_WITH_KEYWORDS = {"expected_keywords": ["smoking", "cardiovascular"], "expected_source_keywords": [], "expected_document_ids": []}
CASE_WITH_SOURCE_KEYWORDS = {"expected_keywords": [], "expected_source_keywords": ["cdc.gov"], "expected_document_ids": []}
CASE_WITH_DOC_IDS = {"expected_keywords": [], "expected_source_keywords": [], "expected_document_ids": ["doc-42"]}


def test_normalize_text_collapses_whitespace_and_lowercases():
    assert normalize_text("  Smoking   AND Heart  ") == "smoking and heart"


def test_is_relevant_chunk_matches_expected_keywords():
    chunk = {"text": "Smoking increases cardiovascular risk.", "source_uri": "https://example.com"}
    assert is_relevant_chunk(chunk, CASE_WITH_KEYWORDS) is True


def test_is_relevant_chunk_no_match():
    chunk = {"text": "Unrelated content about diet only.", "source_uri": "https://example.com"}
    assert is_relevant_chunk(chunk, CASE_WITH_KEYWORDS) is False


def test_is_relevant_chunk_matches_source_keywords():
    chunk = {"text": "irrelevant text", "source_uri": "https://www.cdc.gov/heart"}
    assert is_relevant_chunk(chunk, CASE_WITH_SOURCE_KEYWORDS) is True


def test_is_relevant_chunk_matches_document_id():
    chunk = {"text": "x", "source_uri": "https://x.com", "metadata": {"document_id": "doc-42"}}
    assert is_relevant_chunk(chunk, CASE_WITH_DOC_IDS) is True


def test_is_relevant_chunk_with_no_expectations_is_false():
    case = {"expected_keywords": [], "expected_source_keywords": [], "expected_document_ids": []}
    chunk = {"text": "anything"}
    assert is_relevant_chunk(chunk, case) is False


# --- hit_rate_at_k ---------------------------------------------------------------

def test_hit_rate_at_k_all_hits():
    items = [
        {"case": CASE_WITH_KEYWORDS, "chunks": [{"text": "Smoking and cardiovascular risk."}]},
        {"case": CASE_WITH_KEYWORDS, "chunks": [{"text": "Smoking causes cardiovascular issues."}]},
    ]
    assert hit_rate_at_k(items) == 1.0


def test_hit_rate_at_k_partial_hits():
    items = [
        {"case": CASE_WITH_KEYWORDS, "chunks": [{"text": "Smoking and cardiovascular risk."}]},
        {"case": CASE_WITH_KEYWORDS, "chunks": [{"text": "Nothing related here."}]},
    ]
    assert hit_rate_at_k(items) == 0.5


def test_hit_rate_at_k_empty_input_is_zero():
    assert hit_rate_at_k([]) == 0.0


def test_hit_rate_at_k_no_chunks_is_miss():
    items = [{"case": CASE_WITH_KEYWORDS, "chunks": []}]
    assert hit_rate_at_k(items) == 0.0


# --- precision_at_k ---------------------------------------------------------------

def test_precision_at_k_all_relevant():
    chunks = [{"text": "Smoking and cardiovascular risk."}] * 3
    assert precision_at_k(chunks, CASE_WITH_KEYWORDS, k=3) == 1.0


def test_precision_at_k_no_relevant():
    chunks = [{"text": "unrelated"}] * 3
    assert precision_at_k(chunks, CASE_WITH_KEYWORDS, k=3) == 0.0


def test_precision_at_k_zero_k_returns_zero():
    assert precision_at_k([{"text": "x"}], CASE_WITH_KEYWORDS, k=0) == 0.0


def test_precision_at_k_mixed():
    chunks = [{"text": "Smoking and cardiovascular risk."}, {"text": "unrelated"}]
    assert precision_at_k(chunks, CASE_WITH_KEYWORDS, k=2) == 0.5


# --- expected_source_retrieval_rate -----------------------------------------------

def test_expected_source_retrieval_rate():
    items = [
        {"case": CASE_WITH_SOURCE_KEYWORDS, "chunks": [{"text": "x", "source_uri": "https://cdc.gov/heart"}]},
        {"case": CASE_WITH_SOURCE_KEYWORDS, "chunks": [{"text": "x", "source_uri": "https://other.com"}]},
    ]
    assert expected_source_retrieval_rate(items) == 0.5


def test_expected_source_retrieval_rate_excludes_cases_without_source_expectations():
    case_no_expectation = {"expected_keywords": ["x"], "expected_source_keywords": [], "expected_document_ids": []}
    items = [{"case": case_no_expectation, "chunks": []}]
    assert expected_source_retrieval_rate(items) == 0.0


# --- keyword_coverage --------------------------------------------------------------

def test_keyword_coverage_full():
    assert keyword_coverage("Smoking and cardiovascular risk are related.", ["smoking", "cardiovascular"]) == 1.0


def test_keyword_coverage_partial():
    assert keyword_coverage("Only smoking is mentioned.", ["smoking", "cardiovascular"]) == 0.5


def test_keyword_coverage_no_expected_keywords_is_zero():
    assert keyword_coverage("anything", []) == 0.0


def test_keyword_coverage_none_found():
    assert keyword_coverage("completely unrelated text", ["smoking", "cardiovascular"]) == 0.0


# --- empty_retrieval_rate ------------------------------------------------------------

def test_empty_retrieval_rate():
    items = [
        {"chunks": [{"text": "x"}]},
        {"chunks": []},
        {"chunks": []},
    ]
    assert empty_retrieval_rate(items) == pytest.approx(2 / 3)


def test_empty_retrieval_rate_empty_input():
    assert empty_retrieval_rate([]) == 0.0


# --- mean_retrieval_score ------------------------------------------------------------

def test_mean_retrieval_score_with_scores():
    chunks = [{"score": 0.8}, {"score": 0.6}]
    assert mean_retrieval_score(chunks) == 0.7


def test_mean_retrieval_score_no_scores_returns_none():
    chunks = [{"text": "x"}, {"text": "y"}]
    assert mean_retrieval_score(chunks) is None


def test_mean_retrieval_score_empty_list_returns_none():
    assert mean_retrieval_score([]) is None
