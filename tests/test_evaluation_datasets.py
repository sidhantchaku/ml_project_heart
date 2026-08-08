"""Tests for evaluation/evaluation_models.py dataset loading/validation."""
import json

import pytest

from evaluation.evaluation_models import (
    DatasetValidationError,
    load_rag_test_cases,
    load_safety_test_cases,
)

RAG_PATH = "evaluation/rag_test_cases.json"
SAFETY_PATH = "evaluation/safety_test_cases.json"


def test_rag_test_cases_load_successfully():
    cases = load_rag_test_cases(RAG_PATH)
    assert 20 <= len(cases) <= 30
    assert all(c.query for c in cases)
    assert all(c.expected_keywords for c in cases)


def test_safety_test_cases_load_successfully():
    cases = load_safety_test_cases(SAFETY_PATH)
    assert 15 <= len(cases) <= 25
    assert any(c.expected_allowed for c in cases)
    assert any(not c.expected_allowed for c in cases)


def test_rag_test_case_ids_are_unique():
    cases = load_rag_test_cases(RAG_PATH)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_safety_test_case_ids_are_unique():
    cases = load_safety_test_cases(SAFETY_PATH)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_malformed_rag_dataset_is_rejected(tmp_path):
    bad_file = tmp_path / "bad_rag.json"
    bad_file.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")  # missing required fields
    with pytest.raises(DatasetValidationError):
        load_rag_test_cases(str(bad_file))


def test_malformed_safety_dataset_is_rejected(tmp_path):
    bad_file = tmp_path / "bad_safety.json"
    bad_file.write_text(json.dumps([{"id": "x", "input": "hi"}]), encoding="utf-8")  # missing expected_* fields
    with pytest.raises(DatasetValidationError):
        load_safety_test_cases(str(bad_file))


def test_non_list_dataset_is_rejected(tmp_path):
    bad_file = tmp_path / "not_a_list.json"
    bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_rag_test_cases(str(bad_file))


def test_duplicate_ids_are_rejected(tmp_path):
    bad_file = tmp_path / "dupes.json"
    bad_file.write_text(json.dumps([
        {"id": "dup", "input": "a", "expected_allowed": True, "expected_category": "allowed"},
        {"id": "dup", "input": "b", "expected_allowed": True, "expected_category": "allowed"},
    ]), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_safety_test_cases(str(bad_file))


def test_unparseable_json_is_rejected(tmp_path):
    bad_file = tmp_path / "broken.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_rag_test_cases(str(bad_file))


def test_missing_file_is_rejected():
    with pytest.raises(DatasetValidationError):
        load_rag_test_cases("evaluation/does_not_exist.json")
