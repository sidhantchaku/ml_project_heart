"""Tests for evaluation/metrics.py -- compute_groundedness (lexical-overlap proxy)."""
from evaluation.metrics import compute_groundedness

RETRIEVED_TEXTS = [
    "Smoking damages blood vessels and increases cardiovascular risk over time.",
    "Regular physical activity supports long-term heart health and lowers risk factors.",
]


def test_grounded_claim_matches_retrieved_context():
    claims = ["Smoking increases cardiovascular risk over time."]
    result = compute_groundedness(claims, RETRIEVED_TEXTS)
    assert result["grounded_claim_count"] == 1
    assert result["unsupported_claim_count"] == 0
    assert result["groundedness_ratio"] == 1.0


def test_unsupported_claim_has_no_overlap():
    claims = ["Purple elephants dance under the moonlight every evening."]
    result = compute_groundedness(claims, RETRIEVED_TEXTS)
    assert result["grounded_claim_count"] == 0
    assert result["unsupported_claim_count"] == 1
    assert result["groundedness_ratio"] == 0.0
    assert "Purple elephants" in result["unsupported_claims"][0]


def test_mixed_claims_produce_partial_ratio():
    claims = [
        "Smoking increases cardiovascular risk over time.",
        "Purple elephants dance under the moonlight every evening.",
    ]
    result = compute_groundedness(claims, RETRIEVED_TEXTS)
    assert result["grounded_claim_count"] == 1
    assert result["unsupported_claim_count"] == 1
    assert result["groundedness_ratio"] == 0.5


def test_generic_disclaimer_text_is_ignored_not_counted_either_way():
    claims = [
        "This is an educational, machine-learning-based risk estimate only, "
        "not a medical diagnosis, consult a qualified healthcare professional."
    ]
    result = compute_groundedness(claims, RETRIEVED_TEXTS)
    assert result["grounded_claim_count"] == 0
    assert result["unsupported_claim_count"] == 0
    assert result["groundedness_ratio"] is None


def test_empty_claims_list_returns_none_ratio():
    result = compute_groundedness([], RETRIEVED_TEXTS)
    assert result["groundedness_ratio"] is None
    assert result["grounded_claim_count"] == 0
    assert result["unsupported_claim_count"] == 0


def test_blank_claims_are_skipped():
    result = compute_groundedness(["", "   "], RETRIEVED_TEXTS)
    assert result["groundedness_ratio"] is None


def test_empty_retrieved_context_makes_claims_unsupported():
    claims = ["Smoking increases cardiovascular risk over time."]
    result = compute_groundedness(claims, [])
    assert result["unsupported_claim_count"] == 1
    assert result["groundedness_ratio"] == 0.0


def test_overlap_threshold_is_configurable():
    # "Smoking is dangerous" has two significant terms (smoking, dangerous);
    # only "smoking" overlaps with the retrieved context -> overlap ratio 0.5.
    claims = ["Smoking is dangerous."]
    lenient = compute_groundedness(claims, RETRIEVED_TEXTS, overlap_threshold=0.1)
    strict = compute_groundedness(claims, RETRIEVED_TEXTS, overlap_threshold=0.9)
    assert lenient["grounded_claim_count"] == 1
    assert strict["unsupported_claim_count"] == 1
