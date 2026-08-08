"""Pure, testable metric functions for retrieval, citation, and groundedness
evaluation. No AWS calls, no I/O -- every function here takes plain data in
and returns plain data out, so it's easy to unit test and reason about.

IMPORTANT LIMITATION: relevance and groundedness here are decided by keyword/
lexical overlap, not semantic understanding. This is a lightweight proxy for
"does this look related", not a judgement of medical correctness or true
semantic relevance. Treat all scores below as directional signals, not proof.
"""
import re
from typing import Any, Dict, List, Optional

# --- text helpers --------------------------------------------------------------

_GENERIC_DISCLAIMER_MARKERS = [
    "educational", "not a medical diagnosis", "not clinically validated",
    "qualified healthcare professional", "does not replace", "emergency services",
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for",
    "with", "this", "that", "it", "as", "be", "may", "can", "your", "you",
    "these", "those", "was", "were", "will", "should", "could", "about",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _significant_terms(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", normalize_text(text))
    return [w for w in words if len(w) >= 4 and w not in _STOPWORDS]


def is_generic_disclaimer_text(text: str) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in _GENERIC_DISCLAIMER_MARKERS)


# --- relevance judgement (keyword-based proxy) ---------------------------------

def is_relevant_chunk(chunk: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """Decides whether a retrieved chunk counts as "relevant" to a test case.

    This is a lightweight keyword-based proxy, not real semantic relevance
    judgement. Preference order:
      1. expected_document_ids, matched against chunk metadata's document id
         (if present) or the chunk's source_uri
      2. expected_source_keywords, matched against the chunk's source_uri
      3. expected_keywords, matched against the chunk's text
    """
    text = normalize_text(chunk.get("text", ""))
    source_uri = normalize_text(chunk.get("source_uri") or "")
    metadata = chunk.get("metadata") or {}
    doc_id = normalize_text(str(metadata.get("document_id", "")))

    expected_document_ids = case.get("expected_document_ids") or []
    if expected_document_ids:
        return any(
            normalize_text(doc_id_expected) in (doc_id, source_uri)
            for doc_id_expected in expected_document_ids
        )

    expected_source_keywords = case.get("expected_source_keywords") or []
    if expected_source_keywords:
        return any(normalize_text(kw) in source_uri for kw in expected_source_keywords)

    expected_keywords = case.get("expected_keywords") or []
    if expected_keywords:
        return any(normalize_text(kw) in text for kw in expected_keywords)

    return False


# --- retrieval metrics -----------------------------------------------------------

def hit_rate_at_k(cases_with_chunks: List[Dict[str, Any]]) -> float:
    """Fraction of cases with at least one relevant chunk among the retrieved chunks.

    Each item must have "case" (the test case dict) and "chunks" (retrieved chunks).
    """
    if not cases_with_chunks:
        return 0.0
    hits = 0
    for item in cases_with_chunks:
        chunks = item.get("chunks") or []
        if any(is_relevant_chunk(chunk, item["case"]) for chunk in chunks):
            hits += 1
    return hits / len(cases_with_chunks)


def precision_at_k(chunks: List[Dict[str, Any]], case: Dict[str, Any], k: int) -> float:
    """Relevant chunks in the top k divided by k. Returns 0.0 if k <= 0."""
    if k <= 0:
        return 0.0
    top_k_chunks = chunks[:k]
    relevant = sum(1 for c in top_k_chunks if is_relevant_chunk(c, case))
    return relevant / k


def expected_source_retrieval_rate(cases_with_chunks: List[Dict[str, Any]]) -> float:
    """Percentage of cases where at least one retrieved chunk's source matches
    the case's expected_source_keywords (or expected_document_ids). Cases with
    neither field set are excluded from the denominator (nothing to check)."""
    checkable = [
        item for item in cases_with_chunks
        if item["case"].get("expected_source_keywords") or item["case"].get("expected_document_ids")
    ]
    if not checkable:
        return 0.0
    matched = 0
    for item in checkable:
        chunks = item.get("chunks") or []
        if any(is_relevant_chunk(chunk, item["case"]) for chunk in chunks):
            matched += 1
    return matched / len(checkable)


def keyword_coverage(text: str, expected_keywords: List[str]) -> float:
    """Fraction of expected_keywords found (as a substring) in `text`, case-insensitive."""
    if not expected_keywords:
        return 0.0
    normalized = normalize_text(text)
    found = sum(1 for kw in expected_keywords if normalize_text(kw) in normalized)
    return found / len(expected_keywords)


def empty_retrieval_rate(cases_with_chunks: List[Dict[str, Any]]) -> float:
    if not cases_with_chunks:
        return 0.0
    empty = sum(1 for item in cases_with_chunks if not item.get("chunks"))
    return empty / len(cases_with_chunks)


def mean_retrieval_score(chunks: List[Dict[str, Any]]) -> Optional[float]:
    """Mean of chunk scores, only over chunks that actually have a numeric
    score. Returns None if no chunk has a score -- callers must not compare
    this across incompatible retrieval backends."""
    scores = [c["score"] for c in chunks if isinstance(c.get("score"), (int, float))]
    if not scores:
        return None
    return sum(scores) / len(scores)


# --- citation metrics ------------------------------------------------------------

def citation_presence_rate(results: List[List[Dict[str, str]]]) -> float:
    """Fraction of cases with at least one citation."""
    if not results:
        return 0.0
    present = sum(1 for citations in results if citations)
    return present / len(results)


def citation_uri_presence_rate(citations: List[Dict[str, str]]) -> float:
    """Of all citations across a run, fraction that have a non-empty uri."""
    if not citations:
        return 0.0
    with_uri = sum(1 for c in citations if c.get("uri"))
    return with_uri / len(citations)


def _chunk_identifiers(chunks: List[Dict[str, Any]]) -> List[str]:
    return [normalize_text(c.get("source_uri") or "") for c in chunks if c.get("source_uri")]


def citation_source_match_rate(citations: List[Dict[str, str]], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Fraction of citations whose uri matches an actually-retrieved chunk's
    source_uri. A citation is "supported" only if this is true."""
    if not citations:
        return 0.0
    retrieved_uris = set(_chunk_identifiers(retrieved_chunks))
    supported = sum(1 for c in citations if normalize_text(c.get("uri") or "") in retrieved_uris)
    return supported / len(citations)


def duplicate_citation_rate(citations: List[Dict[str, str]]) -> float:
    if not citations:
        return 0.0
    seen = set()
    duplicates = 0
    for c in citations:
        key = (c.get("id"), c.get("uri"))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates / len(citations)


def unsupported_citation_rate(citations: List[Dict[str, str]], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Complement of citation_source_match_rate -- fraction of citations that
    do NOT correspond to any retrieved chunk. A fabricated/unmatched citation
    counts here, never as a valid one."""
    if not citations:
        return 0.0
    return 1.0 - citation_source_match_rate(citations, retrieved_chunks)


# --- groundedness (lexical-overlap proxy) ----------------------------------------

def compute_groundedness(claims: List[str], retrieved_texts: List[str], overlap_threshold: float = 0.25) -> Dict[str, Any]:
    """Lightweight, deterministic groundedness check.

    For each claim (already split into sentence/bullet-sized items by the
    caller), computes what fraction of its significant terms appear
    somewhere in the retrieved context. A claim is "grounded" if that overlap
    meets `overlap_threshold`. Generic disclaimer-style claims are ignored
    entirely (neither grounded nor unsupported).

    LIMITATION: this is lexical overlap, not semantic entailment, and proves
    nothing about factual/medical correctness -- only whether the claim's
    wording resembles retrieved text.
    """
    context_terms = set()
    for text in retrieved_texts:
        context_terms.update(_significant_terms(text))

    grounded = 0
    unsupported = 0
    unsupported_claims: List[str] = []

    for claim in claims:
        if not claim or not claim.strip() or is_generic_disclaimer_text(claim):
            continue
        claim_terms = _significant_terms(claim)
        if not claim_terms:
            continue
        overlap = sum(1 for term in claim_terms if term in context_terms) / len(claim_terms)
        if overlap >= overlap_threshold:
            grounded += 1
        else:
            unsupported += 1
            unsupported_claims.append(claim)

    total = grounded + unsupported
    ratio = (grounded / total) if total else None

    return {
        "grounded_claim_count": grounded,
        "unsupported_claim_count": unsupported,
        "groundedness_ratio": ratio,
        "unsupported_claims": unsupported_claims,
    }
