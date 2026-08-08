"""Tests for tools/knowledge_retrieval.py. The Bedrock client is mocked."""
from unittest.mock import MagicMock

import pytest

from services.bedrock_client import BedrockServiceError
from tools.knowledge_retrieval import (
    KnowledgeRetrievalTool,
    RetrievedChunk,
    deduplicate,
    normalize_result,
)


def test_normalize_result_extracts_s3_source():
    raw = {
        "content": {"text": "Some educational text."},
        "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/doc.pdf"}},
        "score": 0.87,
        "metadata": {"category": "prevention"},
    }
    chunk = normalize_result(raw)
    assert chunk == RetrievedChunk(
        text="Some educational text.",
        source_uri="s3://bucket/doc.pdf",
        score=0.87,
        metadata={"category": "prevention"},
    )


def test_normalize_result_extracts_web_source():
    raw = {
        "content": {"text": "Web content."},
        "location": {"type": "WEB", "webLocation": {"url": "https://example.gov/heart"}},
    }
    chunk = normalize_result(raw)
    assert chunk.source_uri == "https://example.gov/heart"


def test_normalize_result_returns_none_for_empty_text():
    assert normalize_result({"content": {"text": ""}}) is None
    assert normalize_result({"content": {}}) is None
    assert normalize_result({}) is None


def test_normalize_result_missing_location_gives_none_source():
    chunk = normalize_result({"content": {"text": "text"}})
    assert chunk.source_uri is None


def test_deduplicate_removes_exact_duplicates():
    chunks = [
        RetrievedChunk("same text", "uri-1", 0.9, {}),
        RetrievedChunk("same text", "uri-1", 0.5, {}),
        RetrievedChunk("different text", "uri-2", 0.8, {}),
    ]
    result = deduplicate(chunks)
    assert len(result) == 2
    assert result[0].text == "same text"
    assert result[1].text == "different text"


def test_retrieve_returns_empty_list_for_blank_query():
    tool = KnowledgeRetrievalTool(client=MagicMock())
    assert tool.retrieve("") == []
    assert tool.retrieve("   ") == []


def test_retrieve_normalizes_and_dedupes_results():
    mock_client = MagicMock()
    mock_client.retrieve.return_value = [
        {"content": {"text": "chunk A"}, "location": {"type": "S3", "s3Location": {"uri": "s3://a"}}},
        {"content": {"text": "chunk A"}, "location": {"type": "S3", "s3Location": {"uri": "s3://a"}}},
        {"content": {"text": "chunk B"}, "location": {"type": "S3", "s3Location": {"uri": "s3://b"}}},
        {"content": {"text": ""}},  # dropped: empty text
    ]
    tool = KnowledgeRetrievalTool(client=mock_client)
    results = tool.retrieve("query", top_k=3)
    assert len(results) == 2
    assert {r.text for r in results} == {"chunk A", "chunk B"}


def test_retrieve_returns_empty_list_when_no_results():
    mock_client = MagicMock()
    mock_client.retrieve.return_value = []
    tool = KnowledgeRetrievalTool(client=mock_client)
    assert tool.retrieve("query") == []


def test_retrieve_propagates_client_failure():
    mock_client = MagicMock()
    mock_client.retrieve.side_effect = BedrockServiceError("retrieval broke")
    tool = KnowledgeRetrievalTool(client=mock_client)
    with pytest.raises(BedrockServiceError):
        tool.retrieve("query")


def test_retrieve_never_fabricates_source_when_location_absent():
    mock_client = MagicMock()
    mock_client.retrieve.return_value = [{"content": {"text": "unsourced chunk"}}]
    tool = KnowledgeRetrievalTool(client=mock_client)
    results = tool.retrieve("query")
    assert len(results) == 1
    assert results[0].source_uri is None
