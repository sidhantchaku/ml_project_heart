"""Knowledge Base retrieval tool.

Wraps services/bedrock_client.py's `retrieve()` call and normalizes Bedrock's
raw `retrievalResults` into a small, stable internal structure the rest of the
application (services/explanation_service.py) can rely on. Retrieval logic
lives here; generation logic lives in services/bedrock_client.py /
services/explanation_service.py -- this module never talks to the foundation
model.

This module never fabricates sources: if a chunk has no discoverable URI, the
normalized `source_uri` is None rather than a guessed value, and it is up to
the caller to decide whether an unsourced chunk is citable.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.bedrock_client import BedrockClient, get_default_client

logger = logging.getLogger("cardiorisk.knowledge_retrieval")


@dataclass(frozen=True)
class RetrievedChunk:
    """Normalized representation of a single Knowledge Base retrieval result."""

    text: str
    source_uri: Optional[str]
    score: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_source_uri(location: Dict[str, Any]) -> Optional[str]:
    """Bedrock's `location` field varies by data source type (S3, web,
    Confluence, Salesforce, SharePoint, custom...). Look for a URI under any
    of the known shapes rather than assuming S3.
    """
    if not location:
        return None

    location_type = location.get("type")
    type_key_map = {
        "S3": "s3Location",
        "WEB": "webLocation",
        "CONFLUENCE": "confluenceLocation",
        "SALESFORCE": "salesforceLocation",
        "SHAREPOINT": "sharePointLocation",
        "CUSTOM": "customDocumentLocation",
    }

    key = type_key_map.get(location_type)
    if key and key in location:
        nested = location[key]
        return nested.get("uri") or nested.get("url")

    # Fall back to scanning any nested dict for a uri/url field, in case of a
    # location type not yet in the map above.
    for value in location.values():
        if isinstance(value, dict):
            uri = value.get("uri") or value.get("url")
            if uri:
                return uri
    return None


def normalize_result(raw: Dict[str, Any]) -> Optional[RetrievedChunk]:
    """Converts one raw Bedrock retrievalResults entry into a RetrievedChunk.
    Returns None if the entry has no usable text content.
    """
    text = (raw.get("content") or {}).get("text")
    if not text or not text.strip():
        return None

    return RetrievedChunk(
        text=text.strip(),
        source_uri=_extract_source_uri(raw.get("location") or {}),
        score=raw.get("score"),
        metadata=raw.get("metadata") or {},
    )


def deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Removes chunks with identical (text, source_uri), preserving first occurrence order."""
    seen = set()
    unique: List[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.text, chunk.source_uri)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


class KnowledgeRetrievalTool:
    """Retrieval-only interface used by the explanation service."""

    def __init__(self, client: Optional[BedrockClient] = None):
        self._client = client or get_default_client()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieves and normalizes Knowledge Base chunks for `query`.

        Returns an empty list cleanly if the Knowledge Base has no relevant
        content -- callers must not treat an empty list as an error.
        Raises whatever services.bedrock_client raises (BedrockConfigurationError,
        BedrockAuthError, BedrockThrottledError, BedrockServiceError) on failure.
        """
        if not query or not query.strip():
            return []

        raw_results = self._client.retrieve(query=query, top_k=top_k)
        chunks = [normalize_result(item) for item in raw_results]
        chunks = [chunk for chunk in chunks if chunk is not None]
        deduped = deduplicate(chunks)

        logger.info(
            "knowledge_retrieval_complete raw_count=%d normalized_count=%d",
            len(raw_results), len(deduped),
        )
        return deduped
