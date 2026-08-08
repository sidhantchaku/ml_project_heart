"""Local, dependency-light Knowledge Base retrieval -- a real (if lexical,
not neural) RAG retrieval path that needs no AWS Bedrock Knowledge Base, no
external vector database, and no embedding API key.

Retrieval here is TF-IDF vectorization + cosine similarity over the chunks
built by scripts/build_knowledge_index.py from knowledge/corpus/*.md. The
index artifact (knowledge/index/tfidf_index.joblib) is committed to the repo,
so this works out of the box on a cold serverless start -- no build step, no
network call.

Exposes the exact same interface as tools/knowledge_retrieval.KnowledgeRetrievalTool
(`.retrieve(query, top_k) -> List[RetrievedChunk]`) and raises the same
exception types from services/bedrock_client.py, so services/explanation_service.py
and agent/nodes.py treat this and the Bedrock-backed tool interchangeably.
"""
import logging
from pathlib import Path
from typing import List, Optional

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from services.bedrock_client import BedrockConfigurationError, BedrockServiceError
from tools.knowledge_retrieval import RetrievedChunk, deduplicate

logger = logging.getLogger("cardiorisk.local_knowledge_retrieval")

_DEFAULT_TOP_K = 3
INDEX_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "index" / "tfidf_index.joblib"


def is_index_available(index_path: Optional[Path] = None) -> bool:
    return (index_path or INDEX_PATH).is_file()


class LocalKnowledgeRetrievalTool:
    """Retrieval-only interface, loaded once and reused across requests
    within a warm serverless instance (see get_default_tool)."""

    def __init__(self, index_path: Optional[Path] = None):
        self._index_path = index_path or INDEX_PATH
        self._vectorizer = None
        self._chunk_matrix = None
        self._chunk_metadata = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self._index_path.is_file():
            self._load_error = "Local knowledge index has not been built."
            return
        try:
            index = joblib.load(self._index_path)
            self._vectorizer = index["vectorizer"]
            self._chunk_matrix = index["chunk_matrix"]
            self._chunk_metadata = index["chunk_metadata"]
        except Exception as exc:  # noqa: BLE001 - defensive; corrupt/missing artifact shouldn't crash the app
            self._vectorizer = self._chunk_matrix = self._chunk_metadata = None
            self._load_error = str(exc)
            logger.warning("local_index_load_failed error_type=%s", type(exc).__name__)

    def is_configured(self) -> bool:
        return self._vectorizer is not None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieves and normalizes the top_k most similar chunks for `query`
        by TF-IDF cosine similarity. Returns [] cleanly for an empty query or
        when no chunk clears a minimal similarity bar -- never fabricates a
        result.
        """
        if not self.is_configured():
            raise BedrockConfigurationError(self._load_error or "Local knowledge index is not configured.")
        if not query or not query.strip():
            return []

        k = top_k or _DEFAULT_TOP_K
        try:
            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._chunk_matrix)[0]
        except Exception as exc:  # noqa: BLE001 - vectorizer/matrix mismatch, corrupt index, etc.
            logger.warning("local_retrieval_failed error_type=%s", type(exc).__name__)
            raise BedrockServiceError(f"Local knowledge retrieval failed: {exc}") from exc

        ranked = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        chunks: List[RetrievedChunk] = []
        for i in ranked[:k]:
            score = float(similarities[i])
            if score <= 0.0:
                continue  # no lexical overlap at all -- not a genuine match, don't cite it
            meta = self._chunk_metadata[i]
            chunks.append(RetrievedChunk(
                text=meta["text"],
                source_uri=meta["source_url"],
                score=score,
                metadata={
                    "title": meta["title"],
                    "source_organization": meta["source_organization"],
                    "category": meta["category"],
                },
            ))

        deduped = deduplicate(chunks)
        logger.info("local_knowledge_retrieval_complete candidate_count=%d returned_count=%d", len(ranked), len(deduped))
        return deduped


_default_tool: Optional[LocalKnowledgeRetrievalTool] = None


def get_default_tool() -> LocalKnowledgeRetrievalTool:
    """Process-wide singleton so the index is loaded from disk once per warm instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = LocalKnowledgeRetrievalTool()
    return _default_tool
