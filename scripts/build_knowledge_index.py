"""Builds the local TF-IDF retrieval index from knowledge/corpus/*.md.

Run this whenever a document is added, removed, or edited under
knowledge/corpus/:

    python scripts/build_knowledge_index.py

This produces knowledge/index/tfidf_index.joblib, which is committed to the
repository (it is small -- a handful of KB) so that /api/explain-risk can
retrieve grounding context in production without running this script at
deploy time or depending on any external vector database or embedding API.

Why TF-IDF and not a neural embedding model: this keeps the retrieval path
dependency-free (scikit-learn is already a project dependency for the risk
model) and fast to build/import in a serverless cold start. It is real
lexical-vector retrieval -- chunking, vectorization, cosine similarity, top-k
-- not a fabrication of RAG; it is just not semantic embedding-based
retrieval. See knowledge/README.md for the tradeoffs and how to swap in a
neural embedding backend later without changing the retrieval interface.
"""
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "knowledge" / "corpus"
INDEX_PATH = BASE_DIR / "knowledge" / "index" / "tfidf_index.joblib"

# Chunking: split each document into paragraphs, then greedily group
# consecutive paragraphs up to this word budget per chunk. Keeps a claim and
# any qualifying context (e.g. a caveat sentence) in the same chunk rather
# than splitting mid-thought, per knowledge/README.md's chunking guidance.
_MAX_CHUNK_WORDS = 180

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw_text: str) -> tuple:
    match = _FRONTMATTER_RE.match(raw_text)
    if not match:
        raise ValueError("Document is missing --- frontmatter block.")
    frontmatter_block, body = match.groups()
    metadata: Dict[str, str] = {}
    for line in frontmatter_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def _chunk_body(body: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for paragraph in paragraphs:
        word_count = len(paragraph.split())
        if current and current_words + word_count > _MAX_CHUNK_WORDS:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(paragraph)
        current_words += word_count

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_index() -> Dict[str, Any]:
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    doc_paths = sorted(CORPUS_DIR.glob("*.md"))
    if not doc_paths:
        raise ValueError(f"No .md documents found in {CORPUS_DIR}")

    chunk_texts: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []

    for doc_path in doc_paths:
        metadata, body = _parse_frontmatter(doc_path.read_text(encoding="utf-8"))
        required = {"title", "source_organization", "source_url", "category"}
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"{doc_path.name} is missing required metadata fields: {sorted(missing)}")

        for i, chunk_text in enumerate(_chunk_body(body)):
            chunk_texts.append(chunk_text)
            chunk_metadata.append({
                "text": chunk_text,
                "title": metadata["title"],
                "source_organization": metadata["source_organization"],
                "source_url": metadata["source_url"],
                "category": metadata["category"],
                "document": doc_path.stem,
                "chunk_index": i,
            })

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_matrix = vectorizer.fit_transform(chunk_texts)

    return {
        "vectorizer": vectorizer,
        "chunk_matrix": chunk_matrix,
        "chunk_metadata": chunk_metadata,
    }


def main() -> None:
    index = build_index()
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, INDEX_PATH)
    print(f"Built index: {len(index['chunk_metadata'])} chunks from "
          f"{len(sorted(CORPUS_DIR.glob('*.md')))} documents -> {INDEX_PATH}")


if __name__ == "__main__":
    sys.exit(main())
