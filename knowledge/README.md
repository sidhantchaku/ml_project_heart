# Knowledge base

There are now two retrieval backends, both grounding the same
`/api/explain-risk` generation step. `services/explanation_service.py`
prefers the local index whenever it has been built (the default, since it's
committed to the repo) and falls back to the Bedrock Knowledge Base path
only if the local index is absent.

## Local backend (default, no AWS required)

`knowledge/corpus/*.md` holds ~10 short, real, citable documents (CDC, AHA,
NHLBI -- see each file's frontmatter for `source_organization`/`source_url`).
`scripts/build_knowledge_index.py` chunks them (paragraph-grouped, ~180 words
per chunk) and builds a scikit-learn `TfidfVectorizer` + chunk metadata list,
saved to `knowledge/index/tfidf_index.joblib`. `tools/local_knowledge_retrieval.py`
loads that artifact once per warm instance and answers retrieval queries with
cosine similarity over TF-IDF vectors.

This is real retrieval-augmented generation -- chunking, vectorization, top-k
similarity search, citation-by-actually-retrieved-chunk -- just not neural
embedding-based retrieval. TF-IDF was chosen over a sentence-embedding model
specifically to avoid a heavy (~500MB+) dependency that would hurt Vercel
cold-start time and deployment size; the tradeoff is that it matches on
lexical/term overlap rather than deeper semantic similarity. Swapping in an
embedding model later means writing a new class with the same `.retrieve(query,
top_k) -> List[RetrievedChunk]` interface -- `services/explanation_service.py`
and `agent/nodes.py` don't need to change.

**Rebuild the index after editing the corpus:**

```bash
python scripts/build_knowledge_index.py
```

Commit the resulting `knowledge/index/tfidf_index.joblib` -- it is small
(a few KB for this corpus) and is what lets retrieval work in production
without a build step or external service.

## Bedrock backend (optional, AWS required)

The rest of this document describes what to put into an Amazon Bedrock
Knowledge Base that `BEDROCK_KNOWLEDGE_BASE_ID` points at, for anyone who
wants the AWS-native retrieval path instead of (or alongside) the local one.
**No Bedrock Knowledge Base has been created or populated as part of this
repository** -- this remains a specification to follow, not a working
content set. `tools/knowledge_retrieval.py` and `services/bedrock_client.py`
retrieve whatever is actually indexed there; they do not fabricate content,
and neither does this document.

## Recommended sources

Use only authoritative, publicly available, general-audience cardiovascular
health education material, such as:

- Government health agencies (e.g. CDC, NIH/NHLBI, UK NHS, WHO)
- Recognized cardiovascular organizations (e.g. American Heart Association)
- Peer-reviewed or institutionally reviewed patient-education resources (e.g.
  major academic medical center patient-education pages)

Do not upload:

- Copyrighted books, journal articles, or paywalled content without a license
- Personal health records or any real patient data
- Content that makes specific diagnostic or treatment claims -- this system
  must never present diagnosis or treatment advice, so the source material it
  draws from shouldn't either
- Fabricated or AI-generated "filler" content presented as authoritative

## Recommended document format

- Plain text or well-structured HTML/Markdown/PDF exported from the source
  above, one topic per document where practical (e.g. "general risk factors",
  "physical activity guidance", "diet and cardiovascular health")
- Keep each document self-contained -- avoid documents that rely on context
  from a different document to make sense once chunked

## Chunking considerations

- Prefer semantic or fixed-size chunking in the 300-500 token range (Bedrock
  Knowledge Bases' default chunking strategy is a reasonable starting point)
- Avoid chunk sizes so small that a single chunk loses the qualifying context
  around a claim (e.g. splitting a caveat/exception onto a separate chunk from
  the claim it modifies)

## Required metadata fields

For every document, retain (via Bedrock's metadata support) at minimum:

| Field | Purpose |
|---|---|
| `title` | Human-readable document title, shown in citations |
| `source_organization` | e.g. "CDC", "American Heart Association" |
| `source_url` | Canonical public URL for the original content |
| `last_reviewed_date` | When the source content was last verified as current |
| `category` | e.g. `risk_factors`, `prevention`, `general_education` |

`tools/knowledge_retrieval.py` reads whatever URI/location Bedrock returns for
each chunk and surfaces it as the citation `uri` -- if a document has no
resolvable URI, it will be cited without one rather than a fabricated link.

## Citation requirements

- Every chunk ingested must be traceable back to a real, resolvable public
  source URL. Do not ingest content whose provenance you can't cite.
- The application only ever cites sources that were actually retrieved for a
  given request (see `services/explanation_service.py`'s citation
  sanitization) -- it will not display a citation for a source that wasn't
  retrieved, even if the model hallucinates one.

## Source-quality expectations

- Prefer primary sources (the agency's own published guidance) over secondary
  summaries.
- Re-review and re-index content periodically -- health guidance changes.
- Do not include content describing specific drugs, dosages, or treatment
  protocols. This system is educational and must not produce medication or
  treatment advice regardless of what's in the Knowledge Base -- but keeping
  such content out of the source material is a second layer of defense.

## Status

As of this writing, no Knowledge Base has been created for this project. Until
one is configured (`BEDROCK_KNOWLEDGE_BASE_ID` set and populated per the
above), `BEDROCK_ENABLED=false` should remain the default, and
`/api/explain-risk` will return predictions with `explanation_available: false`.
