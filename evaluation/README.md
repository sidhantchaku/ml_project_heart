# CardioRisk-AI Evaluation Framework

This directory contains a reproducible evaluation suite for the RAG
(retrieval-augmented generation) and LangGraph agent workflow built in
Phases 3–4. It measures retrieval quality, citation correctness, generation
safety/groundedness, safety-screening accuracy, and end-to-end agent
behaviour -- all without requiring AWS credentials by default.

## Why RAG evaluation is necessary

The explanation layer combines three things that can each silently degrade:
a Knowledge Base that may return irrelevant or no results, a foundation model
that may hallucinate facts or citations, and a safety layer that must catch
unsafe requests without over-blocking legitimate educational ones. Without
measuring these separately, a bug in any one of them can hide behind the
others -- e.g. a model that writes confident-sounding text is not the same
as a model that is grounded in what was actually retrieved.

## Retrieval evaluation vs. generation evaluation

- **Retrieval evaluation** (`evaluate_retrieval.py`) asks: *did the Knowledge
  Base return chunks that are actually related to the query?* It never looks
  at generated text.
- **Generation evaluation** (`evaluate_generation.py`) asks: *given some
  retrieved chunks, did the model produce a safe, well-cited, grounded
  answer?* It assumes retrieval already happened and focuses on the model's
  output.

Keeping them separate means a retrieval problem and a generation problem
never get confused with each other in the results.

## What each metric means

### Retrieval metrics (`evaluate_retrieval.py`, `metrics.py`)
- **Hit Rate @ K** -- fraction of test cases where at least one retrieved
  chunk (of the top K) is judged relevant.
- **Precision @ K** -- relevant chunks in the top K, divided by K.
- **Expected-source retrieval rate** -- fraction of cases (that specified an
  expected source) where a chunk from that source was actually retrieved.
- **Keyword coverage** -- fraction of a case's `expected_keywords` that
  appear somewhere in the retrieved text.
- **Empty retrieval rate** -- fraction of cases that returned zero chunks.
- **Mean retrieval score** -- average of chunks' own relevance scores, only
  computed when the retrieval backend actually returns comparable scores.

**Relevance is decided by keyword/lexical overlap, a lightweight proxy for
real semantic relevance** -- not a judgement made by a human or a second
model. Treat Hit Rate/Precision as directional signals, not ground truth.

### Citation metrics (`metrics.py`)
- **Citation presence rate** -- fraction of cases with at least one citation.
- **Citation URI presence rate** -- fraction of citations that have a uri.
- **Citation-to-source match rate** ("support rate") -- fraction of
  citations whose uri matches an actually-retrieved chunk. This is the
  anti-fabrication check: a citation only counts as valid if it's backed by
  something that was really retrieved.
- **Duplicate citation rate** -- fraction of citations that repeat an
  earlier one in the same response.
- **Unsupported citation rate** -- the complement of the support rate.

### Groundedness (`metrics.py: compute_groundedness`)
Splits generated educational claims (already broken into sentence/bullet
items by the caller), strips generic disclaimer language, and checks how
much each claim's significant vocabulary overlaps with the retrieved
context. Produces a grounded/unsupported claim count and a ratio.

**This is lexical overlap, not semantic entailment, and proves nothing about
factual or medical correctness** -- it only tells you whether the wording of
a claim resembles the retrieved text. A claim can overlap heavily and still
be wrong, or use different words and still be correct.

### Safety metrics (`evaluate_safety.py`)
Classification accuracy, block rate, allowed-pass rate, false-positive/
false-negative rates, category-correctness rate, and unsafe-content leakage
rate, all computed by running `evaluation/safety_test_cases.json` through the
real `agent/safety.py` screening and the real LangGraph workflow (Bedrock
disabled, so no AWS calls) -- see Phase 4.

### Agent regression metrics (`evaluate_agent.py`)
End-to-end checks that adding RAG/LangGraph didn't break anything: successful-
explanation rate, each fallback path's success rate, unsafe-output
interception rate, response-schema validity rate, request-ID presence rate,
and **prediction preservation rate** -- confirming the underlying scikit-learn
prediction is identical with or without the explanation layer.

## Mock mode vs. live AWS mode

**Mock mode (default)** uses:
- A small, hand-written illustrative corpus (`mock_data.py: MOCK_CORPUS`) --
  **not** the real Bedrock Knowledge Base, which (as of this writing, see
  `knowledge/README.md`) has not actually been populated.
- A deterministic, template-based mock generator (`mock_data.py:
  mock_generate`) -- **not** a real foundation model call.

Mock mode makes **zero** AWS calls, is fully deterministic, and is safe for
CI. It measures whether the evaluation framework and the deterministic
safety/citation logic work correctly, not the real-world quality of a live
Knowledge Base or model.

**Live mode (`--live`)** calls real AWS Bedrock:
- Explicit opt-in only; never runs automatically.
- Refuses to run (prints a reason, exits non-zero) if Bedrock isn't enabled
  or the model ID / Knowledge Base ID is missing.
- Prints a cost warning before making any calls.
- Records model ID, Knowledge Base ID, region, and a timestamp in the
  report -- never credentials.

**Cost warning:** `--live` mode invokes real Bedrock retrieval and/or
generation calls, which may incur AWS charges depending on your account and
model pricing. Only run it when you intend to and understand your AWS
billing setup.

## How to add test cases

- RAG cases go in `rag_test_cases.json`: give it an `id`, `category`, `query`,
  and either `expected_keywords`, `expected_source_keywords`, or (if you
  actually know real document IDs) `expected_document_ids`. Keep RAG cases
  educational -- never ask for diagnosis/medication/dosage here.
- Safety cases go in `safety_test_cases.json`: an `id`, `input` (free text),
  `expected_allowed` (bool), and `expected_category` (use `"allowed"` for
  allowed cases, or the matching category from `agent/safety.py` otherwise).
- Both files are validated by `evaluation_models.py` before any script runs
  -- a malformed case fails loudly with a clear message, not silently.

## How to run evaluations

```bash
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_retrieval --live
python -m evaluation.evaluate_retrieval --top-k 5
python -m evaluation.evaluate_generation
python -m evaluation.evaluate_safety
python -m evaluation.evaluate_agent
python -m evaluation.evaluate_agent --live
python -m evaluation.run_all
```

`run_all` is the CI-friendly entry point: it runs all four evaluations in
mock mode, writes `evaluation/results/{retrieval,generation,safety,agent}
_report.json` and `evaluation/results/summary.md`, checks the thresholds in
`thresholds.py`, and exits non-zero if any critical gate fails.

Reports are **not** committed by default -- `evaluation/results/` is in
`.gitignore`. If you want to keep a specific report, copy it out or remove it
from `.gitignore` deliberately.

## How to interpret failures

Every report includes a `failed_cases` list with a `case_id` and a `reason`
string -- read that first. A failure means the deterministic check found a
real discrepancy (e.g. "no expected keywords found", "citation not backed by
a retrieved chunk", "expected blocked, was allowed"). This framework does
not fabricate or adjust results to make thresholds pass; if you see a
failure, investigate it rather than assuming it's noise.

## Limitations

- Keyword/lexical matching is a proxy for relevance and groundedness, not a
  semantic judgement -- it will miss paraphrases and can be fooled by
  coincidental word overlap.
- The optional LLM-as-judge mode (not yet implemented as of this phase) would
  add a second, imperfect model's opinion -- it should never be treated as
  ground truth, and is disabled by default when it exists.
- Mock mode's corpus and generator are illustrative, not the real Knowledge
  Base content or a real foundation model's behaviour.
- Safety screening is deterministic pattern matching (see `agent/safety.py`)
  and is not a complete guarantee -- the safety evaluation run in this phase
  found one real guardrail gap (a medication-request phrasing that slipped
  through), which is documented in the safety report rather than hidden.
