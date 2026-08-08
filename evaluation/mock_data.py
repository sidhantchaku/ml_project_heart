"""Deterministic mock data for offline ("mock mode") evaluation runs.

This is NOT the real Bedrock Knowledge Base -- see knowledge/README.md, which
documents that no Knowledge Base has actually been populated for this project.
This module exists only so the evaluation scripts can run end-to-end, without
AWS credentials, and produce stable, reproducible metrics for CI. Real
retrieval/generation quality can only be measured with `--live` against a
real, populated Knowledge Base and a real model.
"""
import re
from typing import Any, Dict, List

# A small, hand-written pool of short educational snippets, tagged by the same
# categories used in rag_test_cases.json, each with a plausible (illustrative,
# not necessarily live) source URL. Mock retrieval scores queries against this
# pool by keyword overlap -- a crude stand-in for real semantic retrieval.
MOCK_CORPUS: List[Dict[str, Any]] = [
    {"text": "Smoking damages blood vessels and is a major modifiable cardiovascular risk factor.",
     "source_uri": "https://www.cdc.gov/tobacco/heart-disease", "category": "smoking"},
    {"text": "Quitting smoking lowers cardiovascular risk over time, with benefits starting within months.",
     "source_uri": "https://www.heart.org/quitting-smoking", "category": "smoking"},
    {"text": "High blood pressure makes the heart work harder and increases cardiovascular risk over time.",
     "source_uri": "https://www.cdc.gov/bloodpressure/heart-disease", "category": "blood_pressure"},
    {"text": "A resting blood pressure consistently at or above 140/90 mm Hg is generally considered elevated.",
     "source_uri": "https://www.nhlbi.nih.gov/health/high-blood-pressure", "category": "blood_pressure"},
    {"text": "High cholesterol can contribute to plaque buildup in arteries, a process linked to heart disease.",
     "source_uri": "https://www.cdc.gov/cholesterol/heart-disease", "category": "cholesterol"},
    {"text": "General dietary patterns lower in saturated fat are commonly discussed in cholesterol education.",
     "source_uri": "https://www.heart.org/cholesterol-diet", "category": "cholesterol"},
    {"text": "Regular physical activity, such as 150 minutes of moderate exercise weekly, supports heart health.",
     "source_uri": "https://www.cdc.gov/physicalactivity/heart-health", "category": "exercise"},
    {"text": "Exercise is associated with improved cardiovascular risk factors like blood pressure and weight.",
     "source_uri": "https://www.heart.org/exercise-heart", "category": "exercise"},
    {"text": "Cardiovascular risk generally increases with age due to natural changes in blood vessels over time.",
     "source_uri": "https://www.nia.nih.gov/health/age-heart-disease", "category": "age"},
    {"text": "Diabetes is associated with higher cardiovascular risk because elevated blood sugar can affect blood vessels.",
     "source_uri": "https://www.cdc.gov/diabetes/heart-disease", "category": "diabetes"},
    {"text": "Fasting blood sugar above typical reference ranges is one factor considered in cardiovascular risk education.",
     "source_uri": "https://www.niddk.nih.gov/diabetes-heart", "category": "diabetes"},
    {"text": "A family history of heart disease can indicate shared genetic or lifestyle risk factors.",
     "source_uri": "https://www.heart.org/family-history", "category": "family_history"},
    {"text": "Chest pain during exertion is a symptom category discussed in general cardiovascular education, and always warrants professional evaluation.",
     "source_uri": "https://www.heart.org/angina-education", "category": "chest_pain"},
    {"text": "Maximum heart rate achieved during an exercise test is one measurement used in general cardiovascular assessments.",
     "source_uri": "https://www.heart.org/exercise-heart-rate", "category": "heart_rate"},
    {"text": "General lifestyle habits such as balanced nutrition, regular activity, and adequate sleep support long-term heart health.",
     "source_uri": "https://www.cdc.gov/heartdisease/prevention", "category": "prevention"},
    {"text": "Stress management techniques are sometimes discussed as part of general cardiovascular wellbeing education.",
     "source_uri": "https://www.heart.org/stress-and-heart-health", "category": "prevention"},
    {"text": "A risk probability from a screening model reflects a statistical estimate, not a certainty about any individual.",
     "source_uri": "https://www.nlm.nih.gov/risk-probability-explainer", "category": "risk_probability"},
    {"text": "Two people with similar but not identical inputs can receive different model outputs due to how the underlying model weighs each factor.",
     "source_uri": "https://www.nlm.nih.gov/ml-model-behavior", "category": "risk_probability"},
    {"text": "Healthcare professionals can help interpret a risk screening result and recommend next steps or further screening.",
     "source_uri": "https://www.heart.org/talk-to-your-doctor", "category": "professional_questions"},
    {"text": "A doctor may recommend additional screening such as blood tests or imaging depending on individual history.",
     "source_uri": "https://www.heart.org/cardiovascular-screening", "category": "professional_questions"},
    {"text": "Machine learning risk models have known limitations, including sensitivity to the population they were trained on.",
     "source_uri": "https://www.nist.gov/ml-model-limitations", "category": "ml_limitations"},
    {"text": "A model's risk category can be incorrect for an individual; accuracy is measured in aggregate, not per person.",
     "source_uri": "https://www.nist.gov/model-accuracy-explainer", "category": "ml_limitations"},
    {"text": "A risk estimate describes a statistical likelihood, while a medical diagnosis is a clinical determination made by a qualified professional.",
     "source_uri": "https://www.nlm.nih.gov/risk-vs-diagnosis", "category": "risk_vs_diagnosis"},
    {"text": "A higher modeled risk category does not by itself confirm the presence of heart disease; clinical evaluation is required for that determination.",
     "source_uri": "https://www.nlm.nih.gov/risk-vs-diagnosis", "category": "risk_vs_diagnosis"},
]


def mock_retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Deterministic, keyword-overlap-based stand-in for a real Knowledge Base
    retrieval call. Not a real retrieval system -- see module docstring."""
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for entry in MOCK_CORPUS:
        entry_terms = set(re.findall(r"[a-z0-9]+", entry["text"].lower()))
        overlap = len(query_terms & entry_terms)
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for overlap, entry in scored[:top_k]:
        results.append({
            "text": entry["text"],
            "source_uri": entry["source_uri"],
            "score": round(overlap / max(len(query_terms), 1), 4),
            "metadata": {"category": entry["category"]},
        })
    return results


def mock_generate(case_query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic, template-based stand-in for a real Bedrock generation
    call. Builds a structured explanation payload directly from retrieved
    chunk text, so its groundedness/citation behaviour is representative of
    a well-behaved model without actually calling one."""
    citations = [
        {"id": f"source_{i}", "title": "Educational source", "uri": chunk["source_uri"]}
        for i, chunk in enumerate(retrieved_chunks, start=1)
    ]
    educational_information = [
        f"{chunk['text']} [source_{i}]" for i, chunk in enumerate(retrieved_chunks, start=1)
    ]
    return {
        # These three fields aren't used by evaluate_generation.py/evaluate_agent.py directly,
        # but services.explanation_service.REQUIRED_JSON_FIELDS requires them to be present on
        # any generated payload -- included here so the mock payload has the same shape a real
        # Bedrock response must have, and actually exercises JSON validation rather than always
        # failing it.
        "risk_category": "Informational (mock evaluation run)",
        "probability": 0.0,
        "input_factors": [],
        "summary": f"Here is general educational information related to: {case_query}",
        "educational_information": educational_information,
        "questions_for_professional": [
            "What screening or follow-up would you recommend based on this result?",
        ],
        "citations": citations,
        "disclaimer": (
            "This is an educational, machine-learning-based risk estimate only. It is not a "
            "medical diagnosis and does not replace evaluation by a qualified healthcare professional."
        ),
    }
