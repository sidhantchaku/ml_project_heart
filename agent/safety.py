"""Deterministic safety screening for /api/explain-risk.

This is a guardrail based on documented pattern matching, not a claim of
complete medical safety or a substitute for a real content-safety system.
Two responsibilities live here:

1. `screen_request` -- blocks free-text `user_message` input that asks for a
   diagnosis, medication, dosage, treatment plan, or emergency triage
   decision, or that attempts to override these system instructions.
2. `validate_generated_output` -- checks the model's generated explanation
   for the same categories of unsafe content slipping through, plus
   structural guardrails (missing disclaimer, citations claimed without
   retrieved context).

Patterns are deliberately narrow and documented so an educational question
like "explain what this result means" or "what questions should I ask a
doctor" is allowed through.
"""
import re
from typing import Any, Dict, List, Optional, TypedDict

# --- Input screening patterns -------------------------------------------------
# Order matters: checked top to bottom, first match wins. Emergency language is
# checked first because it should never be reclassified as an ordinary
# diagnosis/treatment request.

class SafetyResult(TypedDict):
    allowed: bool
    category: Optional[str]
    reason: Optional[str]


_EMERGENCY_PATTERNS = [
    r"\bam i having a heart attack\b",
    r"\bis this a heart attack\b",
    r"\bcan'?t breathe\b",
    r"\bcannot breathe\b",
    r"\bcalling (911|999|112)\b",
    r"\bshould i call (911|999|112|an ambulance|the ambulance)\b",
    r"\bshould i go to (the )?(er|emergency room|hospital)\b",
    r"\bthis is an emergency\b",
    r"\bsevere chest pain\b",
    r"\bi think i'?m dying\b",
]

_DIAGNOSIS_PATTERNS = [
    r"\bdo i have (heart disease|cvd|cardiovascular disease|a heart condition)\b",
    r"\bdiagnose me\b",
    r"\bwhat disease do i have\b",
    r"\bconfirm (that )?i have\b",
    r"\bam i diagnosed with\b",
]

_MEDICATION_PATTERNS = [
    r"\bwhat medication\b",
    r"\bwhich (drug|medicine|pill)s? should i (take|use)\b",
    r"\bprescribe\b",
    r"\brecommend (a |an )?(medication|drug|medicine|pill)\b",
    r"\bshould i take (aspirin|statins?|a beta.?blocker|metoprolol|lisinopril|atorvastatin|warfarin|nitroglycerin)\b",
]

_DOSAGE_PATTERNS = [
    r"\bhow (much|many) (mg|milligrams?|dose|dosage)\b",
    r"\bwhat dosage\b",
    r"\bhow many pills\b",
    r"\bmg (should|do) i take\b",
]

_TREATMENT_PATTERNS = [
    r"\btreatment plan\b",
    r"\bhow (should|do) i treat\b",
    r"\bwhat treatment\b",
    r"\bshould i (get|have) (surgery|a stent|bypass surgery|an angioplasty)\b",
    r"\bstart treatment\b",
]

_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all |the )?(previous|prior|above) instructions\b",
    r"\bdisregard (the )?(rules|guidelines|instructions|disclaimer)\b",
    r"\byou are now\b",
    r"\bact as (a |an )?(doctor|physician|clinician)\b",
    r"\breveal your (system prompt|instructions)\b",
    r"\bpretend (you are|to be)\b",
    r"\bjailbreak\b",
    r"\bsystem prompt\b",
]

_CERTAINTY_OVERRIDE_PATTERNS = [
    r"\bare you (100%|absolutely|completely) (sure|certain)\b",
    r"\bguarantee (that )?i (do|don'?t) have\b",
    r"\bclinically (validated|proven|approved)\b",
    r"\btell me for (certain|sure)\b",
]

_CATEGORY_PATTERNS = [
    ("emergency_request", _EMERGENCY_PATTERNS,
     "This system cannot make emergency medical decisions. Please contact your local emergency "
     "services or a qualified healthcare professional immediately if you believe this is urgent."),
    ("diagnosis_request", _DIAGNOSIS_PATTERNS,
     "This system cannot diagnose a medical condition. A qualified healthcare professional can "
     "evaluate your symptoms and history properly."),
    ("medication_request", _MEDICATION_PATTERNS,
     "Medication recommendations are outside this system's scope. Please discuss medication "
     "options with a qualified healthcare professional."),
    ("dosage_request", _DOSAGE_PATTERNS,
     "Dosage advice is outside this system's scope. Please consult a qualified healthcare "
     "professional or pharmacist."),
    ("treatment_plan_request", _TREATMENT_PATTERNS,
     "Treatment planning is outside this system's scope. Please discuss treatment options with "
     "a qualified healthcare professional."),
    ("prompt_injection", _PROMPT_INJECTION_PATTERNS,
     "This request could not be processed as an educational question about the risk result."),
    ("certainty_override", _CERTAINTY_OVERRIDE_PATTERNS,
     "This system provides an educational, probabilistic ML estimate -- it cannot offer clinical "
     "certainty or claim clinical validation."),
]

_MAX_USER_MESSAGE_LENGTH = 500


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def screen_request(user_message: Optional[str]) -> SafetyResult:
    """Screens free-text user input. A missing/blank message is always allowed
    -- the structured prediction flow never depends on this text."""
    if not user_message or not user_message.strip():
        return {"allowed": True, "category": None, "reason": None}

    normalized = _normalize(user_message)
    for category, patterns, reason in _CATEGORY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized):
                return {"allowed": False, "category": category, "reason": reason}

    return {"allowed": True, "category": None, "reason": None}


def is_user_message_too_long(user_message: Optional[str]) -> bool:
    return bool(user_message) and len(user_message) > _MAX_USER_MESSAGE_LENGTH


# --- Output validation ---------------------------------------------------------

class OutputSafetyResult(TypedDict):
    valid: bool
    category: Optional[str]
    reason: Optional[str]


_DIAGNOSTIC_CERTAINTY_PATTERNS = [
    r"\byou (have|are diagnosed with) (heart disease|cvd|a heart condition)\b",
    r"\byou definitely have\b",
    r"\bthis confirms you have\b",
]

# Illustrative, non-exhaustive list of common cardiovascular drug names/classes.
# This is a guardrail, not a complete pharmacological database.
_MEDICATION_NAMES = [
    "aspirin", "statin", "atorvastatin", "simvastatin", "rosuvastatin",
    "metoprolol", "atenolol", "beta blocker", "lisinopril", "enalapril",
    "warfarin", "clopidogrel", "nitroglycerin", "amlodipine", "losartan",
]

_DOSAGE_LANGUAGE_PATTERNS = [
    r"\b\d+\s?mg\b",
    r"\bmilligrams?\b",
    r"\bonce (a day|daily)\b",
    r"\btwice (a day|daily)\b",
    r"\bdosage of\b",
]

_TREATMENT_DIRECTIVE_PATTERNS = [
    r"\byou should (undergo|get|have) (surgery|a stent|bypass surgery|an angioplasty)\b",
    r"\bstart (taking|treatment with)\b",
    r"\bbegin (this |a )?treatment\b",
]

_UNSUPPORTED_CLAIM_PATTERNS = [
    r"\bclinically validated\b",
    r"\bfda.?approved\b",
    r"\b100% (accurate|certain)\b",
]


def _text_fields(explanation: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("summary", "educational_information", "questions_for_professional"):
        value = explanation.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return _normalize(" ".join(parts))


def validate_generated_output(
    explanation: Dict[str, Any], retrieved_document_count: int,
) -> OutputSafetyResult:
    """Checks a generated explanation payload (the parsed Bedrock JSON) for
    unsafe content and structural problems. Returns valid=False on the first
    problem found.
    """
    combined_text = _text_fields(explanation)

    for pattern in _DIAGNOSTIC_CERTAINTY_PATTERNS:
        if re.search(pattern, combined_text):
            return {"valid": False, "category": "diagnostic_certainty",
                     "reason": "Generated content stated a diagnosis with certainty."}

    for drug_name in _MEDICATION_NAMES:
        if re.search(rf"\b(take|use|try) {re.escape(drug_name)}\b", combined_text):
            return {"valid": False, "category": "medication_recommendation",
                     "reason": "Generated content recommended a specific medication."}

    for pattern in _DOSAGE_LANGUAGE_PATTERNS:
        if re.search(pattern, combined_text):
            return {"valid": False, "category": "dosage_language",
                     "reason": "Generated content included dosage-style language."}

    for pattern in _TREATMENT_DIRECTIVE_PATTERNS:
        if re.search(pattern, combined_text):
            return {"valid": False, "category": "treatment_directive",
                     "reason": "Generated content issued a treatment directive."}

    for pattern in _UNSUPPORTED_CLAIM_PATTERNS:
        if re.search(pattern, combined_text):
            return {"valid": False, "category": "unsupported_clinical_claim",
                     "reason": "Generated content made an unsupported clinical validation claim."}

    disclaimer = explanation.get("disclaimer")
    if not disclaimer or not str(disclaimer).strip():
        return {"valid": False, "category": "missing_disclaimer",
                 "reason": "Generated content did not include a medical disclaimer."}

    citations = explanation.get("citations") or []
    if retrieved_document_count > 0 and not citations:
        return {"valid": False, "category": "missing_citations",
                 "reason": "Retrieved context was used but no citations were included."}

    return {"valid": True, "category": None, "reason": None}
