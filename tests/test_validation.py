"""Tests for input validation: both the Pydantic schema layer (api/schemas.py)
and the service-level validation in tools/risk_prediction.py.
"""
import pytest
from pydantic import ValidationError

from api.schemas import PatientInput
from tools.risk_prediction import PredictionValidationError, RiskPredictionService

VALID_PATIENT = {
    "age": 52,
    "sex": "Male",
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}


@pytest.fixture(scope="module")
def service() -> RiskPredictionService:
    return RiskPredictionService()


# --- Pydantic schema (api/schemas.py) ---------------------------------------

def test_valid_patient_input_passes_schema():
    model = PatientInput(**VALID_PATIENT)
    assert model.age == 52


@pytest.mark.parametrize("field,bad_value", [
    ("age", 0),
    ("age", 121),
    ("trestbps", 49),
    ("trestbps", 251),
    ("chol", 49),
    ("chol", 601),
    ("thalach", 49),
    ("thalach", 251),
    ("oldpeak", -0.1),
    ("oldpeak", 10.1),
])
def test_schema_rejects_out_of_range_numeric_values(field, bad_value):
    payload = {**VALID_PATIENT, field: bad_value}
    with pytest.raises(ValidationError):
        PatientInput(**payload)


@pytest.mark.parametrize("field,bad_value", [
    ("cp", 4),
    ("fbs", 2),
    ("restecg", 3),
    ("exang", 2),
    ("slope", 3),
    ("ca", 5),
    ("thal", 4),
    ("sex", "Other"),
    ("sex", "male"),  # wrong case is rejected -- must match frontend's exact option text
])
def test_schema_rejects_invalid_categorical_values(field, bad_value):
    payload = {**VALID_PATIENT, field: bad_value}
    with pytest.raises(ValidationError):
        PatientInput(**payload)


@pytest.mark.parametrize("missing_field", list(VALID_PATIENT.keys()))
def test_schema_rejects_missing_required_fields(missing_field):
    payload = {k: v for k, v in VALID_PATIENT.items() if k != missing_field}
    with pytest.raises(ValidationError):
        PatientInput(**payload)


def test_schema_rejects_unexpected_extra_fields():
    payload = {**VALID_PATIENT, "unexpected_field": "should not be allowed"}
    with pytest.raises(ValidationError):
        PatientInput(**payload)


def test_schema_rejects_wrong_type():
    payload = {**VALID_PATIENT, "age": "fifty-two"}
    with pytest.raises(ValidationError):
        PatientInput(**payload)


# --- Service-level validation (tools/risk_prediction.py) --------------------

def test_service_rejects_invalid_sex(service):
    payload = {**VALID_PATIENT, "sex": "Unknown"}
    with pytest.raises(PredictionValidationError):
        service.validate(payload)


def test_service_rejects_out_of_range_numeric(service):
    payload = {**VALID_PATIENT, "chol": 9999}
    with pytest.raises(PredictionValidationError):
        service.validate(payload)


def test_service_rejects_invalid_categorical(service):
    payload = {**VALID_PATIENT, "thal": 99}
    with pytest.raises(PredictionValidationError):
        service.validate(payload)


def test_service_rejects_missing_field(service):
    payload = {k: v for k, v in VALID_PATIENT.items() if k != "chol"}
    with pytest.raises(PredictionValidationError):
        service.validate(payload)
