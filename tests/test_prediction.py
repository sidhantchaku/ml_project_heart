"""Tests for tools/risk_prediction.py using the real committed model artifacts.

Only failure modes (missing/corrupt artifacts) are simulated; normal predictions
run against heart_model.pkl / scaler.pkl as committed.
"""
import joblib
import numpy as np
import pytest

from tools.risk_prediction import (
    CATEGORICAL_VALUES,
    FEATURE_ORDER,
    MODEL_PATH,
    SCALER_PATH,
    SEX_ENCODING,
    ModelLoadError,
    PredictionError,
    RiskPredictionService,
)

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


def test_feature_order_matches_training_columns():
    # heart.csv columns, minus "target", in exact training order.
    assert FEATURE_ORDER == [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]


def test_sex_encoding():
    assert SEX_ENCODING == {"Male": 1, "Female": 0}


def test_model_and_scaler_load_successfully(service):
    assert service.is_ready is True


def test_model_and_scaler_expect_thirteen_features(service):
    assert service._model.n_features_in_ == len(FEATURE_ORDER)
    assert service._scaler.n_features_in_ == len(FEATURE_ORDER)


def test_valid_prediction_returns_expected_shape(service):
    result = service.predict(VALID_PATIENT)
    assert result["prediction"] in (0, 1)
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["normalized_input"]["sex"] == 1  # "Male" -> 1


def test_female_sex_is_encoded_as_zero(service):
    patient = {**VALID_PATIENT, "sex": "Female"}
    result = service.predict(patient)
    assert result["normalized_input"]["sex"] == 0


def test_prediction_matches_manual_reference_computation(service):
    """Guards against behavioural drift from the original api/index.py logic:
    build the input the same way the pre-refactor code did, and confirm the
    shared service produces an identical result.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sex_num = 1 if VALID_PATIENT["sex"] == "Male" else 0
    manual_input = np.array([[
        VALID_PATIENT["age"], sex_num, VALID_PATIENT["cp"], VALID_PATIENT["trestbps"],
        VALID_PATIENT["chol"], VALID_PATIENT["fbs"], VALID_PATIENT["restecg"],
        VALID_PATIENT["thalach"], VALID_PATIENT["exang"], VALID_PATIENT["oldpeak"],
        VALID_PATIENT["slope"], VALID_PATIENT["ca"], VALID_PATIENT["thal"],
    ]])
    manual_scaled = scaler.transform(manual_input)
    manual_prediction = int(model.predict(manual_scaled)[0])
    manual_probability = float(model.predict_proba(manual_scaled)[0][1])

    result = service.predict(VALID_PATIENT)

    assert result["prediction"] == manual_prediction
    assert result["probability"] == pytest.approx(manual_probability)


def test_missing_model_artifact_raises_model_load_error(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.pkl")
    service = RiskPredictionService(model_path=missing_path, scaler_path=SCALER_PATH)
    assert service.is_ready is False
    with pytest.raises(ModelLoadError):
        service.predict(VALID_PATIENT)


def test_missing_scaler_artifact_raises_model_load_error(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.pkl")
    service = RiskPredictionService(model_path=MODEL_PATH, scaler_path=missing_path)
    assert service.is_ready is False
    with pytest.raises(ModelLoadError):
        service.predict(VALID_PATIENT)


def test_internal_inference_failure_raises_prediction_error(service, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(service._scaler, "transform", _boom)
    with pytest.raises(PredictionError):
        service.predict(VALID_PATIENT)


def test_categorical_values_match_frontend_options():
    assert CATEGORICAL_VALUES == {
        "cp": {0, 1, 2, 3},
        "fbs": {0, 1},
        "restecg": {0, 1, 2},
        "exang": {0, 1},
        "slope": {0, 1, 2},
        "ca": {0, 1, 2, 3, 4},
        "thal": {0, 1, 2, 3},
    }
