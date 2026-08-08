"""Shared heart-disease risk prediction service.

This is the single source of truth for loading the trained model/scaler and
running inference. It is used by the Vercel FastAPI endpoint (api/index.py)
so prediction logic is not duplicated between deployment targets.

Feature order and the sex encoding below MUST exactly match the training
pipeline in train1.py / the column order in heart.csv. Do not reorder them
without retraining, or predictions will silently be wrong.
"""
import os
import threading
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

# Order of columns the model was trained on (heart.csv columns minus "target").
FEATURE_ORDER: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Matches the encoding used historically in api/index.py and app1.py.
SEX_ENCODING = {"Male": 1, "Female": 0}

# Numeric ranges mirror the min/max constraints already present in public/index.html.
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 600),
    "thalach": (50, 250),
    "oldpeak": (0.0, 10.0),
}

# Categorical fields and their allowed values, mirrors public/index.html <select> options.
CATEGORICAL_VALUES = {
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {0, 1, 2, 3},
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


class ModelLoadError(RuntimeError):
    """Raised when the model or scaler artifacts cannot be loaded from disk."""


class PredictionValidationError(ValueError):
    """Raised when input data fails feature validation (range/category/type)."""


class PredictionError(RuntimeError):
    """Raised when the underlying model fails to produce a prediction."""


class RiskPredictionService:
    """Loads the trained heart-disease model/scaler once and serves predictions.

    A single instance loads its artifacts in __init__ and caches them for the
    lifetime of the process/serverless instance -- callers should reuse one
    instance (see get_default_service) rather than constructing a new one per
    request.
    """

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._model = None
        self._scaler = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            self._model = joblib.load(self._model_path)
            self._scaler = joblib.load(self._scaler_path)
        except Exception as exc:  # noqa: BLE001 - intentionally broad; surfaced via ModelLoadError
            self._model = None
            self._scaler = None
            self._load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._scaler is not None

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate raw patient input and return a normalized dict (sex encoded to int).

        Raises PredictionValidationError describing the first invalid field found.
        """
        normalized: Dict[str, Any] = {}

        sex = data.get("sex")
        if sex not in SEX_ENCODING:
            raise PredictionValidationError(
                f"Invalid value for 'sex': {sex!r}. Expected one of {list(SEX_ENCODING)}."
            )
        normalized["sex"] = SEX_ENCODING[sex]

        for field, (low, high) in NUMERIC_RANGES.items():
            value = data.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PredictionValidationError(f"Field '{field}' must be numeric.")
            if value < low or value > high:
                raise PredictionValidationError(
                    f"Field '{field}' must be between {low} and {high}, got {value}."
                )
            normalized[field] = value

        for field, allowed in CATEGORICAL_VALUES.items():
            value = data.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PredictionValidationError(f"Field '{field}' must be an integer.")
            if value not in allowed:
                raise PredictionValidationError(
                    f"Field '{field}' must be one of {sorted(allowed)}, got {value}."
                )
            normalized[field] = value

        return normalized

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input, run inference, and return the prediction.

        Returns:
            {
                "prediction": int (0 or 1),
                "probability": float | None (probability of class 1, if supported),
                "normalized_input": dict (validated + sex-encoded input actually used),
            }

        Raises:
            ModelLoadError: model/scaler failed to load.
            PredictionValidationError: input failed validation.
            PredictionError: inference itself failed unexpectedly.
        """
        if not self.is_ready:
            raise ModelLoadError(self._load_error or "Model or scaler not loaded.")

        normalized = self.validate(data)
        ordered_values = [normalized[feature] for feature in FEATURE_ORDER]

        try:
            input_array = np.array([ordered_values], dtype=float)
            input_scaled = self._scaler.transform(input_array)
            prediction = int(self._model.predict(input_scaled)[0])

            probability: Optional[float] = None
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(input_scaled)[0]
                classes = list(getattr(self._model, "classes_", [0, 1]))
                positive_index = classes.index(1) if 1 in classes else -1
                probability = float(proba[positive_index])
        except Exception as exc:  # noqa: BLE001 - converted to a domain error for callers
            raise PredictionError(f"Prediction failed: {exc}") from exc

        return {
            "prediction": prediction,
            "probability": probability,
            "normalized_input": normalized,
        }


_default_service: Optional[RiskPredictionService] = None
_default_service_lock = threading.Lock()


def get_default_service() -> RiskPredictionService:
    """Returns a process-wide singleton RiskPredictionService (artifacts loaded once)."""
    global _default_service
    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                _default_service = RiskPredictionService()
    return _default_service
