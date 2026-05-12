"""Tests for ai.predictor module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from ai.predictor import (
    CatalystPredictor,
    PredictionResult,
    _parse_formula,
    _formula_features,
    _condition_features,
    _N_FEATURES,
)


# ── Feature extraction ────────────────────────────────────────────────────────

def test_formula_features_length():
    feats = _formula_features("Fe2O3")
    assert len(feats) == _N_FEATURES

def test_formula_features_empty():
    feats = _formula_features("")
    assert len(feats) == _N_FEATURES
    assert all(f == 0.0 for f in feats)

def test_formula_features_unknown_element():
    # Unknown element should not crash
    feats = _formula_features("Xx2O3")
    assert len(feats) == _N_FEATURES

def test_formula_features_single_element():
    feats = _formula_features("Pt")
    assert len(feats) == _N_FEATURES
    assert any(f != 0.0 for f in feats)

def test_condition_features_length():
    feats = _condition_features({"temperature_k": 500, "ph": 7.0})
    assert len(feats) == 3

def test_condition_features_defaults():
    feats = _condition_features({})
    assert len(feats) == 3
    assert feats[0] == pytest.approx(298.0 / 1000.0)
    assert feats[1] == pytest.approx(7.0 / 14.0)

def test_condition_features_normalised():
    feats = _condition_features({"temperature_k": 1000, "ph": 14.0})
    assert feats[0] == pytest.approx(1.0)
    assert feats[1] == pytest.approx(1.0)


# ── CatalystPredictor.predict ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"formula": "Fe2O3", "source": "Materials Project", "activity_value": -1.2,
         "activity_unit": "eV/atom", "conditions": {}},
        {"formula": "NiO",   "source": "Materials Project", "activity_value": -0.9,
         "activity_unit": "eV/atom", "conditions": {}},
        {"formula": "Pt",    "source": "Open Catalyst",     "activity_value": -0.5,
         "activity_unit": "eV",      "conditions": {"temperature_k": 400}},
        {"formula": "",      "source": "BRENDA",            "activity_value": 0.15,
         "activity_unit": "mM",      "conditions": {"ph": 7.2}, "name": "Hexokinase"},
    ])


def test_predict_adds_columns(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    for col in ["predicted_activity", "predicted_selectivity", "predicted_stability",
                "confidence", "model_version", "rank"]:
        assert col in result.columns, f"Missing column: {col}"


def test_predict_empty_df():
    pred = CatalystPredictor()
    result = pred.predict(pd.DataFrame())
    assert result.empty


def test_predict_confidence_range(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    for val in result["confidence"].dropna():
        assert 0.0 <= val <= 1.0, f"confidence={val} out of [0,1]"


def test_predict_selectivity_range(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    for val in result["predicted_selectivity"].dropna():
        assert 0.0 <= val <= 1.0, f"selectivity={val} out of [0,1]"


def test_predict_rank_sequential(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    ranks = sorted(result["rank"].tolist())
    assert ranks == list(range(1, len(sample_df) + 1))


def test_predict_sorted_by_activity(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    # Results are now sorted by rank_score (per-source percentile), not raw activity.
    # Verify rank_score is monotonically non-decreasing instead.
    rank_scores = result["rank_score"].tolist()
    assert rank_scores == sorted(rank_scores), (
        f"rank_score not sorted: {rank_scores}"
    )


def test_predict_preserves_existing_generated_values():
    """Generated candidates already have predicted_activity set — don't overwrite."""
    df = pd.DataFrame([{
        "formula": "FeNi", "source": "Generative AI",
        "activity_value": None,
        "predicted_activity": -1.999,
        "predicted_selectivity": 0.88,
        "predicted_stability": 0.02,
        "confidence": 0.75,
        "conditions": {},
    }])
    pred = CatalystPredictor()
    result = pred.predict(df)
    assert result.iloc[0]["predicted_activity"] == pytest.approx(-1.999)
    assert result.iloc[0]["confidence"] == pytest.approx(0.75)


def test_predict_model_version_heuristic(sample_df):
    pred = CatalystPredictor()
    result = pred.predict(sample_df)
    assert all(result["model_version"] == "heuristic-v1")


# ── CatalystPredictor.fit ─────────────────────────────────────────────────────

@pytest.fixture
def feedback_df():
    return pd.DataFrame([
        {"formula": "Fe2O3", "conditions": {}, "measured_value": -1.1, "predicted_activity": -1.2},
        {"formula": "NiO",   "conditions": {}, "measured_value": -0.8, "predicted_activity": -0.9},
        {"formula": "CuO",   "conditions": {}, "measured_value": -0.5, "predicted_activity": -0.7},
        {"formula": "Co3O4", "conditions": {}, "measured_value": -1.3, "predicted_activity": -1.0},
        {"formula": "RuO2",  "conditions": {}, "measured_value": -1.5, "predicted_activity": -1.4},
        {"formula": "Pt",    "conditions": {}, "measured_value": -0.3, "predicted_activity": -0.4},
    ])


def test_fit_returns_trained_status(feedback_df):
    pred = CatalystPredictor()
    metrics = pred.fit(feedback_df)
    assert metrics["status"] == "trained"


def test_fit_returns_metrics(feedback_df):
    pred = CatalystPredictor()
    metrics = pred.fit(feedback_df)
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "n_samples" in metrics
    assert metrics["n_samples"] == 6


def test_fit_insufficient_data():
    pred = CatalystPredictor()
    small_df = pd.DataFrame([
        {"formula": "Fe2O3", "conditions": {}, "measured_value": -1.1},
        {"formula": "NiO",   "conditions": {}, "measured_value": -0.8},
    ])
    metrics = pred.fit(small_df)
    assert metrics["status"] == "insufficient_data"


def test_fit_empty_df():
    pred = CatalystPredictor()
    metrics = pred.fit(pd.DataFrame())
    assert metrics["status"] == "insufficient_data"


def test_fit_changes_model_version(feedback_df, sample_df):
    pred = CatalystPredictor()
    pred.fit(feedback_df)
    result = pred.predict(sample_df)
    assert all(result["model_version"] == "ridge-v1")


def test_fit_predict_after_training(feedback_df, sample_df):
    pred = CatalystPredictor()
    pred.fit(feedback_df)
    result = pred.predict(sample_df)
    assert not result["predicted_activity"].isna().all()


# ── analyse_discrepancies ─────────────────────────────────────────────────────

def test_analyse_discrepancies_returns_list(feedback_df):
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(feedback_df)
    assert isinstance(result, list)
    assert len(result) == len(feedback_df)


def test_analyse_discrepancies_sorted_by_abs_error(feedback_df):
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(feedback_df)
    errors = [d["abs_error"] for d in result]
    assert errors == sorted(errors, reverse=True)


def test_analyse_discrepancies_fields(feedback_df):
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(feedback_df)
    required = ["formula", "predicted", "measured", "error", "abs_error", "direction", "hypothesis"]
    for disc in result:
        missing = [k for k in required if k not in disc]
        assert not missing, f"Missing keys: {missing}"


def test_analyse_discrepancies_direction(feedback_df):
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(feedback_df)
    for disc in result:
        assert disc["direction"] in ("exceeded", "underperformed")


def test_analyse_discrepancies_empty():
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(pd.DataFrame())
    assert result == []


def test_analyse_discrepancies_missing_values():
    df = pd.DataFrame([
        {"formula": "Fe2O3", "predicted_activity": None, "measured_value": -1.1},
        {"formula": "NiO",   "predicted_activity": -0.9, "measured_value": None},
    ])
    pred = CatalystPredictor()
    result = pred.analyse_discrepancies(df)
    assert result == []
