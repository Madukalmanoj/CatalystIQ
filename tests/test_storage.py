"""Tests for storage layer — schema, DB CRUD, and feedback loop."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

# Use in-memory SQLite for all storage tests
TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def db_session():
    """Create a fresh in-memory DB for the test module."""
    from storage.schema import Base, QueryEvent, CandidateRecord, ExperimentalResult, ModelVersion
    engine = create_engine(TEST_DB_URL, future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False,
                                class_=Session, future=True)
    yield SessionLocal, engine
    engine.dispose()


@pytest.fixture(scope="module")
def db_helpers(db_session):
    """Patch storage.db to use the in-memory engine."""
    import storage.db as db_mod
    SessionLocal, engine = db_session
    original_session = db_mod.SessionLocal
    original_engine = db_mod.engine
    db_mod.SessionLocal = SessionLocal
    db_mod.engine = engine
    yield db_mod
    db_mod.SessionLocal = original_session
    db_mod.engine = original_engine


# ── Schema ────────────────────────────────────────────────────────────────────

def test_schema_tables_exist(db_session):
    from storage.schema import Base
    _, engine = db_session
    table_names = set(Base.metadata.tables.keys())
    assert "query_events" in table_names
    assert "candidate_records" in table_names
    assert "experimental_results" in table_names
    assert "model_versions" in table_names


def test_query_event_columns(db_session):
    from storage.schema import QueryEvent
    cols = {c.name for c in QueryEvent.__table__.columns}
    assert {"id", "reaction", "sources_queried", "result_count", "timestamp"} <= cols


def test_candidate_record_columns(db_session):
    from storage.schema import CandidateRecord
    cols = {c.name for c in CandidateRecord.__table__.columns}
    assert {"id", "query_event_id", "source", "formula", "activity_value",
            "predicted_activity", "predicted_selectivity", "predicted_stability",
            "confidence", "name"} <= cols


def test_experimental_result_columns(db_session):
    from storage.schema import ExperimentalResult
    cols = {c.name for c in ExperimentalResult.__table__.columns}
    assert {"id", "candidate_id", "measured_value", "predicted_value",
            "outcome", "notes", "researcher"} <= cols


def test_model_version_columns(db_session):
    from storage.schema import ModelVersion
    cols = {c.name for c in ModelVersion.__table__.columns}
    assert {"id", "version_tag", "trigger", "n_training_samples", "rmse", "r2"} <= cols


# ── save_query_event ──────────────────────────────────────────────────────────

def test_save_query_event(db_helpers):
    event = db_helpers.save_query_event("CO2 + H2 -> methanol", ["MP", "BRENDA"], 10)
    assert event.id is not None
    assert event.reaction == "CO2 + H2 -> methanol"
    assert event.result_count == 10


def test_save_query_event_empty_sources(db_helpers):
    event = db_helpers.save_query_event("N2 + H2 -> NH3", [], 0)
    assert event.id is not None


# ── save_candidates ───────────────────────────────────────────────────────────

@pytest.fixture
def candidate_df():
    return pd.DataFrame([
        {"source": "Materials Project", "source_id": "mp-1", "name": "Fe2O3",
         "formula": "Fe2O3", "activity_value": -1.2, "activity_unit": "eV/atom",
         "predicted_activity": -1.15, "predicted_selectivity": 0.72,
         "predicted_stability": 0.03, "confidence": 0.68,
         "conditions": {"temperature_k": 300}, "raw": {"test": True}},
        {"source": "BRENDA", "source_id": "brenda-1", "name": "Hexokinase",
         "formula": "", "activity_value": 0.15, "activity_unit": "mM",
         "predicted_activity": 0.12, "predicted_selectivity": 0.85,
         "predicted_stability": None, "confidence": 0.60,
         "conditions": {"organism": "S. cerevisiae"}, "raw": {}},
    ])


def test_save_candidates_returns_count(db_helpers, candidate_df):
    event = db_helpers.save_query_event("test reaction", ["MP"], 2)
    n = db_helpers.save_candidates(event.id, candidate_df)
    assert n == 2


def test_save_candidates_empty_df(db_helpers):
    event = db_helpers.save_query_event("empty test", [], 0)
    n = db_helpers.save_candidates(event.id, pd.DataFrame())
    assert n == 0


def test_save_candidates_null_activity(db_helpers):
    df = pd.DataFrame([{
        "source": "MP", "source_id": "x", "name": "Test", "formula": "Pt",
        "activity_value": None, "activity_unit": "eV/atom",
        "predicted_activity": None, "predicted_selectivity": None,
        "predicted_stability": None, "confidence": None,
        "conditions": {}, "raw": {},
    }])
    event = db_helpers.save_query_event("null test", ["MP"], 1)
    n = db_helpers.save_candidates(event.id, df)
    assert n == 1


# ── get_candidates_for_feedback ───────────────────────────────────────────────

def test_get_candidates_for_feedback_returns_df(db_helpers, candidate_df):
    event = db_helpers.save_query_event("feedback test", ["MP"], 2)
    db_helpers.save_candidates(event.id, candidate_df)
    result = db_helpers.get_candidates_for_feedback()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_get_candidates_for_feedback_columns(db_helpers, candidate_df):
    event = db_helpers.save_query_event("col test", ["MP"], 2)
    db_helpers.save_candidates(event.id, candidate_df)
    result = db_helpers.get_candidates_for_feedback()
    for col in ["id", "name", "formula", "source", "activity_unit", "predicted_activity"]:
        assert col in result.columns


# ── log_experimental_result ───────────────────────────────────────────────────

def test_log_experimental_result(db_helpers, candidate_df):
    event = db_helpers.save_query_event("exp test", ["MP"], 1)
    db_helpers.save_candidates(event.id, candidate_df.head(1))
    cands = db_helpers.get_candidates_for_feedback()
    cid = int(cands.iloc[0]["id"])

    result = db_helpers.log_experimental_result(
        candidate_id=cid,
        measured_value=-1.05,
        measured_unit="eV/atom",
        measured_metric="formation_energy",
        predicted_value=-1.15,
        outcome="exceeded",
        notes="Better than expected",
        researcher="Dr. Test",
    )
    assert result.id is not None
    assert result.outcome == "exceeded"
    assert result.measured_value == pytest.approx(-1.05)


def test_log_experimental_result_all_outcomes(db_helpers, candidate_df):
    event = db_helpers.save_query_event("outcome test", ["MP"], 1)
    db_helpers.save_candidates(event.id, candidate_df.head(1))
    cands = db_helpers.get_candidates_for_feedback()
    cid = int(cands.iloc[-1]["id"])

    for outcome in ["exceeded", "matched", "underperformed", "unknown"]:
        result = db_helpers.log_experimental_result(
            candidate_id=cid, measured_value=0.5, measured_unit="eV",
            measured_metric="test", predicted_value=0.6, outcome=outcome,
        )
        assert result.outcome == outcome


# ── get_feedback_dataframe ────────────────────────────────────────────────────

def test_get_feedback_dataframe_returns_df(db_helpers, candidate_df):
    event = db_helpers.save_query_event("fb df test", ["MP"], 1)
    db_helpers.save_candidates(event.id, candidate_df.head(1))
    cands = db_helpers.get_candidates_for_feedback()
    cid = int(cands.iloc[0]["id"])
    db_helpers.log_experimental_result(
        candidate_id=cid, measured_value=-1.0, measured_unit="eV/atom",
        measured_metric="fe", predicted_value=-1.1, outcome="matched",
    )
    fb = db_helpers.get_feedback_dataframe()
    assert isinstance(fb, pd.DataFrame)
    assert len(fb) > 0


def test_get_feedback_dataframe_columns(db_helpers, candidate_df):
    fb = db_helpers.get_feedback_dataframe()
    if fb.empty:
        pytest.skip("No feedback data yet")
    for col in ["measured_value", "predicted_activity", "outcome", "formula", "source"]:
        assert col in fb.columns


# ── save_model_version / get_model_history ────────────────────────────────────

def test_save_model_version(db_helpers):
    metrics = {
        "model_version": "ridge-v1", "n_samples": 10,
        "rmse": 0.05, "r2": 0.88, "residual_std": 0.04, "status": "trained",
    }
    mv = db_helpers.save_model_version(metrics, trigger="manual")
    assert mv.id is not None
    assert mv.version_tag == "ridge-v1"
    assert mv.r2 == pytest.approx(0.88)


def test_get_model_history_returns_df(db_helpers):
    history = db_helpers.get_model_history()
    assert isinstance(history, pd.DataFrame)
    assert len(history) > 0


def test_get_model_history_columns(db_helpers):
    history = db_helpers.get_model_history()
    for col in ["id", "version_tag", "trigger", "n_training_samples", "rmse", "r2"]:
        assert col in history.columns


def test_get_model_history_ordered_desc(db_helpers):
    db_helpers.save_model_version(
        {"model_version": "ridge-v2", "n_samples": 15, "rmse": 0.03,
         "r2": 0.92, "residual_std": 0.02, "status": "trained"}, trigger="auto"
    )
    history = db_helpers.get_model_history()
    if len(history) >= 2:
        dates = history["trained_at"].tolist()
        assert dates == sorted(dates, reverse=True)
