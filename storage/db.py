"""Database engine, session management, and CRUD helpers for CatalystIQ."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_URL
from retrieval.demo_data import build_demo_candidates
from storage.schema import Base, CandidateRecord, ExperimentalResult, ModelVersion, QueryEvent

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session, future=True)


def init_db() -> None:
    """Create all database tables.

    Args:
        None.

    Returns:
        None.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If schema creation fails.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at {}", DATABASE_URL)


def save_query_event(reaction: str, sources: list[str], count: int) -> QueryEvent:
    """Persist a query event with source metadata and result count.

    Args:
        reaction: Reaction query string.
        sources: List of source names queried.
        count: Number of records returned.

    Returns:
        Persisted QueryEvent instance with primary key.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If transaction fails.
    """
    with SessionLocal() as session:
        event = QueryEvent(reaction=reaction, sources_queried=sources, result_count=count)
        session.add(event)
        session.commit()
        session.refresh(event)
        return event


def save_candidates(query_event_id: int, df: pd.DataFrame) -> int:
    """Persist candidate rows for a query event.

    Args:
        query_event_id: Foreign-key ID from QueryEvent.
        df: DataFrame with candidate fields.

    Returns:
        Number of records successfully saved.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If insert transaction fails.
    """
    if df.empty:
        return 0

    records: list[CandidateRecord] = []
    for row in df.to_dict(orient="records"):
        conditions: dict[str, Any] = row.get("conditions") or {}
        raw_payload: dict[str, Any] = row.get("raw") or {}
        activity_value = row.get("activity_value")
        numeric_activity = float(activity_value) if activity_value is not None else None

        def _safe_float(val: Any) -> float | None:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        record = CandidateRecord(
            query_event_id=query_event_id,
            source=str(row.get("source", "")),
            source_id=str(row.get("source_id", "")),
            name=str(row.get("name", "")),
            formula=str(row.get("formula", "")),
            activity_value=numeric_activity,
            activity_unit=str(row.get("activity_unit", "")),
            predicted_activity=_safe_float(row.get("predicted_activity")),
            predicted_selectivity=_safe_float(row.get("predicted_selectivity")),
            predicted_stability=_safe_float(row.get("predicted_stability")),
            confidence=_safe_float(row.get("confidence")),
            conditions=conditions if isinstance(conditions, dict) else {"value": conditions},
            raw_payload=raw_payload if isinstance(raw_payload, dict) else {"value": raw_payload},
        )
        records.append(record)

    with SessionLocal() as session:
        session.add_all(records)
        session.commit()
        logger.info("Saved {} candidate records for query_event_id={}", len(records), query_event_id)
        return len(records)


def log_experimental_result(
    candidate_id: int,
    measured_value: float,
    measured_unit: str,
    measured_metric: str,
    predicted_value: float | None,
    outcome: str,
    notes: str = "",
    researcher: str = "",
) -> ExperimentalResult:
    """Persist an experimental result for a candidate.

    Args:
        candidate_id: FK to CandidateRecord.
        measured_value: Experimentally measured activity value.
        measured_unit: Unit of the measured value.
        measured_metric: Metric name (e.g. 'yield', 'Km', 'TOF').
        predicted_value: Model prediction at time of experiment.
        outcome: One of 'exceeded', 'matched', 'underperformed'.
        notes: Researcher free-text notes.
        researcher: Researcher name or identifier.

    Returns:
        Persisted ExperimentalResult instance.
    """
    with SessionLocal() as session:
        result = ExperimentalResult(
            candidate_id=candidate_id,
            measured_value=measured_value,
            measured_unit=measured_unit,
            measured_metric=measured_metric,
            predicted_value=predicted_value,
            outcome=outcome,
            notes=notes,
            researcher=researcher,
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        logger.info(
            "Logged experimental result for candidate_id={}: measured={} {}",
            candidate_id, measured_value, measured_unit,
        )
        return result


def get_feedback_dataframe() -> pd.DataFrame:
    """Return all experimental results joined with candidate data.

    Returns:
        DataFrame suitable for model retraining and discrepancy analysis.
    """
    with SessionLocal() as session:
        rows = session.execute(
            select(
                ExperimentalResult.id.label("result_id"),
                ExperimentalResult.candidate_id,
                ExperimentalResult.measured_value,
                ExperimentalResult.measured_unit,
                ExperimentalResult.measured_metric,
                ExperimentalResult.predicted_value.label("predicted_activity"),
                ExperimentalResult.outcome,
                ExperimentalResult.notes,
                ExperimentalResult.researcher,
                ExperimentalResult.logged_at,
                CandidateRecord.formula,
                CandidateRecord.name,
                CandidateRecord.source,
                CandidateRecord.conditions,
                CandidateRecord.activity_unit,
            )
            .join(CandidateRecord, ExperimentalResult.candidate_id == CandidateRecord.id)
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "result_id", "candidate_id", "measured_value", "measured_unit",
        "measured_metric", "predicted_activity", "outcome", "notes",
        "researcher", "logged_at", "formula", "name", "source",
        "conditions", "activity_unit",
    ])


def get_candidates_for_feedback() -> pd.DataFrame:
    """Return candidates that have been retrieved (for feedback selection UI).

    Returns:
        DataFrame with id, name, formula, source, predicted_activity.
    """
    with SessionLocal() as session:
        rows = session.execute(
            select(
                CandidateRecord.id,
                CandidateRecord.name,
                CandidateRecord.formula,
                CandidateRecord.source,
                CandidateRecord.activity_unit,
                CandidateRecord.predicted_activity,
                CandidateRecord.activity_value,
                CandidateRecord.retrieved_at,
            )
            .order_by(CandidateRecord.retrieved_at.desc())
            .limit(500)
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "id", "name", "formula", "source", "activity_unit",
        "predicted_activity", "activity_value", "retrieved_at",
    ])


def save_model_version(metrics: dict[str, Any], trigger: str = "manual") -> ModelVersion:
    """Persist a model retraining event.

    Args:
        metrics: Dict returned by CatalystPredictor.fit().
        trigger: What triggered retraining ('manual', 'auto', 'threshold').

    Returns:
        Persisted ModelVersion instance.
    """
    with SessionLocal() as session:
        version = ModelVersion(
            version_tag=metrics.get("model_version", "unknown"),
            trigger=trigger,
            n_training_samples=int(metrics.get("n_samples", 0)),
            rmse=metrics.get("rmse"),
            r2=metrics.get("r2"),
            residual_std=metrics.get("residual_std"),
            metrics=metrics,
        )
        session.add(version)
        session.commit()
        session.refresh(version)
        logger.info("Saved model version: {}", metrics)
        return version


def get_model_history() -> pd.DataFrame:
    """Return all model retraining events ordered by most recent first.

    Returns:
        DataFrame with version history.
    """
    with SessionLocal() as session:
        rows = session.execute(
            select(
                ModelVersion.id,
                ModelVersion.version_tag,
                ModelVersion.trigger,
                ModelVersion.n_training_samples,
                ModelVersion.rmse,
                ModelVersion.r2,
                ModelVersion.residual_std,
                ModelVersion.trained_at,
            ).order_by(ModelVersion.trained_at.desc())
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "id", "version_tag", "trigger", "n_training_samples",
        "rmse", "r2", "residual_std", "trained_at",
    ])


def seed_demo_database(minimum_records: int = 90) -> int:
    """Ensure baseline demo candidate records exist for offline testing.

    Args:
        minimum_records: Minimum candidate record count to keep in DB.

    Returns:
        Number of inserted demo candidate records.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If insert transaction fails.
    """
    with SessionLocal() as session:
        current_count = session.execute(select(func.count(CandidateRecord.id))).scalar_one()
        if int(current_count) >= minimum_records:
            return 0

    demo_df = build_demo_candidates(limit=minimum_records)
    seed_event = save_query_event(
        reaction="demo_seed_catalog",
        sources=["Materials Project", "BRENDA", "Open Catalyst"],
        count=len(demo_df),
    )
    inserted = save_candidates(query_event_id=seed_event.id, df=demo_df)
    logger.info("Seeded {} demo records for offline test coverage.", inserted)
    return inserted
