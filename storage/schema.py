"""SQLAlchemy ORM schema for query provenance, candidate records,
experimental feedback, and model version tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return UTC timestamp for default column values.

    Args:
        None.

    Returns:
        Current UTC datetime.

    Raises:
        None.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for CatalystIQ models."""


class QueryEvent(Base):
    """Persist a single user query event and result metadata."""

    __tablename__ = "query_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reaction: Mapped[str] = mapped_column(String(512), nullable=False)
    sources_queried: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    candidates: Mapped[list["CandidateRecord"]] = relationship(
        back_populates="query_event",
        cascade="all, delete-orphan",
    )


class CandidateRecord(Base):
    """Persist normalized candidate details and raw payload provenance."""

    __tablename__ = "candidate_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_event_id: Mapped[int] = mapped_column(ForeignKey("query_events.id"), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    formula: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    activity_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    activity_unit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    predicted_activity: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_selectivity: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_stability: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    query_event: Mapped[QueryEvent] = relationship(back_populates="candidates")
    experimental_results: Mapped[list["ExperimentalResult"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )


class ExperimentalResult(Base):
    """Log experimental outcomes for a candidate — drives the feedback loop."""

    __tablename__ = "experimental_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_records.id"), nullable=False, index=True
    )
    # What was measured
    measured_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured_unit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    measured_metric: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    # What the model predicted at the time
    predicted_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Outcome classification
    outcome: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )  # "exceeded" | "matched" | "underperformed" | "unknown"
    # Free-text researcher notes
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    researcher: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    candidate: Mapped[CandidateRecord] = relationship(back_populates="experimental_results")


class ModelVersion(Base):
    """Track model retraining events and performance metrics."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(String(64), nullable=False)
    trigger: Mapped[str] = mapped_column(String(128), nullable=False, default="manual")
    n_training_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    residual_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
