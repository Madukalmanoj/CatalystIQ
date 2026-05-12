"""Core retrieval abstractions and common catalyst record model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class RetrieverConnectionError(Exception):
    """Raised when a retriever cannot connect to its upstream source."""


class RetrieverParseError(Exception):
    """Raised when a retriever cannot parse upstream response data."""


class CatalystRecord(BaseModel):
    """Normalized catalyst/enzyme candidate payload used across retrievers."""

    source: str
    source_id: str = ""
    name: str = ""
    formula: str = ""
    reaction: str
    activity_metric: str = ""
    activity_value: float | None = None
    activity_unit: str = ""
    conditions: dict[str, Any] = Field(default_factory=dict)
    stability: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface every source retriever must implement."""

    source_name: str

    @abstractmethod
    def search(self, reaction: str) -> list[CatalystRecord]:
        """Return candidate records for a reaction query.

        Args:
            reaction: Free-text reaction or substrate query.

        Returns:
            List of normalized CatalystRecord items.

        Raises:
            RetrieverConnectionError: If upstream cannot be reached.
            RetrieverParseError: If payload mapping fails.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return availability state for the retriever backend.

        Args:
            None.

        Returns:
            True when backend appears available, else False.

        Raises:
            None.
        """
