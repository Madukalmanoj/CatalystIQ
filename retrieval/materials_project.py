"""Materials Project retriever using REST API with normalized CatalystRecord mapping."""

from __future__ import annotations

import re
from typing import Any

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import MP_API_KEY
from retrieval.base import BaseRetriever, CatalystRecord, RetrieverConnectionError, RetrieverParseError

_MP_SUMMARY_URL = "https://api.materialsproject.org/materials/summary/"
_MP_FIELDS = "material_id,formula_pretty,formation_energy_per_atom,energy_above_hull"

# Match element symbols in chemical formulas/reactions:
# uppercase letter optionally followed by lowercase, then digit or non-letter.
# Handles N2, H2, NH3, CO2, Fe2O3, etc.
_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(?=\d|[^a-z]|$)")


class MaterialsProjectRetriever(BaseRetriever):
    """Retrieve catalyst candidates from Materials Project via REST API."""

    source_name = "Materials Project"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize retriever with API key.

        Args:
            api_key: Optional explicit MP API key override.

        Returns:
            None.

        Raises:
            None.
        """
        self.api_key = api_key or MP_API_KEY

    def _parse_elements(self, reaction: str) -> list[str]:
        """Extract probable chemical element symbols from query.

        Args:
            reaction: Free-text reaction string.

        Returns:
            Sorted unique list of element symbols.

        Raises:
            None.
        """
        matches = _ELEMENT_PATTERN.findall(reaction or "")
        # Filter out common non-element words that match the pattern
        _non_elements = {"At", "In", "As", "No", "Is", "Be", "Do", "Go", "He", "Me", "My", "Or", "So", "To", "Up", "Us", "We"}
        return sorted({m for m in matches if m not in _non_elements})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _query_materials(self, elements: list[str]) -> list[dict[str, Any]]:
        """Call Materials Project summary REST endpoint.

        Args:
            elements: Chemical element symbols to filter by.

        Returns:
            List of material summary dicts.

        Raises:
            RetrieverConnectionError: If API key is missing or request fails.
        """
        if not self.api_key:
            raise RetrieverConnectionError("Missing Materials Project API key.")

        params: dict[str, Any] = {
            "elements": ",".join(elements),
            "_fields": _MP_FIELDS,
            "_limit": 100,
        }
        headers = {"X-API-KEY": self.api_key}

        try:
            response = requests.get(
                _MP_SUMMARY_URL,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.HTTPError as exc:
            logger.error("Materials Project HTTP error {}: {}", exc.response.status_code, exc.response.text[:200])
            raise RetrieverConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Materials Project query failed for elements={}", elements)
            raise RetrieverConnectionError(str(exc)) from exc

    def search(self, reaction: str) -> list[CatalystRecord]:
        """Fetch and normalize stable material candidates.

        Args:
            reaction: Reaction text used for element extraction.

        Returns:
            List of CatalystRecord items.

        Raises:
            RetrieverParseError: If response parsing fails unexpectedly.
        """
        if not self.api_key:
            logger.warning("Materials Project API key missing; returning empty result.")
            return []

        elements = self._parse_elements(reaction)
        if not elements:
            logger.warning("No element symbols parsed from reaction='{}'", reaction)
            return []

        try:
            docs = self._query_materials(elements=elements)
            records: list[CatalystRecord] = []
            for doc in docs:
                energy_above_hull = doc.get("energy_above_hull")
                if energy_above_hull is None or float(energy_above_hull) >= 0.1:
                    continue

                formation_energy = doc.get("formation_energy_per_atom")
                material_id = str(doc.get("material_id", ""))
                formula_pretty = str(doc.get("formula_pretty", ""))

                record = CatalystRecord(
                    source=self.source_name,
                    source_id=material_id,
                    name=f"MP {material_id}",
                    formula=formula_pretty,
                    reaction=reaction,
                    activity_metric="formation_energy_per_atom",
                    activity_value=float(formation_energy) if formation_energy is not None else None,
                    activity_unit="eV/atom",
                    conditions={},
                    stability=float(energy_above_hull),
                    raw=doc,
                )
                records.append(record)

            logger.info("Materials Project returned {} stable candidates.", len(records))
            return records
        except RetrieverConnectionError:
            logger.warning("Materials Project connection issue; returning empty results.")
            return []
        except Exception as exc:  # noqa: BLE001
            raise RetrieverParseError(f"Failed to parse Materials Project payload: {exc}") from exc

    def health_check(self) -> bool:
        """Check retriever readiness.

        Args:
            None.

        Returns:
            True if API key is set and a minimal query succeeds.

        Raises:
            None.
        """
        if not self.api_key:
            return False
        try:
            docs = self._query_materials(elements=["H"])
            return isinstance(docs, list)
        except Exception:  # noqa: BLE001
            return False
