"""Open Catalyst (OC20) local data retriever for adsorbate-energy lookup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config import OCP_DATA_DIR
from retrieval.base import BaseRetriever, CatalystRecord, RetrieverParseError

OCP_FILENAME_CANDIDATES = ("oc20_data_mapping.pkl", "oc20_data_mapping.csv")
OCP_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"


class OpenCatalystRetriever(BaseRetriever):
    """Retrieve catalyst candidates from local OC20 mapping artifacts."""

    source_name = "Open Catalyst"

    def __init__(self, data_dir: str | None = None) -> None:
        """Initialize OC20 data directory and lazy dataframe holder.

        Args:
            data_dir: Optional explicit data directory.

        Returns:
            None.

        Raises:
            None.
        """
        self.data_dir = Path(data_dir or OCP_DATA_DIR)
        self._df: pd.DataFrame | None = None

    def _resolve_data_file(self) -> Path | None:
        """Find first available OC20 mapping file from known names.

        Args:
            None.

        Returns:
            Path to data file or None when not found.

        Raises:
            None.
        """
        for name in OCP_FILENAME_CANDIDATES:
            candidate = self.data_dir / name
            if candidate.exists():
                return candidate
        return None

    def _load_data(self) -> pd.DataFrame:
        """Load OC20 mapping file from pickle or CSV into DataFrame.

        Args:
            None.

        Returns:
            DataFrame with OC20 mapping records.

        Raises:
            RetrieverParseError: If file exists but parsing fails.
        """
        if self._df is not None:
            return self._df

        file_path = self._resolve_data_file()
        if file_path is None:
            logger.warning(
                "Open Catalyst mapping not found in {}. Download from {}",
                str(self.data_dir.resolve()),
                OCP_DOWNLOAD_URL,
            )
            self._df = pd.DataFrame()
            return self._df

        try:
            if file_path.suffix.lower() == ".pkl":
                data = pd.read_pickle(file_path)
                if isinstance(data, dict):
                    rows: list[dict[str, Any]] = []
                    for key, payload in data.items():
                        row = {"id": key}
                        if isinstance(payload, dict):
                            row.update(payload)
                        rows.append(row)
                    df = pd.DataFrame(rows)
                else:
                    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            else:
                df = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001
            raise RetrieverParseError(f"Failed to load OC20 mapping file: {exc}") from exc

        if "adsorbate" not in df.columns and "ads_symbols" in df.columns:
            df["adsorbate"] = (
                df["ads_symbols"]
                .astype(str)
                .str.replace("*", "", regex=False)
                .str.strip()
            )
        if "bulk_formula" not in df.columns and "bulk_symbols" in df.columns:
            df["bulk_formula"] = df["bulk_symbols"]
        if "adsorption_energy" not in df.columns:
            if "anomaly" in df.columns:
                df["adsorption_energy"] = pd.to_numeric(df["anomaly"], errors="coerce")
            elif "class" in df.columns:
                df["adsorption_energy"] = pd.to_numeric(df["class"], errors="coerce")
            else:
                df["adsorption_energy"] = None

        self._df = df
        return self._df

    @staticmethod
    def _extract_tokens(reaction: str) -> list[str]:
        """Extract normalized query tokens from reaction text.

        Args:
            reaction: User reaction query.

        Returns:
            Lowercase alphanumeric tokens with length >= 1.

        Raises:
            None.
        """
        cleaned = reaction.replace("→", " ").replace("->", " ").replace("+", " ")
        raw_tokens = re.split(r"[^A-Za-z0-9]+", cleaned.lower().strip())
        return [token for token in raw_tokens if token]

    def search(self, reaction: str) -> list[CatalystRecord]:
        """Filter OC20 local rows by adsorbate keyword and rank candidates.

        Args:
            reaction: Reaction text used to infer adsorbate.

        Returns:
            Top 20 CatalystRecord items by lowest absolute adsorption energy.

        Raises:
            RetrieverParseError: If OC20 parsing fails.
        """
        df = self._load_data()
        if df.empty:
            return []

        tokens = self._extract_tokens(reaction)
        if not tokens:
            return []

        adsorbate_series = df["adsorbate"].astype(str).str.lower()
        bulk_formula_series = df["bulk_formula"].astype(str).str.lower()
        token_pattern = "|".join(re.escape(token) for token in tokens[:6])

        filtered = df[
            adsorbate_series.str.contains(token_pattern, na=False)
            | bulk_formula_series.str.contains(token_pattern, na=False)
        ].copy()

        if filtered.empty:
            logger.warning(
                "No exact OC20 token match for reaction='{}'. Returning globally best live OC20 rows.",
                reaction,
            )
            filtered = df.copy()

        filtered["abs_energy"] = filtered["adsorption_energy"].astype(float).abs()
        filtered = filtered.sort_values(by="abs_energy", ascending=True).head(20)

        records: list[CatalystRecord] = []
        for idx, row in filtered.iterrows():
            payload: dict[str, Any] = row.to_dict()
            adsorption_energy_raw = payload.get("adsorption_energy")
            adsorption_energy = float(adsorption_energy_raw) if adsorption_energy_raw is not None else None

            records.append(
                CatalystRecord(
                    source=self.source_name,
                    source_id=str(payload.get("id") or idx),
                    name=str(payload.get("adsorbate") or "OC20 candidate"),
                    formula=str(payload.get("bulk_formula") or ""),
                    reaction=reaction,
                    activity_metric="adsorption_energy",
                    activity_value=adsorption_energy,
                    activity_unit="eV",
                    conditions={
                        "adsorbate": str(payload.get("adsorbate") or ""),
                        "temperature_k": int(payload["temperature_k"]) if payload.get("temperature_k") is not None else None,
                        "ph": float(payload["ph"]) if payload.get("ph") is not None else None,
                    },
                    stability=None,
                    raw=payload,
                )
            )

        logger.info("Open Catalyst returned {} candidates for tokens={}.", len(records), tokens[:6])
        return records

    def health_check(self) -> bool:
        """Check whether OC20 local data file is available and loadable.

        Args:
            None.

        Returns:
            True if dataset can be loaded and is non-empty.

        Raises:
            None.
        """
        try:
            df = self._load_data()
            return not df.empty
        except Exception:  # noqa: BLE001
            return False
