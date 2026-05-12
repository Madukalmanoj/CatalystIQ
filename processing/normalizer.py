"""Data normalization helpers for units, formulas, and quality flags."""

from __future__ import annotations

import pandas as pd
from loguru import logger

EV_TO_KJ_PER_MOL = 96.485

try:
    from pymatgen.core import Composition
except Exception:  # noqa: BLE001
    Composition = None


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common units to consistent forms for comparison.

    Args:
        df: Candidate dataframe containing activity columns.

    Returns:
        Updated dataframe with normalized units and values.

    Raises:
        None.
    """
    if df.empty:
        return df

    normalized = df.copy()
    if "activity_unit" not in normalized.columns or "activity_value" not in normalized.columns:
        return normalized

    normalized["activity_value"] = pd.to_numeric(normalized["activity_value"], errors="coerce")

    mm_mask = normalized["activity_unit"].astype(str).str.lower().eq("mm")
    normalized.loc[mm_mask, "activity_value"] = normalized.loc[mm_mask, "activity_value"] / 1000.0
    normalized.loc[mm_mask, "activity_unit"] = "mol/L"

    ev_mask = normalized["activity_unit"].astype(str).str.lower().eq("ev")
    normalized.loc[ev_mask, "activity_value"] = normalized.loc[ev_mask, "activity_value"] * EV_TO_KJ_PER_MOL
    normalized.loc[ev_mask, "activity_unit"] = "kJ/mol"

    return normalized


def canonicalize_formula(formula_str: str) -> str:
    """Canonicalize chemical formula when pymatgen is available.

    Args:
        formula_str: Input formula string.

    Returns:
        Canonicalized formula string or stripped original value.

    Raises:
        None.
    """
    formula = (formula_str or "").strip()
    if not formula:
        return formula

    if Composition is None:
        return formula

    try:
        return Composition(formula).reduced_formula
    except Exception as exc:  # noqa: BLE001
        logger.debug("Formula canonicalization failed for '{}': {}", formula, exc)
        return formula


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add data quality status based on required values.

    Formula is only required for material/catalyst sources (Materials Project,
    Open Catalyst). BRENDA enzyme records have no formula by design — they are
    flagged only if activity_value is missing.

    Args:
        df: Candidate dataframe with formula and activity_value columns.

    Returns:
        Dataframe with a data_quality column.

    Raises:
        None.
    """
    flagged = df.copy()
    if flagged.empty:
        flagged["data_quality"] = pd.Series(dtype=str)
        return flagged

    # Sources where an empty formula is expected and acceptable
    _FORMULA_OPTIONAL_SOURCES = {"BRENDA"}

    formula_col = flagged.get("formula", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    source_col = flagged.get("source", pd.Series(dtype=str)).fillna("").astype(str)

    # Formula missing only counts against sources that should have one
    formula_missing = (
        formula_col.eq("")
        & ~source_col.isin(_FORMULA_OPTIONAL_SOURCES)
    )
    activity_missing = flagged.get("activity_value", pd.Series(dtype=float)).isna()

    flagged["data_quality"] = "ok"
    flagged.loc[formula_missing | activity_missing, "data_quality"] = "missing_fields"

    if "formula" in flagged.columns:
        flagged["formula"] = flagged["formula"].astype(str).map(canonicalize_formula)

    return flagged
