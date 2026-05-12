"""Tests for app.py logic functions (filter, source breakdown, fallback)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd


# Import the functions we want to test directly
# We need to avoid triggering Streamlit's page_config at import time,
# so we import the functions after patching st.set_page_config.
import unittest.mock as mock

# Patch streamlit before importing app
with mock.patch("streamlit.set_page_config"):
    import app as app_module


# ── apply_filters ─────────────────────────────────────────────────────────────

@pytest.fixture
def mixed_df():
    return pd.DataFrame([
        # Materials Project rows
        {"source": "Materials Project", "activity_value": -1.5, "activity_unit": "eV/atom",
         "stability": 0.02, "conditions": {}},
        {"source": "Materials Project", "activity_value": 0.5,  "activity_unit": "eV/atom",
         "stability": 0.6,  "conditions": {}},
        # Open Catalyst rows
        {"source": "Open Catalyst", "activity_value": -0.8, "activity_unit": "eV",
         "stability": None, "conditions": {}},
        {"source": "Open Catalyst", "activity_value": 300.0, "activity_unit": "eV",
         "stability": None, "conditions": {}},
        # BRENDA rows
        {"source": "BRENDA", "activity_value": 0.5,  "activity_unit": "mM",
         "stability": None, "conditions": {"ph": 7.0}},
        {"source": "BRENDA", "activity_value": 250.0, "activity_unit": "mM",
         "stability": None, "conditions": {"ph": 3.0}},
    ])


def _broad_filters():
    return {
        "mp_max_formation_energy": 0.0,
        "mp_max_hull": 0.5,
        "oc_max_adsorption_abs": 200.0,
        "brenda_max_km": 200.0,
        "brenda_ph_range": (0.0, 14.0),
    }


def test_apply_filters_broad_passes_all(mixed_df):
    result = app_module.apply_filters(mixed_df, _broad_filters())
    assert len(result) == len(mixed_df)


def test_apply_filters_empty_df():
    result = app_module.apply_filters(pd.DataFrame(), _broad_filters())
    assert result.empty


def test_apply_filters_mp_formation_energy(mixed_df):
    filters = _broad_filters()
    filters["mp_max_formation_energy"] = -1.0
    result = app_module.apply_filters(mixed_df, filters)
    mp_rows = result[result["source"] == "Materials Project"]
    for _, row in mp_rows.iterrows():
        assert float(row["activity_value"]) <= -1.0


def test_apply_filters_mp_hull(mixed_df):
    filters = _broad_filters()
    filters["mp_max_hull"] = 0.1
    result = app_module.apply_filters(mixed_df, filters)
    mp_rows = result[result["source"] == "Materials Project"]
    for _, row in mp_rows.iterrows():
        if row["stability"] is not None:
            assert float(row["stability"]) <= 0.1


def test_apply_filters_oc_adsorption(mixed_df):
    filters = _broad_filters()
    filters["oc_max_adsorption_abs"] = 1.0
    result = app_module.apply_filters(mixed_df, filters)
    oc_rows = result[result["source"] == "Open Catalyst"]
    for _, row in oc_rows.iterrows():
        assert abs(float(row["activity_value"])) <= 1.0


def test_apply_filters_brenda_km(mixed_df):
    filters = _broad_filters()
    filters["brenda_max_km"] = 1.0
    result = app_module.apply_filters(mixed_df, filters)
    brenda_rows = result[result["source"] == "BRENDA"]
    for _, row in brenda_rows.iterrows():
        assert float(row["activity_value"]) <= 1.0


def test_apply_filters_brenda_ph(mixed_df):
    filters = _broad_filters()
    filters["brenda_ph_range"] = (6.5, 7.5)
    result = app_module.apply_filters(mixed_df, filters)
    brenda_rows = result[result["source"] == "BRENDA"]
    for _, row in brenda_rows.iterrows():
        ph = row["conditions"].get("ph")
        if ph is not None:
            assert 6.5 <= float(ph) <= 7.5


def test_apply_filters_non_brenda_passes_ph_filter(mixed_df):
    """MP and OC rows should not be filtered by pH."""
    filters = _broad_filters()
    filters["brenda_ph_range"] = (6.5, 7.5)
    result = app_module.apply_filters(mixed_df, filters)
    mp_count = len(result[result["source"] == "Materials Project"])
    oc_count = len(result[result["source"] == "Open Catalyst"])
    # MP and OC rows should be unaffected by pH filter
    assert mp_count > 0
    assert oc_count > 0


def test_apply_filters_returns_original_if_all_excluded():
    """If all rows would be excluded, return original df."""
    df = pd.DataFrame([
        {"source": "BRENDA", "activity_value": 500.0, "stability": None,
         "conditions": {"ph": 2.0}},
    ])
    filters = _broad_filters()
    filters["brenda_max_km"] = 1.0
    result = app_module.apply_filters(df, filters)
    # Should return original since all would be excluded
    assert len(result) == len(df)


def test_apply_filters_null_activity_value():
    df = pd.DataFrame([
        {"source": "Materials Project", "activity_value": None, "stability": 0.02, "conditions": {}},
    ])
    filters = _broad_filters()
    filters["mp_max_formation_energy"] = -1.0
    # Should not crash on None activity_value
    result = app_module.apply_filters(df, filters)
    assert isinstance(result, pd.DataFrame)


# ── _source_breakdown ─────────────────────────────────────────────────────────

def test_source_breakdown_empty():
    result = app_module._source_breakdown(pd.DataFrame())
    assert "No sources" in result


def test_source_breakdown_single_source():
    df = pd.DataFrame([{"source": "Materials Project"}] * 3)
    result = app_module._source_breakdown(df)
    assert "Materials Project" in result
    assert "3" in result


def test_source_breakdown_multiple_sources():
    df = pd.DataFrame([
        {"source": "Materials Project"},
        {"source": "Materials Project"},
        {"source": "BRENDA"},
    ])
    result = app_module._source_breakdown(df)
    assert "Materials Project" in result
    assert "BRENDA" in result


def test_source_breakdown_no_source_column():
    df = pd.DataFrame([{"formula": "Fe2O3"}])
    result = app_module._source_breakdown(df)
    assert isinstance(result, str)
