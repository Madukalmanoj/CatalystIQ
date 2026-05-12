"""Tests for processing layer — normalizer and cache."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import tempfile
import time
import pandas as pd
from processing.normalizer import normalize_units, flag_missing, canonicalize_formula
from processing.cache import QueryCache


# ── normalize_units ───────────────────────────────────────────────────────────

def test_normalize_units_empty():
    result = normalize_units(pd.DataFrame())
    assert result.empty


def test_normalize_units_mm_to_mol_per_l():
    df = pd.DataFrame([{"activity_value": 1.0, "activity_unit": "mM"}])
    result = normalize_units(df)
    assert result.iloc[0]["activity_value"] == pytest.approx(0.001)
    assert result.iloc[0]["activity_unit"] == "mol/L"


def test_normalize_units_mm_case_insensitive():
    df = pd.DataFrame([{"activity_value": 2.0, "activity_unit": "MM"}])
    result = normalize_units(df)
    assert result.iloc[0]["activity_value"] == pytest.approx(0.002)


def test_normalize_units_ev_to_kj_per_mol():
    df = pd.DataFrame([{"activity_value": 1.0, "activity_unit": "eV"}])
    result = normalize_units(df)
    assert result.iloc[0]["activity_value"] == pytest.approx(96.485)
    assert result.iloc[0]["activity_unit"] == "kJ/mol"


def test_normalize_units_unknown_unit_unchanged():
    df = pd.DataFrame([{"activity_value": 5.0, "activity_unit": "TOF"}])
    result = normalize_units(df)
    assert result.iloc[0]["activity_value"] == pytest.approx(5.0)
    assert result.iloc[0]["activity_unit"] == "TOF"


def test_normalize_units_mixed():
    df = pd.DataFrame([
        {"activity_value": 1.0, "activity_unit": "mM"},
        {"activity_value": 1.0, "activity_unit": "eV"},
        {"activity_value": 1.0, "activity_unit": "eV/atom"},
    ])
    result = normalize_units(df)
    assert result.iloc[0]["activity_unit"] == "mol/L"
    assert result.iloc[1]["activity_unit"] == "kJ/mol"
    assert result.iloc[2]["activity_unit"] == "eV/atom"


def test_normalize_units_no_columns():
    df = pd.DataFrame([{"name": "test"}])
    result = normalize_units(df)
    assert "name" in result.columns


def test_normalize_units_non_numeric_value():
    df = pd.DataFrame([{"activity_value": "not_a_number", "activity_unit": "mM"}])
    result = normalize_units(df)
    # Should not crash; value becomes NaN
    import math
    assert math.isnan(result.iloc[0]["activity_value"])


# ── flag_missing ──────────────────────────────────────────────────────────────

def test_flag_missing_empty():
    result = flag_missing(pd.DataFrame())
    assert result.empty


def test_flag_missing_ok_row():
    df = pd.DataFrame([{
        "formula": "Fe2O3", "source": "Materials Project", "activity_value": -1.2
    }])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "ok"


def test_flag_missing_no_activity():
    df = pd.DataFrame([{
        "formula": "Fe2O3", "source": "Materials Project", "activity_value": None
    }])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "missing_fields"


def test_flag_missing_no_formula_mp():
    df = pd.DataFrame([{
        "formula": "", "source": "Materials Project", "activity_value": -1.2
    }])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "missing_fields"


def test_flag_missing_no_formula_brenda_ok():
    """BRENDA records have no formula by design — should not be flagged."""
    df = pd.DataFrame([{
        "formula": "", "source": "BRENDA", "activity_value": 0.15
    }])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "ok"


def test_flag_missing_brenda_no_activity():
    df = pd.DataFrame([{
        "formula": "", "source": "BRENDA", "activity_value": None
    }])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "missing_fields"


def test_flag_missing_adds_column():
    df = pd.DataFrame([{"formula": "Pt", "source": "MP", "activity_value": -0.5}])
    result = flag_missing(df)
    assert "data_quality" in result.columns


def test_flag_missing_mixed():
    df = pd.DataFrame([
        {"formula": "Fe2O3", "source": "Materials Project", "activity_value": -1.2},
        {"formula": "",      "source": "Materials Project", "activity_value": -0.9},
        {"formula": "",      "source": "BRENDA",            "activity_value": 0.15},
        {"formula": "NiO",   "source": "Open Catalyst",     "activity_value": None},
    ])
    result = flag_missing(df)
    assert result.iloc[0]["data_quality"] == "ok"
    assert result.iloc[1]["data_quality"] == "missing_fields"
    assert result.iloc[2]["data_quality"] == "ok"
    assert result.iloc[3]["data_quality"] == "missing_fields"


# ── canonicalize_formula ──────────────────────────────────────────────────────

def test_canonicalize_formula_empty():
    assert canonicalize_formula("") == ""


def test_canonicalize_formula_none():
    assert canonicalize_formula(None) == ""


def test_canonicalize_formula_returns_string():
    result = canonicalize_formula("Fe2O3")
    assert isinstance(result, str)


# ── QueryCache ────────────────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test_cache.sqlite")
    return QueryCache(db_path, ttl_seconds=60)


def test_cache_miss_returns_none(cache):
    result = cache.get("TestSource", {"query": "test"})
    assert result is None


def test_cache_set_and_get(cache):
    cache.set("TestSource", {"query": "test"}, {"data": [1, 2, 3]})
    result = cache.get("TestSource", {"query": "test"})
    assert result == {"data": [1, 2, 3]}


def test_cache_different_keys(cache):
    cache.set("SourceA", {"q": "a"}, {"val": 1})
    cache.set("SourceB", {"q": "a"}, {"val": 2})
    assert cache.get("SourceA", {"q": "a"}) == {"val": 1}
    assert cache.get("SourceB", {"q": "a"}) == {"val": 2}


def test_cache_overwrite(cache):
    cache.set("S", {"q": "x"}, {"v": 1})
    cache.set("S", {"q": "x"}, {"v": 2})
    assert cache.get("S", {"q": "x"}) == {"v": 2}


def test_cache_ttl_expiry(tmp_path):
    db_path = str(tmp_path / "ttl_cache.sqlite")
    short_cache = QueryCache(db_path, ttl_seconds=1)
    short_cache.set("S", {"q": "expire"}, {"v": 99})
    assert short_cache.get("S", {"q": "expire"}) == {"v": 99}
    time.sleep(2.0)
    assert short_cache.get("S", {"q": "expire"}) is None


def test_cache_key_deterministic(cache):
    k1 = QueryCache._compute_key("S", {"b": 2, "a": 1})
    k2 = QueryCache._compute_key("S", {"a": 1, "b": 2})
    assert k1 == k2  # sort_keys=True in JSON


def test_cache_complex_value(cache):
    val = {"entries": [{"ec": "1.1.1.1", "km": 0.15}, {"ec": "2.7.1.1", "km": 7.5}]}
    cache.set("BRENDA", {"queries": [["1.1.1.1", "org", "sub"]]}, val)
    result = cache.get("BRENDA", {"queries": [["1.1.1.1", "org", "sub"]]})
    assert result == val
