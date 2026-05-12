"""Tests for retrieval layer — demo data, OC20, aggregator, base models."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from retrieval.base import CatalystRecord, BaseRetriever, RetrieverConnectionError, RetrieverParseError
from retrieval.demo_data import (
    build_demo_candidates,
    demo_candidates_for_query,
    brenda_demo_records,
    DEMO_SOURCES,
    DEMO_FORMULAS,
)
from retrieval.aggregator import CatalystAggregator


# ── CatalystRecord ────────────────────────────────────────────────────────────

def test_catalyst_record_defaults():
    r = CatalystRecord(source="Test", reaction="A -> B")
    assert r.source_id == ""
    assert r.name == ""
    assert r.formula == ""
    assert r.activity_value is None
    assert r.stability is None
    assert r.conditions == {}
    assert r.raw == {}


def test_catalyst_record_model_dump():
    r = CatalystRecord(
        source="MP", reaction="CO2 -> CH4",
        formula="Fe2O3", activity_value=-1.2, activity_unit="eV/atom",
    )
    d = r.model_dump()
    assert d["source"] == "MP"
    assert d["formula"] == "Fe2O3"
    assert d["activity_value"] == pytest.approx(-1.2)


def test_catalyst_record_conditions_dict():
    r = CatalystRecord(
        source="BRENDA", reaction="test",
        conditions={"organism": "E. coli", "ph": 7.0},
    )
    assert r.conditions["organism"] == "E. coli"


# ── build_demo_candidates ─────────────────────────────────────────────────────

def test_build_demo_candidates_returns_df():
    df = build_demo_candidates(limit=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_build_demo_candidates_columns():
    df = build_demo_candidates(limit=5)
    required = ["source", "formula", "reaction", "activity_value", "activity_unit", "conditions"]
    for col in required:
        assert col in df.columns


def test_build_demo_candidates_sources():
    df = build_demo_candidates(limit=300)
    sources = set(df["source"].unique())
    assert sources == set(DEMO_SOURCES)


def test_build_demo_candidates_deterministic():
    df1 = build_demo_candidates(limit=50)
    df2 = build_demo_candidates(limit=50)
    pd.testing.assert_frame_equal(df1, df2)


def test_build_demo_candidates_limit():
    for limit in [1, 10, 50, 120]:
        df = build_demo_candidates(limit=limit)
        assert len(df) <= limit


def test_build_demo_candidates_activity_values_numeric():
    df = build_demo_candidates(limit=30)
    assert pd.to_numeric(df["activity_value"], errors="coerce").notna().all()


# ── demo_candidates_for_query ─────────────────────────────────────────────────

def test_demo_candidates_for_query_returns_df():
    df = demo_candidates_for_query(
        reaction="CO2 + H2 -> methanol",
        selected_sources=["Materials Project"],
        temperature_range=(200, 1500),
        ph_range=(0.0, 14.0),
        limit=20,
    )
    assert isinstance(df, pd.DataFrame)


def test_demo_candidates_for_query_source_filter():
    df = demo_candidates_for_query(
        reaction="CO2 + H2 -> methanol",
        selected_sources=["BRENDA"],
        temperature_range=(200, 1500),
        ph_range=(0.0, 14.0),
        limit=50,
    )
    if not df.empty:
        assert all(df["source"] == "BRENDA")


def test_demo_candidates_for_query_limit():
    df = demo_candidates_for_query(
        reaction="N2 + H2 -> NH3",
        selected_sources=list(DEMO_SOURCES),
        temperature_range=(200, 1500),
        ph_range=(0.0, 14.0),
        limit=10,
    )
    assert len(df) <= 10


def test_demo_candidates_for_query_temperature_filter():
    df = demo_candidates_for_query(
        reaction="CO2 + H2 -> methanol",
        selected_sources=list(DEMO_SOURCES),
        temperature_range=(500, 600),
        ph_range=(0.0, 14.0),
        limit=100,
    )
    if not df.empty:
        temps = df["conditions"].map(lambda c: c.get("temperature_k", 0))
        assert all(500 <= t <= 600 for t in temps)


def test_demo_candidates_for_query_ph_filter():
    df = demo_candidates_for_query(
        reaction="CO2 + H2 -> methanol",
        selected_sources=list(DEMO_SOURCES),
        temperature_range=(200, 1500),
        ph_range=(6.0, 8.0),
        limit=100,
    )
    if not df.empty:
        phs = df["conditions"].map(lambda c: float(c.get("ph", 7.0)))
        assert all(6.0 <= p <= 8.0 for p in phs)


# ── brenda_demo_records ───────────────────────────────────────────────────────

def test_brenda_demo_records_returns_list():
    records = brenda_demo_records("N2 + H2 -> NH3")
    assert isinstance(records, list)
    assert len(records) > 0


def test_brenda_demo_records_are_catalyst_records():
    records = brenda_demo_records("CO2 + H2 -> methanol")
    for r in records:
        assert isinstance(r, CatalystRecord)


def test_brenda_demo_records_source():
    records = brenda_demo_records("glucose -> ethanol")
    for r in records:
        assert r.source == "BRENDA"


def test_brenda_demo_records_km_positive():
    records = brenda_demo_records("glucose -> ethanol")
    for r in records:
        if r.activity_value is not None:
            assert r.activity_value > 0, f"Km should be positive, got {r.activity_value}"


def test_brenda_demo_records_limit():
    records = brenda_demo_records("test", limit=3)
    assert len(records) <= 3


def test_brenda_demo_records_conditions():
    records = brenda_demo_records("N2 + H2 -> NH3")
    for r in records:
        assert "organism" in r.conditions
        assert "substrate" in r.conditions
        assert "ec_number" in r.conditions


def test_brenda_demo_records_relevance_ordering():
    """Nitrogenase should rank high for N2 reaction."""
    records = brenda_demo_records("N2 + H2 -> NH3")
    names = [r.name for r in records]
    assert any("nitrogenase" in n.lower() for n in names[:3])


# ── CatalystAggregator ────────────────────────────────────────────────────────

class _MockRetriever(BaseRetriever):
    source_name = "Mock"

    def __init__(self, records, fail=False):
        self._records = records
        self._fail = fail

    def search(self, reaction):
        if self._fail:
            raise RuntimeError("Mock failure")
        return self._records

    def health_check(self):
        return not self._fail


def test_aggregator_combines_sources():
    r1 = CatalystRecord(source="A", reaction="test", formula="Fe2O3", activity_value=-1.0)
    r2 = CatalystRecord(source="B", reaction="test", formula="NiO",   activity_value=-0.8)
    ret_a = _MockRetriever([r1])
    ret_b = _MockRetriever([r2])
    agg = CatalystAggregator([ret_a, ret_b])
    df = agg.retrieve("test")
    assert len(df) == 2
    assert set(df["source"].tolist()) == {"A", "B"}


def test_aggregator_handles_failure():
    r1 = CatalystRecord(source="A", reaction="test", formula="Fe2O3", activity_value=-1.0)
    ret_ok = _MockRetriever([r1])
    ret_fail = _MockRetriever([], fail=True)
    ret_fail.source_name = "Failing"
    agg = CatalystAggregator([ret_ok, ret_fail])
    df = agg.retrieve("test")
    assert len(df) == 1
    assert df.iloc[0]["source"] == "A"


def test_aggregator_empty_retrievers():
    agg = CatalystAggregator([])
    df = agg.retrieve("test")
    assert df.empty


def test_aggregator_deduplicates():
    r1 = CatalystRecord(source="A", reaction="test", formula="Fe2O3", activity_value=-1.0)
    r2 = CatalystRecord(source="A", reaction="test", formula="Fe2O3", activity_value=-1.0)
    ret = _MockRetriever([r1, r2])
    agg = CatalystAggregator([ret])
    df = agg.retrieve("test")
    assert len(df) == 1


def test_aggregator_sorted_by_activity():
    records = [
        CatalystRecord(source="A", reaction="test", formula="Fe2O3", activity_value=-0.5),
        CatalystRecord(source="A", reaction="test", formula="NiO",   activity_value=-1.5),
        CatalystRecord(source="A", reaction="test", formula="CuO",   activity_value=-1.0),
    ]
    ret = _MockRetriever(records)
    agg = CatalystAggregator([ret])
    df = agg.retrieve("test")
    vals = df["activity_value"].tolist()
    assert vals == sorted(vals)


def test_aggregator_returns_dataframe():
    ret = _MockRetriever([])
    agg = CatalystAggregator([ret])
    df = agg.retrieve("test")
    assert isinstance(df, pd.DataFrame)


def test_aggregator_empty_results_has_columns():
    ret = _MockRetriever([])
    agg = CatalystAggregator([ret])
    df = agg.retrieve("test")
    assert "source" in df.columns
    assert "formula" in df.columns
