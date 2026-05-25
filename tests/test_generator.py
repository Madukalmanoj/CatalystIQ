"""Tests for ai.generator module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from ai.generator import (
    CatalystGenerator,
    GeneratedCandidate,
    _parse_formula,
    _formula_from_dict,
    _novelty_score,
    _deterministic_float,
)


# ── _parse_formula ────────────────────────────────────────────────────────────

def test_parse_formula_simple():
    assert _parse_formula("Fe2O3") == {"Fe": 2, "O": 3}

def test_parse_formula_single():
    assert _parse_formula("Pt") == {"Pt": 1}

def test_parse_formula_empty():
    assert _parse_formula("") == {}

def test_parse_formula_multi():
    assert _parse_formula("Co3O4") == {"Co": 3, "O": 4}

def test_parse_formula_binary():
    assert _parse_formula("NiO") == {"Ni": 1, "O": 1}

def test_parse_formula_alloy():
    result = _parse_formula("FeNi")
    assert result == {"Fe": 1, "Ni": 1}


# ── _formula_from_dict ────────────────────────────────────────────────────────

def test_formula_from_dict_roundtrip():
    comp = {"Fe": 2, "O": 3}
    assert _formula_from_dict(comp) == "Fe2O3"

def test_formula_from_dict_single():
    assert _formula_from_dict({"Pt": 1}) == "Pt"

def test_formula_from_dict_alloy():
    result = _formula_from_dict({"Fe": 1, "Ni": 1})
    assert "Fe" in result and "Ni" in result


# ── _novelty_score ────────────────────────────────────────────────────────────

def test_novelty_known_is_zero():
    known = {"Fe2O3", "NiO"}
    assert _novelty_score("Fe2O3", known) == 0.0

def test_novelty_completely_new():
    known = {"Fe2O3", "NiO"}
    score = _novelty_score("RuO2", known)
    assert 0.5 <= score <= 1.0

def test_novelty_empty_known():
    score = _novelty_score("Fe2O3", set())
    assert score == 1.0


# ── _deterministic_float ─────────────────────────────────────────────────────

def test_deterministic_float_range():
    val = _deterministic_float("test_seed", 0.0, 1.0)
    assert 0.0 <= val <= 1.0

def test_deterministic_float_reproducible():
    v1 = _deterministic_float("same_seed", -1.0, 1.0)
    v2 = _deterministic_float("same_seed", -1.0, 1.0)
    assert v1 == v2

def test_deterministic_float_different_seeds():
    v1 = _deterministic_float("seed_a", 0.0, 1.0)
    v2 = _deterministic_float("seed_b", 0.0, 1.0)
    assert v1 != v2


# ── CatalystGenerator ─────────────────────────────────────────────────────────

@pytest.fixture
def mp_df():
    return pd.DataFrame([
        {"formula": "Fe2O3", "source": "Materials Project", "activity_value": -1.2,
         "activity_unit": "eV/atom", "activity_metric": "formation_energy_per_atom", "conditions": {}},
        {"formula": "NiO",   "source": "Materials Project", "activity_value": -0.9,
         "activity_unit": "eV/atom", "activity_metric": "formation_energy_per_atom", "conditions": {}},
        {"formula": "CuO",   "source": "Materials Project", "activity_value": -0.7,
         "activity_unit": "eV/atom", "activity_metric": "formation_energy_per_atom", "conditions": {}},
        {"formula": "Co3O4", "source": "Materials Project", "activity_value": -1.1,
         "activity_unit": "eV/atom", "activity_metric": "formation_energy_per_atom", "conditions": {}},
        {"formula": "RuO2",  "source": "Materials Project", "activity_value": -1.4,
         "activity_unit": "eV/atom", "activity_metric": "formation_energy_per_atom", "conditions": {}},
    ])


@pytest.fixture
def brenda_df():
    return pd.DataFrame([
        {"formula": "", "source": "BRENDA", "activity_value": 0.15, "activity_unit": "mM",
         "activity_metric": "Km", "name": "Hexokinase",
         "conditions": {"organism": "S. cerevisiae", "ec_number": "2.7.1.1"}},
        {"formula": "", "source": "BRENDA", "activity_value": 0.87, "activity_unit": "mM",
         "activity_metric": "Km", "name": "Alcohol dehydrogenase",
         "conditions": {"organism": "S. cerevisiae", "ec_number": "1.1.1.1"}},
        {"formula": "", "source": "BRENDA", "activity_value": 0.04, "activity_unit": "mM",
         "activity_metric": "Km", "name": "Nitrogenase",
         "conditions": {"organism": "A. vinelandii", "ec_number": "1.18.6.1"}},
    ])


def test_generate_returns_list(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    assert isinstance(result, list)


def test_generate_respects_n_candidates(mp_df):
    gen = CatalystGenerator(n_candidates=5)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    assert len(result) <= 5


def test_generate_empty_df():
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", pd.DataFrame())
    assert result == []


def test_generate_no_duplicates_with_known(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    known_formulas = set(mp_df["formula"].tolist())
    for c in result:
        if c.formula:
            assert c.formula not in known_formulas, f"{c.formula} is a known formula"


def test_generate_no_duplicate_formulas_in_output(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    formulas = [c.formula for c in result if c.formula]
    assert len(formulas) == len(set(formulas)), "Duplicate formulas in output"


def test_generate_candidate_fields_valid(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    for c in result:
        assert isinstance(c, GeneratedCandidate)
        assert c.predicted_activity is not None
        assert 0.0 <= c.confidence <= 1.0, f"confidence={c.confidence} out of [0,1]"
        assert 0.0 <= c.novelty_score <= 1.0, f"novelty={c.novelty_score} out of [0,1]"
        assert c.generation_strategy, "missing generation_strategy"
        assert c.rationale, "missing rationale"
        assert c.reaction == "CO2 + H2 -> methanol"


def test_generate_enzyme_mutations(brenda_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("ethanol -> hydrocarbons", brenda_df)
    enzyme_variants = [c for c in result if "enzyme" in c.generation_strategy.lower()]
    assert len(enzyme_variants) > 0, "No enzyme variants generated from BRENDA data"


def test_generate_to_dict_keys(mp_df):
    gen = CatalystGenerator(n_candidates=3)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    required = ["source", "name", "formula", "reaction", "predicted_activity",
                "confidence", "novelty_score", "generation_strategy", "rationale",
                "conditions", "stability", "data_quality"]
    for c in result:
        d = c.to_dict()
        missing = [k for k in required if k not in d]
        assert not missing, f"to_dict missing keys: {missing}"


def test_generate_sorted_by_activity(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", mp_df)
    activities = [c.predicted_activity for c in result if c.predicted_activity is not None]
    assert activities == sorted(activities), "Candidates not sorted by predicted_activity"


def test_generate_deterministic(mp_df):
    gen = CatalystGenerator(n_candidates=8)
    r1 = gen.generate("CO2 + H2 -> methanol", mp_df)
    r2 = gen.generate("CO2 + H2 -> methanol", mp_df)
    assert [c.name for c in r1] == [c.name for c in r2], "Generator not deterministic"


def test_generate_mixed_sources(mp_df, brenda_df):
    combined = pd.concat([mp_df, brenda_df], ignore_index=True)
    gen = CatalystGenerator(n_candidates=8)
    result = gen.generate("CO2 + H2 -> methanol", combined)
    assert len(result) > 0

GEMINI_API_KEY= Agvsdhsdsgvfsjhbdhasdvasdsa
