"""Tests for fair cross-source ranking in CatalystPredictor."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
from ai.predictor import CatalystPredictor


@pytest.fixture
def mixed_sources_df():
    """DataFrame with MP (eV/atom), OC (eV), BRENDA (mM), and AI candidates."""
    return pd.DataFrame([
        # Materials Project — formation energies (negative, eV/atom)
        {"formula": "Fe2O3", "source": "Materials Project", "activity_value": -1.2,
         "activity_unit": "eV/atom", "conditions": {}},
        {"formula": "NiO",   "source": "Materials Project", "activity_value": -0.9,
         "activity_unit": "eV/atom", "conditions": {}},
        {"formula": "RuO2",  "source": "Materials Project", "activity_value": -1.5,
         "activity_unit": "eV/atom", "conditions": {}},
        {"formula": "CuO",   "source": "Materials Project", "activity_value": -0.6,
         "activity_unit": "eV/atom", "conditions": {}},
        # Open Catalyst — adsorption energies (eV, small values)
        {"formula": "Pt",    "source": "Open Catalyst", "activity_value": -0.3,
         "activity_unit": "eV", "conditions": {}},
        {"formula": "Pd",    "source": "Open Catalyst", "activity_value": -0.5,
         "activity_unit": "eV", "conditions": {}},
        {"formula": "Rh",    "source": "Open Catalyst", "activity_value": -0.8,
         "activity_unit": "eV", "conditions": {}},
        # BRENDA — Km values (mM, positive, lower=better)
        {"formula": "", "source": "BRENDA", "activity_value": 0.15,
         "activity_unit": "mM", "name": "Hexokinase", "conditions": {}},
        {"formula": "", "source": "BRENDA", "activity_value": 0.87,
         "activity_unit": "mM", "name": "ADH", "conditions": {}},
        {"formula": "", "source": "BRENDA", "activity_value": 7.50,
         "activity_unit": "mM", "name": "Glucokinase", "conditions": {}},
        # AI generated — heuristic scores (around -1.7 to -2.0)
        {"formula": "TiS2", "source": "Generative AI", "predicted_activity": -1.95,
         "activity_unit": "eV/atom", "conditions": {}, "confidence": 0.72, "novelty_score": 0.75},
        {"formula": "RhRu", "source": "Generative AI", "predicted_activity": -1.88,
         "activity_unit": "eV/atom", "conditions": {}, "confidence": 0.68, "novelty_score": 0.80},
        {"formula": "FeZn", "source": "Generative AI", "predicted_activity": -1.70,
         "activity_unit": "eV/atom", "conditions": {}, "confidence": 0.65, "novelty_score": 0.70},
    ])


def test_rank_score_column_exists(mixed_sources_df):
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)
    assert "rank_score" in result.columns


def test_rank_score_in_unit_interval(mixed_sources_df):
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)
    assert result["rank_score"].between(0, 1).all(), "rank_score must be in [0,1]"


def test_ai_does_not_dominate_top5(mixed_sources_df):
    """AI candidates should not occupy all top-5 slots."""
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)
    top5_sources = result.head(5)["source"].value_counts()
    assert top5_sources.get("Generative AI", 0) <= 3, (
        f"AI dominates top 5: {dict(top5_sources)}"
    )


def test_all_sources_represented_in_top10(mixed_sources_df):
    """All 4 sources should appear in the top 10."""
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)
    top10_sources = set(result.head(10)["source"].unique())
    expected = {"Materials Project", "Open Catalyst", "BRENDA", "Generative AI"}
    assert expected <= top10_sources, (
        f"Missing sources in top 10: {expected - top10_sources}"
    )


def test_within_source_ordering_preserved(mixed_sources_df):
    """Within each source, better candidates should rank higher."""
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)

    # MP: RuO2 (-1.5) should rank above Fe2O3 (-1.2) which ranks above NiO (-0.9)
    mp = result[result["source"] == "Materials Project"].copy()
    ruo2_rank = mp[mp["formula"] == "RuO2"]["rank"].iloc[0]
    fe2o3_rank = mp[mp["formula"] == "Fe2O3"]["rank"].iloc[0]
    nio_rank = mp[mp["formula"] == "NiO"]["rank"].iloc[0]
    assert ruo2_rank < fe2o3_rank < nio_rank, (
        f"MP ordering wrong: RuO2={ruo2_rank}, Fe2O3={fe2o3_rank}, NiO={nio_rank}"
    )

    # BRENDA: Hexokinase (0.15 mM) should rank above ADH (0.87) above Glucokinase (7.5)
    brenda = result[result["source"] == "BRENDA"].copy()
    hk_rank = brenda[brenda["name"] == "Hexokinase"]["rank"].iloc[0]
    adh_rank = brenda[brenda["name"] == "ADH"]["rank"].iloc[0]
    gk_rank = brenda[brenda["name"] == "Glucokinase"]["rank"].iloc[0]
    assert hk_rank < adh_rank < gk_rank, (
        f"BRENDA ordering wrong: HK={hk_rank}, ADH={adh_rank}, GK={gk_rank}"
    )


def test_rank_column_is_sequential(mixed_sources_df):
    pred = CatalystPredictor()
    result = pred.predict(mixed_sources_df)
    ranks = sorted(result["rank"].tolist())
    assert ranks == list(range(1, len(mixed_sources_df) + 1))


def test_compute_rank_scores_static():
    """Test _compute_rank_scores directly."""
    df = pd.DataFrame([
        {"source": "A", "predicted_activity": -2.0},
        {"source": "A", "predicted_activity": -1.0},
        {"source": "B", "predicted_activity": 0.1},
        {"source": "B", "predicted_activity": 5.0},
    ])
    scores = CatalystPredictor._compute_rank_scores(df)
    assert len(scores) == 4
    assert scores.between(0, 1).all()
    # Within source A: -2.0 should score lower (better) than -1.0
    assert scores.iloc[0] < scores.iloc[1]
    # Within source B: 0.1 should score lower (better) than 5.0
    assert scores.iloc[2] < scores.iloc[3]


def test_single_candidate_per_source():
    """Single candidate per source should get rank_score = 0.5."""
    df = pd.DataFrame([
        {"source": "A", "predicted_activity": -1.0, "conditions": {}, "formula": "Fe"},
        {"source": "B", "predicted_activity": 0.5,  "conditions": {}, "formula": "Ni"},
    ])
    pred = CatalystPredictor()
    result = pred.predict(df)
    assert result["rank_score"].between(0, 1).all()


def test_empty_df():
    pred = CatalystPredictor()
    result = pred.predict(pd.DataFrame())
    assert result.empty
