"""Tests for ai.pathway module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from ai.pathway import (
    build_energy_profile,
    build_metabolic_pathway,
    ReactionEnergyProfile,
    MetabolicPathway,
    EnergyStep,
    PathwayNode,
    PathwayEdge,
    _match_profile,
    _match_pathway,
    _ENERGY_PROFILES,
    _PATHWAY_LIBRARY,
)


# ── _match_profile ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reaction,expected_key", [
    ("CO2 + H2 -> methanol",          "co2_methanol"),
    ("CO + H2 -> methane",            "co_methane"),
    ("N2 + H2 -> NH3",                "n2_nh3"),
    ("ethanol -> jet fuel",           "ethanol_jet"),
    ("syngas -> ethanol",             "syngas_ethanol"),
    ("cellulose -> hydrocarbons",     "cellulose_hydrocarbons"),
    ("completely unknown reaction",   "co2_methanol"),  # default
])
def test_match_profile(reaction, expected_key):
    assert _match_profile(reaction) == expected_key


# ── _match_pathway ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reaction,expected_key", [
    ("ethanol -> jet fuel hydrocarbons", "ethanol_hydrocarbons"),
    ("biomass cellulose -> fuels",       "biomass_fuels"),
    ("CO2 -> methanol biological",       "co2_methanol_bio"),
])
def test_match_pathway(reaction, expected_key):
    assert _match_pathway(reaction) == expected_key


# ── build_energy_profile ──────────────────────────────────────────────────────

@pytest.mark.parametrize("reaction", [
    "CO2 + H2 -> methanol",
    "N2 + H2 -> NH3",
    "ethanol -> jet fuel",
    "CO + H2 -> methane",
    "syngas -> ethanol",
    "cellulose -> hydrocarbons",
    "completely unknown reaction xyz",
])
def test_build_energy_profile_returns_object(reaction):
    profile = build_energy_profile(reaction)
    assert isinstance(profile, ReactionEnergyProfile)


def test_energy_profile_has_steps():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    assert len(profile.steps) > 0


def test_energy_profile_steps_are_EnergyStep():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    for step in profile.steps:
        assert isinstance(step, EnergyStep)


def test_energy_profile_first_step_zero():
    """First step (reactants) should have cumulative ΔG = 0."""
    profile = build_energy_profile("CO2 + H2 -> methanol")
    # First step starts at 0 + first delta
    raw = _ENERGY_PROFILES["co2_methanol"]
    assert profile.steps[0].delta_g == pytest.approx(raw[0][1], abs=1e-4)


def test_energy_profile_activation_barrier_positive():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    assert profile.activation_barrier >= 0.0


def test_energy_profile_has_transition_states():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    ts_steps = [s for s in profile.steps if s.is_transition_state]
    assert len(ts_steps) > 0


def test_energy_profile_rate_limiting_step_is_string():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    assert isinstance(profile.rate_limiting_step, str)
    assert len(profile.rate_limiting_step) > 0


def test_energy_profile_step_indices_sequential():
    profile = build_energy_profile("N2 + H2 -> NH3")
    for i, step in enumerate(profile.steps):
        assert step.step_index == i


def test_energy_profile_to_plotly_traces():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    traces = profile.to_plotly_traces()
    assert isinstance(traces, list)
    assert len(traces) == 2  # line + TS markers
    for trace in traces:
        assert "type" in trace
        assert trace["type"] == "scatter"
        assert "x" in trace
        assert "y" in trace


def test_energy_profile_plotly_x_y_same_length():
    profile = build_energy_profile("CO2 + H2 -> methanol")
    traces = profile.to_plotly_traces()
    line_trace = traces[0]
    assert len(line_trace["x"]) == len(line_trace["y"])
    assert len(line_trace["x"]) == len(profile.steps)


# ── build_metabolic_pathway ───────────────────────────────────────────────────

@pytest.mark.parametrize("reaction", [
    "ethanol -> jet fuel hydrocarbons",
    "biomass cellulose -> fuels",
    "CO2 -> methanol biological",
    "completely unknown reaction xyz",
])
def test_build_metabolic_pathway_returns_object(reaction):
    pathway = build_metabolic_pathway(reaction)
    assert isinstance(pathway, MetabolicPathway)


def test_pathway_has_nodes():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    assert len(pathway.nodes) > 0


def test_pathway_has_edges():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    assert len(pathway.edges) > 0


def test_pathway_nodes_are_PathwayNode():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    for node in pathway.nodes:
        assert isinstance(node, PathwayNode)


def test_pathway_edges_are_PathwayEdge():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    for edge in pathway.edges:
        assert isinstance(edge, PathwayEdge)


def test_pathway_flux_range():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    for node in pathway.nodes:
        assert 0.0 <= node.flux <= 1.0, f"flux={node.flux} out of [0,1]"


def test_pathway_predicted_yield_range():
    for key in _PATHWAY_LIBRARY:
        reaction = list(_PATHWAY_LIBRARY[key]["name"].split("→")[0].strip().lower().split())[0]
        pathway = build_metabolic_pathway(reaction)
        assert 0.0 <= pathway.predicted_yield <= 1.0


def test_pathway_bottlenecks_are_node_ids():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    node_ids = {n.id for n in pathway.nodes}
    for bn in pathway.bottlenecks:
        assert bn in node_ids, f"Bottleneck {bn!r} not in node IDs"


def test_pathway_edge_references_valid_nodes():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    node_ids = {n.id for n in pathway.nodes}
    for edge in pathway.edges:
        assert edge.source in node_ids, f"Edge source {edge.source!r} not in nodes"
        assert edge.target in node_ids, f"Edge target {edge.target!r} not in nodes"


def test_pathway_has_knockouts_and_overexpressions():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    assert isinstance(pathway.suggested_knockouts, list)
    assert isinstance(pathway.suggested_overexpressions, list)
    assert len(pathway.suggested_knockouts) > 0
    assert len(pathway.suggested_overexpressions) > 0


def test_pathway_organism_is_string():
    pathway = build_metabolic_pathway("ethanol -> jet fuel")
    assert isinstance(pathway.organism, str)
    assert len(pathway.organism) > 0
