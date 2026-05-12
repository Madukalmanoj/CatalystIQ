"""Reaction energy profile estimator and metabolic pathway mapper.

Provides:
- ReactionEnergyProfile: estimates activation barriers and intermediate
  energies for common catalytic reactions using literature-grounded values.
- MetabolicPathwayMapper: maps synthetic biology routes for bio-based
  fuel/chemical production with flux estimates and bottleneck detection.

All values are estimates for demonstration. In a production system these
would be replaced by DFT calculations or kinetic Monte Carlo outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Reaction energy profile data
# Each entry: reaction_key → list of (step_label, ΔG_eV, is_transition_state)
# Values are representative literature estimates (eV, 298 K, standard state)
# ---------------------------------------------------------------------------
_ENERGY_PROFILES: dict[str, list[tuple[str, float, bool]]] = {
    "co2_methanol": [
        ("CO₂(g) + H₂(g)",          0.00,  False),
        ("CO₂* adsorption",         -0.18,  False),
        ("HCOO* formation",          0.42,  True),
        ("HCOO* → H₂COO*",          0.31,  True),
        ("H₂COO* → H₂CO* + O*",     0.28,  True),
        ("H₂CO* + H* → H₃CO*",      0.15,  True),
        ("H₃CO* + H* → CH₃OH*",     0.09,  True),
        ("CH₃OH* desorption",        0.22,  False),
        ("CH₃OH(g)",                -0.65,  False),
    ],
    "co_methane": [
        ("CO(g) + H₂(g)",            0.00,  False),
        ("CO* adsorption",           -0.52,  False),
        ("CO* + H* → HCO*",          0.38,  True),
        ("HCO* + H* → H₂CO*",        0.29,  True),
        ("H₂CO* + H* → H₃CO*",       0.21,  True),
        ("H₃CO* + H* → CH₄* + O*",   0.18,  True),
        ("CH₄ desorption",            0.12,  False),
        ("CH₄(g)",                   -0.74,  False),
    ],
    "n2_nh3": [
        ("N₂(g) + H₂(g)",            0.00,  False),
        ("N₂* adsorption",           -0.08,  False),
        ("N₂* dissociation → 2N*",    1.05,  True),
        ("N* + H* → NH*",             0.68,  True),
        ("NH* + H* → NH₂*",           0.42,  True),
        ("NH₂* + H* → NH₃*",          0.28,  True),
        ("NH₃* desorption",           0.18,  False),
        ("NH₃(g)",                   -0.92,  False),
    ],
    "ethanol_jet": [
        ("Ethanol(g)",                0.00,  False),
        ("Dehydration → ethylene",    0.55,  True),
        ("Ethylene oligomerisation",  0.38,  True),
        ("C4-C8 alkenes",            -0.22,  False),
        ("Hydrogenation",             0.19,  True),
        ("C4-C8 alkanes",            -0.48,  False),
        ("Isomerisation",             0.24,  True),
        ("Jet-range hydrocarbons",   -0.71,  False),
    ],
    "syngas_ethanol": [
        ("CO + H₂ (syngas)",          0.00,  False),
        ("CO* + H* → HCO*",           0.45,  True),
        ("HCO* + CO* → CH₃CO*",       0.52,  True),
        ("CH₃CO* + H* → CH₃CHO*",     0.31,  True),
        ("CH₃CHO* + H* → C₂H₅O*",    0.22,  True),
        ("C₂H₅O* + H* → C₂H₅OH*",    0.14,  True),
        ("Ethanol desorption",         0.19,  False),
        ("Ethanol(g)",                -0.88,  False),
    ],
    "cellulose_hydrocarbons": [
        ("Cellulose",                  0.00,  False),
        ("Hydrolysis → glucose",       0.35,  True),
        ("Glucose → HMF",              0.48,  True),
        ("HMF → levulinic acid",       0.29,  True),
        ("Levulinic acid → GVL",       0.41,  True),
        ("GVL → pentanoic acid",       0.33,  True),
        ("Pentanoic acid → alkenes",   0.52,  True),
        ("Alkenes → hydrocarbons",    -0.62,  False),
    ],
}

_REACTION_KEYWORDS: dict[str, list[str]] = {
    "co2_methanol":           ["co2", "methanol", "co₂"],
    "co_methane":             ["methane", "methanation"],
    "n2_nh3":                 ["n2", "nh3", "nitrogen", "ammonia", "haber"],
    "ethanol_jet":            ["jet", "atj", "aviation"],
    "syngas_ethanol":         ["syngas"],
    "cellulose_hydrocarbons": ["cellulose", "biomass", "lignocellulose", "hmf"],
}

# Ordered by specificity — checked first wins on tie
_PROFILE_PRIORITY = [
    "cellulose_hydrocarbons",
    "n2_nh3",
    "co2_methanol",
    "syngas_ethanol",
    "ethanol_jet",
    "co_methane",
]


def _match_profile(reaction: str) -> str:
    """Return the best-matching profile key for a reaction string."""
    # Normalise: replace arrows and special chars, lowercase
    normalised = reaction.lower().replace("→", " ").replace("->", " ")
    tokens = set(re.split(r"[^a-z0-9]+", normalised))
    tokens.discard("")

    scores: dict[str, int] = {}
    for key in _PROFILE_PRIORITY:
        keywords = _REACTION_KEYWORDS[key]
        score = sum(1 for kw in keywords if kw in tokens)
        scores[key] = score

    best_score = max(scores.values())
    if best_score == 0:
        return "co2_methanol"  # default

    # Return highest-scoring key, using priority order to break ties
    for key in _PROFILE_PRIORITY:
        if scores[key] == best_score:
            return key
    return "co2_methanol"


@dataclass
class EnergyStep:
    label: str
    delta_g: float          # cumulative ΔG from reactants (eV)
    is_transition_state: bool
    step_index: int


@dataclass
class ReactionEnergyProfile:
    reaction: str
    profile_key: str
    steps: list[EnergyStep]
    activation_barrier: float   # highest TS above preceding minimum (eV)
    overall_delta_g: float      # reactants → products (eV)
    rate_limiting_step: str

    def to_plotly_traces(self) -> list[dict[str, Any]]:
        """Return Plotly-compatible trace dicts for the energy diagram."""
        x_vals = [s.step_index for s in self.steps]
        y_vals = [s.delta_g for s in self.steps]
        labels = [s.label for s in self.steps]

        # Main pathway line
        line_trace = {
            "type": "scatter",
            "x": x_vals,
            "y": y_vals,
            "mode": "lines+markers",
            "name": "Energy pathway",
            "line": {"color": "#00d4ff", "width": 2.5},
            "marker": {"size": 8, "color": "#00d4ff"},
            "text": labels,
            "hovertemplate": "<b>%{text}</b><br>ΔG = %{y:.3f} eV<extra></extra>",
        }

        # Highlight transition states
        ts_x = [s.step_index for s in self.steps if s.is_transition_state]
        ts_y = [s.delta_g for s in self.steps if s.is_transition_state]
        ts_labels = [s.label for s in self.steps if s.is_transition_state]
        ts_trace = {
            "type": "scatter",
            "x": ts_x,
            "y": ts_y,
            "mode": "markers",
            "name": "Transition states",
            "marker": {"size": 14, "color": "#f85149", "symbol": "diamond"},
            "text": ts_labels,
            "hovertemplate": "<b>TS: %{text}</b><br>ΔG = %{y:.3f} eV<extra></extra>",
        }

        return [line_trace, ts_trace]


def build_energy_profile(reaction: str) -> ReactionEnergyProfile:
    """Build a reaction energy profile for the given reaction string.

    Args:
        reaction: Target reaction text.

    Returns:
        ReactionEnergyProfile with cumulative ΔG steps.
    """
    key = _match_profile(reaction)
    raw_steps = _ENERGY_PROFILES[key]

    # Convert relative ΔG to cumulative
    steps: list[EnergyStep] = []
    cumulative = 0.0
    for i, (label, delta, is_ts) in enumerate(raw_steps):
        cumulative += delta
        steps.append(EnergyStep(
            label=label,
            delta_g=round(cumulative, 4),
            is_transition_state=is_ts,
            step_index=i,
        ))

    # Find activation barrier: max TS height above preceding minimum
    min_so_far = 0.0
    max_barrier = 0.0
    rls = steps[0].label
    for step in steps:
        if not step.is_transition_state:
            min_so_far = min(min_so_far, step.delta_g)
        else:
            barrier = step.delta_g - min_so_far
            if barrier > max_barrier:
                max_barrier = barrier
                rls = step.label

    return ReactionEnergyProfile(
        reaction=reaction,
        profile_key=key,
        steps=steps,
        activation_barrier=round(max_barrier, 4),
        overall_delta_g=round(steps[-1].delta_g, 4),
        rate_limiting_step=rls,
    )


# ---------------------------------------------------------------------------
# Metabolic pathway mapper (Synthetic Biology)
# ---------------------------------------------------------------------------

@dataclass
class PathwayNode:
    id: str
    label: str
    node_type: str          # "metabolite" | "enzyme" | "gene"
    flux: float             # relative flux (0–1)
    is_bottleneck: bool
    organism: str
    notes: str = ""


@dataclass
class PathwayEdge:
    source: str
    target: str
    reaction_name: str
    flux: float
    reversible: bool = False


@dataclass
class MetabolicPathway:
    name: str
    description: str
    nodes: list[PathwayNode]
    edges: list[PathwayEdge]
    bottlenecks: list[str]
    suggested_knockouts: list[str]
    suggested_overexpressions: list[str]
    predicted_yield: float      # mol product / mol substrate (0–1)
    organism: str


_PATHWAY_LIBRARY: dict[str, dict[str, Any]] = {
    "ethanol_hydrocarbons": {
        "name": "Ethanol → Jet-range Hydrocarbons",
        "description": (
            "Engineered pathway converting ethanol to C8–C16 hydrocarbons "
            "via dehydration, oligomerisation, and hydrogenation. "
            "Suitable for Saccharomyces cerevisiae or E. coli chassis."
        ),
        "organism": "Saccharomyces cerevisiae",
        "predicted_yield": 0.61,
        "nodes": [
            ("ethanol",      "Ethanol",              "metabolite", 1.00, False),
            ("acetaldehyde", "Acetaldehyde",          "metabolite", 0.85, False),
            ("acetyl_coa",   "Acetyl-CoA",            "metabolite", 0.80, False),
            ("malonyl_coa",  "Malonyl-CoA",           "metabolite", 0.42, True),
            ("fatty_acid",   "Fatty acid (C8–C16)",   "metabolite", 0.38, True),
            ("alkene",       "Terminal alkene",        "metabolite", 0.61, False),
            ("hydrocarbon",  "Jet hydrocarbon",        "metabolite", 0.61, False),
            ("adh",          "ADH (alcohol dehydrogenase)", "enzyme", 0.85, False),
            ("acs",          "ACS (acetyl-CoA synthetase)", "enzyme", 0.80, False),
            ("acc",          "ACC (acetyl-CoA carboxylase)", "enzyme", 0.42, True),
            ("fas",          "FAS (fatty acid synthase)",    "enzyme", 0.38, True),
            ("ole1",         "OLE1 (fatty acid desaturase)", "enzyme", 0.61, False),
            ("car",          "CAR (carboxylic acid reductase)", "enzyme", 0.61, False),
        ],
        "edges": [
            ("ethanol",      "adh",          "Ethanol oxidation",        0.85, True),
            ("adh",          "acetaldehyde", "→ Acetaldehyde",           0.85, False),
            ("acetaldehyde", "acs",          "Acetaldehyde activation",  0.80, False),
            ("acs",          "acetyl_coa",   "→ Acetyl-CoA",             0.80, False),
            ("acetyl_coa",   "acc",          "Carboxylation",            0.42, False),
            ("acc",          "malonyl_coa",  "→ Malonyl-CoA",            0.42, False),
            ("malonyl_coa",  "fas",          "Chain elongation",         0.38, False),
            ("fas",          "fatty_acid",   "→ Fatty acid",             0.38, False),
            ("fatty_acid",   "ole1",         "Desaturation",             0.61, False),
            ("ole1",         "alkene",       "→ Terminal alkene",        0.61, False),
            ("alkene",       "car",          "Reduction",                0.61, False),
            ("car",          "hydrocarbon",  "→ Jet hydrocarbon",        0.61, False),
        ],
        "knockouts": ["FAA1", "FAA4"],
        "overexpressions": ["ACC1", "FAS1", "FAS2", "OLE1"],
    },
    "biomass_fuels": {
        "name": "Biomass → Fuels & Chemicals",
        "description": (
            "Consolidated bioprocessing route from lignocellulosic biomass "
            "to ethanol and higher alcohols using Clostridium thermocellum chassis."
        ),
        "organism": "Clostridium thermocellum",
        "predicted_yield": 0.48,
        "nodes": [
            ("cellulose",    "Cellulose",             "metabolite", 1.00, False),
            ("glucose",      "Glucose",               "metabolite", 0.88, False),
            ("pyruvate",     "Pyruvate",              "metabolite", 0.75, False),
            ("acetyl_coa",   "Acetyl-CoA",            "metabolite", 0.68, False),
            ("acetaldehyde", "Acetaldehyde",          "metabolite", 0.55, True),
            ("ethanol",      "Ethanol",               "metabolite", 0.48, False),
            ("cel",          "Cellulase complex",     "enzyme",     0.88, False),
            ("pgm",          "Phosphoglucomutase",    "enzyme",     0.75, False),
            ("pdc",          "Pyruvate decarboxylase","enzyme",     0.55, True),
            ("adh",          "Alcohol dehydrogenase", "enzyme",     0.48, False),
        ],
        "edges": [
            ("cellulose",    "cel",          "Hydrolysis",               0.88, False),
            ("cel",          "glucose",      "→ Glucose",                0.88, False),
            ("glucose",      "pgm",          "Glycolysis",               0.75, False),
            ("pgm",          "pyruvate",     "→ Pyruvate",               0.75, False),
            ("pyruvate",     "acetyl_coa",   "Pyruvate oxidation",       0.68, False),
            ("acetyl_coa",   "pdc",          "Decarboxylation",          0.55, False),
            ("pdc",          "acetaldehyde", "→ Acetaldehyde",           0.55, False),
            ("acetaldehyde", "adh",          "Reduction",                0.48, False),
            ("adh",          "ethanol",      "→ Ethanol",                0.48, False),
        ],
        "knockouts": ["ldh", "pta"],
        "overexpressions": ["pdc", "adh", "cel"],
    },
    "co2_methanol_bio": {
        "name": "CO₂ → Methanol (Biological)",
        "description": (
            "Synthetic methylotrophic pathway in E. coli for direct CO₂ "
            "fixation to methanol using formate dehydrogenase and methanol "
            "dehydrogenase cascade."
        ),
        "organism": "Escherichia coli",
        "predicted_yield": 0.35,
        "nodes": [
            ("co2",          "CO₂",                  "metabolite", 1.00, False),
            ("formate",      "Formate",               "metabolite", 0.72, False),
            ("formaldehyde", "Formaldehyde",          "metabolite", 0.52, True),
            ("methanol",     "Methanol",              "metabolite", 0.35, False),
            ("fdh",          "Formate dehydrogenase", "enzyme",     0.72, False),
            ("faldh",        "Formaldehyde reductase","enzyme",     0.52, True),
            ("mdh",          "Methanol dehydrogenase","enzyme",     0.35, False),
        ],
        "edges": [
            ("co2",          "fdh",          "CO₂ reduction",            0.72, False),
            ("fdh",          "formate",      "→ Formate",                0.72, False),
            ("formate",      "faldh",        "Formate reduction",        0.52, False),
            ("faldh",        "formaldehyde", "→ Formaldehyde",           0.52, False),
            ("formaldehyde", "mdh",          "Formaldehyde reduction",   0.35, False),
            ("mdh",          "methanol",     "→ Methanol",               0.35, False),
        ],
        "knockouts": ["frmA", "frmB"],
        "overexpressions": ["fdh1", "faldh", "mdh2"],
    },
}

_PATHWAY_KEYWORDS: dict[str, list[str]] = {
    "ethanol_hydrocarbons": ["jet", "atj", "aviation", "alkene"],
    "biomass_fuels":        ["biomass", "cellulose", "lignocellulose", "clostridium", "cbp"],
    "co2_methanol_bio":     ["methylotrophic", "formate", "fdh", "methanol"],
}

# Priority order for tie-breaking
_PATHWAY_PRIORITY = ["biomass_fuels", "co2_methanol_bio", "ethanol_hydrocarbons"]


def _match_pathway(reaction: str) -> str:
    normalised = reaction.lower().replace("→", " ").replace("->", " ")
    tokens = set(re.split(r"[^a-z0-9]+", normalised))
    tokens.discard("")

    scores: dict[str, int] = {}
    for key in _PATHWAY_PRIORITY:
        keywords = _PATHWAY_KEYWORDS[key]
        score = sum(1 for kw in keywords if kw in tokens)
        scores[key] = score

    best_score = max(scores.values())
    if best_score == 0:
        return "ethanol_hydrocarbons"  # default

    for key in _PATHWAY_PRIORITY:
        if scores[key] == best_score:
            return key
    return "ethanol_hydrocarbons"


def build_metabolic_pathway(reaction: str) -> MetabolicPathway:
    """Build a metabolic pathway map for the given reaction.

    Args:
        reaction: Target reaction or process description.

    Returns:
        MetabolicPathway with nodes, edges, bottlenecks, and recommendations.
    """
    key = _match_pathway(reaction)
    data = _PATHWAY_LIBRARY[key]

    nodes: list[PathwayNode] = []
    for node_tuple in data["nodes"]:
        nid, label, ntype, flux, is_bn = node_tuple
        nodes.append(PathwayNode(
            id=nid,
            label=label,
            node_type=ntype,
            flux=flux,
            is_bottleneck=is_bn,
            organism=data["organism"],
        ))

    edges: list[PathwayEdge] = []
    for edge_tuple in data["edges"]:
        src, tgt, rname, flux, rev = edge_tuple
        edges.append(PathwayEdge(
            source=src,
            target=tgt,
            reaction_name=rname,
            flux=flux,
            reversible=rev,
        ))

    bottlenecks = [n.id for n in nodes if n.is_bottleneck]

    return MetabolicPathway(
        name=data["name"],
        description=data["description"],
        nodes=nodes,
        edges=edges,
        bottlenecks=bottlenecks,
        suggested_knockouts=data["knockouts"],
        suggested_overexpressions=data["overexpressions"],
        predicted_yield=data["predicted_yield"],
        organism=data["organism"],
    )
