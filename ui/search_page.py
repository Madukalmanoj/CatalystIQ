"""Streamlit search sidebar — source-aware filters that visibly change results."""

from __future__ import annotations

from typing import Any

import streamlit as st

from retrieval.base import BaseRetriever

SOURCE_ICONS: dict[str, str] = {
    "Materials Project": "⚗️",
    "BRENDA":            "🧬",
    "Open Catalyst":     "⚡",
}

# Presets: (max_formation_energy, max_energy_above_hull, max_adsorption_energy_abs, max_km_mM, min_ph, max_ph)
PRESETS: dict[str, dict[str, Any]] = {
    "Broad Screen": {
        "mp_max_formation_energy":   0.0,
        "mp_max_hull":               0.5,
        "oc_max_adsorption_abs":     200.0,
        "brenda_max_km":             200.0,
        "brenda_ph_range":           (0.0, 14.0),
        "label": "No restrictions — show everything",
    },
    "Stable Materials Only": {
        "mp_max_formation_energy":  -0.5,
        "mp_max_hull":               0.05,
        "oc_max_adsorption_abs":     200.0,
        "brenda_max_km":             200.0,
        "brenda_ph_range":           (0.0, 14.0),
        "label": "MP: only highly stable materials (hull < 0.05 eV)",
    },
    "Strong Binders (OC)": {
        "mp_max_formation_energy":   0.0,
        "mp_max_hull":               0.5,
        "oc_max_adsorption_abs":     2.0,
        "brenda_max_km":             200.0,
        "brenda_ph_range":           (0.0, 14.0),
        "label": "OC: adsorption energy |E| ≤ 2 eV (optimal binding)",
    },
    "High Affinity Enzymes": {
        "mp_max_formation_energy":   0.0,
        "mp_max_hull":               0.5,
        "oc_max_adsorption_abs":     200.0,
        "brenda_max_km":             1.0,
        "brenda_ph_range":           (6.0, 8.0),
        "label": "BRENDA: Km ≤ 1 mM (high substrate affinity), neutral pH",
    },
    "Physiological Conditions": {
        "mp_max_formation_energy":  -0.3,
        "mp_max_hull":               0.1,
        "oc_max_adsorption_abs":     5.0,
        "brenda_max_km":             10.0,
        "brenda_ph_range":           (6.5, 7.8),
        "label": "Balanced: stable materials + moderate binding + physiological pH",
    },
}


def _source_ready(source_name: str, retriever: BaseRetriever) -> bool:
    """Check if a retriever is configured without making network calls."""
    try:
        import config
        if source_name == "Materials Project":
            return bool(config.MP_API_KEY)
        if source_name == "BRENDA":
            return bool(config.BRENDA_EMAIL and config.BRENDA_PASSWORD)
        if source_name == "Open Catalyst":
            from pathlib import Path
            data_dir = Path(config.OCP_DATA_DIR)
            return any(
                (data_dir / name).exists()
                for name in ("oc20_data_mapping.pkl", "oc20_data_mapping.csv")
            )
    except Exception:  # noqa: BLE001
        pass
    return False


def render_search_sidebar(
    retrievers: dict[str, BaseRetriever],
) -> tuple[str, list[str], dict[str, Any], bool]:
    """Render sidebar controls and return user search selections.

    Args:
        retrievers: Mapping from source label to retriever instance.

    Returns:
        Tuple of (reaction, selected_sources, filters, run_clicked).

    Raises:
        None.
    """
    if "reaction_input" not in st.session_state:
        st.session_state["reaction_input"] = ""

    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding:1rem 0 0.5rem 0;">
                <span style="font-size:2.2rem;">⚗️</span>
                <h2 style="margin:0; font-size:1.4rem; font-weight:700;
                    background:linear-gradient(135deg,#00d4ff,#7b2ff7);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    CatalystIQ
                </h2>
                <p style="margin:0; font-size:0.72rem; color:#8892a4; letter-spacing:0.08em;">
                    MOLECULAR DISCOVERY ENGINE
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Quick examples ────────────────────────────────────────────────────
        st.markdown("##### 💡 Quick Examples")
        examples = [
            "CO2 + H2 → methanol",
            "N2 + H2 → NH3",
            "Fe + O2 → Fe2O3",
            "CO + H2 → methane",
            "glucose + ATP → glucose-6-phosphate + ADP",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state["reaction_input"] = ex

        st.markdown("##### 🎯 Target Reaction")
        reaction = st.text_input(
            "Target Reaction",
            placeholder="CO2 + H2 → methanol",
            label_visibility="collapsed",
            key="reaction_input",
        )
        st.divider()

        # ── Data sources ──────────────────────────────────────────────────────
        st.markdown("##### 🗄️ Data Sources")
        source_names = list(retrievers.keys())
        selected_sources = st.multiselect(
            "Data Sources",
            options=source_names,
            default=source_names,
            label_visibility="collapsed",
        )
        st.divider()

        # ── Filter preset ─────────────────────────────────────────────────────
        st.markdown("##### 🎛️ Filter Preset")
        selected_preset = st.selectbox(
            "Filter Preset",
            options=list(PRESETS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        preset = PRESETS[selected_preset]
        st.markdown(
            f"<p style='color:#8892a4; font-size:0.75rem; margin:2px 0 8px 0;'>"
            f"ℹ️ {preset['label']}</p>",
            unsafe_allow_html=True,
        )

        # ── Advanced filters (source-aware) ───────────────────────────────────
        with st.expander("⚙️ Advanced Filters", expanded=False):

            st.markdown(
                "<p style='color:#00d4ff; font-size:0.78rem; margin:0 0 4px 0;'>⚗️ Materials Project</p>",
                unsafe_allow_html=True,
            )
            mp_max_formation_energy = st.slider(
                "Max formation energy (eV/atom)",
                min_value=-5.0, max_value=0.0,
                value=float(preset["mp_max_formation_energy"]),
                step=0.1,
                key=f"mp_fe_{selected_preset}",
                help="Only keep materials with formation energy below this value. More negative = more stable.",
            )
            mp_max_hull = st.slider(
                "Max energy above hull (eV)",
                min_value=0.0, max_value=0.5,
                value=float(preset["mp_max_hull"]),
                step=0.01,
                key=f"mp_hull_{selected_preset}",
                help="0 = perfectly stable. < 0.1 eV = practically stable.",
            )

            st.markdown(
                "<p style='color:#3fb950; font-size:0.78rem; margin:8px 0 4px 0;'>⚡ Open Catalyst</p>",
                unsafe_allow_html=True,
            )
            oc_max_adsorption_abs = st.slider(
                "Max |adsorption energy| (eV)",
                min_value=0.0, max_value=10.0,
                value=float(preset["oc_max_adsorption_abs"]),
                step=0.5,
                key=f"oc_ads_{selected_preset}",
                help="Optimal catalysts bind neither too strongly nor too weakly. 0.5–3 eV is typically ideal.",
            )

            st.markdown(
                "<p style='color:#a371f7; font-size:0.78rem; margin:8px 0 4px 0;'>🧬 BRENDA</p>",
                unsafe_allow_html=True,
            )
            brenda_max_km = st.slider(
                "Max Km (mM) — lower = higher affinity",
                min_value=0.01, max_value=200.0,
                value=float(preset["brenda_max_km"]),
                step=0.5,
                key=f"brenda_km_{selected_preset}",
                help="Km is substrate affinity. Lower Km = enzyme binds substrate more tightly.",
            )
            brenda_ph_range = st.slider(
                "Optimal pH range",
                min_value=0.0, max_value=14.0,
                value=(float(preset["brenda_ph_range"][0]), float(preset["brenda_ph_range"][1])),
                step=0.1,
                key=f"brenda_ph_{selected_preset}",
                help="Filter enzymes by their optimal operating pH.",
            )

            st.divider()
            allow_demo_fallback = st.checkbox(
                "Allow demo fallback",
                value=False,
                help="Show synthetic demo data when live sources return no results.",
            )

        st.divider()

        # ── Source health ─────────────────────────────────────────────────────
        st.markdown("##### 🔌 Source Health")
        for source_name, retriever in retrievers.items():
            ready = _source_ready(source_name, retriever)
            icon = SOURCE_ICONS.get(source_name, "🔬")
            dot = "🟢" if ready else "🔴"
            status_text = "Configured" if ready else "Not configured"
            status_color = "#3fb950" if ready else "#f85149"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"padding:4px 0;font-size:0.85rem;'>"
                f"{dot} {icon} <span style='color:#c9d1d9;'>{source_name}</span>"
                f"<span style='margin-left:auto;font-size:0.75rem;color:{status_color};'>"
                f"{status_text}</span></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        run_clicked = st.button(
            "🔍  Retrieve Candidates",
            type="primary",
            use_container_width=True,
        )

    filters: dict[str, Any] = {
        "preset": selected_preset,
        # Source-specific filters
        "mp_max_formation_energy":  mp_max_formation_energy,
        "mp_max_hull":              mp_max_hull,
        "oc_max_adsorption_abs":    oc_max_adsorption_abs,
        "brenda_max_km":            brenda_max_km,
        "brenda_ph_range":          brenda_ph_range,
        # Legacy keys kept for demo fallback compatibility
        "temperature_range":        (200, 1500),
        "ph_range":                 brenda_ph_range,
        "allow_demo_fallback":      allow_demo_fallback,
    }

    return reaction, selected_sources, filters, run_clicked
