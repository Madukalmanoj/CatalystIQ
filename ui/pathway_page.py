"""Pathway visualisation page — rich visual redesign.

Charts:
  Energy tab:
    1. Filled area energy profile with TS spikes, shaded barrier region,
       colour-coded zones (exothermic/endothermic), and annotated arrows.
    2. Step-by-step ΔG waterfall bar chart.

  Pathway tab:
    3. Sankey diagram with brighter link colours and node type legend.
    4. Horizontal flux lollipop with bottleneck highlighting.
    5. Gene engineering recommendation cards.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai.pathway import (
    MetabolicPathway,
    ReactionEnergyProfile,
    build_energy_profile,
    build_metabolic_pathway,
)

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
)
_AXIS = dict(
    gridcolor="rgba(255,255,255,0.07)",
    zerolinecolor="rgba(255,255,255,0.18)",
    color="#8892a4",
    tickfont=dict(size=11),
)


# ── Energy profile — filled area + TS spikes ──────────────────────────────────

def _build_energy_figure(profile: ReactionEnergyProfile) -> go.Figure:
    steps = profile.steps
    x = [s.step_index for s in steps]
    y = [s.delta_g for s in steps]
    labels = [s.label for s in steps]

    fig = go.Figure()

    # ── Shaded fill under the curve (split exo/endo) ──────────────────────────
    # Exothermic region (y < 0) — green tint
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=[min(v, 0) for v in y] + [0] * len(x),
        fill="toself",
        fillcolor="rgba(63,185,80,0.07)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Endothermic region (y > 0) — red tint
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=[max(v, 0) for v in y] + [0] * len(x),
        fill="toself",
        fillcolor="rgba(248,81,73,0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Zero baseline ─────────────────────────────────────────────────────────
    fig.add_hline(
        y=0,
        line=dict(color="rgba(255,255,255,0.20)", dash="dot", width=1),
    )

    # ── Main pathway line ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Energy pathway",
        line=dict(color="#00d4ff", width=2.5, shape="spline", smoothing=0.3),
        hovertemplate="<b>%{text}</b><br>ΔG = %{y:.3f} eV<extra></extra>",
        text=labels,
        showlegend=True,
    ))

    # ── Intermediate minima (non-TS) ──────────────────────────────────────────
    int_x = [s.step_index for s in steps if not s.is_transition_state]
    int_y = [s.delta_g for s in steps if not s.is_transition_state]
    int_labels = [s.label for s in steps if not s.is_transition_state]
    fig.add_trace(go.Scatter(
        x=int_x, y=int_y,
        mode="markers",
        name="Intermediates",
        marker=dict(
            color="#00d4ff", size=10,
            symbol="circle",
            line=dict(width=2, color="rgba(255,255,255,0.5)"),
        ),
        text=int_labels,
        hovertemplate="<b>%{text}</b><br>ΔG = %{y:.3f} eV<extra></extra>",
    ))

    # ── Transition state spikes ───────────────────────────────────────────────
    ts_steps = [s for s in steps if s.is_transition_state]
    for ts in ts_steps:
        # Vertical spike line
        fig.add_shape(
            type="line",
            x0=ts.step_index, x1=ts.step_index,
            y0=ts.delta_g - 0.08, y1=ts.delta_g,
            line=dict(color="rgba(248,81,73,0.6)", width=1.5, dash="dot"),
        )

    ts_x = [s.step_index for s in ts_steps]
    ts_y = [s.delta_g for s in ts_steps]
    ts_labels = [s.label for s in ts_steps]
    fig.add_trace(go.Scatter(
        x=ts_x, y=ts_y,
        mode="markers",
        name="Transition states",
        marker=dict(
            color="#f85149", size=14,
            symbol="diamond",
            line=dict(width=2, color="rgba(255,255,255,0.4)"),
        ),
        text=ts_labels,
        hovertemplate="<b>TS: %{text}</b><br>ΔG = %{y:.3f} eV<extra></extra>",
    ))

    # ── Activation barrier bracket annotation ────────────────────────────────
    if ts_steps:
        highest_ts = max(ts_steps, key=lambda s: s.delta_g)
        # Find the preceding minimum
        preceding_min = min(
            (s.delta_g for s in steps[:highest_ts.step_index] if not s.is_transition_state),
            default=0.0,
        )
        # Double-headed arrow for barrier
        fig.add_annotation(
            x=highest_ts.step_index,
            y=highest_ts.delta_g + 0.05,
            text=f"<b>Eₐ = {profile.activation_barrier:.3f} eV</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#f85149",
            arrowwidth=1.5,
            font=dict(color="#f85149", size=11, family="Inter, sans-serif"),
            bgcolor="rgba(248,81,73,0.12)",
            bordercolor="rgba(248,81,73,0.3)",
            borderwidth=1,
            borderpad=4,
            ax=55, ay=-35,
        )

    # ── Overall ΔG annotation ─────────────────────────────────────────────────
    last = steps[-1]
    dg_color = "#3fb950" if last.delta_g < 0 else "#f85149"
    fig.add_annotation(
        x=last.step_index,
        y=last.delta_g,
        text=f"<b>ΔG = {profile.overall_delta_g:.3f} eV</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor=dg_color,
        arrowwidth=1.5,
        font=dict(color=dg_color, size=11, family="Inter, sans-serif"),
        bgcolor=f"rgba({_hex_rgb(dg_color)},0.12)",
        bordercolor=f"rgba({_hex_rgb(dg_color)},0.3)",
        borderwidth=1,
        borderpad=4,
        ax=-60, ay=35,
    )

    fig.update_layout(
        **_DARK,
        margin=dict(t=64, b=80, l=64, r=32),
        title=dict(
            text=(
                f"Reaction Energy Profile — {profile.reaction[:55]}"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "🔴 Transition states · 🔵 Intermediates · "
                "Green fill = exothermic · Red fill = endothermic"
                "</sup>"
            ),
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(steps))),
            ticktext=[s.label for s in steps],
            tickangle=-38,
            **_AXIS,
        ),
        yaxis=dict(title="Cumulative ΔG (eV)", **_AXIS),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=460,
    )
    return fig


# ── Waterfall ΔG chart ────────────────────────────────────────────────────────

def _build_waterfall(profile: ReactionEnergyProfile) -> go.Figure:
    """Waterfall bar chart showing incremental ΔG per step."""
    steps = profile.steps
    # Compute incremental deltas from cumulative
    increments = [steps[0].delta_g]
    for i in range(1, len(steps)):
        increments.append(round(steps[i].delta_g - steps[i - 1].delta_g, 5))

    colors = []
    for i, step in enumerate(steps):
        if step.is_transition_state:
            colors.append("#f85149")
        elif increments[i] < 0:
            colors.append("#3fb950")
        else:
            colors.append("rgba(0,212,255,0.6)")

    fig = go.Figure(go.Bar(
        x=[s.label for s in steps],
        y=increments,
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        hovertemplate="<b>%{x}</b><br>Incremental ΔG: %{y:.3f} eV<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.20)", width=1))

    fig.update_layout(
        **_DARK,
        margin=dict(t=48, b=80, l=64, r=24),
        title=dict(
            text=(
                "Step-by-step ΔG Increments"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "🔴 Transition state · 🟢 Exothermic step · 🔵 Endothermic step"
                "</sup>"
            ),
            font=dict(size=13, color="#e6edf3"),
        ),
        xaxis=dict(tickangle=-38, **_AXIS),
        yaxis=dict(title="Incremental ΔG (eV)", **_AXIS),
        height=320,
    )
    return fig


# ── Sankey flux diagram ───────────────────────────────────────────────────────

def _build_flux_sankey(pathway: MetabolicPathway) -> go.Figure:
    node_ids = [n.id for n in pathway.nodes]
    node_labels = [n.label for n in pathway.nodes]

    node_colors = []
    for n in pathway.nodes:
        if n.is_bottleneck:
            node_colors.append("rgba(248,81,73,0.90)")
        elif n.node_type == "enzyme":
            node_colors.append("rgba(163,113,247,0.85)")
        else:
            node_colors.append("rgba(0,212,255,0.75)")

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    sources, targets, values, link_labels, link_colors = [], [], [], [], []
    for edge in pathway.edges:
        if edge.source in id_to_idx and edge.target in id_to_idx:
            sources.append(id_to_idx[edge.source])
            targets.append(id_to_idx[edge.target])
            values.append(max(edge.flux, 0.01))
            link_labels.append(edge.reaction_name)
            # Colour links by target node type
            tgt_node = next((n for n in pathway.nodes if n.id == edge.target), None)
            if tgt_node and tgt_node.is_bottleneck:
                link_colors.append("rgba(248,81,73,0.25)")
            elif tgt_node and tgt_node.node_type == "enzyme":
                link_colors.append("rgba(163,113,247,0.20)")
            else:
                link_colors.append("rgba(0,212,255,0.15)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=22,
            thickness=22,
            line=dict(color="rgba(255,255,255,0.15)", width=0.8),
            label=node_labels,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Node type shown by colour<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Relative flux: %{value:.2f}<extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        **_DARK,
        margin=dict(t=64, b=24, l=24, r=24),
        title=dict(
            text=(
                f"Metabolic Flux — {pathway.name}"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "🔵 Metabolite · 🟣 Enzyme · 🔴 Bottleneck"
                "</sup>"
            ),
            font=dict(size=14, color="#e6edf3"),
        ),
        height=500,
    )
    return fig


# ── Flux lollipop ─────────────────────────────────────────────────────────────

def _build_flux_lollipop(pathway: MetabolicPathway) -> go.Figure:
    """Horizontal lollipop showing relative flux per node, bottlenecks highlighted."""
    nodes = sorted(pathway.nodes, key=lambda n: n.flux, reverse=True)
    labels = [n.label for n in nodes]
    fluxes = [n.flux for n in nodes]
    colors = [
        "#f85149" if n.is_bottleneck else
        "#a371f7" if n.node_type == "enzyme" else
        "#00d4ff"
        for n in nodes
    ]

    fig = go.Figure()

    # Stems
    for i, (node, flux, color) in enumerate(zip(nodes, fluxes, colors)):
        fig.add_trace(go.Scatter(
            x=[0, flux], y=[i, i],
            mode="lines",
            line=dict(color=f"rgba({_hex_rgb(color)},0.25)", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Dots
    fig.add_trace(go.Scatter(
        x=fluxes, y=list(range(len(nodes))),
        mode="markers+text",
        marker=dict(
            color=colors, size=14,
            line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
        ),
        text=[f"  {f:.2f}" for f in fluxes],
        textposition="middle right",
        textfont=dict(size=9, color="#c9d1d9"),
        hovertemplate=[
            f"<b>{n.label}</b><br>Flux: {n.flux:.2f}<br>"
            f"{'⚠️ Bottleneck' if n.is_bottleneck else n.node_type.capitalize()}"
            "<extra></extra>"
            for n in nodes
        ],
        showlegend=False,
    ))

    # Bottleneck threshold line
    if any(n.is_bottleneck for n in nodes):
        bn_flux = max(n.flux for n in nodes if n.is_bottleneck)
        fig.add_vline(
            x=bn_flux,
            line=dict(color="rgba(248,81,73,0.4)", dash="dash", width=1.5),
            annotation_text="Bottleneck threshold",
            annotation_font=dict(color="#f85149", size=10),
            annotation_position="top right",
        )

    fig.update_layout(
        **_DARK,
        margin=dict(t=48, b=32, l=24, r=80),
        title=dict(
            text=(
                "Relative Flux per Node"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "🔴 Bottleneck · 🟣 Enzyme · 🔵 Metabolite"
                "</sup>"
            ),
            font=dict(size=13, color="#e6edf3"),
        ),
        xaxis=dict(title="Relative flux (0–1)", range=[-0.02, 1.25], **_AXIS),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(nodes))),
            ticktext=labels,
            color="#8892a4",
            tickfont=dict(size=10),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        height=max(300, len(nodes) * 30 + 80),
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ── Main render ───────────────────────────────────────────────────────────────

def render_pathway_page(reaction: str) -> None:
    """Render the pathway analysis tab."""
    st.markdown(
        """
        <div style="padding:0.8rem 0 1rem 0;">
            <h3 style="margin:0;color:#e6edf3;">⚡ Reaction & Pathway Analysis</h3>
            <p style="color:#8892a4;font-size:0.85rem;margin:4px 0 0 0;">
                Estimated reaction energy profiles and metabolic pathway flux maps
                for the target reaction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not reaction.strip():
        st.info("Enter a target reaction in the sidebar to see pathway analysis.")
        return

    tab_energy, tab_pathway = st.tabs([
        "⚡ Reaction Energy Profile",
        "🧬 Metabolic Pathway (Synthetic Biology)",
    ])

    # ── Reaction energy profile ───────────────────────────────────────────────
    with tab_energy:
        profile = build_energy_profile(reaction)

        # KPI row
        kpi_cols = st.columns(3)
        for col, label, val, color, sub in [
            (kpi_cols[0], "Activation Barrier",
             f"{profile.activation_barrier:.3f} eV", "#f85149",
             "energy to overcome rate-limiting TS"),
            (kpi_cols[1], "Overall ΔG",
             f"{profile.overall_delta_g:.3f} eV",
             "#3fb950" if profile.overall_delta_g < 0 else "#f85149",
             "exothermic" if profile.overall_delta_g < 0 else "endothermic"),
            (kpi_cols[2], "Rate-Limiting Step",
             profile.rate_limiting_step[:28], "#f0a500",
             "highest transition state"),
        ]:
            with col:
                st.markdown(
                    f"""<div style="background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.09);
                        border-left:3px solid {color};
                        border-radius:12px;padding:0.9rem 1rem;">
                        <p style="margin:0 0 3px 0;font-size:0.68rem;color:#8892a4;
                            letter-spacing:0.07em;text-transform:uppercase;">{label}</p>
                        <p style="margin:0 0 2px 0;font-size:1.15rem;font-weight:700;
                            color:{color};line-height:1.2;">{val}</p>
                        <p style="margin:0;font-size:0.68rem;color:#8892a4;">{sub}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Main energy diagram
        fig_energy = _build_energy_figure(profile)
        st.plotly_chart(fig_energy, use_container_width=True)

        # Waterfall
        fig_wf = _build_waterfall(profile)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown(
            "<p style='color:#8892a4;font-size:0.78rem;'>"
            "⚠️ Energy values are literature-grounded estimates for representative catalysts. "
            "In production these would be replaced by DFT calculations or kinetic Monte Carlo outputs.</p>",
            unsafe_allow_html=True,
        )

        with st.expander("📋 Step-by-step energy table"):
            step_data = [
                {
                    "Step": s.step_index,
                    "Label": s.label,
                    "Cumulative ΔG (eV)": round(s.delta_g, 4),
                    "Type": "⚡ Transition State" if s.is_transition_state else "● Intermediate",
                }
                for s in profile.steps
            ]
            st.dataframe(
                pd.DataFrame(step_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Step": st.column_config.NumberColumn(width="small"),
                    "Cumulative ΔG (eV)": st.column_config.NumberColumn(format="%.4f"),
                },
            )

    # ── Metabolic pathway ─────────────────────────────────────────────────────
    with tab_pathway:
        pathway = build_metabolic_pathway(reaction)

        # Header card
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
                <p style="margin:0 0 4px 0;font-size:1rem;font-weight:600;color:#e6edf3;">
                    {pathway.name}
                </p>
                <p style="margin:0 0 8px 0;font-size:0.82rem;color:#8892a4;">
                    {pathway.description}
                </p>
                <div style="display:flex;gap:24px;flex-wrap:wrap;">
                    <span style="font-size:0.82rem;">
                        🧫 <strong style="color:#a371f7;">Chassis:</strong>
                        <span style="color:#c9d1d9;">{pathway.organism}</span>
                    </span>
                    <span style="font-size:0.82rem;">
                        📈 <strong style="color:#3fb950;">Predicted yield:</strong>
                        <span style="color:#3fb950;font-weight:700;">{pathway.predicted_yield:.0%}</span>
                    </span>
                    <span style="font-size:0.82rem;">
                        ⚠️ <strong style="color:#f85149;">Bottlenecks:</strong>
                        <span style="color:#f85149;">{len(pathway.bottlenecks)}</span>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gene engineering recommendations
        col_bn, col_ko, col_oe = st.columns(3)

        with col_bn:
            st.markdown(
                "<p style='font-size:0.78rem;color:#8892a4;text-transform:uppercase;"
                "letter-spacing:0.06em;margin-bottom:6px;'>🔴 Bottleneck Nodes</p>",
                unsafe_allow_html=True,
            )
            for bn in pathway.bottlenecks:
                node = next((n for n in pathway.nodes if n.id == bn), None)
                label = node.label if node else bn
                st.markdown(
                    f"<div style='background:rgba(248,81,73,0.10);border:1px solid rgba(248,81,73,0.25);"
                    f"border-radius:8px;padding:5px 10px;margin-bottom:4px;"
                    f"font-size:0.82rem;color:#f85149;'>⚠️ {label}</div>",
                    unsafe_allow_html=True,
                )

        with col_ko:
            st.markdown(
                "<p style='font-size:0.78rem;color:#8892a4;text-transform:uppercase;"
                "letter-spacing:0.06em;margin-bottom:6px;'>✂️ Gene Knockouts</p>",
                unsafe_allow_html=True,
            )
            for ko in pathway.suggested_knockouts:
                st.markdown(
                    f"<div style='background:rgba(240,165,0,0.10);border:1px solid rgba(240,165,0,0.25);"
                    f"border-radius:8px;padding:5px 10px;margin-bottom:4px;"
                    f"font-size:0.82rem;color:#f0a500;'>✂ {ko}</div>",
                    unsafe_allow_html=True,
                )

        with col_oe:
            st.markdown(
                "<p style='font-size:0.78rem;color:#8892a4;text-transform:uppercase;"
                "letter-spacing:0.06em;margin-bottom:6px;'>⬆️ Overexpressions</p>",
                unsafe_allow_html=True,
            )
            for gene in pathway.suggested_overexpressions:
                st.markdown(
                    f"<div style='background:rgba(63,185,80,0.10);border:1px solid rgba(63,185,80,0.25);"
                    f"border-radius:8px;padding:5px 10px;margin-bottom:4px;"
                    f"font-size:0.82rem;color:#3fb950;'>⬆ {gene}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Sankey
        fig_sankey = _build_flux_sankey(pathway)
        st.plotly_chart(fig_sankey, use_container_width=True)

        # Flux lollipop
        fig_lollipop = _build_flux_lollipop(pathway)
        st.plotly_chart(fig_lollipop, use_container_width=True)

        st.markdown(
            "<p style='color:#8892a4;font-size:0.78rem;'>"
            "⚠️ Flux values are relative estimates based on literature-curated pathway data. "
            "In production these would be replaced by constraint-based flux balance analysis (FBA).</p>",
            unsafe_allow_html=True,
        )
