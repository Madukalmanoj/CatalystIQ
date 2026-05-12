"""Generative design results page — rich visual redesign.

Charts:
  1. Bubble chart  — activity vs novelty, bubble size = confidence, colour = source/strategy
  2. Ranked lollipop — all candidates ranked, AI stars highlighted, confidence encoded
  3. Radar chart  — per-candidate multi-dimensional score comparison
  4. Strategy breakdown — donut showing generation strategy mix
  5. Candidate cards — structured, colour-coded, with inline score gauges
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_SOURCE_COLORS = {
    "Generative AI":     "#f0a500",
    "Materials Project": "#00d4ff",
    "BRENDA":            "#a371f7",
    "Open Catalyst":     "#3fb950",
}
_STRATEGY_COLORS = {
    "Element substitution":  "#00d4ff",
    "Promoter addition":     "#3fb950",
    "Binary alloy design":   "#a371f7",
    "Enzyme engineering":    "#f0a500",
}
_STRATEGY_ICONS = {
    "Element substitution":  "🔄",
    "Promoter addition":     "➕",
    "Binary alloy design":   "⚗️",
    "Enzyme engineering":    "🧬",
}

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
)
_AXIS = dict(
    gridcolor="rgba(255,255,255,0.07)",
    zerolinecolor="rgba(255,255,255,0.15)",
    color="#8892a4",
    tickfont=dict(size=11),
)


def _confidence_bar(value: float) -> str:
    filled = int(round(value * 10))
    bar = "█" * filled + "░" * (10 - filled)
    color = "#3fb950" if value >= 0.7 else "#f0a500" if value >= 0.5 else "#f85149"
    return (
        f"<span style='font-family:monospace;color:{color};font-size:0.85rem;'>"
        f"{bar}</span> "
        f"<span style='color:{color};font-weight:600;'>{value:.0%}</span>"
    )


def _novelty_badge(score: float) -> str:
    if score >= 0.8:
        color, label = "#f0a500", "HIGH"
    elif score >= 0.5:
        color, label = "#00d4ff", "MED"
    else:
        color, label = "#8892a4", "LOW"
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;"
        f"border-radius:6px;padding:1px 7px;font-size:0.72rem;"
        f"color:{color};font-weight:700;'>✨ {label}</span>"
    )


# ── Chart 1: Bubble chart — activity vs novelty ───────────────────────────────

def _build_bubble_chart(known_df: pd.DataFrame, gen_df: pd.DataFrame) -> go.Figure | None:
    """Bubble chart: x=predicted_activity, y=novelty_score, size=confidence.

    AI-generated candidates are stars; known are circles.
    Quadrant lines divide the space into high/low novelty × good/poor activity.
    """
    frames = []
    if not known_df.empty:
        kdf = known_df.copy()
        kdf["_group"] = kdf["source"].astype(str)
        if "novelty_score" not in kdf.columns:
            kdf["novelty_score"] = 0.1
        else:
            kdf["novelty_score"] = pd.to_numeric(kdf["novelty_score"], errors="coerce").fillna(0.1)
        if "confidence" not in kdf.columns:
            kdf["confidence"] = 0.5
        else:
            kdf["confidence"] = pd.to_numeric(kdf["confidence"], errors="coerce").fillna(0.5)
        frames.append(kdf)
    if not gen_df.empty:
        gdf = gen_df.copy()
        gdf["_group"] = "Generative AI"
        frames.append(gdf)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    # Resolve predicted_activity: prefer that column, fall back to activity_value
    if "predicted_activity" not in combined.columns:
        combined["predicted_activity"] = pd.to_numeric(
            combined.get("activity_value", pd.Series(dtype=float)), errors="coerce"
        )
    else:
        combined["predicted_activity"] = pd.to_numeric(combined["predicted_activity"], errors="coerce")
    if "novelty_score" not in combined.columns:
        combined["novelty_score"] = 0.1
    else:
        combined["novelty_score"] = pd.to_numeric(combined["novelty_score"], errors="coerce").fillna(0.1)
    if "confidence" not in combined.columns:
        combined["confidence"] = 0.5
    else:
        combined["confidence"] = pd.to_numeric(combined["confidence"], errors="coerce").fillna(0.5)
    combined = combined.dropna(subset=["predicted_activity"])

    # Use rank_score for x-axis so all sources are on the same 0–1 scale
    if "rank_score" in combined.columns:
        combined["_x"] = pd.to_numeric(combined["rank_score"], errors="coerce").fillna(0.5)
    else:
        from ai.predictor import CatalystPredictor
        combined["_x"] = CatalystPredictor._compute_rank_scores(combined)

    fig = go.Figure()

    # Quadrant shading — left = ideal (high novelty, low rank score)
    x_vals = combined["_x"].dropna()
    x_mid = 0.5  # rank_score midpoint is always 0.5
    fig.add_shape(type="rect",
        x0=-0.02, x1=x_mid,
        y0=0.5, y1=1.05,
        fillcolor="rgba(63,185,80,0.04)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_annotation(
        x=0.25, y=1.02,
        text="★ Ideal zone",
        showarrow=False,
        font=dict(color="rgba(63,185,80,0.5)", size=10),
    )
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.10)", dash="dot", width=1))
    fig.add_vline(x=x_mid, line=dict(color="rgba(255,255,255,0.10)", dash="dot", width=1))

    for group in combined["_group"].unique():
        sub = combined[combined["_group"] == group]
        color = _SOURCE_COLORS.get(group, "#8892a4")
        is_gen = group == "Generative AI"
        symbol = "star" if is_gen else "circle"
        sizes = (sub["confidence"].clip(0, 1) * 20 + 8).tolist()

        label_col = sub.apply(
            lambda r: str(r.get("formula") or r.get("name") or "?"), axis=1
        )
        strategy_col = sub.get("generation_strategy", pd.Series([""] * len(sub), index=sub.index))
        act_col = sub["predicted_activity"]
        unit_col = sub.get("activity_unit", pd.Series([""] * len(sub), index=sub.index))

        fig.add_trace(go.Scatter(
            x=sub["_x"],
            y=sub["novelty_score"],
            mode="markers",
            name=group,
            marker=dict(
                color=color,
                size=sizes,
                symbol=symbol,
                opacity=0.88,
                line=dict(
                    width=1.5 if is_gen else 0.5,
                    color="rgba(255,255,255,0.5)" if is_gen else "rgba(255,255,255,0.15)",
                ),
            ),
            text=label_col,
            customdata=list(zip(
                sub["confidence"].round(2),
                strategy_col.fillna("Retrieved"),
                act_col.round(4),
                unit_col.fillna(""),
            )),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Rank score: <b>%{x:.3f}</b> (0=best)<br>"
                "Predicted value: <b>%{customdata[2]} %{customdata[3]}</b><br>"
                "Novelty: <b>%{y:.0%}</b><br>"
                "Confidence: <b>%{customdata[0]:.0%}</b><br>"
                "Strategy: %{customdata[1]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_DARK,
        margin=dict(t=56, b=56, l=64, r=32),
        title=dict(
            text=(
                "Candidate Space — Activity vs Novelty"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "Bubble size = confidence · ⭐ = AI-generated · Green zone = high novelty + good activity"
                "</sup>"
            ),
            font=dict(size=15, color="#e6edf3"),
        ),
        xaxis=dict(
            title="Rank Score (0 = best within source, 1 = worst)",
            range=[-0.02, 1.1],
            **_AXIS,
        ),
        yaxis=dict(
            title="Novelty Score",
            tickformat=".0%",
            range=[-0.05, 1.1],
            **_AXIS,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=460,
    )
    return fig


# ── Chart 2: Ranked lollipop ──────────────────────────────────────────────────

def _build_ranked_lollipop(known_df: pd.DataFrame, gen_df: pd.DataFrame) -> go.Figure | None:
    """Lollipop chart of all candidates ranked by rank_score (fair cross-source).

    Uses rank_score (per-source percentile) for ordering so retrieved candidates
    compete fairly with AI-generated ones. Shows original predicted_activity value
    in the label. AI candidates get a gold star; known get a circle.
    """
    frames = []
    if not known_df.empty:
        kdf = known_df.copy()
        kdf["_is_gen"] = False
        kdf["_group"] = kdf["source"].astype(str)
        if "confidence" not in kdf.columns:
            kdf["confidence"] = 0.5
        else:
            kdf["confidence"] = pd.to_numeric(kdf["confidence"], errors="coerce").fillna(0.5)
        frames.append(kdf)
    if not gen_df.empty:
        gdf = gen_df.copy()
        gdf["_is_gen"] = True
        gdf["_group"] = "Generative AI"
        frames.append(gdf)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Resolve predicted_activity
    if "predicted_activity" not in combined.columns:
        combined["predicted_activity"] = pd.to_numeric(
            combined["activity_value"] if "activity_value" in combined.columns
            else pd.Series(dtype=float),
            errors="coerce",
        )
    else:
        combined["predicted_activity"] = pd.to_numeric(combined["predicted_activity"], errors="coerce")

    if "confidence" not in combined.columns:
        combined["confidence"] = 0.5
    else:
        combined["confidence"] = pd.to_numeric(combined["confidence"], errors="coerce").fillna(0.5)

    combined = combined.dropna(subset=["predicted_activity"])

    # Use rank_score for ordering if available, otherwise compute it inline
    if "rank_score" in combined.columns:
        combined["_sort_key"] = pd.to_numeric(combined["rank_score"], errors="coerce").fillna(0.5)
    else:
        # Compute per-source percentile rank inline
        from ai.predictor import CatalystPredictor
        combined["_sort_key"] = CatalystPredictor._compute_rank_scores(combined)

    combined = combined.nsmallest(25, "_sort_key").reset_index(drop=True)
    combined["_rank"] = range(1, len(combined) + 1)

    # Label: show formula/name + original value + unit
    def _row_label(r: pd.Series) -> str:
        prefix = "⭐ " if r.get("_is_gen") else ""
        name = str(r.get("formula") or r.get("name") or "?")
        val = r.get("predicted_activity")
        unit = str(r.get("activity_unit") or "")
        try:
            val_str = f" ({float(val):.3f} {unit})" if pd.notna(val) else ""
        except (TypeError, ValueError):
            val_str = ""
        return prefix + name + val_str

    label = combined.apply(_row_label, axis=1)

    fig = go.Figure()

    # Stem lines — use rank_score as x-axis (0=best, 1=worst)
    for _, row in combined.iterrows():
        color = _SOURCE_COLORS.get(str(row["_group"]), "#8892a4")
        fig.add_trace(go.Scatter(
            x=[0, row["_sort_key"]],
            y=[row["_rank"], row["_rank"]],
            mode="lines",
            line=dict(color=f"rgba({_hex_rgb(color)},0.25)", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Dots — split by source for legend
    for group in combined["_group"].unique():
        sub = combined[combined["_group"] == group]
        color = _SOURCE_COLORS.get(group, "#8892a4")
        is_gen = group == "Generative AI"
        sizes = (sub["confidence"].clip(0, 1) * 14 + 8).tolist()
        labels_sub = label[sub.index]

        act_vals = sub["predicted_activity"]
        units = sub.get("activity_unit", pd.Series([""] * len(sub), index=sub.index))

        fig.add_trace(go.Scatter(
            x=sub["_sort_key"],
            y=sub["_rank"],
            mode="markers+text",
            name=group,
            marker=dict(
                color=color,
                size=sizes,
                symbol="star" if is_gen else "circle",
                opacity=0.92,
                line=dict(
                    width=1.5 if is_gen else 0.5,
                    color="rgba(255,255,255,0.5)" if is_gen else "rgba(255,255,255,0.15)",
                ),
            ),
            text=["  " + lbl for lbl in labels_sub],
            textposition="middle right",
            textfont=dict(size=9, color="#f0a500" if is_gen else "#8892a4"),
            customdata=list(zip(
                act_vals.round(4),
                units.fillna(""),
            )),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Rank: %{y}<br>"
                "Rank score: %{x:.3f}<br>"
                "Predicted value: %{customdata[0]} %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_DARK,
        margin=dict(t=64, b=48, l=48, r=200),
        title=dict(
            text=(
                "Ranked Candidates — Top 25"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "⭐ = AI-generated · X-axis = within-source percentile rank (0=best) · "
                "Dot size = confidence · Labels show original predicted value"
                "</sup>"
            ),
            font=dict(size=15, color="#e6edf3"),
        ),
        xaxis=dict(
            title="Rank Score (0 = best within source, 1 = worst)",
            range=[-0.02, 1.15],
            **_AXIS,
        ),
        yaxis=dict(
            title="Overall Rank",
            autorange="reversed",
            tickmode="linear",
            dtick=1,
            color="#8892a4",
            tickfont=dict(size=9),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=max(380, len(combined) * 22 + 100),
    )
    return fig


# ── Chart 3: Radar chart — multi-dimensional score comparison ─────────────────

def _build_radar_chart(generated_candidates: list) -> go.Figure | None:
    """Radar/spider chart comparing top-5 generated candidates on 4 dimensions."""
    top = generated_candidates[:5]
    if not top:
        return None

    dims = ["Activity\n(inv.)", "Selectivity", "Stability\n(inv.)", "Confidence", "Novelty"]
    fig = go.Figure()

    for cand in top:
        # Normalise each dimension to [0,1] where 1 = best
        act = cand.predicted_activity or 0.0
        # For activity: lower is better → invert by mapping to [0,1] via sigmoid-like
        act_norm = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(act * 2))))
        sel = float(cand.predicted_selectivity or 0.5)
        stab_raw = float(cand.predicted_stability or 0.1)
        # Stability: lower hull energy = better → invert
        stab_norm = max(0.0, min(1.0, 1.0 - stab_raw * 5))
        conf = float(cand.confidence or 0.5)
        nov = float(cand.novelty_score or 0.5)

        values = [act_norm, sel, stab_norm, conf, nov]
        values_closed = values + [values[0]]  # close the polygon

        strategy = cand.generation_strategy or ""
        color = "#f0a500"
        for key, col in _STRATEGY_COLORS.items():
            if key.lower() in strategy.lower():
                color = col
                break

        short_name = (cand.formula or cand.name or "?")[:18]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=dims + [dims[0]],
            fill="toself",
            fillcolor=f"rgba({_hex_rgb(color)},0.10)",
            line=dict(color=color, width=2),
            name=short_name,
            hovertemplate=(
                f"<b>{short_name}</b><br>"
                "%{theta}: %{r:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_DARK,
        margin=dict(t=72, b=48, l=48, r=48),
        title=dict(
            text=(
                "Multi-Dimensional Score Comparison — Top 5 Generated"
                "<br><sup style='color:#8892a4;font-size:11px;'>"
                "All axes normalised to [0,1] where 1 = best"
                "</sup>"
            ),
            font=dict(size=14, color="#e6edf3"),
        ),
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.10)",
                tickfont=dict(size=9, color="#8892a4"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.15)",
                tickfont=dict(size=11, color="#c9d1d9"),
            ),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
        ),
        height=480,
    )
    return fig


# ── Chart 4: Strategy donut ───────────────────────────────────────────────────

def _build_strategy_donut(generated_candidates: list) -> go.Figure | None:
    if not generated_candidates:
        return None

    strategy_counts: dict[str, int] = {}
    for c in generated_candidates:
        # Normalise strategy label to one of the known keys
        strat = "Other"
        for key in _STRATEGY_COLORS:
            if key.lower() in (c.generation_strategy or "").lower():
                strat = key
                break
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    labels = list(strategy_counts.keys())
    values = list(strategy_counts.values())
    colors = [_STRATEGY_COLORS.get(l, "#8892a4") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color="rgba(0,0,0,0.3)", width=2),
        ),
        textfont=dict(size=11, color="#e6edf3"),
        hovertemplate="<b>%{label}</b><br>%{value} candidates (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(generated_candidates)}</b><br><span style='font-size:10px'>generated</span>",
        x=0.5, y=0.5,
        font=dict(size=14, color="#e6edf3"),
        showarrow=False,
    )
    fig.update_layout(
        **_DARK,
        margin=dict(t=48, b=16, l=16, r=16),
        title=dict(
            text="Generation Strategy Mix",
            font=dict(size=13, color="#e6edf3"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
            font=dict(size=10),
            orientation="v",
        ),
        height=300,
        showlegend=True,
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string for rgba() use."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def _score_gauge_html(label: str, value: float, color: str) -> str:
    """Inline SVG progress bar for a score."""
    pct = int(value * 100)
    bar_w = int(value * 120)
    return (
        f"<div style='margin:4px 0;'>"
        f"<span style='font-size:0.72rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>{label}</span>"
        f"<div style='display:flex;align-items:center;gap:8px;margin-top:2px;'>"
        f"<div style='flex:1;height:6px;background:rgba(255,255,255,0.08);"
        f"border-radius:3px;overflow:hidden;'>"
        f"<div style='width:{bar_w}px;max-width:100%;height:100%;"
        f"background:{color};border-radius:3px;'></div></div>"
        f"<span style='font-size:0.78rem;font-weight:600;color:{color};min-width:32px;'>"
        f"{pct}%</span></div></div>"
    )


def _strategy_color(strategy: str) -> str:
    for key, col in _STRATEGY_COLORS.items():
        if key.lower() in strategy.lower():
            return col
    return "#8892a4"


def render_generative_page(
    known_df: pd.DataFrame,
    generated_candidates: list,
    reaction: str,
) -> None:
    """Render the generative design tab with rich visualisations."""
    st.markdown(
        """
        <div style="padding:0.8rem 0 1rem 0;">
            <h3 style="margin:0;color:#e6edf3;">🧬 Generative AI Design</h3>
            <p style="color:#8892a4;font-size:0.85rem;margin:4px 0 0 0;">
                Novel candidates proposed by the AI engine — not present in any database.
                Ranked alongside known candidates by predicted activity.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not generated_candidates:
        st.info("Run a retrieval first to generate novel candidates.")
        return

    gen_df = pd.DataFrame([c.to_dict() for c in generated_candidates])

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_gen = len(generated_candidates)
    n_known = len(known_df) if not known_df.empty else 0
    avg_conf = float(gen_df["confidence"].mean()) if "confidence" in gen_df.columns else 0.0
    avg_novelty = float(gen_df["novelty_score"].mean()) if "novelty_score" in gen_df.columns else 0.0
    best_act = gen_df["predicted_activity"].dropna()
    best_str = f"{float(best_act.min()):.4f}" if not best_act.empty else "—"

    kpi_cols = st.columns(5)
    for col, label, val, icon, sub in [
        (kpi_cols[0], "Generated",      str(n_gen),              "🤖", "novel candidates"),
        (kpi_cols[1], "Known Retrieved", str(n_known),            "🗄️", "from databases"),
        (kpi_cols[2], "Avg Confidence",  f"{avg_conf:.0%}",       "📊", "prediction reliability"),
        (kpi_cols[3], "Avg Novelty",     f"{avg_novelty:.0%}",    "✨", "vs known pool"),
        (kpi_cols[4], "Best Predicted",  best_str,                "🏆", "lowest activity"),
    ]:
        with col:
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                    border-radius:14px;padding:0.9rem 1rem;">
                    <p style="margin:0 0 4px 0;font-size:0.68rem;color:#8892a4;
                        letter-spacing:0.07em;text-transform:uppercase;">{icon}&nbsp;{label}</p>
                    <p style="margin:0 0 2px 0;font-size:1.5rem;font-weight:700;color:#e6edf3;line-height:1.1;">{val}</p>
                    <p style="margin:0;font-size:0.68rem;color:#8892a4;">{sub}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Main charts ───────────────────────────────────────────────────────────
    tab_bubble, tab_rank, tab_radar, tab_cards = st.tabs([
        "🫧 Candidate Space",
        "🏆 Ranked View",
        "🕸️ Score Radar",
        "📋 Candidate Cards",
    ])

    with tab_bubble:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.5rem;'>"
            "Each point is one candidate. X-axis = predicted activity (lower = better for energy metrics). "
            "Y-axis = novelty vs known pool. Bubble size encodes prediction confidence.</p>",
            unsafe_allow_html=True,
        )
        fig_bubble = _build_bubble_chart(known_df, gen_df)
        if fig_bubble:
            st.plotly_chart(fig_bubble, use_container_width=True)

    with tab_rank:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.5rem;'>"
            "Top 25 candidates ranked by predicted activity. "
            "⭐ = AI-generated. Dot size = confidence. Stems show distance from zero.</p>",
            unsafe_allow_html=True,
        )
        fig_lollipop = _build_ranked_lollipop(known_df, gen_df)
        if fig_lollipop:
            st.plotly_chart(fig_lollipop, use_container_width=True)

    with tab_radar:
        col_radar, col_donut = st.columns([3, 2])
        with col_radar:
            st.markdown(
                "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.5rem;'>"
                "Multi-dimensional comparison of the top 5 generated candidates. "
                "All axes normalised — larger area = better overall profile.</p>",
                unsafe_allow_html=True,
            )
            fig_radar = _build_radar_chart(generated_candidates)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
        with col_donut:
            st.markdown(
                "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.5rem;'>"
                "Breakdown of generation strategies used.</p>",
                unsafe_allow_html=True,
            )
            fig_donut = _build_strategy_donut(generated_candidates)
            if fig_donut:
                st.plotly_chart(fig_donut, use_container_width=True)

    with tab_cards:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:1rem;'>"
            "Detailed view of each AI-generated candidate with design rationale and score breakdown.</p>",
            unsafe_allow_html=True,
        )

        for i, cand in enumerate(generated_candidates):
            strat_color = _strategy_color(cand.generation_strategy or "")
            strat_icon = "🔬"
            for key, icon in _STRATEGY_ICONS.items():
                if key.lower() in (cand.generation_strategy or "").lower():
                    strat_icon = icon
                    break

            conf_color = "#3fb950" if cand.confidence >= 0.7 else "#f0a500" if cand.confidence >= 0.5 else "#f85149"
            nov_color = "#f0a500" if cand.novelty_score >= 0.8 else "#00d4ff" if cand.novelty_score >= 0.5 else "#8892a4"

            # Card header
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);
                    border:1px solid {strat_color}33;
                    border-left:3px solid {strat_color};
                    border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.6rem;">
                  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
                    <div style="flex:1;">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span style="font-size:1.1rem;">{strat_icon}</span>
                        <span style="font-weight:600;color:#e6edf3;font-size:0.95rem;">
                          {cand.formula or cand.name or "Novel candidate"}
                        </span>
                        <span style="background:{strat_color}22;border:1px solid {strat_color}44;
                          border-radius:6px;padding:1px 8px;font-size:0.70rem;color:{strat_color};
                          font-weight:600;">{(cand.generation_strategy or "").split(":")[0]}</span>
                      </div>
                      <p style="margin:0 0 6px 0;font-size:0.82rem;color:#8892a4;font-style:italic;">
                        💡 {cand.rationale}
                      </p>
                      <p style="margin:0;font-size:0.78rem;color:#8892a4;">
                        Reaction: <span style="color:#c9d1d9;">{cand.reaction}</span>
                      </p>
                    </div>
                    <div style="min-width:160px;">
                      <div style="text-align:right;margin-bottom:8px;">
                        <span style="font-size:1.4rem;font-weight:700;color:{conf_color};">
                          {cand.predicted_activity:.4f}
                        </span>
                        <span style="font-size:0.75rem;color:#8892a4;"> {cand.activity_unit}</span>
                        <br>
                        <span style="font-size:0.70rem;color:#8892a4;">predicted activity</span>
                      </div>
                      {_score_gauge_html("Confidence", cand.confidence, conf_color)}
                      {_score_gauge_html("Novelty", cand.novelty_score, nov_color)}
                      {_score_gauge_html("Selectivity", cand.predicted_selectivity or 0.5, "#00d4ff")}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    export_df = gen_df.drop(columns=["raw", "conditions"], errors="ignore")
    csv = export_df.to_csv(index=False)
    st.download_button(
        "⬇️ Export Generated Candidates (CSV)",
        data=csv,
        file_name=f"catalystiq_generated_{reaction[:30].replace(' ', '_')}.csv",
        mime="text/csv",
    )
