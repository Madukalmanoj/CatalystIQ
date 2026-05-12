"""Streamlit results renderer — clean, glassy, per-source charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SOURCE_COLORS: dict[str, str] = {
    "Materials Project": "#00d4ff",
    "BRENDA":            "#a371f7",
    "Open Catalyst":     "#3fb950",
}
SOURCE_ICONS: dict[str, str] = {
    "Materials Project": "⚗️",
    "BRENDA":            "🧬",
    "Open Catalyst":     "⚡",
}
SOURCE_METRIC_LABEL: dict[str, str] = {
    "Materials Project": "Formation Energy (eV/atom)",
    "BRENDA":            "Km — substrate affinity (mM)",
    "Open Catalyst":     "Adsorption Energy (eV)",
}
QUALITY_BADGE: dict[str, str] = {
    "ok":             "🟢 ok",
    "missing_fields": "🟡 missing",
}

_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
    margin=dict(t=48, b=48, l=56, r=24),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.10)",
        borderwidth=1,
        font=dict(size=11),
    ),
)
_AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.06)",
    zerolinecolor="rgba(255,255,255,0.12)",
    color="#8892a4",
    tickfont=dict(size=11),
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, icon: str, sub: str = "") -> None:
    sub_html = f"<p style='margin:2px 0 0 0;font-size:0.72rem;color:#3fb950;'>{sub}</p>" if sub else ""
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
            border-radius:14px;padding:1rem 1.1rem;backdrop-filter:blur(8px);">
            <p style="margin:0 0 6px 0;font-size:0.72rem;color:#8892a4;
                letter-spacing:0.07em;text-transform:uppercase;">{icon}&nbsp;{label}</p>
            <p style="margin:0;font-size:1.65rem;font-weight:700;color:#e6edf3;line-height:1;">{value}</p>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def _source_pill(source: str) -> str:
    color = SOURCE_COLORS.get(source, "#8892a4")
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;"
        f"border-radius:6px;padding:2px 8px;font-size:0.75rem;color:{color};"
        f"font-weight:600;white-space:nowrap;'>{source}</span>"
    )


def _chart_layout(**extra) -> dict:
    layout = dict(**_CHART_BASE)
    layout.update(extra)
    return layout


# ── per-source bar chart ──────────────────────────────────────────────────────

def _build_per_source_charts(df: pd.DataFrame) -> go.Figure | None:
    """Build one subplot per source, each with its own y-axis unit."""
    sources = [s for s in SOURCE_COLORS if s in df["source"].unique()]
    if not sources:
        return None

    n = len(sources)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f"{SOURCE_ICONS.get(s,'')} {s}" for s in sources],
        horizontal_spacing=0.10,
    )

    for col_idx, source in enumerate(sources, start=1):
        src_df = df[df["source"] == source].copy()
        src_df = src_df.dropna(subset=["activity_value"])
        src_df["activity_value"] = pd.to_numeric(src_df["activity_value"], errors="coerce")
        src_df = src_df.dropna(subset=["activity_value"])

        if src_df.empty:
            continue

        color = SOURCE_COLORS.get(source, "#8892a4")

        if source == "BRENDA":
            # BRENDA: lollipop chart — cleaner for sparse enzyme Km data
            # Deduplicate by name, keep lowest Km per enzyme
            src_df["_label"] = src_df["name"].replace("", "Unknown").fillna("Unknown")
            src_df = (
                src_df.sort_values("activity_value")
                .drop_duplicates(subset=["_label"])
                .nsmallest(12, "activity_value")
            )
            src_df = src_df.sort_values("activity_value", ascending=True)

            hover_brenda = src_df.apply(
                lambda r: (
                    f"<b>{r['_label']}</b><br>"
                    f"Km = {r['activity_value']:.4f} mM<br>"
                    f"Organism: {r['conditions'].get('organism', '') if isinstance(r.get('conditions'), dict) else ''}<br>"
                    f"Substrate: {r['conditions'].get('substrate', '') if isinstance(r.get('conditions'), dict) else ''}"
                ),
                axis=1,
            ).tolist()

            # Stem lines (horizontal lines from 0 to value)
            for _, row in src_df.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[0, row["activity_value"]],
                        y=[row["_label"], row["_label"]],
                        mode="lines",
                        line=dict(color=_hex_to_rgba(color, 0.35), width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=col_idx,
                )

            # Dots at the tip
            fig.add_trace(
                go.Scatter(
                    x=src_df["activity_value"].tolist(),
                    y=src_df["_label"].tolist(),
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=11,
                        line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
                    ),
                    text=[f"  {v:.3f}" for v in src_df["activity_value"]],
                    textposition="middle right",
                    textfont=dict(size=9, color="#c9d1d9"),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_brenda,
                    showlegend=False,
                    name=source,
                ),
                row=1, col=col_idx,
            )

            # X-axis: linear with nice tick format
            max_val = src_df["activity_value"].max()
            fig.update_xaxes(
                title_text="Km (mM)",
                title_font=dict(size=10, color="#8892a4"),
                range=[-max_val * 0.05, max_val * 1.35],
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.18)",
                color="#8892a4",
                tickfont=dict(size=10),
                tickformat=".3g",
                row=1, col=col_idx,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.04)",
                color="#8892a4",
                tickfont=dict(size=10),
                autorange=True,
                row=1, col=col_idx,
            )

        else:
            # Materials Project & Open Catalyst: vertical bars, formula on x-axis
            label_col = "formula"
            src_df[label_col] = src_df[label_col].replace("", "N/A").fillna("N/A")

            if source == "Materials Project":
                src_df = src_df.nsmallest(12, "activity_value")
            else:
                src_df = src_df.reindex(
                    src_df["activity_value"].abs().nsmallest(12).index
                )

            bar_colors = [
                color if v <= 0 else _hex_to_rgba(color, 0.4)
                for v in src_df["activity_value"]
            ]

            hover = src_df.apply(
                lambda r: (
                    f"<b>{r[label_col]}</b><br>"
                    f"Value: {r['activity_value']:.4f} {r.get('activity_unit','')}<br>"
                    f"Metric: {r.get('activity_metric','')}<br>"
                    f"ID: {r.get('source_id','')}"
                ),
                axis=1,
            ).tolist()

            fig.add_trace(
                go.Bar(
                    x=src_df[label_col].tolist(),
                    y=src_df["activity_value"].tolist(),
                    name=source,
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=0),
                        opacity=0.88,
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover,
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
            y_title = SOURCE_METRIC_LABEL.get(source, "Activity Value")
            fig.update_yaxes(
                title_text=y_title,
                title_font=dict(size=10, color="#8892a4"),
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.12)",
                color="#8892a4",
                tickfont=dict(size=11),
                row=1, col=col_idx,
            )
            fig.update_xaxes(
                tickangle=-40,
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.12)",
                color="#8892a4",
                tickfont=dict(size=9),
                row=1, col=col_idx,
            )

    fig.update_layout(
        **_chart_layout(
            title_text="Activity by Source  <span style='font-size:12px;color:#8892a4'>"
                       "(each panel uses its own unit)</span>",
            title_font=dict(size=15, color="#e6edf3"),
            height=420,
            bargap=0.18,
        )
    )
    # Style subplot title annotations
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#c9d1d9")

    return fig


# ── top-N ranked table ────────────────────────────────────────────────────────

def _build_top_candidates_chart(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart of top candidates per source by activity value."""
    plot_df = df.copy()
    plot_df["activity_value"] = pd.to_numeric(plot_df["activity_value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["activity_value"])
    if plot_df.empty:
        return None

    frames = []
    for source in SOURCE_COLORS:
        src = plot_df[plot_df["source"] == source].copy()
        if src.empty:
            continue
        src = src.nsmallest(8, "activity_value")
        frames.append(src)

    if not frames:
        return None

    top = pd.concat(frames).reset_index(drop=True)
    label_col = top.apply(
        lambda r: r["formula"] if r.get("formula", "") not in ("", "N/A") else r.get("name", "Unknown"),
        axis=1,
    )
    top["_label"] = label_col + "  [" + top["source"].str[:2] + "]"

    fig = go.Figure()
    for source in SOURCE_COLORS:
        mask = top["source"] == source
        if not mask.any():
            continue
        color = SOURCE_COLORS[source]
        fig.add_trace(
            go.Bar(
                y=top.loc[mask, "_label"],
                x=top.loc[mask, "activity_value"],
                name=source,
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Value: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_chart_layout(
            title_text="Top Candidates by Activity Value",
            title_font=dict(size=15, color="#e6edf3"),
            height=max(320, len(top) * 28 + 80),
            barmode="overlay",
            xaxis=dict(title="Activity Value", **_AXIS_STYLE),
            yaxis=dict(autorange="reversed", **_AXIS_STYLE),
        )
    )
    return fig


# ── distribution violin ───────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color to an rgba() string Plotly accepts.

    Args:
        hex_color: Hex string like '#00d4ff'.
        alpha: Opacity between 0 and 1.

    Returns:
        rgba string e.g. 'rgba(0,212,255,0.2)'.

    Raises:
        None.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_distribution_chart(df: pd.DataFrame) -> go.Figure | None:
    """Violin + box plot of activity distribution per source (normalised z-score)."""
    plot_df = df.copy()
    plot_df["activity_value"] = pd.to_numeric(plot_df["activity_value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["activity_value"])
    if plot_df.empty:
        return None

    # Z-score normalise per source so all fit on one axis
    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        return (series - series.mean()) / std if std > 0 else series - series.mean()

    plot_df["_z"] = plot_df.groupby("source")["activity_value"].transform(_zscore)

    fig = go.Figure()
    for source in SOURCE_COLORS:
        src = plot_df[plot_df["source"] == source]
        if src.empty:
            continue
        color = SOURCE_COLORS[source]
        fig.add_trace(
            go.Violin(
                y=src["_z"],
                name=source,
                box_visible=True,
                meanline_visible=True,
                fillcolor=_hex_to_rgba(color, 0.2),
                line_color=color,
                opacity=0.85,
                points="outliers",
                marker=dict(color=color, size=4, opacity=0.6),
            )
        )

    fig.update_layout(
        **_chart_layout(
            title_text="Activity Distribution  <span style='font-size:11px;color:#8892a4'>"
                       "(z-score normalised per source)</span>",
            title_font=dict(size=15, color="#e6edf3"),
            height=360,
            yaxis=dict(title="Normalised Activity (σ)", **_AXIS_STYLE),
            xaxis=dict(**_AXIS_STYLE),
            violingap=0.3,
            violinmode="overlay",
        )
    )
    return fig


# ── main render ───────────────────────────────────────────────────────────────

def render_results(df: pd.DataFrame) -> None:
    """Render retrieval results: KPIs, tabs with table and charts.

    Args:
        df: Candidate dataframe produced by aggregator.

    Returns:
        None.

    Raises:
        None.
    """
    if df.empty:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;
                background:rgba(255,255,255,0.03);
                border:1px dashed rgba(255,255,255,0.12);
                border-radius:16px;margin-top:1rem;">
                <p style="font-size:2.5rem;margin:0;">🔬</p>
                <p style="color:#8892a4;margin:0.5rem 0 0 0;">No candidates found.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    total = len(df)
    sources_count = df["source"].nunique() if "source" in df.columns else 0
    formulas_count = df["formula"].replace("", pd.NA).nunique() if "formula" in df.columns else 0
    ok_pct = (
        int(round((df["data_quality"] == "ok").sum() / total * 100))
        if "data_quality" in df.columns else 0
    )
    best_val = df["activity_value"].dropna()
    best_str = f"{best_val.min():.4f}" if not best_val.empty else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric_card("Candidates", str(total), "🧪")
    with c2:
        _metric_card("Sources", str(sources_count), "🗄️")
    with c3:
        _metric_card("Formulas", str(formulas_count), "⚗️")
    with c4:
        _metric_card("Quality", f"{ok_pct}%", "✅", sub="complete records")
    with c5:
        _metric_card("Best Value", best_str, "🏆", sub="lowest activity")

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_table, tab_charts, tab_dist, tab_prov = st.tabs(
        ["📋 Candidates", "📊 Activity Charts", "🎻 Distribution", "🔍 Provenance"]
    )

    # ── Candidates table ──────────────────────────────────────────────────────
    with tab_table:
        display_df = df.copy()

        # Source pill (HTML rendered via markdown below table)
        display_df["Source"] = display_df["source"]
        display_df["ID"] = display_df["source_id"]
        display_df["Name"] = display_df["name"].fillna("—")
        display_df["Formula"] = display_df["formula"].replace("", "—").fillna("—")
        display_df["Metric"] = display_df["activity_metric"].fillna("—")
        display_df["Value"] = pd.to_numeric(display_df["activity_value"], errors="coerce")
        display_df["Unit"] = display_df["activity_unit"].fillna("—")
        if "stability" in display_df.columns:
            display_df["Stability"] = pd.to_numeric(display_df["stability"], errors="coerce")
        if "data_quality" in display_df.columns:
            display_df["Quality"] = display_df["data_quality"].map(
                lambda v: QUALITY_BADGE.get(str(v), f"⚪ {v}")
            )

        show_cols = ["Source", "ID", "Name", "Formula", "Metric", "Value", "Unit"]
        if "Stability" in display_df.columns:
            show_cols.append("Stability")
        if "Quality" in display_df.columns:
            show_cols.append("Quality")

        col_cfg: dict = {
            "Source":    st.column_config.TextColumn("Source",    width="medium"),
            "ID":        st.column_config.TextColumn("ID",        width="small"),
            "Name":      st.column_config.TextColumn("Name",      width="large"),
            "Formula":   st.column_config.TextColumn("Formula",   width="small"),
            "Metric":    st.column_config.TextColumn("Metric",    width="medium"),
            "Value":     st.column_config.NumberColumn("Value",   format="%.4f", width="small"),
            "Unit":      st.column_config.TextColumn("Unit",      width="small"),
            "Stability": st.column_config.NumberColumn("Stability", format="%.3f", width="small"),
            "Quality":   st.column_config.TextColumn("Quality",   width="small"),
        }

        # Source filter
        all_sources = sorted(display_df["Source"].unique().tolist())
        selected = st.multiselect(
            "Filter by source",
            options=all_sources,
            default=all_sources,
            key="table_source_filter",
            label_visibility="collapsed",
        )
        filtered = display_df[display_df["Source"].isin(selected)] if selected else display_df

        st.dataframe(
            filtered[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
            height=420,
        )

        col_dl, col_info = st.columns([1, 3])
        with col_dl:
            csv = df.drop(columns=["raw", "conditions"], errors="ignore").to_csv(index=False)
            st.download_button(
                "⬇️ Export CSV",
                data=csv,
                file_name="catalystiq_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_info:
            src_summary = "  ·  ".join(
                f"{SOURCE_ICONS.get(s,'🔬')} **{s}**: {(df['source']==s).sum()}"
                for s in all_sources
            )
            st.markdown(
                f"<p style='color:#8892a4;font-size:0.82rem;padding-top:0.5rem;'>{src_summary}</p>",
                unsafe_allow_html=True,
            )

    # ── Activity charts ───────────────────────────────────────────────────────
    with tab_charts:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.8rem;'>"
            "Each panel shows one source with its own unit — comparing across sources "
            "directly is not meaningful due to different metrics.</p>",
            unsafe_allow_html=True,
        )

        fig_src = _build_per_source_charts(df)
        if fig_src:
            st.plotly_chart(fig_src, use_container_width=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        fig_top = _build_top_candidates_chart(df)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)

    # ── Distribution ─────────────────────────────────────────────────────────
    with tab_dist:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.8rem;'>"
            "Activity values are z-score normalised per source so all three "
            "distributions fit on one axis for shape comparison.</p>",
            unsafe_allow_html=True,
        )
        fig_dist = _build_distribution_chart(df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Not enough numeric data for distribution plot.")

    # ── Provenance ────────────────────────────────────────────────────────────
    with tab_prov:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;margin-bottom:0.6rem;'>"
            "Raw payloads from each source for full traceability.</p>",
            unsafe_allow_html=True,
        )

        # Group by source for cleaner display
        if "raw" in df.columns:
            for source in df["source"].unique():
                src_df = df[df["source"] == source]
                color = SOURCE_COLORS.get(source, "#8892a4")
                icon = SOURCE_ICONS.get(source, "🔬")
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"margin:0.8rem 0 0.3rem 0;'>"
                    f"<span style='font-size:1.1rem;'>{icon}</span>"
                    f"<span style='color:{color};font-weight:600;font-size:0.9rem;'>"
                    f"{source}</span>"
                    f"<span style='color:#8892a4;font-size:0.78rem;'>"
                    f"({len(src_df)} records)</span></div>",
                    unsafe_allow_html=True,
                )
                records = src_df[["source_id", "raw"]].to_dict(orient="records")
                st.json(records, expanded=False)
        else:
            st.info("No raw payload data available.")
