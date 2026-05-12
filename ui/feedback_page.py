"""Experimental feedback loop page.

Allows researchers to:
1. Log experimental outcomes for retrieved/generated candidates.
2. Compare predicted vs measured values.
3. Trigger model retraining.
4. View discrepancy analysis and model version history.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.predictor import CatalystPredictor
from storage.db import (
    get_candidates_for_feedback,
    get_feedback_dataframe,
    get_model_history,
    log_experimental_result,
    save_model_version,
)

_OUTCOME_COLORS = {
    "exceeded":       "#3fb950",
    "matched":        "#00d4ff",
    "underperformed": "#f85149",
    "unknown":        "#8892a4",
}
_OUTCOME_ICONS = {
    "exceeded":       "🟢",
    "matched":        "🔵",
    "underperformed": "🔴",
    "unknown":        "⚪",
}


def _build_parity_chart(feedback_df: pd.DataFrame) -> go.Figure | None:
    """Predicted vs measured parity plot."""
    df = feedback_df.copy()
    df["predicted_activity"] = pd.to_numeric(df["predicted_activity"], errors="coerce")
    df["measured_value"] = pd.to_numeric(df["measured_value"], errors="coerce")
    df = df.dropna(subset=["predicted_activity", "measured_value"])
    if df.empty:
        return None

    colors = [_OUTCOME_COLORS.get(str(o), "#8892a4") for o in df.get("outcome", ["unknown"] * len(df))]

    fig = go.Figure()

    # Parity line
    all_vals = pd.concat([df["predicted_activity"], df["measured_value"]])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.1
    fig.add_trace(go.Scatter(
        x=[lo - margin, hi + margin],
        y=[lo - margin, hi + margin],
        mode="lines",
        name="Perfect prediction",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=df["predicted_activity"],
        y=df["measured_value"],
        mode="markers",
        name="Experiments",
        marker=dict(
            color=colors, size=11, opacity=0.85,
            line=dict(width=1, color="rgba(255,255,255,0.3)"),
        ),
        text=df.get("name", df.get("formula", pd.Series(["?"] * len(df)))),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Predicted: %{x:.4f}<br>"
            "Measured: %{y:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
        margin=dict(t=48, b=48, l=56, r=24),
        title=dict(
            text="Predicted vs Measured — Parity Plot",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(
            title="Predicted Activity",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            color="#8892a4",
        ),
        yaxis=dict(
            title="Measured Activity",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            color="#8892a4",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
        ),
        height=400,
    )
    return fig


def _build_error_chart(feedback_df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of prediction errors sorted by magnitude."""
    df = feedback_df.copy()
    df["predicted_activity"] = pd.to_numeric(df["predicted_activity"], errors="coerce")
    df["measured_value"] = pd.to_numeric(df["measured_value"], errors="coerce")
    df = df.dropna(subset=["predicted_activity", "measured_value"])
    if df.empty:
        return None

    df["error"] = df["measured_value"] - df["predicted_activity"]
    df = df.reindex(df["error"].abs().sort_values(ascending=False).index)
    label = df.apply(
        lambda r: str(r.get("name") or r.get("formula") or "?"), axis=1
    )
    colors = ["#f85149" if e > 0 else "#3fb950" for e in df["error"]]

    fig = go.Figure(go.Bar(
        x=label,
        y=df["error"],
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Error: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter, sans-serif", size=12),
        margin=dict(t=48, b=48, l=56, r=24),
        title=dict(
            text="Prediction Error (measured − predicted)  "
                 "<span style='font-size:11px;color:#f85149;'>🔴 underperformed</span>  "
                 "<span style='font-size:11px;color:#3fb950;'>🟢 exceeded</span>",
            font=dict(size=13, color="#e6edf3"),
        ),
        xaxis=dict(
            tickangle=-35,
            gridcolor="rgba(255,255,255,0.06)",
            color="#8892a4",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Error",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.18)",
            color="#8892a4",
        ),
        height=340,
    )
    return fig


def render_feedback_page() -> None:
    """Render the experimental feedback and model retraining tab."""
    st.markdown(
        """
        <div style="padding:0.8rem 0 1rem 0;">
            <h3 style="margin:0;color:#e6edf3;">🔁 Experimental Feedback Loop</h3>
            <p style="color:#8892a4;font-size:0.85rem;margin:4px 0 0 0;">
                Log experimental outcomes, compare predictions vs reality,
                and retrain the model to improve future rankings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_log, tab_analysis, tab_retrain, tab_history = st.tabs([
        "📝 Log Result",
        "📊 Discrepancy Analysis",
        "🔄 Retrain Model",
        "📜 Model History",
    ])

    # ── Log experimental result ───────────────────────────────────────────────
    with tab_log:
        st.markdown("#### Log an Experimental Outcome")
        st.markdown(
            "<p style='color:#8892a4;font-size:0.83rem;'>"
            "Select a candidate that was tested in the lab and enter the measured result.</p>",
            unsafe_allow_html=True,
        )

        # Show persistent success banner from previous submission
        if st.session_state.get("_feedback_success"):
            msg = st.session_state.pop("_feedback_success")
            st.success(msg)

        candidates_df = get_candidates_for_feedback()

        if candidates_df.empty:
            st.info("No candidates in the database yet. Run a retrieval first.")
        else:
            # ── Build unique display labels ───────────────────────────────────
            def _make_label(r: pd.Series, idx: int) -> str:
                raw_name = r.get("name")
                raw_formula = r.get("formula")
                # Guard against NaN values from pandas
                name = ""
                for val in (raw_name, raw_formula):
                    try:
                        if val is not None and pd.notna(val) and str(val).strip():
                            name = str(val).strip()
                            break
                    except (TypeError, ValueError):
                        pass
                name = name or "Unknown"
                source = str(r.get("source") or "").strip()
                pred = r.get("predicted_activity")
                unit = str(r.get("activity_unit") or "").strip()
                try:
                    pred_str = f"pred: {float(pred):.4f} {unit}" if pd.notna(pred) else "no prediction"
                except (TypeError, ValueError):
                    pred_str = "no prediction"
                # Append row index to guarantee uniqueness even with identical names
                return f"[#{idx}] {name} [{source}] ({pred_str})"

            candidates_df = candidates_df.reset_index(drop=True)
            candidates_df["_label"] = [
                _make_label(candidates_df.iloc[i], i)
                for i in range(len(candidates_df))
            ]

            selected_label = st.selectbox(
                "Select candidate",
                options=candidates_df["_label"].tolist(),
                key="feedback_candidate_select",
            )

            # Look up by label — guaranteed unique because of the index prefix
            mask = candidates_df["_label"] == selected_label
            if not mask.any():
                st.warning("Could not find selected candidate. Please refresh.")
                return
            selected_row = candidates_df[mask].iloc[0]

            # ── Show selected candidate info ──────────────────────────────────
            pred_val_raw = selected_row.get("predicted_activity")
            try:
                predicted_val: float | None = float(pred_val_raw) if pd.notna(pred_val_raw) else None
            except (TypeError, ValueError):
                predicted_val = None

            unit_default = str(selected_row.get("activity_unit") or "").strip()

            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px;padding:0.7rem 1rem;margin:0.5rem 0 1rem 0;font-size:0.83rem;">
                    <span style="color:#8892a4;">Selected: </span>
                    <strong style="color:#e6edf3;">{str(selected_row.get("name") or selected_row.get("formula") or "Unknown")}</strong>
                    &nbsp;·&nbsp;
                    <span style="color:#8892a4;">Source: </span>
                    <span style="color:#c9d1d9;">{str(selected_row.get("source") or "")}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#8892a4;">Model prediction: </span>
                    <span style="color:#00d4ff;font-weight:600;">
                        {f"{predicted_val:.4f} {unit_default}" if predicted_val is not None else "N/A"}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Form fields ───────────────────────────────────────────────────
            with st.form(key="log_result_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    measured_value = st.number_input(
                        "Measured value",
                        value=0.0,
                        format="%.5f",
                    )
                    measured_unit = st.text_input(
                        "Unit",
                        value=unit_default,
                        placeholder="e.g. eV/atom, mM, %",
                    )
                    measured_metric = st.text_input(
                        "Metric (e.g. yield, Km, TOF)",
                        value="activity",
                    )
                with col2:
                    outcome = st.selectbox(
                        "Outcome vs prediction",
                        options=["exceeded", "matched", "underperformed", "unknown"],
                    )
                    researcher = st.text_input(
                        "Researcher name / ID",
                        placeholder="e.g. Dr. Sharma",
                    )
                    notes = st.text_area(
                        "Notes (observations, conditions, anomalies)",
                        placeholder="e.g. Catalyst showed unexpected deactivation after 2h...",
                        height=100,
                    )

                submitted = st.form_submit_button(
                    "💾  Log Experimental Result",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                try:
                    candidate_id = int(selected_row["id"])
                    log_experimental_result(
                        candidate_id=candidate_id,
                        measured_value=float(measured_value),
                        measured_unit=measured_unit.strip(),
                        measured_metric=measured_metric.strip(),
                        predicted_value=predicted_val,
                        outcome=outcome,
                        notes=notes.strip(),
                        researcher=researcher.strip(),
                    )
                    cand_name = ""
                    for val in (selected_row.get("name"), selected_row.get("formula")):
                        try:
                            if val is not None and pd.notna(val) and str(val).strip():
                                cand_name = str(val).strip()
                                break
                        except (TypeError, ValueError):
                            pass
                    cand_name = cand_name or "candidate"
                    # Store success message in session state so it survives the rerun
                    st.session_state["_feedback_success"] = (
                        f"✅ Result logged for **{cand_name}** — "
                        f"measured **{measured_value:.5f} {measured_unit}**, "
                        f"outcome: **{outcome}**."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Failed to log result: {exc}")

    # ── Discrepancy analysis ──────────────────────────────────────────────────
    with tab_analysis:
        feedback_df = get_feedback_dataframe()

        if feedback_df.empty:
            st.info("No experimental results logged yet. Use the 'Log Result' tab to add data.")
        else:
            # Summary KPIs
            n_total = len(feedback_df)
            outcome_counts = feedback_df["outcome"].value_counts().to_dict()
            n_exceeded = outcome_counts.get("exceeded", 0)
            n_matched = outcome_counts.get("matched", 0)
            n_under = outcome_counts.get("underperformed", 0)

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val, color in [
                (c1, "Total Experiments", str(n_total), "#e6edf3"),
                (c2, "Exceeded", str(n_exceeded), "#3fb950"),
                (c3, "Matched", str(n_matched), "#00d4ff"),
                (c4, "Underperformed", str(n_under), "#f85149"),
            ]:
                with col:
                    st.markdown(
                        f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                            border-radius:14px;padding:0.9rem 1rem;">
                            <p style="margin:0 0 4px 0;font-size:0.72rem;color:#8892a4;
                                letter-spacing:0.07em;text-transform:uppercase;">{label}</p>
                            <p style="margin:0;font-size:1.5rem;font-weight:700;color:{color};">{val}</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            fig_parity = _build_parity_chart(feedback_df)
            if fig_parity:
                st.plotly_chart(fig_parity, use_container_width=True)

            fig_error = _build_error_chart(feedback_df)
            if fig_error:
                st.plotly_chart(fig_error, use_container_width=True)

            # Discrepancy hypotheses
            st.markdown("#### 🔍 Model Discrepancy Hypotheses")
            st.markdown(
                "<p style='color:#8892a4;font-size:0.83rem;'>"
                "The AI surfaces structural hypotheses for the largest prediction errors.</p>",
                unsafe_allow_html=True,
            )
            predictor = CatalystPredictor()
            discrepancies = predictor.analyse_discrepancies(feedback_df)

            if discrepancies:
                for disc in discrepancies[:5]:
                    direction_color = "#f85149" if disc["direction"] == "underperformed" else "#3fb950"
                    direction_icon = "🔴" if disc["direction"] == "underperformed" else "🟢"
                    with st.expander(
                        f"{direction_icon} {disc['name']}  —  "
                        f"Error: {disc['error']:+.4f}  ({disc['direction']})"
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Predicted", f"{disc['predicted']:.4f}")
                            st.metric("Measured", f"{disc['measured']:.4f}")
                        with col_b:
                            st.metric(
                                "Absolute Error",
                                f"{disc['abs_error']:.4f}",
                                delta=f"{disc['error']:+.4f}",
                                delta_color="inverse",
                            )
                        st.markdown(
                            f"<p style='color:#8892a4;font-size:0.85rem;margin-top:0.5rem;'>"
                            f"💡 <em>{disc['hypothesis']}</em></p>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No discrepancy data available yet.")

            # Raw feedback table
            with st.expander("📋 All logged results"):
                display = feedback_df.drop(columns=["conditions"], errors="ignore")
                st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Retrain model ─────────────────────────────────────────────────────────
    with tab_retrain:
        st.markdown("#### 🔄 Retrain Prediction Model")
        feedback_df = get_feedback_dataframe()
        n_feedback = len(feedback_df)

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
                <p style="margin:0;color:#c9d1d9;">
                    <strong>{n_feedback}</strong> experimental results available for training.
                    Minimum required: <strong>5</strong>.
                </p>
                <p style="margin:6px 0 0 0;color:#8892a4;font-size:0.83rem;">
                    The model uses ridge regression on physicochemical features extracted from
                    chemical formulas and reaction conditions. Retraining incorporates all
                    logged experimental outcomes to improve future predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        trigger = st.selectbox(
            "Retraining trigger",
            options=["manual", "threshold", "scheduled"],
            key="retrain_trigger",
        )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            retrain_clicked = st.button(
                "🔄 Retrain Now",
                type="primary",
                disabled=(n_feedback < 5),
                use_container_width=True,
            )
        with col_info:
            if n_feedback < 5:
                st.warning(
                    f"Need at least 5 experimental results to retrain. "
                    f"Currently have {n_feedback}."
                )

        if retrain_clicked and n_feedback >= 5:
            predictor = CatalystPredictor()
            with st.spinner("Training model on experimental feedback…"):
                metrics = predictor.fit(feedback_df)

            if metrics.get("status") == "trained":
                save_model_version(metrics, trigger=trigger)
                st.success(
                    f"✅ Model retrained on **{metrics['n_samples']}** samples.  \n"
                    f"RMSE: **{metrics['rmse']:.5f}**  |  R²: **{metrics['r2']:.4f}**  |  "
                    f"Residual σ: **{metrics['residual_std']:.5f}**"
                )
                st.info(
                    "The updated model will be used for predictions in the next retrieval session. "
                    "Re-run a search to see improved rankings."
                )
            else:
                st.warning(
                    f"Retraining skipped: {metrics.get('status')}. "
                    f"Need {metrics.get('min_required', 5)} samples, "
                    f"have {metrics.get('n_samples', 0)}."
                )

        # Data quality summary
        if not feedback_df.empty:
            st.markdown("#### 📊 Training Data Quality")
            col1, col2 = st.columns(2)
            with col1:
                outcome_dist = feedback_df["outcome"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=outcome_dist.index.tolist(),
                    values=outcome_dist.values.tolist(),
                    marker=dict(colors=[
                        _OUTCOME_COLORS.get(o, "#8892a4")
                        for o in outcome_dist.index
                    ]),
                    hole=0.4,
                    textfont=dict(size=11),
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
                    margin=dict(t=32, b=16, l=16, r=16),
                    title=dict(text="Outcome Distribution", font=dict(size=13, color="#e6edf3")),
                    height=280,
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                source_dist = feedback_df["source"].value_counts()
                fig_src = go.Figure(go.Bar(
                    x=source_dist.index.tolist(),
                    y=source_dist.values.tolist(),
                    marker=dict(color=["#00d4ff", "#a371f7", "#3fb950", "#f0a500"][:len(source_dist)]),
                ))
                fig_src.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
                    margin=dict(t=32, b=32, l=40, r=16),
                    title=dict(text="Results by Source", font=dict(size=13, color="#e6edf3")),
                    xaxis=dict(color="#8892a4"),
                    yaxis=dict(color="#8892a4", gridcolor="rgba(255,255,255,0.06)"),
                    height=280,
                )
                st.plotly_chart(fig_src, use_container_width=True)

    # ── Model history ─────────────────────────────────────────────────────────
    with tab_history:
        st.markdown("#### 📜 Model Version History")
        history_df = get_model_history()

        if history_df.empty:
            st.info("No model retraining events recorded yet.")
        else:
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id":                  st.column_config.NumberColumn("ID", width="small"),
                    "version_tag":         st.column_config.TextColumn("Version", width="medium"),
                    "trigger":             st.column_config.TextColumn("Trigger", width="small"),
                    "n_training_samples":  st.column_config.NumberColumn("Samples", width="small"),
                    "rmse":                st.column_config.NumberColumn("RMSE", format="%.5f", width="small"),
                    "r2":                  st.column_config.NumberColumn("R²", format="%.4f", width="small"),
                    "residual_std":        st.column_config.NumberColumn("Residual σ", format="%.5f", width="small"),
                    "trained_at":          st.column_config.DatetimeColumn("Trained At", width="medium"),
                },
            )

            # R² trend
            if len(history_df) > 1 and "r2" in history_df.columns:
                valid = history_df.dropna(subset=["r2"])
                if len(valid) > 1:
                    fig_trend = go.Figure(go.Scatter(
                        x=list(range(len(valid))),
                        y=valid["r2"].tolist(),
                        mode="lines+markers",
                        line=dict(color="#3fb950", width=2),
                        marker=dict(size=8, color="#3fb950"),
                        hovertemplate="Version %{x}<br>R² = %{y:.4f}<extra></extra>",
                    ))
                    fig_trend.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
                        margin=dict(t=40, b=40, l=56, r=24),
                        title=dict(text="Model R² Over Retraining Iterations", font=dict(size=13, color="#e6edf3")),
                        xaxis=dict(title="Retraining iteration", color="#8892a4", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(title="R²", color="#8892a4", gridcolor="rgba(255,255,255,0.06)"),
                        height=280,
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
