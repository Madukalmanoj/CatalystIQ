"""CatalystIQ Streamlit entrypoint orchestrating retrieval, normalization, and persistence."""

from __future__ import annotations

from collections import Counter
import sys

import pandas as pd
import streamlit as st
from loguru import logger

from ai.generator import CatalystGenerator
from ai.predictor import CatalystPredictor
from config import CACHE_DB_PATH, LOG_LEVEL
from processing.cache import QueryCache
from processing.normalizer import flag_missing, normalize_units
from retrieval.base import BaseRetriever
from retrieval.brenda import BrendaRetriever
from retrieval.demo_data import demo_candidates_for_query
from retrieval.materials_project import MaterialsProjectRetriever
from retrieval.open_catalyst import OpenCatalystRetriever
from storage.db import init_db, save_candidates, save_query_event, seed_demo_database
from ui.feedback_page import render_feedback_page
from ui.generative_page import render_generative_page
from ui.pathway_page import render_pathway_page
from ui.results_page import render_results
from ui.search_page import render_search_sidebar

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

st.set_page_config(
    page_title="CatalystIQ",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS: glassy dark theme ────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-glass:     rgba(22, 27, 34, 0.75);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-blue:  #00d4ff;
    --accent-purple:#7b2ff7;
    --accent-green: #3fb950;
    --text-primary: #e6edf3;
    --text-muted:   #8892a4;
    --radius-lg:    16px;
    --radius-md:    10px;
}

.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -40%; left: -20%;
    width: 80%; height: 80%;
    background: radial-gradient(ellipse at center, rgba(123,47,247,0.12) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -30%; right: -10%;
    width: 60%; height: 60%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--bg-glass) !important;
    border-right: 1px solid var(--border-glass) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stMain"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* Example buttons — smaller, secondary style */
[data-testid="stButton"] > button[kind="secondary"],
.example-btn button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    padding: 4px 8px !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(0,212,255,0.08) !important;
    border-color: rgba(0,212,255,0.3) !important;
    color: var(--accent-blue) !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-glass) !important;
    padding: 4px !important; gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(123,47,247,0.18)) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
}

[data-testid="stDataFrame"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    backdrop-filter: blur(8px) !important;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
}

[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: var(--radius-md) !important;
}

hr { border-color: var(--border-glass) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: var(--accent-blue) !important;
}

[data-baseweb="tag"] {
    background: rgba(0,212,255,0.15) !important;
    border: 1px solid rgba(0,212,255,0.30) !important;
    border-radius: 6px !important;
    color: var(--accent-blue) !important;
}
</style>
"""


def _inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_retrievers() -> dict[str, BaseRetriever]:
    """Build retriever registry used by sidebar source selection."""
    cache = QueryCache(CACHE_DB_PATH)
    return {
        "Materials Project": MaterialsProjectRetriever(),
        "BRENDA": BrendaRetriever(cache=cache),
        "Open Catalyst": OpenCatalystRetriever(),
    }


def run_retrieval(reaction: str, selected_sources: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run retrieval and return dataframe plus per-source status messages.

    Each source is queried independently so a slow or failing source never
    blocks the others. BRENDA SOAP calls can take 30-60 s.

    Args:
        reaction: Reaction text query.
        selected_sources: Tuple of source labels selected by user.

    Returns:
        Tuple of (DataFrame, dict mapping source name → status string).

    Raises:
        None.
    """
    from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

    retrievers = get_retrievers()
    source_status: dict[str, str] = {}
    all_records: list[pd.DataFrame] = []

    # Per-source timeout: BRENDA SOAP is slow, give it 60 s
    SOURCE_TIMEOUT = {"BRENDA": 60}
    DEFAULT_TIMEOUT = 25

    chosen = {name: retrievers[name] for name in selected_sources if name in retrievers}
    if not chosen:
        return pd.DataFrame(), source_status

    with ThreadPoolExecutor(max_workers=len(chosen)) as executor:
        future_map: dict[Future, str] = {
            executor.submit(retriever.search, reaction): name
            for name, retriever in chosen.items()
        }
        overall_timeout = max(SOURCE_TIMEOUT.get(n, DEFAULT_TIMEOUT) for n in chosen)

        try:
            for future in as_completed(future_map, timeout=overall_timeout):
                name = future_map[future]
                try:
                    records = future.result()
                    if records:
                        df_part = pd.DataFrame([r.model_dump() for r in records])
                        all_records.append(df_part)
                        source_status[name] = f"{len(records)} records"
                    else:
                        source_status[name] = "no results"
                except Exception as exc:  # noqa: BLE001
                    source_status[name] = "failed"
                    logger.warning("Retriever '{}' failed: {}", name, exc)

        except FuturesTimeoutError:
            # Collect whatever finished; mark the rest as timed out
            for future, name in future_map.items():
                if name in source_status:
                    continue  # already recorded
                if future.done():
                    try:
                        records = future.result()
                        if records:
                            df_part = pd.DataFrame([r.model_dump() for r in records])
                            all_records.append(df_part)
                            source_status[name] = f"{len(records)} records"
                        else:
                            source_status[name] = "no results"
                    except Exception as exc:  # noqa: BLE001
                        source_status[name] = "failed"
                        logger.warning("Retriever '{}' failed: {}", name, exc)
                else:
                    source_status[name] = "timed out"
                    future.cancel()
                    logger.warning("Retriever '{}' timed out.", name)

    if not all_records:
        return pd.DataFrame(), source_status

    combined = pd.concat(all_records, ignore_index=True)
    return combined, source_status


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    """Apply source-specific filters to the combined results dataframe.

    Each source has its own meaningful filter:
    - Materials Project: formation energy + energy above hull
    - Open Catalyst:     absolute adsorption energy
    - BRENDA:            Km value + optimal pH range

    Rows from sources with no matching filter pass through unchanged.

    Args:
        df: Combined candidate dataframe after normalization.
        filters: Filter dict from the sidebar.

    Returns:
        Filtered dataframe. Returns original if all rows would be excluded.

    Raises:
        None.
    """
    if df.empty:
        return df

    # Extract filter values with safe defaults
    mp_max_fe   = float(filters.get("mp_max_formation_energy", 0.0))
    mp_max_hull = float(filters.get("mp_max_hull", 0.5))
    oc_max_abs  = float(filters.get("oc_max_adsorption_abs", 200.0))
    brenda_max_km = float(filters.get("brenda_max_km", 200.0))
    brenda_ph   = filters.get("brenda_ph_range", (0.0, 14.0))
    try:
        ph_min, ph_max = float(brenda_ph[0]), float(brenda_ph[1])
    except (TypeError, IndexError):
        ph_min, ph_max = 0.0, 14.0

    # Check if all filters are at their broadest — skip entirely
    all_broad = (
        mp_max_fe >= 0.0
        and mp_max_hull >= 0.5
        and oc_max_abs >= 200.0
        and brenda_max_km >= 200.0
        and ph_min <= 0.0
        and ph_max >= 14.0
    )
    if all_broad:
        return df

    def _passes(row: pd.Series) -> bool:
        source = str(row.get("source", ""))
        val = row.get("activity_value")
        conditions = row.get("conditions") or {}

        if source == "Materials Project":
            # Filter by formation energy
            if val is not None:
                try:
                    if float(val) > mp_max_fe:
                        return False
                except (TypeError, ValueError):
                    pass
            # Filter by energy above hull stored in stability column
            hull = row.get("stability")
            if hull is not None:
                try:
                    if float(hull) > mp_max_hull:
                        return False
                except (TypeError, ValueError):
                    pass

        elif source == "Open Catalyst":
            # Filter by absolute adsorption energy
            if val is not None:
                try:
                    if abs(float(val)) > oc_max_abs:
                        return False
                except (TypeError, ValueError):
                    pass

        elif source == "BRENDA":
            # Filter by Km value
            if val is not None:
                try:
                    if float(val) > brenda_max_km:
                        return False
                except (TypeError, ValueError):
                    pass
            # Filter by optimal pH
            if isinstance(conditions, dict):
                ph = conditions.get("ph")
                if ph is not None:
                    try:
                        if not (ph_min <= float(ph) <= ph_max):
                            return False
                    except (TypeError, ValueError):
                        pass

        return True

    mask = df.apply(_passes, axis=1)
    filtered = df[mask].reset_index(drop=True)

    if filtered.empty:
        logger.warning(
            "Source-specific filters excluded all {} records — returning unfiltered.",
            len(df),
        )
        return df

    logger.info(
        "Source-specific filters kept {}/{} records.",
        len(filtered), len(df),
    )
    return filtered


def _source_breakdown(df: pd.DataFrame) -> str:
    if df.empty or "source" not in df.columns:
        return "No sources returned records."
    counts = Counter(df["source"].astype(str).tolist())
    return " · ".join(f"**{source}**: {count}" for source, count in counts.items())


def _fallback_demo_results(
    reaction: str,
    selected_sources: list[str],
    filters: dict[str, object],
) -> pd.DataFrame:
    temperature_range = filters.get("temperature_range", (200, 1500))
    ph_range = filters.get("ph_range", (0.0, 14.0))
    temp_bounds = (
        (int(temperature_range[0]), int(temperature_range[1]))
        if isinstance(temperature_range, tuple)
        else (200, 1500)
    )
    ph_bounds = (
        (float(ph_range[0]), float(ph_range[1]))
        if isinstance(ph_range, tuple)
        else (0.0, 14.0)
    )
    return demo_candidates_for_query(
        reaction=reaction,
        selected_sources=selected_sources,
        temperature_range=temp_bounds,
        ph_range=ph_bounds,
        limit=60,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div style="padding:1.5rem 0 1rem 0; border-bottom:1px solid rgba(255,255,255,0.07); margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:2.4rem;">⚗️</span>
                <div>
                    <h1 style="margin:0; font-size:2rem; font-weight:700; line-height:1.1;
                        background:linear-gradient(135deg,#00d4ff 0%,#7b2ff7 100%);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        CatalystIQ
                    </h1>
                    <p style="margin:0; font-size:0.85rem; color:#8892a4; letter-spacing:0.05em;">
                        AI-POWERED MOLECULAR DISCOVERY ENGINE
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_source_status(source_status: dict[str, str]) -> None:
    """Render a compact per-source retrieval status row."""
    cols = st.columns(len(source_status))
    icons = {"Materials Project": "⚗️", "BRENDA": "🧬", "Open Catalyst": "⚡"}
    for col, (name, status) in zip(cols, source_status.items()):
        with col:
            is_ok = "records" in status
            color = "#3fb950" if is_ok else "#f85149"
            dot = "🟢" if is_ok else "🔴"
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px; padding:0.6rem 0.8rem; text-align:center;">
                    <div style="font-size:1.3rem;">{icons.get(name,'🔬')}</div>
                    <div style="font-size:0.78rem; color:#c9d1d9; font-weight:600; margin:2px 0;">{name}</div>
                    <div style="font-size:0.72rem; color:{color};">{dot} {status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Run Streamlit app lifecycle and user interaction flow."""
    _inject_css()
    _render_hero()

    init_db()
    seeded = seed_demo_database(minimum_records=120)
    if seeded > 0:
        st.info(f"🗄️ Initialized local demo catalog with **{seeded}** records for offline testing.")

    retrievers = get_retrievers()
    reaction, selected_sources, filters, run_clicked = render_search_sidebar(retrievers)

    # ── Top-level navigation tabs ─────────────────────────────────────────────
    tab_retrieve, tab_generate, tab_pathway, tab_feedback = st.tabs([
        "🔍 Retrieve & Rank",
        "🤖 Generative Design",
        "⚡ Pathway Analysis",
        "🔁 Feedback Loop",
    ])

    # ── Feedback tab is always available ─────────────────────────────────────
    with tab_feedback:
        render_feedback_page()

    # ── Pathway tab is always available ──────────────────────────────────────
    with tab_pathway:
        render_pathway_page(reaction)

    # ── Retrieval + generative tabs need a search run ─────────────────────────
    with tab_retrieve:
        if not run_clicked:
            st.markdown(
                """
                <div style="text-align:center; padding:4rem 2rem;
                    background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.10);
                    border-radius:20px; margin-top:1rem;">
                    <p style="font-size:3rem; margin:0;">🔬</p>
                    <h3 style="color:#e6edf3; margin:0.8rem 0 0.4rem 0;">Ready to Discover</h3>
                    <p style="color:#8892a4; margin:0 auto; max-width:420px;">
                        Pick a quick example or enter a reaction in the sidebar, then click
                        <strong style="color:#00d4ff;">Retrieve Candidates</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with tab_generate:
                st.info("Run a retrieval first to generate novel candidates.")
            return

        if not reaction.strip():
            st.warning("⚠️ Please enter a target reaction.")
            with tab_generate:
                st.info("Enter a reaction and run retrieval to generate candidates.")
            return

        if not selected_sources:
            st.warning("⚠️ Please select at least one data source.")
            with tab_generate:
                st.info("Select at least one data source and run retrieval.")
            return

        with st.spinner("🔍 Retrieving candidates from selected sources…"):
            df, source_status = run_retrieval(
                reaction=reaction.strip(),
                selected_sources=tuple(selected_sources),
            )

        # Always show per-source status
        _render_source_status(source_status)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Respect the sidebar checkbox; also auto-fallback when all sources fail
        allow_demo_fallback: bool = bool(filters.get("allow_demo_fallback", False))
        all_failed = all("records" not in s for s in source_status.values())

        if df.empty:
            if allow_demo_fallback or all_failed:
                df = _fallback_demo_results(
                    reaction=reaction.strip(),
                    selected_sources=selected_sources,
                    filters=filters,
                )
                if all_failed and not allow_demo_fallback:
                    st.warning(
                        "⚠️ All live sources returned no data — showing **demo fallback** candidates. "
                        "Enable *Allow demo fallback* in Advanced Filters to always use this mode."
                    )
                else:
                    st.info("ℹ️ Showing demo fallback candidates (live sources returned no results).")
            else:
                st.error("❌ No records returned from the selected sources.")
                st.info(
                    "💡 **Tips:** Try enabling **Allow demo fallback** in Advanced Filters, "
                    "or use reactions like `N2 + H2 → NH3` / `CO + H2 → methane`."
                )
                with tab_generate:
                    st.info("No candidates retrieved — generative design requires at least some known candidates.")
                return

        df = normalize_units(df)
        df = flag_missing(df)

        total_before = len(df)
        df = apply_filters(df, filters)
        total_after = len(df)

        # Show active filter summary
        filter_tags = []
        mp_max_fe   = float(filters.get("mp_max_formation_energy", 0.0))
        mp_max_hull = float(filters.get("mp_max_hull", 0.5))
        oc_max_abs  = float(filters.get("oc_max_adsorption_abs", 200.0))
        brenda_km   = float(filters.get("brenda_max_km", 200.0))
        brenda_ph   = filters.get("brenda_ph_range", (0.0, 14.0))
        try:
            ph_lo, ph_hi = float(brenda_ph[0]), float(brenda_ph[1])
        except Exception:
            ph_lo, ph_hi = 0.0, 14.0

        if mp_max_fe < 0.0:
            filter_tags.append(f"⚗️ FE ≤ {mp_max_fe:.1f} eV/atom")
        if mp_max_hull < 0.5:
            filter_tags.append(f"⚗️ Hull ≤ {mp_max_hull:.2f} eV")
        if oc_max_abs < 200.0:
            filter_tags.append(f"⚡ |E_ads| ≤ {oc_max_abs:.1f} eV")
        if brenda_km < 200.0:
            filter_tags.append(f"🧬 Km ≤ {brenda_km:.1f} mM")
        if not (ph_lo <= 0.0 and ph_hi >= 14.0):
            filter_tags.append(f"🧬 pH {ph_lo:.1f}–{ph_hi:.1f}")

        if filter_tags:
            removed = total_before - total_after
            removed_str = (
                f" · <span style='color:#f85149;'>−{removed} filtered out</span>"
                if removed > 0
                else " · <span style='color:#3fb950;'>all passed</span>"
            )
            st.markdown(
                "<p style='color:#8892a4; font-size:0.8rem; margin:0 0 0.5rem 0;'>"
                "Active filters: "
                + "  ".join(
                    f"<span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.25);"
                    f"border-radius:6px;padding:1px 8px;color:#00d4ff;font-size:0.75rem;'>{t}</span>"
                    for t in filter_tags
                )
                + removed_str
                + "</p>",
                unsafe_allow_html=True,
            )

        # ── Run predictive scoring on retrieved candidates ────────────────────
        predictor = CatalystPredictor()
        df = predictor.predict(df)

        event = save_query_event(reaction=reaction.strip(), sources=selected_sources, count=len(df))
        saved_rows = save_candidates(query_event_id=event.id, df=df)

        st.markdown(
            f"""
            <div style="background:rgba(63,185,80,0.08); border:1px solid rgba(63,185,80,0.25);
                border-radius:12px; padding:0.75rem 1.2rem; margin-bottom:1rem;
                display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.2rem;">✅</span>
                <div>
                    <span style="color:#3fb950; font-weight:600;">{len(df)} candidates retrieved</span>
                    <span style="color:#8892a4; font-size:0.85rem; margin-left:8px;">{_source_breakdown(df)}</span>
                    <br/>
                    <span style="color:#8892a4; font-size:0.78rem;">{saved_rows} records saved to provenance database</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_results(df)

    # ── Generative design tab ─────────────────────────────────────────────────
    with tab_generate:
        if run_clicked and not df.empty:
            with st.spinner("🤖 Generating novel candidates…"):
                generator = CatalystGenerator(n_candidates=8)
                generated = generator.generate(reaction=reaction.strip(), known_df=df)
                # Score generated candidates too
                if generated:
                    gen_df_raw = pd.DataFrame([c.to_dict() for c in generated])
                    gen_df_scored = predictor.predict(gen_df_raw)
                    # Push scores back into generated objects
                    for i, cand in enumerate(generated):
                        if i < len(gen_df_scored):
                            row = gen_df_scored.iloc[i]
                            if pd.notna(row.get("predicted_activity")):
                                cand.predicted_activity = float(row["predicted_activity"])
                            if pd.notna(row.get("confidence")):
                                cand.confidence = float(row["confidence"])

            render_generative_page(
                known_df=df,
                generated_candidates=generated,
                reaction=reaction.strip(),
            )
        elif run_clicked:
            st.info("No candidates retrieved — generative design requires at least some known candidates.")


if __name__ == "__main__":
    main()
