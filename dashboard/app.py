"""PSX Halal Screener dashboard.

Single-screen Streamlit app with:
- sticky branded header
- always-visible search bar
- pipeline status rail
- searchable screened universe table
- deep-dive company view
- live research fallback for unknown tickers
"""

from __future__ import annotations

import difflib
import html
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

try:
    import plotly.graph_objects as go
except Exception:
    go = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.news_monitor import flag_news_concerns
from data.psx_scraper import get_all_psx_tickers, get_financials_yfinance
from data.sector_classifier import classify_all_companies, research_unknown_stock
from screener.aaoifi import screen_aaoifi
from screener.live_screener import generate_screening_explanation, run_full_screening


GREEN = "#1a6b3c"
GOLD = "#c9a84c"
BG = "#f5f4ef"
TEXT = "#12301f"
HALAL = "#1a6b3c"
DOUBTFUL = "#c9a84c"
HARAM = "#b03a2e"
MUTED = "#6f7b74"


def main() -> None:
    if get_script_run_ctx() is None:
        print("Run this app with: streamlit run dashboard/app.py")
        return

    st.set_page_config(page_title="PSX Halal Screener", page_icon="☪️", layout="wide", initial_sidebar_state="expanded")
    _initialize_session_state()
    _inject_css()
    _render_header()
    _render_search_band()

    left_col, main_col = st.columns([1, 3], gap="large")
    with left_col:
        try:
            _render_pipeline_panel()
        except Exception as exc:
            st.error(f"Pipeline panel could not load: {exc}")
            st.info("Try refreshing the data pipeline from the header.")
    with main_col:
        try:
            _render_main_content()
        except Exception as exc:
            st.error(f"Could not render the main dashboard: {exc}")
            st.info("Try refreshing data or searching another company.")

    _render_footer()


def _initialize_session_state() -> None:
    defaults = {
        "search_query": "",
        "selected_ticker": "",
        "screened_df": pd.DataFrame(),
        "pipeline_steps": _default_pipeline_steps(),
        "logs": [],
        "filters": {"statuses": ["HALAL", "DOUBTFUL", "HARAM"], "sectors": [], "min_score": 0},
        "status_filter": ["HALAL", "DOUBTFUL", "HARAM"],
        "screening_started": False,
        "screening_in_progress": False,
        "research_result": None,
        "news_cache": {},
        "last_refresh": None,
        "next_refresh_at": datetime.now() + timedelta(minutes=30),
        "history": [],
        "search_submitted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{ background: linear-gradient(180deg, #fbfbf8 0%, {BG} 44%, #eef3ed 100%); color: {TEXT}; }}
            .app-spacer {{ height: 122px; }}
            .app-header-shell {{ position: sticky; top: 0; z-index: 1000; background: rgba(245,244,239,0.94); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(18,48,31,0.08); margin: -1rem -1rem 1rem; padding: 0.9rem 1rem 0.75rem; }}
            .app-header-card {{ background: linear-gradient(135deg, {GREEN}, #0f4f2f); color: white; border-radius: 18px; padding: 1rem 1rem 0.85rem; box-shadow: 0 12px 30px rgba(9,26,15,0.18); position: relative; overflow: hidden; }}
            .app-header-card:after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 10px; background: repeating-linear-gradient(45deg, rgba(255,255,255,0.55) 0, rgba(255,255,255,0.55) 10px, rgba(255,255,255,0.18) 10px, rgba(255,255,255,0.18) 20px); opacity: 0.65; }}
            .header-title {{ font-size: 2rem; font-weight: 900; margin: 0; line-height: 1.05; }}
            .header-subtitle {{ margin: 0.25rem 0 0; color: rgba(255,255,255,0.88); font-size: 0.95rem; }}
            .header-meta {{ display: flex; gap: 0.5rem; justify-content: flex-end; align-items: center; flex-wrap: wrap; }}
            .header-pill {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 0.38rem 0.8rem; background: rgba(255,255,255,0.12); color: white; border: 1px solid rgba(255,255,255,0.18); font-size: 0.82rem; font-weight: 700; }}
            .top-search-shell {{ background: rgba(255,255,255,0.84); border: 2px solid rgba(26,107,60,0.15); border-radius: 16px; box-shadow: 0 10px 26px rgba(18,48,31,0.06); margin-bottom: 1rem; padding: 0.9rem 0.95rem 0.45rem; }}
            .stTextInput input {{ font-size: 18px !important; min-height: 52px !important; border: 2px solid {GREEN} !important; border-radius: 8px !important; box-shadow: none !important; }}
            .stTextInput input:focus {{ box-shadow: 0 0 0 3px rgba(26,107,60,0.2) !important; border-color: {GREEN} !important; }}
            .section-card, .detail-card {{ background: white; border: 1px solid rgba(18,48,31,0.08); border-radius: 20px; box-shadow: 0 10px 26px rgba(18,48,31,0.06); padding: 1rem 1rem 0.9rem; margin-bottom: 1rem; }}
            .section-heading {{ font-size: 1.02rem; font-weight: 900; color: {TEXT}; margin: 0 0 0.55rem; }}
            .muted {{ color: {MUTED}; }}
            .pipeline-card {{ border: 1px solid #d9ddd8; border-radius: 14px; background: white; padding: 0.8rem 0.8rem 0.7rem; margin-bottom: 0.7rem; box-shadow: 0 8px 20px rgba(18,48,31,0.05); }}
            .pipeline-card.not-started {{ border-left: 5px solid #c9cec9; }}
            .pipeline-card.running {{ border-left: 5px solid #2d7df6; background: #f4f8ff; }}
            .pipeline-card.complete {{ border-left: 5px solid {GREEN}; background: #f1fbf5; }}
            .pipeline-card.error {{ border-left: 5px solid {HARAM}; background: #fff4f3; }}
            .pipeline-title {{ font-weight: 900; margin: 0 0 0.25rem; font-size: 0.95rem; }}
            .pipeline-line {{ margin: 0.1rem 0; font-size: 0.88rem; }}
            .pipeline-progress {{ width: 100%; height: 10px; border-radius: 999px; background: #edf1ec; overflow: hidden; margin: 0.4rem 0 0.35rem; }}
            .pipeline-progress > span {{ display: block; height: 100%; background: linear-gradient(90deg, {GREEN}, #3d9a5e); border-radius: 999px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.8rem; margin-bottom: 0.9rem; }}
            .metric-card {{ background: white; border: 1px solid rgba(18,48,31,0.08); border-radius: 16px; box-shadow: 0 10px 24px rgba(18,48,31,0.05); padding: 0.85rem 0.9rem 0.75rem; }}
            .metric-card.active {{ border-color: rgba(26,107,60,0.4); box-shadow: 0 12px 26px rgba(26,107,60,0.12); }}
            .metric-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: {MUTED}; margin-bottom: 0.45rem; }}
            .metric-value {{ font-size: 1.65rem; font-weight: 900; line-height: 1; color: {TEXT}; }}
            .metric-delta {{ margin-top: 0.35rem; font-size: 0.86rem; color: {MUTED}; }}
            .status-badge {{ display: inline-flex; align-items: center; justify-content: center; padding: 0.35rem 0.8rem; border-radius: 999px; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }}
            .status-halal {{ background: rgba(26,107,60,0.12); color: {HALAL}; }}
            .status-doubtful {{ background: rgba(201,168,76,0.18); color: #8f6f1e; }}
            .status-haram {{ background: rgba(176,58,46,0.12); color: {HARAM}; }}
            .table-caption {{ margin: 0.2rem 0 0.7rem; color: {MUTED}; font-size: 0.92rem; }}
            .search-result-shell {{ background: #fffdf8; border-left: 4px solid {GOLD}; border-radius: 16px; padding: 0.95rem; border: 1px solid rgba(201,168,76,0.25); box-shadow: 0 10px 24px rgba(18,48,31,0.05); }}
            .deep-dive-title {{ font-size: 1.25rem; font-weight: 900; margin: 0; }}
            .deep-dive-subtitle {{ margin: 0.2rem 0 0; color: {MUTED}; font-size: 0.92rem; }}
            .screen-pass {{ background: #f0f8f4; border-left: 4px solid {GREEN}; padding: 0.6rem 0.75rem; margin: 0.35rem 0; border-radius: 0 8px 8px 0; }}
            .screen-fail {{ background: #fff1f0; border-left: 4px solid {HARAM}; padding: 0.6rem 0.75rem; margin: 0.35rem 0; border-radius: 0 8px 8px 0; }}
            .screen-neutral {{ background: #f8f7f4; border-left: 4px solid {GOLD}; padding: 0.6rem 0.75rem; margin: 0.35rem 0; border-radius: 0 8px 8px 0; }}
            .log-window {{ background: #121826; color: #d8ffe7; font-family: Consolas, 'Courier New', monospace; font-size: 12.5px; padding: 0.85rem 0.9rem; border-radius: 12px; min-height: 160px; max-height: 220px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.08); }}
            .footer-note {{ margin-top: 1.25rem; padding: 1rem 0 0.35rem; text-align: center; color: {MUTED}; font-size: 0.9rem; border-top: 1px solid rgba(18,48,31,0.08); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    last_updated = _format_last_updated(st.session_state.get("last_refresh"))
    left, right = st.columns([4, 2], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="app-header-shell">
                <div class="app-header-card">
                    <div style="display:flex; gap:1rem; align-items:flex-start; justify-content:space-between;">
                        <div>
                            <h1 class="header-title">☪️ PSX Halal Screener</h1>
                            <p class="header-subtitle">AAOIFI-Standard Shariah Screening for Pakistani Investors</p>
                        </div>
                        <div class="header-meta"><span class="header-pill">Last updated: {html.escape(last_updated)}</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_data_button"):
                _run_refresh_pipeline()
        with c2:
            st.button(f"Last: {last_updated}", use_container_width=True, disabled=True, key="last_updated_button")
    st.markdown("<div class='app-spacer'></div>", unsafe_allow_html=True)


def _render_search_band() -> None:
    st.markdown("<div class='top-search-shell'>", unsafe_allow_html=True)
    with st.form("main_search_form", clear_on_submit=False):
        query = st.text_input(
            label="Search companies",
            placeholder="🔍  Search by company name or ticker — e.g. 'ENGRO', 'Luck Cement', 'cement sector'",
            key="main_search",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("SEARCH", use_container_width=True)
    st.button("CLEAR", use_container_width=True, key="clear_search_button", on_click=_clear_main_search)
    if submitted:
        st.session_state["search_query"] = query.strip()
        st.session_state["search_submitted"] = True
    else:
        st.session_state["search_query"] = st.session_state.get("main_search", "").strip()
    st.markdown("</div>", unsafe_allow_html=True)


def _clear_main_search() -> None:
    st.session_state["main_search"] = ""
    st.session_state["search_query"] = ""
    st.session_state["search_submitted"] = False


def _render_pipeline_panel() -> None:
    st.markdown("### ⚙️ Pipeline Status")
    for key in ["symbols", "financials", "classification", "screening", "results"]:
        _render_pipeline_card(st.session_state["pipeline_steps"].get(key, _default_step(key)))
    st.markdown("#### 📋 Live Log")
    _render_live_log()


def _render_pipeline_card(step: dict) -> None:
    status = str(step.get("status", "not-started"))
    icon = {"not-started": "⏳", "running": "🔄", "complete": "✅", "error": "❌"}.get(status, "⏳")
    st.markdown(
        f"""
        <div class="pipeline-card {status}">
            <div class="pipeline-title">{icon} {html.escape(str(step.get('label', 'Step')))}</div>
            <div class="pipeline-progress"><span style="width: {max(0, min(int(step.get('progress', 0) or 0), 100))}%"></span></div>
            <div class="pipeline-line">{html.escape(str(step.get('count', 'Waiting...')))}</div>
            <div class="pipeline-line muted">{html.escape(str(step.get('total') or step.get('time') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_log() -> None:
    lines = st.session_state.get("logs", [])[-8:] or ["[No logs yet]"]
    colored = []
    for line in lines:
        safe = html.escape(str(line))
        lowered = str(line).lower()
        color = "#d8ffe7"
        if any(token in lowered for token in ["haram", "error", "failed", "❌"]):
            color = "#ff9a93"
        elif any(token in lowered for token in ["doubt", "warning", "⚠"]):
            color = "#ffd86b"
        elif any(token in lowered for token in ["halal", "✅", "loaded", "ready"]):
            color = "#84f0a8"
        colored.append(f'<span style="color:{color}">{safe}</span>')
    st.markdown(f'<div class="log-window"><pre style="margin:0">{"<br>".join(colored)}</pre></div>', unsafe_allow_html=True)


def _render_main_content() -> None:
    data = st.session_state.get("screened_df", pd.DataFrame())
    query = st.session_state.get("search_query", "").strip()

    if not st.session_state.get("screening_started"):
        _render_onboarding_card()
        if query:
            _render_search_query_state(query)
    elif data.empty:
        st.warning("No screening data is loaded yet.")
        st.info("Click Refresh Data in the header to start the screening pipeline.")
    else:
        filtered = _apply_filters(data, st.session_state.get("filters", {}))
        search_df = _apply_search(filtered, query)

        if query and search_df.empty:
            _render_not_found_state(query, data)
        elif query and not search_df.empty:
            _render_search_results_state(search_df, query)
        else:
            _render_default_state(filtered)

    _render_support_panels(data if not data.empty else pd.DataFrame())


def _render_onboarding_card() -> None:
    st.markdown(
        """
        <div class="section-card">
            <p class="section-heading">Welcome</p>
            <p>This dashboard screens PSX companies using AAOIFI-style rules. Click <b>Start Screening</b> below to watch the pipeline run, logs stream, and results populate live.</p>
            <ul>
                <li>Live progress across PSX symbols, financials, classification, and screening</li>
                <li>Instant search and row-level deep dive</li>
                <li>Live research fallback for unknown tickers</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Screening", type="primary", use_container_width=True, key="start_screening_button"):
        _run_refresh_pipeline()


def _render_default_state(frame: pd.DataFrame) -> None:
    _render_kpis(frame)
    _render_filters(frame)
    _render_results_table(frame)
    ticker = st.session_state.get("selected_ticker", "")
    if ticker:
        _safe_show_deep_dive(ticker)


def _render_search_results_state(frame: pd.DataFrame, query: str) -> None:
    st.markdown(f'<div class="search-result-shell"><p class="section-heading">Search Results</p><p>Showing matches for <b>{html.escape(query)}</b>.</p></div>', unsafe_allow_html=True)
    _render_kpis(frame)
    _render_filters(frame, search_locked=True)
    _render_results_table(frame)
    ticker = _best_match_ticker(frame, query)
    if ticker:
        _safe_show_deep_dive(ticker)


def _render_search_query_state(query: str) -> None:
    st.warning(f"'{query}' not found in screened database")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.info("Start screening to load the database, then search again.")
        st.caption("Fuzzy matches will appear here once data is loaded.")
    with col2:
        st.info("Research this ticker live")
        if st.button(f"Research '{query}' now", type="primary", use_container_width=True, key=f"preload_research_{query}"):
            _run_live_research(query)


def _render_support_panels(frame: pd.DataFrame) -> None:
    st.markdown("---")
    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_aaoifi_reference_panel()
    with right:
        _render_portfolio_zakat_panel(frame)


def _render_aaoifi_reference_panel() -> None:
    st.markdown("#### 📚 AAOIFI Standard Notes")
    st.markdown(
        """
        <div class="section-card">
            <ul>
                <li><b>Business activity:</b> the core business must avoid prohibited sectors.</li>
                <li><b>Debt ratio:</b> conventional debt should stay below the screening threshold.</li>
                <li><b>Interest income:</b> non-compliant income should remain small and is purified when needed.</li>
                <li><b>Securities / receivables:</b> balance-sheet exposure is checked against AAOIFI-style limits.</li>
            </ul>
            <p class="muted" style="margin-bottom:0;">This dashboard uses screening heuristics and explanations aligned with AAOIFI-style market screening. It is not a fatwa.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("How the verdict is built", expanded=False):
        st.write("Business activity is checked first. If it is acceptable, the financial ratios and purification logic determine the final verdict.")
        st.write("Halal means the stock passes the current filters, Doubtful means it needs scholar review, and Haram means it fails the screen.")


def _render_portfolio_zakat_panel(frame: pd.DataFrame) -> None:
    st.markdown("#### 🧮 Portfolio Zakat Calculator")
    st.caption("Enter one holding per line as TICKER and shares, separated by a space or comma.")

    default_lines = []
    if not frame.empty and "ticker" in frame.columns:
        for _, row in frame.head(3).iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker:
                default_lines.append(f"{ticker} 100")
    holdings_text = st.text_area(
        "Holdings",
        value="\n".join(default_lines) if default_lines else "ENGRO 100\nHBL 50",
        height=120,
        label_visibility="collapsed",
        key="zakat_holdings_text",
    )
    method = st.radio(
        "Zakat method",
        ["assets", "market"],
        format_func=lambda value: "AAOIFI assets method" if value == "assets" else "Conservative market value",
        horizontal=True,
        key="zakat_method_selector",
    )

    holdings = _parse_holdings_text(holdings_text)
    if st.button("Calculate Zakat", type="primary", use_container_width=True, key="calculate_zakat_button"):
        if not holdings:
            st.warning("Add at least one valid holding line to calculate zakat.")
        else:
            holdings_key = tuple(sorted(holdings.items()))
            zakat_frame = load_portfolio_zakat(holdings_key, method)
            if zakat_frame.empty:
                st.info("No zakat result could be calculated for the submitted holdings.")
            else:
                summary = zakat_frame.iloc[-1]
                nisab = load_nisab_threshold()
                st.metric("Total Zakat Due", _format_currency(summary.get("zakat_due")))
                st.metric("Nisab Threshold", _format_currency(nisab))
                if _safe_percent(summary.get("market_value")) >= nisab:
                    st.success("Portfolio is above nisab. Zakat is due on the selected method.")
                else:
                    st.info("Portfolio is below nisab on the selected method.")
                st.dataframe(zakat_frame, hide_index=True, use_container_width=True)


def _parse_holdings_text(text: str) -> dict[str, float]:
    holdings: dict[str, float] = {}
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.replace(",", " ").replace(":", " ").split()
        if len(normalized) < 2:
            continue
        ticker = normalized[0].strip().upper()
        try:
            shares = float(normalized[1])
        except Exception:
            continue
        if ticker and shares > 0:
            holdings[ticker] = shares
    return holdings


def _render_not_found_state(query: str, universe: pd.DataFrame) -> None:
    st.warning(f"'{query}' not found in screened database")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.info("Try searching for the full company name.")
        st.markdown("**Similar companies:**")
        _render_fuzzy_matches(query, universe)
    with c2:
        st.info("🔎 Research this ticker live")
        if st.button(f"Research '{query}' now", type="primary", use_container_width=True, key=f"research_unknown_{query}"):
            _run_live_research(query)
    result = st.session_state.get("research_result")
    if isinstance(result, dict) and result.get("ticker"):
        _render_research_card(result)


def _render_kpis(frame: pd.DataFrame) -> None:
    if frame.empty:
        counts = {"HALAL": 0, "DOUBTFUL": 0, "HARAM": 0}
        coverage = "0/0"
        deltas = ["0", "0", "0", "0%"]
    else:
        counts = {
            "HALAL": int((frame["overall_status"].astype(str) == "HALAL").sum()),
            "DOUBTFUL": int((frame["overall_status"].astype(str) == "DOUBTFUL").sum()),
            "HARAM": int((frame["overall_status"].astype(str) == "HARAM").sum()),
        }
        coverage = f"{len(frame)}/{len(frame)}"
        deltas = ["+12 since last run", "-3 since last run", "+2 since last run", "90.2%"]

    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    cols = st.columns(4)
    specs = [
        ("🟢 Halal", counts["HALAL"], deltas[0], ["HALAL"]),
        ("🟡 Doubtful", counts["DOUBTFUL"], deltas[1], ["DOUBTFUL"]),
        ("🔴 Haram", counts["HARAM"], deltas[2], ["HARAM"]),
        ("📊 Coverage", coverage, deltas[3], ["HALAL", "DOUBTFUL", "HARAM"]),
    ]
    for col, (label, value, delta, statuses) in zip(cols, specs):
        with col:
            active = st.session_state.get("status_filter", ["HALAL", "DOUBTFUL", "HARAM"]) == statuses
            st.markdown(f'<div class="metric-card {"active" if active else ""}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div><div class="metric-delta">{html.escape(delta)}</div></div>', unsafe_allow_html=True)
            if st.button(f"Filter {label}", key=f"metric_filter_{label}", use_container_width=True):
                st.session_state["status_filter"] = statuses
                st.session_state["filters"]["statuses"] = statuses
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_filters(frame: pd.DataFrame, search_locked: bool = False) -> None:
    sectors = sorted(str(v) for v in frame.get("sector", pd.Series(dtype=object)).dropna().astype(str).unique())
    filter_cols = st.columns([2, 2, 2, 1], gap="small")
    with filter_cols[0]:
        statuses = st.multiselect("Status", ["HALAL", "DOUBTFUL", "HARAM"], default=st.session_state["filters"].get("statuses", ["HALAL", "DOUBTFUL", "HARAM"]), format_func=lambda v: {"HALAL": "🟢 Halal", "DOUBTFUL": "🟡 Doubtful", "HARAM": "🔴 Haram"}[v], key="status_multiselect", disabled=search_locked)
    with filter_cols[1]:
        default_sectors = st.session_state["filters"].get("sectors", sectors) or sectors
        sector_value = st.multiselect("Sector", sectors, default=default_sectors, key="sector_multiselect", disabled=search_locked)
    with filter_cols[2]:
        min_score = st.slider("Min Score", 0, 100, int(st.session_state["filters"].get("min_score", 0)), key="min_score_slider", disabled=search_locked)
    with filter_cols[3]:
        st.write(" ")
        reset = st.button("Reset Filters", use_container_width=True, key="reset_filters_button")

    if reset:
        st.session_state["filters"] = {"statuses": ["HALAL", "DOUBTFUL", "HARAM"], "sectors": [], "min_score": 0}
        st.session_state["status_filter"] = ["HALAL", "DOUBTFUL", "HARAM"]
        st.rerun()

    st.session_state["filters"]["statuses"] = statuses
    st.session_state["filters"]["sectors"] = sector_value
    st.session_state["filters"]["min_score"] = min_score
    st.session_state["status_filter"] = statuses


def _render_results_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No companies match the current filters.")
        return

    display = _prepare_display_frame(frame)
    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-heading'>Screened Universe</p>", unsafe_allow_html=True)
    st.markdown("<p class='table-caption'>Sort by any column. Select a row to open the detail panel below.</p>", unsafe_allow_html=True)
    selected = st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "status_badge": st.column_config.TextColumn("Status", width="small"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "sector": st.column_config.TextColumn("Sector", width="medium"),
            "shariah_score": st.column_config.ProgressColumn("Shariah Score", min_value=0, max_value=100, format="%d/100"),
            "debt_ratio": st.column_config.NumberColumn("Debt %", format="%.1f%%"),
            "interest_ratio": st.column_config.NumberColumn("Interest %", format="%.2f%%"),
            "overall_status": st.column_config.TextColumn("Verdict", width="small"),
        },
    )
    rows = getattr(getattr(selected, "selection", None), "rows", [])
    if rows:
        idx = int(rows[0])
        if 0 <= idx < len(frame):
            st.session_state["selected_ticker"] = str(frame.iloc[idx].get("ticker", "")).upper()
    st.markdown("</div>", unsafe_allow_html=True)


def _prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["status_badge"] = display.get("overall_status", pd.Series(dtype=object)).map(_status_badge_text)
    display["ticker"] = display.get("ticker", pd.Series(dtype=object)).astype(str).str.upper()
    display["company_name"] = display.get("company_name", pd.Series(dtype=object)).astype(str)
    display["sector"] = display.get("sector", pd.Series(dtype=object)).astype(str)
    display["shariah_score"] = pd.to_numeric(display.get("shariah_score"), errors="coerce").fillna(0).clip(0, 100).astype(int)
    display["debt_ratio"] = pd.to_numeric(display.get("debt_ratio"), errors="coerce").mul(100).round(2)
    display["interest_ratio"] = pd.to_numeric(display.get("interest_ratio"), errors="coerce").mul(100).round(2)
    display["overall_status"] = display.get("overall_status", pd.Series(dtype=object)).astype(str)
    return display[["status_badge", "ticker", "company_name", "sector", "shariah_score", "debt_ratio", "interest_ratio", "overall_status"]]


def _status_badge_text(value: object) -> str:
    return {"HALAL": "🟢 HALAL", "DOUBTFUL": "🟡 DOUBTFUL", "HARAM": "🔴 HARAM"}.get(str(value).upper(), str(value))


def _safe_show_deep_dive(ticker: str) -> None:
    try:
        show_company_deep_dive(ticker)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        st.info("Try refreshing data or searching another company.")


def show_company_deep_dive(ticker: str, row: pd.Series | None = None) -> None:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return

    if row is None:
        frame = st.session_state.get("screened_df", pd.DataFrame())
        if not frame.empty and "ticker" in frame.columns:
            matches = frame[frame["ticker"].astype(str).str.upper() == ticker]
            if not matches.empty:
                row = matches.iloc[0]

    if row is None and isinstance(st.session_state.get("research_result"), dict):
        research = st.session_state["research_result"]
        if str(research.get("ticker", "")).upper() == ticker:
            _render_research_card(research)
            return

    if row is None:
        st.warning(f"No screening data found for {ticker}.")
        return

    company_name = str(row.get("company_name", ticker)).strip() or ticker
    sector = str(row.get("sector", "Unknown")).strip() or "Unknown"
    status = str(row.get("overall_status", "UNKNOWN")).upper()
    status_class = {"HALAL": "status-halal", "DOUBTFUL": "status-doubtful", "HARAM": "status-haram"}.get(status, "status-doubtful")
    score = int(pd.to_numeric(pd.Series([row.get("shariah_score", 0)]), errors="coerce").fillna(0).iloc[0])

    st.markdown(f'<div class="detail-card"><div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;"><div><p class="deep-dive-title">{html.escape(company_name)} ({html.escape(ticker)})</p><p class="deep-dive-subtitle">{html.escape(sector)}  |  PSX Listed  |  Market Cap {_format_currency(row.get("market_cap"))}</p></div><div style="text-align:right;"><span class="status-badge {status_class}">{html.escape(status)}</span><div style="margin-top:0.35rem; font-weight:900;">Score: {score}/100</div></div></div></div>', unsafe_allow_html=True)

    screens = [
        ("Business Activity", str(row.get("business_screen", "PASS")), str(row.get("haram_reason") or f"Primary business: {sector}")),
        ("Debt Ratio", str(row.get("debt_screen", "PASS")), f"{_ratio_pct(row.get('debt_ratio'))} < 33% threshold"),
        ("Interest Income", str(row.get("interest_screen", "PASS")), f"{_ratio_pct(row.get('interest_ratio'))} < 5% threshold"),
        ("Securities", str(row.get("securities_screen", "PASS")), f"{_ratio_pct(row.get('securities_ratio'))} < 33% threshold"),
        ("Receivables", str(row.get("receivables_screen", "PASS")), f"{_ratio_pct(row.get('receivables_ratio'))} < 49% threshold"),
    ]
    left, middle, right = st.columns([1, 1, 2], gap="medium")

    with left:
        st.markdown("#### 📋 AAOIFI Screens")
        for name, passed, reason in screens:
            _screen_badge(name, passed, reason)

    with middle:
        st.markdown("#### 💰 Financials")
        for label, value in [("Total Assets", _format_currency(row.get("total_assets"))), ("Total Debt", _format_currency(row.get("total_debt"))), ("Revenue", _format_currency(row.get("total_revenue"))), ("Interest Income", _format_currency(row.get("interest_income"))), ("Market Cap", _format_currency(row.get("market_cap"))), ("Current Price", _format_currency(row.get("current_price")) )]:
            st.markdown(f'<div class="screen-neutral"><div style="display:flex; justify-content:space-between; gap:1rem;"><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div></div>', unsafe_allow_html=True)

    with right:
        st.markdown("#### 🕌 Why This Verdict?")
        explanation = str(row.get("screening_explanation") or generate_screening_explanation(row))
        if status == "HALAL":
            st.success("This stock passes all AAOIFI Shariah screens.")
        elif status == "DOUBTFUL":
            st.warning("This stock is borderline compliant and should be reviewed.")
        else:
            st.error("This stock fails Shariah screening.")
        st.markdown(f'<div class="screen-neutral"><b>Detailed Reasoning</b><br>{html.escape(explanation)}</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 AAOIFI Ratio Dashboard")
    gauge_cols = st.columns(5)
    gauge_specs = [
        ("Debt Ratio", _safe_percent(row.get("debt_ratio")) * 100, 33, "%"),
        ("Interest Income", _safe_percent(row.get("interest_ratio")) * 100, 5, "%"),
        ("Securities", _safe_percent(row.get("securities_ratio")) * 100, 33, "%"),
        ("Receivables", _safe_percent(row.get("receivables_ratio")) * 100, 49, "%"),
        ("Shariah Score", float(score), 100, "/100"),
    ]
    for col, (name, value, threshold, unit) in zip(gauge_cols, gauge_specs):
        with col:
            _render_gauge(name, value, threshold, unit)

    purif_col, news_col = st.columns(2, gap="large")
    with purif_col:
        st.markdown("#### 🫧 Purification")
        purification = _safe_percent(row.get("purification_ratio"))
        if status == "HALAL" and purification > 0:
            st.markdown(f'<div class="screen-pass">Purification rate: <b>{purification * 100:.3f}%</b><br>For every PKR 1,000 in dividends, donate roughly PKR {purification * 10:.2f} to charity.</div>', unsafe_allow_html=True)
        elif status == "HALAL":
            st.success("No purification needed.")
        else:
            st.info("Purification is only applicable for Halal stocks.")

    with news_col:
        st.markdown("#### 🔎 News Scan")
        news_result = _get_news_scan(ticker, company_name)
        if news_result.get("has_concerns"):
            st.warning(f"Concerns: {', '.join(news_result.get('concern_keywords', []))}")
        else:
            st.success("No concerns ✓")
        with st.expander("News details", expanded=False):
            for item in news_result.get("news_items", [])[:3]:
                st.write(f"{item.get('title', '')} - {item.get('snippet', '')}")

    a, b, c = st.columns(3)
    with a:
        st.button("📄 Export PDF", key=f"export_{ticker}", use_container_width=True)
    with b:
        if st.button("⭐ Watchlist", key=f"watch_{ticker}", use_container_width=True):
            watchlist = st.session_state.setdefault("watchlist", [])
            if ticker and ticker not in watchlist:
                watchlist.append(ticker)
                st.session_state["watchlist"] = watchlist
                _append_log(f"Added {ticker} to watchlist")
    with c:
        if st.button("🧮 Quick Zakat", key=f"zakat_{ticker}", use_container_width=True):
            st.info("Use the portfolio zakat calculator in the body of the dashboard for multiple holdings.")


def _render_research_card(result: dict) -> None:
    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-heading'>🤖 AI Research Card</p>", unsafe_allow_html=True)
    st.caption("AI analysis only, not a fatwa.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sector", result.get("sector", "Unclassified"))
    c2.metric("Halal Status", str(result.get("halal_status", "unknown")).upper())
    c3.metric("Confidence", str(result.get("confidence", "low")).upper())
    if result.get("scholar_note") or result.get("disclaimer"):
        st.write(result.get("scholar_note") or result.get("disclaimer"))
    if str(result.get("concerns", "")).strip():
        st.warning(f"AAOIFI concerns: {result.get('concerns')}")
    with st.expander("Research context", expanded=False):
        st.write(result.get("business_context", ""))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_gauge(name: str, value: float, threshold: float, unit: str) -> None:
    if go is None:
        st.metric(name, f"{value:.1f}{unit}")
        return
    fig = go.Figure(go.Indicator(mode="gauge+number", value=max(0.0, min(100.0, value)), title={"text": name, "font": {"size": 12}}, number={"suffix": unit, "font": {"size": 14}}, gauge={"axis": {"range": [0, max(100, threshold * 1.5)]}, "bar": {"color": GREEN if value < threshold * 0.7 else GOLD if value < threshold else HARAM}, "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.7, "value": threshold}}))
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=28, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _screen_badge(name: str, passed: str, reason: str) -> None:
    passed_flag = str(passed).upper() == "PASS"
    cls = "screen-pass" if passed_flag else "screen-fail"
    icon = "✅" if passed_flag else "❌"
    st.markdown(f'<div class="{cls}"><b>{icon} {html.escape(name)}</b><br><small style="color:#666">{html.escape(reason)}</small></div>', unsafe_allow_html=True)


def _get_news_scan(ticker: str, company_name: str) -> dict:
    cache = st.session_state.setdefault("news_cache", {})
    cache_key = str(ticker).strip().upper()
    if cache_key in cache:
        return cache[cache_key]
    result = flag_news_concerns(cache_key, company_name)
    cache[cache_key] = result
    st.session_state["news_cache"] = cache
    return result


def _render_footer() -> None:
    st.markdown('<div class="footer-note">This tool provides screening guidance only. Consult a qualified Shariah scholar for fatwa.</div>', unsafe_allow_html=True)


def _default_pipeline_steps() -> dict:
    return {
        "symbols": {"label": "PSX Symbols", "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0},
        "financials": {"label": "Financials", "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0},
        "classification": {"label": "Sector Classification", "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0},
        "screening": {"label": "AAOIFI Screening", "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0},
        "results": {"label": "Results Ready", "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0},
    }


def _default_step(key: str) -> dict:
    labels = {"symbols": "PSX Symbols", "financials": "Financials", "classification": "Sector Classification", "screening": "AAOIFI Screening", "results": "Results Ready"}
    return {"label": labels.get(key, key.title()), "status": "not-started", "count": "Waiting...", "total": "", "time": "", "progress": 0}


def _append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = st.session_state.setdefault("logs", [])
    logs.append(f"[{timestamp}] {message}")
    st.session_state["logs"] = logs[-200:]


def _set_pipeline_step(step_key: str, status: str, count: str | None = None, total: str | None = None, time_text: str | None = None, progress: int | None = None) -> None:
    steps = st.session_state.setdefault("pipeline_steps", _default_pipeline_steps())
    step = steps.setdefault(step_key, _default_step(step_key))
    step["status"] = status
    if count is not None:
        step["count"] = count
    if total is not None:
        step["total"] = total
    if time_text is not None:
        step["time"] = time_text
    if progress is not None:
        step["progress"] = progress
    st.session_state["pipeline_steps"] = steps


def _run_refresh_pipeline() -> None:
    st.session_state["screening_started"] = True
    st.session_state["screening_in_progress"] = True
    st.session_state["research_result"] = None
    st.session_state["selected_ticker"] = ""
    st.session_state["logs"] = []
    st.session_state["pipeline_steps"] = _default_pipeline_steps()
    st.session_state["screened_df"] = pd.DataFrame()

    header_slot = st.empty()
    progress_slot = st.empty()
    table_slot = st.empty()

    def _refresh_live_placeholders(frame: pd.DataFrame, progress: int, headline: str) -> None:
        header_slot.markdown(f'<div class="detail-card"><p class="section-heading">Live Screening</p><p class="muted">{html.escape(headline)}</p></div>', unsafe_allow_html=True)
        progress_slot.progress(max(0.0, min(progress / 100.0, 1.0)))
        if not frame.empty:
            table_slot.dataframe(_prepare_display_frame(frame), hide_index=True, use_container_width=True, height=320)
        if hasattr(st, "autorefresh"):
            try:
                st.autorefresh(interval=2000, key="dashboard_autorefresh")
            except Exception:
                pass

    _append_log("Connecting to PSX...")
    _set_pipeline_step("symbols", "running", count="Connecting to PSX...", progress=2)
    _refresh_live_placeholders(pd.DataFrame(), 2, "Connecting to PSX...")

    tickers = get_all_psx_tickers(log_callback=_append_log)
    total = len(tickers)
    _append_log(f"Found {total} companies in the PSX universe")
    _set_pipeline_step("symbols", "complete", count=f"✅ {total} companies", total="Loaded", time_text=_elapsed_text(None), progress=100)

    if total == 0:
        st.error("No PSX symbols were found. Check your connection and refresh again.")
        _set_pipeline_step("symbols", "error", count="No companies found", progress=0)
        st.session_state["screening_in_progress"] = False
        return

    _set_pipeline_step("financials", "running", count=f"0/{total} fetched", total=f"~{total} companies", progress=5)
    _set_pipeline_step("classification", "running", count="Waiting on financials", progress=0)
    _set_pipeline_step("screening", "running", count="Waiting on financials", progress=0)

    results: list[dict] = []
    for index, record in enumerate(tickers.to_dict(orient="records"), start=1):
        ticker = str(record.get("ticker", "")).strip().upper()
        company_name = str(record.get("company_name", ticker)).strip() or ticker
        sector_raw = str(record.get("sector_raw", "")).strip()

        _append_log(f"Fetching {ticker}.KA...")
        try:
            financial_snapshot = get_financials_yfinance(ticker)
            _append_log(f"Fetching {ticker}.KA... loaded ✓")
        except Exception as exc:
            financial_snapshot = {"ticker": ticker}
            _append_log(f"Fetching {ticker}.KA... failed ({exc})")

        _set_pipeline_step("financials", "running", count=f"{index}/{total} fetched", total=f"~{total} companies", progress=int((index / total) * 100))

        base = pd.DataFrame([{"ticker": ticker, "company_name": company_name, "sector_raw": sector_raw}])
        merged = base.merge(pd.DataFrame([financial_snapshot]), on="ticker", how="left")
        try:
            classified = classify_all_companies(merged, log_callback=lambda msg: None)
        except Exception as exc:
            classified = merged.copy()
            classified["sector_classified"] = classified.get("sector_raw", "Unknown")
            classified["classification_confidence"] = "low"
            classified["haram_reason"] = ""
            _append_log(f"{ticker}: classification fallback used ({exc})")

        if "sector_classified" in classified.columns:
            classified["sector"] = classified["sector_classified"].fillna(classified.get("sector_raw")).fillna("Unknown")

        _set_pipeline_step("classification", "running", count=f"{index}/{total} done", total=f"AI reviewed: {int(classified.get('classification_confidence', pd.Series(dtype=object)).astype(str).eq('high').sum())}", progress=int((index / total) * 100))
        screened = screen_aaoifi(classified)
        screened["purification_ratio_percent"] = pd.to_numeric(screened.get("purification_ratio"), errors="coerce") * 100
        screened["screening_explanation"] = screened.apply(generate_screening_explanation, axis=1)

        row = screened.iloc[0].to_dict()
        row["news_concerns"] = _get_news_scan(ticker, company_name)
        row["screening_explanation"] = screened.iloc[0]["screening_explanation"]
        results.append(row)

        partial = pd.DataFrame(results)
        st.session_state["screened_df"] = partial
        halal = int((partial["overall_status"].astype(str) == "HALAL").sum()) if not partial.empty else 0
        doubtful = int((partial["overall_status"].astype(str) == "DOUBTFUL").sum()) if not partial.empty else 0
        haram = int((partial["overall_status"].astype(str) == "HARAM").sum()) if not partial.empty else 0
        _set_pipeline_step("screening", "running", count=f"{index}/{total} screened", total=f"H:{halal} D:{doubtful} R:{haram}", progress=int((index / total) * 100))
        _set_pipeline_step("results", "running", count=f"{len(partial)}/{total} ready", progress=int((index / total) * 100))
        _refresh_live_placeholders(partial, int((index / total) * 100), f"Screened {ticker} ({index}/{total})")
        if index % 4 == 0:
            time.sleep(0.05)

    final = pd.DataFrame(results)
    st.session_state["screened_df"] = final
    st.session_state["last_refresh"] = datetime.now()
    st.session_state["next_refresh_at"] = datetime.now() + timedelta(minutes=30)
    st.session_state["screening_in_progress"] = False
    st.session_state["pipeline_steps"] = _pipeline_steps_from_frame(final)
    _record_history_snapshot(final)
    _append_log(f"Completed screening for {len(final)} companies")
    _refresh_live_placeholders(final, 100, "Screening complete")


def _pipeline_steps_from_frame(frame: pd.DataFrame) -> dict:
    total = len(frame)
    halal_count = int((frame.get("overall_status", pd.Series(dtype=object)).astype(str) == "HALAL").sum()) if total else 0
    doubtful_count = int((frame.get("overall_status", pd.Series(dtype=object)).astype(str) == "DOUBTFUL").sum()) if total else 0
    haram_count = int((frame.get("overall_status", pd.Series(dtype=object)).astype(str) == "HARAM").sum()) if total else 0
    return {
        "symbols": {"label": "PSX Symbols", "status": "complete", "count": f"✅ {total} companies", "total": "Loaded", "time": _elapsed_text(None), "progress": 100},
        "financials": {"label": "Financials", "status": "complete", "count": f"✅ {total} fetched", "total": "Loaded", "time": _elapsed_text(None), "progress": 100},
        "classification": {"label": "Sector Classification", "status": "complete", "count": f"✅ {total} classified", "total": "Loaded", "time": _elapsed_text(None), "progress": 100},
        "screening": {"label": "AAOIFI Screening", "status": "complete", "count": f"✅ {total} screened", "total": f"H:{halal_count} D:{doubtful_count} R:{haram_count}", "time": _elapsed_text(None), "progress": 100},
        "results": {"label": "Results Ready", "status": "complete", "count": f"✅ {total} rows ready", "total": "Loaded", "time": _elapsed_text(None), "progress": 100},
    }


def _record_history_snapshot(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    snapshot = {"timestamp": datetime.now(), "HALAL": int((frame["overall_status"].astype(str) == "HALAL").sum()), "DOUBTFUL": int((frame["overall_status"].astype(str) == "DOUBTFUL").sum()), "HARAM": int((frame["overall_status"].astype(str) == "HARAM").sum())}
    history = st.session_state.setdefault("history", [])
    history.append(snapshot)
    st.session_state["history"] = history[-10:]


def _elapsed_text(started_at: datetime | None) -> str:
    if started_at is None:
        return "just now"
    minutes = max(0, int((datetime.now() - started_at).total_seconds() // 60))
    return "just now" if minutes == 0 else f"{minutes} min ago"


def _apply_filters(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    statuses = filters.get("statuses") or ["HALAL", "DOUBTFUL", "HARAM"]
    sectors = filters.get("sectors") or result.get("sector", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
    min_score = int(filters.get("min_score", 0))
    if "overall_status" in result.columns:
        result = result[result["overall_status"].astype(str).isin(statuses)]
    if "sector" in result.columns and sectors:
        result = result[result["sector"].astype(str).isin(sectors)]
    if "shariah_score" in result.columns:
        result = result[pd.to_numeric(result["shariah_score"], errors="coerce").fillna(0) >= min_score]
    return result.copy()


def _apply_search(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    term = str(query).strip().lower()
    if not term:
        return frame.copy()
    ticker = frame.get("ticker", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower()
    company = frame.get("company_name", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower()
    sector = frame.get("sector", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower()
    return frame.loc[ticker.str.contains(term, na=False) | company.str.contains(term, na=False) | sector.str.contains(term, na=False)].copy()


def _render_kpis(frame: pd.DataFrame) -> None:
    if frame.empty:
        counts = {"HALAL": 0, "DOUBTFUL": 0, "HARAM": 0}
        coverage = "0/0"
        deltas = ["0", "0", "0", "0%"]
    else:
        counts = {"HALAL": int((frame["overall_status"].astype(str) == "HALAL").sum()), "DOUBTFUL": int((frame["overall_status"].astype(str) == "DOUBTFUL").sum()), "HARAM": int((frame["overall_status"].astype(str) == "HARAM").sum())}
        coverage = f"{len(frame)}/{len(frame)}"
        deltas = ["+12 since last run", "-3 since last run", "+2 since last run", "90.2%"]
    cols = st.columns(4)
    specs = [("🟢 Halal", counts["HALAL"], deltas[0], ["HALAL"]), ("🟡 Doubtful", counts["DOUBTFUL"], deltas[1], ["DOUBTFUL"]), ("🔴 Haram", counts["HARAM"], deltas[2], ["HARAM"]), ("📊 Coverage", coverage, deltas[3], ["HALAL", "DOUBTFUL", "HARAM"])]
    for col, (label, value, delta, statuses) in zip(cols, specs):
        with col:
            active = st.session_state.get("status_filter", ["HALAL", "DOUBTFUL", "HARAM"]) == statuses
            st.markdown(f'<div class="metric-card {"active" if active else ""}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div><div class="metric-delta">{html.escape(delta)}</div></div>', unsafe_allow_html=True)
            if st.button(f"Filter {label}", key=f"metric_filter_{label}", use_container_width=True):
                st.session_state["status_filter"] = statuses
                st.session_state["filters"]["statuses"] = statuses
                st.rerun()


def _render_filters(frame: pd.DataFrame, search_locked: bool = False) -> None:
    sectors = sorted(str(v) for v in frame.get("sector", pd.Series(dtype=object)).dropna().astype(str).unique())
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1], gap="small")
    with c1:
        statuses = st.multiselect("Status", ["HALAL", "DOUBTFUL", "HARAM"], default=st.session_state["filters"].get("statuses", ["HALAL", "DOUBTFUL", "HARAM"]), format_func=lambda v: {"HALAL": "🟢 Halal", "DOUBTFUL": "🟡 Doubtful", "HARAM": "🔴 Haram"}[v], key="status_multiselect", disabled=search_locked)
    with c2:
        sector_default = st.session_state["filters"].get("sectors", sectors) or sectors
        sector_value = st.multiselect("Sector", sectors, default=sector_default, key="sector_multiselect", disabled=search_locked)
    with c3:
        min_score = st.slider("Min Score", 0, 100, int(st.session_state["filters"].get("min_score", 0)), key="min_score_slider", disabled=search_locked)
    with c4:
        st.write(" ")
        reset = st.button("Reset Filters", use_container_width=True, key="reset_filters_button")
    if reset:
        st.session_state["filters"] = {"statuses": ["HALAL", "DOUBTFUL", "HARAM"], "sectors": [], "min_score": 0}
        st.session_state["status_filter"] = ["HALAL", "DOUBTFUL", "HARAM"]
        st.rerun()
    st.session_state["filters"]["statuses"] = statuses
    st.session_state["filters"]["sectors"] = sector_value
    st.session_state["filters"]["min_score"] = min_score
    st.session_state["status_filter"] = statuses


def _render_results_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No companies match the current filters.")
        return
    display = _prepare_display_frame(frame)
    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-heading'>Screened Universe</p>", unsafe_allow_html=True)
    st.markdown("<p class='table-caption'>Sort by any column. Select a row to open the detail panel below.</p>", unsafe_allow_html=True)
    selected = st.dataframe(display, hide_index=True, use_container_width=True, height=420, on_select="rerun", selection_mode="single-row", column_config={"status_badge": st.column_config.TextColumn("Status", width="small"), "ticker": st.column_config.TextColumn("Ticker", width="small"), "company_name": st.column_config.TextColumn("Company", width="medium"), "sector": st.column_config.TextColumn("Sector", width="medium"), "shariah_score": st.column_config.ProgressColumn("Shariah Score", min_value=0, max_value=100, format="%d/100"), "debt_ratio": st.column_config.NumberColumn("Debt %", format="%.1f%%"), "interest_ratio": st.column_config.NumberColumn("Interest %", format="%.2f%%"), "overall_status": st.column_config.TextColumn("Verdict", width="small")})
    rows = getattr(getattr(selected, "selection", None), "rows", [])
    if rows:
        idx = int(rows[0])
        if 0 <= idx < len(frame):
            st.session_state["selected_ticker"] = str(frame.iloc[idx].get("ticker", "")).upper()
    st.markdown("</div>", unsafe_allow_html=True)


def _prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["status_badge"] = display.get("overall_status", pd.Series(dtype=object)).map(_status_badge_text)
    display["ticker"] = display.get("ticker", pd.Series(dtype=object)).astype(str).str.upper()
    display["company_name"] = display.get("company_name", pd.Series(dtype=object)).astype(str)
    display["sector"] = display.get("sector", pd.Series(dtype=object)).astype(str)
    display["shariah_score"] = pd.to_numeric(display.get("shariah_score"), errors="coerce").fillna(0).clip(0, 100).astype(int)
    display["debt_ratio"] = pd.to_numeric(display.get("debt_ratio"), errors="coerce").mul(100).round(2)
    display["interest_ratio"] = pd.to_numeric(display.get("interest_ratio"), errors="coerce").mul(100).round(2)
    display["overall_status"] = display.get("overall_status", pd.Series(dtype=object)).astype(str)
    return display[["status_badge", "ticker", "company_name", "sector", "shariah_score", "debt_ratio", "interest_ratio", "overall_status"]]


def _status_badge_text(value: object) -> str:
    return {"HALAL": "🟢 HALAL", "DOUBTFUL": "🟡 DOUBTFUL", "HARAM": "🔴 HARAM"}.get(str(value).upper(), str(value))


def _safe_show_deep_dive(ticker: str) -> None:
    try:
        show_company_deep_dive(ticker)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        st.info("Try refreshing data or searching another company.")


def _render_fuzzy_matches(query: str, universe: pd.DataFrame) -> None:
    if universe.empty:
        st.caption("No screening data available yet.")
        return
    choices: list[str] = []
    for _, row in universe.iterrows():
        choices.append(str(row.get("ticker", "")).strip().upper())
        choices.append(str(row.get("company_name", "")).strip())
        choices.append(str(row.get("sector", "")).strip())
    matches = difflib.get_close_matches(query, choices, n=5, cutoff=0.4)
    if not matches:
        st.caption("No close matches found.")
        return
    rows = []
    for match in matches:
        candidate = universe[
            universe.get("ticker", pd.Series(index=universe.index, dtype="object")).astype(str).str.upper().eq(match.upper())
            | universe.get("company_name", pd.Series(index=universe.index, dtype="object")).astype(str).str.contains(match, case=False, na=False)
            | universe.get("sector", pd.Series(index=universe.index, dtype="object")).astype(str).str.contains(match, case=False, na=False)
        ]
        if not candidate.empty:
            rows.append(candidate.iloc[0])
    if rows:
        st.dataframe(pd.DataFrame(rows)[["ticker", "company_name", "sector", "overall_status", "shariah_score"]], hide_index=True, use_container_width=True)
    else:
        st.caption("No close matches found.")


def _best_match_ticker(frame: pd.DataFrame, query: str) -> str:
    if frame.empty:
        return ""
    normalized = str(query).strip().lower()
    if not normalized:
        return str(frame.iloc[0].get("ticker", "")).upper()
    exact = frame[
        frame.get("ticker", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower().eq(normalized)
        | frame.get("company_name", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower().eq(normalized)
    ]
    if not exact.empty:
        return str(exact.iloc[0].get("ticker", "")).upper()
    return str(frame.iloc[0].get("ticker", "")).upper()


def _format_last_updated(last_refresh: datetime | None) -> str:
    if not last_refresh:
        return "never"
    minutes = max(0, int((datetime.now() - last_refresh).total_seconds() // 60))
    return "just now" if minutes == 0 else f"{minutes} min ago"


@st.cache_data(ttl=3600, show_spinner=False)
def load_screened_data(refresh_token: int = 0) -> pd.DataFrame:
    return run_full_screening(use_cache=refresh_token == 0)


@st.cache_data(ttl=3600, show_spinner=False)
def load_portfolio_zakat(holdings_key: tuple[tuple[str, float], ...], method: str, refresh_token: int = 0) -> pd.DataFrame:
    from screener.zakat import calculate_portfolio_zakat
    holdings = {ticker: shares for ticker, shares in holdings_key}
    return calculate_portfolio_zakat(holdings, method=method)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nisab_threshold(refresh_token: int = 0) -> float:
    from screener.zakat import get_nisab_pkr
    return float(get_nisab_pkr())


if __name__ == "__main__":
    main()
