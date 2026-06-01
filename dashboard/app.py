"""PSX Halal Screener dashboard.

Search-first Streamlit experience with a background screening pipeline,
always-visible written verdict reasons, and an inline AAOIFI zakat calculator.
"""

from __future__ import annotations

import csv
import html
import difflib
import threading
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data.error_utils import log_exception, log_message
from data.sector_classifier import research_unknown_stock
from data.verdict_store import get_all_verdicts, get_financials, get_screening_stats, get_verdict, search_verdicts
from screener.live_screener import run_full_screening, screen_single_company
from screener.reason_generator import generate_full_reason
from screener import zakat as zakat_utils
from screener.zakat import calculate_nisab_pkr, calculate_portfolio_zakat, calculate_stock_zakat, export_zakat_pdf, generate_zakat_report


BG = "#f5f0e8"
GREEN = "#1a6b3c"
GOLD = "#c9a84c"
TEXT = "#12301f"
HALAL = "#1a6b3c"
DOUBTFUL = "#b8860b"
HARAM = "#8b1a1a"
MUTED = "#6f7b74"

PIPELINE_PHASES = ["symbols", "financials", "classify", "screening", "complete"]
PIPELINE_LABELS = {
    "symbols": "PSX Symbols",
    "financials": "Financials",
    "classify": "Sector AI",
    "screening": "AAOIFI Screen",
    "complete": "Results Ready",
}

TASK_LOCK = threading.Lock()
TASK_STATES: dict[str, dict[str, Any]] = {}


def main() -> None:
    st.set_page_config(
        page_title="PSX Halal Screener",
        page_icon="☪️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _initialize_session_state()
    _inject_css()

    stats = get_screening_stats()
    frame = get_all_verdicts()
    if not frame.empty:
        st.session_state["screened_df"] = frame
    st.session_state["db_stats"] = stats

    if _should_bootstrap_refresh(stats) and not st.session_state.get("bootstrap_refresh_started"):
        st.session_state["bootstrap_refresh_started"] = True
        _start_pipeline_background(force=True, auto=True)

    _sync_task_results()

    _render_header(stats)
    _render_hero_strip(stats)
    _render_search_band()
    _render_action_strip(stats)
    _render_kpi_strip(frame if not frame.empty else st.session_state.get("screened_df", pd.DataFrame()))

    left_col, right_col = st.columns([1, 3])
    with left_col:
        _render_pipeline_panel(stats)
    with right_col:
        _render_results_panel()

    # route to different pages: list (default), detail, zakat
    if st.session_state.get("page") == "zakat":
        _render_zakat_page()

    # Only show the Zakat panel on the main list page unless the user specifically opened the calculator
    if st.session_state.get("page", "list") == "list" or st.session_state.get("show_zakat_focus"):
        _render_zakat_panel()
    _render_footer()


def _render_hero_strip(stats: dict) -> None:
    total = int(stats.get("total", 0) or 0)
    coverage = float(stats.get("coverage_pct", 0.0) or 0.0)
    last_run = _format_last_run(stats.get("last_run"))

    st.markdown(
        f"""
        <div class="hero-band">
            <div class="hero-grid">
                <div class="hero-card">
                    <div class="hero-kicker">Workspace overview</div>
                    <div class="hero-title">Search PSX stocks, inspect the reasoning, and calculate Zakat in one focused workflow.</div>
                    <div class="hero-copy">The layout is intentionally split into clear, high-signal blocks: search and verdicts, match explanations, pipeline status, and a dedicated Zakat calculator.</div>
                    <div>
                        <span class="feature-chip">Typo-tolerant ticker search</span>
                        <span class="feature-chip">Nearest match reasoning</span>
                        <span class="feature-chip">Single-stock and portfolio Zakat</span>
                    </div>
                </div>
                <div class="hero-card">
                    <div class="hero-kicker">Database</div>
                    <div class="hero-metric">{total}</div>
                    <div class="hero-label">Screened companies</div>
                    <div class="hero-copy" style="margin-top:10px;">The cached verdict store is loaded first so the page stays fast and searchable.</div>
                </div>
                <div class="hero-card">
                    <div class="hero-kicker">Freshness</div>
                    <div class="hero-metric">{coverage:.0f}%</div>
                    <div class="hero-label">Local coverage</div>
                    <div class="hero-copy" style="margin-top:10px;">Last updated {html.escape(last_run)}. Refresh the pipeline only when you need new data.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session_state() -> None:
    defaults = {
        "search_box": "",
        "search_query": "",
        "search_match_context": None,
        "selected_ticker": "",
        "screened_df": pd.DataFrame(),
        "status_filter": ["HALAL", "DOUBTFUL", "HARAM"],
        "sector_filter": [],
        "min_score": 0,
        "db_stats": {},
        "bootstrap_refresh_started": False,
        "zakat_holdings_text": "ENGRO, 100, 0, 0\nMEBL, 50, 0, 0",
        "zakat_method": "net_assets",
        "zakat_other_assets": 0.0,
        "zakat_deductible_debts": 0.0,
        "zakat_benchmark": "silver",
        "zakat_result": None,
        "zakat_stock_result": None,
        "zakat_download_html": None,
        "show_zakat_focus": False,
        "research_result": None,
        "research_ticker": "",
        "page": "list",
        "detail_ticker": "",
        "detail_reason_context": None,
        "zakat_single_ticker": "",
        "zakat_single_shares": 100,
        "zakat_single_purchase_price": 0.0,
        "zakat_single_current_price": 0.0,
        "zakat_single_company_name": "",
        "zakat_single_sector": "Unknown",
        "zakat_single_method": "net_assets",
        "zakat_single_benchmark": "silver",
        "zakat_parse_preview": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {BG};
            color: {TEXT};
            font-family: Georgia, serif;
        }}
        [data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ padding: 0 !important; max-width: 100% !important; }}

        .app-header {{
            background: linear-gradient(135deg, #0d3d20, #1a6b3c);
            padding: 18px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid {GOLD};
            position: sticky;
            top: 0;
            z-index: 999;
        }}
        .app-header h1 {{ color: white; margin: 0; font-size: 24px; letter-spacing: 1px; }}
        .app-header .subtitle {{ color: {GOLD}; font-size: 12px; margin-top: 2px; }}
        .header-right {{ display: flex; gap: 12px; align-items: center; color: rgba(255,255,255,0.7); font-size: 12px; flex-wrap: wrap; justify-content: flex-end; }}

        .hero-band {{ padding: 16px 32px 0; }}
        .hero-grid {{ display: grid; grid-template-columns: 1.35fr 0.8fr 0.8fr; gap: 12px; }}
        .hero-card {{ background: rgba(255,255,255,0.92); border: 1px solid #e2d8c8; border-radius: 18px; padding: 18px 20px; box-shadow: 0 12px 26px rgba(17, 34, 22, 0.06); backdrop-filter: blur(6px); }}
        .search-wrapper {{
            background: white;
            padding: 16px 32px;
            border-bottom: 1px solid #e0d8cc;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .search-hint {{ margin-top: 8px; color: {MUTED}; font-size: 12px; }}
        .stTextInput > div > div > input {{
            font-size: 17px !important;
            height: 50px !important;
            border: 2px solid {GREEN} !important;
            border-radius: 10px !important;
            padding: 0 20px 0 18px !important;
            background: #fafaf7 !important;
            color: #1a1a1a !important;
        }}
        .stTextInput > div > div > input::placeholder {{ color: #999 !important; font-style: italic; }}
        .stTextInput > div > div > input:focus {{ background: white !important; box-shadow: 0 0 0 3px rgba(26,107,60,0.15) !important; border-color: {GREEN} !important; }}

        .action-strip {{ padding: 10px 32px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
        .action-note {{ color: {MUTED}; font-size: 12px; margin-left: 4px; }}

        .kpi-row {{ display: flex; gap: 12px; padding: 16px 32px; background: {BG}; }}
        .kpi-card {{ flex: 1; background: white; border-radius: 12px; padding: 14px 18px; border: 1px solid #e0d8cc; box-shadow: 0 6px 16px rgba(0,0,0,0.04); }}
        .kpi-card.active {{ border-color: {GREEN}; background: #f0f8f4; }}
        .kpi-number {{ font-size: 30px; font-weight: bold; margin: 4px 0; }}
        .kpi-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        .kpi-sub {{ font-size: 11px; color: #999; margin-top: 2px; }}
        .kpi-halal .kpi-number {{ color: {HALAL}; }}
        .kpi-doubtful .kpi-number {{ color: {DOUBTFUL}; }}
        .kpi-haram .kpi-number {{ color: {HARAM}; }}
        .kpi-total .kpi-number {{ color: #1a3a6b; }}

        .panel-card {{ background: white; border: 1px solid #e0d8cc; border-radius: 16px; padding: 16px; box-shadow: 0 10px 24px rgba(0,0,0,0.05); margin-bottom: 14px; }}
        .panel-title {{ font-size: 16px; font-weight: 900; margin: 0 0 10px; color: #1a3a1a; }}
        .panel-muted {{ color: {MUTED}; font-size: 12px; }}

        .pipeline-step {{ background: white; border: 1px solid #e0d8cc; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; }}
        .pipeline-step.done {{ border-left: 4px solid {GREEN}; }}
        .pipeline-step.running {{ border-left: 4px solid {GOLD}; background: #fffdf0; }}
        .pipeline-step.waiting {{ border-left: 4px solid #ddd; }}
        .pipeline-step.error {{ border-left: 4px solid {HARAM}; background: #fff1f1; }}
        .step-title {{ font-weight: bold; color: #333; }}
        .step-detail {{ color: #777; font-size: 11px; margin-top: 2px; }}
        .step-progress {{ width: 100%; height: 9px; background: #f0ebe0; border-radius: 999px; overflow: hidden; margin-top: 6px; }}
        .step-progress > span {{ display: block; height: 100%; background: linear-gradient(90deg, {GREEN}, #49a96a); border-radius: 999px; }}

        .company-table-wrapper {{ background: white; border-radius: 16px; border: 1px solid #e0d8cc; overflow: hidden; box-shadow: 0 10px 24px rgba(0,0,0,0.05); }}
        .table-caption {{ margin: 0.2rem 0 0.7rem; color: {MUTED}; font-size: 0.92rem; }}

        .verdict-header {{ border-radius: 16px; padding: 18px 20px; display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; color: white; }}
        .verdict-halal {{ background: linear-gradient(135deg, #0d3d20, #1a6b3c); }}
        .verdict-haram {{ background: linear-gradient(135deg, #3d0d0d, #8b1a1a); }}
        .verdict-doubtful {{ background: linear-gradient(135deg, #3d2e0d, #b8860b); }}
        .verdict-ticker {{ font-size: 28px; font-weight: bold; line-height: 1.1; }}
        .verdict-name {{ font-size: 14px; opacity: 0.9; margin-top: 2px; }}
        .verdict-badge {{ font-size: 18px; font-weight: bold; padding: 8px 20px; border-radius: 20px; background: rgba(255,255,255,0.16); white-space: nowrap; }}
        .reason-box {{ background: #fafaf7; border: 1px solid #e0d8cc; border-radius: 12px; padding: 14px 16px; line-height: 1.7; font-size: 13px; color: #333; white-space: pre-wrap; max-height: 340px; overflow-y: auto; }}
        .match-card {{ background: linear-gradient(135deg, #fffdf5, #f7fbf8); border: 1px solid #e6dbc0; border-radius: 14px; padding: 14px 16px; margin-top: 12px; }}
        .match-title {{ font-weight: 900; color: #21402a; margin-bottom: 4px; }}
        .match-meta {{ color: #5d6c63; font-size: 12px; line-height: 1.55; }}
        .match-score {{ display: inline-block; margin-top: 8px; background: rgba(26,107,60,0.08); color: {GREEN}; padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; }}

        .mini-screen {{ display: flex; align-items: flex-start; gap: 10px; padding: 11px 12px; border-bottom: 1px solid #f0ebe0; }}
        .mini-screen:last-child {{ border-bottom: none; }}
        .mini-main {{ flex: 1; }}
        .mini-title {{ font-weight: bold; font-size: 14px; color: #333; }}
        .mini-reason {{ font-size: 12.5px; color: #555; margin-top: 3px; line-height: 1.5; }}

        .log-box {{ background: #0d1117; border-radius: 12px; padding: 10px; font-family: Consolas, 'Courier New', monospace; font-size: 11px; min-height: 170px; max-height: 220px; overflow-y: auto; border: 1px solid #30363d; color: #d8ffe7; }}
        .zakat-card {{ background: linear-gradient(135deg, #0d3d20, #1a6b3c); color: white; border-radius: 16px; padding: 18px; box-shadow: 0 12px 28px rgba(9,26,15,0.18); }}
        .zakat-amount {{ font-size: 32px; font-weight: 900; color: {GOLD}; }}
        .zakat-sub {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}
        .zakat-section {{ background: white; border: 1px solid #e0d8cc; border-radius: 16px; padding: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.05); margin-top: 18px; }}
        .zakat-summary {{ background: #f9fcfa; border: 1px solid #d8eadf; border-radius: 14px; padding: 14px; }}
        .zakat-quick {{ background: #fbfcfb; border: 1px solid #dfe8e1; border-radius: 14px; padding: 14px; margin-bottom: 14px; }}
        .zakat-result-strip {{ background: #f8fbf8; border: 1px solid #d8eadf; border-radius: 14px; padding: 14px; margin-top: 12px; }}
        .feature-chip {{ display: inline-flex; align-items: center; gap: 7px; padding: 7px 10px; border-radius: 999px; background: #f4f7f5; color: #21402a; font-size: 11px; font-weight: 700; margin-right: 8px; margin-top: 8px; border: 1px solid #dde7df; }}
        .disclaimer {{ background: #1a1a1a; color: rgba(255,255,255,0.5); text-align: center; padding: 16px; font-size: 11px; margin-top: 26px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(stats: dict) -> None:
    last_run = _format_last_run(stats.get("last_run"))
    total = int(stats.get("total", 0) or 0)
    coverage = float(stats.get("coverage_pct", 0.0) or 0.0)
    halal = int(stats.get("halal_count", 0) or 0)
    doubtful = int(stats.get("doubtful_count", 0) or 0)
    haram = int(stats.get("haram_count", 0) or 0)

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>☪️ PSX Halal Screener</h1>
                <div class="subtitle">AAOIFI-standard screening for Pakistan Stock Exchange investors</div>
            </div>
            <div class="header-right">
                <span>Total screened: {total}</span>
                <span>Halal: {halal}</span>
                <span>Doubtful: {doubtful}</span>
                <span>Haram: {haram}</span>
                <span>Coverage: {coverage:.0f}%</span>
                <span>Last updated: {html.escape(last_run)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_search_band() -> None:
    st.markdown("<div class='search-wrapper'>", unsafe_allow_html=True)
    st.text_input(
        "Search companies",
        placeholder="🔍 Search a single ticker, company name, or sector — typos will open the nearest stock",
        key="search_box",
        label_visibility="collapsed",
    )
    st.session_state["search_query"] = str(st.session_state.get("search_box", "")).strip()
    st.markdown("<div class='search-hint'>Single-ticker lookup is supported. If you misspell a symbol, the closest listing is shown with a reason and score.</div>", unsafe_allow_html=True)

    # Autosuggest / quick picks to mimic an Investify-like compact search
    query = st.session_state.get("search_query", "")
    if query:
        df = st.session_state.get("screened_df", pd.DataFrame())
        if df is not None and not df.empty:
            suggestions = _suggest_matches(df, query)
            if suggestions:
                cols = st.columns(min(4, len(suggestions)))
                for idx, suggestion in enumerate(suggestions):
                    with cols[idx]:
                        if st.button(suggestion, key=f"autosuggest_{suggestion}", use_container_width=True):
                            st.session_state["search_box"] = suggestion
                            st.session_state["search_query"] = suggestion
                            st.session_state["selected_ticker"] = suggestion.upper()
                            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_action_strip(stats: dict) -> None:
    st.markdown("<div class='action-strip'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 3])
    with c1:
        if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_data_button"):
            _start_pipeline_background(force=True)
            st.rerun()
    with c2:
        if st.button("🧮 Zakat Calculator", use_container_width=True, key="focus_zakat_button"):
            st.session_state["page"] = "zakat"
            st.rerun()
    with c3:
        if st.button("🧹 Clear Search", use_container_width=True, key="clear_search_button"):
            st.session_state["search_box"] = ""
            st.session_state["search_query"] = ""
            st.session_state["selected_ticker"] = ""
            st.rerun()
    with c4:
        if st.button("▶ Full Screening", use_container_width=True, key="full_screening_button"):
            _start_pipeline_background(force=True)
            st.rerun()
    with c5:
        st.markdown(
            "<div class='action-note'>Background screening never blocks the UI. If the database is empty or stale, a new run starts automatically.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_kpi_strip(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.markdown(
            "<div class='kpi-row'><div class='kpi-card kpi-total'><div class='kpi-label'>Database</div><div class='kpi-number'>0</div><div class='kpi-sub'>No screened companies loaded yet</div></div></div>",
            unsafe_allow_html=True,
        )
        return

    total = len(frame)
    counts = _status_counts(frame)
    coverage = float(get_screening_stats().get("coverage_pct", 0.0) or 0.0)
    cols = st.columns(5)
    cards = [
        (cols[0], "HALAL", counts["HALAL"], "kpi-halal"),
        (cols[1], "DOUBTFUL", counts["DOUBTFUL"], "kpi-doubtful"),
        (cols[2], "HARAM", counts["HARAM"], "kpi-haram"),
        (cols[3], "TOTAL", total, "kpi-total"),
    ]
    for col, label, value, css in cards:
        with col:
            pct = (value / total * 100) if total else 0.0
            if st.button(f"{label}\n{value}\n{pct:.1f}%", key=f"kpi_{label.lower()}", use_container_width=True):
                if label == "TOTAL":
                    st.session_state["status_filter"] = ["HALAL", "DOUBTFUL", "HARAM"]
                else:
                    st.session_state["status_filter"] = [label]
                st.rerun()

    with cols[4]:
        st.markdown(
            f"""
            <div class="kpi-card active {css}">
                <div class="kpi-label">Coverage</div>
                <div class="kpi-number">{coverage:.0f}%</div>
                <div class="kpi-sub">Local DB coverage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_pipeline_panel(stats: dict) -> None:
    # Collapse pipeline panel into an expander to reduce on-page noise
    task = TASK_STATES["pipeline"]
    with st.expander("⚙️ Pipeline Status (click to expand)", expanded=False):
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Pipeline</div>", unsafe_allow_html=True)

        if task["status"] == "running":
            st.info("Background screening is running. The page stays interactive while data updates in the background.")
        elif task["status"] == "error":
            st.error(f"Background screening failed: {task.get('error', 'Unknown error')}")

        current_phase = str(task.get("phase", "symbols"))
        for phase in PIPELINE_PHASES:
            state = _phase_state(phase, current_phase, task["status"])
            detail = task.get("detail", "") if phase == current_phase else ""
            progress = int(task.get("progress", 0) or 0)
            if state == "done":
                progress = 100
            elif state == "running":
                progress = max(progress, 15)

            st.markdown(
                f"""
                <div class="pipeline-step {state}">
                    <div class="step-title">{PIPELINE_LABELS[phase]} {_phase_icon(state)}</div>
                    <div class="step-progress"><span style="width:{max(0, min(progress, 100))}%"></span></div>
                    <div class="step-detail">{html.escape(str(detail or _phase_detail(phase, task, stats)))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📋 Live Log")
        log_lines = task.get("logs", [])[-10:] or ["[No logs yet]"]
        st.markdown(f"<div class='log-box'><pre style='margin:0'>{'<br>'.join(html.escape(str(line)) for line in log_lines)}</pre></div>", unsafe_allow_html=True)

        st.markdown("#### 🗄️ Database")
        st.markdown(
            f"""
            | | |
            |---|---:|
            | Total | {int(stats.get('total', 0) or 0)} |
            | Halal | {int(stats.get('halal_count', 0) or 0)} |
            | Doubtful | {int(stats.get('doubtful_count', 0) or 0)} |
            | Haram | {int(stats.get('haram_count', 0) or 0)} |
            | Coverage | {float(stats.get('coverage_pct', 0.0) or 0.0):.0f}% |
            """
        )
        st.button("Run Full Pipeline", use_container_width=True, key="run_pipeline_button", on_click=_start_pipeline_background)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_results_panel() -> None:
    # If user navigated to a detail page, render that view alone to reduce on-screen density
    if st.session_state.get("page") == "detail":
        detail_ticker = st.session_state.get("detail_ticker", "")
        if detail_ticker:
            _render_detail_page(detail_ticker)
            return

    frame = st.session_state.get("screened_df", pd.DataFrame())
    query = st.session_state.get("search_query", "").strip()

    if frame is None or frame.empty:
        _render_onboarding_card(query)
        if query:
            _render_no_results(query)
        return

    filtered = _apply_filters(frame.copy(), query)
    active_ticker = _resolve_active_ticker(filtered)
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>📊 Screened Universe</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-muted'>Search is live. Every verdict reason appears below the table.</div>", unsafe_allow_html=True)

    if active_ticker:
        _render_company_detail(active_ticker)

    if not query:
        status_opts = ["HALAL", "DOUBTFUL", "HARAM"]
        sector_opts = sorted([str(value) for value in frame.get("sector_classified", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()])
        f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.2, 0.8])
        with f1:
            st.session_state["status_filter"] = st.multiselect("Status", status_opts, default=st.session_state.get("status_filter", status_opts), key="status_filter_widget")
        with f2:
            st.session_state["sector_filter"] = st.multiselect("Sector", sector_opts, default=st.session_state.get("sector_filter", []), key="sector_filter_widget")
        with f3:
            st.session_state["min_score"] = st.slider("Min Shariah Score", 0, 100, int(st.session_state.get("min_score", 0) or 0), key="score_filter_widget")
        with f4:
            if st.button("Reset", use_container_width=True, key="reset_filters_button"):
                st.session_state["status_filter"] = ["HALAL", "DOUBTFUL", "HARAM"]
                st.session_state["sector_filter"] = []
                st.session_state["min_score"] = 0
                st.rerun()
        filtered = _apply_filters(frame.copy(), query)

    st.markdown(f"<div class='table-caption'>{len(filtered)} company(s) matched.</div>", unsafe_allow_html=True)
    selected = _render_company_table(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

    if selected:
        st.session_state["selected_ticker"] = selected

    ticker = st.session_state.get("selected_ticker", "")
    if ticker and ticker != active_ticker:
        _render_company_detail(ticker)

    if query and len(filtered) == 0:
        nearest = _find_nearest_match(frame, query)
        if nearest:
            _render_nearest_match(query, nearest)
        else:
            _render_no_results(query)


def _render_company_table(data: pd.DataFrame) -> str:
    if data is None or data.empty:
        st.info("No screened companies match the current filters.")
        return ""

    display = data.copy()
    for column in ["shariah_score", "debt_ratio", "interest_ratio", "securities_ratio", "receivables_ratio"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce")

    if "overall_status" in display.columns:
        display["status_badge"] = display["overall_status"].astype(str).str.upper().map(_status_badge_text)
    else:
        display["status_badge"] = ""

    display["debt_ratio_pct"] = pd.to_numeric(display.get("debt_ratio"), errors="coerce").mul(100).round(2)
    display["interest_ratio_pct"] = pd.to_numeric(display.get("interest_ratio"), errors="coerce").mul(100).round(2)
    display["score"] = pd.to_numeric(display.get("shariah_score"), errors="coerce").fillna(0).round(0).astype(int)

    columns = ["status_badge", "ticker", "company_name", "sector_classified", "score", "debt_ratio_pct", "interest_ratio_pct", "overall_status"]
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    table = display[columns].rename(
        columns={
            "status_badge": "Status",
            "ticker": "Ticker",
            "company_name": "Company",
            "sector_classified": "Sector",
            "score": "Shariah Score",
            "debt_ratio_pct": "Debt %",
            "interest_ratio_pct": "Interest %",
            "overall_status": "Verdict",
        }
    )

    try:
        event = st.dataframe(
            table,
            use_container_width=True,
            height=260,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "Shariah Score": st.column_config.ProgressColumn("Shariah Score", min_value=0, max_value=100, format="%d/100"),
                "Debt %": st.column_config.NumberColumn("Debt %", format="%.1f%%"),
                "Interest %": st.column_config.NumberColumn("Interest %", format="%.2f%%"),
            },
            key="results_table",
        )
        if event.selection.rows:
            row = table.iloc[event.selection.rows[0]]
            return str(row.get("Ticker", "")).strip().upper()
    except Exception:
        st.dataframe(table, use_container_width=True, height=360, hide_index=True)

    return ""


def _render_company_detail(ticker: str) -> None:
    verdict = get_verdict(ticker)
    financials = get_financials(ticker)

    if not verdict:
        _render_research_prompt(ticker)
        return

    status = str(verdict.get("overall_status", "UNKNOWN")).upper()
    score = int(pd.to_numeric(pd.Series([verdict.get("shariah_score", 0)]), errors="coerce").fillna(0).iloc[0])
    company_name = str(verdict.get("company_name", ticker))
    header_class = {"HALAL": "verdict-halal", "HARAM": "verdict-haram", "DOUBTFUL": "verdict-doubtful"}.get(status, "verdict-doubtful")
    badge = {"HALAL": "✅ HALAL", "HARAM": "❌ HARAM", "DOUBTFUL": "⚠ DOUBTFUL"}.get(status, status)

    st.markdown(
        f"""
        <div class="panel-card" style="padding:0; overflow:hidden; margin-bottom:14px;">
        <div class="verdict-header {header_class}">
            <div>
                <div class="verdict-ticker">{html.escape(ticker)}</div>
                <div class="verdict-name">{html.escape(company_name)}</div>
                <div style="font-size:12px; opacity:0.85; margin-top:4px">{html.escape(str(verdict.get('sector_classified', verdict.get('sector', 'Unknown'))))} · Screened: {html.escape(_format_last_run(verdict.get('screened_at')))}</div>
            </div>
            <div style="text-align:right">
                <div class="verdict-badge">{badge}</div>
                <div style="margin-top:8px; font-size:13px; opacity:0.9">Score: {score}/100</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_detail_reasoning(ticker)

    col_a, col_b = st.columns([1.1, 1.9])
    with col_a:
        st.markdown("#### AAOIFI Screens")
        screen_rows = [
            ("Business Activity", verdict.get("business_screen", "PASS"), verdict.get("business_reason") or verdict.get("haram_reason") or "Primary business appears compliant."),
            (f"Debt Ratio ({_pct(verdict.get('debt_ratio'))})", verdict.get("debt_screen", "PASS"), verdict.get("debt_reason") or "Debt ratio screen reviewed."),
            (f"Interest Income ({_pct(verdict.get('interest_ratio'), 2)})", verdict.get("interest_screen", "PASS"), verdict.get("interest_reason") or "Interest income screen reviewed."),
            (f"Securities ({_pct(verdict.get('securities_ratio'))})", verdict.get("securities_screen", "PASS"), verdict.get("securities_reason") or "Securities screen reviewed."),
            (f"Receivables ({_pct(verdict.get('receivables_ratio'))})", verdict.get("receivables_screen", "PASS"), verdict.get("receivables_reason") or "Receivables screen reviewed."),
        ]
        for name, result, reason in screen_rows:
            result_text = str(result).upper()
            icon = "✅" if result_text == "PASS" else "❌" if result_text == "FAIL" else "⚠"
            css = "screen-pass" if result_text == "PASS" else "screen-fail" if result_text == "FAIL" else "screen-neutral"
            st.markdown(
                f"""
                <div class="{css}">
                    <div><b>{icon} {html.escape(name)}</b></div>
                    <div style="font-size:11.5px; color:#555; margin-top:3px; line-height:1.45">{html.escape(str(reason))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Quick Metrics")
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Debt %", _pct(verdict.get("debt_ratio")))
            st.metric("Receivables %", _pct(verdict.get("receivables_ratio")))
        with metric_cols[1]:
            st.metric("Interest %", _pct(verdict.get("interest_ratio"), 2))
            st.metric("Securities %", _pct(verdict.get("securities_ratio")))

    with col_b:
        st.markdown("#### Why This Verdict?")
        reason_text = generate_full_reason(ticker, verdict, financials or {})
        st.markdown(f"<div class='reason-box'>{html.escape(reason_text)}</div>", unsafe_allow_html=True)

        if verdict.get("ai_explanation"):
            st.markdown("#### AI / Research Note")
            st.info(str(verdict.get("ai_explanation")))

        if status == "HALAL" and verdict.get("purification_ratio") is not None:
            st.success(f"Purification rate: {_pct(verdict.get('purification_ratio'), 3)} of dividends should be donated to charity.")
        elif status == "HARAM":
            st.error("This stock is not permissible under the current AAOIFI screening result.")
        else:
            st.warning("This stock is borderline or has incomplete data. Consult a qualified scholar before investing.")

        button_col_1, button_col_2, button_col_3 = st.columns(3)
        with button_col_1:
            if st.button("Load into Zakat Calculator", key=f"load_zakat_{ticker}", use_container_width=True):
                _load_ticker_into_zakat(ticker, verdict, financials)
                st.rerun()
        with button_col_2:
            if st.button("Re-screen Live", key=f"rescreen_{ticker}", use_container_width=True):
                _start_research_background(ticker)
                st.rerun()
        with button_col_3:
            if st.button("Copy Reason to Report", key=f"copy_reason_{ticker}", use_container_width=True):
                st.session_state["download_reason_text"] = reason_text
        # allow opening a focused detail page to avoid crowding the main list
        if st.button("Open Detail Page", key=f"open_detail_{ticker}"):
            st.session_state["page"] = "detail"
            st.session_state["detail_ticker"] = ticker
            st.rerun()


def _resolve_active_ticker(filtered: pd.DataFrame) -> str:
    selected = str(st.session_state.get("selected_ticker", "")).strip().upper()
    if selected:
        return selected

    if filtered is None or filtered.empty:
        return ""

    query = str(st.session_state.get("search_query", "")).strip()
    if query and len(filtered) == 1:
        return str(filtered.iloc[0].get("ticker", "")).strip().upper()

    return ""


def _render_onboarding_card(query: str = "") -> None:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Start here</div>
            <div class="panel-muted">Search is the first control on the page. Once data is loaded, every verdict includes a written reason and the Zakat calculator stays on the same page.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if query:
        _render_no_results(query)


def _render_no_results(query: str) -> None:
    df = st.session_state.get("screened_df", pd.DataFrame())
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">🔍 No results for "{html.escape(query)}"</div>
            <div class="panel-muted">This ticker or company is not in the cached database yet.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not df.empty:
        suggestions = _suggest_matches(df, query)
        if suggestions:
            st.markdown("**Did you mean:**")
            cols = st.columns(min(3, len(suggestions)))
            for idx, suggestion in enumerate(suggestions[:3]):
                with cols[idx]:
                    if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                        st.session_state["search_box"] = suggestion
                        st.session_state["search_query"] = suggestion
                        st.session_state["selected_ticker"] = suggestion.upper()
                        st.rerun()

    st.markdown("**Live research is available without leaving the page.**")
    a, b = st.columns([1, 3])
    with a:
        if st.button(f"🔎 Research {query.upper()} now", key=f"research_{query}", use_container_width=True, type="primary"):
            _start_research_background(query)
            st.rerun()
    with b:
        st.caption("Background research uses the same screen flow and returns as soon as the worker finishes.")

    research_state = TASK_STATES["research"]
    if research_state["status"] == "running" and str(research_state.get("ticker", "")).upper() == query.upper():
        st.info("Live research is running in the background.")
    elif research_state["status"] == "complete" and str(research_state.get("ticker", "")).upper() == query.upper():
        result = research_state.get("result") or {}
        _render_research_result(result)


def _render_research_prompt(ticker: str) -> None:
    st.warning(f"No cached verdict found for {ticker}.")
    if st.button(f"Research {ticker} live", key=f"research_live_{ticker}"):
        _start_research_background(ticker)
        st.rerun()


def _render_research_result(result: dict) -> None:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Live Research Result</div>", unsafe_allow_html=True)
    st.json(result)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_detail_page(ticker: str) -> None:
    st.markdown("<div style='padding:18px 32px;'>", unsafe_allow_html=True)
    back_col, spacer = st.columns([1, 4])
    with back_col:
        if st.button("← Back to List", key="back_to_list_button"):
            st.session_state["page"] = "list"
            st.session_state["detail_ticker"] = ""
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Render only the selected company detail to keep the page focused and uncluttered
    _render_company_detail(ticker)


def _render_detail_reasoning(ticker: str) -> None:
    context = st.session_state.get("detail_reason_context") or st.session_state.get("search_match_context")
    if not isinstance(context, dict):
        return
    if str(context.get("ticker", "")).strip().upper() != str(ticker).strip().upper():
        return

    st.markdown(
        f"""
        <div class="match-card">
            <div class="match-title">Why this stock was suggested</div>
            <div class="match-meta">Query: <b>{html.escape(str(context.get('query', '')))}</b><br>
            Metric: {html.escape(str(context.get('metric', 'similarity')))}<br>
            Reason: {html.escape(str(context.get('reason', 'Closest overall match')))}</div>
            <div class="match-score">Match score: {float(context.get('score', 0.0)):.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_nearest_match(query: str, match: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">Closest stock match</div>
            <div class="match-card">
                <div class="match-title">{html.escape(str(match.get('ticker', '')))} · {html.escape(str(match.get('company_name', '')))}</div>
                <div class="match-meta">{html.escape(str(match.get('sector_classified', 'Unknown')))}<br>
                Query: {html.escape(query)}<br>
                Reason: {html.escape(str(match.get('reason', 'Closest overall spelling match')))}</div>
                <div class="match-score">Score {float(match.get('score', 0.0)):.3f} via {html.escape(str(match.get('metric', 'similarity')))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"Open {match.get('ticker', '')} detail page", key=f"open_match_{match.get('ticker', '')}", use_container_width=True, type="primary"):
        st.session_state["selected_ticker"] = str(match.get("ticker", "")).strip().upper()
        st.session_state["page"] = "detail"
        st.session_state["detail_ticker"] = str(match.get("ticker", "")).strip().upper()
        st.session_state["detail_reason_context"] = {
            "query": query,
            "ticker": str(match.get("ticker", "")).strip().upper(),
            "score": float(match.get("score", 0.0) or 0.0),
            "metric": str(match.get("metric", "similarity")),
            "reason": str(match.get("reason", "Closest overall spelling match")),
        }
        st.rerun()


def _render_zakat_panel() -> None:
    st.markdown("<div class='zakat-section'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🧮 Zakat Calculator</div>", unsafe_allow_html=True)

    nisab_silver = calculate_nisab_pkr("silver")
    nisab_gold = calculate_nisab_pkr("gold")
    nisab_card = st.columns(2)
    with nisab_card[0]:
        st.markdown(
            f"""
            <div class="zakat-card">
                <div style="font-size:12px; opacity:0.8; text-transform:uppercase; letter-spacing:0.08em">Silver Nisab</div>
                <div class="zakat-amount">{_fmt_pkr(nisab_silver['nisab_pkr'])}</div>
                <div class="zakat-sub">{html.escape(str(nisab_silver.get('note', '')))}<br>Source: {html.escape(str(nisab_silver.get('source', 'unknown')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nisab_card[1]:
        st.markdown(
            f"""
            <div class="zakat-card">
                <div style="font-size:12px; opacity:0.8; text-transform:uppercase; letter-spacing:0.08em">Gold Nisab</div>
                <div class="zakat-amount">{_fmt_pkr(nisab_gold['nisab_pkr'])}</div>
                <div class="zakat-sub">{html.escape(str(nisab_gold.get('note', '')))}<br>Source: {html.escape(str(nisab_gold.get('source', 'unknown')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.get("show_zakat_focus"):
        st.info("Zakat calculator is open on this page. No tab switching required.")

    with st.expander("Single stock calculator", expanded=bool(st.session_state.get("show_zakat_focus"))):
        st.markdown("<div class='zakat-quick'>", unsafe_allow_html=True)
        selected = st.session_state.get("selected_ticker", "")
        if selected:
            st.caption(f"Selected ticker: {selected}. Use it as the starting point for a single-stock Zakat calculation.")

        with st.form("single_zakat_form"):
            q1, q2 = st.columns([1.2, 1])
            with q1:
                st.session_state["zakat_single_ticker"] = st.text_input(
                    "Ticker",
                    value=str(st.session_state.get("zakat_single_ticker", selected or "")).strip().upper(),
                    key="zakat_single_ticker_input",
                )
            with q2:
                st.session_state["zakat_single_shares"] = st.number_input(
                    "Shares",
                    min_value=0,
                    value=int(st.session_state.get("zakat_single_shares", 100) or 100),
                    step=1,
                    key="zakat_single_shares_input",
                )

            p1, p2 = st.columns(2)
            with p1:
                st.session_state["zakat_single_purchase_price"] = st.number_input(
                    "Purchase price",
                    min_value=0.0,
                    value=float(st.session_state.get("zakat_single_purchase_price", 0.0) or 0.0),
                    step=1.0,
                    key="zakat_single_purchase_price_input",
                )
            with p2:
                st.session_state["zakat_single_current_price"] = st.number_input(
                    "Current price",
                    min_value=0.0,
                    value=float(st.session_state.get("zakat_single_current_price", 0.0) or 0.0),
                    step=1.0,
                    key="zakat_single_current_price_input",
                )

            method_col, benchmark_col = st.columns(2)
            with method_col:
                st.session_state["zakat_single_method"] = st.selectbox(
                    "Method",
                    options=["net_assets", "market_value", "dividend_income"],
                    format_func=lambda value: {"net_assets": "Net Assets (AAOIFI)", "market_value": "Market Value", "dividend_income": "Dividend / Income"}[value],
                    index=["net_assets", "market_value", "dividend_income"].index(st.session_state.get("zakat_single_method", "net_assets")),
                    key="zakat_single_method_select",
                )
            with benchmark_col:
                st.session_state["zakat_single_benchmark"] = st.radio(
                    "Nisab benchmark",
                    ["silver", "gold"],
                    horizontal=True,
                    index=0 if st.session_state.get("zakat_single_benchmark", "silver") == "silver" else 1,
                    key="zakat_single_benchmark_select",
                )

            single_submitted = st.form_submit_button("Calculate single stock Zakat", use_container_width=True)

        if single_submitted:
            ticker = str(st.session_state.get("zakat_single_ticker", "")).strip().upper().replace(".KA", "")
            holdings_detail = []
            financials = get_financials(ticker) if ticker else None
            company_name = str((financials or {}).get("company_name") or ticker).strip() or ticker
            sector_name = str((financials or {}).get("sector") or (financials or {}).get("sector_classified") or "Unknown").strip() or "Unknown"
            holding = {
                "ticker": ticker,
                "shares": int(st.session_state.get("zakat_single_shares", 0) or 0),
                "purchase_price": float(st.session_state.get("zakat_single_purchase_price", 0.0) or 0.0),
                "current_price": float(st.session_state.get("zakat_single_current_price", 0.0) or 0.0),
                "company_name": company_name,
                "sector": sector_name,
                "financial_data": financials or {},
            }
            holdings_detail.append(holding)

            stock_result = zakat_utils.calculate_stock_zakat(
                ticker=ticker,
                shares_held=int(st.session_state.get("zakat_single_shares", 0) or 0),
                purchase_price=float(st.session_state.get("zakat_single_purchase_price", 0.0) or 0.0),
                current_price=float(st.session_state.get("zakat_single_current_price", 0.0) or 0.0),
                method=st.session_state.get("zakat_single_method", "net_assets"),
                financial_data=financials or {},
                nisab_metal=st.session_state.get("zakat_single_benchmark", "silver"),
            )
            portfolio_result = calculate_portfolio_zakat(
                holdings_detail,
                method=st.session_state.get("zakat_single_method", "net_assets"),
                include_other_assets=0.0,
                debts_deductible=0.0,
                nisab_metal=st.session_state.get("zakat_single_benchmark", "silver"),
            )
            st.session_state["zakat_stock_result"] = stock_result
            st.session_state["zakat_result"] = portfolio_result
            st.session_state["zakat_download_html"] = generate_zakat_report(portfolio_result)

        stock_result = st.session_state.get("zakat_stock_result")
        if stock_result:
            st.markdown("<div class='zakat-result-strip'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Single-stock Zakat", _fmt_pkr(stock_result.get("zakat_amount", 0.0)))
            with c2:
                st.metric("Purification", _fmt_pkr(stock_result.get("purification_amount", 0.0)))
            with c3:
                st.metric("Total obligation", _fmt_pkr(stock_result.get("total_obligation", 0.0)))
            st.caption(str(stock_result.get("scholarly_note", "")))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])
    with left:
        selected = st.session_state.get("selected_ticker", "")
        if selected:
            st.caption(f"Selected ticker: {selected}. You can load it into the calculator with one click.")
            if st.button("Load selected ticker into calculator", key="load_selected_into_zakat"):
                verdict = get_verdict(selected)
                financials = get_financials(selected)
                _load_ticker_into_zakat(selected, verdict or {}, financials or {})
                st.rerun()

        with st.form("zakat_form"):
            st.text_area(
                "Holdings",
                key="zakat_holdings_text",
                height=140,
                help="One holding per line: ticker, shares, purchase_price, current_price, company_name, sector",
            )
            method_col, benchmark_col = st.columns(2)
            with method_col:
                st.session_state["zakat_method"] = st.selectbox(
                    "Method",
                    options=["net_assets", "market_value", "dividend_income"],
                    format_func=lambda value: {"net_assets": "Net Assets (AAOIFI)", "market_value": "Market Value", "dividend_income": "Dividend / Income"}[value],
                    index=["net_assets", "market_value", "dividend_income"].index(st.session_state.get("zakat_method", "net_assets")),
                    key="zakat_method_select",
                )
            with benchmark_col:
                st.session_state["zakat_benchmark"] = st.radio(
                    "Nisab benchmark",
                    ["silver", "gold"],
                    horizontal=True,
                    index=0 if st.session_state.get("zakat_benchmark", "silver") == "silver" else 1,
                    key="zakat_benchmark_select",
                )

            other_col, debt_col = st.columns(2)
            with other_col:
                st.session_state["zakat_other_assets"] = st.number_input(
                    "Other assets (PKR)",
                    min_value=0.0,
                    value=float(st.session_state.get("zakat_other_assets", 0.0) or 0.0),
                    step=1000.0,
                    key="zakat_other_assets_input",
                )
            with debt_col:
                st.session_state["zakat_deductible_debts"] = st.number_input(
                    "Deductible debts (PKR)",
                    min_value=0.0,
                    value=float(st.session_state.get("zakat_deductible_debts", 0.0) or 0.0),
                    step=1000.0,
                    key="zakat_deductible_debts_input",
                )

            submitted = st.form_submit_button("Calculate Zakat", use_container_width=True)

        if submitted:
            # Parse holdings (this will populate session_state['zakat_parse_preview'] if suggestions found)
            holdings = _parse_holdings_text(st.session_state.get("zakat_holdings_text", ""))
            preview = st.session_state.get("zakat_parse_preview")
            if preview:
                st.warning("Some holdings look like typos or unrecognised tickers. Review suggestions below.")
                st.markdown("**Parse preview:**")
                for note in preview.get("suggestions", []):
                    st.markdown(f"- {html.escape(str(note))}")
                col_apply, col_ignore = st.columns(2)
                with col_apply:
                    if st.button("Apply suggestions and compute", key="apply_suggestions_button"):
                        # apply suggested_ticker if present
                        for h in holdings:
                            if h.get("suggested_ticker"):
                                h["ticker"] = h["suggested_ticker"]
                        result = calculate_portfolio_zakat(
                            holdings,
                            method=st.session_state.get("zakat_method", "net_assets"),
                            include_other_assets=float(st.session_state.get("zakat_other_assets", 0.0) or 0.0),
                            debts_deductible=float(st.session_state.get("zakat_deductible_debts", 0.0) or 0.0),
                            nisab_metal=st.session_state.get("zakat_benchmark", "silver"),
                        )
                        st.session_state["zakat_result"] = result
                        st.session_state["zakat_download_html"] = generate_zakat_report(result)
                        st.session_state["zakat_parse_preview"] = None
                        st.experimental_rerun()
                with col_ignore:
                    if st.button("Compute anyway (use entered tickers)", key="compute_anyway_button"):
                        result = calculate_portfolio_zakat(
                            holdings,
                            method=st.session_state.get("zakat_method", "net_assets"),
                            include_other_assets=float(st.session_state.get("zakat_other_assets", 0.0) or 0.0),
                            debts_deductible=float(st.session_state.get("zakat_deductible_debts", 0.0) or 0.0),
                            nisab_metal=st.session_state.get("zakat_benchmark", "silver"),
                        )
                        st.session_state["zakat_result"] = result
                        st.session_state["zakat_download_html"] = generate_zakat_report(result)
                        st.session_state["zakat_parse_preview"] = None
                        st.experimental_rerun()
            else:
                result = calculate_portfolio_zakat(
                    holdings,
                    method=st.session_state.get("zakat_method", "net_assets"),
                    include_other_assets=float(st.session_state.get("zakat_other_assets", 0.0) or 0.0),
                    debts_deductible=float(st.session_state.get("zakat_deductible_debts", 0.0) or 0.0),
                    nisab_metal=st.session_state.get("zakat_benchmark", "silver"),
                )
                st.session_state["zakat_result"] = result
                st.session_state["zakat_download_html"] = generate_zakat_report(result)

    with right:
        st.markdown(
            """
            <div class="zakat-summary">
                <b>What this calculator does</b><br>
                It estimates Zakat using AAOIFI Standard No. 7, keeps purification separate, and produces a single-stock or portfolio summary that can be exported as HTML or PDF.
            </div>
            """,
            unsafe_allow_html=True,
        )

        stock_result = st.session_state.get("zakat_stock_result")
        if stock_result:
            st.markdown("#### Single Stock Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Market Value", _fmt_pkr(stock_result.get("market_value", 0.0)))
                st.metric("Zakatable Base", _fmt_pkr(stock_result.get("zakatable_total", 0.0)))
            with c2:
                st.metric("Nisab", _fmt_pkr(stock_result.get("nisab_threshold", 0.0)))
                st.metric("Due Above Nisab", "Yes" if stock_result.get("zakat_due") else "No")

        result = st.session_state.get("zakat_result")
        if result:
            summary = result.get("summary", {})
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Zakat", _fmt_pkr(summary.get("total_zakat", 0.0)))
                st.metric("Purification", _fmt_pkr(summary.get("total_purification", 0.0)))
            with c2:
                st.metric("Total Obligation", _fmt_pkr(summary.get("total_obligation", 0.0)))
                st.metric("Above Nisab", "Yes" if summary.get("zakat_due") else "No")

            st.markdown("#### Portfolio Summary")
            st.json(
                {
                    "method": summary.get("method_name", summary.get("method")),
                    "calculation_date": summary.get("calculation_date"),
                    "nisab_pkr": summary.get("nisab_pkr"),
                    "net_zakatable": summary.get("net_zakatable"),
                    "other_assets": summary.get("other_assets"),
                    "deductible_debts": summary.get("deductible_debts"),
                    "hawl_reminder": summary.get("hawl_reminder"),
                }
            )

            holdings_detail = pd.DataFrame(result.get("holdings_detail", []))
            if not holdings_detail.empty:
                st.markdown("#### Per-Holding Breakdown")
                display = holdings_detail[[c for c in ["ticker", "company_name", "shares_held", "market_value", "zakatable_total", "zakat_amount", "purification_amount", "method_name"] if c in holdings_detail.columns]].copy()
                # Render clickable per-holding rows so users can jump to the company detail page
                st.markdown("<div class='panel-card' style='padding:8px;'>", unsafe_allow_html=True)
                for idx, row in display.reset_index(drop=True).iterrows():
                    t = str(row.get("ticker", "")).strip().upper()
                    name = str(row.get("company_name", "")).strip()
                    mv = _fmt_pkr(row.get("market_value", 0.0))
                    zak_amt = _fmt_pkr(row.get("zakat_amount", 0.0))
                    pur_amt = _fmt_pkr(row.get("purification_amount", 0.0))
                    c0, c1, c2, c3 = st.columns([1, 4, 2, 2])
                    with c0:
                        if st.button(t or "-", key=f"zakat_open_{t}_{idx}"):
                            st.session_state["page"] = "detail"
                            st.session_state["detail_ticker"] = t
                            st.rerun()
                    with c1:
                        st.markdown(f"**{html.escape(name or t)}")
                    with c2:
                        st.markdown(f"Market: {mv}")
                    with c3:
                        st.markdown(f"Zakat: {zak_amt} · Purify: {pur_amt}")
                st.markdown("</div>", unsafe_allow_html=True)

            if result.get("non_compliant_holdings"):
                st.warning("Some holdings are non-compliant. Review the list before relying on the numbers.")
                st.json(result.get("non_compliant_holdings"))

            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Download HTML report",
                    data=st.session_state.get("zakat_download_html", ""),
                    file_name="zakat_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with export_cols[1]:
                if st.button("Create PDF report", use_container_width=True):
                    pdf_path = _export_zakat_pdf_to_temp(result)
                    if pdf_path:
                        with open(pdf_path, "rb") as handle:
                            st.download_button(
                                "Download PDF",
                                data=handle.read(),
                                file_name="zakat_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
        else:
            st.caption("Run the calculator to see the full portfolio result here.")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_zakat_page() -> None:
    st.markdown("<div style='padding:18px 32px;'>", unsafe_allow_html=True)
    back_col, spacer = st.columns([1, 4])
    with back_col:
        if st.button("← Back to List", key="back_to_list_from_zakat"):
            st.session_state["page"] = "list"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 32px 32px;'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🧮 Zakat Calculator (Dedicated Page)</div>", unsafe_allow_html=True)
    # Reuse the existing panel UI but ensure it doesn't auto-focus
    prev_show = st.session_state.get("show_zakat_focus")
    st.session_state["show_zakat_focus"] = False
    _render_zakat_panel()
    st.session_state["show_zakat_focus"] = prev_show
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_footer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            ☪️ PSX Halal Screener · Based on AAOIFI Standard No. 21 and Standard No. 7 · This tool is for guidance only and does not constitute a fatwa or religious ruling · Consult a qualified Shariah scholar before making investment decisions · Not affiliated with PSX or AAOIFI.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _start_pipeline_background(force: bool = False, auto: bool = False) -> None:
    task = TASK_STATES["pipeline"]
    if task["status"] == "running" and not force:
        return

    with TASK_LOCK:
        TASK_STATES["pipeline"] = _default_task_state("pipeline")
        TASK_STATES["pipeline"].update(
            {
                "status": "running",
                "phase": "symbols",
                "detail": "Waiting for the PSX universe fetch to begin.",
                "logs": ["Background pipeline started." if not auto else "Auto-refresh pipeline started."],
                "progress": 8,
                "auto": auto,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    thread = threading.Thread(target=_pipeline_worker, daemon=True)
    thread.start()


def _pipeline_worker() -> None:
    try:
        result = run_full_screening(use_cache=False, log_callback=_append_pipeline_log)
        with TASK_LOCK:
            task = TASK_STATES["pipeline"]
            task.update(
                {
                    "status": "complete",
                    "phase": "complete",
                    "detail": f"Finished with {len(result)} screened rows.",
                    "progress": 100,
                    "result": result,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        log_message(f"Pipeline finished with {len(result)} rows")
    except Exception as exc:
        log_exception("Background pipeline failed")
        with TASK_LOCK:
            TASK_STATES["pipeline"].update({"status": "error", "error": str(exc), "detail": str(exc), "phase": "complete"})


def _append_pipeline_log(message: str) -> None:
    with TASK_LOCK:
        task = TASK_STATES["pipeline"]
        logs = task.setdefault("logs", [])
        logs.append(str(message))
        if len(logs) > 200:
            del logs[:-200]

        text = str(message).lower()
        if "step 1" in text or "symbols" in text:
            task.update({"phase": "financials", "progress": 20, "detail": str(message)})
        elif "step 2" in text or "financial" in text:
            task.update({"phase": "classify", "progress": 40, "detail": str(message)})
        elif "step 3" in text or "classif" in text:
            task.update({"phase": "screening", "progress": 60, "detail": str(message)})
        elif "step 4" in text or "screen" in text:
            task.update({"phase": "complete", "progress": 90, "detail": str(message)})
        elif "step 5" in text or "ready" in text:
            task.update({"phase": "complete", "progress": 100, "detail": str(message)})
        else:
            task["detail"] = str(message)


def _start_research_background(ticker: str) -> None:
    normalized = str(ticker).strip().upper().replace(".KA", "")
    if not normalized:
        return

    with TASK_LOCK:
        TASK_STATES["research"] = _default_task_state("research")
        TASK_STATES["research"].update(
            {
                "status": "running",
                "ticker": normalized,
                "phase": "symbols",
                "detail": f"Researching {normalized}.",
                "logs": [f"Research started for {normalized}."],
                "progress": 10,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    thread = threading.Thread(target=_research_worker, args=(normalized,), daemon=True)
    thread.start()


def _research_worker(ticker: str) -> None:
    try:
        summary = research_unknown_stock(ticker)
        verdict = screen_single_company(ticker)
        with TASK_LOCK:
            TASK_STATES["research"].update(
                {
                    "status": "complete",
                    "phase": "complete",
                    "detail": f"Research finished for {ticker}.",
                    "progress": 100,
                    "ticker": ticker,
                    "result": {"research": summary, "screening": verdict},
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "logs": TASK_STATES["research"].get("logs", []) + [f"Research complete for {ticker}."],
                }
            )
    except Exception as exc:
        with TASK_LOCK:
            TASK_STATES["research"].update({"status": "error", "error": str(exc), "detail": str(exc), "phase": "complete"})


def _sync_task_results() -> None:
    pipeline = TASK_STATES["pipeline"]
    if pipeline["status"] == "complete" and isinstance(pipeline.get("result"), pd.DataFrame):
        st.session_state["screened_df"] = pipeline["result"].copy()
        st.session_state["db_stats"] = get_screening_stats()
        st.session_state["last_refresh"] = pipeline.get("finished_at")

    research = TASK_STATES["research"]
    if research["status"] == "complete":
        st.session_state["research_result"] = research.get("result")
        st.session_state["research_ticker"] = research.get("ticker", "")


def _apply_filters(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    filtered = frame.copy()
    if query:
        search_hits = search_verdicts(query)
        if not search_hits.empty:
            filtered = search_hits

    statuses = st.session_state.get("status_filter", ["HALAL", "DOUBTFUL", "HARAM"])
    if statuses and "overall_status" in filtered.columns:
        filtered = filtered[filtered["overall_status"].astype(str).str.upper().isin([status.upper() for status in statuses])]

    sectors = st.session_state.get("sector_filter", [])
    if sectors and "sector_classified" in filtered.columns:
        filtered = filtered[filtered["sector_classified"].astype(str).isin(sectors)]

    min_score = int(st.session_state.get("min_score", 0) or 0)
    if min_score and "shariah_score" in filtered.columns:
        score_series = pd.to_numeric(filtered["shariah_score"], errors="coerce").fillna(0)
        filtered = filtered[score_series >= min_score]

    return filtered.reset_index(drop=True)


def _phase_state(phase: str, current: str, task_status: str) -> str:
    if task_status == "error":
        return "error"
    if task_status == "complete":
        return "done"
    if phase == current:
        return "running"
    if PIPELINE_PHASES.index(phase) < PIPELINE_PHASES.index(current):
        return "done"
    return "waiting"


def _phase_icon(state: str) -> str:
    return {"done": "✅", "running": "🔄", "waiting": "⏳", "error": "❌"}.get(state, "⏳")


def _phase_detail(phase: str, task: dict, stats: dict) -> str:
    if phase == "symbols":
        return "Fetching PSX symbols from the local database or live source."
    if phase == "financials":
        return "Loading financial snapshots from the cached screening store."
    if phase == "classify":
        return "Classifying sectors and business activity."
    if phase == "screening":
        return "Running AAOIFI screening and generating verdicts."
    return f"{int(stats.get('total', 0) or 0)} companies ready for use."


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    statuses = frame.get("overall_status", pd.Series(dtype=object)).astype(str).str.upper()
    return {
        "HALAL": int((statuses == "HALAL").sum()),
        "DOUBTFUL": int((statuses == "DOUBTFUL").sum()),
        "HARAM": int((statuses == "HARAM").sum()),
    }


def _should_bootstrap_refresh(stats: dict) -> bool:
    if int(stats.get("total", 0) or 0) == 0:
        return True
    last_run = stats.get("last_run")
    if not last_run:
        return True
    parsed = _parse_datetime(str(last_run))
    if parsed is None:
        return True
    return datetime.now(timezone.utc) - parsed > timedelta(days=90)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _format_last_run(value: Any) -> str:
    parsed = _parse_datetime(str(value) if value is not None else None)
    if parsed is None:
        return "Never"
    return parsed.strftime("%B %d, %Y")


def _status_badge_text(status: str) -> str:
    return {"HALAL": "🟢 HALAL", "DOUBTFUL": "🟡 DOUBTFUL", "HARAM": "🔴 HARAM"}.get(status.upper(), status)


def _pct(value: Any, places: int = 1) -> str:
    try:
        if value is None:
            return "N/A"
        number = float(value)
        if pd.isna(number):
            return "N/A"
        return f"{number * 100:.{places}f}%"
    except Exception:
        return "N/A"


def _fmt_pkr(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        amount = float(value)
        abs_amount = abs(amount)
        if abs_amount >= 1_000_000_000:
            return f"PKR {amount / 1_000_000_000:.1f}B"
        if abs_amount >= 1_000_000:
            return f"PKR {amount / 1_000_000:.1f}M"
        return f"PKR {amount:,.0f}"
    except Exception:
        return "N/A"


def _suggest_matches(frame: pd.DataFrame, query: str) -> list[str]:
    query_norm = str(query).strip()
    names = frame.get("company_name", pd.Series(dtype=object)).astype(str).tolist()
    tickers = frame.get("ticker", pd.Series(dtype=object)).astype(str).tolist()
    candidates = list(dict.fromkeys(tickers + names))
    return difflib.get_close_matches(query_norm.upper(), candidates, n=3, cutoff=0.35)


def _find_nearest_match(frame: pd.DataFrame, query: str) -> dict[str, Any] | None:
    query_norm = _normalize_query_text(query)
    if not query_norm or frame is None or frame.empty:
        return None

    best: dict[str, Any] | None = None
    alternatives: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper().replace(".KA", "")
        company_name = str(row.get("company_name", "")).strip()
        sector = str(row.get("sector_classified", row.get("sector", "Unknown"))).strip() or "Unknown"

        ticker_score = _sequence_match_score(query_norm, ticker)
        name_score = _sequence_match_score(query_norm, _normalize_query_text(company_name))
        sector_score = _sequence_match_score(query_norm, _normalize_query_text(sector))
        best_metric = max(ticker_score, name_score, sector_score)

        reason = "Closest spelling match"
        metric = "similarity"
        if query_norm == _normalize_query_text(ticker):
            best_metric = 1.0
            reason = "Exact ticker match"
            metric = "ticker exact match"
        elif ticker and query_norm in _normalize_query_text(ticker):
            best_metric = min(1.0, max(best_metric, 0.96))
            reason = f"Ticker contains the query text '{query.strip()}'"
            metric = "ticker contains"
        elif company_name and query_norm in _normalize_query_text(company_name):
            best_metric = min(1.0, max(best_metric, 0.92))
            reason = "Company name contains the query text"
            metric = "company contains"
        elif sector and query_norm in _normalize_query_text(sector):
            best_metric = min(1.0, max(best_metric, 0.88))
            reason = "Sector text contains the query text"
            metric = "sector contains"
        else:
            if ticker_score >= name_score and ticker_score >= sector_score:
                reason = f"Closest ticker spelling match, using SequenceMatcher ratio {ticker_score:.3f}"
                metric = "ticker similarity"
            elif name_score >= sector_score:
                reason = f"Closest company-name spelling match, using SequenceMatcher ratio {name_score:.3f}"
                metric = "company similarity"
            else:
                reason = f"Closest sector spelling match, using SequenceMatcher ratio {sector_score:.3f}"
                metric = "sector similarity"

        candidate = {
            "ticker": ticker,
            "company_name": company_name,
            "sector_classified": sector,
            "score": float(best_metric),
            "metric": metric,
            "reason": reason,
        }
        alternatives.append(candidate)

    alternatives.sort(key=lambda item: (item["score"], item["ticker"]), reverse=True)
    if not alternatives:
        return None

    best = alternatives[0]
    best["alternatives"] = alternatives[1:4]
    best["query"] = query
    return best


def _sequence_match_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(difflib.SequenceMatcher(None, left, right).ratio())


def _normalize_query_text(value: str) -> str:
    return " ".join(str(value).strip().upper().replace(".KA", "").split())


def _load_ticker_into_zakat(ticker: str, verdict: dict, financials: dict) -> None:
    ticker = str(ticker).strip().upper()
    company_name = str(verdict.get("company_name") or financials.get("company_name") or ticker).strip()
    current_price = financials.get("current_price") or verdict.get("current_price") or 0
    line = f"{ticker}, 100, 0, {current_price}, {company_name}, {verdict.get('sector_classified') or verdict.get('sector') or 'Unknown'}"
    st.session_state["zakat_holdings_text"] = line
    st.session_state["selected_ticker"] = ticker
    st.session_state["show_zakat_focus"] = True
    st.session_state["zakat_single_ticker"] = ticker
    st.session_state["zakat_single_shares"] = 100
    st.session_state["zakat_single_purchase_price"] = float(current_price or 0.0)
    st.session_state["zakat_single_current_price"] = float(current_price or 0.0)
    st.session_state["zakat_single_company_name"] = company_name
    st.session_state["zakat_single_sector"] = str(verdict.get("sector_classified") or verdict.get("sector") or "Unknown")


def _parse_holdings_text(text: str) -> list[dict]:
    import difflib

    holdings: list[dict] = []
    suggestions: list[str] = []
    df = st.session_state.get("screened_df", pd.DataFrame())
    known_tickers = set()
    known_names = []
    if df is not None and not df.empty:
        known_tickers = set(df.get("ticker", pd.Series(dtype=object)).astype(str).str.upper().tolist())
        known_names = df.get("company_name", pd.Series(dtype=object)).astype(str).tolist()

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.replace(";", ",").replace("\t", ",")
        try:
            parts = next(csv.reader([normalized], skipinitialspace=True))
        except Exception:
            parts = [segment.strip() for segment in normalized.split(",")]

        parts = [part.strip() for part in parts if part is not None]
        if not parts:
            continue

        raw_ticker = parts[0].upper().replace(".KA", "")
        ticker = raw_ticker
        # If ticker is not known, try fuzzy-match against known tickers and company names
        suggestion = None
        if known_tickers and ticker not in known_tickers:
            # try ticker matches first
            close = difflib.get_close_matches(ticker, list(known_tickers), n=1, cutoff=0.6)
            if close:
                suggestion = close[0]
            else:
                # try matching against company names
                close_name = difflib.get_close_matches(parts[0], known_names, n=1, cutoff=0.5)
                if close_name and df is not None and not df.empty:
                    # find ticker for that company name
                    matched = df[df.get("company_name", pd.Series(dtype=object)).astype(str) == close_name[0]]
                    if not matched.empty:
                        suggestion = str(matched.iloc[0].get("ticker", "")).upper()

        shares = _safe_float(parts[1]) if len(parts) > 1 else 0.0
        purchase_price = _safe_float(parts[2]) if len(parts) > 2 else float("nan")
        current_price = _safe_float(parts[3]) if len(parts) > 3 else float("nan")
        company_name = parts[4] if len(parts) > 4 else ticker
        sector = parts[5] if len(parts) > 5 else "Unknown"

        entry = {
            "raw_ticker": raw_ticker,
            "ticker": ticker,
            "shares": shares,
            "purchase_price": purchase_price,
            "current_price": current_price,
            "company_name": company_name,
            "sector": sector,
        }
        if suggestion and suggestion != ticker:
            entry["suggested_ticker"] = suggestion
            suggestions.append(f"{raw_line} → suggested: {suggestion}")

        holdings.append(entry)

    # store last parse preview in session for UI use
    if suggestions:
        st.session_state["zakat_parse_preview"] = {"holdings": holdings, "suggestions": suggestions}
    else:
        st.session_state["zakat_parse_preview"] = None

    return holdings


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        text = str(value).strip().replace(",", "")
        if not text:
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def _default_task_state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "idle",
        "phase": "symbols",
        "progress": 0,
        "detail": "",
        "logs": [],
        "result": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "ticker": "",
        "auto": False,
    }


TASK_STATES.update(
    {
        "pipeline": _default_task_state("pipeline"),
        "research": _default_task_state("research"),
    }
)


def _export_zakat_pdf_to_temp(result: dict) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            path = handle.name
        export_zakat_pdf(result, path)
        return path
    except Exception as exc:
        log_exception("Failed to export zakat PDF")
        st.error(f"Could not create PDF export: {exc}")
        return None


if __name__ == "__main__":
    main()
