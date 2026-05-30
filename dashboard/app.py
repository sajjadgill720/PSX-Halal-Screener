"""Streamlit dashboard for the PSX Halal Screener.

Run with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import PSX_TICKERS, get_psx_data
from screener.aaoifi import screen_aaoifi
from screener.zakat import calculate_portfolio_zakat, get_nisab_pkr, is_zakat_due

GREEN = "#1a6b3c"
GOLD = "#c9a84c"
BG = "#f5f4ef"
TEXT = "#12301f"
HALAL = "#1a6b3c"
DOUBTFUL = "#c9a84c"
HARAM = "#b03a2e"
MUTED = "#6f7b74"


def main() -> None:
    """Render the PSX Halal Screener dashboard."""

    if get_script_run_ctx() is None:
        print("Run this app with: streamlit run dashboard/app.py")
        return

    st.set_page_config(
        page_title="PSX Halal Screener",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _render_sidebar()

    st.title("PSX Halal Screener")
    st.caption("AAOIFI-style screening, zakat estimates, and educational guidance for PSX equities.")

    tabs = st.tabs(["Screener", "Zakat Calculator", "Learn"])

    with tabs[0]:
        _render_screener_tab()
    with tabs[1]:
        _render_zakat_tab()
    with tabs[2]:
        _render_learn_tab()

    _render_footer()


def _inject_css() -> None:
    """Inject the visual theme and component styles."""

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #fbfbf8 0%, #f5f4ef 40%, #edf2eb 100%);
                color: {TEXT};
            }}
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0f271a 0%, #173b28 55%, #1a6b3c 100%);
                color: #fff;
            }}
            section[data-testid="stSidebar"] * {{
                color: #fff;
            }}
            .brand-card {{
                border-radius: 22px;
                padding: 18px 16px;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.14);
                box-shadow: 0 18px 40px rgba(7, 20, 13, 0.26);
                backdrop-filter: blur(8px);
            }}
            .brand-title {{
                font-size: 1.35rem;
                font-weight: 800;
                margin: 0.2rem 0 0;
                letter-spacing: 0.03em;
            }}
            .brand-subtitle {{
                font-size: 0.86rem;
                color: rgba(255,255,255,0.82);
                margin: 0.35rem 0 0;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
                margin: 0.75rem 0 1.15rem;
            }}
            .metric-card {{
                border-radius: 20px;
                padding: 18px 18px 16px;
                border: 1px solid rgba(18, 48, 31, 0.08);
                box-shadow: 0 10px 26px rgba(18, 48, 31, 0.06);
                background: white;
            }}
            .metric-card .label {{
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: {MUTED};
                margin-bottom: 0.45rem;
            }}
            .metric-card .value {{
                font-size: 2rem;
                font-weight: 800;
                line-height: 1;
            }}
            .metric-card.halal {{ border-left: 6px solid {HALAL}; }}
            .metric-card.doubtful {{ border-left: 6px solid {DOUBTFUL}; }}
            .metric-card.haram {{ border-left: 6px solid {HARAM}; }}
            .metric-card.total {{ border-left: 6px solid {GREEN}; }}
            .status-badge {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.38rem 0.8rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .status-halal {{ background: rgba(26, 107, 60, 0.12); color: {HALAL}; }}
            .status-doubtful {{ background: rgba(201, 168, 76, 0.18); color: #8f6f1e; }}
            .status-haram {{ background: rgba(176, 58, 46, 0.12); color: {HARAM}; }}
            .detail-card {{
                background: white;
                border: 1px solid rgba(18, 48, 31, 0.08);
                border-radius: 22px;
                padding: 18px 18px 10px;
                box-shadow: 0 12px 30px rgba(18, 48, 31, 0.06);
            }}
            .footer-note {{
                margin-top: 2rem;
                padding: 1rem 0 0.35rem;
                color: {MUTED};
                text-align: center;
                font-size: 0.9rem;
                border-top: 1px solid rgba(18, 48, 31, 0.08);
            }}
            .small-muted {{ color: {MUTED}; font-size: 0.88rem; }}
            .geom-wrap {{
                width: 100%;
                border-radius: 20px;
                overflow: hidden;
                margin-bottom: 14px;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12);
            }}
            .section-heading {{
                font-size: 1.05rem;
                font-weight: 800;
                margin: 0 0 0.6rem;
                color: {TEXT};
            }}
            .learn-card {{
                background: white;
                border-radius: 22px;
                padding: 18px 18px 10px;
                border: 1px solid rgba(18, 48, 31, 0.08);
                box-shadow: 0 12px 30px rgba(18, 48, 31, 0.05);
            }}
            .table-caption {{
                margin: 0.25rem 0 0.75rem;
                color: {MUTED};
                font-size: 0.92rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    """Render the branded sidebar and global controls."""

    st.sidebar.markdown(
        """
        <div class="brand-card">
            <div class="geom-wrap">
                <svg viewBox="0 0 320 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
                            <stop offset="0%" stop-color="#c9a84c"/>
                            <stop offset="100%" stop-color="#1a6b3c"/>
                        </linearGradient>
                    </defs>
                    <rect width="320" height="120" fill="#0f271a"/>
                    <g fill="none" stroke="url(#g)" stroke-width="2" opacity="0.9">
                        <path d="M0 60 H320"/>
                        <path d="M160 0 V120"/>
                        <path d="M40 20 L80 60 L40 100 L0 60 Z"/>
                        <path d="M120 20 L160 60 L120 100 L80 60 Z"/>
                        <path d="M200 20 L240 60 L200 100 L160 60 Z"/>
                        <path d="M280 20 L320 60 L280 100 L240 60 Z"/>
                        <circle cx="160" cy="60" r="24"/>
                        <circle cx="160" cy="60" r="11"/>
                    </g>
                </svg>
            </div>
            <p class="brand-title">PSX Halal Screener</p>
            <p class="brand-subtitle">AAOIFI screening and zakat tools for Pakistan equities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Filters")
    if st.sidebar.button("Refresh Data", use_container_width=True):
        _clear_caches()
        st.rerun()


def _render_screener_tab() -> None:
    """Render the screening dashboard tab."""

    data = load_screened_universe(0)
    if data.empty:
        st.warning("No screener data could be loaded.")
        return

    available_sectors = sorted(str(value) for value in data["sector"].dropna().unique())
    available_statuses = ["HALAL", "DOUBTFUL", "HARAM"]

    st.sidebar.markdown("#### Screener Filters")
    selected_statuses = st.sidebar.multiselect(
        "Status",
        available_statuses,
        default=available_statuses,
        help="Filter stocks by their overall AAOIFI result.",
    )
    selected_sectors = st.sidebar.multiselect(
        "Sector",
        available_sectors,
        default=available_sectors,
        help="Filter by sector or industry group.",
    )
    minimum_score = st.sidebar.slider("Minimum Shariah Score", 0, 100, 0, 1)

    filtered = data.loc[
        data["overall_status"].isin(selected_statuses)
        & data["sector"].isin(selected_sectors)
        & (pd.to_numeric(data["shariah_score"], errors="coerce") >= minimum_score)
    ].copy()

    halal_count = int((filtered["overall_status"] == "HALAL").sum())
    doubtful_count = int((filtered["overall_status"] == "DOUBTFUL").sum())
    haram_count = int((filtered["overall_status"] == "HARAM").sum())
    total_screened = int(len(filtered))

    _render_metric_grid(
        [
            ("Total Screened", total_screened, "total"),
            ("Halal Count", halal_count, "halal"),
            ("Doubtful Count", doubtful_count, "doubtful"),
            ("Haram Count", haram_count, "haram"),
        ]
    )

    if filtered.empty:
        st.info("No companies match the current filters.")
        return

    display = filtered[
        [
            "ticker",
            "company_name",
            "sector",
            "shariah_score",
            "debt_ratio",
            "interest_ratio",
            "overall_status",
            "purification_ratio",
        ]
    ].copy()
    display["Overall Status"] = display["overall_status"].map(_badge_text)
    display["Shariah Score"] = pd.to_numeric(display["shariah_score"], errors="coerce").fillna(0).round(0).astype(int)
    display["Debt Ratio"] = pd.to_numeric(display["debt_ratio"], errors="coerce").map(_format_ratio)
    display["Interest Ratio"] = pd.to_numeric(display["interest_ratio"], errors="coerce").map(_format_ratio)
    display["Purification Ratio"] = pd.to_numeric(display["purification_ratio"], errors="coerce").map(_format_ratio)

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Screened Universe</p>', unsafe_allow_html=True)
    st.markdown('<p class="table-caption">Progress bars show how comfortably a stock stays under the AAOIFI thresholds.</p>', unsafe_allow_html=True)

    table = display[
        [
            "ticker",
            "company_name",
            "sector",
            "Shariah Score",
            "Debt Ratio",
            "Interest Ratio",
            "Overall Status",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company Name",
            "sector": "Sector",
        }
    )

    selection = None
    try:
        event = st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            key="screener_table",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Shariah Score": st.column_config.ProgressColumn("Shariah Score", help="0-100 score", min_value=0, max_value=100),
                "Debt Ratio": st.column_config.TextColumn("Debt Ratio"),
                "Interest Ratio": st.column_config.TextColumn("Interest Ratio"),
                "Overall Status": st.column_config.TextColumn("Overall Status"),
            },
        )
        rows = getattr(getattr(event, "selection", None), "rows", [])
        if rows:
            selection = filtered.iloc[int(rows[0])]
    except Exception:
        pass

    if selection is None:
        selected_ticker = st.selectbox("Open company detail card", filtered["ticker"].tolist())
        selection = filtered.loc[filtered["ticker"] == selected_ticker].iloc[0]

    _render_company_detail(selection)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_company_detail(row: pd.Series) -> None:
    """Render a selected company detail card with all five AAOIFI screens."""

    st.markdown(
        f"""
        <div style="margin-top: 1rem;">
            <h3 style="margin-bottom: 0.2rem;">{row.get('company_name', row['ticker'])} ({row['ticker']})</h3>
            <p class="small-muted">Sector: {row.get('sector', 'Unknown')} | Status: {row.get('overall_status', 'UNKNOWN')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    screen_items = [
        ("Business Screen", row.get("business_screen"), "Business activity"),
        ("Debt Screen", row.get("debt_screen"), f"Debt ratio: {_format_ratio(row.get('debt_ratio'))} < 0.33"),
        ("Interest Income Screen", row.get("interest_screen"), f"Interest ratio: {_format_ratio(row.get('interest_ratio'))} < 0.05"),
        ("Securities Screen", row.get("securities_screen"), f"Securities ratio: {_format_ratio(row.get('securities_ratio'))} < 0.33"),
        ("Receivables Screen", row.get("receivables_screen"), f"Receivables ratio: {_format_ratio(row.get('receivables_ratio'))} < 0.49"),
    ]

    cols = st.columns(5)
    for column, (title, result, detail) in zip(cols, screen_items):
        passed = str(result).upper() == "PASS"
        icon = "✅" if passed else "❌"
        color = HALAL if passed else HARAM if title != "Business Screen" or not passed else DOUBTFUL
        if title == "Business Screen" and not passed:
            color = HARAM
        column.markdown(
            f"""
            <div style="border-radius: 18px; border: 1px solid rgba(18,48,31,0.08); padding: 14px; background: #fff; box-shadow: 0 8px 20px rgba(18,48,31,0.05); min-height: 130px;">
                <div style="font-weight: 800; margin-bottom: 0.35rem;">{title}</div>
                <div style="font-size: 1.4rem; font-weight: 900; color: {color};">{icon} {result}</div>
                <div class="small-muted" style="margin-top: 0.55rem;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if str(row.get("overall_status")) == "HALAL" and pd.notna(row.get("purification_ratio")):
        st.info(
            f"Purification ratio: {float(row['purification_ratio']):.4f}. This is the fraction of income that may need to be purified before charitable use."
        )
    elif pd.notna(row.get("purification_ratio")):
        st.caption(
            "Purification ratio is shown for reference when a stock is Halal. Non-Halal names should not be treated as investable until a qualified scholar reviews them."
        )


def _render_zakat_tab() -> None:
    """Render the Zakat calculator tab."""

    st.subheader("Zakat Calculator")
    st.caption("Enter your holdings, choose a calculation method, and compare the result against nisab.")

    default_rows = pd.DataFrame(
        [
            {"Ticker": "ENGRO", "Shares Held": 0.0},
            {"Ticker": "LUCK", "Shares Held": 0.0},
            {"Ticker": "HBL", "Shares Held": 0.0},
        ]
    )

    portfolio_editor = st.data_editor(
        default_rows,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="PSX ticker symbol, for example ENGRO or LUCK."),
            "Shares Held": st.column_config.NumberColumn("Shares Held", min_value=0.0, step=1.0),
        },
        key="portfolio_editor",
    )

    method_label = st.radio("Calculation Method", ["Zakatable Assets", "Market Value"], horizontal=True)
    method = "assets" if method_label == "Zakatable Assets" else "market"

    holdings = _editor_to_holdings(portfolio_editor)
    portfolio_df = load_portfolio_zakat(tuple(sorted(holdings.items())), method, 0)

    total_portfolio_value = float(pd.to_numeric(portfolio_df.loc[portfolio_df["ticker"] != "TOTAL", "market_value"], errors="coerce").fillna(0).sum())
    nisab = load_nisab_threshold(0)
    zakat_due = is_zakat_due(total_portfolio_value)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nisab Threshold (PKR)", _format_currency(nisab))
    c2.metric("Portfolio Market Value (PKR)", _format_currency(total_portfolio_value))
    c3.metric("Zakat Due", "Yes" if zakat_due else "No")

    explanation = (
        "Your portfolio value meets or exceeds nisab, so zakat is due." if zakat_due else
        "Your portfolio value is below nisab, so zakat is not due at this time."
    )
    st.info(explanation)

    if portfolio_df.empty:
        st.warning("Add at least one holding to calculate zakat.")
        return

    result_view = portfolio_df.copy()
    for column in ["market_value", "zakatable_base", "zakat_due"]:
        result_view[column] = pd.to_numeric(result_view[column], errors="coerce")

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Per-Stock Zakat</p>', unsafe_allow_html=True)
    st.dataframe(
        result_view,
        hide_index=True,
        use_container_width=True,
        column_config={
            "market_value": st.column_config.NumberColumn("Market Value (PKR)", format="PKR %.2f"),
            "zakatable_base": st.column_config.NumberColumn("Zakatable Base (PKR)", format="PKR %.2f"),
            "zakat_due": st.column_config.NumberColumn("Zakat Due (PKR)", format="PKR %.2f"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    total_zakat = float(pd.to_numeric(portfolio_df.loc[portfolio_df["ticker"] == "TOTAL", "zakat_due"], errors="coerce").fillna(0).iloc[0]) if not portfolio_df.empty else 0.0
    st.success(f"Total Zakat Due: {_format_currency(total_zakat)}")

    pdf_bytes = build_zakat_pdf(portfolio_df, nisab, zakat_due, method_label)
    st.download_button(
        "Download Zakat Report",
        data=pdf_bytes,
        file_name=f"psx_zakat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _render_learn_tab() -> None:
    """Render the educational content tab."""

    st.subheader("Learn")
    st.caption("A concise reference for the standards and calculations used in this dashboard.")

    sections = [
        (
            "What AAOIFI standards are",
            "AAOIFI standards are widely used Shariah governance and accounting guidelines for Islamic finance. In equity screening, they define which sectors are prohibited and how to apply financial ratios to determine whether a stock may be considered investable.",
        ),
        (
            "Screening ratios with examples",
            "Debt ratio = total debt / total assets. If a company has PKR 20 billion in debt and PKR 80 billion in assets, the ratio is 0.25 and passes the 0.33 threshold. Interest income ratio = interest income / total revenue. If interest income is PKR 2 billion and revenue is PKR 100 billion, the ratio is 0.02 and passes the 0.05 threshold.",
        ),
        (
            "Halal, Doubtful, and Haram",
            "Halal means the business screen and all financial screens pass. Doubtful means the business screen passes but one or two financial screens fail, so a scholar may want to review the case. Haram means the business screen fails or three or more financial screens fail.",
        ),
        (
            "Purification",
            "Purification is the process of removing the impermissible portion of income from Halal holdings before using the proceeds. In this dashboard, the purification ratio equals interest income divided by total revenue for stocks that pass screening.",
        ),
        (
            "Zakat methods",
            "Method 1, the zakatable assets method, calculates zakat on cash and receivables attributable to the shareholding at 2.5775%, which reflects the lunar-year adjustment. Method 2, the market value method, applies the simpler 2.5% rate to the full market value of the holding and is more conservative.",
        ),
    ]

    for title, body in sections:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_screened_universe(refresh_token: int = 0) -> pd.DataFrame:
    """Fetch PSX data, enrich it, and run AAOIFI screening.

    The refresh token is part of the cache signature so the sidebar refresh
    button can invalidate cached results on demand.
    """

    raw = get_psx_data(list(PSX_TICKERS.keys())).copy()
    if raw.empty:
        return raw

    company_map = _load_company_name_map(tuple(sorted(PSX_TICKERS.keys())), refresh_token)
    raw["company_name"] = raw["ticker"].map(company_map).fillna(raw["ticker"])
    raw["non_compliant_investments"] = pd.to_numeric(raw.get("cash_and_equivalents"), errors="coerce").fillna(np.nan)
    raw["non_compliant_investments"] = raw["non_compliant_investments"].fillna(0.0)
    screened = screen_aaoifi(raw)
    screened["company_name"] = raw["company_name"]
    screened["company_name"] = screened["company_name"].fillna(screened["ticker"])
    screened["sector"] = screened["sector"].fillna("Unknown")
    screened["business_screen"] = screened["business_screen"].fillna("PASS")
    screened["overall_status"] = screened["overall_status"].fillna("DOUBTFUL")
    return screened


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _load_company_name_map(tickers: tuple[str, ...], refresh_token: int = 0) -> dict[str, str]:
    """Resolve ticker symbols to friendly company names using Yahoo Finance."""

    import yfinance as yf

    mapping: dict[str, str] = {}
    for ticker in tickers:
        yahoo_symbol = f"{ticker}.KA" if not str(ticker).upper().endswith(".KA") else str(ticker)
        try:
            company = yf.Ticker(yahoo_symbol)
            info = getattr(company, "info", {}) or {}
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            name = ticker
        mapping[str(ticker)] = str(name)
    return mapping


def _load_company_names(ticker: str) -> str:
    """Resolve a single company name using the cached ticker map."""

    company_map = _load_company_name_map(tuple(sorted(PSX_TICKERS.keys())), 0)
    return company_map.get(str(ticker), str(ticker))


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_portfolio_zakat(holdings_key: tuple[tuple[str, float], ...], method: str, refresh_token: int = 0) -> pd.DataFrame:
    """Cache portfolio zakat calculations for the current holdings."""

    holdings = {ticker: shares for ticker, shares in holdings_key}
    return calculate_portfolio_zakat(holdings, method=method)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_nisab_threshold(refresh_token: int = 0) -> float:
    """Cache the current nisab threshold."""

    return float(get_nisab_pkr())


def build_zakat_pdf(portfolio_df: pd.DataFrame, nisab: float, zakat_due: bool, method_label: str) -> bytes:
    """Build a simple PDF report for download.

    The PDF is created without external dependencies so the dashboard can offer
    a download even in minimal environments.
    """

    rows = portfolio_df.copy()
    rows = rows.fillna("")
    lines = [
        "PSX Halal Screener - Zakat Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Method: {method_label}",
        f"Nisab (PKR): {_format_currency(nisab)}",
        f"Zakat Due: {'Yes' if zakat_due else 'No'}",
        "",
        "Ticker | Shares | Market Value | Zakatable Base | Zakat Due | Method",
        "-" * 90,
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{row.get('ticker', '')} | {row.get('shares_held', '')} | {row.get('market_value', '')} | {row.get('zakatable_base', '')} | {row.get('zakat_due', '')} | {row.get('method_used', '')}"
        )
    return _simple_pdf_bytes(lines)


def _simple_pdf_bytes(lines: Iterable[str]) -> bytes:
    """Render plain text lines into a small valid PDF file."""

    pages = []
    lines = list(lines)
    chunk_size = 38
    for start in range(0, len(lines), chunk_size):
        pages.append(lines[start : start + chunk_size])

    if not pages:
        pages = [["PSX Halal Screener"]]

    objects: list[bytes] = []
    offsets: list[int] = []

    def add_object(content: bytes) -> None:
        offsets.append(sum(len(obj) for obj in objects) + len(b"%PDF-1.4\n"))
        objects.append(content)

    font_obj_num = 3 + len(pages) * 2
    catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>\nendobj\n".encode()
    add_object(catalog_obj)
    add_object(pages_obj)

    for index, page_lines in enumerate(pages):
        page_num = 3 + index * 2
        content_num = page_num + 1
        stream_lines = ["BT", "/F1 11 Tf", "50 790 Td"]
        first = True
        for line in page_lines:
            escaped = _escape_pdf_text(str(line))
            if first:
                stream_lines.append(f"({escaped}) Tj")
                first = False
            else:
                stream_lines.append("0 -14 Td")
                stream_lines.append(f"({escaped}) Tj")
        stream_lines.append("ET")
        content_stream = "\n".join(stream_lines).encode("utf-8")
        page_obj = (
            f"{page_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_num} 0 R >>\n"
            f"endobj\n"
        ).encode()
        content_obj = (
            f"{content_num} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n".encode()
            + content_stream
            + b"\nendstream\nendobj\n"
        )
        add_object(page_obj)
        add_object(content_obj)

    font_obj = b"%d 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n" % font_obj_num
    add_object(font_obj)

    body = b"".join(objects)
    xref_offset = len(b"%PDF-1.4\n") + len(body)

    xref_entries = [b"0000000000 65535 f \n"]
    cursor = len(b"%PDF-1.4\n")
    for obj in objects:
        xref_entries.append(f"{cursor:010d} 00000 n \n".encode())
        cursor += len(obj)

    trailer = (
        f"xref\n0 {len(xref_entries)}\n".encode()
        + b"".join(xref_entries)
        + f"trailer\n<< /Size {len(xref_entries)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )

    return b"%PDF-1.4\n" + body + trailer


def _escape_pdf_text(text: str) -> str:
    """Escape parentheses and backslashes for PDF text streams."""

    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _format_ratio(value: object) -> str:
    """Format a ratio for display."""

    try:
        numeric = float(value)
    except Exception:
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    return f"{numeric:.2%}"


def _format_currency(value: object) -> str:
    """Format a PKR amount for display."""

    try:
        numeric = float(value)
    except Exception:
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    return f"PKR {numeric:,.0f}"


def _badge_text(status: object) -> str:
    """Return a compact badge label for table rows."""

    value = str(status).upper()
    if value == "HALAL":
        return "🟩 HALAL"
    if value == "DOUBTFUL":
        return "🟨 DOUBTFUL"
    if value == "HARAM":
        return "🟥 HARAM"
    return value


def _render_metric_grid(items: list[tuple[str, int, str]]) -> None:
    """Render the four top-line KPI cards."""

    html = ["<div class='metric-grid'>"]
    for label, value, kind in items:
        html.append(
            f"""
            <div class="metric-card {kind}">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _editor_to_holdings(editor_df: pd.DataFrame) -> dict[str, float]:
    """Convert the data editor output into a holdings dictionary."""

    holdings: dict[str, float] = {}
    if editor_df is None or editor_df.empty:
        return holdings

    for _, row in editor_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        shares = pd.to_numeric(pd.Series([row.get("Shares Held", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
        if float(shares) <= 0:
            continue
        holdings[ticker] = float(shares)
    return holdings


def _clear_caches() -> None:
    """Clear Streamlit caches and local cached helpers."""

    st.cache_data.clear()
    try:
        get_nisab_pkr.cache_clear()
    except Exception:
        pass


def _render_footer() -> None:
    """Render the disclaimer footer."""

    st.markdown(
        '<div class="footer-note">This tool provides screening guidance only. Consult a qualified Shariah scholar for fatwa.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
