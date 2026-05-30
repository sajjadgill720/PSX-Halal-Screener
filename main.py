"""Entry point for the PSX Halal Screener pipeline.

Run this module directly to fetch the configured PSX universe, apply AAOIFI
screening, and persist the results to ``data/screened_results.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.fetcher import get_psx_data
from screener.aaoifi import screen_aaoifi

PSX_TICKERS: dict[str, str] = {
    "OGDC": "Oil & Gas",
    "PPL": "Oil & Gas",
    "POL": "Oil & Gas",
    "MARI": "Oil & Gas",
    "PSO": "Oil & Gas",
    "ATRL": "Oil & Gas",
    "NRL": "Oil & Gas",
    "SHEL": "Oil & Gas",
    "HUBC": "Energy",
    "KAPCO": "Energy",
    "NCPL": "Energy",
    "KEL": "Energy",
    "PKGP": "Energy",
    "LUCK": "Cement",
    "DGKC": "Cement",
    "MLCF": "Cement",
    "CHCC": "Cement",
    "PIOC": "Cement",
    "FCCL": "Cement",
    "KOHC": "Cement",
    "NML": "Textiles",
    "GATM": "Textiles",
    "NCL": "Textiles",
    "CLOV": "Textiles",
    "ILP": "Textiles",
    "KTML": "Textiles",
    "ENGRO": "Fertilizer",
    "FFC": "Fertilizer",
    "EFERT": "Fertilizer",
    "FATIMA": "Fertilizer",
    "FFBL": "Fertilizer",
    "NESTLE": "Food",
    "UNITY": "Food",
    "QUICE": "Food",
    "TREET": "Food",
    "FCEPL": "Food",
    "SEARL": "Pharma",
    "GLAXO": "Pharma",
    "HINOON": "Pharma",
    "FEROZ": "Pharma",
    "ABOT": "Pharma",
    "TRG": "Tech",
    "NETSOL": "Tech",
    "SYSTEMS": "Tech",
    "AVN": "Tech",
    "AIRLINK": "Tech",
    "ISL": "Steel",
    "ASTL": "Steel",
    "MUGHAL": "Steel",
    "ASL": "Steel",
    "INDU": "Auto",
    "PSMC": "Auto",
    "HCAR": "Auto",
    "GHNI": "Auto",
    "SAZEW": "Auto",
    "HBL": "Banking (Conventional)",
    "MCB": "Banking (Conventional)",
    "UBL": "Banking (Conventional)",
    "NBP": "Banking (Conventional)",
    "BAFL": "Banking (Conventional)",
}


def build_screened_universe() -> pd.DataFrame:
    """Fetch the stock universe and apply AAOIFI screening."""

    raw = get_psx_data(list(PSX_TICKERS.keys())).copy()
    if raw.empty:
        return raw

    raw["sector"] = raw["ticker"].map(PSX_TICKERS).fillna(raw.get("sector", "Unknown"))
    raw["company_name"] = raw["ticker"]

    # The fetcher currently does not provide a separate non-compliant investments field.
    # Using cash and equivalents here keeps the AAOIFI securities screen computable.
    raw["non_compliant_investments"] = pd.to_numeric(raw.get("cash_and_equivalents"), errors="coerce").fillna(0.0)

    screened = screen_aaoifi(raw)
    screened["company_name"] = raw["company_name"]
    screened["sector"] = raw["sector"]
    return screened


def save_results(frame: pd.DataFrame) -> Path:
    """Persist screened results to the data directory."""

    output_path = Path(__file__).resolve().parent / "data" / "screened_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def print_summary(frame: pd.DataFrame) -> None:
    """Print a one-line screening summary to the console."""

    total = int(len(frame))
    halal = int((frame["overall_status"] == "HALAL").sum())
    doubtful = int((frame["overall_status"] == "DOUBTFUL").sum())
    haram = int((frame["overall_status"] == "HARAM").sum())

    print(f"{halal} halal, {doubtful} doubtful, {haram} haram out of {total} total")


def main() -> int:
    """Run the full screening pipeline and save the result set."""

    screened = build_screened_universe()
    if screened.empty:
        print("No screening results could be generated.")
        return 1

    output_path = save_results(screened)
    print_summary(screened)
    print(f"Saved screened results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
