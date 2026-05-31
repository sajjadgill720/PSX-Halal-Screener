"""Entry point for the PSX Halal Screener pipeline.

Run this module directly to fetch the configured PSX universe, apply AAOIFI
screening, and persist the results to ``data/screened_results.csv``.
"""

from __future__ import annotations

import pandas as pd

from screener.live_screener import run_full_screening


def print_summary(frame: pd.DataFrame) -> None:
    """Print a one-line screening summary to the console."""

    total = int(len(frame))
    halal = int((frame["overall_status"] == "HALAL").sum())
    doubtful = int((frame["overall_status"] == "DOUBTFUL").sum())
    haram = int((frame["overall_status"] == "HARAM").sum())

    print(f"{halal} halal, {doubtful} doubtful, {haram} haram out of {total} total")


def main() -> int:
    """Run the full screening pipeline and save the result set."""

    screened = run_full_screening(use_cache=True)
    if screened.empty:
        print("No screening results could be generated.")
        return 1

    print_summary(screened)
    print("Saved screened results to data/screened_results.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
