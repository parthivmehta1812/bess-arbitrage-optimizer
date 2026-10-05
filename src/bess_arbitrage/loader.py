"""
Price data loading utilities.

Accepts CSV or Excel files with hourly electricity prices (EUR/MWh).
Falls back to a bundled sample dataset when no file is provided.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

_SAMPLE_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "Electricity_Price_2023_DE.xlsx"

HOURS_PER_YEAR = 8760


def load_prices(price_bytes: bytes | None = None) -> np.ndarray:
    """
    Load hourly electricity prices and return a numpy array of length 8760.

    Parameters
    ----------
    price_bytes : raw bytes from an uploaded CSV or XLSX file, or None to use
                  the bundled sample (2024 German ENTSO-E subset).

    Returns
    -------
    np.ndarray of shape (8760,), dtype float64, units EUR/MWh.

    Raises
    ------
    ValueError  if the file contains no numeric columns.
    FileNotFoundError  if no bytes are supplied and the sample file is missing.
    """
    if price_bytes:
        try:
            df = pd.read_excel(io.BytesIO(price_bytes), header=0)
        except Exception:
            df = pd.read_csv(io.BytesIO(price_bytes), header=0)
    else:
        if not _SAMPLE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Sample price file not found at {_SAMPLE_DATA_PATH}. "
                "Download ENTSO-E data per data/README.md or pass price_bytes."
            )
        print("[loader] No price file supplied — using bundled 2023 DE ENTSO-E prices.")
        df = pd.read_excel(_SAMPLE_DATA_PATH, header=0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError(
            "Price file contains no numeric columns. "
            "Expected at least one column with EUR/MWh values."
        )

    # Skip columns that look like a sequential row index (e.g. 0,1,2,...,8759)
    price_col = None
    for col in numeric_cols:
        vals = df[col].dropna().values
        if not (vals.min() == 0 and vals.max() == len(vals) - 1):
            price_col = col
            break
    if price_col is None:
        price_col = numeric_cols[-1]

    prices = df[price_col].dropna().values.astype(float)

    # Trim or forward-fill to exactly 8760 hours
    if len(prices) >= HOURS_PER_YEAR:
        prices = prices[:HOURS_PER_YEAR]
    else:
        padding = np.full(HOURS_PER_YEAR - len(prices), prices[-1])
        prices = np.concatenate([prices, padding])

    print(
        f"[loader] Prices loaded: {len(prices)} h | "
        f"min={prices.min():.1f}  max={prices.max():.1f}  "
        f"mean={prices.mean():.1f} EUR/MWh"
    )
    return prices
