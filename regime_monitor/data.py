"""Self-contained OHLCV data access via yfinance.

A thin, dependency-light fetch layer with an optional on-disk parquet cache so
the pipeline can run offline (``--cache-only``) and stays polite to the data
source in CI. No API keys required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ohlcv_cache"

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame to a lowercase OHLCV frame with a tz-naive
    DatetimeIndex, sorted ascending."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    # yfinance can return a column MultiIndex for a single symbol; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [c for c in _COLUMNS if c in df.columns]
    df = df[keep].copy()

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="all")
    # Drop rows with no close — yfinance can return a trailing NaN bar for the
    # current/partial day, which would poison indicators and (worse) emit NaN
    # into the JSON. The close is the one column everything depends on.
    if "close" in df.columns:
        df = df[df["close"].notna()]
    return df


def fetch_ohlcv(
    symbol: str,
    period: str = "3y",
    *,
    cache_only: bool = False,
    refresh: bool = True,
) -> pd.DataFrame:
    """Return a normalized OHLCV frame for ``symbol``.

    Reads the on-disk cache first; if ``cache_only`` is False and ``refresh``
    is True, fetches from yfinance and merges into the cache. Never raises on a
    network failure — falls back to whatever is cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol)

    cached = pd.DataFrame(columns=_COLUMNS)
    if path.exists():
        try:
            cached = _normalize(pd.read_parquet(path))
        except Exception:
            cached = pd.DataFrame(columns=_COLUMNS)

    if cache_only or not refresh:
        return cached

    fetched = pd.DataFrame(columns=_COLUMNS)
    try:
        import yfinance as yf

        raw = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        fetched = _normalize(raw)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[data] warning: fetch failed for {symbol}: {exc}; using cache")

    if fetched.empty:
        return cached

    merged = fetched if cached.empty else pd.concat([cached, fetched])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    try:
        merged.to_parquet(path)
    except Exception as exc:  # pragma: no cover
        print(f"[data] warning: could not write cache for {symbol}: {exc}")
    return merged
