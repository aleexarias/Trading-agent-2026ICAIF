"""Live market data from Yahoo Finance and portfolio parsing for the runner."""

from datetime import datetime

import numpy as np
import pandas as pd

from bot.data import ET, _flatten_yahoo
from bot.market import SYMBOLS, panel_from_long

BAR_BOUNDARIES = [(10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30), (16, 0)]
HISTORY = '400d'
MAX_LAGGING = 3


class DataError(RuntimeError):
    """Live market data is missing, stale or malformed."""


def expected_last_bar_end(now):
    """Latest hourly bar end that Yahoo should have published by `now` (ET, naive).

    Uses weekdays as trading days; on a market holiday this is later than the
    data, which makes the check fail safe (the round is skipped and holdings kept).
    """
    now = pd.Timestamp(now).tz_convert(ET).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
    settle = now - pd.Timedelta(minutes=2)
    day = settle.normalize()
    if day.weekday() < 5:
        today = [day + pd.Timedelta(hours=h, minutes=m) for h, m in BAR_BOUNDARIES]
        done = [b for b in today if b <= settle]
        if done:
            return done[-1]
    previous = day - pd.Timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= pd.Timedelta(days=1)
    return previous + pd.Timedelta(hours=16)


def download_bars(now=None, period=HISTORY):
    """Download hourly bars and return (long frame of completed bars, download time)."""
    import yfinance as yf
    now = now or datetime.now(ET)
    frame = yf.download(SYMBOLS, period=period, interval='1h', auto_adjust=False, prepost=False,
                        progress=False, threads=True)
    if frame is None or frame.empty:
        raise DataError('Yahoo returned no hourly data')
    bars = _flatten_yahoo(frame, SYMBOLS).dropna(subset=['close'])
    bars['timestamp_et'] = pd.to_datetime(bars.timestamp_et, utc=True).dt.tz_convert(ET)
    is_last = bars.timestamp_et.dt.strftime('%H:%M') == '15:30'
    end = bars.timestamp_et + pd.to_timedelta(np.where(is_last, 30, 60), unit='min')
    bars = bars[end <= pd.Timestamp(now)].copy()
    bars['timestamp_et'] = bars.timestamp_et.dt.tz_localize(None)
    return bars, now


def yahoo_bar_ends(index):
    """Yahoo hourly bars last one hour, except the 15:30 bar which ends at the 16:00 close."""
    idx = pd.DatetimeIndex(index)
    minutes = np.where((idx.hour == 15) & (idx.minute == 30), 30, 60)
    return pd.Series(idx + pd.to_timedelta(minutes, unit='min'), index=idx)


def load_live_panel(now=None, bars=None):
    """Fetch and validate live bars; return (panel, diagnostics) or raise DataError.

    Up to MAX_LAGGING symbols may miss only the latest bar (Yahoo occasionally
    publishes a final bar late); their previous close is carried forward.
    """
    if bars is None:
        bars, now = download_bars(now)
    now = pd.Timestamp(now or datetime.now(ET))
    if bars.empty:
        raise DataError('No completed hourly bars')
    present = bars.groupby('ticker').timestamp_et.max()
    missing = sorted(set(SYMBOLS) - set(present.index))
    if missing:
        raise DataError(f'Missing symbols: {missing}')
    panel = panel_from_long(bars)
    panel.end = yahoo_bar_ends(panel.index)
    last_start = panel.index[-1]
    previous_start = panel.index[-2]
    lagging = sorted(present[present < last_start].index)
    too_old = sorted(present[present < previous_start].index)
    if len(lagging) > MAX_LAGGING or too_old:
        raise DataError(f'Symbols without a recent bar at {last_start}: {lagging}')
    last_end = panel.end.iloc[-1]
    expected = expected_last_bar_end(now)
    if last_end < expected:
        raise DataError(f'Stale data: last bar ends {last_end}, expected {expected}')
    prices = panel.close.iloc[-1].values
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DataError('Non-positive or missing latest prices')
    days = panel.close.index.normalize().nunique()
    return panel, {'bars': len(panel.index), 'days': int(days), 'last_bar_start': str(last_start),
                   'last_bar_end': str(last_end), 'expected_last_end': str(expected),
                   'lagging_symbols': lagging}


SYMBOL_KEYS = ('symbol', 'ticker')
SHARE_KEYS = ('shares', 'quantity', 'qty', 'units')


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def current_weights(portfolio, prices):
    """Current weights from the backend portfolio, marked at the latest prices.

    Positions with share counts are revalued at `prices`; otherwise the
    backend's `current_weights` snapshot is used. Returns (weights, method).
    """
    prices = np.asarray(prices, dtype=float)
    cash = _number(portfolio.get('cash'))
    shares = np.zeros(len(SYMBOLS))
    found = False
    positions = portfolio.get('positions') or []
    if isinstance(positions, dict):
        positions = [{'symbol': k, **(v if isinstance(v, dict) else {'shares': v})} for k, v in positions.items()]
    for item in positions:
        if not isinstance(item, dict):
            continue
        symbol = next((item[k] for k in SYMBOL_KEYS if k in item), None)
        qty = next((_number(item[k]) for k in SHARE_KEYS if k in item), None)
        if symbol in SYMBOLS and qty is not None:
            shares[SYMBOLS.index(symbol)] = qty
            found = True
    if cash is not None and (found or not positions):
        values = shares * prices
        total = cash + values.sum()
        if total > 0:
            return values / total, 'positions' if found else 'cash_only'
    snapshot = portfolio.get('current_weights') or {}
    weights = np.array([_number(snapshot.get(s)) or 0.0 for s in SYMBOLS])
    if snapshot and np.isfinite(weights).all() and weights.sum() <= 1 + 1e-6:
        return weights, 'snapshot'
    raise DataError('Could not determine current portfolio weights')
