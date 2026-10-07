"""Causal per-stock features computed from a bar Panel.

Every feature at bar t depends only on bars <= t (rolling windows, shifts and
same-row cross-sectional operations), so features computed on a truncated
panel equal the corresponding rows computed on the full panel.
"""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 3, 7, 14, 35, 70, 140)
SLOT_HISTORY = 20


def _xs_demean(frame):
    return frame.sub(frame.mean(axis=1), axis=0)


def _xs_rank(frame):
    return frame.rank(axis=1, pct=True) - 0.5


def compute_features(panel):
    """Return (names, values, index) with values shaped [bars, tickers, features]."""
    close, high, low, volume = panel.close, panel.high, panel.low, panel.volume
    logret = np.log(close).diff()
    feats = {}
    for k in RETURN_WINDOWS:
        r = close / close.shift(k) - 1
        feats[f'rel_r{k}'] = _xs_demean(r)
        feats[f'rank_r{k}'] = _xs_rank(r)
    for k in (7, 35, 140):
        feats[f'vol{k}'] = logret.rolling(k, min_periods=max(3, k // 2)).std()
    feats['rank_vol35'] = _xs_rank(feats['vol35'])
    rng = (high - low) / close
    feats['range1'] = rng
    feats['range7'] = rng.rolling(7, min_periods=3).mean()
    feats['dist_high35'] = close / high.rolling(35, min_periods=10).max() - 1
    feats['dist_low35'] = close / low.rolling(35, min_periods=10).min() - 1

    days = close.index.normalize()
    slot = pd.Series(close.groupby(days).cumcount().values, index=close.index)
    past_slot_volume = volume.groupby(slot.values).transform(
        lambda x: x.rolling(SLOT_HISTORY, min_periods=5).mean().shift(1))
    feats['volume_z'] = np.log1p(volume) - np.log1p(past_slot_volume)
    feats['rank_volume_z'] = _xs_rank(feats['volume_z'])

    prev_close = close.groupby(days).last().shift(1).reindex(days).set_axis(close.index)
    intraday = close / prev_close - 1
    feats['rel_intraday'] = _xs_demean(intraday)

    market = logret.mean(axis=1)
    market_feats = {
        'mkt_r7': market.rolling(7, min_periods=3).sum(),
        'mkt_r35': market.rolling(35, min_periods=10).sum(),
        'mkt_vol35': market.rolling(35, min_periods=10).std(),
        'dispersion7': (close / close.shift(7) - 1).std(axis=1),
        'slot': slot.astype(float),
        'weekday': pd.Series(close.index.weekday, index=close.index).astype(float),
    }
    names = list(feats) + list(market_feats)
    n_bars, n_tickers = close.shape
    values = np.empty((n_bars, n_tickers, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        if name in feats:
            values[:, :, j] = feats[name].values
        else:
            values[:, :, j] = np.repeat(market_feats[name].values[:, None], n_tickers, axis=1)
    return names, values, close.index


def cutoff_positions(panel, cutoffs):
    """Index of the last bar that ended at or before each cutoff (-1 if none)."""
    return np.searchsorted(panel.end.values, pd.DatetimeIndex(cutoffs).values, side='right') - 1
