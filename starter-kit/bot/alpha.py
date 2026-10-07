"""Approach 2: pooled cross-sectional LightGBM alpha model.

One row per (decision round, ticker). Features are taken at the round's
information cutoff; the label is the forward execution-to-execution return over
`horizon` rounds, demeaned and ranked across the 30 tickers. Training windows
are purged so no label overlaps the following validation period.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bot.features import compute_features, cutoff_positions
from bot.market import SYMBOLS, execution_prices

PARAMS = {'objective': 'regression', 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4,
          'min_data_in_leaf': 500, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
          'lambda_l2': 10.0, 'verbose': -1, 'num_threads': 8, 'seed': 7, 'deterministic': True}
NUM_ROUNDS = 300


@dataclass
class Dataset:
    """Per-round feature matrices and labels."""

    names: list
    X: np.ndarray
    y: np.ndarray
    forward: np.ndarray
    when: pd.DatetimeIndex
    execution: pd.DatetimeIndex


def build_dataset(panel, rounds, horizon):
    """Features at each round cutoff and forward returns over `horizon` rounds."""
    names, values, _ = compute_features(panel)
    pos = cutoff_positions(panel, [r.opens_at for r in rounds])
    X = values[np.clip(pos, 0, None)]
    X[pos < 0] = np.nan
    prices = execution_prices(panel, [r.execution for r in rounds]).values
    fwd = np.full_like(prices, np.nan)
    fwd[:-horizon] = prices[horizon:] / prices[:-horizon] - 1
    with np.errstate(all='ignore'):
        rel = fwd - np.nanmean(np.where(np.isfinite(fwd), fwd, np.nan), axis=1, keepdims=True)
    ranks = pd.DataFrame(rel).rank(axis=1, pct=True).values - 0.5
    return Dataset(names, X, ranks, rel, pd.DatetimeIndex([r.opens_at for r in rounds]),
                   pd.DatetimeIndex([r.execution for r in rounds]))


def _rows(ds, mask):
    X = ds.X[mask].reshape(-1, ds.X.shape[2])
    y = ds.y[mask].reshape(-1)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[keep], y[keep]


class AlphaModel:
    """Thin wrapper around a LightGBM booster with a fixed feature order."""

    def __init__(self, booster, names, horizon):
        self.booster, self.names, self.horizon = booster, list(names), horizon

    def predict(self, names, row):
        if list(names) != self.names:
            raise ValueError('Feature order differs from the trained model')
        row = np.asarray(row, dtype=float)
        if not np.isfinite(row).all():
            row = np.where(np.isfinite(row), row, np.nanmedian(row, axis=0))
        return self.booster.predict(row)

    def predict_rounds(self, X):
        n, k, f = X.shape
        flat = X.reshape(-1, f).astype(float)
        out = self.booster.predict(np.nan_to_num(flat))
        return out.reshape(n, k)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        path.with_suffix('.json').write_text(json.dumps(
            {'features': self.names, 'horizon': self.horizon, 'params': PARAMS, 'num_rounds': NUM_ROUNDS},
            indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        return cls(lgb.Booster(model_file=str(path)), meta['features'], meta['horizon'])


def train(ds, train_mask, horizon):
    """Fit on rows where `train_mask` is true and the label is complete."""
    X, y = _rows(ds, train_mask)
    booster = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=ds.names), num_boost_round=NUM_ROUNDS)
    return AlphaModel(booster, ds.names, horizon)


def purged_mask(ds, start, end, horizon):
    """Rounds in [start, end) whose labels finish before `end`."""
    when = ds.when
    mask = (when >= pd.Timestamp(start)) & (when < pd.Timestamp(end))
    idx = np.where(mask)[0]
    if len(idx) > horizon:
        mask[idx[-horizon:]] = False
    return mask


def information_coefficient(scores, forward):
    """Per-round Spearman correlation between scores and forward relative returns."""
    out = []
    for s, f in zip(scores, forward):
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() >= 10:
            out.append(spearmanr(s[ok], f[ok]).statistic)
        else:
            out.append(np.nan)
    return np.array(out)


FOLDS = [
    ('A', '2021-01-01', '2023-01-01', '2023-07-01'),
    ('B', '2021-01-01', '2023-07-01', '2024-01-01'),
    ('C', '2021-01-01', '2024-01-01', '2024-07-01'),
    ('D', '2021-01-01', '2024-07-01', '2025-01-01'),
    ('holdout', '2021-01-01', '2025-01-01', '2026-01-01'),
]


def walk_forward(ds, horizon, folds=FOLDS):
    """Train one model per fold; return fold models and out-of-sample IC summaries."""
    models, summary = {}, []
    for name, train_start, train_end, test_end in folds:
        model = train(ds, purged_mask(ds, train_start, train_end, horizon), horizon)
        test = (ds.when >= pd.Timestamp(train_end)) & (ds.when < pd.Timestamp(test_end))
        scores = model.predict_rounds(ds.X[test])
        ic = information_coefficient(scores, ds.forward[test])
        summary.append({'fold': name, 'train_end': train_end, 'test_end': test_end,
                        'rounds': int(test.sum()), 'mean_ic': float(np.nanmean(ic)),
                        'ic_t': float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12) * np.sqrt(np.isfinite(ic).sum() / horizon)),
                        'hit_rate': float(np.nanmean(ic > 0))})
        models[name] = model
    return models, summary


def baseline_ic(ds, test_mask):
    """IC of simple single-feature signals on the same rounds, for comparison."""
    col = {n: i for i, n in enumerate(ds.names)}
    out = {}
    for label, name, sign in [('momentum_5d', 'rel_r35', 1), ('momentum_20d', 'rel_r140', 1),
                              ('reversal_1d', 'rel_r7', -1), ('low_vol', 'vol35', -1)]:
        scores = sign * ds.X[test_mask][:, :, col[name]]
        out[label] = float(np.nanmean(information_coefficient(scores, ds.forward[test_mask])))
    return out


def feature_importance(model, top=12):
    gain = model.booster.feature_importance('gain')
    order = np.argsort(gain)[::-1][:top]
    total = gain.sum() or 1
    return {model.names[i]: round(float(gain[i] / total), 3) for i in order}


__all__ = ['AlphaModel', 'Dataset', 'build_dataset', 'train', 'walk_forward', 'SYMBOLS']
