"""Competition-faithful backtester.

Rules reproduced from docs/rules.md and docs/evaluation.md: USD 1,000,000 start,
long-only target weights, 0.1% fee on buy plus sell notional, seven rounds per
day executed at 09:30-15:30 ET, missing decisions hold, positions carry
overnight, period returns between consecutive executions, maximum drawdown over
period endpoints and daily closes, and turnover as mean notional / pre-trade NAV.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.agent import View
from bot.features import compute_features, cutoff_positions
from bot.market import SYMBOLS, execution_prices

FEE = 0.001
INITIAL_NAV = 1_000_000.0


@dataclass
class Market:
    """Precomputed arrays for a panel and its rounds."""

    panel: object
    rounds: list
    exec_prices: np.ndarray
    cutoff_close: np.ndarray
    cutoff_pos: np.ndarray
    day_of_round: np.ndarray
    daily_close: pd.DataFrame
    session_close: pd.Series
    feature_names: list
    features: np.ndarray

    @classmethod
    def build(cls, panel, rounds):
        pos = cutoff_positions(panel, [r.opens_at for r in rounds])
        names, values, _ = compute_features(panel)
        return cls(panel=panel, rounds=rounds,
                   exec_prices=execution_prices(panel, [r.execution for r in rounds]).values,
                   cutoff_close=panel.close.values[pos], cutoff_pos=pos,
                   day_of_round=np.array([r.day for r in rounds], dtype='datetime64[ns]'),
                   daily_close=panel.daily_close(), session_close=panel.session_close(),
                   feature_names=names, features=values)

    def view(self, i):
        r = self.rounds[i]
        pos = self.cutoff_pos[i]
        return View(panel=_DailyOnly(self.daily_close, r.opens_at, self.panel.close.iloc[pos]),
                    cutoff=r.opens_at, features=self.features[pos], feature_names=self.feature_names)


class _DailyOnly:
    """Minimal panel stand-in exposing what the agent needs from precomputed arrays."""

    def __init__(self, daily_close, cutoff, last_close_row):
        self._daily = daily_close
        self._cutoff = cutoff
        self.close = pd.DataFrame([last_close_row.values], columns=SYMBOLS)

    def daily_close(self):
        return self._daily[self._daily.index <= pd.Timestamp(self._cutoff).normalize()]


def precompute_targets(agent, market, model_scores=None, start=None):
    """Target weights for every round (sequential so the online state evolves).

    Returns ({name: [rounds x tickers]}, list of per-round info dicts).
    """
    n = len(market.rounds)
    out, infos, state = {}, [], {}
    first = 0 if start is None else int(np.searchsorted(market.day_of_round, np.datetime64(pd.Timestamp(start))))
    for i in range(first, n):
        scores = None if model_scores is None else model_scores[i]
        if scores is not None and not np.isfinite(scores).all():
            scores = None
        targets, info = agent.targets(market.view(i), model_scores=scores, online_state=state)
        for name, w in targets.items():
            out.setdefault(name, np.full((n, len(SYMBOLS)), np.nan))[i] = w
        infos.append(info)
    return out, infos


def simulate(market, idx, decide, fee=FEE, initial=INITIAL_NAV):
    """Run one competition window over round indices `idx`.

    `decide(i, current_weights)` returns target weights or None (hold). Returns
    the period list, valuation points, metrics and trade count.
    """
    shares = np.zeros(len(SYMBOLS))
    cash = initial
    periods, points = [], [initial]
    nav_before = None
    trades = 0
    days = market.day_of_round[idx]
    for j, i in enumerate(idx):
        price = market.exec_prices[i]
        nav = cash + shares @ price
        if nav_before is not None:
            periods[-1]['nav_after_period'] = nav
            points.append(nav)
        mark = market.cutoff_close[i]
        held = shares * mark
        total = cash + held.sum()
        current = held / total if total > 0 else np.zeros(len(SYMBOLS))
        target = decide(i, current)
        notional = 0.0
        if target is not None:
            target = np.asarray(target, dtype=float)
            fee_paid = 0.0
            for _ in range(4):
                values = target * (nav - fee_paid)
                trade = values - shares * price
                fee_paid = fee * np.abs(trade).sum()
            notional = float(np.abs(trade).sum())
            shares = values / price
            cash = nav - fee_paid - values.sum()
            trades += notional > 0
        periods.append({'nav_before': nav, 'nav_after_period': None, 'traded_notional': notional})
        nav_before = nav
        last_of_day = j + 1 == len(idx) or days[j + 1] != days[j]
        if last_of_day:
            day = pd.Timestamp(days[j])
            close = market.daily_close.loc[day].values
            points.append(cash + shares @ close)
    periods[-1]['nav_after_period'] = points[-1]
    return {'periods': periods, 'points': points, 'trades': trades,
            'metrics': metrics(periods, points, initial)}


def metrics(periods, points, initial=INITIAL_NAV):
    """Fast float version of kit.evaluation.calculate_metrics (same definitions)."""
    before = np.array([p['nav_before'] for p in periods])
    after = np.array([p['nav_after_period'] for p in periods])
    notional = np.array([p['traded_notional'] for p in periods])
    r = after / before - 1
    sharpe = 0.0
    if len(r) > 1 and r.std(ddof=1) > 0:
        sharpe = 42.0 * r.mean() / r.std(ddof=1)
    v = np.asarray(points)
    peak = np.maximum.accumulate(v)
    return {'cumulative_return': after[-1] / initial - 1, 'sharpe_ratio': sharpe,
            'maximum_drawdown': float(((peak - v) / peak).max()), 'turnover': float((notional / before).mean())}


def windows(market, length=15, stride=5, start=None, end=None):
    """Round-index arrays for rolling windows of `length` trading days."""
    days = pd.DatetimeIndex(np.unique(market.day_of_round))
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days < pd.Timestamp(end)]
    out = []
    for s in range(0, len(days) - length + 1, stride):
        chosen = days[s:s + length]
        mask = (market.day_of_round >= chosen[0].to_datetime64()) & (market.day_of_round <= chosen[-1].to_datetime64())
        out.append((chosen[0], np.where(mask)[0]))
    return out


def band_decider(agent, target_matrix):
    """Decision rule used by the live agent: no-trade bands around the target."""
    def decide(i, current):
        target = target_matrix[i]
        if not np.isfinite(target).all():
            return None
        return agent.decide(target, current)
    return decide


def field_strategies(market, seed=11, randoms=8):
    """Plausible competitor strategies as decide functions, for rank-based scoring."""
    rng = np.random.default_rng(seed)
    names = market.feature_names
    col = {n: k for k, n in enumerate(names)}
    feats = market.features[market.cutoff_pos]
    n = len(SYMBOLS)
    first_round = np.r_[True, market.day_of_round[1:] != market.day_of_round[:-1]]

    def topk(score, k, w):
        out = np.zeros(n)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]
        out[order] = w
        return out

    def every_round(builder):
        return lambda i, cur: builder(i)

    def daily(builder):
        return lambda i, cur: builder(i) if first_round[i] or cur.sum() == 0 else None

    def hold(builder):
        return lambda i, cur: builder(i) if cur.sum() == 0 else None

    ew = np.full(n, 1 / n)
    field = {
        'ew_hold': hold(lambda i: ew),
        'ew_daily': daily(lambda i: ew),
        'half_cash_ew': hold(lambda i: ew / 2),
        'momentum_top5_hourly': every_round(lambda i: topk(feats[i, :, col['rel_r7']], 5, 0.2)),
        'momentum_top5_daily': daily(lambda i: topk(feats[i, :, col['rel_r35']], 5, 0.2)),
        'reversal_top5_hourly': every_round(lambda i: topk(-feats[i, :, col['rel_r3']], 5, 0.2)),
        'low_vol_top10_daily': daily(lambda i: topk(-feats[i, :, col['vol35']], 10, 0.1)),
        'trend_top4_hold': hold(lambda i: topk(feats[i, :, col['rel_r140']], 4, 0.25)),
    }
    for k in range(randoms):
        draws = rng.random((len(market.rounds), n))
        size = int(rng.integers(5, 13))
        field[f'random{size}_daily_{k}'] = daily(lambda i, d=draws, s=size: topk(d[i], s, 1 / s))
    return field


def rank_scores(candidate, field_results):
    """Overall rank score of `candidate` among the field (1 = best), averaged ranks per metric."""
    rows = list(field_results) + [candidate]
    table = pd.DataFrame(rows)
    ranks = pd.DataFrame({
        'cumulative_return': table.cumulative_return.rank(ascending=False),
        'sharpe_ratio': table.sharpe_ratio.rank(ascending=False),
        'maximum_drawdown': table.maximum_drawdown.rank(ascending=True),
        'turnover': table.turnover.rank(ascending=True),
    })
    own = ranks.iloc[-1]
    overall = own.mean()
    position = int((ranks.mean(axis=1) < overall).sum()) + 1
    return {'overall_rank_score': float(overall), 'position': position, 'field_size': len(rows),
            **{f'rank_{k}': float(v) for k, v in own.items()}}


def is_sensible(name):
    """Field members that a careful team might plausibly run (no random or hourly churn)."""
    return not name.startswith('random') and 'hourly' not in name


def evaluate(market, strategies, window_list, field=None, fee=FEE):
    """Simulate every strategy on every window; return per-window rows.

    Rank columns are computed against the full field and, prefixed with
    `sensible_`, against the sensible subset only.
    """
    rows = []
    field = field or {}
    for start, idx in window_list:
        field_metrics = {n: simulate(market, idx, d, fee)['metrics'] for n, d in field.items()}
        sensible = [m for n, m in field_metrics.items() if is_sensible(n)]
        for name, decide in strategies.items():
            res = simulate(market, idx, decide, fee)
            row = {'window': start, 'strategy': name, 'trades': res['trades'], **res['metrics']}
            if field_metrics:
                row.update(rank_scores(res['metrics'], list(field_metrics.values())))
                row.update({f'sensible_{k}': v for k, v in rank_scores(res['metrics'], sensible).items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    if field_metrics:
        table['score'] = 0.5 * ((table.overall_rank_score - 1) / (table.field_size - 1)
                                + (table.sensible_overall_rank_score - 1) / (table.sensible_field_size - 1))
    return table


def summarize(table):
    """Mean metrics and rank statistics per strategy across windows."""
    agg = {'cumulative_return': ['mean', 'median', 'min'], 'sharpe_ratio': ['mean', 'median'],
           'maximum_drawdown': ['mean', 'max'], 'turnover': ['mean'], 'trades': ['mean']}
    if 'overall_rank_score' in table:
        agg.update({'overall_rank_score': ['mean'], 'position': ['mean', 'median'], 'field_size': ['max']})
    out = table.groupby('strategy').agg(agg)
    out.columns = ['_'.join(c) for c in out.columns]
    return out.sort_values(out.columns[-3] if 'overall_rank_score' in table else 'sharpe_ratio_mean')
