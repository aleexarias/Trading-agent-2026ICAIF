"""Decisions must not depend on information after the decision cutoff."""

import numpy as np
import pandas as pd

from bot.agent import Agent, AgentConfig, AllocationConfig, View
from bot.backtest import Market, precompute_targets
from bot.features import compute_features
from bot.market import build_rounds, panel_from_long


def test_features_are_truncation_invariant(panel):
    _, full, _ = compute_features(panel)
    for t in (300, 777, len(panel.index) - 2):
        _, part, _ = compute_features(panel.head(t + 1))
        np.testing.assert_array_equal(np.nan_to_num(part[t], nan=-9), np.nan_to_num(full[t], nan=-9))


def test_targets_ignore_future_bars(bars):
    cutoff = pd.Timestamp('2026-08-03 11:43')
    future = bars.timestamp_et > cutoff
    shocked = bars.copy()
    shocked.loc[future, ['open', 'high', 'low', 'close']] *= np.random.default_rng(1).uniform(0.5, 2.0, future.sum())[:, None]
    agent_cfg = AgentConfig(allocation=AllocationConfig(construction='min_variance', vol_target=0.1, exposure=0.8))
    out = []
    for frame in (bars, shocked):
        panel = panel_from_long(frame)
        targets, _ = Agent(agent_cfg).targets(View(panel.upto(cutoff), cutoff), online_state={})
        out.append(targets)
    for name in out[0]:
        np.testing.assert_allclose(out[0][name], out[1][name])


def test_backtest_targets_ignore_future_bars(bars):
    cutoff_day = pd.Timestamp('2026-08-03')
    shocked = bars.copy()
    future = shocked.timestamp_et >= cutoff_day
    shocked.loc[future, 'close'] *= 3.0
    cfg = AgentConfig(allocation=AllocationConfig(construction='risk_parity', vol_target=0.1))
    results = []
    for frame in (bars, shocked):
        panel = panel_from_long(frame)
        market = Market.build(panel, build_rounds(panel))
        targets, _ = precompute_targets(Agent(cfg), market, start='2026-07-20')
        idx = [i for i, r in enumerate(market.rounds) if r.opens_at < cutoff_day and r.day >= pd.Timestamp('2026-07-20')]
        results.append(targets['base'][idx])
    np.testing.assert_allclose(results[0], results[1])
