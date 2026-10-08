"""Portfolio construction and decision-rule invariants."""

import json

import numpy as np
import pandas as pd
import pytest

from bot import risk
from bot.agent import (Agent, AgentConfig, AllocationConfig, OnlineConfig, OnlineExperts, View, finalize,
                       sector_codes, tilt, zscore)
from bot.market import SYMBOLS
from bot.runner import quantize
from kit.contracts import _weights

N = len(SYMBOLS)


def random_cov(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(N, N)) * rng.uniform(0.05, 0.4, N)
    return a @ a.T / N + np.diag(rng.uniform(0.01, 0.1, N))


@pytest.mark.parametrize('construction', sorted(risk.CONSTRUCTIONS))
@pytest.mark.parametrize('seed', range(5))
def test_constructions_respect_caps(construction, seed):
    groups = sector_codes()
    w = risk.CONSTRUCTIONS[construction](random_cov(seed), 0.8, 0.10, groups, 0.30)
    assert np.isfinite(w).all() and (w >= -1e-12).all()
    assert w.max() <= 0.10 + 1e-9
    assert abs(w.sum() - 0.8) < 1e-6
    for g in np.unique(groups):
        assert w[groups == g].sum() <= 0.30 + 1e-6


@pytest.mark.parametrize('seed', range(20))
def test_quantized_weights_pass_kit_validation(seed):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 0.5, N) * rng.integers(0, 2, N)
    raw[rng.integers(0, N)] = np.nan
    w = quantize(finalize(raw, cap=0.30, sector_cap=0.30))
    _weights(w)
    assert set(w) == set(SYMBOLS)
    assert sum(w.values()) <= 1


def test_decide_holds_inside_bands_and_trades_outside():
    agent = Agent(AgentConfig(allocation=AllocationConfig(band_l1=0.2, band_max=0.05)))
    target = np.full(N, 0.8 / N)
    assert agent.decide(target, target * 1.02) is None
    first = agent.decide(target, np.zeros(N))
    assert first is not None and np.allclose(first, target)
    drifted = target.copy()
    drifted[0] += 0.06
    assert agent.decide(target, drifted) is not None


def test_zscore_and_tilt_are_safe():
    z = zscore([np.nan] * 5 + list(range(N - 5)))
    assert np.isfinite(z).all() and np.abs(z).max() <= 3
    base = np.full(N, 0.8 / N)
    tilted = tilt(base, z, 0.5, 2.0, 0.10, sector_codes(), 0.30)
    assert abs(tilted.sum() - 0.8) < 1e-9 and tilted.max() <= 0.10 + 1e-12


def test_short_history_falls_back_to_equal_weight(panel):
    agent = Agent(AgentConfig(online=OnlineConfig(mode='off')))
    cut = panel.end.iloc[7 * 20]
    targets, info = agent.targets(View(panel.upto(cut), pd.Timestamp(cut)))
    assert info.get('fallback') == 'equal_weight'
    assert np.allclose(targets['base'], targets['base'][0])


def test_vol_target_scales_exposure(panel):
    cfg = AgentConfig(allocation=AllocationConfig(construction='equal_weight', vol_target=0.01, exposure=0.8,
                                                  min_exposure=0.3), online=OnlineConfig(mode='off'))
    cut = panel.end.iloc[-1]
    targets, info = Agent(cfg).targets(View(panel, pd.Timestamp(cut)))
    assert abs(targets['base'].sum() - 0.3) < 1e-9
    assert info['exposure'] == pytest.approx(0.3)


def test_online_experts_state_is_serialisable_and_bounded():
    cfg = OnlineConfig(experts=('a', 'b'), eta=1000, max_step=0.05)
    experts = OnlineExperts(cfg, {})
    prices = np.full(N, 100.0)
    signal = {'a': np.arange(N, dtype=float), 'b': -np.arange(N, dtype=float)}
    experts.update(signal, prices, 't0')
    q = experts.update(signal, prices * (1 + np.arange(N) / 1000), 't1')
    assert abs(sum(q.values()) - 1) < 1e-9
    assert q['a'] > q['b'] and q['a'] - 0.5 <= 0.05 + 1e-9
    json.dumps(experts.state)


def test_unreadable_positions_never_look_like_cash():
    from bot.live import DataError, current_weights
    prices = np.full(N, 100.0)
    w, method = current_weights({'cash': '200000', 'positions': [{'symbol': 'AAPL', 'shares': 8000}]}, prices)
    assert method == 'positions' and w[SYMBOLS.index('AAPL')] == pytest.approx(0.8)
    w, method = current_weights({'cash': '200000', 'positions': [{'name': 'AAPL', 'lots': 1}],
                                 'current_weights': {'AAPL': 0.8}}, prices)
    assert method == 'snapshot' and w.sum() == pytest.approx(0.8)
    with pytest.raises(DataError):
        current_weights({'cash': '200000', 'positions': [{'name': 'AAPL', 'lots': 1}], 'current_weights': {}}, prices)


def test_live_panel_uses_yahoo_bar_ends_mid_day(bars):
    from bot.live import DataError, load_live_panel
    day = pd.Timestamp('2026-08-04')
    upto_first = bars[bars.timestamp_et <= day + pd.Timedelta(hours=9, minutes=30)]
    panel, info = load_live_panel(now=day + pd.Timedelta(hours=10, minutes=45), bars=upto_first)
    assert info['last_bar_end'] == '2026-08-04 10:30:00'
    assert panel.end.iloc[-1] == day + pd.Timedelta(hours=10, minutes=30)
    with pytest.raises(DataError):
        load_live_panel(now=day + pd.Timedelta(hours=12, minutes=45), bars=upto_first)
