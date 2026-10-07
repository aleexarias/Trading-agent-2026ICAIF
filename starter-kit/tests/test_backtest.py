"""Backtest accounting against the kit's official metric calculator."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bot.backtest import FEE, Market, simulate, windows
from bot.market import SYMBOLS, build_rounds
from kit.evaluation import calculate_metrics


@pytest.fixture(scope='module')
def market(panel):
    return Market.build(panel, build_rounds(panel))


def test_rounds_follow_competition_timeline(market):
    r1 = [r for r in market.rounds if r.number == 1][5]
    previous = [r for r in market.rounds if r.execution < r1.execution][-1]
    assert r1.execution.strftime('%H:%M') == '09:30'
    assert previous.execution.strftime('%H:%M') == '15:30'
    assert r1.opens_at == previous.execution + pd.Timedelta(minutes=13)


def test_metrics_match_kit_calculator(market):
    _, idx = windows(market, start='2026-05-01')[0]
    rng = np.random.default_rng(3)

    def decide(i, current):
        if rng.random() < 0.3:
            w = rng.uniform(0, 1, len(SYMBOLS))
            return w / w.sum() * 0.9
        return None

    res = simulate(market, idx, decide)
    kit = calculate_metrics([{k: Decimal(repr(float(v))) for k, v in p.items()} for p in res['periods']],
                            [Decimal(repr(float(v))) for v in res['points']], Decimal('1000000.0'))
    for key, value in kit.items():
        assert res['metrics'][key] == pytest.approx(float(value), rel=1e-9, abs=1e-12)


def test_fee_and_turnover_of_initial_allocation(market):
    _, idx = windows(market, start='2026-05-01')[0]
    target = np.full(len(SYMBOLS), 0.5 / len(SYMBOLS))
    res = simulate(market, idx, lambda i, cur: target if cur.sum() == 0 else None)
    first = res['periods'][0]
    assert first['traded_notional'] == pytest.approx(0.5 * 1e6 * (1 - FEE * 0.5), rel=1e-4)
    assert all(p['traded_notional'] == 0 for p in res['periods'][1:])
    assert res['metrics']['turnover'] == pytest.approx(first['traded_notional'] / 1e6 / len(idx))
    assert res['trades'] == 1


def test_all_cash_is_flat(market):
    _, idx = windows(market, start='2026-05-01')[0]
    res = simulate(market, idx, lambda i, cur: None)
    m = res['metrics']
    assert m['cumulative_return'] == 0 and m['maximum_drawdown'] == 0 and m['turnover'] == 0
