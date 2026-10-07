"""Wide bar panels, the competition round timeline and execution prices.

A `Panel` holds completed hourly bars for the 30 symbols as wide frames
(rows = bar start timestamps, naive US/Eastern; columns = tickers). The same
structure is built from the official dataset, from cached Yahoo bars and from
live Yahoo downloads, so the agent sees identical inputs in backtest and live.
"""

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from kit.config import load_symbols

SYMBOLS = sorted(load_symbols())
FIELDS = ('open', 'high', 'low', 'close', 'volume')
EXECUTION_TIMES = [time(9, 30), time(10, 30), time(11, 30), time(12, 30), time(13, 30), time(14, 30), time(15, 30)]
WINDOW_DELAY = pd.Timedelta(minutes=10)


@dataclass
class Panel:
    """Completed bars as wide frames plus per-bar end times."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    end: pd.Series

    @property
    def index(self):
        return self.close.index

    def upto(self, cutoff):
        """Return the panel restricted to bars that ended at or before `cutoff`."""
        n = int(np.searchsorted(self.end.values, np.datetime64(cutoff), side='right'))
        return self.head(n)

    def head(self, n):
        return Panel(*(getattr(self, f).iloc[:n] for f in FIELDS), end=self.end.iloc[:n])

    def daily_close(self):
        """Close of the last bar of each trading day contained in the panel."""
        close = self.close
        return close.groupby(close.index.normalize()).last()

    def session_close(self):
        """End time of the last bar of each trading day (16:00, or 13:00 on half days)."""
        return self.end.groupby(self.end.index.normalize()).max()


def bar_end_times(index):
    """Bar end times for clock-aligned (official) or execution-aligned (Yahoo) bars.

    A bar ends where the next bar of the same day starts. The last bar of a day
    ends at the session close: 13:00 if the day's last bar starts before 15:00
    (NYSE early close), else 16:00.
    """
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    days = s.dt.normalize()
    last_start = s.groupby(days.values).transform('max')
    close = days + pd.to_timedelta(np.where(last_start.dt.hour < 15, 13, 16), unit='h')
    nxt = s.shift(-1)
    end = nxt.where(nxt.dt.normalize() == days, np.maximum(close, s))
    return end.astype('datetime64[ns]')


def panel_from_long(bars):
    """Build a Panel from long bars with timestamp_et, ticker and OHLCV columns.

    Leading bars before every symbol has a price are dropped; later gaps are
    filled flat from the previous close with zero volume.
    """
    frame = bars[bars.ticker.isin(SYMBOLS)]
    wide = {f: frame.pivot_table(index='timestamp_et', columns='ticker', values=f, aggfunc='last')
            .reindex(columns=SYMBOLS).sort_index() for f in FIELDS}
    first_complete = wide['close'].notna().all(axis=1).idxmax()
    wide = {f: w.loc[first_complete:] for f, w in wide.items()}
    close = wide['close'].ffill()
    for f in ('open', 'high', 'low'):
        wide[f] = wide[f].fillna(close)
    wide['close'] = close
    wide['volume'] = wide['volume'].fillna(0.0).astype(float)
    return Panel(**wide, end=bar_end_times(wide['close'].index))


def load_official_panel():
    """Official 2021-2025 bars, spinoff back-adjusted."""
    from bot.data import OFFICIAL, back_adjust
    return panel_from_long(back_adjust(pd.read_parquet(OFFICIAL)))


def load_yahoo_panel():
    """Cached Yahoo hourly bars (execution aligned)."""
    from bot.data import YAHOO_1H
    return panel_from_long(pd.read_parquet(YAHOO_1H))


@dataclass(frozen=True)
class Round:
    """One decision round: info cutoff, execution time and its trading day."""

    day: pd.Timestamp
    number: int
    opens_at: pd.Timestamp
    execution: pd.Timestamp


def build_rounds(panel, decision_delay=pd.Timedelta(minutes=3)):
    """Competition rounds implied by the panel's trading days.

    Rounds whose execution time is at or after the session close are cancelled
    (early-close days). Round 1 opens 10 minutes after the previous trading
    day's last executed round; later rounds open 10 minutes after the previous
    execution. `opens_at` includes the runner's decision delay.
    """
    closes = panel.session_close()
    rounds, previous = [], None
    for day, close in closes.items():
        for number, t in enumerate(EXECUTION_TIMES, start=1):
            execution = day + pd.Timedelta(hours=t.hour, minutes=t.minute)
            if execution >= close:
                continue
            if previous is not None:
                rounds.append(Round(day, number, previous + WINDOW_DELAY + decision_delay, execution))
            previous = execution
    return rounds


def execution_prices(panel, times):
    """Prices at each execution time: the open of a bar starting there, else the bar midpoint.

    Midpoints are only used for clock-aligned official bars, where 10:30-15:30
    executions fall inside a bar.
    """
    starts = panel.index
    pos = np.searchsorted(starts.values, pd.DatetimeIndex(times).values, side='right') - 1
    out = []
    for t, p in zip(times, pos):
        if starts[p] == t:
            out.append(panel.open.iloc[p].values)
        else:
            out.append(((panel.open.iloc[p] + panel.close.iloc[p]) / 2).values)
    return pd.DataFrame(np.vstack(out), index=pd.DatetimeIndex(times), columns=SYMBOLS)
