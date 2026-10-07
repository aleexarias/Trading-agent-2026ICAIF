"""Market data pipeline: official history plus Yahoo Finance top-up.

Run from the starter-kit directory:
    python -m bot.data            # download (if missing), clean, top up, validate
    python -m bot.data --refresh  # also re-download the Yahoo files

Outputs (all under private/data/, which is git-ignored):
    raw/hourly_market_data_2021_2026.parquet  official dataset 1833, untouched
    official_1h.parquet   cleaned official bars on a complete session grid
    yahoo_1h.parquet      Yahoo hourly bars, last ~730 days (bars start at :30)
    yahoo_1d.parquet      Yahoo daily bars since 2021 with dividends and splits
    data_report.json      summary of every check below

Bar conventions (verified against Yahoo, see data_report.json):
    * Timestamps are naive US/Eastern and mark the START of a bar.
    * Official bars: 09:30-10:00, 10:00-11:00, ..., 15:00-16:00 (clock-hour aligned).
    * Yahoo hourly bars: 09:30-10:30, 10:30-11:30, ..., 15:30-16:00, which line
      up with the competition execution times (09:30, 10:30, ..., 15:30).
    * Prices are split-adjusted and NOT dividend-adjusted in both sources.
    * Spinoffs (GE x2, T): official prices are raw, Yahoo back-adjusts them.
      Use spinoff_factor / back_adjust() before computing returns.
"""

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from kit.config import ROOT, load_environment, load_symbols

ET = ZoneInfo('America/New_York')
DATA_DIR = ROOT / 'private' / 'data'
RAW_ZIP = DATA_DIR / 'hourly_market_data_2021_2026.zip'
RAW_PARQUET = DATA_DIR / 'raw' / 'hourly_market_data_2021_2026.parquet'
OFFICIAL = DATA_DIR / 'official_1h.parquet'
YAHOO_1H = DATA_DIR / 'yahoo_1h.parquet'
YAHOO_1D = DATA_DIR / 'yahoo_1d.parquet'
REPORT = DATA_DIR / 'data_report.json'

DATASET_URL = 'https://hackathon2.deepintomlf.ai/datasets/download/8bfeba71-171b-47a1-93f8-2abf846293be/'
DATASET_SHA256 = '396a36e86d5db7447a645276a299177413a86157ddf38e9c2b71134788051652'

REGULAR_BARS = ['09:30', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00']
# NYSE 13:00 early closes. The 13:00 bar holds the closing cross; later bars are after-hours prints.
EARLY_CLOSES = pd.to_datetime([
    '2021-11-26', '2022-11-25', '2023-07-03', '2023-11-24', '2024-07-03',
    '2024-11-29', '2024-12-24', '2025-07-03', '2025-11-28', '2025-12-24'])
EARLY_CLOSE_BARS = REGULAR_BARS[:5]
# Pre/post value ratio at the spinoff open; the official prices are not adjusted for these.
SPINOFFS = {('GE', '2023-01-04'): 1.281, ('GE', '2024-04-02'): 1.253, ('T', '2022-04-11'): 1.324}


def _download_official():
    """Download the official dataset zip from Codabench into RAW_ZIP."""
    import httpx
    load_environment()
    token = os.environ.get('CODABENCH_TOKEN', '')
    if not token:
        raise SystemExit('Set CODABENCH_TOKEN in .env to download the official dataset.')
    response = httpx.get(DATASET_URL, headers={'Authorization': 'Token ' + token}, timeout=30)
    if response.status_code in (301, 302, 303, 307, 308):
        # Signed storage URL: the personal token must not be forwarded.
        response = httpx.get(response.headers['location'], timeout=300)
    response.raise_for_status()
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    RAW_ZIP.write_bytes(response.content)


def load_raw_official():
    """Return the untouched official dataset, downloading and verifying it if needed."""
    if not RAW_PARQUET.exists():
        if not RAW_ZIP.exists():
            _download_official()
        digest = hashlib.sha256(RAW_ZIP.read_bytes()).hexdigest()
        if digest != DATASET_SHA256:
            raise SystemExit(f'Official dataset checksum mismatch: {digest}')
        with zipfile.ZipFile(RAW_ZIP) as archive:
            archive.extractall(RAW_PARQUET.parent)
    return pd.read_parquet(RAW_PARQUET)


def clean_official(raw):
    """Clean the official bars onto a complete (session bar, ticker) grid.

    After-hours prints on early-close days are dropped. Missing bars are filled
    flat from the previous close with zero volume and flagged in `filled`.
    Returns the cleaned frame and a dict of cleaning statistics.
    """
    df = raw.copy()
    df['bar'] = df.timestamp_et.dt.strftime('%H:%M')
    early = df.trading_date.isin(EARLY_CLOSES)
    keep = (~early & df.bar.isin(REGULAR_BARS)) | (early & df.bar.isin(EARLY_CLOSE_BARS))
    dropped = int((~keep).sum())
    df = df[keep]

    days = sorted(df.trading_date.unique())
    grid = pd.DataFrame([(d, b) for d in days
                         for b in (EARLY_CLOSE_BARS if d in EARLY_CLOSES else REGULAR_BARS)],
                        columns=['trading_date', 'bar'])
    grid = grid.merge(pd.DataFrame({'ticker': sorted(load_symbols())}), how='cross')
    grid['timestamp_et'] = pd.to_datetime(grid.trading_date.dt.strftime('%Y-%m-%d ') + grid.bar)
    cols = ['timestamp_et', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    out = grid.merge(df[cols], on=['timestamp_et', 'ticker'], how='left').sort_values(['ticker', 'timestamp_et'])
    out['filled'] = out.close.isna()
    last = out.groupby('ticker').close.ffill()
    for col in ('open', 'high', 'low', 'close'):
        out[col] = out[col].fillna(last)
    out['volume'] = out.volume.fillna(0).astype('int64')
    sector = raw.drop_duplicates('ticker').set_index('ticker').sector_group
    out['sector'] = out.ticker.map(sector)
    out['spinoff_factor'] = [SPINOFFS.get((t, d.strftime('%Y-%m-%d')), 1.0) if b == '09:30' else 1.0
                             for t, d, b in zip(out.ticker, out.trading_date, out.bar)]
    out = out[['timestamp_et', 'trading_date', 'bar', 'ticker', 'sector', 'open', 'high', 'low',
               'close', 'volume', 'filled', 'spinoff_factor']].reset_index(drop=True)
    if out.close.isna().any():
        raise ValueError('A ticker has no observation before a missing bar')
    stats = {'rows_raw': len(raw), 'after_hours_rows_dropped': dropped, 'rows_clean': len(out),
             'bars_forward_filled': int(out.filled.sum()),
             'days_with_filled_bars': sorted({d.strftime('%Y-%m-%d') for d in out[out.filled].trading_date}),
             'trading_days': len(days), 'first_bar': str(out.timestamp_et.min()),
             'last_bar': str(out.timestamp_et.max())}
    return out, stats


def back_adjust(bars):
    """Return a copy with prices before each spinoff divided by its factor.

    Produces continuous series comparable to Yahoo's back-adjusted prices.
    """
    out = bars.sort_values(['ticker', 'timestamp_et']).copy()
    # Cumulative product of all LATER factors for each row.
    later = out.groupby('ticker').spinoff_factor.transform(lambda f: f[::-1].cumprod()[::-1].shift(-1, fill_value=1.0))
    for col in ('open', 'high', 'low', 'close'):
        out[col] = out[col] / later
    return out


def _flatten_yahoo(frame, symbols):
    """Convert a multi-ticker yfinance frame to long format with a `ticker` column."""
    rows = []
    for symbol in symbols:
        part = frame.xs(symbol, axis=1, level=1).copy()
        part['ticker'] = symbol
        rows.append(part)
    out = pd.concat(rows).reset_index()
    out.columns = [str(c).lower().replace(' ', '_') for c in out.columns]
    return out.rename(columns={'datetime': 'timestamp_et', 'date': 'timestamp_et'})


def download_yahoo():
    """Download Yahoo hourly bars (~730 days) and daily bars since 2021.

    Bars that have not finished yet are dropped. Returns (hourly, daily).
    """
    import yfinance as yf
    symbols = sorted(load_symbols())
    now = datetime.now(ET)
    hourly = yf.download(symbols, period='729d', interval='1h', auto_adjust=False,
                         prepost=False, progress=False, threads=True)
    h = _flatten_yahoo(hourly, symbols)
    h['timestamp_et'] = pd.to_datetime(h.timestamp_et, utc=True).dt.tz_convert(ET)
    # A bar ends at the next half hour, or at 16:00 for the 15:30 bar.
    end = (h.timestamp_et + pd.Timedelta(hours=1)).where(h.timestamp_et.dt.strftime('%H:%M') != '15:30',
                                                          h.timestamp_et + pd.Timedelta(minutes=30))
    h = h[end <= now].dropna(subset=['close'])
    h['timestamp_et'] = h.timestamp_et.dt.tz_localize(None)
    h['bar'] = h.timestamp_et.dt.strftime('%H:%M')
    h['trading_date'] = h.timestamp_et.dt.normalize()
    h = h[['timestamp_et', 'trading_date', 'bar', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

    daily = yf.download(symbols, start='2021-01-01', interval='1d', auto_adjust=False,
                        actions=True, progress=False, threads=True)
    d = _flatten_yahoo(daily, symbols).dropna(subset=['close'])
    d['timestamp_et'] = pd.to_datetime(d.timestamp_et).dt.tz_localize(None)
    d = d.rename(columns={'timestamp_et': 'trading_date'})
    # Today's daily bar is still forming before the 16:00 close.
    if now.hour < 16:
        d = d[d.trading_date < pd.Timestamp(now.date())]
    return h.sort_values(['ticker', 'timestamp_et']).reset_index(drop=True), \
        d.sort_values(['ticker', 'trading_date']).reset_index(drop=True)


def _daily_from_bars(bars):
    """Aggregate intraday bars to daily OHLCV indexed by (ticker, trading_date)."""
    g = bars.sort_values('timestamp_et').groupby(['ticker', 'trading_date'])
    return pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(),
                         'close': g.close.last(), 'volume': g.volume.sum()})


def cross_check(official, yahoo_1h, yahoo_1d):
    """Compare daily OHLC between sources.

    Each `<col>_diff_pct` is [median, 99th percentile, max] absolute difference in percent.
    """
    def compare(a, b, label):
        joined = a.join(b, how='inner', lsuffix='_a', rsuffix='_b')
        result = {'days_compared': int(joined.index.get_level_values(1).nunique())}
        for col in ('open', 'high', 'low', 'close'):
            diff = (joined[col + '_a'] / joined[col + '_b'] - 1).abs() * 100
            result[col + '_diff_pct'] = [round(diff.median(), 4), round(diff.quantile(.99), 4), round(diff.max(), 4)]
        close = (joined.close_a / joined.close_b - 1).abs() * 100
        result['ticker_days_close_diff_over_0.5pct'] = int((close > .5).sum())
        result['volume_ratio_median'] = round((joined.volume_a / joined.volume_b).median(), 3)
        return {label: result}

    off = _daily_from_bars(back_adjust(official)[lambda x: ~x.filled])
    y1d = yahoo_1d.set_index(['ticker', 'trading_date'])[['open', 'high', 'low', 'close', 'volume']]
    y1h = _daily_from_bars(yahoo_1h)
    out = {}
    out.update(compare(off, y1d, 'official_vs_yahoo_daily'))
    out.update(compare(y1h, y1d, 'yahoo_hourly_vs_yahoo_daily'))
    out.update(compare(off, y1h, 'official_vs_yahoo_hourly'))
    return out


def yahoo_stats(yahoo_1h, yahoo_1d):
    """Summarise coverage, gaps and corporate actions in the Yahoo data."""
    bars = yahoo_1h.groupby(['trading_date', 'ticker']).size()
    per_day = bars.groupby(level=0).agg(['min', 'max', 'count'])
    actions = yahoo_1d[(yahoo_1d.get('stock_splits', 0) != 0)][['ticker', 'trading_date', 'stock_splits']]
    return {
        'hourly_first_bar': str(yahoo_1h.timestamp_et.min()), 'hourly_last_bar': str(yahoo_1h.timestamp_et.max()),
        'hourly_bar_times': sorted(yahoo_1h.bar.unique()),
        'hourly_days': int(per_day.shape[0]),
        'hourly_days_missing_tickers': [d.strftime('%Y-%m-%d') for d in per_day[per_day['count'] < 30].index],
        'hourly_days_with_short_bars': [d.strftime('%Y-%m-%d') for d in per_day[per_day['min'] < 7].index],
        'daily_first': str(yahoo_1d.trading_date.min().date()), 'daily_last': str(yahoo_1d.trading_date.max().date()),
        'splits_since_2021': [f'{r.ticker} {r.trading_date.date()} x{r.stock_splits:g}' for r in actions.itertuples()],
    }


def main(argv=None):
    """Build all data files and write data_report.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refresh', action='store_true', help='Re-download Yahoo data')
    args = parser.parse_args(argv)

    official, report = clean_official(load_raw_official())
    official.to_parquet(OFFICIAL, index=False)
    if args.refresh or not (YAHOO_1H.exists() and YAHOO_1D.exists()):
        yahoo_1h, yahoo_1d = download_yahoo()
        yahoo_1h.to_parquet(YAHOO_1H, index=False)
        yahoo_1d.to_parquet(YAHOO_1D, index=False)
    yahoo_1h, yahoo_1d = pd.read_parquet(YAHOO_1H), pd.read_parquet(YAHOO_1D)
    full = {'generated_at': datetime.now(ET).isoformat(timespec='seconds'), 'official': report,
            'yahoo': yahoo_stats(yahoo_1h, yahoo_1d), 'cross_check': cross_check(official, yahoo_1h, yahoo_1d)}
    REPORT.write_text(json.dumps(full, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(full, indent=2))


if __name__ == '__main__':
    main()
