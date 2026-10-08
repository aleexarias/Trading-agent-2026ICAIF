"""Shared fixtures: synthetic bar panels and a fake Codabench/backend platform."""

import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.market import SYMBOLS, panel_from_long  # noqa: E402
from kit.original_client import OriginalSession, Profile  # noqa: E402

BAR_STARTS = ['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30']


def synthetic_bars(days=220, seed=0, start='2026-01-02'):
    """Random-walk execution-aligned hourly bars for the 30 symbols on weekdays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    stamps = [pd.Timestamp(f'{d.date()} {t}') for d in dates for t in BAR_STARTS]
    rows = []
    for k, symbol in enumerate(SYMBOLS):
        vol = 0.004 + 0.002 * (k % 5)
        rets = rng.normal(0.0001, vol, len(stamps))
        close = 100 * (1 + k / 10) * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, len(stamps)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, len(stamps)))
        volume = rng.integers(1e5, 1e6, len(stamps))
        rows.append(pd.DataFrame({'timestamp_et': stamps, 'ticker': symbol, 'open': open_, 'high': high,
                                  'low': low, 'close': close, 'volume': volume}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope='session')
def bars():
    return synthetic_bars()


@pytest.fixture(scope='session')
def panel(bars):
    return panel_from_long(bars)


PROFILE = Profile(base_url='https://platform.test', competition_id=99, phases={'unified': 120})
TOKEN = 'test-codabench-token'
TEAM_ID = '11111111-2222-4333-8444-555555555555'
TEAM_TOKEN = 'test-team-token-value'


def make_schedule(now, phase='validation', day='2026-10-08'):
    """Seven rounds on one day (ET), with server time `now` (aware datetime)."""
    rows = []
    previous_exec = pd.Timestamp(f'2026-10-07 15:30', tz='America/New_York')
    for n, t in enumerate(['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30'], start=1):
        execution = pd.Timestamp(f'{day} {t}', tz='America/New_York')
        deadline = execution - pd.Timedelta(minutes=20 if n == 1 else 5)
        rows.append({'id': f'{phase}-{day}-r{n}', 'phase': phase, 'day': day, 'number': n,
                     'opens_at': (previous_exec + pd.Timedelta(minutes=10)).isoformat(),
                     'deadline': deadline.isoformat(), 'execution_time': execution.isoformat(),
                     'close_time': pd.Timestamp(f'{day} 16:00', tz='America/New_York').isoformat(),
                     'status': 'SCHEDULED'})
        previous_exec = execution
    return {'timezone': 'America/New_York', 'current_time': pd.Timestamp(now).isoformat(), 'rounds': rows,
            'symbols': SYMBOLS, 'universe_confirmed': True, 'final_deadline': '2026-11-03T23:59:00-05:00'}


class FakePlatform:
    """In-memory Codabench + ICAIF backend implementing the routes OriginalSession uses."""

    def __init__(self, now, price_fn=None):
        self.now = pd.Timestamp(now)
        self.price_fn = price_fn
        self.portfolio = {'cash': '1000000', 'nav': '1000000', 'positions': [], 'current_weights': {},
                          'as_of': None}
        self.round_decisions = {}
        self.uploads = []
        self.submissions = {}
        self.posts = []
        self.fail_next_submission = False
        self._stored = {}

    def schedule(self):
        return make_schedule(self.now)

    def handler(self, request):
        url = urlsplit(str(request.url))
        path, method = url.path, request.method
        backend = '/extensions/icaif2026/99/backend'
        if method == 'GET' and path == '/api/my_profile/':
            return httpx.Response(200, json={'username': 'tester'})
        if method == 'GET' and path == '/api/competitions/99/':
            return httpx.Response(200, json={'id': 99, 'participant_status': 'approved', 'phases': [{'id': 120}]})
        if method == 'GET' and path == backend + '/api/v1/schedule':
            return httpx.Response(200, json=self.schedule())
        if method == 'GET' and path == backend + '/api/v1/me/portfolio':
            assert request.headers.get('X-ICAIF-Team-Token') == TEAM_TOKEN
            return httpx.Response(200, json=self.portfolio)
        if method == 'GET' and path.startswith(backend + '/api/v1/me/rounds/'):
            rid = path.rsplit('/', 1)[1]
            return httpx.Response(200, json={'round_id': rid, 'decisions': self.round_decisions.get(rid, [])})
        if method == 'POST' and path == '/api/datasets/':
            self.posts.append('dataset')
            key = f'key{len(self.posts)}'
            return httpx.Response(200, json={'key': key,
                                             'sassy_url': f'https://storage.test/{key}?content-type=application/json'})
        if method == 'PUT' and url.hostname == 'storage.test':
            self._stored[path.strip('/')] = request.content
            return httpx.Response(200)
        if method == 'PUT' and path.startswith('/api/datasets/completed/'):
            return httpx.Response(200, json={})
        if method == 'POST' and path == '/api/submissions/':
            self.posts.append('submission')
            if self.fail_next_submission:
                self.fail_next_submission = False
                return httpx.Response(500)
            body = json.loads(request.content)
            raw = self._stored[body['data']]
            sid = 500 + len(self.submissions) + 1
            payload = json.loads(raw)
            self.submissions[sid] = raw
            self.uploads.append(payload)
            self.round_decisions.setdefault(payload['round_id'], []).append(
                {'submission_id': sid, 'selection_status': 'SELECTED', 'validation_status': 'VALID'})
            if self.price_fn is not None:
                prices = self.price_fn(self.now)
                weights = {k: float(v) for k, v in payload['weights'].items()}
                self.portfolio = {'nav': '1000000', 'cash': str(1e6 * (1 - sum(weights.values()))),
                                  'positions': [{'symbol': s, 'shares': w * 1e6 / prices[s]}
                                                for s, w in weights.items() if w > 0],
                                  'current_weights': weights,
                                  'as_of': next(r['execution_time'] for r in self.schedule()['rounds']
                                                if r['id'] == payload['round_id'])}
            return httpx.Response(200, json={'id': sid})
        if method == 'GET' and path == '/api/submissions/':
            rows = [{'id': sid, 'owner': 'tester', 'phase': 120, 'parent': None} for sid in self.submissions]
            return httpx.Response(200, json={'results': rows, 'next': None})
        if method == 'GET' and path.startswith('/api/submissions/') and path.count('/') == 4:
            sid = int(path.split('/')[3])
            return httpx.Response(200, json={'id': sid, 'owner': 'tester', 'phase': 120, 'status': 'Finished'})
        if method == 'GET' and path.startswith('/extensions/icaif2026/99/submissions/') and path.endswith('/receipt'):
            sid = int(path.split('/')[5])
            raw = self.submissions[sid]
            return httpx.Response(200, json={'submission_id': sid, 'owner_username': 'tester', 'phase_id': 120,
                                             'file_name': 'decision.json', 'sha256': hashlib.sha256(raw).hexdigest(),
                                             'status': 'VALID', 'result': {'status': 'ACCEPTED', 'submission_id': sid}})
        return httpx.Response(404, json={'path': path, 'method': method})


@pytest.fixture
def platform_env(tmp_path):
    """A FakePlatform plus a session factory bound to private temp state with saved credentials."""
    platform = FakePlatform(pd.Timestamp('2026-10-08 10:45', tz='America/New_York'))
    transport = httpx.MockTransport(platform.handler)
    checkpoint, credentials = tmp_path / 'icaif' / 'checkpoint.json', tmp_path / 'icaif' / 'credentials.json'

    def factory():
        return OriginalSession(profile=PROFILE, token=TOKEN, checkpoint=checkpoint, credentials=credentials,
                               transport=transport, retry_delay=0, poll_interval=0)

    with factory() as session:
        session.save_credentials(TEAM_ID, TEAM_TOKEN)
    return platform, factory, tmp_path
