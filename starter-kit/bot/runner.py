"""Live competition runner: one decision per round, uploaded through the kit client.

Run from the starter-kit directory:
    python -m bot.runner --phase validation              # trade every Validation round
    python -m bot.runner --phase official                # trade every Official round
    python -m bot.runner --phase validation --dry-run    # decide and validate, never upload
    python -m bot.runner --phase validation --dry-run --simulate-round validation-2026-10-08-r1

Differences from `tools/auto_submit.py watch`: a round is skipped (holdings
kept, zero turnover) when no rebalance is needed or data is unusable; agent
failures never stop the loop; the kit checkpoint lock is only held while
talking to the platform; every decision is logged with its inputs.

Uploads always go through `OriginalSession.decision`, so the kit's schedule,
payload, credential and occupied-slot checks and its crash recovery apply.
A decision file is written before upload and reused verbatim on restart, so an
interrupted upload is resumed instead of replaced.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from bot.agent import Agent, AgentConfig, View
from bot.alpha import AlphaModel
from bot.live import DataError, current_weights, load_live_panel
from bot.market import SYMBOLS, build_rounds
from kit.config import ROOT, load_environment
from kit.contracts import dumps_json, timestamp, validate_payload
from kit.original_client import AutomationError, OriginalSession, load_profile, write_private_json

CONFIG_PATH = ROOT / 'bot' / 'production.json'
STATE_DIR = ROOT / 'private' / 'state'
DECISIONS_DIR = ROOT / 'private' / 'decisions'
LOG_PATH = ROOT / 'private' / 'logs' / 'decisions.jsonl'
DONE = {'submitted', 'held', 'occupied', 'skipped', 'missed', 'dry_run'}

log = logging.getLogger('bot.runner')


def quantize(weights, total_cap=Decimal(1), cap=Decimal('0.30')):
    """Round weights down to 6 decimals as exact Decimals within the competition limits."""
    out = {}
    for symbol, w in zip(SYMBOLS, weights):
        value = Decimal(repr(float(max(w, 0.0)))).quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
        out[symbol] = min(value, cap)
    while sum(out.values()) > total_cap:
        largest = max(out, key=out.get)
        out[largest] -= Decimal('0.000001')
    return out


def code_version():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'bot'], cwd=ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


class Runner:
    """Stateful loop around the kit client for one phase."""

    def __init__(self, phase, *, dry_run=False, config_path=CONFIG_PATH, session_factory=None,
                 data_loader=load_live_panel, clock=None, sleep=time.sleep, state_dir=STATE_DIR,
                 decisions_dir=DECISIONS_DIR, log_path=LOG_PATH, decision_delay=180, data_margin=600,
                 retry_seconds=120, poll_seconds=300):
        self.phase, self.dry_run = phase, dry_run
        self.config_raw = json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.config = AgentConfig.from_dict(self.config_raw['agent'])
        model_path = ROOT / self.config.alpha.model_path
        self.model = AlphaModel.load(model_path) if model_path.exists() else None
        self.agent = Agent(self.config, self.model)
        self.session_factory = session_factory or default_session
        self.data_loader, self.sleep = data_loader, sleep
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.state_path = Path(state_dir) / f'runner_{phase}.json'
        self.decisions_dir, self.log_path = Path(decisions_dir), Path(log_path)
        self.decision_delay, self.data_margin = decision_delay, data_margin
        self.retry_seconds, self.poll_seconds = retry_seconds, poll_seconds
        self.state = self._load_state()
        self.version = code_version()

    def _load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding='utf-8'))
        return {'rounds': {}, 'online': {}}

    def _save_state(self):
        if not self.dry_run:
            write_private_json(self.state_path, self.state)

    def _log(self, entry):
        self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        entry = {'logged_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'phase': self.phase,
                 'dry_run': self.dry_run, 'code_version': self.version, **entry}
        with open(self.log_path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, default=str) + '\n')
        os.chmod(self.log_path, 0o600)
        log.info('%s %s %s', entry.get('round_id'), entry.get('status'), entry.get('note', ''))

    def schedule(self):
        with self.session_factory() as session:
            return session.schedule()

    def now(self, schedule=None):
        if schedule and schedule.get('current_time'):
            return timestamp(schedule['current_time'])
        return self.clock()

    def phase_rounds(self, schedule):
        rows = [r for r in schedule['rounds'] if r['phase'] == self.phase and r['status'] != 'CANCELLED']
        return sorted(rows, key=lambda r: timestamp(r['opens_at']))

    def run(self, once=False, max_seconds=None, simulate_round=None):
        """Main loop. Returns a status dict when the phase ends, time runs out, or after one pass."""
        started = time.monotonic()
        while True:
            try:
                schedule = self.schedule()
            except AutomationError as error:
                log.warning('schedule unavailable: %s', error)
                if once:
                    return {'status': 'SCHEDULE_UNAVAILABLE'}
                self.sleep(60)
                continue
            rows = self.phase_rounds(schedule)
            now = self.now(schedule)
            if simulate_round:
                row = next(r for r in rows if r['id'] == simulate_round)
                return self.handle(row, schedule, simulated_now=timestamp(row['opens_at']))
            if rows and now >= max(timestamp(r['close_time']) for r in rows):
                return {'status': 'PHASE_COMPLETE', 'phase': self.phase}
            row = next((r for r in rows if timestamp(r['opens_at']) <= now < timestamp(r['deadline'])), None)
            result = None
            if row and self.state['rounds'].get(row['id'], {}).get('status') not in DONE:
                ready = timestamp(row['opens_at']).timestamp() + self.decision_delay
                if now.timestamp() >= ready:
                    result = self.handle(row, schedule)
                else:
                    self.sleep(ready - now.timestamp())
                    continue
            if once:
                return result or {'status': 'WAITING_FOR_ROUND', 'next_deadline': schedule.get('next_deadline')}
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                return {'status': 'PAUSED'}
            upcoming = [timestamp(r['opens_at']).timestamp() + self.decision_delay for r in rows
                        if timestamp(r['opens_at']) > now]
            wait = min([self.poll_seconds] + [max(1.0, u - now.timestamp()) for u in upcoming[:1]])
            self.sleep(wait)

    def handle(self, row, schedule, simulated_now=None):
        """Decide and (unless dry-run) submit one round. Never raises for agent or data failures."""
        round_id = row['id']
        record = self.state['rounds'].setdefault(round_id, {})
        path = self.decisions_dir / self.phase / round_id / 'decision.json'
        entry = {'round_id': round_id, 'opens_at': row['opens_at'], 'deadline': row['deadline']}
        try:
            if path.exists() and not self.dry_run:
                return self._finish(record, entry, self._submit(path), 'submitted', 'resumed existing decision file')
            with self.session_factory() as session:
                own = session.round(round_id)
                if any(d.get('selection_status') != 'NOT_ELIGIBLE' for d in own.get('decisions', [])):
                    return self._finish(record, entry, None, 'occupied', 'slot already consumed')
            deadline = timestamp(row['deadline'])
            portfolio = self._fresh_portfolio(schedule, deadline, simulated_now)
            if portfolio is None:
                return self._finish(record, entry, None, 'skipped',
                                    'portfolio not updated after the previous execution')
            panel, data_info = self._fetch(deadline, simulated_now)
            if panel is None:
                return self._finish(record, entry, None, 'skipped', data_info)
            entry['data'] = data_info
            weights, decision, info = self._decide(panel, portfolio)
            entry.update(info)
            if decision is None:
                return self._finish(record, entry, None, 'held', 'within no-trade bands')
            payload = self._payload(round_id, decision)
            entry['submitted_weights'] = {k: str(v) for k, v in payload['weights'].items()}
            if self.dry_run:
                check_time = (simulated_now or self.now(schedule)).isoformat()
                self._validate(payload, schedule, check_time)
                return self._finish(record, entry, {'status': 'DRY_RUN_VALID', 'checked_at': check_time},
                                    'dry_run', 'payload validated, not uploaded')
            self._write_decision(path, payload, panel)
            return self._finish(record, entry, self._submit(path), 'submitted')
        except AutomationError as error:
            past = self.now() >= timestamp(row['deadline'])
            status = 'missed' if past else 'error'
            return self._finish(record, entry, None, status, f'platform: {error}')
        except Exception as error:
            log.exception('round %s failed', round_id)
            return self._finish(record, entry, None, 'skipped', f'agent failure: {type(error).__name__}: {error}')

    def _required_as_of(self, schedule):
        """Execution time of the latest submitted round that has already executed, if any."""
        now = self.now(schedule)
        times = {r['id']: timestamp(r['execution_time']) for r in schedule['rounds']}
        done = [times[rid] for rid, rec in self.state['rounds'].items()
                if rec.get('status') == 'submitted' and rid in times and times[rid] <= now]
        return max(done) if done else None

    def _fresh_portfolio(self, schedule, deadline, simulated_now):
        """Portfolio that reflects our last executed decision, or None if it never catches up.

        The backend updates holdings some minutes after the nominal execution
        time, so a decision taken from an older snapshot would rebuy positions
        that are already held.
        """
        required = self._required_as_of(schedule)
        while True:
            with self.session_factory() as session:
                portfolio = session.portfolio(self.phase)
            as_of = portfolio.get('as_of')
            if required is None or (as_of and timestamp(as_of) >= required):
                return portfolio
            remaining = deadline.timestamp() - self.data_margin - self.clock().timestamp()
            if simulated_now is not None or remaining <= 0:
                return None
            log.warning('portfolio as_of %s is older than the execution at %s; retrying', as_of, required)
            self.sleep(min(self.retry_seconds, remaining))

    def _fetch(self, deadline, simulated_now):
        last_error = None
        while True:
            try:
                return self.data_loader()
            except DataError as error:
                last_error = str(error)
            except Exception as error:
                last_error = f'{type(error).__name__}: {error}'
            remaining = deadline.timestamp() - self.data_margin - self.clock().timestamp()
            if simulated_now is not None or remaining <= 0:
                return None, f'no usable data: {last_error}'
            log.warning('data not ready (%s); retrying', last_error)
            self.sleep(min(self.retry_seconds, remaining))

    def _decide(self, panel, portfolio):
        cutoff = panel.end.iloc[-1]
        view = View(panel=panel, cutoff=pd.Timestamp(cutoff))
        targets, info = self.agent.targets(view, online_state=self.state.setdefault('online', {}))
        prices = panel.close.iloc[-1].values
        current, method = current_weights(portfolio, prices)
        active = targets[self.config.active]
        decision = self.agent.decide(active, current)
        rounded = lambda w: {s: round(float(x), 5) for s, x in zip(SYMBOLS, w)}
        details = {'active_target': self.config.active, 'current_weights': rounded(current),
                   'current_weights_method': method, 'targets': {k: rounded(v) for k, v in targets.items()},
                   'agent_info': {k: v for k, v in info.items()}, 'portfolio': portfolio,
                   'deviation_l1': float(np.abs(active - current).sum())}
        return current, decision, details

    def _payload(self, round_id, weights):
        with self.session_factory() as session:
            creds = session._team()
        return {'submission_type': 'decision', 'team_id': creds['team_id'], 'team_token': creds['team_token'],
                'phase': self.phase, 'round_id': round_id, 'weights': quantize(weights)}

    def _validate(self, payload, schedule, submitted_at):
        path = self.state_path.parent / 'dry_run_schedule.json'
        write_private_json(path, schedule)
        validate_payload(payload, 'decision.json', submitted_at=submitted_at, schedule_path=path)

    def _write_decision(self, path, payload, panel):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private_json(path, payload)
        snapshot = pd.concat({f: getattr(panel, f) for f in ('open', 'high', 'low', 'close', 'volume')}, axis=1)
        snapshot.to_parquet(path.parent / 'inputs.parquet')
        os.chmod(path.parent / 'inputs.parquet', 0o600)

    def _submit(self, path):
        with self.session_factory() as session:
            return session.decision(path)

    def _finish(self, record, entry, result, status, note=''):
        if result is not None:
            entry['result'] = result
            status = {'SLOT_CONSUMED': 'occupied', 'MISSED_DEADLINE': 'missed'}.get(result.get('status'), status)
        entry.update(status=status, note=note)
        if status != 'error':
            record.update(status=status, at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        self._save_state()
        safe = {k: v for k, v in entry.items() if k != 'portfolio'}
        safe['portfolio'] = {k: entry['portfolio'].get(k) for k in ('cash', 'nav', 'positions', 'current_weights',
                                                                     'as_of', 'last_completed_round')} \
            if isinstance(entry.get('portfolio'), dict) else None
        self._log(safe)
        return {'round_id': entry['round_id'], 'status': status, 'note': note,
                'result': result, 'deviation_l1': entry.get('deviation_l1')}


def default_session():
    load_environment()
    token = os.environ.get('CODABENCH_TOKEN', '')
    profile = os.environ.get('ICAIF_PROFILE', '')
    if not token or not profile:
        raise AutomationError('Set CODABENCH_TOKEN and ICAIF_PROFILE in .env')
    return OriginalSession(profile=load_profile(ROOT / profile if not Path(profile).is_absolute() else profile),
                           token=token, checkpoint=ROOT / '.icaif' / 'checkpoint.json',
                           credentials=ROOT / '.icaif' / 'credentials.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=('validation', 'official'), required=True)
    parser.add_argument('--dry-run', action='store_true', help='Decide and validate without uploading')
    parser.add_argument('--once', action='store_true', help='Handle at most the current round, then exit')
    parser.add_argument('--simulate-round', help='Dry-run a specific round now as if its window were open')
    parser.add_argument('--max-hours', type=float, help='Stop after this many hours (default: until phase ends)')
    args = parser.parse_args(argv)
    if args.simulate_round and not args.dry_run:
        parser.error('--simulate-round requires --dry-run')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    runner = Runner(args.phase, dry_run=args.dry_run)
    result = runner.run(once=args.once, simulate_round=args.simulate_round,
                        max_seconds=args.max_hours * 3600 if args.max_hours else None)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
