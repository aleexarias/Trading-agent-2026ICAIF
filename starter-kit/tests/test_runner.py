"""End-to-end runner behaviour through the real kit client against a fake platform."""

import json

import pandas as pd
import pytest

from bot.live import DataError
from bot.market import SYMBOLS
from bot.runner import Runner
from kit.contracts import validate_payload

ET = 'America/New_York'


def write_config(path, **allocation):
    allocation = {'construction': 'equal_weight', 'vol_target': 0.10, 'exposure': 0.8,
                  'band_l1': 0.2, 'band_max': 0.05, **allocation}
    path.write_text(json.dumps({'agent': {'allocation': allocation, 'alpha': {'mode': 'off'},
                                          'online': {'mode': 'shadow'}}}), encoding='utf-8')
    return path


def make_runner(platform_env, panel, *, dry_run=False, loader=None, **allocation):
    platform, factory, tmp = platform_env

    def default_loader():
        now = platform.now.tz_convert(ET).tz_localize(None)
        return panel.upto(now), {'source': 'synthetic'}

    return Runner('validation', dry_run=dry_run, config_path=write_config(tmp / 'config.json', **allocation),
                  session_factory=factory, data_loader=loader or default_loader, clock=lambda: platform.now,
                  sleep=lambda s: setattr(platform, 'now', platform.now + pd.Timedelta(seconds=s)),
                  state_dir=tmp / 'state', decisions_dir=tmp / 'decisions', log_path=tmp / 'log.jsonl')


def log_entries(tmp):
    return [json.loads(line) for line in (tmp / 'log.jsonl').read_text().splitlines()]


def test_submits_one_valid_decision_per_round(platform_env, panel):
    platform, _, tmp = platform_env
    runner = make_runner(platform_env, panel)
    result = runner.run(once=True)
    assert result['status'] == 'submitted' and result['round_id'] == 'validation-2026-10-08-r3'
    assert len(platform.uploads) == 1
    payload = platform.uploads[0]
    schedule_path = tmp / 'schedule.json'
    schedule_path.write_text(json.dumps(platform.schedule()))
    validate_payload(payload, 'decision.json', schedule_path=schedule_path,
                     submitted_at=platform.now.isoformat())
    assert set(payload['weights']) == set(SYMBOLS)
    assert 0.5 < sum(payload['weights'].values()) <= 0.8
    assert (tmp / 'decisions' / 'validation' / 'validation-2026-10-08-r3' / 'decision.json').exists()
    assert runner.run(once=True)['status'] == 'WAITING_FOR_ROUND'
    assert len(platform.uploads) == 1


def test_holds_without_upload_inside_bands(platform_env, panel):
    platform, _, tmp = platform_env
    result = make_runner(platform_env, panel, band_l1=10, band_max=10).run(once=True)
    assert result['status'] == 'held' and platform.posts == []
    assert log_entries(tmp)[-1]['status'] == 'held'


def test_data_failure_skips_round_without_upload(platform_env, panel):
    platform, _, _ = platform_env

    def broken():
        raise DataError('stale')

    runner = make_runner(platform_env, panel, loader=broken)
    runner.data_margin = 10 ** 6
    result = runner.run(once=True)
    assert result['status'] == 'skipped' and 'stale' in result['note'] and platform.posts == []


def test_agent_failure_is_contained(platform_env, panel):
    platform, _, _ = platform_env
    result = make_runner(platform_env, panel, loader=lambda: (object(), {})).run(once=True)
    assert result['status'] == 'skipped' and 'agent failure' in result['note'] and platform.posts == []


def test_occupied_slot_is_respected(platform_env, panel):
    platform, _, _ = platform_env
    platform.round_decisions['validation-2026-10-08-r3'] = [{'selection_status': 'SELECTED'}]
    result = make_runner(platform_env, panel).run(once=True)
    assert result['status'] == 'occupied' and platform.posts == []


def test_dry_run_never_uploads(platform_env, panel):
    platform, _, tmp = platform_env
    result = make_runner(platform_env, panel, dry_run=True).run(once=True)
    assert result['status'] == 'dry_run' and result['result']['status'] == 'DRY_RUN_VALID'
    assert platform.posts == []
    assert not (tmp / 'state' / 'runner_validation.json').exists()


def test_existing_decision_file_is_resubmitted_verbatim(platform_env, panel):
    platform, _, tmp = platform_env
    runner = make_runner(platform_env, panel)
    schedule = platform.schedule()
    row = next(r for r in schedule['rounds'] if r['id'].endswith('r3'))
    path = tmp / 'decisions' / 'validation' / row['id'] / 'decision.json'
    weights = [0.02] * len(SYMBOLS)
    payload = runner._payload(row['id'], weights)
    runner._write_decision(path, payload, panel.head(50))
    result = runner.handle(row, schedule)
    assert result['status'] == 'submitted' and len(platform.uploads) == 1
    assert platform.uploads[0]['weights'] == {k: float(v) for k, v in payload['weights'].items()}


def test_ambiguous_upload_is_never_duplicated(platform_env, panel):
    platform, _, _ = platform_env
    platform.fail_next_submission = True
    runner = make_runner(platform_env, panel)
    first = runner.run(once=True)
    assert first['status'] == 'error'
    second = runner.run(once=True)
    assert second['status'] == 'error'
    assert platform.posts.count('submission') == 1 and platform.uploads == []


def test_full_day_loop_trades_once_then_holds(platform_env, panel):
    platform, _, tmp = platform_env
    platform.now = pd.Timestamp('2026-10-07 15:45', tz=ET)
    platform.price_fn = lambda now: panel.upto(now.tz_convert(ET).tz_localize(None)).close.iloc[-1].to_dict()
    runner = make_runner(platform_env, panel)
    result = runner.run()
    assert result['status'] == 'PHASE_COMPLETE'
    statuses = {k: v['status'] for k, v in runner.state['rounds'].items()}
    assert statuses['validation-2026-10-08-r1'] == 'submitted'
    assert len(statuses) == 7 and all(v in ('submitted', 'held') for v in statuses.values())
    assert len(platform.uploads) == sum(v == 'submitted' for v in statuses.values()) <= 2
    entries = log_entries(tmp)
    assert all(e['round_id'].startswith('validation-2026-10-08') for e in entries)
    assert 'TEAM_TOKEN' not in (tmp / 'log.jsonl').read_text() and 'test-team-token' not in (tmp / 'log.jsonl').read_text()


@pytest.mark.parametrize('minutes_after_open', [0, 2])
def test_waits_for_decision_delay(platform_env, panel, minutes_after_open):
    platform, _, _ = platform_env
    platform.now = pd.Timestamp('2026-10-08 10:40', tz=ET) + pd.Timedelta(minutes=minutes_after_open)
    runner = make_runner(platform_env, panel)
    result = runner.run(once=True)
    assert result['status'] == 'submitted'
    assert platform.now >= pd.Timestamp('2026-10-08 10:43', tz=ET)


def test_submitted_decision_replays_exactly(platform_env, panel):
    from bot.replay import replay
    platform, _, tmp = platform_env
    runner = make_runner(platform_env, panel)
    assert runner.run(once=True)['status'] == 'submitted'
    round_dir = tmp / 'decisions' / 'validation' / 'validation-2026-10-08-r3'
    recomputed, uploaded = replay(round_dir, config_path=tmp / 'config.json', log_path=tmp / 'log.jsonl')
    assert recomputed == uploaded


def test_stale_portfolio_after_execution_is_not_traded_on(platform_env, panel):
    platform, _, _ = platform_env
    runner = make_runner(platform_env, panel)
    runner.state['rounds']['validation-2026-10-08-r2'] = {'status': 'submitted'}
    runner.data_margin = 10 ** 6
    result = runner.run(once=True)
    assert result['status'] == 'skipped' and 'portfolio' in result['note'] and platform.posts == []


def test_waits_for_portfolio_to_catch_up(platform_env, panel):
    platform, _, _ = platform_env
    runner = make_runner(platform_env, panel)
    runner.state['rounds']['validation-2026-10-08-r2'] = {'status': 'submitted'}
    advance = runner.sleep

    def sleep(seconds):
        advance(seconds)
        platform.portfolio = dict(platform.portfolio, as_of='2026-10-08T10:30:00-04:00')

    runner.sleep = sleep
    result = runner.run(once=True)
    assert result['status'] == 'submitted'
    assert platform.now > pd.Timestamp('2026-10-08 10:45', tz=ET)
