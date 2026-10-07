"""Recompute a submitted live decision from its archived inputs.

Run from the starter-kit directory:
    python -m bot.replay private/decisions/validation/validation-2026-10-08-r1

Loads `inputs.parquet` (the exact bars the runner used) and the portfolio
recorded in the decision log, re-runs the agent with the production config and
compares the result with the uploaded `decision.json`.
"""

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

from bot.agent import Agent, AgentConfig, View
from bot.live import current_weights
from bot.market import FIELDS, Panel, bar_end_times
from bot.runner import CONFIG_PATH, LOG_PATH, quantize


def load_inputs(round_dir):
    """Rebuild the Panel saved next to a decision file."""
    snapshot = pd.read_parquet(Path(round_dir) / 'inputs.parquet')
    frames = {f: snapshot[f] for f in FIELDS}
    return Panel(**frames, end=bar_end_times(frames['close'].index))


def logged_portfolio(round_id, log_path=LOG_PATH):
    """Portfolio recorded for the submitted decision of `round_id`."""
    entries = [json.loads(line) for line in Path(log_path).read_text(encoding='utf-8').splitlines()]
    matches = [e for e in entries if e.get('round_id') == round_id and e.get('status') == 'submitted'
               and e.get('portfolio')]
    if not matches:
        raise LookupError(f'No submitted log entry with a portfolio for {round_id}')
    return matches[-1]['portfolio']


def replay(round_dir, config_path=CONFIG_PATH, log_path=LOG_PATH):
    """Return (recomputed weights, uploaded weights) as {symbol: Decimal} dicts."""
    round_dir = Path(round_dir)
    uploaded = json.loads((round_dir / 'decision.json').read_text(encoding='utf-8'), parse_float=Decimal)
    config = AgentConfig.from_dict(json.loads(Path(config_path).read_text(encoding='utf-8'))['agent'])
    panel = load_inputs(round_dir)
    agent = Agent(config)
    targets, _ = agent.targets(View(panel=panel, cutoff=pd.Timestamp(panel.end.iloc[-1])), online_state={})
    current, _ = current_weights(logged_portfolio(uploaded['round_id'], log_path), panel.close.iloc[-1].values)
    decision = agent.decide(targets[config.active], current)
    recomputed = quantize(decision) if decision is not None else None
    return recomputed, uploaded['weights']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('round_dir')
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--log', default=LOG_PATH)
    args = parser.parse_args(argv)
    recomputed, uploaded = replay(args.round_dir, args.config, args.log)
    match = recomputed == uploaded
    print(json.dumps({'match': match, 'recomputed': recomputed, 'uploaded': uploaded}, indent=2, default=str))
    return 0 if match else 1


if __name__ == '__main__':
    sys.exit(main())
