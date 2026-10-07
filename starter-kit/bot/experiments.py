"""Historical experiments for approaches 1, 2 and 4, plus robustness checks.

Run from the starter-kit directory:
    python -m bot.experiments            # full study, writes private/results/
    python -m bot.experiments --quick    # smaller grid for a smoke test

Every strategy is simulated on rolling 15-trading-day windows (the Official
phase length) and ranked against a field of simple competitor strategies on
the four official metrics. Lower overall rank score is better.
"""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd

from bot import alpha
from bot.agent import Agent, AgentConfig, AllocationConfig, AlphaConfig, OnlineConfig
from bot.backtest import (FEE, Market, band_decider, evaluate, field_strategies, precompute_targets,
                          summarize, windows)
from bot.market import build_rounds, load_official_panel, load_yahoo_panel
from kit.config import ROOT

RESULTS = ROOT / 'private' / 'results'
MODEL_PATH = ROOT / 'bot' / 'models' / 'lgbm.txt'
HORIZON = 7
PERIODS = {'yahoo': ('2024-06-01', None), 'official': ('2021-07-01', None)}
ALPHA_PERIODS = {'yahoo': ('2024-06-01', None), 'official': ('2023-01-01', None)}

_STATE = {}


def _load(name):
    if name not in _STATE:
        panel = load_yahoo_panel() if name == 'yahoo' else load_official_panel()
        market = Market.build(panel, build_rounds(panel))
        _STATE[name] = {'market': market, 'field': field_strategies(market)}
    return _STATE[name]


def oos_scores(market, dataset_name, official_market):
    """Walk-forward out-of-sample model scores per round of `market` (NaN where unavailable)."""
    ds = alpha.build_dataset(official_market.panel, official_market.rounds, HORIZON)
    folds = alpha.FOLDS + [('final', '2021-01-01', '2026-01-01', '2100-01-01')]
    target = alpha.build_dataset(market.panel, market.rounds, HORIZON)
    scores = np.full((len(market.rounds), market.features.shape[1]), np.nan)
    models = {}
    for name, start, end, test_end in folds:
        model = alpha.train(ds, alpha.purged_mask(ds, start, end, HORIZON), HORIZON)
        models[name] = model
        mask = (target.when >= pd.Timestamp(end)) & (target.when < pd.Timestamp(test_end))
        if dataset_name == 'official' and name == 'final':
            continue
        if mask.any():
            scores[mask] = model.predict_rounds(target.X[mask])
    return scores, models


def _run_config(args):
    """Worker: precompute targets for one agent config and evaluate band variants."""
    dataset, config_dict, bands, target_name, start, end, scores_key, fee, *tag = args
    state = _load(dataset)
    market = state['market']
    config = AgentConfig.from_dict(config_dict)
    scores = _STATE.get(scores_key)
    targets, infos = precompute_targets(Agent(config), market, model_scores=scores, start=start)
    window_list = windows(market, start=start, end=end)
    strategies = {}
    for band in bands:
        cfg = AgentConfig.from_dict(config_dict)
        cfg.allocation = replace(cfg.allocation, **band)
        label = json.dumps({'core': _core_label(cfg), 'band': band, 'target': target_name,
                            'alpha': cfg.alpha.strength if target_name == 'alpha' else None,
                            'online': cfg.online.strength if target_name == 'online' else None})
        strategies[label] = band_decider(Agent(cfg), targets[target_name])
    table = evaluate(market, strategies, window_list, state['field'], fee)
    table['dataset'] = dataset
    table['variant'] = tag[0] if tag else ''
    table['fee'] = fee
    expert = [i.get('expert_weights') for i in infos if i.get('expert_weights')]
    return table, (expert[-1] if expert else None)


def _core_label(cfg):
    a = cfg.allocation
    return f'{a.construction}|cap{a.cap}|vt{a.vol_target}|ex{a.exposure}|lb{a.lookback_days}|blend{a.blend_equal}'


def run_parallel(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_config, jobs))
    tables = pd.concat([t for t, _ in results], ignore_index=True)
    return tables, [e for _, e in results]


def field_summary(dataset, start, end):
    """Average rank of each field member against the rest of the field."""
    state = _load(dataset)
    market, field = state['market'], state['field']
    rows = []
    from bot.backtest import rank_scores, simulate
    for wstart, idx in windows(market, start=start, end=end):
        runs = {n: simulate(market, idx, d) for n, d in field.items()}
        for n, run in runs.items():
            others = [v['metrics'] for k, v in runs.items() if k != n]
            rows.append({'window': wstart, 'strategy': n, 'trades': run['trades'], **run['metrics'],
                         **rank_scores(run['metrics'], others)})
    return pd.DataFrame(rows)


def stage_core(quick, workers):
    constructions = ['min_variance', 'risk_parity', 'inverse_volatility', 'equal_weight']
    caps = [0.10]
    vol_targets = [None] if quick else [None, 0.10]
    exposures = [1.0] if quick else [0.6, 0.8, 1.0]
    bands = [{'band_l1': 0.10, 'band_max': 0.03}] if quick else [
        {'band_l1': 0.05, 'band_max': 0.02}, {'band_l1': 0.10, 'band_max': 0.03},
        {'band_l1': 0.20, 'band_max': 0.05}, {'band_l1': 0.40, 'band_max': 0.10}]
    jobs = []
    for dataset, (start, end) in PERIODS.items():
        for construction, cap, vt, ex in itertools.product(constructions, caps, vol_targets, exposures):
            cfg = AgentConfig(allocation=AllocationConfig(construction=construction, cap=cap, vol_target=vt,
                                                          exposure=ex),
                              online=OnlineConfig(mode='off'))
            jobs.append((dataset, cfg.to_dict(), bands, 'base', start, end, None, FEE))
    table, _ = run_parallel(jobs, workers)
    return table


def choose_core(table):
    """Rank core configs by mean finishing position (normalised by field size), both fields and datasets."""
    table = table.assign(score=(table.position - 1) / (table.field_size - 1) / 2
                         + (table.sensible_position - 1) / (table.sensible_field_size - 1) / 2)
    means = table.groupby(['strategy', 'dataset']).score.mean().unstack()
    means['both'] = means.mean(axis=1)
    best = means.sort_values('both').index[0]
    return json.loads(best), means.sort_values('both')


def config_from_label(label):
    construction, cap, vt, ex, lb, blend = label['core'].split('|')
    allocation = AllocationConfig(construction=construction, cap=float(cap[3:]),
                                  vol_target=None if vt == 'vtNone' else float(vt[2:]), exposure=float(ex[2:]),
                                  lookback_days=int(lb[2:]), blend_equal=float(blend[5:]), **label['band'])
    return AgentConfig(allocation=allocation)


def stage_overlays(base_cfg, workers, quick):
    """Approaches 2 and 4 against the chosen core on periods with out-of-sample scores."""
    strengths = [0.3] if quick else [0.15, 0.3, 0.6]
    jobs = []
    for dataset, (start, end) in ALPHA_PERIODS.items():
        key = f'scores_{dataset}'
        band = [{'band_l1': base_cfg.allocation.band_l1, 'band_max': base_cfg.allocation.band_max},
                {'band_l1': max(0.2, 2 * base_cfg.allocation.band_l1), 'band_max': max(0.05, 2 * base_cfg.allocation.band_max)}]
        base = replace(base_cfg, online=OnlineConfig(mode='off'), alpha=AlphaConfig(mode='off'))
        jobs.append((dataset, base.to_dict(), band, 'base', start, end, key, FEE))
        for s in strengths:
            cfg = replace(base_cfg, alpha=AlphaConfig(mode='active', strength=s), online=OnlineConfig(mode='off'))
            jobs.append((dataset, cfg.to_dict(), band, 'alpha', start, end, key, FEE))
            cfg = replace(base_cfg, alpha=AlphaConfig(mode='off'), online=OnlineConfig(mode='active', strength=s))
            jobs.append((dataset, cfg.to_dict(), band, 'online', start, end, key, FEE))
    return run_parallel(jobs, workers)


def stage_robustness(base_cfg, workers):
    """Perturb fees, bands, lookback and covariance for the chosen core."""
    jobs = []
    a = base_cfg.allocation
    variants = {
        'chosen': {}, 'lookback_63': {'lookback_days': 63}, 'lookback_252': {'lookback_days': 252},
        'cap_x0.75': {'cap': round(a.cap * 0.75, 3)}, 'cap_x1.5': {'cap': round(min(a.cap * 1.5, 0.3), 3)},
        'cov_ewma': {'cov_method': 'ewma'}, 'blend_equal_0.3': {'blend_equal': 0.3},
        'exposure_-0.1': {'exposure': round(max(a.exposure - 0.1, 0.1), 2)},
        'exposure_+0.1': {'exposure': round(min(a.exposure + 0.1, 1.0), 2)},
        'band_x0.5': {'band_l1': a.band_l1 / 2, 'band_max': a.band_max / 2},
        'band_x2': {'band_l1': a.band_l1 * 2, 'band_max': a.band_max * 2},
    }
    for dataset, (start, end) in PERIODS.items():
        for name, change in variants.items():
            cfg = replace(base_cfg, allocation=replace(a, **change), online=OnlineConfig(mode='off'))
            for fee in (FEE, 2 * FEE):
                band = [{'band_l1': cfg.allocation.band_l1, 'band_max': cfg.allocation.band_max}]
                jobs.append((dataset, cfg.to_dict(), band, 'base', start, end, None, fee, name))
    table, _ = run_parallel(jobs, workers)
    return table


def yearly(table, strategy):
    t = table[table.strategy == strategy].copy()
    t['year'] = pd.to_datetime(t.window).dt.year
    return t.groupby(['dataset', 'year']).agg(windows=('window', 'size'), score=('score', 'mean'),
                                              position=('position', 'mean'), ret=('cumulative_return', 'mean'),
                                              sharpe=('sharpe_ratio', 'mean'), mdd=('maximum_drawdown', 'mean'),
                                              turnover=('turnover', 'mean')).round(4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--core', help='JSON core label to use instead of the grid winner; reuses core_windows.csv')
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for name in PERIODS:
        _load(name)
    official = _STATE['official']['market']
    for name in PERIODS:
        scores, models = oos_scores(_STATE[name]['market'], name, official)
        _STATE[f'scores_{name}'] = scores
    models['final'].save(MODEL_PATH)

    report = {}
    if args.core:
        core_table = pd.read_csv(RESULTS / 'core_windows.csv')
    else:
        core_table = stage_core(args.quick, args.workers)
        core_table.to_csv(RESULTS / 'core_windows.csv', index=False)
    best_label, ranking = choose_core(core_table)
    report['core_ranking'] = ranking.round(3).reset_index().to_dict('records')
    report['grid_winner'] = best_label
    if args.core:
        best_label = json.loads(args.core)
    report['core_choice'] = best_label
    base_cfg = config_from_label(best_label)

    fields = {d: field_summary(d, *PERIODS[d]) for d in PERIODS}
    report['field'] = {d: summarize(t).round(4).reset_index().to_dict('records') for d, t in fields.items()}

    overlay_table, expert_weights = stage_overlays(base_cfg, args.workers, args.quick)
    overlay_table.to_csv(RESULTS / 'overlay_windows.csv', index=False)
    report['overlays'] = (overlay_table.groupby(['dataset', 'strategy'])
                          .agg(score=('score', 'mean'), position=('position', 'mean'),
                               sensible_position=('sensible_position', 'mean'),
                               ret=('cumulative_return', 'mean'), sharpe=('sharpe_ratio', 'mean'),
                               mdd=('maximum_drawdown', 'mean'), turnover=('turnover', 'mean'),
                               trades=('trades', 'mean'))
                          .round(4).reset_index().to_dict('records'))
    report['final_expert_weights'] = [w for w in expert_weights if w]

    robust_table = stage_robustness(base_cfg, args.workers)
    robust_table.to_csv(RESULTS / 'robustness_windows.csv', index=False)
    report['robustness'] = (robust_table.groupby(['dataset', 'variant', 'fee'])
                            .agg(score=('score', 'mean'), position=('position', 'mean'),
                                 sensible_position=('sensible_position', 'mean'),
                                 ret=('cumulative_return', 'mean'), sharpe=('sharpe_ratio', 'mean'),
                                 mdd=('maximum_drawdown', 'mean'), turnover=('turnover', 'mean'))
                            .round(4).reset_index().to_dict('records'))
    chosen = core_table[core_table.strategy == json.dumps(best_label)]
    report['chosen_by_year'] = yearly(chosen, json.dumps(best_label)).reset_index().to_dict('records')
    report['chosen_config'] = base_cfg.to_dict()
    (RESULTS / 'experiments.json').write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({'core_choice': best_label, 'top_core': report['core_ranking'][:8]}, indent=2, default=str))


if __name__ == '__main__':
    main()
