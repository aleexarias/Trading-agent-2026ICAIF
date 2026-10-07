"""The trading agent: risk-based core (approach 1), alpha tilt (approach 2) and
online expert weighting (approach 4).

Target portfolios are pure functions of market data up to the decision cutoff
(plus the online expert state, which also depends only on market data). Only
`Agent.decide` looks at current holdings, applying no-trade bands. The same
code path serves the backtester and the live runner.
"""

from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd

from bot import risk
from bot.features import compute_features
from bot.market import SYMBOLS

COMPETITION_CAP = 0.30
SECTORS = None


def sector_codes():
    """Integer sector code per symbol, in SYMBOLS order."""
    global SECTORS
    if SECTORS is None:
        import json
        from kit.config import ROOT
        sectors = json.loads((ROOT / 'universe.json').read_text(encoding='utf-8'))['sectors']
        lookup = {t: i for i, (_, members) in enumerate(sectors.items()) for t in members}
        SECTORS = np.array([lookup[t] for t in SYMBOLS])
    return SECTORS


@dataclass
class AllocationConfig:
    """Approach 1: long-only risk-based core portfolio and rebalancing policy."""

    construction: str = 'min_variance'
    cov_method: str = 'ledoit_wolf'
    lookback_days: int = 126
    cap: float = 0.10
    sector_cap: float = 0.30
    exposure: float = 1.0
    vol_target: float | None = None
    min_exposure: float = 0.5
    blend_equal: float = 0.0
    band_l1: float = 0.10
    band_max: float = 0.03
    trade_fraction: float = 1.0


@dataclass
class AlphaConfig:
    """Approach 2: multiplicative tilt of the core weights by model scores."""

    mode: str = 'off'
    model_path: str = 'bot/models/lgbm.txt'
    strength: float = 0.3
    clip: float = 2.0


@dataclass
class OnlineConfig:
    """Approach 4: exponentially weighted blend of alpha experts."""

    mode: str = 'shadow'
    experts: tuple = ('momentum', 'long_momentum', 'reversal', 'low_volatility', 'model')
    eta: float = 100.0
    decay: float = 0.995
    max_step: float = 0.05
    strength: float = 0.3
    clip: float = 2.0


@dataclass
class AgentConfig:
    """Full agent configuration; `active` names the target that is traded."""

    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)

    @property
    def active(self):
        if self.online.mode == 'active':
            return 'online'
        if self.alpha.mode == 'active':
            return 'alpha'
        return 'base'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        def build(kind, values):
            known = {f.name for f in fields(kind)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f'Unknown {kind.__name__} fields: {sorted(unknown)}')
            values = dict(values)
            if 'experts' in values:
                values['experts'] = tuple(values['experts'])
            return kind(**values)
        return cls(allocation=build(AllocationConfig, data.get('allocation', {})),
                   alpha=build(AlphaConfig, data.get('alpha', {})),
                   online=build(OnlineConfig, data.get('online', {})))


@dataclass
class View:
    """Market information available at a decision cutoff."""

    panel: object
    cutoff: pd.Timestamp
    features: np.ndarray | None = None
    feature_names: list | None = None

    def latest_close(self):
        return self.panel.close.iloc[-1].values

    def feature_row(self):
        if self.features is None:
            self.feature_names, values, _ = compute_features(self.panel)
            self.features = values[-1]
        return self.feature_names, self.features


def zscore(x, clip=3.0):
    """Cross-sectional z-score with NaNs set to zero, clipped to +-clip."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return np.zeros_like(x)
    sd = x[mask].std()
    z = np.where(mask, (x - x[mask].mean()) / sd if sd > 0 else 0.0, 0.0)
    return np.clip(z, -clip, clip)


def tilt(base, z, strength, clip, cap, groups=None, group_cap=None):
    """Multiply base weights by (1 + strength * clip(z)) and re-impose caps at the same total."""
    total = base.sum()
    if total <= 0 or strength == 0:
        return base.copy()
    raw = base * np.clip(1 + strength * np.clip(z, -clip, clip), 0, None)
    return risk.cap_weights(raw, total, cap, groups, group_cap)


def expert_signals(names, row, model_scores=None):
    """Raw alpha vectors for each expert from a feature row (tickers x features)."""
    col = {n: row[:, i] for i, n in enumerate(names)}
    out = {'momentum': col['rel_r35'], 'long_momentum': col['rel_r140'],
           'reversal': -col['rel_r7'], 'low_volatility': -col['vol35']}
    if model_scores is not None:
        out['model'] = model_scores
    return out


class OnlineExperts:
    """Exponentiated-gradient weights over experts with forgetting and step caps.

    Each update scores every expert by the dollar-neutral return of its
    previous z-scored signal since the previous update. State is a plain dict
    so it can be persisted between live decisions.
    """

    def __init__(self, config, state=None):
        self.config = config
        self.state = state if state is not None else {}

    def weights(self):
        names = list(self.config.experts)
        q = self.state.get('weights')
        if not q or set(q) != set(names):
            return {n: 1 / len(names) for n in names}
        return q

    def update(self, signals, prices, when):
        """Score the previous signals on returns to `prices`, then store the new signals."""
        names = [n for n in self.config.experts if n in signals]
        q = self.weights()
        last = self.state.get('last')
        if last and last.get('when') != str(when):
            prev_prices = np.array(last['prices'])
            returns = prices / prev_prices - 1
            scores = dict(self.state.get('scores', {}))
            for n in names:
                if n not in last['signals']:
                    continue
                z = np.array(last['signals'][n])
                gross = np.abs(z).sum()
                realized = float(z @ returns / gross) if gross > 0 else 0.0
                if np.isfinite(realized):
                    scores[n] = self.config.decay * scores.get(n, 0.0) + realized
            self.state['scores'] = scores
            if names:
                logits = np.array([self.config.eta * scores.get(n, 0.0) for n in names])
                proposal = np.exp(logits - logits.max())
                proposal /= proposal.sum()
                old = np.array([q.get(n, 1 / len(names)) for n in names])
                step = np.clip(proposal - old, -self.config.max_step, self.config.max_step)
                new = np.clip(old + step, 1e-6, None)
                q = dict(zip(names, (new / new.sum()).tolist()))
        self.state['weights'] = q
        self.state['last'] = {'when': str(when), 'prices': [float(p) for p in prices],
                              'signals': {n: zscore(signals[n]).tolist() for n in names}}
        return q

    def blend(self, signals):
        q = self.weights()
        names = [n for n in q if n in signals]
        total = sum(q[n] for n in names)
        if not names or total <= 0:
            return np.zeros(len(SYMBOLS))
        return sum(q[n] / total * zscore(signals[n]) for n in names)


class Agent:
    """Computes target portfolios and applies the rebalancing policy."""

    def __init__(self, config=None, model=None):
        self.config = config or AgentConfig()
        self.model = model
        self._core_cache = {}

    def core(self, view):
        """Approach 1 target (cached per last completed trading day)."""
        a = self.config.allocation
        daily = completed_daily_close(view.panel, view.cutoff)
        key = daily.index[-1] if len(daily) else None
        if key in self._core_cache:
            return self._core_cache[key]
        w, info = core_weights(daily, a)
        self._core_cache[key] = (w, info)
        return w, info

    def targets(self, view, model_scores=None, online_state=None):
        """Return ({'base', 'alpha'?, 'online'?: weights}, diagnostics)."""
        a, al, on = self.config.allocation, self.config.alpha, self.config.online
        base, info = self.core(view)
        out = {'base': base}
        groups = sector_codes()
        if model_scores is None and self.model is not None and (al.mode != 'off' or 'model' in on.experts):
            names, row = view.feature_row()
            model_scores = self.model.predict(names, row)
        if al.mode != 'off' and model_scores is not None:
            out['alpha'] = tilt(base, zscore(model_scores), al.strength, al.clip, a.cap, groups, a.sector_cap)
        if on.mode != 'off':
            names, row = view.feature_row()
            signals = expert_signals(names, row, model_scores)
            experts = OnlineExperts(on, online_state)
            q = experts.update(signals, view.latest_close(), view.cutoff)
            out['online'] = tilt(base, experts.blend(signals), on.strength, on.clip, a.cap, groups, a.sector_cap)
            info['expert_weights'] = q
        return out, info

    def decide(self, target, current):
        """Return new target weights if a rebalance is warranted, else None (hold)."""
        a = self.config.allocation
        current = np.nan_to_num(np.asarray(current, dtype=float))
        deviation = target - current
        if np.abs(deviation).sum() <= a.band_l1 and np.abs(deviation).max() <= a.band_max:
            return None
        new = current + a.trade_fraction * deviation
        return finalize(new, a.cap, a.sector_cap)


def completed_daily_close(panel, cutoff):
    """Daily closes for trading days fully completed by `cutoff`."""
    daily = panel.daily_close()
    cutoff = pd.Timestamp(cutoff)
    today = cutoff.normalize()
    if cutoff < today + pd.Timedelta(hours=16):
        daily = daily[daily.index < today]
    return daily


def core_weights(daily_close, a):
    """Approach 1 weights from completed daily closes; equal weight if history is short."""
    groups = sector_codes()
    returns = risk.daily_returns(daily_close, a.lookback_days)
    info = {'history_days': len(returns)}
    if len(returns) < max(30, a.lookback_days // 2) or returns.shape[1] != len(SYMBOLS):
        w = risk.equal_weight(np.eye(len(SYMBOLS)), a.exposure, a.cap, groups, a.sector_cap)
        info['fallback'] = 'equal_weight'
        return w, info
    cov = risk.covariance(returns, a.cov_method)
    w = risk.CONSTRUCTIONS[a.construction](cov, 1.0, a.cap, groups, a.sector_cap)
    if a.blend_equal > 0:
        w = (1 - a.blend_equal) * w + a.blend_equal / len(w)
    vol = risk.portfolio_volatility(w, cov)
    exposure = a.exposure
    if a.vol_target:
        exposure = float(np.clip(a.vol_target / vol, a.min_exposure, a.exposure))
    w = risk.cap_weights(w, exposure, a.cap, groups, a.sector_cap)
    info.update(predicted_vol=vol, exposure=exposure)
    return w, info


def finalize(w, cap=COMPETITION_CAP, sector_cap=None):
    """Clip to [0, cap], enforce the sector cap and keep the total at most 1."""
    w = np.clip(np.nan_to_num(np.asarray(w, dtype=float)), 0, min(cap, COMPETITION_CAP))
    if sector_cap is not None:
        groups = sector_codes()
        for g in np.unique(groups):
            s = w[groups == g].sum()
            if s > sector_cap:
                w[groups == g] *= sector_cap / s
    if w.sum() > 1:
        w = w / w.sum()
    return w
