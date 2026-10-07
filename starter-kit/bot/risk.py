"""Covariance estimation and long-only portfolio construction under competition caps."""

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def daily_returns(daily_close, lookback):
    """Simple daily returns over the last `lookback` days, dropping incomplete columns."""
    returns = daily_close.pct_change().iloc[1:].tail(lookback)
    return returns.dropna(axis=1, how='any')


def covariance(returns, method='ledoit_wolf', halflife=None):
    """Annualised covariance of daily returns.

    `ledoit_wolf` shrinks the sample covariance toward a scaled identity;
    `ewma` applies exponential weights with the given half-life in days.
    """
    x = returns.values
    if method == 'ledoit_wolf':
        cov = LedoitWolf().fit(x).covariance_
    elif method == 'ewma':
        weights = 0.5 ** (np.arange(len(x))[::-1] / (halflife or 30))
        weights /= weights.sum()
        centered = x - weights @ x
        cov = (centered * weights[:, None]).T @ centered
    elif method == 'sample':
        cov = np.cov(x, rowvar=False)
    else:
        raise ValueError(f'Unknown covariance method {method}')
    return cov * TRADING_DAYS


def cap_weights(w, total, cap, groups=None, group_cap=None):
    """Rescale `w` to sum to `total` with per-asset and per-group caps, redistributing excess.

    Excess above a cap is redistributed proportionally to uncapped names; the
    loop terminates because each pass saturates at least one asset or group.
    """
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum() * total
    for _ in range(100):
        changed = False
        over = w > cap + 1e-12
        if over.any():
            excess = (w[over] - cap).sum()
            w[over] = cap
            free = w < cap - 1e-12
            if free.any() and w[free].sum() > 0:
                w[free] += excess * w[free] / w[free].sum()
            changed = True
        if groups is not None and group_cap is not None:
            for g in np.unique(groups):
                members = groups == g
                gsum = w[members].sum()
                if gsum > group_cap + 1e-12:
                    excess = gsum - group_cap
                    w[members] *= group_cap / gsum
                    free = ~members & (w < cap - 1e-12)
                    for h in np.unique(groups[free]):
                        if w[groups == h].sum() >= group_cap - 1e-12:
                            free &= groups != h
                    if free.any() and w[free].sum() > 0:
                        w[free] += excess * w[free] / w[free].sum()
                    changed = True
        if not changed:
            break
    return w


def min_variance(cov, total, cap, groups=None, group_cap=None):
    """Long-only minimum variance with per-asset and optional per-group caps (SLSQP)."""
    n = len(cov)
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - total, 'jac': lambda w: np.ones(n)}]
    if groups is not None and group_cap is not None:
        for g in np.unique(groups):
            mask = (groups == g).astype(float)
            constraints.append({'type': 'ineq', 'fun': lambda w, m=mask: group_cap - m @ w,
                                'jac': lambda w, m=mask: -m})
    x0 = cap_weights(1 / np.sqrt(np.diag(cov)), total, cap, groups, group_cap)
    result = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w, method='SLSQP',
                      bounds=[(0, cap)] * n, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-12})
    w = result.x if result.success else x0
    return cap_weights(np.clip(w, 0, cap), total, cap, groups, group_cap)


def risk_parity(cov, total, cap, groups=None, group_cap=None):
    """Equal risk contribution weights (log-barrier formulation), then capped."""
    n = len(cov)
    vol = np.sqrt(np.diag(cov))
    y0 = 1 / vol

    def objective(y):
        return 0.5 * y @ cov @ y - np.log(y).sum() / n

    result = minimize(objective, y0, jac=lambda y: cov @ y - 1 / (n * y), method='L-BFGS-B',
                      bounds=[(1e-8, None)] * n)
    y = result.x if result.success else y0
    return cap_weights(y, total, cap, groups, group_cap)


def inverse_volatility(cov, total, cap, groups=None, group_cap=None):
    """Weights proportional to 1 / volatility, then capped."""
    return cap_weights(1 / np.sqrt(np.diag(cov)), total, cap, groups, group_cap)


def equal_weight(cov, total, cap, groups=None, group_cap=None):
    """Equal weights summing to `total`."""
    return cap_weights(np.ones(len(cov)), total, cap, groups, group_cap)


CONSTRUCTIONS = {'min_variance': min_variance, 'risk_parity': risk_parity,
                 'inverse_volatility': inverse_volatility, 'equal_weight': equal_weight}


def portfolio_volatility(w, cov):
    """Annualised volatility of weights `w` under `cov`."""
    return float(np.sqrt(max(w @ cov @ w, 0.0)))
