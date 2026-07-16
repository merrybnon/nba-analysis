"""Fit and compare discrete distributions to per-game count stats.

Game-log counts (FTA, FTM, REB, ...) are non-negative integers with heavy ties —
typically 15-20 distinct values across ~70 games. Every candidate here is therefore
scored on the integers. That matters most for the lognormal: it is discretized onto
the integer grid rather than evaluated as a density, both so its likelihood is
comparable to the genuinely discrete models and because a raw lognormal cannot
accommodate the zeros that real game logs contain.

The negative binomial is the natural default for these counts. A made-shot count is a
binomial thinning of the corresponding attempt count, and the negative binomial is
closed under binomial thinning: if attempts ~ NB(r, q), then makes ~ NB(r, q') with
the same r. See `thinning_check` for a way to test that on real data.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import optimize, stats


class NegBinomFit(NamedTuple):
    """Negative binomial fit, in scipy's (n=r, p) parametrization."""

    r: float
    p: float
    loglik: float

    n_params = 2

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.n_params)

    @property
    def mean(self) -> float:
        return self.r * (1 - self.p) / self.p

    def pmf(self, k):
        return stats.nbinom.pmf(k, self.r, self.p)

    def sf(self, k):
        """Survival function P(X > k)."""
        return stats.nbinom.sf(k, self.r, self.p)


class DiscretizedLognormalFit(NamedTuple):
    """Lognormal discretized onto the integers: P(X=k) = F(k+0.5) - F(k-0.5)."""

    mu: float
    sigma: float
    loglik: float

    n_params = 2

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.n_params)

    def pmf(self, k):
        return np.exp(_discretized_lognormal_logpmf(np.asarray(k), self.mu, self.sigma))

    def sf(self, k):
        ks = np.arange(0, int(np.max(k)) + 1)
        cdf = np.cumsum(self.pmf(ks))
        return 1 - cdf[np.asarray(k)]


class PoissonFit(NamedTuple):
    """Poisson fit. Included mainly as the equidispersed null: it forces var/mean = 1."""

    lam: float
    loglik: float

    n_params = 1

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.n_params)

    def pmf(self, k):
        return stats.poisson.pmf(k, self.lam)

    def sf(self, k):
        return stats.poisson.sf(k, self.lam)


def aic(loglik: float, n_params: int) -> float:
    """Akaike information criterion. Lower is better; only comparable across models
    fit to the *same* observations on the same support."""
    return 2 * n_params - 2 * loglik


def survival(x, ks):
    """Empirical survival function S(k) = P(X > k), evaluated at each k in `ks`."""
    x = np.asarray(x)
    return np.array([(x > k).mean() for k in ks])


def fit_nbinom(x) -> NegBinomFit:
    """MLE fit of a negative binomial to counts.

    Optimized in unconstrained space (log r, logit p) so the search cannot leave the
    parameter domain, started from the method-of-moments estimate.
    """
    x = np.asarray(x)

    def nll(theta):
        r, p = np.exp(theta[0]), _expit(theta[1])
        return -stats.nbinom.logpmf(x, r, p).sum()

    mean, var = x.mean(), x.var(ddof=1)
    # Underdispersed samples have no MoM solution (var - mean <= 0); fall back to a
    # large r, which is where the NB likelihood pushes anyway as it approaches Poisson.
    r0 = mean**2 / (var - mean) if var > mean else 100.0
    r0 = float(np.clip(r0, 0.5, 500.0))
    p0 = r0 / (r0 + mean)

    res = optimize.minimize(
        nll,
        [np.log(r0), np.log(p0 / (1 - p0))],
        method="Nelder-Mead",
        options={"maxiter": 5000, "fatol": 1e-10, "xatol": 1e-8},
    )
    return NegBinomFit(float(np.exp(res.x[0])), float(_expit(res.x[1])), float(-res.fun))


def fit_discretized_lognormal(x) -> DiscretizedLognormalFit:
    """MLE fit of a lognormal discretized onto the integers.

    The k=0 cell runs from 0 to 0.5, which gives zeros positive mass — a raw
    continuous lognormal returns -inf there and cannot be fit to game logs at all.
    """
    x = np.asarray(x)
    positive = x[x > 0]
    if len(positive) < 2:
        raise ValueError("need at least two positive observations to fit a lognormal")

    start = [np.log(positive).mean(), np.log(max(np.log(positive).std(ddof=1), 1e-3))]
    res = optimize.minimize(
        lambda t: -_discretized_lognormal_logpmf(x, t[0], np.exp(t[1])).sum(),
        start,
        method="Nelder-Mead",
        options={"maxiter": 5000, "fatol": 1e-10, "xatol": 1e-8},
    )
    return DiscretizedLognormalFit(float(res.x[0]), float(np.exp(res.x[1])), float(-res.fun))


def fit_poisson(x) -> PoissonFit:
    """MLE fit of a Poisson. The MLE is just the sample mean."""
    x = np.asarray(x)
    lam = x.mean()
    return PoissonFit(float(lam), float(stats.poisson.logpmf(x, lam).sum()))


def compare_models(x) -> dict[str, float]:
    """AIC for each candidate fit to `x`. Lower is better.

    Ranks these candidates against each other; it does not establish that the winner
    is correct, only that it is the best of the three offered.
    """
    return {
        "NegBinom": fit_nbinom(x).aic,
        "discLogNorm": fit_discretized_lognormal(x).aic,
        "Poisson": fit_poisson(x).aic,
    }


def thinning_check(attempts, makes) -> dict[str, float]:
    """Compare the NB dispersion parameter r of attempts against that of makes.

    Makes are a binomial thinning of attempts, and the negative binomial is closed
    under thinning with r preserved, so these two should land close together. They are
    two independent noisy MLEs, so exact equality is not expected.
    """
    a, m = fit_nbinom(attempts), fit_nbinom(makes)
    return {
        "r_attempts": a.r,
        "r_makes": m.r,
        "ratio": m.r / a.r,
        "p_attempts": a.p,
        "p_makes": m.p,
    }


def _expit(z):
    return 1 / (1 + np.exp(-z))


def _discretized_lognormal_logpmf(k, mu, sigma):
    """log P(X=k) for a lognormal discretized onto the integers."""
    k = np.asarray(k, dtype=float)
    lo = np.maximum(k - 0.5, 0.0)
    hi = k + 0.5
    scale = np.exp(mu)
    mass = stats.lognorm.cdf(hi, s=sigma, scale=scale) - stats.lognorm.cdf(lo, s=sigma, scale=scale)
    return np.log(np.clip(mass, 1e-300, None))
