"""Online risk control with audited (partial) feedback.

Implements 02_METHODOLOGY §5.7. In deployment the outcome of a *skipped* test is
never observed, so online methods that need every label cannot be used as is.
We run a random audit: each flagged test is still performed with probability p
(independently of everything else), and the threshold follows an ACI-style
update with an inverse-propensity loss estimate

    flag_t  = 1[u_t <= λ_t],          u_t = F̂_cal(s_t) (rank scale, monotone in s)
    ℓ̂_t    = D_t · flag_t · A_t / p,  A_t ~ Bernoulli(p)  (D_t observed iff test done)
    λ_{t+1} = min(λ_t + η (α − ℓ̂_t), 1).

Deterministic bound (any sequence, any shift): λ never drops below −η(1/p − α), and
the cap at 1 only removes upward steps, so

    (1/T) Σ_t ℓ̂_t <= α + (λ_1 + η (1/p − α)) / (η T).

Because E[ℓ̂_t | past, D_t, s_t] = ℓ_t = D_t · flag_t, the realised miss rate
(1/T) Σ ℓ_t differs from (1/T) Σ ℓ̂_t by a martingale average with increments
bounded by 1/p (Azuma: O(1/(p √T))). Hence the long-run miss rate is at most
α + o(1) without exchangeability. Audited flagged tests are performed, so the
effective reduction is mean(flag · (1 − A)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rank_transform(cal_scores: np.ndarray):
    """Empirical CDF of calibration scores: s -> fraction of calibration scores <= s."""
    ref = np.sort(np.asarray(cal_scores, float))
    n = max(len(ref), 1)

    def f(s: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, np.asarray(s, float), side="right") / n

    return f


def estimate_bound(alpha: float, eta: float, audit_p: float, lam1: float, T: int) -> float:
    """Right-hand side of the deterministic bound on the mean estimated loss."""
    return alpha + (max(lam1, 0.0) + eta * (1.0 / audit_p - alpha)) / (eta * max(T, 1))


@dataclass
class OnlineResult:
    flag: np.ndarray        # skipped-or-audited decision per step
    audit: np.ndarray       # audited (performed anyway) per step
    loss: np.ndarray        # true loss D·flag per step
    loss_hat: np.ndarray    # IPW estimate per step
    lam: np.ndarray         # threshold before each step (rank scale)

    @property
    def miss_rate(self) -> float:
        return float(self.loss.mean()) if len(self.loss) else float("nan")

    @property
    def flag_rate(self) -> float:
        """Share of tests flagged low value (before audits)."""
        return float(self.flag.mean()) if len(self.flag) else float("nan")

    @property
    def reduction(self) -> float:
        """Share of tests actually avoided (flagged and not audited)."""
        return float((self.flag & ~self.audit).mean()) if len(self.flag) else float("nan")

    def worst_window(self, w: int) -> float:
        """Largest miss rate over sliding windows of ``w`` consecutive tests."""
        if len(self.loss) < w or w <= 0:
            return self.miss_rate
        c = np.concatenate([[0.0], np.cumsum(self.loss)])
        return float(((c[w:] - c[:-w]) / w).max())


def online_risk_control(u: np.ndarray, d: np.ndarray, alpha: float, eta: float,
                        audit_p: float, lam1: float, rng: np.random.Generator
                        ) -> OnlineResult:
    """Run the audited online update over a stream (rank-scale scores ``u``, labels ``d``).

    ``audit_p = 1`` is full feedback (every flagged test is still checked: no reduction,
    used as a reference); the labels ``d`` are read only when a test is performed.
    """
    if not 0 < audit_p <= 1:
        raise ValueError("audit_p must be in (0, 1]")
    u = np.asarray(u, float)
    d = np.asarray(d, int)
    T = len(u)
    audit_draw = rng.random(T) < audit_p
    flag = np.zeros(T, bool)
    audit = np.zeros(T, bool)
    loss_hat = np.zeros(T)
    lam = np.empty(T)
    cur, inv_p = float(lam1), 1.0 / audit_p
    ul, dl, al = u.tolist(), d.tolist(), audit_draw.tolist()   # plain floats: fast loop
    lam_l, flag_l, aud_l, lh_l = [0.0] * T, [False] * T, [False] * T, [0.0] * T
    for t in range(T):
        lam_l[t] = cur
        lh = 0.0
        if ul[t] <= cur:
            flag_l[t] = True
            if al[t]:
                aud_l[t] = True
                lh = dl[t] * inv_p         # outcome observed only because it was audited
        # not flagged -> test performed -> D observed, but the loss is 0 by definition
        lh_l[t] = lh
        cur = cur + eta * (alpha - lh)
        if cur > 1.0:
            cur = 1.0
    lam[:], flag[:], audit[:], loss_hat[:] = lam_l, flag_l, aud_l, lh_l
    return OnlineResult(flag=flag, audit=audit, loss=d * flag, loss_hat=loss_hat, lam=lam)


def static_result(u: np.ndarray, d: np.ndarray, lam: float) -> OnlineResult:
    """Fixed threshold (calibrated once), for comparison on the same stream."""
    u, d = np.asarray(u, float), np.asarray(d, int)
    f = u <= lam
    return OnlineResult(flag=f, audit=np.zeros(len(u), bool), loss=d * f,
                        loss_hat=(d * f).astype(float), lam=np.full(len(u), lam))
