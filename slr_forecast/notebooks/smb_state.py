"""Drifting error state for the Greenland SMB emulator (Kalman filter tools).

Observation, in Gt/yr:  r_t = SMB_obs,t - f(x_t) = M + delta_t + eps_t
State:                  delta_t = phi delta_{t-1} + eta_t,  phi = exp(-1/tau)

M is a constant level with a diffuse prior, delta a stationary AR(1) state
with sd sigma_d, and Var(eps_t) = sigma_e^2 + s_t^2 (weather plus the
reported observation error s_t).  Option 2 adds a constant c multiplying a
regressor H_t (c x).  Used by scripts/smb_state_space_diagnostic.py and the
delay-fit figure in component_greenland.ipynb.
"""

import numpy as np

P_M = 2000.0**2                     # diffuse prior variance on M, (Gt/yr)^2
TAU = np.geomspace(3.0, 150.0, 32)
SD_D = np.linspace(0.0, 240.0, 41)
SD_E = np.linspace(20.0, 250.0, 47)


# ── Priors on the grid ──
# Production tau prior: lognormal with 90% range 5-25 yr, so the 90% upper
# bound allows about two relaxation times within the 47-yr Mouginot record.
# The record does not constrain tau (its likelihood is flat from about 10 to
# over 100 yr), so tau is set by this prior.
TAU_PRIOR = ('lognormal', 5.0, 25.0)


def log_prior(tg, sdg, prior='narrow'):
    """Log prior on (tau, sigma_d) over the grid.  `prior` is 'narrow'
    (lognormal, 90% range 10-50 yr), 'wide' (log-uniform over the grid), or
    a tuple ('lognormal', lo, hi) with 90% range lo-hi yr, or
    ('loguniform', lo, hi) with zero prior weight outside lo-hi yr."""
    if prior == 'narrow':
        prior = ('lognormal', 10.0, 50.0)
    if prior == 'wide':
        lp_tau = np.zeros_like(tg)
    elif prior[0] == 'lognormal':
        lo, hi = prior[1], prior[2]
        mu, s = np.log(np.sqrt(lo * hi)), np.log(hi / lo) / (2 * 1.645)
        lp_tau = -0.5 * ((np.log(tg) - mu) / s) ** 2
    elif prior[0] == 'loguniform':
        lp_tau = np.where((tg >= prior[1]) & (tg <= prior[2]), 0.0, -np.inf)
    else:
        raise ValueError(f'unknown tau prior {prior!r}')
    lp_sd = -0.5 * (sdg / 150.0) ** 2
    return lp_tau + lp_sd


# ── Kalman filter, vectorised over a leading batch dimension ──
def kalman(r, s_obs, phi, sd_d, sd_e, H=None, prior_c=None, return_path=False):
    """r: (B, T) or (T,) observations with NaN for missing years; s_obs: (T,)
    reported errors; phi, sd_d, sd_e: (B,).  State [M, (c), delta].  H: (T,)
    regressor for c (option 2) or None.  Returns log-likelihood (B,), final
    mean (B, n) and covariance (B, n, n); with return_path also the filtered
    and smoothed delta means and sds (B, T)."""
    phi, sd_d, sd_e = (np.atleast_1d(np.asarray(v, float)) for v in (phi, sd_d, sd_e))
    B = max(len(phi), 1 if np.ndim(r) == 1 else r.shape[0])
    r = np.broadcast_to(r, (B, np.shape(r)[-1]))
    T = r.shape[1]
    n = 2 if H is None else 3
    a = np.zeros((B, n))
    P = np.zeros((B, n, n))
    P[:, 0, 0] = P_M
    if H is not None:
        P[:, 1, 1] = prior_c ** 2
    P[:, -1, -1] = sd_d ** 2
    F = np.broadcast_to(np.eye(n), (B, n, n)).copy()
    F[:, -1, -1] = phi
    q = sd_d ** 2 * (1 - phi ** 2)
    ll = np.zeros(B)
    keep = []
    for t in range(T):
        if t > 0:
            a = np.einsum('bij,bj->bi', F, a)
            P = np.einsum('bij,bjk,blk->bil', F, P, F)
            P[:, -1, -1] += q
        a_pred, P_pred = a.copy(), P.copy()
        h = np.zeros(n); h[0] = 1.0; h[-1] = 1.0
        if H is not None:
            h[1] = H[t]
        if np.isfinite(r[0, t]):
            v = r[:, t] - a @ h
            Ph = P @ h
            Sv = Ph @ h + sd_e ** 2 + s_obs[t] ** 2
            K = Ph / Sv[:, None]
            a = a + K * v[:, None]
            P = P - np.einsum('bi,bj->bij', K, K) * Sv[:, None, None]
            ll += -0.5 * (np.log(2 * np.pi * Sv) + v ** 2 / Sv)
        if return_path:
            keep.append((a_pred, P_pred, a.copy(), P.copy()))
    if not return_path:
        return ll, a, P
    # Rauch-Tung-Striebel smoother for the delta path
    filt_m = np.stack([k[2][:, -1] for k in keep], 1)
    filt_s = np.sqrt(np.stack([k[3][:, -1, -1] for k in keep], 1))
    a_s, P_s = keep[-1][2], keep[-1][3]
    sm_m = [a_s[:, -1]]; sm_s = [np.sqrt(P_s[:, -1, -1])]
    for t in range(T - 2, -1, -1):
        a_f, P_f = keep[t][2], keep[t][3]
        a_p, P_p = keep[t + 1][0], keep[t + 1][1]
        J = np.einsum('bij,bkj,bkl->bil', P_f, F, np.linalg.inv(P_p))
        a_s = a_f + np.einsum('bij,bj->bi', J, a_s - a_p)
        P_s = P_f + np.einsum('bij,bjk,blk->bil', J, P_s - P_p, J)
        sm_m.append(a_s[:, -1]); sm_s.append(np.sqrt(np.maximum(P_s[:, -1, -1], 0)))
    return ll, a, P, filt_m, filt_s, np.stack(sm_m[::-1], 1), np.stack(sm_s[::-1], 1)


def grid_posterior(r, s_obs, prior='narrow', H=None, prior_c=None, sd_e=SD_E):
    tg, dg, eg = np.meshgrid(TAU, SD_D, sd_e, indexing='ij')
    tg, dg, eg = tg.ravel(), dg.ravel(), eg.ravel()
    ll, a, P = kalman(r, s_obs, np.exp(-1 / tg), dg, eg, H=H, prior_c=prior_c)
    lp = ll + log_prior(tg, dg, prior)
    w = np.exp(lp - lp.max()); w /= w.sum()
    return {'tau': tg, 'sd_d': dg, 'sd_e': eg, 'w': w, 'a': a, 'P': P, 'll': ll}




def ffbs(r, s_obs, phi, sd_d, sd_e, rng):
    """Forward-filter, backward-sample joint draws of the state [M, delta].

    r: (B, T) residuals with NaN for missing years (a missing year is
    predicted, not updated); s_obs: (T,); phi, sd_d, sd_e: (B,).
    Returns (B, T, 2): one sampled path of (M, delta) per batch member.
    """
    phi, sd_d, sd_e = (np.asarray(v, float) for v in (phi, sd_d, sd_e))
    B, T = r.shape
    a = np.zeros((B, 2)); P = np.zeros((B, 2, 2))
    P[:, 0, 0] = P_M; P[:, 1, 1] = sd_d ** 2
    F = np.zeros((B, 2, 2)); F[:, 0, 0] = 1.0; F[:, 1, 1] = phi
    q = sd_d ** 2 * (1 - phi ** 2)
    h = np.array([1.0, 1.0])
    af, Pf = np.empty((B, T, 2)), np.empty((B, T, 2, 2))
    for t in range(T):
        if t > 0:
            a = np.einsum('bij,bj->bi', F, a)
            P = np.einsum('bij,bjk,blk->bil', F, P, F)
            P[:, 1, 1] += q
        ok = np.isfinite(r[:, t])
        if ok.any():
            v = np.where(ok, r[:, t] - a @ h, 0.0)
            Ph = P @ h
            Sv = Ph @ h + sd_e ** 2 + np.nan_to_num(s_obs[t]) ** 2
            K = np.where(ok[:, None], Ph / Sv[:, None], 0.0)
            a = a + K * v[:, None]
            P = P - np.einsum('bi,bj->bij', K, K) * Sv[:, None, None]
        af[:, t], Pf[:, t] = a, P
    # Last year: joint draw from the filtered distribution.
    L = np.linalg.cholesky(Pf[:, -1] + 1e-9 * np.eye(2))
    out = np.empty((B, T, 2))
    out[:, -1] = af[:, -1] + np.einsum('bij,bj->bi', L, rng.standard_normal((B, 2)))
    # Backward: M is constant, so M_t = M_{t+1} exactly.  delta_t given M
    # comes from the filtered joint, then is updated on the transition
    # delta_{t+1} = phi delta_t + eta, eta ~ N(0, q).  With sigma_d = 0
    # (q = 0 and zero prior variance) delta stays at its filtered mean.
    for t in range(T - 2, -1, -1):
        M = out[:, t + 1, 0]
        Pmm, Pdm, Pdd = Pf[:, t, 0, 0], Pf[:, t, 1, 0], Pf[:, t, 1, 1]
        m = af[:, t, 1] + Pdm / Pmm * (M - af[:, t, 0])
        v = np.maximum(Pdd - Pdm ** 2 / Pmm, 0.0)
        pos = (v > 0) & (q > 0)
        vs = np.where(pos, 1.0 / (1.0 / np.where(pos, v, 1.0) + phi ** 2 / np.where(pos, q, 1.0)), 0.0)
        ms = np.where(pos, vs * (m / np.where(pos, v, 1.0) + phi * out[:, t + 1, 1] / np.where(pos, q, 1.0)), m)
        out[:, t, 0] = M
        out[:, t, 1] = ms + np.sqrt(vs) * rng.standard_normal(B)
    return out


def fit_drift_state(smb_obs, sig_obs, x_obs, b1, b2, years_obs, grid_years, seed,
                    prior=TAU_PRIOR):
    """Fit the drift state to an observed SMB record and extend it to a grid.

    smb_obs, sig_obs : observed SMB and its reported error (Gt/yr, mass gain)
        on consecutive years `years_obs`
    x_obs : emulator driver (GMST anomaly) on `years_obs`
    b1, b2 : (n,) emulator members; f_k(x) = b1 x + b2 x^2
    grid_years : consecutive years spanning `years_obs` (the projection grid)

    The hyperparameters come from the grid posterior on the residual about
    the median emulator; each member draws (tau, sigma_d, sigma_e) from it
    and one joint path of (M, delta) through the record by forward
    filtering and backward sampling.  Off the record delta is the member's
    AR(1) continued forward from the last year and backward from the first
    (a stationary AR(1) is time-reversible).

    Returns dict: 'M' (n,), 'delta' (n, len(grid_years)), 'tau', 'sd_d',
    'sd_e' (n,), and 'posterior' (the grid posterior).
    """
    years_obs = np.asarray(years_obs, float)
    grid_years = np.asarray(grid_years, float)
    if np.any(np.diff(years_obs) != 1) or np.any(np.diff(grid_years) != 1):
        raise ValueError('years_obs and grid_years must be consecutive years')
    b1, b2 = np.asarray(b1, float), np.asarray(b2, float)
    f = b1[:, None] * x_obs[None, :] + b2[:, None] * x_obs[None, :] ** 2
    gp = grid_posterior(smb_obs - np.median(f, axis=0), sig_obs, prior)
    rng = np.random.default_rng(seed)
    pk = rng.choice(len(gp['w']), size=len(b1), p=gp['w'])
    tau, sd_d, sd_e = gp['tau'][pk], gp['sd_d'][pk], gp['sd_e'][pk]
    phi = np.exp(-1.0 / tau)
    paths = ffbs(smb_obs[None, :] - f, sig_obs, phi, sd_d, sd_e, rng)
    i0 = int(np.where(grid_years == years_obs[0])[0][0])
    i1 = i0 + len(years_obs) - 1
    q = sd_d * np.sqrt(1.0 - phi ** 2)
    delta = np.empty((len(b1), len(grid_years)))
    delta[:, i0:i1 + 1] = paths[:, :, 1]
    for t in range(i1 + 1, len(grid_years)):
        delta[:, t] = phi * delta[:, t - 1] + q * rng.standard_normal(len(b1))
    for t in range(i0 - 1, -1, -1):
        delta[:, t] = phi * delta[:, t + 1] + q * rng.standard_normal(len(b1))
    return {'M': paths[:, -1, 0], 'delta': delta, 'tau': tau, 'sd_d': sd_d,
            'sd_e': sd_e, 'posterior': gp}
