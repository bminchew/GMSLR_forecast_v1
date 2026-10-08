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
def log_prior(tg, sdg, prior='narrow'):
    if prior == 'narrow':          # lognormal, 5-95% = 10-50 yr
        mu, s = np.log(np.sqrt(10 * 50)), np.log(5.0) / (2 * 1.645)
        lp_tau = -0.5 * ((np.log(tg) - mu) / s) ** 2
    else:                          # log-uniform over the grid
        lp_tau = np.zeros_like(tg)
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
    def draw(m, C):
        L = np.linalg.cholesky(C + 1e-9 * np.eye(2))
        return m + np.einsum('bij,bj->bi', L, rng.standard_normal((B, 2)))
    out = np.empty((B, T, 2))
    out[:, -1] = draw(af[:, -1], Pf[:, -1])
    for t in range(T - 2, -1, -1):
        Pp = np.einsum('bij,bjk,blk->bil', F, Pf[:, t], F); Pp[:, 1, 1] += q
        J = np.einsum('bij,bkj,bkl->bil', Pf[:, t], F, np.linalg.inv(Pp))
        m = af[:, t] + np.einsum('bij,bj->bi', J, out[:, t + 1] - np.einsum('bij,bj->bi', F, af[:, t]))
        C = Pf[:, t] - np.einsum('bij,bjk,blk->bil', J, Pp, J)
        out[:, t] = draw(m, 0.5 * (C + np.transpose(C, (0, 2, 1))))
    return out
