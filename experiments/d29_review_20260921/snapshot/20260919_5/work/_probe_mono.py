"""Throwaway probe -- DELETE BEFORE DELIVERY.

The independent recomputer (`audit_beta.py`) brackets `G_r(k) - T_r` as if `G` were
NON-INCREASING in `k` (`ok_lo = G(lo) >= Tgt`, `ok_hi = G(hi) <= Tgt`), and on this
grid that bracket is empty at every reach for every non-zero beta: the bisection then
returns `hi = 1e6` as the "root" and the audit correctly marks
`solve_reproduced_by_an_independent_root_finder = False` rather than passing silently.

Before touching the audit, MEASURE which way `G` actually goes.  `G(k) = sum_t
av_t(k) p_t(k) W_t Q_r` with `av` from the full self-consistent recurrence: as `k -> 0`
the hazard vanishes so `p -> 0`; as `k -> inf` the hazard saturates so `p -> 1` while
`av` falls to whatever the input alone sustains.  Both ends are finite, so which way it
moves -- and whether a root exists at all -- is a measurement, not an argument.

This file rebuilds the audit's own construction (same peer tree, same model, same
`land`) rather than importing any round-5 module, so what it measures is what the audit
computes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_beta as A                                        # noqa: E402

from closures import scan as frozen_scan                       # noqa: E402

KS = (1e-6, 1e-3, 1e-1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e6)


def build():
    """The audit's `main()` head, verbatim in effect."""
    rec = json.loads((A.PEER / 'outputs' / A.TAG / 'model.json').read_text(encoding='utf-8'))
    design = json.loads(json.dumps(rec['design']))
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = A.sha(A.PEER / 'data/prediction_registry.json')
    x = np.asarray(rec['parameters'], float)
    data = A.cm.load_data('FULL24')
    model = A.cm.make_model(data, None, 'D29_BE', design)
    with torch.no_grad():
        hh, ss, ff, kk = model.flux_parameters(torch.tensor(x))
    Gf = dict(h=np.ascontiguousarray(hh.numpy()), s=np.ascontiguousarray(ss.numpy()),
              f=np.ascontiguousarray(ff.numpy()), k=np.ascontiguousarray(kk.numpy()),
              l=data.lower_release, inp=np.ascontiguousarray(model.inp),
              demand=np.ascontiguousarray(model.demand), cap=bool(model.cap),
              dates=data.dates, vf=float(x[2]))
    nd, nr = Gf['h'].shape

    W = np.asarray(data.fast_water, float) + np.asarray(data.percolation, float) \
        * np.asarray(data.area_ha, float)[None, :] * 10.0
    active = W > A.W_FLOOR
    Weff = np.maximum(W, A.W_FLOOR)
    yrs = np.asarray(Gf['dates']).astype('datetime64[Y]').astype(np.int64) + 1970
    ref = (yrs >= A.REF_YEARS[0]) & (yrs <= A.REF_YEARS[1])
    clim = np.where(active[ref], np.log(Weff[ref]), 0.0).sum(axis=0) \
        / active[ref].sum(axis=0).astype(float)
    u = np.log(Weff) - clim[None, :]

    _f, _s, a0, p0 = frozen_scan(Gf['h'], Gf['s'], Gf['f'], Gf['k'], Gf['l'], Gf['inp'],
                                 Gf['demand'], Gf['cap'])
    Qf = np.asarray(data.fast_water, float)
    Qs = np.asarray(data.percolation, float) * np.asarray(data.area_ha, float)[None, :] * 10.0
    Lf, Ls = np.log(Qf), np.log(Qs)
    L0 = np.logaddexp(Lf, Ls)
    return dict(Gf=Gf, nd=nd, nr=nr, W=W, active=active, yrs=yrs, u=u,
                a0=np.asarray(a0, float), p0=np.asarray(p0, float),
                Lf=Lf, Ls=Ls, L0=L0)


def xi_of(B, beta):
    b = float(beta)
    half = b * B['u'] / 2.0
    Xi = np.exp(np.clip(np.logaddexp(B['Lf'] + half, B['Ls'] - half) - B['L0'],
                        -A.CLIP, A.CLIP))
    return np.where(B['active'], Xi, 1.0)


def Gof(B, Xi, k, Wt, Q):
    kv = np.full(B['nr'], float(k)) if np.ndim(k) == 0 else np.asarray(k, float)
    _f, _s, a, p = A.land(B['Gf']['h'], B['Gf']['s'], B['Gf']['f'], B['Gf']['l'],
                          B['Gf']['inp'], B['Gf']['demand'], Xi * kv[None, :])
    return (a * p * Wt[:, None] * Q[None, :]).sum(axis=0)


def main():
    B = build()
    nr = B['nr']
    Tgt0 = B['a0'] * B['p0']
    rep = {'probe': 'monotonicity of G in k', 'n_reaches': int(nr),
           'ks': list(KS), 'devices': {}}

    kf = json.loads((A.OUT / 'k_field.json').read_text(encoding='utf-8'))['k_field']
    for device in ('N1', 'N3'):
        target, windowname = A.DEVICE_SPEC[device]
        Wt = (B['yrs'] >= A.WINDOWS[windowname][0]) & (B['yrs'] <= A.WINDOWS[windowname][1])
        if target == 'mass':
            Q = np.ones(nr)
        else:
            Q = 1.0 / np.where(Wt[:, None], B['W'], 0.0).sum(axis=0)
        Tgt = (Tgt0 * Wt[:, None] * Q[None, :]).sum(axis=0)
        Xi = xi_of(B, 0.5)

        print('=== %s (%s, %s)  Tgt: min=%.6e med=%.6e max=%.6e ==='
              % (device, target, windowname, Tgt.min(), np.median(Tgt), Tgt.max()))
        print('%12s %16s %16s %16s %10s %10s' %
              ('k', 'G_min', 'G_med', 'G_max', 'frac>=T', 'frac<=T'))
        curve = {}
        for k in KS:
            g = Gof(B, Xi, k, Wt, Q)
            curve['%g' % k] = dict(G_min=float(g.min()), G_med=float(np.median(g)),
                                   G_max=float(g.max()),
                                   frac_ge_Tgt=float((g >= Tgt).mean()),
                                   frac_le_Tgt=float((g <= Tgt).mean()))
            print('%12.3g %16.6e %16.6e %16.6e %10.4f %10.4f'
                  % (k, g.min(), np.median(g), g.max(), (g >= Tgt).mean(),
                     (g <= Tgt).mean()))

        # the two ends, and the production k
        g_lo = Gof(B, Xi, 1e-6, Wt, Q)
        g_hi = Gof(B, Xi, 1e6, Wt, Q)
        kp = np.asarray(kf[device]['points']['0.5']['k'], float)
        g_p = Gof(B, Xi, kp, Wt, Q)
        rel_p = np.abs(g_p - Tgt) / np.maximum(Tgt, 1e-300)
        rep['devices'][device] = dict(
            target=target, window=windowname, curve=curve,
            increasing_on_bracket=float((g_hi >= g_lo).mean()),
            decreasing_on_bracket=float((g_hi <= g_lo).mean()),
            bracket_lo_ge_tgt=float((g_lo >= Tgt).mean()),
            bracket_hi_le_tgt=float((g_hi <= Tgt).mean()),
            valid_bracket_assuming_increasing=float(((g_lo <= Tgt) & (g_hi >= Tgt)).mean()),
            valid_bracket_assuming_decreasing=float(((g_lo >= Tgt) & (g_hi <= Tgt)).mean()),
            production=dict(k_min=float(kp.min()), k_med=float(np.median(kp)),
                            k_max=float(kp.max()),
                            max_rel_G_minus_Tgt=float(rel_p.max()),
                            med_rel_G_minus_Tgt=float(np.median(rel_p)),
                            frac_G_ge_Tgt=float((g_p >= Tgt).mean())))
        print('   G(1e6)>=G(1e-6): %.4f  G(1e6)<=G(1e-6): %.4f'
              % ((g_hi >= g_lo).mean(), (g_hi <= g_lo).mean()))
        print('   bracket valid IF INCREASING [G(lo)<=T<=G(hi)]: %.4f'
              % ((g_lo <= Tgt) & (g_hi >= Tgt)).mean())
        print('   bracket valid IF DECREASING [G(lo)>=T>=G(hi)]: %.4f'
              % ((g_lo >= Tgt) & (g_hi <= Tgt)).mean())
        print('   at PRODUCTION k: max_rel|G-Tgt|/Tgt=%.6e med=%.6e  frac(G>=Tgt)=%.4f'
              % (rel_p.max(), np.median(rel_p), (g_p >= Tgt).mean()))
        print()

    A.OUT.parent.joinpath('work', '_probe_mono.json').write_text(
        json.dumps(rep, indent=1, sort_keys=True), encoding='utf-8')
    print('wrote work/_probe_mono.json')


if __name__ == '__main__':
    main()
