"""Throwaway probe -- DELETE BEFORE DELIVERY.

After the audit's bracket was fixed, its independent root finder still disagrees with
the registered `k` at every non-zero beta (max_rel 2.6e-3 .. 2.7e-2) while agreeing
bitwise at beta=0.  The registered record says why:

    status_counts      = LINEAR 128, FIXED_POINT 83, FIXED_POINT_BISECTED 19
    n rho_lin <= 1e-4  = 128

i.e. the registered route is the plan's section 2.2 TWO-ROUTE route: where the closed
form's exact residual is already <= 1e-4 the closed form IS the product (128 reaches),
and only the rest are iterated to the self-consistent fixed point (102 reaches, whose
`max_rho_state_residual` is 2.65e-11).  The audit solved ONE equation -- the
self-consistent `G(k) = T` -- for all 230 reaches, so on the 128 LINEAR reaches it is
comparing the root of a different equation against a one-shot linearisation and calling
the difference a disagreement.  That is a category error in the AUDIT, not in the field.

This probe measures the hypothesis before anything is restructured:

  * recompute `k_lin` INDEPENDENTLY (sum av0 h / sum av0 h Xi, both over the device
    window) and compare it to the registered `k` on the claimed-LINEAR subset;
  * compare the audit's self-consistent root to the registered `k` on the complement;
  * report how the two subsets split, so the audit can compare like with like.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_beta as A                                          # noqa: E402
import _probe_mono as M                                         # noqa: E402


def routes(B, device, beta):
    target, windowname = A.DEVICE_SPEC[device]
    Wt = (B['yrs'] >= A.WINDOWS[windowname][0]) & (B['yrs'] <= A.WINDOWS[windowname][1])
    nr = B['nr']
    Q = np.ones(nr) if target == 'mass' else \
        1.0 / np.where(Wt[:, None], B['W'], 0.0).sum(axis=0)
    Tgt = (B['a0'] * B['p0'] * Wt[:, None] * Q[None, :]).sum(axis=0)
    Xi = M.xi_of(B, beta) if beta != 0.0 else np.ones_like(B['Gf']['h'])
    aw = Wt[:, None] * B['a0'] * B['Gf']['h'] * Q[None, :]
    k_lin = aw.sum(axis=0) / np.where((aw * Xi).sum(axis=0) > 0,
                                      (aw * Xi).sum(axis=0), np.nan)
    # the registered route's OWN criterion, recomputed here
    p_lin = 1.0 - np.exp(-np.minimum(B['Gf']['h'] * Xi * k_lin[None, :], 700.0))
    rho_lin = np.abs((Wt[:, None] * B['a0'] * p_lin * Q[None, :]).sum(axis=0) - Tgt) \
        / np.where(Tgt > 0, Tgt, 1.0)
    got = A.solve_bisect(B['Gf'], Xi, Tgt, Wt, Q)
    return dict(Tgt=Tgt, Xi=Xi, Wt=Wt, Q=Q, k_lin=k_lin, rho_lin=rho_lin, audit=got)


def main():
    B = M.build()
    kf = json.loads((A.OUT / 'k_field.json').read_text(encoding='utf-8'))['k_field']
    rep = {'probe': 'registered route vs audit route'}
    for device in ('N1', 'N1e', 'N3'):
        for beta in (-0.5, 0.5):
            r = routes(B, device, beta)
            k_reg = np.asarray(kf[device]['points']['%g' % beta]['k'], float)
            lin = r['rho_lin'] <= 1e-4
            k_au = r['audit']['k']
            rel_lin = np.abs(k_reg - r['k_lin']) / np.maximum(1.0, np.abs(k_reg))
            rel_fp = np.abs(k_reg - k_au) / np.maximum(1.0, np.abs(k_reg))
            d = dict(
                device=device, beta=beta,
                n_linear_declared_by_criterion=int(lin.sum()),
                n_fixed_point_complement=int((~lin).sum()),
                k_lin_vs_reg_on_LINEAR_max=float(rel_lin[lin].max()),
                k_lin_vs_reg_on_LINEAR_median=float(np.median(rel_lin[lin])),
                audit_vs_reg_on_LINEAR_max=float(rel_fp[lin].max()),
                audit_vs_reg_on_COMPLEMENT_max=float(rel_fp[~lin].max()) if (~lin).any() else None,
                audit_vs_reg_on_COMPLEMENT_median=(float(np.median(rel_fp[~lin]))
                                                   if (~lin).any() else None),
                audit_agrees_everywhere=bool(rel_fp.max() <= 1e-6),
                audit_agrees_on_complement=bool((rel_fp[~lin] <= 1e-6).all()),
            )
            rep['%s|%g' % (device, beta)] = d
            print('%-4s b=%+5.2f  LINEAR=%3d COMPLEMENT=%3d | k_lin vs reg: max=%.3e med=%.3e'
                  ' | audit vs reg: LINEAR max=%.3e  COMPLEMENT max=%.3e med=%.3e'
                  % (device, beta, d['n_linear_declared_by_criterion'],
                     d['n_fixed_point_complement'], d['k_lin_vs_reg_on_LINEAR_max'],
                     d['k_lin_vs_reg_on_LINEAR_median'], d['audit_vs_reg_on_LINEAR_max'],
                     d['audit_vs_reg_on_COMPLEMENT_max'] or -1.0,
                     d['audit_vs_reg_on_COMPLEMENT_median'] or -1.0))
    (A.ROUND / 'work' / '_probe_route.json').write_text(
        json.dumps(rep, indent=1, sort_keys=True), encoding='utf-8')
    print('wrote work/_probe_route.json')


if __name__ == '__main__':
    main()
