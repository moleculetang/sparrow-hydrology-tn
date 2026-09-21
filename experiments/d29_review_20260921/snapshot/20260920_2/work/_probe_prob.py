"""Throwaway probe 2: the closed-form level crossing band, per cell, and the
monotonicity of the closed form in `q_m`.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C

TAU = [30.0, 90.0, 200.0, 365.0, 1e3, 3.6e3, 1e4, 1e5, 1e6]

model = C.build()
d = model.data
x30 = C.parameters(C.TAG)
with torch.no_grad():
    h_, s_, f_, k_ = [np.ascontiguousarray(v.numpy())
                      for v in model.flux_parameters(torch.tensor(x30))]
nd, nr = h_.shape
prob = -np.expm1(-np.minimum(h_, 700.0))
sv = np.asarray(s_, np.float64)
pack = C.dp_arrays(model, q_m=C.q_m_of(200.0), form='x')
gu = np.asarray(pack['gu'], np.float64)
act = np.asarray(d.contact, np.float64) > 0
interior = np.ones_like(act, bool); interior[0] = False
sel = act & interior
print('sel n', int(sel.sum()))

s2 = np.broadcast_to(sv[None, :], h_.shape)
one_m_s = 1.0 - s2
F2 = gu / (one_m_s + s2 * gu)                     # mobile-side drain
EI_frozen = prob / (one_m_s + s2 * prob)          # frozen kernel steady state
K = EI_frozen / F2
print('F2  median %.6f  min %.6f' % (np.median(F2[sel]), np.min(F2[sel])))
print('EI_frozen median %.6f' % np.median(EI_frozen[sel]))
print('K median %.6f  min %.6f  max %.6f' % (np.median(K[sel]), np.min(K[sel]), np.max(K[sel])))


def q_of_tau(t):
    return -np.expm1(-1.0 / np.asarray(t, float))


def tau_of_q(q):
    return -1.0 / np.log1p(-np.asarray(q, float))


for t in TAU:
    q = q_of_tau(t)
    EI = (q / (one_m_s + s2 * q)) * F2
    NM = q / (one_m_s + s2 * q)
    print('tau %10.1f q %.6e  EI/med %.6f  EI/EF med %.4f  NM med %.6f'
          % (t, q, np.median(EI[sel]), np.median(EI[sel] / EI_frozen[sel]),
             np.median(NM[sel])))

# the crossing: q* with EI(q*) == EI_frozen per cell
with np.errstate(divide='ignore', invalid='ignore'):
    qs = K * one_m_s / (1.0 - K * s2)
ok = sel & np.isfinite(qs) & (qs > 0.0) & (qs < 1.0)
ta = tau_of_q(qs[ok])
print('\nper-cell crossing over %d cells' % ok.sum())
for p in (5, 25, 50, 75, 95):
    print('  tau* p%02d = %.6g d   q* = %.6e' % (p, np.percentile(ta, p),
                                                 np.percentile(qs[ok], p)))
print('  tau* min %.6g  max %.6g' % (ta.min(), ta.max()))
print('  band [min,max] = [%.4g, %.4g] d' % (ta.min(), ta.max()))

# the +-0.5% level window, measured rather than assumed
t0 = float(np.median(ta))
q0 = q_of_tau(t0)
EI0 = (q0 / (one_m_s + s2 * q0)) * F2
for fac in (0.995, 1.005):
    t1 = t0 * fac
    q1 = q_of_tau(t1)
    EI1 = (q1 / (one_m_s + s2 * q1)) * F2
    print('  tau x %.3f -> median EI rel %.6f' % (fac, np.median(EI1[sel] / EI0[sel]) - 1.0))
# solve for the tau factor giving exactly +-0.5% in EI
lo, hi = t0 * 0.9, t0 * 1.1
for _ in range(80):
    mid = 0.5 * (lo + hi)
    qm = q_of_tau(mid)
    r = np.median(((qm / (one_m_s + s2 * qm)) * F2)[sel] / EI0[sel]) - 1.005
    if r < 0:
        lo = mid
    else:
        hi = mid
print('  tau for +0.5%% median EI: %.6f d  (= %.4f%% of tau*)' % (lo, 100 * lo / t0))
