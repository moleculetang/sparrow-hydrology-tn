"""Is `L_pre = cumsum(J - F_s) + F_s` wrong, or merely reassociated?  Measure, do not
assume: report both the relative residual and the residual under an exactly-ordered
recurrence over a short prefix.

Writes `reports/probe_lpre.json`.  THIS WRITE IS THE POINT OF THE FILE: `layers24.py`
quotes four of these readings in its `slow_track` docstring (`max_rel_resid`,
`median_rel_resid`, `frac_exact`, `prefix_exact/prefix_n`), and a reading that exists
only in the console of an ad-hoc run cannot be re-run, cannot be checked against the
code that quotes it, and would leave the `np.cumsum` decision resting on a number no
file on disk contains.  Same shape as `probe_saturation.py`.

The provisional `work/_probe_lpre.npy` (~170 MB: four 23376x230 float64 arrays) is NOT
written.  Nothing reads it -- `layers24.py` cites the .py, no report cites the .npy --
so it was residue that looked like an artifact.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C

OUT = C.ROUND / 'reports' / 'probe_lpre.json'

m = C.build(); pack = C.dp_arrays(m, form='x'); C.install_kernel(m)
import torch
with torch.no_grad():
    t = torch.tensor(C.parameters(C.TAG))
    h_, s_, f_, k_ = m.flux_parameters(t[:30])
    fast, slow, a, p = C.MC.scan_dp_full(h_, s_, f_, k_, m)
fast = np.asarray(fast, float); slow = np.asarray(slow, float)
a = np.asarray(a, float); p = np.asarray(p, float)
Eu = a * p
J = Eu * (1.0 - pack['phi_f'])
cum = np.cumsum(J - slow, axis=0) + slow
res = np.abs(cum * pack['gs'] - slow)
sc = np.maximum(np.abs(slow), 1e-300)
# exactly-ordered recurrence over the first K days
K = 400
gs = pack['gs']; Lx = np.zeros(J.shape[1]); pre = np.zeros((K, J.shape[1])); sl = np.zeros((K, J.shape[1]))
for tt in range(K):
    pr = Lx + J[tt]; sl[tt] = pr * gs[tt]; pre[tt] = pr; Lx = pr - sl[tt]
out = dict(
    n=J.shape,
    max_abs_resid=float(res.max()),
    max_rel_resid=float((res / sc).max()),
    median_rel_resid=float(np.median(res / sc)),
    n_exact=int(np.sum(res == 0.0)), frac_exact=float(np.mean(res == 0.0)),
    prefix_K=int(K),
    prefix_max_abs_rel=float(np.max(np.abs(pre * gs[:K] - slow[:K]) / np.maximum(slow[:K], 1e-300))),
    prefix_exact=int(np.sum(pre * gs[:K] == slow[:K])),
    prefix_n=int(pre.size),
    full_frac_within_1e_14=float(np.mean(res / sc <= 1e-14)),
    full_frac_within_1e_12=float(np.mean(res / sc <= 1e-12)),
)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps(out, indent=1, default=str))
print('\n[probe_lpre] wrote %s' % OUT, flush=True)
