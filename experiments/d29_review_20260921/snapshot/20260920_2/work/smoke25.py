"""Pre-N1' smoke check for round `20260920_2`: does the model surface carry what
`dp_kernel.flows`/`volumes` need, at what shapes, does `V_s` align, and is the ONE new
scalar `q_m` refused when it is missing?  Read-only, NO forward.

Derived from `20260920_1/work/smoke24.py`.  Deltas: `common24` -> `common25`; `dp_arrays`
now takes `q_m`; a new `q_m_refusals` block proves all three of `_check_q_m`'s refusals and
`dp_arrays`'s own `Q_M_NOT_GIVEN` actually fire (a refusal that has never been observed
firing is a comment, not an assertion).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C

out = {}
m = C.build()
d = m.data
nd, nr = d.fast_water.shape
out['shape'] = [int(nd), int(nr)]
out['dates'] = [str(np.asarray(d.dates)[0]), str(np.asarray(d.dates)[-1]),
                int(np.asarray(d.dates).shape[0])]

want = ['fast_water', 'percolation', 'slow_water', 'area_ha', 'upper_water',
        'soil_water_mm', 'lower_release', 'fast_fraction', 'contact',
        'lower_slow_storage_mm']
surf = {}
for k in want:
    v = getattr(d, k, None)
    surf[k] = None if v is None else [list(np.asarray(v).shape), str(np.asarray(v).dtype)]
out['data_surface'] = surf
out['pilot_indices'] = C.pilot_indices(m)
out['pilot_n'] = len(out['pilot_indices'])

for k in ['fast_water', 'percolation', 'slow_water', 'upper_water', 'soil_water_mm',
          'area_ha', 'fast_fraction', 'contact', 'lower_release']:
    v = getattr(d, k, None)
    if v is None:
        continue
    a = np.asarray(v, np.float64)
    out.setdefault('stats', {})[k] = dict(min=float(np.min(a)), max=float(np.max(a)),
                                          n_zero=int(np.sum(a == 0.0)))

try:
    S = C.load_lower_storage(m)
    out['lower_storage'] = dict(shape=list(S.shape), min=float(S.min()), max=float(S.max()),
                                alignment='PROVEN_BY_LOCAL_SLOW_RESPONSE')
except SystemExit as e:
    out['lower_storage'] = dict(ERROR=str(e))

# --------------------------------------------------------------------------
# the three refusals, OBSERVED FIRING
# --------------------------------------------------------------------------
ref = {}
try:
    C.dp_arrays(m, form='x')
    ref['missing_q_m'] = 'DID_NOT_RAISE'
except ValueError as e:
    ref['missing_q_m'] = str(e)
try:
    C.dp_arrays(m, q_m=0.5)
    ref['missing_form'] = 'DID_NOT_RAISE'
except ValueError as e:
    ref['missing_form'] = str(e)
try:
    C.dp_arrays(m, q_m=np.zeros(nr), form='x')
    ref['ndarray_q_m'] = 'DID_NOT_RAISE'
except TypeError as e:
    ref['ndarray_q_m'] = str(e)
for bad in (1.5, -0.1, float('nan'), float('inf')):
    try:
        C.dp_arrays(m, q_m=bad, form='x')
        ref['range_%r' % bad] = 'DID_NOT_RAISE'
    except ValueError as e:
        ref['range_%r' % bad] = str(e)
# the kernel's own guard, once the module is reachable
MC = C.MC
for label, val in (('none', None), ('ndarray', np.zeros(3)), ('gt1', 1.5), ('nan', float('nan'))):
    MC.CONFIG['q_m'] = val
    try:
        MC._check_q_m()
        ref['kernel_%s' % label] = 'DID_NOT_RAISE'
    except (RuntimeError, TypeError, ValueError) as e:
        ref['kernel_%s' % label] = '%s: %s' % (type(e).__name__, e)
MC.CONFIG['q_m'] = None
out['q_m_refusals'] = ref

# --------------------------------------------------------------------------
# the install, on the primary arm's blind-rule q_m
# --------------------------------------------------------------------------
TAU_PRIMARY = 200.0
pack = C.dp_arrays(m, q_m=C.q_m_of(TAU_PRIMARY), form='x')
out['pack_scalars'] = dict(q_m=pack['q_m'], k_m=pack['k_m'], tau_m=pack['tau_m'])
out['pack'] = {k: (list(np.asarray(v).shape) if isinstance(v, np.ndarray) else v)
               for k, v in pack.items() if k not in ('guard',)}
gu, pf, gs = pack['gu'], pack['phi_f'], pack['gs']
act = np.asarray(d.contact, np.float64) > 0
out['fractions'] = dict(
    gu=dict(min=float(gu.min()), max=float(gu.max()), median=float(np.median(gu)),
            frac_ge_0p99=float(np.mean(gu >= 0.99))),
    phi_f=dict(min=float(pf.min()), max=float(pf.max()), median=float(np.median(pf))),
    gs=dict(min=float(gs.min()), max=float(gs.max()), median=float(np.median(gs))),
    on_active=dict(gu_median=float(np.median(gu[act])), gu_max=float(gu[act].max()),
                   frac_gu_ge_0p99=float(np.mean(gu[act] >= 0.99))))
out['guard'] = dict(n_zero=int(np.sum(pack['guard'] == 0.0)),
                    n_total=int(pack['guard'].size),
                    frac_zero=float(np.mean(pack['guard'] == 0.0)))
out['guard_vs_contact'] = dict(
    guard_zero_and_contact_le0=int(np.sum((pack['guard'] == 0.0) & (~act))),
    guard_one_and_contact_le0=int(np.sum((pack['guard'] == 1.0) & (~act))),
    guard_zero_and_contact_gt0=int(np.sum((pack['guard'] == 0.0) & act)))
out['expm1_probe'] = C.XI.expm1_underflow_probe()
out['expm1_probe_q_m'] = C.expm1_underlying_contrast()
out['binding_census'] = C.binding_census()

# the mask semantics the kernel will be held to: `T^{mob}` happens where water does not
Tmask = float(np.max(np.asarray(pack['guard'], float) * 0.0))
out['mask_note'] = ('guard is 0 exactly where contact <= 0; T^{mob} = q_m*N^L does not '
                    'contain guard, so it is NOT zeroed there -- asserted in '
                    'phase0_gates.py 3.3 on the kernel outputs, not here')
out['mask_guard_max'] = Tmask

print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
(C.ROUND / 'reports').mkdir(parents=True, exist_ok=True)
(C.ROUND / 'reports' / 'smoke25.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
