"""Pre-N1 smoke check: does the model surface carry what `dp_kernel.flows`/`volumes`
need, at what shapes, and does `V_s` align?  Read-only, no forward.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C

out = {}
m = C.build()
d = m.data
nd, nr = d.fast_water.shape
out['shape'] = [int(nd), int(nr)]
out['dates'] = [str(np.asarray(d.dates)[0]), str(np.asarray(d.dates)[-1]),
                int(np.asarray(d.dates).shape[0])]

want = ['fast_water', 'percolation', 'slow_water', 'area_ha', 'upper_water',
        'soil_water_mm', 'lower_release', 'fast_fraction', 'contact', 'lower_slow_storage_mm']
surf = {}
for k in want:
    v = getattr(d, k, None)
    surf[k] = None if v is None else [list(np.asarray(v).shape), str(np.asarray(v).dtype)]
out['data_surface'] = surf
out['data_all_attrs'] = sorted(a for a in dir(d) if not a.startswith('_'))[:80]
out['pilot_indices'] = C.pilot_indices(m)
out['pilot_n'] = len(out['pilot_indices'])

# which cache files exist
cache = Path(C.cm.RUN if hasattr(C.cm, 'RUN') else C.PEER)
out['cache_dir_repr'] = repr(getattr(d, 'root', None)) + ' | ' + repr(getattr(d, 'dir', None))

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

pack = C.dp_arrays(m, form='x')
out['pack'] = {k: (list(np.asarray(v).shape) if isinstance(v, np.ndarray) else v)
               for k, v in pack.items()}
gu, pf, gs = pack['gu'], pack['phi_f'], pack['gs']
act = np.asarray(d.contact, np.float64) > 0
out['fractions'] = dict(
    gu=dict(min=float(gu.min()), max=float(gu.max()), median=float(np.median(gu)),
            frac_ge_0p99=float(np.mean(gu >= 0.99))),
    phi_f=dict(min=float(pf.min()), max=float(pf.max()), median=float(np.median(pf))),
    gs=dict(min=float(gs.min()), max=float(gs.max()), median=float(np.median(gs))),
    on_active=dict(
        gu_median=float(np.median(gu[act])), gu_max=float(gu[act].max()),
        frac_gu_ge_0p99=float(np.mean(gu[act] >= 0.99)),
        Eu_over_A_median=float(np.median(gu[act]))))
out['guard'] = dict(n_zero=int(np.sum(pack['guard'] == 0.0)),
                    n_total=int(pack['guard'].size),
                    frac_zero=float(np.mean(pack['guard'] == 0.0)))
out['guard_vs_contact'] = dict(
    guard_zero_and_contact_le0=int(np.sum((pack['guard'] == 0.0) & (~act))),
    guard_one_and_contact_le0=int(np.sum((pack['guard'] == 1.0) & (~act))),
    guard_zero_and_contact_gt0=int(np.sum((pack['guard'] == 0.0) & act)))
out['expm1_probe'] = C.XI.expm1_underflow_probe()
out['binding_census'] = C.binding_census()

print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
(C.ROUND / 'reports').mkdir(parents=True, exist_ok=True)
(C.ROUND / 'reports' / 'smoke24.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
