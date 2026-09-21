"""One forward through `layers24` on the primary arm, to prove the file runs before
`phase1_arms.py` is written.  Writes `reports/smoke_layers24.json`.  Not a gate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C
import layers24 as LY

out = {}
t0 = time.time()
m = C.build()
pack = C.dp_arrays(m, form='x')
out['closure_form'] = pack['form']
import hashlib

def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()

out['Vu_sha'] = arr_sha(pack['Vu'])
out['Vs_sha'] = arr_sha(pack['Vs'])
out['gu_sha'] = arr_sha(pack['gu'])
inst = C.install_kernel(m)
out['install'] = dict(all_bound=inst['all_bound'], closure_form=inst['closure_form'],
                      n_transport_modules=inst['n_transport_modules'],
                      per_name={k: v['all_live'] for k, v in inst['per_name'].items()})
out['pilot_n'] = len(inst['pilot_indices'])
df, ex = LY.forward_layers(m, C.TAG)
out['forward_seconds'] = round(time.time() - t0, 2)
out['frame'] = dict(n_rows=int(len(df)), columns=list(df.columns))
out['delivered_columns'] = list(LY.DELIVERED_COLUMNS)
out['deliverable'] = dict(columns=list(LY.deliverable(df).columns))
for lay in ('L1', 'L2', 'L3'):
    v = df['p' + lay].to_numpy(float)
    out.setdefault('layers', {})[lay] = dict(
        min=float(np.min(v)), max=float(np.max(v)), median=float(np.median(v)),
        n_nan=int(np.sum(~np.isfinite(v))))
out['station_days'] = [int(df.station_key.nunique()), int(df.date.nunique())]
for k in ('C_u', 'C_s'):
    v = np.asarray(ex[k], float)
    out[k] = dict(min=float(np.min(v)), max=float(np.max(v)),
                  median=float(np.median(v)), n_nan=int(np.sum(~np.isfinite(v))))
out['phi_u'] = dict(min=float(np.min(ex['phi_u'])), max=float(np.max(ex['phi_u'])))
out['cumsum_reassociation'] = dict(
    max_rel=ex['cumsum_reassociation_max_rel'],
    median_rel=ex['cumsum_reassociation_median_rel'],
    exact_frac=ex['cumsum_exact_frac'])
out['saved_kernel'] = C.restore_kernel()
out['total_seconds'] = round(time.time() - t0, 2)

(C.ROUND / 'reports').mkdir(parents=True, exist_ok=True)
(C.ROUND / 'reports' / 'smoke_layers24.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
