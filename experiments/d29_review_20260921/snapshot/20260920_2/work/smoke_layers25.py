"""One forward through `layers25` on the primary arm's blind-rule `tau_m`, to prove the
file runs before `phase1_arms.py` is written.  Writes `reports/smoke_layers25.json`.
Not a gate, and NOT one of the ten registered arm forwards.

Derived from `20260920_1/work/smoke_layers24.py`.  Deltas: `common24`/`layers24` ->
`common25`/`layers25`; `dp_arrays` takes the primary arm's `q_m`; the frozen-anchor cross
check is added, because this round's whole claim is that `tau_m = 200 d` is a NEW arm and
`q_m = 1` is the parent -- so the smoke run must show a DIFFERENT number from
`2.8234476727379607`, and a run that happened to reproduce it would mean the scalar never
reached the kernel.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C
import layers25 as LY

TAU_PRIMARY = 200.0
out = {}
t0 = time.time()
m = C.build()
pack = C.dp_arrays(m, q_m=C.q_m_of(TAU_PRIMARY), form='x')
out['closure_form'] = pack['form']
out['q_m'] = pack['q_m']
out['k_m'] = pack['k_m']
out['tau_m'] = pack['tau_m']


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


out['Vu_sha'] = arr_sha(pack['Vu'])
out['Vs_sha'] = arr_sha(pack['Vs'])
out['gu_sha'] = arr_sha(pack['gu'])
inst = C.install_kernel(m)
out['install'] = dict(all_bound=inst['all_bound'], closure_form=inst['closure_form'],
                      n_transport_modules=inst['n_transport_modules'],
                      q_m=inst['q_m'], tau_m=inst['tau_m'],
                      per_name={k: v['all_live'] for k, v in inst['per_name'].items()})
out['pilot_n'] = len(inst['pilot_indices'])
df, ex = LY.forward_layers(m, C.TAG)
out['forward_seconds'] = round(time.time() - t0, 2)
out['frame'] = dict(n_rows=int(len(df)), columns=list(df.columns))
out['delivered_columns'] = list(LY.DELIVERED_COLUMNS)
out['deliverable'] = dict(columns=list(LY.deliverable(df).columns),
                          n_rows=int(len(LY.deliverable(df))))
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

# --------------------------------------------------------------------------
# the two readings that say the scalar actually reached the kernel
# --------------------------------------------------------------------------
frozen_c = C.load_anchors()['mean_concentration']['value']
elig = C.eligible_grid()
obs = C.obs_monthly()
ly = LY.deliverable(df)
st = C.monthly_stats(ly, elig, obs)
out['mean_concentration'] = st['mean_concentration']
out['mean_concentration_frozen'] = frozen_c
out['level_ratio_over_frozen'] = st['mean_concentration'] / frozen_c
out['n_eligible_rows'] = st['n_eligible_rows']
out['n_station_months'] = st['n_station_months']
out['the_scalar_moved_the_level'] = bool(st['mean_concentration'] != frozen_c)

# `T^{mob}` is exactly `q_m * N^L` on every cell, and it is NOT zero where water is not
led = None
with __import__('torch').no_grad():
    led = C.MC.ledger_dp2(m, ex['x'])
tl = led['legacy_pool_before_transfer']
tr = led['transfer']
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(tl > 0.0, tr / np.where(tl > 0.0, tl, 1.0), np.nan)
fin = np.isfinite(ratio)
out['N13_transfer_is_q_m_times_nL'] = dict(
    n_finite=int(fin.sum()),
    max_abs_dev=float(np.max(np.abs(ratio[fin] - pack['q_m']))),
    is_a_single_constant=bool(np.max(np.abs(ratio[fin] - pack['q_m'])) == 0.0))
g = np.asarray(pack['guard'], float)
out['N13_transfer_independent_of_water'] = dict(
    n_transfer_on_masked_cells=int(np.sum((g == 0.0) & (tr > 0.0))),
    max_transfer_on_masked_cells=float(np.max(np.where(g == 0.0, tr, 0.0))),
    max_fast_on_masked_cells=float(np.max(np.where(g == 0.0, led['fast'], 0.0))))
out['ledger'] = dict(local_balance_max_kg=led['local_balance_max_kg'],
                     network_balance_kg=led['network_balance_kg'],
                     ledger_spelling=led['ledger_spelling'])
out['saved_kernel'] = C.restore_kernel()
out['total_seconds'] = round(time.time() - t0, 2)

(C.ROUND / 'reports').mkdir(parents=True, exist_ok=True)
(C.ROUND / 'reports' / 'smoke_layers25.json').write_text(
    json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
