"""Phase 0 probe 4: the volume at the moment outflow begins.

hydrology_core.py:167-181 applies the day's INFLOW before the day's OUTFLOW:
    sm += infiltration ; fast += excess            (167-168)
    percolation = min(perc_mm_day, fast) ; fast -= percolation   (173-174)
    q0, q1 out of fast ; q2 out of slow            (177-182)

So the store holds `S[t-1] + recharge[t]` when the outflow starts, and the
recurrence `S[t] = S[t-1] + recharge[t] - Q_out[t]` makes that equal
`S[t] + Q_out[t]`.  That is exactly the plan's S_pre = S_post + Q_out, computed
from the SAME day -- no shift.

If it holds, then x = Q_out/(S_post + Q_out) <= 1 IDENTICALLY (S_post >= 0), and
S1.3 row 1 fires: g(x) = x.

Three things are measured here:
  1. the order test -- does S[t]+Q_out[t] equal S[t-1]+inflow[t]?
  2. the pi census under the true pre-outflow volume;
  3. the turnover table under the same.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import pandas as pd

CACHE = ROOT / '5_Test/20260917_5/data/domains/FULL24'
HYDRO = ROOT / '5_Test/20260828_38/outputs/tn_hydrology_reach_daily.parquet'
OUT = RUN / 'reports/probe_prevolume.json'
ND, NR = 23376, 230


def f(x):
    x = float(x)
    if not np.isfinite(x):
        return None if np.isnan(x) else ('inf' if x > 0 else '-inf')
    return x


def dst(a, ps=(0, 1, 25, 50, 75, 99, 100)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {'n': 0}
    v = np.percentile(a, ps)
    return {'n': int(a.size), **{f'p{p}': f(z) for p, z in zip(ps, v)}}


def main():
    rep = {}
    arr = {n: np.load(CACHE / f'{n}.npy', mmap_mode='r', allow_pickle=False)
           for n in ['upper_water', 'soil_water_mm', 'percolation', 'fast_water',
                     'slow_water', 'contact', 'fast_fraction', 'area_ha']}
    cols = ['date', 'reach_id', 'soil_storage_mm', 'upper_response_storage_mm',
            'lower_slow_storage_mm', 'percolation_to_lower_mm_day',
            'precipitation_daily_mm', 'actual_aet_mm_day',
            'upper_store_instantaneous_turnover_day',
            'lower_store_instantaneous_turnover_day']
    w = pd.read_parquet(HYDRO, columns=cols).sort_values(['date', 'reach_id'])
    assert np.array_equal(w.date.to_numpy().reshape(-1, NR)[:, 0].astype('datetime64[D]')[:ND],
                          np.load(CACHE / 'dates.npy', allow_pickle=False))

    def col(n):
        return w[n].to_numpy(np.float64).reshape(-1, NR)[:ND]

    soil_p, upper_p, lower_p = col('soil_storage_mm'), col('upper_response_storage_mm'), col('lower_slow_storage_mm')
    perc, rain, aet = col('percolation_to_lower_mm_day'), col('precipitation_daily_mm'), col('actual_aet_mm_day')
    turn_up, turn_lo = col('upper_store_instantaneous_turnover_day'), col('lower_store_instantaneous_turnover_day')

    area_ha = np.asarray(arr['area_ha'], np.float64)
    fast_mm = np.asarray(arr['fast_water'], np.float64) / (area_ha[None, :] * 10.0)
    slow_mm = np.asarray(arr['slow_water'], np.float64) / (area_ha[None, :] * 10.0)
    Qu = fast_mm + perc

    def shift(a, fill=None):
        b = np.empty_like(a)
        b[0] = a[0] if fill is None else fill
        b[1:] = a[:-1]
        return b

    def res(a, b):
        d = np.abs(a - b)
        sc = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
        return dict(max_abs=f(np.max(d)), median_abs=f(np.median(d)),
                    max_rel=f(np.max(d / sc)), median_rel=f(np.median(d / sc)))

    # ------------------------------------------------------- 1. the order test
    # inflow[t] is *recovered* from the recurrence, so use the independent route:
    #   lower : S[t]+slow[t]  ==  S[t-1]+perc[t]          (both known)
    #   upper : S[t]+Qu[t]    ==  S[t-1]+recharge[t]      (recharge recovered)
    #   soil  : S[t]+aet+rech ==  S[t-1]+rain[t]
    recharge = (upper_p[1:] - upper_p[:-1]) + fast_mm[1:] + perc[1:]
    rech_full = np.vstack([np.full((1, NR), np.nan), recharge])

    rep['order_test'] = {
        'lower__S_plus_slow_vs_Sprev_plus_perc': res(
            (lower_p + slow_mm)[1:], (shift(lower_p) + perc)[1:]),
        'upper__S_plus_Qu_vs_Sprev_plus_recharge': res(
            (upper_p + Qu)[1:], (shift(upper_p) + rech_full)[1:]),
        'soil__S_plus_aet_plus_rech_vs_Sprev_plus_rain': res(
            (soil_p + aet + rech_full)[1:], (shift(soil_p) + rain)[1:]),
        'alternative_Sprev_only_vs_S_post': res(
            (upper_p + Qu)[1:], upper_p[1:]),   # must be large: distinguishes the two readings
        'note': ('if the first three close at rounding and the last does not, the '
                 'volume at outflow time is S_post + Q_out, i.e. inflow-then-outflow'),
    }

    # the two competing pre-volume readings, for the record
    V_pre_up = upper_p + Qu          # inflow-then-outflow (predicted)
    V_prev_up = shift(upper_p)       # previous day's close (rejected if the test holds)
    V_pre_soil = soil_p + aet + rech_full
    V_pre_low = lower_p + slow_mm

    contact = np.asarray(arr['contact'], np.float64)
    ff = np.asarray(arr['fast_fraction'], np.float64)
    act = contact > 0
    interiors = np.zeros_like(act)
    interiors[1:] = True
    sel = act & interiors

    def census(V, Q, name):
        x = np.full(V.shape, np.nan)
        x[sel] = (Q / V)[sel]
        xa = x[sel & np.isfinite(x)]
        g = -np.expm1(-xa)
        return {
            'name': name, 'n': int(xa.size),
            'x': dst(xa),
            'frac_x_ge_1': f(np.mean(xa >= 1.0)),
            'frac_x_gt_1': f(np.mean(xa > 1.0)),
            'frac_x_ge_0p99': f(np.mean(xa >= 0.99)),
            'frac_pi_ge_0p99': f(np.mean(g >= 0.99)),
            'Eu_over_A_median': f(np.median(g)),
            'Eu_over_A_max': f(np.max(g)),
            'one_minus_x': dst(1.0 - xa),
        }

    rep['pi_census'] = {
        'selection': 'active (contact>0), interior days',
        'upper__Qu__POST': census(upper_p, Qu, 'V=S_post'),
        'upper__Qu__PRE_Spost_plus_Qu': census(V_pre_up, Qu, 'V=S_post+Qu'),
        'upper__Qu__PRE_prevday': census(V_prev_up, Qu, 'V=S[t-1]'),
        'soil__Qu__POST': census(soil_p, Qu, 'V=soil_post'),
        'soil__Qu__PRE': census(V_pre_soil, Qu, 'V=soil_post+aet+rech'),
    }

    # ------------------------------------------------------ 3. turnover table
    def medlog(a, b, ok):
        r = np.abs(np.log(np.asarray(a, np.float64)[ok] / np.asarray(b, np.float64)[ok]))
        return f(np.median(r))

    okup = np.isfinite(turn_up) & (turn_up > 0) & interiors
    oklo = np.isfinite(turn_lo) & (turn_lo > 0) & interiors
    cands = {
        'upper_post/Qf_producer_spelling': upper_p / np.maximum(fast_mm, 1e-300),
        'upper_post/Qu': upper_p / np.maximum(Qu, 1e-300),
        'upper_pre/Qu': V_pre_up / np.maximum(Qu, 1e-300),
        'upper_pre/Qf': V_pre_up / np.maximum(fast_mm, 1e-300),
        'soil_pre/Qu': V_pre_soil / np.maximum(Qu, 1e-300),
        'soil_pre/recharge': V_pre_soil / np.maximum(rech_full, 1e-300),
        'lower_pre/slow': V_pre_low / np.maximum(slow_mm, 1e-300),
    }
    rep['turnover_table'] = {
        'vs_producer_upper': {k: medlog(v, turn_up, okup) for k, v in cands.items()},
        'vs_producer_lower': {k: medlog(v, turn_lo, oklo) for k, v in cands.items()},
        'lower_pre_vs_producer_lower': medlog(V_pre_low / np.maximum(slow_mm, 1e-300), turn_lo, oklo),
    }

    # ----------------------------------------------------- 4. the g decision
    xp = (Qu / V_pre_up)[sel & np.isfinite(Qu / V_pre_up)]
    xs = (Qu / soil_p)[sel & np.isfinite(Qu / soil_p)]
    rep['g_decision'] = dict(
        post_outflow_carry=True,
        S_pre_rebuildable=True,
        S_pre_max_rel_residual=rep['order_test']['upper__S_plus_Qu_vs_Sprev_plus_recharge']['max_rel'],
        x_le_1_globally_upper_pre=bool(np.all(xp <= 1.0)),
        x_le_1_globally_soil=bool(np.all(xs <= 1.0)),
        max_x_upper_pre=f(np.max(xp)),
        n_x_gt_1_upper_pre=int((xp > 1.0).sum()),
        max_x_soil=f(np.max(xs)),
        n_x_gt_1_soil=int((xs > 1.0).sum()),
    )
    rep['g_decision']['closure_choice'] = (
        'x' if rep['g_decision']['x_le_1_globally_upper_pre'] else '-expm1(-x)')
    rep['g_decision']['reason'] = (
        'S1.3 row 1 requires post-outflow carry AND S_pre rebuildable AND x<=1 globally. '
        'If all three hold the closure is the LINEAR one, and that is a deterministic '
        'output of the timing gate, not the safe default.')

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f'\n[probe4] wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
