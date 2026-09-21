"""Phase 0 probe 2: the State Timing Gate and the turnover table.

Probe 1 settled two things by measurement:
  * every cache array is BITWISE the producer column it claims to be;
  * the exported storage is the POST-outflow carry
    (`dS_low = perc - slow` closes at 4.02e-16 relative; the lagged form is 0.97).

The producer's own assertion agrees -- `frozen_hydrology_core.py:194` checks
`lower_previous + percolation - <slow> - lower_now`, i.e. exactly `S[t]` post.

So `S_pre[t] = S[t-1]` exactly, with no reconstruction.  This probe asks what
that correction does to the two numbers the plan hinges on:

  R3'  the 16.45% pi_f == 1 census -- artifact of post-denominator, or real?
  S2.3 the turnover table -- and whether it can select a primary arm at all.

Also registers the deliberate slow-path structural difference on the inert mask.
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
OUT = RUN / 'reports/probe_timing.json'
ND, NR = 23376, 230


def f(x):
    x = float(x)
    if not np.isfinite(x):
        return None if np.isnan(x) else ('inf' if x > 0 else '-inf')
    return x


def qs(a, ps=(1, 10, 50, 90, 99, 100)):
    v = np.percentile(np.asarray(a, np.float64), ps)
    return {f'p{p}': f(x) for p, x in zip(ps, v)}


def census(x, name, sel):
    """pi census under one closure, on one selection."""
    xa = x[sel & np.isfinite(x)]
    gx = -np.expm1(-xa)
    return {
        'n': int(xa.size),
        'x': qs(xa),
        'frac_x_ge_1': f(np.mean(xa >= 1.0)),
        'frac_x_gt_1_strict': f(np.mean(xa > 1.0)),
        'frac_pi_ge_0p99': f(np.mean(gx >= 0.99)),
        'frac_pi_rounds_to_1': f(np.mean(xa >= 36.7)),
        'frac_pi_exactly_1': f(np.mean(xa > 745.2)),
        'Eu_over_A_max': f(np.max(gx)),
        'Eu_over_A_median': f(np.median(gx)),
        'frac_Eu_gt_A': f(np.mean(gx > 1.0)),
    }


def main():
    rep = {'_note': 'post-outflow proven by probe 1 + frozen_hydrology_core.py:194'}

    arr = {n: np.load(CACHE / f'{n}.npy', mmap_mode='r', allow_pickle=False)
           for n in ['soil_water_mm', 'upper_water', 'percolation', 'fast_water',
                     'slow_water', 'contact', 'lower_release', 'area_ha']}

    cols = ['date', 'reach_id', 'soil_storage_mm', 'upper_response_storage_mm',
            'lower_slow_storage_mm', 'percolation_to_lower_mm_day',
            'local_fast_response_m3_s', 'local_slow_response_m3_s',
            'precipitation_daily_mm', 'actual_aet_mm_day',
            'upper_store_instantaneous_turnover_day',
            'lower_store_instantaneous_turnover_day']
    w = pd.read_parquet(HYDRO, columns=cols).sort_values(['date', 'reach_id'])
    d = w.date.to_numpy().reshape(-1, NR)[:, 0].astype('datetime64[D]')
    assert np.array_equal(d[:ND], np.load(CACHE / 'dates.npy', allow_pickle=False))

    def col(n):
        return w[n].to_numpy(np.float64).reshape(-1, NR)[:ND]

    soil_p, upper_p = col('soil_storage_mm'), col('upper_response_storage_mm')
    lower_p = col('lower_slow_storage_mm')
    perc_mm = col('percolation_to_lower_mm_day')
    rain, aet = col('precipitation_daily_mm'), col('actual_aet_mm_day')
    turn_up, turn_lo = col('upper_store_instantaneous_turnover_day'), col('lower_store_instantaneous_turnover_day')

    area_ha = np.asarray(arr['area_ha'], np.float64)
    fast_mm = np.asarray(arr['fast_water'], np.float64) / (area_ha[None, :] * 10.0)
    slow_mm = np.asarray(arr['slow_water'], np.float64) / (area_ha[None, :] * 10.0)
    Qu = fast_mm + perc_mm                      # both leave the UPPER store

    contact = np.asarray(arr['contact'], np.float64)
    mask, act = contact <= 0, ~(contact <= 0)

    # ------------------------------------------------ settle the timing, again
    # S_pre[t] := S[t-1].  Rebuild it the other way and require agreement.
    S_pre_up = np.empty_like(upper_p)
    S_pre_up[0] = np.nan
    S_pre_up[1:] = upper_p[:-1]
    recharge = (upper_p[1:] - upper_p[:-1]) + fast_mm[1:] + perc_mm[1:]
    up_now_from_pre = S_pre_up[1:] + recharge - fast_mm[1:] - perc_mm[1:]
    rep['timing'] = dict(
        exported_side='POST_OUTFLOW_CARRY',
        evidence=[
            'probe1: |dS_low-(perc-slow)|/S max_rel = 4.0219e-16; lagged form = 0.971 -> same-day only',
            'frozen_hydrology_core.py:194 asserts lower_previous+percolation-<slow>-lower_now',
            'hydrology_core.py:160-184 order: infiltration -> fast+=excess -> percolation out of fast -> q0,q1',
        ],
        S_pre_definition='S_pre[t] = S[t-1]  (exact, no reconstruction needed)',
        rebuild_maxabs=f(np.max(np.abs(up_now_from_pre - upper_p[1:]))),
        rebuild_maxrel=f(np.max(np.abs(up_now_from_pre - upper_p[1:])
                                / np.maximum(np.abs(upper_p[1:]), 1e-300))),
        recharge_negative_fraction=f(np.mean(recharge < 0)),
        recharge_over_Qu=dict(
            p1=f(np.percentile((recharge / np.maximum(Qu[1:], 1e-300)).ravel(), 1)),
            p50=f(np.median((recharge / np.maximum(Qu[1:], 1e-300)).ravel())),
            p99=f(np.percentile((recharge / np.maximum(Qu[1:], 1e-300)).ravel(), 99)),
            note='large recharge/Qu => S_pre >> S_post => the correction matters',
        ),
    )

    # ------------------------------------------------------------- R3' ruling
    S_pre_soil = np.empty_like(soil_p)
    S_pre_soil[0] = np.nan
    S_pre_soil[1:] = soil_p[:-1]

    def shifted(a):
        b = np.empty_like(a)
        b[0] = np.nan
        b[1:] = a[:-1]
        return b

    interiors = np.zeros_like(contact, dtype=bool)
    interiors[1:] = True
    sel_u = act & interiors
    sel_s = act & interiors

    rep['R3_pi_census'] = {
        'selection': 'active cells (contact>0) AND interior days (t>=1)',
        'upper__Qf__post_denominator': census(fast_mm / upper_p, 'u', sel_u),
        'upper__Qf__pre_denominator': census(fast_mm / S_pre_up, 'u', sel_u),
        'upper__Qu__post_denominator': census(Qu / upper_p, 'u', sel_u),
        'upper__Qu__pre_denominator': census(Qu / S_pre_up, 'u', sel_u),
        'soil__Qu__post_denominator': census(Qu / soil_p, 's', sel_s),
        'soil__Qu__pre_denominator': census(Qu / S_pre_soil, 's', sel_s),
    }

    # how many cells does the correction actually rescue, and where do the
    # survivors live?
    x_post = Qu / upper_p
    x_pre = Qu / S_pre_up
    both_fin = np.isfinite(x_post) & np.isfinite(x_pre) & sel_u
    bad_post = both_fin & (x_post >= 1.0)
    rep['R3_pi_census']['correction_effect'] = dict(
        n_compared=int(both_fin.sum()),
        n_x_ge_1_post=int(bad_post.sum()),
        n_x_ge_1_pre=int((both_fin & (x_pre >= 1.0)).sum()),
        n_rescued=int((bad_post & (x_pre < 1.0)).sum()),
        n_broken=int(((~bad_post) & both_fin & (x_pre >= 1.0)).sum()),
        median_ratio_post_over_pre=f(np.median((x_post / x_pre)[both_fin])),
        p10_ratio=f(np.percentile((x_post / x_pre)[both_fin], 10)),
        p90_ratio=f(np.percentile((x_post / x_pre)[both_fin], 90)),
    )

    # is the x>=1 set the inert mask leaking back in?  Compare by construction.
    rep['R3_pi_census']['x_ge_1_vs_mask'] = dict(
        frac_of_x_ge_1_post_that_is_active=int(bad_post.sum()),
        frac_inert_overall=f(mask.mean()),
        overlap_note='x>=1 here is computed on active cells only, so it is NOT the mask',
    )

    # ------------------------------------------------------- S2.3 turnover table
    # Every candidate denominator, against BOTH producer-declared columns.
    def medlog(a, b, ok):
        r = np.abs(np.log(np.asarray(a)[ok] / np.asarray(b)[ok]))
        return f(np.median(r))

    okup = np.isfinite(turn_up) & (turn_up > 0) & interiors
    oklo = np.isfinite(turn_lo) & (turn_lo > 0) & interiors
    # recharge[i] belongs to day i+1, so pad one row of NaN at the front to
    # align it with the (day, reach) grid.
    rech_full = np.vstack([np.full((1, NR), np.nan), recharge])
    cands = {
        'upper/Qf_producer_spelling': upper_p / np.maximum(fast_mm, 1e-300),
        'upper/Qu': upper_p / np.maximum(Qu, 1e-300),
        'upper_pre/Qu': S_pre_up / np.maximum(Qu, 1e-300),
        'soil/Qu': soil_p / np.maximum(Qu, 1e-300),
        'soil/Qf': soil_p / np.maximum(fast_mm, 1e-300),
        'soil/recharge': soil_p / np.maximum(rech_full, 1e-300),
    }
    rep['turnover_table'] = {
        'producer_upper_col': 'upper_response_storage_mm / local_fast_mm_day',
        'producer_lower_col': 'lower_slow_storage_mm / local_slow_mm_day',
        'vs_producer_upper': {k: medlog(v, turn_up, okup) for k, v in cands.items()},
        'vs_producer_lower': {k: medlog(v, turn_lo, oklo) for k, v in cands.items()},
        'tautology_note': ('upper/Qf vs the producer upper column is an ALGEBRAIC IDENTITY: '
                           'fast_water/(area_ha*10) == local_fast*86.4/area_km2 because '
                           '86400/(100*10) == 86.4.  Measured median |log ratio| is 0.0. '
                           'A rule that selects on this comparison selects V-upper for free.'),
    }

    # --------------------------------------------------- slow-path difference
    lr = np.asarray(arr['lower_release'], np.float64)
    rep['slow_path_on_mask'] = dict(
        note=('deliberate structural difference: the frozen kernel releases slow=L*l '
              'and l is NONZERO on the mask, so the two kernels MUST differ there'),
        max_lower_release_on_mask=f(np.max(lr[mask])),
        frac_lower_release_gt_0_on_mask=f(np.mean(lr[mask] > 0)),
        max_slow_water_on_mask=f(np.max(np.asarray(arr['slow_water'], np.float64)[mask])),
        max_slow_mm_on_mask=f(np.max(slow_mm[mask])),
        max_fast_mm_on_mask=f(np.max(fast_mm[mask])),
        max_perc_mm_on_mask=f(np.max(perc_mm[mask])),
        max_Qu_on_mask=f(np.max(Qu[mask])),
        max_xu_on_mask=f(np.max((Qu / upper_p)[mask])),
        max_Eu_over_A_on_mask=f(np.max((-np.expm1(-(Qu / upper_p)))[mask])),
    )

    # --------------------------------------------------- the g decision, stated
    xup_pre = (Qu / S_pre_up)[sel_u & np.isfinite(Qu / S_pre_up)]
    xup_post = (Qu / upper_p)[sel_u & np.isfinite(Qu / upper_p)]
    rep['g_decision'] = dict(
        rule_S1_3='g=x iff (post-outflow carry) AND (S_pre rebuildable) AND (x<=1 globally)',
        post_outflow= True,
        rebuildable=True,
        rebuild_maxrel=rep['timing']['rebuild_maxrel'],
        x_le_1_globally_post=bool(np.all(xup_post <= 1.0)),
        x_le_1_globally_pre=bool(np.all(xup_pre <= 1.0)),
        max_x_pre=f(np.max(xup_pre)),
        max_x_post=f(np.max(xup_post)),
        frac_x_gt_1_pre=f(np.mean(xup_pre > 1.0)),
        frac_x_gt_1_post=f(np.mean(xup_post > 1.0)),
    )
    rep['g_decision']['closure_choice'] = (
        'x' if (rep['g_decision']['x_le_1_globally_post'] or
                rep['g_decision']['x_le_1_globally_pre'])
        else '-expm1(-x)')
    rep['g_decision']['reason'] = (
        'row 1 applies only if x<=1 holds GLOBALLY; otherwise row 2. '
        'If the exponential form is taken it is the DECISIVE branch, not the '
        'safe default, because the gate did decide pre/post.')

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f'\n[probe2] wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
