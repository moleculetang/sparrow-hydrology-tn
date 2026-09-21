"""Phase 0 probe 3: is the ~16.5% pi==1 census hydrology or a float floor?

S1.3 and S3.3 both require this to be measured, not argued: the admissible
criterion replaced the old SAT_BOUND gate with "does pi==1 come from float
saturation or from real water".  So: split the ACTIVE cells by x>=1 and look at
what V_u and Q_u actually are on each side.

If x>=1 lives where V_u is at the denormal floor (1e-200) and Q_u is ~0, the
ratio is physically meaningless there -- both numerator and denominator are
numerical residue.  If instead V_u is O(1 mm) and Q_u genuinely exceeds it, the
reach really does turn over its upper store more than once a day.
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
OUT = RUN / 'reports/probe_saturation.json'
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
    return {'n': int(a.size), **{f'p{p}': f(z) for p, z in zip(ps, v)},
            'frac_eq_0': f(np.mean(a == 0.0)),
            'frac_lt_1e_100': f(np.mean(a < 1e-100)),
            'frac_lt_1e_10': f(np.mean(a < 1e-10))}


def main():
    rep = {}
    arr = {n: np.load(CACHE / f'{n}.npy', mmap_mode='r', allow_pickle=False)
           for n in ['upper_water', 'soil_water_mm', 'percolation', 'fast_water',
                     'slow_water', 'contact', 'fast_fraction', 'lower_release',
                     'area_ha']}
    w = pd.read_parquet(HYDRO, columns=['date', 'reach_id', 'upper_response_storage_mm'])
    w = w.sort_values(['date', 'reach_id'])
    assert np.array_equal(w.date.to_numpy().reshape(-1, NR)[:, 0].astype('datetime64[D]')[:ND],
                          np.load(CACHE / 'dates.npy', allow_pickle=False))

    area_ha = np.asarray(arr['area_ha'], np.float64)
    fast_mm = np.asarray(arr['fast_water'], np.float64) / (area_ha[None, :] * 10.0)
    perc_mm = np.asarray(arr['percolation'], np.float64)
    Qu = fast_mm + perc_mm
    upper = np.asarray(arr['upper_water'], np.float64)
    soil = np.asarray(arr['soil_water_mm'], np.float64)
    contact = np.asarray(arr['contact'], np.float64)
    ff = np.asarray(arr['fast_fraction'], np.float64)
    lr = np.asarray(arr['lower_release'], np.float64)

    act = contact > 0
    interiors = np.zeros_like(act)
    interiors[1:] = True
    sel = act & interiors

    x = np.full(upper.shape, np.nan)
    x[sel] = (Qu / upper)[sel]
    hi = sel & np.isfinite(x) & (x >= 1.0)
    lo = sel & np.isfinite(x) & (x < 1.0)

    rep['split'] = dict(n_hi=int(hi.sum()), n_lo=int(lo.sum()),
                        frac_hi=f(hi.sum() / (hi.sum() + lo.sum())))

    rep['hi_side'] = dict(
        V_upper_mm=dst(upper[hi]), Q_fast_mm=dst(fast_mm[hi]),
        Q_perc_mm=dst(perc_mm[hi]), Q_u_mm=dst(Qu[hi]),
        soil_mm=dst(soil[hi]), x=dst(x[hi]),
        frac_Qu_lt_1e_10=f(np.mean(Qu[hi] < 1e-10)),
        frac_Qu_lt_1e_6=f(np.mean(Qu[hi] < 1e-6)),
        frac_V_lt_1e_100=f(np.mean(upper[hi] < 1e-100)),
        frac_V_lt_1e_10=f(np.mean(upper[hi] < 1e-10)),
        frac_V_eq_0_exactly=f(np.mean(upper[hi] == 0.0)),
    )
    rep['lo_side'] = dict(
        V_upper_mm=dst(upper[lo]), Q_fast_mm=dst(fast_mm[lo]),
        Q_perc_mm=dst(perc_mm[lo]), Q_u_mm=dst(Qu[lo]),
        soil_mm=dst(soil[lo]), x=dst(x[lo]),
    )

    # The decisive conditional: on the x>=1 set, is the OUTFLOW real?
    # A cell that genuinely turns over >1/day must carry a real flux.
    q_hi = Qu[hi]
    rep['decisive'] = dict(
        question=('on x>=1, is Q_u real water or numerical residue?  '
                  'threshold 1e-6 mm/day = 1e-3 m3/km2/day, i.e. negligible'),
        frac_hi_with_Qu_ge_1e_6=f(np.mean(q_hi >= 1e-6)),
        frac_hi_with_Qu_ge_1e_3=f(np.mean(q_hi >= 1e-3)),
        frac_hi_with_Qu_ge_1e_0=f(np.mean(q_hi >= 1.0)),
        median_Qu_on_hi=f(np.median(q_hi)),
        median_V_on_hi=f(np.median(upper[hi])),
        # what does the ratio look like when the outflow is real?
        x_on_real_outflow=dst(np.where(q_hi >= 1e-3, x[hi], np.nan)),
        verdict=None,
    )
    r = rep['decisive']
    if r['frac_hi_with_Qu_ge_1e_3'] < 0.01:
        r['verdict'] = ('FLOAT_FLOOR: essentially every x>=1 cell has Q_u negligible; '
                        'the ratio is residue/residue and carries no hydrology')
    elif r['frac_hi_with_Qu_ge_1e_3'] > 0.5:
        r['verdict'] = ('REAL_TURNOVER: most x>=1 cells carry a real flux; the upper '
                        'store genuinely turns over more than once per day')
    else:
        r['verdict'] = 'MIXED'

    # and the soil denominator on the same cells -- the same cells, other V
    with np.errstate(invalid='ignore'):
        xs = Qu / soil
    rep['same_cells_soil_denominator'] = dict(
        x_soil_on_hi_cells=dst(xs[hi]),
        frac_xs_ge_1=f(np.mean(xs[hi][np.isfinite(xs[hi])] >= 1.0)),
    )

    # lower_release on the hi set -- the slow-path structural difference size
    rep['slow_path_difference_size'] = dict(
        lower_release_on_hi=dst(lr[hi]),
        lower_release_on_lo=dst(lr[lo]),
        note=('the frozen kernel releases slow = L*l everywhere; the new kernel '
              'releases L*g(Q_s/V_s).  Their ratio is g/l, so report l to size it.'),
    )

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f'\n[probe3] wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
