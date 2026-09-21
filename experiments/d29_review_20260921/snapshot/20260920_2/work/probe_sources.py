"""Phase 0 probe: source identity, water-balance pairing, storage timing.

Read-only.  Writes one JSON.  Nothing here is a criterion yet -- it is the
measurement that decides which criterion is even available.

Three questions, in this order:

  1. IDENTITY.  Which cache array is which producer column, bit for bit?
  2. BALANCE.   Which store does `percolation` actually leave and which does it
     enter?  This is the non-tautological pairing test.  (`upper_water/fast`
     equals the producer's own `upper_store_instantaneous_turnover_day` by
     algebra -- 86400/(100*10) == 86.4 -- so the turnover table cannot select
     a pairing; the balance can.)
  3. TIMING.    Is the exported storage the pre-outflow volume or the
     post-outflow carry?  The lower store is closed in known terms
     (dS_low = percolation - slow), so it decides without any assumption.

Also adjudicates the inert-mask question: does `contact <= 0` imply
`fast_water == 0` AND `percolation == 0`, or not?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
import pandas as pd

CACHE = ROOT / '5_Test/20260917_5/data/domains/FULL24'
HYDRO = ROOT / '5_Test/20260828_38/outputs/tn_hydrology_reach_daily.parquet'
OUT = RUN / 'reports/probe_sources.json'

ND, NR = 23376, 230


def f(x):
    """JSON-safe scalar."""
    x = float(x)
    if not np.isfinite(x):
        return None if np.isnan(x) else ('inf' if x > 0 else '-inf')
    return x


def q(a, ps=(0, 1, 50, 99, 100)):
    """Quantiles as a plain dict."""
    v = np.percentile(np.asarray(a, dtype=np.float64), ps)
    return {f'p{p}': f(v) for p, v in zip(ps, v)}


def main():
    rep = {}

    # ---------------------------------------------------------------- load
    arr = {}
    for name in ['soil_water_mm', 'upper_water', 'percolation', 'fast_water',
                 'slow_water', 'contact', 'fast_fraction', 'lower_release',
                 'h_day', 'area_ha']:
        arr[name] = np.load(CACHE / f'{name}.npy', mmap_mode='r', allow_pickle=False)
    dates = np.load(CACHE / 'dates.npy', allow_pickle=False)
    rep['cache'] = dict(dir=str(CACHE), shape=list(arr['upper_water'].shape),
                        dates_first=str(dates[0]), dates_last=str(dates[-1]))

    cols = ['date', 'reach_id', 'soil_storage_mm', 'upper_response_storage_mm',
            'lower_slow_storage_mm', 'percolation_to_lower_mm_day',
            'local_fast_response_m3_s', 'local_slow_response_m3_s',
            'precipitation_daily_mm', 'actual_aet_mm_day',
            'upper_store_instantaneous_turnover_day',
            'lower_store_instantaneous_turnover_day', 'catchment_area_km2']
    water = (pd.read_parquet(HYDRO, columns=cols)
             .sort_values(['date', 'reach_id']))
    rep['hydro'] = dict(rows=int(len(water)),
                        days=int(len(water) // NR),
                        date_first=str(water.date.iloc[0]),
                        date_last=str(water.date.iloc[-1]))
    assert len(water) % NR == 0, len(water)

    d = water.date.to_numpy().reshape(-1, NR)[:, 0].astype('datetime64[D]')
    rep['hydro']['calendar_matches_cache'] = bool(np.array_equal(d[:ND], dates))
    assert rep['hydro']['calendar_matches_cache'], 'calendar mismatch -- stop'

    def col(name, dtype=np.float64):
        return water[name].to_numpy(dtype).reshape(-1, NR)[:ND]

    soil_p = col('soil_storage_mm')
    upper_p = col('upper_response_storage_mm')
    lower_p = col('lower_slow_storage_mm')
    perc_p = col('percolation_to_lower_mm_day')
    fast_s = col('local_fast_response_m3_s')
    slow_s = col('local_slow_response_m3_s')
    rain = col('precipitation_daily_mm')
    aet = col('actual_aet_mm_day')
    turn_up = col('upper_store_instantaneous_turnover_day')
    turn_lo = col('lower_store_instantaneous_turnover_day')
    area_km2 = col('catchment_area_km2')
    area_ha = np.asarray(arr['area_ha'], dtype=np.float64)

    # ------------------------------------------------- 1. identity, bitwise
    def maxabs(a, b):
        return f(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    ident = {
        'soil_water_mm__eq__soil_storage_mm': bool(np.array_equal(arr['soil_water_mm'], soil_p)),
        'upper_water__eq__upper_response_storage_mm': bool(np.array_equal(arr['upper_water'], upper_p)),
        'percolation__eq__percolation_to_lower_mm_day': bool(np.array_equal(arr['percolation'], perc_p)),
        'fast_water__eq__local_fast_s_x86400': bool(np.array_equal(arr['fast_water'], fast_s * 86400.0)),
        'slow_water__eq__local_slow_s_x86400': bool(np.array_equal(arr['slow_water'], slow_s * 86400.0)),
        'area_ha__eq__area_km2_x100': bool(np.array_equal(area_ha, area_km2[0] * 100.0)),
    }
    ident['fast_water__maxabs'] = maxabs(arr['fast_water'], fast_s * 86400.0)
    ident['slow_water__maxabs'] = maxabs(arr['slow_water'], slow_s * 86400.0)
    ident['percolation__maxabs'] = maxabs(arr['percolation'], perc_p)
    ident['area_ha_maxrel'] = f(np.max(np.abs(area_ha - area_km2[0] * 100.0)
                                       / np.maximum(area_km2[0] * 100.0, 1e-12)))
    rep['identity'] = ident

    # ------------------------------------------------------- unit conversion
    # mm/day, same convention as the producer's own turnover definition.
    fast_mm = np.asarray(arr['fast_water'], np.float64) / (area_ha[None, :] * 10.0)
    slow_mm = np.asarray(arr['slow_water'], np.float64) / (area_ha[None, :] * 10.0)
    perc_mm = np.asarray(arr['percolation'], np.float64)
    rep['units'] = dict(
        fast_mm__eq__producer_local_fast_mm=bool(np.allclose(
            fast_mm, fast_s * 86.4 / area_km2, rtol=0, atol=0)),
        fast_mm_maxabs_vs_producer=f(np.max(np.abs(fast_mm - fast_s * 86.4 / area_km2))),
        slow_mm_maxabs_vs_producer=f(np.max(np.abs(slow_mm - slow_s * 86.4 / area_km2))),
        fast_mm_min=f(np.nanmin(fast_mm)), fast_mm_max=f(np.nanmax(fast_mm)),
        perc_mm_min=f(np.nanmin(perc_mm)), perc_mm_max=f(np.nanmax(perc_mm)),
        slow_mm_min=f(np.nanmin(slow_mm)), slow_mm_max=f(np.nanmax(slow_mm)),
    )

    # ----------------------------------------- 2. water balance, both stores
    # Only interior days (drop the first row of each column, which has no
    # previous state).  Residual uses finite differences of the exported state.
    def resid(state, inflow, outflow):
        dS = state[1:] - state[:-1]
        return np.abs(dS - (inflow[1:] - outflow[1:]))

    # LOWER: dS = percolation - slow.  No unknown term.  This one DECIDES.
    r_lower = resid(lower_p, perc_mm, slow_mm)
    # Same equation, fluxes lagged one day: if the exported state were the
    # START-of-day volume, day t's fluxes would close against S[t+1]-S[t]
    # read as S[t]-S[t-1].
    r_lower_lag = np.abs((lower_p[1:] - lower_p[:-1]) - (perc_mm[:-1] - slow_mm[:-1]))
    scale_lower = np.maximum(lower_p[1:], 1e-12)
    rep['balance_lower'] = dict(
        equation='dS_low = percolation - slow',
        same_day_max_abs=f(np.max(r_lower)),
        same_day_median_abs=f(np.median(r_lower)),
        same_day_max_rel=f(np.max(r_lower / scale_lower)),
        lagged_max_abs=f(np.max(r_lower_lag)),
        lagged_median_abs=f(np.median(r_lower_lag)),
        verdict=('END_OF_DAY_POST_OUTFLOW' if np.max(r_lower / scale_lower) < 1e-9
                 else ('START_OF_DAY_PRE_OUTFLOW' if np.max(r_lower_lag / scale_lower) < 1e-9
                       else 'NEITHER_CLOSES')),
    )

    # UPPER: dS = recharge - fast - percolation; recharge unknown, so recover it
    # and then check it against the soil store.
    recharge = (upper_p[1:] - upper_p[:-1]) + fast_mm[1:] + perc_mm[1:]
    rep['balance_upper'] = dict(
        equation='dS_up = recharge - fast - percolation  (recharge recovered)',
        recharge_q=q(recharge),
        recharge_negative_fraction=f(np.mean(recharge < 0)),
        recharge_max_abs=f(np.max(np.abs(recharge))),
    )
    # SOIL: dS_soil = rain - aet - recharge (the recovered one).  If the two
    # balances are consistent the same recharge closes both.
    r_soil = np.abs((soil_p[1:] - soil_p[:-1]) - (rain[1:] - aet[1:] - recharge))
    rep['balance_soil'] = dict(
        equation='dS_soil = rain - aet - recharge(recovered from upper)',
        max_abs=f(np.max(r_soil)),
        median_abs=f(np.median(r_soil)),
        max_rel=f(np.max(r_soil / np.maximum(soil_p[1:], 1e-12))),
        note='residual here is what the upper balance leaves unexplained in the soil store',
    )

    # ------------------------------------------------ 3. turnover, as declared
    # Producer's own denominator is `local_fast` alone.  Report BOTH candidates
    # so the tautology is visible rather than assumed.
    turn_cand_fast = arr['upper_water'] / np.maximum(fast_mm, 1e-300)
    turn_cand_qu = arr['upper_water'] / np.maximum(fast_mm + perc_mm, 1e-300)
    ok = np.isfinite(turn_up) & (turn_up > 0)
    rep['turnover'] = dict(
        producer_definition='upper_response_storage_mm / local_fast_mm_day',
        vs_fast_only=dict(
            identical=bool(np.array_equal(np.asarray(turn_cand_fast)[ok], turn_up[ok])),
            median_abs_log_ratio=f(np.median(np.abs(np.log(
                np.asarray(turn_cand_fast)[ok] / turn_up[ok])))),
            n=int(ok.sum())),
        vs_fast_plus_perc=dict(
            median_abs_log_ratio=f(np.median(np.abs(np.log(
                np.asarray(turn_cand_qu)[ok] / turn_up[ok])))),
            q90_abs_log_ratio=f(np.percentile(np.abs(np.log(
                np.asarray(turn_cand_qu)[ok] / turn_up[ok])), 90))),
        note=('vs_fast_only is an ALGEBRAIC IDENTITY (86400/(100*10)==86.4), so the '
              'producer turnover table cannot select a pairing; the balance above can.'),
        lower_vs_producer=dict(
            median_abs_log_ratio=f(np.median(np.abs(np.log(
                lower_p / np.maximum(slow_mm, 1e-300) / np.where(turn_lo > 0, turn_lo, np.nan))))),
            identical=bool(np.array_equal(
                (lower_p / np.maximum(slow_mm, 1e-300))[np.isfinite(turn_lo) & (turn_lo > 0)],
                turn_lo[np.isfinite(turn_lo) & (turn_lo > 0)]))),
    )

    # --------------------------------------- 4. inert mask: adjudicate R4'
    contact = np.asarray(arr['contact'], np.float64)
    ff = np.asarray(arr['fast_fraction'], np.float64)
    mask = contact <= 0
    act = ~mask
    rep['inert_mask'] = dict(
        n_inert=int(mask.sum()), n_active=int(act.sum()),
        frac_inert=f(mask.mean()),
        max_fast_water_on_mask=f(np.max(np.asarray(arr['fast_water'], np.float64)[mask])),
        max_percolation_on_mask=f(np.max(perc_mm[mask])),
        max_fast_fraction_on_mask=f(np.max(ff[mask])),
        max_fast_water_on_active=f(np.max(np.asarray(arr['fast_water'], np.float64)[act])),
        min_fast_water_on_active=f(np.min(np.asarray(arr['fast_water'], np.float64)[act])),
        max_upper_water_on_mask=f(np.max(np.asarray(arr['upper_water'], np.float64)[mask])),
        max_slow_water_on_mask=f(np.max(np.asarray(arr['slow_water'], np.float64)[mask])),
    )

    # ---------------------------------------------------------- 5. the x census
    # Both pairings, so the size of the R3' distortion is measured rather than
    # argued.  `frac_expm1_rounds_to_1` is the x where -expm1(-x) rounds to 1.0
    # in float64 (x >= 36.7); `frac_pi_eq_1_exactly` is the x where exp(-x)
    # underflows to 0 so that expm1 returns exactly -1.0 (x > 745).  Different
    # quantities; reported separately.
    out = {}
    for vname in ['upper_water', 'soil_water_mm']:
        V = np.asarray(arr[vname], np.float64)
        for qname, Q in (('fast', fast_mm), ('fast_plus_perc', fast_mm + perc_mm)):
            with np.errstate(divide='ignore', invalid='ignore'):
                x = Q / V
            xa = x[act & np.isfinite(x)]
            out[f'{vname}|{qname}'] = dict(
                min=f(np.min(xa)), p50=f(np.median(xa)), p90=f(np.percentile(xa, 90)),
                p99=f(np.percentile(xa, 99)), max=f(np.max(xa)),
                frac_x_ge_1=f(np.mean(xa >= 1.0)),
                frac_pi_ge_0p99=f(np.mean(-np.expm1(-xa) >= 0.99)),
                frac_pi_eq_1_exactly=f(np.mean(-np.expm1(-xa) == 1.0)),
                frac_expm1_rounds_to_1=f(np.mean(xa >= 36.7)),
            )
    rep['x_census_active_cells'] = out

    # also on the mask, to size the inert contribution under the new kernel
    V = np.asarray(arr['upper_water'], np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        xm = (fast_mm + perc_mm) / V
    xm = xm[mask & np.isfinite(xm)]
    rep['x_census_inert_cells'] = dict(
        n=int(xm.size), p50=f(np.median(xm)), p99=f(np.percentile(xm, 99)),
        frac_pi_ge_0p99=f(np.mean(-np.expm1(-xm) >= 0.99)),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f'\n[probe] wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
