"""20260919_5 -- sections 2.7, 2.8 and C7: level vs variance, catchment k, dCbar_pred.

WHY SECTION 2.7 EXISTS
----------------------
The round's whole question is whether the device removes a LEVEL without removing the
STRUCTURE that buys event amplitude.  A monthly NSE that collapses is consistent with two
very different worlds -- the level moved, or the shape moved -- and round 4 never separated
them, because it only ever computed `nse`.  So for every device and every beta this script
reports THREE numbers on the frozen eligible station-month grid:

  * the LEVEL   -- mean(C_cand)/mean(C_null)
  * the AMPLITUDE -- Var(C_cand)/Var(C_null)
  * the LEVEL-REMOVED NSE -- the candidate month series recentred onto the baseline mean

and the plan is explicit that only a RECOVERED level-removed NSE is evidence that the
collapse was a level effect.  If it is not recovered, this round may not claim that
neutralisation fixes the monthly scale, and must say so.

THE MASS-WEIGHTING DIAGNOSTIC, AND A LINE THIS SCRIPT DOES NOT CROSS
-------------------------------------------------------------------
The exact form of the concentration/weighting mismatch is `cov(dmass_t, 1/water_t)`.  That
numerator is a concentration change TIMES A WATER VOLUME, i.e. a mass, and section 0.1
keeps load off every criterion in this round.  The resolution is not to drop the diagnostic
-- the plan registers it -- but to keep it where it belongs: it is computed here, reported
under a name that says what it is, and used in NO gate.  No load is reported as a
standalone quantity anywhere, and nothing in `verdict_beta.py` reads this file.

SECTION 2.8, THE THING NOBODY HAS MEASURED
------------------------------------------
`c_base_L1` is about 32% of `c_base_L3` at the event base medians, so roughly two thirds of
the concentration the criteria score arrives from UPSTREAM reaches.  A per-reach constant
`k_r` therefore only behaves as a per-STATION constant to the extent that the catchment's
`k_r` agree -- and that spatial consistency has never been measured in this lineage.  It is
reported here as a mass-weighted mean and CV per station, beside a SHARE-weighted mean, and
the tension is written down rather than discovered later: exact-constant `k` leaves every
ratio statistic exactly invariant and only moves the level, so ALL of this device's control
and ALL of its damage come from the cross-reach variation of `k_r` -- one quantity.

C7, AND WHY IT IS CHEAP
-----------------------
`routing.py:130` is linear in `local` and its coefficients do not contain `Xi`, so ONE
adjoint pass of the FROZEN model yields `share[t,r] = d(mean p)/d local[t,r]` valid for
every beta and every device.  `dlocal` then comes from the land-phase recursion alone.
"""
import numpy as np
import pandas as pd

# ORDER IS LOAD-BEARING: `common23` injects the frozen vendor directory into `sys.path`,
# so the vendor import must come AFTER it.  See the note in `phase0_freeze.py`.
import common23 as C
import closures as _closed
# `xi_k`, not `xi_base`: `xi_star` exists only in `xi_k`, which section 3.2 item 1 makes the
# single constructor of `Xi_star`.  Same defect as `phase1_full.py` had, fixed here before
# it could fire rather than after -- `xi_base` has no `xi_star` to call.
import xi_k as XI

if not hasattr(XI, 'xi_star'):
    raise SystemExit('XI_MODULE_DOES_NOT_CONSTRUCT_XI_STAR -- got %r'
                     % getattr(XI, '__name__', XI))

R = C.ROUND
OUT = R / 'reports'
TAG = C.TAG
N_STATIONS = 15


def _p(msg, **kw):
    """Print, always flushing.  `flush` is accepted and ignored -- three call sites in this
    file pass it from force of habit while the definition did not take it, the same latent
    `TypeError` `phase1_full.py` actually died of.  Found by RUNNING, fixed here before it
    could fire."""
    kw.pop('flush', None)
    print(msg, flush=True, **kw)


def recentred_nse(z, znull):
    """Pooled NSE after recentring the candidate onto the baseline mean.

    This is the ONLY number in this file that bears on "was the collapse a level effect".
    `z` and `znull` are on the SAME joined index, so a mean shift is the only thing removed.
    """
    if not z.index.equals(znull.index):
        raise SystemExit('RECENTRING_ON_A_DIFFERENT_INDEX')
    o = z.obs.to_numpy(float)
    p = z.pred.to_numpy(float)
    pn = znull.pred.to_numpy(float)
    p_r = p - p.mean() + pn.mean()
    return float(1.0 - np.mean((p_r - o) ** 2) / np.var(o))


def record_mask(model, elig):
    """Boolean `(n_record,)` selecting the eligible station-days of the daily record.

    The record's row order is built by `temporal_model.daily_metadata`: metadata rows in
    the calendar's own order, each expanded over its own days.  Rather than re-derive that
    order (and be silently wrong if it ever changes), the row identities are READ back out
    of it as `(station_key, day)` pairs and tested for membership in the eligible grid.
    """
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    c, rec, _w = model.daily_metadata(meta)
    keys = meta.station_key.to_numpy().astype(str)[rec.numpy()]
    days = np.asarray(model.data.dates)[c['ti'].numpy()]
    want = set(zip(elig.station_key.astype(str),
                   elig.date.to_numpy().astype('datetime64[D]').astype(str)))
    m = np.fromiter(((k, str(d)[:10]) in want for k, d in zip(keys, days)),
                    dtype=bool, count=len(keys))
    if int(m.sum()) != len(elig):
        raise SystemExit('RECORD_MASK_SELECTS_%d_BUT_THE_ELIGIBLE_GRID_HAS_%d'
                         % (int(m.sum()), len(elig)))
    return m, len(keys)


def land_local(B, Xi_star):
    """`local = fast + slow` per (day, reach) at the modulated hazard.

    `closures.scan` computes `prob = -expm1(-min(risk, 700))` with `risk = h` in the frozen
    `mode=False` path, and the round's kernel change is `risk -> risk * Xi`.  Feeding
    `h * Xi_star` into the FROZEN scan therefore reproduces exactly the modulated product
    -- including the clip -- without reimplementing a single line of the recurrence, and
    without installing anything globally.  The identity `h*Xi*` == `(h*k)*Xi_raw` is the
    same one Phase 0 asserts in both spellings, so a factor-of-k error cannot hide here.
    """
    h = np.ascontiguousarray(B['h'] * Xi_star, float)
    fast, slow, _a, _p = _closed.scan(h, B['s'], B['f'], B['k'],
                                      B['lower_release'], B['inp'], B['demand'], B['cap'])
    return np.ascontiguousarray(fast + slow)


def main():
    rep = {'phase': 'level_variance', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'reads_gates_from': 'phase1_full.json (no recomputation of any gate)',
           'load_red_line': 'no load is reported as a standalone quantity and no number in '
                            'this file enters any gate; the section 2.7 diag is a '
                            'concentration change times a water volume, and is named as '
                            'such rather than quoted as a load'}

    p1 = C.read_json(OUT / 'phase1_full.json')
    kf = C.read_json(OUT / 'k_field.json')['k_field']
    rep['source_sha'] = dict(phase1_full=C.sha(OUT / 'phase1_full.json'),
                             k_field=C.sha(OUT / 'k_field.json'),
                             daily_layers=C.sha(OUT / 'daily_layers.parquet'))

    model = C.build(TAG)
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    if not C.hazard_is_frozen():
        raise SystemExit('Predictor.hazard_IS_REBOUND')
    elig = C.eligible_grid()
    obs_m = C.obs_monthly()
    daily = pd.read_parquet(OUT / 'daily_layers.parquet')
    rep['daily_layers'] = dict(n_rows=int(len(daily)), columns=list(daily.columns),
                               n_eligible=int(len(elig)),
                               n_points=int(daily[['device', 'beta']].drop_duplicates()
                                            .shape[0]))

    # ------------------------------------------------------------- the null
    null = daily[(daily.device == 'N1') & (daily.beta == 0.0)]
    if len(null) != len(elig):
        raise SystemExit('NULL_FRAME_ROW_COUNT %d vs %d' % (len(null), len(elig)))
    _, znull = C.monthly_join(null, elig, obs_m)
    null_m = null.pL3.to_numpy(float)
    rep['null'] = dict(n_eligible_rows=int(len(null)), n_station_months=int(len(znull)),
                       mean_concentration=float(null_m.mean()),
                       var_concentration=float(np.var(null_m)),
                       nse=C.nse_of(znull.obs.to_numpy(float), znull.pred.to_numpy(float)))

    # --------------------------------------------- section 2.7, per device per beta
    _p('=== section 2.7: level / variance / level-removed NSE ===', flush=True)
    lv = {}
    for device in C.DEVICES:
        for beta in C.BETA_GRID:
            sub = daily[(daily.device == device) & (daily.beta == float(beta))]
            if len(sub) != len(elig):
                raise SystemExit('POINT_ROW_COUNT %s %g %d' % (device, beta, len(sub)))
            _, z = C.monthly_join(sub, elig, obs_m)
            pv = sub.pL3.to_numpy(float)
            level = float(pv.mean() / null_m.mean())
            amp = float(np.var(pv) / np.var(null_m))
            dmass = (pv - null_m) * sub.water_m3_day.to_numpy(float)
            invw = 1.0 / sub.water_m3_day.to_numpy(float)
            cc = float(np.corrcoef(dmass, invw)[0, 1])
            lv['%s|%g' % (device, beta)] = dict(
                device=device, beta=float(beta),
                level_ratio=level, amplitude_ratio=amp,
                var_cand=float(np.var(pv)), var_null=float(np.var(null_m)),
                nse=C.nse_of(z.obs.to_numpy(float), z.pred.to_numpy(float)),
                nse_level_removed=recentred_nse(z, znull),
                n_station_months=int(len(z)),
                corr_dmass_invwater=cc,
                cov_dmass_invwater=float(np.cov(dmass, invw, ddof=0)[0, 1]),
                dmass_units='concentration_change_times_water_volume',
                dmass_note='reported only as the numerator of the section 2.7 weighting '
                           'mismatch; it is never quoted as a load and no gate reads it')
    rep['level_variance'] = lv
    for device in C.DEVICES:
        for beta in (0.0, 0.5, -0.5):
            r = lv['%s|%g' % (device, beta)]
            _p('   %-4s b=%+5.2f  level=%.6f  amp=%.6f  nse=%.6f  nse_level_removed=%.6f  '
               'corr(dm,1/w)=%.4f'
               % (device, beta, r['level_ratio'], r['amplitude_ratio'], r['nse'],
                  r['nse_level_removed'], r['corr_dmass_invwater']))

    # ------------------------------------------------ section 2.8: catchment k
    _p('=== section 2.8: catchment k consistency ===', flush=True)
    cm = C.catchment_map(model)
    s2r, _reaches = C.station_reaches(model)
    Wt_ref = C.window_mask(B['dates'], C.REF_YEARS)
    av0h_h = np.where(Wt_ref[:, None], B['a0'] * B['h'], 0.0).sum(axis=0)
    own_share = {}
    adj = C.routing_adjoint(model, B, mask=record_mask(model, elig)[0])
    sh_w = np.abs(adj['share']).sum(axis=0)
    rep['C7_adjoint'] = dict(shape=adj['shape'], n_rows_selected=adj['n_rows_selected'],
                             n_rows=adj['n_rows'], objective=adj['objective'],
                             note=adj['note'])
    catch = {}
    for sk, r0 in sorted(s2r.items()):
        mem = np.asarray(cm['catchment'][int(r0)], int)
        own_share[sk] = float(av0h_h[int(r0)] / av0h_h[mem].sum())
        w = av0h_h[mem]
        ks = np.array([np.asarray(kf['N1']['points']['%g' % 0.5]['k'], float)[int(r)]
                       for r in mem])
        mw = float((w * ks).sum() / w.sum()) if w.sum() > 0 else None
        sw_ = sh_w[mem]
        shm = float((sw_ * ks).sum() / sw_.sum()) if sw_.sum() > 0 else None
        catch[sk] = dict(reach_1based=int(r0) + 1, n_catchment=int(len(mem)),
                         k_mass_weighted_mean=mw,
                         k_share_weighted_mean=shm,
                         k_mass_weighted_cv=(None if mw in (None, 0.0) else
                                             float(np.sqrt((w * (ks - mw) ** 2).sum()
                                                           / w.sum()) / abs(mw))),
                         k_spread=float(ks.max() - ks.min()),
                         own_reach_share_of_catchment_mobilisation=own_share[sk],
                         k_own_reach=float(ks[int(np.where(mem == int(r0))[0][0])]))
    rep['catchment_k'] = dict(
        per_station=catch, beta=0.5, device='N1',
        tension='an EXACT constant k leaves every ratio statistic exactly invariant and '
                'moves only the level, so all of this device control and all of its damage '
                'come from the CROSS-REACH variation of k_r -- the same quantity.  This is '
                'written here so it is not discovered after the fact.',
        own_reach_share_note='c_base_L1 is about 32% of c_base_L3 at the event base '
                             'medians, so a station own reach supplies only part of the '
                             'scored concentration')
    _p('   stations=%d  own-reach share of catchment mobilisation: min=%.4f med=%.4f max=%.4f'
       % (len(catch), min(own_share.values()),
          float(np.median(list(own_share.values()))), max(own_share.values())))

    # ------------------------------------------------- C7: dCbar_pred vs measured
    _p('=== C7: dCbar_pred vs measured dCbar ===', flush=True)
    local0 = land_local(B, np.ones_like(B['h']))
    share = adj['share']
    pred = {}
    for device in C.DEVICES:
        for beta in C.BETA_GRID:
            k = np.asarray(kf[device]['points']['%g' % beta]['k'], float)
            if beta == 0.0:
                dlocal = np.zeros_like(local0)
            else:
                dlocal = land_local(B, XI.xi_star(G, beta, k)) - local0
            d_pred = float((share * dlocal).sum())
            meas = (p1['points']['%s|%g' % (device, beta)]['mean_concentration']
                    - p1['baseline']['mean_concentration'])
            pred['%s|%g' % (device, beta)] = dict(
                device=device, beta=float(beta), dCbar_pred=d_pred, dCbar_measured=meas,
                ratio=(None if d_pred == 0 else float(meas / d_pred)),
                abs_error=float(abs(meas - d_pred)),
                linearisation_holds=bool(abs(meas - d_pred) <= 1e-6 * max(1.0, abs(meas))))
    rep['C7_predicted_vs_measured'] = pred
    for device in C.DEVICES:
        for beta in (0.0, 0.5, -0.5):
            r = pred['%s|%g' % (device, beta)]
            _p('   %-4s b=%+5.2f  pred=%+.6e  measured=%+.6e  ratio=%s'
               % (device, beta, r['dCbar_pred'], r['dCbar_measured'],
                  'n/a' if r['ratio'] is None else '%.4f' % r['ratio']))

    rep['deviations'] = [
        'the section 2.7 weighting diagnostic is computed from concentration x water and '
        'is labelled as such; section 0.1 keeps load off every criterion and no gate in '
        'this round reads it',
        'the C7 adjoint mask is built by reading (station_key, day) back out of '
        'temporal_model.daily_metadata rather than re-deriving the record row order; the '
        'mask is asserted to select exactly the eligible grid',
        'land_local calls the FROZEN closures.scan with h -> h*Xi_star instead of running '
        'the modulated kernel globally; the two spellings produce the same clipped product',
    ]
    C.write_json(OUT / 'level_variance.json', rep)
    _p('=== wrote %s ===' % (OUT / 'level_variance.json'))
    return rep


if __name__ == '__main__':
    main()
