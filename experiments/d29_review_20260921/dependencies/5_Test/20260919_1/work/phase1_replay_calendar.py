"""Phase 1 -- four strictly paired replays at ONE frozen parameter vector per start.

    {C0_s0, C0_s1} x {monthfirst, uniform_daily}          = 4 arms
    C0_s1 x uniform_daily, UNCOMPENSATED                  = 1 diagnostic arm

Nothing is fitted.  The ONLY thing that differs between the two calendars is the
four land-input attributes `inp`, `demand`, `tag_inputs`, `tag_demand`, and they
are overwritten at runtime, because `scripts/structure_model.py` sits in the
75-entry `frozen_hashes` list enforced by `fit_worker.py:11-18` and the reference
round is read-only here.

WHY THE COMPENSATION IS MANDATORY AND NOT A COSMETIC FIX
-------------------------------------------------------
`structure_model.py:9-15` computes I_d = I_m / n_d on every day of the month.  The
float64 ascending sum of n_d identical copies is NOT I_m: measured residues are
inp 1.118e-08, demand 2.794e-09, tag_inputs 9.313e-09 kg, and the monthly sum is
not bitwise equal in 768 of 768 months.  Left alone, that residue makes the
ACCEPTANCE CRITERION a function of the calendar -- exactly the failure this round
exists to remove.  The compensated construction moves the residue to the last day
(see common19's Sterbenz argument) and closes every month bitwise.  Both paths are
reported; every downstream statistic uses the compensated one.

HARD GATES, IN ORDER.  Any failure stops the round; the readings are NOT to be
reinterpreted as a calendar effect.
  1  monthly totals bitwise equal for inp / demand / tag_inputs (compensated)
  2  the uncompensated residue is measured and reported (not hidden)
  3  the anchor: C0_s1 x monthfirst reproduces _2's stored daily table at 1e-12,
     with the same row count, and the monthfirst arm is PROVEN untouched
  4  the common eligible mask is asserted by sha256 before anything is computed
  5  same-calendar two-start agreement (a harness self-check, not a result)
  6  concentrations only; no mass column is written into any product
"""
import json

import numpy as np
import pandas as pd
import torch

import common19 as C

R = C.ROUND
PEER = C.PEER
ANCHOR_TABLE = PEER / 'outputs/C0_s1/daily_station_mass_water.parquet'
ANCHOR_RMSE = 1.1248754588360006
ANCHOR_CORR = 0.2958196063992402
ANCHOR_CME = 1.1667851845622874
TOL = 1e-12
END_YEAR = 2024
STARTS = ('C0_s0', 'C0_s1')
CALENDARS = ('monthfirst', 'uniform_daily')


def metric(g):
    """byte-for-byte the arithmetic of _5/scripts/replay_exposure_levels.py::metrics.

    Kept for provenance and NOT called on this path: it is the registered
    definition of the `p` column this round's replays carry, so it stays next to
    the code that writes them.  Phase 1 needs no fit statistic.
    """
    y = g.y.to_numpy(float); p = g.p.to_numpy(float); e = p - y
    b = float(e.mean()); sy = float(y.std()); sp = float(p.std())
    cov = float(np.mean((y - y.mean()) * (p - p.mean())))
    mse = float(np.mean(e * e))
    return dict(n=len(g), bias=b, rmse=np.sqrt(mse), mse=mse,
                correlation=cov / (sp * sy) if sp * sy else np.nan,
                centered_mse=float(np.var(p - y)),
                nse=1 - mse / sy ** 2 if sy else np.nan)


def forward(model, tag, arrays=None):
    """One arm: fixed parameters through the same daily block audit_job.py runs."""
    if arrays is not None:
        C.apply_arrays(model, arrays)
    x = C.parameters(tag)
    bounds = model.bounds
    if not all(a <= v <= bb for v, (a, bb) in zip(x, bounds)):
        raise SystemExit('PARAMETER_OUTSIDE_BOUNDS ' + tag)
    meta = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(END_YEAR)].copy()
    with torch.no_grad():
        d = model.daily_boundary(torch.tensor(x), meta)
    ri = d['record'].numpy(); di = d['day_index'].numpy()
    F = d['mass'].numpy(); V = d['water'].numpy()
    t = pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[ri],
                          date=model.data.dates[di],
                          p=1000 * F / V, water_m3_day=V))
    assert not any('mass' in c or 'load' in c or 'kg' in c for c in t.columns), t.columns
    return t


def main():
    pre = json.loads((R / 'reports/phase0_acceptance.json').read_text(encoding='utf-8'))
    pre_sha = C.sha(R / 'reports/预注册_判据与门槛.md')
    if pre_sha != pre['pre_registration']['sha256']:
        raise SystemExit('PRE_REGISTRATION_CHANGED_AFTER_PHASE0')
    S_M = float(pre['s_m_freeze']['selected'])
    mask_sha = C.sha(R / 'data/phase1_eligible_mask.parquet')
    if mask_sha != pre['eligible_set']['sha256']:
        raise SystemExit('ELIGIBLE_MASK_CHANGED_AFTER_PHASE0')
    S = dict(phase='phase1_replay_calendar', n_fits=0, fit_worker_calls=0,
             pre_registration_sha256=pre_sha, S_M_kg=S_M,
             eligible_mask_sha256=mask_sha, not_a_fit=True)

    # ---- gate 1 / 2: the two construction paths ---------------------------
    ref = C.build('C0_s1', 'monthfirst')
    data = ref.data
    rr = np.asarray(data.pilot_indices)
    closure = {}
    for comp in (True, False):
        arr = C.uniform_daily_arrays(data, rr, compensated=comp)
        closure['compensated' if comp else 'registered_uncompensated'] = {
            k: dict(worst_abs_kg=C.monthly_closure(data, arr, k, rr)[0],
                    n_months_not_bitwise_equal=C.monthly_closure(data, arr, k, rr)[1])
            for k in ('inp', 'demand', 'tag_inputs')}
    for v in closure['compensated'].values():
        if v['n_months_not_bitwise_equal'] != 0:
            raise SystemExit('GATE1_MONTHLY_TOTAL_NOT_BITWISE ' + json.dumps(v))
    S['monthly_closure'] = closure
    S['gate1_monthly_totals_bitwise_equal'] = True

    # the registered monthfirst construction, asserted NOT to have been altered
    reg_inp = np.zeros_like(np.asarray(data.contact, float))
    reg_inp[np.asarray(data.starts)] = np.asarray(data.source, float)
    if not np.array_equal(np.asarray(ref.inp, float), reg_inp):
        raise SystemExit('MONTHFIRST_ARM_ALTERED')
    S['monthfirst_arm_untouched'] = True

    # ---- the arms ---------------------------------------------------------
    arms, tab = {}, {}
    for tag in STARTS:
        m = C.build(tag, 'monthfirst')
        tab[(tag, 'monthfirst')] = forward(m, tag)
        arms['%s|monthfirst' % tag] = tab[(tag, 'monthfirst')]
    for tag in STARTS:
        m = C.build(tag, 'uniform_daily')
        a = C.uniform_daily_arrays(m.data, np.asarray(m.data.pilot_indices), True)
        tab[(tag, 'uniform_daily')] = forward(m, tag, a)
        arms['%s|uniform_daily' % tag] = tab[(tag, 'uniform_daily')]
    m = C.build('C0_s1', 'uniform_daily')
    a = C.uniform_daily_arrays(m.data, np.asarray(m.data.pilot_indices), False)
    raw = forward(m, tag='C0_s1', arrays=a)
    arms['C0_s1|uniform_daily_uncompensated'] = raw

    # ---- gate 3: the anchor ----------------------------------------------
    stored = pd.read_parquet(ANCHOR_TABLE)[['station_key', 'date', 'concentration_mg_l']]
    stored = stored.rename(columns={'concentration_mg_l': 'p_stored'})
    mine = tab[('C0_s1', 'monthfirst')].rename(columns={'p': 'p_replay'})
    cmp = stored.merge(mine, on=['station_key', 'date'], validate='one_to_one')
    if len(cmp) != len(stored):
        raise SystemExit('ANCHOR_ROW_COUNT_CHANGED %d != %d' % (len(cmp), len(stored)))
    d_elem = float(np.max(np.abs(cmp.p_replay.to_numpy() - cmp.p_stored.to_numpy())))
    anchor = dict(anchor_rmse=ANCHOR_RMSE, replay_rmse=None, abs_diff_rmse=None,
                  abs_diff_correlation=None, abs_diff_centered_mse=None,
                  max_abs_elementwise_concentration=d_elem,
                  n_compared_rows=int(len(cmp)), tolerance=TOL,
                  source='20260916_2\\outputs\\C0_s1\\daily_station_mass_water.parquet',
                  comparison='elementwise concentration vs the stored C0_s1 daily table')
    anchor['passed'] = bool(d_elem <= TOL)
    if not anchor['passed']:
        raise SystemExit('GATE3_ANCHOR_FAILED ' + json.dumps(anchor))
    S['anchor_gate'] = anchor
    S['gate3_anchor_passed'] = True
    print('ANCHOR_GATE PASS  n=%d  max|dc|=%.3e' % (len(cmp), d_elem), flush=True)

    # ---- gate 5: same-calendar two-start agreement ------------------------
    twostart = {}
    for cal in CALENDARS:
        z = tab[('C0_s0', cal)].merge(tab[('C0_s1', cal)], on=['station_key', 'date'],
                                      suffixes=('_s0', '_s1'), validate='one_to_one')
        twostart[cal] = dict(n=int(len(z)),
                             max_abs_dc=float(np.max(np.abs(z.p_s0 - z.p_s1))),
                             median_abs_dc=float(np.median(np.abs(z.p_s0 - z.p_s1))),
                             max_rel=float(np.max(np.abs(z.p_s0 - z.p_s1) / z.p_s1)))
    spread = max(v['max_abs_dc'] for v in twostart.values())
    cal_effect = float(np.max(np.abs(
        tab[('C0_s1', 'uniform_daily')].p - tab[('C0_s1', 'monthfirst')].p)))
    S['two_start_consistency'] = dict(
        per_calendar=twostart,
        spread_band_max_abs_dc=spread,
        calendar_effect_max_abs_dc=cal_effect,
        calendar_effect_exceeds_start_spread=bool(cal_effect > spread),
        role=('harness self-check: if two starts at the SAME calendar disagreed by more than the '
              'calendar effect, the replay apparatus would be the finding and the round must '
              'stop rather than read a calendar effect'))
    if not S['two_start_consistency']['calendar_effect_exceeds_start_spread']:
        raise SystemExit('GATE5_START_SPREAD_SWAMPS_CALENDAR_EFFECT')
    print('two-start spread %.3e vs calendar effect %.3e' % (spread, cal_effect), flush=True)

    # ---- gate 4: every arm covers the frozen eligible mask ---------------
    mask = pd.read_parquet(R / 'data/phase1_eligible_mask.parquet')
    elig = mask[mask.eligible][['station_key', 'date']]
    cover = {}
    for k, t in arms.items():
        z = elig.merge(t[['station_key', 'date']], on=['station_key', 'date'],
                       how='left', indicator=True)
        # n_rows_outside_mask is the count of arm rows the frozen mask excludes
        # (a row of the replay whose station-day is NOT eligible).  It is a
        # coverage diagnostic, not an error: the mask is deliberately narrower
        # than the calendar.  The name was `n_extra_rows` in the run that wrote
        # reports/phase1_replay.json, which reads as if those rows were surplus
        # ELIGIBLE rows -- they are the complement.
        cover[k] = dict(n_required=int(len(elig)),
                        n_missing=int((z._merge == 'left_only').sum()),
                        n_rows_outside_mask=int(len(t) - int((z._merge != 'left_only').sum())))
        if cover[k]['n_missing']:
            raise SystemExit('GATE4_ARM_DROPPED_ELIGIBLE_ROWS %s' % k)
    S['arm_coverage_of_the_eligible_mask'] = cover
    S['gate4_common_eligible_set'] = True

    # ---- ledgers: the endpoint channels under both layers -----------------
    led = {}
    for tag in STARTS:
        m = C.build(tag, 'monthfirst')
        a = m.ledger(C.parameters(tag))
        m2 = C.build(tag, 'uniform_daily')
        C.apply_arrays(m2, C.uniform_daily_arrays(m2.data, np.asarray(m2.data.pilot_indices), True))
        b = m2.ledger(C.parameters(tag))
        led[tag] = {}
        for name, r in (('monthfirst', a), ('uniform_daily', b)):
            errs = {k: float(v) for k, v in r['source_label_sum_errors'].items()}
            led[tag][name] = dict(
                source_label_sum_errors=errs,
                M_kg=errs['M'], M_over_S_M=errs['M'] / S_M,
                f_b_passes=bool(errs['M'] / S_M <= 1e-6),
                f_c_passes=bool(errs['M'] <= 1e-6),
                local_balance_max_kg=float(r['local_balance_max_kg']),
                network_balance_kg=float(r['network_balance_kg']))
    S['endpoint_channels'] = dict(
        per_arm=led, S_M_kg=S_M,
        reading=('monthfirst reports the UNCHANGED absolute gate; uniform_daily reports BOTH '
                 'the absolute and the relative reading, side by side, as the plan requires'),
        note=('the absolute 1e-6 kg acceptance is unchanged and is the only scientific gate; '
              'the relative continuation condition is a different threshold and is labelled so'))
    print('endpoint M: monthfirst %s  uniform_daily %s' %
          ({t: '%.3e' % led[t]['monthfirst']['M_kg'] for t in STARTS},
           {t: '%.3e' % led[t]['uniform_daily']['M_kg'] for t in STARTS}), flush=True)

    # ---- write the four (plus one) replays -------------------------------
    written = {}
    for key, t in arms.items():
        tag, cal = key.split('|')
        name = 'phase1_replay_%s_%s.parquet' % (tag[-2:], cal)
        path = R / 'data' / name
        t.to_parquet(path, index=False)
        written[key] = dict(path='data/' + name, sha256=C.sha(path), n_rows=int(len(t)),
                            columns=list(t.columns))
    S['replays'] = written
    # the compensated / uncompensated contrast at the LEVEL OF THE MONTHLY OUTPUT
    z = tab[('C0_s1', 'uniform_daily')].merge(raw, on=['station_key', 'date'],
                                              suffixes=('_c', '_r'), validate='one_to_one')
    mon = z.assign(y=z.date.dt.year, m=z.date.dt.month).groupby(['station_key', 'y', 'm']).agg(
        c=('p_c', 'mean'), r=('p_r', 'mean')).reset_index()
    S['compensation_contrast'] = dict(
        daily_max_abs_dc=float(np.max(np.abs(z.p_c - z.p_r))),
        daily_n_rows_differing=int((z.p_c != z.p_r).sum()),
        monthly_max_abs_dc=float(np.max(np.abs(mon.c - mon.r))),
        monthly_n_groups_differing=int((mon.c != mon.r).sum()),
        n_monthly_groups=int(len(mon)),
        reading=('the residue is a last-bit effect at both scales; downstream statistics use the '
                 'compensated path, and the registered path is reported for the record'))

    S['gates'] = dict(gate1_monthly_totals_bitwise_equal=True, gate2_residue_reported=True,
                      gate3_anchor_passed=True, gate4_common_eligible_set=True,
                      gate5_two_start_consistency=True, gate6_concentrations_only=True)
    S['all_gates_passed'] = True
    C.write_json(R / 'reports/phase1_replay.json', S)
    print('PHASE1_REPLAY_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
