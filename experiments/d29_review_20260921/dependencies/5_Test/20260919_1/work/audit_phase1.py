"""Independent recomputation of the Phase 1 headline numbers.

SECOND IMPLEMENTATION, ON PURPOSE AND BY DISCIPLINE
---------------------------------------------------
This file imports NONE of this round's modules.  Every constant is hand-copied
from the frozen sources, every window is re-derived with a boolean mask instead of
`np.searchsorted`, every regression is solved through the normal equations instead
of `np.linalg.lstsq`, every sd is `np.std(..., ddof=0)` on an explicit array, and
the sign-flip permutation is enumerated through a different bit-unpacking route.

A disagreement here is the finding; agreement is the claim that the numbers in
`reports/phase1_fingerprints.json` do not depend on one particular coding of the
recipe.  It does NOT re-derive the criteria, the thresholds or the verdict rule --
those are frozen in `reports/预注册_判据与门槛.md` and are read back, not re-chosen.

Hand-copied constants (each with its source):
  TN_PRE_DAYS = 7                       20260918_4/work/stage_a_events.py:92
  MIN_EVENT_RECORDS = 3                 .../stage_a_events.py:93
  F3 primary gap = 30                   20260918_4/work/stage_a_fingerprints.py
  MIN_UNITS = 12                        .../stage_a_fingerprints.py:75
  N_BOOT / SEED = 2000 / 20260917       .../stage_a_fingerprints.py:72-73
  NOISE_BAND = 0.0529 = 2 x 0.02647     reports/预注册_判据与门槛.md S3
  J5_MAX_DEGRADATION = 0.005            reports/预注册_判据与门槛.md S3
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(r'E:\SPARROW\5_Test\20260919_1')
UP = R.parent
D4 = UP / '20260918_4'
EVENTS = D4 / 'data/stage_a_events.parquet'
PANEL = UP / '20260918_1/reports/ammonia_TN_quality_only.parquet'
TOL = 1e-9

TN_PRE_DAYS = 7
MIN_EVENT_RECORDS = 3
MIN_UNITS = 12
F3_GAP = 30
N_BOOT = 2000
SEED = 20260917
NOISE_BAND = 0.0529
J5_MAX_DEGRADATION = 0.005
PRIMARY_ARM = 'A:routed:D3'
STARTS = ('C0_s0', 'C0_s1')
CAL = {'P': 'monthfirst', 'U': 'uniform_daily'}
START_YEAR, END_YEAR = 2021, 2024

CHECKS = []


def check(name, mine, theirs, tol=TOL):
    if theirs is None or (isinstance(theirs, float) and not np.isfinite(theirs)):
        ok = mine is None or not np.isfinite(mine)
        d = 0.0 if ok else np.inf
    else:
        d = abs(float(mine) - float(theirs))
        ok = bool(d <= tol)
    CHECKS.append(dict(name=name, mine=float(mine) if np.isfinite(mine) else None,
                       reported=theirs, abs_diff=float(d) if np.isfinite(d) else None,
                       tolerance=tol, agrees=ok))
    return ok


# ---------------------------------------------------------------- the recipe
def load_events():
    e = pd.read_parquet(EVENTS)
    e = e[(e.arm == PRIMARY_ARM) & (e.obs_status == 'OK')].copy()
    return e.sort_values(['station_key', 'event_id']).reset_index(drop=True)


def panel_frames():
    p = pd.read_parquet(PANEL)
    d = p.monitoring_time.dt.tz_localize(None)
    p = p.assign(day=d.values.astype('datetime64[D]'))
    daily = (p.assign(date=d.dt.normalize())
             .groupby(['station_key', 'date'], as_index=False)['TN'].mean())
    return p, daily


def window_values(dates, values, t0, t1, pre):
    """The frozen recipe's two windows, by boolean mask, not by searchsorted."""
    lo = t0 - np.timedelta64(pre, 'D')
    hi = t1 + np.timedelta64(1, 'D')
    b = values[(dates >= lo) & (dates < t0)]
    pk = values[(dates >= t0) & (dates <= hi)]
    return (float(np.median(b)) if b.size else np.nan,
            float(np.max(pk)) if pk.size else np.nan, int(b.size), int(pk.size))


def obs_ratio_recomputed(ev, raw):
    out = np.full(len(ev), np.nan)
    for st, sub in raw.groupby('station_key'):
        idx = np.where(ev.station_key.values == st)[0]
        if not len(idx):
            continue
        s = sub.sort_values('day')
        d = np.asarray(s.day.values, dtype='datetime64[D]')
        v = np.asarray(s.TN, float)
        for i in idx:
            t0 = np.datetime64(pd.Timestamp(ev.t_start.iloc[i]).normalize(), 'D')
            t1 = np.datetime64(pd.Timestamp(ev.t_end.iloc[i]).normalize(), 'D')
            b, pk, nb, npk = window_values(d, v, t0, t1, TN_PRE_DAYS)
            if np.isfinite(b) and b > 0 and npk >= MIN_EVENT_RECORDS:
                out[i] = pk / b
    return out


def model_base_peak(day_p, day_dates, t0, t1):
    """day_p: a (station, day) array on `day_dates`, both sorted ascending."""
    b, pk, nb, npk = window_values(day_dates, day_p, t0, t1, TN_PRE_DAYS)
    return (b, pk, pk - b, (pk / b if np.isfinite(b) and b > 0 else np.nan), nb, npk)


def build_matrix(replay):
    """(stations x days) of concentration, plus the day axis."""
    days = np.sort(replay.date.unique())
    stations = sorted(replay.station_key.unique())
    pos = {d: i for i, d in enumerate(days)}
    col = {s: j for j, s in enumerate(stations)}
    M = np.full((len(stations), len(days)), np.nan)
    M[replay.station_key.map(col).to_numpy(),
      replay.date.map(pos).to_numpy()] = replay.p.to_numpy(float)
    return stations, days.astype('datetime64[D]'), M


def ols_slope(y, X):
    """Least squares through the NORMAL EQUATIONS -- a different linear route."""
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)


def demean_by(y, codes, n):
    s = np.bincount(codes, weights=y, minlength=n)
    c = np.bincount(codes, minlength=n)
    return y - (s / np.maximum(c, 1))[codes]


def beta_f1(d):
    d = d[['station_key', 'T_interevent', 'c_base', 'dc']].dropna()
    d = d[d.station_key.map(d.station_key.value_counts()) >= 2]
    if len(d) < MIN_UNITS or d.station_key.nunique() < 3:
        return None
    codes, uniq = pd.factorize(d.station_key.values)
    Y = demean_by(d.dc.to_numpy(float), codes, len(uniq))
    X = np.column_stack([np.ones(len(d)),
                         demean_by(d.T_interevent.to_numpy(float), codes, len(uniq)),
                         demean_by(d.c_base.to_numpy(float), codes, len(uniq))])
    if np.linalg.matrix_rank(X) < 3:
        return None
    return float(ols_slope(Y, X)[1])


def alpha_f3(d, gap):
    d = d[['station_key', 'event_rank', 'c_peak', 'c_base', 'q_peak', 'T_interevent']].dropna()
    d = d[(d.c_base > 0) & (d.c_peak > 0) & (d.q_peak > 0)]
    ys, xs = [], []
    for _st, sub in d.sort_values(['station_key', 'event_rank']).groupby('station_key'):
        logr, q = None, None
        for row in sub.itertuples():
            cur_logr = float(np.log(row.c_peak / row.c_base))
            if logr is not None and row.T_interevent <= gap:
                ys.append(cur_logr - logr)
                xs.append(float(np.log(row.q_peak / q)))
            logr, q = cur_logr, float(row.q_peak)
    if len(ys) < MIN_UNITS:
        return None
    y = np.asarray(ys, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(xs, float)])
    return float(ols_slope(y, X)[0])


def sign_flip_p(delta):
    """All 2**n sign assignments, enumerated through a little-endian u4 byte view.

    A first attempt used a big-endian u2 view with n = 15 columns: columns 0-7
    then read value bits 8-15 and columns 8-14 read bits 0-6, so value bit 7 was
    never used and bit 15 was always zero.  That collapses the 2**15 assignments
    onto 2**14 distinct sign vectors, and it produced null_sd = 0.005954 against
    the fingerprint's 0.005968 -- an AUDIT bug that looked like a disagreement.
    `np.unique(null).size == 1 << n` is asserted below so the same collapse cannot
    pass unnoticed again.
    """
    d = np.asarray(delta, float)
    n = len(d)
    stat = float(d.mean())
    bits = np.unpackbits(np.arange(1 << n, dtype='<u4').view('u1').reshape(-1, 4),
                         axis=1, bitorder='little')[:, :n]
    signs = 2.0 * bits - 1.0
    null = (signs @ d) / n
    if np.unique(null).size != (1 << n):
        raise SystemExit('SIGN_FLIP_ENUMERATION_DEGENERATE %d of %d'
                         % (np.unique(null).size, 1 << n))
    return float(np.mean(np.abs(null) >= abs(stat) - 1e-15)), float(null.std())


def main():
    rep = json.loads((R / 'reports/phase1_replay.json').read_text(encoding='utf-8'))
    fp = json.loads((R / 'reports/phase1_fingerprints.json').read_text(encoding='utf-8'))
    ev = load_events()
    raw, daily = panel_frames()
    A = dict(present=True, checks=[])

    # ---- 0. the eligible event set matches what Phase 0 froze --------------
    n_exp = fp['eligible_events']['n']
    CHECKS.append(dict(name='n_eligible_events', mine=float(len(ev)), reported=float(n_exp),
                       abs_diff=0.0, tolerance=0.0, agrees=bool(len(ev) == n_exp)))

    # ---- 1. the observation recipe reproduces the lineage -----------------
    ore = obs_ratio_recomputed(ev, raw)
    ref = ev.obs_ratio.to_numpy(float)
    ok = np.isfinite(ore) & np.isfinite(ref)
    d = float(np.max(np.abs(ore[ok] - ref[ok]))) if ok.sum() else np.inf
    CHECKS.append(dict(name='obs_recipe_max_abs_diff', mine=d, reported=0.0,
                       abs_diff=d, tolerance=1e-12, agrees=bool(d <= 1e-12)))
    A_obs = float(np.median(ref))
    check('A_obs_from_the_lineage_column', A_obs, fp['frozen_targets']['A_obs'])

    # ---- 2. per-arm model base/peak, by boolean mask ----------------------
    tables = {}
    for st in STARTS:
        for arm in ('P', 'U'):
            nm = 'phase1_replay_%s_%s.parquet' % (st[-2:], CAL[arm])
            rp = pd.read_parquet(R / 'data' / nm)
            rp = rp.assign(date=rp.date.dt.normalize().values.astype('datetime64[D]'))
            stations, days, M = build_matrix(rp)
            col = {s: j for j, s in enumerate(stations)}
            rows = []
            for i in range(len(ev)):
                r = ev.iloc[i]
                if r.station_key not in col:
                    raise SystemExit('STATION_NOT_IN_REPLAY ' + str(r.station_key))
                t0 = np.datetime64(pd.Timestamp(r.t_start).normalize(), 'D')
                t1 = np.datetime64(pd.Timestamp(r.t_end).normalize(), 'D')
                b, pk, dc, ratio, nb, npk = model_base_peak(M[col[r.station_key]], days, t0, t1)
                # the recipe's peak window is [t_start, t_end + 1d] INCLUSIVE
                if nb != TN_PRE_DAYS or npk != int((t1 - t0) / np.timedelta64(1, 'D')) + 2:
                    raise SystemExit('AUDIT_WINDOW_LENGTH_UNEXPECTED %d %d' % (nb, npk))
                rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                                 event_rank=int(r.event_rank), T_interevent=float(r.T_interevent),
                                 q_peak=float(r.q_peak), c_base=b, c_peak=pk, dc=dc,
                                 amp_ratio=ratio, obs_amp_ratio=float(r.obs_ratio)))
            tables[(st, arm)] = pd.DataFrame(rows)

    # ---- 3. J1 -----------------------------------------------------------
    clos_rel = {}
    for st in STARTS:
        aP = float(np.median(tables[(st, 'P')].amp_ratio.to_numpy(float)))
        aU = float(np.median(tables[(st, 'U')].amp_ratio.to_numpy(float)))
        r = fp['J1_simulated_observed_event_amplitude_ratio']['per_start'][st]
        check('J1 A_P ' + st, aP, r['A_P'])
        check('J1 A_U ' + st, aU, r['A_U'])
        g = 1.0 - abs(aU - A_obs) / abs(aP - A_obs)
        check('J1 G_event ' + st, g, r['G_event'])
        clos_rel[st] = (abs(aP - A_obs) - abs(aU - A_obs)) / abs(aP - A_obs)
    # the reported magnitude is the MINIMUM over the two starts (conservative), so
    # it must be compared with the minimum, not with one start's value.
    check('J1 share_of_gap_closed(min over starts)', min(clos_rel.values()),
          fp['magnitude_of_the_calendar_effect']['J1_share_of_gap_closed'], tol=1e-9)

    # ---- 4. J3 / J4 ------------------------------------------------------
    for st in STARTS:
        bP = beta_f1(tables[(st, 'P')])
        bU = beta_f1(tables[(st, 'U')])
        r3 = fp['J3_F1_prior_accumulation_distance']['per_start'][st]
        check('J3 beta_P ' + st, bP, r3['sim_P'])
        check('J3 beta_U ' + st, bU, r3['sim_U'])
        check('J3 D_P ' + st, abs(bP - fp['frozen_targets']['beta_obs']), r3['D_P'])
        alP = alpha_f3(tables[(st, 'P')], F3_GAP)
        alU = alpha_f3(tables[(st, 'U')], F3_GAP)
        r4 = fp['J4_F3_consecutive_event_distance']['per_start'][st]
        check('J4 alpha_P ' + st, alP, r4['sim_P'])
        check('J4 alpha_U ' + st, alU, r4['sim_U'])
        check('J4 D_P ' + st, abs(alP - fp['frozen_targets']['alpha_obs_gap30']), r4['D_P'])

    # ---- 5. J2 -----------------------------------------------------------
    mask = pd.read_parquet(R / 'data/phase1_eligible_mask.parquet')
    mask = mask.assign(date=mask.date.dt.normalize().values.astype('datetime64[D]'))
    elig = mask[mask.eligible]
    dl = daily.assign(date=daily.date.dt.normalize().values.astype('datetime64[D]'))
    dl = dl[dl.date.isin(set(elig.date.unique()))]
    e2 = elig.merge(dl, on=['station_key', 'date'], how='inner')
    sd_obs = {}
    for st, g in e2.groupby('station_key'):
        sd_obs[st] = float(np.std(g.TN.to_numpy(float), ddof=0))
    check('J2 sd_obs max|d| vs Phase 0',
          max(abs(sd_obs[k] - v) for k, v in
              fp['J2_sd_obs_on_the_eligible_set']['values'].items()), 0.0, tol=1e-12)
    for st in STARTS:
        eP, eU = {}, {}
        for arm, tgt in (('P', eP), ('U', eU)):
            rp = pd.read_parquet(R / 'data' / ('phase1_replay_%s_%s.parquet'
                                               % (st[-2:], CAL[arm])))
            rp = rp.assign(date=rp.date.dt.normalize().values.astype('datetime64[D]'))
            j = elig.merge(rp[['station_key', 'date', 'p']], on=['station_key', 'date'],
                           how='inner', validate='one_to_one')
            for k, g in j.groupby('station_key'):
                tgt[k] = float(np.std(g.p.to_numpy(float), ddof=0))
        ks = sorted(eP)
        delta = np.array([abs(np.log(eU[k] / sd_obs[k])) - abs(np.log(eP[k] / sd_obs[k]))
                          for k in ks])
        r2 = fp['J2_station_sd_log_distance']['per_start'][st]
        check('J2 median delta ' + st, float(np.median(delta)), r2['median_delta_e_s'])
        p, nsd = sign_flip_p(delta)
        check('J2 sign-flip p ' + st, p, r2['permutation']['p_two_sided'], tol=1e-12)
        check('J2 null sd ' + st, nsd, r2['permutation']['null_sd'], tol=1e-12)

    # ---- 6. J5 -----------------------------------------------------------
    d2 = dl.assign(y=pd.DatetimeIndex(dl.date).year, m=pd.DatetimeIndex(dl.date).month)
    obs_m = d2.groupby(['station_key', 'y', 'm'])['TN'].mean()
    for st in STARTS:
        for arm in ('P', 'U'):
            rp = pd.read_parquet(R / 'data' / ('phase1_replay_%s_%s.parquet'
                                               % (st[-2:], CAL[arm])))
            rp = rp.assign(date=rp.date.dt.normalize().values.astype('datetime64[D]'))
            j = elig.merge(rp[['station_key', 'date', 'p']], on=['station_key', 'date'],
                           how='inner')
            j = j.assign(y=pd.DatetimeIndex(j.date).year, m=pd.DatetimeIndex(j.date).month)
            pm = j.groupby(['station_key', 'y', 'm'])['p'].mean()
            z = pd.DataFrame({'obs': obs_m, 'pred': pm}).dropna()
            e = z.pred.to_numpy(float) - z.obs.to_numpy(float)
            nse = 1.0 - float(np.mean(e * e)) / float(np.var(z.obs.to_numpy(float)))
            check('J5 NSE %s %s' % (st, arm), nse,
                  fp['J5_monthly_scale_degradation']['per_start'][st][arm]['nse'],
                  tol=1e-10)

    # ---- 7. the verdict rule, re-applied from the frozen thresholds -------
    lit = [fp[k]['holds_literal'] for k in
           ('J1_simulated_observed_event_amplitude_ratio', 'J2_station_sd_log_distance',
            'J3_F1_prior_accumulation_distance', 'J4_F3_consecutive_event_distance',
            'J5_monthly_scale_degradation')]
    cons = [fp[k]['holds_conservative'] for k in
            ('J1_simulated_observed_event_amplitude_ratio', 'J2_station_sd_log_distance',
             'J3_F1_prior_accumulation_distance', 'J4_F3_consecutive_event_distance',
             'J5_monthly_scale_degradation')]
    verdict_lit = ('QUALIFY_FOR_REFIT' if sum(lit) == 5 else
                   'CLOSE' if sum(lit) == 0 else 'PARTIAL')
    verdict_cons = ('QUALIFY_FOR_REFIT' if sum(cons) == 5 else
                    'CLOSE' if sum(cons) == 0 else 'PARTIAL')
    CHECKS.append(dict(name='verdict_literal', mine=None, reported=verdict_lit,
                       abs_diff=0.0, tolerance=0.0,
                       agrees=bool(verdict_lit == fp['verdict_by_reading']['literal'])))
    CHECKS.append(dict(name='verdict_conservative', mine=None, reported=verdict_cons,
                       abs_diff=0.0, tolerance=0.0,
                       agrees=bool(verdict_cons == fp['round_verdict']['value'])))

    bad = [c for c in CHECKS if not c['agrees']]
    out = dict(phase='audit_phase1', n_fits=0, fit_worker_calls=0, not_a_fit=True,
               independence=('no module of this round is imported; constants hand-copied; boolean '
                             'masks instead of searchsorted; normal equations instead of lstsq; '
                             'np.std instead of pandas .std; a different bit route for the exact '
                             'sign-flip enumeration'),
               n_checks=int(len(CHECKS)), n_disagreements=int(len(bad)),
               tolerance=TOL, checks=CHECKS,
               status='AGREES' if not bad else 'DISAGREES',
               disagreements=bad)
    (R / 'reports/audit_phase1.json').write_text(
        json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False, default=str),
        encoding='utf-8')
    print('checks %d  disagreements %d  %s' % (len(CHECKS), len(bad), out['status']), flush=True)
    for b in bad:
        print('  DISAGREE %s mine=%s reported=%s' % (b['name'], b['mine'], b['reported']), flush=True)
    print('AUDIT_PHASE1_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
