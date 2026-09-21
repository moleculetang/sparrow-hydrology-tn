"""Independent recomputation of the envelope's criteria -- SECOND implementation.

WHAT INDEPENDENCE MEANS HERE, AND WHAT IT DOES NOT
--------------------------------------------------
This module imports NONE of this round's modules: not `common20`, not
`closures_r`, not `eventlib`, not `round_tools`, not either phase.  It re-derives
the criteria from the artifacts on disk with a from-scratch implementation.

The part that is genuinely re-implemented is the part where the previous round's
B1 defect lived: the EVENT WINDOW.  `eventlib.base_peak` uses four `searchsorted`
bounds with pinned `side` arguments; this module walks the date axis explicitly
and selects by comparison.  Two different mechanisms, one specification, and the
specification is asserted (`n_peak == (t_end - t_start).days + 2`, `n_base == 7`).

WHAT THIS MODULE CANNOT INDEPENDENTLY REDO, AND WHY THAT IS STATED RATHER THAN
PAPERED OVER: the `legal` conjuncts require re-installing `closures_r`'s kernel
into the model and running the frozen ledger.  That re-installation is exactly
what `gate_noop.py` already does, as its own registered producer.  Duplicating it
here would create a second place that binds the kernel, which is the failure mode
this round's discipline exists to prevent.  So the conjunct VERDICTS are checked
here for consistency against the recorded measurements and the frozen tolerances,
and that limitation is written down rather than hidden.

Source of every previous-round constant: the values here were typed from
`20260918_4/reports/stage_a_fingerprints.json`, `reports/frozen_anchors.json` and
the pre-registration, NOT imported -- an audit that imports its subject's
constants is not an audit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

UP = Path(r'E:\SPARROW\5_Test')
R = UP / '20260919_2'
EVENTS = UP / '20260918_4/data/stage_a_events.parquet'
PANEL = UP / '20260918_1/reports/ammonia_TN_quality_only.parquet'
MASK = UP / '20260919_1/data/phase1_eligible_mask.parquet'
ANCHOR = UP / '20260916_2/outputs/C0_s1/daily_station_mass_water.parquet'

PRIMARY_ARM = 'A:routed:D3'
TN_PRE_DAYS = 7
MIN_UNITS = 12
F3_GAP = 30
NULL_LABEL = 'null_delta_1'

MEDIAN_OBS_RATIO = 1.2758737517831669
BETA_OBS = 0.004063600875414098
ALPHA_OBS = -0.0704045722214265
C_PEAK_P = 2.094720376603729
A_P = 1.023217381861253
C_PEAK_OBS = 3.11
J1A_MIN_CLOSURE = 0.5
MONTHLY_GATE = 0.005
LABEL_TOL = 1e-6
NETWORK_SCALE_TOL = 1e-10
# The non-negativity conjunct is `min(...) >= -1e-7`.  The first version of this
# audit re-typed it as `+1e-7`, which made `min == 0` fail and reported all 14
# points as violating conjunct 3 when the envelope's own record shows `min == 0`
# at every point.  A re-typed constant is the residual risk of an independent
# implementation, and the sign of a tolerance is exactly where it hides.
NEG_TOL = -1e-7
UPTAKE_TOL = 1e-7


def _eq(a, b, tol=1e-12):
    """Both-nan is agreement.  `limit_delta_0` drives every concentration to 0, so
    every ratio there is 0/0 and every criterion value is nan on both sides."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


def eligible():
    ev = pd.read_parquet(EVENTS)
    ev = ev[(ev.arm == PRIMARY_ARM) & (ev.obs_status == 'OK')].copy()
    ev = ev.sort_values(['station_key', 'event_id']).reset_index(drop=True)
    assert len(ev) == 214 and ev.station_key.nunique() == 15, (len(ev), ev.station_key.nunique())
    return ev


def window(cbase_col, dates, values, t_start, t_end):
    """The frozen window, walked explicitly: base `[t0-7d, t0)`, peak `[t0, t1+1d]`.

    No `searchsorted`, no `side` argument, no slice arithmetic -- the whole class of
    off-by-one that produced the previous round's B1 defect is absent by method,
    not by care.
    """
    t0 = np.datetime64(pd.Timestamp(t_start).normalize(), 'D')
    t1 = np.datetime64(pd.Timestamp(t_end).normalize(), 'D')
    lo = t0 - np.timedelta64(TN_PRE_DAYS, 'D')
    hi = t1 + np.timedelta64(1, 'D')
    v = np.asarray(values, float)
    b = v[(dates >= lo) & (dates < t0)]
    p = v[(dates >= t0) & (dates <= hi)]
    b = b[np.isfinite(b)]
    p = p[np.isfinite(p)]
    return (float(np.median(b)) if b.size else np.nan,
            float(np.max(p)) if p.size else np.nan, int(b.size), int(p.size))


def events_from_replay(rep, ev):
    piv = rep.pivot_table(index='date', columns='station_key', values='p').sort_index()
    dates = np.asarray(pd.to_datetime(piv.index).values.astype('datetime64[D]'))
    rows = []
    for r in ev.itertuples():
        v = piv[r.station_key].to_numpy(float)
        cb, cp, nb, npk = window(None, dates, v, r.t_start, r.t_end)
        want = (pd.Timestamp(r.t_end).normalize() - pd.Timestamp(r.t_start).normalize()).days + 2
        assert npk == want, ('PEAK_WINDOW_LENGTH_WRONG', r.event_id, npk, want)
        assert nb == TN_PRE_DAYS, ('BASE_WINDOW_LENGTH_WRONG', r.event_id, nb)
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         event_rank=int(r.event_rank), T_interevent=float(r.T_interevent),
                         q_peak=float(r.q_peak), c_base=cb, c_peak=cp))
    return pd.DataFrame(rows).sort_values(['station_key', 'event_id']).reset_index(drop=True)


def f1(d, xcol):
    """within-station demeaned OLS, re-derived rather than imported."""
    d = d[['station_key', 'dc', xcol, 'c_base']].dropna().copy()
    if len(d) < MIN_UNITS:
        return None
    cnt = d.station_key.map(d.station_key.value_counts())
    d = d[cnt >= 2]
    if d.station_key.nunique() < 3 or len(d) < MIN_UNITS:
        return None
    for c in ('dc', xcol, 'c_base'):
        d[c] = d[c] - d.groupby('station_key')[c].transform('mean')
    X = np.column_stack([np.ones(len(d)), d[xcol].to_numpy(float), d.c_base.to_numpy(float)])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return float(np.linalg.lstsq(X, d.dc.to_numpy(float), rcond=None)[0][1])


def f3(d, maxgap):
    d = d[['station_key', 'event_rank', 'c_peak', 'c_base', 'q_peak', 'T_interevent']].dropna()
    d = d[(d.c_base > 0) & (d.c_peak > 0) & (d.q_peak > 0)]
    ys, xs = [], []
    for _st, sub in d.sort_values(['station_key', 'event_rank']).groupby('station_key'):
        prev = None
        for r in sub.itertuples():
            if prev is not None and r.T_interevent <= maxgap:
                ys.append(np.log(r.c_peak / r.c_base) - prev[0])
                xs.append(np.log(r.q_peak / prev[1]))
            prev = (np.log(r.c_peak / r.c_base), r.q_peak)
    y, x = np.asarray(ys, float), np.asarray(xs, float)
    if len(y) < MIN_UNITS:
        return None
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), y, rcond=None)[0][0])


def obs_monthly():
    p = pd.read_parquet(PANEL)
    p = p.assign(date=p.monitoring_time.dt.tz_localize(None).dt.normalize())
    d = p.groupby(['station_key', 'date'], as_index=False).TN.mean()
    d = d[(d.date.dt.year >= 2021) & (d.date.dt.year <= 2024)]
    d = d.assign(y=d.date.dt.year, mo=d.date.dt.month)
    return d.groupby(['station_key', 'y', 'mo']).TN.mean().rename('obs')


def obs_anchors(ev):
    """Re-derive the observation side from the raw panel, independently."""
    p = pd.read_parquet(PANEL)
    p = p.assign(d=p.monitoring_time.dt.tz_localize(None).values.astype('datetime64[D]'))
    got = np.full(len(ev), np.nan)
    cb = np.full(len(ev), np.nan)
    cp = np.full(len(ev), np.nan)
    for st, sub in p.groupby('station_key'):
        s = sub.sort_values('d')
        d = np.asarray(s.d.values, dtype='datetime64[D]')
        v = np.asarray(s.TN, float)
        for i in np.where(ev.station_key.values == st)[0]:
            b, pk, nb, npk = window(None, d, v, ev.t_start.values[i], ev.t_end.values[i])
            if np.isfinite(b) and b > 0 and np.isfinite(pk):
                got[i] = pk / b
                cb[i] = b
                cp[i] = pk
    ref = ev.obs_ratio.to_numpy(float)
    ok = np.isfinite(got) & np.isfinite(ref)
    return dict(n_reproduced=int(ok.sum()), n_expected=int(np.isfinite(ref).sum()),
                max_abs_diff_ratio=float(np.max(np.abs(got[ok] - ref[ok]))),
                c_base_median=float(np.median(cb[np.isfinite(cb)])),
                c_peak_median=float(np.median(cp[np.isfinite(cp)])),
                median_ratio=float(np.median(got[np.isfinite(got)])))


def main():
    ev = eligible()
    elig = pd.read_parquet(MASK)
    elig = elig[elig.eligible][['station_key', 'date']].copy()
    elig['date'] = pd.to_datetime(elig.date).dt.normalize()
    obs_m = obs_monthly()
    env = json.loads((R / 'reports/phase1_envelope.json').read_text(encoding='utf-8'))
    sc = json.loads((R / 'reports/phase1_scores.json').read_text(encoding='utf-8'))

    A = {}
    OA = obs_anchors(ev)
    if OA['max_abs_diff_ratio'] != 0.0:
        raise SystemExit('OBS_ANCHORS_DO_NOT_REPRODUCE %.6g' % OA['max_abs_diff_ratio'])

    for lab in env['points']:
        rep = pd.read_parquet(R / 'data' / ('phase1_replay_%s.parquet' % lab))
        t = events_from_replay(rep, ev)
        t = t.assign(dc=t.c_peak - t.c_base)
        f = np.isfinite(t.c_base) & np.isfinite(t.c_peak)
        m = elig.merge(rep[['station_key', 'date', 'p']], on=['station_key', 'date'],
                       how='left', validate='one_to_one')
        m = m.assign(y=m.date.dt.year, mo=m.date.dt.month)
        z = obs_m.to_frame().join(
            m.groupby(['station_key', 'y', 'mo']).p.mean().rename('pred'), how='inner').dropna()
        e = z.pred.to_numpy(float) - z.obs.to_numpy(float)
        o = z.obs.to_numpy(float)
        A[lab] = dict(
            A=float((t.c_peak / t.c_base)[f].median()), n_events=int(f.sum()),
            c_base=float(t.c_base[f].median()), c_peak=float(t.c_peak[f].median()),
            beta=f1(t, 'T_interevent'), alpha=f3(t, F3_GAP),
            nse=float(1.0 - np.mean(e ** 2) / np.var(o)),
            mean_concentration=float(m.p.to_numpy(float).mean()),
            max_abs_peak_diff_vs_null=None)

    n = A[NULL_LABEL]
    if abs(n['A'] - A_P) > 1e-12:
        raise SystemExit('NULL_A_IS_NOT_THE_FROZEN_BASELINE %.15g' % n['A'])
    if abs(n['c_peak'] - C_PEAK_P) > 1e-9:
        raise SystemExit('NULL_CPEAK_IS_NOT_THE_FROZEN_BASELINE %.15g' % n['c_peak'])

    rep_n = pd.read_parquet(R / 'data' / ('phase1_replay_%s.parquet' % NULL_LABEL))
    tn = events_from_replay(rep_n, ev)
    rows, agree = {}, {}
    for lab, v in A.items():
        rep = pd.read_parquet(R / 'data' / ('phase1_replay_%s.parquet' % lab))
        t = events_from_replay(rep, ev)
        j = t[['station_key', 'event_id']].assign(dp=t.c_peak.to_numpy() - tn.c_peak.to_numpy())
        G = (v['A'] - n['A']) / (MEDIAN_OBS_RATIO - n['A'])
        s = sc['points'][lab]
        rows[lab] = dict(
            A=v['A'], c_base=v['c_base'], c_peak=v['c_peak'], G_event=float(G),
            j1a_holds=bool(G >= J1A_MIN_CLOSURE),
            j1b_holds=bool(abs(v['c_peak'] - C_PEAK_OBS) <= abs(n['c_peak'] - C_PEAK_OBS)),
            beta=v['beta'], alpha=v['alpha'],
            d_beta=float(abs(v['beta'] - BETA_OBS)) if v['beta'] is not None else None,
            d_beta_null=float(abs(n['beta'] - BETA_OBS)),
            d_alpha=float(abs(v['alpha'] - ALPHA_OBS)) if v['alpha'] is not None else None,
            d_alpha_null=float(abs(n['alpha'] - ALPHA_OBS)),
            nse_degradation=float(n['nse'] - v['nse']),
            mean_concentration_relative_change=float(
                abs(v['mean_concentration'] - n['mean_concentration'])
                / n['mean_concentration']),
            max_abs_peak_diff_vs_null=float(np.max(np.abs(j.dp.to_numpy()))))
        rows[lab]['j2_holds'] = (None if v['beta'] is None
                                 else bool(rows[lab]['d_beta'] <= rows[lab]['d_beta_null']))
        rows[lab]['j3_holds'] = (None if v['alpha'] is None
                                 else bool(rows[lab]['d_alpha'] <= rows[lab]['d_alpha_null']))
        rows[lab]['j4_holds'] = bool(rows[lab]['nse_degradation'] <= MONTHLY_GATE)
        rows[lab]['j5_holds'] = bool(
            rows[lab]['mean_concentration_relative_change'] <= MONTHLY_GATE)
        agree[lab] = dict(
            A=_eq(rows[lab]['A'], s['amp']['A']),
            c_base=_eq(v['c_base'], s['amp']['c_base']),
            c_peak=_eq(v['c_peak'], s['amp']['c_peak']),
            beta=_eq(v['beta'], s['beta']),
            alpha=_eq(v['alpha'], s['alpha']),
            j1a=bool(rows[lab]['j1a_holds'] == s['j1a_holds']),
            j1b=bool(rows[lab]['j1b_holds'] == s['j1b_holds']),
            j2=bool(rows[lab]['j2_holds'] == s['j2_holds']),
            j3=bool(rows[lab]['j3_holds'] == s['j3_holds']),
            j4=bool(rows[lab]['j4_holds'] == s['j4_holds']),
            j5=bool(rows[lab]['j5_holds'] == s['j5_holds']),
            peak_diff_vs_null=_eq(rows[lab]['max_abs_peak_diff_vs_null'],
                                  s['max_abs_peak_diff_vs_null']))

    # the conjunct verdicts, checked against the recorded measurements
    legal = {}
    for lab, p in env['points'].items():
        g = p['legal']
        legal[lab] = dict(
            conj1=bool(g['local_balance_max_kg'] <= 1e-6),
            conj2=bool(abs(g['network_balance_kg']) <= g['network_scale_kg'] * NETWORK_SCALE_TOL),
            conj3=bool(min(g['min_over_channels'].values()) >= NEG_TOL
                       and g['R_min'] >= NEG_TOL),
            conj4=bool(g['max_uptake_minus_demand'] <= UPTAKE_TOL),
            conj5=bool(g['source_label_sum_errors_max'] <= LABEL_TOL
                       and not g['source_label_channels_missing']),
            matches_envelope=bool(
                (g['local_balance_max_kg'] <= 1e-6
                 and abs(g['network_balance_kg']) <= g['network_scale_kg'] * NETWORK_SCALE_TOL
                 and min(g['min_over_channels'].values()) >= NEG_TOL
                 and g['R_min'] >= NEG_TOL
                 and g['max_uptake_minus_demand'] <= UPTAKE_TOL
                 and g['source_label_sum_errors_max'] <= LABEL_TOL
                 and not g['source_label_channels_missing']) == g['all_conjuncts']))

    # The one implication that MUST hold, stated so it is checked rather than
    # assumed: if the cap binds on NO cell of the whole domain, the kernel is the
    # frozen one and the event table must be bitwise the null's.  The converse is
    # deliberately NOT asserted -- capping cells far from any event still moves the
    # peak through the slow `M` channel, which is the nonlocal feedback the
    # pre-registration's section 2.2 flagged and P1's falsification confirmed.
    no_cap_implies_identical = {}
    for lab, v in rows.items():
        total = env['points'][lab]['capped']['n_capped_cells']
        no_cap_implies_identical[lab] = dict(
            n_capped_cells_domain=int(total),
            max_abs_peak_diff_vs_null=float(v['max_abs_peak_diff_vs_null']),
            implication_holds=bool(total > 0 or v['max_abs_peak_diff_vs_null'] == 0.0))
    # the anchor replay must still be the frozen stored table
    a = pd.read_parquet(ANCHOR)[['station_key', 'date', 'concentration_mg_l']]
    rp = rep_n.rename(columns={'p': 'p_rep'})
    j = a.merge(rp, on=['station_key', 'date'], validate='one_to_one')
    anchor = dict(n_rows=int(len(j)), max_abs_diff=float(
        np.max(np.abs(j.concentration_mg_l.to_numpy() - j.p_rep.to_numpy()))))

    out = dict(audit=True, second_implementation=True,
               imports_of_this_round=[],
               observation_anchors=OA, null_point=n,
               points=rows, agreement_with_envelope=agree,
               all_agree=bool(all(all(x for k, x in v.items() if k != 'peak_diff_vs_null')
                                  for v in agree.values())),
               legal_verdicts=legal,
               all_legal_verdicts_match=bool(all(v['matches_envelope'] for v in legal.values())),
               no_cap_implies_identical=no_cap_implies_identical,
               all_implications_hold=bool(all(v['implication_holds']
                                              for v in no_cap_implies_identical.values())),
               anchor_replay_check=anchor,
               limitations=('the `legal` conjuncts are re-checked from the recorded '
                            'measurements, not re-derived: re-installing the kernel is '
                            '`gate_noop.py`\'s registered job and duplicating the binding '
                            'here would create a second place that installs it.'))
    (R / 'reports/audit_envelope.json').write_text(
        json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False, default=str),
        encoding='utf-8')
    bad = {k: {kk: vv for kk, vv in v.items() if vv is False} for k, v in agree.items()}
    bad = {k: v for k, v in bad.items() if v}
    print('AUDIT all_agree=%s legal_match=%s anchor=%s'
          % (out['all_agree'], out['all_legal_verdicts_match'], anchor), flush=True)
    if bad:
        print('DISAGREEMENTS', json.dumps(bad, ensure_ascii=False), flush=True)
    print('AUDIT_DONE', flush=True)


if __name__ == '__main__':
    main()
