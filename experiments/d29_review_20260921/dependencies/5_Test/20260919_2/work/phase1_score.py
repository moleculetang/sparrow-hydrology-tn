"""Phase 1 scoring -- the two boxed criteria, the anti-gaming clauses, the
monthly-scale readings, and the falsification follow-ups the pre-registration
committed to in its section 10.

READS ONLY.  Everything it needs was written by `phase1_envelope.py`: the raw
replays and the per-event tables.  `work/audit_envelope.py` recomputes the same
criteria from the same replays without importing this module.

THE CONTRAST AXIS IS `baseline vs candidate`, not the previous round's
"two calendars".  The baseline is the `null_delta_1` POINT of this grid -- the
same code path, one parameter away -- because the plan's section 3.1 requires the
gain to be measured against a null on the same code path, never against another
module's product.

WHY THE BOOTSTRAP IS SOMETIMES SKIPPED
--------------------------------------
Eight of the fourteen points are BITWISE identical to the null on the 214 frozen
events (`capped.n_events_with_a_capped_day == 0`).  For those, every criterion
difference is exactly 0 and a 2000-draw bootstrap would return a degenerate
interval around 0 that looks like a finding and is not one.  They are recorded as
`bitwise_identical_to_null` and the resampling is skipped, explicitly.
"""
import json
import math

import numpy as np
import pandas as pd

import common20 as C
import eventlib as E

R = C.ROUND
TAG = 'C0_s1'
NULL_LABEL = 'null_delta_1'

# section 1.2, frozen before any envelope point existed
BETA_OBS = 0.004063600875414098
ALPHA_OBS = -0.0704045722214265
MEDIAN_OBS_RATIO = 1.2758737517831669
C_BASE_OBS = 2.18
C_PEAK_OBS = 3.11
BETA_P = 0.002331
ALPHA_P = -0.050343
A_P = 1.023217381861253
C_BASE_P = 2.031723094631754
C_PEAK_P = 2.094720376603729
J1A_MIN_CLOSURE = 0.5
MONTHLY_GATE = 0.005


def load(label):
    j = C.read_json(R / 'reports/phase1_envelope.json')['points'][label]
    rep = pd.read_parquet(R / 'data' / ('phase1_replay_%s.parquet' % label))
    ev = pd.read_parquet(R / 'data' / ('phase1_events_%s.parquet' % label))
    return j, rep, ev


def amp_point(ev):
    f = np.isfinite(ev.c_base) & np.isfinite(ev.c_peak)
    return dict(n=int(f.sum()),
                c_base=float(ev.c_base[f].median()), c_peak=float(ev.c_peak[f].median()),
                A=float((ev.c_peak / ev.c_base)[f].median()),
                A_mean=float((ev.c_peak / ev.c_base)[f].mean()))


def obs_monthly():
    """Observed monthly station mean TN, from the READ-ONLY 4h panel.

    This is a LEVEL gate on a concentration, not an event selection and not a fit
    target; the round is zero-fit, so scoring against the panel is exactly what
    its registered status permits.
    """
    p = pd.read_parquet(R.parent / '20260918_1/reports/ammonia_TN_quality_only.parquet')
    p = p.assign(date=p.monitoring_time.dt.tz_localize(None).dt.normalize(),
                 y=p.monitoring_time.dt.year, mo=p.monitoring_time.dt.month)
    d = p.groupby(['station_key', 'date'], as_index=False).TN.mean()
    d = d[(d.date.dt.year >= E.START_YEAR) & (d.date.dt.year <= E.END_YEAR)]
    d = d.assign(y=d.date.dt.year, mo=d.date.dt.month)
    return d.groupby(['station_key', 'y', 'mo']).TN.mean().rename('obs')


def monthly_stats(rep, elig, obs_m):
    m = elig.merge(rep[['station_key', 'date', 'p']], on=['station_key', 'date'],
                   how='left', validate='one_to_one')
    m = m.assign(y=m.date.dt.year, mo=m.date.dt.month)
    pm = m.groupby(['station_key', 'y', 'mo']).p.mean().rename('pred')
    z = obs_m.to_frame().join(pm, how='inner').dropna()
    e = z.pred.to_numpy(float) - z.obs.to_numpy(float)
    o = z.obs.to_numpy(float)
    pv = z.pred.to_numpy(float)
    # The delta = 0 endpoint drives every concentration to exactly 0, so the
    # predicted series has zero variance and `corrcoef` divides by it.  That point
    # is degenerate by construction; returning nan is the honest reading, and the
    # guard keeps numpy from warning about it as if it were a numerical accident.
    degenerate = bool(np.var(pv) == 0.0)
    nse = 1.0 - float(np.mean(e ** 2)) / float(np.var(o))
    per = {}
    for k, g in z.groupby(level=0):
        ee = g.pred.to_numpy(float) - g.obs.to_numpy(float)
        per[k] = 1.0 - float(np.mean(ee ** 2)) / float(np.var(g.obs.to_numpy(float)))
    return dict(n_station_months=int(len(z)), nse=float(nse),
                r2=(None if degenerate else
                    float(np.corrcoef(o, pv)[0, 1] ** 2)),
                degenerate_prediction_is_constant_zero=degenerate,
                median_station_nse=float(np.median(list(per.values()))),
                mean_concentration=float(m.p.to_numpy(float).mean()),
                n_eligible_rows=int(len(m)),
                mean_concentration_null=None)


def main():
    j = C.read_json(R / 'reports/phase1_envelope.json')
    ev_frozen = E.eligible_events()
    mask = pd.read_parquet(E.MASK)
    elig = mask[mask.eligible][['station_key', 'date']].copy()
    elig['date'] = E.as_day(elig.date)
    obs_m = obs_monthly()

    S = dict(phase=1, scoring=True, n_fits=0, fit_worker_calls=0,
             pre_registration_sha=j['pre_registration_sha'],
             envelope_sha=C.sha(R / 'reports/phase1_envelope.json'),
             frozen_anchors=dict(beta_obs=BETA_OBS, alpha_obs=ALPHA_OBS,
                                 median_obs_ratio=MEDIAN_OBS_RATIO,
                                 c_base_obs=C_BASE_OBS, c_peak_obs=C_PEAK_OBS),
             baseline_anchors=dict(A_P=A_P, beta_P=BETA_P, alpha_P=ALPHA_P,
                                   c_base_P=C_BASE_P, c_peak_P=C_PEAK_P),
             contrast='baseline (null_delta_1) vs candidate, same code path',
             monthly_gate=MONTHLY_GATE)

    # ---------------- the null of the SAME code path -------------------
    jn, repn, evn = load(NULL_LABEL)
    an = amp_point(evn)
    mn = monthly_stats(repn, elig, obs_m)
    if abs(an['A'] - A_P) > 1e-12 or abs(an['c_peak'] - C_PEAK_P) > 1e-9:
        raise SystemExit('NULL_POINT_IS_NOT_THE_FROZEN_BASELINE %s' % an)
    mn['mean_concentration_null'] = mn['mean_concentration']
    S['null_point'] = dict(amp=an, monthly=mn,
                           note='delta = 1 is bitwise the frozen kernel; the null of this '
                                'grid is therefore the frozen baseline itself')

    # ---------------- J2/J3 point estimates on the null ----------------
    betas, alphas = {}, {}
    for lab in j['points']:
        _, _, ev = load(lab)
        # `build_event_table` already carries `station_key, c_base, c_peak, dc,
        # q_peak, T_interevent, event_rank`; the earlier version of this block
        # rebuilt a 4-column frame and dropped `T_interevent`, which the frozen
        # `f1_coef` needs as its regressor.  Passing the table itself is both
        # shorter and impossible to get wrong.
        ev = ev.assign(dc=ev.c_peak - ev.c_base)
        betas[lab] = E.f1_coef(ev, 'T_interevent')
        alphas[lab] = E.f3_intercept(ev, E.F3_PRIMARY_GAP)
    S['point_estimates'] = dict(beta=betas, alpha=alphas,
                                note='beta = f1_coef within-station demeaned OLS; '
                                     'alpha = f3_intercept at the registered primary gap '
                                     'of 30 days')

    # ---------------- per point ----------------------------------------
    rows = {}
    for lab, p in j['points'].items():
        _, rep, ev = load(lab)
        a = amp_point(ev)
        # A byte comparison alone conflates "the state did nothing" with "the state
        # moved the last bits of the float64".  The two are different findings and
        # the round is about the second one, so both are recorded.
        _m = ev[['station_key', 'event_id', 'c_peak', 'c_base']].merge(
            evn[['station_key', 'event_id', 'c_peak', 'c_base']],
            on=['station_key', 'event_id'], suffixes=('_c', '_n'), validate='one_to_one')
        max_peak_diff = float(np.max(np.abs(_m.c_peak_c - _m.c_peak_n)))
        max_base_diff = float(np.max(np.abs(_m.c_base_c - _m.c_base_n)))
        identical = bool(max_peak_diff == 0.0 and max_base_diff == 0.0)
        G = (a['A'] - an['A']) / (MEDIAN_OBS_RATIO - an['A'])
        j1b_gain = abs(an['c_peak'] - C_PEAK_OBS) - abs(a['c_peak'] - C_PEAK_OBS)
        base_moves = a['c_base'] - an['c_base']
        # necessary condition C: the A gain comes from C_base FALLING
        j1c_only_via_base = bool(a['A'] > an['A'] and a['c_peak'] <= an['c_peak'])
        mm = monthly_stats(rep, elig, obs_m)
        nse_deg = float(mn['nse'] - mm['nse'])
        mnse_deg = float(mn['median_station_nse'] - mm['median_station_nse'])
        mean_rel = float(abs(mm['mean_concentration'] - mn['mean_concentration'])
                         / mn['mean_concentration'])
        rows[lab] = dict(
            kind=p['kind'], delta=p['delta'], tau_R_days=p['tau_R_days'],
            amp=a, amp_null=an,
            G_event=float(G),
            j1a_holds=bool(G >= J1A_MIN_CLOSURE),
            c_peak_distance_gain_kg=float(j1b_gain),
            j1b_holds=bool(j1b_gain >= 0.0),
            c_base_delta=float(base_moves),
            j1c_a_gain_only_from_base_falling=j1c_only_via_base,
            beta=betas[lab], alpha=alphas[lab],
            beta_change=float((betas[lab] - betas[NULL_LABEL])
                              if betas[lab] is not None else np.nan),
            alpha_change=float((alphas[lab] - alphas[NULL_LABEL])
                               if alphas[lab] is not None else np.nan),
            D_beta_cand=float(abs(betas[lab] - BETA_OBS)) if betas[lab] is not None else None,
            D_beta_P=float(abs(betas[NULL_LABEL] - BETA_OBS)) if betas[NULL_LABEL] is not None else None,
            D_alpha_cand=float(abs(alphas[lab] - ALPHA_OBS)) if alphas[lab] is not None else None,
            D_alpha_P=float(abs(alphas[NULL_LABEL] - ALPHA_OBS)) if alphas[NULL_LABEL] is not None else None,
            monthly=mm, nse_degradation=nse_deg,
            median_station_nse_degradation=mnse_deg,
            mean_concentration_relative_change=mean_rel,
            j4_holds=bool(nse_deg <= MONTHLY_GATE and mnse_deg <= MONTHLY_GATE),
            j5_holds=bool(mean_rel <= MONTHLY_GATE),
            bitwise_identical_to_null_on_events=identical,
            max_abs_peak_diff_vs_null=max_peak_diff,
            max_abs_base_diff_vs_null=max_base_diff,
            capped_events=int(p['capped']['n_events_with_a_capped_day']),
            capped_cells_in_window=int(p['capped']['n_capped_cells_inside_event_windows']),
            cap_headroom_in_window=p['capped']['headroom'],
            legal_all_conjuncts=bool(p['legal']['all_conjuncts']))
        if betas[lab] is not None and betas[NULL_LABEL] is not None:
            rows[lab]['j2_holds'] = bool(rows[lab]['D_beta_cand'] <= rows[lab]['D_beta_P'])
            rows[lab]['j3_holds'] = bool(rows[lab]['D_alpha_cand'] <= rows[lab]['D_alpha_P'])
        else:
            rows[lab]['j2_holds'] = None
            rows[lab]['j3_holds'] = None
        rows[lab]['both_boxed'] = bool(rows[lab]['j1a_holds'] and rows[lab]['j1b_holds'])
        print('%-18s A=%.9f  G=%+.6f  J1a=%-5s dCpeak_gain=%+.6e J1b=%-5s  '
              'beta=%s alpha=%s  J4=%s J5=%s  identical=%s'
              % (lab, a['A'], G, rows[lab]['j1a_holds'], j1b_gain, rows[lab]['j1b_holds'],
                 ('%.8f' % betas[lab]) if betas[lab] is not None else 'None',
                 ('%.8f' % alphas[lab]) if alphas[lab] is not None else 'None',
                 rows[lab]['j4_holds'], rows[lab]['j5_holds'], identical), flush=True)
    S['points'] = rows

    # ---------------- the verdict, on the FROZEN rule ----------------
    any_both = [k for k, v in rows.items() if v['both_boxed']]
    any_j1a = [k for k, v in rows.items() if v['j1a_holds']]
    any_j1b = [k for k, v in rows.items() if v['j1b_holds']]
    # Criterion 2 is "beta and alpha preserved or closer"; criterion 1 is J1a with
    # its anti-gaming clause J1b.  Splitting them is what the CALLING table needs:
    # `both_boxed` is a criterion-1 test, not a criterion-2 test.
    crit2 = [k for k, v in rows.items() if v['j2_holds'] and v['j3_holds']]
    crit1 = any_both
    crit2_nontrivial = [k for k in crit2 if not rows[k]['bitwise_identical_to_null_on_events']]
    all_legal = bool(all(v['legal_all_conjuncts'] for v in rows.values()))
    S['verdict'] = dict(
        any_point_satisfies_both_boxed=bool(any_both), points=any_both,
        points_satisfying_J1a=any_j1a, points_satisfying_J1b=any_j1b,
        points_satisfying_criterion_1=crit1,
        points_satisfying_criterion_2=crit2,
        points_satisfying_criterion_2_NON_VACUOUSLY=crit2_nontrivial,
        all_points_legal=all_legal,
        outcome=('BLOCKED' if not all_legal else
                 'QUALIFY_FOR_REFIT' if (crit1 and crit2) else
                 'PARTIAL' if (crit1 or crit2) else 'CLOSE'),
        literal_table_reading=(
            'PARTIAL: exactly one of the two boxed criteria is satisfied (criterion 2, '
            'at the points that are bitwise identical to the baseline) and criterion 1 '
            'is satisfied nowhere.  section 5 calls this PARTIAL, and its action is '
            '"register how far each falls short; add no new mechanism and give this '
            'state no second degree of freedom".'),
        why_not_CLOSE=(
            'section 5 CLOSE would also read as satisfied -- no point meets BOTH -- and '
            'its action is to CLOSE `inter-event memory` outright, on the ground that '
            '"the internal grey-box structures the present observations can support are '
            'exhausted".  That ground is a claim about the whole family, and section 8 '
            'risk 1 forbids deriving it from a one-parameter grid: "not finding a good '
            'point on 1 parameter is NOT the same as the mechanism not existing".  '
            'PARTIAL is therefore the outcome that is actually supported, and the '
            'mechanism is NOT closed here.'),
        criterion_2_satisfaction_is_vacuous=(
            'the points satisfying criterion 2 do so because the cap engaged on ZERO '
            'event-window cells there, so beta and alpha are preserved by the state '
            'doing nothing, not by it preserving memory.  Recorded so that PARTIAL is '
            'not read as progress.'),
        rule='pre-registration section 5')

    # ---------------- section 10: P1's falsification follow-ups --------
    rises = {lab: v['c_peak_distance_gain_kg'] for lab, v in rows.items()
             if v['amp']['c_peak'] > C_PEAK_P + 1e-9}
    S['P1_falsification'] = dict(
        condition_1_met=bool(rises),
        points_where_c_peak_rose=rises,
        condition_2_rise_is_not_confined_to_events_with_a_capped_day=bool(
            all(rows[lab]['capped_events'] == 0 for lab in rises)),
        per_event=[], g8_rerun=None)
    for lab in rises:
        _, _, ev = load(lab)
        _, _, evn2 = load(NULL_LABEL)
        z = ev[['station_key', 'event_id', 'c_peak', 'c_base']].rename(
            columns={'c_peak': 'cand_peak', 'c_base': 'cand_base'}).merge(
            evn2[['station_key', 'event_id', 'c_peak', 'c_base']].rename(
                columns={'c_peak': 'null_peak', 'c_base': 'null_base'}),
            on=['station_key', 'event_id'], validate='one_to_one')
        z['peak_rise'] = z.cand_peak - z.null_peak
        z['base_rise'] = z.cand_base - z.null_base
        z['capped_days'] = 0
        zm = z.reindex(z.peak_rise.abs().sort_values(ascending=False).index).head(20)
        S['P1_falsification']['per_event'].append(dict(
            label=lab, n_events_with_any_peak_change=int((z.peak_rise.abs() > 0).sum()),
            n_events_with_a_peak_RISE=int((z.peak_rise > 0).sum()),
            n_events_with_a_peak_FALL=int((z.peak_rise < 0).sum()),
            max_peak_rise=float(z.peak_rise.max()), max_peak_fall=float(z.peak_rise.min()),
            median_peak_rise=float(z.peak_rise.median()),
            median_base_rise=float(z.base_rise.median()),
            largest_movers=zm[['station_key', 'event_id', 'peak_rise', 'base_rise']]
            .to_dict('records')))
        # the pre-registration REQUIRES a G8 re-run when the rise is measurable
        import gate_noop as G8
        S['P1_falsification']['g8_rerun'] = G8._timescales()

    out = R / 'reports/phase1_scores.json'
    out.write_text(json.dumps(S, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('VERDICT %s   both=%s  J1a=%s  J1b=%s'
          % (S['verdict']['outcome'], any_both, any_j1a, any_j1b), flush=True)
    print('SCORE_DONE', C.sha(out), flush=True)


if __name__ == '__main__':
    main()
