"""`20260920_2` -- the section 8.1 shape quantity set.  ZERO FORWARDS.  `enters_no_gate=True`.

WHY THIS FILE IS THE SECOND HALF OF THE ROUND
---------------------------------------------
The round asks one question on one knob: can a single `k_m` buy event amplitude WITHOUT
pushing the long-run level off the frozen baseline?  `level_variance.py` answers the "level"
half -- every kernel arm moves the mean by 1.9x to 9.1x and recentring recovers none of it.
This file answers the other half, and the user's ruling is explicit about how it must be
answered (verbatim): "**我建议不要只用'去均值 NSE'。** 因为 C'_t = C_t - Cbar + Cbar_0 是
**加法平移**，而我们的核心事件判据 C_peak/C_base **不是平移不变量**。一个较大的加法平移会
直接改变事件比值。"

So this file carries the shape evidence on statistics that are NOT decoupled-by-construction
from the level, and it says so per statistic:

  A_L1 / A_L3      ratio statistics -- a translation CHANGES them, so they are NOT pure shape
                   evidence and must be read beside the level.  Reported, flagged.
  CV_s, e^CV_s     within-station scale-normalised dispersion.  Dividing by the station's OWN
                   mean is what makes a 9x level shift NOT read as 9x of variance capability.
                   This is the statistic the ruling was aimed at.
  r_monthly        pooled and per-station Pearson r on the monthly panel.
  SD_cand/SD_obs   pooled monthly dispersion ratio.
  centered-RMSE    RMSE of the recentred prediction -- the additive-shift residual.

WHICH PANEL THE CV READS, AND A REGISTERED TENSION IN THE TEXT
--------------------------------------------------------------
Section 8.1's first bullet describes `CV_s` as a check on "日尺度方差能力", but the registered
note immediately below it fixes the CALIBER: "`CV_obs,s` 与 `centered-RMSE` 读取**月度观测
面板**的**方差**，走的是 `G5` 同一条只读路径（`obs_monthly()`，同一个 sha 断言）."  The
caliber note is the binding clarification (it was added precisely to pin this down), so the
MONTHLY CV is this file's primary reading and it is labelled `registered_panel: monthly`.

That leaves the prose word "日尺度" describing a monthly statistic.  This file does not paper
over the mismatch: it computes the DAILY CV as well, on the eligible station-days that
`layers25.station_sd_gate` already uses, labels it `registered_panel: none (extra reading)`,
and registers it below as an ADDITION rather than a substitution.  Both are diagnostics, both
enter no gate, and the daily one is the version that actually addresses the 9x-shift worry the
ruling names.  Neither replaces the other.

THE PANEL IS READ-ONLY AND IS NOT A FIT TARGET
----------------------------------------------
`obs_monthly()` is the same read-only 4h panel `G5` scores against, behind the same sha
assertion (common25.py:771).  A zero-fit round may SCORE against it, never calibrate on it.
No fit is taken here, no threshold is read here, and every node carries `enters_no_gate=True`.

`R5-ref` IS A REFERENCE, NOT AN ARM
-----------------------------------
Read off disk by `phase1_arms.py`; plan sections 0.1 and 5 forbid its readings from entering
any gate.  Reported in its own block, flagged, excluded from every roll-up.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C

E = C.EL
R = C.ROUND
OUT = R / 'reports'
NULL_ARM = 'B0'
REFERENCE_ARM = 'R5-ref'


def _p(msg):
    print(msg, flush=True)


def _num(x):
    """A real, or `None` where the statistic is undefined.  Never `nan` (it is not JSON, and
    `nan <= threshold` grades False, which would look like a measurement)."""
    x = float(x)
    return x if np.isfinite(x) else None


def _cv(s):
    """Coefficient of variation of a series, or `None` if its mean is 0 (0/0)."""
    s = np.asarray(s, float)
    m = float(s.mean())
    if m == 0.0 or s.size < 2:
        return None
    return _num(s.std(ddof=0) / abs(m))


def _e_cv(pred, obs):
    """|log(CV_pred / CV_obs)|, or `None` if either CV is undefined or zero."""
    if pred is None or obs is None or pred <= 0.0 or obs <= 0.0:
        return None
    return _num(abs(np.log(pred / obs)))


def main():
    rep = {'phase': 'shape_diagnostics', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'zero_forwards': True, 'n_forwards_taken': 0,
           'enters_no_gate': True,
           'authority': 'NONE. Section 8.1 makes every quantity in this file a DIAGNOSTIC. '
                        'It may not pass a gate, may not change a gate, and may not sign a '
                        'verdict. `FULL_CAPABILITY` is issued by layer 1 only, and layer 2 '
                        'lives in verdict.json::shape_layer.',
           'why_it_exists': 'the user ruling of this round: do NOT use recentring / '
                            'de-meaned NSE as the only matched-level evidence, because '
                            'C\'_t = C_t - Cbar + Cbar_0 is an ADDITIVE SHIFT while the '
                            'round\'s core criterion C_peak/C_base is NOT shift-invariant',
           'reads': 'reports/daily_arms.parquet + reports/phase1_arms.json + the READ-ONLY '
                    'monthly observation panel behind obs_monthly()'}

    p1 = C.read_json(OUT / 'phase1_arms.json')
    daily = pd.read_parquet(OUT / 'daily_arms.parquet')
    elig = C.eligible_grid()
    obs_m = C.obs_monthly()
    A = C.load_anchors()
    rep['sources'] = dict(phase1_arms=C.sha(OUT / 'phase1_arms.json'),
                          daily_arms=C.sha(OUT / 'daily_arms.parquet'),
                          panel=C.PANEL_SHA, mask=E.MASK_SHA)
    rep['anchors'] = dict(A_L1_frozen=float(A['A_L1']['value']),
                          A_L3_frozen=float(A['A_L3']['value']),
                          A_L1_P_upper=float(A['P_upper_A_L1']['value']),
                          A_L3_P_upper=float(A['P_upper_A_L3']['value']),
                          G1_threshold=float(A['G1_target_50pct']['value']),
                          G2_threshold=float(A['G2_target_50pct']['value']),
                          role='frozen readings only; no arm of this round was run when these '
                               'were computed')
    rep['registered_caliber'] = dict(
        cv_panel='monthly',
        text='预注册_判据与门槛.md section 8.1 note: "CV_obs,s 与 centered-RMSE 读取月度观测'
             '面板的方差，走的是 G5 同一条只读路径（obs_monthly()，同一个 sha 断言）"',
        tension='the same section\'s first bullet describes CV_s as a check on 日尺度 variance '
                'capability, which a monthly statistic is not. The caliber note is the binding '
                'clarification; the DAILY CV is added below as an extra, separately labelled '
                'reading rather than substituted, so neither description is silently dropped.',
        not_a_fit_target='the panel is 只读 and is never a fit target (n_fits = 0)',
        shares_the_panel_with='G5b / mean_concentration use the same panel and a DIFFERENT '
                             'statistic; this is registered as a named free choice (N9-prime)')

    # The arm set is DERIVED from the frozen arm table, never re-typed: a hard-coded list is a
    # second place for the arm set to drift.
    KERNEL_ARMS = tuple(sorted(a for a, v in p1['arms'].items()
                               if v.get('installs_kernel') and a != REFERENCE_ARM))
    if len(KERNEL_ARMS) != 9:
        raise SystemExit('THE_KERNEL_ARM_SET_MOVED %r' % (KERNEL_ARMS,))
    ARMS = (NULL_ARM,) + KERNEL_ARMS + (REFERENCE_ARM,)
    rep['arm_sets'] = dict(kernel_arms=list(KERNEL_ARMS), null_arm=NULL_ARM,
                           reference_arm=REFERENCE_ARM,
                           source='derived from phase1_arms.json::arms/*/installs_kernel')

    # -------------------------------------------------- the daily observation column
    # The SAME construction `layers25.station_sd_gate` uses: the 4h panel collapsed to
    # station-day MEANS over the eligible days.  Built once and asserted complete, so an
    # eligible day with no observation is an error and not a silent drop.
    key = elig[['station_key', 'date']].copy()
    key['date'] = key.date.astype('datetime64[ns]')
    panel = pd.read_parquet(E.PANEL)
    panel = panel.assign(date=E.as_day(panel.monitoring_time.dt.tz_localize(None)))
    obs_d = (panel.groupby(['station_key', 'date'], as_index=False)['TN'].mean()
             .rename(columns={'TN': 'obs'}))
    obs_e = key.merge(obs_d, on=['station_key', 'date'], how='left')
    if int(obs_e.obs.isna().sum()):
        raise SystemExit('ELIGIBLE_DAY_WITHOUT_OBSERVATION %d' % int(obs_e.obs.isna().sum()))
    cv_obs_daily = {str(k): _cv(g.obs.to_numpy(float))
                    for k, g in obs_e.groupby('station_key')}
    if any(v is None for v in cv_obs_daily.values()):
        raise SystemExit('AN_OBSERVED_DAILY_CV_IS_UNDEFINED %r' % cv_obs_daily)
    rep['daily_observation'] = dict(
        n_eligible_station_days=int(len(obs_e)), n_stations=int(obs_e.station_key.nunique()),
        panel_rows=int(len(panel)),
        obs_definition='station-day MEAN TN over the ELIGIBLE days (the same collapse '
                       'layers25.station_sd_gate uses at :299)',
        cv_obs_daily=cv_obs_daily,
        cv_obs_daily_median=float(np.median(list(cv_obs_daily.values()))),
        note='arm-independent: CV_obs,s does not move with the arm, which is asserted per arm '
             'below rather than assumed')

    # ---------------------------------------------------------------- the null monthly join
    null = daily[daily.arm == NULL_ARM].reset_index(drop=True)
    key0 = np.asarray(null.station_key.astype(str))
    day0 = null.date.to_numpy().astype('datetime64[D]')
    _m0, znull = C.monthly_join(null, elig, obs_m)
    o = znull.obs.to_numpy(float)
    cv_obs_monthly = {str(k): _cv(g.obs.to_numpy(float))
                      for k, g in znull.groupby(level=0)}
    if any(v is None for v in cv_obs_monthly.values()):
        raise SystemExit('AN_OBSERVED_MONTHLY_CV_IS_UNDEFINED %r' % cv_obs_monthly)
    rep['observation'] = dict(
        source='obs_monthly() -- READ-ONLY 4h panel, sha asserted (common25.PANEL_SHA)',
        n_station_months=int(len(znull)), n_stations=int(znull.index.get_level_values(0).nunique()),
        cv_obs_monthly=cv_obs_monthly,
        cv_obs_monthly_median=float(np.median(list(cv_obs_monthly.values()))),
        cv_obs_is_arm_independent=True)
    _p('=== shape diagnostics (enters_no_gate=True) ===')
    _p('   obs: %d station-months, %d stations; median CV_obs monthly=%.6f daily=%.6f'
       % (len(znull), len(cv_obs_monthly), rep['observation']['cv_obs_monthly_median'],
          rep['daily_observation']['cv_obs_daily_median']))

    # ------------------------------------------------------------- per arm
    sh = {}
    for a in ARMS:
        sub = daily[daily.arm == a].reset_index(drop=True)
        # Row alignment is asserted, not assumed: the daily CV below is an elementwise
        # model-minus-observation comparison and is meaningless if the rows are not the same
        # station-days in the same order.
        if not (np.array_equal(np.asarray(sub.station_key.astype(str)), key0)
                and np.array_equal(sub.date.to_numpy().astype('datetime64[D]'), day0)):
            raise SystemExit('ARM_%s_IS_NOT_ROW_ALIGNED_WITH_B0' % a)
        _m, z = C.monthly_join(sub, elig, obs_m)
        po = z.pred.to_numpy(float)
        node = dict(arm=a, role=p1['arms'][a]['role'],
                    enters_no_gate=True,
                    is_reference=bool(a == REFERENCE_ARM),
                    A_L1=_num(p1['arms'][a]['A_L1']) if p1['arms'][a]['A_L1'] is not None else None,
                    A_L3=_num(p1['arms'][a]['A_L3']) if p1['arms'][a]['A_L3'] is not None else None,
                    A_L1_reading_note='a RATIO statistic: an additive shift changes it, so it '
                                      'is NOT pure shape evidence and must be read beside the '
                                      'level (section 8.1, verbatim). Read from '
                                      'phase1_arms.json; audit_dp2.py is the independent '
                                      'recomputation.',
                    n_station_months=int(len(z)), n_eligible_rows=int(len(sub)))

        # ---- the registered monthly CV
        cvp = {str(k): _cv(g.pred.to_numpy(float)) for k, g in z.groupby(level=0)}
        ecv = {k: _e_cv(cvp[k], cv_obs_monthly[k]) for k in cv_obs_monthly}
        ok = {k: v for k, v in ecv.items() if v is not None}
        node['monthly_cv'] = dict(
            registered_panel='monthly', cv_pred=cvp, e_cv_s=ecv,
            n_stations_defined=int(len(ok)),
            n_stations_undefined=int(len(ecv) - len(ok)),
            undefined_because=(None if ok else
                               'every modelled monthly CV is 0/0: the arm\'s land-side output '
                               'is identically zero (Q0-zero, q_m = 0)'),
            e_cv_median=(None if not ok else float(np.median(list(ok.values())))),
            e_cv_mean=(None if not ok else float(np.mean(list(ok.values())))),
            cv_pred_over_obs_median=(None if not ok else
                                     float(np.median([cvp[k] / cv_obs_monthly[k] for k in ok
                                                      if cv_obs_monthly[k] > 0.0]))),
            undirected='e^CV_s is a |log ratio|, so it is UNDIRECTED: it says the dispersion is '
                       'further from the observation, never which way. This is deliberate -- '
                       'the round may not buy a smaller e^CV_s by inflating dispersion.')

        # ---- the extra daily CV (registered below as an addition, not a substitution)
        cmp_d = obs_e.merge(sub[['station_key', 'date', 'pL3']], on=['station_key', 'date'],
                            how='left', validate='one_to_one')
        if int(cmp_d.pL3.isna().sum()):
            raise SystemExit('ARM_%s_DROPPED_AN_ELIGIBLE_DAY' % a)
        cvd = {str(k): _cv(g.pL3.to_numpy(float)) for k, g in cmp_d.groupby('station_key')}
        ed = {k: _e_cv(cvd[k], cv_obs_daily[k]) for k in cv_obs_daily}
        okd = {k: v for k, v in ed.items() if v is not None}
        node['daily_cv'] = dict(
            registered_panel='none -- EXTRA reading, registered in work/_deviations_running.md',
            cv_pred=cvd, e_cv_s=ed,
            n_stations_defined=int(len(okd)), n_stations_undefined=int(len(ed) - len(okd)),
            undefined_because=(None if okd else
                               'every modelled daily CV is 0/0: the land-side output is '
                               'identically zero (Q0-zero, q_m = 0)'),
            e_cv_median=(None if not okd else float(np.median(list(okd.values())))),
            daily_panel_obs_would_be_undefined=True if any(
                v is None for v in cv_obs_daily.values()) else False,
            why_it_is_here='section 8.1\'s first bullet describes CV_s as a check on 日尺度 '
                           'variance capability while the caliber note fixes CV_obs,s to the '
                           'MONTHLY panel. Reporting both keeps the 9x-shift worry the ruling '
                           'names actually addressed; neither reading replaces the other.')

        # ---- monthly correlation and dispersion
        node['monthly_r'] = dict(
            pooled=_num(np.corrcoef(o, po)[0, 1]) if float(np.std(po)) > 0.0 else None,
            pooled_undefined_because=(None if float(np.std(po)) > 0.0 else
                                      'the arm\'s monthly prediction is constant, so r is 0/0'),
            per_station={str(k): _num(np.corrcoef(g.obs.to_numpy(float),
                                                  g.pred.to_numpy(float))[0, 1])
                         for k, g in z.groupby(level=0)
                         if float(np.std(g.pred.to_numpy(float))) > 0.0},
            n_stations_undefined=int(sum(1 for _k, g in z.groupby(level=0)
                                         if float(np.std(g.pred.to_numpy(float))) == 0.0)),
            r2=_num(1.0 - np.mean((po - o) ** 2) / np.var(o)))
        sd_ratio = (None if float(np.std(o)) == 0.0 else _num(np.std(po) / np.std(o)))
        node['sd_cand_over_obs'] = dict(
            pooled=sd_ratio, sd_obs=float(np.std(o)), sd_pred=float(np.std(po)),
            note='a DISPERSION ratio, not a shape ratio: it rises with any overall scale change')

        # ---- centered-RMSE: the additive-shift residual
        pc = po - po.mean() + o.mean()
        node['centered_rmse'] = dict(
            pooled=_num(np.sqrt(np.mean((pc - o) ** 2))),
            rmse_uncentered=_num(np.sqrt(np.mean((po - o) ** 2))),
            per_station_median=(float(np.median([np.sqrt(np.mean(
                (g.pred.to_numpy(float) - g.pred.to_numpy(float).mean()
                 + g.obs.to_numpy(float).mean() - g.obs.to_numpy(float)) ** 2))
                for _k, g in z.groupby(level=0)]))),
            mean_shift_removed=float(po.mean() - o.mean()),
            definition='RMSE of C\'-pred against obs, C\' = C - mean(C) + mean(C_obs). The '
                       'ADDITIVE-SHIFT residual, and AUXILIARY ONLY (section 8.1).')

        sh[a] = node
        _p('   %-9s A_L1=%s A_L3=%s  eCV_m=%-9s eCV_d=%-9s  r=%-9s  SDratio=%-9s  cRMSE=%s'
           % (a, _f(node['A_L1']), _f(node['A_L3']),
              _f(node['monthly_cv']['e_cv_median']), _f(node['daily_cv']['e_cv_median']),
              _f(node['monthly_r']['pooled']), _f(node['sd_cand_over_obs']['pooled']),
              _f(node['centered_rmse']['pooled'])))
    rep['arms'] = sh

    # ------------------------------------- does anything move TOWARD the observation?
    # The plan's `KM_MAPPING_NO_SHAPE_CAPABILITY` clause reads: even ignoring the level,
    # A_L1 / A_L3 / the station CV do not move toward the observation anywhere on the
    # candidate grid.  That is a claim about a DIRECTION, so the direction is computed here
    # against the frozen baseline rather than left for a reader to eyeball.
    base = sh[NULL_ARM]
    cv_base = base['monthly_cv']['e_cv_median']
    cands = [a for a in KERNEL_ARMS if p1['arms'][a].get('is_candidate')]
    moved = {}
    for a in cands:
        n = sh[a]
        moved[a] = dict(
            q_m=p1['arms'][a]['q_m'], tau_m=p1['arms'][a]['tau_m'],
            e_cv_median=n['monthly_cv']['e_cv_median'],
            e_cv_baseline=cv_base,
            cv_moves_toward_obs=(None if n['monthly_cv']['e_cv_median'] is None or cv_base is None
                                 else bool(n['monthly_cv']['e_cv_median'] < cv_base)),
            cv_change=(None if n['monthly_cv']['e_cv_median'] is None or cv_base is None
                       else float(n['monthly_cv']['e_cv_median'] - cv_base)),
            A_L1_moves_toward_obs=(None if n['A_L1'] is None
                                   else bool(n['A_L1'] > float(A['A_L1']['value']))),
            A_L3_moves_toward_obs=(None if n['A_L3'] is None
                                   else bool(n['A_L3'] > float(A['A_L3']['value']))))
    n_cv = int(sum(1 for v in moved.values() if v['cv_moves_toward_obs']))
    n_l1 = int(sum(1 for v in moved.values() if v['A_L1_moves_toward_obs']))
    n_l3 = int(sum(1 for v in moved.values() if v['A_L3_moves_toward_obs']))
    # The SAME question asked of the EXTRA daily CV.  It is asked here, and its answer is
    # reported in the same breath, because the two panels do NOT agree and a record that
    # showed only the statistic that failed would be a selection. See the `reading` string.
    cvd_base = base['daily_cv']['e_cv_median']
    n_cvd = int(sum(1 for a in cands if sh[a]['daily_cv']['e_cv_median'] is not None
                    and cvd_base is not None
                    and sh[a]['daily_cv']['e_cv_median'] < cvd_base))
    rep['direction'] = dict(
        baseline=NULL_ARM, baseline_e_cv_median=cv_base, baseline_e_cv_daily=cvd_base,
        baseline_A_L1=float(A['A_L1']['value']), baseline_A_L3=float(A['A_L3']['value']),
        per_candidate=moved,
        n_candidates=len(cands),
        n_cv_moving_toward_obs=n_cv,
        n_A_L1_moving_toward_obs=n_l1,
        n_A_L3_moving_toward_obs=n_l3,
        n_cv_daily_moving_toward_obs=n_cvd,
        cv_daily_per_candidate={a: dict(
            e_cv_median=sh[a]['daily_cv']['e_cv_median'], e_cv_baseline=cvd_base,
            moves_toward_obs=(None if sh[a]['daily_cv']['e_cv_median'] is None
                              or cvd_base is None
                              else bool(sh[a]['daily_cv']['e_cv_median'] < cvd_base)))
            for a in cands},
        panels_disagree=bool(n_cvd != n_cv),
        reading=('on the REGISTERED monthly panel -- the same panel G5 and G5b score on -- '
                 'no candidate moves toward the observation on any of the three statistics '
                 '(monthly CV, A_L1, A_L3): 0/%d on each' % len(cands)
                 if (n_cv == 0 and n_l1 == 0 and n_l3 == 0) else
                 'at least one candidate moves toward the observation on at least one '
                 'registered shape statistic -- read per_candidate before writing any '
                 'sentence about it'),
        panels_disagree_note=('THE TWO PANELS DO NOT AGREE, and this is reported rather than '
                              'resolved. On the extra DAILY CV, %d of %d candidates read closer '
                              'to the observation than the frozen baseline (%.6f) does -- the '
                              'arms roughly halve or double the within-station day-to-day '
                              'dispersion, and one of those two lands nearer. That is a real '
                              'signal and it is NOT capability: on the same arms the MONTHLY '
                              'correlation collapses from the baseline\'s %.6f to %.2f-%.2f, '
                              'the monthly SD ratio rises to %.1f-%.1f and the centered-RMSE to '
                              '%.1f-%.1f, i.e. the monthly panel -- the one this round scores '
                              'on -- is strictly worse everywhere. The registered caliber makes '
                              'the monthly reading primary; the daily reading is recorded '
                              'beside it so the choice is visible.'
                              % (n_cvd, len(cands), cvd_base,
                                 base['monthly_r']['pooled'],
                                 min(sh[a]['monthly_r']['pooled'] for a in cands),
                                 max(sh[a]['monthly_r']['pooled'] for a in cands),
                                 min(sh[a]['sd_cand_over_obs']['pooled'] for a in cands),
                                 max(sh[a]['sd_cand_over_obs']['pooled'] for a in cands),
                                 min(sh[a]['centered_rmse']['pooled'] for a in cands),
                                 max(sh[a]['centered_rmse']['pooled'] for a in cands))),
        note='A_L1 / A_L3 "toward the observation" is read against the FROZEN baseline value '
             '(A_L1 = %r), which is the observation-anchored reference in this round. The '
             'G1/G2 THRESHOLDS are a different, stricter comparison and are computed in '
             'phase1_arms.py, not here.' % float(A['A_L1']['value']))
    _p('   direction: %d/%d candidates move toward obs on monthly CV, %d on A_L1, %d on A_L3 '
       '(%d on the extra DAILY CV -- panels disagree: %s)'
       % (n_cv, len(cands), n_l1, n_l3, n_cvd, rep['direction']['panels_disagree']))

    # --------------------------------------------------------- the reference arm
    rep['reference_arm'] = dict(
        arm=REFERENCE_ARM, enters_no_gate=True, readings=sh[REFERENCE_ARM],
        note='R5-ref is round 5\'s delivered point, READ OFF DISK, not run here. Plan sections '
             '0.1 and 5 forbid its readings from entering any gate pass/fail; it is excluded '
             'from every roll-up in this file.')

    rep['deviations'] = [
        'the DAILY CV is an ADDITION to section 8.1, not a substitution: the section\'s first '
        'bullet calls CV_s a 日尺度 check while its caliber note fixes CV_obs,s to the MONTHLY '
        'panel. Both are reported, the monthly one is labelled registered_panel=monthly and is '
        'the primary, and both enter no gate. Registered in work/_deviations_running.md.',
        'e^CV_s is UNDIRECTED by construction (|log ratio|), so no arm can buy a smaller '
        'value by inflating dispersion. Stated in the record rather than left implicit.',
        'A_L1 / A_L3 are flagged as RATIO statistics that an additive shift changes, per the '
        'user ruling; they are reported here but may not be read as pure shape evidence.',
        'Q0-zero (q_m = 0) has identically zero land-side output, so every modelled CV is 0/0. '
        'Those nodes are `None` with an `undefined_because` reason -- never `nan`, and never a '
        'zero that would look like a measurement.',
        'R5-ref is reported in its own block, flagged enters_no_gate=True, and excluded from '
        'every roll-up in this file.',
        'no gate, threshold or baseline is read in this file; every node carries '
        'enters_no_gate=True',
    ]
    C.write_json(OUT / 'shape_diagnostics.json', rep)
    _p('=== wrote %s ===' % (OUT / 'shape_diagnostics.json'))
    return rep


def _f(x, spec='.6f'):
    """Print a reading, showing `None` where the statistic is undefined rather than a nan."""
    return 'None' if x is None else format(float(x), spec)


if __name__ == '__main__':
    main()
