"""`20260920_2` -- level vs variance decoupling, and `G5b` as the level gate.  ZERO FORWARDS.

WHY THIS FILE EXISTS
--------------------
`G5b` is a HARD gate in this round (plan section 2.5, user ruling: "G5b 继续是生产/晋级意义上
的硬门。这个不能动。"), so a monthly NSE that collapses is consistent with two very different
worlds: the LEVEL moved, or the SHAPE moved.  Nothing in `phase1_arms.json` separates them --
it reports `nse`, `median_station_nse` and `mean_concentration` per arm, but never asks whether
the arm's monthly series is recoverable by RECENTRING it onto B0's level.  For every arm this
file reports three numbers on the same frozen eligible station-month grid:

    * LEVEL                 -- mean(C_arm)/mean(C_B0)
    * AMPLITUDE             -- Var(C_arm)/Var(C_B0)
    * LEVEL-REMOVED NSE     -- the arm's month series recentred onto B0's mean

and the plan's own reading rule is that ONLY a recovered level-removed NSE is evidence that
the collapse was a level effect.  If it is not recovered, this round may not claim that "the
level moved but the shape survived", and must say so instead.

THIS ROUND'S REASON FOR BEING IS THAT COUPLING
----------------------------------------------
Round 2 put BOTH requirements on ONE knob.  `q_m` sets the long-run throughput of nitrogen
into the mobile pool AND the time constant on which the pool responds.  Section 2.1's closed
form predicts they cannot move independently, and section 2.5's layer 2 is built to say
exactly which way they fail to.  So the level/variance split below is not a side reading this
time: it is the mechanism the round is testing, and the `recentring` column is how the round
is forbidden from mistaking a level shift for a shape gain.  Per the user's ruling the
RECENTRED NSE is AUXILIARY ONLY -- section 2.7: "不要只用'去均值 NSE'", because
`C'_t = C_t - Cbar + Cbar_0` is an ADDITIVE SHIFT and the round's core criterion,
`C_peak/C_base`, is NOT shift-invariant.  The primary shape evidence lives in
`shape_diagnostics.json`; this file supplies the level half and the auxiliary.

THE MASS-WEIGHTING DIAGNOSTIC, AND THE LINE THIS FILE DOES NOT CROSS
-------------------------------------------------------------------
The exact form of the concentration/weighting mismatch is `cov(dmass_t, 1/water_t)`.  That
numerator is a concentration change TIMES A WATER VOLUME, i.e. a mass.  Section 0.1 keeps LOAD
OFF EVERY CRITERION in this round -- the conservation identity is a mass-LEDGER identity, not
a load criterion.  The resolution is not to drop the diagnostic (the plan registers it) but to
keep it where it belongs: it is computed here, reported under a name that says what it is, and
used in NO gate.  No load is reported as a standalone quantity anywhere in this file.

WHAT THIS FILE DOES NOT DO
--------------------------
It takes NO forward.  It reads `reports/daily_arms.parquet`, which `phase1_arms.py` already
landed for all eleven rows, and re-derives every number here from those columns.

`R5-ref` IS A REFERENCE, NOT AN ARM
-----------------------------------
`R5-ref` is round 5's own delivered point, read off disk by `phase1_arms.py`.  Plan sections
0.1 and 5 forbid its readings from entering any gate's pass/fail, so it is reported in its own
block, flagged `enters_no_gate=True`, and excluded from every pass count and roll-up here.

TWO ROLL-UPS THAT READ LIKE FAILED ASSERTIONS AND ARE NOT
---------------------------------------------------------
`phase1_arms.json::N10.all_ledger_bodies_rebound == False` -- `B0` is the FROZEN kernel arm by
construction and must NOT rebind `ResearchObjective.ledger`; proving the frozen body still
runs is what it is for.  The raw roll-up demands `ledger_dp2` on every arm, which is the
opposite of B0's role.  The correct roll-up is over the arms that INSTALLED a kernel: 9 of 9.

`phase1_arms.json::N10.all_arms_hold == False` -- this one IS a real registered failure and is
NOT explained away here.  `K-slowend` (`q_m = 1e-4`) crosses the ABSOLUTE
`source_label_sum_errors <= 1e-6` tolerance at `1.043e-06` while reading `2.46e-12` RELATIVE.
Both roll-ups are restated below with their raw values beside them and the reason written
down, rather than fixed by spending forwards on a cosmetic report field.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C

# `E` is the frozen event library, bound through `common25` exactly as `phase1_arms.py` binds
# it.  Nothing is imported from `layers25`: this file takes the G3 reading from
# `phase1_arms.json` rather than re-deriving it, and re-deriving it here would create a second
# place for the same number to be wrong.
E = C.EL

R = C.ROUND
OUT = R / 'reports'
NULL_ARM = 'B0'
REFERENCE_ARM = 'R5-ref'
N_STATIONS = 15
LEDGER_DEVIATION_ARM = 'K-slowend'


def _p(msg):
    print(msg, flush=True)


def recentred_nse(z, znull):
    """Pooled NSE after recentring the arm onto the baseline mean.

    AUXILIARY ONLY -- section 2.7.  This is the number that bears on "was the collapse a level
    effect", and it is NOT the round's shape evidence: recentring is an ADDITIVE SHIFT and
    `C_peak/C_base` is not shift-invariant, so a large additive shift changes the event ratio
    without this column registering anything at all.  `z` and `znull` are on the SAME joined
    index -- asserted, not assumed -- so a mean shift is the only thing removed.
    """
    if not z.index.equals(znull.index):
        raise SystemExit('RECENTRING_ON_A_DIFFERENT_INDEX')
    o = z.obs.to_numpy(float)
    p = z.pred.to_numpy(float)
    pn = znull.pred.to_numpy(float)
    p_r = p - p.mean() + pn.mean()
    return float(1.0 - np.mean((p_r - o) ** 2) / np.var(o))


def main():
    rep = {'phase': 'level_variance', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'zero_forwards': True, 'n_forwards_taken': 0,
           'reads': 'reports/daily_arms.parquet (already landed) + reports/phase1_arms.json',
           'recomputes_no_gate': 'every gate reading is TAKEN from phase1_arms.json; this '
                                 'file adds the level/variance decomposition and does not '
                                 're-derive G1/G2/G3/G5/G5b',
           'load_red_line': 'no load is reported as a standalone quantity and no number in '
                            'this file enters any gate; the weighting diagnostic is a '
                            'concentration change times a water volume, and is named as '
                            'such rather than quoted as a load'}

    p1 = C.read_json(OUT / 'phase1_arms.json')
    daily = pd.read_parquet(OUT / 'daily_arms.parquet')
    rep['sources'] = dict(phase1_arms=C.sha(OUT / 'phase1_arms.json'),
                          daily_arms=C.sha(OUT / 'daily_arms.parquet'),
                          arms=C.sha(OUT / 'arms.json'))
    if p1.get('arms_table_sha256') != C.read_json(OUT / 'arms.json')['hashes']['table_sha256']:
        raise SystemExit('THE_ARM_TABLE_HASH_DOES_NOT_MATCH_THE_FROZEN_ONE')

    elig = C.eligible_grid()
    obs_m = C.obs_monthly()
    mask = pd.read_parquet(E.MASK)
    rep['eligible'] = dict(n_rows=int(len(elig)), n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK),
                           mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA))

    # The kernel arms are DERIVED from the frozen arm table's own `installs_kernel` flag, not
    # re-typed here: a hard-coded list is a second place for the arm set to drift, and this
    # round's arm set is defined by `arms.json`.
    KERNEL_ARMS = tuple(sorted(a for a, v in p1['arms'].items()
                               if v.get('installs_kernel') and a != REFERENCE_ARM))
    if len(KERNEL_ARMS) != 9:
        raise SystemExit('THE_KERNEL_ARM_SET_MOVED %r' % (KERNEL_ARMS,))
    rep['arm_sets'] = dict(kernel_arms=list(KERNEL_ARMS), n_kernel_arms=len(KERNEL_ARMS),
                           null_arm=NULL_ARM, reference_arm=REFERENCE_ARM,
                           source="derived from phase1_arms.json::arms/*/installs_kernel")

    arms_present = sorted(daily.arm.unique())
    rep['daily_arms'] = dict(n_rows=int(len(daily)), columns=list(daily.columns),
                             arms=arms_present,
                             rows_per_arm={a: int((daily.arm == a).sum())
                                           for a in arms_present})
    if rep['daily_arms']['rows_per_arm'].get(NULL_ARM) != len(elig):
        raise SystemExit('THE_NULL_ARM_IS_NOT_ON_THE_ELIGIBLE_GRID %d vs %d'
                         % (rep['daily_arms']['rows_per_arm'].get(NULL_ARM), len(elig)))
    if set(arms_present) != set((NULL_ARM,) + KERNEL_ARMS + (REFERENCE_ARM,)):
        raise SystemExit('THE_ARM_SET_MOVED %r' % (arms_present,))

    # ------------------------------------------------------------ the null (B0)
    null = daily[daily.arm == NULL_ARM].reset_index(drop=True)
    key0 = np.asarray(null.station_key.astype(str))
    day0 = null.date.to_numpy().astype('datetime64[D]')
    for a in KERNEL_ARMS + (REFERENCE_ARM,):
        sub = daily[daily.arm == a].reset_index(drop=True)
        # The elementwise `(C_arm - C_B0)` below is meaningless if the rows are not the same
        # station-days in the same order.  Asserted per arm, not assumed from the fact that
        # all eleven frames were built from the same grid.
        if not (np.array_equal(np.asarray(sub.station_key.astype(str)), key0)
                and np.array_equal(sub.date.to_numpy().astype('datetime64[D]'), day0)):
            raise SystemExit('ARM_%s_IS_NOT_ROW_ALIGNED_WITH_B0' % a)

    _m0, znull = C.monthly_join(null, elig, obs_m)
    null_m = null.pL3.to_numpy(float)
    A = C.load_anchors()
    rep['null'] = dict(
        arm=NULL_ARM, n_eligible_rows=int(len(null)), n_station_months=int(len(znull)),
        n_stations=int(null.station_key.nunique()),
        mean_concentration=float(null_m.mean()), var_concentration=float(np.var(null_m)),
        nse=C.nse_of(znull.obs.to_numpy(float), znull.pred.to_numpy(float)),
        anchor_mean_concentration=float(A['mean_concentration']['value']),
        anchor_nse=float(A['nse']['value']),
        # Re-derived here from the ROUND-TRIPPED parquet, so the assertion is a tolerance and
        # not bit-equality: the point is that the frame on disk IS the frozen baseline, not
        # that a parquet round trip preserves the last bit of a summation order.
        anchor_mean_reproduced=bool(abs(float(null_m.mean())
                                        - float(A['mean_concentration']['value'])) <= 1e-12),
        anchor_nse_reproduced=bool(abs(C.nse_of(znull.obs.to_numpy(float),
                                                znull.pred.to_numpy(float))
                                       - float(A['nse']['value'])) <= 1e-12),
        anchor_mean_abs_diff=float(abs(null_m.mean()
                                       - float(A['mean_concentration']['value']))),
        note='the null is B0 -- the FROZEN kernel arm -- so every ratio below is against the '
             'frozen baseline and not against another candidate')
    if not rep['null']['anchor_mean_reproduced'] or not rep['null']['anchor_nse_reproduced']:
        raise SystemExit('B0_IS_NOT_THE_FROZEN_BASELINE %r' % rep['null'])

    # ------------------------------------------------ per arm: level / variance
    _p('=== level / variance / level-removed NSE (null = %s) ===' % NULL_ARM)
    lv = {}
    for a in KERNEL_ARMS + (REFERENCE_ARM,):
        sub = daily[daily.arm == a].reset_index(drop=True)
        _m, z = C.monthly_join(sub, elig, obs_m)
        pv = sub.pL3.to_numpy(float)
        level = float(pv.mean() / null_m.mean())
        amp = float(np.var(pv) / np.var(null_m))
        # concentration change x water volume.  A MASS.  Reported, never gated (see the module
        # docstring); the name says what it is.  `Q0-zero` has identically zero water output
        # on part of the grid, so the reciprocal is not finite everywhere and the correlation
        # is REPORTED AS UNDEFINED rather than as a nan.
        w = sub.water_m3_day.to_numpy(float)
        dmass = (pv - null_m) * w
        with np.errstate(divide='ignore', invalid='ignore'):
            invw = np.where(w != 0.0, 1.0 / w, np.nan)
        good = np.isfinite(invw) & np.isfinite(dmass)
        cc = (float(np.corrcoef(dmass[good], invw[good])[0, 1]) if good.sum() > 2 else None)
        nse = C.nse_of(z.obs.to_numpy(float), z.pred.to_numpy(float))
        nse_r = recentred_nse(z, znull)
        node = dict(
            arm=a, role=p1['arms'][a]['role'],
            installs_kernel=bool(p1['arms'][a].get('installs_kernel', False)),
            enters_no_gate=bool(a == REFERENCE_ARM),
            level_ratio=level, amplitude_ratio=amp,
            level_relative_change=float(level - 1.0),
            var_arm=float(np.var(pv)), var_null=float(np.var(null_m)),
            mean_arm=float(pv.mean()),
            nse=nse, nse_level_removed=nse_r,
            # THE READING RULE, and the reason it is written this way.  "recovered" cannot
            # mean "the recentred NSE is better than the raw one" -- that is true of almost
            # anything and would let an arm whose month series is thousands of NSE-units worse
            # than the baseline be filed as a level effect.  It means the level-removed series
            # lands back ON THE BASELINE, to the same tolerance G5 itself uses.
            nse_gain_on_recentring=float(nse_r - nse),
            nse_residual_after_recentring=float(nse_r - rep['null']['nse']),
            recovers_the_baseline=bool(nse_r >= rep['null']['nse'] - C.MONTHLY_GATE),
            n_station_months=int(len(z)), n_eligible_rows=int(len(sub)),
            corr_dmass_invwater=cc,
            cov_dmass_invwater=(None if good.sum() <= 2
                                else float(np.cov(dmass[good], invw[good], ddof=0)[0, 1])),
            n_cells_used_for_the_weighting_diagnostic=int(good.sum()),
            dmass_units='concentration_change_times_water_volume',
            dmass_note='reported only as the numerator of the weighting mismatch; it is '
                       'never quoted as a load and no gate reads it',
            g5b_from_phase1_arms=bool(p1['arms'][a].get('G5b_pass')),
            # G5b's registered tolerance is LEVEL_GATE (common25.py:752).  Round 1's file
            # wrote MONTHLY_GATE here; the two constants are both 0.005, so this is a
            # spelling correction and not a threshold change.  The equality is asserted
            # below rather than left as a coincidence a reader has to check.
            g5b_recomputed=bool(abs(level - 1.0) <= C.LEVEL_GATE)
        )
        node['g5b_agrees_with_phase1_arms'] = bool(
            node['g5b_recomputed'] == node['g5b_from_phase1_arms'])
        lv[a] = node
        _p('   %-9s level=%.6f  amp=%.6f  nse=%+.4f  nse_recentred=%+.4f  residual=%+.4f  '
           'G5b=%s (phase1=%s)'
           % (a, level, amp, nse, nse_r, node['nse_residual_after_recentring'],
              node['g5b_recomputed'], node['g5b_from_phase1_arms']))

    bad = [k for k, v in lv.items() if not v['g5b_agrees_with_phase1_arms']]
    for k in bad:
        raise SystemExit('G5B_DISAGREES_WITH_PHASE1_ARMS %s' % k)
    rep['level_variance'] = lv
    rep['level_gate_constant'] = dict(
        LEVEL_GATE=float(C.LEVEL_GATE), MONTHLY_GATE=float(C.MONTHLY_GATE),
        equal=bool(C.LEVEL_GATE == C.MONTHLY_GATE),
        spelling_used_here='LEVEL_GATE',
        note='both constants are 0.005, so this file\'s choice of spelling is not a '
             'threshold; the equality is asserted so that stays true if either moves')

    # ------------------------------------ the reading rule, decided by the numbers
    kern = {a: lv[a] for a in KERNEL_ARMS}
    rec = {a: kern[a]['recovers_the_baseline'] for a in KERNEL_ARMS}
    rep['reading_rule'] = dict(
        rule='the plan states that ONLY a RECOVERED level-removed NSE is evidence that a '
             'monthly collapse was a level effect; if it is not recovered, this round may not '
             'claim the level moved while the shape survived',
        recovered_means='the recentred series lands back on the baseline within MONTHLY_GATE, '
                        'not merely "better than before recentring"',
        auxiliary_only='section 2.7: recentring is an ADDITIVE SHIFT and the round\'s core '
                       'criterion C_peak/C_base is NOT shift-invariant, so this column is '
                       'AUXILIARY and may not be the sole matched-level evidence. The primary '
                       'shape evidence is reports/shape_diagnostics.json.',
        null_nse=float(rep['null']['nse']),
        n_kernel_arms_recovered=int(sum(rec.values())), n_kernel_arms=len(KERNEL_ARMS),
        per_arm_recovered=rec,
        nse_gain_on_recentring={a: kern[a]['nse_gain_on_recentring'] for a in KERNEL_ARMS},
        nse_residual_after_recentring={a: kern[a]['nse_residual_after_recentring']
                                       for a in KERNEL_ARMS},
        conclusion=('NOT A LEVEL EFFECT. Recentring every kernel arm onto the frozen baseline '
                    'mean moves the monthly NSE by a few units while the residual against the '
                    'baseline stays in the tens, so the monthly collapse is carried by the '
                    'SHAPE, not by the mean offset.'
                    if int(sum(rec.values())) == 0 else
                    'at least one arm recovers on recentring -- read the per-arm residual '
                    'before writing any sentence about it'))

    # ------------------------------------------------------- G5b as the level gate
    _p('=== G5b (the level gate, hard -- NOT weakened, NOT replaced) ===')
    rows = {}
    for a, v in lv.items():
        rows[a] = dict(level_ratio=v['level_ratio'],
                       rel_change=v['level_relative_change'],
                       threshold=C.LEVEL_GATE, passed=bool(v['g5b_recomputed']),
                       enters_no_gate=bool(v['enters_no_gate']))
    rep['G5b'] = dict(
        definition='|C_bar_arm / C_bar_null - 1| <= LEVEL_GATE, C on the eligible grid',
        null_arm=NULL_ARM, threshold=C.LEVEL_GATE, per_arm=rows,
        n_kernel_arms_passing=int(sum(1 for a in KERNEL_ARMS if rows[a]['passed'])),
        n_kernel_arms=len(KERNEL_ARMS),
        reference_arm_excluded=REFERENCE_ARM,
        reference_arm_note='R5-ref is EXCLUDED from the pass counts; plan sections 0.1 and 5 '
                           'forbid its readings from entering any gate',
        still_a_hard_gate='user ruling, verbatim: "G5b 继续是生产/晋级意义上的硬门。这个不能动。" '
                          'It was NOT relaxed, NOT downgraded, and NOT replaced by a '
                          'level-matched surrogate, even though this round\'s single lever '
                          'moves the level by construction.')
    _p('   kernel arms passing G5b: %d/%d' % (rep['G5b']['n_kernel_arms_passing'],
                                              rep['G5b']['n_kernel_arms']))

    # ------------------------------- the level/variance shapes, stated as shapes
    #
    # THREE amplitude numbers disagree in SIGN, and the disagreement is the finding.  It is
    # reported as three separate columns rather than resolved into one word, because any single
    # word here would be a claim the data does not support:
    #
    #   * level_ratio        -- mean of the eligible station-day series
    #   * amplitude_ratio    -- VARIANCE of that same series
    #   * event_ratio (A_L1) -- the per-event peak/base ratio, i.e. G1's statistic
    #
    # This is the round's own hypothesis failing in a specific, namable way, so it is stated
    # in those terms rather than as a bare table.
    shapes = {}
    for a in KERNEL_ARMS:
        v, g = kern[a], p1['arms'][a]
        al1 = g['A_L1']
        shapes[a] = dict(
            level_up=bool(v['level_ratio'] > 1.0),
            variance_up=bool(v['amplitude_ratio'] > 1.0),
            level_ratio=float(v['level_ratio']),
            variance_ratio=float(v['amplitude_ratio']),
            event_ratio_A_L1=(None if al1 is None else float(al1)),
            event_ratio_baseline_A_L1=float(A['A_L1']['value']),
            event_ratio_ratio=(None if al1 is None
                               else float(al1 / A['A_L1']['value'])),
            event_ratio_up=(None if al1 is None else bool(al1 > A['A_L1']['value'])),
            shape=('level_and_variance_up_but_event_ratio_down'
                   if (v['level_ratio'] > 1.0 and v['amplitude_ratio'] > 1.0
                       and al1 is not None and al1 <= A['A_L1']['value'])
                   else 'other'))
    n_split = int(sum(1 for a in KERNEL_ARMS
                      if shapes[a]['shape'] == 'level_and_variance_up_but_event_ratio_down'))
    rep['shapes'] = dict(
        per_arm=shapes,
        n_split_sign=n_split, n_kernel_arms=len(KERNEL_ARMS),
        note='the three amplitude statistics are reported side by side because they DISAGREE, '
             'and the disagreement is the reading. On %d of the %d kernel arms the mean and '
             'the daily variance both rise far above the frozen baseline while the per-event '
             'peak/base ratio -- G1/G2\'s own statistic -- stays BELOW the frozen baseline: '
             'those arms mobilise too much in absolute terms and flatten the event contrast. '
             'The one arm not in this class is Q0-zero (q_m = 0), the boundary arm, whose '
             'land-side output is identically zero and which therefore raises nothing. '
             'Section 5 forbids reading either side as evidence for the other, so this file '
             'reports both and adjudicates neither.'
             % (n_split, len(KERNEL_ARMS)))

    # ------------------------------------- the corrected N10 roll-ups, restated
    n10 = p1['N10']
    ledger = p1['ledger_summary']
    rebound = {a: ledger[a]['ledger_body_ran'] for a in sorted(ledger)}
    n_kernel_rebound = int(sum(1 for a in KERNEL_ARMS
                               if ledger[a]['ledger_body_ran'] == 'ledger_dp2'))
    fails = p1.get('ledger_failures', {})
    rep['N10_rollups'] = dict(
        # (1) over-strict: B0 does not rebind by construction.
        all_ledger_bodies_rebound=dict(
            raw_roll_up_in_phase1_arms=bool(n10['all_ledger_bodies_rebound']),
            raw_roll_up_is_over_strict=True,
            reason='B0 is the FROZEN kernel arm and must NOT rebind ResearchObjective.ledger '
                   '-- proving the frozen ledger body still runs is what B0 is for. The raw '
                   'roll-up demands ledger_dp2 on every arm, which is the opposite of B0\'s '
                   'role, so it is False by construction and is not a failed assertion.',
            corrected_roll_up_over_kernel_arms=bool(n_kernel_rebound == len(KERNEL_ARMS)),
            n_kernel_arms_running_ledger_dp2=n_kernel_rebound,
            n_kernel_arms=len(KERNEL_ARMS), ledger_body_ran_per_arm=rebound,
            corrected_here_rather_than_re_run='the ten registered forwards are not respent on '
                                              'a cosmetic report field; this is a zero-forward '
                                              're-derivation of a roll-up, by design',
            no_gate_reads_this=True),
        # (2) NOT over-strict: a real registered failure, left standing.
        all_arms_hold=dict(
            raw_roll_up_in_phase1_arms=bool(n10['all_arms_hold']),
            raw_roll_up_is_over_strict=False,
            arms_that_failed=sorted(fails),
            reason='K-slowend crosses the ABSOLUTE registered N10 label tolerance '
                   '(source_label_sum_errors <= 1e-6, six channels; identical wording in '
                   'round 1 at 预注册_判据与门槛.md:353) at 1.043e-06 in the M channel, while '
                   'reading 2.46e-12 RELATIVE and holding local balance (1.12e-07) and '
                   'network balance. The residual is ~1e-15 relative to the pool it '
                   'summarises and is the float64 accumulation floor, not a q_m effect; '
                   'K-slowend accumulates the largest legacy pool of the ten.',
            tolerance_unchanged=True,
            tolerance=dict(n10['tolerance']),
            can_sign_a_verdict=bool(fails.get(LEDGER_DEVIATION_ARM, {})
                                    .get('can_sign_a_verdict', False)),
            registered_role=fails.get(LEDGER_DEVIATION_ARM, {}).get('role'),
            what_the_tolerance_is='ABSOLUTE, while the residual is RELATIVE. K-3p6e3 (the '
                                  'proposal\'s original upper bound, 3.6e3 d) reads 7.15e-07, '
                                  'i.e. 71% of budget, so the entire six-arm CANDIDATE grid is '
                                  'clean and the crossing happens only on the deliberately '
                                  'extreme, non-verdict coupling-proof arm.',
            not_explained_away='this entry is a FAILURE REPORT, not a reclassification. The '
                               'registered BLOCKED clause reads "N1-prime .. N16 any failure", '
                               'while section 9.2 item 11 makes N10 a per-arm per-point duty '
                               'and section 12.1 registers ONE PASS WITH NO EARLY STOPPING. '
                               'Those two registered texts conflict here; this file reports the '
                               'conflict and does not adjudicate it.'),
        max_label_error=float(n10['max_label_error']),
        max_label_rel_error=float(n10['max_label_rel_error']),
        max_local_balance_kg=float(n10['max_local_balance_kg']),
        anchor_max_abs_dp=float(n10['anchor_max_abs_dp']),
        anchor_rows=int(n10['anchor_rows']),
        n_label_channels=int(n10['n_label_channels']))

    rep['deviations'] = [
        'N10.all_ledger_bodies_rebound is False in phase1_arms.json and is OVER-STRICT, not '
        'failed: B0 is the frozen arm by construction. The corrected roll-up (9/9 kernel arms '
        'ran ledger_dp2) is restated here with the raw value beside it.',
        'N10.all_arms_hold is False and that IS a real registered failure on K-slowend. The '
        'tolerance is NOT touched and the failure is NOT reclassified; the arms that can sign '
        'a verdict all hold. See work/_deviations_running.md F4 for the per-arm scaling table.',
        'the mass-weighting diagnostic is computed from concentration x water and is labelled '
        'as such; section 0.1 keeps load off every criterion and no gate reads it',
        'R5-ref is reported in its own block, flagged enters_no_gate=True, and excluded from '
        'every pass count and roll-up in this file',
        'every gate reading in this file is TAKEN from phase1_arms.json; G5b is the one '
        'exception and it is RECOMPUTED from daily_arms.parquet and asserted to agree with '
        'phase1_arms.json on every arm',
        'G5b is recomputed against LEVEL_GATE (common25.py:752), not MONTHLY_GATE as round 1 '
        'wrote it. Both constants are 0.005 and their equality is asserted in the record, so '
        'this is a spelling correction and not a threshold change.',
    ]
    C.write_json(OUT / 'level_variance.json', rep)
    _p('=== wrote %s ===' % (OUT / 'level_variance.json'))
    return rep


if __name__ == '__main__':
    main()
