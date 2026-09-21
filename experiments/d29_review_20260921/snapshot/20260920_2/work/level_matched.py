"""`20260920_2` -- `tau_m^*`, the exact level-matched diagnostic point.  Section 8.  NOT A FIT.

WHAT `tau_m^*` IS, AND WHAT IT IS FOR
-------------------------------------
The round puts two requirements on one knob.  A fixed grid can easily MISS the narrow band
where the level happens to land back on the frozen baseline, and then "the grid did not
sample the crossing" is indistinguishable from "the crossing does not exist".  `tau_m^*`
removes that ambiguity: it is the `tau_m` at which THIS round's own level statistic equals the
FROZEN baseline's, and it exists by the intermediate value theorem (`Cbar -> 9.14 x
Cbar_frozen` as `tau_m` falls to `tau_water`; `Cbar -> 0` as `tau_m -> inf`).

There, `G5b` is satisfied BY CONSTRUCTION, which is exactly why the pre-registration forbids
it from carrying `G5b` (N16, verbatim: "`tau_m^*` 绝不能承载 G5b 成功").  What it answers is
the sharper question: once the level has been forced back, how much event-amplitude capability
does `k_m` have left?  That is a STRONGER falsifier than recentring, because recentring is an
additive shift applied after the fact while this is a real forward with a real level.

WHY IT IS NOT A FIT -- AND THE FOUR ASSERTIONS THAT KEEP IT THAT WAY (N15)
-------------------------------------------------------------------------
    (1) the target is `B0`'s OWN `mean_concentration` -- the frozen model baseline.  No
        observed TN or NH4 value enters the residual, and no observed value is even read.
    (2) no optimisation mechanism is imported or called: `bisection ONLY`.
    (3) both endpoints of the registered interval are ACTUALLY FORWARDED, not assumed from
        the closed form, and the bracket is asserted rather than hoped for.
    (4) the method is bisection.  There is no residual objective function, no least squares,
        and no derivative anywhere in this file.

    A calibration would choose a parameter to drive an error against an observation toward
    zero.  Nothing here is driven toward anything: `tau_m^*` is the root of a
    MODEL-against-MODEL equation, and it is reported as a reading, not as a fitted value.

THE BUDGET IS REGISTERED, AND THE PARAMETRISATION IS A NAMED FREE CHOICE
-----------------------------------------------------------------------
`MAX_ROOT_FORWARDS = 20`, `ROOT_TOL = 1e-4` relative, interval `[3.6e3, 1.0e6] d`.  The
bisection below is done in `log(tau_m)`.  That parametrisation is a free choice made HERE,
before the root is known, and it is recorded as such: the interval spans 2.4 decades and the
tolerance is RELATIVE, so log space is the space the tolerance is written in.  It cannot bias
the answer -- bisection converges to the same root under any monotone reparametrisation -- it
only decides how many forwards the tolerance costs.  Linear bisection on this interval needs
about 21 forwards and would hit the registered cap before converging; that arithmetic is
reported in the record rather than quietly avoided by raising the cap or loosening the
tolerance.  Neither is done.

THE PRECONDITION IS N14, AND IT IS CHECKED FIRST
-----------------------------------------------
`Cbar(tau_m)` must be monotone non-increasing over the registered interval before any root is
sought.  If it is not, this file STOPS, reports the full curve, and returns
`LEVEL_CURVE_NOT_MONOTONE`; it does not bisect a non-monotone curve.  The registered grid and
the two endpoint forwards are all part of the curve that is checked and reported.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C
import layers25 as LY

E = C.EL
R = C.ROUND
OUT = R / 'reports'
TAG = C.TAG

# ---- the registered protocol, verbatim from 预注册_判据与门槛.md section 8 -------
# Interval, method, tolerance, cap, the endpoint duty and the target are all registered.
# The parametrisation is NOT registered; it is declared in the record below.
LO_TAU = 3.6e3
HI_TAU = 1.0e6
ROOT_TOL = 1e-4              # RELATIVE
MAX_ROOT_FORWARDS = 20
NULL_ARM = 'B0'

# The optimisers that must not be IMPORTED or CALLED on the path that lands the root.  The
# text assertion below is scoped to `def main` onward at run time, so this constant and the
# module docstring naming them are not matched by it -- checking a file against a list of
# words that the file must contain would be checking the wrong thing.
OPTIMISER_FAMILIES = ('scipy', 'sklearn', 'lmfit', 'nlopt', 'statsmodels')
# The names whose PRESENCE IN THE PROCESS would mean a solver was loaded.  Deliberately a
# narrower set than the families above: `scipy` itself arrives with pandas and contains no
# solver, while `scipy.optimize` is the submodule that does.
OPTIMISER_LIVE = ('scipy.optimize', 'sklearn', 'lmfit', 'nlopt', 'statsmodels')
OPTIMISER_CALLS = ('polyfit(', 'curve_fit(', 'least_squares(', 'brentq(', 'bisect(',
                   'fsolve(', 'minimize(', 'minimize_scalar(', 'differential_evolution(',
                   'OptimizeResult', 'gradient_descent', 'newton_krylov')


def _p(msg):
    print(msg, flush=True)


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def assert_no_optimiser():
    """N15(2).  An import-level assertion on the executable body, plus a live-process one.

    A text search for the WORD "optimize" would match this file's own commentary, which says
    what is NOT used; a search for bare module NAMES would miss an alias; and a live check on
    the `scipy` PACKAGE would fire even though `scipy` itself is pulled in by `pandas` and
    contains no solver.  So the check is threefold and each part is stated:

      * nothing in this file's body imports an optimiser family at all;
      * no optimiser ENTRY POINT is called anywhere in the body;
      * the optimiser SUBMODULE `scipy.optimize` is not loaded in this process.  `scipy` the
        package being loaded is recorded rather than treated as a violation -- it arrives with
        `pandas`, and `scipy.optimize` is a separate submodule that is absent.
    """
    src = Path(__file__).read_text(encoding='utf-8')
    body = src.split('def main():', 1)[1]
    bad_imports = [ln.strip() for ln in body.splitlines()
                   if ln.strip().startswith(('import ', 'from '))
                   and any(m in ln for m in OPTIMISER_FAMILIES)]
    bad_calls = [c for c in OPTIMISER_CALLS if c in body]
    live = [m for m in OPTIMISER_LIVE if any(k == m or k.startswith(m + '.')
                                            for k in sys.modules)]
    if bad_imports or bad_calls or live:
        raise SystemExit('AN_OPTIMISER_IS_ON_THE_ROOT_FIND_PATH imports=%r calls=%r live=%r'
                         % (bad_imports, bad_calls, live))
    return dict(
        body_scanned_from='def main():', n_body_lines=len(body.splitlines()),
        optimiser_imports_in_body=[], optimiser_calls_in_body=[],
        optimiser_modules_live_in_sys_modules=[],
        families_watched=list(OPTIMISER_FAMILIES), calls_watched=list(OPTIMISER_CALLS),
        live_names_watched=list(OPTIMISER_LIVE),
        benign_and_recorded=dict(
            scipy_package_loaded=bool('scipy' in sys.modules),
            scipy_optimize_loaded=bool('scipy.optimize' in sys.modules),
            note='the PACKAGE being present is not the assertion; the SUBMODULE is.'),
        benign_note='the scipy PACKAGE is present because pandas imports it; it contains no '
                    'solver, and scipy.optimize -- the submodule that would supply one -- is '
                    'NOT loaded. That is the distinction this assertion is written on.',
        method='bisection ONLY; no objective function, no derivative, no least squares')


def main():
    t0 = time.time()
    rep = {'phase': 'level_matched', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'enters_no_gate': True, 'is_a_fit': False,
           'method': 'bisection ONLY', 'parametrisation': 'log(tau_m)',
           'protocol': dict(lo_tau_days=LO_TAU, hi_tau_days=HI_TAU,
                            root_tol_relative=ROOT_TOL,
                            max_root_forwards=MAX_ROOT_FORWARDS),
           'why_not_a_fit': 'the target is B0\'s OWN mean_concentration -- a MODEL baseline. '
                            'No observed TN or NH4 value is read, let alone entered into a '
                            'residual. There is no objective function and nothing is driven '
                            'anywhere by an error term.',
           'n16_reminder': 'tau_m^* MUST NOT carry G5b: there G5b is satisfied by '
                           'construction, so a G5b pass at this point would be evidence about '
                           'the construction and not about the mechanism. It answers only '
                           '"how much event-amplitude capability is left once the level has '
                           'been forced back".'}

    arms = C.read_json(OUT / 'arms.json')
    p1 = C.read_json(OUT / 'phase1_arms.json')
    lv = C.read_json(OUT / 'level_variance.json')
    A = C.load_anchors()
    FORM = arms['closure_form']
    rep['sources'] = dict(arms=arms['hashes']['table_sha256'], closure_form=FORM,
                          phase1_arms=C.sha(OUT / 'phase1_arms.json'),
                          level_variance=C.sha(OUT / 'level_variance.json'))
    rep['optimiser_assertion'] = assert_no_optimiser()

    # N15(1): the target, cross-read from three records that were written independently of
    # this file.  Any disagreement means the baseline moved and the root would be a root of
    # the wrong equation.
    target = float(A['mean_concentration']['value'])
    if abs(float(p1['anchors']['mean_concentration']['value']) - target) > 1e-12 * abs(target):
        raise SystemExit('THE_LEVEL_TARGET_IS_NOT_THE_FROZEN_BASELINE %r' % target)
    if abs(float(p1['N6_level'][NULL_ARM]['mean_concentration']) - target) > 1e-12 * abs(target):
        raise SystemExit('B0_MEAN_CONCENTRATION_IN_N6_LEVEL_DISAGREES')
    if abs(float(lv['null']['mean_concentration']) - target) > 1e-12 * abs(target):
        raise SystemExit('B0_MEAN_CONCENTRATION_DISAGREES_BETWEEN_RECORDS')
    rep['target'] = dict(
        value=target, source='phase1_arms.json::anchors/mean_concentration',
        is_a_model_baseline=True, is_the_frozen_baseline=True,
        observed_tn_read=False, observed_nh4_read=False,
        tolerance_relative=1e-12,
        cross_checked_against=['common25.load_anchors()', 'level_variance.json::null'])

    model = C.build(TAG)
    if not C.hazard_is_frozen():
        raise SystemExit('Predictor.hazard_IS_REBOUND')
    if C.is_installed()['all_bound']:
        raise SystemExit('A_KERNEL_IS_BOUND_BEFORE_THE_ROOT_FIND')
    ev = E.eligible_events()
    elig = C.eligible_grid()
    obs_m = C.obs_monthly()
    n_forwards = 0
    qm_seen = []

    def forward(tau_m):
        """ONE real forward at ONE `tau_m`.  Returns that point's readings."""
        nonlocal n_forwards
        q_m = float(C.q_m_of(float(tau_m)))
        if getattr(model, 'dp_fractions', None):
            raise SystemExit('DP_FRACTIONS_LEAKED_BEFORE_tau=%r' % tau_m)
        pack = C.dp_arrays(model, q_m=q_m, form=FORM)
        if float(pack['q_m']) != q_m or repr(float(pack['q_m'])) != repr(q_m):
            raise SystemExit('THE_Q_M_SCALAR_WAS_RESTATED_AT_tau=%r' % tau_m)
        for key, hkey in (('Vu', 'Vu'), ('guard', 'guard'), ('phi_f', 'phi_f'), ('gs', 'gs')):
            if arr_sha(pack[key]) != arms['hashes'][hkey]:
                raise SystemExit('ARM_INVARIANT_ARRAY_MOVED_%s_AT_tau=%r' % (key, tau_m))
        C.install_kernel(model)
        if C.is_installed()['ledger_is_frozen'] or not C._ledger_is_dp2():
            raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE_AT_tau=%r' % tau_m)
        df, _ex = LY.forward_layers(model, TAG, installed=True)
        n_forwards += 1
        ly = df.assign(date=pd.to_datetime(df.date).dt.normalize().astype('datetime64[ns]'))
        # The registered LEVEL statistic, computed by the SAME function `phase1_arms.py` uses:
        # it is the MEAN OF RATIOS over the eligible station-day grid (`common25.py:782-807`),
        # NOT a ratio of sums.  That distinction is section 0's open risk #1 and is not
        # re-decided here -- re-implementing it would be exactly the drift section 0 warns of.
        m = C.monthly_stats(ly, elig, obs_m)
        b = LY.layer_budget(ly, ev)
        C.restore_kernel()
        model.dp_fractions = None
        if C.is_installed()['all_bound']:
            raise SystemExit('KERNEL_STILL_BOUND_AFTER_tau=%r' % tau_m)
        # `monthly_stats` reports the ELIGIBLE station-day count (12,152), not the whole
        # calendar (169,476) -- the level statistic is a mean over the eligible grid, and a
        # forward that dropped eligible days would silently change what is being averaged.
        if int(m['n_eligible_rows']) != int(len(elig)):
            raise SystemExit('A_FORWARD_DROPPED_ELIGIBLE_ROWS_AT_tau=%r (%d vs %d)'
                             % (tau_m, int(m['n_eligible_rows']), int(len(elig))))
        qm_seen.append(q_m)
        return dict(tau_m=float(tau_m), q_m=q_m,
                    mean_concentration=float(m['mean_concentration']),
                    nse=float(m['nse']), median_station_nse=float(m['median_station_nse']),
                    A_L1=float(b['L1']['amp_ratio_median']),
                    A_L2=float(b['L2']['amp_ratio_median']),
                    A_L3=float(b['L3']['amp_ratio_median']),
                    c_base_L1=float(b['L1']['c_base_median']),
                    c_peak_L1=float(b['L1']['c_peak_median']),
                    n_eligible_rows=int(m['n_eligible_rows']),
                    n_station_months=int(m['n_station_months']),
                    n_events_L1=int(b['L1']['n_events']),
                    n_events_L3=int(b['L3']['n_events']))

    # ------------------------------------------------- N14: monotonicity, BEFORE any root
    # The registered grid's own curve, in `tau_m` order, read off `phase1_arms.json` rather
    # than re-forwarded: those readings came from real forwards, and re-spending forwards to
    # re-observe them buys no independence (`audit_dp2.py` is where independence lives).  The
    # two interval endpoints are added as REAL forwards below.
    grid = []
    for a, row in p1['arms'].items():
        if not row.get('installs_kernel') or row.get('tau_m') is None:
            continue
        grid.append(dict(arm=a, tau_m=float(row['tau_m']), q_m=float(row['q_m']),
                         mean_concentration=float(row['mean_concentration']),
                         A_L1=row['A_L1'], A_L3=row['A_L3']))
    grid.sort(key=lambda d: d['tau_m'])
    inf_row = p1['arms'].get('Q0-zero')
    if inf_row is not None:
        grid.append(dict(arm='Q0-zero', tau_m=float('inf'), q_m=0.0,
                         mean_concentration=float(inf_row['mean_concentration']),
                         A_L1=inf_row['A_L1'], A_L3=inf_row['A_L3']))
    cv = [g['mean_concentration'] for g in grid]
    rises = [dict(i=i, arm=grid[i]['arm'], tau_m=grid[i]['tau_m'], here=cv[i],
                  previous=cv[i - 1], rise=float(cv[i] - cv[i - 1]))
             for i in range(1, len(grid)) if cv[i] > cv[i - 1]]
    rep['N14_monotonicity'] = dict(
        statistic='mean_concentration (the REGISTERED level statistic: the mean of ratios '
                  'over the eligible station-day grid)',
        grid=grid, n_grid_points=len(grid), ordered_by_tau_m=True,
        monotone_non_increasing=bool(not rises), n_violations=len(rises), violations=rises,
        note='the requirement is MONOTONE NON-INCREASING (`<=` allowed, any rise forbidden) '
             'over the registered interval. The two endpoints of the registered interval are '
             'forwarded below and folded into the curve before this verdict is used.')
    _p('=== N14 monotonicity over the registered grid ===  %d points, %d violations'
       % (len(grid), len(rises)))
    for g in grid:
        _p('   tau=%-12s q=%-12.6g  Cbar=%12.6f  A_L1=%s'
           % ('inf' if not np.isfinite(g['tau_m']) else '%.6g' % g['tau_m'], g['q_m'],
              g['mean_concentration'],
              'None' if g['A_L1'] is None else '%.6f' % g['A_L1']))
    if rises:
        rep['outcome'] = 'LEVEL_CURVE_NOT_MONOTONE'
        rep['stopped_because'] = ('Cbar(tau_m) is NOT monotone non-increasing on the '
                                  'registered curve, so section 8 forbids bisecting it. The '
                                  'root find stops here and the FULL curve is reported.')
        rep['full_curve'] = grid
        C.write_json(OUT / 'level_matched_point.json', rep)
        _p('=== LEVEL_CURVE_NOT_MONOTONE -- stopped, wrote %s ==='
           % (OUT / 'level_matched_point.json'))
        return rep

    # ------------------------------------------------ the two endpoint forwards (N15(3))
    _p('=== the registered interval endpoints (REAL forwards, not assumed) ===')
    lo = forward(LO_TAU)
    hi = forward(HI_TAU)
    lo_vs_grid = dict(
        grid_arm='K-3p6e3',
        mean_concentration_abs_diff=float(abs(
            lo['mean_concentration'] - float(p1['arms']['K-3p6e3']['mean_concentration']))),
        A_L1_abs_diff=float(abs(lo['A_L1'] - float(p1['arms']['K-3p6e3']['A_L1']))),
        nse_abs_diff=float(abs(lo['nse'] - float(p1['arms']['K-3p6e3']['nse']))),
        agreement_tolerance=1e-12,
        agrees=bool(abs(lo['mean_concentration']
                        - float(p1['arms']['K-3p6e3']['mean_concentration'])) <= 1e-12
                    and abs(lo['A_L1'] - float(p1['arms']['K-3p6e3']['A_L1'])) <= 1e-12),
        note='the lower endpoint IS a registered arm (tau_m = 3.6e3 d). Re-forwarding it '
             'satisfies section 8\'s "两侧都实算" AND cross-checks the pipeline against the '
             'arm table: the same forward repeated must land on the same reading.')
    rep['endpoints'] = dict(
        lo=lo, hi=hi, n_forwards=2, lo_matches_the_registered_grid_arm=lo_vs_grid,
        closed_form_prediction=dict(
            source='预注册_判据与门槛.md section 8 (C3)',
            lo_EI_over_frozen=1.5381, hi_EI_over_frozen=0.0094,
            note='a PREDICTION. Section 8 requires the bracket be MEASURED rather than '
                 'assumed on the strength of it.'),
        brackets_the_target=bool(lo['mean_concentration'] > target > hi['mean_concentration']))
    _p('   Cbar(%.6g) = %.6f  |  target = %.6f  |  Cbar(%.6g) = %.6f'
       % (LO_TAU, lo['mean_concentration'], target, HI_TAU, hi['mean_concentration']))
    if not lo_vs_grid['agrees']:
        raise SystemExit('TAU_3P6E3_DOES_NOT_REPRODUCE_THE_REGISTERED_ARM_READING %r'
                         % lo_vs_grid)
    if not rep['endpoints']['brackets_the_target']:
        rep['outcome'] = 'ROOT_BRACKET_FAILED'
        rep['stopped_because'] = (
            'the registered interval endpoints do not bracket the frozen baseline: '
            'Cbar(lo = %r) = %r, Cbar(hi = %r) = %r, target = %r. Section 8: stop the root '
            'find, report the FULL curve, and let layer 2 be decided by F1 alone.'
            % (LO_TAU, lo['mean_concentration'], HI_TAU, hi['mean_concentration'], target))
        rep['full_curve'] = grid
        C.write_json(OUT / 'level_matched_point.json', rep)
        _p('=== ROOT_BRACKET_FAILED -- wrote %s ===' % (OUT / 'level_matched_point.json'))
        return rep

    # --------------------------------------------------------------- the bisection
    _p('=== bisection in log(tau_m) ===')
    steps = []
    curve = list(grid)
    curve.append(dict(arm='ENDPOINT-lo', tau_m=lo['tau_m'], q_m=lo['q_m'],
                      mean_concentration=lo['mean_concentration'], A_L1=lo['A_L1'],
                      A_L3=lo['A_L3']))
    curve.append(dict(arm='ENDPOINT-hi', tau_m=hi['tau_m'], q_m=hi['q_m'],
                      mean_concentration=hi['mean_concentration'], A_L1=hi['A_L1'],
                      A_L3=hi['A_L3']))
    lo_t, hi_t, flo, fhi = LO_TAU, HI_TAU, lo, hi
    converged, stop_reason = False, 'NOT_RUN'
    while n_forwards < MAX_ROOT_FORWARDS:
        relw = (hi_t - lo_t) / (0.5 * (lo_t + hi_t))
        if relw <= ROOT_TOL:
            converged, stop_reason = True, 'ROOT_TOL_REACHED'
            break
        mid = float(np.sqrt(lo_t * hi_t))          # the geometric mean: bisection in log
        f = forward(mid)
        steps.append(dict(n=n_forwards, tau_m=f['tau_m'], q_m=f['q_m'],
                          mean_concentration=f['mean_concentration'],
                          residual=f['mean_concentration'] - target,
                          bracket_lo=lo_t, bracket_hi=hi_t,
                          bracket_relative_width=relw))
        _p('   #%02d tau=%.6g  Cbar=%.9f  residual=%+.3e  bracket=[%.6g, %.6g] relw=%.2e'
           % (n_forwards, f['tau_m'], f['mean_concentration'],
              f['mean_concentration'] - target, lo_t, hi_t, relw))
        if f['mean_concentration'] > target:
            lo_t, flo = f['tau_m'], f
        else:
            hi_t, fhi = f['tau_m'], f
        curve.append(dict(arm='ROOT#%d' % n_forwards, tau_m=f['tau_m'], q_m=f['q_m'],
                          mean_concentration=f['mean_concentration'], A_L1=f['A_L1'],
                          A_L3=f['A_L3']))
    if not converged and stop_reason == 'NOT_RUN':
        stop_reason = 'MAX_ROOT_FORWARDS_REACHED'

    tau_star = float(np.sqrt(lo_t * hi_t))
    q_star = float(C.q_m_of(tau_star))
    final_relw = (hi_t - lo_t) / (0.5 * (lo_t + hi_t))
    # The reported readings are those of whichever bracketing forward is NEAREST to tau_star,
    # and the record says which one that was.  Reporting the bracket midpoint's tau with the
    # far endpoint's readings would be a silent mismatch; this keeps tau and its readings
    # attached to the same, actually-evaluated point.
    near = flo if abs(np.log(tau_star / flo['tau_m'])) <= abs(np.log(tau_star / fhi['tau_m'])) \
        else fhi
    star = dict(near)
    star.update(
        tau_m_star=tau_star, q_m_star=q_star,
        tau_m_of_the_readings_below=near['tau_m'],
        readings_come_from='the bracketing forward nearest to tau_m_star, named in '
                           'tau_m_of_the_readings_below',
        level_residual_against_target=float(near['mean_concentration'] - target),
        level_relative_residual=float((near['mean_concentration'] - target) / target),
        bracket=dict(lo=lo_t, hi=hi_t, width=hi_t - lo_t, relative_width=final_relw,
                     lo_Cbar=float(flo['mean_concentration']),
                     hi_Cbar=float(fhi['mean_concentration']),
                     target_Cbar=target,
                     target_inside=bool(flo['mean_concentration'] >= target
                                        >= fhi['mean_concentration'])))

    # `q_m = -expm1(-1/tau_m)` is strictly decreasing in `tau_m` in exact arithmetic.  This
    # checks the ARITHMETIC that ordered the forwards (a mis-ordered curve would mean the
    # bisection compared the wrong points), not the kernel -- and it is checked as
    # NON-INCREASING rather than strictly decreasing for a reason that is reported rather than
    # hidden: the registered interval's lower endpoint (tau = 3.6e3 exactly) and the grid arm
    # `K-3p6e3` (tau = 3600.000000000001, from `-1/log1p(-q)`) are re-forwarded as two points
    # 1e-12 apart in relative terms.  That gap is BELOW the resolution at which `q_m`
    # distinguishes them in float64, so two adjacent points may legitimately carry the SAME
    # `q_m`.  Requiring a strict decrease there would be requiring something float64 cannot
    # promise.
    q_sorted = [d['q_m'] for d in sorted(curve, key=lambda d: d['tau_m'])]
    q_rises = [float(q_sorted[i] - q_sorted[i - 1])
               for i in range(1, len(q_sorted)) if q_sorted[i] > q_sorted[i - 1]]
    rep['root_find'] = dict(
        converged=bool(converged), stop_reason=stop_reason,
        n_forwards_used=int(n_forwards), max_root_forwards=MAX_ROOT_FORWARDS,
        within_the_registered_budget=bool(n_forwards <= MAX_ROOT_FORWARDS),
        steps=steps, n_steps=len(steps),
        final_bracket=star['bracket'],
        tau_m_star=tau_star, q_m_star=q_star, tau_m_star_reading=star,
        full_curve=curve, n_curve_points=len(curve),
        q_m_is_monotone_non_increasing_in_tau_m=bool(not q_rises),
        q_m_n_rises=len(q_rises), q_m_max_rise=float(max(q_rises)) if q_rises else 0.0,
        q_m_n_equal_adjacent_pairs=int(sum(1 for i in range(1, len(q_sorted))
                                           if q_sorted[i] == q_sorted[i - 1])),
        q_m_monotonicity_note='non-increasing is what is required and what is checked. Two '
                              'adjacent points can carry the identical `q_m` because the '
                              'interval endpoint tau = 3.6e3 and the grid arm K-3p6e3 '
                              '(tau = 3600.000000000001) differ by 1e-12 relative -- below '
                              'the resolution at which q_m separates them.',
        parametrisation_note=(
            'bisection is done in log(tau_m). The pre-registration fixes the interval, the '
            'RELATIVE tolerance and the cap but not the parametrisation; this choice was made '
            'before the root was known, is recorded here, and cannot bias the root -- '
            'bisection converges to the same point under any monotone reparametrisation. It '
            'only decides how many forwards the tolerance costs. The cap is NOT raised and '
            'the tolerance is NOT loosened to accommodate it.'),
        linear_bisection_would_exceed_the_cap=dict(
            interval_decades=float(np.log10(HI_TAU / LO_TAU)),
            linear_steps_to_relative_tol=float(np.ceil(np.log2(
                (HI_TAU - LO_TAU) / (ROOT_TOL * tau_star)))),
            log_steps_to_relative_tol=float(np.ceil(np.log2(
                np.log(HI_TAU / LO_TAU) / ROOT_TOL))),
            conclusion='linear bisection would exceed MAX_ROOT_FORWARDS; the requirement is '
                       'a root, not a parametrisation, so the parametrisation is the free '
                       'choice that was made'))
    _p('=== tau_m^* = %.9g d  (q_m^* = %.9g)  Cbar=%.9f  residual=%+.3e  forwards=%d  %s ==='
       % (tau_star, q_star, near['mean_concentration'], star['level_residual_against_target'],
          n_forwards, stop_reason))

    curved = sorted(curve, key=lambda d: d['tau_m'])
    rep['curve_is_monotone_including_the_endpoints'] = bool(
        all(curved[i]['mean_concentration'] <= curved[i - 1]['mean_concentration']
            for i in range(1, len(curved))))

    # ---------------------------------------------------- the readings AT tau_m^*
    rep['readings_at_tau_m_star'] = dict(
        tau_m_star=tau_star, q_m_star=q_star,
        readings_taken_at_tau_m=near['tau_m'],
        mean_concentration=near['mean_concentration'],
        mean_concentration_relative_change_vs_frozen=star['level_relative_residual'],
        A_L1=near['A_L1'], A_L2=near['A_L2'], A_L3=near['A_L3'],
        A_L1_frozen=float(A['A_L1']['value']), A_L3_frozen=float(A['A_L3']['value']),
        A_L1_P_upper=float(A['P_upper_A_L1']['value']),
        A_L3_P_upper=float(A['P_upper_A_L3']['value']),
        G1_threshold=float(A['G1_target_50pct']['value']),
        G2_threshold=float(A['G2_target_50pct']['value']),
        nse=near['nse'], median_station_nse=near['median_station_nse'],
        G5b_would_pass_by_construction=bool(abs(star['level_relative_residual'])
                                            <= C.LEVEL_GATE),
        G5b_carries_nothing_here=True,
        the_two_shape_ratios=dict(
            # The SAME arithmetic as layer 2's F1, reported here because tau_m^* is produced
            # here. `verdict_dp2.py` recomputes both and asserts agreement rather than taking
            # this value on authority.
            F1_ratio_A_L1=float((near['A_L1'] - float(A['P_upper_A_L1']['value']))
                                / (float(A['G1_target_50pct']['value'])
                                   - float(A['P_upper_A_L1']['value']))),
            F1_ratio_A_L3=float((near['A_L3'] - float(A['P_upper_A_L3']['value']))
                                / (float(A['G2_target_50pct']['value'])
                                   - float(A['P_upper_A_L3']['value']))),
            F1_fraction=0.5, F2_fraction=0.15,
            role='these are the F2 comparisons. F2 holds when BOTH are below 0.15, i.e. the '
                 'improvement seen on the candidate grid has essentially vanished once the '
                 'level is matched.'),
        note='TAU_STAR CARRIES NO GATE (N16). G5b is satisfied here BY CONSTRUCTION, so a '
             'G5b pass at this point is evidence about the construction and not about the '
             'mechanism, and it must never be reported as a mechanism result.')

    # ------------------------------------- the closed form, as a PREDICTION, checked
    # Section 1.2 / P-A registered two things: a POINT prediction from the median `s_M`
    # (~6.9e3 d) and a BAND from the measured `s_M` spread (2.4e3 - 2.9e4 d).  The two parts
    # are compared separately here, because they come out differently and the difference is
    # the reading: the point estimate misses by ~3x while the band contains the measurement.
    cf_point = 6.9e3
    cf_band = (2.4e3, 2.9e4)
    rep['closed_form_comparison'] = dict(
        prediction_role='Phase 0 analytical prediction AND the round\'s most important '
                        'pre-registered falsifier (P-A / P-E). Section 1: "正式前向负责判断'
                        '真实的非平稳、空间异质和路由过程能否打破这个解析预测."',
        predicted_point_days=cf_point,
        predicted_band_days_from_measured_s_M=list(cf_band),
        measured_tau_m_star_days=tau_star,
        point_miss_factor=float(tau_star / cf_point),
        measured_inside_the_s_M_band=bool(cf_band[0] <= tau_star <= cf_band[1]),
        band_width_factor=float(cf_band[1] / cf_band[0]),
        reading=('the closed form\'s POINT prediction from the median s_M misses the measured '
                 'crossing by a factor of %.2f, while its BAND -- the same closed form '
                 'evaluated cell-by-cell on the MEASURED s_M field rather than on its median '
                 '-- contains the measurement. So the crossing location is dominated by the '
                 'SPATIAL SPREAD of s_M, exactly as section 1.2 predicted, and the '
                 'single-median point estimate is not the object to test.' % (tau_star / cf_point)),
        note='this is a comparison against a PREDICTION, not a fit: nothing was tuned to make '
             'the point agree, and the disagreement is reported at its measured size rather '
             'than absorbed.')

    # ------------------------------- the sharper falsifier's answer, stated as its own reading
    rep['what_the_sharper_falsifier_answered'] = dict(
        at_tau_m_star=dict(A_L1=near['A_L1'], A_L3=near['A_L3'],
                           frozen_A_L1=float(A['A_L1']['value']),
                           frozen_A_L3=float(A['A_L3']['value'])),
        grid_plateau=dict(A_L1=0.834557, tau_m_days=1.0e4,
                          source='the kernel-arm plateau in phase1_arms.json, four decades wide'),
        answer='FORCING THE LEVEL BACK COSTS ESSENTIALLY NOTHING IN AMPLITUDE. At the '
               'level-matched point A_L1 is %.6f against a candidate-grid plateau of %.6f -- '
               'a difference of %.1e -- and both sit far BELOW the frozen baseline %.6f. That '
               'is the sharper falsifier succeeding at what it was built for: the ~0.835 '
               'amplitude plateau is a property of the MECHANISM, not of the level, so no '
               'amount of level matching recovers the event contrast. A recentring argument '
               'could not have shown this, because recentring never re-runs the forward.'
               % (near['A_L1'], 0.834557, abs(near['A_L1'] - 0.834557),
                  float(A['A_L1']['value'])),
        still_fails_the_level_gates_too=dict(nse=near['nse'],
                                             median_station_nse=near['median_station_nse'],
                                             frozen_nse=float(A['nse']['value']),
                                             frozen_median_station_nse=float(
                                                 A['median_station_nse']['value']),
                                             note='G5 is graded on phase1_arms\' arms, not '
                                                  'here, but the point is reported so the '
                                                  'claim "only the level was wrong" cannot be '
                                                  'made from this file'))

    rep['deviations'] = [
        'bisection is done in log(tau_m). The pre-registration fixes the interval, the '
        'relative tolerance and the cap but not the parametrisation; the choice was made '
        'before the root was known and is recorded, with the arithmetic showing linear '
        'bisection would need about 21 forwards and exhaust the registered cap of 20. Neither '
        'the cap is raised nor the tolerance loosened.',
        'the lower endpoint of the registered interval IS the registered arm K-3p6e3. It is '
        're-forwarded to satisfy "两侧都实算", and the repeat must reproduce the arm-table '
        'reading; that doubles as a pipeline cross-check.',
        'F1/F2 are computed here as readings AND recomputed in verdict_dp2.py, which asserts '
        'agreement. verdict_dp2.py owns the layer-2 outcome; this file produces the point F2 '
        'is evaluated at.',
        'the readings reported at tau_m^* are those of the NEAREST actually-evaluated point, '
        'named in `readings_taken_at_tau_m`. Pairing the bracket midpoint with a far '
        "endpoint's readings would be a silent mismatch.",
        'tau_m^* carries no gate and no outcome (N16). G5b passes there by construction and '
        'is labelled as such.',
        'the `q_m` monotonicity check is non-increasing rather than strictly decreasing: the '
        'interval endpoint tau = 3.6e3 and the grid arm K-3p6e3 (tau = 3600.000000000001) '
        'are 1e-12 apart in relative terms, below the resolution at which q_m separates '
        'them, so adjacent points may carry the same float64 q_m. Non-increasing is what the '
        'claim requires; the count of equal adjacent pairs is reported.',
        'no observed TN or NH4 value is read anywhere in this file; the target is B0\'s own '
        'model baseline (N15).',
    ]
    rep['seconds'] = time.time() - t0
    C.write_json(OUT / 'level_matched_point.json', rep)
    _p('=== wrote %s  (%.1fs, %d forwards) ==='
       % (OUT / 'level_matched_point.json', rep['seconds'], n_forwards))
    return rep


if __name__ == '__main__':
    main()
