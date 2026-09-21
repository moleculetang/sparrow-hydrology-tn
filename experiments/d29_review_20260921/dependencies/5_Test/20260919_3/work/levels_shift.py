"""Why the split can move 16 points and the amplitude 2: the levels, and the algebra.

THE READING THIS EXPLAINS
-------------------------
At `g = 2` the intervention moves the carrier split by a median of
`0.09473163262398915`, yet `A_L1` moves from `1.0400772083480145` to `1.0607464383`.
A criterion built as `median(C_peak / C_base)` is a RATIO OF TWO LEVELS, and the
intervention is a DAILY driver that does not know which window a day belongs to.  So if
it lifts the base window and the peak window together, the lift CANCELS in the criterion
while remaining plainly visible in the levels.  This module reports the levels so that
statement is a reading rather than a story.

WHY THE COMMON MODE IS EXPECTED, FROM THE FROZEN KERNEL
-------------------------------------------------------
`vendor/research/closures.py:18-27`, `mode = owner.cap = False`:

    av   = max(M + inp - demand, 0)
    prob = 1 - exp(-h)
    E    = av * prob                       # the mobilised total: NO `f` HERE
    fast = E * f
    pre  = L_prev + E * (1 - f)
    slow = pre * l                         # l = lower_release, ~7.4e-3
    L    = pre * (1 - l)

`L1_local = fast + slow`, so on any day, with `L_prev` the pool carried in,

    L1 = E*f + (L_prev + E*(1-f))*l                                        (A)

and at CONSTANT `E`, `f` the pool settles at `L* = E(1-f)(1-l)/l`, giving

    slow* = (L* + E(1-f))*l = E(1-f)   =>   L1* = E*f + E(1-f) = E         (B)

so `dL1*/df = 0`: **at steady state the partition does not appear in the level at all.**
It only re-times.  The excursion from (B) decays on the pool's own timescale `1/l`, which
is ~136 days, while the event windows are 6-20 days long.  The fraction of the partition
excursion that has ESCAPED the pool by day `k` is `1 - (1-l)^k`, so the order of the
expected fractional amplitude response is

    ~ [1 - (1-l)^k] * (typical |delta f|)                                  (C)

DIFFERENTIATING (A) GIVES THE MECHANISM EXACTLY
------------------------------------------------
Unrolling `L`:  `L_t = (1-l) * sum_{j>=0} (1-l)^j E_{t-j} (1-f)`, so

    dL1_t/df = (1-l) * ( E_t - l * sum_{j>=0} (1-l)^j E_{t-1-j} )          (D)

i.e. the partition moves the level on day `t` exactly to the extent that TODAY'S
mobilisation exceeds a trailing exponentially-weighted mean of mobilisation with mean lag
`1/l`.  (D) vanishes identically for constant `E`, which is (B) again -- so (B) and (D) are
the same fact.  It also fixes the SIGN on either side of an event:

  * on the peak day `E_t` is high  => `dL1/df > 0`  -- the split HELPS;
  * in the weeks after, `E_t` is low against a high recent mean => `dL1/df < 0` -- the
    drawdown CLAWS THE GAIN BACK.

That is the registered side-effect of §1.4 (post-event compensatory deficit) appearing as
algebra rather than as a worry, and it is why moving `f` harder (g=4) stops paying.
(D) is a DERIVATION, not a measurement: nothing in this module evaluates it.  It is stated
so a reader can check it, and it is the reason the measured gains below have the scale they
have and not a larger one.

Both terms are measured here, not assumed: `l` from `lower_release.npy`, `k` from the
event windows, `|delta f|` from `reports/phase1_l1.json`.  (C) is an ADIABATIC argument --
it treats `E`, `f` as locally constant, which they are not -- so it is reported as a
SCALE, never as a gate, and it cannot change the verdict.

This module builds no model, installs no hazard and computes no amplitude: it reads
`reports/phase1_l1.json` and one frozen `.npy`, and writes derived ratios so that every
number the report cites has a producer.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(r'E:\SPARROW\5_Test')
REPORTS = ROOT / '20260919_3/reports'
LR = ROOT / '20260916_2/data/domains/FULL24/lower_release.npy'
FF = ROOT / '20260916_2/data/domains/FULL24/fast_fraction.npy'
OUT = REPORTS / 'levels_shift.json'
GRID = ('0', '0.25', '0.5', '1', '2', '4')


def main():
    d = json.loads((REPORTS / 'phase1_l1.json').read_text(encoding='utf-8'))
    l = np.load(LR).astype(np.float64)
    l_med, l_min, l_max = float(np.median(l)), float(l.min()), float(l.max())
    tau = 1.0 / l_med

    out = dict(purpose='levels behind the ratio criterion, and the adiabatic scale of the '                       'common-mode cancellation',
               lower_release=dict(median=l_med, min=l_min, max=l_max,
                                  turnover_days_median=tau,
                                  note='l is fitted and varies in time and space; the '
                                       'steady-state identity (B) is exact only for '
                                       'constant E and f, so this is an adiabatic scale'),
               layers={}, expectation={})
    # The reachability ceiling of this whole family: f0 == 0 is absorbing under
    # fN = aq_eff*f0/(aq_eff*f0 + 1 - f0) for EVERY gamma, so those cells are outside the
    # intervention's reach no matter how hard it is pushed.  Counted here so the report can
    # cite an interior fraction that has a producer rather than an implied complement.
    ff = np.load(FF).astype(np.float64)
    out['f0_census'] = dict(
        n_cells=int(ff.size),
        frac_exactly_zero=float((ff == 0.0).mean()),
        frac_exactly_one=float((ff == 1.0).mean()),
        frac_strictly_interior=float(((ff > 0.0) & (ff < 1.0)).mean()),
        max=float(ff.max()), min_positive=float(ff[ff > 0.0].min()),
        note='f0 == 0 => fN == 0 for every gamma (absorbing), so the intervention cannot '
             'reach those cells at all; this is a PRIOR ceiling on the family, independent '
             'of the grid and of the gate')
    for name, fam in (('grid', GRID), ('limit_case', None)):
        keys = fam if fam else ['limit_case']
        for k in keys:
            node = d['per_gamma'][k] if fam else d['limit_case']
            for lay in ('L1', 'L2', 'L3'):
                b = d['per_gamma']['0'][lay]
                z = node[lay]
                t = out['layers'].setdefault(lay, {})
                t[k] = dict(
                    c_base_median=z['c_base_median'], c_peak_median=z['c_peak_median'],
                    A=z['amp_ratio_median'],
                    base_ratio_to_g0=z['c_base_median'] / b['c_base_median'],
                    peak_ratio_to_g0=z['c_peak_median'] / b['c_peak_median'],
                    ratio_of_medians_shift=((z['c_peak_median'] / z['c_base_median'])
                                            / (b['c_peak_median'] / b['c_base_median'])),
                    median_of_ratios_shift=z['amp_ratio_median'] / b['amp_ratio_median'],
                    n_peak_min=int(z['n_peak_min']), n_peak_max=int(z['n_peak_max']))

    # ---- the adiabatic scale (C), evaluated at the measured inputs ----------------
    for k in GRID[1:]:
        s = d['per_gamma'][k]['split_population']
        df = s['median_abs_change']
        kmax = d['per_gamma'][k]['L1']['n_peak_max']
        kmin = d['per_gamma'][k]['L1']['n_peak_min']
        esc = {('%d' % kk): float(1.0 - (1.0 - l_med) ** kk) for kk in (kmin, kmax)}
        out['expectation'][k] = dict(
            median_abs_delta_f=df, k_min=int(kmin), k_max=int(kmax),
            escaped_fraction=esc,
            predicted_amplitude_gain_min=float(esc['%d' % kmin] * df),
            predicted_amplitude_gain_max=float(esc['%d' % kmax] * df),
            measured_median_of_ratios_gain=float(
                out['layers']['L1'][k]['median_of_ratios_shift'] - 1.0),
            measured_ratio_of_medians_gain=float(
                out['layers']['L1'][k]['ratio_of_medians_shift'] - 1.0),
            note='(C) with the measured |delta f|; measured gains are given under BOTH '
                 'summaries because they disagree in sign at some g -- `A_L1` is a median '
                 'of PAIRED ratios, while a ratio of medians is a different functional')
    # ---- what the gate ASKS FOR, against what the algebra PERMITS ---------------
    a = d['anchors']
    need = {lay: a['G%d_target_50pct' % i] / a['A_%s' % lay] - 1.0
            for lay, i in (('L1', 1), ('L3', 2))}
    band = [out['expectation'][k]['predicted_amplitude_gain_max'] for k in GRID[1:]]
    out['gate_requirement'] = dict(
        required_fractional_gain=need,
        required_note='(G1 target / g=0 baseline) - 1: how much of a RELATIVE rise in '
                      'median(C_peak/C_base) the 50%-of-gap rule demands',
        permitted_by_adiabatic_scale=dict(
            max_over_grid=float(max(band)), mean_over_grid=float(sum(band) / len(band)),
            n_points=len(band)),
        shortfall_factor_at_best=float(need['L1'] / max(band)),
        saturated_fraction_at_g2=float(
            out['layers']['L1']['2']['median_of_ratios_shift'] - 1.0) / max(band),
        measured_best_gain=float(d['per_gamma']['2']['closure_L1']),
        measured_best_gain_note='closure_L1 at g=2, the grid maximum: this is the fraction '
                                'of the L1 gap actually closed, not a fractional gain',
        reading='the gate demands a relative amplitude gain an order of magnitude above '
                'what the kernel\'s own pool timescale permits for a partition change of '
                'the measured size, so this is a BOUND, not an under-tuned knob',
        summary_statistic_disagreement=dict(
            note='`A` is the median of per-event PAIRED ratios; a ratio of the two medians '
                 'is a different functional of the same 214 events, and they disagree in '
                 'SIGN at some g and about WHICH g is best -- so "the response" is not one '
                 'number even before it is compared with a gate',
            per_g={k: dict(median_of_ratios_shift=out['layers']['L1'][k]['median_of_ratios_shift'],
                           ratio_of_medians_shift=out['layers']['L1'][k]['ratio_of_medians_shift'],
                           signs_agree=bool(
                               (out['layers']['L1'][k]['median_of_ratios_shift'] - 1.0)
                               * (out['layers']['L1'][k]['ratio_of_medians_shift'] - 1.0) > 0))
                  for k in GRID[1:]},
            argmax_by_median_of_ratios=max(
                GRID[1:], key=lambda k: out['layers']['L1'][k]['median_of_ratios_shift']),
            argmax_by_ratio_of_medians=max(
                GRID[1:], key=lambda k: out['layers']['L1'][k]['ratio_of_medians_shift']),
            n_g_with_sign_disagreement=int(sum(
                1 for k in GRID[1:]
                if not (out['layers']['L1'][k]['median_of_ratios_shift'] - 1.0)
                       * (out['layers']['L1'][k]['ratio_of_medians_shift'] - 1.0) > 0))))
    out['summary'] = dict(
        common_mode_at_g2=dict(
            L1_base_ratio=out['layers']['L1']['2']['base_ratio_to_g0'],
            L1_peak_ratio=out['layers']['L1']['2']['peak_ratio_to_g0'],
            L1_ratio_of_medians_shift=out['layers']['L1']['2']['ratio_of_medians_shift'],
            L1_median_of_ratios_shift=out['layers']['L1']['2']['median_of_ratios_shift']),
        reading='both windows are lifted by a few percent at g=2, so the criterion -- a '
                'ratio of the two -- moves far less than either level')
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding='utf-8')
    print('lower_release median=%.10g  turnover ~%.6g d' % (l_med, tau))
    for k in GRID[1:]:
        e = out['expectation'][k]
        print('  g=%-5s |df|=%.6f  escaped[%d..%d]=%.4f..%.4f  predicted %+.5f..%+.5f  '
              'measured median-of-ratios %+.5f  ratio-of-medians %+.5f'
              % (k, e['median_abs_delta_f'], e['k_min'], e['k_max'],
                 e['escaped_fraction']['%d' % e['k_min']],
                 e['escaped_fraction']['%d' % e['k_max']],
                 e['predicted_amplitude_gain_min'], e['predicted_amplitude_gain_max'],
                 e['measured_median_of_ratios_gain'], e['measured_ratio_of_medians_gain']))
    print('WROTE reports/levels_shift.json')


if __name__ == '__main__':
    raise SystemExit(main())
