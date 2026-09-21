"""Phase 1's verdict, as a PURE FUNCTION of readings and frozen thresholds.

WHY THIS IS A SEPARATE MODULE
-----------------------------
`phase1_l1.py` both runs the scan and decides.  That is one module too few: the decision
then cannot be checked without re-running the forward passes that produced it, and a
reader has only the author's word that the decision follows from the readings.  Here the
decision is a function of two JSON files, so it can be re-derived by hand, by a different
implementation, or by a reader with a calculator.

This module therefore:
  * verifies the pre-registration hash before reading anything (the criteria may not be
    edited after the readings exist);
  * reads the readings ONLY from `reports/phase1_l1.json`;
  * does NOT import `common21`, `eventlib`, `layers21` or `hazard_g`, touches no model,
    and computes no amplitude;
  * recomputes the two thresholds from the published baselines rather than trusting a
    literal, and reports the 1-ULP gap to the md's rendering.

The two questions §七 asks are kept apart, because here they give different answers and
one of them is the comfortable one:
  * the FORMAL condition for `INSUFFICIENT` -- "no gamma reaches G1, limit case included";
  * the WORDING condition for the `INSUFFICIENT_BUT_RESPONSIVE` refinement -- "A_L1 明显随
    g 上升", i.e. a monotone rise.
`A_L1` is not monotone in `g` (it peaks at `g=2` and falls), so the wording condition
fails while my first version's weakest-possible test passed.  The md's wording binds.
"""
import hashlib
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parent.parent
REPORTS = ROUND / 'reports'
PRE = REPORTS / 'pre_registration.json'
MD = REPORTS / '预注册_判据与门槛.md'
READINGS = REPORTS / 'phase1_l1.json'
OUT = REPORTS / 'phase1_verdict.json'
TOL_ULP = 1e-15


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decide(readings, pre, md_text):
    """(readings, frozen thresholds) -> outcome.  No I/O, no model, no amplitude."""
    fa = readings['anchors']
    base_L1, base_L3, obs = fa['A_L1'], fa['A_L3'], fa['A_obs']
    lit = {'L1': '1.1579754800655907', 'L3': '1.14954556682221'}
    for s in lit.values():
        if s not in md_text:
            raise SystemExit('MD_LITERAL_NOT_FOUND %s' % s)
    thr = {k: base + 0.5 * (obs - base) for k, base in (('L1', base_L1), ('L3', base_L3))}
    for k in thr:
        if abs(thr[k] - fa['G%d_target_50pct' % (1 if k == 'L1' else 2)]) > 0.0:
            raise SystemExit('FROZEN_THRESHOLD_DISAGREES_WITH_THE_RULE ' + k)

    grid = dict(readings['per_gamma'])
    order = sorted((float(k) for k in grid))
    a1 = [grid['%g' % g]['A_L1'] for g in order]
    d = [a1[i + 1] - a1[i] for i in range(len(a1) - 1)]
    g1_ok = sorted(g for g in order if grid['%g' % g]['A_L1'] >= thr['L1'])
    g2_ok = sorted(g for g in order if grid['%g' % g]['A_L3'] >= thr['L3'])
    both = sorted(set(g1_ok) & set(g2_ok))
    lim = readings['limit_case']
    lim_pass = bool(lim['A_L1'] >= thr['L1'])
    any_response = bool(max(a1) > a1[0])
    monotone = bool(all(x > 0 for x in d))

    if g1_ok or lim_pass:
        verdict, branch = 'PASS', ('L1_AND_L3' if both else
                                   'GRID_TRUNCATED__LIMIT_CASE_PASSES' if (lim_pass and not g1_ok)
                                   else 'L1_ONLY')
        qualifier, action = None, 'proceed to Phase 2 (G2-G5)'
    else:
        verdict, branch = 'STOP', 'INSUFFICIENT'
        qualifier = ('RESPONSIVE_NONMONOTONE' if (any_response and not monotone)
                     else 'RESPONSIVE_MONOTONE' if any_response else 'NO_RESPONSE')
        action = ('stop changing the LAND-PHASE TN internal structure; turn to an '
                  'additional event-N source the current model does not have at all '
                  '(bed/bank sediment resuspension, erosive organic N)')
    return dict(
        verdict=verdict, branch=branch, qualifier=qualifier, action=action,
        thresholds=dict(L1=thr['L1'], L3=thr['L3'],
                        md_literal_L1=float(lit['L1']), md_literal_L3=float(lit['L3']),
                        ulp_L1=float(float(lit['L1']) - thr['L1']),
                        ulp_L3=float(float(lit['L3']) - thr['L3']),
                        md_literal_changes_no_verdict=True),
        readings_used=dict(
            grid_A_L1={k: grid[k]['A_L1'] for k in readings['per_gamma']},
            grid_A_L3={k: grid[k]['A_L3'] for k in readings['per_gamma']},
            grid_order=order, increments=d, max_A_L1=max(a1), argmax_g=order[a1.index(max(a1))],
            A_L1_at_top_of_grid=a1[-1],
            limit_case_g=lim['g'], limit_case_A_L1=lim['A_L1'],
            limit_case_A_L3=lim['A_L3']),
        conditions=dict(
            formal_INSUFFICIENT='no gamma reaches G1, limit case included',
            formal_satisfied=bool(not (g1_ok or lim_pass)),
            wording_RESPONSIVE='A_L1 clearly rises with g (monotone)',
            wording_satisfied=monotone,
            weak_test_any_positive_span=any_response,
            weak_test_would_say=('INSUFFICIENT_BUT_RESPONSIVE' if any_response
                                 else 'INSUFFICIENT'),
            weak_test_was_an_undefined_term=('§七 names 明显 and defines no number; the '
                                             'weakest-possible reading of it passes while '
                                             'the wording fails, so the wording binds '
                                             '(tightening can never flatter)')),
        g_passing_G1=g1_ok, g_passing_G2=g2_ok, g_passing_both=both,
        limit_case_passes_G1=lim_pass,
        selected_g=(both[0] if both else g1_ok[0] if g1_ok else None),
        selection_rule='smallest g passing every gate; else smallest passing G1+G2; else none',
        proceed_to_phase2=bool(verdict == 'PASS'),
        phase2_not_run_reason=(None if verdict == 'PASS' else
                               'the early-stop mandate: L1 did not pass, so the complex '
                               'spatial validation is not run'),
        prediction_check=dict(
            registered='§2.3: dA_L1/dg > 0 AND monotone',
            observed_sign='positive overall (A_L1 max exceeds its g=0 value)',
            observed_shape='NON-monotone: peaks at g=%g then falls' % order[a1.index(max(a1))],
            verdict_on_prediction='shape FALSIFIED, sign not'),
        n_fits=readings['n_fits'], fit_worker_calls=readings['fit_worker_calls'])


def main():
    pre = json.loads(PRE.read_text(encoding='utf-8'))
    md_text = MD.read_text(encoding='utf-8')
    if sha(MD) != pre['pre_registration']['sha256']:
        raise SystemExit('PRE_REGISTRATION_CHANGED')
    if sha(REPORTS / 'frozen_anchors.json') != pre['frozen_anchors']['sha256']:
        raise SystemExit('FROZEN_ANCHORS_CHANGED')
    readings = json.loads(READINGS.read_text(encoding='utf-8'))
    out = decide(readings, pre, md_text)
    out['sources'] = dict(pre_registration=pre['pre_registration']['sha256'],
                          readings=sha(READINGS), readings_file='reports/phase1_l1.json',
                          decided_without_recomputing_any_amplitude=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding='utf-8')
    print('VERDICT %s / %s / %s' % (out['verdict'], out['branch'], out['qualifier']))
    print('  G1 target=%.17g  G2 target=%.17g' % (out['thresholds']['L1'],
                                                  out['thresholds']['L3']))
    print('  g passing G1=%s  G2=%s  both=%s  limit=%s  selected=%s'
          % (out['g_passing_G1'], out['g_passing_G2'], out['g_passing_both'],
             out['limit_case_passes_G1'], out['selected_g']))
    print('  formal_satisfied=%s  wording_satisfied=%s  weak_test_would_say=%s'
          % (out['conditions']['formal_satisfied'], out['conditions']['wording_satisfied'],
             out['conditions']['weak_test_would_say']))
    print('  action: %s' % out['action'])
    print('WROTE reports/phase1_verdict.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
