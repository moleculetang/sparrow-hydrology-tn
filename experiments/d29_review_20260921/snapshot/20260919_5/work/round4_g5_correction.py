"""20260919_5 -- section 5: the round-4 G5 correction, from the STORED round-4 JSONs.

WHAT IS BEING CORRECTED, AND WHAT IS *NOT*
------------------------------------------
Round 4 gated the monthly fit with

    20260919_4/work/phase2_full.py:247
        g5 = bool(nse - bl_nse <= MONTHLY_GATE and mnse - bl_mnse <= MONTHLY_GATE)

that is CANDIDATE MINUS BASELINE (`:262-263` store the same reversed quantity under the
name `nse_degradation`).  A candidate that destroys the monthly fit therefore has
`nse - bl_nse` strongly NEGATIVE, and passes.

The upstream original it is cited as copying is the opposite:

    20260919_2/work/phase1_score.py:178
        nse_deg = float(mn['nse'] - mm['nse'])          # mn = NULL, mm = candidate
    20260919_2/work/phase1_score.py:203
        j4_holds = bool(nse_deg <= MONTHLY_GATE and mnse_deg <= MONTHLY_GATE)

that is NULL MINUS CANDIDATE, which is protective: a destroyed candidate has
`null - candidate` strongly POSITIVE and fails.

So the correction is a RESTORATION of the upstream direction, not a new judgement.  The
`g5_direction_finding.what` text stored in round 4 asserts this was copied "verbatim ...
with `nse_deg = null - candidate`".  On that one point -- its own implementation -- the
assertion is FALSE, and this script measures the discrepancy rather than repeating it.

WHAT THIS DOES NOT DO
---------------------
It does NOT re-run a single forward.  Every number is read from the round-4 JSONs, which
are READ-ONLY in this round (their sha256 values are quoted in round-4 prose).  `<=` is
written, never `not (x > gate)`, so a NaN fails instead of passing.

TWO LIMITS THAT MUST BE REPORTED AND NOT SOFTENED
-------------------------------------------------
1. `nse` / `median_station_nse` exist in round 4 for only FOUR beta values
   (`points` = -4, -8, +0.5, +1).  The other FIFTEEN beta values have no monthly reading
   anywhere on disk, so the corrected G5 table CANNOT be extended over the 19-point grid.
2. The correction does not move the round-4 verdict.  G4 already failed at every evaluated
   point; this makes the gate table honest about G5 as well, and nothing more.
"""
import json

import common23 as C

R = C.ROUND
OUT = R / 'reports'
R4 = C.R4

# The upstream source lines this correction restores, quoted so a reader can check them
# rather than take the citation on trust.
UPSTREAM_CITATION = dict(
    file='20260919_2/work/phase1_score.py',
    line=178,
    text="nse_deg = float(mn['nse'] - mm['nse'])",
    line_203="j4_holds = bool(nse_deg <= MONTHLY_GATE and mnse_deg <= MONTHLY_GATE)",
    direction='null - candidate  (PROTECTIVE)')

ROUND4_CITATION = dict(
    file='20260919_4/work/phase2_full.py',
    line=247,
    text='g5 = bool(nse - bl_nse <= MONTHLY_GATE and mnse - bl_mnse <= MONTHLY_GATE)',
    line_262='nse_degradation=float(nse - bl_nse)',
    direction='candidate - baseline  (DESTROYED CANDIDATES PASS)')


def _p(msg):
    print(msg, flush=True)


def main():
    _p('=== %s section 5: round-4 G5 correction (read-only, no forward) ===' % R.name)
    pre = C.round4_json_shas()
    _p('    round-4 JSON sha BEFORE: verdict=%s phase2_full=%s phase1_l1=%s'
       % (pre['verdict.json'][:12], pre['phase2_full.json'][:12], pre['phase1_l1.json'][:12]))

    p2 = json.loads((R4 / 'reports' / 'phase2_full.json').read_text(encoding='utf-8'))
    v4 = json.loads((R4 / 'reports' / 'verdict.json').read_text(encoding='utf-8'))

    gate = float(p2['monthly_gate'])                     # MONTHLY_GATE, read not typed
    bl = p2['baseline']['monthly']
    bl_nse = float(bl['nse'])
    bl_mnse = float(bl['median_station_nse'])
    _p('    MONTHLY_GATE=%.6g (read from phase2_full.json::monthly_gate)' % gate)
    _p('    baseline.monthly: nse=%r  median_station_nse=%r' % (bl_nse, bl_mnse))

    # ---------------------------------------------------------------- per beta
    rows, n_stored_pass, n_corr_pass = {}, 0, 0
    for lab, q in sorted(p2['points'].items(), key=lambda kv: float(kv[1]['beta'])):
        m = q['monthly']
        nse = float(m['nse'])
        mnse = float(m['median_station_nse'])
        # the upstream direction: null MINUS candidate
        deg_c = float(bl_nse - nse)
        mdeg_c = float(bl_mnse - mnse)
        corr_pass = bool(deg_c <= gate and mdeg_c <= gate)
        stored_pass = bool(q['G5_pass'])
        n_stored_pass += int(stored_pass)
        n_corr_pass += int(corr_pass)
        rows[lab] = dict(
            beta=float(q['beta']), group=q.get('group'),
            nse=nse, median_station_nse=mnse,
            nse_degradation_STORED=float(q['nse_degradation']),
            median_station_nse_degradation_STORED=float(q['median_station_nse_degradation']),
            G5_pass_STORED=stored_pass,
            nse_degradation_corrected=deg_c,
            median_station_nse_degradation_corrected=mdeg_c,
            G5_pass_corrected=corr_pass,
            stored_degradation_is_the_negated_corrected_value=bool(
                q['nse_degradation'] == -deg_c
                and q['median_station_nse_degradation'] == -mdeg_c),
            mean_concentration_relative_change=float(q['mean_concentration_relative_change']),
            j5_holds=bool(q['j5_holds']),
            per_gate_STORED={g: bool(q[g + '_pass']) for g in ('G1', 'G2', 'G3', 'G4', 'G5')},
            per_gate_corrected={**{g: bool(q[g + '_pass']) for g in ('G1', 'G2', 'G3', 'G4')},
                                'G5': corr_pass})

    for lab, r in rows.items():
        _p('    beta=%-4s  nse_deg stored=%+.6e corrected=%+.6e  G5 %s -> %s  (negated=%s)'
           % (lab, r['nse_degradation_STORED'], r['nse_degradation_corrected'],
              'PASS' if r['G5_pass_STORED'] else 'FAIL',
              'PASS' if r['G5_pass_corrected'] else 'FAIL',
              r['stored_degradation_is_the_negated_corrected_value']))

    # ------------------------------------------------- the round-4 own attribution
    g5f = v4.get('g5_direction_finding', {})
    attribution_false = bool('null - candidate' in str(g5f.get('what', ''))
                             and any(not r['G5_pass_corrected'] for r in rows.values()))

    # ------------------------------------------------- section 6 selection, unchanged
    decided = p2.get('phase2_decision', {})
    per_gate_stored = dict(decided.get('per_gate', {}))
    per_gate_corrected = dict(per_gate_stored)
    per_gate_corrected['G5_pass'] = n_corr_pass

    sel_beta = v4.get('selected')
    _p('    selected beta (round 4, unchanged): %r' % (sel_beta,))

    out = dict(
        round=R.name, reads=[str(R4 / 'reports' / 'phase2_full.json'),
                             str(R4 / 'reports' / 'verdict.json')],
        runs_no_forward=True, edits_round4_json=False,
        monthly_gate=gate,
        baseline_monthly=dict(nse=bl_nse, median_station_nse=bl_mnse),
        upstream_citation=UPSTREAM_CITATION, round4_citation=ROUND4_CITATION,
        points=rows,
        per_gate_stored=per_gate_stored, per_gate_corrected=per_gate_corrected,
        n_G5_pass_stored=n_stored_pass, n_G5_pass_corrected=n_corr_pass,
        g5_direction_finding_what=dict(
            stored_text=str(g5f.get('what', '')),
            stored_parenthetical_is_false_about_round4_own_code=attribution_false,
            why=('the stored text says it was copied verbatim with `nse_deg = null - '
                 'candidate`; round 4 implementation is `nse - bl_nse`, i.e. '
                 'candidate - baseline, and the stored degradation fields are the '
                 'NEGATED corrected values (measured, per point, above)')),
        # limit 1: the grid cannot be covered
        n_beta_grid=len(C.BETA_GRID),
        n_beta_with_a_monthly_reading_on_disk=len(rows),
        n_beta_WITHOUT_a_monthly_reading=len(C.BETA_GRID) - len(rows),
        cannot_be_plotted_note=(
            '`nse` and `median_station_nse` exist in round 4 for only the four evaluated '
            'beta values; the remaining fifteen beta values have NO monthly reading on '
            'disk, so the corrected G5 table covers four points and MUST NOT be drawn as a '
            'curve over the 19-point grid.  Round 5 computes the monthly reading at every '
            'beta for its own devices, which is a different panel and does not backfill '
            'this one.'),
        # limit 2: the verdict does not move
        verdict_unchanged=dict(
            qualifies=v4.get('qualifies'),
            verdict=v4.get('verdict'),
            mechanism_family_closed=v4.get('mechanism_family_closed'),
            selection_rule=v4.get('selection_rule'),
            selected=sel_beta,
            n_all_five_pass_stored=decided.get('n_all_five_pass'),
            n_all_five_pass_corrected=int(sum(
                all(r['per_gate_corrected'][g] for g in ('G1', 'G2', 'G3', 'G4', 'G5'))
                for r in rows.values())),
            unchanged_because=('G4 failed at every evaluated point before and after; the '
                               'correction only stops G5 from reporting the reverse of '
                               'what it measured.  qualifies, the section-6 selection and '
                               'mechanism_family_closed are all untouched.'),
        ),
        round4_json_sha_before=pre,
        fit_worker_calls=int(p2.get('fit_worker_calls', 0)),
        n_fits=int(p2.get('n_fits', 0)),
    )

    post = C.round4_json_shas()
    out['round4_json_sha_after'] = post
    out['round4_json_unchanged'] = bool(post == pre)
    if post != pre:
        raise SystemExit('ROUND4_JSON_CHANGED_WHILE_CORRECTING %r' % post)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'round4_G5_correction.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')

    _p('    G5 passes: stored=%d -> corrected=%d  (n_points=%d)'
       % (n_stored_pass, n_corr_pass, len(rows)))
    _p('    per_gate stored=%r' % per_gate_stored)
    _p('    per_gate corrected=%r' % per_gate_corrected)
    _p('    round-4 JSON unchanged: %s' % out['round4_json_unchanged'])
    _p('    monthlies cover %d of %d betas -> %d CANNOT be plotted'
       % (len(rows), len(C.BETA_GRID), out['n_beta_WITHOUT_a_monthly_reading']))
    _p('    wrote %s' % (OUT / 'round4_G5_correction.json'))
    return out


if __name__ == '__main__':
    main()
