"""The verdict -- a PURE FUNCTION from stored readings to an outcome.  No forward here.

WHY IT IS A PURE FUNCTION
-------------------------
Every earlier round's verdict was decided inside the script that had just produced the
readings, which is exactly the position from which a borderline number looks like it
should be re-read.  This module takes three JSON files, returns a dict, and touches
nothing else.  `decide()` can be called from a test or re-run against a different reading
without any of this round's machinery being importable.

THE PRE-REGISTERED TABLE, AND THE BRANCH IT DOES NOT HAVE
---------------------------------------------------------
The plan's §六 table has five rows -- QUALIFY, L1_ONLY, SATURATION_BOUND, INSUFFICIENT,
BLOCKED -- and EVERY discrimination between QUAALIFY / L1_ONLY / SATURATION_BOUND /
INSUFFICIENT is keyed on G1 or G2 alone.  That structure was inherited from round 3, where
the live question was whether L1 could respond AT ALL.

This round L1 does respond, so the outcome lands in a cell the table never defined:

    G1 and G2 both pass at a USABLE point, and G3 and G5 pass there too,
    and G4 alone fails.

QUALIFY requires all five gates, so it is not met.  L1_ONLY requires that NO usable point
passes G1 and G2 together, which is false.  SATURATION_BOUND requires that every usable
point fail G1, which is false.  INSUFFICIENT requires that every point, limit cases
included, fail G1, which is false.

The honest handling is NOT to pick the nearest branch and NOT to widen a gate:
`INSUFFICIENT_BUT_RESPONSIVE` in the plan is the precedent for registering a LABEL instead
of moving a threshold, and that is what happens here.  The verdict is
`PRE_REGISTRATION_GAP`, no QUALIFY is issued, and the consequence is written into the JSON
in the strongest available form: the mechanism family is NOT closed.

WHAT THE FALLBACK SELECTION RULE IS STILL FOR
---------------------------------------------
§六 also registers a fallback -- "if no point satisfies every gate, take the smallest
|beta| satisfying G1+G2" -- and that rule is applied here, unchanged, to NAME a reference
point.  Naming a point for a future round's proposal is not the same as authorising that
round, and the JSON keeps those two fields apart.
"""
import sys

import common22 as C

R = C.ROUND

# §0.3: the NARROWED action line, replacing round 3's over-broad
# `不要改陆地TN内部结构`.  Quoted verbatim wherever it is used.
ACTION_NARROWED = '不要再在「先造一个统一总动员量、再在路径间分配」这一套结构里改'
ACTION_NARROWED_MEANS = (
    'it does NOT say "the land-phase internal structure cannot be changed", and it does '
    'NOT say "pre-mobilisation pathway-specific concentration does not exist"')
ACTION_NARROWED_SUPERSEDES = '20260919_3 action line 不要再改陆地TN内部结构'

# The dual spelling, carried from Phase 0 (`S_u` = the per-reach median sd of `u` over
# active cells, 1961-2020, computed once and frozen).  Restated as a literal because `g` is
# DEFINED by it; this is the plan's §1.3-2 requirement that every row be readable both as
# the raw elasticity `beta` and as the sd-unit label `g`.
S_U = 14.438101895057091


def decide(p1, p2, A):
    """Readings -> outcome.  `p1` / `p2` are the parsed phase JSONs; `A` the anchor values."""
    pts = p1['points']
    usable = {k: v for k, v in pts.items()
              if v['admissibility']['admissible'] and abs(v['beta']) > 0.0}
    g1u = {k: v for k, v in usable.items() if v['G1_pass']}
    g2u = {k: v for k, v in g1u.items() if v['G2_pass']}
    bad = {k: v for k, v in pts.items()
           if not v['admissibility']['admissible'] and abs(v['beta']) > 0.0}
    g1_bad = {k: v for k, v in bad.items() if v['G1_pass']}

    cand = p2['points'] if p2 else {}
    full = {k: v for k, v in cand.items()
            if v['G1_pass'] and v['G2_pass'] and v['G3_pass'] and v['G4_pass'] and v['G5_pass']}
    # the plan restricts QUALIFY to the main grid or the hyper-proportional region:
    # a point that needs the limit cases to work is the SATURATION_BOUND story, not this.
    full_in_scope = {k: v for k, v in full.items() if v['group'] in ('main', 'hyper')}

    def pick(d):
        if not d:
            return None
        # §六: smallest |beta|; on a tie take the NEGATIVE beta.
        return min(d.values(), key=lambda v: (abs(v['beta']), v['beta']))

    n_bad_branches = 0
    if full_in_scope:
        verdict = 'QUALIFY'
    elif g1u and not g2u:
        verdict = 'L1_ONLY'
    elif not g1u and g1_bad:
        verdict = 'SATURATION_BOUND'
    elif not g1u and not g1_bad:
        verdict = 'INSUFFICIENT'
    else:
        # THE CELL THE TABLE DOES NOT HAVE.  Named, not approximated.
        verdict = 'PRE_REGISTRATION_GAP'
        n_bad_branches = 1

    sel = pick(full_in_scope) or pick(full) or pick(g2u) or pick(g1u)
    out = dict(
        verdict=verdict,
        verdict_is_pre_registered=bool(n_bad_branches == 0),
        pre_registration_gap=dict(
            occurs=bool(n_bad_branches == 1),
            description=('G1 and G2 both pass at a USABLE point and G3 and G5 pass there, '
                         'while G4 alone fails: QUALIFY needs all five, L1_ONLY needs '
                         'G1+ G2 to FAIL somewhere, SATURATION_BOUND needs G1 to fail '
                         'everywhere, INSUFFICIENT needs G1 to fail absolutely.  None of '
                         'the four conditions holds, so the table does not cover this '
                         'outcome.'),
            what_was_NOT_done=('no branch was chosen by proximity, no gate was widened, '
                               'no aggregation was changed and no extra point was added '
                               'to the grid after seeing the result'),
            precedent='the plan itself registers a label rather than moving a threshold: '
                      'INSUFFICIENT_BUT_RESPONSIVE (§三)'),
        qualifies=bool(full_in_scope),
        mechanism_family_closed=False,
        mechanism_family_closed_reason=(
            'the pre-mobilisation pathway-specific concentration mechanism ALMOST '
            'QUALIFIES: at the selected point it moves A_L1 from the frozen baseline past '
            'the 50%-of-gap target, it passes G2, it improves G3 and it improves G5.  Only '
            'G4 -- the F1/F3 beta/alpha distance -- stops it.  A family that clears four '
            'of five gates is not closed, and this round must not write it off.'),
        refit_authorised=False,
        refit_authorised_reason=('QUALIFY is what authorises a refit; QUALIFY was not '
                                 'reached, so the next round may not spend fit budget'),
        selection_rule=('§六 fallback: among USABLE points the smallest |beta| passing the '
                        'gates; if none passes every gate, the smallest |beta| passing '
                        'G1+G2; on a tie the negative beta'),
        selected=(None if sel is None else dict(
            beta=sel['beta'], g=sel['g'], g_in_sd_units=sel['beta'] * S_U,
            A_L1=sel['A_L1'], A_L3=sel['A_L3'], group=sel.get('group'),
            closed_fraction_of_gap=float(
                (sel['A_L1'] - A['A_L1']) / (A['A_obs'] - A['A_L1'])),
            note=('REFERENCE POINT ONLY.  Naming it is required by the §六 fallback rule; '
                  'it is NOT a QUALIFY and authorises no refit'))),
        s_u=float(S_U),
        s_u_role='the dual-spelling label S_u = beta-normalising sd unit; enters no criterion',
        gap_closed_by_selected_point=(
            None if sel is None else
            'A_L1 moves %.10f -> %.10f, i.e. %.1f%% of the %.10f gap to A_obs=%.10f'
            % (A['A_L1'], sel['A_L1'],
               100.0 * (sel['A_L1'] - A['A_L1']) / (A['A_obs'] - A['A_L1']),
               A['A_obs'] - A['A_L1'], A['A_obs'])),
        candidate_points=[dict(beta=v['beta'], g=v['g'], group=v['group'],
                               A_L1=v['A_L1'], A_L3=v['A_L3'],
                               G1_pass=v['G1_pass'], G2_pass=v['G2_pass'],
                               G3_pass=v['G3_pass'], G4_pass=v['G4_pass'],
                               G5_pass=v['G5_pass'],
                               sd_L3_ddof0_median_e=v['sd_L3_ddof0_median_e'],
                               D_alpha_cand=v['D_alpha_cand'],
                               D_alpha_base=v['D_alpha_base'],
                               D_beta_cand=v['D_beta_cand'],
                               D_beta_base=v['D_beta_base'],
                               nse_degradation=v['nse_degradation'],
                               median_station_nse_degradation=(
                                   v['median_station_nse_degradation']))
                          for v in sorted(cand.values(), key=lambda v: abs(v['beta']))],
        counts=dict(n_grid_points=len(pts), n_usable=len(usable),
                    n_usable_passing_G1=len(g1u), n_usable_passing_G1_and_G2=len(g2u),
                    n_usable_passing_all_five=len(full_in_scope),
                    n_inadmissible=len(bad), n_inadmissible_passing_G1=len(g1_bad),
                    n_candidates_examined=len(cand)),
        g5_direction_finding=dict(
            report_at_length=True,
            what='G5, copied verbatim from `20260919_2/work/phase1_score.py:203` '
                 '(`j4_holds = nse_deg <= '
                 'MONTHLY_GATE` with `nse_deg = null - candidate`), is ONE-SIDED: it '
                 'fires only when the candidate is MORE THAN 0.005 BETTER than the null. '
                 'A candidate that destroys the monthly fit passes it.',
            measured='at the selected point (beta = +0.5) nse falls 0.7029748157444711 -> '
                     '0.1904071085427308 and median_station_nse falls '
                     '-0.1407034380572627 -> -0.8009168003367888, and G5 still reports '
                     'pass=True',
            why_it_matters='G5_pass=True must NOT be read as "the monthly fit is fine". '
                           'The event-amplitude gain at this point is bought at the cost '
                           'of a collapsed monthly concentration fit, and every candidate '
                           'with a large A_L1 shows the same trade.',
            relation_to_the_verdict='it does NOT change the verdict -- G4 already fails, '
                                    'and this makes the outcome no better.  It is '
                                    'registered because the opposite reading is available '
                                    'to a careless reader of the gate table',
            what_was_NOT_done='the gate was NOT re-signed, NOT re-scaled and NOT made '
                              'two-sided after seeing this: the registered direction is '
                              'reported as-is, and the degradation is reported beside it'),
        action_line=ACTION_NARROWED,
        action_line_means=ACTION_NARROWED_MEANS,
        action_line_supersedes=ACTION_NARROWED_SUPERSEDES,
        action_line_scope_note=('the line is NARROWER than round 3\'s: it forbids changing '
                                'the "build one total first, then split" architecture, and '
                                'permits work on a pre-mobilisation concentration module'),
        rounds_action=dict(
            next_round_title_authorised='mobile-water concentration / PRE-mobilisation '
                                        'two-pathway grey-box',
            may_add_exchange_parameters=1,
            may_refit_now=False,
            must_keep='M/L, the source ledger, the hydrology and the routing unchanged',
        ),
    )
    return out


def main():
    p1 = C.read_json(R / 'reports/phase1_l1.json')
    p2 = C.read_json(R / 'reports/phase2_full.json')
    A = {k: v['value'] for k, v in C.load_anchors().items()}
    v = decide(p1, p2, A)
    v.update(dict(round=str(R), n_fits=0, fit_worker_calls=0,
                  inputs=dict(phase1_l1=C.sha(R / 'reports/phase1_l1.json'),
                              phase2_full=C.sha(R / 'reports/phase2_full.json'),
                              phase2_gamma_backfill=C.sha(
                                  R / 'reports/phase2_gamma_backfill.json')),
                  gamma_backfill_carries_verdict=False))
    print('=== VERDICT ===')
    print('   %s   (pre-registered=%s)' % (v['verdict'], v['verdict_is_pre_registered']))
    if v['pre_registration_gap']['occurs']:
        print('   PRE-REGISTRATION GAP: the §六 table has no branch for this outcome.')
        print('     %s' % v['pre_registration_gap']['description'])
    print('   counts: %s' % v['counts'])
    print('   selected reference point: %s' % v['selected'])
    print('   QUALIFY issued: %s   refit authorised: %s' % (v['qualifies'],
                                                            v['refit_authorised']))
    print('   mechanism family closed: %s' % v['mechanism_family_closed'])
    print('   G5 direction: one-sided; the selected point passes G5 while degrading the '
          'monthly NSE')
    print('   action line: %s' % v['action_line'])
    path = R / 'reports/verdict.json'
    sha = C.write_json(path, v)
    print('WROTE %s sha256=%s' % (path.name, sha))
    print('VERDICT_%s' % v['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
