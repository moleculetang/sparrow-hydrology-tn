"""Phase 1 -- the L1 amplitude scan over the frozen 19-point beta grid.

The round's single question, as the plan states it: can making the FAST water carry
its OWN N concentration -- a concentration difference that exists BEFORE the total
mobilised mass is formed -- produce the missing event amplitude?  The frozen baseline
is `A_L1 = 1.0400772083480145`, the 50%-of-gap target is `1.1579754800655908`, and the
observed amplitude is `1.2758737517831669`.

WHAT IS REPORTED AND WHY EACH THING IS HERE
-------------------------------------------
* A_L1/A_L2/A_L3 for ALL NINETEEN points, including the ones that fail.  A grid
  reported only at its best point cannot be checked, and the plan makes reporting
  every point a hard gate (delivery check 12).
* `g = beta * S_u` beside every raw `beta` (pure relabelling, zero extra compute), so
  neither reading can be quoted in the wrong unit.
* the saturating-selector census at every point.  `Xi` is unbounded and exponentially
  ASYMMETRIC, so a large-beta row read without this census looks like a modest tilt.
* the PRE-REGISTERED mechanical admissibility rule, applied unchanged.  It is not a
  judgement made after seeing an amplitude.
* per-event improve/worsen counts against beta = 0.  Round 3's strongest intervention
  made MORE THAN HALF the events worse -- an upper-tail effect a median can hide.
* A_L1 split by `sign(Q_f - Q_s)`.  Phase 0 measures that the sign is mixed in space and
  time, so no single global monotone direction may be pre-registered.
* `A_obs` beside every A_L1, because the registered gate `A_L1 >= G1_target` is
  ONE-SIDED and therefore a point that OVERSHOOTS the observation also passes it.
  Overshooting is not the same finding as explaining the amplitude.  The gate is NOT
  tightened here -- that would be re-tuning after seeing results -- the overshoot is
  counted and reported instead.
"""
import sys

import numpy as np
import pandas as pd
import torch

import censuses22 as CENS
import common22 as C
import eventlib as E
import layers22 as LY
import xi as XI

R = C.ROUND
TAG = C.TAG


def censuses(model, G, layers0, info0, h):
    """Delegate to `censuses22`, so Phase 0 and Phase 1 read ONE implementation."""
    return CENS.all_censuses(model, G, layers0, info0, h)

def group_best(rows, betas):
    sub = {b: rows['%g' % b] for b in betas}
    ok = [v for v in sub.values() if v['admissibility']['admissible']]
    best = max(sub.values(), key=lambda v: v['A_L1'])
    bestok = max(ok, key=lambda v: v['A_L1']) if ok else None
    return dict(n_points=len(sub), n_admissible=len(ok),
                best_A_L1=best['A_L1'], best_beta=best['beta'],
                best_admissible_A_L1=(None if bestok is None else bestok['A_L1']),
                best_admissible_beta=(None if bestok is None else bestok['beta']),
                any_G1=bool(any(v['G1_pass'] for v in sub.values())),
                any_G1_admissible=bool(any(v['G1_pass'] for v in ok)))


def main():
    rep = {'phase': 1, 'n_fits': 0, 'fit_worker_calls': 0, 'round': str(R),
           'grid': dict(main=list(C.BETA_MAIN), hyper=list(C.BETA_HYPER),
                        extreme=list(C.BETA_EXTREME), n_points=len(C.BETA_GRID)),
           'tolerances': dict(sat_bound=C.SAT_BOUND, degen_floor=C.DEGEN_FLOOR,
                              w_floor=C.W_FLOOR)}

    print('=== frozen anchors ===', flush=True)
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    G1, G2, AOBS = A['G1_target_50pct'], A['G2_target_50pct'], A['A_obs']
    print('   A_L1=%.16g  A_L2=%.16g  A_L3=%.16g  A_obs=%.16g'
          % (A['A_L1'], A['A_L2'], A['A_L3'], AOBS))
    print('   G1=%.16g  G2=%.16g' % (G1, G2))

    print('=== model + geometry ===', flush=True)
    model = C.build(TAG)
    G = C.geometry(model)
    rep['geometry_diag'] = G['diag']
    assert G['diag']['n_nonpositive'] == 0, 'W must be strictly positive'
    assert G['diag']['n_reaches_without_active_ref_day'] == 0
    assert C.hazard_is_frozen(), 'Predictor.hazard must NOT be rebound this round'
    print('   W_min=%.6g  frac_active=%.6f  S_u=%.16g  ref_days=%d  hazard_frozen=%s'
          % (G['diag']['w_min'], G['diag']['frac_active'], G['S_u'],
             G['diag']['n_ref_days'], C.hazard_is_frozen()))

    ev = E.eligible_events()
    rep['events'] = dict(n=int(len(ev)), n_stations=int(ev.station_key.nunique()))
    assert rep['events']['n'] == A['n_events'] and rep['events']['n_stations'] == 15
    print('   events=%d stations=%d' % (rep['events']['n'], rep['events']['n_stations']))

    print('=== frozen baseline (kernel NOT installed) ===', flush=True)
    assert not C.is_installed()['all_bound'], 'kernel must start UNINSTALLED'
    ly0, info0 = LY.forward_layers(model, TAG)
    b0 = LY.layer_budget(ly0, ev)
    for L in ('L1', 'L2', 'L3'):
        assert b0[L]['amp_ratio_median'] == A['A_' + L], (L, b0[L]['amp_ratio_median'])
    ev0 = E.build_event_table(
        ly0[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
    beta0 = E.f1_coef(ev0.assign(dc=ev0.c_peak - ev0.c_base), 'T_interevent')
    alpha0 = E.f3_intercept(ev0, E.F3_PRIMARY_GAP)
    rep['frozen_baseline'] = dict(
        layer_budget=b0, n_basis=int(len(ly0)), beta=beta0, alpha=alpha0,
        D_beta_P=float(abs(beta0 - A['beta_obs'])),
        D_alpha_P=float(abs(alpha0 - A['alpha_obs'])),
        reproduces_registered_A_L1=bool(b0['L1']['amp_ratio_median'] == A['A_L1']))
    print('   A_L1=%.16g A_L2=%.16g A_L3=%.16g  (all == registered)'
          % (b0['L1']['amp_ratio_median'], b0['L2']['amp_ratio_median'],
             b0['L3']['amp_ratio_median']))
    print('   beta=%.16g D_beta_P=%.6g   alpha=%.16g D_alpha_P=%.6g'
          % (beta0, abs(beta0 - A['beta_obs']), alpha0, abs(alpha0 - A['alpha_obs'])))

    with torch.no_grad():
        h = np.ascontiguousarray(model.flux_parameters(
            torch.tensor(C.parameters(TAG)))[0].numpy())
    cens, sg = censuses(model, G, ly0, info0, h)
    rep['censuses'] = cens
    assert cens['eligible_set']['mask_sha_matches'], 'ELIGIBLE MASK DRIFTED'

    sq = cens['sign_qf_minus_qs']
    print('=== census 1: sign(Qf-Qs) ===')
    print('   eligible station-days n=%d  frac(Qf>Qs)=%.6f  frac(Qf<Qs)=%.6f  zero=%d'
          % (sq['n'], sq['frac_qf_gt_qs'], sq['frac_qf_lt_qs'], sq['n_zero']))
    print('   reaches uniformly Qf<Qs: %d / %d   (uniformly Qf>Qs: %d)'
          % (sq['n_reaches_uniformly_qf_lt_qs'], sq['n_reaches'],
             sq['n_reaches_uniformly_qf_gt_qs']))
    ac = cens['active']
    print('=== census 2: active vs h==0 ===')
    print('   frac_active=%.6f  frac_h_zero=%.6f  masks_agree=%s  '
          'h==0|inactive=%.6f  h==0|active=%.6f'
          % (ac['frac_active'], ac['frac_h_zero'], ac['masks_agree'],
             ac['inactive_and_h_zero_frac_of_inactive'],
             ac['active_and_h_zero_frac_of_active']))
    # Same finding Phase 0 recorded as `D_pin_mask`, asserted at the SAME strength here
    # rather than a stronger one: `h == 0 <=> W < 1e-6` is exact, `active` is `W > 1e-3`,
    # and they differ on the 5-cell band between them.  Phase 0 owns the footprint
    # number; this phase must not re-assert a claim Phase 0 already refuted.
    assert ac['h_zero_iff_W_lt_1e_6'], \
        'h == 0 is not the W < 1e-6 mask: the plan\'s identity does not hold'
    assert ac['n_active_and_h_zero'] == 0, \
        'an active cell has h == 0: Xi != 1 would multiply a zero risk'
    assert ac['pin_is_within_the_band'] and ac['n_inactive_and_h_nonzero'] <= 5, \
        ('the pin footprint grew beyond the Phase-0 measurement: %d'
         % ac['n_inactive_and_h_nonzero'])
    print('   h==0 <=> W<1e-6 EXACT=%s ; identical to `~active`=%s (pin footprint %d '
          'cells, Phase 0 recorded 5)'
          % (ac['h_zero_iff_W_lt_1e_6'], ac['masks_agree'],
             ac['n_inactive_and_h_nonzero']))
    uf = ac['unfloored_vs_floored']
    ks = ('0', '1', '50', '99', '100')
    print('   floored   u on the usably-nonzero cells =%s'
          % [round(uf['u_floored_quantiles_same_cells'][k], 4) for k in ks])
    print('   UNfloored u on the usably-nonzero cells =%s'
          % [round(uf['u_unfloored_quantiles'][k], 4) for k in ks])
    print('   max floor shift on those cells = %.6g   usable@1e-3=%.6f usable@1e-6=%.6f'
          % (uf['max_floor_shift_on_usable_cells'], uf['frac_usable_at_floor_1e3'],
             uf['frac_usable_at_1e6']))
    ao = cens['axis_overlap']
    print('=== census 3: axis overlap with round 3 ===')
    print('   corr(u,z3) active=%.6f  eligible-days=%.6f  corr(logWeff,logQfQs)=%.6f'
          % (ao['corr_u_z3_active_cells'], ao['corr_u_z3_eligible_station_days'],
             ao['corr_logWeff_logQfQr_active_cells']))
    print('=== census 4: pilot reaches %s (S_u=%.16g) ==='
          % (cens['pilot_reaches']['pilot_indices'], cens['pilot_reaches']['S_u']))
    for k, v in cens['pilot_reaches']['per_reach'].items():
        print('   col %s -> reach %d: Qs/W=%.6f Qf/W=%.6f u_med=%+.6f sd_u=%.4f '
              'frac_active=%.4f' % (k, v['reach_id'], v['qs_over_w_median'],
                                    v['qf_over_w_median'], v['u_median'],
                                    v['per_reach_sd_u'], v['frac_active']))
    ew = cens['event_window_signs']
    print('   event-window sign(Qf-Qs): peak +%d/-%d/0=%d  base +%d/-%d/0=%d'
          % (ew['peak_positive'], ew['peak_negative'], ew['peak_zero'],
             ew['base_positive'], ew['base_negative'], ew['base_zero']))

    print('=== the 19-point beta scan ===', flush=True)
    base0 = ev0.set_index(['station_key', 'event_id']).amp_ratio
    rows = {}
    for beta in C.BETA_GRID:
        X, XT, _ = C.build_xi(model, beta)
        assert bool(np.isfinite(X).all()) and bool((X > 0).all()), beta
        cen = XI.saturation_census(X, h, G['active'])
        adm = XI.admissible(cen, C.SAT_BOUND, C.DEGEN_FLOOR)
        info = C.install_kernel(model, X, XT)
        assert info['all_bound'] and info['hazard_is_frozen'], (beta, info)
        try:
            ly, _ = LY.forward_layers(model, TAG)
            b = LY.layer_budget(ly, ev)
            evb = E.build_event_table(
                ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
            bh = E.f1_coef(evb.assign(dc=evb.c_peak - evb.c_base), 'T_interevent')
            ah = E.f3_intercept(evb, E.F3_PRIMARY_GAP)
            cur = evb.set_index(['station_key', 'event_id']).amp_ratio
            join = pd.concat([base0.rename('a0'), cur.rename('a1')], axis=1)
            join = join[np.isfinite(join.a0) & np.isfinite(join.a1)]
            better = int((join.a1 > join.a0).sum())
            worse = int((join.a1 < join.a0).sum())
            sig = CENS.grouped_A(evb, sg, 'peak')
        finally:
            C.restore_kernel()
        rows['%g' % beta] = dict(
            beta=float(beta), g=float(beta * G['S_u']),
            group=('main' if beta in C.BETA_MAIN else
                   'hyper' if beta in C.BETA_HYPER else 'extreme'),
            A_L1=b['L1']['amp_ratio_median'], A_L1_mean=b['L1']['amp_ratio_mean'],
            A_L1_sd=b['L1']['amp_ratio_sd'], n_events_L1=b['L1']['n_events'],
            A_L2=b['L2']['amp_ratio_median'], A_L3=b['L3']['amp_ratio_median'],
            c_base_L1=b['L1']['c_base_median'], c_peak_L1=b['L1']['c_peak_median'],
            c_base_L2=b['L2']['c_base_median'], c_peak_L2=b['L2']['c_peak_median'],
            c_base_L3=b['L3']['c_base_median'], c_peak_L3=b['L3']['c_peak_median'],
            G1_pass=bool(b['L1']['amp_ratio_median'] >= G1),
            G2_pass=bool(b['L3']['amp_ratio_median'] >= G2),
            overshoots_observation=bool(b['L1']['amp_ratio_median'] > AOBS),
            delta_A_L1_vs_obs=float(b['L1']['amp_ratio_median'] - AOBS),
            n_events_better=better, n_events_worse=worse,
            n_events_compared=int(len(join)),
            n_events_tied=int(len(join) - better - worse),
            frac_events_worse=float(worse / max(1, len(join))),
            beta_hat=bh, alpha_hat=ah,
            D_beta_cand=(None if bh is None else float(abs(bh - A['beta_obs']))),
            D_alpha_cand=(None if ah is None else float(abs(ah - A['alpha_obs']))),
            j2_holds=(None if bh is None else
                      bool(abs(bh - A['beta_obs']) <= abs(beta0 - A['beta_obs']))),
            j3_holds=(None if ah is None else
                      bool(abs(ah - A['alpha_obs']) <= abs(alpha0 - A['alpha_obs']))),
            saturation=cen, admissibility=adm, **sig)
        r = rows['%g' % beta]
        print('   b=%7.3f g=%9.3f %-7s A_L1=%.10f A_L2=%.10f A_L3=%.10f '
              'G1=%-5s dA_obs=%+.6f admiss=%-5s f99=%.6f better/worse=%d/%d'
              % (beta, r['g'], r['group'], r['A_L1'], r['A_L2'], r['A_L3'],
                 r['G1_pass'], r['delta_A_L1_vs_obs'], r['admissibility']['admissible'],
                 r['saturation']['frac_prob_ge_0p99'], better, worse), flush=True)

    rep['points'] = rows
    assert len(rows) == len(C.BETA_GRID), 'ALL 19 POINTS MUST BE REPORTED'
    usable = {k: v for k, v in rows.items() if v['admissibility']['admissible']}
    g1u = {k: v for k, v in usable.items() if v['G1_pass']}
    g1a = {k: v for k, v in rows.items() if v['G1_pass']}
    g1g2u = {k: v for k, v in g1u.items() if v['G2_pass']}
    best = max(rows.values(), key=lambda v: v['A_L1'])
    bu = max(usable.values(), key=lambda v: v['A_L1']) if usable else None
    rep['phase1_decision'] = dict(
        n_usable=len(usable), n_usable_passing_G1=len(g1u),
        n_passing_G1_including_unusable=len(g1a),
        n_usable_passing_G1_and_G2=len(g1g2u),
        best_A_L1_overall=dict(beta=best['beta'], group=best['group'],
                               admissible=best['admissibility']['admissible'],
                               A_L1=best['A_L1']),
        best_A_L1_among_usable=(None if bu is None else
                                dict(beta=bu['beta'], A_L1=bu['A_L1'],
                                     group=bu['group'])),
        smallest_abs_beta_usable_and_G1=(None if not g1u else
                                         min(g1u.values(),
                                             key=lambda v: abs(v['beta']))['beta']),
        smallest_abs_beta_usable_G1_and_G2=(None if not g1g2u else
                                            min(g1g2u.values(),
                                                key=lambda v: abs(v['beta']))['beta']),
        n_points_overshooting_observation=int(
            sum(1 for v in rows.values() if v['overshoots_observation'])),
        n_points_G1_pass_but_overshoot=int(
            sum(1 for v in rows.values() if v['G1_pass'] and v['overshoots_observation'])),
        G1_target=float(G1), G2_target=float(G2), A_obs=float(AOBS),
        per_group=dict(main=group_best(rows, C.BETA_MAIN),
                       hyper=group_best(rows, C.BETA_HYPER),
                       extreme=group_best(rows, C.BETA_EXTREME)),
        early_stop_triggered=bool(len(g1u) == 0),
        G1_passes_sorted=[dict(beta=v['beta'], group=v['group'],
                               admissible=v['admissibility']['admissible'],
                               A_L1=v['A_L1'], overshoots=v['overshoots_observation'])
                          for v in sorted(g1a.values(), key=lambda v: abs(v['beta']))])
    pd_ = rep['phase1_decision']
    print('=== PHASE 1 DECISION ===')
    print('   usable=%d  usable&G1=%d  usable&G1&G2=%d  (any group passing G1=%d)'
          % (pd_['n_usable'], pd_['n_usable_passing_G1'],
             pd_['n_usable_passing_G1_and_G2'], pd_['n_passing_G1_including_unusable']))
    print('   best A_L1 overall: beta=%g group=%s admissible=%s A_L1=%.10f'
          % (best['beta'], best['group'], best['admissibility']['admissible'],
             best['A_L1']))
    print('   best USABLE A_L1: %s' % pd_['best_A_L1_among_usable'])
    print('   smallest |beta| usable&G1: %s ; usable&G1&G2: %s'
          % (pd_['smallest_abs_beta_usable_and_G1'],
             pd_['smallest_abs_beta_usable_G1_and_G2']))
    print('   points overshooting A_obs=%.10f: %d (of which G1-pass: %d)'
          % (AOBS, pd_['n_points_overshooting_observation'],
             pd_['n_points_G1_pass_but_overshoot']))
    for g, v in pd_['per_group'].items():
        print('   %-7s: n=%d admissible=%d best A_L1=%.10f at beta=%g '
              '(admissible best: %s @ %s)  any_G1=%s'
              % (g, v['n_points'], v['n_admissible'], v['best_A_L1'], v['best_beta'],
                 v['best_admissible_A_L1'], v['best_admissible_beta'], v['any_G1']))
    print('   EARLY STOP: %s' % pd_['early_stop_triggered'])

    path = R / 'reports/phase1_l1.json'
    sha = C.write_json(path, rep)
    print('WROTE %s sha256=%s' % (path.name, sha))
    print('PHASE1_PASSED' if not pd_['early_stop_triggered'] else 'PHASE1_EARLY_STOP')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
