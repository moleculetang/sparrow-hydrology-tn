"""Phase 1 -- the zero-fit capability envelope of the minimal `R` state.

WHAT RUNS
---------
The frozen grid (`reports/frozen_grid.json`, sha
`5e462683ac4cac275a536999cfa63fedcad6b37c91f127ec409ba797fd07307e`): 8 main
`tau_R` points, the `delta = 1` null, and 5 limit cases -- 14 forwards, no more
and no fewer.  The grid is READ, never reconstructed here, because a grid
rebuilt at run time is a grid that can differ from the one that was frozen.

Per point this module records, and nothing else:

  * the daily concentration replay (only `p` is ever read by a criterion);
  * the per-event `C_base / C_peak / A` table on the 214 frozen events;
  * the five `legal` conjuncts from the FROZEN ledger path
    (`common20.ledger_with` -> `model.ledger`, which is the registered
    `campaign_model.Matched.ledger`), plus the explicit `R >= -1e-7`
    assertion the plan's section 3.2 item 6 requires, because `R` is not in
    `fit_worker.py:70`'s minimum list;
  * where the cap binds, and whether it binds inside the event windows.
    This is the falsifier the plan's section 10 registers for prediction P1.

No criterion is evaluated here.  `phase1_score.py` does that, reading these
artifacts.  Splitting them is what lets `work/audit_envelope.py` recompute the
criteria from the raw replays without importing this module.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

import common20 as C
import closures_r as KR
import eventlib as E
import round_tools as RT

R = C.ROUND
TAG = 'C0_s1'
GRID_JSON = R / 'reports/frozen_grid.json'

# The five conjuncts.  Conjunct 1 and 2 come from the frozen ledger; conjunct 3
# and 4 are recomputed here on the arrays the ledger returns, because the ledger
# exposes arrays rather than verdicts; conjunct 5 is the tagged channel identity,
# which `campaign_model.Matched.ledger` computes and `closures.ledger` does not.
LABEL_CHANNELS = {0: 'fast', 1: 'slow', 3: 'M', 4: 'L', 5: 'uptake', 6: 'mineral_loss'}
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, negatives=-1e-7,
           uptake_over_demand=1e-7, label_sum=1e-6, R_lower=-1e-7)


def load_grid():
    g = C.read_json(GRID_JSON)
    pts = ([dict(label='null_delta_1', tau_R_days=None, delta=float(g['null_delta']),
                 kind='null')]
           + [dict(label='tau_%g' % p['tau_R_days'], tau_R_days=float(p['tau_R_days']),
                   delta=float(p['delta']), kind='main') for p in g['main_grid']]
           + [dict(label='limit_delta_%g' % p['delta'], tau_R_days=None,
                   delta=float(p['delta']), kind='limit', note=p['note'])
              for p in g['limit_cases']])
    want = int(g['total_forwards'])
    if len(pts) != want:
        raise SystemExit('GRID_POINT_COUNT_CHANGED %d %d' % (len(pts), want))
    return g, pts


def legal_gate(model, x, delta, r_init=0.0):
    """The five conjuncts, on the frozen code path, plus `R >= -1e-7`."""
    led = C.ledger_with(model, x, delta, r_init)
    # The scale and the label-error reduction are copied from `fit_worker.py:70`,
    # which is the registered definition of these conjuncts -- `scale` is the
    # network inflow, not the largest input element, and `source_label_sum_errors`
    # is a DICT keyed by channel, not an array.
    scale = max(1.0, float(led.get('river_input', led['fast'] + led['slow']).sum()))
    mins = {k: float(np.min(np.asarray(led[k], float)))
            for k in ('M', 'L', 'available', 'uptake', 'fast', 'slow', 'mineral_loss')}
    over = float(np.max(np.asarray(led['uptake'], float)
                        - np.asarray(led['demand'], float)))
    lab = led.get('source_label_sum_errors') or {}
    lab_max = max([float(v) for v in lab.values()], default=0.0)
    lab_missing = sorted(set(LABEL_CHANNELS.values()) - set(lab))
    full = KR.scan_r_full(*model.flux_parameters(torch.tensor(x)), owner=model,
                          delta=delta, r_init=r_init)
    Rmin = float(np.min(full[5]))
    kmin = float(np.min(full[4]))
    kmax = float(np.max(full[4]))
    out = dict(
        local_balance_max_kg=float(led['local_balance_max_kg']),
        network_balance_kg=float(led['network_balance_kg']),
        network_scale_kg=scale,
        min_over_channels=mins,
        max_uptake_minus_demand=over,
        source_label_sum_errors_max=lab_max,
        source_label_sum_errors=lab, source_label_channels_missing=lab_missing,
        R_min=Rmin, R_max=float(np.max(full[5])), kappa_min=kmin, kappa_max=kmax,
        conj1=bool(led['local_balance_max_kg'] <= TOL['local_balance_kg']),
        conj2=bool(abs(led['network_balance_kg']) <= scale * TOL['network_scale']),
        conj3=bool(min(mins.values()) >= TOL['negatives'] and Rmin >= TOL['R_lower']),
        conj4=bool(over <= TOL['uptake_over_demand']),
        conj5=bool(lab_max <= TOL['label_sum'] and not lab_missing),
        R_lower_bound=bool(Rmin >= TOL['R_lower']),
        kappa_in_unit_interval=bool(kmin >= -1e-15 and kmax <= 1.0 + 1e-15))
    out['all_conjuncts'] = bool(out['conj1'] and out['conj2'] and out['conj3']
                                and out['conj4'] and out['conj5'])
    return out, led, full


def main():
    t0 = time.time()
    grid, pts = load_grid()
    ev = E.eligible_events()
    S = dict(phase=1, n_fits=0, fit_worker_calls=0, not_a_fit=True,
             grid_source=str(GRID_JSON.name), grid_sha=C.sha(GRID_JSON),
             grid_definition=grid['definition'],
             n_grid_points=len(pts), grid=pts,
             eligible_set=dict(n_events=int(len(ev)),
                               n_stations=int(ev.station_key.nunique())),
             pre_registration_sha=C.read_json(R / 'reports/预注册_hash.json')
             ['pre_registration']['sha256'])
    base = C.build(TAG)
    x = C.parameters(TAG)
    reach_of = RT.station_reach_map(base)
    # `daily_metadata` covers EVERY station the prediction calendar carries, so the
    # map is wider than the eligible set.  What must hold is that the 15 eligible
    # stations all resolve -- a 15-key assertion would have been the wrong check.
    missing = sorted(set(ev.station_key.astype(str)) - set(reach_of))
    if missing:
        raise SystemExit('ELIGIBLE_STATION_NOT_IN_REACH_MAP %s' % missing)
    S['reach_map'] = dict(n_stations_in_calendar=len(reach_of),
                          n_eligible=15,
                          eligible_reaches=[reach_of[str(s)] for s in
                                            sorted(set(ev.station_key.astype(str)))])

    results = {}
    timings = {}
    for p in pts:
        lab, delta = p['label'], p['delta']
        t1 = time.time()
        C.install_kernel(base, delta, 0.0)
        gate, led, full = legal_gate(base, x, delta, 0.0)
        rep = C.replay(base, TAG)
        C.restore_kernel()
        if delta >= 1.0:
            gate['anchor'] = C.anchor_gate(rep)
        tbl = RT.event_table_for(rep, ev)
        nd, nr = np.shape(full[4])
        mask, _ro, dates = RT.event_window_mask(base, ev, nd, nr, reach_of=reach_of)
        diag = RT.capped_diagnostics(full[4], mask, ev, reach_of, dates)
        diag['headroom'] = RT.cap_headroom(full, mask)
        timings[lab] = time.time() - t1

        rep.to_parquet(R / 'data' / ('phase1_replay_%s.parquet' % lab), index=False)
        tbl.to_parquet(R / 'data' / ('phase1_events_%s.parquet' % lab), index=False)
        fin = np.isfinite(tbl.c_base) & np.isfinite(tbl.c_peak)
        results[lab] = dict(
            tau_R_days=p['tau_R_days'], delta=delta, kind=p['kind'],
            legal=gate,
            n_events=int(fin.sum()),
            c_base_median=float(tbl.c_base[fin].median()),
            c_peak_median=float(tbl.c_peak[fin].median()),
            amp_ratio_median=float((tbl.c_peak / tbl.c_base)[fin].median()),
            amp_ratio_mean=float((tbl.c_peak / tbl.c_base)[fin].mean()),
            capped=diag,
            n_peak_min=int(tbl.n_peak.min()), n_peak_max=int(tbl.n_peak.max()),
            n_base_unique=[int(v) for v in sorted(tbl.n_base.unique())])
        print('%-16s delta=%-22.16g  C_base=%.6f C_peak=%.6f A=%.9f  '
              'capped_cells=%d in_window=%d events_with_cap=%d  conj=%s  %.1fs'
              % (lab, delta, results[lab]['c_base_median'], results[lab]['c_peak_median'],
                 results[lab]['amp_ratio_median'], diag['n_capped_cells'],
                 diag['n_capped_cells_inside_event_windows'],
                 diag['n_events_with_a_capped_day'], gate['all_conjuncts'], timings[lab]),
              flush=True)

    S['points'] = results
    S['runtime_seconds'] = dict(total=time.time() - t0, per_point=timings,
                                note='measured this round, on this machine, with '
                                     'configs/campaign.json resources unchanged')
    S['all_points_legal'] = bool(all(v['legal']['all_conjuncts'] for v in results.values()))
    S['all_points_kappa_in_unit_interval'] = bool(
        all(v['legal']['kappa_in_unit_interval'] for v in results.values()))
    S['outputs'] = dict(replays='data/phase1_replay_<label>.parquet',
                        events='data/phase1_events_<label>.parquet')
    if not S['all_points_legal']:
        S['BLOCKED'] = [k for k, v in results.items() if not v['legal']['all_conjuncts']]
    out = R / 'reports/phase1_envelope.json'
    out.write_text(json.dumps(S, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('ENVELOPE_%s %s  %.1fs'
          % ('DONE' if S['all_points_legal'] else 'BLOCKED', C.sha(out),
             S['runtime_seconds']['total']), flush=True)


if __name__ == '__main__':
    main()
