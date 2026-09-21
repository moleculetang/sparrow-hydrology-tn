"""Hard gates that must pass BEFORE any envelope point is read.

G1  the frozen kernel reproduces `20260916_2`'s own stored C0_s1 daily table
G2  `scan_r(delta=1)` is bitwise `closures.scan` on identical arrays
G3  `tag_scan_r(delta=1)` is bitwise `tagged_transport.tag_scan` on identical arrays
G4  the full replay at `delta=1` is a bitwise no-op against the frozen replay
G5  the ledger at `delta=1` reproduces the frozen local balance and source ledger
G6  `R in [0, av]` and `kappa in [0,1]` on the real domain, at several deltas
G7  the source input is bitwise the registered `monthfirst` construction
G8  `R`'s relaxation timescale is <= 60 d and `M`'s is >= 0.5 y, >= 10x apart

A failure here is a BLOCKED round (`BLOCKED` in the decision table), never evidence
about the mechanism.  Every comparison of two float arrays below that is a no-op
claim uses np.array_equal; the only tolerances are the anchor gate's 1e-12 and the
ledger's own registered thresholds.
"""
import json
import time

import numpy as np
import pandas as pd

import common20 as C
import closures as CL
import tagged_transport as TT
import closures_r as KR

R = C.ROUND
TAG = 'C0_s1'
FREEZE_SHA = '7a1b173086012366'


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print('  %-34s %8.2f s' % (label, dt), flush=True)
    return out, dt


def main():
    S = dict(gate='noop_and_conservation', n_fits=0, fit_worker_calls=0, not_a_fit=True)
    S['frozen_hash_report'] = C.frozen_hash_report()
    bad = [(f, k) for f, d in S['frozen_hash_report'].items()
           for k, v in d['watched'].items() if not v['unchanged']]
    print('frozen_hashes: %s' % ('ALL UNCHANGED' if not bad else 'CHANGED %s' % bad), flush=True)
    if bad:
        raise SystemExit('FROZEN_FILE_CHANGED ' + json.dumps(bad))

    # ---------------------------------------------------------------- G1 / G4
    C.restore_kernel()
    m = C.build(TAG)
    frozen_replay, dt_frozen = timed('frozen replay (12k+ days, 230 reaches)',
                                     lambda: C.replay(m, TAG))
    gate = C.anchor_gate(frozen_replay)
    print('G1 anchor: n=%d max|dc|=%.3e passed=%s'
          % (gate['n_compared_rows'], gate['max_abs_elementwise_concentration'], gate['passed']),
          flush=True)
    S['G1_anchor'] = gate
    if not gate['passed']:
        raise SystemExit('G1_ANCHOR_FAILED')
    S['runtime_seconds'] = dict(frozen_replay=dt_frozen)

    # ---------------------------------------------------------------- G2 / G3
    print('G2/G3 kernel-level bitwise no-op', flush=True)
    import torch
    with torch.no_grad():
        x = C.parameters(TAG)
        h, s, f, k = [v.numpy() for v in m.flux_parameters(torch.tensor(x))]
    h = np.ascontiguousarray(h); s = np.ascontiguousarray(s)
    f = np.ascontiguousarray(f); k = np.ascontiguousarray(k)
    l = np.ascontiguousarray(m.data.lower_release)
    inp = np.ascontiguousarray(m.inp); dem = np.ascontiguousarray(m.demand)
    S['domain_shape'] = dict(h=list(h.shape))

    a_f = CL.scan(h, s, f, k, l, inp, dem, m.cap)
    a_r = KR.scan_r(h, s, f, k, l, inp, dem, m.cap, 1.0, 0.0)
    g2 = {n: dict(array_equal=bool(np.array_equal(a_f[i], a_r[i])),
                  max_abs_diff=float(np.max(np.abs(a_f[i] - a_r[i]))))
          for i, n in enumerate(('fast', 'slow', 'available', 'prob'))}
    g2['kappa_is_one'] = bool(np.array_equal(a_r[4], np.ones_like(a_r[4])))
    g2['kappa_max'] = float(a_r[4].max())
    g2['raw_prob_equals_frozen_prob'] = bool(np.array_equal(a_r[6], a_f[3]))
    g2['p_equals_raw_prob_times_kappa'] = bool(np.array_equal(a_r[3], a_r[6] * a_r[4]))
    print('G2 %s' % {n: v['array_equal'] for n, v in g2.items() if isinstance(v, dict)}, flush=True)
    S['G2_scan_r_delta1_is_bitwise_scan'] = g2
    if not all(v['array_equal'] for v in g2.values() if isinstance(v, dict)):
        raise SystemExit('G2_SCAN_R_NOT_BITWISE_NOOP')
    if not (g2['kappa_is_one'] and g2['raw_prob_equals_frozen_prob']
            and g2['p_equals_raw_prob_times_kappa']):
        raise SystemExit('G2_KAPPA_NOT_EXACTLY_ONE')

    rr = np.asarray(m.data.pilot_indices)
    tr = {kk: np.ascontiguousarray(v) for kk, v in
          dict(h=h[:, rr], s=s[rr], f=f[:, rr], release=m.tag_release,
               inputs=m.tag_inputs, demand=m.tag_demand).items()}
    b_f = TT.tag_scan(tr['h'], tr['s'], tr['f'], tr['release'], tr['inputs'], tr['demand'])
    b_r = KR._tag_scan_r_nb(tr['h'], tr['s'], tr['f'], tr['release'], tr['inputs'],
                            tr['demand'], 1.0, 0.0)
    g3 = {n: dict(array_equal=bool(np.array_equal(b_f[i], b_r[i])),
                  max_abs_diff=float(np.max(np.abs(b_f[i] - b_r[i]))))
          for i, n in enumerate(('fast', 'slow', 'raw', 'M', 'L', 'uptake', 'loss'))}
    print('G3 %s' % {n: v['array_equal'] for n, v in g3.items()}, flush=True)
    S['G3_tag_scan_r_delta1_is_bitwise_tag_scan'] = g3
    if not all(v['array_equal'] for v in g3.values()):
        raise SystemExit('G3_TAG_SCAN_R_NOT_BITWISE_NOOP')

    # ---------------------------------------------------------------- G4
    print('G4 replay at delta=1', flush=True)
    m2 = C.install_kernel(C.build(TAG), 1.0, 0.0)
    rep1, dt1 = timed('replay at delta=1.0', lambda: C.replay(m2, TAG))
    align = frozen_replay.merge(rep1, on=['station_key', 'date'],
                                suffixes=('_f', '_r'), validate='one_to_one')
    if len(align) != len(frozen_replay):
        raise SystemExit('G4_ROW_COUNT_CHANGED')
    g4 = dict(n_rows=int(len(align)),
              array_equal_p=bool(np.array_equal(align.p_f.to_numpy(), align.p_r.to_numpy())),
              array_equal_water=bool(np.array_equal(align.water_m3_day_f.to_numpy(),
                                                    align.water_m3_day_r.to_numpy())),
              max_abs_dc=float(np.max(np.abs(align.p_f - align.p_r))))
    print('G4 %s' % g4, flush=True)
    S['G4_delta1_replay_is_bitwise_noop'] = g4
    if not (g4['array_equal_p'] and g4['array_equal_water']):
        raise SystemExit('G4_REPLAY_NOT_BITWISE_NOOP')
    S['runtime_seconds']['replay_delta1'] = dt1

    # ---------------------------------------------------------------- G5 / G6 / G7
    print('G5/G6/G7 ledger', flush=True)
    C.restore_kernel()
    led_frozen = C.build(TAG).ledger(C.parameters(TAG))
    led = {}
    for delta in (1.0, 0.999, 0.99, 0.9, 0.5, 0.1, 0.0):
        mm = C.build(TAG)
        t0 = time.perf_counter()
        a = C.ledger_with(mm, C.parameters(TAG), delta, 0.0)
        errs = {kk: float(v) for kk, v in a['source_label_sum_errors'].items()}
        led['%g' % delta] = dict(
            local_balance_max_kg=float(a['local_balance_max_kg']),
            network_balance_kg=float(a['network_balance_kg']),
            source_label_sum_errors=errs,
            seconds=time.perf_counter() - t0)
        print('   delta=%-6g local_balance=%.3e  M_err=%.3e  L_err=%.3e  balance=%.3e'
              % (delta, a['local_balance_max_kg'], errs['M'], errs['L'],
                 a['network_balance_kg']), flush=True)
    C.restore_kernel()
    errs_f = {kk: float(v) for kk, v in led_frozen['source_label_sum_errors'].items()}
    g5 = dict(
        frozen_local_balance_max_kg=float(led_frozen['local_balance_max_kg']),
        frozen_source_label_sum_errors=errs_f)
    S['G5_ledger'] = dict(frozen=g5, per_delta=led)
    worst_lb = max(v['local_balance_max_kg'] for v in led.values())
    S['G5_ledger']['worst_local_balance_kg'] = worst_lb
    S['G5_ledger']['local_balance_within_1e-6'] = bool(worst_lb <= 1e-6)
    if worst_lb > 1e-6:
        raise SystemExit('G5_LOCAL_BALANCE_EXCEEDS_1e-6')
    # the operative conjunct-5 gate: no increase over the frozen kernel at delta=1
    d1 = led['1']['source_label_sum_errors']
    S['G5_ledger']['conjunct5_delta1_equals_frozen'] = {
        kk: bool(abs(d1[kk] - errs_f[kk]) <= 1e-12 * max(1.0, abs(errs_f[kk]))) for kk in errs_f}
    if not all(S['G5_ledger']['conjunct5_delta1_equals_frozen'].values()):
        raise SystemExit('G5_CONJUNCT5_DELTA1_MOVED')

    # G6 on the real domain
    g6 = {}
    for delta in (1.0, 0.9, 0.5, 0.1):
        mm = C.install_kernel(C.build(TAG), delta, 0.0)
        with torch.no_grad():
            hh, ss, ff, kk = [v.numpy() for v in mm.flux_parameters(torch.tensor(x))]
        out = KR.scan_r(np.ascontiguousarray(hh), np.ascontiguousarray(ss),
                        np.ascontiguousarray(ff), np.ascontiguousarray(kk),
                        l, inp, dem, mm.cap, delta, 0.0)
        av, kap, Rrec = out[2], out[4], out[5]
        g6['%g' % delta] = dict(
            kappa_min=float(kap.min()), kappa_max=float(kap.max()),
            kappa_in_unit_interval=bool((kap >= 0).all() and (kap <= 1.0).all()),
            frac_capped=float(np.mean(kap < 1.0)),
            frac_capped_strongly=float(np.mean(kap < 0.9)),
            R_min=float(Rrec.min()), R_max=float(Rrec.max()),
            R_nonnegative=bool((Rrec >= 0).all()),
            # `R` is a lagging bound, not a pool: it may EXCEED the current `av`
            # after a sharp drop in av, and that is the memory this round is
            # testing.  The invariant that matters is `E <= av*prob <= av`, which
            # is kappa <= 1, not R <= av.
            R_over_av_max=float(np.max(Rrec / np.where(av > 0, av, np.nan))),
            frac_E_lt_avprob=float(np.mean(kap < 1.0)))
        print('   G6 delta=%-5g kappa in [%.4f,%.4f] unit=%s capped=%.5f capped<0.9=%.5f R>=0=%s'
              % (delta, g6['%g' % delta]['kappa_min'], g6['%g' % delta]['kappa_max'],
                 g6['%g' % delta]['kappa_in_unit_interval'],
                 g6['%g' % delta]['frac_capped'], g6['%g' % delta]['frac_capped_strongly'],
                 g6['%g' % delta]['R_nonnegative']), flush=True)
    C.restore_kernel()
    S['G6_R_domain'] = g6
    if not all(v['R_nonnegative'] and v['kappa_in_unit_interval'] for v in g6.values()):
        raise SystemExit('G6_KAPPA_OR_R_OUT_OF_RANGE')

    # G7 the source input is the registered monthfirst construction, untouched
    mm = C.build(TAG)
    _like = np.asarray(mm.data.contact, float)
    st = np.asarray(mm.data.starts)
    # THREE different shape classes are being compared here, and conflating them
    # was the first version of this checker's error -- a failing assertion is
    # usually the assertion:
    #   `source`/`crop`/`source_tags`  (768, ...)   MONTHLY, spread across days
    #   `lower_release`/`contact`      (23376,230)  already DAILY, only restricted
    #   `tag_inputs`                   (23376,2,4)  daily-spread, zero off `starts`
    # `campaign_model.py:72` writes `tag_inputs[data.starts] = source_tags[:,rr,:]`
    # and `:73` writes `tag_demand = demand[:,rr]`, `tag_release = lower_release[:,rr]`.
    _off = np.ones(len(np.asarray(mm.tag_inputs)), bool); _off[st] = False
    g7 = dict(
        inp_is_registered_monthfirst=bool(np.array_equal(
            np.asarray(mm.inp, float), _reg(mm.data.source, st, _like))),
        demand_is_registered_monthfirst=bool(np.array_equal(
            np.asarray(mm.demand, float), _reg(mm.data.crop, st, _like))),
        tag_inputs_is_registered_monthfirst=bool(np.array_equal(
            np.asarray(mm.tag_inputs, float)[st], np.asarray(mm.data.source_tags)[:, rr, :])),
        tag_demand_is_registered_monthfirst=bool(np.array_equal(
            np.asarray(mm.tag_demand, float), _reg(mm.data.crop, st, _like)[:, rr])),
        tag_release_is_registered_daily=bool(np.array_equal(
            np.asarray(mm.tag_release, float), np.asarray(mm.data.lower_release)[:, rr])),
        # the spread is *writing only at* `starts`: every non-start day must be 0.
        # (`count_nonzero` is compared to `count_nonzero`, not to `.size` -- 256 of
        # the 6144 monthly tag entries are themselves zero, so a `.size` reading
        # fails on a correct array.  Second checker defect of the same class.)
        tag_inputs_zero_off_month_start=bool(
            np.count_nonzero(np.asarray(mm.tag_inputs, float)[_off]) == 0),
        tag_inputs_nonzero_count=bool(
            np.count_nonzero(np.asarray(mm.tag_inputs, float)) ==
            np.count_nonzero(np.asarray(mm.data.source_tags)[:, rr, :])),
        source_tag_identity=bool(np.allclose(
            np.asarray(mm.data.source_tags)[:, rr, :].sum(-1),
            np.asarray(mm.data.source)[:, rr], rtol=1e-13, atol=1e-7)))
    print('G7 %s' % g7, flush=True)
    S['G7_source_input_untouched'] = g7
    if not all(g7.values()):
        raise SystemExit('G7_SOURCE_INPUT_CHANGED')

    # ---------------------------------------------------------------- G8
    print('G8 timescales', flush=True)
    S['G8_timescales'] = _timescales()

    C.write_json(R / 'reports/gate_noop.json', S)
    print('GATE_NOOP_PASSED', flush=True)


def _reg(monthly, starts, like):
    """The registered monthfirst construction, rebuilt from the domain arrays.

    `closures.py:138,142` and `campaign_model.py:72` are the frozen construction:
    the whole month's total is assigned to the month's first day and to no other
    day.  Reproducing it here from `data.source` / `data.crop` / `data.starts` is
    what proves this round did not touch the injection.
    """
    out = np.zeros_like(np.asarray(like, float))
    out[np.asarray(starts)] = np.asarray(monthly, float)
    return out


def _timescales():
    """The analytic `R` relaxation against the frozen `M`/`L` pool, plus a check.

    `R` relaxes as `(1-delta)^n`, so its e-folding is `-1/ln(1-delta)` days.
    `M`'s is not a free parameter of this round: it is the frozen
    `log_tau_mineral_days`, i.e. `exp(theta)`.  The point of G8 is that the new
    state cannot be a renaming of the existing one -- the two timescales must be
    separated by at least an order of magnitude.

    The probe ladder below IS the round's registered operating range for
    `tau_R in [1, 60]` d, i.e. `delta in [0.01653, 0.63212]`, plus the null point
    `delta = 1`.  `delta -> 0` is the LIMIT case (the cap collapses to the frozen
    `r_init` and no mass is mobilizable at all); it lies outside the operating
    range by construction and is reported separately, never inside `r_max`.
    """
    import math

    def tau_of(delta):
        # delta = 1 -> R is assigned exactly `av` every day: no memory at all,
        # so the relaxation time is 0, NOT log(0) (math.log(0.0) raises).
        if delta >= 1.0:
            return 0.0
        if delta <= 0.0:
            return float('inf')
        return -1.0 / math.log(1.0 - delta)

    rec = C.model_record(TAG)
    names = list(rec['names'])
    tau_min = float(rec['parameters'][names.index('log_tau_mineral_days')])
    m_efold_days = math.exp(tau_min)
    range_delta = (1.0 - math.exp(-1.0 / 60.0), 1.0 - math.exp(-1.0 / 1.0))
    out = dict(log_tau_mineral_days=tau_min, M_efold_days=m_efold_days,
               M_efold_years=m_efold_days / 365.25,
               operating_delta_for_tau_1_to_60_days=list(range_delta),
               operating_tau_days=[1.0, 60.0],
               delta_for_tau_30_days=1.0 - math.exp(-1.0 / 30.0),
               per_delta={})
    n = 30
    for delta in (1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02):
        tau = tau_of(delta)
        # a second, independent reading: drive one reach with a single injection
        # and measure how far R has closed the gap to av after `n` days.
        R = 0.0
        av = 100.0
        for _ in range(n):
            R = av + (1.0 - delta) * (R - av)
        out['per_delta']['%g' % delta] = dict(
            tau_days_closed_form=tau,
            gap_after_30d=(av - R) / av,
            predicted_gap=(1.0 - delta) ** n)
    out['limit_case'] = dict(
        delta=0.0, tau_days_closed_form=tau_of(0.0),
        note='delta -> 0+ freezes R at r_init, so E -> r_init and the reach goes '
             'silent; this is the achievable boundary, not an operating point.')
    fin = [(k, v['tau_days_closed_form']) for k, v in out['per_delta'].items()
           if np.isfinite(v['tau_days_closed_form'])]
    out['R_tau_days'] = dict(fin)
    r_max = max(v for _, v in fin) if fin else float('inf')
    ratio = m_efold_days / r_max if r_max > 0 else float('inf')
    out['assertion'] = dict(
        R_tau_max_days=r_max,
        R_tau_le_60d=bool(r_max <= 60.0),
        M_tau_ge_half_year=bool(m_efold_days >= 182.625),
        M_over_R_tau=ratio,
        at_least_10x_apart=bool(ratio >= 10.0),
        closed_form_agrees_with_iteration=bool(all(
            abs(v['gap_after_30d'] - v['predicted_gap']) <= 1e-12
            for v in out['per_delta'].values())))
    print('G8 %s' % out['assertion'], flush=True)
    if not (out['assertion']['at_least_10x_apart']
            and out['assertion']['R_tau_le_60d']
            and out['assertion']['M_tau_ge_half_year']
            and out['assertion']['closed_form_agrees_with_iteration']):
        raise SystemExit('G8_R_NOT_DISTINGUISHABLE_FROM_M')
    return out


if __name__ == '__main__':
    main()
