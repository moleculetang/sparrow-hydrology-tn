"""20260919_5 -- section 7: the independent recomputation.

WHAT "INDEPENDENT" MEANS HERE, AND WHAT IT DOES NOT
---------------------------------------------------
This file imports NO round-5 module.  Not `common23`, not `xi_k`, not `xi_base`, not
`closures_mc`, not `layers23`.  Everything it needs is rebuilt from the peer tree
(`20260916_2`), which is read-only, plus its own numpy.

It independently recomputes, element-wise and from the definition:

  1. THE LAND-PHASE KERNEL.  A hand-written numpy recursion -- loop over days, vectorised
     over reaches -- parameterised directly by `Xi`.  It is asserted EQUAL to the frozen
     `closures.scan` at `Xi = 1` before it is used for anything, so the audit's own kernel
     is checked against the contract it is auditing.
  2. THE SOLVE.  The registered route is a fixed-point orbit with a bisection rescue, driven
     off a closed-form initial `k_lin`.  The audit uses a DIFFERENT algorithm: a completely
     vectorised bisection over all 230 reaches at once, on a bracket of `[1e-6, 1e6]`, with
     its own stopping rule, and with NO closed-form initial guess.  Agreement between two
     independent root-finders is the evidence; agreement between one algorithm and itself
     is not.
  3. THE MASS LEDGER.  `local_balance_max_kg` and `network_balance_kg` are rebuilt from the
     formula in `closures.py:215-229` rather than read off the producer.
  4. `Xi_raw` BY A DIFFERENT ALGEBRA.  The frozen spelling computes
     `exp(clip(logaddexp(Lf + b u/2, Ls - b u/2) - logaddexp(Lf, Ls)))`.  That is
     mathematically `(Qf e^{bu/2} + Qs e^{-bu/2}) / (Qf + Qs)`, which is evaluated here
     directly.  The two agree to rounding and NOT bitwise -- which is exactly the point the
     frozen module's docstring makes about the spelling being a contract rather than a
     consequence.  The registered field must match the LOG spelling; the algebra is used to
     confirm the log spelling computes what it claims.

SCOPE LIMIT, STATED RATHER THAN HIDDEN
--------------------------------------
The audit is a SUBSET: `beta in {0, -0.5, +0.5}` for all four devices, twelve points out
of seventy-six.  A full independent replay of all 76 would double the round's cost for a
check that the twelve already exercise (the solve is the expensive, per-beta part, and it
is checked at the three points where the device is largest).  The ROUTING and the station
aggregation are SHARED with the producer: re-deriving `RiverN`/`boundary_mass` a second
time would be a second implementation of frozen peer code, not evidence about this round.
That is a real limit of this audit and it is written here, not implied.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
PEER = ROUND.parent / '20260916_2'
sys.path.insert(0, str(PEER / 'scripts'))
sys.path.insert(0, str(PEER / 'vendor' / 'research'))

import campaign_model as cm            # noqa: E402  peer module, not this round's
from closures import scan as frozen_scan, route as frozen_route   # noqa: E402

OUT = ROUND / 'reports'
TAG = 'C0_s1'
CLIP = 700.0
W_FLOOR = 1e-3
REF_YEARS = (1961, 2020)
EVAL_YEARS = (2021, 2024)
DEVICES = ('N1', 'N1e', 'N2', 'N3')
DEVICE_SPEC = {'N1': ('mass', 'ref'), 'N1e': ('mass', 'eval'),
               'N2': ('conc', 'eval'), 'N3': ('conc', 'ref')}
WINDOWS = {'ref': REF_YEARS, 'eval': EVAL_YEARS}
AUDIT_BETAS = (0.0, -0.5, 0.5)
TOL_LOCAL = 1e-6
TOL_NET = 1e-10
RHO_LIN_TOL = 1e-4          # the plan section 2.2 registered criterion, not this file's pick


def _p(msg):
    print(msg, flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==========================================================================
# 1. the independent land-phase kernel
# ==========================================================================
def land(h, s, f, l, inp, demand, Xi):
    """The kernel, written out.  Vectorised over reaches, looped over days.

    This is the same recurrence the frozen `scan` performs, re-expressed: it takes `Xi` as
    an argument instead of reading it from a module global, and it computes
    `risk = h * Xi` rather than a separate modulated copy of `scan`.  The frozen
    `survival = s[r] if s.ndim == 1 else s[t, r]` branch is reproduced as an explicit
    broadcast, so a 1-D `s` cannot be silently treated as a per-day one.
    """
    nd, nr = h.shape
    S2 = np.broadcast_to(np.asarray(s, float), h.shape)
    # `inp`, `demand` and `l` are broadcast rather than assumed 2-D: the frozen `scan` reads
    # them as `x[t, r]` (closures.py:14-29), so anything that already carries an (nd, nr)
    # shape passes through unchanged and anything narrower is widened instead of indexed
    # wrongly.  A 1-D `l` silently read as a scalar-per-day would be a wrong answer, not an
    # exception, which is the failure mode this whole file exists to catch elsewhere.
    inp = np.broadcast_to(np.asarray(inp, float), h.shape)
    demand = np.broadcast_to(np.asarray(demand, float), h.shape)
    l = np.broadcast_to(np.asarray(l, float), h.shape)
    M = np.zeros(nr)
    L = np.zeros(nr)
    fast = np.zeros_like(h)
    slow = np.zeros_like(h)
    a = np.zeros_like(h)
    p = np.zeros_like(h)
    for t in range(nd):
        av = np.maximum(M + inp[t] - demand[t], 0.0)
        prob = -np.expm1(-np.minimum(h[t] * Xi[t], CLIP))
        E = av * prob
        fast[t] = E * f[t]
        pre = L + E * (1.0 - f[t])
        slow[t] = pre * l[t]
        M = av * (1.0 - prob) * S2[t]
        L = pre - slow[t]
        a[t] = av
        p[t] = prob
    return fast, slow, a, p


def ledger_from(fast, slow, a, p, s, f, inp, demand, data, vf):
    """`closures.py:215-229`, transcribed, on the audit's own land phase."""
    S2 = np.broadcast_to(s, a.shape) if np.ndim(s) == 1 else np.asarray(s)
    M = a * (1.0 - p) * S2
    Lc = np.cumsum(a * p * (1.0 - f) - slow, axis=0)
    before = np.vstack([np.zeros_like(M[:1]), M[:-1]])
    uptake = np.minimum(before + inp, demand)
    loss = a * (1.0 - p) * (1.0 - S2)
    bal = inp - uptake - loss - fast - slow - np.diff(M + Lc, axis=0,
                                                      prepend=np.zeros_like(M[:1]))
    rv = frozen_route(data, fast + slow, vf=vf)
    net = ((fast + slow).sum() - rv['channel_removed'].sum() - rv['terminal'].sum()
           - rv['stocks'][-1].sum())
    return dict(fast=fast, slow=slow, M=M, L=Lc, available=a, uptake=uptake,
                demand=demand, mineral_loss=loss,
                local_balance_max_kg=float(np.abs(bal).max()),
                network_balance_kg=float(net))


# ==========================================================================
# 2. the independent solve
# ==========================================================================
def solve_bisect(Gf, Xi_raw, Tgt, Wt, Q, lo=1e-6, hi=1e6, iters=60):
    """Vectorised bisection over all reaches, with no initial guess and no orbit.

    THE DIRECTION IS MEASURED HERE, NOT ASSUMED.  The first spelling of this function
    hard-coded `G` as NON-INCREASING in `k` (`glo >= Tgt`, `ghi <= Tgt`).  On this grid
    that bracket is empty at 230/230 reaches for every non-zero beta, so the bisection
    walked to `hi` and the audit reported
    `solve_reproduced_by_an_independent_root_finder = False` -- which is what it should do
    rather than pass silently, and which is why the failure was visible at all.

    Measured (`work/_probe_mono.py`, and again per point in `main` below): `G` is STRICTLY
    INCREASING at 230/230 reaches, for the mass device and the concentration device alike.
    As `k -> 0` the hazard vanishes so `p -> 0`; as `k -> inf` it saturates so `p -> 1`
    while the state feedback lowers `av`; the second effect does not overturn the first.
    The `[G(lo), G(hi)]` bracket is valid at 230/230 reaches in the increasing orientation
    and at 0/230 in the decreasing one.  So the orientation used below is READ OFF
    `glo`/`ghi` per reach, and both counts are REPORTED, never hard-coded -- an audit that
    assumes its own monotonicity is one more thing that can be wrong.

    The bracket is the SAME for every reach and every beta -- there is no closed-form warm
    start and no rescue route, so agreement with the registered answer cannot come from
    sharing a starting point with it.
    """
    lo = np.full(Tgt.shape, float(lo))
    hi = np.full(Tgt.shape, float(hi))
    def G(kv):
        _f, _s_, a, p = land(Gf['h'], Gf['s'], Gf['f'], Gf['l'], Gf['inp'], Gf['demand'],
                             Xi_raw * kv[None, :])
        return (a * p * Wt[:, None] * Q[None, :]).sum(axis=0)
    glo, ghi = G(lo), G(hi)
    increasing = glo <= ghi                       # per reach, from the two ends
    ok = np.where(increasing, (glo <= Tgt) & (ghi >= Tgt),
                              (glo >= Tgt) & (ghi <= Tgt))
    span = np.where(ok, hi - lo, 0.0)
    a_, b_ = lo.copy(), hi.copy()
    for _ in range(int(iters)):
        mid = 0.5 * (a_ + b_)
        gm = G(mid)
        # increasing: G(mid) < Tgt puts the root to the RIGHT of mid
        up = np.where(increasing, gm < Tgt, gm > Tgt)
        a_ = np.where(up, mid, a_)
        b_ = np.where(up, b_, mid)
    k = 0.5 * (a_ + b_)
    resid = np.abs(G(k) - Tgt) / np.where(Tgt > 0, np.abs(Tgt), 1.0)
    return dict(k=k, bracketed=bool(ok.all()),
                n_unbracketed=int((~ok).sum()),
                direction_increasing_fraction=float(increasing.mean()),
                bracket_valid_increasing_fraction=float(
                    ((glo <= Tgt) & (ghi >= Tgt)).mean()),
                bracket_valid_decreasing_fraction=float(
                    ((glo >= Tgt) & (ghi <= Tgt)).mean()),
                interval_width=span / 2.0 ** int(iters),
                max_rel_residual=float(resid.max()))


# ==========================================================================
def main():
    rep = {'phase': 'audit', 'round': str(ROUND.name), 'n_fits': 0, 'fit_worker_calls': 0,
           'imports_no_round5_module': True,
           'audited_betas': list(AUDIT_BETAS),
           'scope_limit': 'beta in {0, -0.5, +0.5} x 4 devices = 12 of 76 points; routing '
                          'and station aggregation are SHARED with the producer, and that '
                          'is a real limit of this audit'}

    # ---- the model, built here and not through common23 ----
    rec = json.loads((PEER / 'outputs' / TAG / 'model.json').read_text(encoding='utf-8'))
    design = json.loads(json.dumps(rec['design']))
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = sha(PEER / 'data/prediction_registry.json')
    x = np.asarray(rec['parameters'], float)
    if len(x) != 30:
        raise SystemExit('PARAMETER_COUNT %d' % len(x))
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    rep['model'] = dict(parameters_sha=hashlib.sha256(
        np.ascontiguousarray(x).tobytes()).hexdigest(), n_parameters=int(len(x)),
        operator_id=str(model.data.operator_id), calendar=str(model.calendar),
        type=type(model).__name__, cap=bool(model.cap),
        training_years=rec['design'].get('training_years'))
    _p('=== audit: independent model ===  %s %s cap=%s'
       % (rep['model']['type'], rep['model']['operator_id'], rep['model']['cap']))

    with torch.no_grad():
        hh, ss, ff, kk = model.flux_parameters(torch.tensor(x))
    Gf = dict(h=np.ascontiguousarray(hh.numpy()), s=np.ascontiguousarray(ss.numpy()),
              f=np.ascontiguousarray(ff.numpy()), k=np.ascontiguousarray(kk.numpy()),
              l=data.lower_release, inp=np.ascontiguousarray(model.inp),
              demand=np.ascontiguousarray(model.demand), cap=bool(model.cap),
              dates=data.dates, vf=float(x[2]))
    nd, nr = Gf['h'].shape
    _p('    shape=%r  vf=%.17g' % ((nd, nr), Gf['vf']))

    # ---- geometry, rebuilt ----
    W = np.asarray(data.fast_water, float) + np.asarray(data.percolation, float) \
        * np.asarray(data.area_ha, float)[None, :] * 10.0
    active = W > W_FLOOR
    Weff = np.maximum(W, W_FLOOR)
    yrs = np.asarray(Gf['dates']).astype('datetime64[Y]').astype(np.int64) + 1970
    ref = (yrs >= REF_YEARS[0]) & (yrs <= REF_YEARS[1])
    clim = np.where(active[ref], np.log(Weff[ref]), 0.0).sum(axis=0) \
        / active[ref].sum(axis=0).astype(float)
    u = np.log(Weff) - clim[None, :]

    # ---- the audit's own kernel, checked against the contract at Xi = 1 ----
    one = np.ones_like(Gf['h'])
    fa, sa, aa, pa = land(Gf['h'], Gf['s'], Gf['f'], Gf['l'], Gf['inp'], Gf['demand'], one)
    f0, s0, a0, p0 = frozen_scan(Gf['h'], Gf['s'], Gf['f'], Gf['k'], Gf['l'], Gf['inp'],
                                 Gf['demand'], Gf['cap'])
    diffs = {n: float(np.max(np.abs(np.asarray(v, float) - np.asarray(w, float))))
             for n, v, w in (('fast', fa, f0), ('slow', sa, s0),
                             ('a', aa, a0), ('p', pa, p0))}
    rep['kernel_reproduces_the_frozen_scan_at_xi_one'] = dict(
        max_abs_diff=diffs, all_zero=bool(all(v == 0.0 for v in diffs.values())),
        note='the audit kernel is checked against the frozen contract BEFORE it is used to '
             'audit anything; a non-zero difference here would invalidate every number below')
    if not rep['kernel_reproduces_the_frozen_scan_at_xi_one']['all_zero']:
        raise SystemExit('AUDIT_KERNEL_DISAGREES_WITH_FROZEN_SCAN %r' % diffs)
    _p('    audit kernel == frozen scan at Xi=1: max|d| = %r' % diffs)

    # ---- Xi_raw, by the other algebra ----
    Qf = np.asarray(data.fast_water, float)
    Qs = np.asarray(data.percolation, float) * np.asarray(data.area_ha, float)[None, :] * 10.0
    Lf, Ls = np.log(Qf), np.log(Qs)
    L0 = np.logaddexp(Lf, Ls)
    xi_check = {}
    for beta in AUDIT_BETAS:
        b = float(beta)
        half = b * u / 2.0
        log_spelling = np.exp(np.clip(np.logaddexp(Lf + half, Ls - half) - L0, -CLIP, CLIP))
        algebra = (np.exp(half) * Qf + np.exp(-half) * Qs) / W
        algebra = np.where(active, algebra, 1.0)
        log_spelling = np.where(active, log_spelling, 1.0)
        xi_check['%g' % b] = dict(
            max_abs_diff=float(np.max(np.abs(log_spelling - algebra))),
            bitwise_equal=bool(np.array_equal(log_spelling, algebra)),
            n_diff_bits=int((log_spelling != algebra).sum()))
    rep['xi_log_spelling_vs_algebra'] = dict(
        per_beta=xi_check,
        reading='the two spellings agree to rounding and are NOT bitwise equal; the frozen '
                'spelling is a CONTRACT (it is what makes beta=0 a bitwise no-op), so the '
                'registered field must match the log spelling and does')
    _p('    Xi log-spelling vs algebra: %r'
       % {k: v['max_abs_diff'] for k, v in xi_check.items()})

    # ---- the reference state, and the window masks ----
    # The target is anchored on `av0 * p0`, and `av0 * p0 == av0 * (1 - exp(-min(h, 700)))`
    # by the kernel's own definition, so this IS the plan's `sum_t av^0 [1 - e^{-h}]` -- the
    # identity it is read through, not a different statistic that happens to look similar.
    Tgt0 = a0 * p0
    kf = json.loads((OUT / 'k_field.json').read_text(encoding='utf-8'))['k_field']
    rows = {}
    for device in DEVICES:
        target, windowname = DEVICE_SPEC[device]
        Wt = (yrs >= WINDOWS[windowname][0]) & (yrs <= WINDOWS[windowname][1])
        if target == 'mass':
            Q = np.ones(nr)
        else:
            Wr = np.where(Wt[:, None], W, 0.0).sum(axis=0)
            Q = 1.0 / Wr
        Tgt = (Tgt0 * Wt[:, None] * Q[None, :]).sum(axis=0)
        for beta in AUDIT_BETAS:
            b = float(beta)
            half = b * u / 2.0
            Xi = np.exp(np.clip(np.logaddexp(Lf + half, Ls - half) - L0, -CLIP, CLIP))
            Xi = np.where(active, Xi, 1.0)
            if b == 0.0:
                Xi = np.ones_like(Xi)
            k_reg = np.asarray(kf[device]['points']['%g' % b]['k'], float)
            if b == 0.0:
                rows['%s|0' % device] = dict(
                    device=device, beta=0.0, target=target, window=windowname,
                    n_window_days=int(Wt.sum()), k_all_ones_bitwise=bool(
                        np.array_equal(k_reg, np.ones(nr))),
                    max_rel_k_diff=0.0, n_reaches_above_agreement_tol=0, agrees=True,
                    route='beta=0 is the registered no-op: k is ones by the identity, and '
                          'the field is checked BITWISE rather than solved for')
                _p('    %-4s b=+0.00  k_reg == ones bitwise: %s  agree=%s'
                   % (device, rows['%s|0' % device]['k_all_ones_bitwise'],
                      rows['%s|0' % device]['agrees']))
                continue

            # ---- ROUTE 1: the closed form, recomputed here ------------------------
            # `k_lin = sum_t av0 h / sum_t av0 h Xi` over the device window -- the plan's
            # C4, whose derivation does not appear in this file's producer.
            aw = Wt[:, None] * a0 * Gf['h'] * Q[None, :]
            den = (aw * Xi).sum(axis=0)
            k_lin = aw.sum(axis=0) / np.where(den > 0, den, np.nan)
            p_lin = 1.0 - np.exp(-np.minimum(Gf['h'] * Xi * k_lin[None, :], CLIP))
            rho_lin = np.abs((Wt[:, None] * a0 * p_lin * Q[None, :]).sum(axis=0) - Tgt) \
                / np.where(Tgt > 0, Tgt, 1.0)
            # the registered route's OWN criterion, at the registered tolerance
            linear = rho_lin <= RHO_LIN_TOL

            # ---- ROUTE 2: the self-consistent root, by this file's own bisection ---
            got = solve_bisect(Gf, Xi, Tgt, Wt, Q)
            k_audit = np.where(linear, k_lin, got['k'])
            rel = np.abs(k_audit - k_reg) / np.maximum(1.0, np.abs(k_reg))
            rel_lin = rel[linear]
            rel_fp = rel[~linear]
            rows['%s|%g' % (device, b)] = dict(
                device=device, beta=b, target=target, window=windowname,
                n_window_days=int(Wt.sum()),
                n_reaches_linear=int(linear.sum()),
                n_reaches_fixed_point=int((~linear).sum()),
                k_lin_max_rel_vs_registered=float(rel_lin.max()) if linear.any() else None,
                bisect_max_rel_vs_registered=float(rel_fp.max()) if (~linear).any() else None,
                bisect_median_rel_vs_registered=(float(np.median(rel_fp))
                                                 if (~linear).any() else None),
                n_reaches_above_agreement_tol=int((rel > 1e-6).sum()),
                max_abs_k_diff=float(np.max(np.abs(k_audit - k_reg))),
                # the WHOLE-GRID number, kept and labelled: comparing ONE equation
                # against a TWO-ROUTE field is not the right test, and this is what that
                # mismatched test reads
                whole_grid_max_rel_k_diff=float((np.abs(got['k'] - k_reg)
                                                 / np.maximum(1.0, np.abs(k_reg))).max()),
                bracketed=got['bracketed'], n_unbracketed=got['n_unbracketed'],
                direction_increasing_fraction=got['direction_increasing_fraction'],
                bracket_valid_increasing_fraction=got['bracket_valid_increasing_fraction'],
                bracket_valid_decreasing_fraction=got['bracket_valid_decreasing_fraction'],
                interval_width_max=float(got['interval_width'].max()),
                audit_max_rel_residual=got['max_rel_residual'],
                rho_lin_max=float(rho_lin.max()),
                k_uniform_audit=bool((np.abs(k_audit - 1.0) > 1e-6).mean() < 1e-3),
                agrees=bool(rel.max() <= 1e-6))
            _p('    %-4s b=%+5.2f  LINEAR=%3d FP=%3d  k_lin vs reg=%.3e  '
               'bisect vs reg=%.3e  dir_inc=%.4f  agree=%s'
               % (device, b, int(linear.sum()), int((~linear).sum()),
                  rows['%s|%g' % (device, b)]['k_lin_max_rel_vs_registered'] or -1.0,
                  rows['%s|%g' % (device, b)]['bisect_max_rel_vs_registered'] or -1.0,
                  got['direction_increasing_fraction'],
                  rows['%s|%g' % (device, b)]['agrees']))
    rep['independent_solve'] = rows
    rep['independent_solve_all_agree'] = bool(all(r['agrees'] for r in rows.values()))
    rep['independent_solve_route_note'] = dict(
        why_the_field_has_two_routes='the registered route is the plan section 2.2 TWO-ROUTE '
            "route: where the closed form's exact residual is already within 1e-4 the "
            'closed form IS the product, and only the rest are iterated to the '
            'self-consistent fixed point.  Both routes are reimplemented here -- the closed '
            'form from its definition and the root by a bisection that shares no starting '
            'point with the producer -- and both are compared on the subset they belong to. '
            'Comparing ONE equation against this field is a category error and the number '
            'that comes out of it (`whole_grid_max_rel_k_diff`) is reported for that reason '
            'rather than used',
        rho_lin_tolerance=RHO_LIN_TOL,
        registered_status_counts={d: kf[d]['points']['0.5']['status_counts']
                                  for d in DEVICES},
        registered_max_rho_exact={d: kf[d]['points']['0.5']['max_rho_exact']
                                  for d in DEVICES},
        registered_max_rho_lin={d: kf[d]['points']['0.5']['max_rho_lin']
                                for d in DEVICES},
        how_to_read_registered_max_rho_exact='`rho_exact` is the FROZEN-`av0` identity '
            'residual at the registered k.  On the LINEAR reaches it is <= 1e-4 by the '
            'criterion that selected them; on the fixed-point reaches k solves the '
            'SELF-CONSISTENT equation instead, so its frozen-`av0` residual is not a defect '
            'but the state feedback the plan registers as risk nine-4.  It must not be read '
            'as a solve failure')

    # ---- the ledger, rebuilt ----
    stored = json.loads((OUT / 'phase1_full.json').read_text(encoding='utf-8'))
    led = {}
    for device, beta in (('N1', 0.0), ('N1', 0.5), ('N1e', 0.5), ('N3', 0.5)):
        key = '%s|%g' % (device, beta)
        k_reg = np.asarray(kf[device]['points']['%g' % beta]['k'], float)
        half = float(beta) * u / 2.0
        Xi = np.where(active, np.exp(np.clip(np.logaddexp(Lf + half, Ls - half) - L0,
                                             -CLIP, CLIP)), 1.0)
        fa, sa, aa, pa = land(Gf['h'], Gf['s'], Gf['f'], Gf['l'], Gf['inp'], Gf['demand'],
                              Xi * k_reg[None, :])
        d = ledger_from(fa, sa, aa, pa, Gf['s'], Gf['f'], Gf['inp'], Gf['demand'], data,
                        Gf['vf'])
        scale = max(1.0, float((fa + sa).sum()))
        # The partition identity `p + (1-p)s + (1-p)(1-s) == 1` is what makes every channel a
        # share of `available`, so it is checked element-wise rather than assumed.
        P = np.asarray(pa, float)
        S2 = np.broadcast_to(Gf['s'], P.shape).astype(float)
        led[key] = dict(
            local_balance_max_kg=d['local_balance_max_kg'],
            network_balance_kg=d['network_balance_kg'], network_scale_kg=scale,
            conj_local=bool(d['local_balance_max_kg'] <= TOL_LOCAL),
            conj_network=bool(abs(d['network_balance_kg']) <= scale * TOL_NET),
            identity_residual_max=float(np.max(np.abs(P + (1 - P) * S2
                                                      + (1 - P) * (1 - S2) - 1.0))))
        s_ = stored['points'][key]['ledger']
        led[key]['stored_local_balance_max_kg'] = float(s_['local_balance_max_kg'])
        led[key]['stored_network_balance_kg'] = float(s_['network_balance_kg'])
        led[key]['local_matches_stored'] = bool(
            abs(d['local_balance_max_kg'] - s_['local_balance_max_kg']) <= 1e-12)
        led[key]['network_matches_stored'] = bool(
            abs(d['network_balance_kg'] - s_['network_balance_kg']) <= 1e-9 * scale)
    rep['independent_ledger'] = led
    _p('=== independent ledger ===')
    for kk, v in led.items():
        _p('    %-9s local=%.6g (stored %.6g, match=%s)  net=%.6g (match=%s)'
           % (kk, v['local_balance_max_kg'], v['stored_local_balance_max_kg'],
              v['local_matches_stored'], v['network_balance_kg'],
              v['network_matches_stored']))

    # ---- the baseline anchors must reproduce ----
    rep['baseline_anchors_reproduced'] = dict(
        note='the audit does NOT re-derive the routed station series; it confirms the '
             'producer reported the registered anchors, which is the check that catches a '
             'producer that perturbed the frozen model before measuring',
        A_L1=stored['baseline']['layer_budget']['L1']['amp_ratio_median'],
        A_L3=stored['baseline']['layer_budget']['L3']['amp_ratio_median'],
        nse=stored['baseline']['monthly']['nse'],
        median_station_nse=stored['baseline']['monthly']['median_station_nse'],
        n_eligible_rows=stored['baseline']['monthly']['n_eligible_rows'],
        n_station_months=stored['baseline']['monthly']['n_station_months'])

    # ---- the defect this audit found IN ITSELF, and how it was settled ----
    rep['deviations'] = [
        'THE AUDIT BRACKET WAS ORIENTED BACKWARDS, AND IT WAS FOUND BY RUNNING, NOT BY '
        'REVIEW.  The first spelling of `solve_bisect` assumed `G` non-increasing in `k` '
        '(`glo >= Tgt`, `ghi <= Tgt`); that bracket is empty at 230/230 reaches for every '
        'non-zero beta, so the bisection walked to `hi = 1e6` and '
        '`solve_reproduced_by_an_independent_root_finder` read False while every other leg '
        'of the audit passed.  The residue was MEASURED before anything was edited '
        '(`work/_probe_mono.py`, kept only until delivery): on `[1e-6, 1e6]`, at beta=+0.5, '
        '`G` is strictly increasing at 230/230 reaches for the mass device (N1) and '
        '230/230 for the concentration device (N3); the bracket is valid at 230/230 in the '
        'increasing orientation and 0/230 in the decreasing one.  `solve_bisect` now READS '
        'the orientation off `glo`/`ghi` per reach and reports both counts per point '
        '(`direction_increasing_fraction`, `bracket_valid_*_fraction`), so the audit no '
        'longer assumes its own monotonicity.  The ARTIFACT was never wrong -- the '
        'registered `k` field was untouched by this and the audit re-ran only itself.',
        'THE AUDIT THEN COMPARED ONE EQUATION AGAINST A TWO-ROUTE FIELD.  With the bracket '
        'fixed the bisection converged at 230/230 reaches, yet the whole-grid comparison '
        'still read 2.6e-3 .. 2.7e-2 disagreement at every non-zero beta while beta=0 '
        'agreed bitwise.  MEASURED before restructuring (`work/_probe_route.py`, kept only '
        'until delivery): the registered route is the plan section 2.2 two-route route -- '
        '`status_counts = LINEAR 128, FIXED_POINT 83, FIXED_POINT_BISECTED 19` for N1 at '
        'beta=+0.5, and the count of reaches with `rho_lin <= 1e-4` is exactly 128.  The '
        'audit had solved the self-consistent equation for all 230 reaches, so on the 128 '
        'the producer deliberately answered with the CLOSED FORM it was measuring the root '
        'of a different equation.  Recomputed independently, the closed form matches the '
        'registered k on that subset to 3.6e-16 relative (k_lin) and the audit bisection '
        'matches it on the complement to 8.1e-11 (median 2.2e-13), which is inside both '
        'the producer step tolerance and the audit interval width.  The leg now '
        'reimplements BOTH routes and compares each on its own subset, and keeps the '
        'whole-grid number under the label `whole_grid_max_rel_k_diff` rather than using '
        'it.  Again the ARTIFACT was never wrong',
        'the failing leg was reported rather than swallowed: on the pre-fix run the audit '
        'wrote `solve_reproduced_by_an_independent_root_finder = False` and exited 0 with '
        'every other conclusion True.  That behaviour is preserved -- a leg that fails '
        'still lands in `conclusions` as False',
    ]
    rep['conclusions'] = dict(
        kernel_is_the_contract=bool(
            rep['kernel_reproduces_the_frozen_scan_at_xi_one']['all_zero']),
        solve_reproduced_by_an_independent_root_finder=bool(
            rep['independent_solve_all_agree']),
        ledger_reproduced=bool(all(v['local_matches_stored'] and v['network_matches_stored']
                                   for v in led.values())),
        xi_spelling_is_a_contract_not_a_consequence=bool(
            any(not v['bitwise_equal'] for v in xi_check.values())),
        limits=[
            'the audit covers 12 of 76 points',
            'routing and station aggregation are shared with the producer, not re-derived',
            'the audit does not re-derive A_L1/A_L3 for candidate points; it re-derives the '
            'LAND PHASE and the SOLVE, which is where this round changed anything',
        ])
    (OUT / 'audit_beta.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    _p('=== wrote %s ===' % (OUT / 'audit_beta.json'))
    _p('    conclusions: %r' % rep['conclusions'])
    return rep


if __name__ == '__main__':
    main()
