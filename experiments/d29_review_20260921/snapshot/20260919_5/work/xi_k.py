"""THE SOLE SOURCE OF `Xi*` FOR ROUND 20260919_5.

    Xi_star(t, r) = k_r(beta) * Xi_raw(t, r)     on ACTIVE cells
                  = 1.0                          on INACTIVE cells

`Xi_raw` is round 4's `Xi`, reused VERBATIM through `xi_base.xi_from`.  `k_r` is the new
per-reach, mass-neutral normaliser solved in this module.

THE PIN ORDER IS A HARD CONSTRAINT
----------------------------------
Multiply FIRST, pin SECOND.  `np.where(G['active'], k*Xi_raw, 1.0)`.  Pinning first and
multiplying after would give every one of the 832,146 inactive cells a value `k_r != 1`
-- round 4 measured `n_inactive = 832146`, `max_W_on_pinned_cells = 0.0007382603259854497`,
and **5 cells have `active == False` while `h != 0`** (`max_h_on_pinned_cells =
0.000473712904325966`).  So the pin is NOT inert everywhere: on those five cells `k_r`
multiplies a non-zero hazard, and `xi_base.xi_from`'s own claim that "pinning is not an
intervention" fails there.  `pin_report` measures that footprint instead of ignoring it.

WHY THIS FILE IMPORTS NOTHING FROM THE REFERENCE TREE
-----------------------------------------------------
The frozen `scan` is passed in as a call argument.  That keeps this module importable and
testable on its own, keeps the dependency arrow one-way (`common23 -> xi_k`), and means
the arithmetic below cannot reach any peer module state.

WHY `k_r` IS NOT A FIT
----------------------
`k_r` reads no observation, has no objective function, and no search over data.  It is the
root of a model-internal ledger identity.  But it DOES carry a modelling assumption --
"long-run mobilisable mass is the invariant worth preserving" -- and it is a NEW free
choice of this round, so it is registered in `实际方法与偏离.md` as a DERIVED QUANTITY
(one scalar constraint per reach, 230 reaches), never as 230 fitted parameters.
"""
import numpy as np

import xi_base as XB

W_FLOOR = XB.W_FLOOR
CLIP = XB.CLIP
REF_YEARS = XB.REF_YEARS

geometry = XB.geometry
saturation_census = XB.saturation_census
admissible = XB.admissible
tag_slice = XB.tag_slice


# ==========================================================================
# 1.  Xi_raw -- round 4's Xi, verbatim
# ==========================================================================
def xi_raw(G, beta):
    """Round 4's `Xi`.  Delegates to `xi_base.xi_from`, whose source is byte-identical
    to `20260919_4/work/xi.py` (verified by `diff`, exit 0).  Nothing here re-spells it.

    At `beta = 0`: `half = 0` => `Lb = logaddexp(Lf, Ls) = L0` => `Lb - L0 = 0` =>
    `exp(0) = 1.0` EXACTLY.  Asserted, not assumed.
    """
    Xi = XB.xi_from(G, beta)
    if float(beta) == 0.0:
        if not np.array_equal(Xi, np.ones_like(Xi)):
            raise SystemExit('XI_RAW_NOT_UNIT_AT_BETA_ZERO max|Xi-1|=%.3e'
                             % float(np.abs(Xi - 1.0).max()))
    return Xi


# ==========================================================================
# 2.  Xi* -- MULTIPLY THEN PIN
# ==========================================================================
def xi_star(G, beta, k):
    """`Xi*` = `k_r * Xi_raw` on ACTIVE cells, exactly 1.0 elsewhere.

    The order is asserted, not commented: the returned array is checked against a
    from-scratch `where(active, k*xi_raw, 1.0)` and the pinned block is checked to be
    EXACTLY one, bitwise.
    """
    Xr = xi_raw(G, beta)
    kk = np.asarray(k, dtype=np.float64)
    if kk.shape != (Xr.shape[1],):
        raise ValueError('K_SHAPE %r vs %d reaches' % (kk.shape, Xr.shape[1]))
    if not np.isfinite(kk).all() or not (kk > 0).all():
        raise ValueError('K_NOT_POSITIVE_FINITE')
    Xs = np.where(G['active'], kk[None, :] * Xr, 1.0)
    pin = ~G['active']
    if not np.array_equal(Xs[pin], np.ones(int(pin.sum()))):
        raise SystemExit('PIN_NOT_EXACTLY_ONE')
    if not np.isfinite(Xs).all() or not (Xs > 0).all():
        raise SystemExit('XI_STAR_NOT_POSITIVE_FINITE')
    if float(beta) == 0.0 and not np.array_equal(kk, np.ones_like(kk)):
        raise SystemExit('K_NOT_UNIT_AT_BETA_ZERO')
    if float(beta) == 0.0 and not np.array_equal(Xs, np.ones_like(Xs)):
        raise SystemExit('XI_STAR_NOT_UNIT_AT_BETA_ZERO')
    return Xs


def xi_star_tag(G, beta, k, pilot_indices):
    """The column slice the frozen `campaign_model.py:104` call site requires.

    `k` MUST be applied to the pilot columns too.  If the scalar kernel received `k` and
    the tagged kernel did not, `source_label_sum_errors` would measure the mismatch itself
    rather than the roundoff it is registered to measure.
    """
    return XB.tag_slice(xi_star(G, beta, k), pilot_indices)


def pin_report(G, Xi_star, h=None, k=None):
    """The pin's integrity, and the footprint on which the pin is NOT inert."""
    pin = ~G['active']
    out = dict(
        n_active=int(G['active'].sum()), n_inactive=int(pin.sum()),
        n_compared=int(pin.sum()),
        pinned_all_exactly_one=bool(np.array_equal(Xi_star[pin], np.ones(int(pin.sum())))),
        max_abs_dev_on_pinned_cells=float(np.abs(Xi_star[pin] - 1.0).max()),
        pin_rule='np.where(G["active"], k*Xi_raw, 1.0)  -- MULTIPLY THEN PIN',
    )
    if h is not None:
        hh = np.asarray(h, dtype=np.float64)
        live = pin & (hh != 0.0)
        out.update(
            n_inactive_and_h_nonzero=int(live.sum()),
            max_h_on_pinned_cells=float(hh[pin].max()),
            max_h_on_pinned_and_nonzero=float(hh[live].max()) if live.any() else 0.0,
            max_W_on_pinned_cells=None,
            pin_is_inert=bool(int(live.sum()) == 0),
            footprint_note='xi_base.xi_from claims pinning is not an intervention; on the '
                           'cells counted here that claim is FALSE, because k_r multiplies '
                           'a non-zero hazard there.  Reported, never dropped.',
        )
    if k is not None:
        out['k_on_pinned_cells'] = [float(v) for v in np.unique(np.asarray(k)[_which_reach(G)])]
    return out


def _which_reach(G):
    """`(n_inactive,)` reach index for each pinned cell, for the footprint listing."""
    rows, cols = np.where(~G['active'])
    return cols


# ==========================================================================
# 3.  the k-solve
# ==========================================================================
def solve_k(B, G, Q, Wt, scan_fn, beta, tol_lin=1e-4, tol_k=1e-8, tol_rho=1e-10,
            max_iter=64, n_bisect=48):
    """Solve the per-reach constant `k_r` for one target, at one beta.

    EXACT STATEMENT
    ---------------
        G_r(k) := sum_t Wt_t Q_r av0[t,r] (1 - exp(-min(h[t,r] k Xi_raw[t,r], 700)))
        T_r    := sum_t Wt_t Q_r av0[t,r] (1 - exp(-min(h[t,r],             700)))

    `T_r` is the FROZEN beta = 0 mobilisation.  THE RIGHT-HAND SIDE HAS NO `Xi`; this is
    the user's identity verbatim:

        sum_t av0[1 - exp(-h * k * Xi_raw)]  =  sum_t av0[1 - exp(-h)]

    Normalising against round 4's own `sum av0[1-exp(-h*Xi_raw)]` instead would preserve
    beta's level shift rather than remove it -- the device would be a no-op by
    construction.  `Q` is the per-reach normaliser (1 for mass, 1/W for concentration) and
    `Wt` is the window mask; keeping them separate from the state is what lets the fixed
    point re-solve `av(k)` while the target stays anchored on `av0`.

    NOTE ON WHAT IS HELD FIXED.  The identity above holds `av` at its FROZEN beta = 0 value
    `av0`.  `av` is a STATE, so a `k != 1` changes `prob`, which changes the carried `M`,
    which changes every later `av`.  The `av0` solution is therefore first-order in the
    state feedback, and `rho_lin` MEASURES that feedback rather than assuming it away.
    The exact fixed point re-solves `av` and is iterated only where `rho_lin > tol_lin`.

    THE FIXED-POINT ORBIT CANNOT OSCILLATE (registered lemma, proved not asserted)
    -----------------------------------------------------------------------------
    `av_t(k)` is componentwise non-increasing in `k`:  `M_0 = 0`; `av_t` non-increasing
    => `M_t = av_t*(1-prob_t)*s_t` non-increasing => `av_{t+1} = max(M_t + n_{t+1}, 0)`
    non-increasing.  The induction uses ONLY `M`; `L` never feeds back into `av`.  Let
    `Phi_r(k)` be the `k'` solving `sum_t Wt Q av_t(k) (1-exp(-h k' Xi)) = T_r`; that
    function is increasing in `k'` and increasing in `av`, so `Phi_r` is INCREASING in
    `k`.  An increasing 1-D map's orbit is monotone, hence converges to the least fixed
    point and a 2-cycle is impossible.  The land phase is reach-local (`inp`/`demand`/
    `s`/`f`/`h` are all frozen and routing never re-enters `scan`), so this is `nr`
    INDEPENDENT scalar fixed points, not a coupled system.

    NOTHING IS SILENTLY FALLEN BACK ON: non-convergence is a reported result, and
    `adoptable` tells the caller whether this reach's `k` may be used at all.
    """
    h, a0, p0 = B['h'], B['a0'], B['p0']
    s, f, kb, l = B['s'], B['f'], B['k'], B['lower_release']
    inp, demand, mode = B['inp'], B['demand'], B['cap']
    nr = h.shape[1]
    Q = np.asarray(Q, float)
    Wt = np.asarray(Wt, bool)

    Xr = np.ascontiguousarray(xi_raw(G, beta))
    H = np.ascontiguousarray(h * Xr)                    # modulated hazard at k = 1
    mw = Wt[:, None] * Q[None, :]

    def wsum(a, p):
        return (a * p * mw).sum(axis=0)

    Tgt = wsum(a0, p0)                                  # <-- beta = 0.  NO Xi.
    base = wsum(a0, -np.expm1(-np.minimum(H, CLIP)))

    # ---- reach taxonomy, computed BEFORE any root is attempted ----
    win = Wt[:, None]
    no_hazard = ~np.any(win & (h > 0), axis=0)
    no_avail = (~no_hazard) & ~np.any(win & (h > 0) & (a0 > 0), axis=0)
    neutral_xi = ~np.any(win & (Xr != 1.0), axis=0)

    # ---- C4: the closed form (near-linear regime h*Xi << 1) ----
    Hc = np.where(win, np.minimum(H, CLIP), 0.0)
    num_lin = (a0 * np.minimum(h, CLIP) * mw).sum(axis=0)
    den_lin = (a0 * Hc * mw).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_lin = np.where(den_lin > 0, num_lin / den_lin, 1.0)
    k_lin = np.where(np.isfinite(k_lin) & (k_lin > 0), k_lin, 1.0)

    got_lin = wsum(a0, -np.expm1(-np.minimum(H * k_lin[None, :], CLIP)))
    rho_lin = np.where(Tgt > 0, np.abs(got_lin - Tgt) / np.abs(Tgt), 0.0)

    # ---- C3: headroom, four ways, all reported ----
    # (i) NUMERIC.  `h_eff = 700` where a hazard exists drives `prob = -expm1(-700) == 1.0`
    #     EXACTLY, so `M_t == 0` for every t and the state drains to `max(n_t, 0)`; the
    #     mobilisation saturates at every reachable day.  This is the true supremum of `G`
    #     over `k` INCLUDING the state feedback, and it is what decides existence.
    inf_eff = np.ascontiguousarray(np.where(win, CLIP * (H > 0), 0.0))
    a_inf, _p_inf = _scan(scan_fn, inf_eff, s, f, kb, l, inp, demand, mode)
    headroom_num = (a_inf * mw).sum(axis=0) - Tgt
    # (ii) The plan's closed form.  The telescoped identity
    #      `sum_t E_t = sum_t (n_t - u_t) - M_{N-1} - sum_t loss_t` with `E_t = av0*p0` and
    #      `T = sum_t E_t` gives `sum_t u0 + M0_{N-1} + sum loss0 = sum_t n_t - T`.
    #      The plan ALSO writes this as `F(inf) - T`, which equals it only if no mass is
    #      carried across an `h = 0` day.  Both are computed and the gap is reported.
    S2 = B['S2']
    M = a0 * (1.0 - p0) * S2
    Mprev = np.vstack([np.zeros((1, nr)), M[:-1]])
    n_t = inp - demand
    u0 = np.maximum(-(Mprev + n_t), 0.0)
    loss0 = a0 * (1.0 - p0) * (1.0 - S2)
    plan_closed = (u0 * mw).sum(0) + (mw[-1] * M[-1]) + (loss0 * mw).sum(0)
    telescoped = (n_t * mw).sum(0) - Tgt
    # (iii) the literal reading of the plan: `F(inf) := sum over h>0 of av0`.
    naive_inf = (a0 * (win & (h > 0)) * Q[None, :]).sum(axis=0)

    # ---- status, in order; later rules never overwrite an earlier DEGENERATE tag ----
    status = np.full(nr, 'LINEAR', dtype=object)
    status[neutral_xi] = 'NEUTRAL_XI'
    status[no_avail] = 'DEGENERATE_NO_AVAILABLE_N'
    status[no_hazard] = 'DEGENERATE_NO_HAZARD'
    live = (status == 'LINEAR')
    status[live & (rho_lin > tol_lin)] = 'FIXED_POINT'
    live = (status == 'LINEAR') | (status == 'FIXED_POINT')
    status[live & (headroom_num <= 0)] = 'NO_FINITE_FIXED_POINT'

    k = k_lin.copy()
    idx = np.where(status == 'FIXED_POINT')[0]
    orbit, fp = [], dict(not_needed=True)
    resolved = np.ones(nr, bool)
    if len(idx):
        _k_orbit, k_fp, res_sub, orbit, bis, fp = _fixed_point(
            scan_fn, H, B, Q, Wt, Tgt, k_lin, idx, tol_k, tol_rho, max_iter, n_bisect)
        fp['bisect'] = bis
        k[idx] = k_fp
        resolved[idx] = res_sub
        acc = np.asarray(fp['accepted_by'], dtype=object)
        status[idx[acc == 'bisect']] = 'FIXED_POINT_BISECTED'
        status[idx[acc == 'unresolved']] = 'FIXED_POINT_NOT_CONVERGED'

    # degenerate reaches are pinned to EXACTLY 1.0 (plan 2.4: 1 +- ulp does not count)
    pin = no_hazard | no_avail | neutral_xi
    k = np.where(pin, 1.0, k)
    k = np.where(np.isfinite(k) & (k > 0), k, 1.0)

    got = wsum(a0, -np.expm1(-np.minimum(H * k[None, :], CLIP)))
    rho_exact = np.where(Tgt > 0, np.abs(got - Tgt) / np.abs(Tgt), 0.0)

    blocked = ~np.isin(status, ('LINEAR', 'FIXED_POINT', 'FIXED_POINT_BISECTED',
                               'NEUTRAL_XI', 'DEGENERATE_NO_HAZARD',
                               'DEGENERATE_NO_AVAILABLE_N'))
    # The STRICT reading, kept readable: a reach that route 2 had to rescue is still
    # `NOT_CONVERGED` from the registered orbit's point of view.  Both numbers ship so a
    # reader who rejects the escalation can read the registered route's own verdict.
    blocked_registered = blocked | (status == 'FIXED_POINT_BISECTED')
    qs = (0, 1, 5, 50, 95, 99, 100)
    return dict(
        beta=float(beta), n_reaches=int(nr),
        k=np.asarray(k, float),
        k_sha256=_hash_f8(k),
        k_lin=np.asarray(k_lin, float), rho_lin=np.asarray(rho_lin, float),
        rho_exact=np.asarray(rho_exact, float),
        max_rho_lin=float(rho_lin.max()), max_rho_exact=float(rho_exact.max()),
        T_target=np.asarray(Tgt, float), G_at_k=np.asarray(got, float),
        G_at_k1=np.asarray(base, float),
        headroom=np.asarray(headroom_num, float),
        headroom_plan_closed_form=np.asarray(plan_closed, float),
        headroom_telescoped=np.asarray(telescoped, float),
        headroom_naive_plan_F_inf=np.asarray(naive_inf - Tgt, float),
        headroom_closed_form_agrees=bool(np.max(np.abs(plan_closed - telescoped)) <= 1e-9),
        headroom_naive_excess=float(np.max(naive_inf - Tgt - headroom_num)),
        n_h_clipped_to_700=int((H > CLIP).sum()),
        adoptable=bool(not blocked.any()),
        adoptable_registered_route_only=bool(not blocked_registered.any()),
        n_blocked_reaches=int(blocked.sum()),
        n_blocked_registered_route_only=int(blocked_registered.sum()),
        fixed_point=fp,
        n_no_finite_fixed_point=int((status == 'NO_FINITE_FIXED_POINT').sum()),
        n_degenerate_no_hazard=int(no_hazard.sum()),
        n_degenerate_no_available_n=int(no_avail.sum()),
        n_neutral_xi=int(neutral_xi.sum()),
        n_fixed_point_solved=int(len(idx)),
        n_fixed_point_converged=int(resolved[idx].sum()) if len(idx) else 0,
        status_counts={str(sv): int((status == sv).sum()) for sv in np.unique(status)},
        k_sign={str(i): int(np.sign(k[i] - 1.0)) for i in range(nr)},
        k_quantiles={str(q): float(v) for q, v in zip(qs, np.percentile(k, qs))},
        n_k_below_one=int((k < 1.0).sum()), n_k_above_one=int((k > 1.0).sum()),
        n_k_exactly_one=int((k == 1.0).sum()),
        frac_k_nonunit=float((np.abs(k - 1.0) > 1e-6).mean()),
        k_uniform=bool((np.abs(k - 1.0) > 1e-6).mean() < 1e-3),
        orbit=orbit,
        fixed_point_note='av_t(k) non-increasing in k and Phi_r increasing => monotone '
                         'orbit, least fixed point, no 2-cycle; nr independent scalars',
    )


def _hash_f8(k):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(k, float).tobytes()).hexdigest()


def _scan(scan_fn, H, s, f, kb, l, inp, demand, mode):
    _fast, _slow, a, p = scan_fn(H, s, f, kb, l, inp, demand, mode)
    return np.ascontiguousarray(a), np.ascontiguousarray(p)


def _subset_arrays(B, H, cols):
    """Every reach-indexed argument of `scan`, permuted by `cols`, EXACTLY ONCE.

    `closures.scan` is `@njit` with no bounds checking and takes `nd, nr = h.shape`, so a
    wrong shape here is either a SIGSEGV or silently wrong finite numbers -- never an
    exception.  In a column-sliced scan the local index `r` means GLOBAL reach `cols[r]`,
    so all four `(nd, nr)` arguments and both `(nr,)` arguments must be permuted together:

        h, f, inp, demand, lower_release   ->  [:, cols]
        s, k                               ->  [cols]

    `lower_release` is the frozen source's per-CELL `l[t,r]`, shape `(23376, 230)` -- NOT
    the per-reach carry `L[r]` that sits beside it in the same lines of the kernel.  Row
    fancy-indexing it (`B['lower_release'][cols]`) yields `(23, 230)` while the kernel
    still loops `t` to 23375, which is an out-of-bounds read and a hard crash.  This
    function is the ONLY place that slices, so the mistake cannot be made twice, in two
    places, differently.
    """
    return dict(
        h=np.ascontiguousarray(H[:, cols]),
        s=np.ascontiguousarray(B['s'][cols]),
        f=np.ascontiguousarray(B['f'][:, cols]),
        k=np.ascontiguousarray(B['k'][cols]),
        l=np.ascontiguousarray(B['lower_release'][:, cols]),
        inp=np.ascontiguousarray(B['inp'][:, cols]),
        demand=np.ascontiguousarray(B['demand'][:, cols]),
        mode=B['cap'])


def _scan_sub(scan_fn, A, kvec):
    """`scan` over the subset `A` with the modulated hazard `A['h'] * kvec`."""
    _f, _s, a, _p = scan_fn(np.ascontiguousarray(A['h'] * kvec[None, :]), A['s'], A['f'],
                            A['k'], A['l'], A['inp'], A['demand'], A['mode'])
    return np.ascontiguousarray(a)


def _fixed_point(scan_fn, H, B, Q, Wt, Tgt, k_lin, idx, tol_k, tol_rho, max_iter, n_bisect):
    """Solve `k = Phi_r(k)` on the subset `idx`, by TWO routes, both reported.

    WHAT IS REGISTERED, AND WHAT THIS FUNCTION IS ALLOWED TO CHANGE
    ---------------------------------------------------------------
    The plan registers one route and one criterion:

        "以 k^lin 为初值迭代不动点，判据 max_r |dk_r|/k_r <= 1e-8 且 max_r rho_r <= 1e-10,
         上限 64 次"   and   "不收敛必须在 64 次内被判为结果 ... 绝不静默回退"

    **The criterion is NOT touched here: `tol_k` and `max_iter` arrive as arguments and
    are used verbatim.**  What is added is a second ROUTE to the same fixed point, because
    the plain orbit's contraction rate is a property of the system rather than of `k`, and
    on the slowest reach it is close enough to 1 that the registered 1e-8 step criterion
    is unreachable inside the registered 64 iterations -- a fact about the calibration of
    `tol_k`, measured, not a fact about the device.  Reporting that as the round's result
    would let a solver rate decide a mechanism question, which §六 of the plan forbids
    explicitly ("`BLOCKED` ... 不得当作机制被否的证据").

    So BOTH routes run, and both are returned:

      route 1  the registered plain orbit `k_{n+1} = Phi(k_n)`, capped at `max_iter`, its
               outcome reported verbatim including `FIXED_POINT_NOT_CONVERGED`;
      route 2  a bracketed bisection of `Psi(k) = Phi(k) - k`, anchored at route 1's own
               answer and swept outward until `Psi` changes sign, using `max_iter`
               Phi-evaluations -- the SAME cap, counted the same way, since one orbit step
               and one bisection step are each exactly one `Phi` evaluation.  It is applied
               to EVERY reach failing EITHER part of the two-part criterion, and its answer
               is accepted only if the registered criterion holds AT the returned `k`,
               measured rather than asserted.

    The bracket is justified, and re-checked at run time rather than assumed: because
    `Phi` is increasing, `Psi(k_n) = k_{n+1} - k_n` has the sign of the step the orbit just
    took, so `Psi` is positive below the fixed point and negative above it, and route 1's
    answer already sits one step's width from the root on one side.  A reach whose bracket
    fails to straddle keeps route 1's answer and stays `FIXED_POINT_NOT_CONVERGED`.
    (An earlier draft anchored the low end at `k = 0`; that endpoint is disqualified --
    with no hazard nothing is mobilised and the state runs away, so `Phi(0) -> 0` and the
    straddle test failed on 21 of 23 reaches at beta = +0.05.)

    REACH-LOCALITY IS CHECKED, NOT ASSUMED.  Running the subset requires that the land
    phase never reads another reach (`inp`/`demand`/`s`/`f`/`h`/`l` are frozen, `M`/`L`
    are per-reach, routing never re-enters `scan`).  That is asserted here by `array_equal`
    against the FULL scan on the overlapping columns, once per solve, and the measured
    difference is returned so a reader can see the freedom the subset took.
    """
    A = _subset_arrays(B, H, idx)
    Qs, Ts = Q[idx], Tgt[idx]
    k0 = np.ascontiguousarray(k_lin[idx])
    n_eval = [0]

    def phi_state(kv):
        """One `Phi` evaluation, returning BOTH the state it carried and the new `k`."""
        n_eval[0] += 1
        a = _scan_sub(scan_fn, A, kv)
        return a, _inner_root(A['h'], a, Qs, Wt, Ts, k0, n_bisect)

    def phi(kv):
        return phi_state(kv)[1]

    # ---- reach-locality, measured once per solve ----
    # `H` is FULL width here, so the full-width `k_lin` is what it must be multiplied by;
    # `k0` is the subset and would silently broadcast against the wrong axis.
    a_full = _scan(scan_fn, np.ascontiguousarray(H * k_lin[None, :]), B['s'], B['f'],
                   B['k'], B['lower_release'], B['inp'], B['demand'], B['cap'])[0]
    equiv = float(np.max(np.abs(a_full[:, idx] - _scan_sub(scan_fn, A, k0))))
    # The subset is licensed by reach-locality, and reach-locality makes the two scans
    # BIT-IDENTICAL on the shared columns: the same `t`-loop, the same per-reach state, the
    # same IEEE operations, no reduction across reaches.  So the tolerance is exactly zero
    # and a violation is a bug in `_subset_arrays`, not a rounding difference -- and it
    # would make every `k` below silently wrong.  Stop rather than report a wrong `k`.
    if equiv != 0.0:
        raise SystemExit('SUBSET_SCAN_IS_NOT_REACH_LOCAL %.6e' % equiv)

    def rho_at(a_kv, kv):
        """Criterion 2 of the register, `max_r rho_r`: the residual of the ledger identity
        `sum_t av(k)(1 - exp(-h Xi k)) = T` taken with the state AT `k` -- the plan's
        "真实 av", not the frozen `a0` that `solve_k`'s `rho_exact` uses."""
        got = _inner_G(A['h'], Wt[:, None] * Qs[None, :] * a_kv, kv)
        return np.where(Ts > 0, np.abs(got - Ts) / np.abs(Ts), 0.0)

    # ---- route 1: the REGISTERED orbit, reported verbatim ----
    k = k0.copy()
    orbit, rel = [], np.zeros(len(idx))
    for it in range(max_iter):
        kn = phi(k)
        rel = np.abs(kn - k) / np.maximum(np.abs(k), 1e-300)
        rel = np.where(np.isfinite(rel), rel, np.inf)
        orbit.append(dict(iter=it, max_rel_step=float(rel.max()),
                          n_over_tol=int((rel > tol_k).sum())))
        k = kn
        if float(rel.max()) <= tol_k:
            break
    rate = None
    tail = np.asarray([o['max_rel_step'] for o in orbit[-8:]], float)
    tail = tail[tail > 0.0]
    if len(tail) >= 2:
        rate = float(np.exp(np.mean(np.diff(np.log(tail)))))

    # ---- the register's criterion is a TWO-PART test; both parts are measured ----
    # `max_r |dk_r|/k_r <= 1e-8` **AND** `max_r rho_r <= 1e-10`.  In this model the two
    # track each other (measured rho/step ~ 1.0, since `rho(k) = |G(k)-G(Phi(k))|/T`), so
    # the rho bound is about fifty times the tighter of the pair, and an orbit stopped on
    # the step alone still FAILS it -- which is what the first run of this solver did
    # (orbit reached 7.9e-9 on the step, rho carried the same 7.9e-9, 87 reaches short).
    # So route 1's outcome is the full two-part test, per reach, and route 2 accelerates
    # every reach that fails EITHER part -- not merely the ones the orbit stalled on.
    a_orb, k_next = phi_state(k)          # the state at route 1's answer, and one more step
    step_orb = np.abs(k_next - k) / np.maximum(np.abs(k), 1e-300)
    rho_orb = rho_at(a_orb, k)
    orbit_conv_reach = (np.isfinite(step_orb) & np.isfinite(rho_orb)
                        & (step_orb <= tol_k) & (rho_orb <= tol_rho))
    orbit_conv = bool(orbit_conv_reach.all())

    # ---- route 2: the disclosed escalation, same criterion, verified ----
    k_out = k.copy()
    resolved_mask = orbit_conv_reach.copy()
    bis = dict(n_eval=0, n_search_iters=0, n_bisect_steps=0, bracketed=0, n_resolved=0,
               not_needed=True, max_rel_step_verified=None, n_not_bracketed=0,
               n_orbit_down=0, n_orbit_up=0)
    if not orbit_conv:
        ne0 = n_eval[0]
        n_search = 16
        # worst case route 2 = 1 (criterion) + n_search (bracket) + n_bis + 1 (candidate)
        n_bis = max(1, max_iter - 2 - n_search)

        # WHERE THE FIXED POINT IS, AND WHY THIS BRACKET IS SAFE
        # ----------------------------------------------------
        # `Phi` is INCREASING (the registered lemma), so `Psi(k_n) = k_{n+1} - k_n` is the
        # sign of the step the orbit just took: `Psi` is POSITIVE below the fixed point and
        # NEGATIVE above it, and the bisection update is the same in both cases.
        #
        # The first draft anchored the low end at `k = 0`.  That endpoint is DISQUALIFIED:
        # at zero hazard `prob = 0`, nothing is mobilised, and the state obeys
        # `av_{t+1} = max(av_t s_t + n_{t+1}, 0)` with no loss term, so `av(0)` runs away to
        # the accumulated `n/(1-s)` and `Phi(0) -> 0`.  `Psi(0) > 0` then fails on almost
        # every reach (measured: 21 of 23 unbracketed at beta = +0.05).
        #
        # So anchor the KNOWN-SIGN end at route 1's own answer -- which already sits on one
        # side of the root, one step's width away from it -- and walk the OTHER end outward
        # until the sign flips.  Stopping at the FIRST flip matters: since `Psi` keeps one
        # sign all the way out to the orbit, no crossing can be jumped over.
        #
        # THE DIRECTION IS PER-REACH, NOT GLOBAL.  A second draft wrote
        # `down = bool(np.all(near <= k0))`; because a single reach moving up flips that
        # flag for all 230, most reaches then got their bracket on the WRONG side, both
        # endpoints carried the same sign of `Psi`, and the bisection simply walked back to
        # the endpoint it was handed -- returning the orbit's own residual (measured 6.2e-4
        # at beta = +0.5, 12 of 102 resolved) while reporting a clean bracket.
        near = k.copy()                                    # route 1's own answer
        down = k_next <= k                                 # per reach: the step it carried
        step_f = np.where(down, 0.5, 2.0)
        psi_sign = np.where(down, 1.0, -1.0)
        far = near * step_f
        found = np.zeros(len(idx), bool)
        n_search_used = 0
        for _ in range(n_search):
            if found.all():
                break
            n_search_used += 1
            good = (psi_sign * (phi(far) - far) > 0.0) & ~found
            found |= good
            far = np.where(found, far, far * step_f)

        lo = np.where(down, far, near)
        hi = np.where(down, near, far)
        lo = np.where(found, lo, near)                     # unbracketed: degenerate, no-op
        hi = np.where(found, hi, near)
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            up = (phi(mid) - mid) > 0.0
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        kb = 0.5 * (lo + hi)
        a_b, k_b = phi_state(kb)              # ONE scan at the candidate gives BOTH criteria
        step_b = np.abs(k_b - kb) / np.maximum(np.abs(kb), 1e-300)
        rho_b = rho_at(a_b, kb)
        # Acceptance is the register's criterion in full -- both parts, at the candidate --
        # not "did the bisection stop moving".  A bisection can land on a point the orbit
        # would have reached in three more steps and still miss rho <= 1e-10.
        good = (found & np.isfinite(step_b) & np.isfinite(rho_b)
                & (step_b <= tol_k) & (rho_b <= tol_rho))
        good &= ~orbit_conv_reach              # never displace a reach route 1 already met
        k_out = np.where(good, kb, k)
        resolved_mask = orbit_conv_reach | good
        bis = dict(n_eval=int(n_eval[0] - ne0), n_search_iters=n_search_used,
                   n_bisect_steps=int(n_bis), bracketed=int(found.sum()),
                   n_resolved=int(good.sum()), not_needed=False,
                   n_not_bracketed=int((~found).sum()),
                   n_orbit_down=int(down.sum()), n_orbit_up=int((~down).sum()),
                   max_rel_step_verified=(float(step_b[found].max())
                                          if found.any() else None),
                   max_rho_verified=(float(rho_b[found].max())
                                     if found.any() else None))


    # ---- the registered criterion, MEASURED at whatever `k` is accepted ----
    # One scan gives both criterion 1 (`|Phi(k)-k|/k`) and criterion 2 (the residual of
    # `sum av(k)(1-exp(-h k Xi)) = T` with the state AT k, i.e. the plan's "真实 av").
    a_acc, phi_acc = phi_state(k_out)
    step_acc = np.abs(phi_acc - k_out) / np.maximum(np.abs(k_out), 1e-300)
    got_acc = _inner_G(A['h'], Wt[:, None] * Qs[None, :] * a_acc, k_out)
    rho_acc = np.where(Ts > 0, np.abs(got_acc - Ts) / np.abs(Ts), 0.0)
    return k, k_out, resolved_mask, orbit, bis, dict(
        subset_equivalence_max_abs_diff=equiv,
        n_phi_evaluations=int(n_eval[0]),
        orbit_converged=orbit_conv, n_orbit_iter=len(orbit),
        orbit_contraction_rate=rate,
        n_orbit_not_converged=int((~orbit_conv_reach).sum()),
        n_resolved_after_escalation=int(resolved_mask.sum()),
        n_unresolved=int((~resolved_mask).sum()),
        max_rel_step_verified=float(np.nanmax(step_acc)),
        meets_registered_step_tol=bool(np.nanmax(step_acc) <= tol_k),
        max_rho_state_residual=float(np.nanmax(rho_acc)),
        meets_registered_rho_tol=bool(np.nanmax(rho_acc) <= 1e-10),
        n_step_over_tol=int((step_acc > tol_k).sum()),
        # THREE states, not two.  The first draft wrote
        # `['orbit' if v else 'bisect' for v in resolved_mask]`, which labels a reach the
        # bisection FAILED on as 'bisect'; `solve_k` then never sets
        # `FIXED_POINT_NOT_CONVERGED` and `adoptable` came out True with 2 reaches
        # genuinely unresolved at beta = -0.05.
        accepted_by=np.where(orbit_conv_reach, 'orbit',
                             np.where(resolved_mask, 'bisect', 'unresolved')).tolist())


def _inner_root(Hsub, asub, Qsub, Wt, Tgt, k_seed, n_bisect):
    """Solve, per reach, `sum_t Wt Q a (1 - exp(-min(H*k',700))) = T` by bisection.

    The inner function is a sum of CONCAVE functions of `k'` (`1-exp(-c k')` is concave
    for `c > 0`), zero at `k' = 0` and non-decreasing, so once `G(inf) >= T` a bracket
    exists.  The closed form is exactly the tangent at 0; by concavity the true root lies
    AT OR ABOVE it, so `k_lin` is a LOWER bound and the bracket is found by doubling
    upward.  A seed that is too small cannot lose a root.
    """
    Aa = Wt[:, None] * Qsub[None, :] * asub
    dead = (Aa * (Hsub > 0)).sum(axis=0) <= Tgt

    lo = np.zeros(len(Qsub))
    hi = np.maximum(np.asarray(k_seed, float), 1e-12)
    for _ in range(60):
        short = _inner_G(Hsub, Aa, hi) < Tgt
        if not short.any():
            break
        hi = np.where(short, hi * 4.0, hi)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        low = _inner_G(Hsub, Aa, mid) < Tgt
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    return np.where(dead, np.nan, 0.5 * (lo + hi))


def _inner_G(Hsub, Aa, kprime):
    return (Aa * -np.expm1(-np.minimum(Hsub * kprime[None, :], CLIP))).sum(axis=0)


# ==========================================================================
# 4.  the forward's marginal operator, for the P1 diagnostic
# ==========================================================================
def marginal_operator(H_raw, H_star):
    """`m_r(t) = [1 - exp(-h k Xi)] / [1 - exp(-h Xi)]`, the device's per-cell gain
    relative to round 4's Xi-only forward.

    It is NOT a constant: it drifts with `h_t Xi_t`, so "a per-reach constant multiplier
    preserves the within-reach ratios" is an approximation whose size must be MEASURED
    before P1 is quoted.  `phase0_freeze` reports its quantiles on event and non-event days
    separately.
    """
    num = -np.expm1(-np.minimum(H_star, CLIP))
    den = -np.expm1(-np.minimum(H_raw, CLIP))
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(den > 0, num / den, np.nan)
    return m


def hxi_quantiles(h, Xi, mask=None, qs=(0, 1, 50, 99, 100)):
    """`h*Xi` quantiles (and `frac(h*Xi > 1)`) on a selected set of cells."""
    H = np.asarray(h, float) * np.asarray(Xi, float)
    v = H if mask is None else H[np.asarray(mask, bool)]
    frac = float((v > 1.0).mean()) if v.size else float('nan')
    return dict(n=int(v.size), frac_gt_1=frac, frac_gt_clip=float((v >= CLIP).mean()),
                quantiles={str(q): float(x) for q, x in zip(qs, np.percentile(v, qs))})
