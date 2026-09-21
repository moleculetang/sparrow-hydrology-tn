"""The ONE source of `Xi`, the pre-mobilisation mobile-water concentration modulator.

WHAT THIS FILE IMPLEMENTS
-------------------------
    Qf   = fast_water                                  (m^3/day, LOCAL, per cell)
    Qs   = percolation * area_ha * 10                  (m^3/day, LOCAL, per cell)
    W    = Qf + Qs
    Weff = max(W, W_FLOOR)
    clim = mean over ACTIVE cells in 1961-2020 of log(Weff)          (per reach)
    u    = log(Weff) - clim
    Xi   = exp( clip( logaddexp(Lf + b*u/2, Ls - b*u/2)
                    - logaddexp(Lf, Ls) , -700, +700 ) )             (b := beta)

`Xi` multiplies the land-phase HAZARD INSIDE the exponential, so
`prob = -expm1(-min(h*Xi, 700))`: it changes HOW MUCH N the day can mobilise,
upstream of the fast/slow split `f`, which is left untouched.

WHY THE FLOOR (a registered free choice, measured before it was made)
---------------------------------------------------------------------
`fast_water` bottoms out at 4.184469060922251e-200 and `percolation` at
1.1282505570800604e-204, both strictly positive, so `log` is finite everywhere and
no epsilon is needed for FINITENESS.  But `W < 1e-6` on 15.477431% of cells
(832,141 of 5,376,480) with a minimum of 1.1123379224548185e-199, i.e. the driver
is not usably non-zero over a sixth of the record: `log W` reaches -458.108 and
`u` would reach about -470 there.  A modulator that swings by exp(470) on cells
that carry no water is a numerical artefact, not a concentration.  Hence
`Weff = max(W, 1e-3)` and `Xi := 1` on inactive cells.  The plateau is flat
(active fraction 0.845225 / 0.845153 / 0.844914 at floors 1e-3 / 1e-1 / 1), so the
floor's VALUE is not load-bearing -- it is reported to show that.

THE INACTIVE MASK IS NOT AN AD HOC CUT
--------------------------------------
Measured on the frozen FULL24 domain, all three of these are the SAME 832,141 cells:

    fast_fraction == 0            (the frozen carrier fraction)
    W < 1e-6                      (the mobile water)
    h == 0                        (the hazard, gated by the `self.positive`
                                   mask = `contact > 0`, BUILT at model.py:61
                                   and APPLIED at
                                   vendor/expert/tn_challenge/model.py:74)

so on every cell where `Xi` is pinned to 1 the hazard is already exactly 0 and
`risk*Xi = 0.0` regardless.  The pin is therefore not an intervention on those
cells; it only fixes what gets REPORTED for them.

WHY THE LOG FORM, AND WHY THE SPELLING IS FROZEN
------------------------------------------------
Red-team measurement: `W/W` is exactly 1 everywhere, but `W*(1/W)` differs from 1
by >=1 ulp on 13.9% of cells.  So "bitwise no-op at beta = 0" is a property of the
SPELLING, not of the mathematics.  The spelling above is the one that is asserted,
and it is exact for this reason at `b == 0.0`:

    b*u/2      is +-0.0            (u is finite; 0.0*finite never yields NaN)
    Lf +- 0.0  is Lf               exactly
    Lb, L0     are given bit-identical inputs, so Lb - L0 == 0.0 exactly
    exp(0.0)   is 1.0              exactly
    clip(0.0)  is 0.0              exactly

Equivalent rewritings (`np.reciprocal`, `W**-1`, `exp(b)**0.5`, `sqrt(exp(b))`)
destroy that and MUST NOT be used.  Assertions live in `common22.assert_bindings`
and `phase0_freeze`.

WHY THE CLIP IS ON THE EXPONENT AND NOT ON `Xi`
-----------------------------------------------
Red-team measurement, without the clip: at beta = 4 there are 34 cells with
`Xi = +inf` and ALL 34 coincide with `h == 0`, so `risk*Xi = 0.0*inf = NaN`; the
NaN then propagates through `prob` and `E` into the kernel's CARRIED state `M[r]`
and poisons every remaining day of that reach.  `clip(.,-700,700)` keeps
`Xi` in [9.86e-305, 1.014e304]: finite AND strictly positive, so `0*Xi == 0.0`
exactly.  This does NOT replace the product-level cap `min(risk*Xi, 700)`, which
`closures.scan`'s counterpart keeps verbatim.  Clipping `Xi` itself instead would
be a different (and wrong) operator, because the cap in the frozen kernel is on the
PRODUCT.

REFERENCE WINDOW
----------------
`clim` is a 1961-2020 per-reach mean.  EVALUATION is 2021-2024, so the climatology
is strictly pre-evaluation, exactly as in `20260919_3/work/common21.py::build_z`.

UNITS
-----
`W` here is UNSCALED.  `closures.py:143` builds a `carrier` that is this same bundle
divided by 1000.  A multiplicative rescaling cancels in `u = log(Weff) - clim`
(so `beta` is scale-free), but the FLOOR does not cancel, which is why the floor is
stated in THIS file's units and the near-duplicate `self.carrier` is explicitly NOT
the object being used.
"""
import numpy as np

W_FLOOR = 1e-3
REF_YEARS = (1961, 2020)
CLIP = 700.0


def mobile_water(fast_water, percolation, area_ha):
    """`W = fast_water + percolation * area_ha * 10`, unscaled, in m^3/day."""
    fw = np.asarray(fast_water, dtype=np.float64)
    pc = np.asarray(percolation, dtype=np.float64)
    ah = np.asarray(area_ha, dtype=np.float64)
    assert fw.ndim == 2 and pc.shape == fw.shape, (fw.shape, pc.shape)
    assert ah.shape == (fw.shape[1],), (ah.shape, fw.shape)
    return fw + pc * ah[None, :] * 10.0


def geometry(fast_water, percolation, area_ha, years, ref_years=REF_YEARS,
             w_floor=W_FLOOR):
    """Everything that does NOT depend on beta.  Computed once, hashed, frozen."""
    Qf = np.asarray(fast_water, dtype=np.float64)
    Qs = np.asarray(percolation, dtype=np.float64) \
        * np.asarray(area_ha, dtype=np.float64)[None, :] * 10.0
    W = Qf + Qs
    n_nonpos = int((Qf <= 0).sum() + (Qs <= 0).sum())
    if n_nonpos:
        raise SystemExit('MOBILE_WATER_NOT_STRICTLY_POSITIVE %d' % n_nonpos)
    if not (np.isfinite(Qf).all() and np.isfinite(Qs).all()):
        raise SystemExit('MOBILE_WATER_NOT_FINITE')
    Weff = np.maximum(W, w_floor)
    active = W > w_floor                       # the floor itself is INACTIVE
    logWeff = np.log(Weff)
    # on active cells `maximum` returns W itself, so log(Weff) is bitwise log(W)
    assert np.array_equal(logWeff[active], np.log(W)[active]), 'FLOOR_NOT_EXACT'

    yrs = np.asarray(years).astype('datetime64[Y]').astype(np.int64) + 1970
    ref = (yrs >= ref_years[0]) & (yrs <= ref_years[1])
    n_ref = int(ref.sum())
    if n_ref == 0:
        raise SystemExit('EMPTY_REFERENCE_WINDOW')
    Wref = Weff[ref]
    Aref = active[ref]
    n_active_ref_per_reach = Aref.sum(axis=0)
    if int((n_active_ref_per_reach < 2).sum()):
        raise SystemExit('REACH_WITHOUT_USABLE_REFERENCE_DAYS %d'
                         % int((n_active_ref_per_reach < 2).sum()))
    # per-reach mean over ACTIVE reference cells only
    denom = n_active_ref_per_reach.astype(np.float64)
    clim = np.where(Aref, logWeff[ref], 0.0).sum(axis=0) / denom
    u = logWeff - clim[None, :]

    # ---- S_u: the sd-unit reading, a PURE RELABELLING of the same grid ----------
    # per-reach sd of u over the ACTIVE reference cells, then the median over reaches
    ctr = np.where(Aref, u[ref] - clim[None, :], 0.0)
    sq = np.where(Aref, ctr * ctr, 0.0).sum(axis=0)
    sd_r = np.sqrt(sq / denom)
    S_u = float(np.median(sd_r))

    diag = dict(
        w_floor=float(w_floor), ref_years=[int(ref_years[0]), int(ref_years[1])],
        n_ref_days=n_ref, shape=list(W.shape),
        w_min=float(W.min()), w_max=float(W.max()),
        qf_min=float(Qf.min()), qs_min=float(Qs.min()),
        n_nonpositive=int(n_nonpos),
        frac_w_lt_1e_6=float((W < 1e-6).mean()),
        frac_active=float(active.mean()),
        active_frac_at_floors={str(f): float((W > f).mean()) for f in (1e-3, 1e-1, 1.0)},
        n_reaches_without_active_ref_day=int((n_active_ref_per_reach == 0).sum()),
        n_active_ref_days_min=int(n_active_ref_per_reach.min()),
        n_active_ref_days_median=float(np.median(n_active_ref_per_reach)),
        n_active_ref_days_max=int(n_active_ref_per_reach.max()),
        clim_min=float(clim.min()), clim_max=float(clim.max()),
        u_active_min=float(u[active].min()), u_active_max=float(u[active].max()),
        u_active_abs_max=float(np.abs(u[active]).max()),
        u_active_quantiles={str(q): float(v) for q, v in
                            zip((0, 1, 50, 99, 100),
                                np.percentile(u[active], (0, 1, 50, 99, 100)))},
        S_u=S_u, S_u_rule='median over all reaches of the per-reach sd of u over the '
                          'ACTIVE cells of 1961-2020 (pre-evaluation)',
        per_reach_sd_min=float(sd_r.min()), per_reach_sd_max=float(sd_r.max()),
    )
    return dict(Qf=Qf, Qs=Qs, W=W, Weff=Weff, active=active, clim=clim, u=u,
                S_u=S_u, sd_r=sd_r, diag=diag, ref=ref)


def xi_from(G, beta):
    """`Xi` at this beta.  THE frozen spelling -- see the module docstring."""
    b = float(beta)
    if b == 0.0:
        # exact by construction; not special-cased for speed, for ASSURANCE
        pass
    Lf = np.log(G['Qf'])
    Ls = np.log(G['Qs'])
    L0 = np.logaddexp(Lf, Ls)
    half = b * G['u'] / 2.0
    Lb = np.logaddexp(Lf + half, Ls - half)
    Xi = np.exp(np.clip(Lb - L0, -CLIP, CLIP))
    Xi = np.where(G['active'], Xi, 1.0)
    return Xi


def tag_slice(Xi, pilot_indices):
    """The column slice the frozen `campaign_model.py:104` call site requires.

    `ledger` passes `h[:,rr]`, so a FULL `Xi` handed to the tagged kernel would be
    read column-by-column from index 0 and silently modulate reaches 1..2 instead of
    the pilot reaches.  numba performs no cross-argument shape check, so this is a
    silent corruption, not a crash.
    """
    rr = np.asarray(pilot_indices, dtype=np.int64)
    return np.ascontiguousarray(Xi[:, rr])


def saturation_census(Xi, h, active):
    """Everything a reader needs to see that `Xi` is a SELECTOR, not a tilt.

    Red-team measurement: `Xi` is unbounded and exponentially asymmetric.  Reported
    per beta so no large-beta row can be read as a modest tilt.
    """
    pos = active
    x = Xi[pos]
    prob = -np.expm1(-np.minimum(h * Xi, CLIP))
    prod = h * Xi
    out = dict(
        frac_abs_xi_minus_1_gt_1e6=float((np.abs(Xi - 1.0) > 1e-6).mean()),
        frac_xi_gt_10=float((Xi > 10.0).mean()),
        frac_xi_lt_0p1=float((Xi < 0.1).mean()),
        frac_prod_ge_700=float((prod >= CLIP).mean()),
        frac_prob_ge_0p99=float((prob >= 0.99).mean()),
        frac_prob_ge_1e_3=float((prob >= 1e-3).mean()),
        xi_min=float(Xi.min()), xi_max=float(Xi.max()),
        xi_active_quantiles={str(q): float(v) for q, v in
                             zip((0, 1, 50, 99, 100),
                                 np.percentile(x, (0, 1, 50, 99, 100)))},
        med_abs_log_xi=float(np.median(np.abs(np.log(x)))),
        n_nonfinite=int((~np.isfinite(Xi)).sum()),
        n_nonpositive=int((Xi <= 0).sum()),
        prob_baseline_frac_ge_0p99=float(
            (np.where(pos, -np.expm1(-np.minimum(h, CLIP)), 0.0) >= 0.99).mean()),
    )
    return out


def admissible(census, sat_bound=0.10, degen_floor=1e-3):
    """The PRE-REGISTERED mechanical availability rule (plan 1.3-3).

    A beta may carry a QUALIFY / L1_ONLY verdict only if it is BOTH
    non-degenerate (it really modulates: frac(|Xi-1|>1e-6) >= degen_floor) and
    not a saturation-dominated binary selector (frac(prob>=0.99) <= sat_bound).

    This is a registered rule applied to a reported number, not a judgement made
    after seeing an amplitude.  It must not be relaxed post hoc.
    """
    non_degenerate = census['frac_abs_xi_minus_1_gt_1e6'] >= degen_floor
    not_saturated = census['frac_prob_ge_0p99'] <= sat_bound
    return dict(non_degenerate=bool(non_degenerate),
                not_saturated=bool(not_saturated),
                sat_bound=float(sat_bound), degen_floor=float(degen_floor),
                admissible=bool(non_degenerate and not_saturated))
