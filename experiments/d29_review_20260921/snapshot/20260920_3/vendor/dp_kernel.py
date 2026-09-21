"""The conserving dual-pathway mobile-water-concentration kernel: the ONE file in this
round that owns a frozen spelling.

WHAT THIS FILE IS, AND WHY IT IS SEPARATE FROM `closures_dp.py`
---------------------------------------------------------------
This is round 5's `xi_base.py` analogue, and it is deliberately SIMPLER than that file:
there is no hazard, no mask, no floor, no `clip`, and no log-sum-exp.  What lives here
is exactly three things:

  1. the VOLUMES (`V_u`, `V_s`) and the three water pathways (`Q_f`, `Q_p`, `Q_s`),
     each one a deterministic read of a producer-owned array;
  2. the closure `g`, which has EXACTLY TWO admissible spellings;
  3. the concentration form `phi(x) = -expm1(-x)/x`, whose denominator contains no `Q`.

`closures_dp.py` holds the recursion; it receives `g_u`, `phi_f`, `g_s` as ARGUMENTS.
The split matters because `closures_dp` must stay a bitwise reduction to the frozen
kernel when the three fractions are injected (assertion N1), and a file that both owns
the closure and consumes it cannot be audited that way.

THE PHASE 0 RULING THAT FIXES `g`  (plan S1.3, row 1)
-----------------------------------------------------
The producer `20260825_3/scripts/hydrology_core.py:167-181` applies the day's INFLOW
BEFORE the day's OUTFLOW:

    sm += infiltration ; fast += excess                       (167-168)
    percolation = min(perc_mm_day, fast) ; fast -= percolation (173-174)
    q0 = min(k0*max(fast-uzl,0), fast) ; fast -= q0            (177-178)
    q1 = min(k1*fast, fast)            ; fast -= q1            (179-180)
    q2 = min(k2*slow, slow)            ; slow -= q2            (181-182)

so the store holds `S[t] + Q_out[t]` at the instant outflow begins, where `S[t]` is the
exported (POST-outflow) carry.  Measured on all three stores, `S[t] + Q_out[t]` equals
the inflow-based rebuild at `max_rel <= 3.58e-14`, while the alternative reading
`S[t-1]` is off by `median_rel = 0.191`.  With `V = S_post + Q_out` and `S_post >= 0`:

    x = Q_out / (S_post + Q_out) <= 1        IDENTICALLY

`probe_prevolume.json::g_decision` reports `n_x_gt_1 = 0` at `p100 = 0.9999999999999986`
for the upper store and `max = 0.2293` for the soil store.  S1.3's row-1 condition is
therefore MET, and the closure is the LINEAR one.  **This is a deterministic output of
the timing gate, not the registered safe default** -- the safe default (`1-e^{-x}`)
fires only when the gate cannot decide between pre and post, which it did.

The ruling also REFUTES the earlier red-team hypothesis that the observed saturation was
a pre/post artifact: correcting the denominator takes `frac(x >= 1)` from 0.2155 to
`0.0`, but `S[t-1]` (the reading that was actually tested first) takes it only to
0.2011.  `probe_saturation.json` shows the surviving saturation was real turnover.

THE INERT GUARD, AND WHY IT SURVIVES
------------------------------------
`probe_timing.json::slow_path_on_mask` measures, on `contact <= 0`:
`max(fast_water) = 4.45e-21 m3/day`, `max(percolation) = 2.51e-14 mm/day`,
`max(fast_fraction) = 0.0`.  The fluxes vanish.  But `g` depends on the RATIO
`Q_u / V_u`, and on those cells `upper_water -> 1e-28 mm` collapses FASTER than
`Q_u -> 0`, so `x -> 1` and `g -> 1`; `probe_timing` reports
`max_Eu_over_A_on_mask = 1.0`.  Under the linear closure the same thing happens by
algebra: `x = Q_u/(S+Q_u) -> 1` as `S -> 0`.

So the earlier derivation ("`Q -> 0` therefore `E -> 0`, the guard is unnecessary") is
WRONG, and the guard is retained.  Its form is chosen so that N1 stays consistent: the
guard is evaluated on the FROZEN `fast_fraction == 0` set, which is exactly the set on
which the frozen kernel's own `prob = -expm1(-min(h,700)) = 0` (`h` is zero there), so
injecting `g_u := p_frozen` and applying the guard are the SAME array.  This is also why
the guard introduces no second free parameter: it reproduces the frozen semantics rather
than inventing one.

Registering it is mandatory (plan S3.3, S3.5 N9).  It is NOT registered as `inert_mask`
in the `xi_base.saturation_census` sense -- there is no floor and no clip here.

WHAT IS DELIBERATELY *NOT* HERE
-------------------------------
* No `min()`, no cap, no clip.  `x <= 1` is proven, not enforced; enforcing it would
  hide a timing error instead of exposing one.
* No per-reach or station-level parameter.  Every array is a read.
* `g` has exactly two spellings.  `G_FORMS` is the closed set; adding a third is a
  plan violation, not an implementation choice.
* No `1 - np.exp(-x)` anywhere.  See `expm1_underflow_probe()` and assertion N12: at
  `x = 1e-200` the direct subtraction gives exactly `0.0` while `-expm1(-x)` gives
  exactly `x`.  That is a numerical-correctness fact, not a style preference.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# units, registered rather than guessed (plan S1.5)
# --------------------------------------------------------------------------
#   Qf = fast_water / (area_ha * 10)      mm/day    (verified bitwise against the
#   Qp = percolation                      mm/day     producer's own local_fast_mm:
#   Qs = slow_water / (area_ha * 10)      mm/day     `fast_mm == fast_s*86.4/area_km2`
#   V_u, V_s                              mm         to 0.0 absolute in probe 1)
#   x = Q/V                               1/day
# Area cancels identically inside x, so no reach geometry enters the mobilisation.

G_FORMS = ('x', '-expm1(-x)')          # the CLOSED set.  Two, never three.
MM_PER_HA = 10.0                       # 1 mm over 1 ha == 10 m3


# --------------------------------------------------------------------------
# 1. the volumes -- deterministic reads, no reconstruction from a flux
# --------------------------------------------------------------------------
def flows(fast_water_m3_day, percolation_mm_day, slow_water_m3_day, area_ha):
    """The three pathways in mm/day.  `area_ha` is `(nr,)`; the fluxes are `(nd, nr)`.

    All three are STRICTLY POSITIVE on the interior of the domain -- probe 1 reports
    `fast_water >= 4.184469060922251e-200`, `percolation >= 1.1282505570800604e-204` --
    so `x = Q/V` is well defined everywhere and no `0/0` can arise.  The assertion is
    made rather than assumed because a future producer that emits an exact zero would
    silently turn `phi` into a NaN.
    """
    A = np.asarray(area_ha, np.float64)[None, :]
    Qf = np.asarray(fast_water_m3_day, np.float64) / (A * MM_PER_HA)
    Qp = np.asarray(percolation_mm_day, np.float64)
    Qs = np.asarray(slow_water_m3_day, np.float64) / (A * MM_PER_HA)
    for name, v in (('Qf', Qf), ('Qp', Qp), ('Qs', Qs)):
        if not np.all(np.isfinite(v)) or not np.all(v > 0.0):
            raise ValueError('NONPOSITIVE_PATHWAY:%s min=%r' % (name, float(np.min(v))))
    return dict(Qf=Qf, Qp=Qp, Qs=Qs, Qu=Qf + Qp)


def volumes(upper_water_mm, lower_slow_storage_mm, F):
    """`V_u` and `V_s` at the INSTANT OUTFLOW BEGINS: the exported carry plus the day's
    outflow.  See the module docstring for the producer lines that make this the right
    reading and for the measured residual (`max_rel <= 3.58e-14` on three stores).

    `upper_water_mm` and `lower_slow_storage_mm` are BOTH the exported POST-outflow
    carry -- settled twice, by `closures`-side balance and by the producer's own
    assertion in `frozen_hydrology_core.py:194`.
    """
    Vu = np.asarray(upper_water_mm, np.float64) + F['Qu']
    Vs = np.asarray(lower_slow_storage_mm, np.float64) + F['Qs']
    for name, v in (('Vu', Vu), ('Vs', Vs)):
        if not np.all(np.isfinite(v)) or not np.all(v > 0.0):
            raise ValueError('NONPOSITIVE_VOLUME:%s min=%r' % (name, float(np.min(v))))
    return dict(Vu=Vu, Vs=Vs)


# --------------------------------------------------------------------------
# 2. the closure -- two spellings, and only two
# --------------------------------------------------------------------------
def g_of(x, form):
    """The single nonlinear operator of this round.

    `form='x'` returns `x` unchanged.  `form='-expm1(-x)'` returns the exponential
    closure.  The exponential is written with `expm1` and never as `1 - exp(-x)`: at
    `x = 1e-200` the subtraction is exactly 0.0 in float64 (assertion N12), which would
    make a legitimate small mobilisation silently vanish on the deep-tail cells where
    `Q` reaches `1e-200`.
    """
    if form == 'x':
        return np.asarray(x, np.float64)
    if form == '-expm1(-x)':
        return -np.expm1(-np.asarray(x, np.float64))
    raise ValueError('G_FORM_NOT_ADMISSIBLE:%r (closed set is %r)' % (form, G_FORMS))


def phi_of(x, form):
    """`phi(x) = g(x)/x`, the factor in the STABLE concentration form (plan S1.2).

        C_u = 1000 * A / ((area_ha*10) * V_u) * phi(x_u)
        C_s = 1000 * L / ((area_ha*10) * V_s) * phi(x_s)

    The denominator contains no `Q`, so `Q -> 0` underflows nothing and raises nothing:
    the analytic `x -> 0` limit is produced by the SAME expression rather than by a
    special case.  For `g = x` the factor is identically 1.0 and the division is skipped
    rather than evaluated (so no `0/0` can appear at `x = 0`).  For the exponential form
    `phi = -expm1(-x)/x`, which is the standard `exprel` and is `1 - x/2 + ...` near 0.
    """
    x = np.asarray(x, np.float64)
    if form == 'x':
        return np.ones_like(x)
    if form == '-expm1(-x)':
        out = np.ones_like(x)
        nz = x != 0.0
        out[nz] = -np.expm1(-x[nz]) / x[nz]
        return out
    raise ValueError('G_FORM_NOT_ADMISSIBLE:%r' % (form,))


# --------------------------------------------------------------------------
# 3. the three fractions handed to the recursion
# --------------------------------------------------------------------------
def fractions(F, V, form, guard=None):
    """`g_u`, `phi_f`, `g_s`, all `(nd, nr)`, as the recursion's ARGUMENTS.

    `phi_f = Q_f/Q_u` is a WATER split between two exits of the SAME store, not a
    second concentration -- so `F_f` and `J` are `E_u` times two water shares and there
    is no fast-before-percolation ordering anywhere.  `phi_f in (0, 1)` strictly because
    both pathways are strictly positive.

    `guard` is the registered inert guard: a `(nd, nr)` array that is `1.0` where the
    cell is live and `0.0` on the frozen `fast_fraction == 0` set.  It multiplies `g_u`
    only -- `phi_f` is a ratio of two vanishing fluxes and `g_s` is the slow path, which
    the frozen kernel also leaves live on those cells (registered as a DELIBERATE
    structural difference, not an error).
    """
    Qu, Qf = F['Qu'], F['Qf']
    xu = Qu / V['Vu']
    xs = F['Qs'] / V['Vs']
    gu = g_of(xu, form)
    if guard is not None:
        gu = gu * np.asarray(guard, np.float64)
    pf = Qf / Qu
    gs = g_of(xs, form)
    for name, v in (('gu', gu), ('phi_f', pf), ('gs', gs)):
        if v.shape != Qu.shape:
            raise ValueError('FRACTION_SHAPE:%s %r vs %r' % (name, v.shape, Qu.shape))
        if not np.all(np.isfinite(v)):
            raise ValueError('FRACTION_NOT_FINITE:%s' % name)
    if np.any(gu < 0.0) or np.any(gu > 1.0):
        raise ValueError('GU_OUT_OF_RANGE min=%r max=%r'
                         % (float(np.min(gu)), float(np.max(gu))))
    if np.any(pf <= 0.0) or np.any(pf >= 1.0):
        raise ValueError('PHI_F_OUT_OF_RANGE min=%r max=%r'
                         % (float(np.min(pf)), float(np.max(pf))))
    if np.any(gs < 0.0) or np.any(gs > 1.0):
        raise ValueError('GS_OUT_OF_RANGE min=%r max=%r'
                         % (float(np.min(gs)), float(np.max(gs))))
    return dict(gu=gu, phi_f=pf, gs=gs, xu=xu, xs=xs)


def guard_from(fast_fraction):
    """`1.0` where live, `0.0` on the frozen `fast_fraction == 0` set.

    NOT a new parameter: `fast_fraction == 0` is the producer's OWN statement that the
    cell has no hydraulic contact with the reach, and on exactly that set the frozen
    kernel's `prob` is already `0`.  Asserted rather than trusted -- a guard applied to
    a set the frozen kernel does not zero would make N1 fail, and N1 failing after the
    guard is applied is a real bug, not a tolerance.
    """
    ff = np.asarray(fast_fraction, np.float64)
    if ff.shape[0] == 0:
        raise ValueError('EMPTY_FAST_FRACTION')
    if np.any(~np.isfinite(ff)):
        raise ValueError('FAST_FRACTION_NOT_FINITE')
    if np.any((ff < 0.0) | (ff > 1.0)):
        raise ValueError('FAST_FRACTION_OUT_OF_RANGE')
    return np.where(ff == 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# 4. the two zero-cost gates the plan requires to be REPORTED, not merely passed
# --------------------------------------------------------------------------
def timing_gate(S_post, Q_out, inflow, prev_S):
    """S1.3's rebuild check, on one store.

    `S_pre = S_post + Q_out` must equal `prev_S + inflow`.  On the real arrays this
    closes at float64 rounding (measured `max_rel` 3.68e-16 / 3.57e-14 / 4.35e-16 on the
    lower / upper / soil stores).  `x_le_1` is then a theorem, not a fit; it is reported
    so the theorem is checked against the array rather than asserted.
    """
    Spost = np.asarray(S_post, np.float64)
    Q = np.asarray(Q_out, np.float64)
    rebuild = np.asarray(prev_S, np.float64) + np.asarray(inflow, np.float64)
    V = Spost + Q
    d = np.abs(V - rebuild)
    sc = np.maximum(np.maximum(np.abs(V), np.abs(rebuild)), 1e-300)
    x = Q / np.where(V > 0.0, V, np.nan)
    return dict(max_abs=float(np.max(d)), max_rel=float(np.max(d / sc)),
                n_x_gt_1=int(np.sum(np.isfinite(x) & (x > 1.0))),
                max_x=float(np.nanmax(x)), x_le_1_globally=bool(np.all(x <= 1.0)))


def turnover_candidates(V, Q, names):
    """`V/Q` per candidate, for the S2.3 comparison table.

    Reported, NOT used as the primary-arm selector.  `probe_timing.json` measures that
    `V-upper` against the producer's own `upper_store_instantaneous_turnover_day` is an
    ALGEBRAIC IDENTITY (`median |log ratio| = 0.0` exactly, because
    `86400/(100*10) == 86.4`), so a rule that selects on that comparison selects
    `V-upper` for free and carries no evidence.  The water balance decides instead.
    """
    out = {}
    for n in names:
        with np.errstate(divide='ignore', invalid='ignore'):
            out[n] = V[n] / np.where(Q[n] > 0.0, Q[n], np.nan)
    return out


def expm1_underflow_probe(x=1e-200):
    """Assertion N12's evidence, measured rather than argued.

    `1 - exp(-x)` evaluates to exactly `0.0` at `x = 1e-200` in float64 because
    `exp(-1e-200)` rounds to `1.0`; `-expm1(-x)` returns the argument itself.  The
    difference is not a rounding curiosity here: `Q` reaches `1e-200` on real cells, so
    the naive form would report "no mobilisation at all" on them.
    """
    x = float(x)
    naive = float(1.0 - np.exp(-x))
    stable = float(-np.expm1(-x))
    return dict(x=x, one_minus_exp=naive, neg_expm1=stable,
                naive_is_exactly_zero=bool(naive == 0.0),
                stable_equals_x=bool(stable == x),
                note='the direct subtraction is EXACTLY zero; expm1 returns the argument')
