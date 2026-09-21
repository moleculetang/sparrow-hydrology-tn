"""The conserving dual-pathway mobile-water-concentration recursion WITH a legacy/mobile
accessibility split: `20260920_1`'s kernel plus ONE new state pair and ONE rate.

DERIVED FROM `20260920_1/work/closures_dp.py`.  THE DELTA, EXHAUSTIVELY
---------------------------------------------------------------------
  1. NEW module-global `CONFIG['q_m']`, the scalar daily fraction of the post-uptake
     legacy pool that becomes mobile-accessible.  It is a SCALAR on purpose: a `(nr,)`
     vector would be 230 parameters wearing one name (asserted in `_check_q_m`).
  2. `scan_dp` -> `scan_dp2`, `_tag_dp_nb` -> `_tag_dp2_nb`,
     `TransportDP` -> `TransportDP2`, `ledger_dp` -> `ledger_dp2`,
     `tag_scan_dp` -> `tag_scan_dp2`, `scan_dp_full` -> `scan_dp2_full`,
     `scan_dp_ledger` -> `scan_dp2_ledger`.  Apart from the appended `q_m` argument and the
     appended fifth output (items 3 and 5 below), ONLY the names changed:
     `TransportDP2`, `scan_dp2_ledger` and `tag_scan_dp2` have round 1's exact arities, and
     `tag_scan_dp2` still takes EXACTLY SIX positional arguments (the bare name
     `campaign_model.Matched.ledger:104` supplies six and cannot be handed a seventh).
  3. The single mobile pool `M[r]` becomes a PAIR `ML[r]` (legacy) + `MM[r]` (mobile).
     Inputs enter `ML`; `q_m` moves mass `ML -> MM`; everything downstream is unchanged.
  4. `ledger_dp2` respells the two pools and returns five extra diagnostic fields.
  5. `scan_dp2` returns a FIFTH output, `N^{M,pre}`.  Round 1's `scan_dp` returned four;
     the first four keep their order and meaning, so `scan_dp2_ledger` (exactly four),
     `TransportDP2.apply` (two) and the bare-name call sites are unaffected.
     `scan_dp2_full` is the only caller that sees the fifth.  Reason: `layers25.cu_cs`
     needs the pool the mobilisation draws from, which is no longer `a`.

THE ONE ARITHMETIC CHANGE, AND WHY IT IS SPELLED THIS WAY
---------------------------------------------------------
Round 1 (`closures_dp.py:134-143`):

    av   = max(M[r] + inp[t,r] - demand[t,r], 0.)
    Eu   = av * gu
    ...
    M[r] = av * (1. - gu) * survival

Round 2, per (t, r):

    uL      = ML[r] + inp[t, r]          # the day's input enters the LEGACY side
    uP      = uL + MM[r]                 # total available before crop uptake
    av      = max(uP - demand[t, r], 0.) # total available after crop uptake  == round 1's av
    rL      = uL / uP                    # parameter-free proportional share
    nL      = av * rL                    # legacy after uptake
    T       = q_m * nL                   # legacy -> mobile, the ONE new rate
    nl_star = nL - T
    pre     = av - nl_star               # mobile pool BEFORE mobilisation
    Eu      = pre * gu
    fast    = Eu * phi_f
    J       = Eu * (1. - phi_f)
    Lpre    = L[r] + J
    slow    = Lpre * gs
    MM[r]   = pre * (1. - gu) * survival
    ML[r]   = nl_star * survival

WHY `pre = av - nl_star` AND NOT `pre = nM + T`
-----------------------------------------------
The two are the same in exact arithmetic (`nM + T = av - nL + q_m*nL = av - nl_star`) and
`nL = NL - U_L` equals `av*(uL/uP)` exactly, because under proportional uptake
`NL - U*(NL/P) = NL*(P-U)/P`.  But only the `av - nl_star` spelling makes the PARENT
REDUCTION EXACT IN FLOAT64, which assertion N1' requires bitwise:

  q_m = 1  =>  T = 1.0*nL (exact), nl_star = nL - nL = 0.0 (exact), pre = av - 0.0 = av
           =>  MM[r] = av*(1.-gu)*survival   -- term for term round 1's own expression
           =>  ML[r] = 0.0*survival = 0.0    -- and it stays 0, so uL = 0 + inp = inp
           =>  uP = inp + MM = MM + inp (IEEE addition commutes) = round 1's M + inp
           =>  av  = max((M + inp) - demand, 0.)  -- round 1's own expression

The `nM + T` spelling instead forces `pre = fl(fl(av - nL) + nL)`, a two-sum round trip
that returns `av` only when no tie occurs.  A bitwise gate that holds "almost always" is
not a bitwise gate.  THE SPELLING IS THEREFORE LOAD-BEARING, not stylistic.

`uP > 0` IS GUARDED, AND THE GUARD IS INERT AT THE REDUCTION POINT
------------------------------------------------------------------
`uL/uP` is `0/0` when the reach has no N at all.  In that case `av = max(0 - demand, 0) =
0`, so `nL = av*rL` would be `0*NaN = NaN` if `rL` were evaluated.  `rL` is therefore
`0.0` when `uP <= 0`, and `nL` is `0`, which is what round 1 produces too (its `av = 0`
nulls every term downstream).  The guard adds no free parameter and does not weaken N1'.

`T^{mob}` IS NOT A FUNCTION OF WATER
------------------------------------
`T = q_m * nL` with `q_m` a scalar read from `CONFIG`.  No `Q`, no `V`, no `x`, no mask
enters it, and on `contact <= 0` cells the transfer still happens: the guard zeroes
`g_u`, hence `Eu`, hence `fast`/`J`, but `ML` and `MM` keep evolving.  Assertion N13
checks both the source text and the array (`T` must be exactly `q_m * nL` elementwise).

WHAT IS *NOT* HERE
------------------
* No `k_ex`, no second loss parameter.  BOTH pools share the frozen `s_M`
  (`survival`), which is the discipline that keeps this a ONE-parameter extension.
* No temperature or moisture modulation, no source-specific mobile fraction.
* No `min`, no cap, no clip, no floor, and no `1 - exp(-x)` anywhere.
* `k_m` does not appear in this file at all: `q_m` is the argument.  `k_m = 1/tau_m`
  is computed once, in `common25.q_m_of`, and never here.
"""
import numpy as np
import torch
from numba import njit

# See `closures_dp.py:69-75` for why the fractions cannot travel as call arguments at two
# of the entry points, and why numba module globals are compile-time constants.
# `q_m` joins them: `scan_dp2_ledger` and `tag_scan_dp2` are reached through BARE names
# with fixed arity, so the scalar has to come through CONFIG like the arrays.
CONFIG = {'gu': None, 'phi_f': None, 'gs': None,
          'gu_tag': None, 'phi_f_tag': None, 'gs_tag': None, 'q_m': None}


def set_config(gu, phi_f, gs, q_m=None, gu_tag=None, phi_f_tag=None, gs_tag=None):
    CONFIG.update(gu=gu, phi_f=phi_f, gs=gs, q_m=q_m,
                  gu_tag=gu_tag, phi_f_tag=phi_f_tag, gs_tag=gs_tag)
    return {k: (None if v is None else (float(v) if k == 'q_m' else list(v.shape)))
            for k, v in CONFIG.items()}


def _shape(v):
    return None if v is None else tuple(v.shape)


def _check(h, keys):
    """Assert every named CONFIG array is installed, agrees, and matches `h`'s shape.

    numba performs NO cross-argument shape check (memory: `closures-scan-is-njit-with-no-
    shape-checking`), so a fraction array of the wrong shape is a silent read of the wrong
    columns rather than an exception.  Every Python wrapper goes through here once per
    array, as assertion N11 requires.
    """
    out = []
    for k in keys:
        v = CONFIG[k]
        if v is None:
            raise RuntimeError('FRACTIONS_NOT_INSTALLED:%s' % k)
        if v.shape != h.shape:
            raise ValueError('FRACTION_SHAPE_MISMATCH:%s %r vs h %r'
                             % (k, tuple(v.shape), tuple(h.shape)))
        if not np.all(np.isfinite(v)):
            raise ValueError('FRACTION_NOT_FINITE:%s' % k)
        out.append(np.ascontiguousarray(v))
    if not (0.0 <= float(np.min(out[0])) and float(np.max(out[0])) <= 1.0):
        raise ValueError('GU_OUT_OF_RANGE')
    return out


def _check_q_m():
    """The scalar-type assertion N11 adds on top of the three array assertions.

    THREE separate refusals, because each one is a different way to smuggle a parameter
    in: an unset slot (the kernel would silently use a stale arm's value), an ARRAY
    (`(nr,)` is 230 reach-specific timescales behind one name), and an out-of-range or
    non-finite value (`q_m > 1` would move more N than the pool holds; `q_m = NaN` would
    poison every state downstream without raising).
    """
    q = CONFIG['q_m']
    if q is None:
        raise RuntimeError('Q_M_NOT_INSTALLED')
    if isinstance(q, np.ndarray):
        raise TypeError('Q_M_MUST_BE_A_SCALAR: got ndarray %r -- one scalar is one shared '
                        'timescale; a vector is %d parameters' % (q.shape, int(q.size)))
    q = float(q)
    if not np.isfinite(q) or not (0.0 <= q <= 1.0):
        raise ValueError('Q_M_OUT_OF_RANGE:%r' % q)
    return q


@njit(cache=True)
def scan_dp2(h, s, f, k, l, inp, demand, mode, gu, phi_f, gs, q_m):
    """`scan_dp` with the single land-phase pool split into legacy and mobile.

    Same name slot, same argument order, same four outputs in the same order (round 1's
    `scan_dp` ends with the three fractions; `q_m` is appended LAST so that the frozen
    argument prefix is untouched and the reduction test can hand both kernels the same
    first eleven arguments).

    `h`, `f`, `k`, `l` and `mode` are retained and UNREAD, exactly as in round 1: `h` and
    `f` carried the frozen mobilisation and split, which the volumes now supply.

    THE RETURNED `p` IS STILL `g_u`.  `ResearchObjective.ledger` derives the retention
    term from `a` and `p` alone, so returning anything else would put the replacement into
    the fluxes but not into the retention, and the ledger would measure the replacement
    instead of the rounding.

    A FIFTH OUTPUT, `prea`, IS NEW AND IS A DIAGNOSTIC, NOT A FIFTH FLUX.
    -------------------------------------------------------------------
    `prea[t,r] = N^{M,pre}` is the mobile pool the mobilisation actually draws from, and it
    is what `layers25.cu_cs` needs: round 1's `C_u` used `X = a` because `E = a*g_u` there,
    but here `E_M = pre*g_u` with `pre <= a`, so the round-1 spelling would pin `C_u` to
    the wrong quantity and the `EU_PHI_F_DOES_NOT_REPRODUCE_FAST` assertion would fire.
    Returning it from the SAME call is the only way to avoid a second spelling of the
    recursion.  The first four outputs keep round 1's order and meaning, so
    `scan_dp2_ledger` (which unpacks exactly four) and `TransportDP2.apply` (which takes
    two) are unaffected.
    """
    nd, nr = h.shape
    fast = np.zeros_like(h); slow = np.zeros_like(h)
    a = np.zeros_like(h); p = np.zeros_like(h); prea = np.zeros_like(h)
    ML = np.zeros(nr); MM = np.zeros(nr); L = np.zeros(nr)
    for t in range(nd):
        for r in range(nr):
            uL = ML[r] + inp[t, r]                 # the input enters the LEGACY side
            uP = uL + MM[r]
            av = max(uP - demand[t, r], 0.)
            rL = (uL / uP) if uP > 0. else 0.
            nL = av * rL
            T = q_m * nL                           # legacy -> mobile, the ONE new rate
            nl_star = nL - T
            pre = av - nl_star                     # see the docstring: spelling is load-bearing
            Eu = pre * gu[t, r]
            fast[t, r] = Eu * phi_f[t, r]
            J = Eu * (1. - phi_f[t, r])
            Lpre = L[r] + J
            slow[t, r] = Lpre * gs[t, r]
            survival = s[r] if s.ndim == 1 else s[t, r]
            MM[r] = pre * (1. - gu[t, r]) * survival
            ML[r] = nl_star * survival
            L[r] = Lpre - slow[t, r]
            a[t, r] = av; p[t, r] = gu[t, r]; prea[t, r] = pre
    return fast, slow, a, p, prea


@njit(cache=True)
def _tag_dp2_nb(h, s, f, release, inputs, demand, gu, phi_f, gs, q_m):
    """`_tag_dp_nb` with the per-tag pool split into legacy and mobile.

    Same order, same SEVEN outputs.  `h`, `f` and `release` are retained and unread.

    The per-tag structure mirrors round 1's exactly, so the reduction is term for term:
    round 1 accumulates `B += M[r,j] + inputs[t,r,j]` and then `A = raw*ratio`; this
    version accumulates `B += (ML[r,j] + inputs[t,r,j]) + MM[r,j]` and then
    `A = raw*ratio` from the SAME stored `raw`.  With `ML == 0` the two accumulands are
    `0 + inputs + M` and `inputs + M` -- IEEE addition commutes, so they are the same bits.

    `legpre` is RE-READ from `ML[r,j]` at the top of the second loop rather than stored.
    That is safe because `ML[r,j]` is written only at the end of iteration `j`, so at the
    top of iteration `j` it still holds the previous day's value for that tag -- and only
    that tag is read.

    SLOT 3 IS THE *TAGGED LAND STATE*, NOT THE MOBILE STATE.  `ms[t,r,j]` must be the object
    `ledger_dp2` reports under the name `M`, because `campaign_model.Matched.ledger:107`
    grades the label residual as `|ms.sum(-1) - a['M'][:, rr]|`.  Round 1's `_tag_dp_nb`
    wrote `ms[t,r,j] = M[r,j]`, i.e. the whole retained land pool `A*(1-p)*s`; the ledger
    named the same object `M`.  Round 2 splits that pool, so the state is the SUM
    `ML + MM` -- the same association `ledger_dp2:447` uses (`M = ML + MM`).  Writing `MM`
    alone here is a real defect, not a rounding: it makes the residual equal the legacy
    pool (measured 1.83e7 kg at the primary arm against a 1e-6 kg tolerance), and it is the
    only one of the six channels that fails.  The reduction is preserved: at `q_m = 1`,
    `nl_star` is exactly `0.0`, so `ML[r,j] = 0.0` and `0.0 + MM[r,j] == MM[r,j]` -- the
    slot's bits at the parent control are exactly round 1's.
    """
    nd, nr = h.shape; ns = inputs.shape[2]
    fast = np.zeros_like(inputs); slow = np.zeros_like(inputs); raw = np.zeros_like(inputs)
    ms = np.zeros_like(inputs); ls = np.zeros_like(inputs)
    uptake = np.zeros_like(inputs); loss = np.zeros_like(inputs)
    ML = np.zeros((nr, ns)); MM = np.zeros((nr, ns)); L = np.zeros((nr, ns))
    for t in range(nd):
        for r in range(nr):
            B = 0.
            for j in range(ns):
                raw[t, r, j] = (ML[r, j] + inputs[t, r, j]) + MM[r, j]
                B += raw[t, r, j]
            ratio = max(1 - demand[t, r] / B, 0.) if B > 0 else 0.
            pp = gu[t, r]
            for j in range(ns):
                legpre = ML[r, j] + inputs[t, r, j]
                A = raw[t, r, j] * ratio
                rL = (legpre / raw[t, r, j]) if raw[t, r, j] > 0. else 0.
                nL = A * rL
                T = q_m * nL
                nl_star = nL - T
                pre = A - nl_star
                E = pre * pp
                Lpre = L[r, j] + E * (1. - phi_f[t, r])
                fast[t, r, j] = E * phi_f[t, r]; slow[t, r, j] = Lpre * gs[t, r]
                uptake[t, r, j] = raw[t, r, j] - A
                # The land-side remainder that is NOT exported: N^{L,*} plus N^{M,rem}.
                # `nl_star` is added LAST so that at q_m = 1 the term is exactly round 1's
                # `A * (1. - p)`: `0.0 + av*(1.-pp) = av*(1.-pp)` (0.0 + x == x for x >= 0),
                # and the outer association `(...) * (1. - s[r])` is round 1's.
                loss[t, r, j] = (nl_star + pre * (1. - pp)) * (1. - s[r])
                MM[r, j] = pre * (1. - pp) * s[r]
                ML[r, j] = nl_star * s[r]
                L[r, j] = Lpre - slow[t, r, j]
                ms[t, r, j] = ML[r, j] + MM[r, j]; ls[t, r, j] = L[r, j]
    return fast, slow, raw, ms, ls, uptake, loss


class TransportDP2:
    """Drop-in for `closures.Transport` and for round 1's `TransportDP`: same signature,
    same two outputs, and the scalar travels on the model like the arrays do.

    Plain `@staticmethod`, NOT a `torch.autograd.Function`: there is no `backward`, so an
    attempt to differentiate this reports loudly instead of silently returning the frozen
    adjoint of a DIFFERENT function.
    """

    @staticmethod
    def apply(h, s, f, k, owner):
        hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
        gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
        q_m = _check_q_m()
        own = getattr(owner, 'dp_fractions', None)
        if own is not None:
            for kk_, v in zip(('gu', 'phi_f', 'gs'), (gu, pf, gs)):
                if v.shape != own[kk_].shape or not np.array_equal(v, own[kk_]):
                    raise ValueError('FRACTION_SOURCES_DISAGREE:%s' % kk_)
            if float(own['q_m']) != float(q_m):
                raise ValueError('Q_M_SOURCES_DISAGREE:%r vs %r' % (own['q_m'], q_m))
        out = scan_dp2(hh, ss, ff, kk, owner.data.lower_release,
                       owner.inp, owner.demand, owner.cap, gu, pf, gs, q_m)
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


def scan_dp2_full(h, s, f, k, owner):
    """The audit entry point: the same call, with `h`/`f`/`k`/`l` passed through so an
    auditor can confirm they are genuinely unread.

    FIVE outputs, where round 1's `scan_dp_full` had four: the fifth is `N^{M,pre}` (see
    `scan_dp2`).  This is the ONLY caller that sees it.
    """
    hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
    gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
    q_m = _check_q_m()
    return scan_dp2(hh, ss, ff, kk, owner.data.lower_release,
                    owner.inp, owner.demand, owner.cap, gu, pf, gs, q_m)


def scan_dp2_ledger(h, s, f, k, l, inp, demand, mode):
    """The 4-output entry the bare name `scan` resolves to inside `closures`.

    Only `closures.ResearchObjective.ledger:220` resolves it, and it unpacks exactly four
    values.  Binding this keeps `binding_census` honest.
    """
    hh = np.ascontiguousarray(h)
    gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
    q_m = _check_q_m()
    out = scan_dp2(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                   np.ascontiguousarray(k), l, inp, demand, mode, gu, pf, gs, q_m)
    return out[0], out[1], out[2], out[3]


def tag_scan_dp2(h, s, f, release, inputs, demand):
    """The bare-name entry `campaign_model.Matched.ledger:104` resolves to.

    EXACTLY SIX positional arguments -- round 1's arity, unchanged.  A seventh raises
    `TypeError` and a jit function cannot read the module global instead, so the scalar
    comes through `CONFIG` beside the three sliced arrays.
    """
    hh = np.ascontiguousarray(h)
    rr = _TAG_SLICE['rr']
    if rr is None:
        raise RuntimeError('TAG_SLICE_NOT_INSTALLED')
    gu, pf, gs = _check(hh, ('gu_tag', 'phi_f_tag', 'gs_tag'))
    q_m = _check_q_m()
    for nm, v in (('gu_tag', gu), ('phi_f_tag', pf), ('gs_tag', gs)):
        if v.shape[1] != len(rr):
            raise ValueError('TAG_SLICED_WIDTH:%s %d vs %d' % (nm, v.shape[1], len(rr)))
    return _tag_dp2_nb(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                       release, inputs, demand, gu, pf, gs, q_m)


_TAG_SLICE = {'rr': None}


def set_tag_slice(pilot_indices):
    _TAG_SLICE['rr'] = [int(i) for i in pilot_indices]
    return list(_TAG_SLICE['rr'])


def ledger_dp2(self, x):
    """`closures.ResearchObjective.ledger` for the two-pool kernel.

    MINIMAL, DELIBERATE CHANGE FROM ROUND 1
    ---------------------------------------
    Round 1 substituted `f -> phi_f` in the `L` recurrence only.  This round keeps that
    substitution AND respells the pool term `M = a*(1-p)*s` as the pair

        ML = nl_star * s        MM = pre * (1 - p) * s

    which is the two-pool recurrence, and computes `nl_star`/`pre` from `a`, `p`, `s`,
    `phi_f`, `inp`, `demand` and `q_m` -- NOT from the kernel's returned `fast`/`slow`.

    WHERE THE TEETH ARE  (assertion N3')
    ------------------------------------
    `fast` and `slow` come from the KERNEL'S RETURN.  `ML`, `MM`, `L`, `loss` and `uptake`
    are respelled here.  So `balance` asks "is the flux the kernel returned consistent
    with a separately written two-pool recurrence?" -- that is a real check, and
    `phase0_gates.py` proves it has teeth by injecting `fast := 1.1*Eu*phi_f` and showing
    the balance moves.  It is NOT a physical conservation law: with the kernel's own
    `fast`/`slow` the identity telescopes for ANY admissible fraction triple, which is
    exactly why the injection test is the only thing that gives it teeth.

    The two-pool recurrence is genuinely recursive (`ML_t` feeds `uL_{t+1}`), so unlike
    `L` it cannot be written as a `cumsum`; the loop below is the honest form.  `L` IS
    still a `cumsum`, so the slow path keeps round 1's independent spelling.

    THE SIX EXTRA RETURN FIELDS
    ---------------------------
    `legacy_state`, `mobile_state`, `transfer`, `legacy_pool_before_transfer`,
    `legacy_pool_after_transfer` and `mobile_pre_mobilisation` are diagnostics for
    assertions N2'/N3'/N13 and for the report.  They are NOT part of the frozen `ledger`
    contract (which round 1 already extended), and no gate reads a load from them:
    `transfer` is reported as a RATIO and a CELL COUNT, never as a mass total (plan S0.1:
    no loads are computed or reported).

    `legacy_pool_before_transfer` is `N-tilde^L = av * rL`, the pool the transfer draws
    FROM; `legacy_pool_after_transfer` is `N^{L,*} = N-tilde^L - T`, what it leaves
    behind.  N13's data half needs the FORMER: `T / N-tilde^L` must equal the scalar
    `q_m` on every cell, whereas `T / N^{L,*}` equals `q_m/(1-q_m)` and drifts by
    `q_m^2/(1-q_m)` -- a constant, so a test that divides by the wrong one still reports
    "approximately a single constant" and passes for the wrong reason.
    """
    from routing import route                      # the frozen router, byte for byte
    with torch.no_grad():
        h, s, f, k = [v.numpy() for v in self.flux_parameters(torch.tensor(x))]
    gu, pf, gs = _check(np.ascontiguousarray(h), ('gu', 'phi_f', 'gs'))
    q_m = _check_q_m()
    fast, slow, a, p, _prea = scan_dp2(np.ascontiguousarray(h), np.ascontiguousarray(s),
                                       np.ascontiguousarray(f), np.ascontiguousarray(k),
                                       self.data.lower_release, self.inp, self.demand,
                                       self.cap, gu, pf, gs, q_m)
    nd, nr = a.shape
    S = np.ascontiguousarray(s, np.float64)
    ML = np.zeros_like(a); MM = np.zeros_like(a)
    nL_a = np.zeros_like(a)
    nlstar_a = np.zeros_like(a); pre_a = np.zeros_like(a)
    transfer = np.zeros_like(a)
    MLprev = np.zeros(nr); MMprev = np.zeros(nr)
    for t in range(nd):
        uL = MLprev + self.inp[t]
        uP = uL + MMprev
        rL = np.where(uP > 0.0, uL / np.where(uP > 0.0, uP, 1.0), 0.0)
        nL = a[t] * rL
        nL_a[t] = nL
        transfer[t] = q_m * nL
        nlstar_a[t] = nL - transfer[t]
        pre_a[t] = a[t] - nlstar_a[t]
        sv = S if S.ndim == 1 else S[t]
        MM[t] = pre_a[t] * (1.0 - p[t]) * sv
        ML[t] = nlstar_a[t] * sv
        MLprev = ML[t]; MMprev = MM[t]
    Eu = pre_a * p
    J = Eu * (1.0 - pf)
    L = np.cumsum(J - slow, axis=0)
    beforeL = np.vstack([np.zeros_like(ML[:1]), ML[:-1]])
    beforeM = np.vstack([np.zeros_like(MM[:1]), MM[:-1]])
    # The parenthesisation reproduces the KERNEL's, so `uP - uptake == a` exactly and the
    # availability the kernel returned is the one the ledger balances against.
    uptake = np.minimum((beforeL + self.inp) + beforeM, self.demand)
    # `loss` IS THE LAND-SIDE REMAINDER THAT IS NOT EXPORTED, and spelling it as
    # `(N^{L,*} + N^{M,pre}) * (1 - s)` -- i.e. the WHOLE post-uptake pool -- was WRONG by
    # the exported part of the mobile pool.  Round 1's loss is `a*(1-p)*(1-s)`: the pool
    # MINUS what the mobilisation took out (`a*p = E`).  Here the mobilisation removes
    # `pre*g_u = Eu`, so the remainder is `nlstar + pre*(1-p)` and the loss is that times
    # `(1-s)`.  With the wrong spelling the balance did not telescope and measured
    # `pre*g_u*(s-1)`-sized residuals, 6134.68 kg at the primary arm instead of ~1e-8.
    # `nlstar_a` is added LAST so the parent reduction keeps round 1's association.
    landside = nlstar_a + pre_a * (1.0 - p)
    loss = landside * (1.0 - S)       # (nr,) broadcasts against (nd, nr) when s is 1-D
    river = route(self.data, fast + slow, vf=float(x[2]))
    balance = (self.inp - uptake - loss - fast - slow
               - np.diff(ML + MM + L, axis=0, prepend=np.zeros_like(ML[:1])))
    net = ((fast + slow).sum() - river['channel_removed'].sum()
           - river['terminal'].sum() - river['stocks'][-1].sum())
    return dict(fast=fast, slow=slow, M=ML + MM, L=L, available=a, uptake=uptake,
                demand=self.demand, mineral_loss=loss,
                local_balance_max_kg=float(np.abs(balance).max()),
                network_balance_kg=float(net), terminal=river['terminal'],
                reservoir_stocks=river['stocks'],
                channel_loss=river['channel_removed'],
                legacy_state=ML, mobile_state=MM, transfer=transfer,
                # THREE names for THREE different quantities of the same day.  Earlier
                # drafts put `nL - T` under the name `legacy_pool_before_transfer`, which
                # made `transfer / legacy_pool_before_transfer` read `q_m/(1-q_m)` and
                # turned assertion N13's residual into the constant `q_m^2/(1-q_m)` --
                # a bug that looks exactly like a single-constant law, so it had to be
                # found by algebra rather than by eyeballing the number.
                legacy_pool_before_transfer=nL_a,          # N-tilde^L  (before T)
                legacy_pool_after_transfer=nlstar_a,       # N^{L,*}    (after T)
                mobile_pre_mobilisation=pre_a, q_m=float(q_m),
                ledger_spelling='independent: ML=nl_star*s; MM=pre*(1-gu)*s; '
                                'L=cumsum(pre*gu*(1-phi_f)-F_s); '
                                'loss=(nlstar+pre*(1-gu))*(1-s)')


def assert_no_autograd():
    """A criterion must never be evaluated outside `torch.no_grad()`.

    The frozen adjoints carry the FROZEN `h -> p` map.  `TransportDP2` has no `backward`,
    so differentiating it fails loudly -- asserted here rather than assumed.
    """
    for name, obj in (('TransportDP2', TransportDP2), ('scan_dp2', scan_dp2),
                      ('_tag_dp2_nb', _tag_dp2_nb), ('ledger_dp2', ledger_dp2)):
        if hasattr(obj, 'backward'):
            raise AssertionError('DP2_KERNEL_HAS_BACKWARD:%s' % name)
    return True
