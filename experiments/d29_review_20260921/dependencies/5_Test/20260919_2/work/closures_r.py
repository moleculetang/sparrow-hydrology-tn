"""The minimal inter-event availability kernel: a COPY of the frozen land-phase
recursion with ONE added state and ONE added factor.

WHY THIS IS A COPY AND NOT AN EDIT
----------------------------------
`vendor/research/closures.py` (02433a200de4911f...) and
`vendor/research/tagged_transport.py` (b1a7df8a559e00a4...) are two of the 75
entries in `20260916_2/reports/launch_by_fold/{C0,C2}.json::frozen_hashes`, and
they carry IDENTICAL hashes across the `20260916_2` and `20260918_1` trees.
Editing either file invalidates `fit_worker.py:11-18::identity()` in both trees.
The two functions below are therefore copies living in this round, installed over
the frozen classes by rebinding module attributes in the calling process only.

Both a forward and a tagged copy are required.  The precedent in this lineage for
adding a state by adding a NEW kernel rather than editing `closures.py` is
`vendor/expert/tn_challenge/sc_kernel.py` -- and it can never satisfy the `legal`
gate's conjunct 5 for exactly one reason: it has no tagged counterpart, while
`campaign_model.py:104` sends every source-ledger reading through `tag_scan`
regardless of which forward kernel ran.  A forward-only copy is therefore
guaranteed to report a source-ledger error from the first trial point onward.

THE ONE CHANGE, IN THE SCALAR KERNEL
------------------------------------
    frozen (closures.py:14-29)                    this round
      av   = max(M+inp-demand, 0)                   av   = max(M+inp-demand, 0)   [same]
      risk = h/(av+k) if mode else h                risk = ...                    [same]
      prob = -expm1(-min(risk,700))                 prob = ...                    [same]
      E    = av*prob                                R    = av + (1-delta)*(R-av)  [new state]
                                                    E    = min(av*prob, R)        [only edit]
      M    = av*(1-prob)*survival                   M    = av*(1-prob*kappa)*surv

`prob` IS NOT TOUCHED.  The release TIMING of the land phase is still decided
entirely by the existing hydrologic hazard; `R` produces no flux of its own and
only CEILINGS the amount that may leave.  That is the separation this round is
built on: available mass is not transport opportunity.

WHY THE RECHARGE TARGET IS `av` AND NOT `min(M,av)`
---------------------------------------------------
A pre-implementation check of the no-op identity killed the draft form
`R = min(R + delta*(min(M,av) - R), M)`, registered as deviation D5.  On a
month-first injection day `av = M + inp - demand` can exceed `M` substantially, so
the outer `min(., M)` ceiling can hold `R` BELOW `av*prob`; at `delta = 1` the
kernel would then return `E = M < av*prob` and NOT be a no-op.  That would destroy
the envelope's baseline, because every gain is measured against the same code path
at its null point.  The form used here,

    R = av + (1-delta)*(R-av)

is a convex interpolation between the previous `R` and `av`, so:
  * `delta = 1`  -> `R` is assigned `av + 0.0*(R-av)`, which is bitwise `av`
                    (0.0*finite is a signed zero and `av + -0.0 == av`), hence
                    `E = min(av*prob, av) = av*prob` because `prob <= 1`, hence
                    `kappa = 1.0`, hence `prob*kappa == prob` exactly.  The no-op
                    is exact and is asserted with np.array_equal, not allclose.
  * `delta = 0`  -> `R` never moves: the limit case in which nothing is mobilizable.
  * `R` stays in `[0, av]` for every `delta` in `[0,1]` by convexity, and `av >= 0`.

MASS CONSERVATION IS EXACT FOR EVERY `delta`, AND IS NOT A CLAIM ABOUT `R`
-------------------------------------------------------------------------
`R` holds no mass.  Because `E <= av*prob`, the retention term becomes
`av - E = av*(1 - prob*kappa)` and the frozen ledger identity

    inp - uptake - loss - fast - slow - diff(M+L) == 0

telescopes to zero identically for any `kappa` in `[0,1]`:

    balance = av - av*(1-q) - av*q*f - pre + L_prev
            = av - av*(1-q) - av*q*f - L_prev - av*q*(1-f) + L_prev
            = av - av*(1-q) - av*q  =  0,     q := prob*kappa

So `local_balance_max_kg` is unchanged BY CONSTRUCTION rather than by tolerance, and
no channel is added to any ledger.  This differs from the approved plan's
justification line ("R is a re-partition of M") and is registered as deviation D4:
the implemented object is a bound on mobilizable `av`, which is a STRONGER
conservation statement than a re-partition, not a weaker one.  `ledger` carries no
`R` term because there is no `R` storage to carry.

WHY THE CAP IS A SCALAR, AND WHY THAT NEEDS NO NEW TAG CHANNEL
--------------------------------------------------------------
`E_j = A_j * p * kappa` with the SAME `kappa` for every channel j is the exact
shape of `tag_scan`'s own demand handling (`ratio = max(1-demand/B,0)` then
`A = raw*ratio`), so

    sum_j tags[fast][t,r,:] = kappa*p*f*sum_j A_j = kappa*p*f*A_total ~ fast[t,r]

holds to the same rounding that already separates the tagged and scalar ledgers.
`campaign_model.py:72-74`'s 4-channel `tag_inputs` shape and `SOURCE_TAG_IDENTITY`
are untouched because this design splits no input: `inp` and `demand` reach `M`/`L`
at full value, exactly as before.
"""
import numpy as np
import torch
from numba import njit

# Single source of truth for the tagged path, which cannot take a call argument:
# `campaign_model.Matched.ledger` calls the bare name `tag_scan`.
CONFIG = {'delta': 1.0, 'r_init': 0.0}


def set_config(delta, r_init=0.0):
    CONFIG['delta'] = float(delta)
    CONFIG['r_init'] = float(r_init)
    return dict(CONFIG)


@njit(cache=True)
def scan_r(h, s, f, k, l, inp, demand, mode, delta, r_init):
    """`closures.scan` verbatim -- same name, same order, same outputs -- plus `R`.

    Signature is positional-identical to `closures.scan(h,s,f,k,l,inp,demand,mode)`.
    Returns (fast, slow, av, prob, kappa, R); `TransportR` uses the first two and
    the audit uses the rest.
    """
    nd, nr = h.shape
    fast = np.zeros_like(h); slow = np.zeros_like(h); a = np.zeros_like(h); p = np.zeros_like(h)
    kappa = np.ones_like(h); rrec = np.zeros_like(h); praw = np.zeros_like(h)
    M = np.zeros(nr); L = np.zeros(nr); R = np.full(nr, r_init)
    for t in range(nd):
        for r in range(nr):
            av = max(M[r] + inp[t, r] - demand[t, r], 0.)
            risk = h[t, r] / (av + k[r]) if mode else h[t, r]
            prob = -np.expm1(-min(risk, 700.))
            # inter-event recharge only: no flux of its own, no new timing operator
            R[r] = av + (1.0 - delta) * (R[r] - av)
            efull = av * prob
            E = efull if efull < R[r] else R[r]
            kap = E / efull if efull > 0. else 1.
            kappa[t, r] = kap
            fast[t, r] = E * f[t, r]
            pre = L[r] + E * (1 - f[t, r]); slow[t, r] = pre * l[t, r]
            survival = s[r] if s.ndim == 1 else s[t, r]
            M[r] = av * (1 - prob * kap) * survival
            L[r] = pre - slow[t, r]
            # `p` IS THE EFFECTIVE PROBABILITY, and it must be.  The frozen
            # `closures.ResearchObjective.ledger:221-223` derives the retention and
            # mineralisation channels from `a` and this `p` alone --
            #     M = a*(1-p)*s;  loss = a*(1-p)*(1-s);  L = cumsum(a*p*(1-f) - slow)
            # so returning the raw `prob` would leave the cap out of the retention
            # term while `fast`/`slow` carried it, and `local_balance_max_kg` would
            # then measure the CAP and not the frozen rounding.  With p := prob*kap
            # the frozen ledger telescopes to zero again, exactly:
            #     balance = av - av(1-q)[(1-s)+s] - av*q*f - av*q*(1-f) - av(1-q)s ... = 0
            # `prob*kap` is bitwise `prob` whenever kap is 1.0, which is what keeps
            # the delta=1 no-op exact.
            a[t, r] = av; p[t, r] = prob * kap
            rrec[t, r] = R[r]; praw[t, r] = prob
    return fast, slow, a, p, kappa, rrec, praw


@njit(cache=True)
def _tag_scan_r_nb(h, s, f, release, inputs, demand, delta, r_init):
    """`tagged_transport.tag_scan` verbatim -- same order, same outputs -- plus the cap.

    `tag_scan` hard-codes the scalar kernel's `mode=False` branch
    (`p = -expm1(-min(h,700))`, no `av+k`), which is the branch this round runs, so
    the two kernels see the same `prob` by construction.
    """
    nd, nr = h.shape; ns = inputs.shape[2]
    fast = np.zeros_like(inputs); slow = np.zeros_like(inputs); raw = np.zeros_like(inputs)
    ms = np.zeros_like(inputs); ls = np.zeros_like(inputs)
    uptake = np.zeros_like(inputs); loss = np.zeros_like(inputs)
    M = np.zeros((nr, ns)); L = np.zeros((nr, ns)); R = np.full(nr, r_init)
    for t in range(nd):
        for r in range(nr):
            B = 0.
            for j in range(ns):
                raw[t, r, j] = M[r, j] + inputs[t, r, j]; B += raw[t, r, j]
            ratio = max(1 - demand[t, r] / B, 0.) if B > 0 else 0.
            p = -np.expm1(-min(h[t, r], 700.))
            atot = ratio * B
            R[r] = atot + (1.0 - delta) * (R[r] - atot)
            efull = atot * p
            E = efull if efull < R[r] else R[r]
            kap = E / efull if efull > 0. else 1.
            for j in range(ns):
                A = raw[t, r, j] * ratio
                Ej = A * p * kap
                pre = L[r, j] + Ej * (1 - f[t, r])
                fast[t, r, j] = Ej * f[t, r]; slow[t, r, j] = pre * release[t, r]
                uptake[t, r, j] = raw[t, r, j] - A
                loss[t, r, j] = A * (1 - p * kap) * (1 - s[r])
                M[r, j] = A * (1 - p * kap) * s[r]; L[r, j] = pre - slow[t, r, j]
                ms[t, r, j] = M[r, j]; ls[t, r, j] = L[r, j]
    return fast, slow, raw, ms, ls, uptake, loss


class TransportR:
    """Drop-in for `closures.Transport`: same signature, same two outputs.

    The frozen call site is `Transport.apply(h,s,f,k,self)`, so `owner` is the model
    and the new parameters travel on the model object -- no call site changes.
    """

    @staticmethod
    def apply(h, s, f, k, owner):
        d = float(getattr(owner, 'R_delta', CONFIG['delta']))
        r0 = float(getattr(owner, 'R_init', CONFIG['r_init']))
        hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
        out = scan_r(hh, ss, ff, kk, owner.data.lower_release,
                     owner.inp, owner.demand, owner.cap, d, r0)
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


def scan_r_full(h, s, f, k, owner, delta=None, r_init=0.0):
    """The audit entry point: the same call, but `kappa` and `R` are kept."""
    d = float(CONFIG['delta'] if delta is None else delta)
    hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
    return scan_r(hh, ss, ff, kk, owner.data.lower_release,
                  owner.inp, owner.demand, owner.cap, d, float(r_init))


def scan_r_ledger(h, s, f, k, l, inp, demand, mode):
    """The 4-output entry `closures.ResearchObjective.ledger:220` resolves to.

    `ledger` unpacks exactly four values, so this wrapper drops `kappa` and `R`.
    Without this binding the ledger's SCALAR channel would come from the frozen
    `scan` while its TAGGED channel came from the capped `tag_scan_r`, and
    `source_label_sum_errors` would then measure the cap itself rather than the
    tagged/scalar rounding it is registered to measure.
    """
    out = scan_r(np.ascontiguousarray(h), np.ascontiguousarray(s),
                 np.ascontiguousarray(f), np.ascontiguousarray(k),
                 l, inp, demand, mode, CONFIG['delta'], CONFIG['r_init'])
    return out[0], out[1], out[2], out[3]


def tag_scan_r(h, s, f, release, inputs, demand):
    """The bare-name entry `campaign_model.Matched.ledger:104` resolves to.

    It cannot receive `delta` as an argument, so it reads `CONFIG`, which
    `common20.ledger_with` sets and then asserts before every ledger call.
    """
    return _tag_scan_r_nb(np.ascontiguousarray(h), np.ascontiguousarray(s),
                          np.ascontiguousarray(f), release, inputs, demand,
                          CONFIG['delta'], CONFIG['r_init'])
