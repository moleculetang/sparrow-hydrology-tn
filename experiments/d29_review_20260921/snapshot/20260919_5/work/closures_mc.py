"""The pre-mobilisation pathway-concentration kernel: a COPY of the frozen
land-phase recursion with ONE added factor on the HAZARD, and NOTHING else.

WHY THIS IS A COPY AND NOT AN EDIT
----------------------------------
`vendor/research/closures.py` and `vendor/research/tagged_transport.py` are two of
the entries in `20260916_2/reports/launch_by_fold/{C0,C2}.json::frozen_hashes`, and
they carry IDENTICAL hashes across the `20260916_2` and `20260918_1` trees.  Editing
either file invalidates `fit_worker.py:11-18::identity()` in both trees.  The
functions below are therefore copies living in this round, installed over the frozen
names by rebinding module attributes in the calling process only.

Both a forward and a tagged copy are required, for the reason recorded in
`20260919_2/work/closures_r.py:14-20`: `campaign_model.py:104` sends every
source-ledger reading through the bare name `tag_scan` regardless of which forward
kernel ran, so a forward-only copy reports a source-ledger error from the first
trial point onward, and that error would be the modification itself.

THE ONE CHANGE, IN BOTH KERNELS
-------------------------------
    frozen (closures.py:14-29)              this round
      risk = h/(av+k) if mode else h          risk = ...                       [same]
      prob = -expm1(-min(risk, 700))          prob = -expm1(-min(risk*Xi, 700)) [ONE edit]
      E    = av*prob                          E    = av*prob                   [same]
      fast = E*f                              fast = E*f                       [same]
      pre  = L + E*(1-f);  slow = pre*l       pre  = ...                       [same]
      M    = av*(1-prob)*survival             M    = ...                       [same]
      L    = pre - slow                       L    = ...                       [same]

    frozen (tagged_transport.py:6-23)       this round
      p = -expm1(-min(h[t,r], 700.))          p = -expm1(-min(h[t,r]*Xi[t,r], 700.))

This round runs `cap=False`, so `mode=False` and `risk = h[t,r]`: the `av+k` branch is
present for signature fidelity only and is never taken.  `f` IS NOT TOUCHED -- the
user's governing instruction for this round is explicitly "不要再改 `f`".  `M`/`L`,
the source ledger, the hydrology and the routing are all untouched.

WHAT THE EDIT MEANS, AND WHY IT IS NOT A REPARAMETERISATION
-----------------------------------------------------------
`f` is frozen as `aq*Qf/(aq*Qf + Qs)` with `aq = e^{t[20]} = 1.0155379765842147`,
which is algebraically ALREADY the two-path expression
`C_fast*Qf/(C_fast*Qf + C_slow*Qs)` for a CONSTANT contrast `C_fast/C_slow = aq`.
So any `Xi` that is (near-)constant merely re-parameterises the fitted `aq` and
proves nothing; `xi.admissible`'s degeneracy floor `frac(|Xi-1| > 1e-6) >= 1e-3` is
the hard gate that guards against exactly that, and a degenerate `Xi` is reported as
`BLOCKED`, NOT as evidence against the mechanism.

`Xi` multiplies the hazard INSIDE the exponential, so it changes how much N the day
can mobilise AT ALL -- upstream of the fast/slow split -- which is the structural
layer this round exists to test.  `Xi = 1` is a BITWISE no-op (`xi.py`'s frozen
spelling; red-team measured `min(h*1.0,700) == min(h,700)` on all 5,376,480 cells).

WHY `Xi` IS A CALL ARGUMENT AND NEVER A MODULE GLOBAL IN THE JIT LAYER
---------------------------------------------------------------------
numba treats module globals as COMPILE-TIME CONSTANTS: rebinding a global of the same
type does not force a recompile, so the cached compiled artefact keeps the OLD
reference and the "installed" array is silently ignored.  `CONFIG` is therefore read
only in the non-jit Python wrappers below, exactly as `closures_r.py:107/:151` pass
`delta`/`r_init` as arguments and read `CONFIG` only at `:196/:223/:235`.

WHY THE CLIP IS ON THE PRODUCT HERE TOO
---------------------------------------
The frozen cap `min(risk, 700)` is retained VERBATIM as `min(risk*Xi, 700)`.  `Xi` is
kept finite and strictly positive by `xi.py` (its clip is on the EXPONENT), so the
product `0.0 * Xi` is exactly `0.0` and no NaN can enter `prob`, `E`, or the carried
state `M[r]`.  Clipping `Xi` here instead would be a different operator.

NO AUTOGRAD, AND THAT IS ASSERTED
---------------------------------
`TransportMC` is a plain `@staticmethod` with no `backward`.  Every criterion in this
round runs inside `torch.no_grad()`.  The frozen adjoints (`closures.py:45-52::reverse`,
`tagged_transport.py:32::tag_reverse`) each carry their own UNMODULATED `h -> p` map;
if the modulated kernel were ever differentiated, the gradient would silently NOT be
the gradient of the function being evaluated.  `assert_no_autograd()` makes that
loud, and `common22.install_kernel` calls it.
"""
import numpy as np
import torch
from numba import njit

# Single source of truth for the entries that cannot take a call argument.
# `closures.ResearchObjective.ledger:220` resolves the BARE name `scan`, and
# `campaign_model.Matched.ledger:104` resolves the BARE name `tag_scan` with exactly
# SIX positional arguments -- so neither can be handed `Xi` at the call site.
CONFIG = {'Xi': None, 'Xi_tag': None}


def set_config(Xi, Xi_tag):
    CONFIG['Xi'] = Xi
    CONFIG['Xi_tag'] = Xi_tag
    return {'Xi_shape': None if Xi is None else list(Xi.shape),
            'Xi_tag_shape': None if Xi_tag is None else list(Xi_tag.shape)}


def _check(owner, h, what):
    """Return the installed `Xi`, assert it agrees with `owner.Xi`, assert the shape.

    numba performs NO cross-argument shape check, so a `Xi` of the wrong shape is not
    a crash -- it is a silent read of the wrong columns.  Every Python wrapper goes
    through here so that every entry point is guarded.
    """
    X = CONFIG[what]
    if X is None:
        raise RuntimeError('XI_NOT_INSTALLED:%s' % what)
    own = getattr(owner, 'Xi' if what == 'Xi' else 'Xi_tag', None)
    if own is not None and own is not X:
        if own.shape != X.shape or not np.array_equal(own, X):
            raise ValueError('XI_SOURCES_DISAGREE:%s' % what)
    if X.shape != h.shape:
        raise ValueError('XI_SHAPE_MISMATCH:%s %r vs h %r' % (what, X.shape, h.shape))
    return X


@njit(cache=True)
def scan_mc(h, s, f, k, l, inp, demand, mode, Xi):
    """`closures.scan` verbatim -- same name, same order, same four outputs.

    Diagnostic extra outputs `hh`/`pr` (the modulated `risk*Xi` and the raw pre-cap
    product) are appended so an audit can see what the cap did without re-deriving it.
    `TransportMC` uses only the first two.

    `p` RETURNED IS THE MODULATED `prob`.  This is the `closures_r.py:134-145` lesson:
    `ResearchObjective.ledger:221-223` derives retention and mineralisation from `a`
    and this `p` alone (`M = a*(1-p)*s; loss = a*(1-p)*(1-s); L = cumsum(a*p*(1-f) - slow)`),
    so returning anything else would put the modification into `fast`/`slow` but not
    into the retention term, and `local_balance_max_kg` would measure the modification
    instead of the frozen float64 rounding.  With `p := prob` the ledger telescopes to
    zero for ANY `Xi`, by construction.
    """
    nd, nr = h.shape
    fast = np.zeros_like(h); slow = np.zeros_like(h); a = np.zeros_like(h); p = np.zeros_like(h)
    risk_out = np.zeros_like(h); product = np.zeros_like(h)
    M = np.zeros(nr); L = np.zeros(nr)
    for t in range(nd):
        for r in range(nr):
            av = max(M[r] + inp[t, r] - demand[t, r], 0.)
            risk = h[t, r] / (av + k[r]) if mode else h[t, r]
            prod = risk * Xi[t, r]
            prob = -np.expm1(-min(prod, 700.))
            E = av * prob; fast[t, r] = E * f[t, r]
            pre = L[r] + E * (1 - f[t, r]); slow[t, r] = pre * l[t, r]
            survival = s[r] if s.ndim == 1 else s[t, r]
            M[r] = av * (1 - prob) * survival; L[r] = pre - slow[t, r]
            a[t, r] = av; p[t, r] = prob
            risk_out[t, r] = risk; product[t, r] = prod
    return fast, slow, a, p, risk_out, product


@njit(cache=True)
def _tag_scan_mc_nb(h, s, f, release, inputs, demand, Xi):
    """`tagged_transport.tag_scan` verbatim -- same order, same seven outputs.

    `tag_scan` hard-codes the scalar kernel's `mode=False` branch
    (`p = -expm1(-min(h,700))`, no `av+k`), which is the branch this round runs, so
    the two kernels see the same `prob` by construction even though this one does not
    receive `mode`.

    `h` here is ALREADY column-sliced: `campaign_model.py:104` passes `h[:,rr]` with
    `rr = data.pilot_indices`.  `Xi` must be sliced the same way -- see
    `xi.tag_slice` and `tag_scan_mc` below.
    """
    nd, nr = h.shape; ns = inputs.shape[2]
    fast = np.zeros_like(inputs); slow = np.zeros_like(inputs); raw = np.zeros_like(inputs)
    ms = np.zeros_like(inputs); ls = np.zeros_like(inputs)
    uptake = np.zeros_like(inputs); loss = np.zeros_like(inputs)
    M = np.zeros((nr, ns)); L = np.zeros((nr, ns))
    for t in range(nd):
        for r in range(nr):
            B = 0.
            for j in range(ns):
                raw[t, r, j] = M[r, j] + inputs[t, r, j]; B += raw[t, r, j]
            ratio = max(1 - demand[t, r] / B, 0.) if B > 0 else 0.
            p = -np.expm1(-min(h[t, r] * Xi[t, r], 700.))
            for j in range(ns):
                A = raw[t, r, j] * ratio; E = A * p; pre = L[r, j] + E * (1 - f[t, r])
                fast[t, r, j] = E * f[t, r]; slow[t, r, j] = pre * release[t, r]
                uptake[t, r, j] = raw[t, r, j] - A; loss[t, r, j] = A * (1 - p) * (1 - s[r])
                M[r, j] = A * (1 - p) * s[r]; L[r, j] = pre - slow[t, r, j]
                ms[t, r, j] = M[r, j]; ls[t, r, j] = L[r, j]
    return fast, slow, raw, ms, ls, uptake, loss


class TransportMC:
    """Drop-in for `closures.Transport`: same signature, same two outputs.

    The frozen call sites are all `Transport.apply(h, s, f, k, self)` -- five of them,
    every one passing `self` -- so `owner` is the model and the new parameter travels
    on the model object with NO call-site change.

    Plain `@staticmethod`, NOT a `torch.autograd.Function`: there is no `backward`, so
    an attempt to differentiate this reports loudly instead of silently returning the
    frozen UNMODULATED adjoint.
    """

    @staticmethod
    def apply(h, s, f, k, owner):
        hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
        X = _check(owner, hh, 'Xi')
        out = scan_mc(hh, ss, ff, kk, owner.data.lower_release,
                      owner.inp, owner.demand, owner.cap, np.ascontiguousarray(X))
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


def scan_mc_full(h, s, f, k, owner):
    """The audit entry point: the same call, but `risk` and the pre-cap product kept."""
    hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
    X = _check(owner, hh, 'Xi')
    return scan_mc(hh, ss, ff, kk, owner.data.lower_release,
                   owner.inp, owner.demand, owner.cap, np.ascontiguousarray(X))


def scan_mc_ledger(h, s, f, k, l, inp, demand, mode):
    """The 4-output entry `closures.ResearchObjective.ledger:220` resolves to.

    `ledger` unpacks exactly four values, so this wrapper drops `risk`/`product`.
    Without this binding the ledger's SCALAR channel would come from the frozen `scan`
    while its TAGGED channel came from the modulated `tag_scan_mc`, and
    `source_label_sum_errors` would then measure the modulation itself rather than the
    tagged/scalar rounding it is registered to measure.
    """
    hh = np.ascontiguousarray(h)
    X = CONFIG['Xi']
    if X is None:
        raise RuntimeError('XI_NOT_INSTALLED:Xi')
    if X.shape != hh.shape:
        raise ValueError('XI_SHAPE_MISMATCH:Xi %r vs h %r' % (X.shape, hh.shape))
    out = scan_mc(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                  np.ascontiguousarray(k), l, inp, demand, mode,
                  np.ascontiguousarray(X))
    return out[0], out[1], out[2], out[3]


def tag_scan_mc(h, s, f, release, inputs, demand):
    """The bare-name entry `campaign_model.Matched.ledger:104` resolves to.

    EXACTLY SIX positional arguments, because that is all the frozen call site
    supplies.  Red-team finding 4: a seven-argument install raises `TypeError`, and a
    jit function cannot read the module global instead.  So `Xi` travels through
    `CONFIG['Xi_tag']`, which is `Xi[:, pilot_indices]` -- sliced ONCE by
    `xi.tag_slice` and asserted here against the already-sliced `h`.
    """
    hh = np.ascontiguousarray(h)
    X = CONFIG['Xi_tag']
    if X is None:
        raise RuntimeError('XI_NOT_INSTALLED:Xi_tag')
    if X.shape != hh.shape:
        raise ValueError('XI_SHAPE_MISMATCH:Xi_tag %r vs h[:,pilot] %r'
                         % (X.shape, hh.shape))
    return _tag_scan_mc_nb(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                           release, inputs, demand, np.ascontiguousarray(X))


def assert_no_autograd():
    """A criterion must never be evaluated outside `torch.no_grad()`.

    The frozen adjoints carry an UNMODULATED `h -> p` map.  `TransportMC` has no
    `backward`, so differentiating it fails loudly -- but the failure mode is bad
    enough that it is asserted here rather than assumed.
    """
    for name, obj in (('TransportMC', TransportMC), ('scan_mc', scan_mc),
                      ('_tag_scan_mc_nb', _tag_scan_mc_nb)):
        if hasattr(obj, 'backward'):
            raise AssertionError('MODULATED_KERNEL_HAS_BACKWARD:%s' % name)
    if isinstance(TransportMC.apply, torch.autograd.Function):
        raise AssertionError('MODULATED_KERNEL_IS_AUTOGRAD_FUNCTION')
    return True
