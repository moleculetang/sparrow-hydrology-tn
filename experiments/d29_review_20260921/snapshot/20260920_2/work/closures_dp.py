"""The conserving dual-pathway mobile-water-concentration recursion: a COPY of the
frozen land-phase recursion with the mobilisation term REPLACED, and nothing else.

WHY A COPY AND NOT AN EDIT
--------------------------
`vendor/research/closures.py` and `vendor/research/tagged_transport.py` are entries in
`20260916_2/reports/launch_by_fold/{C0,C2}.json::frozen_hashes` and carry identical
hashes across the `20260916_2` and `20260918_1` trees.  Editing either invalidates
`fit_worker.py:11-18::identity()` in both.  The functions below are copies living in
this round, installed over the frozen names by rebinding module attributes in the
calling process only.

THE ONE CHANGE
--------------
    frozen (closures.py:14-29)                   this round
      risk = h/(av+k) if mode else h               [DELETED -- no hazard at all]
      prob = -expm1(-min(risk, 700.))              g_u = <argument, see dp_kernel>
      E    = av*prob                               Eu = av*g_u
      fast = E*f                                   F_f = Eu*phi_f
      pre  = L + E*(1-f)                           pre = L + Eu*(1-phi_f)
      slow = pre*l                                 F_s = pre*g_s
      M    = av*(1-prob)*survival                  M = av*(1-g_u)*survival
      L    = pre - slow                            L = pre - F_s

`h` is not used.  `f` is not used.  `l` (`lower_release`) is not used.  The three
replacement fractions arrive as ARGUMENTS, which is what makes assertion N1 possible.

WHY THE ARGUMENTS ARE ARGUMENTS AND NOT MODULE GLOBALS
------------------------------------------------------
numba treats module globals as COMPILE-TIME CONSTANTS: rebinding a global of the same
type does not force a recompile, so a cached compiled artefact keeps the OLD reference
and the "installed" array is silently ignored.  `closures_mc.py:53-59` records the same
lesson for round 5's `Xi`.  `CONFIG` is read only in the non-jit Python wrappers.

N1: THE BITWISE REDUCTION, AND WHY THE SPELLING IS WHAT IT IS
------------------------------------------------------------
Injecting

    g_u   := p_frozen        (the frozen scan's 4th output)
    phi_f := f_frozen        (the model's own `f`)
    g_s   := l_frozen        (the model's own `lower_release`)

must reproduce the frozen kernel BITWISE, not approximately.  That constrains the
association order, so the body below is written to match `closures.py:14-29` term for
term:

    Eu  = av * g_u          <->  E    = av * prob
    F_f = Eu * phi_f        <->  fast = E * f            (both are (av*gu)*f)
    J   = Eu * (1.0 - phi_f)<->  E * (1-f)               (binding a name is free)
    pre = L[r] + J          <->  pre = L[r] + E*(1-f)
    F_s = pre * g_s         <->  slow = pre * l
    M   = av * (1.0-g_u)*survival <-> av*(1-prob)*survival

`dp_kernel`'s `guard_from` sets `g_u = 0` exactly where `fast_fraction == 0`, which is
also exactly where the frozen `prob` is already `0`.  So applying the guard and
injecting `p_frozen` produce the SAME array, and N1 is not weakened by the guard.

NO `min`, NO CAP, NO FLOOR
--------------------------
The frozen `min(risk, 700.)` guarded the exponential's overflow.  Here `g_u = Q/V <= 1`
is a theorem from the timing gate (`probe_prevolume.json::g_decision`), and `phi_f` is a
water ratio in `(0,1)`, so there is nothing to cap.  Adding one would hide a timing
error rather than expose it.  `dp_kernel.fractions` ASSERTS the ranges instead.
"""
import numpy as np
import torch
from numba import njit

# The three fractions cannot travel as call arguments at two of the entry points, for
# the reason recorded in `closures_mc.py:82-84`:
#   `closures.ResearchObjective.ledger` resolves the BARE name `scan` with exactly EIGHT
#   positional arguments, and `campaign_model.Matched.ledger:104` resolves the BARE name
#   `tag_scan` with exactly SIX.  Neither can be handed the fractions at the call site.
CONFIG = {'gu': None, 'phi_f': None, 'gs': None,
          'gu_tag': None, 'phi_f_tag': None, 'gs_tag': None}


def set_config(gu, phi_f, gs, gu_tag=None, phi_f_tag=None, gs_tag=None):
    CONFIG.update(gu=gu, phi_f=phi_f, gs=gs,
                  gu_tag=gu_tag, phi_f_tag=phi_f_tag, gs_tag=gs_tag)
    return {k: (None if v is None else list(v.shape)) for k, v in CONFIG.items()}


def _shape(v):
    return None if v is None else tuple(v.shape)


def _check(h, keys):
    """Assert every named CONFIG array is installed, agrees, and matches `h`'s shape.

    numba performs NO cross-argument shape check, so a fraction array of the wrong shape
    is not a crash -- it is a silent read of the wrong columns (`xi_base.tag_slice`'s
    standing warning).  Every Python wrapper goes through here so every entry point is
    guarded, once per array, as assertion N11 requires.
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


@njit(cache=True)
def scan_dp(h, s, f, k, l, inp, demand, mode, gu, phi_f, gs):
    """`closures.scan` with the mobilisation replaced.  Same name, same order, same
    shape of return (the frozen four first, in the frozen order).

    `h`, `f`, `k`, `l` and `mode` are retained in the SIGNATURE for call-site fidelity
    and are NOT READ.  `k` and `l` are in fact passed by every frozen call site, so
    dropping them would change `Transport.apply`'s arity; `h` and `f` are the frozen
    carrier of the mobilisation and the split, and both are now supplied by the volumes.

    THE RETURNED `p` IS `g_u`.  This matters: `ResearchObjective.ledger:221` derives the
    retention term from `a` and `p` alone (`M = a*(1-p)*s`), so returning anything else
    would put the replacement into `fast`/`slow` but not into retention, and the ledger
    would measure the replacement rather than the rounding.  With `p := g_u` the ledger
    telescopes to zero for any admissible fraction triple.
    """
    nd, nr = h.shape
    fast = np.zeros_like(h); slow = np.zeros_like(h)
    a = np.zeros_like(h); p = np.zeros_like(h)
    M = np.zeros(nr); L = np.zeros(nr)
    for t in range(nd):
        for r in range(nr):
            av = max(M[r] + inp[t, r] - demand[t, r], 0.)
            Eu = av * gu[t, r]
            fast[t, r] = Eu * phi_f[t, r]
            J = Eu * (1. - phi_f[t, r])
            pre = L[r] + J
            slow[t, r] = pre * gs[t, r]
            survival = s[r] if s.ndim == 1 else s[t, r]
            M[r] = av * (1. - gu[t, r]) * survival
            L[r] = pre - slow[t, r]
            a[t, r] = av; p[t, r] = gu[t, r]
    return fast, slow, a, p


@njit(cache=True)
def _tag_dp_nb(h, s, f, release, inputs, demand, gu, phi_f, gs):
    """`tagged_transport.tag_scan` with the mobilisation replaced.  Same order, same
    seven outputs.  `h`, `f` and `release` are retained and unread.

    `h` here is ALREADY column-sliced by `campaign_model.py:104` (`h[:, rr]`,
    `rr = data.pilot_indices`), so the three fraction arrays must be sliced the same way
    -- done once in `tag_scan_dp` below, exactly as round 5 sliced its `Xi`.
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
            p = gu[t, r]
            for j in range(ns):
                A = raw[t, r, j] * ratio; E = A * p
                pre = L[r, j] + E * (1. - phi_f[t, r])
                fast[t, r, j] = E * phi_f[t, r]; slow[t, r, j] = pre * gs[t, r]
                uptake[t, r, j] = raw[t, r, j] - A
                loss[t, r, j] = A * (1. - p) * (1. - s[r])
                M[r, j] = A * (1. - p) * s[r]; L[r, j] = pre - slow[t, r, j]
                ms[t, r, j] = M[r, j]; ls[t, r, j] = L[r, j]
    return fast, slow, raw, ms, ls, uptake, loss


class TransportDP:
    """Drop-in for `closures.Transport`: same signature, same two outputs.

    Every frozen call site is `Transport.apply(h, s, f, k, self)`, so `owner` is the
    model and the fraction arrays travel on the model object with NO call-site change.

    Plain `@staticmethod`, NOT a `torch.autograd.Function`: there is no `backward`, so
    an attempt to differentiate this reports loudly instead of silently returning the
    frozen adjoint of a DIFFERENT function.
    """

    @staticmethod
    def apply(h, s, f, k, owner):
        hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
        gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
        own = getattr(owner, 'dp_fractions', None)
        if own is not None:
            for kk_, v in zip(('gu', 'phi_f', 'gs'), (gu, pf, gs)):
                if v.shape != own[kk_].shape or not np.array_equal(v, own[kk_]):
                    raise ValueError('FRACTION_SOURCES_DISAGREE:%s' % kk_)
        out = scan_dp(hh, ss, ff, kk, owner.data.lower_release,
                      owner.inp, owner.demand, owner.cap, gu, pf, gs)
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])


def scan_dp_full(h, s, f, k, owner):
    """The audit entry point: the same call, with `h`/`f`/`k`/`l` passed through so an
    auditor can confirm they are genuinely unread."""
    hh, ss, ff, kk = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, k)]
    gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
    return scan_dp(hh, ss, ff, kk, owner.data.lower_release,
                   owner.inp, owner.demand, owner.cap, gu, pf, gs)


def scan_dp_ledger(h, s, f, k, l, inp, demand, mode):
    """The 4-output entry the bare name `scan` resolves to inside `closures`.

    Only `closures.ResearchObjective.ledger:220` resolves it, and it unpacks exactly four
    values.  Binding this keeps `binding_census` honest: without it `closures.scan` would
    still be the frozen object while the tagged channel came from `_tag_dp_nb`, and
    `source_label_sum_errors` would measure the replacement rather than the rounding.
    """
    hh = np.ascontiguousarray(h)
    gu, pf, gs = _check(hh, ('gu', 'phi_f', 'gs'))
    out = scan_dp(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                  np.ascontiguousarray(k), l, inp, demand, mode, gu, pf, gs)
    return out[0], out[1], out[2], out[3]


def tag_scan_dp(h, s, f, release, inputs, demand):
    """The bare-name entry `campaign_model.Matched.ledger:104` resolves to.

    EXACTLY SIX positional arguments, because that is all the frozen call site supplies;
    a seventh raises `TypeError`, and a jit function cannot read the module global
    instead.  So the fractions travel through `CONFIG`, sliced ONCE here by the model's
    own `pilot_indices` and asserted against the already-sliced `h`.
    """
    hh = np.ascontiguousarray(h)
    rr = _TAG_SLICE['rr']
    if rr is None:
        raise RuntimeError('TAG_SLICE_NOT_INSTALLED')
    gu, pf, gs = _check(hh, ('gu_tag', 'phi_f_tag', 'gs_tag'))
    for nm, v in (('gu_tag', gu), ('phi_f_tag', pf), ('gs_tag', gs)):
        if v.shape[1] != len(rr):
            raise ValueError('TAG_SLICED_WIDTH:%s %d vs %d' % (nm, v.shape[1], len(rr)))
    return _tag_dp_nb(hh, np.ascontiguousarray(s), np.ascontiguousarray(f),
                      release, inputs, demand, gu, pf, gs)


_TAG_SLICE = {'rr': None}


def set_tag_slice(pilot_indices):
    _TAG_SLICE['rr'] = [int(i) for i in pilot_indices]
    return list(_TAG_SLICE['rr'])


def ledger_dp(self, x):
    """`closures.ResearchObjective.ledger`, with `f` -> `phi_f` in the L recurrence.

    MINIMAL, DELIBERATE CHANGE
    --------------------------
    The frozen body (`closures.py:215-229`) is reproduced term for term except line 221,
    where `a*p*(1-f)` becomes `a*p*(1-phi_f)`.  That substitution is not cosmetic: this
    round's split is `phi_f = Q_f/Q_u`, a water share, while `f` is the frozen fitted
    `aq*Qf/(aq*Qf + Qs)`.  Leaving `f` in would put a DIFFERENT split into the `L`
    recurrence than the kernel used, and `local_balance_max_kg` would then measure the
    split difference instead of the float64 rounding it is registered to measure.

    WHERE THE TEETH ARE  (plan R1', assertion N3)
    ---------------------------------------------
    `fast` and `slow` come from the KERNEL'S RETURN.  `M`, `L` and `loss` are re-spelled
    independently from `a`, `p`, `s` and the INJECTED `phi_f`.  So `balance` asks
    "is the flux the kernel returned consistent with a separately written recurrence?" --
    that is a real check.  Injecting `fast := 1.1*A*gu*phi_f` makes it fail analytically
    by `-0.1*A*gu*phi_f`.  It is NOT a physical conservation law, and the report must say
    so: with the kernel's own `fast`/`slow` the identity telescopes for ANY admissible
    fraction triple, which is exactly why the injection test is the only thing that
    gives it teeth.
    """
    from routing import route                      # the frozen router, byte for byte
    with torch.no_grad():
        h, s, f, k = [v.numpy() for v in self.flux_parameters(torch.tensor(x))]
    gu, pf, gs = _check(np.ascontiguousarray(h), ('gu', 'phi_f', 'gs'))
    fast, slow, a, p = scan_dp(np.ascontiguousarray(h), np.ascontiguousarray(s),
                               np.ascontiguousarray(f), np.ascontiguousarray(k),
                               self.data.lower_release, self.inp, self.demand,
                               self.cap, gu, pf, gs)
    M = a * (1 - p) * s
    L = np.cumsum(a * p * (1 - pf) - slow, axis=0)
    before = np.vstack([np.zeros_like(M[:1]), M[:-1]])
    uptake = np.minimum(before + self.inp, self.demand); loss = a * (1 - p) * (1 - s)
    river = route(self.data, fast + slow, vf=float(x[2]))
    balance = (self.inp - uptake - loss - fast - slow
               - np.diff(M + L, axis=0, prepend=np.zeros_like(M[:1])))
    net = ((fast + slow).sum() - river['channel_removed'].sum()
           - river['terminal'].sum() - river['stocks'][-1].sum())
    return dict(fast=fast, slow=slow, M=M, L=L, available=a, uptake=uptake,
                demand=self.demand, mineral_loss=loss,
                local_balance_max_kg=float(np.abs(balance).max()),
                network_balance_kg=float(net), terminal=river['terminal'],
                reservoir_stocks=river['stocks'],
                channel_loss=river['channel_removed'],
                ledger_spelling='independent: M=a*(1-gu)*s; L=cumsum(a*gu*(1-phi_f)-F_s)')


def assert_no_autograd():
    """A criterion must never be evaluated outside `torch.no_grad()`.

    The frozen adjoints carry the FROZEN `h -> p` map.  `TransportDP` has no `backward`,
    so differentiating it fails loudly -- asserted here rather than assumed.
    """
    for name, obj in (('TransportDP', TransportDP), ('scan_dp', scan_dp),
                      ('_tag_dp_nb', _tag_dp_nb), ('ledger_dp', ledger_dp)):
        if hasattr(obj, 'backward'):
            raise AssertionError('DP_KERNEL_HAS_BACKWARD:%s' % name)
    return True
