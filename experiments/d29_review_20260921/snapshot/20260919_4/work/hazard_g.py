"""The ONE changed upstream: `Predictor.hazard`, with the fast-fraction log-odds
modulated by `gamma * z`.

    odds(f^N_t) = exp(gamma * z_t) * odds(f_t)
    aq_eff      = exp(t[20] + gamma * z_t)          <- the whole change
    f^N         = aq_eff * f0 / (aq_eff * f0 + 1 - f0)

`f0` is `data.fast_fraction`, the carrier partition
`fast_water / (fast_water + percolation*area_ha*10)`, verified against the frozen
arrays at max abs diff 2.220446049250313e-16.  `t[20] = log_aq` is the frozen
model's OWN scalar log-odds multiplier on that fraction -- this round does not add
a mechanism, it lets that multiplier vary with the day.

WHY gamma = 0 IS BITWISE AND NOT MERELY CLOSE
---------------------------------------------
`0.0 * z` is exactly `0.0` for every finite `z`; `t[20] + 0.0` is exactly `t[20]`;
`exp` of the same input is the same float.  The expression is therefore the frozen
one with the scalar `aq` broadcast instead of stored, and every element is
computed from the same operands by the same operations.  `phase0_freeze.py` asserts
this with `np.array_equal`, never `allclose`.

The general expression is used UNCONDITIONALLY -- there is deliberately no
`if gamma == 0` fast path, so the no-op assertion exercises the real arithmetic
rather than a branch that bypasses it.

WHY THE BINDING MUST BE ON THE CLASS, NOT THE INSTANCE
------------------------------------------------------
`f` is constructed in exactly one place, `Predictor.hazard`, but it is reached
four ways and ONE of them bypasses attribute lookup on the instance:

    Predictor.tensor_predict:89          self.hazard(t)                 instance-ok
    Predictor.ledger:96                  self.hazard(torch.tensor(x))   instance-ok
    campaign_model.py:130                Predictor.hazard(self,t[:29])  CLASS ONLY
        ^ Endpoints.flux_parameters, reached by
          Matched.tag_values:92 / Matched.tensor_predict:97 / Matched.ledger:103
          and by StructureEndpoints.daily_boundary:39 -- i.e. by the forward
          replay and by BOTH ledger branches.

`Predictor.hazard(self, t[:29])` is an explicit class-level call.  Binding to the
instance (`model.hazard = MethodType(...)`) would leave it pointing at the frozen
function and would NOT raise: the run would complete, nothing would differ, and
`gamma` would be silently inert on the forward path.  Binding to the class covers
all four.  This mirrors `common20.install_kernel`'s module-attribute rebinding, one
level over: module -> class.  `_ORIG` + `restore()` make it reversible.

`t[:29]` TRUNCATES NOTHING THAT MATTERS
---------------------------------------
`hazard` reads indices 0, 1, 3, 4:11, 11:18, 18, 19, 20.  `20 < 29`, so `t[20]` is
the same number under the 29-slice and the full 30-vector.

`is not` note: `from model import Predictor` and `cm.Predictor` are the SAME class
object (`campaign_model.py:13` imports it from `model`), which is what makes a
single rebinding reach every call site.  `assert_identity()` below pins that.
"""
import numpy as np
import torch

from model import Predictor, ALPHA, TAU

_ORIG = Predictor.hazard
_ORIG_WAS = Predictor.hazard

CONFIG = {'gamma': 0.0, 'z': None}


def assert_identity():
    """`cm.Predictor` must be the same object as the `model.Predictor` rebound here."""
    import campaign_model as cm
    assert cm.Predictor is Predictor, (cm.Predictor, Predictor)
    return True


def set_config(z, gamma):
    """Freeze (z, gamma) for the next `hazard` call.  z is validated, not trusted."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    assert z.ndim == 2, z.shape
    assert np.isfinite(z).all(), 'Z_NOT_FINITE'
    CONFIG['z'] = torch.tensor(z)
    CONFIG['gamma'] = float(gamma)
    return CONFIG['z'].shape


def hazard_g(self, t):
    """`Predictor.hazard` verbatim except for the single `aq` line."""
    a = self.regional(t[0], t[4:11], ALPHA)
    tau = self.regional(t[3], t[11:18], TAU)
    logh = a[None, :] + t[1] * self.logcontact + t[18] * self.dyn[0] + t[19] * self.dyn[1]
    h = torch.where(self.positive, torch.exp(logh), torch.zeros_like(logh))
    s = torch.exp(-torch.exp(-tau))
    f0 = torch.tensor(self.data.fast_fraction)
    aq = torch.exp(t[20] + CONFIG['gamma'] * CONFIG['z'])   # <- the whole change
    f = aq * f0 / (aq * f0 + 1 - f0)
    return h, s, f


def install():
    """Bind on the CLASS.  See the module docstring for why the instance is wrong."""
    Predictor.hazard = hazard_g
    return Predictor.hazard


def restore():
    Predictor.hazard = _ORIG
    return Predictor.hazard


def is_installed():
    return Predictor.hazard is hazard_g
