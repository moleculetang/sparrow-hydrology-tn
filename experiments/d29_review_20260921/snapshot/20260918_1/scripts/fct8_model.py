"""FCT8 — level/timing decoupling of the D29_BE daily mobilization closure.

The decoupling happens on the latent score `u` *inside* the single soft
saturation, never on the final multiplier (that would put `exp(A_t)` back into
its own weight term). With `b_L = b + delta/2`, `b_T = b - delta/2` and
`mu_{r,m}(v)` the `w`-weighted mean of `u(v)` over the days of month `m`:

    u^FCT_{r,t} = mu_{r,m}(b_L) + [ u_{r,t}(b_T) - mu_{r,m}(b_T) ]
    A^FCT_{r,t} = B * tanh(u^FCT_{r,t} / B),   B = ln 10
    h^FCT_{r,t} = h^{preD29}_{r,t} * exp(A^FCT_{r,t})

`h_preD29` is `Predictor.hazard(t[:29])` times `exp(shift*logcontact)` — i.e.
`Endpoints.flux_parameters` with the last line (the tanh factor) removed. That is
the only difference from D29_BE, so at delta = 0 this is *algebraically* D29_BE:

    u^FCT(delta=0) = mu(b) + u(b) - mu(b) = u(b)     for ANY weight w

which is why the D29_BE source lines in `campaign_model.py` need no change and
the published H1-D29_BE baseline is reusable at zero new fits.

`w` is frozen to `h_preD29` at the fold's published theta_ref. `w` enters only
the delta direction, so its choice cannot affect the delta = 0 identity.

Everything downstream of `flux_parameters` (M/L recurrence, river routing,
reservoir mixing, observation operator, objective, prior) is inherited
unchanged.
"""
import math
import numpy as np
import torch
from campaign_model import RUN, Endpoints, Predictor, sha
from hf_model import HFEndpoints as TemporalEndpoints

#: appended after the 30 D29_BE parameters; `dynamic_0..7` stay at 21..28.
DELTA_NAMES = ['delta_%d' % i for i in range(8)]
DELTA_INDEX = slice(30, 38)
#: the single soft saturation's bound, identical to D29_BE.
B = math.log(10)

#: identical to the D29_BE dynamic block's hard bound (no hidden asymmetry).
DELTA_BOUND = (-12., 12.)
#: identical to the D29_BE dynamic block's variable_scale.
DELTA_SCALE = .5
#: identical to the *effective* D29_BE dynamic prior scale: .5 in the block spec
#: times Matched.prior's sqrt(nstation/17) against ScientificModel's
#: sqrt(.03/nstation).
DELTA_PRIOR_SCALE = math.sqrt(.03 / 17.)


def build_phi_bar(weight, basis, starts, months, nreaches, count):
    """`w`-weighted within-month latent mean.

    `weight` is (T,R) and may be exactly zero (`contact == 0` days); `basis` is
    (T,R,8) and equals phi/sqrt(8). Returns (M,R,8). Cells whose monthly weight
    mass is zero have no weighted mean at all, so they fall back to the uniform
    monthly mean of the basis — the natural limit, and safe because the delta = 0
    identity holds for any weight.
    """
    mass = np.add.reduceat(weight, starts, axis=0)
    degenerate = mass == 0
    phibar = np.divide(np.add.reduceat(weight[:, :, None] * basis, starts, axis=0),
                       mass[:, :, None],
                       out=np.zeros((months, nreaches, 8)), where=~degenerate[:, :, None])
    if degenerate.any():
        uniform = np.add.reduceat(basis, starts, axis=0) / count[:, None, None]
        phibar[degenerate] = uniform[degenerate]
    return phibar, mass, degenerate


class FCT8Tail:
    """Mixin appending the 8-parameter delta block.

    Placed *first* in the MRO so its `initial` / `variable_scale` / `prior` /
    `flux_parameters` wrap the D29_BE ones rather than replace them: every
    `super()` call below lands on the unchanged D29_BE implementation.
    """

    def __init__(self, data, train, design):
        super().__init__(data, train, design)
        cfg = design.get('fct8')
        if cfg is None:
            raise ValueError('MISSING_FCT8_DESIGN')
        self.names = self.names + DELTA_NAMES
        self.bounds = self.bounds + [DELTA_BOUND] * 8
        path = RUN / cfg['phi_bar']
        if sha(path) != cfg['phi_bar_sha256']:
            raise ValueError('FCT8_PHI_BAR_HASH_CHANGED')
        phibar = np.load(path, allow_pickle=False)
        want = (len(self.data.months), self.data.source.shape[1], 8)
        if phibar.shape != want or not np.isfinite(phibar).all():
            raise ValueError('BAD_FCT8_PHI_BAR %s != %s' % (phibar.shape, want))
        self.phi_bar = torch.tensor(phibar)
        self.mid = torch.tensor(np.asarray(self.data.mid))
        self.fct8 = cfg

    def initial(self, index=0):
        """delta starts exactly at 0, so both starts are the D29_BE starts."""
        return np.r_[super().initial(index), np.zeros(8)]

    def variable_scale(self):
        return np.r_[super().variable_scale(), np.repeat(DELTA_SCALE, 8)]

    def prior(self, t):
        """Zero at delta = 0, so J_FCT8(b, delta=0) == J_D29_BE(b) in full."""
        return torch.cat([super().prior(t[:30]),
                          DELTA_PRIOR_SCALE * (t[DELTA_INDEX] / DELTA_SCALE)])

    def flux_parameters(self, t):
        h, s, f = Predictor.hazard(self, t[:29])
        shift = self.pi * (t[29] - t[1])
        h = h * torch.exp(shift[None, :] * self.logcontact)
        b = t[21:29]
        d = t[DELTA_INDEX]
        bL, bT = b + d / 2, b - d / 2
        muL = torch.einsum('mrj,j->mr', self.phi_bar, bL)[self.mid]
        muT = torch.einsum('mrj,j->mr', self.phi_bar, bT)[self.mid]
        u = muL + (torch.einsum('trj,j->tr', self.dynamic_basis, bT) - muT)
        h = h * torch.exp(B * torch.tanh(u / B))
        return h, s, f, torch.ones(self.data.source.shape[1])


class FCT8Daily(FCT8Tail, TemporalEndpoints):
    """D-fold FCT8: daily station boundary, D29_BE operator."""


class FCT8Monthly(FCT8Tail, Endpoints):
    """M-fold FCT8: monthly exposure, D29_BE operator."""


def FCT8(data, train, design):
    """Resolve the base exactly as `make_model` does for D29_BE."""
    if 'observation_operator' in design:
        return FCT8Daily(data, train, design)
    return FCT8Monthly(data, train, design)
