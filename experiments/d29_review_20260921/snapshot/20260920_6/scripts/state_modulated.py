"""Hydrologic-state modulation OUTSIDE the original D29 soft saturation.

No labels enter the reference construction or hazard. Qp is percolation,
not lower-layer discharge. The inherited transport caps hazard at 700.
"""
import math
import numpy as np
import torch
from hf_model import HFEndpoints

def reference(fast, percolation, reference_days):
    if not np.isfinite(fast).all() or not np.isfinite(percolation).all():
        raise ValueError('NONFINITE_WATER')
    if (fast < 0).any() or (percolation < 0).any():
        raise ValueError('NEGATIVE_WATER')
    total = fast + percolation
    active = total > 1e-3
    ref = active & reference_days[:, None]
    count = ref.sum(0)
    if np.any((count == 0) & active.any(0)):
        raise ValueError('NO_REFERENCE_ACTIVE_DAYS')
    logtotal = np.zeros_like(total)
    np.log(total, out=logtotal, where=active)
    mean = np.divide((logtotal * ref).sum(0), count,
                     out=np.zeros(total.shape[1]), where=count > 0)
    u = np.where(active, logtotal - mean, 0.)
    a = np.divide(fast, total, out=np.zeros_like(total), where=active)
    return active, u, a, count

def log_modulation(eta, u, a, active):
    # Avoid log(0), including unused torch.where branches.
    positive_a = torch.where(a > 0, a, torch.ones_like(a))
    positive_b = torch.where(a < 1, 1-a, torch.ones_like(a))
    v = eta * u / 2
    both = torch.logaddexp(torch.log(positive_a)+v,
                          torch.log(positive_b)-v)
    out = torch.where(a == 0, -v, torch.where(a == 1, v, both))
    # logaddexp(log(a),log(1-a)) has roundoff: remove exactly that constant.
    base = torch.logaddexp(torch.log(positive_a), torch.log(positive_b))
    out = out - torch.where((a > 0) & (a < 1), base, torch.zeros_like(base))
    return torch.where(active, out, torch.zeros_like(out))

def safe_hazard(h, logxi):
    positive = h > 0
    if not bool(torch.isfinite(h).all()) or bool((h < 0).any()):
        raise ValueError('INVALID_PARENT_HAZARD')
    logh = torch.log(torch.where(positive, h, torch.ones_like(h)))
    total = logh + logxi
    # Product is formed only on a branch known safe; unused branches are safe too.
    direct = positive & (total < math.log(700.)) & (torch.abs(logxi) < 300.)
    product = h * torch.exp(torch.where(direct, logxi, torch.zeros_like(logxi)))
    stable = torch.exp(torch.clamp(total, max=math.log(700.)))
    return torch.where(positive, torch.where(direct, product, stable), torch.zeros_like(h))

class StateModulated(HFEndpoints):
    def __init__(self, data, train, design):
        super().__init__(data, train, design)
        self.names = self.names + ['eta_hydrologic_state']
        self.bounds = self.bounds + [(-1., 1.)]
        qf = np.asarray(data.fast_water)
        qp = np.asarray(data.percolation) * data.area_ha[None, :] * 10
        active, u, a, count = reference(qf, qp, np.asarray(data.dates.year <= 2020))
        self.state_u = torch.tensor(u)
        self.state_a = torch.tensor(a)
        self.state_active = torch.tensor(active)
        self.reference_count = count

    def initial(self, index=0):
        return np.r_[super().initial(index), 0. if index == 0 else .25]

    def variable_scale(self):
        return np.r_[super().variable_scale(), .5]

    def prior(self, t):
        return torch.cat([super().prior(t[:30]), math.sqrt(.03/17.)*t[30:31]/.5])

    def flux_parameters(self, t):
        h, s, f, k = super().flux_parameters(t[:30])
        lx = log_modulation(t[30], self.state_u, self.state_a, self.state_active)
        return safe_hazard(h, lx), s, f, k

    def value_gradient(self, x):
        value, grad = super().value_gradient(x)
        extra = .5*(.03/17.)*(float(x[30])/.5)**2
        self.last_terms['new_prior'] = extra
        self.last_terms['original_prior'] = self.last_terms['prior'] - extra
        return value, grad
