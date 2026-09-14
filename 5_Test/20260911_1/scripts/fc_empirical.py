"""Shared empirical models: no TN history, site embeddings or postfit correction."""
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from fc_io import RUN


def softplus_numpy(margin):
    return np.logaddexp(0., np.asarray(margin, dtype=np.float64))


def sigmoid_numpy(margin):
    x = np.asarray(margin, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -x))


def xgb_gradient_hessian(margin, labels, weights):
    """Exact gradient, positive GN Hessian of 0.5 sum w*(softplus(z)-y)^2.

    The true Hessian also contains w*(p-y)*p'' and may be negative. The
    registered GN approximation intentionally omits that term. A tiny positive
    floor allows recovery from a saturated margin; it does not change loss.
    """
    p = softplus_numpy(margin)
    derivative = sigmoid_numpy(margin)
    w = np.asarray(weights, dtype=np.float64)
    grad = w*(p-np.asarray(labels, dtype=np.float64))*derivative
    hess = np.maximum(w*derivative**2, 1e-12)
    return grad, hess


def xgb_objective(weights):
    weights = np.asarray(weights, dtype=np.float64).copy()
    def objective(margin, dmatrix):
        return xgb_gradient_hessian(margin, dmatrix.get_label(), weights)
    return objective


def import_xgboost():
    sys.path.insert(0, str(RUN.parent / '20260907_4/vendor'))
    import xgboost
    if xgboost.__version__ != '3.1.3':
        raise RuntimeError(f'Unregistered XGBoost version: {xgboost.__version__}')
    return xgboost


class EALSTM(nn.Module):
    """Static-only input gate, dynamic forget/output/candidate gates.

    State is supplied/returned explicitly. Chunking is a memory decision, never
    a history reset or truncated backpropagation decision. No dropout or batch
    normalization, so station batching and ordering cannot change predictions.
    """
    def __init__(self, n_static, n_dynamic, hidden, seed=1729):
        super().__init__()
        if hidden not in [8, 16]:
            raise ValueError(hidden)
        torch.manual_seed(seed)
        self.hidden = hidden
        self.input_gate = nn.Linear(n_static, hidden, dtype=torch.float32)
        self.dynamic_gates = nn.Linear(n_dynamic+hidden, 3*hidden, dtype=torch.float32)
        self.head = nn.Linear(hidden, 1, dtype=torch.float32)
        for layer in [self.input_gate, self.dynamic_gates, self.head]:
            nn.init.normal_(layer.weight, std=1/math.sqrt(layer.in_features))
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.dynamic_gates.bias[:hidden].fill_(1.)

    def negative_log_prior(self):
        prior = self.head.bias.new_zeros(())
        for layer in [self.input_gate, self.dynamic_gates, self.head]:
            prior = prior + .5*(layer.weight.square().sum()*layer.in_features + layer.bias.square().sum())
        return prior.double()

    def forward(self, dynamic, static, state=None, chunk_days=128, checkpoint_chunks=True):
        if state is None:
            h = dynamic.new_zeros((dynamic.shape[1], self.hidden))
            c = torch.zeros_like(h)
        else:
            h, c = state
        gate = torch.sigmoid(self.input_gate(static))

        def block(hidden, cell, drivers):
            daily = []
            for x in drivers:
                f, o, g = self.dynamic_gates(torch.cat([x, hidden], dim=-1)).chunk(3, dim=-1)
                cell = torch.sigmoid(f)*cell + gate*torch.tanh(g)
                hidden = torch.sigmoid(o)*torch.tanh(cell)
                daily.append(torch.nn.functional.softplus(self.head(hidden)).squeeze(-1))
            return hidden, cell, torch.stack(daily)

        outputs = []
        for a in range(0, len(dynamic), chunk_days):
            args = (h, c, dynamic[a:a+chunk_days])
            if checkpoint_chunks and torch.is_grad_enabled():
                h, c, output = checkpoint(block, *args, use_reentrant=False)
            else:
                h, c, output = block(*args)
            outputs.append(output)
        return {'daily_latent_readout': torch.cat(outputs), 'state': (h, c)}

    def predict_monthly(self, dynamic, static, stops, **kwargs):
        result = self.forward(dynamic, static, **kwargs)
        # Readout is supervised only at month ends: no constructed daily labels.
        index = torch.as_tensor(np.asarray(stops)-1, device=dynamic.device)
        return result['daily_latent_readout'][index], result['state']
