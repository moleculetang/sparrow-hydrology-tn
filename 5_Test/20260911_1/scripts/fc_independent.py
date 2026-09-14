"""NumPy equation replay independent of Torch step, including source tags."""
import math
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def regional_numpy(base, gamma, static, low, high):
    a = (base-low)/(high-low)
    return low+(high-low)*sigmoid(np.log(a)-np.log1p(-a)+static@gamma)


def decode(model):
    # Copy numeric checkpoint values only, not live Torch calculations.
    value = {k: p.detach().cpu().numpy().copy() for k, p in model.values.items()}
    network = []
    if model.network is not None:
        for layer in model.network:
            if hasattr(layer, 'weight'):
                network.append((layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy()))
    return {'family': model.family, 'values': value, 'network': network}


def replay(description, source, demand, contact, fast_fraction, lower_release, static, dynamic, eta_drivers, area_ha, state=None,source_parent_index=None):
    v = description['values']
    shape = (source.shape[1], 3, source.shape[-1])
    stock = np.zeros(shape) if state is None else np.array(state, dtype=float).copy()
    split = sigmoid(v['source_logits'] + (static@v['source_gamma'] if 'source_gamma' in v else np.zeros((shape[0], len(v['source_logits'])))))
    if source_parent_index is not None:split=split[:,np.asarray(source_parent_index,dtype=int)]
    if split.shape!=(shape[0],shape[-1]):raise ValueError('Source labels do not map to registered source classes')
    life = regional_numpy(v['log_tau_mineral'], v['gamma_lifetime'], static, math.log(182.625), math.log(3652.5))
    survival = np.exp(-np.exp(-life))
    is_kernel = description['family'] == 'P_KERNEL'
    if not is_kernel:
        alpha = regional_numpy(v['log_alpha'], v['gamma_contact'], static, -9.21, 4.605170186)
    stocks, outputs = [], []
    for d in range(len(source)):
        before = stock.copy()
        stock[:, 0] += source[d]*split
        stock[:, 1] += source[d]*(1-split)
        a = stock[:, :2].sum(axis=(1, 2))
        withdrawal = np.minimum(a, demand[d])
        proportion = np.divide(withdrawal, a, out=np.zeros_like(a), where=a>0)
        removed = stock[:, :2]*proportion[:, None, None]
        uptake = removed.sum(axis=1)
        stock[:, :2] -= removed
        fraction = np.exp(v['log_aq'])*fast_fraction[d]
        fraction = fraction/(fraction+1-fast_fraction[d])
        transfer = np.zeros_like(source[d])
        if is_kernel:
            e1 = np.exp(-contact[d]/np.exp(v['log_tau_short']))
            e2 = np.exp(-contact[d]/np.exp(v['log_tau_long']))
            moved = stock[:, 0]*(-np.expm1(-contact[d]/np.exp(v['log_tau_short'])))[:,None]
            moved += stock[:, 1]*(-np.expm1(-contact[d]/np.exp(v['log_tau_long'])))[:,None]
            fast, recharge = moved*fraction[:,None], moved*(1-fraction[:,None])
            stock[:, 0] *= e1[:,None]
            stock[:, 1] *= e2[:,None]
            loss = stock[:,:2].sum(axis=1)*(1-survival[:,None])
            stock[:,:2] *= survival[:,None,None]
        else:
            features = np.concatenate([static, dynamic[d], np.log1p(stock.sum(axis=-1)/area_ha[:,None])], axis=-1)
            w, b = description['network'][0]
            hidden = np.tanh(features@w.T+b)
            w, b = description['network'][1]
            multiplier = np.exp(np.log(10)*np.tanh(hidden@w.T+b))
            safe = np.where(contact[d]>0, contact[d], 1.)
            hazard = np.where(contact[d]>0, np.exp(np.minimum(alpha+v['beta']*np.log(safe)+eta_drivers[d]@v['eta'], np.log(700))), 0.)
            hazards1 = np.column_stack([multiplier[:,0]/30., multiplier[:,1]*hazard*fraction, multiplier[:,2]*hazard*(1-fraction)])
            hazards2 = np.column_stack([multiplier[:,3]*hazard*fraction, multiplier[:,4]*hazard*(1-fraction)])
            emitted = []
            for bank, hazards in enumerate([hazards1, hazards2]):
                total = hazards.sum(axis=-1)
                ratios = np.divide(hazards, total[:,None], out=np.zeros_like(hazards), where=total[:,None]>0)
                flux = stock[:,bank,None,:] * (ratios*(-np.expm1(-total))[:,None])[:,:,None]
                stock[:,bank] *= np.exp(-total)[:,None]
                emitted.append(flux)
            transfer = emitted[0][:,0]
            fast = emitted[0][:,1]+emitted[1][:,0]
            recharge = emitted[0][:,2]+emitted[1][:,1]
            stock[:,1] += transfer
            loss = stock[:,1]*(1-survival[:,None])
            stock[:,1] *= survival[:,None]
        stock[:,2] += recharge
        slow = stock[:,2]*lower_release[d,:,None]
        stock[:,2] *= (1-lower_release[d,:,None])
        balance = before.sum(axis=1)+source[d]-stock.sum(axis=1)-fast-slow-uptake-loss
        if np.max(np.abs(balance)) > 2e-10 * max(1., np.max(before), np.max(source[d])):
            raise AssertionError('Independent daily source conservation failed')
        outputs.append(np.stack([fast, slow, uptake, loss, transfer, recharge], axis=1))
        stocks.append(stock.copy())
    return {'state': stock, 'fluxes': np.stack(outputs), 'stocks': np.stack(stocks)}
