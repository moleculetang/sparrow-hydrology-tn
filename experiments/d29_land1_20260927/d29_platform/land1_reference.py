"""Independent small Torch reference. Never use this graph for a full 64-year run."""
import torch


def torch_reference(*, initial, sources, plant_target, plant_outflows,
                    probabilities, transitions=None,plant_activity_mode='strict_prescribed'):
    """Inputs are float64 Torch tensors; caller controls differentiation leaves."""
    state = initial
    states = [state]
    fluxes = []
    for t in range(len(sources)):
        if transitions and t in transitions:
            state = torch.einsum("rjk,rjl->rkl", transitions[t], state)
        P, a, b, n, lower = state.unbind(-1)
        ip, ia, ib, im = sources[t].unbind(-1)
        export, ret_a, ret_b = plant_outflows[t].unbind(-1)
        probs = {name: value[t] for name, value in probabilities.items()}
        ka = a * probs["mineralize_active"]
        kb = b * probs["mineralize_protected"]
        available = n + im + ka + kb
        plant_out = export + ret_a + ret_b
        raw_need = plant_target[t] + plant_out - P - ip
        need = torch.where(raw_need > 0, raw_need, torch.zeros_like(raw_need))
        uptake = torch.where(raw_need > 0, torch.where(available <= need, available, need),
                             torch.zeros_like(available))
        pre_plant = P + ip + uptake
        if torch.any(plant_out > pre_plant + 1e-10):
            if plant_activity_mode=='strict_prescribed':raise ValueError("REFERENCE_PLANT_INFEASIBLE")
        if plant_activity_mode=='potential_with_shortfall':
            safe=torch.where(plant_out>0,plant_out,torch.ones_like(plant_out))
            factor=torch.where(plant_out>pre_plant,pre_plant/safe,torch.ones_like(plant_out))
            export,ret_a,ret_b=export*factor,ret_a*factor,ret_b*factor
            plant_out=export+ret_a+ret_b
        mobilized = probs["mobilize"] * (available - uptake)
        fast = probs["fast_fraction"] * mobilized
        pre_lower = lower + (1 - probs["fast_fraction"]) * mobilized
        slow = probs["lower_release"] * pre_lower
        remaining = (1 - probs["mobilize"]) * (available - uptake)
        loss = probs["available_loss"] * remaining
        state = torch.stack((pre_plant - plant_out, a - ka + ia + ret_a,
                             b - kb + ib + ret_b,
                             (1 - probs["available_loss"]) * remaining,
                             (1 - probs["lower_release"]) * pre_lower), -1)
        states.append(state)
        fluxes.append(torch.stack((fast, slow, loss, export), -1))
    return torch.stack(fluxes), torch.stack(states)
