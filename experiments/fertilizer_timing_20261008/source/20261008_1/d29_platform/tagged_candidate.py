"""Experimental provenance repair: replay planned uptake, allocate realized outflows.
Separate acceptance required; does not modify the physical state recurrence.
"""
import numpy as np
from numba import njit
from .land1 import Land1Result, TaggedLand1Result, _array, _transfer, _accurate_sum

@njit(cache=True)
def _tag_forward(initial, source, total_source, target, out, planned_out, probs, event,
                 matrices, nr, nl, total_states, total_fluxes, record, compensated, initial_compensation):
    nt, nu, nk, _ = source.shape
    current = initial.copy()
    pa, pp, pm, pl, ff, lr = probs
    history = np.empty((nt + 1 if record else 0, nu, nk, 5), np.float64)
    flux_history = np.empty((nt if record else 0, nu, nk, 4), np.float64)
    if record:
        history[0] = current
    acc = np.zeros((nu, nk, 4), np.float64)
    worst, state_error, flux_error = 0.0, 0.0, 0.0
    organic_correction = initial_compensation.copy()
    for t in range(nt):
        if event[t] >= 0:
            moved = _transfer(current.reshape(nu,nk*5),matrices[event[t]],nr,nl).reshape(nu,nk,5)
            moved_correction = np.zeros_like(organic_correction)
            mat = matrices[event[t]]
            for r in range(nr):
                for old in range(nl):
                    for new in range(nl):
                        moved_correction[r*nl+new] += mat[r,old,new]*organic_correction[r*nl+old]
                for k in range(nk):
                    transfer_error = _accurate_sum(np.concatenate((moved[r*nl:(r+1)*nl,k].ravel(),-current[r*nl:(r+1)*nl,k].ravel())))
                    worst = max(worst, abs(transfer_error))
            current = moved
            organic_correction = moved_correction
        following = np.empty_like(current)
        for u in range(nu):
            P, SA, SP, N, L = np.zeros(5, np.float64)
            for k in range(nk):
                P += current[u, k, 0]
                SA += current[u, k, 1]
                SP += current[u, k, 2]
                N += current[u, k, 3]
                L += current[u, k, 4]
            if total_states.shape[0]>0:
                # A passive provenance calculation must follow the authoritative
                # physical trajectory, not recompute nonlinear uptake branches
                # from independently rounded sums of tracers.
                if event[t]>=0:
                    physical=np.zeros(5,np.float64);r=u//nl;land=u%nl;terms=np.empty(nl,np.float64)
                    for j in range(5):
                        for old in range(nl):terms[old]=matrices[event[t],r,old,land]*total_states[t,r*nl+old,j]
                        physical[j]=_accurate_sum(terms)
                    P,SA,SP,N,L=physical
                else:P,SA,SP,N,L=total_states[t,u]
            X = N + total_source[t, u, 3] + pa[t, u] * SA + pp[t, u] * SP
            O = planned_out[t, u].sum()  # Replay uptake from prescribed plan, allocate only realized destinations.
            need = max(0.0, target[t, u] + O - P - total_source[t, u, 0])
            uptake = min(X, need)
            uptake_fraction = uptake / X if X > 0 else 0.0
            plant_pre = P + total_source[t, u, 0] + uptake
            raw_need = target[t, u] + O - P - total_source[t, u, 0]
            plant_after = target[t, u] if raw_need > 0 and X >= need else (P + total_source[t, u, 0] - O) + uptake
            if total_states.shape[0]>0:plant_after=total_states[t+1,u,0]
            flux_sum = np.zeros(4, np.float64)
            state_sum = np.zeros(5, np.float64)
            for k in range(nk):
                p, a, b, n, lower = current[u, k]
                ka, kb = pa[t, u] * a, pp[t, u] * b
                x = n + source[t, u, k, 3] + ka + kb
                take = uptake_fraction * x
                p_pre = p + source[t, u, k, 0] + take
                share = p_pre / plant_pre if plant_pre > 0 else 0.0
                exp, ret_a, ret_b = share * out[t, u]
                av = x - take
                e = pm[t, u] * av
                fast = ff[t, u] * e
                low_pre = lower + (1 - ff[t, u]) * e
                slow = lr[t, u] * low_pre
                rem = (1 - pm[t, u]) * av
                loss = pl[t, u] * rem
                following[u, k, 0] = share * plant_after
                for j, stock, release, returned in ((0,a,ka,ret_a),(1,b,kb,ret_b)):
                    delta=source[t,u,k,j+1]+returned-release
                    corrected=delta-organic_correction[u,k,j] if compensated else delta
                    updated=stock+corrected
                    organic_correction[u,k,j]=(updated-stock)-corrected if compensated else 0.
                    following[u,k,j+1]=updated
                following[u, k, 3] = (1 - pl[t, u]) * rem
                following[u, k, 4] = (1 - lr[t, u]) * low_pre
                values = np.array([fast, slow, loss, exp])
                acc[u, k] += values
                flux_sum += values
                state_sum += following[u, k]
                if record:
                    flux_history[t, u, k] = values
                err = (following[u, k] - current[u, k]).sum() - source[t, u, k].sum() + values.sum()
                worst = max(worst, abs(err))
            for j in range(4):
                flux_error = max(flux_error, abs(flux_sum[j] - total_fluxes[t, u, j]))
            if total_states.shape[0] > 0:
                for j in range(5):
                    state_error = max(state_error, abs(state_sum[j] - total_states[t + 1, u, j]))
        current = following
        if record:
            history[t + 1] = current
    return current, acc, flux_history, history, worst, state_error, flux_error, organic_correction


def propagate_source_labels(result: Land1Result, *, tagged_initial, tagged_sources,
                            labels, record_history=False, diagnostic_restart_error_kg=None,
                            compensated=False, initial_compensation=None):
    """Conservative source labels on all land units, including initial stocks.

    Tagged shapes add a label axis immediately before state/entry. Default
    streaming bookkeeping retains final stocks and cumulative fluxes, avoiding
    a 64-year five-stock array per source label. Full histories are opt-in.
    Labels preserve provenance through internal transfers, not chemical species.
    """
    i = result._inputs
    nt, nr, nl = i["shape"]
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("SOURCE_LABELS_MUST_BE_NONEMPTY_AND_UNIQUE")
    nk = len(labels)
    initial = _array(tagged_initial, "tagged_initial")
    source = _array(tagged_sources, "tagged_sources")
    if initial.shape != (nr, nl, nk, 5) or source.shape != (nt, nr, nl, nk, 4):
        raise ValueError("TAGGED_SHAPE")
    if np.any(initial < 0) or np.any(source < 0):
        raise ValueError("NEGATIVE_SOURCE_LABEL")
    # Source-accounting contract is 1e-6 kg, distinct from the 1e-7 kg
    # nonnegativity/uptake contract. Do not use a tighter unrelated threshold
    # for a source-label restart at real soil-stock magnitudes.
    restart_difference = np.abs(initial.sum(axis=-2).reshape(nr * nl, 5)-i["initial"])
    if np.max(restart_difference)>1e-6:
        # Diagnostic continuation only. No stock/label values are changed and
        # the caller must carry the previous measured discrepancy. This does
        # not waive the 1e-6 source acceptance criterion in run receipts.
        if diagnostic_restart_error_kg is None or not np.isfinite(diagnostic_restart_error_kg) or diagnostic_restart_error_kg<np.max(restart_difference):
            raise ValueError("INITIAL_LABEL_SUM_IDENTITY")
        envelope=64*366*np.spacing(np.maximum(np.abs(i['initial']),1.))
        if np.any(restart_difference>np.maximum(envelope,1e-6)):
            raise ValueError('LABEL_DIFFERENCE_NOT_FLOAT64_ROUNDOFF_SCALE')
    if not np.allclose(source.sum(axis=-2).reshape(nt, nr * nl, 4), i["sources"], rtol=0, atol=1e-7):
        raise ValueError("SOURCE_LABEL_SUM_IDENTITY")
    states = np.empty((0, nr * nl, 5)) if result.states is None else result.states.reshape(nt + 1, nr * nl, 5)
    corr=np.zeros((nr*nl,nk,2)) if initial_compensation is None else np.asarray(initial_compensation,dtype=float).reshape(nr*nl,nk,2)
    if not np.isfinite(corr).all() or (not compensated and np.any(corr!=0)):raise ValueError('TAG_COMPENSATION_STATE_IDENTITY')
    ans = _tag_forward(initial.reshape(nr * nl, nk, 5), source.reshape(nt, nr * nl, nk, 4),
        i["sources"], i["target"], i.get('realized_outflows',i["outflows"]), i["outflows"], i["probabilities"], i["event"], i["matrices"],
        nr, nl, states, result.fluxes.reshape(nt, nr * nl, 4), record_history, compensated, corr)
    final = ans[0].reshape(nr, nl, nk, 5)
    final_err = float(np.max(np.abs(final.sum(axis=-2) - result.final)))
    return TaggedLand1Result(labels, final, ans[1].reshape(nr, nl, nk, 4),
        ans[2].reshape(nt, nr, nl, nk, 4) if record_history else None,
        ans[3].reshape(nt + 1, nr, nl, nk, 5) if record_history else None,
        float(ans[4]), max(float(ans[5]), final_err), float(ans[6]), ans[7].reshape(nr,nl,nk,2))
