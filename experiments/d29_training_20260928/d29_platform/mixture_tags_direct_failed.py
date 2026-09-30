"""Experimental provenance repair: replay planned uptake, allocate realized outflows.
Separate acceptance required; does not modify the physical state recurrence.
"""
import numpy as np
from numba import njit
from .precision_candidate import Land1Result, TaggedLand1Result, _array, _transfer, _accurate_sum

from .precision_transfer import transfer_expansion

@njit(cache=True)
def _tag_forward(initial, source, total_source, target, out, planned_out, probs, event,
                 matrices, nr, nl, total_states, total_fluxes, record, compensated, initial_compensation, physical_correction):
    nt, nu, nk, _ = source.shape
    current = initial.copy()
    pa, pp, pm, pl, ff, lr = probs
    history = np.empty((nt + 1 if record else 0, nu, nk, 5), np.float64)
    flux_history = np.empty((nt if record else 0, nu, nk, 4), np.float64)
    if record:
        history[0] = current
    acc = np.zeros((nu, nk, 4), np.float64)
    worst, state_error, flux_error = 0.0, 0.0, 0.0
    rounding_allocation_max=0.;rounding_bound_failed=False
    largest_rounding_ratio=0.;largest_rounding_day=-1;largest_rounding_unit=-1;largest_rounding_state=-1
    organic_correction = initial_compensation.copy()
    for t in range(nt):
        if event[t] >= 0:
            old_low=np.zeros_like(current)
            old_low[:,:,1:5]=-organic_correction
            moved,low,_=transfer_expansion(current.reshape(nr,nl,nk*5),old_low.reshape(nr,nl,nk*5),matrices[event[t]])
            moved=moved.reshape(nu,nk,5);low=low.reshape(nu,nk,5)
            moved_correction=-low[:,:,1:5].copy()
            for r in range(nr):
                sl=slice(r*nl,(r+1)*nl)
                for k in range(nk):
                    err=_accurate_sum(np.concatenate((moved[sl,k].ravel(),-current[sl,k].ravel(),-moved_correction[sl,k].ravel(),organic_correction[sl,k].ravel())))
                    worst=max(worst,abs(err))
            current=moved;organic_correction=moved_correction
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
                    all_low=np.zeros_like(total_states[t])
                    all_low[:,1:5]=-physical_correction[t]
                    # Only the current reach is needed for physical replay.
                    r=u//nl;land=u%nl
                    moved,_,_=transfer_expansion(total_states[t,r*nl:(r+1)*nl].reshape(1,nl,5),all_low[r*nl:(r+1)*nl].reshape(1,nl,5),matrices[event[t],r:r+1])
                    P,SA,SP,N,L=moved[0,land]
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
            mixture=np.empty(nk,np.float64)
            for label in range(nk):
                p0,a0,b0,n0,l0=current[u,label]
                x0=n0+source[t,u,label,3]+pa[t,u]*a0+pp[t,u]*b0
                mixture[label]=p0+source[t,u,label,0]+uptake_fraction*x0
            mixture_total=_accurate_sum(mixture)
            flux_sum = np.zeros(4, np.float64)
            state_sum = np.zeros(5, np.float64)
            old_corrections=organic_correction[u].copy()
            daily_flux=np.empty((nk,4),np.float64)
            for k in range(nk):
                old_correction=organic_correction[u,k].copy()
                p, a, b, n, lower = current[u, k]
                ka, kb = pa[t, u] * a, pp[t, u] * b
                x = n + source[t, u, k, 3] + ka + kb
                take = uptake_fraction * x
                p_pre = p + source[t, u, k, 0] + take
                share = p_pre / mixture_total if mixture_total > 0 else 0.0
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
                # Same direct nonnegative transport state as the physical
                # kernel. Rounding remains visible to the independent budget.
                organic_correction[u,k,2]=0.
                organic_correction[u,k,3]=0.
                values = np.array([fast, slow, loss, exp])
                daily_flux[k]=values
                acc[u, k] += values
                flux_sum += values
                state_sum += following[u, k]
                if record:
                    flux_history[t, u, k] = values
                err = _accurate_sum(np.concatenate((following[u,k],-current[u,k],-source[t,u,k],values,old_correction,-organic_correction[u,k])))
                worst = max(worst, abs(err))
            if total_states.shape[0]>0:
                # Tags passively partition the one physical transaction. Put
                # the bounded final rounding remainder in the largest tag's
                # low part; never alter the physical stock or its fluxes.
                for state in (3,4):
                    terms=np.concatenate((np.array([total_states[t+1,u,state],
                        -physical_correction[t+1,u,state-1]]),
                        -following[u,:,state],organic_correction[u,:,state-1]))
                    residual=_accurate_sum(terms)
                    pivot=np.argmax(following[u,:,state])
                    scale=max(1.,abs(total_states[t+1,u,state]),
                              _accurate_sum(np.abs(following[u,:,state])))
                    # Five independently rounded source mixtures can differ
                    # from one physical float64 transaction over a long
                    # history. The allocation is capped at one tenth of the
                    # existing 1e-6 kg source-accounting gate as well as a
                    # scale-dependent 512-epsilon bound.
                    allowed=min(1e-7,512.*2.220446049250313e-16*scale)
                    if abs(residual)>allowed:rounding_bound_failed=True
                    ratio=abs(residual)/allowed
                    if ratio>largest_rounding_ratio:
                        largest_rounding_ratio=ratio;largest_rounding_day=t
                        largest_rounding_unit=u;largest_rounding_state=state
                    rounding_allocation_max=max(rounding_allocation_max,abs(residual))
                    old_high=following[u,pivot,state]
                    new_high=old_high+residual
                    used=new_high-old_high
                    following[u,pivot,state]=new_high
                    organic_correction[u,pivot,state-1]-=(residual-used)
                    err=_accurate_sum(np.concatenate((following[u,pivot],
                        -current[u,pivot],-source[t,u,pivot],daily_flux[pivot],
                        old_corrections[pivot],-organic_correction[u,pivot])))
                    worst=max(worst,abs(err))
            for j in range(4):
                flux_error = max(flux_error, abs(flux_sum[j] - total_fluxes[t, u, j]))
            if total_states.shape[0] > 0:
                for j in range(5):
                    terms=np.concatenate((following[u,:,j],np.array([-total_states[t+1,u,j]])))
                    if j>=1:
                        terms=np.concatenate((terms,-organic_correction[u,:,j-1],np.array([physical_correction[t+1,u,j-1]])))
                    state_error=max(state_error,abs(_accurate_sum(terms)))
        current = following
        if record:
            history[t + 1] = current
    return current, acc, flux_history, history, worst, state_error, flux_error, organic_correction,rounding_allocation_max,rounding_bound_failed,largest_rounding_ratio,largest_rounding_day,largest_rounding_unit,largest_rounding_state


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
    import math
    initial_corr=np.zeros((nr,nl,nk,4)) if initial_compensation is None else np.asarray(initial_compensation).reshape(nr,nl,nk,4)
    physical_corr=i['compensation_history'][0].reshape(nr,nl,4)
    restart_difference=np.zeros((nr,nl,5))
    for r in range(nr):
        for l in range(nl):
            for state in range(5):
                terms=[*initial[r,l,:,state],-i['initial'].reshape(nr,nl,5)[r,l,state]]
                if state in (1,2,3,4):terms.extend([*-initial_corr[r,l,:,state-1],physical_corr[r,l,state-1]])
                restart_difference[r,l,state]=abs(math.fsum(terms))
    restart_difference=restart_difference.reshape(nr*nl,5)
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
    corr=np.zeros((nr*nl,nk,4)) if initial_compensation is None else np.asarray(initial_compensation,dtype=float).reshape(nr*nl,nk,4)
    if not np.isfinite(corr).all() or (not compensated and np.any(corr!=0)):raise ValueError('TAG_COMPENSATION_STATE_IDENTITY')
    ans = _tag_forward(initial.reshape(nr * nl, nk, 5), source.reshape(nt, nr * nl, nk, 4),
        i["sources"], i["target"], i.get('realized_outflows',i["outflows"]), i["outflows"], i["probabilities"], i["event"], i["matrices"],
        nr, nl, states, result.fluxes.reshape(nt, nr * nl, 4), record_history, compensated, corr, i['compensation_history'])
    if ans[9]:
        raise ValueError(f'SOURCE_ROUNDING_ALLOCATION_EXCEEDS_FLOAT64_BOUND max_kg={ans[8]:.17g} ratio={ans[10]:.17g} day={ans[11]} reach={ans[12]//nl} land={ans[12]%nl} state={ans[13]}')
    final = ans[0].reshape(nr, nl, nk, 5)
    final_err=0.
    for r in range(nr):
        for l in range(nl):
            for state in range(5):
                terms=[*final[r,l,:,state],-result.final[r,l,state]]
                if state in (1,2,3,4):terms.extend([*-ans[7][r*nl+l,:,state-1],result.compensation[r,l,state-1]])
                final_err=max(final_err,abs(math.fsum(terms)))
    return TaggedLand1Result(labels, final, ans[1].reshape(nr, nl, nk, 4),
        ans[2].reshape(nt, nr, nl, nk, 4) if record_history else None,
        ans[3].reshape(nt + 1, nr, nl, nk, 5) if record_history else None,
        float(ans[4]), max(float(ans[5]), final_err), float(ans[6]), ans[7].reshape(nr,nl,nk,4),float(ans[8]))
