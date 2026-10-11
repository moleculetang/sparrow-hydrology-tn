# EXPERIMENTAL represented-state budget; formal acceptance pending.
"""Explicit conservative five-stock land kernel; no observations or legacy lifetime.

Units: stocks kg N, daily sources/outflows kg N/day, probabilities dimensionless.
State order P, SON_ACTIVE, SON_PROTECTED, AVAILABLE, SLOW.
Source entry order PLANT, ACTIVE_ORGANIC, PROTECTED_ORGANIC, MINERAL.
Plant outflow order EXPORT, RETURN_ACTIVE, RETURN_PROTECTED.
New organic sources and plant returns mineralize starting the next day.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(_ROOT / "cache" / "numba_land1"))
for _key in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from numba import njit

from .precision_transfer import transfer_expansion

STATE_NAMES = ("plant", "son_active", "son_protected", "available", "slow")
SOURCE_ENTRIES = ("plant", "active_organic", "protected_organic", "mineral")
OUTFLOW_NAMES = ("export", "return_active", "return_protected")
FLUX_NAMES = ("fast", "slow", "available_loss", "plant_export")
PROBABILITY_NAMES = ("mineralize_active", "mineralize_protected", "mobilize",
                     "available_loss", "fast_fraction", "lower_release")


class MissingLand1Input(ValueError):
    pass


class PlantBudgetInfeasible(ValueError):
    def __init__(self, day, reach, land, shortfall):
        self.day, self.reach, self.land, self.shortfall_kg = day, reach, land, shortfall
        super().__init__(f"PLANT_BUDGET_INFEASIBLE day={day} reach={reach} land={land} shortfall_kg={shortfall:.17g}")


@dataclass
class Land1Result:
    fluxes: np.ndarray
    final: np.ndarray
    states: np.ndarray | None
    max_local_balance_kg: float
    _inputs: dict
    compensation: np.ndarray | None = None
    unmet_plant_outflows: np.ndarray | None = None


def _array(value, name):
    if value is None:
        raise MissingLand1Input(f"MISSING_LAND1_INPUT {name}")
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise MissingLand1Input(f"NONFINITE_OR_MISSING_LAND1_INPUT {name}")
    return a


def _field(value, shape, name, probability=False):
    a = _array(value, name)
    try:
        a = np.broadcast_to(a, shape)
    except ValueError as exc:
        raise ValueError(f"LAND1_SHAPE {name}: {a.shape} cannot broadcast to {shape}") from exc
    if np.any(a < 0) or (probability and np.any(a > 1)):
        raise ValueError(f"LAND1_RANGE {name}")
    return a


def hazard_to_probability(hazard):
    """Original capped-risk convention; derivative at h=700 uses capped branch."""
    h = _array(hazard, "hazard")
    if np.any(h < 0):
        raise ValueError("NEGATIVE_HAZARD")
    p = -np.expm1(-np.minimum(h, 700.0))
    dpdh = np.where(h < 700.0, np.exp(-h), 0.0)
    return p, dpdh


def prepare_inputs(*, initial, sources, plant_target, plant_outflows,
                   probabilities: Mapping, transitions=None,
                   intermediate_roundoff_tolerance=0.0):
    initial = _array(initial, "initial")
    if (not np.isfinite(intermediate_roundoff_tolerance) or
            intermediate_roundoff_tolerance < 0 or
            intermediate_roundoff_tolerance > 1e-9):
        raise ValueError("INVALID_INTERMEDIATE_ROUNDOFF_TOLERANCE")
    if (initial.ndim != 3 or initial.shape[-1] != 5 or
            np.any(initial < -intermediate_roundoff_tolerance)):
        raise ValueError("INITIAL_MUST_BE_NONNEGATIVE_REACH_LAND_5")
    nr, nl, _ = initial.shape
    sources = _array(sources, "sources")
    if sources.ndim != 4 or sources.shape[1:] != (nr, nl, 4) or np.any(sources < 0):
        raise ValueError("SOURCES_MUST_BE_NONNEGATIVE_DAY_REACH_LAND_4")
    nt = sources.shape[0]
    if nt == 0:
        raise ValueError("EMPTY_HISTORY")
    shape = (nt, nr, nl)
    target = _field(plant_target, shape, "plant_target")
    out = _field(plant_outflows, (*shape, 3), "plant_outflows")
    if probabilities is None or set(probabilities) != set(PROBABILITY_NAMES):
        raise MissingLand1Input("EXACT_EXPLICIT_PROBABILITY_KEYS_REQUIRED " + str(PROBABILITY_NAMES))
    probs = tuple(_field(probabilities[n], shape, n, True).reshape(nt, nr * nl)
                  for n in PROBABILITY_NAMES)
    if np.any(probs[1] > probs[0]):
        raise ValueError("PROTECTED_MINERALIZATION_EXCEEDS_ACTIVE")
    event = np.full(nt, -1, dtype=np.int64)
    mats = []
    for day, mat in sorted((transitions or {}).items()):
        if not isinstance(day, (int, np.integer)) or day < 0 or day >= nt:
            raise ValueError("TRANSITION_DAY_OUTSIDE_HISTORY")
        m = _field(mat, (nr, nl, nl), "transition")
        if not np.allclose(m.sum(axis=-1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("TRANSITION_ROWS_MUST_SUM_TO_ONE")
        event[int(day)] = len(mats)
        mats.append(m)
    matrices = np.asarray(mats, dtype=np.float64).reshape(-1, nr, nl, nl)
    return dict(initial=np.ascontiguousarray(initial.reshape(nr * nl, 5)),
                sources=sources.reshape(nt, nr * nl, 4), target=target.reshape(nt, nr * nl),
                outflows=out.reshape(nt, nr * nl, 3), probabilities=probs,
                event=event, matrices=matrices, shape=shape,
                transition_days=tuple(sorted((transitions or {}).keys())))


@njit(cache=True)
def _accurate_sum(values):
    """Float64 expansion summation; retains cancellation terms, no mass repair."""
    partials=np.empty(len(values),np.float64);n=0
    for value in values:
        x=value;i=0
        for j in range(n):
            y=partials[j]
            if abs(x)<abs(y):x,y=y,x
            hi=x+y;lo=y-(hi-x)
            if lo!=0.:partials[i]=lo;i+=1
            x=hi
        partials[i]=x;n=i+1
    total=0.
    for j in range(n):total+=partials[j]
    return total


@njit(cache=True)
def _transfer(before, mat, nr, nl):
    after = np.zeros_like(before)
    terms=np.empty(nl,np.float64)
    for r in range(nr):
        for new in range(nl):
            for k in range(before.shape[1]):
                for old in range(nl):terms[old]=mat[r,old,new]*before[r*nl+old,k]
                after[r*nl+new,k]=_accurate_sum(terms)
    return after


@njit(cache=True)
def _forward(initial, source, target, out, probs, event, matrices, nr, nl, keep, compensated, initial_compensation, potential_activity):
    nt, nu, _ = source.shape
    states = np.empty((nt + 1 if keep else 2, nu, 5), np.float64)
    states[0] = initial
    flux = np.empty((nt, nu, 4), np.float64)
    pa, pp, pm, pl, ff, lr = probs
    worst = 0.0
    organic_correction = initial_compensation.copy()
    unmet = np.zeros_like(out)
    correction_history=np.empty((nt+1 if keep else 0,nu,4),np.float64)
    if keep:correction_history[0]=organic_correction
    for t in range(nt):
        oldi = t if keep else t % 2
        newi = t + 1 if keep else (t + 1) % 2
        before = states[oldi]
        if event[t] >= 0:
            old_low=np.zeros_like(before)
            old_low[:,1:5]=-organic_correction
            moved,low,_=transfer_expansion(before.reshape(nr,nl,5),old_low.reshape(nr,nl,5),matrices[event[t]])
            before=moved.reshape(nu,5)
            moved_low=low.reshape(nu,5)
            # Organic pools have persistent hi-minus-compensation state. Other
            # pools retain their ordinary float64 representation; any discarded
            # low part remains visible to this strict independent budget check.
            organic_correction=-moved_low[:,1:5].copy()
            represented_low=np.zeros_like(before)
            represented_low[:,1:5]=-organic_correction
            for r in range(nr):
                sl=slice(r*nl,(r+1)*nl)
                err=_accurate_sum(np.concatenate((before[sl].ravel(),represented_low[sl].ravel(),-states[oldi,sl].ravel(),-old_low[sl].ravel())))
                worst=max(worst,abs(err))
        for u in range(nu):
            old_correction=organic_correction[u].copy()
            P, SA, SP, N, L = before[u]
            ka, kp = pa[t, u] * SA, pp[t, u] * SP
            X = N + source[t, u, 3] + ka + kp
            O = out[t, u, 0] + out[t, u, 1] + out[t, u, 2]
            raw_need = target[t, u] + O - P - source[t, u, 0]
            need = max(0.0, raw_need)
            uptake = min(X, need)
            plant_pre = P + source[t, u, 0] + uptake
            # Algebraic branch evaluation, not post-hoc clipping: if uptake
            # satisfies the positive demand, P_new is exactly the target.
            # Computing (P + source + need) - O instead produced tiny negative
            # states at a zero target with real-scale inputs (saved regression).
            plant_after = target[t, u] if raw_need > 0 and X >= need else (P + source[t, u, 0] - O) + uptake
            realized=out[t,u].copy()
            if plant_after < 0:
                if not potential_activity or O <= 0:
                    return flux, states, worst, t, u, -plant_after, organic_correction, unmet, correction_history
                # This mode declares planned activities POTENTIAL. Supply limits
                # each destination proportionally and records every unmet kg.
                # No stock clipping or unrecorded nitrogen supplementation.
                realized=out[t,u]*(plant_pre/O)
                unmet[t,u]=out[t,u]-realized
                plant_after=0.
            A = X - uptake
            E = pm[t, u] * A
            lower_pre = L + (1 - ff[t, u]) * E
            rem = (1 - pm[t, u]) * A
            fast = ff[t, u] * E
            slow = lr[t, u] * lower_pre
            loss = pl[t, u] * rem
            states[newi, u, 0] = plant_after
            # Compensated additions retain small daily changes to multi-billion
            # kg research initial stocks; equations and probability derivatives
            # are unchanged. No mass-residual adjustment is applied.
            for j, stock, release in ((0, SA, ka), (1, SP, kp)):
                delta = source[t, u, j+1] + realized[j+1] - release
                corrected = delta - organic_correction[u,j] if compensated else delta
                updated = stock + corrected
                organic_correction[u,j] = (updated-stock)-corrected if compensated else 0.
                states[newi,u,j+1] = updated
            # A compensated flux-difference update preserves long-history
            # small changes. Near total exhaustion it can nonetheless round
            # a truly positive remainder below zero. In that specific case
            # the nonnegative flow product is the authoritative state; this
            # numerical fallback is recorded by independent budget tests.
            new_n = (1 - pl[t, u]) * rem
            new_l = (1 - lr[t, u]) * lower_pre
            states[newi, u, 3] = new_n
            states[newi, u, 4] = new_l
            if compensated:
                for j,stock,delta,empty,direct,scale in ((2,N,_accurate_sum(np.array([
                    source[t,u,3],ka,kp,-uptake,-E,-loss])),A==0. or pl[t,u]==1.,new_n,X),
                    (3,L,_accurate_sum(np.array([(1-ff[t,u])*E,-slow])),lr[t,u]==1.,new_l,lower_pre)):
                    if empty:
                        states[newi,u,j+1]=0.;organic_correction[u,j]=0.
                    else:
                        corrected=delta-organic_correction[u,j]
                        updated=stock+corrected
                        new_correction=(updated-stock)-corrected
                        near_empty=direct<=min(1e-7,64.*2.220446049250313e-16*max(1.,scale))
                        if near_empty or updated<0. or updated-new_correction<0.:
                            states[newi,u,j+1]=direct
                            organic_correction[u,j]=0.
                        else:
                            states[newi,u,j+1]=updated
                            organic_correction[u,j]=new_correction
            else:
                organic_correction[u,2]=0.;organic_correction[u,3]=0.
            flux[t, u, 0], flux[t, u, 1] = fast, slow
            flux[t, u, 2], flux[t, u, 3] = loss, realized[0]
            # Sum stock differences, not the difference of two rounded huge
            # totals. This is an independent budget evaluation, not a correction.
            balance = _accurate_sum(np.concatenate((states[newi,u],-before[u],-source[t,u],flux[t,u],old_correction,-organic_correction[u])))
            worst = max(worst, abs(balance))
        if keep:correction_history[t+1]=organic_correction
    return flux, states, worst, -1, -1, 0.0, organic_correction, unmet, correction_history


def run_land1(*, initial, sources, plant_target, plant_outflows, probabilities,
              transitions=None, keep_history=True, compensated=False, initial_compensation=None,
              plant_activity_mode='strict_prescribed', intermediate_roundoff_tolerance=0.0):
    inp = prepare_inputs(initial=initial, sources=sources, plant_target=plant_target,
                         plant_outflows=plant_outflows, probabilities=probabilities,
                         transitions=transitions,
                         intermediate_roundoff_tolerance=intermediate_roundoff_tolerance)
    nt, nr, nl = inp["shape"]
    corr=np.zeros((nr*nl,4)) if initial_compensation is None else np.asarray(initial_compensation,dtype=float).reshape(nr*nl,4)
    if not np.isfinite(corr).all() or (not compensated and np.any(corr!=0)):raise ValueError('COMPENSATION_STATE_IDENTITY')
    if plant_activity_mode not in ('strict_prescribed','potential_with_shortfall'):raise ValueError('UNKNOWN_PLANT_ACTIVITY_MODE')
    inp['plant_activity_mode']=plant_activity_mode
    flux, states, err, day, unit, short, correction, unmet, correction_history = _forward(inp["initial"], inp["sources"], inp["target"],
        inp["outflows"], inp["probabilities"], inp["event"], inp["matrices"], nr, nl, keep_history, compensated, corr, plant_activity_mode=='potential_with_shortfall')
    if day >= 0:
        raise PlantBudgetInfeasible(int(day), int(unit // nl), int(unit % nl), float(short))
    final = states[-1 if keep_history else nt % 2].reshape(nr, nl, 5).copy()
    inp['realized_outflows']=inp['outflows']-unmet
    inp['compensation_history']=correction_history
    return Land1Result(flux.reshape(nt, nr, nl, 4), final,
                       states.reshape(nt + 1, nr, nl, 5) if keep_history else None,
                       float(err), inp, correction.reshape(nr,nl,4), unmet.reshape(nt,nr,nl,3))


@njit(cache=True)
def _reverse(states, source, target, out, probs, event, matrices, nr, nl,
             gflux, gfinal, gstates, potential_activity):
    nt, nu, _ = source.shape
    gi = np.zeros_like(source)
    go = np.zeros_like(out)
    gt = np.zeros_like(target)
    gp = np.zeros((6, nt, nu), np.float64)
    gm = np.zeros_like(matrices)
    adj = gfinal.copy()
    pa, pp, pm, pl, ff, lr = probs
    for t in range(nt - 1, -1, -1):
        if gstates.shape[0] > 0:
            adj += gstates[t + 1]
        before = states[t]
        if event[t] >= 0:
            before = _transfer(before, matrices[event[t]], nr, nl)
        prev = np.empty_like(adj)
        for u in range(nu):
            P, SA, SP, N, L = before[u]
            X = N + source[t, u, 3] + pa[t, u] * SA + pp[t, u] * SP
            O = out[t, u].sum()
            raw_need = target[t, u] + O - P - source[t, u, 0]
            need = max(0.0, raw_need)
            uptake = min(X, need)
            plant_pre=P+source[t,u,0]+uptake
            # Replay the exact stable forward predicate. At a zero plant target,
            # sufficient uptake makes plant_after exactly zero, while separately
            # rounded plant_pre can be one ulp below O. Testing plant_pre<O then
            # wrongly selects the supply-limited branch and erases transport
            # derivatives. This is a branch mismatch, not a physical kink.
            plant_after=target[t,u] if raw_need>0 and X>=need else (P+source[t,u,0]-O)+uptake
            if potential_activity and plant_after<0:
                gP,gSA,gSP,gN,gL=adj[u];gf,gs,gloss,ge=gflux[t,u]
                weights=out[t,u]/O
                gz=ge*weights[0]+gSA*weights[1]+gSP*weights[2]
                go[t,u,0]=(ge-gz)*plant_pre/O
                go[t,u,1]=(gSA-gz)*plant_pre/O
                go[t,u,2]=(gSP-gz)*plant_pre/O
                gi[t,u,0]=gz;gi[t,u,1]=gSA;gi[t,u,2]=gSP;gi[t,u,3]=gz
                prev[u,0]=gz;prev[u,1]=gSA*(1-pa[t,u])+gz*pa[t,u]
                prev[u,2]=gSP*(1-pp[t,u])+gz*pp[t,u];prev[u,3]=gz
                prev[u,4]=gs*lr[t,u]+gL*(1-lr[t,u])
                gp[0,t,u]=(gz-gSA)*SA;gp[1,t,u]=(gz-gSP)*SP
                gp[5,t,u]=(gs-gL)*L
                continue
            A = X - uptake
            E = pm[t, u] * A
            lower_pre = L + (1 - ff[t, u]) * E
            rem = (1 - pm[t, u]) * A
            gP, gSA, gSP, gN, gL = adj[u]
            gf, gs, gloss, ge = gflux[t, u]
            glower = gs * lr[t, u] + gL * (1 - lr[t, u])
            gp[5, t, u] = (gs - gL) * lower_pre
            gE = gf * ff[t, u] + glower * (1 - ff[t, u])
            gp[4, t, u] = (gf - glower) * E
            grem = gloss * pl[t, u] + gN * (1 - pl[t, u])
            gp[3, t, u] = (gloss - gN) * rem
            gA = gE * pm[t, u] + grem * (1 - pm[t, u])
            gp[2, t, u] = (gE - grem) * A
            gX = gA
            gU = gP - gA
            gneed = 0.0
            if raw_need > 0:
                if X <= need:
                    gX += gU
                else:
                    gneed = gU
            gt[t, u] = gneed
            gi[t, u, 0] = gP - gneed
            gi[t, u, 1] = gSA
            gi[t, u, 2] = gSP
            gi[t, u, 3] = gX
            go[t, u, 0] = ge - gP + gneed
            go[t, u, 1] = gSA - gP + gneed
            go[t, u, 2] = gSP - gP + gneed
            prev[u, 0] = gP - gneed
            prev[u, 1] = gSA * (1 - pa[t, u]) + gX * pa[t, u]
            prev[u, 2] = gSP * (1 - pp[t, u]) + gX * pp[t, u]
            prev[u, 3] = gX
            prev[u, 4] = glower
            gp[0, t, u] = (gX - gSA) * SA
            gp[1, t, u] = (gX - gSP) * SP
        if event[t] >= 0:
            idx = event[t]
            adj = np.zeros_like(prev)
            for r in range(nr):
                for old in range(nl):
                    for new in range(nl):
                        for k in range(5):
                            adj[r * nl + old, k] += matrices[idx, r, old, new] * prev[r * nl + new, k]
                            gm[idx, r, old, new] += states[t, r * nl + old, k] * prev[r * nl + new, k]
        else:
            adj = prev
    if gstates.shape[0] > 0:
        adj += gstates[0]
    return adj, gi, gt, go, gp, gm


def land1_adjoint(result: Land1Result, *, grad_fluxes=None, grad_final=None, grad_states=None):
    """Exact reverse recurrence of the entire supplied history, no warmup truncation.

    Return gradients on expanded daily fields. At uptake equality choose the
    supply-limited derivative; at zero need choose zero uptake derivative.
    Transition gradients are unconstrained matrix partials: use row-sum-zero
    directions or a stochastic-matrix chain rule for feasible differentiation.
    """
    if result.states is None:
        raise ValueError("ADJOINT_REQUIRES_KEEP_HISTORY")
    i = result._inputs
    nt, nr, nl = i["shape"]
    gf = np.zeros_like(result.fluxes) if grad_fluxes is None else np.asarray(grad_fluxes, dtype=np.float64)
    gfin = np.zeros_like(result.final) if grad_final is None else np.asarray(grad_final, dtype=np.float64)
    gst = np.empty((0, nr * nl, 5)) if grad_states is None else np.asarray(grad_states, dtype=np.float64).reshape(nt + 1, nr * nl, 5)
    if gf.shape != result.fluxes.shape or gfin.shape != result.final.shape or not np.isfinite(gf).all() or not np.isfinite(gfin).all() or not np.isfinite(gst).all():
        raise ValueError("ADJOINT_COTANGENT_SHAPE_OR_FINITE")
    ans = _reverse(result.states.reshape(nt + 1, nr * nl, 5), i["sources"], i["target"],
        i["outflows"], i["probabilities"], i["event"], i["matrices"], nr, nl,
        gf.reshape(nt, nr * nl, 4), gfin.reshape(nr * nl, 5), gst,i.get('plant_activity_mode')=='potential_with_shortfall')
    return dict(initial=ans[0].reshape(nr, nl, 5), sources=ans[1].reshape(nt, nr, nl, 4),
                plant_target=ans[2].reshape(nt, nr, nl), plant_outflows=ans[3].reshape(nt, nr, nl, 3),
                probabilities={n: ans[4][k].reshape(nt, nr, nl) for k, n in enumerate(PROBABILITY_NAMES)},
                transitions={d: ans[5][k] for k, d in enumerate(i["transition_days"])})


@dataclass
class TaggedLand1Result:
    labels: tuple[str, ...]
    final: np.ndarray
    accumulated_fluxes: np.ndarray
    flux_history: np.ndarray | None
    state_history: np.ndarray | None
    max_local_balance_kg: float
    max_state_sum_error_kg: float
    max_flux_sum_error_kg: float
    compensation: np.ndarray | None = None
    max_rounding_allocation_kg: float = 0.0


@njit(cache=True)
def _tag_forward(initial, source, total_source, target, out, probs, event,
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
            O = out[t, u].sum()
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
        i["sources"], i["target"], i.get('realized_outflows',i["outflows"]), i["probabilities"], i["event"], i["matrices"],
        nr, nl, states, result.fluxes.reshape(nt, nr * nl, 4), record_history, compensated, corr)
    final = ans[0].reshape(nr, nl, nk, 5)
    final_err = float(np.max(np.abs(final.sum(axis=-2) - result.final)))
    return TaggedLand1Result(labels, final, ans[1].reshape(nr, nl, nk, 4),
        ans[2].reshape(nt, nr, nl, nk, 4) if record_history else None,
        ans[3].reshape(nt + 1, nr, nl, nk, 5) if record_history else None,
        float(ans[4]), max(float(ans[5]), final_err), float(ans[6]), ans[7].reshape(nr,nl,nk,2))
