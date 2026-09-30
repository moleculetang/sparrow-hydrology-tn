"""Read-only reproduction of the rejected trial's 1971/1972 state boundary.

Stops after 1971, before any objective or held-out evaluation. This diagnostic
does not change the scientific solver or accepted gradient identity.
"""
import json
import pickle
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/legacy22/scripts')]
from serial_solvers import restore_solver
from d29_platform.runtime import dispatch_allowed, sha, write_json
from d29_platform import precision_candidate as precise
import d29_training.annual_chain as annual_module

annual_module.run_land1 = precise.run_land1
annual_module.land1_adjoint = precise.land1_adjoint
from d29_training.land1_adapter import Land1Training


class Captured(Exception):
    pass


def main():
    ok, resources = dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:
        raise RuntimeError('RESOURCE_GATE')
    job_id = 'LAND1_F23_T0_s0'
    job = next(j for j in json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf-8'))
               if j['id'] == job_id)
    with (ROOT / 'outputs/jobs' / job_id / 'optimizer.pkl').open('rb') as file:
        saved = pickle.load(file)
    solver = restore_solver(saved['solver'])
    if solver.step(lambda x: None) != 'request_evaluation':
        raise RuntimeError('ORIGINAL_TRIAL_STATE_CHANGED')
    point = solver.s['x'].copy()
    original_run = annual_module.AnnualChain._run
    receipt = dict(role='1971 final state at the formerly rejected trial; no TN evaluation',
                   year=1971,
                   resources=resources, code_sha256=sha(ROOT/'d29_platform/precision_candidate.py'),
                   point_sha256=__import__('hashlib').sha256(point.tobytes()).hexdigest())

    def capture(self, year, initial, eta, correction):
        result, mask = original_run(self, year, initial, eta, correction)
        if year == 1971:
            r, land, state = np.unravel_index(np.argmin(result.final), result.final.shape)
            receipt.update(reach_zero_based=int(r),land_zero_based=int(land),state_index=int(state),
                           all_state_min_kg=float(np.min(result.final)))
            high = float(result.final[r, land, state])
            corr = float(result.compensation[r, land, state - 1]) if state else 0.
            full = high - corr
            history = result.states[:, r, land, state]
            lows = (-result._inputs['compensation_history'][:, r*result.final.shape[1]+land, state-1]
                    if state else np.zeros_like(history))
            negative = np.flatnonzero(history < 0)
            indices = sorted(set(range(max(0,len(history)-6),len(history))) |
                             set(int(i) for i in negative[:5]))
            sources=result._inputs['sources'][:,r*result.final.shape[1]+land]
            targets=result._inputs['target'][:,r*result.final.shape[1]+land]
            planned=result._inputs['outflows'][:,r*result.final.shape[1]+land]
            probs=[field[:,r*result.final.shape[1]+land] for field in result._inputs['probabilities']]
            def step(i):
                if i==0:return None
                t=i-1
                P,SA,SP,N,L=(float(v) for v in result.states[t,r,land])
                ka=float(probs[0][t])*SA;kp=float(probs[1][t])*SP
                X=N+float(sources[t,3])+ka+kp
                O=float(np.sum(planned[t]));need=max(0.,float(targets[t])+O-P-float(sources[t,0]))
                uptake=min(X,need);A=X-uptake
                pm=float(probs[2][t]);pl=float(probs[3][t]);rem=(1-pm)*A
                loss=pl*rem
                delta=float(np.sum(np.array([sources[t,3],ka,kp,-uptake,-pm*A,-loss],dtype=float)))
                with localcontext() as decimal_context:
                    decimal_context.prec=60
                    reference=((Decimal(1)-Decimal.from_float(pl)) *
                               (Decimal(1)-Decimal.from_float(pm)) *
                               (Decimal.from_float(X)-Decimal.from_float(uptake)))
                return dict(input_N_kg=N,source_mineral_kg=float(sources[t,3]),
                            mineralized_kg=ka+kp,X_kg=X,need_kg=need,uptake_kg=uptake,
                            available_after_uptake_kg=A,direct_N_next_kg=(1-pl)*rem,
                            decimal_same_binary_operands_N_next_kg=float(reference),
                            delta_sum_kg=delta,old_compensation_kg=float(-lows[t]),
                            mobilize_probability=pm,loss_probability=pl)
            receipt.update(high_kg=high,correction_kg=corr,represented_kg=full,
                           high_negative=high<0,represented_negative=full<0,
                           negative_high_days=int(negative.size),
                           trace=[dict(day_index=int(i),high_kg=float(history[i]),
                                       low_kg=float(lows[i]),represented_kg=float(history[i]+lows[i]),
                                       preceding_step=step(i))
                                  for i in indices],
                           local_balance_kg=float(result.max_local_balance_kg),
                           final_compensation_range_kg=[float(np.min(result.compensation)),
                                                        float(np.max(result.compensation))])
            raise Captured
        return result, mask

    annual_module.AnnualChain._run = capture
    adapter = Land1Training(job, diagnostic=True)
    try:
        adapter.value_gradient(point, forward_only=True)
    except Captured:
        pass
    finally:
        annual_module.AnnualChain._run = original_run
    if 'represented_kg' not in receipt:
        raise RuntimeError('DID_NOT_CAPTURE_1971_BOUNDARY')
    write_json(ROOT/'outputs/roundoff_state_origin_1971.json', receipt)


if __name__ == '__main__':
    main()
