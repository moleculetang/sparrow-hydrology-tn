"""Bounded-memory numerical thread comparison, separate from speed inference."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from compare_hosts import arrays,design_difference
from d29_platform.runtime import write_json,sha


def design_without_output_namespace(folder):
    design=json.loads((folder/'portable_design.json').read_text(encoding='utf8'))
    prefix=folder.name+'_'
    original=design['training_ids']
    if not all(isinstance(v,str) and v.startswith(prefix) for v in original):
        raise RuntimeError('UNEXPECTED_TRAINING_ID_NAMESPACE')
    # These are generated row IDs, not station identities. Verify the exact
    # output namespace and retain every source row suffix and its order.
    design['training_ids']=[v[len(prefix):] for v in original]
    return design


def main():
    jid='TIM_F23_K_V1_s0'
    base=ROOT/'outputs/imported_workstation_diagnostics/outputs/jobs'/('diagnostic_'+jid)
    reference=json.loads((base/'diagnostic_oracle.json').read_text(encoding='utf8'))
    cases=[]
    for threads in (2,4):
        folder=ROOT/'outputs/imported_workstation_threads/outputs/jobs'/f'diagnostic_threads{threads}_{jid}'
        result=json.loads((folder/'diagnostic_oracle.json').read_text(encoding='utf8'))
        compared={name:arrays(base/name,folder/name,tol,relative) for name,tol,relative in (
            ('point.npy',0.,False),('gradient.npy',1e-6,True),('training_prediction.npy',1e-8,True),
            ('local_kg_day.npy',1e-6,False),('final_state.npy',1e-6,False))}
        design=design_difference(design_without_output_namespace(base),design_without_output_namespace(folder))
        error=abs(reference['cycles'][-1]['objective']-result['cycles'][-1]['objective'])
        objective_ok=error<=1e-8*(1+abs(reference['cycles'][-1]['objective']))
        thread_ok=all(p['num_threads']==threads for p in result['actual_library_pools'])
        thread_ok &= result['torch_threads']==threads and result['torch_interop_threads']==1
        cases.append(dict(threads=threads,passed=all(c['passed'] for c in compared.values()) and design['passed'] and objective_ok and thread_ok,
            arrays=compared,design=design,objective_abs_difference=error,actual_threads_verified=thread_ok,
            full_oracle_seconds=[c['seconds'] for c in result['cycles']],
            reference_oracle_seconds=[c['seconds'] for c in reference['cycles']],
            loads_identical=False,generated_training_id_namespace_removed_only=True,
            scope='same_point_full_target_thread_probe_not_controlled_speedup'))
    receipt=dict(passed=all(c['passed'] for c in cases),cases=cases,
        scope='full1961_2024_thread_numerical_consistency_not_fit_or_controlled_speed_benchmark',
        production_threads=1,multithread_speed_benefit_demonstrated=False,identities={'deployment/compare_threads.py':sha(Path(__file__))})
    write_json(ROOT/'evidence/thread_numerical_consistency.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],cases=[dict(threads=c['threads'],passed=c['passed'],J_abs=c['objective_abs_difference'],
        arrays={k:v['max_abs_difference'] for k,v in c['arrays'].items()},seconds=c['full_oracle_seconds']) for c in cases])))
    return 0 if receipt['passed'] else 2


if __name__=='__main__':sys.exit(main())
