import ast,time
import native_runtime as rt
R=rt.RUN
def main():
    assert not (R/'work/campaign.json').exists()
    gates=['input_preparation','daily_fixtures','infrastructure_checks','solver_resume_test','label_counterfactual','evaluation_pipeline_fixture','gate_runner']
    gates += [p.stem for p in (R/'reports').glob('preflight_daily_*.json')]
    assert len([g for g in gates if g.startswith('preflight_daily_')])==12
    for g in gates:assert rt.read(R/'reports'/f'{g}.json')['status']=='PASS',g
    jobs=rt.read(R/'configs/jobs.json');assert len(jobs)==24
    peaks=[rt.read(R/'reports'/f'{g}.json')['process']['peak_gib'] for g in gates if g.startswith('preflight_daily_')]
    shared=[R/'scripts'/s for s in ['campaign_model.py','daily_inputs.py','balanced_tags.py','source_corrected.py','regional_response.py','temporal_model.py','hf_model.py','native_runtime.py','fit_worker.py','serial_solvers.py']]+list((R/'vendor').rglob('*.py'))+list((R/'configs').glob('*.json'))+list((R/'data/daily_inputs').glob('*'))
    shared += [R/'data'/p for p in ['spatial_support.json','domains/FULL24C/arrays.json','domains/FULL24C/topology.json']]
    for f in rt.read(R/'configs/folds.json'):
        files=shared+[R/'data/folds'/f/'train.parquet',R/'data/folds'/f/'registry.json',R/'data/designs'/f'{f}.json',R/'data/bases'/f'{f}.npy']
        rt.write(R/'reports/launch_by_fold'/f'{f}.json',dict(fold=f,frozen_hashes={p.relative_to(R).as_posix():rt.sha(p) for p in files}))
    for j in jobs:
        if j.get('reuse_from'):
            old=rt.read(__import__('pathlib').Path(j['reuse_from'])/'audit.json')
            rt.write(R/'work/jobs'/j['tag']/'status.json',dict(status='NUMERICALLY_SUFFICIENT' if old['numerical_sufficient'] else 'NUMERICALLY_INSUFFICIENT_STATIONARY',reuse_pending_independent_audit=True,source=j['reuse_from']))
    for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',gates=gates,created=time.time(),peak_reservations_gib={k:max(peaks) for k in ['SOURCE_UNIFIED','REGIONAL_L3']},peak_note='Full-history cold, objective/adjoint, P paired state and physical tag ledger peak; 1.2 and live growth applied by controller',new_fits=16,reused=8,logical_paths=24))
    print('SEALED',max(peaks),flush=True)
if __name__=='__main__':main()
