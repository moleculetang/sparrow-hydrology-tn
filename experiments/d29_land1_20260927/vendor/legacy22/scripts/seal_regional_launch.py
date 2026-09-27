"""Seal actual regional acceptance and fold identities; no historical gate substitution."""
import ast,time
import native_runtime as rt
R=rt.RUN

def main():
    assert not (R/'work/campaign.json').exists(),'ALREADY_LAUNCHED'
    gates=['regional_tiny','preparation_regional','infrastructure_checks','solver_resume_test','evaluation_pipeline_fixture','label_counterfactual','full_source_fixture']
    gates += [p.stem for p in (R/'reports').glob('preflight_*_REGIONAL_*.json')]
    assert len([x for x in gates if x.startswith('preflight_')])==7
    for n in gates:assert rt.read(R/'reports'/f'{n}.json')['status']=='PASS',n
    assert rt.read(R/'reports/lowrank_freeze.json')['status']=='FROZEN'
    for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    jobs=rt.read(R/'configs/jobs.json');assert len(jobs)==28
    peaks=[rt.read(R/'reports'/f'{n}.json')['process']['peak_gib'] for n in gates if n.startswith('preflight_') or n=='full_source_fixture']
    for j in jobs:
        if j.get('reuse_from'):
            a=rt.read(R/'outputs'/j['tag']/'audit.json');assert a['physical_reasonable'];peaks.append(a['process']['peak_gib'])
    core=['campaign_model.py','temporal_model.py','hf_model.py','source_corrected.py','regional_response.py','balanced_tags.py','fit_worker.py','serial_solvers.py','native_runtime.py']
    shared=[R/'scripts'/p for p in core]+list((R/'vendor').rglob('*.py'))+[R/'configs'/p for p in ['campaign.json','jobs.json','folds.json']]+[R/'data'/p for p in ['spatial_support.json','spatial_blocks.json','domains/FULL24C/arrays.json','domains/FULL24C/topology.json']]
    for f in rt.read(R/'configs/folds.json'):
        files=shared+[R/'data/folds'/f/'train.parquet',R/'data/folds'/f/'registry.json',R/'data/designs'/f'{f}.json',R/'data/bases'/f'{f}.npy',R/'data/bases'/f'{f}.json']
        rt.write(R/'reports/launch_by_fold'/f'{f}.json',dict(fold=f,frozen_hashes={p.relative_to(R).as_posix():rt.sha(p) for p in files}))
    # Hydrology arrays independently hash-checked by every completed full-history preflight.
    # Event identities are inherited frozen support, never accessible to fitting workers.
    for p,h in rt.read(R/'data/event_identity.json').items():assert rt.sha(p)==h
    peak=max(peaks)
    rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',created=time.time(),gates=gates,peak_reservations_gib={k:peak for k in ['SOURCE_UNIFIED','REGIONAL_L1','REGIONAL_L3','REGIONAL_N3']},peak_note='Observed complete cold/gradient/all-reach source and prediction ledger peaks; admission multiplies by 1.2 plus process growth.',new_fits=24,reused=4,logical_paths=28,full_history_preflight_calls=sum(rt.read(R/'reports'/f'{n}.json').get('full_history_calls',0) for n in gates)))
    print('SEALED',peak,flush=True)
if __name__=='__main__':main()
