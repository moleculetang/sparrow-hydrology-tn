"""Source experiment gate, immutable identities and measured full-history peaks."""
import time,ast
import native_runtime as rt
R=rt.RUN
def main():
    if (R/'work/campaign.json').exists():raise RuntimeError('ALREADY_LAUNCHED')
    gates=['preflight_F23_G_D','preflight_T24_G_D_H1','infrastructure_checks','solver_resume_test','evaluation_fixture_tests','label_counterfactual','full_source_fixture']
    for n in gates:assert rt.read(R/'reports'/f'{n}.json')['status']=='PASS',n
    for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    jobs=rt.read(R/'configs/jobs.json');peaks=[rt.read(R/'reports'/f'preflight_{f}.json')['process']['peak_gib'] for f in rt.read(R/'configs/folds.json')]
    for j in jobs:
        if j['kind']=='D29_BE':
            a=rt.read(R/'outputs'/j['tag']/'audit.json');assert a['physical_reasonable'] and a['numerical_sufficient'];peaks.append(a['process']['peak_gib'])
    # Reconcile every inherited byte registered at the start, including original observations.
    manifest=rt.read(R/'evidence/inherited_manifest.json')
    for p,h in manifest.items():assert rt.sha(p)==h,('OLD_INPUT_CHANGED',p)
    core=['campaign_model.py','temporal_model.py','hf_model.py','source_corrected.py','balanced_tags.py','fit_worker.py','serial_solvers.py','native_runtime.py']
    shared=[R/'scripts'/p for p in core]+list((R/'vendor').rglob('*.py'))+[R/'configs/campaign.json',R/'configs/jobs.json',R/'configs/folds.json',R/'data/spatial_support.json',R/'data/domains/FULL24C/arrays.json',R/'data/domains/FULL24C/topology.json']
    for f in rt.read(R/'configs/folds.json'):
        files=shared+[R/'data/folds'/f/'train.parquet',R/'data/folds'/f/'registry.json',R/'data/designs'/f'{f}.json']
        rt.write(R/'reports/launch_by_fold'/f'{f}.json',dict(fold=f,frozen_hashes={p.relative_to(R).as_posix():rt.sha(p) for p in files}))
    peaks.append(rt.read(R/'reports/full_source_fixture.json')['process']['peak_gib'])
    peak=max(peaks)
    rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',created=time.time(),gates=gates,peak_reservations_gib={k:peak for k in ['D29_BE','SOURCE_UNIFIED','SOURCE_GROUPED','SOURCE_SEPARATE']},peak_note='Observed cold complete source-adjoint preflight and full prediction/ledger audit; controller reserves peak x 1.2 plus process growth.',new_fits=12,reused=4))
    print('SEALED',peak,flush=True)
if __name__=='__main__':main()
