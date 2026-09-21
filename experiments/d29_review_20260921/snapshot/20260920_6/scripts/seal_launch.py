"""Freeze fold-specific training identities without reading other-fold labels in workers."""
import shutil,time
from pathlib import Path
import native_runtime as rt
R=rt.RUN
def main():
    if (R/'work/campaign.json').exists():raise RuntimeError('ALREADY_LAUNCHED')
    gates=['preflight_F23_G_D','preflight_T24_G_D_H1','infrastructure_checks','extended_acceptance','label_counterfactual']
    for name in gates:assert rt.read(R/'reports'/f'{name}.json')['status']=='PASS',name
    for tag in ['F24_R_s0','F24_R_s1']:
        a=rt.read(R/'outputs'/tag/'audit.json');assert a['physical_reasonable'] and a['numerical_sufficient']
    # Fixed event IDs and values copied with explicit provenance, inaccessible to workers.
    ed=R/'data/heldout_labels/events';ed.mkdir(parents=True,exist_ok=True);eventfiles={}
    for name in ['events_frozen.parquet','observed_days.parquet']:
        src=R.parent/'20260920_4/data/evaluation'/name;dst=ed/name;shutil.copy2(src,dst);assert rt.sha(src)==rt.sha(dst);eventfiles[str(src)]=rt.sha(src)
    rt.write(R/'data/event_identity.json',eventfiles)
    core=['campaign_model.py','temporal_model.py','hf_model.py','state_modulated.py','fit_worker.py','serial_solvers.py','native_runtime.py']
    shared=[R/'scripts'/x for x in core]+list((R/'vendor').rglob('*.py'))+[R/'configs/campaign.json',R/'configs/jobs.json',R/'configs/folds.json',R/'data/spatial_support.json',R/'data/domains/FULL24C/arrays.json',R/'data/domains/FULL24C/topology.json']
    for fold in rt.read(R/'configs/folds.json'):
        files=shared+[R/'data/folds'/fold/'train.parquet',R/'data/folds'/fold/'registry.json',R/'data/designs'/f'{fold}.json']
        rt.write(R/'reports/launch_by_fold'/f'{fold}.json',dict(fold=fold,frozen_hashes={str(p.relative_to(R)):rt.sha(p) for p in files}))
    peak=max(rt.read(R/'reports'/f'preflight_{fold}.json')['process']['peak_gib'] for fold in rt.read(R/'configs/folds.json'))
    peak=max(peak,*(rt.read(R/'outputs'/tag/'audit.json').get('process',{}).get('peak_gib',0) for tag in ['F24_R_s0','F24_R_s1']))
    rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',created=time.time(),gates=gates,peak_reservations_gib={'D29_BE':peak,'STATE_MODULATED':peak},peak_note='Measured cold/full-history acceptance and full prediction-ledger audit; admission applies an additional 1.2 factor plus running-process growth.',new_fits=6,reused=2))
    print('SEALED',peak)
if __name__=='__main__':main()
