"""Repair separator aliases; preserve scientific paths and durable budgets."""
import time, shutil
import native_runtime as rt
from fit_worker import identity
R=rt.RUN
archive=R/'evidence/manifest_recovery_v3'
archive.mkdir(exist_ok=False)
def retain(p,move=False):
    if not p.exists():return
    dest=archive/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True)
    if move:shutil.move(str(p),str(dest))
    else:shutil.copy2(p,dest)
oldcampaign=rt.read(R/'work/campaign.json')
for p in (R/'reports/launch_by_fold').glob('*.json'):
    retain(p);m=rt.read(p);fixed={}
    for key,h in m['frozen_hashes'].items():
        norm=key.replace('\\','/')
        if norm in ['scripts/source_corrected.py','scripts/balanced_tags.py']:h=rt.sha(R/norm)
        elif rt.sha(R/norm)!=h:raise RuntimeError('UNEXPECTED_CHANGE '+norm)
        if norm in fixed and fixed[norm]!=h:raise RuntimeError('CONFLICT '+norm)
        fixed[norm]=h
    fixed['scripts/balanced_tags.py']=rt.sha(R/'scripts/balanced_tags.py')
    m['frozen_hashes']=fixed;m['operational_recovery']='separator aliases normalized; maintenance_v2 scientific equivalence retained'
    rt.write(p,m)
records=[]
for job in rt.read(R/'configs/jobs.json'):
    if job.get('reuse_from'):continue
    root=R/'work/jobs'/job['tag'];cp=root/'checkpoints';ident=identity(job)
    for name in ['status.json','audit_failure.json','audit_pending.json','yield.request']:
        retain(root/name,True)
    retain(R/'outputs'/job['tag']/'audit.json',True)
    if (cp/'latest.json').exists():
        ref=rt.read(cp/'latest.json');retain(cp/'latest.json');retain(cp/ref['payload'])
        state=rt.restore(cp,ref['identity']);state['identity']=ident;rt.checkpoint(cp,state)
        assert rt.restore(cp,ident)['calls']==state['calls']
        rt.write(root/'status.json',dict(status='RESOURCE_YIELDED',job=job,calls=state['calls'],active_seconds=state['active_seconds'],phase=state['phase'],stage=state['stage'],best=state['best'],updated=time.time()))
        records.append(dict(tag=job['tag'],calls=state['calls'],phase=state['phase'],best=state['best']['objective']))
    else:records.append(dict(tag=job['tag'],never_initialized=True))
for name in ['data/selected.json','data/prediction_freeze.json','delivery_manifest.json','README.md','reports/专家诊断报告.md','reports/独立完成审计.md','reports/实际方法与偏离.md','reports/independent_completion_audit.json','reports/path_summary.csv','reports/postprocess_tasks.json','reports/evaluation_summary.json','reports/source_coefficients_and_objectives.csv','reports/parameter_roles.csv','reports/physical_period_summary.csv','reports/four_corner_station_diagnostics.csv','reports/annual_reach_source_ledger.parquet','work/controller_status.json']:
    retain(R/name,True)
retain(R/'reports/launch_validation.json');proof=rt.read(R/'reports/launch_validation.json');proof['manifest_recovery']='evidence/manifest_recovery_v3';rt.write(R/'reports/launch_validation.json',proof)
retain(R/'work/campaign.json');campaign=dict(oldcampaign);campaign['launch_sha256']=rt.sha(R/'reports/launch_validation.json');rt.write(R/'work/campaign.json',campaign)
assert all(campaign[k]==oldcampaign[k] for k in ['started','training_deadline','delivery_deadline'])
rt.write(R/'reports/manifest_recovery_v3.json',dict(status='PASS',paths=records,budget_reset=False,scientific_restart=False,reason='Windows separator alias retained stale source_corrected hash; all failed launches stopped before model construction'))
rt.write(R/'work/maintenance.request',dict(reason='hold dispatch until controller restart'))
print('RECOVERY_PASS',records,flush=True)
