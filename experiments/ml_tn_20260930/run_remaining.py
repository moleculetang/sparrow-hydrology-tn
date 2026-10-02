"""Single continuation after direct matrix completion, not a scheduled task."""
import os,time,json,subprocess,datetime,platform
from mltn.common import ROOT,read,write
from mltn.resources import registry
from controller import execute,jobid
def select_joint(routes):
    best={};scores=[]
    for family in routes.values():
        vals=[]
        for c in range(8):
            p=ROOT/'jobs'/f'joint_screen_{family}_c{c}_s1729'/'result.json'
            if p.exists():
                r=read(p);vals.append((r['selection_score'],c));scores.append(dict(family=family,config=c,selection_score=r['selection_score']))
        if vals:best[family]=min(vals)[1]
    write(ROOT/'outputs/frozen_joint_selection.json',dict(selected=best,scores=scores,selection='2022Jan-Sep joint loss; no evaluation outcomes'))
    return best

def auxiliary_jobs(selection):
    jobs=[]
    for task,family in selection['winners'].items():
        c=selection['selected'][family+'_'+task]
        for stage in ['F23','F24']:
            for lead in ([1] if task=='monthly' else [1,7,30]):
                for seed in [1729,1730,1731]:jobs.append(dict(role='auxiliary',stage=stage,task=task,family=family,config=c,seed=seed,lead=lead))
    return jobs

def matched_jobs(selection,best):
    """Already registered same-recipe references need frozen families, not spatial results."""
    jobs=[dict(stage='F24',task=t,family=f,config=0,seed=1729,fixed_recipe=True) for t,f in selection['winners'].items()]
    jobs +=[dict(role='joint',stage='F24',task='joint',family=f,config=0,seed=1729,fixed_recipe=True) for f in best]
    return jobs
def main():
    while not (ROOT/'outputs/direct_matrix_done.json').exists():
        mod,state=registry();owner=read(ROOT/'outputs/controller_owner.json')
        if mod is not None and mod.identity(owner['pid']) is None:raise RuntimeError('DIRECT_CONTROLLER_STOPPED_BEFORE_COMPLETION')
        time.sleep(10)
    write(ROOT/'outputs/remaining_owner.json',dict(pid=os.getpid(),host=platform.node(),start=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    # Identity and origin acceptance for new joint and feedback interfaces.
    if subprocess.call([os.sys.executable,'-B',str(ROOT/'acceptance.py')]):raise RuntimeError('UPDATED_ACCEPTANCE_FAILED')
    from reconcile_warmup import main as reconcile
    reconcile()
    selection=read(ROOT/'outputs/frozen_selection.json');routes=selection['joint_routes'];jobs=[dict(role='joint',stage='screen',task='joint',family=f,config=c,seed=1729) for f in routes.values() for c in range(8)]
    # Direct winners are already frozen. These CPU jobs do not depend on joint selection.
    auxiliary=auxiliary_jobs(selection)
    execute(jobs+auxiliary,'joint_screen_and_independent_historical_TN');best=select_joint(routes)
    jobs=[]
    for f,c in best.items():
        for stage in ['F23','F24']:
            for seed in [1729,1730,1731]:jobs.append(dict(role='joint',stage=stage,task='joint',family=f,config=c,seed=seed))
        for block in [56,113,191]:
            for stage in ['S23','S24']:
                for seed in [1729,1730,1731]:jobs.append(dict(role='joint',stage=stage,task='joint',family=f,config=0,seed=seed,block=block))
    # These references have no dependency on spatial predictions. Queue them now,
    # so external checkpoint waits do not unnecessarily leave Linux slots empty.
    matched=matched_jobs(selection,best)
    execute(jobs+matched,'joint_final')
    execute(auxiliary,'historical_TN')
    # Same recipe reference separates spatial exclusion from changed hyperparameters.
    # Keep the legacy receipt; these calls only reuse already completed jobs.
    execute(matched,'matched_spatial_recipe_reference')
    write(ROOT/'outputs/all_training_done.json',dict(status='finite_training_matrix_finished',direct='outputs/direct_final_receipt.json',joint='outputs/joint_final_receipt.json',auxiliary='outputs/historical_TN_receipt.json',expert_review='pending',required_followup=['nonnegative reserve ensemble','post-freeze evaluation','causal and feature diagnostics','independent checks','expert reports']))
if __name__=='__main__':main()
