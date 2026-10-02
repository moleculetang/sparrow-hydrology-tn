"""Register result roles from training selection, never from held-out scores."""
from mltn.common import ROOT,read,write
from controller import final_jobs,jobid

def main():
    s=read(ROOT/'outputs/frozen_selection.json');expected={jobid(j) for j in final_jobs(s)}
    for task,family in s['winners'].items():
        expected.add(jobid(dict(stage='F24',task=task,family=family,config=0,seed=1729,fixed_recipe=True)))
        for stage in ['F23','F24']:
            for lead in ([1] if task=='monthly' else [1,7,30]):
                for seed in [1729,1730,1731]:expected.add(jobid(dict(role='auxiliary',stage=stage,task=task,family=family,config=s['selected'][family+'_'+task],seed=seed,lead=lead)))
    jp=ROOT/'outputs/frozen_joint_selection.json'
    if jp.exists():
        for family,c in read(jp)['selected'].items():
            for stage in ['F23','F24']:
                for seed in [1729,1730,1731]:expected.add(jobid(dict(role='joint',stage=stage,family=family,config=c,seed=seed)))
            for block in [56,113,191]:
                for stage in ['S23','S24']:
                    for seed in [1729,1730,1731]:expected.add(jobid(dict(role='joint',stage=stage,family=family,config=0,seed=seed,block=block)))
            expected.add(jobid(dict(role='joint',stage='F24',family=family,config=0,seed=1729,fixed_recipe=True)))
    attempts=[];include=[]
    for folder in sorted((ROOT/'jobs').iterdir()):
        if not folder.is_dir():continue
        r=read(folder/'result.json') if (folder/'result.json').exists() else None
        name=folder.name
        reference=bool(r and r.get('role') in ['simple_reference','descriptive historical reference only','descriptive legacy comparator','reserved-segment nonnegative ensemble'])
        # Exclude superseded spatial baseline naming from the initial local smoke test.
        if reference and r.get('role')=='simple_reference' and '_B' in name and name.startswith('F'):reference=False
        eligible=(name in expected or reference) and r is not None
        if eligible:include.append(name)
        role='registered_final' if name in expected else 'reference' if reference else 'screen' if 'screen_' in name else 'superseded_or_deferred'
        attempts.append(dict(job=name,role=role,included=eligible,complete=r is not None,failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None))
    for folder in sorted((ROOT/'superseded').rglob('*')) if (ROOT/'superseded').exists() else []:
        if not folder.is_dir() or not ((folder/'result.json').exists() or (folder/'failure.json').exists()):continue
        attempts.append(dict(job=folder.name,archive=folder.relative_to(ROOT).as_posix(),role='superseded_implementation_or_input_cohort',included=False,complete=(folder/'result.json').exists(),failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None))
    write(ROOT/'outputs/result_roles.json',dict(rule='training-only frozen configurations; no evaluation-based choice',include=include,expected=sorted(expected),missing=sorted(expected-set(include)),attempts=attempts))
if __name__=='__main__':main()
