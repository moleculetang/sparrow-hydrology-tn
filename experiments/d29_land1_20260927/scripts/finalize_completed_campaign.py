"""Finite post-training pipeline; refuses live, stale or unaccepted results.

This does not change models or training choices after evaluation. Exceptions
preserve all files and require review, rather than silently dropping failures.
"""
import sys,json,time,subprocess
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()


def main():
    clock=json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))
    deadline=datetime.fromisoformat(clock['deadline'])
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
    for model in ['u','land1']:
        stop=ROOT/f'outputs/{model}_queue_stop.json'
        if not stop.exists():raise RuntimeError('TRAINING_QUEUE_NOT_FINISHED '+model)
    def call(script,*args,reserve=9_000_000_000):
        paused=False
        while True:
            if datetime.now(deadline.tzinfo)>=deadline:raise RuntimeError('FINAL_DEADLINE_CHECKPOINT_PRESERVED')
            ok,res=dispatch_allowed(paused=paused,reserve_bytes=reserve)
            if ok:break
            paused=True;write_json(ROOT/'outputs/finalization_resource_pause.json',dict(script=script,resources=res));time.sleep(10)
        with (ROOT/'outputs/finalization_console.log').open('ab') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    # The queue stops dispatching at hour 80; its last worker may still be
    # completing a call before writing a safe terminal checkpoint.
    while True:
        live=[]
        for j in jobs:
            path=ROOT/'outputs/jobs'/j['id']/'status.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8'))['status'] in ('running','continuing_same_path_zero_ftol'):
                live.append(j['id'])
        if not live:break
        if datetime.now(deadline.tzinfo)>=deadline:raise RuntimeError('LIVE_WORKERS_AT_FINAL_DEADLINE')
        write_json(ROOT/'outputs/finalization_waiting.json',dict(live_checkpoints=live));time.sleep(10)
    exclusions={}
    for j in jobs:
        folder=ROOT/'outputs/jobs'/j['id'];sp=folder/'status.json'
        if not (folder/'best.npy').exists():
            status=json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else {'status':'not_dispatched'}
            exclusions[j['id']]='No fitted checkpoint: '+status['status']+'; see finite queue and worker records; not evidence that the strategy failed.'
            continue
        status=json.loads(sp.read_text(encoding='utf-8'))
        if status['status']=='resource_checkpoint':
            # Never guess whether a paused worker has a successor. Such a
            # checkpoint needs explicit sealing/continuation before readout.
            raise RuntimeError('RESOURCE_CHECKPOINT_REQUIRES_SEALING '+j['id'])
        audit=folder/'independent_audit.json';ledger=folder/'physical_ledger.json';blocks=folder/'training_block_gradients.json'
        if j['model']=='LAND1':
            if not audit.exists() or not blocks.exists():call('audit_land1_solution.py',j['id'])
            if not ledger.exists():call('verify_land1_ledger.py',j['id'])
        else:
            if not audit.exists():call('audit_u_solution.py',j['id'],reserve=4_000_000_000)
            if not ledger.exists():call('verify_u_ledger.py',j['id'],reserve=4_000_000_000)
            if not blocks.exists():call('diagnose_training_blocks.py',j['id'],reserve=4_000_000_000)
        a=json.loads(audit.read_text(encoding='utf-8'));l=json.loads(ledger.read_text(encoding='utf-8'))
        if not a['objective_passed'] or not l['passed']:raise RuntimeError('POSTFIT_ACCEPTANCE_FAILED '+j['id'])
        if not (folder/'prediction_freeze.json').exists():
            call('freeze_land1_predictions.py' if j['model']=='LAND1' else 'freeze_u_predictions.py',j['id'],reserve=9_000_000_000 if j['model']=='LAND1' else 4_000_000_000)
    write_json(ROOT/'outputs/campaign_exclusions.json',exclusions)
    manifest=ROOT/'outputs/campaign_manifest.json'
    if not manifest.exists():call('build_campaign_manifest.py',str(ROOT/'outputs/campaign_exclusions.json'),reserve=1_000_000_000)
    if not (ROOT/'outputs/evaluation/evaluation_completed.json').exists():
        call('evaluate_frozen_campaign.py',str(manifest),reserve=4_000_000_000)
    call('diagnose_frozen_auxiliary.py',reserve=2_000_000_000)
    call('report_frozen_campaign.py',reserve=1_000_000_000)
    write_json(ROOT/'outputs/finalization_completed.json',dict(completed=datetime.now(deadline.tzinfo).isoformat(),
        computational_pipeline_completed=True,expert_interpretation_review_pending=True,
        no_evaluation_based_refitting=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(ROOT/'outputs/finalization_error.json',dict(error=repr(exc),checkpoints_preserved=True,no_silent_exclusion=True))
        raise
