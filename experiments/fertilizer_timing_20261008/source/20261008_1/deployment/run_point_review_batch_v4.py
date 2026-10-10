"""Two explicit original-contract plateau reviews, serial safe checkpoints.

No production fitting, solver editing, scientific signals or repeated restarts.
Native-stop races are retained for the ordinary train/audit/freeze controller.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from host_resources import registry
from controller_lock import acquire
from approved_parent_lineage import current

JOBS=['TIM_F24_C2_V1_s0','TIM_S113_C1_V0_s0']


def main():
    if os.name!='posix':raise RuntimeError('ORIGINAL_LINUX_PATH_OWNER_REQUIRED')
    f=ROOT/'outputs/checkpoint_point_review_v4';f.mkdir(exist_ok=True)
    lock=acquire(f/'batch.lock')
    if (f/'batch_owner.json').exists():raise RuntimeError('BATCH_ALREADY_LAUNCHED_NO_RETRY')
    parent=current(ROOT)['new_owner']
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(f/'batch_owner.json',owner)
    results=[]
    for jid in JOBS:
        while True:
            state=json.loads((ROOT/'outputs/campaign_workstation/status.json').read_text())
            if state['owner']!=parent or registry.identity(parent['pid'])!=parent['created']:
                raise RuntimeError('EXACT_CURRENT_APPROVED_PARENT_REQUIRED')
            if state['phase']=='running' and not (ROOT/'outputs/campaign_workstation/pause_requested.json').exists():break
            if state['phase'] not in ('running','draining_safe_pause'):
                raise RuntimeError('NONACTIVE_PARENT_REQUIRES_REVIEW')
            time.sleep(2)
        state=json.loads((ROOT/'outputs/campaign_workstation/status.json').read_text())
        if state['owner']!=parent or registry.identity(parent['pid'])!=parent['created']:
            raise RuntimeError('EXACT_CURRENT_APPROVED_PARENT_REQUIRED')
        row=next((r for r in state['running'] if r['job']==jid),None)
        status=json.loads((ROOT/'outputs/jobs'/jid/'status.json').read_text())
        if row is None or row['stage']!='train':
            results.append(dict(job=jid,operation='not_requested_native_lifecycle_already_advanced',actual_status=status['status']))
            continue
        if status['projected_gradient']>1e-5:
            results.append(dict(job=jid,operation='not_requested_current_PG_above_trigger',PG=status['projected_gradient']))
            continue
        write_json(f/'batch_status.json',dict(owner=owner,parent=parent,phase='cooperative_review_running',current_job=jid,results=results))
        p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'deployment/checkpoint_point_review_v4.py'),jid],
            text=True,encoding='utf8',capture_output=True)
        jobfile=f/jid/'complete.json'
        (f/(jid+'.stdout.json')).write_text(p.stdout);(f/(jid+'.stderr.txt')).write_text(p.stderr)
        if p.returncode:
            results.append(dict(job=jid,operation='review_operation_requires_diagnosis',exit_code=p.returncode,
                status_file=str(f/jid/'status.json'),automatic_retry=False))
            continue
        receipt=json.loads(jobfile.read_text())
        results.append(dict(job=jid,operation=receipt['phase'],receipt_sha256=sha(jobfile),
            before_calls=receipt['before_calls'],after_calls=receipt['after_calls'],passed=receipt['passed']))
    result=dict(owner=owner,parent=parent,phase='explicit_point_review_batch_finished',results=results,
        scientific_OS_signals=0,scientific_source_changes=0,numerical_sufficiency_certified=False)
    write_json(f/'batch_complete.json',result);write_json(f/'batch_status.json',result);print(json.dumps(result))


if __name__=='__main__':main()
