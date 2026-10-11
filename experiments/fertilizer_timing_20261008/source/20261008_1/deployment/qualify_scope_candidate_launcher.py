"""One finite Linux check in a naturally released, temporarily held slot."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE=Path('/home/pc/TWY/SPARROW/Test')
ROOT=BASE/'20261008_1_maize_scope_v2'
ORIGINAL=BASE/'20261008_1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ['D29_RESOURCE_STATE']=str(BASE/'.compute_coordination/state')
from host_resources import registry,registry_root
from d29_platform.runtime import write_json,sha

PINNED=dict(pid=2026897,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1357287288',host='pc',role='workstation')


def check_parent():
    state=json.loads((ORIGINAL/'outputs/campaign_workstation/status.json').read_text())
    if state['owner']!=PINNED or registry.identity(PINNED['pid'])!=PINNED['created']:
        raise RuntimeError('PINNED_ORIGINAL_CONTROLLER_CHANGED')
    return state


def main():
    folder=ROOT/'outputs/scope_remote_point_launcher'
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'owner.json'
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    with marker.open('x',encoding='utf8') as f:json.dump(owner,f)
    staged=json.loads((ROOT/'evidence/remote_candidate_staging.json').read_text())
    manifest=ROOT/'deployment/candidate_manifest.json'
    if not staged['passed'] or sha(manifest)!=staged['candidate_manifest_sha256']:
        raise RuntimeError('VERIFIED_CURRENT_CANDIDATE_REQUIRED')
    check_parent()
    pause=ORIGINAL/'outputs/campaign_workstation/pause_requested.json'
    pause_owner=dict(reason='finite_candidate_dualhost_slot_hold_no_child_signal',owner=owner,
        controller=PINNED,manifest_sha256=sha(manifest),scientific_children_signaled=False)
    # Only dispatch is held. Existing fits and audits reach their normal exit.
    with pause.open('x',encoding='utf8') as f:json.dump(pause_owner,f)
    lease=None;child=None;peak=0
    try:
        while True:
            state=check_parent()
            leases=registry.status(root=registry_root())['leases']
            active=sum(v['project']=='20261008_1' for v in leases.values())
            decision=dict(granted=False,reason='WAIT_NATURAL_SLOT_WITHIN_MEASURED_13',active_project_leases=active)
            if active<=12:
                decision=registry.acquire('20261008_1','scope_revision_probe/maize_scope_v4_linux',
                    threads=1,peak_bytes=6500000000,root=registry_root())
            write_json(folder/'status.json',dict(owner=owner,phase='resource_or_natural_slot_wait',
                decision=decision,scientific_children_interrupted=False,production_training_dispatched=False))
            if decision.get('granted'):
                lease=decision;break
            time.sleep(20)
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',
            D29_H1_ROOT=str(BASE/'20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'))
        with (folder/'worker.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                lease['token'],'analysis/qualify_scope_remote_point.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            write_json(folder/'status.json',dict(owner=owner,phase='strict_expanded_point_running',
                child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,
                scientific_children_interrupted=False,production_training_dispatched=False))
            while child.poll() is None:
                peak=max(peak,registry.rss(child.pid));time.sleep(1)
            code=child.returncode
        registry.release(lease['token'],root=registry_root());lease=None
        check_parent()
        if json.loads(pause.read_text())!=pause_owner:
            raise RuntimeError('DISPATCH_HOLD_OWNER_CHANGED_NO_REMOVAL')
        pause.unlink()
        write_json(folder/'status.json',dict(owner=owner,phase='qualification_finished_dispatch_restored',
            exit_code=code,peak_rss_bytes=peak,original_controller=PINNED,
            scientific_children_interrupted=False,lease_released=True,
            production_training_dispatched=False,dualhost_comparison_pending=True))
        return code
    except BaseException as exc:
        live=child is not None and child.poll() is None
        if lease and not live:
            registry.release(lease['token'],root=registry_root())
        # A live child's lease is never released, nor is its source changed.
        restored=False
        if not live:
            try:
                check_parent()
                if pause.exists() and json.loads(pause.read_text())==pause_owner:
                    pause.unlink();restored=True
            except BaseException:pass
        write_json(folder/'status.json',dict(owner=owner,phase='qualification_failure_review_required',
            error=repr(exc),child_live=live,original_dispatch_restored=restored,
            scientific_children_interrupted=False,production_training_dispatched=False))
        raise


if __name__=='__main__':sys.exit(main())
