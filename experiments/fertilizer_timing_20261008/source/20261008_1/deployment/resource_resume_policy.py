"""Explicit user-authorized4:1new-grant priority; active leases unchanged."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path('/home/pc/TWY/SPARROW/Test/20261008_1')
sys.path.insert(0,str(ROOT.parent/'.compute_coordination'))
import resource_registry as registry


def main():
    state=ROOT.parent/'.compute_coordination/state';path=state/'fair_policy.json'
    previous=path.read_bytes()
    result=dict(enabled=True,weights={'20261008_1':4,'20261003_1':1},request_fresh_seconds=30,
        lease_TTL=False,active_lease_rewrite=False,physical_core_cap=40,
        user_instruction='本项目优先；撤销工作站暂停，允许SSH及资源协调')
    with registry.transaction(state) as data:
        leases_before=json.dumps(data['leases'],sort_keys=True)
        target=path.with_name('fair_policy.json.resume_'+str(os.getpid())+'.tmp')
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');os.replace(target,path)
        # Reset only the new-grant turn after replacing the policy. No PID,
        # ownership, resource reservation or CPU affinity is rewritten.
        data.setdefault('fair',dict(index=0,requests={},grants=[]))['index']=0
        if json.dumps(data['leases'],sort_keys=True)!=leases_before:raise RuntimeError('ACTIVE_LEASES_CHANGED')
    receipt=dict(passed=True,scope='resource_priority_configuration_not_scientific_dispatch',
        previous_sha256=hashlib.sha256(previous).hexdigest(),new_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        new_policy=result,active_leases_unchanged=True,scientific_signals=0,
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'evidence/fair_policy_before_explicit_resume.json').write_bytes(previous)
    (ROOT/'evidence/resource_priority_resume.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
