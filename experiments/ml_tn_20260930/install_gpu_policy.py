"""Guarded shared GPU policy installation; CPU acquire API stays unchanged."""
import os,argparse,datetime
from pathlib import Path
from mltn.common import ROOT,write,sha
from mltn.resources import registry

def main(a):
    base=ROOT.parent/'.compute_coordination' if os.name=='posix' else ROOT.parent/'compute_coordination'
    target=base/'resource_registry.py';incoming=base/'resource_registry_gpu_slots.incoming.py'
    if a.expected_new:
        assert sha(target)==a.expected_old,'CONCURRENT_REGISTRY_CHANGE'
        assert sha(incoming)==a.expected_new,'INCOMING_REGISTRY_HASH'
        backup=ROOT/'evidence/resource_registry_remote_before_gpu_slots.py';backup.write_bytes(target.read_bytes());os.replace(incoming,target)
    mod,state=registry();write(state/'gpu_policy.json',dict(max_same_project_jobs=a.slots,project='20260930_1',phase=a.phase,device_peak_reservation_GiB=4,peak_multiplier=1.35,pause_percent=90,resume_percent=85,updated=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write(ROOT/'evidence/gpu_policy_latest.json',dict(registry_sha256=sha(target),state_root=str(state),slots=a.slots,phase=a.phase,cpu_api='same acquire/status/pending signatures and Linux CPU selection; existing leases retained'))
    print(dict(registry_sha256=sha(target),slots=a.slots,state=str(state)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-old');p.add_argument('--expected-new');p.add_argument('--slots',type=int,required=True);p.add_argument('--phase',required=True);main(p.parse_args())
