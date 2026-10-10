"""One resource-bound local trace at an independently accepted frozen fit."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import write_json,sha
from assemble_approved_delivery import validate_frozen


def main(jid,namespace):
    if os.name!='nt':raise RuntimeError('NEW_SMALL_FROZEN_POINT_TRACE_USES_LOCAL_HOST')
    if not namespace.startswith('internal_trace_') or any(c in namespace for c in '/\\:'):
        raise RuntimeError('NEW_SEPARATE_INTERNAL_TRACE_NAMESPACE_REQUIRED')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    source=ROOT/'outputs/jobs'/jid
    gate=sha(ROOT/'evidence/formal_gate.json')
    bound=validate_frozen(source,job,gate,json.loads((ROOT/'evidence/expected_station_support.json').read_text()))
    folder=ROOT/'outputs'/(namespace+'_'+jid+'_launch');folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists() or (folder/'returned.json').exists():
        raise RuntimeError('EXISTING_TRACE_OPERATION_REVIEW_NO_RESPAWN')
    target=ROOT/'outputs/jobs'/(namespace+'_'+jid)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('EXISTING_TRACE_RESULTS_REVIEW_NO_OVERWRITE')
    lease=registry.acquire('20261008_1','analysis/'+namespace+'/'+jid,threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(folder/'resource_decision.json',lease)
    if not lease.get('granted'):return 4
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONPATH=str(ROOT/'vendor/legacy22/scripts'))
    child=None;peak=dict(rss=0,private_commit=0,peak_commit=0);started=time.monotonic()
    try:
        with (folder/'trace.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
              lease['token'],'analysis/trace_accepted_internal_fluxes.py',str(source),'--namespace',namespace],
              cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),
              child_pid=child.pid,child_created=registry.identity(child.pid),lease=lease)
            write_json(folder/'owner.json',owner)
            write_json(folder/'status.json',dict(phase='accepted_fit_full_history_trace_running',owner=owner,
              registered_job=jid,accepted_source_bound_files=bound,source_freeze_sha256=sha(source/'prediction_freeze.json'),
              scientific_fit_signals=0,new_training_calls=0,evaluation_labels_read=False))
            while child.poll() is None:
                m=registry.process_memory(child.pid)
                for k in peak:
                    if m.get(k) is not None:peak[k]=max(peak[k],m[k])
                time.sleep(5)
        complete=target/'complete.json'
        passed=child.returncode==0 and complete.exists() and json.loads(complete.read_text()).get('passed') is True
        if passed:
            result=json.loads(complete.read_text())
            passed=(result['formal_gate_sha256']==gate and result['source_point_sha256']==bound['frozen_parameters.npy']
              and result['no_evaluation_outcomes_read'] and result['new_scientific_training_calls']==0)
        record=dict(passed=bool(passed),exit_code=child.returncode,owner=owner,native_peak=peak,
          seconds=time.monotonic()-started,complete_sha256=sha(complete) if passed else None,
          accepted_source_freeze_sha256=bound['prediction_freeze.json'],physical_sources_modified=False,
          new_training_calls=0,evaluation_labels_read=False)
        write_json(folder/'returned.json',record)
        write_json(folder/'status.json',dict(phase='accepted_fit_trace_passed' if passed else 'accepted_fit_trace_requires_review',**record))
        print(json.dumps(record));return 0 if passed else 2
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--namespace',default='internal_trace_acceptedfit_v1')
    a=p.parse_args();sys.exit(main(a.job,a.namespace))
