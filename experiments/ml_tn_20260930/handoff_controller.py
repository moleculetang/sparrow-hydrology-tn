"""Freeze only dispatch supervisor; let every existing training child finish naturally."""
import os,sys,time,signal,datetime,argparse
from pathlib import Path
from mltn.common import ROOT,read,write

def state(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return None
    try:return p.read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:return None

def freeze_supervisor(pid,mod,resource_root):
    # Never suspend an owner of registry.lock: doing so would block all workers.
    with mod.transaction(resource_root):
        os.kill(pid,signal.SIGSTOP)
        while state(pid) not in ['T','t']:
            assert state(pid) is not None,'SUPERVISOR_DISAPPEARED';time.sleep(.05)

def main(previous):
    assert os.name=='posix'
    argv=[p.decode() for p in (Path('/proc')/str(previous)/'cmdline').read_bytes().split(b'\0') if p]
    valid=(argv[1:]==['-B',str(ROOT/'run_remaining.py')] or
           (len(argv)==5 and argv[1:3]==['-B',str(ROOT/'handoff_controller.py')] and argv[3]=='--previous' and argv[4].isdigit()))
    assert valid,argv
    assert read(ROOT/'outputs/remaining_owner.json')['pid']==previous
    write(ROOT/'outputs/remaining_owner.json',dict(pid=os.getpid(),host=__import__('platform').node(),start=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='safe dispatch handoff',previous_pid=previous))
    from mltn.resources import registry
    mod,resource_root=registry()
    freeze_supervisor(previous,mod,resource_root)
    children=[int(v) for v in (Path('/proc')/str(previous)/'task'/str(previous)/'children').read_text().split()]
    child_identities={pid:mod.identity(pid) for pid in children}
    write(ROOT/'outputs/dispatch_handoff.json',dict(status='waiting_for_natural_training_completion',previous_pid=previous,new_pid=os.getpid(),training_pids=children,training_signals_sent=False))
    while any(state(pid) not in [None,'Z','X'] for pid in children):
        # Keep only actual immediate requests alive during the supervisor boundary.
        # This replaces stale CPU-only counts; no future matrix jobs are claimed.
        leases=mod.status(resource_root)['leases']
        registered={v['pid'] for v in leases.values() if v['project']=='20260930_1'}
        requesting=[]
        for pid in children:
            if child_identities[pid] is None or mod.identity(pid)!=child_identities[pid] or pid in registered:continue
            try:
                args=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
            except FileNotFoundError:continue
            if not any(str(ROOT/s) in args for s in ['joint.py','train.py','auxiliary.py']):continue
            # Each launched scientific worker's first operation is its lease request.
            requesting.append(dict(pid=pid,created=child_identities[pid],argv=args))
        mod.pending('20260930_1',len(requesting),resource_root)
        write(ROOT/'outputs/dispatch_handoff_priority.json',dict(supervisor_pid=os.getpid(),actual_unleased_children=requesting,priority_count=len(requesting),leased_children=sorted(registered.intersection(children)),future_jobs_counted=False))
        time.sleep(10)
    mod.pending('20260930_1',0,resource_root)
    # The old supervisor is stopped and all its children have exited. No training PID is signalled.
    os.kill(previous,signal.SIGTERM);os.kill(previous,signal.SIGCONT)
    for _ in range(100):
        if state(previous) in [None,'Z','X']:break
        time.sleep(.1)
    else:raise RuntimeError('SUPERVISOR_DID_NOT_EXIT_SAFELY')
    write(ROOT/'outputs/dispatch_handoff.json',dict(status='completed_at_natural_child_boundary',previous_pid=previous,new_pid=os.getpid(),training_pids=children,training_signals_sent=False))
    from run_remaining import main as resume
    resume()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=int,required=True);main(p.parse_args().previous)
