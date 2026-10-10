"""Native two-child signal/adoption fixture. No actual science job is touched."""
import json,os,signal,subprocess,sys,time
from pathlib import Path


def main(root):
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from d29_platform.runtime import write_json,sha
    from controller_lock import acquire
    from repartition_policy_v1 import verify_adoption
    from handoff_local_repartition_v1 import stopped
    if os.name!='posix':raise RuntimeError('NATIVE_LINUX_FIXTURE_ONLY')
    folder=root/'outputs/repartition_native_controls_v1';folder.mkdir(exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('EXISTING_NATIVE_FIXTURE_NO_RERUN')
    write_json(folder/'owner.json',dict(pid=os.getpid(),created=registry.identity(os.getpid())))
    code='''import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);from host_resources import registry
children=[subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(180)']) for _ in range(2)]
Path(sys.argv[2]).write_text(json.dumps(dict(owner=dict(pid=os.getpid(),created=registry.identity(os.getpid())),children=[dict(pid=p.pid,created=registry.identity(p.pid)) for p in children])))
time.sleep(180)
'''
    parent=None;children=[];leases=[];replacement=None
    try:
        parent=subprocess.Popen([sys.executable,'-B','-c',code,str(root/'scripts'),str(folder/'toy.json')])
        for _ in range(200):
            if (folder/'toy.json').exists():break
            if parent.poll() is not None:raise RuntimeError('FIXTURE_PARENT_EARLY_EXIT')
            time.sleep(.02)
        toy=json.loads((folder/'toy.json').read_text());children=toy['children']
        scope=dict(epochs={'fixtureA':'original_nonK','fixtureK':'approved_maize_K'},gates={'fixtureA':'A','fixtureK':'K'})
        running=[]
        for i,child in enumerate(children):
            l=registry.acquire('20261008_1',f'software_native/repartition_fixture_v1_{i}',threads=1,
                               peak_bytes=40000000,root=registry_root())
            if not l.get('granted'):raise RuntimeError('NATIVE_FIXTURE_RESOURCE_GATE_DENIED '+str(l))
            leases.append(l);registry.bind(l['token'],child['pid'],root=registry_root())
            bound=registry.status(root=registry_root())['leases'][l['token']]
            jid=('fixtureA','fixtureK')[i]
            running.append(dict(job=jid,pid=child['pid'],created=child['created'],stage='train',
                epoch=scope['epochs'][jid],job_gate_sha256=scope['gates'][jid],lease=dict(token=l['token'],**bound)))
        before=dict(owner=toy['owner'],approved_scope=scope,running=running)
        with registry.transaction(registry_root()):
            if registry.identity(parent.pid)!=toy['owner']['created']:raise RuntimeError('FIXTURE_PARENT_CHANGED')
            os.kill(parent.pid,signal.SIGSTOP)
            for _ in range(100):
                if stopped(parent.pid):break
                time.sleep(.01)
            else:raise RuntimeError('FIXTURE_STOP_NOT_ACKNOWLEDGED')
            if any(registry.identity(c['pid'])!=c['created'] or stopped(c['pid']) for c in children):
                raise RuntimeError('FIXTURE_CHILD_NOT_PRESERVED_DURING_PARENT_STOP')
        os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5)
        # Native replacement rereads the same PID/creation/token/CPU records.
        write_json(folder/'before.json',before)
        adopt='''import json,sys,time
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
from host_resources import registry
from repartition_policy_v1 import verify_adoption
before=json.loads(Path(sys.argv[3]).read_text());after=dict(approved_scope=before['approved_scope'],max_active=13,running=before['running'])
assert verify_adoption(before,after,before['approved_scope'],registry.identity)
Path(sys.argv[4]).write_text(json.dumps(dict(passed=True,after=after,pid=__import__('os').getpid())))
time.sleep(180)
'''
        replacement=subprocess.Popen([sys.executable,'-B','-c',adopt,str(root/'scripts'),str(root/'deployment'),
            str(folder/'before.json'),str(folder/'adopted.json')])
        for _ in range(200):
            if (folder/'adopted.json').exists():break
            if replacement.poll() is not None:raise RuntimeError('FIXTURE_ADOPTION_EARLY_EXIT')
            time.sleep(.02)
        adopted=json.loads((folder/'adopted.json').read_text())
        assert verify_adoption(before,adopted['after'],scope,registry.identity)
        active=registry.status(root=registry_root())['leases']
        if any(active[r['lease']['token']]['pid']!=r['pid'] or active[r['lease']['token']]['created']!=r['created']
               for r in running):raise RuntimeError('NATIVE_LEASE_BINDING_CHANGED')
        result=dict(passed=True,child_identities_preserved=True,children_not_stopped=True,
            same_two_tokens_and_CPUs=True,original_and_corrected_epochs_covered=True,
            scientific_signals=0,scientific_training_calls=0,actual_repartition_executed=False,
            scope='native_parent_only_signal_and_both_epoch_adoption_not_actual_science_handoff',
            toy=toy,fixture_sha256=sha(Path(__file__)),helper_sha256=sha(root/'deployment/handoff_local_repartition_v1.py'))
        write_json(folder/'complete.json',result);write_json(root/'evidence/repartition_native_controls_v1.json',result)
        print(json.dumps(result))
    finally:
        if parent is not None and parent.poll() is None:
            os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=5)
        if replacement is not None and replacement.poll() is None:replacement.terminate();replacement.wait(timeout=5)
        for child in children:
            if registry.identity(child['pid'])==child['created']:
                os.kill(child['pid'],signal.SIGTERM)
                for _ in range(100):
                    if registry.identity(child['pid'])!=child['created']:break
                    time.sleep(.02)
        for l in leases:registry.release(l['token'],root=registry_root())


if __name__=='__main__':main(sys.argv[1])
