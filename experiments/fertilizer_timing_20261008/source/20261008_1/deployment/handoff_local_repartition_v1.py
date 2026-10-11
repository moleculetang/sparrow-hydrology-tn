"""Linux dispatcher-only adoption and held 39/5 ownership transition.

No scientific child is signalled. The new parent remains under an owned dispatch
hold until Windows acknowledges its exact ownership and local launch receipt.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def stopped(pid):
    p=Path('/proc')/str(pid)/'status'
    return p.exists() and any(s.startswith('State:') and s.split()[1] in ('T','t')
                             for s in p.read_text().splitlines())


def main(root,preflight):
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from d29_platform.runtime import sha,write_json
    from controller_lock import acquire
    from approved_scope_policy import load_qualified,qualify_record
    from controller_recovery_policy import resolve_record_lease
    from repartition_policy_v1 import propose,verify_adoption,IDS
    import repartition_scope_controller_v1 as controller
    if os.name!='posix':raise RuntimeError('EXACT_NATIVE_LINUX_HANDOFF_REQUIRED')
    plan=json.loads((root/'deployment/local_repartition_v1.json').read_text())
    for name,digest in plan['source_files'].items():
        if sha(root/name)!=digest:raise RuntimeError('STAGED_OPERATIONAL_BYTES_CHANGED '+name)
    before_path=root/'deployment/local_repartition_v1_before.json'
    after_path=root/'deployment/local_repartition_v1_after.json'
    before=json.loads(before_path.read_text());after=json.loads(after_path.read_text())
    capacity_path=root/'outputs/windows_capacity_probe_v1/independent_readback.json'
    if (sha(before_path)!=plan['before_ownership_sha256'] or sha(after_path)!=plan['after_ownership_sha256']
        or sha(capacity_path)!=plan['capacity_sha256'] or
        after!=propose(before,json.loads((root/'config/jobs.json').read_text()),json.loads(capacity_path.read_text()))):
        raise RuntimeError('EXACT_BYTE_BOUND_39_5_PROPOSAL_REQUIRED')
    scope=load_qualified(root);controller.SCIENCE_SCOPE=scope
    if scope!=plan['scope']:raise RuntimeError('SCIENTIFIC_SCOPE_CHANGED')
    original_gate=sha(root/'evidence/formal_gate.json')
    if original_gate!=before['formal_gate_sha256']:raise RuntimeError('ORIGINAL_GATE_CHANGED')
    controller.validate_admission()
    native=json.loads((root/'evidence/repartition_native_controls_v1.json').read_text())
    if (native.get('passed') is not True or native.get('scientific_signals')!=0
        or native.get('child_identities_preserved') is not True
        or native.get('helper_sha256')!=sha(Path(__file__))
        or native.get('fixture_sha256')!=sha(root/'deployment/test_repartition_native_v1.py')):
        raise RuntimeError('ACTUAL_NATIVE_PARENT_SIGNAL_CONTROLS_REQUIRED')
    policy=json.loads((root/'evidence/repartition_policy_controls_v1.json').read_text())
    if policy.get('passed') is not True or policy['source_sha256']!=sha(root/'deployment/repartition_policy_v1.py'):
        raise RuntimeError('CURRENT_REPARTITION_REFUSAL_CONTROLS_REQUIRED')
    expected=plan['old_owner'];artifact=plan['old_artifact_owner']
    campaign=root/'outputs/campaign_workstation'
    artifact_folder=root/'outputs/approved_frozen_allocations_workstation'

    def snapshot(resources=None):
        if sha(root/'deployment/path_ownership.json')!=plan['before_ownership_sha256']:
            raise RuntimeError('PRIOR_OWNERSHIP_CHANGED')
        if registry.identity(expected['pid'])!=expected['created']:
            raise RuntimeError('EXACT_CURRENT_SCIENCE_PARENT_REQUIRED')
        args=(Path('/proc')/str(expected['pid'])/'cmdline').read_bytes().split(b'\0')
        if str(root/'deployment/approved_scope_controller.py').encode() not in args:
            raise RuntimeError('PRIOR_SCIENCE_PARENT_COMMAND_CHANGED')
        state=json.loads((campaign/'status.json').read_text())
        if (state['owner']!=expected or state.get('approved_scope')!=scope or state['max_active']!=13
            or not 1<=len(state['running'])<=13):raise RuntimeError('CURRENT13_SCIENCE_SNAPSHOT_REQUIRED')
        leases=resources['leases'] if resources is not None else registry.status(root=registry_root())['leases']
        if any(v['pid'] in (expected['pid'],artifact['pid']) for v in leases.values()):
            raise RuntimeError('UNBOUND_DISPATCH_LEASE_DEFER_HANDOFF')
        records=state['running']
        if len({r['pid'] for r in records})!=len(records) or len({r['job'] for r in records})!=len(records):
            raise RuntimeError('DUPLICATE_CURRENT_CHILD_RECORD')
        children=set(map(int,(Path('/proc')/str(expected['pid'])/'task'/str(expected['pid'])/'children').read_text().split()))
        if not children<={r['pid'] for r in records}:raise RuntimeError('UNRECORDED_OS_SCIENCE_CHILD')
        for r in records:
            qualify_record(r,scope)
            if registry.identity(r['pid'])!=r['created']:continue
            lease=resolve_record_lease(r,leases);controller.check_child_command(r,lease)
        for jid in IDS:
            if (root/'outputs/jobs'/jid).exists() or any(r['job']==jid for r in records):
                raise RuntimeError('CANDIDATE_ALREADY_STARTED '+jid)
            if any(v['project']=='20261008_1' and v['job'].split('/')[-1]==jid for v in leases.values()):
                raise RuntimeError('CANDIDATE_ALREADY_LEASED '+jid)
        if registry.identity(artifact['pid'])!=artifact['created']:raise RuntimeError('EXACT_OLD_ARTIFACT_PARENT_REQUIRED')
        a=json.loads((artifact_folder/'status.json').read_text())
        if a['owner']!=artifact or a.get('running') is not None or a.get('approved_scope')!=scope:
            raise RuntimeError('OLD_ARTIFACT_MUST_BE_IDLE_AND_EXACT')
        args=(Path('/proc')/str(artifact['pid'])/'cmdline').read_bytes().split(b'\0')
        if str(root/'deployment/approved_frozen_allocation_controller.py').encode() not in args:
            raise RuntimeError('OLD_ARTIFACT_COMMAND_CHANGED')
        children=(Path('/proc')/str(artifact['pid'])/'task'/str(artifact['pid'])/'children').read_text().split()
        if children:raise RuntimeError('OLD_ARTIFACT_HAS_NATIVE_CHILD_NO_HANDOFF')
        return state,a

    state,astate=snapshot()
    if preflight:
        print(json.dumps(dict(passed=True,preflight_only=True,old_owner=expected,old_artifact_owner=artifact,
            current_children=len(state['running']),candidates=list(IDS),scientific_signals=0)))
        return 0
    folder=root/'outputs/local_repartition_operations_v1';folder.mkdir(exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('EXISTING_OPERATION_REVIEW_NO_RESTART')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    pause=campaign/'pause_requested.json'
    hold=dict(reason='exact_three_never_started_paths_repartition_hold',owner=owner,old_owner=expected,
              after_ownership_sha256=plan['after_ownership_sha256'],scientific_signals=0)
    stopped_parents=[];terminated=[];replacement=None
    try:
        with registry.transaction(registry_root()) as resources:
            if pause.exists():raise RuntimeError('EXISTING_SCIENCE_HOLD_NO_REPLACEMENT')
            with pause.open('x') as stream:json.dump(hold,stream,indent=2)
            for p in (expected,artifact):
                if registry.identity(p['pid'])!=p['created']:raise RuntimeError('PARENT_CHANGED_UNDER_LOCK')
                os.kill(p['pid'],signal.SIGSTOP);stopped_parents.append(p)
                for _ in range(200):
                    if stopped(p['pid']):break
                    time.sleep(.01)
                else:raise RuntimeError('PARENT_STOP_NOT_ACKNOWLEDGED')
            state,astate=snapshot(resources)
            write_json(folder/'before_science_status.json',state)
            write_json(folder/'before_artifact_status.json',astate)
        for p in (artifact,expected):
            os.kill(p['pid'],signal.SIGTERM);os.kill(p['pid'],signal.SIGCONT)
            for _ in range(200):
                if registry.identity(p['pid'])!=p['created']:break
                time.sleep(.05)
            else:raise RuntimeError('OLD_PARENT_DID_NOT_EXIT')
            terminated.append(p)
        # All checked science PIDs/tokens are retained. Only operation files change.
        target=root/'deployment/path_ownership.json'
        if sha(target)!=plan['before_ownership_sha256']:raise RuntimeError('OWNERSHIP_CHANGED_AFTER_PARENT_EXIT')
        temporary=target.with_name(target.name+'.repartition.tmp');temporary.write_bytes(after_path.read_bytes())
        os.replace(temporary,target)
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()))
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        helper=root/'deployment/repartition_scope_controller_v1.py'
        with (folder/'new_science_controller.log').open('ab') as log:
            replacement=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),'--role','workstation','--max-active','13'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(600):
            if replacement.poll() is not None:raise RuntimeError('NEW_PARENT_EXITED_KEEP_HOLD')
            new=json.loads((campaign/'status.json').read_text())
            if new['owner']['pid']==replacement.pid:break
            time.sleep(.1)
        else:raise RuntimeError('NEW_PARENT_STATUS_PENDING_KEEP_HOLD')
        if registry.identity(replacement.pid)!=new['owner']['created']:raise RuntimeError('NEW_CREATION_IDENTITY_CHANGED')
        verify_adoption(state,new,scope,registry.identity)
        expected_remote={j for j,h in after['owners'].items() if h=='workstation'}
        if set(new['owned_paths'])!=expected_remote or len(expected_remote)!=39:
            raise RuntimeError('EXACT39_REMOTE_NEW_PARENT_REQUIRED')
        if json.loads(pause.read_text())!=hold:raise RuntimeError('OWNED_HOLD_CHANGED')
        result=dict(passed=True,version=1,old_owner=expected,new_owner=new['owner'],scope=scope,
            candidates=list(IDS),before_ownership_sha256=plan['before_ownership_sha256'],
            after_ownership_sha256=plan['after_ownership_sha256'],scientific_signals=0,optimizer_edits=0,
            native_adoption_passed=True,candidates_absent_under_stopped_dispatcher=True,
            science_gate_bytes_unchanged=load_qualified(root)==scope,old_artifact_idle_exit_verified=True,
            old_artifact_owner=artifact,max_active=13,controller_sha256=sha(helper),
            verified_children=state['running'],after_children=new['running'],
            native_controls_sha256=sha(root/'evidence/repartition_native_controls_v1.json'),
            hold=hold,phase='native_remote_adoption_waiting_local_ack',source_sha256=sha(Path(__file__)))
        write_json(folder/'complete.json',result);write_json(folder/'status.json',result)
        print(json.dumps(result));return 0
    except BaseException as error:
        for p in stopped_parents:
            if p not in terminated and registry.identity(p['pid'])==p['created']:os.kill(p['pid'],signal.SIGCONT)
        if not terminated and replacement is None and pause.exists() and json.loads(pause.read_text())==hold:pause.unlink()
        write_json(folder/'status.json',dict(passed=False,error=repr(error),terminated_parents=terminated,
            replacement_pid=None if replacement is None else replacement.pid,scientific_signals=0,
            automatic_restart=False,dispatch_hold_retained=bool(terminated)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();sys.exit(main(a.root,a.preflight_only))
