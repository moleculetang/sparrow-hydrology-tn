"""One explicit snapshot before or after verified two-contract handoff."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'analysis'), str(ROOT/'deployment')]
from host_resources import registry
from approved_scope_progress import collect, combine, sha, read
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON
from approved_parent_lineage import current


def check_local_parent(state):
    live = registry.identity(state['owner']['pid']) == state['owner']['created']
    terminal = state['phase'] in ('all_owned_paths_frozen', 'remaining_paths_require_same_point_review', 'safely_drained')
    if not live and not terminal:
        raise RuntimeError('NONTERMINAL_LOCAL_PARENT_MISSING_REVIEW')
    return live


def main():
    jobs = read(ROOT/'config/jobs.json')
    ownership = read(ROOT/'deployment/path_ownership.json')
    original_gate = sha(ROOT/'evidence/formal_gate.json')
    candidate = ROOT.with_name('20261008_1_maize_scope_v2')
    corrected = candidate/'evidence/formal_gate.json'
    corrected_gate = None
    if corrected.exists():
        value = read(corrected)
        if not value.get('passed') or not value.get('sealed'):
            raise RuntimeError('PENDING_CANDIDATE_NOT_A_FORMAL_PROGRESS_GATE')
        corrected_gate = sha(corrected)
    handoff = current(ROOT)
    if handoff:
        if not handoff.get('passed'):
            raise RuntimeError('ACTUAL_HANDOFF_RETURN_NOT_PASSED')
        expected = handoff['new_owner']
    else:
        expected = read(ROOT/'outputs/campaign_admission/status.json')['remote_owner']
    local_state = read(ROOT/'outputs/campaign_local/status.json')
    local_live = check_local_parent(local_state)
    local_records = collect(ROOT, jobs, ownership, 'local', original_gate, corrected_gate)
    source = (ROOT/'analysis/approved_scope_progress.py').read_text(encoding='utf8')
    code = f'''
import json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
ns={{'__name__':'readonly_approved_scope_progress'}}
exec(compile({source!r},'readonly_approved_scope_progress','exec'),ns)
state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
expected={expected!r}
live=reg.identity(expected['pid'])==expected['created']
terminal=state['phase'] in ('all_owned_paths_frozen','remaining_paths_require_same_point_review','safely_drained')
if state['owner']!=expected or (not live and not terminal):raise RuntimeError('EXACT_REMOTE_PARENT_OR_TERMINAL_IDENTITY_REQUIRED')
jobs=json.loads((root/'config/jobs.json').read_text());ownership=json.loads((root/'deployment/path_ownership.json').read_text())
if jobs!={jobs!r} or ownership!={ownership!r}:raise RuntimeError('ACTUAL_HOST_MATRIX_OR_OWNERSHIP_CHANGED')
records=ns['collect'](root,jobs,ownership,'workstation',{original_gate!r},{corrected_gate!r})
leases=reg.status(root=root.parent/'.compute_coordination/state')['leases']
head=sum(max(0,r['reserved_bytes']-reg.rss(r['pid'])) for r in leases.values())
memory=reg.memory();a=reg.cpu_sample();time.sleep(1);b=reg.cpu_sample()
running=[dict(job=r['job'],stage=r['stage'],pid=r['pid'],created=r['created'],
 live=reg.identity(r['pid'])==r['created'],rss_bytes=reg.rss(r['pid']),cpus=r['lease']['cpus'],
 epoch=r.get('epoch','original_nonK')) for r in state['running']]
print(json.dumps(dict(records=records,owner=expected,controller_live=live,phase=state['phase'],
 max_active=state['max_active'],running=running,memory=memory,
 cpu_percent=100*(1-(b[1]-a[1])/(b[0]-a[0])),commit_with_reserve_fraction=(memory['commit_used']+head)/memory['commit_total'],
 pause_requested=(root/'outputs/campaign_workstation/pause_requested.json').exists())))
'''
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr+run.stdout)
    remote = json.loads(run.stdout)
    result = combine(local_records, remote['records'], jobs)
    result.update(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  remote=remote, local=dict(owner=local_state['owner'], controller_live=local_live,
                         phase=local_state['phase'], memory=registry.memory(), running=local_state['running']))
    folder = ROOT/'outputs/approved_scope_progress'
    folder.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    path = folder/(stamp+'.json')
    with path.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(snapshot=str(path), counts=result['counts'], remote_running=len(remote['running']),
                          remote_phase=remote['phase'], remote_cpu_percent=remote['cpu_percent'],
                          remote_commit_with_reserve_fraction=remote['commit_with_reserve_fraction'])))


if __name__ == '__main__':
    main()
