"""Exact three-path operational change; scientific scope stays 30+14 of 44."""
import copy
import json
from pathlib import Path,PurePosixPath

IDS=('TIM_S191_C0_V0_s0','TIM_S191_C1_V0_s0','TIM_S191_C2_V0_s0')
BASE_LOCAL={'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'}


def portable_relative(relative):
    if not isinstance(relative,str):raise RuntimeError('STRING_RELATIVE_SOURCE_PATH_REQUIRED')
    normalized=relative.replace('\\','/')
    path=PurePosixPath(normalized)
    if (not normalized or path.is_absolute() or '..' in path.parts or ':' in normalized
        or path.as_posix()!=normalized):raise RuntimeError('SAFE_PORTABLE_RELATIVE_SOURCE_PATH_REQUIRED')
    return normalized


def propose(before,jobs,capacity):
    ids={j['id'] for j in jobs}
    if len(jobs)!=44 or len(ids)!=44 or set(before['owners'])!=ids:
        raise RuntimeError('EXACT_UNIQUE44_OWNERSHIP_REQUIRED')
    if ({j for j,h in before['owners'].items() if h=='local'}!=BASE_LOCAL or
        any(h not in ('local','workstation') for h in before['owners'].values())):
        raise RuntimeError('EXACT_ORIGINAL42_2_REQUIRED')
    if (capacity.get('passed') is not True or capacity.get('gate_sha256')!=before['formal_gate_sha256']
        or capacity.get('lease_closeout_verified') is not True
        or capacity.get('scientific_paths_dispatched')!=0 or capacity.get('ownership_changed') is not False
        or capacity.get('all44_completion_proven') is not False
        or not 0<capacity.get('maximum_private_commit',0)<=before['peak_bytes']['local']
        or len(capacity.get('comparisons',[]))!=4
        or {(c.get('workers'),c.get('index')) for c in capacity['comparisons']}!={(1,0),(3,0),(3,1),(3,2)}
        or any(c.get('numerical',{}).get('passed') is not True for c in capacity['comparisons'])):
        raise RuntimeError('ACTUAL_COMPLETE_SAME_GATE_WINDOWS_CAPACITY_REQUIRED')
    if before['max_active']!={'local':2,'workstation':12}:
        raise RuntimeError('ORIGINAL_MEASURED_HOST_LIMITS_REQUIRED')
    for jid in IDS:
        job=next(j for j in jobs if j['id']==jid)
        if (before['owners'][jid]!='workstation' or job['calendar']=='K'
            or job['plant_version']!='V0' or job['space_block']!=191
            or job['entry']!=0 or job['fixed_log_source_multiplier']!=0.):
            raise RuntimeError('EXACT_S191_V0_NONK_CANDIDATES_REQUIRED')
    after=copy.deepcopy(before)
    after['owners'].update({j:'local' for j in IDS})
    after['max_active']['local']=3
    return after


def verify_adoption(before,after,scope,identity):
    if before['approved_scope']!=scope or after.get('approved_scope')!=scope or after['max_active']!=13:
        raise RuntimeError('UNCHANGED_SCOPE_AND13_CAPACITY_REQUIRED')
    old=before['running'];new={r['job']:r for r in after['running']}
    if len(new)!=len(after['running']) or any(r['job'] in IDS for r in old):
        raise RuntimeError('NO_CANDIDATE_ALREADY_RUNNING_OR_DUPLICATE_CHILD')
    if not set(new)<={r['job'] for r in old}:
        raise RuntimeError('DISPATCH_HOLD_REQUIRED_NO_NEW_CHILDREN')
    for r in old:
        if r.get('epoch')!=scope['epochs'][r['job']] or r.get('job_gate_sha256')!=scope['gates'][r['job']]:
            raise RuntimeError('EXACT_BOTH_EPOCHS_REQUIRED')
        if identity(r['pid'])!=r['created']:continue
        n=new.get(r['job'])
        if not n or any(n[k]!=r[k] for k in ('pid','created','stage','epoch','job_gate_sha256')):
            raise RuntimeError('LIVE_CHILD_IDENTITY_NOT_PRESERVED')
        if n['lease']['token']!=r['lease']['token'] or n['lease']['cpus']!=r['lease']['cpus']:
            raise RuntimeError('LIVE_CHILD_LEASE_OR_CPU_CHANGED')
    return True


def read_ownership(root):
    """Only original 42/2 or the exact proven proposal; arbitrary partitions fail."""
    from d29_platform.runtime import sha
    root=Path(root)
    path=root/'deployment/path_ownership.json'
    value=json.loads(path.read_text(encoding='utf8'))
    local={j for j,h in value['owners'].items() if h=='local'}
    if local==BASE_LOCAL:
        if len(value['owners'])!=44 or any(h not in ('local','workstation') for h in value['owners'].values()):
            raise RuntimeError('EXACT_ORIGINAL42_2_REQUIRED')
        return value
    if local!=BASE_LOCAL|set(IDS):raise RuntimeError('UNAPPROVED_HOST_PARTITION')
    plan=json.loads((root/'deployment/local_repartition_v1.json').read_text())
    before_path=root/'deployment/local_repartition_v1_before.json'
    after_path=root/'deployment/local_repartition_v1_after.json'
    capacity_path=root/'outputs/windows_capacity_probe_v1/independent_readback.json'
    if (plan.get('version')!=1 or plan.get('candidates')!=list(IDS)
        or sha(before_path)!=plan['before_ownership_sha256']
        or sha(after_path)!=plan['after_ownership_sha256'] or sha(path)!=sha(after_path)
        or sha(capacity_path)!=plan['capacity_sha256']):
        raise RuntimeError('EXACT_BYTE_BOUND_REPARTITION_DECLARATION_REQUIRED')
    before=json.loads(before_path.read_text());capacity=json.loads(capacity_path.read_text())
    jobs=json.loads((root/'config/jobs.json').read_text())
    if value!=propose(before,jobs,capacity):raise RuntimeError('ONLY_EXACT_PROVED_THREE_PATH_CHANGE_ALLOWED')
    normalized=[portable_relative(p) for p in capacity['source_sha256']]
    expected={'deployment/verify_windows_capacity_probe_v1.py','deployment/windows_capacity_probe_v1.py','deployment/measure_windows_parallel_targets_v1.py'}
    if len(normalized)!=3 or set(normalized)!=expected:raise RuntimeError('EXACT_THREE_PORTABLE_CAPACITY_SOURCES_REQUIRED')
    for relative,digest in capacity['source_sha256'].items():
        if sha(root/portable_relative(relative))!=digest:raise RuntimeError('CAPACITY_SOURCE_BYTES_CHANGED')
    return value


def project_parent(base,receipt,plan,ownership_hash):
    if (receipt.get('passed') is not True or receipt.get('version')!=1
        or receipt.get('old_owner')!=base['new_owner'] or receipt.get('scope')!=base['scope']
        or receipt.get('before_ownership_sha256')!=plan['before_ownership_sha256']
        or receipt.get('after_ownership_sha256')!=plan['after_ownership_sha256']
        or ownership_hash!=plan['after_ownership_sha256']
        or receipt.get('candidates')!=list(IDS) or receipt.get('scientific_signals')!=0
        or receipt.get('optimizer_edits')!=0 or receipt.get('native_adoption_passed') is not True
        or receipt.get('candidates_absent_under_stopped_dispatcher') is not True
        or receipt.get('science_gate_bytes_unchanged') is not True
        or receipt.get('old_artifact_idle_exit_verified') is not True
        or receipt.get('max_active')!=13):
        raise RuntimeError('ACTUAL_COMPLETE_REPARTITION_PARENT_RECEIPT_REQUIRED')
    owner=receipt.get('new_owner',{})
    if (owner==base['new_owner'] or type(owner.get('pid')) is not int or owner['pid']<=0
        or not owner.get('created') or owner.get('host')!='pc' or owner.get('role')!='workstation'):
        raise RuntimeError('EXACT_NEW_REPARTITION_PARENT_REQUIRED')
    result=copy.deepcopy(base)
    result.update(new_owner=owner,local_repartition_transition=copy.deepcopy(receipt),
                  original_legacy_observer_parent_preserved=True)
    return result
