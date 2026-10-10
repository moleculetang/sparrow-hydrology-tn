"""Register minimal two-contract routing changes to the qualified exporter."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'deployment')]
from d29_platform.runtime import sha, write_json


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('EXACT_ALLOCATION_BLOCK_REQUIRED '+old[:80])
    return text.replace(old, new)


def main():
    base = ROOT/'deployment/frozen_allocation_controller.py'
    controls = json.loads((ROOT/'evidence/frozen_allocation_controller_known_errors.json').read_text(encoding='utf8'))
    policy = json.loads((ROOT/'evidence/approved_allocation_policy_controls.json').read_text(encoding='utf8'))
    if not controls['passed'] or controls['source_sha256'] != sha(base):
        raise RuntimeError('ACTUAL_CURRENT_BASE_LIFECYCLE_CONTROLS_REQUIRED')
    if not policy['passed'] or policy['source_sha256'] != sha(ROOT/'deployment/approved_allocation_policy.py'):
        raise RuntimeError('ACTUAL_CURRENT_TWO_CONTRACT_ACK_CONTROLS_REQUIRED')
    original = base.read_text(encoding='utf8')
    start = original.index('def transport_admitted(')
    end = original.index('\n\ndef main(role):', start)
    old_transport = original[start:end]
    new_transport = '''def transport_admitted(root,role,gate,cache=None):
    if role!='workstation':raise RuntimeError('TWO_CONTRACT_ALLOCATIONS_CURRENTLY_WORKSTATION_ONLY')
    current_owner=read(root/'outputs/campaign_workstation/owner.json')
    if not current_owner:return False
    ack_path=root/'evidence'/('approved_scope_terminal_import_ack_'+owner_key(current_owner)+'.json')
    ack=read(ack_path)
    ownership_path=root/'deployment/path_ownership.json';ownership=read(ownership_path)
    jobs=sorted(j for j,host in ownership['owners'].items() if host==role)
    if not qualify_ack(ack,current_owner,jobs,SCIENCE_SCOPE,sha(ownership_path)):return False
    key=(sha(ack_path),json.dumps(current_owner,sort_keys=True),sha(ownership_path))
    if cache is not None and cache.get('admitted_key')==key:return True
    path=(root/ack['archive_relative']).resolve()
    if not path.is_relative_to((root/'transfer').resolve()) or sha(path)!=ack['archive_sha256']:
        raise RuntimeError('EXACT_TWO_CONTRACT_IMPORTED_SNAPSHOT_REQUIRED')
    if cache is not None:cache['admitted_key']=key
    return True
'''
    pairs = [
        ('from d29_platform.runtime import sha,write_json\n', '''from d29_platform.runtime import sha,write_json
from approved_scope_policy import load_qualified,route
from approved_allocation_policy import qualify_ack,owner_key

SCIENCE_SCOPE=None


def job_root(jid):
    if SCIENCE_SCOPE is None:raise RuntimeError('ACTUAL_TWO_CONTRACT_SCOPE_REQUIRED')
    return route(ROOT,jid,SCIENCE_SCOPE)


def stage_for_job(job_id):
    return stage(job_root(job_id),job_id,SCIENCE_SCOPE['gates'][job_id])
'''),
        (old_transport, new_transport),
        ('    _,gate=validate_admission()\n', '''    _,gate=validate_admission()
    global SCIENCE_SCOPE
    SCIENCE_SCOPE=load_qualified(ROOT)
    if role!='workstation':raise RuntimeError('EXISTING_LOCAL_ARTIFACT_CONTROLLER_RETAINED')
    legacy=read(ROOT/'outputs/frozen_allocations_workstation/status.json')
    if legacy and registry.identity(legacy['owner']['pid'])==legacy['owner']['created']:
        if legacy.get('running'):
            raise RuntimeError('LIVE_LEGACY_ARTIFACT_CHILD_NO_DUPLICATE_CONTROLLER')
        from frozen_allocation_controller import transport_admitted as legacy_admission
        if legacy_admission(ROOT,role,gate):
            raise RuntimeError('ADMITTED_LEGACY_ARTIFACT_CONTROLLER_REQUIRES_SAFE_HANDOFF')
'''),
        ("folder=ROOT/'outputs'/('frozen_allocations_'+role)", "folder=ROOT/'outputs'/('approved_frozen_allocations_'+role)"),
        ("    old=read(folder/'status.json')\n", "    old=read(folder/'status.json')\n    if old and (old.get('approved_scope')!=SCIENCE_SCOPE or old.get('source_sha256')!=sha(Path(__file__))):\n        raise RuntimeError('EXACT_ARTIFACT_CONTROLLER_SCOPE_AND_SOURCE_REQUIRED_FOR_RECOVERY')\n"),
        ("ROOT/'evidence/frozen_allocation_controller_known_errors.json'", "ROOT/'evidence/approved_allocation_controller_controls.json'"),
        ("source_sha256=sha(Path(__file__)),\n", "source_sha256=sha(Path(__file__)),approved_scope=SCIENCE_SCOPE,\n"),
        ("str(ROOT/'scripts/guarded_worker.py'),lease['token'],relative,jid],cwd=ROOT,env=env",
         "str(job_root(jid)/'scripts/guarded_worker.py'),lease['token'],relative,jid],cwd=job_root(jid),env=env"),
        ("                if os.name=='posix':env['D29_RESOURCE_STATE']=str(registry_root())\n",
         "                if os.name=='posix':\n                    env['D29_RESOURCE_STATE']=str(registry_root())\n                    env['D29_H1_ROOT']='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'\n"),
    ]
    text = original
    for old, new in pairs:
        text = replace_once(text, old, new)
    lookups = [("stage(ROOT,running['job'],gate)", "stage_for_job(running['job'])", 2),
               ('stage(ROOT,jid,gate)', 'stage_for_job(jid)', 1)]
    for old, new, count in lookups:
        if text.count(old) != count:
            raise RuntimeError('EXACT_ALLOCATION_JOB_ROOT_LOOKUP_COUNT_REQUIRED')
        text = text.replace(old, new)
    reversed_text = text
    for old, new, count in reversed(lookups):
        if reversed_text.count(new) != count:
            raise RuntimeError('ALLOCATION_REVERSE_LOOKUP_COUNT_CHANGED')
        reversed_text = reversed_text.replace(new, old)
    for old, new in reversed(pairs):
        reversed_text = replace_once(reversed_text, new, old)
    if reversed_text != original:
        raise RuntimeError('UNREGISTERED_ARTIFACT_LIFECYCLE_CHANGE')
    target = ROOT/'deployment/approved_frozen_allocation_controller.py'
    if target.exists():
        raise RuntimeError('EXISTING_APPROVED_ALLOCATION_CONTROLLER_NO_OVERWRITE')
    compile(text, str(target), 'exec')
    target.write_bytes(text.encode('utf8'))
    proof = dict(passed=True, base_sha256=sha(base), controller_sha256=sha(target),
                 policy_sha256=sha(ROOT/'deployment/approved_allocation_policy.py'),
                 registered_blocks=['fixed_two_contract_stage_and_child_root', 'true_two_contract_import_admission',
                                    'separate_remote_artifact_controller_namespace', 'guard_against_legacy_live_exports',
                                    'exact_source_and_scope_before_artifact_recovery',
                                    'explicit_verified_H1_runtime_root'],
                 original_source_byte_equal_after_registered_reverse=True,
                 max_active=1, resource_peak_reservation_changed=False,
                 scientific_sources_changed=False, production_launched=False)
    write_json(ROOT/'evidence/approved_allocation_controller_build.json', proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
