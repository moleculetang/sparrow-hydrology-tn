"""Generate reviewed dual-epoch dispatch routing, retaining original lifecycle."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json


def replace_once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('EXACT_CONTROLLER_BLOCK_REQUIRED '+old[:100])
    return text.replace(old,new)


def main():
    base=ROOT/'deployment/contract_hold_controller_v3.py'
    proof=json.loads((ROOT/'evidence/contract_hold_controller_build_v3.json').read_text(encoding='utf8'))
    controls=json.loads((ROOT/'evidence/approved_scope_policy_controls.json').read_text(encoding='utf8'))
    if not proof['passed'] or proof['controller_sha256']!=sha(base):raise RuntimeError('QUALIFIED_BASE_CONTROLLER_CHANGED')
    if not controls['passed'] or controls['policy_sha256']!=sha(ROOT/'deployment/approved_scope_policy.py'):
        raise RuntimeError('ACTUAL_CURRENT_SCOPE_REFUSAL_CONTROLS_REQUIRED')
    original=base.read_text(encoding='utf8')
    pairs=[]
    pairs.append(('from train_job import validate_admission\n', '''from train_job import validate_admission
from approved_scope_policy import load_qualified,route,qualify_record,corrected_reviews

SCIENCE_SCOPE=None


def job_root(jid):
    if SCIENCE_SCOPE is None:raise RuntimeError('ACTUAL_TWO_GATE_SCOPE_NOT_LOADED')
    return route(ROOT,jid,SCIENCE_SCOPE)


def check_child_command(record,lease):
    if os.name!='posix':raise RuntimeError('APPROVED_EPOCH_CONTROLLER_CURRENTLY_WORKSTATION_ONLY')
    args=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes().split(b'\\0')
    args=[v for v in args if v]
    script={'train':'train_job.py','audit':'audit_solution.py','same_point_audit':'audit_solution.py','freeze':'freeze_job.py'}[record['stage']]
    if (str(job_root(record['job'])/'scripts/guarded_worker.py').encode() not in args
            or args[-3:]!=[lease['token'].encode(),('scripts/'+script).encode(),record['job'].encode()]):
        raise RuntimeError('LIVE_CHILD_SCIENTIFIC_ROOT_COMMAND_MISMATCH '+record['job'])
'''))
    pairs.append(("        pid=record['pid'];created=record['created'];jid=record['job']\n",
        "        pid=record['pid'];created=record['created'];jid=record['job']\n        epoch,child_gate=qualify_record(record,SCIENCE_SCOPE)\n"))
    pairs.append(("        lease=resolve_record_lease(record,resources);token=lease['token']\n",
        "        lease=resolve_record_lease(record,resources);token=lease['token']\n        check_child_command(record,lease)\n"))
    pairs.append(("            peak_rss=record['peak_rss_bytes'],adopted=True)\n",
        "            peak_rss=record['peak_rss_bytes'],adopted=True,epoch=epoch,job_gate_sha256=child_gate)\n"))
    pairs.append(("    _,gate_hash=validate_admission()\n",
        "    _,gate_hash=validate_admission()\n    global SCIENCE_SCOPE\n    SCIENCE_SCOPE=load_qualified(ROOT)\n"))
    pairs.append(("    from contract_hold_policy import held_jobs,append_holds\n    reviews=append_holds(reviews,held_jobs(ROOT,gate_hash))\n",
        "    reviews=corrected_reviews(reviews,SCIENCE_SCOPE)\n"))
    pairs.append(("        resource_peak_bytes=ownership['peak_bytes'][role],max_active=max_active,no_hard_training_budget=True)\n",
        "        resource_peak_bytes=ownership['peak_bytes'][role],max_active=max_active,no_hard_training_budget=True,\n        approved_scope=SCIENCE_SCOPE,epoch_selection='fixed_contract_without_training_or_evaluation_score_selection')\n"))
    pairs.append(("                        'scripts/'+script,jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)\n",
        "                        'scripts/'+script,jid],cwd=job_root(jid),env=env,stdout=log,stderr=subprocess.STDOUT)\n"))
    pairs.append(("str(ROOT/'scripts/guarded_worker.py'),lease['token'],\n",
        "str(job_root(jid)/'scripts/guarded_worker.py'),lease['token'],\n"))
    pairs.append(("stage=stage,peak_rss=0)\n",
        "stage=stage,peak_rss=0,epoch=SCIENCE_SCOPE['epochs'][jid],job_gate_sha256=SCIENCE_SCOPE['gates'][jid])\n"))
    pairs.append(("lease=r['lease'],peak_rss_bytes=r['peak_rss']) for j,r in running.items()],events=events)\n",
        "lease=r['lease'],peak_rss_bytes=r['peak_rss'],epoch=r['epoch'],job_gate_sha256=r['job_gate_sha256'],\n                science_root=str(job_root(j))) for j,r in running.items()],events=events)\n"))
    # Exactly three job-output root selections: recovery, exit and queue lookup.
    needle="ROOT/'outputs/jobs'/jid"
    if original.count(needle)!=3:raise RuntimeError('EXACT_THREE_JOB_FOLDER_LOOKUPS_REQUIRED')
    text=original
    for old,new in pairs:text=replace_once(text,old,new)
    text=text.replace(needle,"job_root(jid)/'outputs/jobs'/jid")
    reverse=text.replace("job_root(jid)/'outputs/jobs'/jid",needle)
    for old,new in reversed(pairs):reverse=replace_once(reverse,new,old)
    if reverse!=original:raise RuntimeError('UNREGISTERED_LIFECYCLE_CHANGE')
    target=ROOT/'deployment/approved_scope_controller.py'
    if target.exists():raise RuntimeError('EXISTING_REVIEWED_CONTROLLER_NO_OVERWRITE')
    compile(text,str(target),'exec');target.write_bytes(text.encode('utf8'))
    write_json(ROOT/'evidence/approved_scope_controller_build.json',dict(passed=True,
        base_sha256=sha(base),controller_sha256=sha(target),policy_sha256=sha(ROOT/'deployment/approved_scope_policy.py'),
        controls_sha256=sha(ROOT/'evidence/approved_scope_policy_controls.json'),
        exact_modified_blocks=['qualified_two_gate_job_root_selection','original_nonK_child_adoption_with_live_command_check',
            'remove_only_explicit_old_K_contract_holds','per_job_epoch_and_gate_provenance'],
        remaining_source_byte_equal_after_registered_reverse=True,
        existing_optimizer_or_scientific_sources_changed=False,active_controller_changed=False,
        concurrency_changed=False,scientific_jobs_dispatched=0,actual_gate_and_native_handoff_pending=True))
    print(json.dumps(dict(passed=True,controller_sha256=sha(target),dispatched=0)))


if __name__=='__main__':main()
