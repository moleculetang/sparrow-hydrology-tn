"""Derive a passive staged entry; preserve the audited numerical replay body.

Only input admission and output namespace differ. Formal44 admission stays closed.
"""
from pathlib import Path
import json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
SOURCE=ROOT/'analysis/trace_approved_internal_fluxes.py'
TARGET=ROOT/'analysis/trace_staged_internal_fluxes.py'


def derive(source):
    old="""    if mode == 'preflight':
        if jid != 'TIM_F24_C0_V0_s0':
            raise RuntimeError('PREFLIGHT_ONLY_FULLY_ACCEPTED_BASELINE')
        folder = ROOT/'outputs/jobs'/jid
    elif mode == 'final':
        from approved_epoch_delivery import admitted_folders
        folder = admitted_folders(ROOT)[jid]
    else:
        raise RuntimeError('DECLARED_TRACE_MODE_REQUIRED')
"""
    new="""    if mode != 'staged':
        raise RuntimeError('STAGED_PHYSICAL_ONLY_ENTRY_REQUIRED')
    from staged_frozen_sources import resolve
    folder = resolve(jid)
"""
    if source.count(old)!=1:raise RuntimeError('EXACT_AUDITED_UPSTREAM_ENTRY_REQUIRED')
    result=source.replace(old,new)
    # Receipt and output namespaces already depend on the explicit mode.
    compile(result,str(TARGET),'exec')
    return result


def main():
    source=SOURCE.read_text(encoding='utf8')
    from d29_platform.runtime import sha
    controls=json.loads((ROOT/'evidence/approved_internal_trace_controls.json').read_text())
    if not controls.get('passed') or controls.get('worker_sha256')!=sha(SOURCE):
        raise RuntimeError('AUDITED_UPSTREAM_CONTROL_REQUIRED')
    result=derive(source)
    if TARGET.exists():raise RuntimeError('IMMUTABLE_STAGED_WORKER_ALREADY_BUILT')
    TARGET.write_text(result,encoding='utf8')
    proof=dict(passed=True,upstream_sha256=sha(SOURCE),staged_worker_sha256=sha(TARGET),
        scientific_body_identical=True,only_admission_changed=True,
        evaluation_admitted=False,training_ownership_changed=False)
    (ROOT/'evidence/staged_trace_derivation.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    print(json.dumps(proof))


if __name__=='__main__':main()
