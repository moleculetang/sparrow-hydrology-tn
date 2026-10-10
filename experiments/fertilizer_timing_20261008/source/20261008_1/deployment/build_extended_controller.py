"""Exact sealed controller copy with one explicit operational admission change."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'scripts/campaign_controller.py';raw=source.read_bytes();gate=json.loads((ROOT/'evidence/formal_gate.json').read_text())
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=gate['identities']['scripts/campaign_controller.py']:raise RuntimeError('SEALED_BASE_CONTROLLER_CHANGED')
    text=raw.decode('utf8')
    needle="    if ownership['formal_gate_sha256']!=gate_hash or ownership['max_active'][role]!=max_active:\n        raise RuntimeError('MEASURED_DISPATCH_CONTRACT_MISMATCH')"
    if text.count(needle)!=1:raise RuntimeError('ONE_EXACT_ADMISSION_BLOCK_REQUIRED')
    replacement="    from concurrency_extension import admit\n    extension_admission=admit(ROOT,ownership,role,max_active,gate_hash)"
    extended=text.replace(needle,replacement)
    # The lifecycle, stage scripts, oracle, counters, ownership and lease code
    # are byte-identical outside this explicitly bounded admission block.
    if extended.replace(replacement,needle)!=text:raise RuntimeError('UNEXPECTED_LIFECYCLE_CHANGE')
    target=ROOT/'deployment/extended_campaign_controller.py';compile(extended,str(target),'exec')
    target.write_bytes(extended.encode('utf8'))
    result=dict(passed=True,sealed_base_sha256=digest,extension_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        only_changed_block='operational max_active admission',scientific_sources_changed=False,
        stage_lifecycle_and_lease_code_identical=True,new_jobs_dispatched=0)
    (ROOT/'evidence/extended_controller_build.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
