"""Resume terminal failed phase queues, retaining all compatible fit artifacts."""
from pathlib import Path
import sys
import subprocess
import json
import os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,utc_now,sha256


def main():
    run=ROOT/'5_Test/20260905_5';statepath=run/'reports/resumed_program_queue.json'
    proof=json.loads((ROOT/'5_Test/20260905_1/reports/atomic_io_validation.json').read_text(encoding='utf-8'))
    if proof['status']!='PASS_ATOMIC_IO_RETRY' or proof['common_sha256']!=sha256(ROOT/'5_Test/20260905_1/scripts/common.py'):
        raise RuntimeError('Atomic IO recovery has not been verified')
    residual=json.loads((ROOT/'5_Test/20260905_4/reports/new_residual_evidence.json').read_text(encoding='utf-8'))
    if residual['results']['conditional_air_temperature']['probe_supported']:raise RuntimeError('Temperature mechanism review required')
    state=dict(status='RUNNING',pid=os.getpid(),runtime=RUNTIME,started_utc=utc_now(),completed=[],
        resumed_after='terminal Windows checkpoint replace failure in calendar_late_t2022_s4',
        supersedes=['20260905_4/reports/stage4_dependency_queue.json','20260905_5/reports/later_phase_queue.json'],
        storage_code_sha256=proof['common_sha256'])
    scripts=['20260905_4/scripts/run_structural_trials.py','20260905_5/scripts/run_nested_validation.py',
        '20260905_6/scripts/run_confirmation_and_final.py','20260905_6/scripts/completion_audit.py']
    for relative in scripts:
        script=ROOT/'5_Test'/relative
        state.update(status='RUNNING',active=relative,active_sha256=sha256(script),updated_utc=utc_now());atomic_json(state,statepath)
        log=run/'logs'/f'resumed_{script.stem}.log';log.parent.mkdir(parents=True,exist_ok=True)
        print('RESUMED_PHASE_STARTED',relative,flush=True)
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='REQUIRES_DIAGNOSIS',returncode=result.returncode,log=str(log),updated_utc=utc_now());atomic_json(state,statepath)
            print('RESUMED_PHASE_FAILED',relative,result.returncode,flush=True);return
        state['completed'].append(relative);atomic_json(state,statepath)
        print('RESUMED_PHASE_FINISHED',relative,flush=True)
    state.update(status='NUMERICAL_PHASES_FINISHED_REQUIRES_SCIENTIFIC_REVIEW',active=None,updated_utc=utc_now());atomic_json(state,statepath)
    print('RESUMED_NUMERICAL_PHASES_COMPLETE_REVIEW_REQUIRED',flush=True)


if __name__=='__main__':main()
