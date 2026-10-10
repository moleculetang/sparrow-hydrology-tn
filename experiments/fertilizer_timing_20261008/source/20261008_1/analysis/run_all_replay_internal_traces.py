"""One local sequential diagnostic queue; full measured lease per ready child.

Does not expand the44fit matrix or interrupt scientific workers. Reuses the
completed pilot after byte checks, then traces the remaining11accepted replays.
"""
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import write_json,sha
from launch_internal_trace_pilot import main as run_one


def main():
    folder=ROOT/'outputs/replay_internal_traces';folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('ALL_REPLAY_INTERNAL_TRACES_ALREADY_COMPLETED')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    proof=json.loads((ROOT/'evidence/fixed_replays.json').read_text())
    if not proof['passed'] or proof['count']!=12:raise RuntimeError('ACCEPTED12REPLAYS_REQUIRED')
    state=dict(owner=owner,phase='sequential_local_full_history_trace',completed=[],
               fixed_replays_sha256=sha(ROOT/'evidence/fixed_replays.json'),
               new_training_calls=0,no_evaluation_outcomes_read=True)
    for case in proof['cases']:
        if sha(ROOT/case['receipt'])!=case['sha256']:raise RuntimeError('FIXED_REPLAY_CHANGED '+case['job'])
        namespace='internal_trace_pilot_v2' if case['job']=='TIM_F23_C2_V1_s0' else 'internal_trace_replays_v2'
        complete=ROOT/'outputs/jobs'/(namespace+'_'+case['job'])/'complete.json'
        if not complete.exists():
            while True:
                state.update(current_job=case['job']);write_json(folder/'status.json',state)
                try:
                    code=run_one(case['job'],namespace)
                except Exception as exc:
                    state.update(phase='diagnostic_requires_review',failed_job=case['job'],error=str(exc))
                    write_json(folder/'status.json',state)
                    raise
                if code==4:
                    state['phase']='ready_trace_waiting_shared_resource';write_json(folder/'status.json',state)
                    time.sleep(20);continue
                if code!=0:
                    state.update(phase='diagnostic_requires_review',failed_job=case['job'],exit_code=code)
                    write_json(folder/'status.json',state);return 2
                state['phase']='sequential_local_full_history_trace';break
        result=json.loads(complete.read_text())
        if not result['passed'] or result['registered_job']['id']!=case['job']:
            raise RuntimeError('TRACE_JOB_RECEIPT_MISMATCH')
        if result['source_point_sha256']!=json.loads((ROOT/case['receipt']).read_text())['point_sha256']:
            raise RuntimeError('TRACE_PARAMETER_CHANGED')
        for year in result['years']:
            if sha(complete.parent/year['file'])!=year['sha256']:raise RuntimeError('TRACE_YEAR_ARTIFACT_CHANGED')
        if sha(complete.parent/'routing_trace.npz')!=result['routing_trace_sha256'] or sha(complete.parent/'internal_and_network_shapes.csv')!=result['shapes_sha256']:
            raise RuntimeError('TRACE_NETWORK_OR_SHAPES_CHANGED')
        state['completed'].append(dict(job=case['job'],receipt=str(complete.relative_to(ROOT)),sha256=sha(complete)))
        write_json(folder/'status.json',state)
    state.update(phase='all12full_history_internal_traces_passed',passed=True,count=len(state['completed']))
    write_json(folder/'complete.json',state);write_json(folder/'status.json',state)
    return 0


if __name__=='__main__':sys.exit(main())
