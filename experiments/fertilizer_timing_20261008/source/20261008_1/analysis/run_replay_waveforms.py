"""Generate the12accepted fixed replay diagnostics under one native lease."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from d29_platform.runtime import write_json,sha


def main():
    proof=json.loads((ROOT/'evidence/fixed_replays.json').read_text())
    if not proof['passed'] or proof['count']!=12:raise RuntimeError('TWELVE_FIXED_REPLAYS_REQUIRED')
    target=ROOT/'outputs/physical_shape_diagnostics_v3';target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists():raise RuntimeError('REPLAY_SHAPE_DIAGNOSTICS_ALREADY_RETURNED')
    lease=registry.acquire('20261008_1','analysis/fixed_replay_waveforms_v3',threads=1,
                           peak_bytes=1500000000,root=registry_root())
    if not lease.get('granted'):raise RuntimeError('LIGHT_ANALYSIS_RESOURCE_WAIT '+str(lease))
    env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    results=[]
    try:
        # Child processes inherit the parent's reserved core. The completed
        # v3run preceded this explicit binding; retain its source snapshot.
        registry.affinity(lease['cpus'])
        for case in proof['cases']:
            receipt=ROOT/case['receipt']
            if sha(receipt)!=case['sha256']:raise RuntimeError('FIXED_REPLAY_CHANGED_BEFORE_ANALYSIS')
            destination=target/case['job']
            result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'analysis/physical_waveform_diagnostics.py'),
                str(receipt.parent),str(destination)],cwd=ROOT,env=env,capture_output=True,encoding='utf8')
            if result.returncode:raise RuntimeError('PHYSICAL_SHAPE_ANALYSIS_FAILURE '+result.stderr)
            results.append(dict(job=case['job'],receipt=str((destination/'complete.json').relative_to(ROOT)),
                                sha256=sha(destination/'complete.json')))
            write_json(target/'progress.json',dict(completed=results,scope='fixed_replay_not_TN_evaluation'))
        write_json(target/'complete.json',dict(passed=True,count=len(results),cases=results,
            fixed_replays_sha256=sha(ROOT/'evidence/fixed_replays.json'),observations_read=False,
            scientific_sources_modified=False,model_rerun=False,lease=lease))
        print(json.dumps(dict(passed=True,count=len(results))))
    finally:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':main()
