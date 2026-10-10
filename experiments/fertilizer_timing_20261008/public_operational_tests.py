"""Explicit synthetic routing controls in a temporary, private-data-free root."""
from pathlib import Path
import tempfile,shutil,json,sys,os,subprocess
HERE=Path(__file__).resolve().parent

def child(root,test):
    root=Path(root).resolve()
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if str(p).replace('\\','/').lower().startswith('e:/sparrow/') and not p.is_relative_to(HERE):
                raise PermissionError('PRIVATE_EXPERIMENT_READ_BLOCKED')
        if event in ('subprocess.Popen','os.system','os.kill','os.killpg'):
            raise PermissionError('PUBLIC_OPERATIONAL_CONTROL_MAY_NOT_DISPATCH_OR_SIGNAL')
    sys.addaudithook(guard)
    sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    import runpy
    runpy.run_path(str(root/test),run_name='__main__')

def main():
    results=[]
    tests=['deployment/test_repartition_policy_v1.py','deployment/test_repartition_allocation_controller_v1.py','deployment/test_approved_parent_lineage.py','deployment/test_allocation_launch_identity_v1.py']
    with tempfile.TemporaryDirectory(prefix='public-timing-routing-') as tmp:
        parent=Path(tmp)
        root=parent/'20261008_1'
        shutil.copytree(HERE/'source/20261008_1',root,ignore=shutil.ignore_patterns('cache','outputs','evidence','__pycache__'))
        shutil.copytree(HERE/'source/compute_coordination',parent/'compute_coordination')
        (root/'evidence').mkdir()
        jobs=json.loads((root/'config/jobs.json').read_text(encoding='utf8'))
        base_local={'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'}
        ownership=dict(owners={j['id']:'local' if j['id'] in base_local else 'workstation' for j in jobs},formal_gate_sha256='SYNTHETIC_NOT_A_SCIENTIFIC_GATE',max_active={'local':2,'workstation':12},peak_bytes={'local':6500000000})
        capacity=dict(passed=True,gate_sha256=ownership['formal_gate_sha256'],lease_closeout_verified=True,scientific_paths_dispatched=0,ownership_changed=False,all44_completion_proven=False,maximum_private_commit=1234,comparisons=[dict(workers=w,index=i,numerical=dict(passed=True)) for w,i in [(1,0),(3,0),(3,1),(3,2)]])
        for name in ['path_ownership.json','local_repartition_v1_before.json']:
            (root/'deployment'/name).write_text(json.dumps(ownership),encoding='utf8')
        f=root/'outputs/windows_capacity_probe_v1';f.mkdir(parents=True)
        (f/'independent_readback.json').write_text(json.dumps(capacity),encoding='utf8')
        env=os.environ.copy()
        for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='1'
        for test in tests:
            r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(root),test],env=env,encoding='utf8',capture_output=True)
            results.append(dict(test=test,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            print(test,r.returncode,flush=True)
            if r.returncode:break
    report=dict(passed=len(results)==len(tests) and all(r['exit_code']==0 for r in results),scope='synthetic_software_only_no_scientific_admission_no_dispatch',tests=results)
    (HERE/'public_operational_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if report['passed'] else 1

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child':child(sys.argv[2],sys.argv[3])
    else:raise SystemExit(main())
