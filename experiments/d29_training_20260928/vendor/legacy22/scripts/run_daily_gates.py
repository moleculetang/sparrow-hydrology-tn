"""Sequential bounded cold-process acceptance with durable per-gate logs."""
import subprocess,sys,time
import native_runtime as rt
R=rt.RUN
def run(script,*args):
    p=R/'work'/('gate_'+script.replace('.py','')+'_'+'_'.join(args)+'.log')
    with p.open('ab') as f:
        result=subprocess.run([sys.executable,'-B',str(R/'scripts'/script),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    rt.log('gate_exits.jsonl',dict(script=script,args=args,exit_code=result.returncode))
    if result.returncode:raise RuntimeError(str(p))
    print('PASS',script,*args,flush=True)
if __name__=='__main__':
    run('daily_fixtures.py')
    for j in rt.read(R/'configs/jobs.json'):
        if j['start']!=0:continue
        if not (R/'reports'/f"preflight_daily_{j['tag']}.json").exists():run('preflight_daily.py',j['tag'])
    run('infrastructure_checks.py');run('test_solver_resume.py');run('check_label_isolation.py');run('test_evaluation_pipeline.py')
    rt.write(R/'reports/gate_runner.json',dict(status='PASS',finished=time.time()))
