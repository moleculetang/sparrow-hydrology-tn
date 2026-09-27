"""Bounded auxiliary study; independent resource admission and task exit receipts."""
import subprocess,sys,time
import native_runtime as rt
R=rt.RUN
def run(script,*args):
    deadline=rt.read(R/'work/experiment_clock.json')['training_deadline'];rt.resources();paused=False
    while True:
        now=rt.resources();live=[]
        cp=R/'work/controller_status.json'
        if cp.exists():
            for a in rt.read(cp).get('live',[]):
                try:
                    q=rt.process(a['pid'],a['created']);q['peak_gib']=max(q['peak_gib'],max(rt.read(R/'reports/launch_validation.json')['peak_reservations_gib'].values()));live.append(q)
                except (OSError,RuntimeError):pass
        if now['ram_percent']>=90 or (now['cpu_percent'] or 0)>=90:paused=True
        elif now['ram_percent']<85 and now['cpu_percent'] is not None and now['cpu_percent']<85:paused=False
        if time.time()>=deadline:return False
        if not paused and rt.admission(now,5.,live):break
        time.sleep(5)
    with (R/'work'/('_'.join([script.replace('.py',''),*map(str,args)])+'.log')).open('ab') as log:
        child=subprocess.Popen([sys.executable,'-B',str(R/'scripts'/script),*map(str,args)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        proc=rt.process(child.pid);rt.write(R/'work/auxiliary_process.json',dict(process=proc,reserved_peak_gib=5.,script=script,args=args));code=child.wait()
    rt.log('auxiliary_exits.jsonl',dict(script=script,args=args,exit_code=code,process=proc));return code==0
def main():
    tasks=[]
    for seed in [1729,1730,1731]:
        for structure in ['U','L3']:
            status=R/f'outputs/synthetic/{structure}_{seed}/status.json'
            if status.exists() and rt.read(status)['status']=='COMPLETE':continue
            ok=run('synthetic_study.py',structure,seed);tasks.append(dict(script='synthetic_study.py',structure=structure,seed=seed,passed=ok));rt.write(R/'reports/synthetic_suite.json',dict(status='RUNNING',tasks=tasks))
            if not ok:raise RuntimeError('SYNTHETIC_FAILED_OR_DEADLINE')
    for operation in ['pulses','controls','demand_calendar']:
        for structure in ['U','L3']:
            ok=run('signal_diagnostics.py',operation,structure);tasks.append(dict(script='signal_diagnostics.py',operation=operation,structure=structure,passed=ok));rt.write(R/'reports/synthetic_suite.json',dict(status='RUNNING',tasks=tasks))
            if not ok:raise RuntimeError('DIAGNOSTIC_FAILED_OR_DEADLINE')
    rt.write(R/'reports/synthetic_suite.json',dict(status='PASS',tasks=tasks,finished=time.time()))
if __name__=='__main__':main()
