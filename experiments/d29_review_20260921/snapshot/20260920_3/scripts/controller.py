"""Independent finite controller; no scheduler. Consumes every worker exit."""
from runtime import *
import subprocess
def main():
 lock=R/'work/controller.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 started=time.time();active={};pending=['H1','H0'];handles=[];paused=False
 try:
  assert read(R/'reports/preflight.json')['status']=='PASS'
  assert sha(R/'data/events_frozen.parquet')==read(R/'data/events_freeze.json')['sha256']
  freeze={str(p.relative_to(R)):sha(p) for p in (R/'scripts').glob('*.py')}
  put(R/'data/execution_freeze.json',dict(time=started,scripts=freeze,protocol=sha(R/'data/protocol.json')))
  while pending or active:
   cpu,ram=resource('controller_sample');paused=(cpu>=90 or ram>=90) if not paused else not(cpu<85 and ram<85)
   if time.time()-started>10*3600:pending=[];log('controller.jsonl',dict(event='compute_budget_exhausted'))
   # Two independent hydrology tasks, not an arbitrary worker cap.
   if pending and not paused:
    h=pending.pop(0);out=(R/'logs'/f'{h}.log').open('a',encoding='utf-8');handles.append(out)
    p=subprocess.Popen([sys.executable,'-B',str(R/'scripts/runner.py'),h],stdout=out,stderr=subprocess.STDOUT,cwd=R,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    active[h]=p;log('controller.jsonl',dict(event='dispatch',hydro=h,pid=p.pid))
   for h,p in list(active.items()):
    rc=p.poll()
    if rc is not None:
     files=list((R/'outputs').glob(h+'_*/summary.json'));audit=[dict(path=str(f),sha256=sha(f),status=read(f)['physical_status']) for f in files]
     put(R/'reports'/f'exit_audit_{h}.json',dict(exit_code=rc,results=audit))
     log('controller.jsonl',dict(event='exit_audited',hydro=h,exit_code=rc,results=len(audit)));del active[h]
   if active or pending:time.sleep(15)
  put(R/'reports/controller_completion.json',dict(elapsed_seconds=time.time()-started,finished=time.time(),status='WORKERS_EXITED'))
 finally:
  for h in handles:h.close()
  lock.unlink()
if __name__=='__main__':main()
