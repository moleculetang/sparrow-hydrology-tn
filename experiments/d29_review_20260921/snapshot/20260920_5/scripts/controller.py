"""Measured-memory dispatch; child exit 75 means a resumable day-block yield."""
from runtime import *
import subprocess
def child_memory(pid):
 class Mem(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','working','qpp','qp','qnp','qn','pf','ppf')]
 op=ctypes.windll.kernel32.OpenProcess;op.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];op.restype=ctypes.c_void_p
 handle=op(0x410,False,pid)
 if not handle:return None
 try:
  m=Mem();m.cb=ctypes.sizeof(m);fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
  if not fn(handle,ctypes.byref(m),m.cb):return None
  return dict(pid=pid,rss=m.working,peak=m.peak)
 finally:
  fn=ctypes.windll.kernel32.CloseHandle;fn.argtypes=[ctypes.c_void_p];fn(handle)
def policy(cpu,ram,paused,available,total,peak,children):
 paused=(cpu>=90 or ram>=90) if not paused else not(cpu<85 and ram<85)
 headroom=sum(max(1.2*max(peak,c['peak'])-c['rss'],0) for c in children)
 needed=1.2*peak+headroom
 return paused,bool(not paused and available-.1*total>=needed),needed
def main():
 # Hysteresis and growing-worker reservation are tested before dispatch.
 assert policy(91,40,False,1000,1000,1,[])[0]
 assert policy(86,40,True,1000,1000,1,[])[0]
 assert not policy(84,84,True,1000,1000,1,[])[0]
 assert policy(1,1,False,1000,1000,100,[dict(peak=200,rss=100)])[2]==260
 profile=read(R/'reports/profile_MIX.json');assert profile['regression_pass'] and profile['peak_rss_bytes']>0
 put(R/'reports/controller_policy_tests.json',dict(hysteresis=True,growth_headroom=True,measured_peak_bytes=profile['peak_rss_bytes']))
 lock=R/'work/controller.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 protocol=read(R/'data/protocol.json')
 assert read(R/'reports/extended_checks.json')['status']=='PASS'
 pending=[x['id'] for x in protocol['configs'] if x['family'] not in ('MIX','REFERENCE') and read(R/'reports/admission.json')[x['id']]['status']=='READY'];active={};handles=[];paused=False;expired=False;started=time.time();peak=profile['peak_rss_bytes'];deadline=protocol['created']+18*3600
 try:
  while pending or active:
   for name,proc in list(active.items()):
    rc=proc.poll()
    if rc is None:continue
    out=R/'outputs'/name
    if rc==75:
     assert (out/'checkpoint.json').exists();audit=dict(exit_code=rc,resumable=True,checkpoint=read(out/'checkpoint.json')['completed_days'])
     if not expired:pending.append(name)
    elif rc==0:
     summary=read(out/'summary.json');audit=dict(exit_code=rc,status=summary['status'],summary_sha256=sha(out/'summary.json'));peak=max(peak,summary['peak']['peak_rss_bytes'])
    else:audit=dict(exit_code=rc,status='PROCESS_FAILURE',log=str(R/'logs'/f'{name}.log'))
    put(R/'reports'/f'exit_audit_{name}.json',audit);log('controller_events.jsonl',dict(event='exit_audited',arm=name,**audit));del active[name]
   snap=resource('controller',True);children=[v for proc in active.values() if (v:=child_memory(proc.pid)) is not None]
   paused,dispatch,needed=policy(snap['cpu'],snap['ram'],paused,snap['available_bytes'],snap['total_bytes'],peak,children)
   if snap['ram']>=90:(R/'work/pause.request').write_text('RAM90: checkpoint and exit75')
   elif snap['ram']<85 and (R/'work/pause.request').exists():(R/'work/pause.request').unlink()
   if time.time()>=deadline:
    expired=True;pending=[];(R/'work/stop.request').write_text('COMPUTE_BUDGET_END')
   log('controller_events.jsonl',dict(event='resource_decision',paused=paused,can_dispatch=dispatch,needed_bytes=needed,active=children,pending=pending,expired=expired))
   if pending and dispatch and not expired:
    name=pending.pop(0);f=(R/'logs'/f'{name}.log').open('a',encoding='utf-8');handles.append(f)
    p=subprocess.Popen([sys.executable,'-B',str(R/'scripts/worker.py'),name],stdout=f,stderr=subprocess.STDOUT,cwd=R,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active[name]=p
    log('controller_events.jsonl',dict(event='dispatch',arm=name,pid=p.pid,reserved_bytes=needed))
   if active or pending:time.sleep(2)
  put(R/'reports/controller_completion.json',dict(status='FINITE_TASKS_EXITED',seconds=time.time()-started,expired=expired))
 finally:
  for f in handles:f.close()
  lock.unlink()
if __name__=='__main__':main()
