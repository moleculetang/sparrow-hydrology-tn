"""Stop only the verified experiment controller; workers yield safely."""
import os,time,ctypes
import native_runtime as rt
R=rt.RUN;s=rt.read(R/'work/controller_status.json');p=s['process']
verified=rt.process(p['pid'],p['created']);assert verified['alive']
h=rt.k32.OpenProcess(0x1000|0x100000|1,False,p['pid']);assert h
ts=[rt.w.FILETIME() for _ in range(4)];assert rt.k32.GetProcessTimes(h,*map(ctypes.byref,ts));assert rt.ticks(ts[0])==p['created']
rt.write(R/'evidence/isolation_pause.json',dict(reason='Global checksum pass read other-fold file bytes, not numeric labels; replace with per-path manifests',controller=s,time=time.time(),verified_creation=p['created']))
rt.k32.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];rt.k32.TerminateProcess.restype=ctypes.c_int
assert rt.k32.TerminateProcess(h,0);rt.k32.WaitForSingleObject(h,10000);rt.k32.CloseHandle(h)
jobs=rt.read(R/'configs/jobs.json')
for j in jobs:
 root=R/'work/jobs'/j['tag']
 if root.exists():rt.write(root/'yield.request',dict(reason='ISOLATION_MANIFEST_REPAIR',preserve_optimizer=True))
time.sleep(5)
until=time.monotonic()+120
while True:
 live=[]
 for j in jobs:
  q=R/'work/jobs'/j['tag']/'status.json'
  if not q.exists():continue
  z=rt.read(q)
  if z.get('status')=='RUNNING':
   try:
    a=rt.process(z['process']['pid'],z['process']['created'])
    if a['alive']:live.append(j['tag'])
   except OSError:pass
 if not live:break
 if time.monotonic()>until:raise RuntimeError(('WORKER_NOT_YIELDED',live))
 time.sleep(2)
rt.write(R/'evidence/isolation_pause_complete.json',dict(time=time.time(),statuses={j['tag']:rt.read(R/'work/jobs'/j['tag']/'status.json')['status'] if (R/'work/jobs'/j['tag']/'status.json').exists() else 'PENDING' for j in jobs}))
print('SAFE_PAUSE_COMPLETE',flush=True)
