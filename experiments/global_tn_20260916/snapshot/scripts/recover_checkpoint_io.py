"""Resume the same logical path after a transient Windows checkpoint replace failure."""
import ctypes,time,sys,json,shutil,subprocess
import native_runtime as rt
from recover_controller import Adopted
R=rt.RUN;TAG='T24_G_M_s1'
def event(kind,**data):
 with (R/'work/resource_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(time=time.time(),event=kind,**data))+'\n')
def main():
 folder=R/'reports/checkpoint_io_recovery';folder.mkdir(exist_ok=True)
 assert not (folder/'complete.json').exists()
 c=rt.read(R/'work/controller_status.json');pid=c['process'];rt.process(pid['pid'],pid['created'])
 retained=[]
 for item in c['live']:
  try:retained.append((item,Adopted(item['pid'],item['created'])))
  except OSError:pass
 rt.write(folder/'controller_before.json',c)
 handle=rt.k32.OpenProcess(1|0x100000,False,pid['pid']);assert handle
 rt.k32.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];rt.k32.TerminateProcess.restype=ctypes.c_int
 rt.process(pid['pid'],pid['created']);assert rt.k32.TerminateProcess(handle,73);rt.k32.WaitForSingleObject(handle,60000);rt.k32.CloseHandle(handle)
 event('CHECKPOINT_IO_CONTROLLER_PAUSE',controller=pid,tag=TAG)
 # Allow any already-dispatched audits to finish; other fitting processes continue.
 for item,h in retained:
  if item['audit']:
   while h.poll() is None:rt.k32.WaitForSingleObject(h._handle,60000)
  code=h.poll()
  if code is not None:event('CHILD_EXIT',tag=item['tag'],audit=item['audit'],exit_code=code,recovered_during_io_pause=True)
  rt.k32.CloseHandle(h._handle)
 root=R/'work/jobs'/TAG;status=rt.read(root/'status.json');assert status['status']=='FAILED'
 assert not (R/'work/jobs/T24_G_D_s0/checkpoints/latest.json').exists(),'Dependent path already initialized; explicit reconciliation required'
 shutil.copy2(root/'status.json',folder/'failed_status.json');shutil.copy2(root/'fit.log',folder/'failed_fit.log')
 ref=rt.read(root/'checkpoints/latest.json');state=rt.restore(root/'checkpoints',ref['identity'])
 assert state['calls']>=status['calls'] and state['active_seconds']>=status['active_seconds']
 assert 'PermissionError' in state['exception'] and 'latest.json' in state['exception']
 rt.write(folder/'checkpoint_before.json',ref)
 # Same contents, atomic replacement, verifies the previous access failure has cleared.
 rt.write(root/'checkpoints/latest.json',ref);assert rt.restore(root/'checkpoints',ref['identity'])['calls']==state['calls']
 out=R/'outputs'/TAG
 if out.exists():
  archive=folder/'interrupted_point_audit';assert not archive.exists()
  assert out.resolve().is_relative_to(R.resolve()) and archive.resolve().is_relative_to(R.resolve())
  shutil.move(str(out),str(archive))
 for name in ['audit_pending.json','audit_failure.json']:
  p=root/name
  if p.exists():shutil.move(str(p),str(folder/name))
 status['status']='RESOURCE_YIELDED';status['resume_reason']='Transient checkpoint index replacement PermissionError; same checkpoint and budget';rt.write(root/'status.json',status)
 event('CHECKPOINT_IO_RESUME_AUTHORED',tag=TAG,calls=state['calls'],active_seconds=state['active_seconds'],checkpoint_sha256=ref['sha256'])
 log=(R/'work/controller_io_recovery.log').open('ab');child=subprocess.Popen([sys.executable,'-B',str(R/'scripts/campaign_controller.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);log.close()
 rt.write(folder/'complete.json',dict(time=time.time(),controller=rt.process(child.pid),calls=state['calls'],active_seconds=state['active_seconds'],same_scientific_path=True,core_files_unchanged=True,checkpoint_identity_unchanged=True,exception='Transient Windows PermissionError; cause of external handle not established',probe='Same-content atomic replacement and hash restore passed'))
 print('CHECKPOINT_IO_RECOVERED',child.pid,flush=True)
if __name__=='__main__':main()
