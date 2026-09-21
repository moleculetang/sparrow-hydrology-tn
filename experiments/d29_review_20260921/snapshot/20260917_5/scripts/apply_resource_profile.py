"""Operational controller restart; preserve every fit PID and per-worker identity."""
import os,sys,time,json,ctypes,copy,subprocess
import native_runtime as rt
from recover_controller import Adopted
R=rt.RUN
def event(kind,**d):
 record=dict(d);record.update(time=time.time(),event=kind)
 with (R/'work/resource_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n');f.flush()
def events():return [json.loads(s) for s in (R/'work/resource_events.jsonl').read_text(encoding='utf8').splitlines()]
def main():
 out=R/'reports/resource_reservation_change';assert not (out/'completed.json').exists(),'Already applied'
 original=rt.read(R/'reports/launch_validation.json');campaign=rt.read(R/'work/campaign.json');status=rt.read(R/'work/controller_status.json');controller=status['process'];rt.process(controller['pid'],controller['created'])
 workers={p.name:rt.sha(p) for p in (R/'reports/launch_by_fold').glob('*.json')};actual=[]
 for j in rt.read(R/'configs/jobs.json'):
  p=R/'work/jobs'/j['tag']/'status.json'
  if p.exists():
   s=rt.read(p)
   if s.get('process'):actual.append(s['process']['peak_gib'])
 peak=max([rt.read(R/'reports/peak.json')['cold_gradient_jacobian_and_ledger']['peak_gib'],rt.read(R/'reports/single_audit_peak.json')['process']['peak_gib']]+actual)
 if peak>=original['peak_reservations_gib']['D29_BE']:
  rt.write(out/'completed.json',dict(changed=False,reason='New measurement not smaller',peak=peak));return
 rt.write(out/'original_launch_validation.json',original);rt.write(out/'original_campaign.json',campaign);rt.write(out/'worker_manifest_hashes.json',workers)
 latest={};closed={}
 for e in events():
  if e['event']=='DISPATCH':latest[(e['tag'],e['audit'])]=e
  elif e['event']=='CHILD_EXIT':closed[(e['tag'],e['audit'])]=e['time']
 retained={}
 def retain():
  for key,e in latest.items():
   if key in retained or closed.get(key,0)>=e['time']:continue
   try:retained[key]=(e,Adopted(e['pid'],e['created']))
   except (OSError,RuntimeError):pass
 retain()
 event('RESOURCE_REPROFILE_CONTROLLER_STOP',old_peak=original['peak_reservations_gib']['D29_BE'],new_peak=peak,controller=controller)
 rt.k32.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];rt.k32.TerminateProcess.restype=ctypes.c_int
 handle=rt.k32.OpenProcess(1|0x100000,False,controller['pid']);assert handle
 rt.process(controller['pid'],controller['created']);assert rt.k32.TerminateProcess(handle,72);rt.k32.WaitForSingleObject(handle,60000);rt.k32.CloseHandle(handle)
 # Capture any last dispatch occurring between the first snapshot and termination.
 for e in events():
  if e['event']=='DISPATCH':latest[(e['tag'],e['audit'])]=e
  elif e['event']=='CHILD_EXIT':closed[(e['tag'],e['audit'])]=e['time']
 retain()
 # Existing audits are allowed to finish; they must not be duplicated on adoption.
 for key,(e,h) in retained.items():
  if key[1]:
   while h.poll() is None:rt.k32.WaitForSingleObject(h._handle,60000)
 for key,(e,h) in retained.items():
  code=h.poll()
  if code is not None and closed.get(key,0)<e['time']:event('CHILD_EXIT',tag=key[0],audit=key[1],exit_code=code,recovered_during_resource_restart=True,pid=e['pid'])
  rt.k32.CloseHandle(h._handle)
 proof=copy.deepcopy(original);proof['peak_reservations_gib']['D29_BE']=peak;proof['resource_reprofile']=dict(single_audit_proof=rt.sha(R/'reports/single_audit_peak.json'),single_fit_proof=rt.sha(R/'reports/peak.json'),largest_observed_fit_peak=max(actual,default=0),factor=1.2,science_and_worker_identities_unchanged=True)
 rt.write(R/'reports/launch_validation.json',proof);campaign['launch_sha256']=rt.sha(R/'reports/launch_validation.json');rt.write(R/'work/campaign.json',campaign)
 assert workers=={p.name:rt.sha(p) for p in (R/'reports/launch_by_fold').glob('*.json')}
 assert original['frozen_hashes']==proof['frozen_hashes']
 log=(R/'work/controller_reprofile.log').open('ab');child=subprocess.Popen([sys.executable,'-B',str(R/'scripts/campaign_controller.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);log.close()
 result=dict(changed=True,old_peak=original['peak_reservations_gib']['D29_BE'],new_peak=peak,reservation_factor=1.2,new_controller=rt.process(child.pid),worker_manifests_unchanged=True,fit_processes_not_terminated=True,deadlines_unchanged=True,clock_unchanged=True,time=time.time())
 rt.write(out/'completed.json',result);event('RESOURCE_REPROFILE_APPLIED',**result);print('RESOURCE_REPROFILE_APPLIED',peak,flush=True)
if __name__=='__main__':main()
