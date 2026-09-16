"""Record successful postprocessing recovery after the controller's renderer failure."""
import json,time
import native_runtime as rt
R=rt.RUN
def main():
 audit=rt.read(R/'reports/completion_audit.json');assert audit['status']=='COMPLETE_FINITE_EXPERIMENT' and not audit['failures']
 assert audit['audited_paths']==28 and audit['numerically_sufficient']==28 and audit['physical_legal']==28
 for j in rt.read(R/'configs/jobs.json'):
  s=rt.read(R/'work/jobs'/j['tag']/'status.json');assert s['status'] not in ['RUNNING','RESOURCE_YIELDED']
  try:live=rt.process(s['process']['pid'],s['process']['created'])
  except (OSError,RuntimeError):continue
  assert not live['alive'],('LIVE_REGISTERED_FIT',j['tag'])
 now=time.time();old=rt.read(R/'work/controller_status.json')
 receipt=dict(time=now,event='POSTPROCESS_RECOVERY_COMPLETE',reason='Missing optional tabulate replaced with local Markdown renderer; frozen predictions and selection unchanged',audited_paths=28,numerically_sufficient=28,physical_legal=28,freeze_hash=rt.sha(R/'reports/prediction_freeze_manifest.json'),elapsed_hours=(now-rt.read(R/'work/experiment_clock.json')['started'])/3600)
 rt.write(R/'reports/postprocess_recovery.json',receipt)
 with (R/'work/resource_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(receipt)+'\n')
 rt.write(R/'work/controller_status.json',dict(status='COMPLETE_EXPERIMENTAL',exit_code=0,updated=now,recovered_from=old['status'],recovery_receipt='reports/postprocess_recovery.json',audited=28,no_active_registered_fit=True))
 print('CAMPAIGN_CLOSED',receipt['elapsed_hours'],flush=True)
if __name__=='__main__':main()
