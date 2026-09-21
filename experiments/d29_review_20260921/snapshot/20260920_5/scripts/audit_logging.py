from runtime import *
import subprocess
def parsed(name):
 good=[];bad=[]
 for i,line in enumerate((R/'logs'/name).read_text(encoding='utf-8').splitlines(),1):
  try:good.append(json.loads(line))
  except (ValueError,TypeError):bad.append(dict(line=i,text=line))
 return good,bad
def main():
 run=str(time.time_ns());children=[]
 for i in range(8):children.append(subprocess.Popen([sys.executable,'-B',__file__,'child',run,str(i)],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
 assert all(p.wait()==0 for p in children)
 fixture,bad=parsed('concurrent_fixture.jsonl');fixture=[x for x in fixture if x['run']==run];assert not bad and len(fixture)==800 and len({(x['worker'],x['index']) for x in fixture})==800
 attempts,ab=parsed('block_attempts.jsonl');resources,rb=parsed('resources.jsonl');ledger=[];missing=[]
 for root in [R/'outputs',R/'outputs_attempts']:
  for cp in root.glob('*/checkpoint.json'):
   c=read(cp);arm=c['identity']['config']['id'];version=cp.parent.name
   for ch in c['chunks']:
    ledger.append(dict(arm=arm,version=version,start=ch['start'],stop=ch['stop'],checkpoint_path=str(cp.relative_to(R)),sha256=ch['sha256'],completed=True,timestamp_reconstructed=False))
 # Compare multiplicities, including the preserved failed A-FULL attempt.
 from collections import Counter
 expected=Counter((x['arm'],x['start'],x['stop']) for x in ledger);observed=Counter((x['arm'],x['start'],x['stop']) for x in attempts)
 for k,n in (expected-observed).items():missing.append(dict(arm=k[0],start=k[1],stop=k[2],missing_count=n,evidence='completed checkpoint; original attempt timestamp unavailable'))
 assert sum(expected.values())==1152 and sum(x['missing_count'] for x in missing)==3
 pd.DataFrame(ledger).to_csv(R/'reports/completed_block_ledger.csv',index=False)
 put(R/'reports/log_integrity_audit.json',dict(status='COMPLETE_WITH_DISCLOSED_LOG_GAPS',block_valid_rows=len(attempts),checkpoint_completed_blocks=len(ledger),missing_block_rows=missing,malformed_block_rows=ab,malformed_resource_rows=rb,resource_valid_rows=len(resources),original_logs_preserved=True,missing_resource_timestamps_recoverable=False,fix='msvcrt single-byte cross-process lock around append',fix_concurrency_test=dict(processes=8,rows=800,unique_rows=800,pass_test=True)))
 print('LOG_AUDIT_COMPLETE',len(attempts),len(ledger),len(rb))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='child':
  for j in range(100):log('concurrent_fixture.jsonl',dict(run=sys.argv[2],worker=int(sys.argv[3]),index=j))
 else:main()
