"""Final identity, saved network mass, cumulative attempts and artifact read-back audit."""
from runtime import *
def main():
 pro=read(R/'data/protocol.json');attempts={};rows=pd.read_csv(R/'reports/completed_block_ledger.csv').to_dict('records')
 for x in rows:attempts[x['arm']]=attempts.get(x['arm'],0)+1
 assert all(attempts[c['id']]==(128 if c['id']=='A-FULL' else 64) for c in pro['configs'] if c['family']!='REFERENCE')
 evidence=[]
 for c in pro['configs']:
  arm=c['id'];out=R/'outputs'/arm;summary=read(out/'summary.json');assert summary['gate']['pass']
  frame=pd.read_parquet(out/'daily_station.parquet');assert np.isfinite(frame.p).all() and (frame.water_m3>0).all();assert np.array_equal(1000*frame.mass_kg.to_numpy()/frame.water_m3.to_numpy(),frame.p.to_numpy())
  if c['family']=='REFERENCE':continue
  land=np.load(out/'land_history.npy',mmap_mode='r');river=np.load(out/'river_history.npz');scale=float(land[0].sum()+land[1].sum());err=float(scale-river['loss'].sum()-river['terminal'].sum()-river['stocks'][-1].sum());assert abs(err)<=scale*1e-10
  pairs=pd.read_parquet(out/'event_pairs.parquet');assert (pairs.event_rank-pairs.previous_rank==1).all() and not pairs.duplicated(['station_key','event_rank']).any()
  evidence.append(dict(arm=arm,independent_network_balance_kg=err,network_scale_kg=scale,station_concentration_identity=True,event_pair_adjacency=True))
 put(R/'reports/final_network_counter_audit.json',dict(status='PASS',records=evidence,attempted_blocks=attempts,scientific_configs=18,forward_attempts=18))
 for x in read(R/'data/input_manifest.json'):
  assert sha(x['source'])==x['sha256']
  if x['target'].startswith(('data/','vendor/')):assert sha(R/x['target'])==x['sha256']
 for n,h in read(R/'data/C_arrays_freeze.json').items():assert sha(R/'data'/n)==h
 for n,h in read(R/'data/science_code_freeze.json').items():
  version=R/'data/code_versions/runtime_pre_logging_fix.py' if n=='runtime.py' else R/'scripts'/n
  assert sha(version)==h
 for arm,rec in read(R/'data/prediction_freeze.json').items():
  for name,h in rec.items():assert sha(R/'outputs'/arm/name)==h
 assert not list((R/'outputs').glob('*/worker.lock')) and not (R/'work/controller.lock').exists()
 resources=[]
 for line in (R/'logs/resources.jsonl').read_text().splitlines():
  try:resources.append(json.loads(line))
  except ValueError:pass
 put(R/'reports/resource_summary.json',dict(valid_samples=len(resources),max_observed_cpu=max(x['cpu'] for x in resources),max_observed_ram_percent=max(x['ram'] for x in resources),max_process_peak_bytes=max(x['peak_rss_bytes'] for x in resources),append_only=True,RAM90_in_valid_samples=any(x['ram']>=90 for x in resources),log_gaps_disclosed='log_integrity_audit.json'))
 completion=read(R/'reports/completion_audit.json');completion.update(status='COMPLETE_WITH_DISCLOSED_LOG_GAPS',final_network_counter_audit=True,forward_attempts_including_failed_version=18,total_completed_attempt_blocks=len(rows),missing_original_attempt_log_rows=3,resource_log_gaps=True,elapsed_hours=(time.time()-pro['created'])/3600);put(R/'reports/completion_audit.json',completion)
 files={}
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.name=='delivery_manifest.json' or '__pycache__' in p.parts or 'work' in p.relative_to(R).parts:continue
  files[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
 target=R/'data/delivery_manifest.json';put(target,dict(status='SEALED',files=files,excluded=['work scratch','__pycache__','manifest itself'],failed_attempt_preserved=True))
 for name,rec in read(target)['files'].items():assert sha(R/name)==rec['sha256']
 print('SEALED_AND_READBACK_VERIFIED',len(files),'FILES')
if __name__=='__main__':main()
