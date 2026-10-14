"""Portable frozen H1 replay. Original equation files are hash-preserved."""
from pathlib import Path
import os,sys,json,importlib.abc,importlib.util,argparse,hashlib,io,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
BASE=Path(__file__).resolve().parent
MODULES=json.loads((BASE/'module_map.json').read_text())
class SourceLoader(importlib.abc.Loader):
 def __init__(self,name):self.name=name
 def create_module(self,spec):return None
 def exec_module(self,module):
  path=BASE/MODULES[self.name]
  source=path.read_text(encoding='utf-8-sig')
  # Redirect only the workspace root for portability. Original files remain byte-identical.
  source=source.replace('Path(r"E:\\SPARROW")','Path('+repr(str(BASE/'portable_workspace'))+')')
  module.__file__=str(path)
  exec(compile(source,str(path),'exec'),module.__dict__)
class SourceFinder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in MODULES:return importlib.util.spec_from_loader(fullname,SourceLoader(fullname))
sys.meta_path.insert(0,SourceFinder())
import numpy as np,pandas as pd,torch
import frozen_hydrology_core as core
import run_locked_long_simulation as long
from reservoir_development_core import build_runtimes,SECONDS_PER_DAY
from reservoir_network_router import build_reach_graph,route_network_steps
from reservoir_operator import ReservoirState
from export_final_tn_hydrology_interface import build_reach_products,build_reservoir_products
torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
core.CHECKPOINT=BASE/'data/model.pt';core.SCORE=BASE/'data/regionalized_slow_score.parquet'
core.Q72=BASE/'data/area.parquet';core.STATIC=BASE/'data/static.parquet'
core.SCALING=BASE/'data/dynamic_feature_scaling.json';core.TOPOLOGY=BASE/'data/topology_edges.csv'

def forcing(end):
 frames=[pd.read_parquet(BASE/row['path']) for row in json.loads((BASE/'forcing/manifest.json').read_text())['years'] if row['year']<=pd.Timestamp(end).year]
 frame=pd.concat(frames,ignore_index=True)
 dates=pd.date_range('1961-01-01',end)
 frame=frame[frame.date.between(dates[0],dates[-1])].sort_values(['date','reach_id'])
 assert len(frame)==len(dates)*230
 return dates,frame.precipitation_daily_mm.to_numpy().reshape(-1,230),frame.pet_fao56_mm_day.to_numpy().reshape(-1,230)

def metadata():
 static=pd.read_parquet(BASE/'data/historical_reservoir_static_metadata.parquet')
 fields=['name','control_reaches','outflow_reach','historical_active_start','local_capture_fraction','capacity_m3','capacity_source','annual_inflow_m3','capacity_annual_inflow_ratio','hydrologic_wet_months','regulation_class','dead_fraction_r2','target_fraction_r2','hierarchical_capacity_ratio']
 result={}
 for row in static[['reservoir_entity_id',*fields]].to_dict('records'):
  entity=str(row.pop('reservoir_entity_id'));row['control_reaches']=tuple(int(x) for x in row['control_reaches'])
  row['hydrologic_wet_months']=tuple(int(x) for x in row['hydrologic_wet_months'])
  row['active_start']=pd.Timestamp(row.pop('historical_active_start'));result[entity]=row
 return result,static[['reservoir_entity_id','domain_storage_scale','notes']].copy()

def run(end,precipitation_scale=1.0,recompute_spinup=False):
 dates,p,pet=forcing(end);p=p*precipitation_scale
 context=core.load_frozen_context()
 spin_dates,spin_p,spin_pet=forcing('1970-12-31');spin_p=spin_p*precipitation_scale
 initial=torch.from_numpy(np.load(BASE/'data/historical_initial_land_state_1961.npz')['state_mm'].copy())
 spin_audit=None
 if recompute_spinup:
  initial,spin_audit=core.periodic_spinup_actual_dates(spin_p,spin_pet,spin_dates,context)
  if not spin_audit['converged']:raise RuntimeError('Spin-up did not converge')
 parent,audit,fast,slow=core.simulate_process(dates,p,pet,initial,context,periodic_prefix=spin_p)
 meta,priority=metadata();graph=build_reach_graph(pd.read_csv(BASE/'data/topology_edges.csv'),list(range(1,231)))
 runtimes=build_runtimes(meta,'R2',180.0,0.25,0.0)
 res_init=json.loads((BASE/'data/historical_initial_reservoir_states_1961.json').read_text())
 for runtime in runtimes:
  if runtime.entity_id in res_init:runtime.state=ReservoirState(**{k:float(v) for k,v in res_init[runtime.entity_id].items()})
 routed=route_network_steps(fast*SECONDS_PER_DAY,slow*SECONDS_PER_DAY,graph,runtimes,dates=dates,values_are_volumes_per_step=True,keep_reservoir_diagnostics=True)
 reach,monthly=build_reach_products(parent,routed.routed_fast,routed.routed_slow,routed.routed_direct,routed.routed_age_moment)
 daily,res_monthly,res_static=build_reservoir_products(long.collapse_disabled_shared_diagnostics(routed.reservoir_diagnostics),meta,priority)
 return reach,daily,{'process':audit,'spinup_recomputed':recompute_spinup,'spinup_audit':spin_audit,'reservoir_states_initialization':'original 1961 donor state retained; changing long-term parameters requires separately rederiving reservoir initial states','precipitation_scale':precipitation_scale}

def compare(daily,end):
 expected=pd.read_parquet(BASE/'data/tn_hydrology_reservoir_daily.parquet')
 expected=expected[expected.date<=pd.Timestamp(end)].sort_values(['date','reservoir_entity_id']).reset_index(drop=True)
 actual=daily.sort_values(['date','reservoir_entity_id']).reset_index(drop=True)
 assert actual[['date','reservoir_entity_id']].equals(expected[['date','reservoir_entity_id']])
 errors={}
 for col in ['storage_m3','total_release_m3','release_origin_fast_m3','release_origin_slow_m3','release_age_moment_m3_day','local_capture_fraction']:
  errors[col]=float(np.max(np.abs(actual[col].to_numpy()-expected[col].to_numpy())))
  if errors[col]>0:raise RuntimeError({'field':col,'absolute_difference':errors[col]})
 if not np.array_equal(actual.enabled,expected.enabled):raise RuntimeError('Enabled differs')
 return {'reservoir_fields_exact':errors,'reservoir_rows':len(actual)}

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--end',default='2024-12-31');parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--recompute-spinup',action='store_true')
 parser.add_argument('--perturbation-control',action='store_true')
 args=parser.parse_args();start=time.time()
 reach,daily,audit=run(args.end,recompute_spinup=args.recompute_spinup)
 audit.update(compare(daily,args.end))
 if args.perturbation_control:
  changed_reach,changed_daily,_=run(args.end,1.01)
  delta=float(np.max(np.abs(changed_daily.total_release_m3.to_numpy()-daily.total_release_m3.to_numpy())))
  if delta<=0:raise RuntimeError('Reservoir output did not respond to upstream forcing')
  audit['perturbation_control']={'precipitation_scale':1.01,'maximum_release_change_m3':delta,'fresh_water_routing':True,'original_reservoir_output_used_in_forward':False}
 args.output.mkdir(parents=True,exist_ok=True)
 reach.to_parquet(args.output/'replayed_reach_daily.parquet',index=False,compression='zstd')
 daily.to_parquet(args.output/'replayed_reservoir_daily.parquet',index=False,compression='zstd')
 audit.update(passed=True,date_start='1961-01-01',date_end=args.end,days=int(reach.date.nunique()),reach_count=int(reach.reach_id.nunique()),elapsed_seconds=time.time()-start,source_checkpoint_sha256=hashlib.sha256((BASE/'data/model.pt').read_bytes()).hexdigest(),TN_read=False,observed_discharge_read=False,full_joint_calibration_implemented=False)
 (args.output/'replay_receipt.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
