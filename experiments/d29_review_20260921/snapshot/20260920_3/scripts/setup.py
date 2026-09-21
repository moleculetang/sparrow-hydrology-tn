"""Freeze a new, finite experiment. Prior experiments are read-only."""
import json, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone
R=Path(__file__).resolve().parents[1]
P=R.parent
for name in ('data','reports','outputs','work/numba','logs','vendor'):(R/name).mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
arms=[dict(id=f'{h}_{a}',hydro=h,kind=k,gamma=g) for h,items in [('H1',[('D29','D29',None),('Q1','Q1',None),('G0','main',0),('G05','main',.5),('G1','main',1),('G2','main',2),('G4','main',4),('CLIM1','clim',1),('LEVEL0','level',0),('LEVEL1','level',1)]),('H0',[('D29','D29',None),('Q1','Q1',None),('G0','main',0),('G1','main',1)])] for a,k,g in items]
protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),version='WET_MOB_V1',arms=arms,history=[1961,2024],reference=[1961,2020],parameters='20260916_2/outputs/C0_s1/model.json',tau=200.,root_bounds=[1e-8,1e2],root_grid_points=21,root_budget=512,root_relative_tolerance=1e-8,wall_budget_hours=12,report_reserve_hours=2,seed=1729,bootstrap=1000,primary='H1_G1',baseline='H1_G0',physical_tolerances=dict(local_kg=1e-6,tag_kg=1e-6,network_relative=1e-10,negative_kg=1e-7),shape=dict(relative_error_reduction=.25,improved_station_fraction=.6,min_stations=5,min_events=20),events=dict(water_quantile=.9,background_days=7,min_background_observed_days=4,min_event_observed_days=1,pair_gap_days=30),coverage=dict(min_readings=4,min_span_hours=12,min_month_days=10),new_fits=0)
p=R/'data/protocol.json'
if p.exists():raise SystemExit('ALREADY_INITIALIZED')
p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
files=[P/'20260920_2/work'/x for x in ('dp_kernel.py','closures_dp2.py','closures_dp.py')]
records=[]
for src in files:
 dst=R/'vendor'/src.name;shutil.copyfile(src,dst)
 records.append(dict(source=str(src),copy=str(dst),sha256=sha(src)))
(R/'data/algorithm_sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(R/'data/protocol.sha256').write_text(sha(p),encoding='ascii')
(R/'README.md').write_text('# 湿润状态控制的氮动员结构检验\n\n状态：准备中。14个正式前向配置，零参数拟合。主水文H1，有限H0桥接。\n\n协议见`data/protocol.json`；历史目录只读。正式结果和限制以最终独立审计为准。\n',encoding='utf-8')
print(json.dumps(dict(root=str(R),protocol_sha256=sha(p),arms=len(arms))))
