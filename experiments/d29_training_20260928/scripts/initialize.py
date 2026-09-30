from pathlib import Path
import datetime,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent/'20260927_1'
for n in ('config','data','evidence','outputs','reports','work','tests','d29_training'): (ROOT/n).mkdir(exist_ok=True)
clock=ROOT/'config/clock.json'
if not clock.exists():
    start=datetime.datetime.fromisoformat('2026-09-27T13:15:11.2317225+08:00')
    clock.write_text(json.dumps(dict(started=start.isoformat(),deadline=(start+datetime.timedelta(hours=96)).isoformat(),dispatch_deadline=(start+datetime.timedelta(hours=80)).isoformat(),budget_hours=96),indent=2),encoding='utf-8')
manifest=[]
for folder in ('d29_platform','vendor','tests'):
    for p in (PARENT/folder).rglob('*'):
        if not p.is_file() or any(x in p.relative_to(PARENT).parts for x in ('work','cache','__pycache__','daily_inputs')):continue
        q=ROOT/p.relative_to(PARENT);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():continue
        shutil.copyfile(p,q);manifest.append(dict(path=p.relative_to(PARENT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'evidence/parent_snapshot.json').write_text(json.dumps(dict(parent=str(PARENT),files=manifest),indent=2),encoding='utf-8')
for f in ('conditional_reference.json','mineralization_reference.json'):shutil.copyfile(PARENT/'config'/f,ROOT/'config'/f)
jobs=[]
for model in ('U','LAND1'):
    for fold,end in [('F23',2022),('F24',2023)]:
        for strategy in ('T0','T1','T2'):
            for entry in ([0,1] if strategy!='T1' else [0]):
                priority=3 if entry else (2 if strategy=='T1' else 0)
                jobs.append(dict(id=f'{model}_{fold}_{strategy}_s{entry}',model=model,fold=fold,strategy=strategy,entry=entry,priority=priority,train_end=end,evaluate=[end+1],space_block=None))
    for block in (56,113,191):
        for strategy in ('T0','T2'):
            jobs.append(dict(id=f'{model}_S{block}_{strategy}_s0',model=model,fold='F24',strategy=strategy,entry=0,priority=1,train_end=2023,evaluate=[2023,2024],space_block=block))
assert len(jobs)==32
(ROOT/'config/jobs.json').write_text(json.dumps(sorted(jobs,key=lambda x:(x['priority'],x['id'])),indent=2),encoding='utf-8')
study=dict(name='D29 long history and all-domain training',source_monthly_identity='User-confirmed automatic monitoring; arithmetic mean of valid readings',formal_years=[2016,2024],physics_start=1961,prior_strength='unchanged within each model',strategies={'T0':{'start':2021,'monthly':.8,'HF_anomaly':.2},'T1':{'start':2016,'monthly':.8,'HF_anomaly':.2},'T2':{'start':2016,'long_monthly':.4,'other_monthly':.4,'HF_anomaly':.2}},long_group={'years':[2016,2020],'minimum_months_each_year':6},parameter_counts={'U':31,'LAND1':23},NH4_DO='diagnostic_only_no_loss_no_predictor',official_monthly_wins=True,read_count_complete_support_only=True,source_calibration='bounded research correction, not identified emissions',scope='conditional research calibration; not independent complete-source certification',publication=False,schedules=False)
(ROOT/'config/study.json').write_text(json.dumps(study,ensure_ascii=False,indent=2),encoding='utf-8')
print('initialized',len(manifest),'snapshot files',len(jobs),'jobs')
