"""All-site TN evaluation only after complete, immutable prediction accounting.

Entry selection uses independently audited training totals before reading labels.
No fit, initializer or controller imports this module.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha

def clean(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def key(j):
    return (j['calendar'],j['plant_version'],j['fold'],j['space_block'])

def admitted_hf_rows(frame):
    required={'day_coverage_eligible','n_unique_times','span_hours','read_count'}
    if not required<=set(frame.columns):raise RuntimeError('HF_EVALUATION_COVERAGE_IDENTITY_REQUIRED')
    return frame[frame.day_coverage_eligible & frame.n_unique_times.ge(4) & frame.span_hours.ge(12) & frame.read_count.gt(0)].copy()

def select(jobs,records):
    if {j['id'] for j in jobs}!=set(records):raise RuntimeError('EXACT_PATH_ACCOUNTING_REQUIRED')
    groups={}
    for j in jobs:
        r=records[j['id']]
        if not all(r.get(k) for k in ['objective_passed','physical_passed','numerically_sufficient','freeze_verified']):
            raise RuntimeError('UNFINISHED_OR_UNACCEPTED_PATH '+j['id'])
        groups.setdefault(key(j),[]).append((float(r['objective']),j['entry'],j['id']))
    selected={k:min(v)[2] for k,v in groups.items()}
    for k in selected:
        base=('C0',k[1],k[2],k[3])
        if base not in selected:raise RuntimeError('MISSING_MATCHED_MULTIPLIER_ONE_C0')
    return selected

def pair_definitions(selected):
    pairs=set()
    for (calendar,plant,fold,block),jid in selected.items():
        if calendar!='C0':
            pairs.add((selected[('C0',plant,fold,block)],jid,'complete_timing_configuration_vs_C0'))
        if calendar=='C1':
            pairs.add((selected[('C0',plant,fold,block)],jid,'intermonth_information'))
        if calendar=='C2':
            pairs.add((selected[('C1',plant,fold,block)],jid,'intramonth_information_same_monthly_input_mass'))
        if calendar=='K':
            pairs.add((selected[('C2',plant,fold,block)],jid,'bounded_joint_timing_share_calibration'))
        if plant=='V1':
            pairs.add((selected[(calendar,'V0',fold,block)],jid,'complete_plant_configuration'))
    return sorted(pairs)

def scopes(job,frame,blocks):
    import pandas as pd
    years=pd.to_datetime(frame.date).dt.year
    held=set();excluded=set()
    if job['space_block'] is not None:
        b=blocks[str(job['space_block'])];held=set(b['held_stations']);excluded=held|set(b['buffer_stations'])
    result=[('training_2016_'+str(job['train_end']),frame[years.between(2016,job['train_end'])&~frame.station_key.isin(excluded)],'training_reconstruction',False)]
    for y in range(2016,2025):
        g=frame[years.eq(y)]
        if held:
            result.append((str(y)+'_all',g,'all_sites_descriptive',False))
            if y>=2023:result.append((str(y)+'_held',g[g.station_key.isin(held)],'spatial_holdout' if y==2023 else 'spatiotemporal_holdout',True))
            result.append((str(y)+'_training_support',g[~g.station_key.isin(excluded)],'training_reconstruction' if y<=job['train_end'] else 'time_holdout_on_training_sites',False))
        else:
            result.append((str(y)+'_all',g,'training_reconstruction' if y<=job['train_end'] else 'retrospective_temporal_holdout',y==(2023 if job['fold']=='F23' else 2024)))
    return result

def main():
    import numpy as np,pandas as pd
    from approved_epoch_delivery import admitted_folders
    folder_map=admitted_folders(ROOT)
    def approved_folder(jid):return folder_map[jid]
    jobs=json.loads((ROOT/'config/jobs.json').read_text());byid={j['id']:j for j in jobs};records={};frozen={}
    for j in jobs:
        f=approved_folder(j['id'])
        audit=json.loads((f/'independent_audit.json').read_text());freeze=json.loads((f/'prediction_freeze.json').read_text());status=json.loads((f/'status.json').read_text())
        if not status.get('audit_completed'):raise RuntimeError('PATH_AUDIT_NOT_FINAL '+j['id'])
        if sha(f/'independent_audit.json')!=freeze.get('audit_sha256'):
            raise RuntimeError('AUDIT_CHANGED_AFTER_FREEZE '+j['id'])
        if audit.get('parameter_sha256')!=freeze.get('parameter_sha256'):
            raise RuntimeError('AUDIT_FREEZE_PARAMETER_IDENTITY_MISMATCH '+j['id'])
        if sha(f/'physical_ledger.json')!=freeze.get('physical_ledger_receipt_sha256'):
            raise RuntimeError('PHYSICAL_RECEIPT_CHANGED_AFTER_FREEZE '+j['id'])
        physical_files=freeze.get('physical_artifact_sha256',{})
        if len(physical_files)!=66:raise RuntimeError('INCOMPLETE_FROZEN_SOURCE_PLANT_ARTIFACT_ACCOUNTING '+j['id'])
        for relative,digest in physical_files.items():
            p=(f/relative).resolve()
            if not p.is_relative_to(f.resolve()) or not p.exists() or sha(p)!=digest:
                raise RuntimeError('FROZEN_SOURCE_PLANT_BYTES_CHANGED '+j['id'])
        for filename,k in [('frozen_station_days.parquet','days_sha256'),('frozen_station_months.parquet','months_sha256'),('frozen_parameters.npy','parameter_sha256')]:
            if sha(f/filename)!=freeze[k]:raise RuntimeError('FROZEN_BYTES_CHANGED '+j['id']+'/'+filename)
        if freeze.get('station_count')!=len(json.loads((ROOT/'evidence/expected_station_support.json').read_text(encoding='utf8'))['station_keys']) or not freeze.get('no_evaluation_outcomes_read'):raise RuntimeError('INCOMPLETE_OR_LABEL_CONTAMINATED_FREEZE '+j['id'])
        records[j['id']]={k:audit.get(k) for k in ['objective','objective_passed','physical_passed','numerically_sufficient','projected_gradient']}
        records[j['id']]['freeze_verified']=True;frozen[j['id']]=freeze
    selected=select(jobs,records);pairs=pair_definitions(selected)
    manifest=dict(accounted_paths=sorted(records),records=records,selected=[dict(configuration=list(k),job=v) for k,v in selected.items()],selection_training_only=True,selection_rule='minimum independently audited total training objective within calendar/plant_version/support; no evaluation selection',frozen=frozen,pairs=pairs)
    # Seal the selection before the first evaluation-label read.
    write_json(ROOT/'outputs/campaign_manifest.json',clean(manifest))
    m=pd.read_parquet(ROOT/'data/monthly_tn.parquet');h=admitted_hf_rows(pd.read_parquet(ROOT/'data/hf_daily.parquet'));events=pd.read_parquet(ROOT/'data/frozen_evaluation_events.parquet')
    blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text())
    from d29_training.evaluation import join_pair,pair_tables,evaluate_events
    from d29_training.bootstrap import paired_resamples
    from d29_training.metrics import eligible_nse_stations
    out=ROOT/'outputs/evaluation';out.mkdir(exist_ok=True);summaries=[]
    for baseid,candidateid,comparison in pairs:
        job=byid[candidateid];base=approved_folder(baseid);candidate=approved_folder(candidateid);target=out/(candidateid+'__vs__'+baseid);target.mkdir(exist_ok=True)
        for scope,observed,filename,daily in [('monthly',m,'frozen_station_months.parquet',False),('daily',h,'frozen_station_days.parquet',True)]:
            frame=join_pair(observed,pd.read_parquet(base/filename),pd.read_parquet(candidate/filename),monthly=not daily);frame['date']=pd.to_datetime(frame.date)
            for period,g,role,bootstrap in scopes(job,frame,blocks):
                if g.empty:
                    summaries.append(dict(baseline=baseid,candidate=candidateid,comparison=comparison,scope=scope,period=period,role=role,summary=dict(status='no_common_support',common_stations=0)));continue
                table,summary=pair_tables(g,daily);table.to_csv(target/(scope+'_'+period+'_stations.csv'),index=False)
                summaries.append(dict(baseline=baseid,candidate=candidateid,comparison=comparison,scope=scope,period=period,role=role,summary=summary))
                if bootstrap:
                    # Persist the exact primary support for independent error
                    # decomposition and statistics, without rerunning physics.
                    g.to_parquet(target/(scope+'_'+period+'_common_support.parquet'),index=False)
                    for nblock in [1,2]:
                        for centered in ([False,True] if daily else [False]):
                            samples,receipt=paired_resamples(g,1000,nblock,1729,centered,eligible_nse_stations(g,daily));name=f'{scope}_{period}_block{nblock}_centered{int(centered)}'
                            samples.to_csv(target/(name+'.csv'),index=False);write_json(target/(name+'.json'),clean(receipt))
                if daily and period.endswith(('_all','_held')):
                    evaluate_events(g,events[events.station_key.isin(g.station_key.unique())],int(period[:4])).to_csv(target/('events_'+period+'.csv'),index=False)
        write_json(out/'progress.json',dict(completed_pair=[baseid,candidateid],completed_summary_rows=len(summaries),registered_pairs=len(pairs)))
    write_json(out/'summaries.json',clean(summaries))
    write_json(out/'complete.json',dict(passed=True,accounted_paths=len(records),selected_configurations=len(selected),pairs=len(pairs),NH4_DO=False,seed=1729,bootstrap_replicates=1000,selection_manifest_sha256=sha(ROOT/'outputs/campaign_manifest.json'),evaluation_label_hashes={n:sha(ROOT/'data'/n) for n in ['monthly_tn.parquet','hf_daily.parquet','frozen_evaluation_events.parquet']}))
    return 0

if __name__=='__main__':sys.exit(main())
