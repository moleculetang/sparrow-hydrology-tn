"""Unified same-support metrics after predictions freeze; labels never select jobs."""
import re,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from mltn.common import ROOT,write,read,sha
from mltn.metrics import station_table,paired_summary,basic,nse_coverage
from mltn.bootstrap import paired
def event_table(frame,column='prediction',events=None,reference=None):
    events=pd.read_parquet(ROOT/'data/frozen_evaluation_events.parquet') if events is None else events
    reference=pd.read_parquet(ROOT/'data/hf_daily_accepted.parquet') if reference is None else reference
    reference=reference.copy();reference['date']=pd.to_datetime(reference.date);rows=[]
    for e in events[events.eligible].itertuples():
        z=frame[frame.station_key.eq(e.station_key)];g=z[z.date.between(e.start,e.end)].sort_values('date');bg=z[z.date.ge(e.background_start)&z.date.lt(e.start)]
        if len(g)<2 or len(bg)<2:continue
        y=g.observed.to_numpy(float);p=g[column].to_numpy(float);iy=int(np.argmax(y));ip=int(np.argmax(p));by=float(bg.observed.mean());bp=float(bg[column].mean())
        original=reference[reference.station_key.eq(e.station_key)&reference.date.between(e.start,e.end)].sort_values('date')
        peak_dates=original.loc[original.tn_mg_l.eq(original.tn_mg_l.max()),'date']
        canonical=peak_dates.iloc[0] if len(peak_dates) else pd.NaT
        rows.append(dict(station_key=e.station_key,event_rank=e.event_rank,start=e.start,end=e.end,common_event_days=len(g),common_background_days=len(bg),frozen_canonical_peak_date=canonical,frozen_canonical_peak_present=bool(canonical in set(g.date)),frozen_observed_peak_tie_count=len(peak_dates),prediction_peak_tie_count=int(np.sum(p==p[ip])),observed_peak=float(y[iy]),prediction_peak=float(p[ip]),observed_background=by,prediction_background=bp,observed_amplitude=float(y[iy]-by),prediction_amplitude=float(p[ip]-bp),peak_day_shift=int((g.date.iloc[ip]-g.date.iloc[iy])/pd.Timedelta(days=1)),peak_value_error=float(p[ip]-y[iy]),**basic(y,p)))
    return pd.DataFrame(rows)

def paired_event_table(frame,events=None,reference=None):
    a=event_table(frame,'baseline',events,reference);b=event_table(frame,'candidate',events,reference)
    if a.empty or b.empty:return pd.DataFrame()
    keys=['station_key','event_rank','start','end']
    pair=a.merge(b,on=keys,suffixes=('_baseline','_candidate'),validate='one_to_one')
    pair['amplitude_absolute_error_change']=np.abs(pair.prediction_amplitude_candidate-pair.observed_amplitude_candidate)-np.abs(pair.prediction_amplitude_baseline-pair.observed_amplitude_baseline)
    pair['background_absolute_error_change']=np.abs(pair.prediction_background_candidate-pair.observed_background_candidate)-np.abs(pair.prediction_background_baseline-pair.observed_background_baseline)
    pair['peak_value_absolute_error_change']=np.abs(pair.peak_value_error_candidate)-np.abs(pair.peak_value_error_baseline)
    pair['peak_day_absolute_error_change']=np.abs(pair.peak_day_shift_candidate)-np.abs(pair.peak_day_shift_baseline)
    pair['unique_peak_phase_supported']=pair.frozen_canonical_peak_present_candidate & pair.frozen_observed_peak_tie_count_candidate.eq(1) & pair.prediction_peak_tie_count_candidate.eq(1) & pair.prediction_peak_tie_count_baseline.eq(1)
    pair['NSE_change']=pair.NSE_candidate-pair.NSE_baseline
    return pair
def load_predictions():
    groups={};manifest={}
    roles=ROOT/'outputs/result_roles.json';include=set(read(roles)['include']) if roles.exists() else None
    for folder in sorted((ROOT/'jobs').glob('*')):
        if not (folder/'result.json').exists() or 'screen_' in folder.name:continue
        if include is not None and folder.name not in include:continue
        files=[('monthly','prediction_monthly.parquet'),('daily','prediction_daily_hf.parquet')] if folder.name.startswith('joint_') else []
        if not files:
            task='monthly' if '_monthly_' in folder.name else 'daily';files=[(task,'prediction.parquet')]
            if folder.name.startswith('aux_'):files.append((task,'closed_loop_prediction.parquet'))
        match=re.search(r'(F23|F24|S23|S24)',folder.name)
        if match is None:continue
        stage=match[1];block=re.search(r'_B(56|113|191)',folder.name);context=stage+('' if block is None else '_B'+block[1])
        for task,file in files:
            path=folder/file
            if not path.exists():continue
            q=pd.read_parquet(path);q['date']=pd.to_datetime(q.date);q=q.rename(columns={'tn_mg_l':'observed'});q=q[np.isfinite(q.prediction)&np.isfinite(q.observed)]
            name=folder.name+('_closed_loop' if file=='closed_loop_prediction.parquet' else '');groups.setdefault((context,task),{})[name]=q;manifest[str(path.relative_to(ROOT))]=sha(path)
    derived=ROOT/'outputs/monthly_readouts/manifest.json'
    if derived.exists():
        for item in read(derived)['records']:
            path=ROOT/item['file'];assert sha(path)==item['sha256'],'DERIVED_READOUT_HASH'
            q=pd.read_parquet(path).rename(columns={'tn_mg_l':'observed'});q['date']=pd.to_datetime(q.date)
            groups.setdefault((item['context'],item['task']),{})[item['configuration']]=q
            manifest[item['file']]=item['sha256']
            if 'daily_prediction_file' in item['identity']:
                meta=item['identity'];assert sha(ROOT/meta['daily_prediction_file'])==meta['daily_prediction_sha256']
                manifest[meta['daily_prediction_file']]=meta['daily_prediction_sha256']
    write(ROOT/'outputs/prediction_freeze.json',manifest);return groups
def main(bootstrap=True):
    out=ROOT/'outputs/evaluation';out.mkdir(parents=True,exist_ok=True);groups=load_predictions();allrows=[];pairs=[]
    blocks=read(ROOT/'data/spatial_blocks.json')
    for (context,task),preds in list(groups.items()):
        if not context.startswith(('S23_B','S24_B')):continue
        block=context.split('_B')[1];held=set(blocks[block]['held_stations'])
        for name,q in groups.get(('F24',task),{}).items():
            if 'fixed_recipe_' not in name:continue
            if context.startswith('S24'):
                ref=q[q.station_key.isin(held)].copy();readout='same_recipe_global_forecast'
            else:
                folder=ROOT/'jobs'/name
                filename=('training_prediction_monthly.parquet' if task=='monthly' else 'training_prediction_daily_hf.parquet') if name.startswith('joint_') else 'training_prediction.parquet'
                if not (folder/filename).exists():continue
                ref=pd.read_parquet(folder/filename).rename(columns={'tn_mg_l':'observed'});manifest=read(ROOT/'outputs/prediction_freeze.json');manifest[str((folder/filename).relative_to(ROOT))]=sha(folder/filename);write(ROOT/'outputs/prediction_freeze.json',manifest);ref['date']=pd.to_datetime(ref.date);ref=ref[ref.date.dt.year.eq(2023)&ref.station_key.isin(held)];readout='same_recipe_global_training_reconstruction'
            preds[name+'_'+readout+'_B'+block]=ref
    for (context,task),preds in groups.items():
        # Mean of registered three seed models only if all three complete on exact support.
        seedgroups={}
        for name,q in list(preds.items()):
            key=re.sub(r'_s(?:1729|1730|1731)', '_seedmean',name)
            if key!=name:seedgroups.setdefault(key,[]).append((name,q))
        for name,parts in seedgroups.items():
            if len(parts)!=3:continue
            keys=['station_key','date'];join=parts[0][1][keys+['observed','prediction']].rename(columns={'prediction':'p0'})
            for i,(_,g) in enumerate(parts[1:],1):join=join.merge(g[keys+['prediction']].rename(columns={'prediction':'p'+str(i)}),on=keys,validate='one_to_one')
            join['prediction']=join[['p0','p1','p2']].mean(1);base=parts[0][1].drop(columns='prediction').merge(join[keys+['prediction']],on=keys,validate='one_to_one');preds[name]=base
        for name,q in preds.items():
            if q.empty:continue
            daily=task.startswith('daily');table=station_table(q,predictions=('prediction',),daily=daily,minimum_coverage=True);table['configuration']=name;table['context']=context;table['task']=task;allrows.append(table)
            q.to_parquet(out/f'{name}_{task}_frozen.parquet',index=False)
            if daily:event_table(q).assign(configuration=name,context=context,task=task).to_csv(out/f'{name}_{task}_events.csv',index=False,encoding='utf-8-sig')
        # Comparators are chosen by registered identities, not evaluation performance.
        base_name=next((name for name in preds if '_ridge' in name),None)
        comparisons=[]
        if base_name is not None:comparisons.extend((base_name,n) for n in preds if n!=base_name)
        # Region exclusion comparison holds config/seed fixed. These full-domain
        # references are descriptive, and S23 uses training reconstruction explicitly.
        for ref in preds:
            if 'same_recipe_global_' not in ref:continue
            core=ref.split('_same_recipe_global_',1)[0].replace('fixed_recipe_','').replace('F24_',context.split('_B')[0]+'_')
            candidate=core+'_B'+context.split('_B')[1]
            if candidate in preds:comparisons.append((ref,candidate))
        selection=read(ROOT/'outputs/frozen_selection.json') if (ROOT/'outputs/frozen_selection.json').exists() else None
        if selection and task in ['daily','monthly']:
            family=selection['winners'][task];config=selection['selected'][family+'_'+task] if not context.startswith('S') else 0
            stage=context.split('_B')[0];suffix='' if '_B' not in context else '_B'+context.split('_B')[1]
            mean=f'{stage}_{task}_{family}_c{config}_seedmean{suffix}';single=f'{stage}_{task}_{family}_c{config}_s1729{suffix}'
            direct=mean if mean in preds else single if single in preds else None
            if direct is not None:comparisons.extend((direct,n) for n in preds if n!=direct and ('joint_' in n or 'aux_' in n or 'reference_' in n or 'convex_ensemble' in n or 'dailyagg_' in n))
        # Forecast feedback must also beat the simplest same-visibility comparator.
        for name in preds:
            if not name.startswith('aux_') or 'persistence' in name:continue
            lead=re.search(r'_lead(\d+)',name)
            if lead is None:continue
            persistence=f'aux_{context}_{task}_persistence_lead{lead[1]}'+('_closed_loop' if name.endswith('_closed_loop') else '')
            if persistence in preds:comparisons.append((persistence,name))
        for base_name,name in dict.fromkeys(comparisons):
            base=preds[base_name];q=preds[name]
            keys=['station_key','date'];frame=q.merge(base[keys+['prediction']].rename(columns={'prediction':'baseline'}),on=keys,validate='one_to_one');frame['candidate']=frame.prediction
            if frame.empty:continue
            table=station_table(frame,predictions=('baseline','candidate'),daily=task.startswith('daily'),minimum_coverage=True);summary=paired_summary(table);summary.update(context=context,task=task,baseline=base_name,candidate=name)
            if task.startswith('daily'):summary['centered']=paired_summary(table,'month_centered_NSE')
            pairs.append(summary);tag=f'{name}__versus__{base_name}_{task}';table.to_csv(out/f'{tag}_paired.csv',index=False,encoding='utf-8-sig')
            if task.startswith('daily'):paired_event_table(frame).to_csv(out/f'{tag}_paired_events.csv',index=False,encoding='utf-8-sig')
            # Every reported registered comparison, including reserved ensembles
            # and direct-family controls, receives the same uncertainty protocol.
            if bootstrap:
                valid=set(table[table.nse_eligible].station_key);frame=frame[frame.station_key.isin(valid)]
                if len(frame):
                    for block in [1,2]:paired(frame,1000,block,1729).to_parquet(out/f'{tag}_bootstrap_{block}month.parquet',index=False)
    if allrows:pd.concat(allrows,ignore_index=True).to_csv(out/'all_station_metrics.csv',index=False,encoding='utf-8-sig')
    write(out/'paired_summary.json',pairs);write(out/'evaluation_receipt.json',dict(contexts=len(groups),configurations=sum(len(v) for v in groups.values()),station_metric_rows=sum(len(x) for x in allrows),pair_comparisons=len(pairs),bootstrap_replicates=1000 if bootstrap else 0,seed=1729,centered='read-count weighted month centering, month-equal SSE and variance',undefined='zero variance/registered low coverage; no epsilon'))
if __name__=='__main__':main()
