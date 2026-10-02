"""Repair only exact-constant HF mean rows and all affected evaluation pairs."""
import shutil
import numpy as np
import pandas as pd
from mltn.common import ROOT,read,write,sha
from monthly_readouts import hf_months
from mltn.metrics import station_table,paired_summary
from mltn.bootstrap import paired


def main():
    out=ROOT/'outputs/evaluation';manifest_path=ROOT/'outputs/monthly_readouts/manifest.json'
    manifest=read(manifest_path);freeze=read(ROOT/'outputs/prediction_freeze.json')
    table=pd.read_csv(out/'all_station_metrics.csv');pairs=read(out/'paired_summary.json')
    changes=[];affected=set();backup=ROOT/'superseded/constant_HF_mean'
    def preserve(path):
        dest=backup/path.relative_to(ROOT)
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for item in manifest['records']:
        meta=item['identity']
        if 'parent_prediction_file' not in meta:continue
        parent=ROOT/meta['parent_prediction_file'];assert sha(parent)==meta['parent_prediction_sha256']
        source=ROOT/item['file'];assert sha(source)==item['sha256']
        before=pd.read_parquet(source);expected=hf_months(pd.read_parquet(parent))
        keys=['station_key','date'];assert before[keys].equals(expected[keys])
        changed=np.any(before[['tn_mg_l','prediction']].to_numpy()!=expected[['tn_mg_l','prediction']].to_numpy(),axis=1)
        if not changed.any():continue
        error=float(np.max(np.abs(before[['tn_mg_l','prediction']].to_numpy()-expected[['tn_mg_l','prediction']].to_numpy())))
        # This correction preserves the mathematically constant parent value;
        # it is never a near-constant threshold or an NSE epsilon.
        preserve(source);expected.to_parquet(source,index=False);item['sha256']=sha(source)
        freeze[item['file']]=item['sha256'];name=item['configuration'];affected.add(name)
        q=expected.rename(columns={'tn_mg_l':'observed'});frozen=out/f'{name}_hf_monthly_frozen.parquet';preserve(frozen);q.to_parquet(frozen,index=False)
        t=station_table(q,predictions=('prediction',),daily=False,minimum_coverage=True)
        t['configuration']=name;t['context']=item['context'];t['task']='hf_monthly'
        table=table[~((table.configuration==name)&table.task.eq('hf_monthly'))]
        table=pd.concat([table,t],ignore_index=True)
        changes.append(dict(configuration=name,rows_changed=int(changed.sum()),maximum_absolute_prediction_or_label_change=error,parent_sha256=sha(parent),before_sha256=sha(backup/source.relative_to(ROOT)),after_sha256=sha(source)))
    # Propagate corrected parent means to the registered three-seed prediction
    # means, including their shared observed mean support. No model is rerun.
    for name in sorted(set(table.loc[table.task.eq('hf_monthly'),'configuration'])):
        if '_seedmean' not in name:continue
        parents=[name.replace('_seedmean','_s'+str(seed)) for seed in [1729,1730,1731]]
        if not (set(parents)&affected):continue
        parts=[pd.read_parquet(out/f'{p}_hf_monthly_frozen.parquet') for p in parents]
        keys=['station_key','date'];q=parts[0].copy()
        for part in parts:
            assert part[keys].equals(q[keys]);np.testing.assert_array_equal(part.observed,q.observed)
        q['prediction']=np.mean(np.stack([part.prediction.to_numpy(float) for part in parts]),axis=0)
        frozen=out/f'{name}_hf_monthly_frozen.parquet';preserve(frozen);q.to_parquet(frozen,index=False)
        old=table[(table.configuration==name)&table.task.eq('hf_monthly')]
        assert old.context.nunique()==1
        t=station_table(q,predictions=('prediction',),daily=False,minimum_coverage=True)
        t['configuration']=name;t['context']=old.context.iloc[0];t['task']='hf_monthly'
        table=table[~((table.configuration==name)&table.task.eq('hf_monthly'))]
        table=pd.concat([table,t],ignore_index=True);affected.add(name)
        changes.append(dict(configuration=name,kind='propagated three-seed prediction mean',parents=parents,after_sha256=sha(frozen)))
    if affected:
        preserve(out/'all_station_metrics.csv');preserve(out/'paired_summary.json')
        for pair in pairs:
            if pair['task']!='hf_monthly' or not ({pair['baseline'],pair['candidate']}&affected):continue
            a=pd.read_parquet(out/f"{pair['baseline']}_hf_monthly_frozen.parquet")
            b=pd.read_parquet(out/f"{pair['candidate']}_hf_monthly_frozen.parquet")
            f=b.merge(a[['station_key','date','prediction']].rename(columns={'prediction':'baseline'}),on=['station_key','date'],validate='one_to_one');f['candidate']=f.prediction
            t=station_table(f,predictions=('baseline','candidate'),daily=False,minimum_coverage=True)
            identity={k:pair[k] for k in ['context','task','baseline','candidate']};pair.clear();pair.update(paired_summary(t));pair.update(identity)
            tag=f"{pair['candidate']}__versus__{pair['baseline']}_hf_monthly";csv=out/(tag+'_paired.csv');preserve(csv);t.to_csv(csv,index=False,encoding='utf-8-sig')
            valid=set(t[t.nse_eligible].station_key);f=f[f.station_key.isin(valid)]
            if len(f):
                for block in [1,2]:
                    path=out/f'{tag}_bootstrap_{block}month.parquet';preserve(path);paired(f,1000,block,1729).to_parquet(path,index=False)
        table.to_csv(out/'all_station_metrics.csv',index=False,encoding='utf-8-sig');write(out/'paired_summary.json',pairs)
        write(manifest_path,manifest);write(ROOT/'outputs/prediction_freeze.json',freeze)
    receipt_path=ROOT/'evidence/constant_HF_mean_repair.json'
    if changes or not receipt_path.exists():
        write(receipt_path,dict(passed=True,readouts=changes,affected_readouts=len(affected),fit_calls=0,scientific_training_changed=False,official_monthly_or_raw_daily_changed=False,rule='exact constant parent only: np.ptp==0, preserve its literal value; no epsilon or tolerance relaxation',further_gate='all actual scalar metrics must pass after this repair'))
    print('CONSTANT_HF_REPAIR',len(affected),flush=True)


if __name__=='__main__':main()
