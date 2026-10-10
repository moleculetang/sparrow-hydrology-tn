"""Separate recomputation process; not external expert review."""
import argparse,time
from campaign_model import *
import native_runtime as rt
from serial_solvers import projected_gradient

def main(tag):
    start=time.time();job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==tag)
    rt.label_barrier(job['fold']);out=RUN/'outputs'/tag;record=rt.read(out/'model.json');old=rt.read(out/'audit.json')
    for name,h in old['files'].items():assert rt.sha(out/name)==h,('CHANGED_OUTPUT',name)
    m=for_job(job);x=np.array(record['parameters']);value,g=m.value_gradient(x)
    assert abs(value-record['objective'])<=1e-8*(1+abs(value))
    pg=float(np.max(abs(projected_gradient(x,g,m.bounds))));a=m.ledger(x);saved=np.load(out/'daily_physical_ledger.npz')
    errors={k:float(abs(a[k]-saved[k]).max()) for k in ['fast','slow','M','L','uptake','mineral_loss']}
    assert max(errors.values())<=1e-6
    flux=float((a['fast']+a['slow']).sum());physical=a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=flux*1e-10 and max(a['source_label_sum_errors'].values(),default=0)<=1e-6 and min(a[k].min() for k in ['M','L','fast','slow','uptake','mineral_loss'])>=-1e-7 and np.max(a['uptake']-a['demand'])<=1e-7
    nested=None
    if job.get('nested_tags'):
        refs=[rt.read(RUN/'outputs'/p/'model.json')['objective'] for p in job['nested_tags'] if (RUN/'outputs'/p/'audit.json').exists() and rt.read(RUN/'outputs'/p/'audit.json').get('physical_reasonable')]
        if refs:
            nested=min(refs)
            assert value<=nested+1e-8*(1+abs(nested)),'LOST_NESTED_REFERENCE'
    elif job['kind']=='STATE_MODULATED':
        b=rt.read(RUN/'data/selected.json')[tag[:3]+'_R'];base=rt.read(RUN/'outputs'/b/'model.json');nested=float(base['objective'])
        assert value<=nested+1e-8*(1+abs(nested)),'LOST_NESTED_REFERENCE'
    del a,saved
    m.design=dict(m.design,observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(RUN/'data/prediction_registry.json'))
    m.registry=rt.read(RUN/'data/prediction_registry.json')['records'];m._daily_meta_cache={}
    meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');end=rt.read(RUN/'configs/folds.json')[job['fold']]['end_year'];meta=meta[meta.year.le(end)]
    with torch.no_grad():daily=m.daily_boundary(torch.tensor(x),meta)
    stored=pd.read_parquet(out/'daily_station_mass_water.parquet')
    assert np.array_equal(meta.station_key.to_numpy()[daily['record'].numpy()],stored.station_key.to_numpy())
    assert np.array_equal(m.data.dates[daily['day_index'].numpy()].to_numpy(),stored.date.to_numpy())
    station_errors=dict(mass=float(abs(daily['mass'].numpy()-stored.mass_kg_day.to_numpy()).max()),water=float(abs(daily['water'].numpy()-stored.water_m3_day.to_numpy()).max()))
    assert station_errors['mass']<=1e-6 and station_errors['water']==0
    rt.write(RUN/'reports/independent'/f'{tag}.json',dict(status='PASS' if physical else 'PHYSICAL_FAILURE',objective=value,pg=pg,numerical_sufficient=pg<=1e-5,physical=bool(physical),state_errors=errors,station_errors=station_errors,nested_objective=nested,full_history_calls=3,elapsed=time.time()-start))
    assert physical,'INDEPENDENT_PHYSICAL_FAILURE'
    print(tag,'independent',physical,pg,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');main(p.parse_args().tag)
