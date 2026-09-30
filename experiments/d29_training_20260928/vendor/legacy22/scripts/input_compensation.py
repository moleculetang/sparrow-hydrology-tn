"""Four-corner diagnostic; no refitting and no unique allocation of interaction."""
import argparse,gc
import native_runtime as rt
from campaign_model import *
def main(tag):
    selected=rt.read(RUN/'data/selected.json');rec=rt.read(RUN/'outputs'/tag/'model.json');job=rec['job'];structure='L3' if job['kind']=='REGIONAL_L3' else 'U';short=tag[:3]
    parent_tag=selected[f'{short}_{structure}_P'];parent=rt.read(RUN/'outputs'/parent_tag/'model.json');out=RUN/'outputs'/tag/'counterfactuals';out.mkdir(exist_ok=True)
    rt.label_barrier(job['fold']);rows=[]
    for mode in ['input_and_c_only','process_only']:
        jj=job if mode=='input_and_c_only' else parent['job'];x=np.array(parent['parameters'] if mode=='input_and_c_only' else rec['parameters']);x[30]=rec['parameters'][30] if mode=='input_and_c_only' else parent['parameters'][30]
        m=for_job(jj);m.design=dict(m.design,observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(RUN/'data/prediction_registry.json'));m.registry=rt.read(RUN/'data/prediction_registry.json')['records'];m._daily_meta_cache={}
        meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');meta=meta[meta.year.le(rt.read(RUN/'configs/folds.json')[job['fold']]['end_year'])]
        with torch.no_grad():a=m.daily_boundary(torch.tensor(x),meta)
        i=a['record'].numpy();di=a['day_index'].numpy();mass=a['mass'].numpy();water=a['water'].numpy()
        pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[i],date=m.data.dates[di],mass_kg_day=mass,water_m3_day=water,concentration_mg_l=1000*mass/water)).to_parquet(out/(mode+'.parquet'),index=False)
        ledger=m.ledger(x);scale=max(1.,float((ledger['fast']+ledger['slow']).sum()))
        assert ledger['local_balance_max_kg']<=1e-6 and abs(ledger['network_balance_kg'])<=scale*1e-10
        rows.append(dict(mode=mode,parameters=x.tolist(),input_mode=jj['input_mode'],source_c=float(np.exp(x[30])),uptake_kg=float(ledger['uptake'].sum()),loss_kg=float(ledger['mineral_loss'].sum()),end_land_kg=float((ledger['M'][-1]+ledger['L'][-1]).sum()),terminal_kg=float(ledger['terminal'].sum()),local_balance_kg=float(ledger['local_balance_max_kg'])))
        del m,a,ledger;gc.collect()
    pd.DataFrame(dict(parameter=rec['names'],P=parent['parameters'],candidate=rec['parameters'],change=np.array(rec['parameters'])-parent['parameters'])).to_csv(out/'parameter_changes.csv',index=False)
    rt.write(out/'audit.json',dict(status='PASS',baseline=parent_tag,joint=tag,rows=rows,interpretation='Input expression plus c vs process parameters (including regional response). Nonlinear interaction is not uniquely apportioned.'))
    print('PASS INPUT COMPENSATION',tag,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');main(p.parse_args().tag)
