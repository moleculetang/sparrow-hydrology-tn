"""Independent checks of cached simulator and fabricated-input transformations."""
import gc
import native_runtime as rt
from synthetic_study import FrozenSimulator,ROOT
from synthetic_inputs import version,MODES
from campaign_model import *
def main():
    rows=[]
    for structure in ['U','L3']:
        sim=FrozenSimulator(structure);m=sim.model
        with torch.no_grad():a=m.daily_boundary(torch.tensor(sim.x),sim.meta)
        reference=(1000*a['mass']/a['water']).numpy();cached=sim.forward(np.array(m.raw_daily));error=float(abs(reference-cached).max());assert error<=1e-10,error
        future=np.array(m.raw_daily);future[m.data.dates.year==2024]*=2
        changed=sim.forward(future);past=sim.years<2024;assert np.array_equal(changed[past],cached[past])
        rows.append(dict(structure=structure,cached_daily_prediction_error=error,full_history_future_input_preserves_past=True));del a,m,sim,future;gc.collect()
    from campaign_model import load_data
    d=load_data('FULL24C');truth=np.load(ROOT/'U_1729/fabricated_true_daily_sources.npy',mmap_mode='r');area=pd.read_parquet(RUN/'data/daily_inputs/harvested_area_ha.parquet').to_numpy();checks=[]
    monthly=d.monthly_sum(truth);annual=monthly.reshape(64,12,230,4).sum(1)
    for mode in MODES:
        x=version(d,truth,mode,area);months=d.monthly_sum(x);years=months.reshape(64,12,230,4).sum(1)
        assert x.min()>=-1e-7 and np.isfinite(x).all()
        if mode in ['true','uniform','month_first']:err=float(abs(months-monthly).max());assert err<=1e-6
        elif mode=='shift14':err=float(abs(years-annual).max());assert err<=1e-5
        elif mode=='previous_year':err=float(abs(years[1:]-annual[:-1]).max());assert err<=1e-5
        elif mode=='area':err=float(abs(x.sum(1)-truth.sum(1)).max());assert err<=1e-6
        elif mode=='mass_067':err=float(abs(x-truth*.67).max());assert err==0
        else:err=float(abs(years[1:].sum(1)-annual[:-1].sum(1)).max());assert err<=1e-4
        checks.append(dict(mode=mode,aggregation_roundoff_kg=err));del x,months,years;gc.collect()
    rt.write(RUN/'reports/synthetic_engine_audit.json',dict(status='PASS',cached_simulator=rows,input_transformations=checks,note='Calendar/spatial aggregated sums use floating-point accumulation tolerances, not physical mass-balance tolerance relaxation. Per-model local physical tolerance stays 1e-6 kg.'))
    print('PASS SYNTHETIC ENGINE',flush=True)
if __name__=='__main__':main()
