"""Full-history nonnegative and plant-budget scan, isolated candidate only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
s=(ROOT/'scripts/diagnose_tagged_full_history.py').read_text(encoding='utf-8')
s=s.replace('d29_platform.land1','d29_platform.precision_candidate').replace('d29_platform.tagged_candidate','d29_platform.mixture_tags_candidate')
s=s.replace('d29_platform/land1.py','d29_platform/precision_candidate.py').replace('d29_platform/tagged_candidate.py','d29_platform/mixture_tags_candidate.py')
s=s.replace('outputs/tag_precision_candidate','outputs/precision_safety_history')
s=s.replace("paths=[Path(__file__),", "paths=[Path(__file__),ROOT/'d29_platform/precision_transfer.py',")
s=s.replace('existing full-history kernel validated; annual streaming here is forward only, no truncated-gradient training','forward/source/nonnegative audit only; full-history derivative acceptance is a separate required gate')
needle="POTENTIAL='--potential-activity' in sys.argv"
instrument='''
from d29_platform.precision_transfer import transfer_expansion
native_run=run_land1
safety=[]
def run_land1(**kwargs):
    b=native_run(**kwargs);i=b._inputs;nt,nr,nl=i['shape']
    represented=b.states.copy()
    represented[:,:,:,1:]-=i['compensation_history'].reshape(nt+1,nr,nl,4)
    minimum=float(represented.min())
    before=b.states[:-1].reshape(nt,nr*nl,5).copy()
    for t in i['transition_days']:
        low=np.zeros_like(before[t]);low[:,1:]=-i['compensation_history'][t]
        moved,_,_=transfer_expansion(before[t].reshape(nr,nl,5),low.reshape(nr,nl,5),i['matrices'][i['event'][t]])
        before[t]=moved.reshape(nr*nl,5)
    X=before[:,:,3]+i['sources'][:,:,3]+i['probabilities'][0]*before[:,:,1]+i['probabilities'][1]*before[:,:,2]
    need=np.maximum(0,i['target']+i['outflows'].sum(-1)-before[:,:,0]-i['sources'][:,:,0])
    uptake=np.minimum(X,need)
    realized=i['realized_outflows'].sum(-1)
    excess=float(np.maximum(realized-before[:,:,0]-i['sources'][:,:,0]-uptake,0).max())
    safety.append({'year':1961+len(safety),'minimum_represented_stock_kg':minimum,'minimum_flux_kg':float(b.fluxes.min()),'minimum_unmet_kg':float(b.unmet_plant_outflows.min()),'plant_output_excess_kg':excess,'passed':bool(minimum>=-1e-7 and b.fluxes.min()>=-1e-7 and b.unmet_plant_outflows.min()>=-1e-7 and excess<=1e-7)})
    write_json(ROOT/'outputs/precision_safety_history/safety_progress.json',safety)
    return b
'''
s=s.replace(needle,instrument+'\n'+needle)
s=s.replace("write_json(OUT/'receipt.json',receipt);", "receipt['safety_rows']=safety;receipt['nonnegative_and_plant_budget_pass']=all(r['passed'] for r in safety);receipt['strict_mass_pass']=receipt['strict_mass_pass'] and receipt['nonnegative_and_plant_budget_pass'];write_json(OUT/'receipt.json',receipt);")
exec(compile(s,str(ROOT/'scripts/verify_precision_safety_history.py'),'exec'))
