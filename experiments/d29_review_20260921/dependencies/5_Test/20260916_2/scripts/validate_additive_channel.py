"""Independent source-channel additivity, zero removal and land invariance."""
from campaign_model import *
from validate_structure_edges import physical
import native_runtime as rt
def main():
 d=load_data('FULL24');des=rt.read(RUN/'data/designs/T24_G_D.json');fixture=RUN/'data/validation/synthetic_population.npy';checks=[]
 for calendar in ['monthfirst','uniform_daily']:
  cfg=dict(des,structure=dict(human=True,calendar=calendar,population_file=str(fixture.relative_to(RUN)),population_sha256=sha(fixture)))
  m=make_model(d,None,'D29_BE',cfg);x=m.initial(1);x[2]=0.;a=m.ledger(x);zero=x.copy();zero[-1]=0.;b=m.ledger(zero);physical(a);physical(b)
  for n in ['M','L','fast','slow','uptake','mineral_loss']:assert np.array_equal(a[n],b[n]),n
  assert np.max(abs(a['channel_loss']))==0
  expected=float(np.load(fixture).sum());actual=float(a['human_input'].sum());assert abs(expected-actual)<=1e-8*max(1,expected)
  delivered=float(a['terminal'].sum()-b['terminal'].sum()+a['reservoir_stocks'][-1].sum()-b['reservoir_stocks'][-1].sum());assert abs(delivered-actual)<=1e-10*max(1,actual)
  checks.append(dict(calendar=calendar,land_states_bitwise_equal=True,expected_human_kg=expected,inserted_human_kg=actual,terminal_plus_storage_increment_kg=delivered,network_balance_kg=a['network_balance_kg']))
 rt.write(RUN/'reports/additive_channel_validation.json',dict(status='PASS_IMPLEMENTATION_ONLY',checks=checks,scientific_source_admitted=False,full_history_ledgers=4))
if __name__=='__main__':main()
