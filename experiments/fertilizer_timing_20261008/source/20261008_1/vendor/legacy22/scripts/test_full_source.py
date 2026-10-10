"""Full 230-reach, four-source acceptance at a fixed non-unit coefficient fixture."""
import native_runtime as rt
from campaign_model import *
from source_corrected import embed
from routing import route
from balanced_tags import balanced_scan
def main():
    job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']=='F24_S_s0');rt.label_barrier(job['fold']);m=for_job(job);d=m.data
    x=embed(rt.read(RUN/'outputs/F24_R_s0/model.json')['parameters'],'D29_BE','SOURCE_SEPARATE');x[30:]=np.log([.7,1.1,1.3,.9]);base=m.ledger(x)
    inp=np.zeros((*d.contact.shape,4));inp[d.starts]=d.source_tags*np.exp(x[30:])[None,None,:]
    with torch.no_grad():h,s,f,k=[v.numpy() for v in m.flux_parameters(torch.tensor(x))]
    ti=np.zeros_like(h);ti[d.starts]=m.corrected_source(torch.tensor(x)).numpy()
    v=balanced_scan(h,s,f,d.lower_release,inp,ti,m.demand);errors={key:float(np.max(abs(v[i].sum(-1)-base[key]))) for key,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]}
    assert v[8]<=1e-6 and v[9]>=-1e-7,(v[8],v[9])
    channel=np.zeros_like(base['channel_loss']);terminal=np.zeros_like(base['terminal']);stocks=np.zeros_like(base['reservoir_stocks'])
    for i in range(4):
        r=route(d,v[0][:,:,i]+v[1][:,:,i],vf=float(x[2]));channel+=r['channel_removed'];terminal+=r['terminal'];stocks+=r['stocks']
    errors.update(channel=float(np.max(abs(channel-base['channel_loss']))),terminal=float(np.max(abs(terminal-base['terminal']))),reservoir=float(np.max(abs(stocks-base['reservoir_stocks']))))
    assert max(errors.values())<=1e-6,errors
    rt.write(RUN/'reports/full_source_fixture.json',dict(status='PASS',errors=errors,maximum_rounding_adjustment=v[7],per_source_balance=v[8],minimum=v[9],reaches=230,sources=4,days=len(d.dates),scientific_configuration=False,process=rt.process(os.getpid())))
    print('PASS full source fixture',errors,flush=True)
if __name__=='__main__':main()
