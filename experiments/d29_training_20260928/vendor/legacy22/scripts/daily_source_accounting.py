"""All-reach daily source tags and independent routed source sums."""
import argparse
import native_runtime as rt
from campaign_model import *
from balanced_tags import balanced_scan
from routing import route
def main(tag):
    out=RUN/'outputs'/tag;rec=rt.read(out/'model.json');m=for_job(rec['job']);d=m.data;x=np.array(rec['parameters']);base=m.ledger(x)
    inp=np.array(m.raw_daily_tags)*np.exp(x[30]);total=m.corrected_source(torch.tensor(x)).detach().numpy()
    with torch.no_grad():h,s,f,k=[v.numpy() for v in m.flux_parameters(torch.tensor(x))]
    # Chunk by reaches to bound peak memory. No temporal truncation.
    frames=[];network=[];errors={};balance=0.;minimum=0.;adjust=0.;nd,nr=h.shape;ends=d.stops-1
    fast=np.zeros_like(inp);slow=np.zeros_like(inp)
    for start in range(0,nr,16):
        stop=min(start+16,nr);rr=slice(start,stop)
        v=balanced_scan(h[:,rr],s[rr],f[:,rr],d.lower_release[:,rr],inp[:,rr],total[:,rr],m.demand[:,rr]);fast[:,rr]=v[0];slow[:,rr]=v[1]
        balance=max(balance,v[8]);minimum=min(minimum,v[9]);adjust=max(adjust,v[7])
        for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]:errors[name]=max(errors.get(name,0),float(abs(v[i].sum(-1)-base[name][:,rr]).max()))
        for j,name in enumerate(m.source_names):
            row=pd.DataFrame(dict(year=np.repeat(d.months.year,stop-start),month=np.repeat(d.months.month,stop-start),global_reach_id=np.tile(d.global_reach_ids[rr],len(d.months)),source=name,original_input_kg=d.monthly_sum(m.raw_daily_tags[:,rr,j]).ravel(),corrected_input_kg=d.monthly_sum(inp[:,rr,j]).ravel()))
            for key,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]:row[key+'_kg']=(v[i][ends,:,j] if key in ['M','L'] else d.monthly_sum(v[i][:,:,j])).ravel()
            frames.append(row)
        del v
    sumchannel=np.zeros_like(base['channel_loss']);sumterminal=np.zeros_like(base['terminal']);sumstocks=np.zeros_like(base['reservoir_stocks'])
    for j,name in enumerate(m.source_names):
        rr=route(d,fast[:,:,j]+slow[:,:,j],vf=float(x[2]));sumchannel+=rr['channel_removed'];sumterminal+=rr['terminal'];sumstocks+=rr['stocks']
        network.append(pd.DataFrame(dict(year=d.months.year,month=d.months.month,source=name,terminal_kg=d.monthly_sum(rr['terminal']),reservoir_end_kg=rr['stocks'][ends].sum(1))))
    for name,a,b in [('channel',sumchannel,base['channel_loss']),('terminal',sumterminal,base['terminal']),('reservoir',sumstocks,base['reservoir_stocks'])]:errors[name]=float(abs(a-b).max())
    assert balance<=1e-6 and minimum>=-1e-7 and max(errors.values())<=1e-6,(balance,minimum,errors)
    pd.concat(frames).to_parquet(out/'full_monthly_source_ledger.parquet',index=False);pd.concat(network).to_parquet(out/'full_monthly_source_network.parquet',index=False)
    rt.write(out/'full_source_audit.json',dict(status='PASS',sources=m.source_names,max_errors=errors,per_source_balance=balance,minimum=minimum,rounding_adjustment=adjust,process=rt.process(os.getpid())))
    print('PASS full tags',tag,errors,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');main(p.parse_args().tag)
