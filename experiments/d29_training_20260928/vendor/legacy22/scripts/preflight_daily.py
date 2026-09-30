"""New complete-history gates; historical evidence never replaces this run."""
import argparse,time,gc
import native_runtime as rt
from campaign_model import *
from daily_inputs import daily_reverse
from closures import scan

def physical(a):
    out=dict(local=float(a['local_balance_max_kg']),network=float(a['network_balance_kg']),scale=max(1.,float((a['fast']+a['slow']).sum())),source=max(a['source_label_sum_errors'].values(),default=0),minimum=min(float(a[k].min()) for k in ['M','L','available','uptake','fast','slow','mineral_loss']),uptake_excess=float((a['uptake']-a['demand']).max()))
    assert out['local']<=1e-6 and abs(out['network'])<=out['scale']*1e-10 and out['source']<=1e-6 and out['minimum']>=-1e-7 and out['uptake_excess']<=1e-7,out
    return out

def main(tag):
    start=time.time();j=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==tag)
    m=for_job(j);short=tag[:3];structure='L3' if j['kind']=='REGIONAL_L3' else 'U'
    oldtag=f'{short}_{structure}_s'+('0' if structure=='L3' else '1')
    rec=rt.read(RUN.parent/'20260921_3/outputs'/oldtag/'model.json');x=np.array(rec['parameters'])
    value,g=m.value_gradient(x);a=m.ledger(x);ph=physical(a);del a;gc.collect()
    checks=[];nest=None
    if j['input_mode']=='P':
        design=dict(m.design);design.pop('daily_input')
        old=make_model(m.data,m.train,j['kind'],design);jo,go=old.value_gradient(x)
        pred=m.predict(x,m.meta);po=old.predict(x,old.meta)
        aa=m.ledger(x);ao=old.ledger(x)
        errors={k:float(abs(aa[k]-ao[k]).max()) for k in ['M','L','fast','slow','uptake','mineral_loss']}
        nest=dict(objective=abs(value-jo),gradient=float(abs(g-go).max()),prediction=float(abs(pred-po).max()),state_errors=errors,saved_objective_error=abs(value-rec['objective']))
        assert nest['objective']<=1e-8*(1+abs(jo)) and nest['saved_objective_error']<=1e-8*(1+abs(jo))
        assert nest['gradient']<=1e-8 and nest['prediction']<=1e-9 and max(errors.values())<=1e-6,nest
        del old,aa,ao;gc.collect()
    # Changed daily source derivative and process pathways, full-history objective.
    for index in ([0,30,31,54] if structure=='L3' else [0,30,29]):
        results=[]
        for step in [2e-5,1e-5]:
            p=x.copy();n=x.copy();p[index]+=step;n[index]-=step
            lo,hi=m.bounds[index]
            if n[index]<lo or p[index]>hi:
                sign=1 if n[index]<lo else -1;p=x.copy();n=x.copy();p[index]+=sign*step;n[index]+=sign*2*step
                rp=residual(m,p);rn=residual(m,n);fd=sign*(-3*value+2*(rp@rp)-.5*(rn@rn))/(2*step)
            else:
                rp=residual(m,p);rn=residual(m,n);fd=(rp@rp-rn@rn)/(4*step)
            err=abs(fd-g[index]);tol=2e-4*(1+abs(fd)+abs(g[index]));results.append(dict(index=index,step=step,analytic=float(g[index]),finite_difference=float(fd),error=float(err),tolerance=float(tol)))
        assert all(c['error']<=c['tolerance'] for c in results),results
        checks+=results
    rt.write(RUN/'reports'/f'preflight_daily_{tag}.json',dict(status='PASS',tag=tag,physical=ph,nesting=nest,gradient=checks,objective=value,process=rt.process(os.getpid()),seconds=time.time()-start))
    print('PASS',tag,'seconds',time.time()-start,'peak',rt.process(os.getpid())['peak_gib'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');main(p.parse_args().tag)
