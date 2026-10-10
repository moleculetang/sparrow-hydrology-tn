"""Full-history signed and bounded source-coordinate derivatives under new calendars."""
import time,gc
import native_runtime as rt
from campaign_model import *
def main():
    rt.resources();time.sleep(.2);now=rt.resources();live=[]
    for a in rt.read(RUN/'work/controller_status.json').get('live',[]):
        try:live.append(rt.process(a['pid'],a['created']))
        except (OSError,RuntimeError):pass
    assert rt.admission(now,4.,live),'RESOURCE_ADMISSION_WAIT'
    rt.log('auxiliary_admissions.jsonl',dict(task='supplemental_daily_acceptance',resources=now,measured_live=live,reserved_cold_peak_gib=4.))
    rt.label_barrier('T24_G_D_H1');checks=[];jobs=rt.read(RUN/'configs/jobs.json')
    for structure,kind in [('U','SOURCE_UNIFIED'),('L3','REGIONAL_L3')]:
        old=RUN.parent/'20260921_3/outputs'/('F24_U_s1' if structure=='U' else 'F24_L3_s0')/'model.json';base=np.array(rt.read(old)['parameters'])
        for mode in ['D','A']:
            job=next(j for j in jobs if j['kind']==kind and j['fold']=='T24_G_D_H1' and j['input_mode']==mode and j['start']==0);m=for_job(job)
            for eta in [-math.log(4),-.3,.3,math.log(4)]:
                x=base.copy();x[30]=eta;j,g=m.value_gradient(x);local=[]
                for step in [2e-5,1e-5,5e-6,2.5e-6]:
                    p=x.copy();n=x.copy();lo,hi=m.bounds[30]
                    if eta-step<lo or eta+step>hi:
                        sign=1 if eta-step<lo else -1;p[30]+=sign*step;n[30]+=sign*2*step;rp=residual(m,p);rn=residual(m,n);fd=sign*(-3*j+2*(rp@rp)-.5*(rn@rn))/(2*step)
                    else:
                        p[30]+=step;n[30]-=step;rp=residual(m,p);rn=residual(m,n);fd=(rp@rp-rn@rn)/(4*step)
                    err=abs(fd-g[30]);tol=2e-4*(1+abs(fd)+abs(g[30]));row=dict(structure=structure,input_mode=mode,eta=eta,step=step,analytic=float(g[30]),finite_difference=float(fd),error=float(err),tolerance=float(tol));local.append(row);checks.append(row)
                    if len(local)>=2 and all(c['error']<=c['tolerance'] for c in local[-2:]):break
                assert len(local)>=2 and all(c['error']<=c['tolerance'] for c in local[-2:]),local
            del m;gc.collect();print('PASS BOUNDED DAILY',structure,mode,flush=True)
    rt.write(RUN/'reports/supplemental_daily_acceptance.json',dict(status='PASS',checks=checks,process=rt.process(os.getpid()),note='One-sided feasible differences at multiplier bounds; non-smooth depleted cells remain original max recurrence, no smoothing'))
if __name__=='__main__':main()
