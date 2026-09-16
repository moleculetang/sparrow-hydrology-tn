"""Fixed preset resource fixture, not an optimizer path or a scientific start."""
import gc,time,copy
import native_runtime as rt
from campaign_model import *
from audit_job import audit
R=RUN
def main():
 job=copy.deepcopy(next(j for j in rt.read(R/'configs/jobs.json') if j['tag']=='T25S_G_D_s1'));job['tag']='RESOURCE_PROFILE_FIXED_POINT';job['profile_only']=True
 m=for_job(job);x=m.initial(1);value,g=m.value_gradient(x);del m,g;gc.collect()
 root=R/'work/jobs'/job['tag'];state=dict(identity={'profile_only':True,'preset':1},best=dict(x=x.tolist(),objective=float(value)),calls=1,active_seconds=0.,cpu_seconds=0.)
 rt.checkpoint(root/'checkpoints',state);rt.write(root/'status.json',dict(status='RESOURCE_PROFILE_FIXED_POINT',best=state['best'],profile_only=True,not_registered_fit=True))
 audit(job)
 snap=rt.process(os.getpid());rt.write(R/'reports/single_audit_peak.json',dict(process=snap,kind='D29_BE',scope='FULL25 largest daily objective plus all 116 station outputs and independent physical ledger',fixed_preset=1,optimization_iterations=0,registered_path=False,script_sha256=rt.sha(R/'scripts/profile_single_audit.py')))
 print('SINGLE_AUDIT_PEAK',snap['peak_gib'],flush=True)
if __name__=='__main__':main()
