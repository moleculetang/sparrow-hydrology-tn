"""One registered shared-parameter path; resumable LBFGSB and deterministic TRF."""
import argparse,time,os,traceback,hashlib,pickle
import native_runtime as rt
from campaign_model import *
from serial_solvers import LBFGSB,projected_gradient
from scipy.optimize import least_squares

class Yield(Exception):pass
class Budget(Exception):pass

def identity(job):
    manifest=RUN/'reports/launch_by_fold'/f"{job['fold']}.json"
    proof=rt.read(manifest)
    for p,h in proof['frozen_hashes'].items():
        if rt.sha(RUN/p)!=h:raise RuntimeError('REGISTERED_HASH_CHANGED '+p)
    folds=rt.read(RUN/'configs/folds.json');fold=job['fold']
    if rt.sha(RUN/'data/folds'/fold/'train.parquet')!=folds[fold]['train_sha256']:raise RuntimeError('TRAIN_HASH_CHANGED')
    return dict(job=job,launch_sha256=rt.sha(manifest))

class Runner:
    def __init__(self,job):
        self.job=job;self.root=RUN/'work/jobs'/job['tag'];self.root.mkdir(parents=True,exist_ok=True)
        rt.label_barrier(job['fold']);self.ident=identity(job);self.cfg=rt.read(RUN/'configs/campaign.json')
        self.campaign=rt.read(RUN/'work/campaign.json');self.started=time.monotonic();self.lastsave=self.started
        self.proc=rt.process(os.getpid());self.last_cpu=self.proc['cpu_seconds']
        if (self.root/'checkpoints/latest.json').exists():self.s=rt.restore(self.root/'checkpoints',self.ident)
        else:self.s=dict(identity=self.ident,calls=0,active_seconds=0.,cpu_seconds=0.,stage=0,phase='lbfgsb',engine=None,best=None,trace=[],diagnostic_used=False,repair_used=False,decisions=[],sessions=[],trf_start=None,trf_count=0,trf_attempts=0)
        self.s['sessions'].append(dict(pid=os.getpid(),created=self.proc['created'],started=time.time()))
        self.base_active=self.s['active_seconds'];self.base_cpu=self.s['cpu_seconds']
        self.model=for_job(job);self.scale=self.model.variable_scale()
        if job.get('parent_tags') and job['start']==0 and self.s['best'] is None:
            parents=[rt.read(RUN/'outputs'/tag/'model.json') for tag in job.get('parent_tags',job['dependencies']) if rt.read(RUN/'outputs'/tag/'audit.json').get('physical_reasonable') is True]
            parent=min(parents,key=lambda x:x['objective'])
            self.initial_override=np.asarray(parent['parameters'],float) if job.get('observation_operator')=='MATCH' else np.r_[parent['parameters'],parent['parameters'][1]]
            if job.get('reverse_initial'):
                beta=parent['parameters'][1];self.initial_override[1]=min(2.,beta+.1);self.initial_override[29]=max(.25,beta-.1)
            self.s['parent_identity']=dict(tag=parent['job']['tag'],model_sha256=rt.sha(RUN/'outputs'/parent['job']['tag']/'model.json'))
        rt.write(self.root/'design.json',self.model.design)
        rt.write(self.root/'preprocessing.json',dict(floor_policy='frozen per-cohort denominators',variance=self.model.train.fit_variance.tolist(),weights=self.model.weight.tolist(),training_ids=self.model.train.observation_id.tolist()))
        self.engine=None;self.save('RUNNING')
        if job.get('nested_tags') and self.s['best'] is None:
            parents=[rt.read(RUN/'outputs'/tag/'model.json') for tag in job['nested_tags'] if (RUN/'outputs'/tag/'audit.json').exists() and rt.read(RUN/'outputs'/tag/'audit.json').get('physical_reasonable')]
            from regional_response import embed
            references=[]
            for parent in parents:
                nested=embed(parent['parameters'],parent['job']['kind'],job['kind'])
                self.charge();j,g=self.model.value_gradient(nested)
                if abs(j-parent['objective'])>1e-8*(1+abs(j)):raise RuntimeError('NESTED_OBJECTIVE_MISMATCH')
                self.record(nested,j,g)
                references.append(dict(x=nested.tolist(),objective=float(j),parent=parent['job']['tag']))
            self.s['nested_references']=references
            self.s['nested_reference']=min(references,key=lambda a:a['objective'])
            self.s['initial_x']=(self.s['nested_reference']['x'] if job['start']==0 else self.model.initial(1).tolist())
            self.save()


    def save(self,status='RUNNING'):
        self.s['active_seconds']=self.base_active+time.monotonic()-self.started
        p=rt.process(os.getpid(),self.proc['created']);self.s['cpu_seconds']=self.base_cpu+p['cpu_seconds']-self.last_cpu
        if self.engine is not None:self.s['engine']=self.engine.s
        rt.checkpoint(self.root/'checkpoints',self.s)
        rt.write(self.root/'status.json',dict(status=status,job=self.job,process=p,calls=self.s['calls'],active_seconds=self.s['active_seconds'],phase=self.s['phase'],stage=self.s['stage'],best=self.s['best'],updated=time.time()))
        self.lastsave=time.monotonic()

    def guard(self):
        elapsed=self.base_active+time.monotonic()-self.started
        if time.time()>=self.campaign['training_deadline']:raise Budget('CAMPAIGN_TRAINING_DEADLINE')
        limits=self.cfg['solver'];stage=self.s['stage']
        if self.s['calls']>=limits['cumulative_calls'][stage] or elapsed>=limits['cumulative_hours'][stage]*3600:raise Budget('PATH_STAGE_BUDGET')
        if (self.root/'yield.request').exists():raise Yield('RESOURCE_CHECKPOINT_YIELD')
        if time.monotonic()-self.lastsave>=60:self.save()

    def charge(self):
        self.guard();self.s['calls']+=1
        # Durable precharge prevents a crash from erasing an attempted full recurrence.
        self.save()

    def record(self,x,j,g=None):
        if not np.isfinite(j) or (g is not None and not np.isfinite(g).all()):raise FloatingPointError('NONFINITE_OBJECTIVE_OR_GRADIENT')
        candidate=self.s['best'] is None or j<self.s['best']['objective']
        legal=True
        if candidate:
            # Complete independent ledger calls are charged to the same path budget.
            self.charge();a=self.model.ledger(x);scale=max(1.,float((a['fast']+a['slow']).sum()))
            legal=bool(a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=scale*1e-10 and min(float(a[k].min()) for k in ['M','L','available','uptake','fast','slow','mineral_loss'])>=-1e-7 and float((a['uptake']-a['demand']).max())<=1e-7 and max(a.get('source_label_sum_errors',{}).values(),default=0)<=1e-6)
            if not legal:self.s.setdefault('rejected_physical_points',[]).append(dict(call=self.s['calls'],objective=float(j),local=float(a['local_balance_max_kg']),source_errors=a.get('source_label_sum_errors',{})))
        pg=None if g is None else float(np.max(np.abs(projected_gradient(x,g,self.model.bounds))))
        row=dict(terms=dict(self.model.last_terms) if g is not None else None,call=self.s['calls'],objective=float(j),pg=pg,phase=self.s['phase'],physical_legal=legal,active_seconds=self.base_active+time.monotonic()-self.started)
        self.s['trace'].append(row)
        self.s['trace']=self.s['trace'][-100:]
        if g is not None and legal:
            self.s.setdefault('progress_trace',[]).append(row.copy())
            self.s['progress_trace']=self.s['progress_trace'][-100:]
        best=self.s['best']
        if legal and (best is None or j<best['objective']):
            self.s['best']=dict(x=x.tolist(),objective=float(j),pg=pg,call=self.s['calls'])
        elif best is not None and np.array_equal(x,np.asarray(best['x'])) and pg is not None:best['pg']=pg
        with (self.root/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        self.save()
        return pg

    def vg(self,z):
        x=np.asarray(z)*self.scale
        for i,(lo,hi) in enumerate(self.model.bounds):
            if x[i]<lo or x[i]>hi:
                b=lo if x[i]<lo else hi
                if abs(x[i]-b)>8*np.finfo(float).eps*max(1.,abs(b)):raise ValueError('BOUND_VIOLATION')
                x[i]=b
        self.charge();j,g=self.model.value_gradient(x);self.record(x,j,g)
        return j,g*self.scale

    def trf_residual(self,x):
        self.guard();i=self.cursor;folder=self.root/'trf';folder.mkdir(exist_ok=True)
        header=folder/f'{i:06d}.json';request=hashlib.sha256(x.tobytes()).hexdigest()
        if header.exists():
            ref=rt.read(header)
            if ref['request']!=request:raise RuntimeError('NONDETERMINISTIC_TRF_REPLAY')
            payload=folder/ref['payload']
            if rt.sha(payload)!=ref['sha256']:raise RuntimeError('TRF_CACHE_HASH_CHANGED')
            with payload.open('rb') as f:v=pickle.load(f)
            self.cursor+=1;return v
        self.charge();r=residual(self.model,x)
        if not np.isfinite(r).all():raise FloatingPointError('NONFINITE_RESIDUAL')
        name=f'{i:06d}_{self.s["calls"]}.pkl';payload=folder/name
        with payload.open('xb') as f:pickle.dump(r,f,protocol=5);f.flush();os.fsync(f.fileno())
        rt.write(header,dict(payload=name,sha256=rt.sha(payload),request=request))
        self.cursor+=1;self.s['trf_count']=self.cursor;self.record(x,float(.5*r@r))
        return r

    def run_phase(self):
        phase=self.s['phase']
        if phase in ['lbfgsb','polish','repair']:
            bounds=[(None if not np.isfinite(a) else a/self.scale[i],None if not np.isfinite(b) else b/self.scale[i]) for i,(a,b) in enumerate(self.model.bounds)]
            x=np.array(self.s['initial_x']) if phase=='lbfgsb' and 'solver_exit' not in self.s and 'initial_x' in self.s else (getattr(self,'initial_override',self.model.initial(self.job['start'])) if self.s['best'] is None else np.array(self.s['best']['x']))
            self.engine=LBFGSB(x/self.scale,bounds,stage=0 if phase=='lbfgsb' else 1,maxiter=24000,state=self.s['engine'])
            while not self.engine.s['done']:
                self.guard()
                if phase=='lbfgsb' and self.engine.s['evaluations']>=1500:break
                self.engine.step(self.vg)
            self.s['solver_exit']=dict(task=self.engine.s['task'].tolist(),iterations=self.engine.s['iterations'],evaluations=self.engine.s['evaluations'])
            self.engine=None;self.s['engine']=None
            if phase=='lbfgsb':self.s['phase']='trf';self.s['trf_start']=self.s['best']['x']
            else:self.s['phase']='assess'
            self.save()
        elif phase=='trf':
            self.cursor=0;lo,hi=np.array(self.model.bounds).T
            def accepted(intermediate_result):
                # FD perturbations are not optimizer progress windows. Replay
                # callbacks already committed before yield are deduplicated.
                key=hashlib.sha256(intermediate_result.x.tobytes()).hexdigest()
                seen=self.s.setdefault('trf_accepted_hashes',[])
                if key not in seen:
                    seen.append(key)
                    self.s.setdefault('progress_trace',[]).append(dict(objective=float(intermediate_result.cost),pg=None,phase='trf',call=self.s['calls']))
                    self.s['progress_trace']=self.s['progress_trace'][-100:]
                    self.save()
            result=least_squares(self.trf_residual,np.array(self.s['trf_start']),bounds=(lo,hi),jac='3-point',method='trf',tr_solver='exact',x_scale=self.scale,ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=500,callback=accepted)
            self.s['trf_exit']=dict(status=int(result.status),message=result.message,nfev=int(result.nfev),njev=int(result.njev))
            self.s['phase']='polish';self.save()
        elif phase=='assess':
            x=np.array(self.s['best']['x']);self.vg(x/self.scale)
            b=self.s['best']
            if b['pg'] is not None and b['pg']<=1e-5:return 'NUMERICALLY_SUFFICIENT'
            if not self.s['repair_used']:
                # A reversible coordinate change only: scale spread diagnosed from the
                # current original gradient, bounded to a factor of ten per coordinate.
                self.charge();j,g=self.model.value_gradient(x);self.record(x,j,g)
                scaled=np.abs(g*self.scale);positive=scaled[scaled>1e-12]
                if len(positive) and positive.max()/positive.min()>100:
                    factor=np.clip(np.sqrt(np.median(positive)/np.maximum(scaled,1e-12)),.1,10.)
                    self.s['repair_factor']=factor.tolist();self.scale=self.model.variable_scale()*factor
                    self.s['repair_used']=True;self.s['phase']='repair'
                    self.s['decisions'].append(dict(reason='SCALED_GRADIENT_SPREAD',spread=float(positive.max()/positive.min()),factor=factor.tolist()))
                    self.save();return None
            return 'NUMERICALLY_INSUFFICIENT_STATIONARY'
        else:raise RuntimeError('UNKNOWN_PHASE '+phase)

    def run(self):
        if self.s.get('repair_factor') is not None:self.scale=self.model.variable_scale()*np.array(self.s['repair_factor'])
        while True:
            try:
                status=self.run_phase()
                if status:self.save(status);return status
            except Yield:
                self.save('RESOURCE_YIELDED');return 'RESOURCE_YIELDED'
            except Budget as e:
                decision=rt.continuation(self.s.get('progress_trace',[]),self.s['diagnostic_used']);self.s['decisions'].append(dict(trigger=str(e),**decision))
                if str(e)=='PATH_STAGE_BUDGET' and self.s['stage']<2 and decision['continue_training']:
                    self.s['stage']+=1;self.s['diagnostic_used']|=decision['diagnostic'];self.save();continue
                self.save('BUDGET_STOPPED');return 'BUDGET_STOPPED'

def main():
    p=argparse.ArgumentParser();p.add_argument('tag');args=p.parse_args()
    job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==args.tag)
    with rt.exclusive(job['tag']):
        runner=None
        try:runner=Runner(job);runner.run()
        except Exception:
            error=traceback.format_exc()
            if runner is not None:
                runner.s['exception']=error;runner.save('FAILED')
            else:rt.write(RUN/'work/jobs'/job['tag']/'status.json',dict(status='FAILED',error=error,updated=time.time()))
            raise

if __name__=='__main__':main()
