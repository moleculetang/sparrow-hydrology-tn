"""Independent objective, feasible derivative probes and block conflicts at a fitted LAND1 point."""
import sys,json,ast,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np
from d29_training.land1_adapter import Land1Training
from d29_training.experiment_context import ExperimentContext


def main(job_id):
    ok,_=dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    context=ExperimentContext.load();job=context.job(job_id)
    context.check_worker(job_id);folder=context.folder(job_id)
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status'] in ('running','resource_checkpoint','continuing_same_path_zero_ftol'):
        raise RuntimeError('LIVE_CHECKPOINT')
    a=Land1Training(job);x=np.load(folder/'best.npy');saved={};evaluate=a.objective.evaluate
    def capture(p):
        saved['prediction']=p.copy();return evaluate(p)
    a.objective.evaluate=capture
    started=time.monotonic();value,g=a.value_gradient(x);prior=a.last['prior']
    from serial_solvers import projected_gradient
    tree=ast.parse((ROOT/'scripts/audit_u_solution.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recompute')
    namespace={'math':math,'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<independent_scalar_objective>','exec'),namespace)
    terms=namespace['recompute'](a.objective.rows,saved['prediction'],a.objective.scales,job['strategy'],a.objective.long_sites)
    independent=math.fsum(terms.values())+prior;pg=float(np.max(abs(projected_gradient(x,g,a.bounds))))
    receipt=dict(checkpoint_sha256=sha(folder/'best.npy'),objective=value,independent_objective=independent,
        objective_error=abs(value-independent),objective_passed=abs(value-independent)<=1e-8*(1+abs(value)),
        projected_gradient=pg,numerically_sufficient=pg<=1e-5,source_multiplier=float(np.exp(x[-1])),
        data_terms=terms,prior=prior,NSE='not applicable to numerical audit',direction_checks=[])
    write_json(folder/'independent_audit_partial.json',receipt)
    rng=np.random.default_rng(1729)
    for k in range(3):
        d=rng.normal(size=len(x));d[(x-a.bounds[:,0]<1e-3)|(a.bounds[:,1]-x<1e-3)]=0
        norm=np.linalg.norm(d)
        if norm==0:
            receipt['direction_checks'].append({'status':'no_free_tangent_coordinates','passed':False});continue
        d/=norm;samples=[]
        for h in [1e-5,3e-6,1e-6]:
            vp,_=a.value_gradient(x+h*d,forward_only=True);vm,_=a.value_gradient(x-h*d,forward_only=True)
            fd=(vp-vm)/(2*h);error=abs(fd-g@d)
            samples.append(dict(step=h,fd=fd,analytic=float(g@d),error=error,passed=bool(error<=1e-6*(1+abs(g@d)))))
        passed=any(samples[i]['passed'] and samples[i+1]['passed'] for i in range(len(samples)-1))
        receipt['direction_checks'].append(dict(direction=d.tolist(),samples=samples,passed=passed,
            boundary_scope='interior tangent only; full coordinate and boundary gate is separate'))
        write_json(folder/'independent_audit_partial.json',receipt)
    receipt['direction_gate_passed']=all(r['passed'] for r in receipt['direction_checks'])
    receipt['calls']=a.calls;receipt['elapsed_seconds']=time.monotonic()-started
    write_json(folder/'independent_audit.json',receipt)
    # Each block VJP includes the same prior. Subtract a separately computed
    # prior VJP, avoiding changes to the frozen training adapter.
    def block_objective(index):
        def run(p):
            _,_,terms0,_=evaluate(p);dp=np.zeros_like(p)
            if index is None:return 0.,dp,{},None
            b=a.objective.blocks[index];e=b.center(p[b.indices]-a.objective.truth[b.indices])
            dp[b.indices]=b.transpose_center(b.sqrt_weight*b.sqrt_weight*e)
            return float(.5*np.dot(b.sqrt_weight*e,b.sqrt_weight*e)),dp,terms0,None
        return run
    a.objective.evaluate=block_objective(None);_,prior_g=a.value_gradient(x)
    gradients={'prior':prior_g};values={'prior':prior}
    for i,b in enumerate(a.objective.blocks):
        a.objective.evaluate=block_objective(i);v,bg=a.value_gradient(x)
        gradients[b.name]=bg-prior_g;values[b.name]=v-prior
    a.objective.evaluate=evaluate
    difference=float(np.max(abs(np.sum(list(gradients.values()),axis=0)-g)))
    if difference>1e-8*(1+np.max(abs(g))):raise RuntimeError('BLOCK_GRADIENT_SUM_MISMATCH')
    scales=a.model.variable_scale()[a.keep]
    def cosine(u,v):
        den=np.linalg.norm(u)*np.linalg.norm(v)
        return float(np.dot(u,v)/den) if den>0 else None
    names=list(gradients);pairs=[]
    for i,k in enumerate(names):
        for l in names[i+1:]:pairs.append(dict(first=k,second=l,raw_coordinate_cosine=cosine(gradients[k],gradients[l]),fixed_native_scaled_coordinate_cosine=cosine(gradients[k]*scales,gradients[l]*scales)))
    write_json(folder/'training_block_gradients.json',dict(parameter_names=a.names,block_values=values,
        raw_gradients={k:v.tolist() for k,v in gradients.items()},fixed_parameter_scales=scales.tolist(),cosines=pairs,
        gradient_sum_max_error=difference,NSE='not applicable',calls_including_independent_audit=a.calls,
        interpretation='local gradient conflict is descriptive; not a unique attribution to observation or process errors'))


if __name__=='__main__':main(sys.argv[1])
