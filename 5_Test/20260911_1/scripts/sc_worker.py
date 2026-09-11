"""One registered parameter path; cumulative budgets survive every safe yield."""
import os
from pathlib import Path
os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'work/numba_cache')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import copy
import time
import traceback
import numpy as np
import pandas as pd
import torch
from fc_io import *
from fc_checkpoint import ResourceYield,BudgetStop
from fc_resources import process_snapshot
from fc_solvers import LBFGSB,projected_gradient
from gb_solver import CallReplay,trf_fit
from sc_model import SharedBox
from sc_runtime import (exclusive,label_barrier,identity,checkpoint,restore,continuation,
                        legal_point,sufficient,validate_policy)


def run(tag,deadline):
    require_environment();torch.set_num_threads(1);label_barrier()
    fold,variant,start=tag.split('_');start=int(start)
    if fold not in ('F23','F24') or variant not in ('M0','SC') or start not in (0,1):
        raise ValueError('UNREGISTERED_PATH')
    ident=identity(fold,variant,start)
    cp=RUN/f'work/checkpoints/{tag}.pt';sp=RUN/f'work/status/{tag}.json'
    fitpath=RUN/f'reports/fits/{tag}.json'
    if fitpath.exists():raise RuntimeError('PATH_ALREADY_HAS_TERMINAL_REPORT')
    if cp.exists() or Path(str(cp)+'.sha256.json').exists():state=restore(cp,ident)
    else:state={'identity':ident,'calls':0,'active_seconds':0.,'stage':0,'phase':'TRF','trace':[],
       'best':None,'engine':None,'repair_used':False,'diagnostic_used':False,'extensions':[],
       'fit_done':False,'complete':False,'max_call_seconds':0.,'session_open':False}
    if state['complete']:raise RuntimeError('PATH_ALREADY_TERMINAL')
    # A forced/crashed session cannot be resumed with time silently erased.
    if state.get('session_open'):
        closure=RUN/f'work/closed_sessions/{tag}/{state["session_key"]}.json'
        if not closure.exists():raise RuntimeError('UNCLOSED_SESSION_REQUIRES_NATIVE_TIME_ACCOUNTING')
        closed=read(closure)
        if closed['session_key']!=state['session_key'] or closed['identity']!=ident:
            raise RuntimeError('SESSION_CLOSURE_IDENTITY_CHANGED')
        elapsed=float(closed['elapsed_process_seconds'])
        if not np.isfinite(elapsed) or elapsed<state['session_elapsed_seconds']:
            raise RuntimeError('SESSION_CLOSURE_LOSES_EXECUTION_TIME')
        state['active_seconds']=state['session_base_seconds']+elapsed
        state.setdefault('closed_sessions',[]).append(closed)
    elapsed_base=state['active_seconds'];me=process_snapshot(os.getpid())
    # Native lifetime includes imports, validation and loading.
    started=time.monotonic()-max(0.,time.time()-me['process_created'])
    state.update(session_key=f'{me["pid"]}_{me["process_created"]:.6f}',session_open=True,
                 session_base_seconds=elapsed_base,session_elapsed_seconds=0.)
    policy=read(RUN/'configs/campaign.json');validate_policy(policy)
    cfg=read(RUN/f'configs/{fold}.json')
    status={'tag':tag,'status':'LOADING','calibration_ready':False,**me}

    def active_seconds():return elapsed_base+time.monotonic()-started

    def persist(close=False):
        state['active_seconds']=active_seconds();state['session_elapsed_seconds']=time.monotonic()-started
        if close:state['session_open']=False
        digest=checkpoint(cp,state)
        status.update(calls=state['calls'],active_seconds=state['active_seconds'],stage=state['stage'],phase=state['phase'],
          best=state['best'],fit_done=state['fit_done'],utc=now(),checkpoint_manifest=str(cp)+'.sha256.json',
          checkpoint_sha256=digest,session_key=state['session_key'],session_base_seconds=elapsed_base,
          session_elapsed_seconds=state['session_elapsed_seconds'],**process_snapshot(os.getpid()))
        write(sp,status)

    def operational_guard():
        request=RUN/f'work/requests/{tag}.json'
        if request.exists():
            action=read(request)['action']
            if action=='resource_yield':raise ResourceYield('RAM90_CHECKPOINT_YIELD')
            if action=='budget_stop':raise BudgetStop('CONTROLLER_DEADLINE')
            raise RuntimeError('UNKNOWN_CONTROLLER_REQUEST '+str(action))
        if memory()['used_percent']>=90:raise ResourceYield('RAM90_CHECKPOINT_YIELD')

    def guard(reserve_calls=8,terminal=False):
        operational_guard()
        # Training, two terminal calls and the two-call independent child audit
        # share the registered time/call ceilings; no extra scientific budget.
        reserve_seconds=(max(30.,2.4*state['max_call_seconds']+10.) if terminal else
                         max(60.,4.8*state['max_call_seconds']+20.))
        if time.time()>=deadline-reserve_seconds:raise BudgetStop('CAMPAIGN_TRAINING_DEADLINE')
        if state['calls']>=policy['max_calls'][state['stage']]-reserve_calls:
            raise BudgetStop('CUMULATIVE_CALL_SEGMENT')
        if active_seconds()>=policy['max_hours'][state['stage']]*3600-reserve_seconds:
            raise BudgetStop('CUMULATIVE_TIME_SEGMENT')

    def charge(kind,terminal=False):
        guard(2 if terminal else 8,terminal)
        state['calls']+=1
        state['inflight']={'call':state['calls'],'kind':kind,'started_active_seconds':active_seconds()}
        persist()

    def finished_call():
        duration=active_seconds()-state['inflight']['started_active_seconds']
        state['max_call_seconds']=max(state['max_call_seconds'],duration)
        state['inflight']=None

    persist()
    try:
        operational_guard()
        data=torch.load(cfg['cache']['path'],map_location='cpu',weights_only=False)
        train=pd.read_parquet(RUN/f'outputs/data/{fold}_train.parquet')
        model=SharedBox(data,train,variant,np.array(cfg['scale']),cfg['designs'][variant])
        expected_names=cfg['names'][variant] if isinstance(cfg['names'],dict) else cfg['names']
        if model.names!=expected_names:raise RuntimeError('PARAMETER_ORDER_CHANGED')
        if 'x0' not in state:
            if variant=='SC' and start==0:
                candidates=[read(RUN/f'reports/fits/{fold}_M0_{k}.json') for k in (0,1)]
                for v in candidates:
                    if not read(RUN/f'reports/audits/{v["tag"]}.json')['physical_pass']:
                        raise RuntimeError('M0_INITIALIZATION_NOT_PHYSICAL')
                    if sha(v['model_path'])!=v['model_sha256']:raise RuntimeError('M0_INITIALIZATION_HASH_CHANGED')
                best=min(candidates,key=lambda v:v['objective'])
                a=torch.load(best['model_path'],weights_only=False,map_location='cpu')
                state['x0']=np.r_[a['theta'],np.zeros(6)]
                state['nested_origin']={'tag':best['tag'],'sha256':best['model_sha256']}
            else:state['x0']=np.array(cfg['initials'][f'{variant}_{start}'])
            state['x0']=legal_point(state['x0'],model.bounds);persist()

        def evaluate(x,gradient=False):
            x=legal_point(x,model.bounds);charge('value_gradient' if gradient else 'residual')
            if gradient:f,g=model.value_gradient(x);g=np.asarray(g,float);res=None
            else:
                res=np.asarray(model.residual(x),float)
                if res.ndim!=1 or not np.isfinite(res).all():raise FloatingPointError('NONFINITE_RESIDUAL')
                f=float(.5*res@res);g=None
            if not np.isfinite(f) or (g is not None and (g.shape!=x.shape or not np.isfinite(g).all())):
                raise FloatingPointError('NONFINITE_POINT')
            pg=float(abs(projected_gradient(x,g,model.bounds)).max()) if g is not None else None
            finished_call()
            row={'call':state['calls'],'objective':float(f),'projected_gradient':pg,
                 'active_seconds':active_seconds(),'x':x.tolist()}
            if g is not None:row['gradient']=g.tolist()
            state['trace'].append(row)
            if state['best'] is None or f<state['best']['objective']:state['best']=row
            if sufficient(row,state['best']):state['sufficient']=row
            if not sufficient(state.get('sufficient'),state['best']):state.pop('sufficient',None)
            status.update(status='FITTING',calibration_ready=True);persist()
            return (f,g) if gradient else res

        while not state['fit_done']:
            segment_stop=False
            try:
                if state['phase']=='TRF':
                    def trfguard():
                        guard()
                        if state['calls']>=3000:raise BudgetStop('RESERVE_ANALYTIC_REFINEMENT')
                    replay=CallReplay(RUN/f'work/trf/{tag}',ident,guard=trfguard,max_calls=3000)
                    try:
                        class Proxy:
                            bounds=model.bounds
                            variable_scale=model.variable_scale
                            def residual(self,x):return evaluate(x,False)
                        state['trf_result']=trf_fit(Proxy(),state['x0'],replay)
                    except BudgetStop as e:
                        state['trf_stop']=str(e)
                        if str(e) not in ('RESERVE_ANALYTIC_REFINEMENT','FOUR_THOUSAND_FORWARD_CALLS'):raise
                    state['phase']='ANALYTIC';persist()
                if state['best'] is None:raise RuntimeError('NO_TRF_POINT_FOR_ANALYTIC_REFINEMENT')
                scale=np.asarray(state.get('scale',model.variable_scale()),float)
                if scale.shape!=(len(model.bounds),) or not np.isfinite(scale).all() or np.any(scale<=0):
                    raise ValueError('INVALID_REVERSIBLE_PARAMETER_SCALE')
                if state['engine'] is None:
                    x=legal_point(state['best']['x'],model.bounds)
                    bounds=[(None if a is None else a/scale[i],None if b is None else b/scale[i]) for i,(a,b) in enumerate(model.bounds)]
                    engine=LBFGSB(x/scale,bounds,stage=1,maxiter=8000);engine.s['pgtol']=0.;state['scale']=scale
                else:engine=LBFGSB([],[],state=state['engine'])
                def fg(z):
                    x=legal_point(np.asarray(z)*scale,model.bounds,rounding=True)
                    f,g=evaluate(x,True);return f,g*scale
                while True:
                    guard();state['engine']=copy.deepcopy(engine.s);persist()
                    event=engine.step(fg);state['engine']=copy.deepcopy(engine.s)
                    if sufficient(state.get('sufficient'),state['best']):
                        state['fit_done']=True;state['reason']='ORIGINAL_COORDINATE_SUFFICIENT';break
                    if event=='done':
                        state['native_termination']=engine.s['task'].tolist()
                        x=legal_point(state['best']['x'],model.bounds);f,g=evaluate(x,True)
                        if sufficient(state.get('sufficient'),state['best']):
                            state['fit_done']=True;state['reason']='ORIGINAL_COORDINATE_SUFFICIENT';break
                        # One registered coordinate heuristic, not proof of bad derivatives.
                        mags=abs(g)*model.variable_scale();nz=mags[mags>1e-12]
                        if not state['repair_used'] and len(nz) and nz.max()/nz.min()>1e4:
                            state['repair_used']=True
                            state['repair_evidence']={'scaled_gradient_ratio':float(nz.max()/nz.min()),
                                'x':x.tolist(),'objective':float(f),'gradient':g.tolist(),
                                'original_coordinate_pg':float(abs(projected_gradient(x,g,model.bounds)).max()),
                                'origin_scale':scale.tolist(),'native_task':engine.s['task'].tolist(),
                                'type':'one_reversible_diagonal_coordinate_restart_same_best_point'}
                            state['scale']=model.variable_scale()/np.clip(np.sqrt(mags/max(np.median(nz),1e-12)),.1,10.)
                            state['engine']=None;break
                        state['reason']='NATIVE_STOP_WITH_INSUFFICIENT_ORIGINAL_GRADIENT';state['fit_done']=True;break
                    persist()
            except BudgetStop as e:
                if str(e) in ('CAMPAIGN_TRAINING_DEADLINE','CONTROLLER_DEADLINE'):
                    state['reason']=str(e);state['fit_done']=True
                else:segment_stop=True;state['segment_reason']=str(e)
            if segment_stop:
                trend=continuation(state['trace'],state['diagnostic_used'])
                trend.update(stage=state['stage'],calls=state['calls'],active_seconds=active_seconds(),
                             reason_for_segment_stop=state['segment_reason'])
                allowed=state['stage']<2 and trend['continue'];trend['extension_granted']=allowed
                state['extensions'].append(trend)
                if allowed:
                    state['stage']+=1;state['diagnostic_used']|=trend['diagnostic']
                    # A loading/first-call timeout has no point to hand off.
                    state['phase']='ANALYTIC' if state['best'] is not None else 'TRF'
                else:state['fit_done']=True;state['reason']='FINITE_CONTINUATION_EXHAUSTED_OR_UNSUPPORTED'
            persist()

        if state['best'] is None:raise RuntimeError('NO_LEGAL_EVALUATION')
        state['phase']='FINALIZE';status['status']='FINALIZING'
        selected=state.get('sufficient') if sufficient(state.get('sufficient'),state['best']) else state['best']
        x=legal_point(selected['x'],model.bounds)
        final=state.setdefault('final',{'x':x.tolist()})
        if not np.array_equal(final['x'],x):raise RuntimeError('FINAL_PARAMETER_SELECTION_CHANGED')
        persist()
        if 'gradient' not in final:
            charge('terminal_value_gradient',terminal=True);f,g=model.value_gradient(x);g=np.asarray(g,float)
            if not np.isfinite(f) or g.shape!=x.shape or not np.isfinite(g).all():raise FloatingPointError('NONFINITE_FINAL_GRADIENT')
            finished_call();final.update(objective=float(f),gradient=g.tolist(),
                projected_gradient=float(abs(projected_gradient(x,g,model.bounds)).max()));persist()
        f=final['objective'];pg=final['projected_gradient']
        metadata=pd.read_parquet(RUN/f'outputs/data/{fold}_heldout_metadata.parquet')
        from fc_data import prediction_metadata
        combined=pd.concat([prediction_metadata(train),metadata],ignore_index=True)
        if 'prediction' not in final:
            charge('terminal_prediction',terminal=True);pred=np.asarray(model.predict(x,combined),float)
            if pred.shape!=(len(combined),) or not np.isfinite(pred).all() or np.any(pred<0):
                raise FloatingPointError('INVALID_FINAL_PREDICTIONS')
            finished_call();final['prediction']=pred;persist()
        operational_guard()
        artifact=model.snapshot(x);artifact.update(identity=ident,initial=state['x0'],nested_origin=state.get('nested_origin'))
        from fc_checkpoint import save
        path=RUN/f'outputs/models/{tag}.pt';digest=save(path,artifact)
        combined['prediction_mg_l']=final['prediction'];pp=local(RUN/f'outputs/predictions/{tag}.parquet')
        combined.to_parquet(pp,index=False)
        within_budget=(state['calls']<=policy['max_calls'][state['stage']] and
                       active_seconds()<=policy['max_hours'][state['stage']]*3600 and time.time()<=deadline)
        result={'tag':tag,'status':'FIT_FINISHED' if within_budget else 'BUDGET_LIMITED',
          'model_path':str(path),'model_sha256':digest,'prediction_path':str(pp),'prediction_sha256':sha(pp),
          'objective':f,'projected_gradient':pg,'numerically_sufficient':sufficient(final,state['best']),
          'lowest_legal_objective':state['best']['objective'],'calls':state['calls'],'active_seconds':active_seconds(),
          'reason':state['reason'],'identity':ident,'stage':state['stage'],'extensions':state['extensions'],
          'repair':state.get('repair_evidence'),'within_registered_budget':within_budget,
          'checkpoint_manifest':str(cp)+'.sha256.json','session_key':state['session_key']}
        write(fitpath,result);state['complete']=True;status.update(status=result['status']);persist(close=True)
        return 0 if within_budget else 76
    except ResourceYield as e:
        status.update(status='RESOURCE_YIELD',reason=str(e));persist(close=True);return 75
    except BudgetStop as e:
        state['reason']=str(e);status.update(status='FINALIZATION_BUDGET_LIMITED',reason=str(e))
        partial={'tag':tag,'status':'BUDGET_LIMITED','reason':str(e),'identity':ident,
                 'calls':state['calls'],'active_seconds':active_seconds(),'stage':state['stage'],
                 'checkpoint_manifest':str(cp)+'.sha256.json','best':state['best'],
                 'numerically_sufficient':False,'complete_artifacts':False}
        write(fitpath,partial);state['complete']=True;persist(close=True);return 76
    except Exception as e:
        status.update(status='FAILED',error=str(e),traceback=traceback.format_exc());persist(close=True)
        print(status['traceback'],flush=True);return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--deadline',required=True,type=float);a=p.parse_args()
    with exclusive(a.tag):raise SystemExit(run(a.tag,a.deadline))
