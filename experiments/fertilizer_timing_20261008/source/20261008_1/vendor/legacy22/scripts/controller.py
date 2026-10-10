"""Finite registered-path controller; child-exit audit and dependency-aware admission."""
import os,sys,time,subprocess,traceback,ctypes,hashlib,json
from pathlib import Path
import native_runtime as rt
RUN=rt.RUN
TERMINAL={'NUMERICALLY_SUFFICIENT','NUMERICALLY_INSUFFICIENT_STATIONARY','BUDGET_STOPPED','FAILED','NOT_STARTED_DEADLINE','DEPENDENCY_FAILED','SKIPPED_CONDITION'}

def event(kind,**details):
    rt.log('resource_events.jsonl',dict(event=kind,**details))

def main():
    with rt.exclusive('campaign_controller'):
        # Signal exceptional exits to the quiet external waiter; no status polling.
        event_name='Local\\SPARROW_attention_'+hashlib.sha256(str(RUN).encode()).hexdigest()[:16]
        rt.k32.CreateEventW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p];rt.k32.CreateEventW.restype=ctypes.c_void_p
        attention=rt.k32.CreateEventW(None,True,False,event_name)
        rt.k32.SetEvent.argtypes=[ctypes.c_void_p]
        proof=rt.read(RUN/'reports/launch_validation.json')
        if proof['status']!='PASS_LAUNCH_VALIDATION':raise RuntimeError('UNVALIDATED_LAUNCH')
        cfg=rt.read(RUN/'configs/campaign.json');jobs=rt.read(RUN/'configs/jobs.json');stamp=RUN/'work/campaign.json'
        if not stamp.exists():
            now=time.time();clock=rt.read(RUN/'work/experiment_clock.json');rt.write(stamp,dict(started=now,training_deadline=clock['training_deadline'],delivery_deadline=clock['delivery_deadline'],launch_sha256=rt.sha(RUN/'reports/launch_validation.json')))
        campaign=rt.read(stamp)
        if campaign['launch_sha256']!=rt.sha(RUN/'reports/launch_validation.json'):raise RuntimeError('LAUNCH_CHANGED')
        from recover_controller import adopt
        live=adopt(jobs,proof);paused=False;rt.resources();lastupdate=0
        event('CONTROLLER_START',pid=os.getpid(),adopted=list(live))
        def launch(job,audit=False):
            tag=job['tag'];root=RUN/'work/jobs'/tag;root.mkdir(parents=True,exist_ok=True)
            log=(root/('audit.log' if audit else 'fit.log')).open('ab')
            script='audit_job.py' if audit else 'fit_worker.py'
            child=subprocess.Popen([sys.executable,'-B',str(RUN/'scripts'/script),tag],cwd=RUN,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            birth=rt.process(child.pid)
            live[tag]=dict(child=child,log=log,job=job,audit=audit,created=birth['created'],peak=proof['peak_reservations_gib'][job['kind']])
            event('DISPATCH',tag=tag,audit=audit,pid=child.pid,created=birth['created'],peak_gib=proof['peak_reservations_gib'][job['kind']])
        while True:
            # Each process exit is consumed independently; audit gets priority.
            for tag,item in list(live.items()):
                code=item['child'].poll()
                if code is None:continue
                item['log'].close();del live[tag];job=item['job'];root=RUN/'work/jobs'/tag
                event('CHILD_EXIT',tag=tag,audit=item['audit'],exit_code=code)
                if item['audit']:
                    if code!=0:
                        rt.write(root/'audit_failure.json',dict(exit_code=code,log=str(root/'audit.log'),time=time.time()));rt.k32.SetEvent(attention)
                    continue
                state=rt.read(root/'status.json') if (root/'status.json').exists() else {}
                if state.get('status')=='RESOURCE_YIELDED':continue
                if state.get('status') not in TERMINAL:
                    rt.write(root/'status.json',dict(status='FAILED',reason='UNEXPECTED_CHILD_EXIT',exit_code=code,previous=state,updated=time.time()))
                if code!=0 or state.get('status')=='FAILED':rt.k32.SetEvent(attention)
                # Queue an audit marker; reserve memory before starting it.
                rt.write(root/'audit_pending.json',dict(exit_code=code,time=time.time()))
            now=rt.resources();snapshots=[]
            for tag,item in live.items():
                try:snap=rt.process(item['child'].pid,item['created'])
                except OSError:
                    if item['child'].poll() is not None:continue
                    raise
                snap['peak_gib']=max(snap['peak_gib'],item['peak']);snapshots.append(snap)
            if now['ram_percent']>=90 or (now['cpu_percent'] is not None and now['cpu_percent']>=90):
                if not paused:event('PAUSE_DISPATCH',resources=now)
                paused=True
                # One safe-boundary yield at a time, then reassess actual resources.
                candidates=[(t,v) for t,v in live.items() if now['ram_percent']>=90 and not v['audit'] and not (RUN/'work/jobs'/t/'yield.request').exists()]
                if candidates:
                    tag,item=max(candidates,key=lambda v:v[1]['peak']);rt.write(RUN/'work/jobs'/tag/'yield.request',dict(reason='RAM_90',resources=now))
                    event('SAFE_YIELD_REQUEST',tag=tag,resources=now)
            elif now['ram_percent']<85 and now['cpu_percent'] is not None and now['cpu_percent']<85:
                if paused:event('RESUME_DISPATCH',resources=now)
                paused=False
            formal_unfinished=any(j.get('priority',0)==0 and not ((RUN/'outputs'/j['tag']/'audit.json').exists() or (RUN/'work/jobs'/j['tag']/'audit_failure.json').exists()) for j in jobs)
            pending=[]
            for job in jobs:
                tag=job['tag'];root=RUN/'work/jobs'/tag
                if tag in live:continue
                state=rt.read(root/'status.json') if (root/'status.json').exists() else {}
                done=(RUN/'outputs'/tag/'audit.json').exists() or (root/'audit_failure.json').exists()
                if state.get('status') in TERMINAL and not done:pending.append((0,job,True));continue
                if state.get('status') in TERMINAL:continue
                if time.time()>=campaign['training_deadline']:
                    if not state:rt.write(root/'status.json',dict(status='NOT_STARTED_DEADLINE',updated=time.time()))
                    else:
                        state['status']='BUDGET_STOPPED';state['reason']='CAMPAIGN_DEADLINE_WHILE_YIELDED';rt.write(root/'status.json',state)
                    continue
                if job.get('priority',0)>0 and formal_unfinished:continue
                deps=[RUN/'outputs'/t/'audit.json' for t in job.get('dependencies',[])]
                if deps:
                    if not all(p.exists() or (RUN/'work/jobs'/t/'audit_failure.json').exists() for p,t in zip(deps,job['dependencies'])):continue
                    dep_results=[rt.read(p) for p in deps if p.exists()]
                    if job.get('conditional'):
                        sufficient=all(a.get('numerical_sufficient') and a.get('physical_reasonable') for a in dep_results)
                        values=[a.get('objective',float('inf')) for a in dep_results]
                        gap=(max(values)-min(values))/max(abs(min(values)),1e-30) if sufficient else None
                        if not sufficient or gap<.01:
                            rt.write(root/'status.json',dict(status='SKIPPED_CONDITION',reason='PRIMARY_NOT_BOTH_SUFFICIENT' if not sufficient else 'MAP_GAP_BELOW_1_PERCENT',relative_gap=gap,updated=time.time()));continue
                        rt.write(root/'conditional_admission.json',dict(relative_gap=gap,threshold=.01,training_only=True))
                    if not any(a['status']=='AUDITED_FIT' and a.get('physical_reasonable') is True for a in dep_results):
                        rt.write(root/'status.json',dict(status='DEPENDENCY_FAILED',reason='No baseline legal point; no formula replacement',updated=time.time()));continue
                pending.append((1,job,False))
            pending.sort(key=lambda v:(v[0],v[1].get('priority',0)))
            if (RUN/'work/maintenance.request').exists():
                paused=True
                for tag,item in live.items():
                    if not item['audit'] and not (RUN/'work/jobs'/tag/'yield.request').exists():
                        rt.write(RUN/'work/jobs'/tag/'yield.request',dict(reason='CONTROLLED_MAINTENANCE'))
            if not paused:
                for _,job,is_audit in pending:
                    peak=proof['peak_reservations_gib'][job['kind']]
                    if not rt.admission(now,peak,snapshots):continue
                    req=RUN/'work/jobs'/job['tag']/'yield.request'
                    if req.exists():req.unlink()
                    launch(job,is_audit)
                    # Reserve full peak immediately, even before native RSS grows.
                    snapshots.append(dict(peak_gib=peak,rss_gib=0.))
            states={j['tag']:(rt.read(RUN/'work/jobs'/j['tag']/'status.json').get('status') if (RUN/'work/jobs'/j['tag']/'status.json').exists() else 'PENDING') for j in jobs}
            audited=sum((RUN/'outputs'/j['tag']/'audit.json').exists() or (RUN/'work/jobs'/j['tag']/'audit_failure.json').exists() for j in jobs)
            if time.monotonic()-lastupdate>10:
                event('RESOURCE_SAMPLE',resources=now,processes=snapshots,live=[dict(tag=t,pid=v['child'].pid,audit=v['audit']) for t,v in live.items()],paused=paused,audited=audited)
                rt.write(RUN/'work/controller_status.json',dict(status='RUNNING',process=rt.process(os.getpid()),resources=now,live=[dict(tag=t,pid=v['child'].pid,audit=v['audit'],created=v['created']) for t,v in live.items()],states=states,audited=audited,updated=time.time()))
                lastupdate=time.monotonic()
            if not live and all(s in TERMINAL for s in states.values()) and audited==len(jobs):break
            # Process handles wake on exit. The controller itself does not send messages.
            handles=[int(v['child']._handle) for v in live.values()]
            if handles:
                array=(ctypes.c_void_p*len(handles))(*handles)
                rt.k32.WaitForMultipleObjects.argtypes=[ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.c_ulong]
                rt.k32.WaitForMultipleObjects(len(handles),array,False,2000)
            else:time.sleep(2)
        log=(RUN/'work/report.log').open('ab')
        report=subprocess.run([sys.executable,'-B',str(RUN/'scripts/finalize_daily.py')],cwd=RUN,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        log.close()
        status='COMPLETE_EXPERIMENTAL' if report.returncode==0 else 'FINAL_AUDIT_FAILED'
        rt.write(RUN/'work/controller_status.json',dict(status=status,exit_code=report.returncode,updated=time.time(),process=rt.process(os.getpid())))
        event('CONTROLLER_TERMINAL',status=status,exit_code=report.returncode)
        rt.k32.SetEvent(attention)

if __name__=='__main__':
    try:main()
    except Exception:
        rt.write(RUN/'work/controller_failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise

