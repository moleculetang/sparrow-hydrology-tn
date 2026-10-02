"""Recoverable event-driven controller; no cron and no repeated external dispatch."""
import os,sys,time,subprocess,datetime,traceback,platform
from pathlib import Path
from mltn.common import ROOT,read,write
NEURAL=['MLP','LSTM','GRU','TCN','Transformer','GraphTCN']
def memory_percent():
    if os.name=='posix':
        d={a.split(':')[0]:int(a.split()[1]) for a in Path('/proc/meminfo').read_text().splitlines() if len(a.split())>1 and a.split()[1].isdigit()}
        physical=100*(1-d['MemAvailable']/d['MemTotal']);commit=100*d.get('Committed_AS',0)/d.get('CommitLimit',max(d['MemTotal'],1));return physical,commit
    import ctypes
    class MEM(ctypes.Structure):_fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),*[(n,ctypes.c_ulonglong) for n in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]]
    m=MEM();m.dwLength=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.dwMemoryLoad,100*(1-m.ullAvailPageFile/m.ullTotalPageFile)
def cpu_percent():
    if os.name=='posix':
        vals=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]];now=(sum(vals),vals[3]+vals[4])
    else:
        from mltn.resources import registry
        mod,_=registry()
        if mod is None:raise RuntimeError('WINDOWS_CPU_PROBE_REQUIRED')
        now=mod.cpu_sample()
    prev=getattr(cpu_percent,'previous',now);cpu_percent.previous=now
    return 100*(1-(now[1]-prev[1])/max(1,now[0]-prev[0]))
def gpu_percent():
    try:
        r=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0].split(',')
        return 100*float(r[0])/float(r[1])
    except Exception:return 100.
def jobid(j):
    prefix=('joint_' if j.get('role')=='joint' else 'aux_' if j.get('role')=='auxiliary' else '')+('fixed_recipe_' if j.get('fixed_recipe') else '')
    core=f'{j["stage"]}_{j["family"]}' if j.get('role')=='joint' else f'{j["stage"]}_{j["task"]}_{j["family"]}'
    return prefix+core+f'_c{j["config"]}_s{j["seed"]}'+('' if j.get('block') is None else f'_B{j["block"]}')+('' if j.get('role')!='auxiliary' else f'_lead{j["lead"]}')

def ready_for_dispatch(j,external,hold=False):
    jid=jobid(j);folder=ROOT/'jobs'/jid
    if hold or (folder/'result.json').exists() or (folder/'failure.json').exists():return False
    if jid in external and external[jid]['host']!=platform.node():return False
    if j['stage']=='S24' and j.get('block') is not None and not (ROOT/'jobs'/jobid(j|{'stage':'S23'})/'result.json').exists():return False
    return True
def immediate_pending_count(pending,active,cfg,external,leased):
    """Ready CPU and GPU jobs share host RAM; distant/foreign jobs get no priority."""
    hold=cfg.get('dispatch_hold',False)
    cpu_free=max(0,cfg['cpu_jobs']-sum(not a['gpu'] for a in active.values()))
    gpu_free=max(0,cfg['gpu_jobs']-sum(a['gpu'] for a in active.values()))
    cpu_ready=min(cpu_free,sum(j['family'] not in NEURAL and ready_for_dispatch(j,external,hold) for j in pending))
    gpu_ready=min(gpu_free,sum(j['family'] in NEURAL and ready_for_dispatch(j,external,hold) for j in pending))
    waiting=sum(k not in leased for k in active)
    return cpu_ready+gpu_ready+waiting
def execute(jobs,phase):
    cfg=read(ROOT/'config/execution.json');active={};failed=[];paused=False;pending=list(jobs);logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    write(ROOT/'config'/f'{phase}_jobs.json',jobs)
    while pending or active:
        cfg=read(ROOT/'config/execution.json')
        external_file=ROOT/'config/external_ownership.json'
        external=read(external_file).get('jobs',{}) if external_file.exists() else {}
        external_waiting=[]
        dependency_waiting=[]
        from mltn.resources import registry
        resource_mod,resource_root=registry()
        priority_count=0;priority_unleased=[]
        if resource_mod is not None:
            leased={v['job'] for v in resource_mod.status(resource_root)['leases'].values() if v['project']=='20260930_1'}
            priority_count=immediate_pending_count(pending,active,cfg,external,leased)
            priority_unleased=[dict(job=k,pid=v['process'].pid) for k,v in active.items() if k not in leased and v['process'].poll() is None]
            resource_mod.pending('20260930_1',priority_count,resource_root)
        physical,commit=memory_percent();cpu=cpu_percent();gpu_mem=gpu_percent();paused=(max(physical,commit,cpu,gpu_mem)>=90) or (paused and max(physical,commit,cpu,gpu_mem)>=85)
        now=datetime.datetime.now(datetime.timezone.utc);stop=datetime.datetime.fromisoformat(read(ROOT/'study.json')['stop_new_dispatch_utc'].replace('Z','+00:00'))
        for j in list(pending):
            jid=jobid(j);folder=ROOT/'jobs'/jid
            if (folder/'result.json').exists():pending.remove(j);continue
            if (folder/'failure.json').exists():failed.append(jid);pending.remove(j);continue
            if jid in external and external[jid]['host']!=platform.node():
                external_waiting.append(jid);continue
            if cfg.get('dispatch_hold',False):continue
            if j['stage']=='S24' and j.get('block') is not None:
                parent=ROOT/'jobs'/jobid(j|{'stage':'S23'})
                if not (parent/'result.json').exists():
                    dependency_waiting.append(jid);continue
            if paused or now>=stop:break
            gpu=j['family'] in NEURAL;limit=cfg['gpu_jobs'] if gpu else cfg['cpu_jobs'];used=sum(k['gpu']==gpu for k in active.values())
            if used>=limit:continue
            if (folder/'owner.lock').exists():
                from mltn.ownership import reclaim
                reclaim(folder)
            script='joint.py' if j.get('role')=='joint' else 'auxiliary.py' if j.get('role')=='auxiliary' else 'train.py'
            cmd=[sys.executable,'-B',str(ROOT/script)]
            argnames=['family','stage','config','seed','block'] if j.get('role')=='joint' else ['family','task','stage','config','seed','block','lead']
            for name in argnames:
                if j.get(name) is not None:cmd.extend(['--'+name,str(j[name])])
            if j.get('fixed_recipe'):cmd.append('--fixed-recipe')
            threads=cfg['neural_threads'] if gpu else cfg['tree_threads'];cmd+=['--device','cuda' if gpu else 'cpu','--threads',str(threads)]
            env=os.environ.copy();env['ML_THREADS']=str(threads);env['PYTHONIOENCODING']='utf-8';log=(logs/(jid+'.log')).open('ab')
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);active[jid]=dict(process=p,log=log,gpu=gpu,created=now.isoformat(),job=j);pending.remove(j);time.sleep(cfg['stagger_seconds'])
        write(ROOT/'outputs/controller_status.json',dict(phase=phase,created=now.isoformat(),pending=len(pending),external_waiting=external_waiting,dependency_waiting=dependency_waiting,dispatch_hold=cfg.get('dispatch_hold',False),active={k:{'pid':v['process'].pid,'gpu':v['gpu'],'created':v['created']} for k,v in active.items()},priority_requested=priority_count,priority_unleased_active=priority_unleased,failed=failed,physical_percent=physical,commit_percent=commit,cpu_percent=cpu,gpu_memory_percent=gpu_mem,paused=paused))
        if now>=stop and pending and not active:break
        if not active and pending and not paused and not external_waiting and not dependency_waiting and not cfg.get('dispatch_hold',False):raise RuntimeError('NO_DISPATCH_PROGRESS')
        if active:
            time.sleep(5)
            for jid,v in list(active.items()):
                code=v['process'].poll()
                if code is not None:
                    v['log'].close()
                    if code:
                        folder=ROOT/'jobs'/jid;write(folder/'failure.json',dict(exit_code=code,log='logs/'+jid+'.log',owner=platform.node()));failed.append(jid)
                    del active[jid]
        elif paused or external_waiting or dependency_waiting or cfg.get('dispatch_hold',False):time.sleep(5)
    write(ROOT/'outputs'/f'{phase}_receipt.json',dict(completed=[jobid(j) for j in jobs if (ROOT/'jobs'/jobid(j)/'result.json').exists()],failed=failed,undispatched=[jobid(j) for j in pending]))
    if resource_mod is not None:resource_mod.pending('20260930_1',0,resource_root)
    return failed
def screen_jobs():
    des=read(ROOT/'config/design.json');return [dict(family=f,task=t,config=c['id'],seed=1729,stage='screen') for f in des['families'] for t in des['direct_tasks'] for c in des['configs'][f]]
def select():
    des=read(ROOT/'config/design.json');selected={};scores=[]
    for f in des['families']:
        for task in des['direct_tasks']:
            rows=[]
            for c in des['configs'][f]:
                j=dict(family=f,task=task,config=c['id'],seed=1729,stage='screen');p=ROOT/'jobs'/jobid(j)/'result.json'
                if p.exists():
                    r=read(p);rows.append((r['selection_score'],c['id']));scores.append(dict(family=f,task=task,config=c['id'],selection_score=r['selection_score']))
            if rows:selected[f+'_'+task]=min(rows)[1]
    if not selected:raise RuntimeError('NO_VALID_SCREEN')
    winners={}
    for task in des['direct_tasks']:
        rows=[r for r in scores if selected.get(r['family']+'_'+task)==r['config']];winners[task]=min(rows,key=lambda r:(r['selection_score'],r['family']))['family']
    routes={}
    for name,families in [('tree',['XGBoost','LightGBM']),('sequence',['LSTM','GRU','TCN','Transformer']),('graph',['GraphTCN'])]:
        rows=[r for r in scores if r['task']=='daily' and r['family'] in families and selected.get(r['family']+'_daily')==r['config']]
        if rows:routes[name]=min(rows,key=lambda r:(r['selection_score'],r['family']))['family']
    out=dict(selected=selected,winners=winners,joint_routes=routes,selection_support='2022 Jan-Sep; no 2023/2024 access',scores=scores);write(ROOT/'outputs/frozen_selection.json',out);return out
def final_jobs(s):
    jobs=[]
    for key,c in s['selected'].items():
        family,task=key.rsplit('_',1)
        for stage in ['F23','F24']:jobs.append(dict(family=family,task=task,config=c,seed=1729,stage=stage))
    for task,family in s['winners'].items():
        c=s['selected'][family+'_'+task]
        for seed in [1730,1731]:
            for stage in ['F23','F24']:jobs.append(dict(family=family,task=task,config=c,seed=seed,stage=stage))
        for block in [56,113,191]:
            for stage in ['S23','S24']:
                for seed in [1729,1730,1731]:jobs.append(dict(family=family,task=task,config=0,seed=seed,stage=stage,block=block))
    return jobs
if __name__=='__main__':
    if os.name=='posix':
        # Select one logical CPU for each of at most forty physical cores.
        import json
        rows=json.loads(subprocess.check_output(['lscpu','--json','--extended=CPU,CORE,SOCKET'],text=True))['cpus'];chosen={}
        for row in rows:chosen.setdefault((row['socket'],row['core']),int(row['cpu']))
        os.sched_setaffinity(0,list(chosen.values())[:40])
    if not read(ROOT/'evidence'/f'acceptance_{platform.system()}.json')['passed']:raise SystemExit('ACCEPTANCE_FAILED')
    write(ROOT/'outputs/controller_owner.json',dict(pid=os.getpid(),host=platform.node(),start=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    try:
        execute(screen_jobs(),'screen');s=select();execute(final_jobs(s),'direct_final')
        write(ROOT/'outputs/direct_matrix_done.json',dict(status='direct_training_complete',remaining=['joint models','historical TN auxiliary','ensemble','unified validation','expert reports']))
    except Exception:traceback.print_exc();raise
