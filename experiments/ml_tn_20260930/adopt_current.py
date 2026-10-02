"""Adopt existing identified ML workers without interrupting model computation."""
import os,time,json,subprocess,datetime
from pathlib import Path
from mltn.common import ROOT,read,write
from mltn.resources import registry
def main():
    mod,state=registry()
    if mod is None:raise RuntimeError('REGISTRY_NOT_YET_PUBLISHED')
    continuation=ROOT/'outputs/remaining_owner.json'
    if continuation.exists() and mod.identity(read(continuation)['pid']) is not None:
        write(ROOT/'outputs/adoption_retired.json',dict(reason='modern identified continuation owns priority and worker leases',priority_written=False,leases_changed=False))
        return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0].strip();adopted=[]
    while True:
        p=ROOT/'outputs/controller_status.json'
        if not p.exists():time.sleep(5);continue
        status=read(p);leases=mod.status(state)['leases']
        if continuation.exists() and mod.identity(read(continuation)['pid']) is not None:
            write(ROOT/'outputs/adoption_retired.json',dict(reason='modern identified continuation owns priority and worker leases',priority_written=False,leases_changed=False))
            return
        for jid,j in status['active'].items():
            pid=j['pid'];identity=mod.identity(pid)
            if identity is None:continue
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
            except FileNotFoundError:continue
            if str(ROOT/'train.py') not in cmd:
                if mod.identity(pid) is None:continue
                write(ROOT/'outputs/resource_command_mismatch.json',dict(pid=pid,job=jid,observed_command=cmd,action='not adopted; recheck at next event boundary'))
                continue
            if any(v['project']=='20260930_1' and v['job']==jid for v in leases.values()):continue
            n=1 if j['gpu'] else read(ROOT/'config/execution.json')['tree_threads'];used={c for v in leases.values() for c in v['cpus']};free=[c for c in mod.physical_cores() if c not in used];cpus=free[:n]
            if len(cpus)!=n:raise RuntimeError('INSUFFICIENT_CORES_FOR_EXISTING_OWNER')
            grant=mod.acquire('20260930_1',jid,n,4*1024**3 if j['gpu'] else max(2*1024**3,mod.rss(pid)),pid=pid,gpu=gpu if j['gpu'] else None,root=state,adopt_cpus=cpus)
            if not grant.get('granted'):continue
            if mod.identity(pid)!=identity:continue
            os.sched_setaffinity(pid,cpus);adopted.append(grant);leases[grant['token']]=grant
        write(ROOT/'outputs/shared_resource_adoption.json',dict(adopted=adopted,live=mod.status(state)['leases'],time=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy='unique physical CPU sets; active computation retained'))
        # ML CPU priority applies to pending CPU jobs, while GPU-only queue allows graybox CPU use.
        jobsfile=ROOT/'config'/f'{status["phase"]}_jobs.json';count=0
        if jobsfile.exists():
            for job in read(jobsfile):
                if job['family'] not in ['MLP','LSTM','GRU','TCN','Transformer','GraphTCN']:
                    from controller import jobid
                    folder=ROOT/'jobs'/jobid(job)
                    if jobid(job) not in status['active'] and not (folder/'result.json').exists() and not (folder/'failure.json').exists():count+=1
        live=mod.status(state)['leases'];leased={v['job'] for v in live.values() if v['project']=='20260930_1'}
        cpu_active=sum(not j['gpu'] for j in status['active'].values())
        asking=sum(not j['gpu'] and name not in leased for name,j in status['active'].items())
        ready=min(max(0,read(ROOT/'config/execution.json')['cpu_jobs']-cpu_active),count)
        mod.pending('20260930_1',ready+asking,state)
        controller=read(ROOT/'outputs/controller_owner.json')
        if mod.identity(controller['pid']) is None:mod.pending('20260930_1',0,state);break
        time.sleep(5)
if __name__=='__main__':main()
