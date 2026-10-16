"""Bounded local staged replays, preserving all live training and final gates."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from d29_platform.runtime import write_json,sha
from host_resources import registry,registry_root
from controller_lock import acquire
from staged_frozen_sources import resolve


def main():
    if os.name!='nt':raise RuntimeError('LOCAL_PASSIVE_STAGE_REQUIRED')
    ids=sys.argv[1:]
    bnf=bool(ids and ids[0]=='--bnf')
    if bnf:ids=ids[1:]
    kind='bnf' if bnf else 'internal'
    upstream='trace_approved_bnf_history.py' if bnf else 'trace_approved_internal_fluxes.py'
    worker='trace_staged_bnf_history.py' if bnf else 'trace_staged_internal_fluxes.py'
    proof_name='staged_bnf_derivation.json' if bnf else 'staged_trace_derivation.json'
    namespace='bnf_trace_approved_staged_v1' if bnf else 'internal_trace_approved_staged_v1'
    if set(ids)!={'TIM_F23_C0_V0_s0','TIM_F23_C0_V1_s0'} or len(ids)!=2:raise RuntimeError('EXACT_ADDITIVE_TWO_FROZEN_BASELINES_REQUIRED')
    proof=json.loads((ROOT/'evidence'/proof_name).read_text())
    if (not proof.get('passed') or proof['upstream_sha256']!=sha(ROOT/'analysis'/upstream) or
        proof['staged_worker_sha256']!=sha(ROOT/'analysis'/worker)):
        raise RuntimeError('UNCHANGED_DERIVED_WORKER_REQUIRED')
    sources={jid:resolve(jid) for jid in ids}
    out=ROOT/'outputs'/('staged_bnf_trace_002' if bnf else 'staged_internal_trace_002');out.mkdir(exist_ok=True)
    lifetime=acquire(out/'lifetime.lock')
    if (out/'complete.json').exists():raise RuntimeError('STAGED_CONTROLLER_NO_RERUN')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(out/'owner.json',owner)
    queue=list(ids);running={};events=[];completed=[]
    while queue or running:
        for jid,item in list(running.items()):
            child=item['child'];memory=registry.process_memory(child.pid)
            for key in item['peak']:
                if memory.get(key) is not None:item['peak'][key]=max(item['peak'][key],memory[key])
            if child.poll() is None:continue
            item['log'].close();registry.release(item['lease']['token'],root=registry_root())
            event=dict(job=jid,exit_code=child.returncode,peak=item['peak']);events.append(event)
            del running[jid]
            if child.returncode:
                write_json(out/'failure.json',event)
                queue.clear()
            else:
                p=ROOT/'outputs/jobs'/(namespace+'_'+jid)/'complete.json'
                v=json.loads(p.read_text())
                if (v.get('passed') is not True or v['mode']!='staged' or
                    (v.get('evaluation_labels_read') is not False if bnf else v.get('no_evaluation_outcomes_read') is not True) or
                    v.get('new_training_calls' if bnf else 'new_scientific_training_calls')!=0 or
                    v['source_folder']!=str(sources[jid].resolve()) or
                    v.get('worker_sha256' if bnf else 'source_sha256')!=proof['staged_worker_sha256']):
                    raise RuntimeError('STAGED_COMPLETION_BINDING_REQUIRED')
                jobs=json.loads((ROOT/'config/jobs.json').read_text())
                job=next(j for j in jobs if j['id']==jid)
                if bnf:
                    from run_approved_bnf_delivery import verified_case
                else:
                    from run_approved_internal_trace_delivery import verified_case
                verified_case(p.parent,sources[jid],job,'staged')
                completed.append(dict(job=jid,complete=str(p),sha256=sha(p)))
        if not (out/'failure.json').exists():
            while queue and len(running)<3:
                jid=queue[0]
                lease=registry.acquire('20261008_1','analysis/'+namespace+'/'+jid,
                    threads=1,peak_bytes=6500000000,root=registry_root())
                write_json(out/'resource_decision.json',lease)
                if not lease.get('granted'):break
                queue.pop(0);log=(out/(jid+'.log')).open('ab')
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                        lease['token'],'analysis/'+worker,jid,'staged'],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(lease['token'],child.pid,root=registry_root())
                except Exception:
                    log.close();registry.release(lease['token'],root=registry_root());raise
                running[jid]=dict(child=child,log=log,lease=lease,peak=dict(rss=0,private_commit=0,peak_commit=0))
        write_json(out/'status.json',dict(owner=owner,phase='passive_staged_physics_running',
            queue=queue,running=[dict(job=jid,pid=r['child'].pid,created=registry.identity(r['child'].pid),
                lease=r['lease'],peak=r['peak']) for jid,r in running.items()],completed=completed,events=events,
            max_active=3,evaluation_labels_read=False,training_ownership_changed=False))
        if queue or running:time.sleep(2)
    passed=len(completed)==len(ids)
    write_json(out/'complete.json',dict(passed=passed,partial_only=True,completed=completed,events=events,
        evaluation_admitted=False,full44_completed_claimed=False,training_ownership_changed=False,
        worker_sha256=proof['staged_worker_sha256']))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
