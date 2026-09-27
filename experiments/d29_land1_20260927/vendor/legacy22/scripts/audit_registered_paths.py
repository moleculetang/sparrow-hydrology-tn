"""Finite path, immutable fit identity, checkpoint and dispatch accounting."""
import json,time
import native_runtime as rt
R=rt.RUN

def main():
    jobs=rt.read(R/'configs/jobs.json');assert len(jobs)==28 and len({j['tag'] for j in jobs})==28
    dims={'SOURCE_UNIFIED':31,'REGIONAL_L1':39,'REGIONAL_L3':55,'REGIONAL_N3':79};rows=[]
    for fold in rt.read(R/'configs/folds.json'):
        for p,h in rt.read(R/'reports/launch_by_fold'/f'{fold}.json')['frozen_hashes'].items():assert rt.sha(R/p)==h,(fold,p)
    terminal={'NUMERICALLY_SUFFICIENT','NUMERICALLY_INSUFFICIENT_STATIONARY','BUDGET_STOPPED','FAILED','NOT_STARTED_DEADLINE','DEPENDENCY_FAILED','SKIPPED_CONDITION'}
    for j in jobs:
        root=R/'work/jobs'/j['tag'];status=rt.read(root/'status.json');assert status['status'] in terminal,(j['tag'],status['status'])
        if j['tag'].startswith('S'):
            assert not j.get('reuse_from') and all(t.split('_')[0]==j['tag'].split('_')[0] for t in j['nested_tags'])
        ref=root/'checkpoints/latest.json';calls=status.get('calls',0);active=status.get('active_seconds',0)
        if ref.exists():
            identity=rt.read(ref)['identity'];state=rt.restore(root/'checkpoints',identity);assert state['calls']==calls and state['active_seconds']==active
            assert calls<=8000,(j['tag'],calls)
            if state.get('best'):assert len(state['best']['x'])==dims[j['kind']]
        rows.append(dict(tag=j['tag'],status=status['status'],reused=bool(j.get('reuse_from')),fit_calls=calls,active_seconds=active))
    events=[]
    for line in (R/'work/resource_events.jsonl').read_text(encoding='utf-8').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:raise AssertionError('BROKEN_APPEND_LOG')
    dispatch={e['tag'] for e in events if e.get('event')=='DISPATCH' and not e.get('audit')}
    registered={j['tag'] for j in jobs if not j.get('reuse_from')};assert dispatch<=registered and len(dispatch)<=24
    check=rt.read(R/'work/experiment_clock.json');rt.write(R/'reports/registered_path_audit.json',dict(status='PASS',logical_paths=len(jobs),distinct_new_fit_paths=len(dispatch),reused_paths=sum(bool(j.get('reuse_from')) for j in jobs),rows=rows,source_identities_unchanged=True,elapsed_hours=(time.time()-check['started'])/3600,resource_events=len(events),completed_exit_audits=sum((R/'outputs'/j['tag']/'audit.json').exists() for j in jobs)))
    print('PASS registered paths',len(dispatch),flush=True)
if __name__=='__main__':main()
