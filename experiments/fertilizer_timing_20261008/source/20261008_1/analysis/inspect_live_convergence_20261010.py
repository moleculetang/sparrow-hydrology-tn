"""Read-only trajectory review; does not change workers or read evaluation labels."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deployment'))
from wait_prerequisites import OPTIONS, REMOTE_PYTHON

READER = r'''
import json, math, pickle
from pathlib import Path
def inspect(root, approved_k=False):
    result=[]
    for folder in sorted((Path(root)/'outputs/jobs').glob('TIM_*')):
        status_path=folder/'status.json'
        if not status_path.exists(): continue
        # Original-root K files are superseded, sealed evidence, not current jobs.
        if ('_K_' in folder.name) != approved_k: continue
        status=json.loads(status_path.read_text())
        path=folder/'trajectory.jsonl'
        rows=[]
        if path.exists():
            for line in path.read_bytes().splitlines(keepends=True):
                if line.endswith(b'\n'): rows.append(json.loads(line))
        # Only committed calls; unfinished writes/trials are not treated as stops.
        rows=[r for r in rows if r['calls']<=status['calls']]
        objectives=[r['objective'] for r in rows]
        out=dict(job=folder.name,status=status['status'],calls=status['calls'],
            J=status['best_objective'],PG=status['projected_gradient'],
            source_multiplier=status.get('source_multiplier'),
            stop_reason=status.get('stop_reason'),committed_rows=len(rows))
        if rows:
            out.update(first_J=objectives[0],first_PG=rows[0]['projected_gradient'],
                total_best_decrease=objectives[0]-min(objectives),
                max_local_balance=max(r.get('local_balance_kg',0.) for r in rows),
                minimum_year_start_state=min(r.get('minimum_year_start_state_kg',0.) for r in rows),
                invalid_rows=sum(not math.isfinite(r['objective']) or not math.isfinite(r['projected_gradient']) for r in rows),
                duplicate_calls=len(rows)-len({r['calls'] for r in rows}),
                recent_20_pg_range=[min(r['projected_gradient'] for r in rows[-20:]),max(r['projected_gradient'] for r in rows[-20:])])
            for n in (20,50):
                if len(rows)>n:
                    out[f'best_decrease_last_{n}']=min(objectives[:-n])-min(objectives)
            out['PG_met_first_call']=next((r['calls'] for r in rows if r['projected_gradient']<=1e-5),None)
            out['last_5']=[{k:r.get(k) for k in ('calls','objective','projected_gradient','seconds')} for r in rows[-5:]]
        p=folder/'optimizer.pkl'
        if p.exists():
            saved=pickle.loads(p.read_bytes());s=saved['solver']
            out['native']=dict(pgtol=float(s['pgtol']),factr=float(s['factr']),
                task=str(s.get('task')),done=s.get('done'),checkpoint_calls=saved['calls'],
                checkpoint_status=saved['status'])
        for name in ('solution_audit_progress.json','audit_current_probe.json','independent_audit.json'):
            p=folder/name
            if p.exists():
                a=json.loads(p.read_text());checks=a.get('checks',[])
                out[name]=dict(completed=len(checks),coordinate=a.get('coordinate'),calls=a.get('calls'),
                    result_status=a.get('result',{}).get('status'),
                    coordinate_statuses=[dict(coordinate=c.get('coordinate'),classification=c.get('classification'),derivative_qualified=c.get('derivative_qualified')) for c in checks],
                    numerically_sufficient=a.get('numerically_sufficient'))
        result.append(out)
    return result
'''

def main():
    local={}
    exec(READER,local)
    remote_root='/home/pc/TWY/SPARROW/Test/20261008_1'
    code=READER+'\nprint(json.dumps(inspect('+repr(remote_root)+')+inspect('+repr(remote_root+'_maize_scope_v2')+', approved_k=True)))\n'
    run=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
    if run.returncode: raise RuntimeError(run.stderr+run.stdout)
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),
        local=local['inspect'](ROOT)+local['inspect'](ROOT.with_name('20261008_1_maize_scope_v2'),approved_k=True),remote=json.loads(run.stdout),
        superseded_original_K_excluded=True,
        read_only=True,evaluation_labels_read=False,worker_changes=0)
    folder=ROOT/'outputs/live_convergence_review';folder.mkdir(exist_ok=True)
    target=folder/(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
    with target.open('x',encoding='utf8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(receipt=str(target),**result)))

if __name__=='__main__': main()
