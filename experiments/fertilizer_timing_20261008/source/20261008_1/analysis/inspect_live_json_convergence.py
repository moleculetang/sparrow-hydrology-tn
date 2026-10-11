"""Read committed JSON trajectories and audits; never deserialize live checkpoints."""
import datetime
import json
import subprocess
from pathlib import Path
from inspect_live_convergence_20261010 import ROOT, READER, OPTIONS, REMOTE_PYTHON


def main():
    start = READER.index("        p=folder/'optimizer.pkl'")
    end = READER.index("        for name in", start)
    reader = READER[:start] + READER[end:]
    namespace = {}
    exec(reader, namespace)
    remote_root = '/home/pc/TWY/SPARROW/Test/20261008_1'
    code = reader + '\nroot=Path(' + repr(remote_root) + ')\n'
    code += "controller=json.loads((root/'outputs/campaign_workstation/status.json').read_text())\n"
    code += "print(json.dumps(dict(paths=inspect(root)+inspect(str(root)+'_maize_scope_v2',approved_k=True),controller=controller)))\n"
    run = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                         input=code, text=True, encoding='utf8', capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr + run.stdout)
    result = dict(at=datetime.datetime.now().astimezone().isoformat(),
                  local=namespace['inspect'](ROOT)+namespace['inspect'](ROOT.with_name('20261008_1_maize_scope_v2'),approved_k=True),
                  remote=json.loads(run.stdout), read_only=True, checkpoint_deserialization=False,
                  evaluation_labels_read=False, worker_changes=0, superseded_original_K_excluded=True)
    target = ROOT/'outputs/live_convergence_review'/('json_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
    with target.open('x',encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(receipt=str(target),at=result['at'])))
    for item in result['local']+result['remote']['paths']:
        print(json.dumps({key:item.get(key) for key in ('job','status','calls','J','PG','first_J','first_PG','best_decrease_last_20','max_local_balance','invalid_rows','duplicate_calls','source_multiplier','solution_audit_progress.json','audit_current_probe.json','independent_audit.json')}))
    controller=result['remote']['controller']
    print(json.dumps(dict(controller_phase=controller.get('phase'),running=[{k:r.get(k) for k in ('job','stage','pid')} for r in controller.get('running',[])],reviews=controller.get('reviews'),completed=controller.get('completed'))))


if __name__ == '__main__':
    main()
