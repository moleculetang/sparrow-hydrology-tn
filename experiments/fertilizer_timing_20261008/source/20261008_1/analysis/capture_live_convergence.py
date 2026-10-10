"""Explicit read-only dual-host snapshot; never invokes a physical oracle."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'deployment')]
from inspect_live_convergence import inspect
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def main():
    local = inspect(ROOT)
    source = (ROOT / 'analysis/inspect_live_convergence.py').read_text(encoding='utf8')
    code = (
        'import json\n'
        '_ns={"__name__":"readonly_convergence_probe"}\n'
        f'exec(compile({source!r},"readonly_convergence_probe","exec"),_ns)\n'
        f'print(json.dumps(_ns["inspect"]({REMOTE_ROOT!r}),allow_nan=False))\n'
    )
    result = subprocess.run(
        ['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON + ' -X utf8 -B -'],
        input=code, text=True, encoding='utf8', capture_output=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    remote = json.loads(result.stdout)
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    folder = ROOT / 'outputs/live_convergence_review' / stamp
    folder.mkdir(exist_ok=False, parents=True)
    snapshot = dict(local=local, remote=remote, physical_oracle_calls=0,
                    evaluation_labels_read=False, training_modified=False)
    (folder / 'snapshot.json').write_text(
        json.dumps(snapshot, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(snapshot=str(folder / 'snapshot.json'), records=[
        {k: r[k] for k in ['job', 'status', 'checkpoint_calls', 'objective_best',
            'PG_best_recomputed', 'classification', 'windows', 'native_iterations',
            'native_done', 'completed_audit_coordinates', 'current_probe',
            'failed_trial_present', 'max_recorded_local_balance_kg',
            'canonical_bound_encoding_passed', 'all_recorded_oracles_finite']}
        for host in [local, remote] for r in host['records']
    ]), allow_nan=False))


if __name__ == '__main__':
    main()
