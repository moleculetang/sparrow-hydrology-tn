"""One-shot local evidence finalization after a byte-verified remote import.

No training, selection or GPU jobs are started here. All descriptive references
and independent metric checks remain separate from the frozen training choices.
"""
import datetime
import os
import subprocess
import sys
import time

from mltn.common import ROOT, read, sha, write
from mltn.resources import lease


def main():
    recovery = read(ROOT / 'evidence/results_recovery.json')
    assert recovery.get('passed'), 'REMOTE_RESULT_IMPORT_NOT_VERIFIED'
    assert read(ROOT / 'evidence/historical_append_acceptance.json')['passed']
    assert read(ROOT / 'outputs/postprocess_done.json')['status'] == 'computational_postprocess_done'
    lease('local_final_evidence', 1, 'cpu')
    stages = [
        'references.py', 'freeze_results.py', 'repair_constant_HF_means.py',
        'append_historical_comparators.py',
        'training_metrics.py', 'diagnostics.py', 'independent_review.py',
        'training_seedmeans.py', 'expert_tables.py', 'audit_effect_metrics.py',
        'audit_final_protocol.py',
        'report_draft.py',
    ]
    records = []
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    write(ROOT / 'outputs/local_final_evidence_owner.json', dict(
        pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        resource='one CPU lease; no GPU, no new model fit', stages=stages))
    for script in stages:
        start = time.monotonic()
        with (ROOT / 'logs' / ('local_final_' + script + '.log')).open('ab') as log:
            code = subprocess.call([sys.executable, '-B', str(ROOT / script)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(script=script, exit_code=code, elapsed_s=time.monotonic() - start,
                      code_sha256=sha(ROOT / script))
        records.append(record)
        write(ROOT / 'outputs/local_final_stages' / script.replace('.py', '.json'), record)
        print(script, code, round(record['elapsed_s'], 2), flush=True)
        if code:
            raise RuntimeError('LOCAL_EVIDENCE_STAGE_FAILED ' + script)
    missing = read(ROOT / 'outputs/result_roles.json')['missing']
    assert not missing, ('REGISTERED_FINAL_RESULTS_MISSING', missing)
    assert read(ROOT / 'evidence/independent_actual_effect_metrics.json')['passed']
    write(ROOT / 'outputs/local_final_evidence_done.json', dict(
        passed=True, stages=records, missing=missing,
        import_archive_sha256=recovery['archive_sha256'],
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        expert_report_status='draft only; numerical interpretation and signoff still required'))


if __name__ == '__main__':
    main()
