"""Actual immutable imports plus tamper controls for additive physical batches.

No evaluation labels, solver states, training dispatch or physical reruns.
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'analysis'), str(ROOT/'scripts')]
from d29_platform.runtime import sha, write_json
import staged_frozen_sources as sources
import staged_batch_receipts as batches


def main():
    cases = []
    reader = Path.read_text

    def reject(name, function, message):
        try:
            function()
        except (RuntimeError, ValueError) as error:
            if message not in str(error):
                raise
            cases.append(dict(case=name, passed=True, rejection=str(error)))
        else:
            raise AssertionError('TAMPERED_STAGE_ADMITTED '+name)

    def altered(path, value, function):
        def read(self, *args, **kwargs):
            return json.dumps(value) if self == path else reader(self, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            function()

    for jid in sorted(sources.IMPORTS['staged_frozen_import_002']):
        folder = sources.resolve(jid)
        cases.append(dict(case='actual_frozen_import_'+jid, passed=True, folder=str(folder)))
    for jid in ('', 'TIM_F23_K_V0_s0', 'unregistered', '../escape'):
        reject('unknown_import_'+repr(jid), lambda j=jid:sources.resolve(j),
               'EXPLICIT_UNIQUE_STAGED_NONK_ID_REQUIRED')
    proof_path = ROOT/'outputs/staged_frozen_import_002/complete.json'
    original = json.loads(proof_path.read_text())
    jid = 'TIM_F23_C0_V1_s0'
    for key, value in [('passed', False), ('partial_only', False),
                       ('evaluation_admitted', True), ('training_ownership_changed', True)]:
        changed = dict(original); changed[key] = value
        altered(proof_path, changed, lambda k=key:reject('import_flag_'+k,
            lambda:sources.resolve(jid), 'PARTIAL_PHYSICAL_ONLY_IMPORT_REQUIRED'))
    changed = dict(original); changed['target'] = str(ROOT/'outputs/staged_frozen_import/data')
    altered(proof_path, changed, lambda:reject('cross_batch_target',lambda:sources.resolve(jid),
        'EXACT_PRIVATE_STAGED_TARGET_REQUIRED'))
    changed = dict(original); changed['manifest_sha256'] = '0'*64
    altered(proof_path, changed, lambda:reject('changed_manifest_bytes',lambda:sources.resolve(jid),
        'STAGED_MANIFEST_CHANGED'))
    manifest_path = Path(original['target'])/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, mutation in [
        ('empty_manifest', lambda v:v.update(records=[])),
        ('missing_baseline', lambda v:v['records'].pop()),
        ('duplicate_baseline', lambda v:v['records'].__setitem__(1,copy.deepcopy(v['records'][0]))),
        ('K_in_nonK_import', lambda v:v['records'][0]['job'].update(calendar='K')),
    ]:
        changed = copy.deepcopy(manifest); mutation(changed)
        altered(manifest_path, changed, lambda n=name:reject(n,lambda:sources.resolve(jid),
            'EXACT_IMMUTABLE_STAGED_BATCH_REQUIRED'))

    rows = batches.records('internal')
    if len(rows) != 8:
        raise RuntimeError('ACTUAL_EIGHT_INTERNAL_REPLAYS_REQUIRED')
    cases.append(dict(case='actual_eight_batch_receipts',passed=True,rows=len(rows)))
    reject('unknown_kind',lambda:batches.records('unknown'),'DECLARED_STAGED_DIAGNOSTIC_KIND_REQUIRED')
    batch_path = ROOT/'outputs/staged_internal_trace_002/complete.json'
    batch = json.loads(batch_path.read_text())
    for key, value in [('passed',False),('partial_only',False),('evaluation_admitted',True),
                       ('full44_completed_claimed',True),('training_ownership_changed',True)]:
        changed = copy.deepcopy(batch); changed[key] = value
        altered(batch_path,changed,lambda k=key:reject('batch_flag_'+k,
            lambda:batches.records('internal'),'EXACT_COMPLETE_KNOWN_STAGE_BATCH_REQUIRED'))
    for name, mutation, message in [
        ('empty_batch',lambda v:v.update(completed=[]),'EXACT_COMPLETE_KNOWN_STAGE_BATCH_REQUIRED'),
        ('missing_receipt',lambda v:v['completed'].pop(),'EXACT_COMPLETE_KNOWN_STAGE_BATCH_REQUIRED'),
        ('duplicate_receipt',lambda v:v['completed'].__setitem__(1,copy.deepcopy(v['completed'][0])),
            'EXACT_COMPLETE_KNOWN_STAGE_BATCH_REQUIRED'),
        ('cross_job_receipt',lambda v:v['completed'][0].update(complete=v['completed'][1]['complete']),
            'EXACT_KNOWN_STAGE_RECEIPT_BYTES_REQUIRED'),
        ('path_escape',lambda v:v['completed'][0].update(complete=str(ROOT/'reports/not_a_receipt.json')),
            'EXACT_KNOWN_STAGE_RECEIPT_BYTES_REQUIRED'),
        ('changed_receipt_bytes',lambda v:v['completed'][0].update(sha256='0'*64),
            'EXACT_KNOWN_STAGE_RECEIPT_BYTES_REQUIRED'),
    ]:
        changed = copy.deepcopy(batch); mutation(changed)
        altered(batch_path,changed,lambda n=name,m=message:reject(n,lambda:batches.records('internal'),m))
    exists = Path.exists
    with patch.object(Path,'exists',lambda self:False if self == batch_path else exists(self)):
        reject('live_or_partial_additive_batch',lambda:batches.records('internal'),
               'LIVE_OR_PARTIAL_STAGED_BATCH_NOT_ADMISSIBLE')
    result = dict(passed=True,cases=cases,actual_frozen_imports=2,actual_internal_receipts=8,
        evaluation_labels_read=False,physical_reruns=0,new_training_calls=0,
        full44_completed_claimed=False,training_ownership_changed=False,
        resolver_sha256=sha(Path(sources.__file__)),batch_reader_sha256=sha(Path(batches.__file__)),
        test_sha256=sha(Path(__file__)))
    write_json(ROOT/'evidence/additive_staged_admission_controls.json',result)
    print(json.dumps(dict(passed=True,controls=len(cases),actual_frozen_imports=2,
                         actual_internal_receipts=8)))


if __name__ == '__main__':
    main()
