import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from approved_epoch_delivery import validate_rows
from d29_platform.runtime import sha,write_json

def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    rows=[dict(job=j['id'],epoch='approved_maize_K' if j['calendar']=='K' else 'original_nonK',
        formal_gate_sha256='corrected' if j['calendar']=='K' else 'original',source_multiplier=1.,
        selection='fixed_contract_epoch_without_score_selection',evaluation_labels_used=False) for j in jobs]
    assert len(validate_rows(jobs,rows,'original','corrected'))==44
    negatives=[]
    for name,alter in (
        ('missing_path',lambda r:r.pop()),
        ('duplicate_path',lambda r:r.__setitem__(0,r[1])),
        ('narrow_K',lambda r:next(x for x in r if x['epoch']=='approved_maize_K').update(epoch='original_nonK',formal_gate_sha256='original')),
        ('nonK_reassigned',lambda r:next(x for x in r if x['epoch']=='original_nonK').update(epoch='approved_maize_K',formal_gate_sha256='corrected')),
        ('mixed_K_gate',lambda r:next(x for x in r if x['epoch']=='approved_maize_K').update(formal_gate_sha256='original')),
        ('multiplier_changed',lambda r:r[0].update(source_multiplier=.99)),
        ('score_selected',lambda r:r[0].update(selection='best_NSE_epoch')),
        ('evaluation_access',lambda r:r[0].update(evaluation_labels_used=True)),
        ('missing_access_identity',lambda r:r[0].pop('evaluation_labels_used')),
    ):
        test=copy.deepcopy(rows);alter(test)
        try:validate_rows(jobs,test,'original','corrected')
        except RuntimeError as exc:negatives.append(dict(control=name,reason=str(exc)))
        else:raise AssertionError(name)
    for name,gate in [('missing_new_gate',None),('same_old_gate','original')]:
        try:validate_rows(jobs,rows,'original',gate)
        except RuntimeError as exc:negatives.append(dict(control=name,reason=str(exc)))
        else:raise AssertionError(name)
    write_json(ROOT/'evidence/approved_epoch_delivery_known_errors.json',dict(passed=True,
        controls=negatives,scope='accounting_policy_only_not_actual44completion',
        code_sha256=sha(ROOT/'analysis/approved_epoch_delivery.py'),
        all_paths_completed=False,evaluation_labels_read=False))
    print('passed: exact30+14 approved accounting; 11 rejection controls')

if __name__=='__main__':main()
