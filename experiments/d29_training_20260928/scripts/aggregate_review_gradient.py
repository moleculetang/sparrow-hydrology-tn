"""Accept only complete, identical-reference 23-coordinate LAND1 probes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json
from d29_training.derivative_gate import refinement_passes

paths=[ROOT/'outputs/land1_gradient_v6_shard_00_12.json',ROOT/'outputs/land1_gradient_v6_shard_12_23.json']
parts=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
if any(p['status'] not in ('passed','requires_branch_review') for p in parts):
    raise RuntimeError('INCOMPLETE_GRADIENT_SHARD')
reference=parts[0]
for part in parts[1:]:
    if part['identity']!=reference['identity'] or part['objective']!=reference['objective'] or part['analytic_gradient']!=reference['analytic_gradient']:
        raise RuntimeError('GRADIENT_SHARD_REFERENCE_MISMATCH')
for path,digest in reference['identity'].items():
    if sha(path)!=digest:raise RuntimeError('GRADIENT_IMPLEMENTATION_CHANGED '+path)
rows=sorted((row for part in parts for row in part['coordinates']),key=lambda row:row['index'])
if [r['index'] for r in rows]!=list(range(23)):raise RuntimeError('INCOMPLETE_OR_OVERLAPPING_GRADIENT_COVERAGE')
reviews=[]
for part in parts:
    for row in part['coordinates']:
        if row['passed']:continue
        path=ROOT/f"outputs/land1_review_coordinate_{row['index']}_branch_refinement.json"
        accepted=path.exists() and refinement_passes(part,row['index'],json.loads(path.read_text(encoding='utf-8')))
        reviews.append({'coordinate':row['index'],'accepted':bool(accepted),'original_scan_passed':False,
                        'path':str(path) if path.exists() else None,'sha256':sha(path) if path.exists() else None})
write_json(ROOT/'outputs/land1_review_gradient.json',{'status':'passed_with_recorded_refinement' if reviews and all(r['accepted'] for r in reviews) else 'passed' if all(r['passed'] for r in rows) else 'requires_branch_review',
    'coordinates':rows,'identity':reference['identity'],'objective':reference['objective'],
    'analytic_gradient':reference['analytic_gradient'],'calls':sum(part['calls'] for part in parts),
    'refinement_reviews':reviews,
    'shards':[{'path':str(path),'sha256':sha(path)} for path in paths],
    'NSE':'not applicable to numerical gradient audit'})
