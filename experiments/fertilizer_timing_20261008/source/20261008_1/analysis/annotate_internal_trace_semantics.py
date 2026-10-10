"""Correct derived unit labels in new artifacts; never rewrite accepted traces."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
import pandas as pd
from d29_platform.runtime import sha,write_json
from internal_flux_observer import stage_semantics


def main():
    proofpath=ROOT/'outputs/replay_internal_traces/complete.json'
    proof=json.loads(proofpath.read_text())
    if not proof['passed'] or proof['count']!=12:raise RuntimeError('ALL12ACCEPTED_TRACES_REQUIRED')
    target=ROOT/'outputs/replay_internal_traces/annotated_shapes_v2';target.mkdir(exist_ok=True)
    if (target/'complete.json').exists():raise RuntimeError('ANNOTATED_SHAPES_ALREADY_COMPLETED')
    cases=[]
    for case in proof['completed']:
        receipt=ROOT/case['receipt']
        if sha(receipt)!=case['sha256']:raise RuntimeError('ACCEPTED_TRACE_RECEIPT_CHANGED')
        result=json.loads(receipt.read_text());source=receipt.parent/'internal_and_network_shapes.csv'
        if sha(source)!=result['shapes_sha256']:raise RuntimeError('ORIGINAL_DERIVED_SHAPES_CHANGED')
        frame=pd.read_csv(source);numeric=frame.select_dtypes(include='number').copy()
        rows=frame.to_dict('records')
        for row in rows:row.update(stage_semantics(row['stage']))
        corrected=pd.DataFrame(rows)
        if not corrected[numeric.columns].equals(numeric):
            raise RuntimeError('UNIT_ANNOTATION_CHANGED_PHYSICAL_SHAPE_VALUES')
        out=target/(case['job']+'.csv');corrected.to_csv(out,index=False)
        cases.append(dict(job=case['job'],source_receipt_sha256=case['sha256'],
            original_shapes_sha256=sha(source),annotated_file=str(out.relative_to(ROOT)),annotated_sha256=sha(out)))
    receipt=dict(passed=True,count=len(cases),cases=cases,source_complete_sha256=sha(proofpath),
        physical_values_changed=False,original_artifacts_overwritten=False,model_rerun=False,observations_read=False,
        reason='pre-uptake availability is a stock; recurrent requests are not external input or annual harvest debt',
        identities={n:sha(ROOT/n) for n in ['analysis/internal_flux_observer.py','analysis/annotate_internal_trace_semantics.py']})
    write_json(target/'complete.json',receipt);print(json.dumps(dict(passed=True,count=len(cases))))


if __name__=='__main__':main()
