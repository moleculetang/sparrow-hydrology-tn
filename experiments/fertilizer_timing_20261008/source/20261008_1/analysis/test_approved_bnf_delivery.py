"""Reject false BNF delivery claims without changing frozen experiment files."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json
import run_approved_bnf_delivery as delivery
from bnf_endpoint_readback import verify as verify_endpoint


def main():
    controls=[]
    delivery.require_controls();controls.append('current13_partition_controls_byte_bound')
    acquired=[]
    with patch.object(delivery.registry,'acquire',side_effect=lambda *a,**k:acquired.append(True)):
        try:delivery.main('final')
        except RuntimeError as error:
            assert str(error)=='FINAL_CORRECTED44DELIVERY_NOT_READY',repr(error)
            controls.append('actual_incomplete44_refused_before_resource_or_label_access')
        else:raise AssertionError('ACTUAL_INCOMPLETE44_BNF_ACCEPTED')
    assert not acquired
    jid='TIM_F24_C0_V0_s0'
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    source=ROOT/'outputs/jobs'/jid
    target=ROOT/'outputs/jobs'/('bnf_trace_approved_preflight_v1_'+jid)
    path=target/'complete.json'
    if not path.exists():raise RuntimeError('ACTUAL64YEAR_BNF_PREFLIGHT_REQUIRED_FOR_DELIVERY_CONTROLS')
    actual=delivery.verified_case(target,source,job,'preflight')
    endpoint=verify_endpoint(target,source)
    controls.extend(['actual64year230reach_BNF_arrays_independent_readback',
                     'actual_regrouped_original_BNF_and_other_inventory_equivalence'])
    original=json.loads(path.read_text());reader=Path.read_text
    edits=[('wrong_gate',lambda v:v.update(formal_gate_sha256='wrong')),
        ('wrong_science_root',lambda v:v.update(science_root=str(ROOT.with_name('20261008_1_maize_scope_v2')))),
        ('changed_parameter',lambda v:v.update(source_point_sha256='wrong')),
        ('changed_frozen_dependency',lambda v:v['bound_files'].update(physical_ledger_json='wrong')),
        ('source_multiplier_changed',lambda v:v.update(source_multiplier=2.)),
        ('BNF_timing_changed',lambda v:v.update(BNF_timing_changed=True)),
        ('evaluation_labels_read',lambda v:v.update(evaluation_labels_read=True)),
        ('scientific_training_added',lambda v:v.update(new_training_calls=1)),
        ('missing_year',lambda v:v['years'].pop()),
        ('changed_year_order',lambda v:v['years'][0].update(year=1962)),
        ('missing_runtime',lambda v:v['runtime_files'].pop('timing.profiles')),
        ('missing_independent_input',lambda v:v['input_bindings'].pop('inputs/calendar_support/reach_profiles.json')),
        ('source_error_above_original_tolerance',lambda v:v.update(local_source_sum_max_error_kg=1.000001e-6)),
        ('changed_annual_bytes',lambda v:v['years'][0].update(sha256='wrong')),
        ('changed_crop_table',lambda v:v.update(crop_table_sha256='wrong')),
        ('incorrect_partition_code',lambda v:v.update(partition_sha256='wrong'))]
    for name,edit in edits:
        altered=copy.deepcopy(original);edit(altered)
        def substitute(p,*args,**kwargs):
            return json.dumps(altered) if p.resolve()==path.resolve() else reader(p,*args,**kwargs)
        with patch.object(Path,'read_text',substitute):
            try:delivery.verified_case(target,source,job,'preflight')
            except (RuntimeError,KeyError,ValueError):controls.append(name+'_rejected')
            else:raise AssertionError('FALSE_BNF_DELIVERY_ACCEPTED '+name)
    result=dict(passed=True,controls=controls,actual_cases=1,actual_case=actual,endpoint=endpoint,
        actual44_BNF_delivery_completed=False,evaluation_labels_read=False,new_training_calls=0,
        files={p:sha(ROOT/p) for p in ('analysis/run_approved_bnf_delivery.py',
            'analysis/trace_approved_bnf_history.py','analysis/bnf_calendar_partition.py',
            'analysis/bnf_endpoint_readback.py','analysis/test_approved_bnf_delivery.py')})
    write_json(ROOT/'evidence/approved_bnf_delivery_controls.json',result)
    print(json.dumps(dict(passed=True,controls=len(controls),actual_cases=1,actual44_BNF_delivery_completed=False)))


if __name__=='__main__':main()
