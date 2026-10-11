"""Independent negative controls for cross-root full-history trace admission."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
import numpy as np
from d29_platform.runtime import sha,write_json
from trace_approved_internal_fluxes import science_root,require_scientific_module,export_comparison,prediction_comparison


def main():
    controls=[]
    sys.path[:0]=[str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
    from native_bound_adapter_v2 import canonical_bounds
    pairs=[(-2.,3.),(-np.inf,np.inf)]
    converted=canonical_bounds(pairs)
    assert [list(pair) for pair in converted]==json.loads(json.dumps(converted))
    assert [list(pair) for pair in converted]!=[[-2.,3.],[0.,None]]
    controls.append('JSON_tuple_list_boundary_equivalence_finite_sides_preserved')
    for c in ('C0','C1','C2'):
        assert science_root(dict(calendar=c))==ROOT;controls.append(c+'_original_runtime')
    candidate=ROOT.with_name('20261008_1_maize_scope_v2')
    assert science_root(dict(calendar='K'))==candidate;controls.append('K_corrected_runtime')
    for actual,expected in [(ROOT,candidate),(candidate,ROOT)]:
        module=SimpleNamespace(__file__=str(actual/'timing/adapter.py'),__name__='timing.adapter')
        try:require_scientific_module(module,expected)
        except RuntimeError:controls.append('cross_root_rejected_'+actual.name)
        else:raise AssertionError('cross-root adapter admitted')
    zero=np.zeros((3,2));close=zero.copy();close[0,0]=1e-6
    assert export_comparison(close,zero)['passed'];controls.append('existing_absolute_mass_tolerance')
    close[0,0]=np.nextafter(1e-6,np.inf)
    assert not export_comparison(close,zero)['passed'];controls.append('above_mass_tolerance_rejected')
    for values in (np.zeros((2,2)),np.full((3,2),np.nan)):
        try:export_comparison(values,zero)
        except RuntimeError:controls.append('invalid_export_support_rejected')
        else:raise AssertionError('invalid export accepted')
    reference=np.array([0.,2.])
    assert prediction_comparison(reference,reference,3.,3.)['bitwise_training_prediction']
    controls.append('identical_prediction_and_objective')
    bad=reference.copy();bad[0]=1.01e-8
    assert not prediction_comparison(bad,reference,3.,3.)['prediction_passed']
    controls.append('zero_reference_no_epsilon_prediction_rejection')
    assert not prediction_comparison(reference,reference,3.+1e-6,3.)['objective_passed']
    controls.append('changed_objective_rejected')
    try:prediction_comparison(reference[:1],reference,3.,3.)
    except RuntimeError:controls.append('prediction_support_mismatch_rejected')
    else:raise AssertionError('support mismatch admitted')
    from approved_epoch_delivery import admitted_folders
    try:admitted_folders(ROOT)
    except RuntimeError as exc:
        if 'FINAL_CORRECTED44DELIVERY_NOT_READY' not in str(exc):raise
        controls.append('actual_incomplete44_final_admission_rejected')
    else:raise AssertionError('all44 state changed; review rather than assume missing')
    actual_cases=0;actual_result=None
    jid='TIM_F24_C0_V0_s0'
    actual=ROOT/'outputs/jobs'/('internal_trace_approved_preflight_v1_'+jid)
    if (actual/'complete.json').exists():
        from run_approved_internal_trace_delivery import verified_case
        job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
        source=ROOT/'outputs/jobs'/jid
        actual_result=verified_case(actual,source,job,'preflight')
        actual_cases=1;controls.append('actual_accepted64year_original_runtime_readback')
        path=actual/'complete.json';original=json.loads(path.read_text())
        reader=Path.read_text
        for name,edit in [
            ('wrong_gate',lambda v:v.update(formal_gate_sha256='wrong')),
            ('wrong_science_root',lambda v:v.update(science_root=str(candidate.resolve()))),
            ('changed_parameter_binding',lambda v:v.update(source_point_sha256='wrong')),
            ('evaluation_read',lambda v:v.update(no_evaluation_outcomes_read=False)),
            ('training_call_added',lambda v:v.update(new_scientific_training_calls=1)),
            ('extra_forward_call',lambda v:v.update(diagnostic_forward_calls=2)),
            ('changed_reference_bytes',lambda v:v['bound_files'].update(physical_ledger_json='wrong')),
            ('missing_year',lambda v:v['years'].pop()),
            ('wrong_year_sequence',lambda v:v['years'][0].update(year=1962)),
            ('failed_year',lambda v:v['years'][0].update(passed=False)),
            ('missing_runtime',lambda v:v['runtime_files'].pop('timing.adapter')),
            ('changed_runtime',lambda v:v['runtime_files']['timing.adapter'].update(sha256='wrong')),
            ('changed_trace_output',lambda v:v.update(routing_trace_sha256='wrong')),
        ]:
            value=json.loads(json.dumps(original));edit(value)
            def read_override(self,*args,**kwargs):
                return json.dumps(value) if self==path else reader(self,*args,**kwargs)
            with patch.object(Path,'read_text',read_override):
                try:verified_case(actual,source,job,'preflight')
                except RuntimeError:controls.append(name+'_rejected')
                else:raise AssertionError(name)
        prior=ROOT/'outputs/jobs'/('internal_trace_acceptedfit_v1_'+jid)
        for year in range(1961,2025):
            with np.load(actual/'yearly'/f'{year}.npz',allow_pickle=False) as a, np.load(prior/'yearly'/f'{year}.npz',allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                assert all(np.array_equal(a[key],b[key]) for key in a.files)
        controls.append('all64year_internal_arrays_exact_vs_original_reference')
        with np.load(actual/'routing_trace.npz',allow_pickle=False) as a,np.load(prior/'routing_trace.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[key],b[key]) for key in a.files)
        controls.append('all_network_and_reservoir_arrays_exact_vs_original_reference')
    evidence=dict(passed=True,controls=controls,actual_replay_cases=actual_cases,actual_K_cases=0,
        synthetic_controls_only=actual_cases==0,actual_result=actual_result,full44_internal_replay_completed=False,
        evaluation_labels_read=False,worker_sha256=sha(ROOT/'analysis/trace_approved_internal_fluxes.py'),
        launcher_sha256=sha(ROOT/'analysis/run_approved_internal_trace_delivery.py'),test_sha256=sha(Path(__file__)))
    write_json(ROOT/'evidence/approved_internal_trace_controls.json',evidence)
    print(json.dumps(dict(passed=True,controls=len(controls),actual_replay_cases=actual_cases,actual_K_cases=0)))


if __name__=='__main__': main()
