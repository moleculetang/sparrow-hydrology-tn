"""Cross-contract and immutable-output controls; no actual K fit is invented."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis')]
from d29_platform.runtime import sha, write_json
from diagnose_approved_waveforms import approved_gate, accepted_point
from run_approved_waveform_delivery import verified_case
from test_approved_delivery_assembly import fixture, put


def fixture_gate(folder, job, gate):
    for name in ('independent_audit.json', 'prediction_freeze.json'):
        path = folder / name; value = json.loads(path.read_text())
        value['formal_gate_sha256'] = gate
        if name == 'prediction_freeze.json':
            value['job'] = job; value['audit_sha256'] = sha(folder / 'independent_audit.json')
        put(path, value)
    path = folder / 'training_contract.json'; value = json.loads(path.read_text()); value['job'] = job; put(path, value)


def main():
    controls = []; parent = ROOT / 'outputs/approved_waveform_controls'; parent.mkdir(exist_ok=True)
    original = approved_gate({'calendar': 'C0'}); corrected = approved_gate({'calendar': 'K'})
    assert corrected != original
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        f = root / 'synthetic_K_contract_only'; f.mkdir(); job = fixture(f); job['calendar'] = 'K'
        fixture_gate(f, job, original)
        try:
            accepted_point(f, job, {'station_keys': ['test']})
        except RuntimeError as exc:
            assert 'FROZEN_MASS_OR_EPOCH_CHANGED' in str(exc)
            controls.append('K_original_gate_rejected')
        else:
            raise AssertionError('old K epoch admitted')
        fixture_gate(f, job, corrected)
        _, _, files = accepted_point(f, job, {'station_keys': ['test']})
        assert len(files) == 75
        controls.append('synthetic_corrected_K_receipt_admitted_not_actual_K_run')
        nonK = copy.deepcopy(job); nonK['calendar'] = 'C0'; fixture_gate(f, nonK, corrected)
        try:
            accepted_point(f, nonK, {'station_keys': ['test']})
        except RuntimeError as exc:
            assert 'FROZEN_MASS_OR_EPOCH_CHANGED' in str(exc)
            controls.append('nonK_corrected_gate_rejected')
        else:
            raise AssertionError('non-K epoch changed')
        jid = 'TIM_F24_C0_V0_s0'; source = ROOT / 'outputs/jobs' / jid
        actual = ROOT / 'outputs/approved_waveforms_preflight' / jid
        bound = verified_case(actual, source, jid, original)
        controls.append('actual_baseline_preflight_outputs_verified')
        copied = root / 'copied_output_controls'; shutil.copytree(actual, copied)
        for name, edit in (
            ('gate_changed', lambda v: v.update(formal_gate_sha256=corrected)),
            ('wrong_job', lambda v: v['registered_job'].update(id='wrong')),
            ('evaluation_access', lambda v: v.update(evaluation_labels_read=True)),
            ('model_rerun', lambda v: v.update(physical_oracle_calls=1)),
            ('input_freeze_changed', lambda v: v.update(source_freeze_sha256='wrong')),
            ('extractor_changed', lambda v: v.update(source_sha256='wrong')),
            ('shape_helper_changed', lambda v: v.update(immutable_shape_helper_sha256='wrong')),
            ('missing_shape_file', lambda v: v['output_sha256'].pop('domain_flux_shapes.csv')),
        ):
            v=json.loads((actual/'complete.json').read_text()); edit(v); put(copied/'complete.json', v)
            try:
                verified_case(copied, source, jid, original)
            except RuntimeError:
                controls.append(name)
            else:
                raise AssertionError(name)
        put(copied/'complete.json', json.loads((actual/'complete.json').read_text()))
        (copied/'domain_flux_shapes.csv').write_bytes(b'corrupted output')
        try:
            verified_case(copied, source, jid, original)
        except RuntimeError as exc:
            assert 'DIAGNOSTIC_OUTPUT_BYTES_CHANGED' in str(exc)
            controls.append('shape_bytes_changed')
        else:
            raise AssertionError('output corruption admitted')
    # Numerical reference is the previously completed extraction at this exact
    # accepted baseline. This is evidence for the reader, not a new fit.
    import pandas as pd
    prior = ROOT / 'outputs/frozen_fit_waveforms_v1' / jid
    for filename in ('domain_flux_shapes.csv', 'station_boundary_shapes.csv'):
        a = pd.read_csv(prior / filename); b = pd.read_csv(actual / filename)
        keys = ['stage', 'year', 'station_key', 'source']
        a = a.sort_values(keys).reset_index(drop=True); b = b.sort_values(keys).reset_index(drop=True)
        # New reader explicitly distinguishes zero-water concentration support.
        shared = [c for c in a if c in b and c not in ('nse_status', 'mass_timing_status', 'sum_value_units')]
        pd.testing.assert_frame_equal(a[shared], b[shared], check_exact=True)
        concentration = b.stage.eq('OU_concentration_prediction_shape')
        assert a.loc[~concentration, 'sum_value_units'].equals(b.loc[~concentration, 'sum_value_units'])
        if concentration.any():
            assert b.loc[concentration, 'sum_value_units'].eq(
                'mg N day/L_on_positive_water_days_not_basin_mass').all()
        controls.append('actual_unchanged_' + filename)
    write_json(ROOT / 'evidence/approved_waveform_controls.json', dict(passed=True,
        controls=controls, actual_baseline=jid, actual_cases=1, actual_K_cases=0,
        source_sha256=sha(ROOT / 'analysis/diagnose_approved_waveforms.py'),
        launcher_sha256=sha(ROOT / 'analysis/run_approved_waveform_delivery.py'),
        test_sha256=sha(Path(__file__)), baseline_receipt_sha256=bound['complete_sha256'],
        physical_oracle_calls=0, evaluation_labels_read=False, all44_completed=False))
    print(json.dumps(dict(passed=True, controls=len(controls), actual_cases=1, actual_K_cases=0)))


if __name__ == '__main__':
    main()
