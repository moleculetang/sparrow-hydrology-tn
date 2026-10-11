import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'deployment')]
from d29_platform.runtime import sha, write_json
from assemble_approved_delivery import validate_frozen, publish


def put(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,sort_keys=True),encoding='utf8')


def fixture(folder):
    job={'id':'TEST_ONLY_NO_SCIENTIFIC_JOB','calendar':'C0'}
    np.save(folder/'frozen_parameters.npy',np.array([.2,.4],dtype=np.float64))
    point=sha(folder/'frozen_parameters.npy')
    contract={'job':job,'names':['a','b']}
    put(folder/'training_contract.json',contract)
    audit=dict(job=job['id'],numerically_sufficient=True,source_multiplier=1.,formal_gate_sha256='gate',
      objective_passed=True,physical_passed=True,gradient_recovery_passed=True,derivative_passed=True,
      projected_gradient=1e-7,measured_feasible_descent=False,full_history=[1961,2024],reaches=230,
      parameter_sha256=point,checks=[dict(coordinate=n,derivative_qualified=True,measured_feasible_descent=False)
                                  for n in ['a','b']])
    put(folder/'independent_audit.json',audit)
    artifacts={}
    for name in ['full_history_physical_source_ledger.npz','station_source_ledger.npz']:
        (folder/name).write_bytes(b'test-only-ledger-hash-fixture')
        artifacts[name]=sha(folder/name)
    rows=[]
    for year in range(1961,2025):
        relative=f'plant_activity_yearly/{year}.npz'
        p=folder/relative;p.parent.mkdir(exist_ok=True);p.write_bytes(str(year).encode())
        artifacts[relative]=sha(p)
        rows.append(dict(year=year,plant_activity=dict(reach_land_file=relative,reach_land_sha256=sha(p))))
    physical=dict(passed=True,year_ledger=rows,
      parameter_vector_sha256=hashlib.sha256(np.array([.2,.4],dtype='<f8').tobytes()).hexdigest(),
      full_history_ledger_sha256=artifacts['full_history_physical_source_ledger.npz'],
      station_source_ledger_sha256=artifacts['station_source_ledger.npz'])
    put(folder/'physical_ledger.json',physical)
    for name in ['frozen_station_days.parquet','frozen_station_months.parquet','frozen_local_kg_day.npy']:
        (folder/name).write_bytes(b'test-only-frozen-bytes')
    freeze=dict(job=job,numerically_sufficient=True,source_multiplier=1.,formal_gate_sha256='gate',
      station_count=1,simulation_reaches=230,history_start=1961,no_evaluation_outcomes_read=True,
      evaluation_labels_used=False,audit_sha256=sha(folder/'independent_audit.json'),parameter_sha256=point,
      days_sha256=sha(folder/'frozen_station_days.parquet'),months_sha256=sha(folder/'frozen_station_months.parquet'),
      local_sha256=sha(folder/'frozen_local_kg_day.npy'),physical_ledger_receipt_sha256=sha(folder/'physical_ledger.json'),
      physical_artifact_sha256=artifacts)
    put(folder/'prediction_freeze.json',freeze)
    put(folder/'status.json',dict(audit_completed=True,numerically_sufficient=True))
    return job


def change(folder, name, edit):
    p=folder/name;v=json.loads(p.read_text());edit(v);put(p,v)


def main():
    parent=ROOT/'outputs/delivery_assembly_controls';parent.mkdir(exist_ok=True)
    rejected=[]
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        base=Path(temporary)
        f=base/'valid';f.mkdir();job=fixture(f)
        files=validate_frozen(f,job,'gate',{'station_keys':['test']})
        assert len(files)==75
        negatives=[
          ('unfinished_status',lambda p:change(p,'status.json',lambda r:r.update(audit_completed=False))),
          ('incorrect_gate',lambda p:change(p,'independent_audit.json',lambda r:r.update(formal_gate_sha256='narrow'))),
          ('source_multiplier',lambda p:change(p,'independent_audit.json',lambda r:r.update(source_multiplier=.99))),
          ('empty_coordinates',lambda p:change(p,'independent_audit.json',lambda r:r.update(checks=[]))),
          ('duplicate_coordinates',lambda p:change(p,'independent_audit.json',lambda r:r['checks'].__setitem__(1,r['checks'][0]))),
          ('unqualified_coordinate',lambda p:change(p,'independent_audit.json',lambda r:r['checks'][0].update(derivative_qualified=False))),
          ('gradient_above_gate',lambda p:change(p,'independent_audit.json',lambda r:r.update(projected_gradient=2e-5))),
          ('feasible_descent',lambda p:change(p,'independent_audit.json',lambda r:r.update(measured_feasible_descent=True))),
          ('evaluation_access',lambda p:change(p,'prediction_freeze.json',lambda r:r.update(evaluation_labels_used=True))),
          ('wrong_job',lambda p:change(p,'prediction_freeze.json',lambda r:r['job'].update(id='another_job'))),
          ('missing_station',lambda p:change(p,'prediction_freeze.json',lambda r:r.update(station_count=0))),
          ('changed_prediction',lambda p:(p/'frozen_station_days.parquet').write_bytes(b'changed')),
          ('changed_parameter',lambda p:np.save(p/'frozen_parameters.npy',np.array([.2,.5]))),
          ('missing_plant_year',lambda p:(p/'plant_activity_yearly/1961.npz').unlink()),
          ('changed_plant_bytes',lambda p:(p/'plant_activity_yearly/2024.npz').write_bytes(b'changed')),
          ('missing_physical_artifact',lambda p:change(p,'prediction_freeze.json',lambda r:r['physical_artifact_sha256'].pop('station_source_ledger.npz'))),
          ('escaped_artifact',lambda p:change(p,'prediction_freeze.json',lambda r:r['physical_artifact_sha256'].__setitem__('../escape',r['physical_artifact_sha256'].pop('station_source_ledger.npz')))),
        ]
        for index,(name,edit) in enumerate(negatives):
            f=base/str(index);f.mkdir();job=fixture(f);edit(f)
            try:validate_frozen(f,job,'gate',{'station_keys':['test']})
            except RuntimeError as exc:rejected.append(dict(control=name,reason=str(exc)))
            else:raise AssertionError(name)
        publishing=base/'publishing';(publishing/'evidence').mkdir(parents=True)
        p=publish(publishing,{'test_only':True});assert publish(publishing,{'test_only':True})==p
        try:publish(publishing,{'test_only':False})
        except RuntimeError as exc:rejected.append(dict(control='immutable_delivery_conflict',reason=str(exc)))
        else:raise AssertionError('publish overwrote existing receipt')
    write_json(ROOT/'evidence/approved_delivery_assembly_controls.json',dict(passed=True,
      rejection_controls=rejected,positive_bound_files=75,fixture_scope='byte_binding_and_complete_point_checks_only',
      test_data_are_not_scientific_results=True,all44completed=False,evaluation_labels_read=False,
      source_sha256={name:sha(ROOT/'analysis'/name) for name in ['assemble_approved_delivery.py','approved_epoch_delivery.py']}))
    print(json.dumps(dict(passed=True,rejections=len(rejected),scientific_paths_certified=0)))


if __name__=='__main__':main()
