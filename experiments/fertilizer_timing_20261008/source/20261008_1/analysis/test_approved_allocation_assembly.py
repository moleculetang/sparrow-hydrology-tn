import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from assemble_approved_allocation_delivery import validate_allocation
from d29_platform.runtime import sha,write_json


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True),encoding='utf8')


def fixture(base):
    folder=base/'result';science=base/'science';directory=folder/'input_allocation'
    directory.mkdir(parents=True)
    sources=['timing/inputs.py','timing/profiles.py','scripts/export_timing_allocations.py',
      'analysis/verify_timing_allocation_artifacts.py','inputs/reference/agriculture_reference/annual_crop_source_reference.parquet']
    for name in sources:
        p=science/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture-only-source-bytes')
    files={}
    for year in range(1961,2025):
        for kind in ['budgets','profiles']:
            name=f'{kind}_{year}.parquet';(directory/name).write_bytes(name.encode());files[name]=sha(directory/name)
    job={'id':'TEST_ONLY_NO_ACTUAL_ALLOCATION','calendar':'C2'};freeze={'parameter_sha256':'test-point'}
    allocation=dict(passed=True,job=job['id'],calendar='C2',source_multiplier=1.,formal_gate_sha256='gate',
      parameter_sha256='test-point',observation_data_read=False,BNF_and_deposition_unchanged=True,
      artifact_sha256=files,identities={name:sha(science/name) for name in sources[:3]},
      records=[dict(year=y,crops=21,records=230*21*2) for y in range(1961,2025)],
      maxima_kg=dict(annual=0.,month_sum=0.,reconstruction=0.))
    put(directory/'complete.json',allocation)
    review=dict(passed=True,job=job['id'],years=64,reaches=230,crops=21,formal_gate_sha256='gate',
      parameter_sha256='test-point',no_TN_or_prediction_values_read=True,
      allocation_receipt_sha256=sha(directory/'complete.json'),verifier_sha256=sha(science/sources[3]),
      annual_source_sha256=sha(science/sources[4]),
      records=[dict(year=y,rows=230*21*2,maxima_kg=dict(annual=0.,month_sum=0.)) for y in range(1961,2025)])
    put(directory/'independent_artifact_review.json',review)
    return folder,job,freeze,science


def mutate(folder,name,edit):
    p=folder/'input_allocation'/name;value=json.loads(p.read_text());edit(value);put(p,value)


def main():
    parent=ROOT/'outputs/allocation_assembly_controls';parent.mkdir(exist_ok=True)
    negative=[]
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        base=Path(temporary);f,j,z,s=fixture(base/'valid')
        assert len(validate_allocation(f,j,'gate',z,s))==130
        controls=[
          ('narrow_gate',lambda f,s:mutate(f,'complete.json',lambda r:r.update(formal_gate_sha256='narrow'))),
          ('wrong_point',lambda f,s:mutate(f,'complete.json',lambda r:r.update(parameter_sha256='another-point'))),
          ('missing_year',lambda f,s:mutate(f,'complete.json',lambda r:r['artifact_sha256'].pop('budgets_1961.parquet'))),
          ('mass_multiplier',lambda f,s:mutate(f,'complete.json',lambda r:r.update(source_multiplier=.99))),
          ('wrong_calendar',lambda f,s:mutate(f,'complete.json',lambda r:r.update(calendar='C0'))),
          ('read_labels',lambda f,s:mutate(f,'complete.json',lambda r:r.update(observation_data_read=True))),
          ('BNF_changed',lambda f,s:mutate(f,'complete.json',lambda r:r.update(BNF_and_deposition_unchanged=False))),
          ('no_readback',lambda f,s:mutate(f,'independent_artifact_review.json',lambda r:r.update(passed=False))),
          ('duplicate_review_year',lambda f,s:mutate(f,'independent_artifact_review.json',lambda r:r['records'][0].update(year=1962))),
          ('missing_crop_support',lambda f,s:mutate(f,'independent_artifact_review.json',lambda r:r['records'][0].update(rows=230*20*2))),
          ('absolute_error',lambda f,s:mutate(f,'independent_artifact_review.json',lambda r:r['records'][0]['maxima_kg'].update(annual=2e-6))),
          ('changed_daily_profile',lambda f,s:(f/'input_allocation/profiles_2024.parquet').write_bytes(b'changed')),
          ('changed_annual_source',lambda f,s:(s/'inputs/reference/agriculture_reference/annual_crop_source_reference.parquet').write_bytes(b'changed')),
          ('changed_verifier',lambda f,s:(s/'analysis/verify_timing_allocation_artifacts.py').write_bytes(b'changed')),
        ]
        for i,(name,edit) in enumerate(controls):
            f,j,z,s=fixture(base/str(i));edit(f,s)
            try:validate_allocation(f,j,'gate',z,s)
            except RuntimeError as exc:negative.append(dict(control=name,reason=str(exc)))
            else:raise AssertionError(name)
    write_json(ROOT/'evidence/approved_allocation_assembly_controls.json',dict(passed=True,
      negative_controls=negative,positive_bound_files=130,source_sha256=sha(ROOT/'analysis/assemble_approved_allocation_delivery.py'),
      scope='serialized_artifact_binding_controls_only',test_data_are_not_scientific_results=True,
      actual44delivery_executed=False,evaluation_labels_read=False))
    print(json.dumps(dict(passed=True,rejection_controls=len(negative),actual44delivery_executed=False)))


if __name__=='__main__':main()
