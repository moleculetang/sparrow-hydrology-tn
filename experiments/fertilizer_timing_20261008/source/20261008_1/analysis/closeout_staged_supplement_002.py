"""One-shot actual process, receipt and reuse-proof closeout of two replays.

Does not release leases, read evaluation labels, or declare the44-path finish.
"""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'scripts')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from staged_batch_receipts import records
from staged_frozen_sources import IMPORTS


def main():
    expected=IMPORTS['staged_frozen_import_002']
    leases=registry.status(root=registry_root())['leases']
    result=[]
    for kind,base,proof_name in (
        ('internal','staged_internal_trace','staged_trace_reuse_controls.json'),
        ('bnf','staged_bnf_trace','staged_bnf_reuse_controls.json')):
        rows=records(kind)
        if len(rows)!=8:raise RuntimeError('ACTUAL_EIGHT_DIAGNOSTIC_REPLAYS_REQUIRED')
        proof_path=ROOT/'evidence'/proof_name
        proof=json.loads(proof_path.read_text())
        if (proof.get('passed') is not True or proof.get('actual_full_history_cases')!=8
                or {c['job'] for c in proof['cases']}!={r['job'] for r in rows}
                or len(proof['cases'])!=8 or proof.get('evaluation_admitted') is not False
                or proof.get('resolver_sha256')!=sha(ROOT/'analysis/staged_frozen_sources.py')
                or proof.get('batch_reader_sha256')!=sha(ROOT/'analysis/staged_batch_receipts.py')
                or proof.get('launcher_sha256')!=sha(ROOT/'analysis'/('run_approved_internal_trace_delivery.py'
                    if kind=='internal' else 'run_approved_bnf_delivery.py'))):
            raise RuntimeError('CURRENT_ACTUAL_EIGHT_REUSE_PROOF_REQUIRED')
        folder=ROOT/'outputs'/(base+'_002')
        immutable_proof=folder/'reuse_controls_eight_actual.json'
        proof_bytes=proof_path.read_bytes()
        if immutable_proof.exists() and immutable_proof.read_bytes()!=proof_bytes:
            raise RuntimeError('EIGHT_CASE_HISTORICAL_PROOF_NO_OVERWRITE')
        if not immutable_proof.exists():immutable_proof.write_bytes(proof_bytes)
        complete=json.loads((folder/'complete.json').read_text())
        status=json.loads((folder/'status.json').read_text())
        owner=status['owner']
        if registry.identity(owner['pid'])==owner['created']:
            raise RuntimeError('STAGED_PARENT_STILL_LIVE')
        namespace='internal_trace_approved_staged_v1' if kind=='internal' else 'bnf_trace_approved_staged_v1'
        ids={c['job'] for c in complete['completed']}
        events=complete['events']
        if (ids!=expected or len(complete['completed'])!=2 or len(events)!=2
                or {e['job'] for e in events}!=expected or any(e['exit_code']!=0 for e in events)
                or status['running'] or status['queue']):
            raise RuntimeError('ACTUAL_TWO_ZERO_EXIT_COMPLETIONS_REQUIRED')
        if any(l['project']=='20261008_1' and l['job'] in
               {'analysis/'+namespace+'/'+j for j in expected} for l in leases.values()):
            raise RuntimeError('STAGED_CHILD_LEASE_NOT_RECOVERED')
        cases=[]
        for row in complete['completed']:
            path=Path(row['complete']);value=json.loads(path.read_text())
            if sha(path)!=row['sha256'] or value['full_history']!=[1961,2024]:
                raise RuntimeError('EXACT_SUPPLEMENT64YEAR_RECEIPT_REQUIRED')
            case=dict(job=row['job'],receipt_sha256=sha(path),seconds=value['seconds'],
                objective_replay_error=value['objective_replay_error'],
                training_prediction_max_error_mg_l=value['training_prediction_max_error_mg_l'],
                network_daily_relative_error=value['network']['daily_basin_relative_max'])
            if kind=='internal':
                case['max_internal_balance_kg']=max(abs(x) for y in value['years']
                    for x in y['internal_receipt']['errors'].values())
                case['max_local_source_replay_kg']=max(c['max_abs_difference_kg']
                    for y in value['years'] for c in y['source_ledger_export_comparison'].values())
            else:
                case['max_local_source_replay_kg']=value['local_source_sum_max_error_kg']
                endpoint=next(e for e in proof['endpoints'] if e['job']==row['job'])
                case['BNF_endpoint_max_kg']=endpoint['max_abs_error_kg']['all_BNF']
                case['other_initial_endpoint_max_kg']=endpoint['max_abs_error_kg']['other_sources_and_initial']
            cases.append(case)
        closed=dict(passed=True,kind=kind,actual_supplemental_cases=2,aggregate_actual_cases=8,
            cases=cases,all_exit_codes_zero=True,parent_no_longer_live=True,
            staged_leases_released=True,max_peak_commit_bytes=max(e['peak']['peak_commit'] for e in events),
            full44_completed=False,evaluation_labels_read=False,training_ownership_changed=False,
            controller_complete_sha256=sha(folder/'complete.json'),
            aggregate_reuse_controls_sha256=sha(proof_path),
            aggregate_reuse_controls_copy=str(immutable_proof),source_sha256=sha(Path(__file__)))
        write_json(folder/'verified_closeout.json',closed)
        result.append(closed)
    combined=dict(passed=True,batches=result,actual_physical_paths=8,actual_BNF_paths=8,
        full44_completed=False,evaluation_labels_read=False,training_ownership_changed=False,
        admission_controls_sha256=sha(ROOT/'evidence/additive_staged_admission_controls.json'))
    write_json(ROOT/'outputs/staged_frozen_import_002/diagnostic_closeout.json',combined)
    print(json.dumps(dict(passed=True,actual_physical_paths=8,actual_BNF_paths=8,
                         full44_completed=False,evaluation_labels_read=False)))


if __name__=='__main__':main()
