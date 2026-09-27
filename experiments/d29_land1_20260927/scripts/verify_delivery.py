"""Bounded closeout checks; report failures without weakening scientific gates."""
from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
OUT=ROOT/'outputs/final_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def record(name,passed,**kw):checks.append(dict(name=name,passed=bool(passed),**kw))
reports=list((ROOT/'reports').glob('*.md'))
broken=[]
for p in reports:
    content=p.read_text(encoding='utf-8-sig')
    for target in re.findall(r'\]\(<?([^\n)]+?)>?\)',content):
        target=target.strip('<>')
        if target.startswith(('http:','https:')):continue
        if target.startswith('/E:/'):target=target[1:]
        target=re.sub(r':\d+$','',target)
        q=Path(target)
        if not q.is_absolute():q=p.parent/q
        if not q.exists():broken.append(dict(report=str(p),target=str(q)))
record('all_local_report_links_resolve',not broken,broken=broken)
g=json.loads((OUT/'gap_crosswalk_37.json').read_text(encoding='utf-8'))
record('37_unique_reviewed_no_false_scientific_close',len({x['id'] for x in g['items']})==37 and all(not x['scientific_closed'] for x in g['items']))
r=json.loads((OUT/'completion_receipt.json').read_text(encoding='utf-8'))
record('failed_mass_gate_and_formal_block_preserved',not r['strict_mass_pass'] and not r['formal_calibration_allowed'] and r['formal_paths_started']==0)
record('frozen_parent_and_actual_implementation_preserved',r['old_snapshot_all_unchanged'] and r['run_implementation_unchanged'])
record('72_tests_current_log_pass',r['inherited_72_tests_passed'])
for name in ['precision_validation_passed','potential_activity_validation_passed','input_contract_validation_passed']:record(name,r[name])
chain=json.loads((ROOT/'outputs/chain/receipt.json').read_text(encoding='utf-8'))
record('full_history_synthetic_directions_pass',chain['days']==23376 and all(x['passed'] for x in chain['gradient_directions']))
for year in (2023,2024):
    s=pd.read_csv(ROOT/f'outputs/supply_limited_reference/evaluation/{year}_daily_station.csv')
    summary=pd.read_csv(ROOT/f'outputs/supply_limited_reference/evaluation/{year}_daily_summary.csv').set_index('metric')
    for metric in ('nse','month_centered_nse'):
        status='nse_status' if metric=='nse' else 'month_centered_status'
        p=s[s['baseline_'+status].eq('defined')&s['candidate_'+status].eq('defined')]
        a=p['baseline_'+metric];b=p['candidate_'+metric];rr=summary.loc[metric]
        values=[len(p),a.median(),b.median(),b.median()-a.median(),(b-a).median(),float((b>a).mean())]
        expected=rr[['n_common_stations','baseline_median','candidate_median','difference_of_medians','median_paired_difference','improvement_fraction']].astype(float).to_numpy()
        record(f'{year}_{metric}_paired_summary_recomputed',np.allclose(values,expected,rtol=1e-12,atol=1e-12))
    for block in (1,2):
        b=pd.read_csv(ROOT/f'outputs/supply_limited_reference/evaluation/{year}_bootstrap_{block}month_intervals.csv')
        record(f'{year}_{block}month_bootstrap_identity',(b.n_requested_replicates==1000).all() and (b.seed==1729).all())
stocks=np.load(ROOT/'outputs/supply_limited_reference/annual_end_stocks.npy',mmap_mode='r')
record('annual_stocks_finite_nonnegative',np.isfinite(stocks).all() and (stocks>=0).all())
precision=dict(shape=list(stocks.shape),maximum_annual_stock_kg=float(stocks.max()),
              maximum_float64_spacing_kg=float(np.spacing(stocks).max()),
              entries_spacing_exceeds_1e_6=int((np.spacing(stocks)>1e-6).sum()),
              meaning='Representation spacing describes roundoff scale, not permission to change registered absolute tolerance.')
(OUT/'real_stock_precision_scale.json').write_text(json.dumps(precision,ensure_ascii=False,indent=2),encoding='utf-8')
v=dict(passed=all(x['passed'] for x in checks),checks=checks,report_sha256={str(p):sha(p) for p in reports},
       not_a_scientific_success_certificate=True,NSE='reported separately; verification quantities not applicable')
(OUT/'delivery_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=v['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],precision=precision),ensure_ascii=False))
if not v['passed']:raise SystemExit(1)
