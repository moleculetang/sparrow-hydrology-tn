"""Verify and seal the private, full final experiment without starting training.

The original workstation archive stays untouched. This package contains frozen
processed inputs, checkpoints, failures, corrected statistics and final reports.
No extraction, publication, keys or environment installation is performed.
"""
import datetime
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from mltn.common import ROOT, read, write, sha
from mltn.resources import lease


def main():
    lease('final_delivery_seal', 1, 'cpu')
    required=['evidence/acceptance_Windows.json','evidence/acceptance_Linux.json',
              'evidence/independent_actual_effect_metrics.json','evidence/independent_final_protocol.json',
              'evidence/independent_HF_reading_mean.json','evidence/independent_predictions_review.json',
              'evidence/results_recovery.json','outputs/local_final_evidence_done.json','outputs/seed_repair_postprocess_done.json','evidence/joint_seed_forwarding_acceptance.json','outputs/amplitude_diagnostic/report_receipt.json']
    for f in required: assert read(ROOT/f)['passed'], ('FAILED_FINAL_GATE',f)
    roles=read(ROOT/'outputs/result_roles.json'); assert len(roles['expected'])==189 and not roles['missing']
    assert sum(x['role']=='screen' for x in roles['attempts'])==200
    signature=read(ROOT/'outputs/expert_interpretation_receipt.json')
    assert signature['interpretation_authored']
    assert sha(ROOT/signature['report'])==signature['report_sha256']
    assert '待最终独立解读签署' not in (ROOT/signature['report']).read_text(encoding='utf-8')
    links=[]
    for file in [ROOT/'README.md',*(ROOT/'reports').glob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=target.split('#',1)[0].strip('<>')
            resolved=(file.parent/target).resolve()
            assert resolved.exists(), ('BROKEN_REPORT_LINK',file,target)
            links.append(dict(source=str(file.relative_to(ROOT)),target=target))
    domain=[]
    for file in sorted((ROOT/'outputs/full_domain').glob('*.parquet')):
        q=pd.read_parquet(file)
        assert np.isfinite(q.prediction).all() and (q.prediction>=0).all()
        assert q.station_key.nunique()==230 and not q.duplicated(['station_key','date']).any()
        year=2023 if file.name.startswith('F23') else 2024
        n=12 if '_monthly_' in file.name else 365 if year==2023 else 366
        assert len(q)==230*n and q.groupby('date').size().eq(230).all()
        assert pd.to_datetime(q.date).dt.year.eq(year).all()
        domain.append(dict(file=str(file.relative_to(ROOT)),rows=len(q),sha256=sha(file),
                           interpretation='unobserved reach extrapolation, not station validation'))
    assert len(domain)==10
    pairs=read(ROOT/'outputs/evaluation/paired_summary.json');assert pairs
    actual=read(ROOT/'evidence/independent_actual_effect_metrics.json')
    checked=0
    for p in pairs:
        assert p['common_stations']>0, 'EMPTY_COMMON_SUPPORT'
        for b in [1,2]:
            file=ROOT/'outputs/evaluation'/f"{p['candidate']}__versus__{p['baseline']}_{p['task']}_bootstrap_{b}month.parquet"
            q=pd.read_parquet(file);assert len(q)==1000
            checked+=1
    write(ROOT/'evidence/final_delivery_audit.json',dict(passed=True,registered_final_paths=189,
        valid_screen_identities=200,missing=[],bootstrap_tables=checked,replicates_per_table=1000,
        no_zero_common_support=True,full_domain=domain,links=links,links_checked=len(links),
        scalar_bootstrap_scope='separate independent_final_protocol receipt; not all1000 scalar recomputations',
        scientific_boundary='no universal spatial skill, no numerical optimality certificate, no actual source/water certification'))
    now=datetime.datetime.now(datetime.timezone.utc)
    study=read(ROOT/'study.json');start=datetime.datetime.fromisoformat(study['start_utc'].replace('Z','+00:00'))
    study.update(status='complete',completed_utc=now.isoformat(),elapsed_hours=(now-start).total_seconds()/3600,
        completed_scope='finite registered ML experiment, frozen evaluation, independent recomputation and expert interpretation',
        scientific_success='partial temporal skill; spatial and waveform limitations remain',new_dispatch=False)
    write(ROOT/'study.json',study)
    write(ROOT/'outputs/completion.json',dict(status='complete',completed_utc=now.isoformat(),
        valid_screen_identities=200,registered_final_paths=189,legal_spatial_checkpoint_readouts=45,
        missing=[],reports=['reports/专家机器学习与全域验证报告.md','reports/实际方法与偏离.md','reports/模型路径与观测支持说明.md','reports/振幅控制因子数值诊断.md'],
        independent_actual_station_rows=actual["station_rows"],independent_actual_scalar_checks=actual["scalar_checks"],
        local_root=str(ROOT),remote_root=study['remote_root'],publication=False,scheduled_tasks=False,
        training='finished; no new fits',original_workstation_archive='preserved as pre-final-correction recovery snapshot',
        package='transfer/machine_learning_results_final.tar.gz; SHA256 and byte readback receipt outside archive to avoid self-reference'))
    signature['final_package_and_link_audit']='passed; final archive byte seal receipt is external'
    write(ROOT/'outputs/expert_interpretation_receipt.json',signature)
    # Only explicit experiment surfaces; never include transfer archives or .ssh.
    paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md','.sh','.ps1','.txt']]
    for directory in ['config','data','evidence','jobs','literature','logs','mltn','outputs','reports','superseded']:
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts and p.suffix not in ['.pyc','.tmp']]
    excluded={'delivery_manifest_final.json','outputs/final_delivery_receipt.json'}
    paths=sorted(set(p for p in paths if p.relative_to(ROOT).as_posix() not in excluded))
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
    write(ROOT/'delivery_manifest_final.json',dict(experiment='20260930_1',version='final local correction and interpretation',
        created_utc=now.isoformat(),private_processed_inputs=True,keys_included=False,
        original_archive_sha256=read(ROOT/'evidence/results_recovery.json')['archive_sha256'],files=manifest))
    target=ROOT/'transfer/machine_learning_results_final.tar.gz'
    partial=target.with_suffix('.gz.part')
    print('Final gates passed; packaging',len(paths),'files',flush=True)
    with tarfile.open(partial,'w:gz',compresslevel=1) as archive:
        for p in paths:archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        archive.add(ROOT/'delivery_manifest_final.json',arcname='delivery_manifest_final.json',recursive=False)
    verified=set()
    with tarfile.open(partial,'r|gz') as archive:
        for member in archive:
            key=PurePosixPath(member.name)
            assert member.isfile() and not key.is_absolute() and '..' not in key.parts
            assert member.name not in verified, 'DUPLICATE_ARCHIVE_ENTRY'
            expected=manifest.get(member.name)
            if member.name=='delivery_manifest_final.json':
                expected=dict(bytes=(ROOT/member.name).stat().st_size,sha256=sha(ROOT/member.name))
            assert expected is not None, ('UNREGISTERED_ARCHIVE_ENTRY',member.name)
            h=hashlib.sha256();size=0
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block);size+=len(block)
            assert size==expected['bytes'] and h.hexdigest()==expected['sha256'], ('ARCHIVE_BYTE_MISMATCH',member.name)
            verified.add(member.name)
    assert len(verified)==len(manifest)+1
    partial.replace(target)
    write(ROOT/'outputs/final_delivery_receipt.json',dict(passed=True,status='complete',
        archive=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha(target),
        verified_archive_files=len(verified),all_manifest_entries_read_back=True,
        manifest_sha256=sha(ROOT/'delivery_manifest_final.json'),reports={f:sha(ROOT/f) for f in read(ROOT/'outputs/completion.json')['reports']},
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        limitation='private experiment; no release, optimality or universal predictive-skill certification'))
    print('FINAL_DELIVERY_VERIFIED',target,flush=True)


if __name__=='__main__':main()
