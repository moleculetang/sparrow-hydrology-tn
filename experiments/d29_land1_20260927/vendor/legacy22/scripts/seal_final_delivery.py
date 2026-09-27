"""Seal reviewed artifacts, verify immutable predictions and record terminal process identities."""
import json,time,shutil
import native_runtime as rt
R=rt.RUN

def main():
    previous=R/'evidence/delivery_manifest_before_final_review.json'
    if not previous.exists():shutil.copyfile(R/'delivery_manifest.json',previous)
    for manifest in [R/'data/prediction_freeze.json',R/'reports/lowrank_freeze.json']:
        for p,h in rt.read(manifest)['files'].items():assert rt.sha(R/p)==h,('FROZEN_CHANGED',p)
    for p,h in rt.read(R/'evidence/inherited_manifest.json').items():assert rt.sha(p)==h
    for p in (R/'reports/launch_by_fold').glob('*.json'):
        for name,h in rt.read(p)['frozen_hashes'].items():assert rt.sha(R/name)==h
    state=rt.read(R/'work/controller_status.json');assert state['status']=='COMPLETE_EXPERIMENTAL' and state['exit_code']==0
    identities=[state['process']]
    for p in (R/'work/jobs').glob('*/status.json'):
        s=rt.read(p)
        assert s['status'] not in ['RUNNING','PENDING']
        if s.get('process'):identities.append(s['process'])
    checks=[]
    for x in identities:
        try:
            p=rt.process(x['pid'],x['created']);assert not p['alive'],('STILL_RUNNING',p);reason='exited'
        except OSError as e:
            if e.winerror!=87:raise
            reason='PID absent'
        except RuntimeError as e:
            if str(e)!='PID_CREATION_MISMATCH':raise
            reason='PID reused; original identity ended'
        checks.append(dict(pid=x['pid'],created=x['created'],terminal_reason=reason))
    qa=rt.read(R/'figures/figure_qa.json')
    qa.update(status='PASS_VISUAL_REVIEW',overview_reviewed=True,reviewed_event_pages=[1,40,41,97],event_pdf_pages=97,station_names_rendered=True,fix='Explicit Microsoft YaHei font for Chinese event station titles; numerical data unchanged',review_scope='Overview and four representative event pages visually inspected; all 97 pages generated from frozen event curves',time=time.time())
    rt.write(R/'figures/figure_qa.json',qa)
    audit=rt.read(R/'reports/completion_audit.json');assert all(c['passed'] for c in audit['checks']) and audit['selected']==14 and audit['paths']==28
    audit.update(final_visual_review='PASS',final_interpretation='结果解读与最终结论.md',all_registered_processes_ended=True,delivery_reviewed=time.time())
    rt.write(R/'reports/completion_audit.json',audit)
    p=R/'独立完成审计.md';s=p.read_text(encoding='utf-8').replace('图像生成后仍须人工视觉检查，结果单列于 figures/figure_qa.json。','总览与事件PDF第1、40、41、97页已完成视觉检查，修复中文站名显示；97页均已生成，未声称逐页目视检查。结果见 figures/figure_qa.json。')
    p.write_text(s,encoding='utf-8')
    rt.write(R/'reports/final_delivery_checks.json',dict(status='PASS',predictions_unchanged=True,lowrank_freeze_unchanged=True,fit_identities_unchanged=True,old_experiments_unchanged=True,processes=checks,controller_status=state['status'],visual_qa=True,science_configs_unchanged=True,finished=time.time()))
    files={p.relative_to(R).as_posix():rt.sha(p) for folder in ['scripts','configs','reports','figures','data','diagnostics','outputs'] for p in (R/folder).rglob('*') if p.is_file() and p.suffix!='.pyc'}
    files.update({p.relative_to(R).as_posix():rt.sha(p) for p in R.glob('*.md')})
    rt.write(R/'delivery_manifest.json',dict(created=time.time(),files=files,status='AUDITED_WITH_NUMERICAL_LIMITATIONS',notes='Two selected neural points below numerical sufficiency; all physical and arithmetic checks passed.'))
    written=rt.read(R/'delivery_manifest.json');assert all(rt.sha(R/p)==h for p,h in written['files'].items())
    print('FINAL DELIVERY PASS',len(files),'hashed files;',len(checks),'ended process identities',flush=True)

if __name__=='__main__':main()
