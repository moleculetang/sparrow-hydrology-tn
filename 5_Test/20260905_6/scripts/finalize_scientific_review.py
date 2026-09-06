"""Record the completed scientific review and verify delivery artifacts."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now
import numpy as np
import pandas as pd


def main():
    run=ROOT/'5_Test/20260905_6';sources={}
    def read(relative):
        path=ROOT/'5_Test'/relative;sources[str(path)]=sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    contract=read('20260905_1/experiment_contract.json')
    audit=read('20260905_6/reports/completion_audit.json')
    assert audit['completed_count']==audit['requirement_count']==22 and all(r['passed'] for r in audit['checks'])
    manifest=read('20260905_6/reports/final_report_manifest.json')
    assert all(sha256(item['path'])==item['sha256'] for item in manifest['sources'])
    sources.pop(str(run/'reports/final_report_manifest.json'))
    sources.update({item['path']:item['sha256'] for item in manifest['sources']})
    source_manifest=read('20260905_1/reports/input_manifest.json')
    assert len(source_manifest['files'])==31 and all(sha256(r['path'])==r['sha256'] for r in source_manifest['files'])
    spatial=read('20260905_5/reports/nested_validation_summary.json')
    temporal=read('20260905_6/reports/temporal_confirmation.json')
    prior=read('20260905_5/reports/prior_sensitivity.json')
    ident=read('20260905_5/reports/identifiability.json')
    evidence=read('20260905_6/reports/temporal_evidence_audit.json')
    assert evidence['status']=='PASS_COMPLETE_TEMPORAL_EVIDENCE' and not evidence['errors']
    assert all(sha256(p)==h for p,h in evidence['sources'].items())
    assert len(spatial['selections'])==8 and set(temporal)=={'2024','2025'} and len(prior['results'])==4
    assert len(ident['profiles'])==11 and all(p['converged'] and p['delta_generalized_energy']>=-1e-5 for p in ident['profiles'])
    assert not spatial['comparisons']['tree']['passes_nonregression']
    counts={stage:len(list((ROOT/'5_Test'/stage/'reports').glob('*_selected.json'))) for stage in ['20260905_4','20260905_5','20260905_6']}
    assert counts=={'20260905_4':16,'20260905_5':180,'20260905_6':6}
    delivery=[]
    for label,end,rows,train_rows in [('f24',2024,176640,8785),('f25',2025,179400,9850)]:
        export=read(f'20260905_6/reports/{label}_export_audit.json')
        fitpath=Path(export['selected_report']['path'])
        fit=json.loads(fitpath.read_text(encoding='utf-8'))
        assert sha256(fitpath)==export['selected_report']['sha256']
        assert {g['parameter'] for g in export['gradient_checks']}==set(fit['parameters'])
        assert all(g['passed'] for g in export['gradient_checks']) and export['process_decoded_without_TN_history']
        assert export['train_rows']==train_rows and export['fitted_metrics_are_not_validation'] and not export['promotion']
        assert export['independent_decoder_sha256']==sha256(ROOT/'5_Test/20260905_4/scripts/inference_only.py')
        for item in export['exports']:
            assert sha256(item['path'])==item['sha256'];sources[item['path']]=item['sha256']
        monthly=pd.read_parquet(run/'outputs'/f'{label}_reach_monthly_1961_{end}.parquet')
        assert len(monthly)==rows and set(monthly.reach_id)==set(range(1,231))
        assert not monthly.duplicated(['year','month','reach_id']).any()
        assert set(zip(monthly.year,monthly.month))=={(y,m) for y in range(1961,end+1) for m in range(1,13)}
        assert monthly.product_status.eq('EXPERIMENTAL_NOT_PROMOTED').all()
        water=monthly.reach_official_boundary_water_m3.to_numpy();mass=monthly.reach_official_boundary_load_kg_n.to_numpy()
        assert (water>=0).all() and (mass>=0).all()
        np.testing.assert_array_equal(monthly.dry_month,water<=0)
        assert monthly.loc[monthly.dry_month,'tn_mg_l'].isna().all()
        assert np.isfinite(monthly.loc[~monthly.dry_month,'tn_mg_l']).all()
        np.testing.assert_allclose(monthly.loc[~monthly.dry_month,'tn_mg_l'],1000*mass[water>0]/water[water>0],rtol=1e-12,atol=1e-12)
        assert not ((water<=0)&(mass>contract['numeric']['mass_atol_kg_n'])).any()
        assert monthly.loc[monthly.year.lt(2025),'quality_flags'].eq('').all()
        if end==2025:
            assert monthly.loc[monthly.year.eq(2025),'quality_flags'].map(lambda s:set(contract['final_fits']['F25_flags'])<=set(s.split(';'))).all()
        parameter=pd.read_parquet(run/'outputs'/f'{label}_process_parameters.parquet')
        assert len(parameter)==230 and set(parameter.reach_id)==set(range(1,231))
        assert np.isfinite(parameter.select_dtypes('number')).all().all()
        assert parameter.tau_mineral_days.between(*contract['legacy']['tau_days_bounds']).all()
        reservoir=pd.read_parquet(run/'outputs'/f'{label}_reservoir_ledger.parquet')
        assert not reservoir.duplicated(['year','month','reservoir_id']).any()
        assert reservoir[['captured_kg_n','released_kg_n','stock_end_kg_n']].ge(0).all().all()
        delivery.append(dict(label=label,monthly_rows=rows,parameter_reaches=len(parameter),dry_months=int(monthly.dry_month.sum()),
            reservoir_rows=len(reservoir),checked_gradient_parameters=len(export['gradient_checks']),all_products_experimental=True))
    requirements=[
        ('同一空间参数框架及全域TN推断','共享H7环境映射生成全部230个reach的过程参数；独立移除TN历史推断已通过。','20260905_4/reports/inference_only_validation.json'),
        ('水文单向显式驱动TN','锁定水文的快慢水、储水、下渗和路由直接驱动TN；水量与氮账本逐日回放通过。','20260905_2/reports/water_boundary_replay.json'),
        ('拟合器作用于参数且没有输出趋势校正','新模型为过程参数优化与双库存递推；旧输出头仅保留在独立对照。','20260905_4/process_equations.md'),
        ('Bayes约束与敏感性','完成带先验的generalized Bayes/MAP及0.5/2倍尺度试验；未声称完整后验。','20260905_5/reports/prior_sensitivity.json'),
        ('保留legacy','矿质氮lifetime与慢水氮库存均保留，未引入Active/Fresh有机库。','20260905_4/process_equations.md'),
        ('数值漏洞与全参数导数','修复真实日历路由、首日下层分母与vf梯度脱离；最终所有参数有限差分、质量守恒通过。','20260905_6/reports/f25_export_audit.json'),
        ('单站目标及完整比较','140开发、80结构、880嵌套、20年度确认、20先验、10最终起点全部执行；另有11点剖面。','20260905_6/reports/completion_audit.json'),
        ('文献和源码支撑','核查MPR/可微过程/多站目标/氮遗留/SWAT+等资料，区分元数据、摘要、全文及源码；没有理论上限宣称。','20260905_1/reports/literature_and_history.md'),
        ('点源失败历史','构建通过、12模型预测点估计均变差且非劣性未通过；不等同12个显著恶化检验；本轮保持排除。','20260905_1/reports/point_source_history_recheck.json'),
        ('全域导出及科学结论','F24/F25全部230个reach已导出，精度未通过且不替换主线。','20260905_6/reports/completion_audit.json')]
    for _,_,relative in requirements:
        path=ROOT/'5_Test'/relative;assert path.exists();sources[str(path)]=sha256(path)
    draft=run/'final_report_draft.md'
    assert sha256(draft)==manifest.get('draft_sha256',manifest['report_sha256'])
    text=draft.read_text(encoding='utf-8')
    text=text.replace('# 20260905 TN实验汇总（待最终科学复核）','# 20260905 TN实验最终报告')
    text=text.replace('未通过本轮升级门槛，保持实验状态。','本轮注册实验已完整执行并完成科学复核；结构统一和数值修复已完成，但TN预测精度问题尚未解决，未通过升级门槛。')
    text=text.replace('当前数值/产物验收为22/22项通过；本文件尚待逐项科学复核。','数值/产物验收22/22项通过；进一步复核了实际导出中的全域覆盖、质量标记、浓度计算、参数范围和全部参数梯度覆盖。共1150个计划起点与11项参数剖面，不把恢复迭代重复计为独立起点。')
    text=text.replace('核对完整观测、排除域及训练年份。\n## 4.','核对完整观测、排除域及训练年份。\n\n## 4.')
    insert='''
## 科学复核结论

区域化与过程拟合已经可以在没有站点TN历史的reach执行，但这种结构一致性没有转化为可靠的迁移精度。reach留出的两个模型NSE中位数之差为+0.3831，按站95%区间[-0.0872, 0.8917]跨零；逐站差值的中位数为-0.1017，这两个统计量不能混用。终端树留出中位数差为-1.6248，站点区间[-3.3805, -0.0735]；仅3棵树的聚类区间精度有限。终端树log-RMSE增加约10.9%，超过注册的5%容许值。

开发期NSE改善不能解释为已捕捉真实时间变化：开发站内相关中位数约-0.013，2024/2025分别为-0.360/-0.220。减小错误波动及均值变化可提高部分NSE，但不等于动态过程已被正确识别。2024有1154条主评价记录、118站，其中86站可计算NSE；2025有1065条、118站，其中88站可计算NSE，其余站不足8条。年度中位数下降的按站重采样区间跨零，因此不能把两年点估计变差写成均已统计显著变差。

先验尺度0.5意味着更强的收缩，2意味着更弱的收缩。弱化收缩在已测试的2023/2024中改善了点估计，但中位NSE仍为负，且这些诊断没有用于更改已冻结候选。F24的eta_upper=1触及上界，原始梯度约-0.2052；这是有界最优下向界外下降的压力，并非无约束梯度为零。自由参数面的局部曲率正定、条件数约2612以及11个剖面收敛，都不能证明全局唯一识别，也不构成贝叶斯可信区间。保持原边界完成本轮，不能用确认期表现事后放宽边界并冒充原验证。

本轮不能证明“两库存结构已经达到理论上限”。数值修复的固定参数影响多数较小，而时间相关、跨水系迁移、先验/边界敏感性与观测统计口径仍有问题。下一轮应先核对原始采样日期、样品类型及月浓度聚合方式，并以有来源的源项时序和固定留出设计检验缺失的动态信号；若重新考虑温度过程，应在现有lifetime损失账本内重新注册跨年检验，不把硝态氮形态转移直接当TN消失。不依据本轮结果重开已经失败的估算WWTP账本，也不恢复局地输出趋势校正。

技术路线中的历史记忆由1961年以来的源输入、冻结水文和两种氮库存递推表达；拟合的是决定响应强度及记忆长度的共享参数。训练的站点归一化MSE是面向单站表现的可微替代目标，候选选择使用单站NSE中位数；并未声称直接对不可平滑的中位数做逐步优化。

'''
    text=text.replace('## 1. 实际完成范围',insert+'## 1. 实际完成范围')
    final=run/'final_report.md';final.write_text(text,encoding='utf-8')
    matrix=[dict(requirement=name,conclusion=conclusion,evidence=str(ROOT/'5_Test'/relative),status='EXECUTED_AND_REVIEWED') for name,conclusion,relative in requirements]
    review=dict(status='EXPERIMENT_COMPLETE_PRECISION_NOT_ACHIEVED',reviewer='Codex root scientific review',created_utc=utc_now(),runtime=RUNTIME,
        contract_sha256=sha256(ROOT/'5_Test/20260905_1/experiment_contract.json'),planned_starts=1150,nuisance_profiles=11,
        requirement_review=matrix,delivery_checks=delivery,sources=sources,code_sha256=sha256(Path(__file__)),
        scientific_conclusion='Structural/numerical requirements completed; TN accuracy and terminal-tree transfer failed. No promotion and no proof of theoretical structural limit.',
        promotion=False,remaining_registered_experiments=[],unresolved_scientific_limitations=['poor predictive dynamics','terminal-tree transfer regression','prior and boundary sensitivity','unverified observation aggregation','2025 sensitivity confounding','reach17 positioning and open-lake domain'])
    atomic_json(review,run/'reports/scientific_review.json')
    atomic_json(dict(status='FINAL_SCIENTIFIC_REVIEW_COMPLETE_EXPERIMENTAL_NOT_PROMOTED',created_utc=utc_now(),runtime=RUNTIME,
        report_path=str(final),report_sha256=sha256(final),draft_path=str(draft),draft_sha256=sha256(draft),
        scientific_review_path=str(run/'reports/scientific_review.json'),scientific_review_sha256=sha256(run/'reports/scientific_review.json'),
        sources=[dict(path=p,sha256=h) for p,h in sources.items()],promotion=False),run/'reports/final_report_manifest.json')
    print('FINAL_SCIENTIFIC_REVIEW_COMPLETE',1150,11,delivery,flush=True)


if __name__=='__main__':main()
