"""Generate inspectable quantitative report only from frozen completed artifacts."""
import time
import native_runtime as rt
from pathlib import Path
import numpy as np,pandas as pd
R=rt.RUN
def table(frame,columns):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'NA' if not np.isfinite(v) else f'{v:.6g}'
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(fmt(r[k]) for k in columns)+' |' for r in frame.to_dict('records')])
def main():
    selected=rt.read(R/'data/selected.json');paths=pd.read_csv(R/'reports/path_summary.csv');rows=[]
    for key,tag in selected.items():
        short,structure,mode=key.split('_');m=rt.read(R/'outputs'/tag/'model.json');a=rt.read(R/'outputs'/tag/'audit.json')
        contrast='D-P' if mode in ['P','D'] else 'A-P';suffix='R' if mode=='P' else 'X'
        core=pd.read_csv(R/f'reports/{structure}_{contrast}/1month/core_paired_table.csv');core=core[(core.fold==short)&(core.group=='ALL')]
        e=pd.read_csv(R/f'reports/{structure}_{contrast}/1month/event_centered_summary.csv');e=e[e.fold==short].iloc[0]
        get=lambda scale,metric:float(core[core.scale==scale].iloc[0][metric+'_'+suffix])
        row=dict(fold=short,structure=structure,input=mode,tag=tag,c=float(np.exp(m['parameters'][30])),objective=m['objective'],data=m['terms']['data'],prior=m['terms']['prior'],pg=m['pg'],numerical=a['numerical_sufficient'],daily_RMSE=get('HF_day','rmse'),daily_bias=get('HF_day','bias'),daily_r=get('HF_day','r'),monthly_NSE_median=get('PUB_month','nse_median'),monthly_RMSE=get('PUB_month','rmse'),centered_MSE=float(e['centered_mse_'+suffix]),event_amplitude=float(e['amplitude_error_'+suffix]),event_peak=float(e['peak_error_'+suffix]),event_background=float(e['base_error_'+suffix]))
        if mode=='A':row['effective_agricultural_kg_N_ha_year']=row['c']*rt.read(R/'data/daily_inputs/activity_reference.json')['kappa0_kg_N_ha_year']
        rows.append(row)
    a=pd.DataFrame(rows);a.to_csv(R/'reports/selected_model_statistics.csv',index=False)
    paired=[];intervals=[]
    for structure in ['U','L3']:
        for contrast in ['D-P','A-D','A-P']:
            p=R/f'reports/{structure}_{contrast}/1month';x=pd.read_csv(p/'core_paired_table.csv');paired.append(x.assign(structure=structure,contrast=contrast));intervals.append(pd.read_csv(p/'bootstrap_intervals.csv').assign(structure=structure,contrast=contrast))
    pairs=pd.concat(paired);pairs.to_csv(R/'reports/all_core_paired_statistics.csv',index=False);pd.concat(intervals).to_csv(R/'reports/all_primary_intervals.csv',index=False)
    synth=pd.concat([pd.read_csv(p) for p in (R/'outputs/synthetic').glob('*_*/metrics.csv')],ignore_index=True);synth.to_csv(R/'reports/all_synthetic_metrics.csv',index=False)
    q=synth[(synth.fold=='F24')&(synth.target=='clean')&(synth.support=='observed_dates')&(synth.treatment!='true_input_reference')]
    s=q.groupby(['structure','input_mode','treatment'],sort=False).agg(rmse_mean=('rmse','mean'),rmse_min=('rmse','min'),rmse_max=('rmse','max'),c_mean=('c','mean'),seeds=('seed','nunique')).reset_index();s.to_csv(R/'reports/synthetic_clean_2024_summary.csv',index=False)
    controls=pd.concat([pd.read_csv(p) for p in (R/'outputs/synthetic/negative_controls').glob('*_metrics.csv')]);controls.to_csv(R/'reports/negative_control_metrics.csv',index=False)
    literature=rt.read(R/'reports/literature_audit.json');audit=rt.read(R/'reports/independent_completion_audit.json')
    text=['# 氮输入动态损失、可恢复性与活动量源强：专家报告','',
    '本轮真实比较只使用冻结H1和两类D29结构。P为现有月初输入，D为逐月同质量日展开，A为活动量代理农业N加原沉降。下面的真实评价均为已见资料的回顾性检验；合成浓度明确为虚构。','',
    '## 真实浓度、数值充分性与来源倍率','',
    table(a,['fold','structure','input','c','objective','data','pg','numerical','daily_RMSE','monthly_NSE_median','centered_MSE']), '',
    '日RMSE为站点RMSE等权均值（mg/L），月NSE为符合资格站点的中位数；月内MSE单位为(mg/L)²。两时间折的训练期分别为2021—2022、2021—2023，评价期为2023、2024。连续历史均从1961开始。','',
    '## 事件与背景','',table(a,['fold','structure','input','event_amplitude','event_peak','event_background']), '',
    '事件先站内取事件误差中位数，再站等权。F23为40事件/11站，F24为57事件/15站；幅度为对数比误差，峰值与背景为mg/L。','',
    '## 逐站配对结果','',
    table(pairs[(pairs.group=='ALL')&(pairs.scale=='PUB_month')],['fold','structure','contrast','nse_paired_stations','nse_difference_of_medians','nse_median_paired_change','nse_improved_fraction']), '',
    '总体中位数之差与配对差中位数分别报告；前者不能代替典型站改善。逐站恶化/改善清单与1000次整月抽样区间位于各比较目录，连续两个月块为敏感性。区间不包含模型重新拟合不确定性。','',
    '## 合成可恢复性','',table(s,['structure','input_mode','treatment','rmse_mean','rmse_min','rmse_max','c_mean','seeds']), '',
    '这里的RMSE是虚构无噪声2024浓度、真实观测日期支持下的误差。最小—最大为三个固定种子范围，不是置信区间；无噪声已知输入参照按构造恢复真值，不能作为现实预测上限。噪声10%、完整每日支持及合成F23结果全部在all_synthetic_metrics.csv。','',
    '## 正确输入仍不足的反例','',table(controls,['structure','control','fit','c','rmse','nse']), '',
    '反例中的水量错误和遗漏入河源由实验者已知注入。若倍率移动后拟合改善，这说明来源倍率能吸收其他错误，不能说明反演得到了真实来源。结构交叉还包含两套冻结过程参数的差异，并非纯粹只改变一个公式。','',
    '## 证据范围与交付审计','',
    f"文献去重筛查{literature['screened']}项，详细登记{literature['reviewed_cases']}个案例，另有{literature['metadata_only']}个仅元数据线索。没有大型数据下载。独立完成审计状态：{audit['status']}。",'',
    f"有限路径：{len(paths)}；独立评价模型：{len(selected)}。来源标签、局地/网络账本、预测身份和配对统计的审计文件与结果同目录保存。",'',
    '本轮不能直接认证真实输入准确性。已有资料尚缺15个HF站的同站界、同期独立流量认证；活动量A仍继承面积年份替代、原月份额和历史源强基准。后续动作必须区分改进实测依据、改变输入表达、改变输入到浓度的转换。','',
    '详见：实际方法与边界.md、文献与代码证据.md、independent_completion_audit.json。最终优先级解释另见“结论与下一步.md”。','']
    (R/'reports/专家报告.md').write_text('\n'.join(text),encoding='utf-8')
    print('REPORT GENERATED',flush=True)
if __name__=='__main__':main()
