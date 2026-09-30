"""Post-freeze interpretation and compact evidence tables; no fit or selection changes."""
import json,time
import numpy as np
import pandas as pd
import native_runtime as rt
from report_regional import md
R=rt.RUN

def main():
    core=pd.read_csv(R/'reports/all_comparison_core.csv')
    event=pd.read_csv(R/'reports/all_event_summary.csv')
    boot=pd.read_csv(R/'reports/all_bootstrap_intervals.csv')
    models=pd.read_csv(R/'reports/selected_models.csv')
    primary=core[core.group.eq('ALL')&core.comparison.isin(['L3-U','S56_L3-U','S113_L3-U','S191_L3-U'])].copy()
    primary['rmse_percent_change']=100*(primary.rmse_X/primary.rmse_R-1)
    primary.to_csv(R/'reports/final_primary_summary.csv',index=False)
    compress=[]
    for root in (R/'diagnostics/lowrank').glob('*/*/*'):
        if not (root/'training.npz').exists():continue
        a=np.load(root/'training.npz');y=a['values'];w=a['weights'];mask=w>0;total=np.sum(w[mask]*y[mask]**2)
        for rank in [1,2,3]:
            z=np.load(root/f'rank{rank}.npz');error=float(z['history'][-1])
            compress.append(dict(fold=root.parts[-3],scale=root.parts[-2],target=root.parts[-1],rank=rank,observations=int(mask.sum()),weighted_zero_error=float(total),weighted_reconstruction_error=error,training_reduction_fraction=1-error/total))
    comp=pd.DataFrame(compress);comp.to_csv(R/'reports/lowrank_compressibility.csv',index=False)
    stations=pd.read_csv(R/'reports/L3-U/1month/paired_station_changes.csv')
    daily=stations[stations.scale.eq('HF_day')][['fold','station_key','rmse_R','rmse_X','delta_rmse']]
    neural=[]
    for fold in ['F23','F24']:
        d=rt.read(R/'reports'/f'{fold}_neural_zero_output_directions.json')
        v=[x['linear_parent_explained_fraction'] for x in d['rows']]
        neural.append(dict(fold=fold,minimum=min(v),median=float(np.median(v)),maximum=max(v)))
    pd.DataFrame(neural).to_csv(R/'reports/neural_parent_overlap_summary.csv',index=False)
    physics=pd.read_csv(R/'reports/physical_period_summary.csv')
    phys=physics[physics.period.eq('evaluation')&physics.model.isin(['F23_U','F23_L3','F24_U','F24_L3'])]
    temporal=primary[primary.fold.isin(['F23','F24'])]
    eventmain=event[event.comparison.eq('L3-U')]
    intervals=boot[boot.comparison.eq('L3-U')&boot.scale.isin(['HF_day','HF_centered'])&boot.metric.isin(['rmse_change','centered_mse_change'])]
    table=temporal[['fold','scale','paired_stations','nse_paired_stations','rmse_R','rmse_X','rmse_percent_change','rmse_median_paired_change','rmse_improved_fraction','nse_median_R','nse_median_X','nse_difference_of_medians','nse_median_paired_change','nse_improved_fraction']]
    text='''# 最终结果解读：平均误差改善，但事件与空间收益未成立

本轮已完成全部28条逻辑路径（24条新拟合、4条核验复用），14个模型折全部交付。完整物理、来源、独立目标和评价表重算均通过；12个最终选择点数值充分，两个神经选择点不充分。没有追加起点或扩大参数范围。

**正式主模型获得有限的跨年平均浓度预测收益，但没有解决事件动态，也没有在三个冻结支流留出中保持日尺度收益。因此本轮不足以支持其作为可泛化的改进方案继续扩展或替换主线。** 这是对本次固定表达和测试范围的判断，不是否定所有低维学习。

## 1. 真实浓度改善多少？

相对统一来源校正D29，三模式线性主模型的15站等权平均日RMSE：2023年由0.920163降至0.873433 mg/L（−5.08%），2024年由0.930710降至0.884673 mg/L（−4.95%）。改善站分别为11/15和8/15；2024逐站差中位数仅−0.008257 mg/L，平均收益分布不均。两年HF日NSE中位数仍为负，绝对预测水平仍有限。

下表中RMSE为站均值；NSE仅使用满足覆盖和正方差的共同合格站。116站月面板分别只有78、85站满足NSE资格，不能将116误写为NSE配对站数。

'''+md(table)+'''

2024月报的“NSE总体中位数之差”为−0.010278，但“逐站NSE差中位数”为+0.087181，且51/85站NSE改善。两个量确实方向不同，不能互相代替。2023对应为+0.397666、+0.158351，50/78站改善。

全部15站日误差配对如下；负值为改善。2024收益较大的是厂房大桥（−0.331928）、盘溪大桥（−0.273892）和叮当（−0.108550 mg/L）；自良渡口恶化+0.195586 mg/L。

'''+md(daily)+'''

## 2. 代价落在哪个尺度？

两年事件振幅、峰值及背景的站等权误差均上升。2024月内中心化MSE由0.410896升至0.472782 (mg/L)²（+15.06%）；2023仅由0.425934降至0.421032（−1.15%）。因此日/月平均RMSE改善不能解读为事件响应得到修复。

'''+md(eventmain[['fold','stations','events','amplitude_error_R','amplitude_error_X','peak_error_R','peak_error_X','base_error_R','base_error_X','centered_mse_R','centered_mse_X']])+'''

1000次冻结模型配对月份重采样的2.5%—97.5%区间如下。F24日RMSE下降和月内MSE上升在单月、连续两月块下均不跨零；F23日RMSE的单月区间微跨零，两月块不跨零。两折事件三误差的区间均跨零，因此报告点估计代价及不稳定性，不宣称事件恶化已被确认。

'''+md(intervals[['fold','scale','metric','block_months','lower','upper','crosses_zero']])+'''

## 3. 能否保持空间收益？

不能。S56、S113、S191的站均日RMSE分别由0.379679→0.386077、0.323861→0.417516、3.874086→4.047858 mg/L，变化+1.69%、+28.92%、+4.49%；改善站为1/2、0/1、0/4。月报部分指标改善，但不能抵消日/事件绝对浓度代价。HF支持仅2、1、4站，应限定到这些被观测支持的留出站，不推断全部230河段。

'''+md(primary[primary.fold.str.startswith('S')][['fold','scale','paired_stations','nse_paired_stations','rmse_R','rmse_X','rmse_median_paired_change','rmse_improved_fraction','nse_difference_of_medians','nse_median_paired_change']])+'''

## 4. 观测是否可压缩，输入能否预测这些模式？

训练观测子场有可压缩部分。下表为掩码、原训练权重和站点标准化下，相对零异常重建的训练平方误差下降比例；它不是留出解释方差，也不是230河段真实浓度场的可识别性证据。三模式对日异常保留约70%—72%的训练加权变化；月异常约43%—49%，三模式并未覆盖全部变化。

'''+md(comp[['fold','scale','target','rank','observations','training_reduction_fraction']])+'''

可重建没有转化为稳定的输入预测增益。相对同32维输入的不限秩岭回归，三模式日异常预测的逐站log-RMSE差中位数为F23 −0.0000378、F24 +0.0000907；日基线残差预测则为F23 +0.0000256、F24 −0.0002958。月尺度也存在时期反转。改进幅度很小且不一致，不能据训练重建效果宣称现有输入已能预测这些模式。评价期TN辅助重建单列为重建诊断，从未作为预测成绩。

## 5. 地区响应、非线性与参数补偿

三模式主模型的训练数据项F23由0.967725降至0.892802，F24由0.985987降至0.906156，分别下降7.74%、8.10%；并非仅靠先验下降。来源倍率为1.54239和1.53297，基线为1.55009和1.50414，均未触边。来源校正被保留，但这些倍率仍是当前H1和过程结构条件下的估计校正，不是真实排放量。

24个区域系数方向的原31参数局部投影解释比例中位数为41.50%、37.20%；投影后数值秩均为24。它们提供了不同的局部预测方向，但不保证空间迁移有效。

神经探索的F23日RMSE为0.895896、F24为0.937181 mg/L，均差于线性主模型；F24甚至差于基线。四条神经路径均达到8000次调用限制；最终两点投影梯度为5.37e−4、1.60e−3，高于1e−5，不能宣布充分优化。零输出锚点的6个V方向与55参数线性父空间高度重叠（下表），不代表训练后的全局函数等价。H、a在V=0处梯度为零是结构性质。当前证据不支持额外神经复杂度的继续价值，也不能将有限预算结果当成全部神经表达的失败。

'''+md(pd.DataFrame(neural))+'''

评价年物理账本如下，单位kg N。来源、损失、库存和出口共同变化，不把全部收益唯一归因于地区响应。汇总需求满足率均为1；这不认证真实作物需求准确。

'''+md(phys[['model','source_multiplier','source_kg','uptake_kg','demand_satisfaction','mineral_loss_kg','channel_loss_kg','fast_kg','slow_kg','M_end_kg','L_end_kg','terminal_kg']])+'''

## 交付判断

- 观测可压缩：训练子场部分支持；不认证全河网日场。
- 现有输入预测低维模式：未见稳定、实质性的跨年新增收益。
- 区域响应增加浓度信息：局部方向及训练数据项支持；时间平均误差有有限收益，事件代价明确。
- 神经非线性：本次预算内未提供优于线性主模型的稳定日尺度收益，且求解不充分。
- 冻结支流留出：主模型未保持日尺度收益。本轮不建议据此扩大结构或替换生产主线。

所有评价是已见资料的回顾性检验；重采样区间条件于已冻结模型。实验到此结束，未启动新校准、网格、GitHub发布或后续实验。
'''
    (R/'结果解读与最终结论.md').write_text(text,encoding='utf-8')
    expert=R/'专家诊断报告.md';s=expert.read_text(encoding='utf-8')
    note='**最终判读：两年平均日RMSE约下降5%，但事件误差增加，三个空间留出的日RMSE均恶化；不支持作为可泛化改进替换主线。详见[结果解读与最终结论](结果解读与最终结论.md)。**\n\n'
    if note not in s:expert.write_text(s.replace('\n\n','\n\n'+note,1),encoding='utf-8')
    readme=R/'README.md';s=readme.read_text(encoding='utf-8')
    link='优先阅读：[结果解读与最终结论](结果解读与最终结论.md)，其中包含五个科学问题的定量答案。\n\n'
    if link not in s:readme.write_text(s.replace('\n\n','\n\n'+link,1),encoding='utf-8')
    rt.write(R/'reports/final_interpretation.json',dict(time=time.time(),primary='limited temporal mean-error gain; event and spatial costs; no promotion',new_fits=24,logical_paths=28,selected=14,numerically_sufficient_selected=12,neural_budget_limited=True,no_new_experiment=True))
    print('Final scientific review written',flush=True)

if __name__=='__main__':main()
