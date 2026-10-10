"""Readable interpretation of frozen comparisons; no model choice or new fit."""
import time
import pandas as pd
import native_runtime as rt
from report_source import md
R=rt.RUN
audit=rt.read(R/'reports/final_requirements_audit.json');assert audit['status']=='PASS'
c=pd.read_csv(R/'reports/source_coefficients_and_objectives.csv');selected=rt.read(R/'data/selected.json');rows=[];increments=[]
for fold in ['F23','F24']:
    for arm,pair in [('U','U-R'),('G','G-R'),('S','S-R')]:
        core=pd.read_csv(R/'reports'/pair/'1month/core_paired_table.csv');d=core[(core.fold==fold)&(core.group=='ALL')&(core.scale=='HF_day')].iloc[0];m=core[(core.fold==fold)&(core.group=='ALL')&(core.scale=='PUB_month')].iloc[0]
        rows.append(dict(评价年=2023 if fold=='F23' else 2024,配置={'U':'统一校正','G':'分组校正（主检验）','S':'分源校正'}[arm],日RMSE=d.rmse_X,日RMSE下降百分比=100*(d.rmse_R-d.rmse_X)/d.rmse_R,月报RMSE=m.rmse_X,月报NSE配对差中位数=m.nse_median_paired_change,月报NSE改善站比例=m.nse_improved_fraction,月报NSE合格配对站=int(m.nse_paired_stations)))
    for low,high in [('R','U'),('R','G'),('U','G'),('G','S')]:
        a=c[c.tag==selected[fold+'_'+low]].iloc[0];b=c[c.tag==selected[fold+'_'+high]].iloc[0]
        increments.append(dict(fold=fold,comparison=high+'-'+low,total_reduction=a.objective-b.objective,data_reduction=a.data-b.data,original_prior_reduction=a.original_prior-b.original_prior,new_prior_reduction=a.new_prior-b.new_prior,data_fraction_of_total_reduction=(a.data-b.data)/(a.objective-b.objective)))
pd.DataFrame(rows).to_csv(R/'reports/结果速览.csv',index=False,encoding='utf-8-sig');pd.DataFrame(increments).to_csv(R/'reports/objective_increment_decomposition.csv',index=False)
summary='''# 实验结论与下一阶段衔接

**来源估计校正存在预测价值，但本轮不能把倍率当作产品误差的真实反演。正式主检验“分组校正”呈现跨尺度权衡；预注册的统一校正对照表现更稳定，值得优先进入独立空间验证。** 不将统一校正追认为本轮主检验，也不替换主线。

## 1. 完成范围与数值、物理资格

12条候选拟合、4条复用基线逻辑路径均已终止并逐任务审计。16条路径物理检查通过，15条达到原坐标投影梯度≤1e-5。F24_U_s0在4000调用处停止，梯度2.0113e-5，未称数值充分；同模型最终选择的s1检查点梯度3.4308e-6，达到标准。两折共8个最终模型均通过数值、独立目标重算和完整四源账本检查；12个四角固定参数回放均已交付。

正式拟合保持来源倍率0.25—4、需求不变、H1及D29过程不变。局部最优仍存在：统一/分组的两个入口收敛到不同目标，不能把本轮最优点称为全局最优。没有因评价结果追加起点或扩大倍率范围。

## 2. 浓度改善和尺度代价

以下RMSE为站等权平均，单位mg/L。HF为15站共同日期；月报RMSE覆盖116站，NSE仅使用达到既定资格的78站（2023）/85站（2024），不能混用分母。

'''+md(pd.DataFrame(rows))+'''

原D29的HF日RMSE为2023年0.989592、2024年1.108372；月报RMSE为0.727093、0.745837。

正式分组主检验：日RMSE仅下降0.17%和2.35%，两年均9/15个HF站改善。月报2023年的NSE逐站差中位数−0.109913、32/78站改善；2024年为+0.105467、51/85站改善。总体NSE中位数之差分别−0.046107、+0.130986，不能用它替代逐站配对变化。

分组校正的H组月报RMSE两年分别增加0.114114、0.147362 mg/L；N组分别下降0.201292、0.262518 mg/L。这是可定位的组间代价，不能将全域平均改善写成所有站改善。完整改善/恶化站清单在G-R/1month/paired_station_changes.csv。

主检验事件采用2023年40事件/11站、2024年57事件/15站：对数振幅误差0.176769→0.142914和0.227846→0.203867；峰值绝对误差1.121734→0.868066和1.437568→1.317508；背景绝对误差0.703840→0.580711和1.189398→1.108111。月内中心化MSE也下降。但主检验两年的日RMSE重采样区间均跨零，2024振幅和两年背景误差变化的区间也跨零；不能称为稳定的全面改进。单月和两个月块结论方向相近，区间仅条件于冻结模型。

统一校正的日RMSE下降7.02%和16.03%，日RMSE重采样区间均不跨零；月报RMSE、事件三误差和月内中心化误差的两年总体方向均改善。它是较有继续价值的预注册对照。但2024事件指标区间仍跨零，OTHER月报NSE配对中位数两年仍为负，不能称全空间验证通过。分源校正也有预测收益，但化肥、粪肥触及0.25下界，BNF到达/接近4上界，仅登记为边界受限的探索候选。

## 3. 总量、组成与过程补偿

统一倍率为1.550093、1.504138。分组主检验的农业三源倍率为0.480088、0.472371，沉降为3.932532、3.907020；沉降虽未触边但接近4上限。这种稳定只表明在当前输入与过程下拟合偏好相近，不证明农业产品高估或沉降产品低估。

相对原D29，分组训练数据项下降约17.62%、16.53%，主要不是先验下降；但从统一升级到分组，总目标改善中只有约2.64%、20.56%来自数据项，其余主要是原参数先验下降。这解释了为什么更低训练总目标未伴随更好的留出RMSE。完整拆分见objective_increment_decomposition.csv。

主检验四角对照的HF日RMSE（基线/仅来源变化/仅过程变化/联合）为：2023年0.989592/1.072259/1.109152/0.987866；2024年1.108372/1.184474/1.241162/1.082281。单独替换任何一角均不复制联合收益，存在明显参数补偿和非线性交互；不得把联合收益全部分配给来源校正。

基线处四源局部加权方向有约93.4%—98.9%可由原30参数方向解释，剩余方向非零；高重叠不等于完全重复，也不用于否定拟合。逐参数方向余弦和残差奇异值在source_directions及source_process_direction_cosines文件中。

分组校正使评价期总输入由2.9412×10^9 kg增加至3.5530×10^9和3.5193×10^9 kg，但出口质量从5.9529×10^8、8.0699×10^8变为5.6231×10^8、8.0041×10^8 kg，并伴随损失和库存变化。需求保持原值，满足率从约99.997%/99.981%升至100%。来源增加不等于到河输出同比增加；逐源、河段、年份与历史/训练/评价分期账本已保留。

## 4. 解释边界与后续行动

本轮支持继续验证“TN约束下、依赖当前水文与过程结构的来源估计校正”，不能直接订正原产品或认证真实来源比例。2024农业源采用2023、沉降和面积采用2020、月初加载、流量支持不足、遗漏来源与站界限制均可能被倍率吸收。

下一阶段优先验证统一校正在冻结空间留出上的稳定性；分组校正优先解释H组恶化和沉降高倍率，而非扩上界；分源触边候选先补充独立来源约束。不在本轮继续拟合或替换主线。

## 5. 独立复核及偏离

除原独立重算外，最终逐项审计重建了8个最终模型的全部共同日期事件，独立重算所有比较的1000次事件振幅重采样，并核验预算、冻结输入、来源质量和终止状态。见final_requirements_audit.json（202项通过）。

最终人工复核发现事件资格表含不合格行时ratio_defined为object型，按位取反误把可定义比例标作不可定义。已显式转换为bool并重新计算10套评价；拟合、参数和冻结预测字节均不变。原错误报告保存在evidence/evaluation_boolean_v1，不作为结论。合成测试新增混合资格反例。分源浮点分账和Windows清单修复详见实际方法与偏离；没有放宽任何科学容差。
'''
(R/'reports/结论与下一阶段衔接.md').write_text(summary,encoding='utf-8')
report=R/'reports/专家诊断报告.md';report.write_text('# 专家审阅入口\n\n先读[结论与下一阶段衔接](结论与下一阶段衔接.md)，再核对下列全表。最终逐项审计见[final_requirements_audit.json](final_requirements_audit.json)。\n\n'+report.read_text(encoding='utf-8'),encoding='utf-8')
with (R/'reports/实际方法与偏离.md').open('a',encoding='utf-8') as f:f.write('\n- 最终事件汇总修复：混合资格表的object布尔列按位取反造成不可定义计数错误。显式bool转换后重算10套评价；预测、目标、参数不变。原件及修复证据见evidence/evaluation_boolean_v1与reports/evaluation_boolean_repair.json。\n')
with (R/'reports/独立完成审计.md').open('a',encoding='utf-8') as f:f.write('\n\n最终补充：final_requirements_audit.json逐项202项通过，含从原观测支持独立重建全部事件与全部1000次振幅重采样。所有最终模型数值及物理通过；16条路径中F24_U_s0未达到数值充分，已如实保留。\n')
elapsed=(time.time()-rt.read(R/'work/experiment_clock.json')['started'])/3600
(R/'README.md').write_text(f'''# D29氮源估计校正：20260921_2

**实验已完成，最终逐项审计通过；累计约{elapsed:.2f}小时。** 12条候选拟合和4条基线复用均已终止，8个最终模型通过数值与物理验收。

分组主检验存在跨时期和组间权衡。统一校正对照的日RMSE两年下降约7.02%/16.03%，更值得后续独立空间验证；不替换主检验，不认证真实来源量，不替换主线。

- [结论与下一阶段衔接](reports/结论与下一阶段衔接.md)
- [专家报告和完整表](reports/专家诊断报告.md)
- [逐项完成审计](reports/final_requirements_audit.json)
- [独立审计说明](reports/独立完成审计.md)
- [全部拟合路径](reports/path_summary.csv)
- [来源倍率](reports/source_coefficients_and_objectives.csv)
- [实际方法及修复记录](reports/实际方法与偏离.md)

评价表在reports/G-R、U-R、G-U、S-G、S-R，每组包含单月与连续两个月重采样。旧目录保持只读，未发布GitHub，未新建定时任务。
''',encoding='utf-8')
files={p.relative_to(R).as_posix():dict(sha256=rt.sha(p),bytes=p.stat().st_size) for folder in ['scripts','configs','reports'] for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
files['README.md']=dict(sha256=rt.sha(R/'README.md'),bytes=(R/'README.md').stat().st_size)
rt.write(R/'delivery_manifest.json',dict(files=files,prediction_freeze='data/prediction_freeze.json',selected=selected,final_audit='reports/final_requirements_audit.json',elapsed_hours=elapsed))
print('FINAL_NARRATIVE_SEALED',elapsed,flush=True)
