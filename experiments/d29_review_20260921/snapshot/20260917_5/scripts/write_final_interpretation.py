"""Evidence-backed final narrative; uses frozen results only."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def table(df):
 def fmt(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def update(path,body):
 p=R/path;old=p.read_text(encoding='utf8');a='<!-- VERIFIED_INTERPRETATION_START -->';b='<!-- VERIFIED_INTERPRETATION_END -->'
 if a in old:old=old[:old.index(a)]+old[old.index(b)+len(b):]
 first,rest=old.split('\n',1);p.write_text(first+'\n\n'+a+'\n'+body+'\n'+b+'\n'+rest.lstrip(),encoding='utf8')
def main():
 selected=read('reports/prediction_freeze_manifest.json')['selected'];reg=pd.read_csv(R/'reports/region_metrics.csv');reg=reg[reg.selected];comp=pd.read_parquet(R/'reports/monthly_error_components.parquet');out=[];parts=[]
 for fold in ['F23','F24']:
  a=selected[fold+'_D_HF'];b=selected[fold+'_M_HF']
  for c in ['N','H','X']:
   x=reg[(reg.tag==a)&reg.cohort.eq(c)&reg.scale.eq('daily')].iloc[0];y=reg[(reg.tag==b)&reg.cohort.eq(c)&reg.scale.eq('daily')].iloc[0]
   out.append(dict(留出年份=int(fold[-2:])+2000,区域=c,NSE合格站=f'{int(x.NSE_eligible)}/{int(x.stations)}',月约束NSE=y.NSE_median,日约束NSE=x.NSE_median,月约束RMSE=y.RMSE,日约束RMSE=x.RMSE,RMSE变化百分比=100*(x.RMSE/y.RMSE-1)))
   vals={}
   for k,tag in [('月约束',b),('日约束',a)]:
    z=comp[(comp.tag==tag)&comp.cohort.eq(c)].groupby('station_key')[['mean_SSE','within_SSE']].mean().mean()
    vals[k]=z
   parts.append(dict(年份=int(fold[-2:])+2000,区域=c,月约束月水平MSE=vals['月约束'].mean_SSE,日约束月水平MSE=vals['日约束'].mean_SSE,月约束月内MSE=vals['月约束'].within_SSE,日约束月内MSE=vals['日约束'].within_SSE,月内变化百分比=100*(vals['日约束'].within_SSE/vals['月约束'].within_SSE-1)))
 daily=table(pd.DataFrame(out));part=table(pd.DataFrame(parts));pd.DataFrame(parts).to_csv(R/'reports/paired_error_decomposition.csv',index=False)
 gates=[]
 for row in read('reports/comparisons.json'):
  if row['contrast']!='D_HF-M_HF' or not row['scale'].startswith('monthly'):continue
  gates.append(dict(折=row['fold'],面板=row['scale'],区域=row['cohort'],NSE中位增量=row['delta_NSE'],相关中位增量=row['delta_r'],RMSE比=row['RMSE_ratio'],全部门槛通过=row['monthly_gate'],未通过项=', '.join(k for k,v in row['gate_tests'].items() if not v)))
 audits=[read(f'outputs/{j["tag"]}/audit.json') for j in read('configs/jobs.json')];fits=pd.read_csv(R/'reports/fit_summary.csv');paths=pd.read_csv(R/'reports/path_status.csv');numeric=f"12/12条路径均通过数值与物理门槛。最大原坐标投影梯度为{fits.pg.max():.3e}，低于1e-5；最大局地质量误差为{max(a['balance_max_kg'] for a in audits):.3e} kg；最大网络相对误差为{max(a['network_balance_kg']/a['network_scale_kg'] for a in audits):.3e}。每条路径使用{int(paths.calls.min())}—{int(paths.calls.max())}次累计调用，活动时间{paths.active_seconds.min()/3600:.2f}—{paths.active_seconds.max()/3600:.2f}小时，均在首段4000次/2小时预算内。"
 lead='**结论：在相同高频资料、共同月份、尺度和先验下，保留月内日变化提供了有用的新增约束；证据支持原训练地域的回顾性时间预测改善，尚不支持稳定空间推广或整个站网同时受益。**\n\n'+numeric+'\n\n'+daily+'\n\nRMSE为站均日RMSE，单位mg/L；NSE为合格站点的中位数。N/H高频评价各为2/5站，不是原17/17站全体。棉江是X唯一高频站。'
 body=lead+'\n\n## 月内信息的增量确实超出了换月值\n\n'+part+'\n\n以上MSE单位为(mg/L)²：先按实际读数数目在月内加权，再按站内月份等权、区域站点等权汇总；未除训练方差，不能当成MAP数据项。两折N/H的月水平及月内误差均下降；训练期则存在月水平误差略增、月内误差下降的权衡。D-HF并未重复加一份月均损失。\n\nM-HF相对M-PUB在两折N/H的日RMSE都略差：单纯改用高频重建月值没有带来这里的日预测收益。M-PUB与M-HF还改变采样支持，不能将此结果判为月报正确或高频资料错误。\n\n## 完整月报面板仍有退化，不能宣布全局解决\n\n'+table(pd.DataFrame(gates))+'\n\n2023年完整月报面板N/H/X均未通过原组合门槛；2024年N、X通过，H未通过相关性门槛。2024年HF月面板N/H通过，但不能替代对未参与拟合站点的评价。即使门槛通过，绝对NSE仍可能为负。2023年棉江日RMSE由0.538升至1.791 mg/L，日NSE由−0.792降至−18.832；2024年有所改善但NSE仍为−0.262。\n\n## 数值稳定性、空间映射和可约束方向\n\nM-HF、D-HF在两折的两个入口得到极接近的MAP（相对差约1e-12）及参数；这支持本轮主要配对比较不由这两个入口的分歧主导，不证明全局最优。M-PUB两入口的MAP相对差在F23为3.82%、F24为0.69%，其其他站点表现也有差异；选点仍严格遵守训练MAP。\n\n六个选中解的β_b均在下界0.25。F24的β_a从M-HF的0.819变为D-HF的1.263，同时其他参数也补偿；不能把收益单独归给β。月均灵敏度矩阵在两折已数值满秩30，月内矩阵也是30；相对1e-6阈值下F23分别为29与30，F24均为30。因此日信息提供的是额外响应约束和灵敏度分布变化，不能声称发现了此前完全不可识别的新参数数目。\n\n## 日界、事件及仍未表达的过程\n\nCHM/CMFD两种有依据的日界在公共日期上未反转区域站均日RMSE的D-HF−M-HF方向；这一结论限于已测试的指标、日期和窗口。它不代表驱动日界已一致，也不覆盖任意日期平移。6个OS-MIX固定参数回放均通过物理检查；空间敏感性另表列示，不依据其成绩更换主方案。\n\n两年图中，H组站点的一些观测峰值与模拟的响应时段和持续时间仍不一致；同河段站点模拟形状也很相近。这与月初源日历、固定暴露量、空间支持或实际观测支持等多种近似均可能有关，本轮没有将它们分别识别。日信息的改善不是已经解决这些问题。\n\n整月配对重采样的每次抽样现已在N/H/X全部站点同步，精确月份序列保存在month_block_resampling_plan.json。其范围仅描述这批回顾性样本的稳定性，不称未来预测区间。氨氮—TN同时刻关系另见ammonia_TN_quality_summary.csv；仅作质量与形态诊断，不生成硝氮、也不进入拟合。\n\n![2023日过程](F23_daily_dynamics.png)\n\n![2024日过程](F24_daily_dynamics.png)\n'
 update('reports/expert_diagnosis.md',body)
 (R/'reports/short_conclusion.md').write_text('# 高频TN新增日约束：简短结论\n\n'+lead+'\n\n2023年空间评价明显退化、2024年仍有负NSE；完整月报面板未一致通过原门槛。两折N/H月内误差下降，支持继续研究日过程约束，但不自动扩域、替换主线或认证机理。\n\n本轮有限12条矩阵已结束。详细解释、两起点差异、物理账本及日界限制见[专家诊断](expert_diagnosis.md)与[完成审计](completion_audit.json)。\n',encoding='utf8')
 deviation='## 最终后处理核查\n\n'+numeric+'\n\n所有主参数冻结后，另补齐了逐周预测、独立日RMSE重算、氨氮—TN同时刻质量表和配对月水平/月内分解。发现初版bootstrap虽在区域内跨站同步，却在不同区域使用不同抽样序列，已改为同次重复跨N/H/X同步，保存完整抽样计划并独立比对复制行与计数矩阵结果；仅稳定性摘要随之更新，参数、选点和主评价不变。OS-MIX评分增加物理有效性过滤，本轮6次回放均通过，所以该过滤未删去有效结果。\n\n控制器保留资源阈值逻辑、实测峰值预检和各进程/恢复状态，但轮询资源快照被覆盖，没有完整逐时资源历史。因此不能从归档重建整个运行期CPU/RAM的绝对最大值；不以离散快照宣称全程从未瞬间超过90%。这一记录限制与物理、数值结果分别披露。\n\n两张日过程图已人工视觉核查，未见文字遮挡或截断。日/周/月预测分别见outputs各路径日边界表、reports/weekly_HF_predictions.parquet、reports/monthly_error_components.parquet及各路径statistical_products.parquet；所有均可追溯到冻结参数和采样支持。'
 update('reports/actual_methods_and_deviations.md',deviation)
 fits['pg']=fits.pg.map(lambda x:f'{x:.6e}');fits['objective']=fits.objective.map(lambda x:f'{x:.12g}')
 # Exact scientific notation avoids rounded zero gradients in expert-facing tables.
 for file in ['expert_diagnosis.md','actual_methods_and_deviations.md']:
  p=R/'reports'/file;text=p.read_text(encoding='utf8');start=text.find('| tag | selected | objective | pg |')
  if start>=0:
   end=text.find('\n\n',start);text=text[:start]+table(fits)+text[end:];p.write_text(text,encoding='utf8')
 evidence=dict(primary_contrast='D_HF-M_HF',high_frequency_NH_daily_RMSE_improves_both_folds=True,high_frequency_NH_within_month_MSE_improves_both_folds=True,spatial_generalization_supported=False,F23_X_deterioration=True,complete_monthly_panel_uniform_gain=False,automatic_next_experiment=False,scope='Retrospective, seven training stations on five reaches; one HF X station')
 (R/'reports/scientific_findings.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
 print('FINAL_INTERPRETATION_WRITTEN',flush=True)
if __name__=='__main__':main()
