"""Independent saved-output scoring and evidence-backed final reports; no forward runs."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def put(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
arms=['MIX','HYDRO-SELECT','FULL-BYPASS']; cols=['ratio_error','peak_error','base_error']
obs=pd.read_parquet(R/'data/evaluation/observed_days.parquet').set_index(['station_key','date'])
events=pd.read_parquet(R/'data/evaluation/events_frozen.parquet')
audit=[]; totals=[]
for arm in arms:
 pred=pd.read_parquet(R/'outputs'/arm/'daily_station.parquet').set_index(['station_key','date'])
 rows=[]
 for e in events.itertuples():
  if e.cross_period or e.background_start.year<2021 or (e.period=='2024' and e.background_start.year<2024):continue
  if e.station_key not in obs.index.get_level_values(0):continue
  g=obs.loc[e.station_key];b=g.loc[(g.index>=e.background_start)&(g.index<e.start)];p=g.loc[(g.index>=e.start)&(g.index<=e.end)]
  if len(b)<4 or len(p)<1:continue
  pp=pred.loc[e.station_key];yb=float(b.y.median());yp=float(p.y.max());pb=float(pp.loc[b.index,'p'].median());pk=float(pp.loc[p.index,'p'].max())
  assert min(yb,yp,pb,pk)>0
  rows.append(dict(station_key=e.station_key,event_rank=e.event_rank,period=e.period,ratio_error=abs(np.log(pk/pb)-np.log(yp/yb)),peak_error=abs(pk-yp),base_error=abs(pb-yb)))
 independent=pd.DataFrame(rows);saved=pd.read_parquet(R/'outputs'/arm/'event_scores.parquet')
 z=independent.merge(saved,on=['station_key','event_rank','period'],validate='one_to_one',suffixes=('_ind','_saved'))
 assert len(z)==127 and len(saved)==127
 err=max(float(abs(z[c+'_ind']-z[c+'_saved']).max()) for c in cols);assert err<1e-12
 pairs=pd.read_parquet(R/'outputs'/arm/'event_pairs.parquet')
 if len(pairs): assert (pairs.event_rank-pairs.previous_rank==1).all()
 audit.append(dict(arm=arm,events=len(z),independent_score_max_difference=err,adjacent_pairs=len(pairs)))
 ledger=pd.read_parquet(R/'outputs'/arm/'annual_reach_ledger.parquet')
 for period,mask in [('1961-2024',ledger.year>=1961),('2021-2023',ledger.year.between(2021,2023)),('2024',ledger.year==2024)]:
  sub=ledger.loc[mask];sums=sub.select_dtypes('number').sum().to_dict()
  for c in ['year','reach','reach_id']:sums.pop(c,None)
  for c in sub.columns:
   if c.endswith('_yearend'):sums[c]=float(sub.loc[sub.year==sub.year.max(),c].sum())
  totals.append(dict(arm=arm,period=period,**sums))
put(R/'reports/independent_evaluation_audit.json',audit)
pd.DataFrame(totals).to_csv(R/'reports/mass_redistribution.csv',index=False)
boots=pd.read_csv(R/'reports/bootstrap_changes.csv'); bs=[]
for (a,p),g in boots.groupby(['candidate','period']):
 assert len(g)==1000
 for c in cols:
  q=g[c].quantile([.025,.5,.975]).values;bs.append(dict(candidate=a,period=p,metric=c,q025=q[0],median=q[1],q975=q[2]))
pd.DataFrame(bs).to_csv(R/'reports/bootstrap_summary.csv',index=False)
monthly=pd.read_csv(R/'reports/monthly_station_metrics.csv'); ms=[]
for (a,y),g in monthly.groupby(['arm','year']):
 eligible=g[g.nse_eligible];ms.append(dict(arm=a,year=y,stations=len(g),qualified=len(eligible),median_nse=eligible.nse.median(),mean_bias=g.bias.mean(),mean_rmse=g.rmse.mean()))
pd.DataFrame(ms).to_csv(R/'reports/monthly_overall.csv',index=False)
# Verify monthly labels directly against inherited original model-eligible table.
raw=pd.read_parquet(R.parent/'20260918_1/data/heldout_labels/monthly_original.parquet')
raw=raw[raw.model_eligible&raw.year.between(2021,2024)&raw.tn_mg_l.notna()]
saved=pd.read_parquet(R/'outputs/MIX/monthly_predictions.parquet')
z=raw.merge(saved,on=['station_key','year','month'],suffixes=('_raw','_saved'),validate='one_to_one')
assert len(z)==len(raw)==len(saved) and np.array_equal(z.tn_mg_l_raw,z.tn_mg_l_saved)
put(R/'reports/monthly_label_audit.json',dict(rows=len(z),raw_sha256=digest(R.parent/'20260918_1/data/heldout_labels/monthly_original.parquet'),exact=True))
print('INDEPENDENT_EVALUATION_PASS');print(pd.DataFrame(totals).to_string(index=False));print(pd.DataFrame(ms).query('year==2024').to_string(index=False));print(pd.DataFrame(bs).query("period=='2024'").to_string(index=False))

# Add explicit daily qualification instead of presenting short records as annual NSE evidence.
daily=pd.read_csv(R/'reports/daily_station_metrics.csv');qualification=[]
for arm in arms+['D29_REFERENCE']:
 folder=R/'reports/reference' if arm=='D29_REFERENCE' else R/'outputs'/arm
 hf=pd.read_parquet(folder/'HF_daily_predictions.parquet')
 for (s,p),g in hf.groupby(['station_key','period']):qualification.append(dict(arm=arm,station_key=s,period=p,n_months=g.ym.nunique(),nse_eligible=len(g)>=30 and g.ym.nunique()>=3 and np.var(g.y)>0))
daily=daily.drop(columns=['n_months','nse_eligible'],errors='ignore').merge(pd.DataFrame(qualification),on=['arm','station_key','period'],validate='one_to_one');daily.to_csv(R/'reports/daily_station_metrics.csv',index=False)

expert='''# 专家诊断报告：新近动员氮快路旁路

结论：本次固定表达未获得继续验证的联合预测支持。三个配置均物理合法，旁路确实改变了质量分配，但没有同时改善事件振幅、峰值和背景。本报告为项目内部科学诊断，不是外部专家评审。

## 同支持主结果

采用站内事件误差中位数、再对站等权平均。振幅误差无量纲；峰值、背景误差单位 mg/L。背景是事件前7日合格日浓度的中位数，峰值为事件内相同合格日期的最大日均值。

| 时期/配置 | 振幅误差 | 峰值绝对误差 | 背景绝对误差 |
|---|---:|---:|---:|
| 2021—2023 MIX | 0.685438 | 1.980787 | 6.172814 |
| 2021—2023 HYDRO-SELECT | 0.760750 | 2.757457 | 7.791618 |
| 2021—2023 FULL-BYPASS | 0.852346 | 1.956340 | 6.858587 |
| 2024 MIX | 0.605422 | 1.683656 | 5.690534 |
| 2024 HYDRO-SELECT | 0.616750 | 2.683603 | 7.551666 |
| 2024 FULL-BYPASS | 0.714472 | 2.084593 | 7.409233 |

前期70事件/14站，2024年57事件/15站，三个配置完全同支持。HYDRO-SELECT在2024年三项误差增加1.87%、59.39%、32.71%；前期增加10.99%、39.21%、26.22%。FULL在2024年也三项皆差；前期峰值的微小改善不能构成联合支持。

同步整月1000次重采样的2024年HYDRO变化2.5%—97.5%范围：振幅[-0.03853,0.08375]、峰值[0.04727,1.12645]、背景[0.06387,2.53265]。振幅方向不稳定；绝对误差恶化较一致。这些是已见资料的描述性稳定性区间，不是独立未来验证置信区间。

2024年MIX原本高估峰值40事件、低估17事件。HYDRO在高估组峰值误差增加1.1033、低估组增加0.1694 mg/L。高估组振幅误差略降0.01979，却伴随峰值与背景误差上升，说明比值响应不等于浓度预测改善。分层站数不同，不能把分层均值直接相加恢复总指标。

## 质量重分配及适用性

1961—2024年HYDRO累计快路输出由3.3251e10增至4.5286e10 kg，慢路由4.5588e10降至3.3806e10 kg。累计转移量由8.0112e10降至7.9800e10 kg；累计损失由9.4743e9降至9.3090e9 kg。摄取总量本轮基本不变。固定q并没有固定T，旁路不是纯粹改变时间形状。

HYDRO旁路占其累计快路质量77.05%。1000B/Qf的全历史河段日中位数2.914、P90 21.34、P99 5003.01、最大约1.62e10 mg/L。该量是旁路分量的隐含局地浓度，不是站点TN；极小载水会造成极值，未设阈值或裁剪。各河段参考期快水P10以下日期累计旁路仅约1.468 kg，故不能仅由极大浓度推断它主导全域质量。实际输入零快水单元数为0，合成边界测试另行覆盖零水。

2024完整月报116站中85站符合NSE资格。NSE中位数：原H1-D29 -0.72565、MIX -338.62072、HYDRO -292.55394、FULL -384.78766；站等权偏差分别-0.4068、+8.7978、+8.3460、+9.2448 mg/L。HYDRO月报水平有所改善，但仍极不适用。不能隐去这个改善，也不能据此签发事件结构支持。分N/H/X/OTHER结果见monthly_group_metrics.csv。

## 解释边界及衔接

本轮支持“表达能显著改变快慢输出”，不支持“改变方向解决了当前事件浓度问题”。新近动员氮不是年轻水；FULL不是振幅数学上界。试验未分离长期输出与时序重分配，不识别真实来源，也未排除完整SAS、空间加载或其他选择机制。

下一阶段衔接应先保留本轮失败证据与严重水平失真，不能沿此结果自动扩参或校准。本轮按三配置终止；不替换主线。

证据：three_error_summary.csv、comparisons.json、peak_strata.csv、bootstrap_summary.csv、mass_redistribution.csv、monthly_overall.csv，以及逐事件曲线event_curves.parquet。月内中心化误差逐站月见centered_errors.csv，日尺度资格与误差见daily_station_metrics.csv。
'''
(R/'reports/专家诊断报告.md').write_text(expert,encoding='utf-8')
methods='''# 实际方法与偏离

独立目录20260920_4，科学配置恰为MIX、HYDRO-SELECT、FULL-BYPASS；优化0次、求根0次。原目录只读。固定H1完整输入、230河段、1961—2024连续23376日、原30参数、gamma=1、k=0.0070389132605078565/d与已保存q。q哈希29abf7a4cf65b4ce540d75cc32cca4ba63b9bc660b879280adfa9f269e46206e。

源与需求仍月初施加，需求按原比例分抽；T=q×摄取后legacy，B=bT，P=摄取后mobile+T−B，快路=B+Pguφ，入慢层=Pgu(1−φ)。legacy/mobile按原共同损失率更新，慢层、河网水库及OU接口继承。接触掩码继承fast_fraction>0，并加fast_water>0；HYDRO的分母为快水与渗漏水。禁用旁路质量保留在mobile。

MIX运算顺序保留父实现的逐位回归。三个分支各自从原初始化演化，未复用MIX库存。来源标签仅覆盖继承试点河段158、225及四来源；全域总账本不代表全域来源已被识别。

评价从统计删除前正常TN继承日覆盖产品与冻结事件；未重清洗、选时滞或改峰窗。背景日浓度中位数、事件内日最大值沿用20_3公式。月报标签从父表剥离预测，并与20260918_1原始model_eligible月报表独立逐条一致核验。月报读出为整月等日浓度均值的条件假设。

日输出评价补充显式NSE资格（至少30日、3个月、正方差），未改变预测。事件对先按原序号确定，重采样只复制既有事件统计，不重建事件对。抽样台账继承seed1729共1000次，跨站同步；缺少观测的抽样月份不虚构事件。

运行使用sparrow CPU float64、每worker一线程。MIX冷启动及完整前向峰值4.1623 GB，控制器预留4.9948 GB并计入运行进程增长；追加式日志保存资源与退出。CPU/RAM90%停止派发、85%恢复策略通过夹具；本次正式运行未触发真实高内存压力，因此不声称做过机器满载压力试验。未创建定时任务。

每366日保存完整陆地和来源状态、日期、累计块计数、块哈希。正式每配置64块；实际检查点恢复核验涵盖1964闰日且逐位一致。暂停在日递推块边界生效；河网阶段使用保存陆地输出重新回放，不提供河网内部逐日断点。该工程限制不改变本次完整结果。正式任务全部在预算内完成，无预算缺项；计时见protocol.created、controller_completion及资源日志。

启动前冻结物理核；完整脚本交付哈希在最终审计时生成，不能称所有后处理代码在预测前已冻结。独立物理审计重算三个相同配置，不计作新增科学配置；MIX性能测量直接作为正式MIX，不重复科学点。

可复现入口：scripts/setup.py、startup_checks.py、worker.py、controller.py、evaluate_round.py、audit_round.py、finish_reports.py。请在归档副本复现，不覆盖已封存输出。精确CLI参数见各脚本，环境Python为D:/ProgramData/anaconda3/envs/sparrow/python.exe。
'''
(R/'reports/实际方法与偏离.md').write_text(methods,encoding='utf-8')
(R/'reports/简短结论.md').write_text('# 简短结论\n\n三个配置完成并通过物理、来源与MIX回归；独立评分一致。HYDRO-SELECT在2024振幅、峰值、背景误差分别增加1.87%、59.39%、32.71%，前期也均变差。FULL未获联合改善。当前旁路表达未支持解决模拟问题，完成本轮后结束，不扩参、不拟合、不替换主线。\n',encoding='utf-8')
elapsed=time.time()-read(R/'data/protocol.json')['created']
completion=dict(status='COMPLETE_NO_JOINT_SUPPORT',scientific_configs=3,fits=0,roots=0,physical_pass=True,independent_score_pass=True,monthly_raw_labels_exact=True,elapsed_hours=elapsed/3600,budget_hours=6,external_expert_review=False,source_scope=[158,225],limitations=['retrospective_seen_labels','timing_and_longterm_mass_not_identified_separately','no_SAS_or_young_water_identification','source_labels_only_inherited_pilot_reaches','routing_replay_not_internal_checkpoint'])
assert elapsed<6*3600
put(R/'reports/completion_audit.json',completion)
(R/'reports/独立完成审计.md').write_text('''# 独立完成审计

结论：交付完成；实现与物理通过，预测联合收益未通过。审计程序与正式前向/评分程序分离，但由同一项目内部执行，不是外部专家复核。

| 审计面 | 结果与证据 |
|---|---|
| 配置范围 | 3个完整前向，0拟合、0求根；未新增科学点 |
| MIX回归 | 全历史状态、快慢输出、日站界及试点来源逐位恢复20_3 H1_G1 |
| 独立递推 | 另写向量参考完整重算，最大通道差约1.60e-7 kg；见independent_physical_audit.json |
| 守恒与来源 | 三配置最大局地残差1.90e-8 kg，最大来源差6.34e-8 kg；非负/摄取通过；网络误差按通量尺度判定，通过原1e-10相对门槛，不能拿网络全域绝对残差与局地1e-6混比 |
| 检查点 | 实际历史状态与来源恢复逐位一致，含闰日；哈希与64块累计计数核验 |
| 事件评价 | 独立代码从每日预测/实测/事件重新选日，127事件三指标一致至1e-12；不存在按配置删除日期 |
| 月报 | 原始model_eligible标签逐条完全一致，116站，2024年85站满足NSE资格 |
| 因果与隔离 | 物理入口标签读取屏障、未来旁路不改过去、坏哈希与重复启动检查通过；没有声称做全标签反事实重新训练（本轮无训练） |
| 资源 | 冷启动/前向峰值、追加日志、控制器退出审计齐全；无实际90%压力事件，策略及暂停信号有夹具/检查点证据 |
| 预测收益 | HYDRO及FULL均NO_JOINT_SUPPORT；并非数值或物理失败 |
| 限制 | 来源范围为试点；时序与长期质量效果未完全分离；回顾性已见资料；不认证SAS或年轻水 |

机器可读依据：startup_checks.json、independent_physical_audit.json、checkpoint_control_audit.json、independent_evaluation_audit.json、monthly_label_audit.json、completion_audit.json。最终文件哈希见data/delivery_manifest.json。
''',encoding='utf-8')
(R/'README.md').write_text('''# 20260920_4 新近动员N快路旁路结构筛查

**已完成。三个配置物理通过；HYDRO-SELECT与FULL-BYPASS均未获事件三误差联合改善。不进行拟合或升级主线。**

- [专家诊断报告](reports/专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [简短结论](reports/简短结论.md)

## 数据与输出导航

data/protocol.json登记科学协议；input_manifest.json登记旧目录来源哈希；lineage_H1.json及parent_lineage_H1.json登记H1身份；evaluation目录保存冻结观测/事件/抽样；science_code_freeze.json为预测前物理核身份；prediction_freeze.json为评价前预测身份；delivery_manifest.json为交付文件身份。

outputs每配置：land_history.npy轴为通道×23376日×230河段，日期1961-01-01至2024-12-31，河段顺序随lineage/运行域。通道依次fast、slow、legacy、mobile、lower、uptake、loss、transfer、mobile_pre、available、bypass、mixed_fast、to_lower；库存kg，通量kg/日。source_tags.npy为通道×日×试点河段×四源，具体通道/河段见summary.json。bypass_fraction无量纲，q为每日概率；bypass_implied_concentration单位mg/L。

daily_station.parquet为2021—2024的116站日质量kg、水量m³、浓度mg/L；daily_HF_history为固定HF站完整历史。annual_reach_ledger按年/河段登记累计通量和年末库存。mass_redistribution.csv中的库存是报告时期最后年末值，不是多年年末库存相加。源标签仅158/225继承范围。river_history保存河网库存、损失和出口。

event_scores、event_coverage、event_pairs为逐事件与资格账本；reports/event_curves.parquet可重画逐事件曲线。three_error_summary同时保留站等权和事件直接平均，两种分母不可混用。centered_errors为逐站月中心化MSE。monthly_predictions与monthly_station/group/overall是116站原始月报面板；HF_daily_predictions另存，不能混合标签。bootstrap_changes为1000次配对变化，抽样源在data/evaluation。

state_*.npz和checkpoint.json保存实际日递推断点；logs保存追加资源、累计尝试和退出。所有运行脚本在scripts；旧代码只作为已登记继承算法。复现应在归档副本中按实际方法执行，以免覆盖哈希固定交付。
''',encoding='utf-8')
