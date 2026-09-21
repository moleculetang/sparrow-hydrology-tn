"""Evidence-based reports, final artifact census and independent scoring check."""
from runtime import *
from datetime import datetime,timezone
def mdtable(frame):
 def f(x):
  if isinstance(x,(float,np.floating)):return f'{x:.6g}'
  return str(x)
 return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
def main():
 required=['preflight.json','additional_checks.json','independent_physical_audit.json','input_and_execution_audit.json','controller_completion.json','comparisons.json','base_peak_audit.json']
 for n in required:assert (R/'reports'/n).exists(),n
 protocol=read(R/'data/protocol.json');summaries={a['id']:read(R/'outputs'/a['id']/'summary.json') for a in protocol['arms']}
 comp=read(R/'reports/comparisons.json');bp=read(R/'reports/base_peak_audit.json');audit=read(R/'reports/independent_physical_audit.json')
 # Independent primary statistic from the delivered raw daily-pair table; no score_events import.
 frozen=pd.read_parquet(R/'data/events_frozen.parquet');recomputed={}
 for name in ('H1_G0','H1_G1'):
  daily=pd.read_parquet(R/'outputs'/name/'HF_daily_predictions.parquet');errs=[]
  for row in frozen[frozen.period.eq('2024')].itertuples():
   if row.cross_period or row.background_start.year<2024:continue
   g=daily[daily.station_key.eq(row.station_key)];background=g[(g.date>=row.background_start)&(g.date<row.start)];event=g[(g.date>=row.start)&(g.date<=row.end)]
   if len(background)<4 or len(event)<1:continue
   yy=np.array([np.median(background.y),np.max(event.y)]);pp=np.array([np.median(background.p),np.max(event.p)])
   if (yy<=0).any() or (pp<=0).any():continue
   errs.append((row.station_key,abs(np.log(pp[1])-np.log(pp[0])-np.log(yy[1])+np.log(yy[0]))))
  df=pd.DataFrame(errs,columns=['station','error']);recomputed[name]=float(df.groupby('station').error.median().mean())
 expected=comp['H1_G1 minus H1_G0']['2024'];assert abs(recomputed['H1_G0']-expected['baseline'])<1e-12 and abs(recomputed['H1_G1']-expected['candidate'])<1e-12
 independent_reduction=1-recomputed['H1_G1']/recomputed['H1_G0']
 roots=[read(p) for p in (R/'outputs').glob('*_root.json')];assert len(roots)==8 and all(x['status']=='PASS' and abs(x['relative_error'])<=1e-8 for x in roots)
 # Actual formal transfer sums, independently compared with the registered target.
 norm={}
 for name,s in summaries.items():
  if s['arm']['kind'] in ('main','clim') and s['arm']['gamma']!=0:
   target=summaries[name[:2]+'_G0']['reference_transfer'];norm[name]=abs(s['reference_transfer']/target-1)
  elif name=='H1_LEVEL1':norm[name]=abs(s['reference_transfer']/summaries['H1_LEVEL0']['reference_transfer']-1)
 assert max(norm.values())<=1e-8
 lineage0=read(R/'data/lineage_H0.json');lineage1=read(R/'data/lineage_H1.json')
 rootcount=read(R/'reports/input_and_execution_audit.json')['root_calls'];blocked=[n for n,s in summaries.items() if not s['gate']['pass']]
 rows=[]
 for name,s in summaries.items():
  g=s['gate'];tags=g.get('source_sum_errors',{});rows.append(dict(配置=name,k=s['arm'].get('k','原D29'),物理来源检查=s['physical_status'],局地平衡kg=g['local_balance_max_kg'],最大来源误差kg=max(tags.values(),default=0),参考期站均浓度=s['reference_station_mean']))
 table=mdtable(pd.DataFrame(rows));comparison=[]
 for k in ('G05','G1','G2','G4'):
  a=comp[f'H1_{k} minus H1_G0'];comparison.append(dict(配置=k,前期改善百分比=100*a['2021-2023']['relative_reduction'],后期改善百分比=100*a['2024']['relative_reduction'],后期改善站比例=a['2024']['improved_fraction']))
 ctable=mdtable(pd.DataFrame(comparison));obs=read(R/'data/observation_manifest.json')
 evsummary=pd.read_csv(R/'reports/event_base_peak_summary.csv');primaryA=evsummary[(evsummary.period.astype(str)=='2024')&evsummary.arm.isin(['H1_D29','H1_G0','H1_G1','H1_G4'])]
 basescore=[]
 for name in ('H1_D29','H1_G0','H1_G1','H1_G4'):
  e=pd.read_parquet(R/'outputs'/name/'event_evaluation.parquet');e=e[e.period.eq('2024')&e.log_amplitude_error.notna()]
  basescore.append(dict(arm=name,error=float(e.groupby('station_key').log_amplitude_error.median().mean())))
 bootstrap=pd.read_csv(R/'reports/bootstrap_changes.csv');sel=bootstrap[bootstrap.comparison.eq('H1_G1 minus H1_G0')&bootstrap.period.eq('2024')];band=sel.change.quantile([.025,.5,.975]).to_dict()
 red=pd.read_csv(R/'reports/reach_transfer_redistribution.csv');red=red[red.arm.eq('H1_G1')].relative_change.dropna()
 levels={n:max(summaries[n]['gate']['source_sum_errors'].values()) for n in blocked}
 resources=[json.loads(s) for s in (R/'logs/resources.jsonl').read_text().splitlines()];maxram=max(x['ram'] for x in resources);maxcpu=max(x['cpu'] for x in resources)
 elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds()/3600
 expert=f'''# 专家诊断报告：状态依赖动员有响应，但未恢复所需事件结构

## 判决

本轮完成14个正式配置、零拟合及{rootcount}次无标签求根评估。12个配置通过物理/来源硬门；两个均值匹配配置因来源库存求和超限被隔离。主配置在2024的事件误差下降{100*independent_reduction:.3f}%，未达到25%预注册门槛。因此结论是**有限响应存在，但未取得预注册的初步结构支持**，不是“湿润动员完全无效”，也不是“结构已可用于校准”。

## 1. 哪些证据是真正新增的

使用17_3的H1完整水文状态、容量与水库产品，H1/H0动态标准化及β门控均由26个参考河段重新核验。原30参数、四类源、需求、温度与静态属性未改变。既有H0结果没有冒充H1基线。

从canonical重新取得{obs['records']}条正常唯一TN，形成{obs['eligible_days']}个合格日。未施加全历史统计删除。冻结H1水量产生252个候选事件，最终2021—2023为70事件/14站，2024为57事件/15站。背景取相同已观测日期的中位数，峰值取相同已观测日期的日均最大值；不能与旧四小时峰值门槛直接比较。

{ctable}

主配置前期误差下降8.976%，2024下降4.506%；2024为10/15站改善。gamma=4也仅下降13.290%，不能在本轮替换主配置。

## 2. 改善不是单纯背景下降，但也不是事件浓度拟合成功

2024主配对57个事件中，只有{bp['2024']['ratio_up_without_peak_up']}个符合“比值上升而峰值不升、背景下降”。峰值对数变化中位数为{bp['2024']['median_log_peak_change']:.6f}，背景贡献中位数为{bp['2024']['median_minus_log_background_change']:.6f}。多数事件发生了实际模型峰值上升。

但事件峰值绝对误差均值从{bp['2024']['peak_absolute_error_before']:.4f}升至{bp['2024']['peak_absolute_error_after']:.4f} mg/L，背景绝对误差也由{bp['2024']['background_absolute_error_before']:.4f}升至{bp['2024']['background_absolute_error_after']:.4f} mg/L。比值改善不能等同于绝对预测改善。

以下为“各站事件中位数再站等权平均”，不是所有事件的中位数，也不能把汇总后的峰/背景再相除当作振幅：

{mdtable(primaryA[['arm','obs_A','pred_A','obs_base','pred_base','obs_peak','pred_peak']])}

200天常数核本来就是很弱的对照；湿润度改动未消除其事件稀释倾向。原D29是另一个更严格的参照：

{mdtable(pd.DataFrame(basescore))}

## 3. 时间效应、空间再分配和水文效应不能混在一起

相对气候态gamma=1，实际逐日湿润度的2024事件误差仅再改善{100*comp['H1_G1 minus H1_CLIM1']['2024']['relative_reduction']:.3f}%。这是有限的逐日状态增量，不能把相对常数核的全部改善归因于事件触发。

全域累计转移匹配最大相对残差为{max(norm.values()):.3g}，但主配置各河段累计转移相对变化范围为{red.min():.3%}至{red.max():.3%}。全域相同质量并不消除空间再分配、损失和出口量变化。

H0主配对2024改善{100*comp['H0_G1 minus H0_G0']['2024']['relative_reduction']:.3f}%，H1为4.506%。H1常数基准相对H0常数基准改善{100*comp['H1_G0 minus H0_G0']['2024']['relative_reduction']:.3f}%。这支持分开报告输入修正与新结构响应；不是水文已被完全排除的证明。

1000次同步整月重采样的主误差差值分位数（2.5%、50%、97.5%）为{list(band.values())}。这些是已见回顾性资料的稳定性描述，不是未来泛化置信区间。

## 4. 物理合格不等于实际可用

{table}

均值匹配来源最大误差为{levels} kg，原门槛仍为1e-6 kg。它们的浓度诊断可以查看，但不能用于签发水平稳健性。被动四源标签只覆盖继承的试点河段；230河段交付总库存及通量，不宣称全域来源已识别。

N/H/X/OTHER的2024完整月报组合门槛，在主配置相对200天常数核和相对原D29两类比较中均未全部通过。逐站和区域数据见annual_monthly_station_metrics.csv、annual_monthly_group_metrics.csv和monthly_combined_gates.csv。不能以对一个严重失真的基线有所改善掩盖绝对误差。

## 5. 这一轮支持什么后续方向

可以保留“湿润状态具有一定动员时间信息”的窄结论；不支持直接把本函数加入生产校准，也不支持继续按本轮成绩上调gamma。两个来源不合格的水平敏感性使水平稳健性仍有缺项。

本轮没有检验独立浓度组成的选择性混合，也没有修改源日历、输移时间或空间站界。后续若另立结构实验，应要求候选同时解释同支持的峰值、背景与浓度差，并明确区分水量稀释和不同来源浓度被采样的变化。不要把本次有限网格失败写成对所有湿润动员或库存机制的证伪。是否启动下一机制另行制定计划，本轮到此结束。
'''
 (R/'reports/专家诊断报告.md').write_text(expert,encoding='utf-8')
 method=f'''# 实际方法与偏离

## 固定实现

14配置按冻结protocol.json运行。全域230河段、1961—2024、CPU float64、每进程一线程、OU站界、原30参数，无拟合。湿润度W=(soil_storage_mm+actual_aet_mm_day)/FC，来源与配套容量核验；W未新增裁剪。gamma=0明确定义W^0=1。转移q=-expm1(-k W^gamma)。保留20_2输入进入legacy、比例摄取、共同损失和快慢分水顺序。

动态总量求根的ML/MM递推贯穿完整历史；总转移求根省略不反馈到这两个池的L及河道路由，独立核验其转移总量与完整前向一致。均值求根执行完整路由。先21点对数括区，再Brent求根；“唯一”指登记网格只检测到一个括区，不是数学上证明整个连续区间唯一。未扩边界，八个求根任务均达标，共{rootcount}次评估。

所有配置预测完成并冻结哈希后，才运行evaluate.py揭示TN评价。2024是回顾性再检验。实际参数来源曾使用2021—2023标签；design.training_years中的2021—2022只是标准化参考，不能宣称原参数仅见过两年。

## 数值修正

初版被动标签使用了代数等价但浮点顺序不同的比例分抽。发现均值敏感性来源误差后，在首次TN评价之前恢复20_2原标签逐步拼写；常数臂的六通道逐位回归通过。修正只涉及被动标签，不改变物理轨迹、浓度或求根。初始门结果另存initial_source_gate.json，差异见source_arithmetic_review.json。两敏感性臂修正后仍不合格，没有裁剪、补差或放宽容差。

## 评价和审计补充

峰窗严格为水文事件起止日，不沿用旧代码额外+1日。前7天背景取中位数；模拟与实测共用日期。保留单日事件覆盖标记。F3先按完整事件序号检查相邻性，再计算真正间隔，最后构造固定事件对；复制整份样本的回归不变性已通过。整月重复抽样保留bootstrap_copy身份，不在拼接副本后重新连接事件。

统计量和科学门槛来自用户批准计划；评价脚本及附加审计代码是在前向运行期间/之后完成，不声称全部实现文件在前向前已一次性冻结。execution_freeze.json保留当时版本，最终脚本另列哈希。没有根据评价结果调整gamma网格、事件、阈值或选点。

区域摄取账本使用各组登记站母河段及模型上游并集，组间可能重叠，定义和河段ID全部交付；其门槛是此明确支持下的诊断，不升级为空间泛化认证。区域支持代码审查纠正了零基拓扑索引到全域河段ID的映射，不影响前向或浓度评价。

## 执行偏离和限制

独立后台控制器分别消费H0/H1退出事件，未建定时任务。原环境没有psutil，资源采样改用Windows API。执行期CPU/RAM追加记录最高分别为{maxcpu:.2f}%/{maxram:.0f}%，未达到90%。但执行期首次资源实现未采集进程峰值，后来新增的峰值查询又因句柄声明问题返回0，已修正；这些0不代表真实零内存。正式派发没有完成“实测完整峰值×1.2＋增长”规则，不能认证该项通过。审计补测与早期缺项分开登记。

当前控制器的时间预算在派发层执行，缺少运行中安全检查点退让实现；本次任务在约3.3分钟内退出，两项限制未改变有限矩阵结果，但控制器不能原样用于长时间训练。总实际用时截至本报告约{elapsed:.2f}小时，未接近12小时。本轮不以已有检查点冒充未完成任务。

原目录受写入屏障保护。零水拒绝、共享水库单次释放、未来概率不改过去、来源标签、H0站界回归、H1标准化重建均有单独证据。标签隔离采用物理进程读取屏障和无TN接口；未修改整份原始观测归档做破坏性反事实测试。
'''
 (R/'reports/实际方法与偏离.md').write_text(method,encoding='utf-8')
 auditdoc=f'''# 独立完成审计

这是与正式递推循环分离的重算程序审计，不是外部专家审查，也不能证明不存在任何未发现缺陷。

| 项目 | 结论 | 证据 |
|---|---|---|
| 数据及水文身份 | 通过已实施核验 | input_and_execution_audit.json、additional_checks.json |
| 新核与父核回归 | 通过 | preflight.json；H0逐站差≤1e-12 |
| 独立前向 | 12个非D29配置全部通道重算差≤1e-6 kg | independent_physical_audit.json |
| 物理/来源硬门 | 12/14通过；两个水平敏感性隔离 | 各配置summary.json |
| 求根数值充分 | 8/8满足登记残差；{rootcount}/512次 | *_root.json、root_results.jsonl |
| 评价独立重算 | 主比较2024改善{100*independent_reduction:.6f}% | 从交付日表与冻结事件另行计算 |
| 结构收益门槛 | 未通过25% | shape_verdict.json |
| 水平稳健性 | 条件性缺项 | H1_LEVEL0/1来源误差超限 |
| 月面板实际可用性 | 未通过 | monthly_combined_gates.csv |
| 运行资源要求 | 部分通过，峰值预留/安全检查点实现不完整 | 实际方法与偏离.md |
| 交付 | 14配置及缺项完整登记 | delivery_manifest.json |

主比较使用同一57事件/15站；2024站等权误差独立重算为{recomputed}。归一化实际转移误差最大{max(norm.values()):.3g}。局地质量检查、独立前向差、来源总和误差是不同指标，不用其中一个替代另一个。

本轮状态：**有限矩阵完成，未取得注册结构支持，敏感性及执行偏离已明确登记；停止，不自动增加模型或重拟合。**
'''
 (R/'reports/独立完成审计.md').write_text(auditdoc,encoding='utf-8')
 conclusion=f'''# 简短结论

完成14个正式配置、零拟合；12个通过物理/来源检查，2个均值匹配配置因来源库存求和超限隔离。

- 主配置2024事件振幅误差下降 **{100*independent_reduction:.2f}%**，10/15站改善，低于25%门槛。
- gamma=4也只改善13.29%；不能改选它作为主结果。
- 模型确有事件响应，改善并非主要由压低背景制造；但事件峰值绝对误差略增，完整月面板仍严重失真。
- H1已连同容量、库存、标准化和水库完整继承；H0仅作桥接。
- 结论是 **结构有有限响应，但尚未证明具有所需的事件表达能力**。不替换主线，不继续扩参数。

阅读顺序：专家诊断报告 → 实际方法与偏离 → 独立完成审计。
'''
 (R/'reports/简短结论.md').write_text(conclusion,encoding='utf-8')
 (R/'README.md').write_text('''# 20260920_3：湿润状态控制的氮动员

状态：有限矩阵完成，结论未通过预注册结构门槛；两个水平敏感性配置来源硬门不合格。没有拟合或替换主线。

## 阅读入口

- [简短结论](reports/简短结论.md)
- [专家诊断报告](reports/专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [证据图](reports/figures/structure_evidence.png)

## 数据接口

`data/protocol.json`是冻结配置；`lineage_H0/H1.json`记录配套水文、容量和数组身份。`observed_days.parquet`为UTC+8自然日、正常TN、唯一读数≥4且跨度≥12小时、每月≥10合格日的统计删除前视图，原记录ID可回溯canonical。

每个`outputs/<arm>/`包含：`daily_station.parquet`（2021—2024、116站、日质量kg/日水量m³/浓度mg/L）；`daily_HF_history.parquet`（1961—2024固定15站）；`land_history.npz`（day×230、kg）；`source_tags.npz`（channel×day×继承试点×4来源，通道fast/slow/legacy/mobile/lower/uptake/loss/transfer）；`river_history.npz`；`q.npy`（day×230概率）；年度账本、事件与月产品和硬门summary。D29没有新增q/legacy/mobile。

NPZ历史日轴固定1961-01-01至2024-12-31，河段轴依照lineage域的global_reach_ids，source_tags的reaches另存全域ID。不能将局地通量当成下游站点质量，也不能用算术浓度×月水量替代负荷。

## 复核

所有脚本使用conda sparrow下python -B运行。正式协议已有完成记录；不要直接再次运行setup.py或覆盖归档。`independent_audit.py`、`additional_checks.py`、`finalize.py`用于同目录复核并更新审计产物，旧实验只读。物理worker安装标签读取屏障；观测准备和评价另进程运行。

关键限制：回顾性资料；均值匹配两个配置不合格；资源峰值预留与长任务安全检查点有实现缺项。详见偏离报告。本轮完成后不自动启动下一机制。
''',encoding='utf-8')
 put(R/'reports/final_audit.json',dict(status='FINITE_MATRIX_COMPLETE_WITH_CONDITIONAL_GAPS',arms=14,physical_pass=12,blocked=blocked,root_calls=rootcount,root_relative_max=max(norm.values()),independent_score=recomputed,independent_relative_reduction=independent_reduction,registered_shape_support=False,production_ready=False,resource_dispatch_conformance=False,external_expert=False,elapsed_hours=elapsed))
 v=read(R/'reports/shape_verdict.json');v['criteria']['background_not_sole_driver']=True;v['base_peak_caveat']='Most event ratios improve with peak rising, but absolute peak error increases';v['criteria']['level_same_direction']='UNDETERMINED_PHYSICAL_GATE_FAILED';put(R/'reports/shape_verdict.json',v)
 put(R/'reports/figure_QA.json',dict(backend='Python/matplotlib',visually_reviewed=True,no_overlap=True,source_csvs_present=True,editable_svg_text=True,format='Internal research report, not a journal-submission claim'))
 put(R/'data/final_code_hashes.json',{str(p.relative_to(R)):sha(p) for p in (R/'scripts').glob('*.py')})
 census=[]
 for p in sorted(R.rglob('*')):
  if p.is_file() and 'numba' not in p.parts and 'matplotlib' not in p.parts and p.name!='delivery_manifest.json':census.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
 put(R/'reports/delivery_manifest.json',census)
 print('FINALIZED',len(census),'files',elapsed,'hours')
if __name__=='__main__':main()
