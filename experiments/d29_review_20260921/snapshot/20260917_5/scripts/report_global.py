"""Scientific interpretation from frozen paired results, without model selection."""
import json,time
import numpy as np,pandas as pd
import native_runtime as rt
from hf_metrics import clean
R=rt.RUN;P=R/'reports'
def tbl(df):
 if not len(df):return '无可评分记录。'
 def cell(v):
  if pd.isna(v):return '—'
  s=format(v,'.6g') if isinstance(v,(float,np.floating)) else str(v)
  return s.replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
def main():
 prep=rt.read(P/'preparation.json');paths=pd.read_csv(P/'path_status.csv');comp=pd.DataFrame(rt.read(P/'comparisons.json'));metrics=pd.read_parquet(P/'station_metrics.parquet');sens=pd.read_parquet(P/'sensitivity_station_metrics.parquet');fit=pd.read_csv(P/'fit_summary.csv');sel=rt.read(P/'prediction_freeze_manifest.json')['selected'];blocks=rt.read(R/'data/spatial_blocks.json');cfg=rt.read(R/'configs/folds.json')
 # Two-start stability is independent of projected-gradient sufficiency.
 stability=[]
 for key,g in paths.groupby('config'):
  legal=g[g.physical_reasonable.fillna(False)]
  vals=legal.objective.to_numpy();gap=(vals.max()-vals.min())/max(abs(vals.min()),1e-30) if len(vals)==2 else None
  stability.append(dict(config=key,legal_starts=len(legal),sufficient_starts=int(legal.numerical_sufficient.sum()),MAP_relative_gap=gap,selected=sel.get(key),interpretation='distinct local solutions' if gap is not None and gap>=.01 else 'insufficient two-start evidence' if len(vals)<2 else 'small MAP gap; not proof of same optimum'))
 pd.DataFrame(stability).to_csv(P/'start_stability.csv',index=False)
 # Interaction on exactly the same per-station scores.
 interaction=[]
 for fold in ['T24','T25S']:
  configs=[fold+'_'+v for v in ['L_M','L_D','G_M','G_D']]
  if not all(c in sel for c in configs):continue
  parts=[]
  for c in configs:
   z=metrics[metrics.tag.eq(sel[c])][['station_key','year','scale','NSE','r','RMSE','centered_MSE']].set_index(['station_key','year','scale']);z.columns=[c.split('_',1)[1]+'_'+v for v in z];parts.append(z)
  z=pd.concat(parts,axis=1,join='inner')
  for v in ['NSE','r','RMSE','centered_MSE']:z['interaction_'+v]=(z['G_D_'+v]-z['G_M_'+v])-(z['L_D_'+v]-z['L_M_'+v])
  z=z.reset_index();z['fold']=fold;interaction.append(z)
 if interaction:pd.concat(interaction,ignore_index=True).to_parquet(P/'factor_interaction.parquet',index=False)
 # Time support can reverse the paired D-M direction: compare common dates only.
 sensitivity_direction=[]
 for scope in ['T24_L','T24_G','S56','S113','S191']:
  if scope+'_D' not in sel or scope+'_M' not in sel or sens.empty:continue
  a=sens[sens.tag.eq(sel[scope+'_D'])&sens.scale.eq('daily')&sens.role.eq('evaluation')];b=sens[sens.tag.eq(sel[scope+'_M'])&sens.scale.eq('daily')&sens.role.eq('evaluation')]
  z=a.merge(b,on=['station_key','view'],suffixes=('_D','_M'),validate='one_to_one');z['delta_RMSE']=z.RMSE_D-z.RMSE_M
  for view in ['CHM','CMFD']:
   natural=z[z.view.eq('BJT_common_'+view)].set_index('station_key');alt=z[z.view.eq(view+'_common')].set_index('station_key');names=natural.index.intersection(alt.index)
   for s in names:sensitivity_direction.append(dict(scope=scope,station_key=s,window=view,natural_delta=float(natural.loc[s,'delta_RMSE']),alternative_delta=float(alt.loc[s,'delta_RMSE']),direction_reversed=bool(natural.loc[s,'delta_RMSE']*alt.loc[s,'delta_RMSE']<0)))
 pd.DataFrame(sensitivity_direction).to_csv(P/'time_support_direction.csv',index=False)
 # Inspect where errors go, including severe failure cases; no station removal.
 pairedpath=P/'paired_station_changes.parquet';risk=pd.DataFrame()
 if pairedpath.exists():
  paired=pd.read_parquet(pairedpath);risk=paired[paired.RMSE_over_2x|paired.wrong_correlation|paired.persistent_overestimate];risk.to_csv(P/'risk_stations.csv',index=False)
 summaries=[]
 if not comp.empty:
  for contrast,g in comp.groupby('contrast'):
   if g.year.eq(2025).all():
    mm=g[g.scale.eq('monthly_PUB')];passed=int(mm.monthly_gate.fillna(False).sum())
    summaries.append(f'- **{contrast}**：2025年1—11月月报敏感性组合门槛通过{passed}/{len(mm)}组；无合格日尺度主评价，与2024分开解释。')
    continue
   formal=g[g.year.eq(2024)&~g.panel.eq('SPACE_TRAIN')];d=formal[formal.scale.eq('daily')];m=formal[formal.scale.eq('monthly_PUB')]
   good=int(d.RMSE_ratio.lt(1).sum());bad=int(d.RMSE_ratio.gt(1).sum());gates=int(m.monthly_gate.fillna(False).sum()) if len(m) else 0
   summaries.append(f"- **{contrast}**：2024分组日RMSE下降{good}组、上升{bad}组；月报组合门槛通过{gates}/{len(m)}组。不同区域和空间面板分别判断，不能用组数代替站点收益或独立流域样本量。")
 caveat='本轮为算法隔离的回顾性检验。2024为正式留出；2025仅1—11月月报敏感性，水文/PET延伸及重复2024氮源与需求不能与2024合并认证。2025无合格高频站月。未知月报采用等日均值仍属条件假设；同名同坐标不能认证全部站址历史和审核血缘。自然日对应混合日界的冻结水文日期是日尺度近似，未建立四小时物理模型。'
 n=int(paths.numerical_sufficient.fillna(False).sum());p=int(paths.physical_reasonable.fillna(False).sum());detail=paths[['tag','status','calls','objective','pg','numerical_sufficient','physical_reasonable']]
 columns=['contrast','scale','panel','cohort','stations','eligible_NSE','delta_NSE','delta_r','RMSE_ratio','numerically_sufficient']
 chosen=comp[[c for c in columns if c in comp]] if len(comp) else pd.DataFrame()
 interpretation='\n'.join(summaries) or '没有足够合法配对点，不能判断预测收益。'
 text='# 全域月水平＋高频日变化联合校准：专家诊断\n\n'+f'28条登记路径中，{n}条数值充分，{p}条有通过原物理容差的结果。数值不足路径保留预测诊断，但其配对结论为条件性。\n\n'+caveat+'\n\n## 固定比较的结果\n\n'+interpretation+'\n\n'+tbl(chosen)+'\n\n## 目标与参数补偿\n\n月水平采用唯一站月来源：合格HF优先，否则月报；M/D共同月份与方差完全相同。D仅增加HF月内中心化误差，不重复计月均值。L/G共有站点的数据相同，但1/S随合法站数变化，这是登记的站等权设计。\n\n数据项、先验项和全部30参数见fit_summary.csv；两入口MAP差见start_stability.csv。不同目标的MAP大小不可直接用来宣称预测更优，梯度小也不保证同一全局解。\n\n## 时间收益与空间收益\n\n时间折G包含棉江训练标签，棉江2024结果属于训练地点的时间预测。空间能力仅看S56、S113、S191的SPACE_HELD；SPACE_BUFFER为禁用下游诊断，SPACE_TRAIN不混入空间成绩。三个支流并不足以代表众多独立流域样本。交互量见factor_interaction.parquet。\n\n逐站NSE资格及总覆盖、均值偏差、中心化误差、幅度、月峰位相、高浓度缺口均保留在station_metrics.csv。月内残差分解见monthly_error_components.parquet；严重退化和持续高估见risk_stations.csv，不因退化删除站点。日/周/月尺度分开，不将浓度相加解释为负荷。\n\n## 过程与支持限制\n\n高低流分组使用冻结训练水量阈值，相关记录见flow_diagnostics.parquet。同河段站的共同日期浓度差和比例见same_reach_diagnostics.parquet，不假定同时观测属于同一水团。\n\n每路径daily_station_mass_water.parquet保存日质量与水量；monthly_physical_ledger.parquet保存全域库存、入河、摄取、需求和损失；network_ledger.parquet保存终端与水库库存。四源标签仅继承158、225试点范围，不称全域已识别历史来源。预测改善不证明真实legacy来源或施肥日历已识别。\n\n## 稳定性\n\nCHM/CMFD比较同时报告公共日期与覆盖变化，time_support_direction.csv列示收益反转。出现反转的站点应称时间支持下尚不可辨别，不挑选最好的日界。维护状态及训练删除前资料只作固定参数诊断。整月同步重采样仅描述本资料稳定性，不是未来预测或机制置信区间。\n\n## 逐路径数值状态\n\n'+tbl(detail)+'\n'
 text+='\n## 月水平与月内训练损失\n\n'+tbl(fit[fit.selected][['tag','month_level','within_month','prior','objective','pg']])+'\n\n逐站月贡献保存在training_loss_components.parquet。G相较L扩大了站数，也改变具有日资料站点在总目标中的占比；共有站仍采用同一来源与尺度，但整体1/S归一化会变化。因此G-D与L-D的差异应结合覆盖和贡献构成解释，不能全归因为某一新增站点的因果作用。\n\n选定参数的完整历史物理重算见selected_routing_recompute.json；selected_routing目录包含各河段入口/出口、各水库捕获/释放/库存，以及观测年份每日源输入—快慢入河—出口响应链。OU沿用158、225号试点积分范围，其他河段仍使用原表达。\n'
 olddir=R/'diagnostics/old_extrapolation';environment=pd.read_csv(olddir/'environment_support.csv');names=rt.read(R/'data/domains/FULL24/topology.json')['static_fields'];environment['field']=environment.feature.map(lambda i:names[int(i)])
 oldrisk=pd.read_csv(olddir/'mapping_inventory_risk.csv');oldrisk=oldrisk[oldrisk.reach_id.eq(113)&oldrisk.year.eq(2024)][['old_tag','beta','hazard_max','M_end','L_end','fast_kg','slow_kg','loss_kg']]
 text+='\n## 棉江旧点的有限外推检查\n\n'+f'七站仅覆盖五个河段；棉江有{int(environment.outside_seven_station_support.sum())}/7个静态属性超出这五河段的范围。该判断描述环境覆盖，不表示超出原26参考河段的所有标准化范围，也不单独证明退化原因。\n\n'+tbl(environment[['field','mianjiang_raw','training_min','training_max','standardized','outside_seven_station_support']])+'\n\n四个旧保存点均从1961重新传播，只作诊断，没有参与新路径初始化或预热库存。它们在棉江的动员、库存、快慢通量及损失差异如下；端点β本身不能概括所有参数补偿。\n\n'+tbl(oldrisk)+'\n'
 (P/'expert_diagnosis.md').write_text(text,encoding='utf8')
 clock=rt.read(R/'work/experiment_clock.json');deviation='预处理第一次重算在无TN标准化浮点归约的精确字典比较处停止；改为1e-12相对/绝对容差核验后仍保留原冻结标准化值。启动前隔离测试发现Windows反斜线导致清单分区筛选不生效；读取屏障已拒绝外折文件，尚未启动拟合。将清单路径统一为POSIX表示后真实身份测试通过。上述修复没有修改目标、科学起点、数据阈值或重置预算。'
 resource_note='完整验收同时持有多个模型，原预约使用了该进程6.56 GiB峰值。随后对最大FULL25日目标及116站完整审计单独测量，单worker峰值为约3.17 GiB；按它与实测拟合峰值的最大值×1.2重新预约。只重启控制器并接管原PID，未停止拟合worker、变更每worker身份或重置调用/时钟。RESOURCE_PROFILE_FIXED_POINT是零次优化的原预设1资源测试，不是第29条拟合路径。重启后的收据曾因重复time字段写入失败，控制器已成功接管；独立核验随后补记了收据。证据见resource_reservation_change/verified.json及追加事件日志。'
 (P/'actual_methods_and_deviations.md').write_text('# 实际方法与偏离\n\n'+deviation+'\n\n'+resource_note+'\n\n## 实际准入\n\n'+tbl(pd.DataFrame(prep['cohorts']))+'\n\n## 执行与预算\n\n'+f'总预算72小时，最后4小时收尾。时钟起点{clock["started"]}；本报告时刻{time.time()}。完整conda sparrow、CPU float64、每worker一线程；按实测峰值×1.2与增长空间派发，没有固定worker上限或定时任务。优先2024时间与空间任务，之后2025敏感性。资源与每次派发、退让、退出事件追加于work/resource_events.jsonl。\n\n'+tbl(detail)+'\n\n## 证据等级与缺项\n\n'+caveat+'\n\n来源标签覆盖仅继承试点，不扩展动力学。新域来自v3正式产品，230河段全部检查字段差异为零；与原68河段完整支持的公共预测/目标/梯度回归通过。原始与旧实验只读。所有条件性未完成路径按path_status.csv保留，不把检查点当作完成。\n',encoding='utf8')
 (P/'short_conclusion.md').write_text('# 简短结论\n\n'+f'已处理28条有限路径；数值充分{n}/28，具有合法物理结果{p}/28。\n\n'+interpretation+'\n\n'+caveat+'\n\n不自动替换主线；是否扩域或改机制需另行决策。\n',encoding='utf8')
 (R/'README.md').write_text('# 全域月水平＋高频日变化联合校准\n\n[简短结论](reports/short_conclusion.md) · [专家诊断](reports/expert_diagnosis.md) · [实际方法与偏离](reports/actual_methods_and_deviations.md) · [独立完成审计](reports/completion_audit.json)\n\n独立实验：v3与auto_4h只读。230河段、30参数OU、28条有限路径。2024正式与2025敏感性分别报告，空间留出包含下游禁用闭包。没有新增模型结构、四小时物理模型或定时任务。\n\n复现入口：scripts/campaign_controller.py；报告入口scripts/finalize_global.py。所有运行须使用conda sparrow，重复启动与累计预算保护生效。scripts中带hf前缀的继承报告不是本轮交付入口。\n',encoding='utf8')
 with (P/'actual_methods_and_deviations.md').open('a',encoding='utf8') as f:
  f.write('\n## 报告渲染与后续时间授权\n\n首轮报告渲染因sparrow环境缺少tabulate而停止；改为实验目录内的Markdown表格函数，不安装依赖、不改预测或选点。随后重跑报告与独立审计。用户追加允许时间不足且未收敛的原路径续算；本轮28条独立审计全部数值充分，未触发额外优化。S113_M_s0虽保留预算停止状态，其保存点梯度已满足原标准。\n')
  f.write('\n图件复核后的重复生成曾在重新打开既有PNG时发生Invalid argument；没有修改文件权限，改为仅复用与既有清单哈希一致的图件，未匹配清单的图件仍重新生成。冻结预测身份保持不变。各后处理子任务现在使用单独追加日志，避免多进程共享重定向日志造成诊断丢失。\n')
  if (P/'checkpoint_io_recovery/complete.json').exists():
   f.write('\n## 检查点写入中断与恢复\n\nT24_G_M_s1在Windows原子替换latest.json时持续30秒拒绝访问。异常退出前保存了1393次累计调用及11212.44秒活动时间；同内容原子替换与哈希恢复测试随后通过。外部句柄的具体来源未证实。控制器短暂停止派发并接管其他原进程，该路径从原状态恢复，未增加起点、变更核心代码或重置预算。中断点的审计若已启动则另存为恢复证据，不用它完成主路径。详见checkpoint_io_recovery。\n')
  if (P/'checkpoint_status_io_recovery/complete.json').exists():
   recovery=rt.read(P/'checkpoint_status_io_recovery/complete.json')
   f.write(f'\nT24_G_D_s1随后在替换status.json时发生同类访问异常，保留{recovery["calls"]}次调用及{recovery["active_seconds"]:.2f}秒活动时间后恢复。证据见checkpoint_status_io_recovery；两次中断均不算新增科学路径，重复发生的Windows文件访问风险未被证明彻底消除。\n')
  if (P/'t25_checkpoint_io_recovery/complete.json').exists():
   recovery=rt.read(P/'t25_checkpoint_io_recovery/complete.json')
   f.write(f'\nT25S_G_D_s1在精修阶段替换latest.json时也发生同类异常，保留{recovery["calls"]}次调用及{recovery["active_seconds"]:.2f}秒活动时间后从原状态恢复。证据见t25_checkpoint_io_recovery。这是第三次文件访问中断，不增加逻辑拟合路径。\n')
  f.write('\n## 计算计数的已知偏离\n\n拟合计数包含优化、差分与恢复检查；独立审计与报告重算未纳入这些计数。computation_accounting.csv/json另列每条路径已知审计接口调用、选定点重算及观察到的审计时长。总调用仅能给出下界，不能声称所有计算均已满足每路径调用或活动时间上限。实验总时钟没有重置。\n\n## 地理摄取支持与风险标记\n\nN/H/X沿用原物理河段集合；OTHER为新增月报站完整上游支持的并集，允许与原区域重叠，各区域摄取比例不得相加。ALL覆盖230河段，空间面板使用各外层登记的支持掩码。详见evaluation_physical_region_registration.json。风险表中的persistent_overestimate仅指正偏差超过0.5倍RMSE的高估候选，并不证明每月持续高估。\n')
 print('REPORT_GLOBAL_COMPLETE',n,p,flush=True)
if __name__=='__main__':main()
