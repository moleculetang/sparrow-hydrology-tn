"""Concrete interpretation of frozen comparisons, with no model selection changes."""
import numpy as np,pandas as pd
import native_runtime as rt
from hf_metrics import summaries,clean
from report_global import tbl
R=rt.RUN;P=R/'reports'
def main():
 paths=pd.read_csv(P/'path_status.csv');assert len(paths)==28 and paths.numerical_sufficient.all() and paths.physical_reasonable.all()
 selected=rt.read(P/'prediction_freeze_manifest.json')['selected'];metrics=pd.read_parquet(P/'station_metrics.parquet');comp=pd.DataFrame(rt.read(P/'comparisons.json'));starts=pd.read_csv(P/'start_stability.csv');direction=pd.read_csv(P/'time_support_direction.csv');rows=[]
 for config,tag in selected.items():
  z=metrics[metrics.tag.eq(tag)]
  if config.startswith('S'):z=z[z.panel.eq('SPACE_HELD')]
  for scale,g in z.groupby('scale'):rows.append(dict(config=config,scale=scale,**summaries(g)))
 overall=pd.DataFrame(rows);overall.to_csv(P/'overall_selected_scores.csv',index=False)
 paired=pd.read_parquet(P/'paired_station_changes.parquet')
 main=paired[paired.contrast.eq('T24_G_D minus T24_G_M')&paired.scale.eq('daily')]
 expanded=comp[comp.contrast.eq('T24_G_D minus T24_L_D')&comp.scale.eq('monthly_PUB')]
 dyn=comp[comp.contrast.eq('T24_G_D minus T24_G_M')&comp.scale.eq('daily')]
 space=comp[comp.panel.eq('SPACE_HELD')&comp.scale.isin(['daily','monthly_PUB'])]
 mian=metrics[metrics.selected&metrics.station_key.eq('棉江')&metrics.scale.eq('daily')&metrics.year.eq(2024)][['tag','n','NSE','r','RMSE','bias','centered_MSE']]
 monthlygates=comp[comp.year.eq(2024)&comp.contrast.eq('T24_G_D minus T24_G_M')&comp.scale.eq('monthly_PUB')]
 unstable=starts[starts.MAP_relative_gap.ge(.01)]
 # Compare the other legal entrance to the identical M baseline without selecting by scores.
 entrances=[]
 for scope in ['S191','T25S_L']:
  baseline=metrics[metrics.tag.eq(selected[scope+'_M'])]
  for tag in [scope+'_D_s0',scope+'_D_s1']:
   z=metrics[metrics.tag.eq(tag)].merge(baseline,on=['station_key','year','scale'],suffixes=('_D','_M'),validate='one_to_one')
   if scope.startswith('S'):z=z[z.panel_D.eq('SPACE_HELD')]
   for scale,g in z.groupby('scale'):entrances.append(dict(scope=scope,tag=tag,scale=scale,stations=len(g),RMSE_ratio=float(g.RMSE_D.mean()/g.RMSE_M.mean()),delta_NSE_median=float(g.NSE_D.median()-g.NSE_M.median())))
 pd.DataFrame(entrances).to_csv(P/'unstable_start_prediction_diagnostics.csv',index=False)
 cm=pd.read_parquet(P/'monthly_error_components.parquet');dynamic=[]
 for scope in ['T24_L','T24_G','S56','S113','S191']:
  a=cm[cm.tag.eq(selected[scope+'_D'])];b=cm[cm.tag.eq(selected[scope+'_M'])];q=a.merge(b,on=['station_key','year','month'],suffixes=('_D','_M'),validate='one_to_one')
  if scope.startswith('S'):q=q[q.panel_D.eq('SPACE_HELD')]
  for cohort,g in q.groupby('cohort_D'):
   means=g.groupby('station_key')[['mean_SSE_D','mean_SSE_M','within_SSE_D','within_SSE_M']].mean().mean()
   dynamic.append(dict(scope=scope,cohort=cohort,stations=g.station_key.nunique(),month_level_ratio=float(means.mean_SSE_D/means.mean_SSE_M),within_month_ratio=float(means.within_SSE_D/means.within_SSE_M)))
 pd.DataFrame(dynamic).to_csv(P/'paired_month_level_within_changes.csv',index=False)
 evidence=dict(all_paths_numerically_sufficient=True,all_paths_physically_legal=True,global_daily_improved_stations=int(main.delta_RMSE.lt(0).sum()),global_daily_total_stations=len(main),global_daily_monthly_gate_groups=int(monthlygates.monthly_gate.sum()),time_support_direction_reversals=int(direction.direction_reversed.sum()),time_support_comparisons=len(direction),distinct_start_configs=unstable.config.tolist(),selected_mianjiang=mian.to_dict('records'),within_month=dynamic)
 rt.write(P/'key_results.json',clean(evidence))
 head='# 简短结论\n\n28条路径全部完成独立重算，28/28数值充分、28/28通过原物理及来源一致性要求。S113_M_s0保留预算停止状态，但其独立投影梯度4.58×10⁻⁶已达标；无需因时间再续算。\n\n'
 head+='**扩域有明显但不均衡的收益，尚未解决区域偏差或棉江式外推问题。** 2024年G-D相对L-D，新增75站的站均月RMSE降至0.299倍，X组降至0.446倍，两组通过登记的完整月门槛；N组升至1.440倍，H组升至1.084倍。N组月NSE中位数从−0.560降至−2.128。\n\n'
 head+=f'**日信息在全域联合目标中的额外收益较小。** G-D相对G-M，15个合格HF站中{evidence["global_daily_improved_stations"]}站日RMSE下降；N/H组站均日RMSE分别下降约3.0%/0.4%，OTHER和棉江略升。四个原月面板组均未通过日信息增量的完整改善门槛。\n\n'
 head+='**棉江并未因加入全域训练而改善。** L-D与G-D的2024日NSE分别为0.357和−0.016，RMSE从0.348升至0.438 mg/L。S113独立空间留出下，M/D日NSE为0.335/0.328；日约束没有改善这一外层。S56与S191的日RMSE仅分别下降约0.8%和0.5%，三个空间主留出块均未通过完整月门槛，绝对精度仍不足。\n\n'
 head+=f'**收敛不等于起点稳定。** S191-D、T25S-L-M、T25S-L-D的两入口训练MAP差约11.6%、8.0%、7.4%；均按训练目标选点，其他入口结果保留。{evidence["time_support_comparisons"]}项公共日期日界比较中，D−M的日RMSE方向反转{evidence["time_support_direction_reversals"]}项，这只支持本轮所测试日界下的方向稳定。\n\n'
 head+='S191-D尤其需要保留条件性结论：另一个训练MAP较差的合法入口，其空间留出日RMSE为M基准的0.617倍，而按训练MAP选中的入口为0.995倍。不能根据已见留出成绩改选，但这表明空间预测明显依赖求解所到达的局部解。\n\n'
 head+='2025仅为1—11月月报敏感性延伸，OTHER继续获益、N/H仍有退化；没有2025日尺度主评价，不能与2024合并认证。全部属于回顾性检验，不自动替换主线。\n\n详见[专家诊断](expert_diagnosis.md)、[逐站指标](station_metrics.csv)、[起点诊断](unstable_start_prediction_diagnostics.csv)和[独立审计](completion_audit.json)。\n'
 (P/'short_conclusion.md').write_text(head,encoding='utf8')
 detail='\n## 主要发现与具体数值\n\n'+head.split('\n\n',1)[1]+'\n### 扩域：2024月报同队列\n\n'+tbl(expanded[['cohort','stations','eligible_NSE','delta_NSE','delta_r','RMSE_ratio','monthly_gate']])+'\n\n### 全域新增日信息：2024日预测\n\n'+tbl(dyn[['cohort','stations','eligible_NSE','delta_NSE','delta_r','RMSE_ratio','improved_RMSE_fraction']])+'\n\n### 月水平与月内动态误差\n\n先按站点平均站月误差、再站等权比较；比值小于1表示D优于M。\n\n'+tbl(pd.DataFrame(dynamic))+'\n\n### 三个空间主留出块\n\n'+tbl(space[['contrast','scale','cohort','stations','eligible_NSE','delta_NSE','delta_r','RMSE_ratio']])+'\n\n### 棉江：时间与空间面板分开\n\n'+tbl(mian)+'\n\n### 对求解入口敏感的配置\n\n'+tbl(unstable)+'\n\n'+tbl(pd.DataFrame(entrances))+'\n\n这些不同局部解均达小梯度标准；不能靠延长同一局部最优点的时间保证得到另一个解。本轮没有追加入口。\n'
 boot=[]
 for b in rt.read(P/'month_block_bootstrap.json'):
  if b['contrast']!='T24_G_D minus T24_G_M' or b['year']!=2024:continue
  for k,name in enumerate(b['columns']):
   q=b['quantiles'];boot.append(dict(cohort=b['cohort'],scale=b['scale'],quantity=name,q025=q[0][k],median=q[1][k],q975=q[2][k]))
 detail+='\n### 同步整月重采样\n\nG-D−G-M的1000次同步整月重采样分位数如下。N/H/OTHER月内中心化误差在这一重采样下倾向下降，但不等于所有区域月水平或日总误差改善；它只是已见月份的描述性稳定性，不是未来预测置信区间。\n\n'+tbl(pd.DataFrame(boot))+'\n'
 expert=P/'expert_diagnosis.md';base=expert.read_text(encoding='utf8');marker='\n## 主要发现与具体数值\n';base=base.split(marker)[0];expert.write_text(base+detail,encoding='utf8')
 print('CONCRETE_FINDINGS_SYNTHESIZED',flush=True)
if __name__=='__main__':main()
