"""Generate final narrative directly from completed evidence tables."""
from runtime import *
def table(frame):
 cols=list(frame.columns);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
 return '\n'.join(lines)
def main():
 pro=read(R/'data/protocol.json');verdict=read(R/'reports/verdict.json');summary=pd.read_csv(R/'reports/three_error_summary.csv');monthly=pd.read_csv(R/'reports/monthly_group_metrics.csv');primary=['MIX','H1-D29']+[c['id'] for c in pro['configs'] if c.get('primary')];support=[k for k,v in verdict.items() if v['status'] in ['PRIMARY_SUPPORT','EXPLORATORY_CANDIDATE']]
 physical={c['id']:read(R/'outputs'/c['id']/'summary.json')['gate']['pass'] for c in pro['configs']};pa=read(R/'reports/independent_physical_audit.json');ea=read(R/'reports/independent_evaluation_audit.json');assert len(pa)==17 and len(ea)==18
 matrices=[]
 for c in pro['configs']:matrices.append(dict(配置=c['id'],家族=c['family'],主配置=c.get('primary',False),物理通过=physical[c['id']],结论=verdict.get(c['id'],{}).get('status','CONTROL')))
 pd.DataFrame(matrices).to_csv(R/'reports/configuration_verdicts.csv',index=False)
 main=summary[summary.arm.isin(primary)][['arm','period','ratio_error','peak_error','base_error']]
 met=monthly[(monthly.year==2024)&monthly.cohort.eq('ALL')&monthly.arm.isin(primary)][['arm','stations','nse_qualified','nse_median','bias','rmse']]
 comparison=pd.read_csv(R/'reports/comparisons.csv');bootstrap=pd.read_csv(R/'reports/bootstrap_summary.csv');mass=pd.read_csv(R/'reports/mass_redistribution.csv');cent=pd.read_csv(R/'reports/centered_errors.csv');cent=cent.groupby(['arm','period']).apply(lambda x:x.groupby('station_key').centered_mse.mean().mean(),include_groups=False).rename('station_equal_centered_mse').reset_index();cent.to_csv(R/'reports/centered_summary.csv',index=False)
 statement='没有配置获得相对H1-D29、跨时期一致的三误差联合支持。' if not support else '达到相对H1-D29跨时期联合标准的配置：'+', '.join(support)+'。主配置与探索点资格必须分开。'
 expert=f'''# 潜力审计与专家诊断报告

{statement} 本报告为项目内部科学诊断，不是外部专家评审。

本轮完成18个登记配置（17个本轮全历史前向、1个H1-D29归档控制）；零拟合、零求根。所有候选均在相同70个前期事件与57个2024事件上评价。物理通过{sum(physical.values())}/18；不通过者只保留诊断，不授予结构支持。

## 主配置的实际观测结果

振幅为绝对对数比误差；峰值、背景为mg/L绝对误差。先取站内事件中位数，再对站等权。背景采用合格日中位数，峰值为相同观测日期的最大日均值。

{table(main)}

全部16个候选及其与MIX、D29的两类比较见comparisons.csv；不从2024成绩中挑一个点改称主检验。

具体而言，A-HALF的2024背景误差从MIX的5.6905增至10.8254 mg/L，A-FULL进一步增至16.3221，说明新增记忆并未选择性纠正事件峰。B主配置将背景误差降回6.4461，但仍差于MIX与D29。C-W0.8-P0.5的峰值误差1.4253低于D29的1.5615，却伴随0.62245的振幅误差和4.9537的背景误差，不能构成联合支持。C-W0.8-P1振幅误差较MIX略低，但峰值、背景更差。C-W0.8-P0.8只在2024对MIX有极小联合改善，前期不一致，且该比例配置接近共同浓度退化，不能放大为机制发现。

## 机制覆盖究竟增加了什么

1. A增加跨日快路氮记忆。由于gF=gu×phi，分配当天转移量不改变同状态当天快路输出；收益或损失只能通过后续库存、慢层供给和损失反馈产生。
2. B有限交换能逐步消除分配偏好。rho=1时两个pi配置的净输出与状态在代数上相同，内部交换账本不同。这两个配置只提供一致性验证，不是两份机制支持。
3. C为有限水量域，水域体积没有重复使用完整上层水量。但pi=omega、持续接触时两域浓度保持相等，接近共同池控制；源分配不等于水分配的点同时改变源可达性，不能把效应全部归于不完全混合。
4. 原接触掩码阻止氮输出但不阻止水离开；这会打破C的等浓度不变量。C主配置的偏离可能含该继承近似，不可直接当作真实域分离证据。

这些是在揭示本轮TN评分前登记的代数审计。脉冲诊断见pulse_screen.csv：每个1kg新mobile脉冲使用真实水文、原共同损失，不施加需求，共256个季节起始日×230河段；报告1/7/30/90日快慢输出和剩余。它是结构能力诊断，不是实测预测评分。

## 完整月报与平均水平

2024完整116站月报面板，NSE资格至少8个月且方差为正；分组明细未混入HF标签。

{table(met)}

日资格、月内中心化误差及全部站点分别保存在daily_station_metrics.csv、centered_summary.csv和monthly_station_metrics.csv。严重平均水平失真不能被振幅或个别站改善掩盖。

## 质量再分配与稳定性

{table(mass[(mass.period=='2024')&mass.arm.isin(primary)][['arm','fast','slow','transfer','uptake','loss','fast_store_yearend']])}

单位kg；库存是2024年末值。固定q不固定T；本轮没有长期总输出中性化，不把变化解释为纯时间整形。日状态、分域摄取/损失、交换与有效浓度完整输出；A/B有效浓度不是两个独立水域浓度。极小水量不裁剪、不加epsilon，完整极值见effective_concentrations.csv。

seed1729的1000次同步整月配对重采样见bootstrap_summary.csv与抽样原台账。区间只描述本次多配置回顾性结果稳定性，不是搜索后确认性显著性，也不是新时期验证。

## 结论边界和后续衔接

{statement} 无论结论正负，都仅适用于这些固定输入、结构与网格。新近动员氮不是年轻水，不识别DON；没有完成SAS机制家族穷尽。只改善MIX而不能超过D29的点，不视为解决当前问题。若只有非主点通过，只列探索候选，下一轮必须另行预注册，不能直接拟合或替换主线。
'''
 (R/'reports/潜力审计与专家诊断报告.md').write_text(expert,encoding='utf-8')
 methods='''# 实际方法、失效参数与偏离

## 科学身份

继承20_4的H1域、完整容量/库存、水库水量、四源、月初源与需求和固定q。原30参数数值保存，但本轮核中仅共同损失的8参数和河道去除1参数继续发挥作用；21个原动员/分水参数不再作用于新陆相。参数逐项见parameter_effects.csv。未调用优化器、求根、训练清洗或2025驱动。

共同顺序：源进legacy→全上层比例摄取→q转移→分池→交换→输出→共同上层损失。下层仍按原递推。四源被动标签只覆盖158、225，不能宣传成230河段来源识别。

A和B按共享上层参考体积抽取，C使用VF=omega*Spost+Qf、VP=(1-omega)*Spost+Qp。C的内部水交换携带供水域交换前浓度；正H为P向F，输出exchange正值则定义为F向P氮质量，符号须区分。所有体积换算明确：数据水深mm乘area_ha×10得到m³，浓度1000kg/m³得到mg/L。

## C准入与浮点处理

净入水Vu−Sprev出现635199个微小负值，最小−5.68434e-14 mm，负值总和约−4.18688e-10 mm。未直接裁剪。核查生产者101—123行可得excess=p−infiltration、upper_available=upper+excess；守恒非负分配保持上层总水。独立土壤水账本的净入水与上层差最大1.42109e-13 mm，在继承生产者atol=rtol=1e-8回放精度内。

据此将负值登记为重建相消残差；实际分域供水体积采用代数等价omega*Vu及(1−omega)*Vu，避免先减后加。没有修改冻结水文，没有把负入水截成零，没有扩大浓度分母。两域水量平衡最大约1.14e-13 mm。此判断、源代码与初态哈希见C_water_admission.json。

## 执行与工程修复

每366日保存四库存、四源状态、累计块计数及块哈希。17个完整前向各64块。MIX首次已完成回归，但写summary时列表被数组索引导致类型错误；修复元数据索引后从已保存全历史状态重新执行输出与审计，没有重新生成或改变物理轨迹。独立脉冲脚本首次整数初始化导致dtype错误，改为float64后通过；不涉及科学参数调整。

A-FULL首次运行的共同mobile入池量理论为零，却因残差相减积累出−1.11534e-7 kg的下层库存，未通过原非负门槛。保留outputs_attempts/A-FULL-v1全部失败结果及data/code_versions物理核v1；用非负比例分量直接计算该端点的共同库存，未裁剪、未放宽容差，重算同一配置后通过。其余配置的运算分支未变。实际完整陆地运行尝试18次、最终科学配置18个（17新前向身份＋1归档控制），A-FULL累计128块，其余各64块；重试不清除全局计数或预算。

B完全交换的代数重复在脉冲结果中识别，仍保留两个登记输出作一致性检查，不作为独立证据。C主配置的等浓度退化在评分揭示前登记；没有根据结果更换主配置。

MIX冷启动和完整前向峰值用于控制器内存预留，按峰值×1.2加运行进程增长；CPU/RAM90%停派、85%恢复。追加日志记录资源、派发和退出。每worker一线程、CPU float64、独立控制器；无定时任务。日递推可断点恢复，河网阶段从保存的陆地通量重放，不提供河网内部断点。

最终日志审计发现共享追加文件未使用跨进程锁，存在交错：有效块尝试行1149条，完整检查点证明1152块完成，3条原始尝试时间记录缺失；资源日志有4条非空残片和2条空行，丢失采样时间无法可靠重建。原始日志未修改，completed_block_ledger.csv明确标记检查点证据而不补造时间戳。新增msvcrt跨进程锁，经8进程×100条并发测试无缺失/重复。该修复发生在正式前向后，不能追认原资源日志无缺口；物理轨迹、检查点、子任务退出和独立重算仍完整。

预测冻结后才加载共同事件和原始model_eligible月报。月报整月等日均值仍是条件观测假设。评价没有重选事件、日界、峰窗或时滞；无配置通过删观测日改善成绩。数据门槛和逐站分母随产品交付。

原目录只读。独立审计为项目内部不同代码重算，不是外部专家审查。全部代码最终哈希是交付身份；仅声明核心递推和协议于前向前冻结，不将后处理脚本说成早已冻结。
'''
 (R/'reports/实际方法与偏离.md').write_text(methods,encoding='utf-8')
 maxerr=max(max(x['reference_errors_kg'].values()) for x in pa);maxlocal=max(read(R/'outputs'/a/'summary.json')['gate']['local_balance_max_kg'] for a in physical if a!='H1-D29');elapsed=(time.time()-pro['created'])/3600
 audit=dict(status='COMPLETE',scientific_configurations=18,new_full_forwards=17,reused_D29=1,physical_pass_count=sum(physical.values()),full_independent_audits=len(pa),independent_event_audits=len(ea),reference_max_error_kg=maxerr,local_max_kg=maxlocal,elapsed_hours=elapsed,budget_hours=24,fits=0,roots=0,supported=support,external_review=False)
 put(R/'reports/completion_audit.json',audit)
 (R/'reports/独立完成审计.md').write_text(f'''# 独立完成审计

完成18个登记配置：17个完整历史前向、1个H1-D29读取控制。物理通过{sum(physical.values())}/18，独立完整递推审计17个，独立事件复算18个。最大独立通道差{maxerr:.6g} kg，最大局地残差{maxlocal:.6g} kg；没有放宽1e-6局地/来源、1e-7非负/摄取或网络相对1e-10门槛。

MIX十个公共历史通道和站界逐位恢复20_4；B零交换恢复A。C强制完全混合夹具恢复共同浓度输出。全部实际检查点恢复覆盖1964闰日，陆地和来源逐位一致，64块哈希逐一核验。事件独立重建三指标一致至1e-12。全部预测在标签评价前冻结，继承事件/观测哈希通过。

本轮存在的解释限制已登记：B完全交换重复、C比例源的等浓度退化、无接触水氮脱耦、C入水浮点相消、21个失效参数、试点来源范围。没有把实现正确等同预测改善。

交付追溯另有明确缺项：共享追加日志并发交错造成3条原始块尝试时间记录缺失，部分资源采样行不可恢复。完整1152块可由保留检查点核验，但不得称原日志完整。跨进程锁修复及8×100条并发验证通过，详情见log_integrity_audit.json；本轮终审状态为完成但披露日志缺口。

结果：{statement}

运行约{elapsed:.3f}小时，未用完预算不构成需要扩大实验的理由。全部轨迹、评价、独立审计与资源日志交付。审计程序与生产worker分离，仍属于内部重算，非外部专家审查。最终文件完整性见data/delivery_manifest.json。
''',encoding='utf-8')
 (R/'reports/简短结论.md').write_text('# 简短结论\n\n'+statement+'\n\n三家族分别检验快路记忆、有限交换和有限水量域；配置多不等于机制证据独立。完成有限矩阵后结束，不拟合、不增加结构、不替换主线。\n',encoding='utf-8')
 (R/'README.md').write_text('''# 20260920_5 有限机制家族筛查

已完成18个登记配置，零拟合、零求根。结果与限制请从以下报告阅读：

- [潜力审计与专家诊断](reports/潜力审计与专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [简短结论](reports/简短结论.md)

## 数据接口

data/protocol.json为固定矩阵；input_manifest、physical_arrays、C_arrays_freeze、science_code_freeze及prediction_freeze分别登记来源、物理数组、C水域、代码与评价前预测。原始来源保持只读，未使用2025。

各配置land_history.npy轴为通道×23376日×230河段；source_tags.npy为通道×日×2试点河段×4源。日期1961-01-01起，河段顺序为lineage_H1域topology的global_reach_ids；来源名沿用父模型。通道顺序见summary.json及scripts/family_kernel.py的CHANNELS，库存kg、通量kg/日，exchange为带符号的F向M氮转移。effective_*_concentration单位mg/L；A/B为参考体积有效浓度，C为分域浓度，不与站点TN混称。

daily_station.parquet包含2021—2024的116站质量kg、水量m³、p浓度mg/L；daily_HF_history为固定HF站全历史。event_scores/coverage/pairs、HF_daily_predictions和monthly_predictions分别登记事件、HF日与原月报支持。H1-D29复用已验证历史控制，不新增前向。

annual_reach_ledger为年通量与年末库存；reports/mass_redistribution中的库存取时期末年，非多年库存相加。原共享水库账本存river_history.npz。data/C_*保存mm水域体积、交换水量和无量纲抽取率；每个C配置通过omega索引这些共同数组。

reports/three_error_summary保留站等权及事件直接平均；comparisons和bootstrap_summary为两类控制的配对变化；monthly_group_metrics包括ALL及N/H/X/OTHER；event_curves可用于重绘；pulse_screen为无标签潜力诊断。资格和分母不可跨产品混用。

追溯限制：原共享追加日志存在少量并发交错，3条尝试时间记录缺失、部分资源采样不可恢复；原日志保留，完成块由独立检查点账本核验。锁修复仅保证修复后写入，不能补回历史。详见reports/log_integrity_audit.json。

## 重现

使用完整conda sparrow、Python float64、每worker一线程。在归档副本中依次运行scripts下setup、prepare、admit_water、extended_checks、algebra_audit、pulse_screen、worker MIX、controller、evaluate、audit、delivery_analysis、write_reports和seal。不要覆盖已封存目录；脚本默认新目录R由自身位置决定。检查点支持日递推恢复，旧文件和既有结果受身份检查保护。
''',encoding='utf-8')
 print('REPORTS_COMPLETE',audit)
if __name__=='__main__':main()
