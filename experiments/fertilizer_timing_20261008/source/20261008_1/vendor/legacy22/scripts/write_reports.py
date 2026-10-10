"""Source-backed Chinese reports; no causal mechanism certification."""
import json
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def table(df):
    def cell(v):
        if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return '—'
        value=f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
        return value.replace('|','\\|').replace('\n','<br>')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(lines)
def put(name,text):
    (R/'reports'/name).write_text(text,encoding='utf-8')
def main():
    c=rt.read(R/'reports/completion.json');paths=pd.read_csv(R/'reports/path_summary.csv')
    core=pd.read_csv(R/'reports/core_paired_table.csv') if (R/'reports/core_paired_table.csv').exists() else pd.DataFrame()
    events=pd.read_csv(R/'reports/event_centered_summary.csv') if (R/'reports/event_centered_summary.csv').exists() else pd.DataFrame()
    allcore=core[core.group.eq('ALL')] if len(core) else core
    selected=paths[paths.tag.isin(c['selected'].values())]
    conclusions=[]
    for fold in ['F23','F24']:
        if fold+'_R' not in c['selected'] or fold+'_X' not in c['selected']:
            conclusions.append(f'- {fold}：合法配对缺项，不签发结构收益结论。');continue
        r=selected[selected.tag.eq(c['selected'][fold+'_R'])].iloc[0];x=selected[selected.tag.eq(c['selected'][fold+'_X'])].iloc[0]
        conclusions.append(f'- {fold}：候选调制参数={x.eta:.6g}；训练总目标变化={x.objective-r.objective:+.6g}，数据项变化={x.data-r.data:+.6g}，原先验变化={x.original_prior-r.original_prior:+.6g}，新增先验={x.new_prior:.6g}。')
        if x.objective<r.objective and (r.original_prior-x.original_prior-x.new_prior)>max(r.data-x.data,0):
            conclusions.append('  总目标下降主要由先验项下降贡献，不能将这部分收益称为数据拟合改善。')
        q=allcore[allcore.fold.eq(fold)];ev=events[events.fold.eq(fold)]
        if len(q) and len(ev):
            day=q[q.scale.eq('HF_day')].iloc[0];mo=q[q.scale.eq('PUB_month')].iloc[0];e=ev.iloc[0]
            same=bool(r.numerical and x.numerical and r.physical and x.physical) and day.rmse_change<0 and e.centered_mse_X<e.centered_mse_R and e.peak_error_change<=0 and e.base_error_change<=0 and (e.peak_error_change<0 or e.base_error_change<0) and mo.rmse_change<=0 and mo.abs_bias_change<=0 and mo.nse_difference_of_medians>=0
            conclusions.append(f'  日RMSE变化={day.rmse_change:+.6g} mg/L；月报NSE中位数之差={mo.nse_difference_of_medians:+.6g}，逐站差中位数={mo.nse_median_paired_change:+.6g}，改善站比例={mo.nse_improved_fraction:.1%}。'+('满足本轮多尺度同向的保守继续建议。' if same else '存在尺度或站点间权衡，未满足全部同向的保守继续建议；这不等于机制无效。'))
            if mo.nse_difference_of_medians>0 and mo.nse_median_paired_change<=0:
                conclusions.append('  总体NSE中位数上升，但逐站配对差中位数未上升，不能据此声称典型站改善。')
    verdict_text = "实现与物理验收通过，但本轮不支持把状态调制D29作为恢复缺失事件振幅的主要方案。2023年日RMSE下降约2.38%，2024年却上升约1.41%，月报误差也在2024年增加；两折事件振幅误差均增加。结论只限于本次表达、正则和试验范围，不排除其他水文状态响应。"
    conclusions.insert(0, verdict_text)
    cols=['fold','scale','nse_paired_stations','nse_median_R','nse_median_X','nse_difference_of_medians','nse_median_paired_change','nse_improved_fraction']
    text='# 专家诊断报告\n\n本轮检验水文状态调制的有效动员表达，采用已见资料的回顾性配对评价。正式基线是在H1上重新校准的D29，不是近期结构筛查的旧参数回放。\n\n'
    text+='\n'.join(conclusions)+'\n\n## 配对证据\n\n'+(table(allcore[cols]) if len(core) else '配对结果缺项。')+'\n\n'
    if len(core):text+='## 绝对浓度误差\n\n'+table(allcore[['fold','scale','paired_stations','rmse_R','rmse_X','rmse_change','abs_bias_R','abs_bias_X']])+'\n\nRMSE及偏差单位均为mg/L，按站等权。误差下降不等于绝对精度已经充分。\n\n'
    if len(events):text+='## 同支持事件与月内误差\n\n'+table(events[['fold','stations','events','amplitude_error_change','peak_error_change','base_error_change','centered_mse_R','centered_mse_X']])+'\n\n'
    if len(events):text+=table(events[['fold','amplitude_error_R','amplitude_error_X','peak_error_R','peak_error_X','base_error_R','base_error_X']])+'\n\n峰值、背景为mg/L绝对误差；振幅为无量纲对数误差。先取站内事件中位数，再站等权。\n\n'
    stability=R/'reports/start_stability.csv'
    if stability.exists():text+='## 两入口与局部解\n\n'+table(pd.read_csv(stability))+'\n\n目标接近不证明全局最优；数值未充分的入口不能用于认证入口稳定性。\n\n'
    bp=R/'reports/bootstrap_intervals.csv'
    if bp.exists():
        intervals=pd.read_csv(bp);focus=intervals[intervals.metric.isin(['rmse_change','nse_median_paired_change','peak_error_change','base_error_change','amplitude_error_change'])]
        text+='## 同步整月重采样的描述性稳定性\n\n'+table(focus)+'\n\n区间跨零时，改善方向的稳定性不足；这些不是独立未来验证置信区间。\n\n'
    text+='## 新增方向与参数职责\n\n'
    for fold in ['F23','F24']:
        path=R/'reports'/f'direction_{fold}.json'
        if not path.exists():continue
        d=rt.read(path)
        if d.get('status')!='DIAGNOSTIC_COMPLETE':text+=f'{fold}：诊断缺项。\n\n';continue
        fraction=d['explained_fraction'];value='不可可靠解释' if fraction is None else f'{fraction:.2%}'
        text+=f"{fold}：训练数据加权下新增一阶方向被原30参数列空间解释的比例为{value}，机器精度数值秩{d['rank']}，新增方向范数{d['weighted_new_direction_norm']:.6g}。另报告以步长差异界定的可分辨秩{d.get('resolved_rank')}及其解释比例，避免有限差分噪声被包装成新增方向。该局部诊断不决定是否拟合；对称扰动的二阶响应见direction_{fold}.json。\n\n"
    text+='## 解释边界\n\n调制位于D29软饱和外部，扩大了可达动态范围。收益可以来自范围扩大、参数职责重分配及水文响应表达，不能单凭TN判定真实生地化来源、年轻水或SAS机制。局部列空间投影不等同于有界参数可行方向的复制能力，也不代替非线性有限变化检验。站界、缺失人为源和水量偏差等既有问题没有因本轮而消失。\n\n所有站点和事件的改善与退化均保留。重采样使用1000次同步整月抽样，只描述已见资料的稳定性。某一指标微小反向不自动否定结构；本轮不支持也只限于当前表达、先验和测试范围。只有在数值、物理与跨时期收益分别核实后，才讨论后续空间检验；本轮不替换主线。\n'
    phys=R/'reports/physical_period_comparison.csv'
    if phys.exists():
        p=pd.read_csv(phys);p=p[p.period.eq('evaluation')]
        text+='\n## 质量再分配与补偿\n\n'+table(p[['fold','uptake_kg_change','mineral_loss_kg_change','terminal_kg_change','M_end_kg_change','L_end_kg_change','uptake_demand_ratio_change']])+'\n\n质量单位为kg；摄取/需求为无量纲。全部年度及1961—2020参考期见physical_year_comparison.csv与physical_period_comparison.csv，原参数变化见parameter_compensation.csv。浓度收益不能自动解释为纯时间形状收益。\n'
    text+='\n\n## 综合判读与后续衔接\n\n两折总目标下降约73%和88%来自净先验下降，数据损失相对改善仅约0.076%和0.047%。结合新增一阶方向约95%的局部列空间重叠，更符合原参数职责和有效正则的重新分配，不能视为新增机制已经识别。仍存在约4%—5%的加权平方方向未被解释，二阶响应也非零；不能称新项完全重复。\n\nF24的HF月面板特别说明配对口径的重要性：NSE中位数之差为+0.03354，但逐站差中位数为−0.07276，仅4/14站改善。不能据前者宣称典型站改善。F24日与月报的配对变化也总体不利，原模型和候选的NSE中位数仍为负，绝对精度没有达标。\n\n同步整月重采样中，2023日RMSE差的区间跨零，2024日及月报RMSE差的区间均在恶化方向；两折振幅误差差值区间均在恶化方向。事件峰值和背景区间跨零，不把这些小变化解释为稳定收益。\n\n月内训练损失占比小不等于梯度弱：F24其尺度化梯度范数约0.0486，与月水平的0.0525接近；两者夹角余弦约−0.917。最优点附近分项梯度相互抵消部分来自驻点条件，不能仅凭此认定信息不兼容或建议增大日权重。F23候选月内训练项略增，F24略降。\n\nF23原D29两个入口到达不同局部解，目标差约0.03418；本轮选择较优基线。两折候选入口目标近乎一致，但这不是全局最优证明。没有证据将当前预测表现归因于未达到既定收敛标准。\n\nF24旧参数H1回放的日RMSE约1.16143，重新校准原D29后为1.10837，说明重新校准本身有收益；加入调制后为1.12400，未延续该收益。评价年出口质量约下降0.93%和2.44%，伴随损失与库存变化，并非只重排日内或事件时间。\n\n建议保留本轮作为已完成的负向结构证据，不自动进入本表达的空间推广或生产替换。后续若提出新实验，应明确其不同于本轮的输入信息或结构响应，并继续同时约束绝对浓度、事件振幅及逐站代价。\n'
    put('专家诊断报告.md',text)
    prep=rt.read(R/'reports/preparation.json')
    methods=f'''# 实际方法与偏离

总状态：{c['status']}；累计墙钟时间约{c['wall_hours']:.3f}小时，预算48小时。详细调用计数见computation_accounting.json。

## 数据与身份

H1全部26个实际数组核验SHA256，显式只读引用20260917_5/data/domains/FULL24C。F24训练表、注册表和设计逐字节继承并独立重算；两个入口复用，不重复拟合。F24历史D入口0来自当时同折最佳M点，并非原预设0；入口1为原预设1。本轮F23基线从两原预设独立开始，候选入口0来自本折最佳基线，候选入口1为原预设1加调制参数0.25。F23从统计删除前HF及原始月报重建，独立反事实修改留出值后清洗、来源并集和尺度完全一致。F23正式规模：{prep['cohorts'][0]['stations']}站、{prep['cohorts'][0]['months']}站月、{prep['cohorts'][0]['hf_months']}HF站月、{prep['cohorts'][0]['hf_days']}日值。

## 结构与数值

原30参数和共享调制参数联合估计。新参数范围[-1,1]，先验0.5*(0.03/17)*(eta/0.5)^2；0.03是固定正则设定。调制在原软饱和外部，使用快水与渗漏；对数域保护避免乘法先溢出。零调制时完整历史预测、库存、通量及原梯度恢复。每折候选保存基线加零调制参数的合法点，不因优化器返回更差点而丢弃它。

继承L-BFGS-B、三点差分有界TRF和解析精修；检查、差分及恢复调用在路径预算内累计。两入口分别交付，不根据留出成绩选入口。原坐标投影梯度≤1e-5才称数值充分。共享先验和物理容差没有放宽。

## 明确的方法实现说明

输出敏感度矩阵采用完整历史预测有限差分，并用解析目标伴随核验，未实现新的解析输出Jacobian。这一方法与完整历史目标梯度验收区分登记；使用步长比较、SVD截断容差及二阶差分报告数值稳定性。它不增加科学模型或拟合路径。

历史数组延续至2024用于完整历史一致性验收，F23目标及评价只使用相应年份；未来输入反事实检验确认不能改变过去预测。训练worker在文件哈希验证前安装标签屏障。

## 资源与审计

CPU float64、每worker一线程，完整冷启动峰值实测约5.061 GiB，另乘1.2并预留运行进程增长。90%停止派发、85%恢复；RAM压力请求检查点退让。追加日志使用跨进程锁，8进程共800行测试无丢失。无定时任务。独立审计为与拟合循环分离的重算程序，不是外部专家审查。

首次控制器启动因恢复辅助脚本仍引用旧文件名而在派发前退出；已修复该运行层引用并保留controller_failure.json及首次stderr。该修复未改变科学代码、起点、参数或累计时钟。拟合全部结束后，报告入口另有一处旧finalize_global.py文件名引用；已修复为本轮finalize.py后继续审计，没有重跑拟合。两个失败记录均保留。

缺项：{c['missing']}。详细数值及资源事件见path_summary.csv和work/resource_events.jsonl；不将条件性缺项写成结构失败。
'''
    put('实际方法与偏离.md',methods)
    audit=f"# 独立完成审计\n\n状态：{c['status']}。\n\n- 数据支持：折内重建、F24复用身份及留出标签反事实检查已登记。\n- 实现正确性：完整历史嵌套、31方向、正负调制、端点、独立小核及基础设施验收见对应JSON。\n- 数值充分：所选四配置全部充分={c['all_selected_numerically_sufficient']}；完整八入口状态见下表。\n- 物理合理性：所选点独立重算见reports/independent；保留原局地、网络、来源及非负容差。\n- 预测收益：以逐站配对表和专家报告为准，不把数值收敛当成预测有效。\n- 时间支持稳健性：本轮仅采用冻结共同日期和事件，未重新开展日界敏感性，不认证额外时间支持稳健性。\n- 空间推广：本轮没有独立空间外层，不能认证。\n- 审查性质：独立程序重算，不是外部专家审查。\n\n"+table(paths[['tag','reused','physical','numerical','objective','pg']])+'\n'
    put('独立完成审计.md',audit)
    put('简短结论.md','# 简短结论\n\n'+'\n'.join(conclusions)+f"\n\n交付状态：{c['status']}。数值充分、物理合法和预测收益分别判断。本轮结束，不自动启动下一轮或替换主线。\n")
    (R/'README.md').write_text('# 水文状态调制的氮动员联合校准\n\n'+f"状态：{c['status']}。两折、八条逻辑路径（F24原D29两条复用，六条新拟合）。\n\n"+'\n'.join(conclusions)+'\n\n阅读顺序：reports/简短结论.md、reports/专家诊断报告.md、reports/实际方法与偏离.md、reports/独立完成审计.md。逐站配对见reports/paired_station_changes.csv，总体与逐站NSE差口径并列于reports/core_paired_table.csv。完整身份与交付哈希见data；资源与调用轨迹见work。\n\n旧目录只读；所有结论均为已见资料的回顾性检验，不自动替换主线。\n',encoding='utf-8')
if __name__=='__main__':main()
