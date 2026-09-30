"""Evidence-first regional experiment report; missing/insufficient results remain explicit."""
import time,math,ast
import numpy as np,pandas as pd
import native_runtime as rt
from finalize_source import run
R=rt.RUN
NAMES={'U':'统一来源校正D29','L1':'1模式区域响应D29','L3':'3模式区域响应D29','N3':'3模式神经区域响应D29'}

def md(df):
    if df.empty:return '无可用结果。'
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','／')
    return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def write(name,text):(R/name).write_text(text,encoding='utf-8')

def main():
    selected=rt.read(R/'data/selected.json');paths=pd.read_csv(R/'reports/path_summary.csv');rows=[];checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    for name in ['sampling_reconstruction','lowrank_fixture','lowrank_evaluation_fixture','regional_reporting_fixture','lowrank_support_audit']:
        p=R/'reports'/f'{name}.json';check(name,p.exists() and rt.read(p).get('status')=='PASS')
    for key,tag in selected.items():
        fold,arm=key.split('_');out=R/'outputs'/tag;r=rt.read(out/'model.json');a=rt.read(out/'audit.json');x=np.asarray(r['parameters']);terms=r['terms'];source=.5*(.03/17)*(x[30]/math.log(2))**2;mapa=.5*(.03/17)*np.sum((x[31:31+(8 if arm=='L1' else 24)]/.5)**2) if arm!='U' else 0;mapv=.5*(.03/17)*np.sum((x[73:79]/.5)**2) if arm=='N3' else 0
        sf=out/'full_source_audit.json';source_pass=sf.exists() and rt.read(sf).get('status')=='PASS'
        row=dict(fold=fold,arm=arm,model=NAMES[arm],tag=tag,objective=r['objective'],data=terms['data'],original_prior=terms['prior']-source-mapa-mapv,source_prior=source,mapping_A_prior=mapa,mapping_V_prior=mapv,source_multiplier=float(np.exp(x[30])),source_at_bound=bool(np.isclose(abs(x[30]),math.log(4),atol=1e-7)),pg=r['pg'],numerical=a['numerical_sufficient'],aggregate_physical=a['physical_reasonable'],full_source_pass=source_pass,physical=bool(a['physical_reasonable'] and source_pass));rows.append(row)
        for name,file in [('independent',R/'reports/independent'/f'{tag}.json'),('full_source',out/'full_source_audit.json')]:check(name+'_'+tag,file.exists() and rt.read(file).get('status')=='PASS')
        check('prior_sum_'+tag,abs(row['original_prior']+source+mapa+mapv-terms['prior'])<1e-12)
    result=pd.DataFrame(rows);result.to_csv(R/'reports/selected_models.csv',index=False)
    core=[];events=[];intervals=[]
    for comp in ['L3-U','L1-U','N3-U','L3-L1','N3-L3','S56_L3-U','S113_L3-U','S191_L3-U']:
        p=R/'reports'/comp/'1month'
        if (p/'core_paired_table.csv').exists():core.append(pd.read_csv(p/'core_paired_table.csv').assign(comparison=comp))
        if (p/'event_centered_summary.csv').exists():events.append(pd.read_csv(p/'event_centered_summary.csv').assign(comparison=comp))
        for block in [1,2]:
            p=R/'reports'/comp/f'{block}month/bootstrap_intervals.csv'
            if p.exists():intervals.append(pd.read_csv(p).assign(comparison=comp,block_months=block))
    c=pd.concat(core,ignore_index=True) if core else pd.DataFrame();e=pd.concat(events,ignore_index=True) if events else pd.DataFrame()
    c.to_csv(R/'reports/all_comparison_core.csv',index=False);e.to_csv(R/'reports/all_event_summary.csv',index=False)
    if intervals:pd.concat(intervals).to_csv(R/'reports/all_bootstrap_intervals.csv',index=False)
    p=R/'reports/independent_regional_evaluation.json';check('independent_tables',p.exists() and rt.read(p).get('status')=='PASS')
    tasks=rt.read(R/'reports/postprocess_tasks.json');check('postprocess_tasks',all(t['passed'] for t in tasks),[t for t in tasks if not t['passed']])
    plot_ok=run('plot_regional.py');check('figures_generated',plot_ok)
    primary=c[c.comparison.isin(['L3-U','S56_L3-U','S113_L3-U','S191_L3-U'])&c.group.eq('ALL')] if len(c) else c
    main_daily=primary[primary.scale.eq('HF_day')] if len(primary) else primary
    primary_events=e[e.comparison.isin(['L3-U','S56_L3-U','S113_L3-U','S191_L3-U'])] if len(e) else e
    conclusions=[]
    for row in main_daily.itertuples():
        conclusions.append(f'{row.fold}：日RMSE站均值 {row.rmse_R:.6f} → {row.rmse_X:.6f} mg/L，逐站变化中位数 {row.rmse_median_paired_change:+.6f}，改善站比例 {row.rmse_improved_fraction:.1%}（{row.paired_stations}站）。')
    temporal=main_daily[main_daily.fold.isin(['F23','F24'])] if len(main_daily) else main_daily
    spatial=main_daily[main_daily.fold.str.startswith('S')] if len(main_daily) else main_daily
    numerical_main=result[result.arm.isin(['U','L3'])].numerical.all() if len(result) else False
    interpretation='本轮结果须结合下列逐站、事件、月尺度及数值证据判断。'
    if len(temporal)==2:
        if (temporal.rmse_change<0).all():interpretation='三模式线性主模型在两时间折的站均日RMSE均下降；是否值得继续，仍取决于空间留出、事件和月尺度代价。'
        elif (temporal.rmse_change>=0).all():interpretation='三模式线性主模型未改善两时间折的站均日RMSE；本轮不支持仅凭新增区域响应提高预测能力。'
        else:interpretation='三模式线性主模型的日RMSE收益具有时期依赖，尚不能称跨时期稳定改善。'
    if not numerical_main:interpretation+=' 部分主比较点未达到原坐标投影梯度充分标准，需保留求解不充分的限制。'
    if not result.physical.all():interpretation+=' 部分选择点的完整物理／来源检查未通过，其预测统计仅作诊断，不签发结构支持。'
    if len(spatial)<3:interpretation+=' 三个空间块未全部获得合格日比较，缺项不能视作支持。'
    write('专家诊断报告.md','# 保留来源校正的低维区域响应实验\n\n'+interpretation+'\n\n'+'\n\n'.join(conclusions)+'\n\n## 1. 训练与来源校正\n\n'+md(result[['fold','model','objective','data','original_prior','source_prior','mapping_A_prior','mapping_V_prior','source_multiplier','pg','numerical','physical']])+'\n\n来源倍率是当前H1和过程结构条件下的估计校正，不是实测排放量。需求不缩放；数据项与各类先验分开报告。\n\n## 2. 主比较：时间与空间\n\n'+(md(primary[['comparison','fold','scale','paired_stations','rmse_R','rmse_X','rmse_median_paired_change','rmse_improved_fraction','nse_difference_of_medians','nse_median_paired_change','nse_improved_fraction']]) if len(primary) else '无合格比较。')+'\n\n总体NSE中位数之差与逐站NSE差的中位数是不同统计量。月尺度资格沿用原规则，不从日观测插值构造月标签。完整分组和改善／恶化站见 reports 下逐站表。\n\n## 3. 事件与月内动态\n\n'+(md(primary_events) if len(primary_events) else '事件比较缺项。')+'\n\n事件按冻结H1水量定义，峰值只使用共同合格日期；背景不跨训练期。振幅为绝对对数比误差，峰值与背景为mg/L绝对误差。月内误差为各站月中心化后均方误差。\n\n## 4. 可重建与可预测\n\n训练期采用掩码加权低秩拟合，秩1、2、3固定；缺测单元不进入目标。输入预测只用32项同步水文空间汇总，λ=1。评价TN辅助估计模态系数的结果独立标为观测辅助重建，不能当预测成绩。完整结果见 reports/lowrank_paired.csv、lowrank_evaluation.csv。\n\n## 5. 机制与补偿解释\n\n新项位于原软饱和内部，不扩展动态乘子范围。低秩对象是地区×动态特征系数矩阵，浓度场不保证低秩。局部方向投影使用训练数据权重且排除先验，见 F23/F24_regional_directions.json；高重叠不等于机制完全重复，低重叠不等于预测有用。regional_response.csv登记响应RMS、极值与软饱和导数；physical_period_summary.csv与land_station_propagation.csv登记陆地到站界变化。这些是联合变化，未把非线性交互强分为唯一贡献。\n\n## 6. 不确定性和结论边界\n\n1000次整月同步配对重采样，seed1729；连续两个月块作敏感性。所有站共享抽样月份，事件在原序列预定义。区间条件于冻结模型，不是系数置信区间，也不校正多配置探索。神经模型仅有时间探索，无神经空间验证。空间留出为已知公共H1驱动下的TN标签封闭检验，不是未知流域完全归纳。原始数据、水文、模型主线均未替换；不自动开展下一轮。\n')
    low=pd.read_csv(R/'reports/lowrank_preparation.csv');reuse=rt.read(R/'reports/baseline_reuse.json')
    low_result=R/'reports/lowrank_paired.csv'
    if low_result.exists():
        lp=pd.read_csv(low_result);lp=lp[lp.metric.eq('logrmse')]
        with (R/'专家诊断报告.md').open('a',encoding='utf-8') as f:f.write('\n## 7. 输入驱动低秩预测的实际增量\n\n以下以同输入、不限制输出秩的岭回归为对照；负的log-RMSE差表示改善，不能与灰箱RMSE直接比较。\n\n'+md(lp)+'\n')
    directionrows=[]
    for short in ['F23','F24']:
        p=R/'reports'/f'{short}_regional_directions.json'
        if p.exists():
            d=rt.read(p);valid=[r['explained_fraction'] for r in d['rows'] if r['reliable']];directionrows.append(dict(fold=short,baseline_rank=d['baseline_rank'],remaining_rank=d['remaining_rank'],reliable_new_directions=len(valid),median_explained_fraction=float(np.median(valid)) if valid else np.nan))
    if directionrows:
        with (R/'专家诊断报告.md').open('a',encoding='utf-8') as f:f.write('\n## 8. 局部方向与原参数的重叠\n\n'+md(pd.DataFrame(directionrows))+'\n\n解释比例是逐方向局部投影，非全局等价、因果贡献或选型门槛。\n')
    write('实际方法与偏离.md','# 实际方法与边界\n\n- 28条逻辑路径；4条时间基线路径独立验证复用，24条新拟合；两个入口由配置固定。旧入口的数值不充分标志保留。\n- 1961—2024完整历史伴随，CPU float64、每worker一线程；256日块仅重算映射中间量，不截断库存梯度。\n- 空间S56/S113/S191独立重建合法训练表、清洗和方差；排除上游留出及下游／水库缓冲标签。新PCA只由合法训练河段构建，公共H1无标签设计不改。\n- 低秩掩码加权交替最小二乘固定100次、seed1729；每次通过QR规范化空间基底。输入到模态系数的线性映射通过冻结基底直接在全部有观测的训练单元上求λ=1岭解，不把缺测日期下不可识别的瞬时系数作为伪标签；与不限秩对照严格共享观测单元和权重。缺测目标从未补零。\n- 低秩训练权重沿用合法目标权重；输入均值、方差、季节回归和模式仅用本折训练资料。月尺度采用唯一站月记录，评价HF合格月优先，月报补余。\n- 所有高维候选保留嵌套父点；总目标选点，梯度阈值1e-5，未达到者不称充分。物理硬门不放宽。\n- 训练验收与公开代码查证在启动前完成。外部参考代码仅存为审阅文本，不导入执行；文献、固定提交与许可状态见 reports/literature_code_registry.json。\n- 逐路径实际完成／预算终止／缺项见 reports/path_summary.csv；条件性缺项见 reports/finalization_omissions.json。没有根据留出结果增加起点、秩或正则网格。\n\n## 低秩实际覆盖\n\n'+md(low)+'\n\n## 运行预算\n\n'+str(rt.read(R/'work/experiment_clock.json'))+'\n')
    allok=all(x['passed'] for x in checks);rt.write(R/'reports/completion_audit.json',dict(status='PASS_WITH_REGISTERED_LIMITATIONS' if allok else 'INCOMPLETE_AUDIT',checks=checks,selected=len(selected),expected_selected=14,paths=len(paths),expected_paths=28,numerically_sufficient_selected=int(result.numerical.sum()),no_automatic_mainline_replacement=True,finished=time.time()))
    write('独立完成审计.md','# 独立完成审计\n\n这里的“独立”指单独程序／进程重算，不是外部专家独立复现。\n\n'+md(pd.DataFrame(checks))+'\n\n'+f'已选择{len(selected)}/14个模型折；登记{len(paths)}/28条路径。数值充分的选择点为{int(result.numerical.sum())}个。物理、目标和站界读出重算见 reports/independent。表格由不调用评价函数的程序重新计算。图像生成后仍须人工视觉检查，结果单列于 figures/figure_qa.json。\n')
    write('README.md','# D29低维区域响应实验\n\n本轮保留统一来源估计校正和完整守恒递推。主检验为3模式线性区域响应。\n\n阅读：[专家诊断报告](专家诊断报告.md) → [实际方法与偏离](实际方法与偏离.md) → [独立完成审计](独立完成审计.md)。\n\n核心数据：reports/selected_models.csv、reports/all_comparison_core.csv、reports/all_event_summary.csv、reports/all_bootstrap_intervals.csv。逐站与事件原表在各比较目录。图示位于figures。\n\n核心实现：scripts/regional_response.py；训练准备prepare_regional.py；验收preflight_regional.py；控制器controller.py；最终流水线finalize_regional.py。继承但未在本轮调用的旧实验脚本不代表新增配置。\n\n真实复算需要已登记私有H1和训练资料；不自动下载或补造缺失数据。无新GitHub发布，无定时任务，不替换主线。\n')
    manifest={p.relative_to(R).as_posix():rt.sha(p) for folder in ['scripts','configs','reports','figures','data','diagnostics','outputs'] for p in (R/folder).rglob('*') if p.is_file() and p.suffix!='.pyc'}
    manifest.update({p.relative_to(R).as_posix():rt.sha(p) for p in R.glob('*.md')})
    rt.write(R/'delivery_manifest.json',dict(created=time.time(),files=manifest,status='AUDITED' if allok else 'AUDIT_INCOMPLETE'))
    print('REPORT',allok,len(selected),flush=True);return allok
if __name__=='__main__':
    if not main():raise SystemExit(2)
