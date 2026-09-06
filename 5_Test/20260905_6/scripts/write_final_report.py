"""Assemble a source-backed scientific draft after the numerical requirements pass."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now
import pandas as pd


def main():
    run=ROOT/'5_Test/20260905_6';sources=[]
    def read(relative):
        path=ROOT/'5_Test'/relative;sources.append(path)
        return json.loads(path.read_text(encoding='utf-8'))
    audit=read('20260905_6/reports/completion_audit.json')
    if not all(c['passed'] for c in audit['checks']):raise RuntimeError('Do not write a completed-stage report from incomplete evidence')
    dev=read('20260905_3/reports/development_summary.json')
    structural=read('20260905_4/reports/structural_summary.json')
    comparison=read('20260905_4/reports/structure_vs_old_control.json')
    choice=read('20260905_6/reports/development_choice.json')
    spatial=read('20260905_5/reports/nested_validation_summary.json')
    outer_audit=read('20260905_5/reports/nested_selection_audit.json')
    temporal=read('20260905_6/reports/temporal_confirmation.json')
    temporal_audit=read('20260905_6/reports/temporal_evidence_audit.json')
    prior=read('20260905_5/reports/prior_sensitivity.json')
    ident=read('20260905_5/reports/identifiability.json')
    f24=read('20260905_6/reports/f24_export_audit.json');f25=read('20260905_6/reports/f25_export_audit.json')
    temp=read('20260905_4/reports/temperature_mechanism_decision.json')
    selected=choice['chosen']['candidate'];key=selected['candidate_id']
    selected_summary=choice['chosen']['summary']
    control=dev['summaries']['CONTROL_H7__STUDENT_T4_LOG1P']
    gates=comparison['comparisons'][key]['development_gates'] if key in comparison['comparisons'] else dev['comparisons'][selected['model']+'__'+selected['loss']]['temporal_gates']
    spatial_pass=all(r['passes_nonregression'] for r in spatial['comparisons'].values())
    statistical_pass=all(gates.values()) and spatial_pass
    result='未通过本轮升级门槛，保持实验状态。' if not statistical_pass else '通过开发与空间统计门槛；仍需结合确认期、可识别性和科学复核判断。'
    def fmt(value):return '未定义' if value is None else f'{value:.4f}'
    def row(label,s):
        return f"| {label} | {fmt(s['median_nse'])} | {fmt(s['q25_nse'])} | {s['fraction_nse_positive']:.1%} | {fmt(s['median_time_r'])} | {fmt(s['mean_station_log_rmse'])} |"
    header=['| 模型/评价 | 单站NSE中位数 | NSE q25 | NSE>0比例 | 站内r中位数 | 平均站点log-RMSE |',
            '|---|---:|---:|---:|---:|---:|']
    text=['# 20260905 TN实验汇总（待最终科学复核）','',result,'',
        f"开发冻结候选为 `{key}`：{selected['model']}，{selected['loss']}，月历{selected['calendar']}，dynamic={selected['dynamic']}。",
        f"开发期单站NSE中位数为{selected_summary['median_nse']:.4f}，同折重拟合旧对照为{control['median_nse']:.4f}。",
        '水文正式产品未被替换，所有最终输出仍标记EXPERIMENTAL_NOT_PROMOTED。','',
        '## 1. 实际完成范围','',
        '140次开发起点全部达到投影梯度门槛；四项结构单改动分别完成4个前向年份×5起点。',
        f"空间验证完成{len(spatial['selections'])}个外层reach/终端树试验，每个外层使用独立内层选模。",
        '2024回顾性确认、2025敏感性、先验尺度敏感性、F24/F25各五起点和参数剖面均已执行。',
        f"当前数值/产物验收为{audit['completed_count']}/{audit['requirement_count']}项通过；本文件尚待逐项科学复核。",'',
        '## 2. 开发与结构比较','',*header,row('重拟合旧H7对照',control)]
    for name,s in dev['summaries'].items():
        if not name.startswith('CONTROL'):text.append(row(name,s))
    for name,item in structural['outcomes'].items():text.append(row(name,item['summary']))
    text+=['','训练起点只按训练目标选取；表格合并2020—2023 OOF相同观测队列。pooled NSE没有参与选模。',
        '均匀日输入和EARLY/LATE改变的是输入时间分配，年总质量已独立核对。只有单项独立通过时才允许组合。','',
        '本候选开发门槛：','']
    text += [f"- {name}：{'通过' if passed else '未通过'}" for name,passed in gates.items()]
    if key in comparison['comparisons']:
        paired=comparison['comparisons'][key]['paired_vs_old_control']
        text += ['',f"相对旧对照的中位NSE增量：{paired['delta_median_nse']:.4f}；按站95%区间{paired['station_ci95']}；按终端树区间{paired['tree_cluster_ci95']}。",
            '两种重采样支持的范围不同；终端树数量较少，不能把按站区间替代跨水系一致性的证据。']
    text+=['','## 3. 零TN历史空间迁移与时间确认','',*header]
    for kind,result in spatial['comparisons'].items():
        text += [row(kind+'嵌套选模过程',result['new']),row(kind+'旧对照',result['control'])]
    for year,result in temporal.items():text += [row(year+'固定候选',result['new']),row(year+'旧对照',result['control'])]
    text += ['','逐外层诊断（描述区域异质性，不用于重新选择候选或替代预注册汇总门槛）：','',
        '| 外层 | 选中候选 | 站点数 | 新模型NSE中位数 | 对照NSE中位数 | 中位数差 | q25差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for item in outer_audit['outer_diagnostics']:
        text.append(f"| {item['outer_id']} | {item['chosen']} | {item['new']['stations']} | {fmt(item['new']['median_nse'])} | {fmt(item['control']['median_nse'])} | {fmt(item['median_nse_delta'])} | {fmt(item['q25_nse_delta'])} |")
    text += ['','逐外层的时间变化诊断（均为站点指标的中位数）：','',
        '| 外层 | 新模型站内r | 对照站内r | 新模型SD比 | 对照SD比 | 新模型绝对偏差mg/L | 对照绝对偏差mg/L |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in outer_audit['outer_diagnostics']:
        n,c=item['error_dynamics']['new'],item['error_dynamics']['control']
        text.append(f"| {item['outer_id']} | {fmt(n['median_time_r'])} | {fmt(c['median_time_r'])} | {fmt(n['median_sd_ratio'])} | {fmt(c['median_sd_ratio'])} | {fmt(n['median_absolute_bias_mg_l'])} | {fmt(c['median_absolute_bias_mg_l'])} |")
    text += ['','SD比为预测/观测标准差。减少错误波动也可能提高NSE，须联合相关性、幅度与均值偏差解释；',
        '误差分解核对MSE恒等式，不生成任何均值校正后的模型输出。']
    text += ['','空间表评价的是嵌套选择流程；不同外层可能选择不同候选。每个外层的全部TN数据从其内层预处理和选择中排除。',
        '这里的无历史指TN站点历史；水文此前已经校准并冻结。2024曾被历史实验使用，不能称全新盲测；2025为敏感性而非正式主线验证。','',
        '预注册排除域的独立诊断如下。这些观测未进入训练、选模或主升级门槛；位置/过程适用性仍未解决。','',
        '| 年份 | 排除域 | 观测数 | 新模型NSE中位数 | 对照NSE中位数 | 新模型log-RMSE | 对照log-RMSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for year,result in temporal.items():
        for domain,item in result['excluded_domains'].items():
            n,c=item['new'] or {},item['control'] or {}
            text.append(f"| {year} | {domain} | {item['rows']} | {fmt(n.get('median_nse'))} | {fmt(c.get('median_nse'))} | {fmt(n.get('mean_station_log_rmse'))} | {fmt(c.get('mean_station_log_rmse'))} |")
    text += ['','观测数为0的分组没有可验证数据，指标未定义；不能解释为该域已验证通过。','',
        '年度确认与先验敏感性另经独立审计：从所选起点预测重算单站NSE中位数、q25、log-RMSE和相关性，核对完整观测、排除域及训练年份。',
        '## 4. 参数与legacy','',
        '新主预测保留矿质氮lifetime与慢水氮库存。共享环境映射生成230个reach的参数，拟合器通过守恒过程方程优化参数。',
        '没有新增Active/Fresh库，没有站点残差、浓度C-Q输出头、自由reach截距或事后历史趋势曲线。',
        '独立推断移除TN观测、站点ID与TN方差统计后仍能还原过程字段；该验证单独保存。',
        'Bayesian部分是带先验的generalized Bayes/MAP；没有全后验样本，也没有将曲率或profile冒充可信区间。','',
        '| 先验系数尺度 | 测试年 | 单站NSE中位数 | 平均站点log-RMSE |','|---:|---:|---:|---:|']
    for item in prior['results']:text.append(f"| {item['prior_scale']:g} | {item['year']} | {fmt(item['summary']['median_nse'])} | {fmt(item['summary']['mean_station_log_rmse'])} |")
    bounds=[r['parameter'] for r in ident['boundary'] if r['at_lower'] or r['at_upper']]
    pressure=[r['parameter'] for r in ident['boundary'] if r['outward_pressure']]
    text += ['',f"F24活动边界参数：{', '.join(bounds) or '无'}；仍有向边界外降低目标压力：{', '.join(pressure) or '无'}。",
        f"自由边界面局部曲率正定：{ident['curvature']['positive_definite']}；条件数：{ident['curvature']['condition_number']}。",
        '11项剖面跟踪局部 nuisance 最优，并非全局后验积分；有限数据不能唯一识别每一项生物地球化学过程。','',
        '## 5. 最终实验产品与质量账本','',
        '| 产品 | 拟合年份 | 训练行数 | reach月行数 | 全系统质量相对误差 |','|---|---|---:|---:|---:|']
    for label,a in [('F24',f24),('F25',f25)]:text.append(f"| {label} | {a['years'][0]}—{a['years'][1]} | {a['train_rows']} | {a['monthly_rows']} | {a['mass']['full_system_relative']:.3e} |")
    text += ['','源输入=作物摄取+其他损失+最终土壤/慢水氮库存+河道移除+水库库存+终端输出，已做全历史回放。',
        '输出包括全部230个reach的月浓度/负荷/水量/氮库存、站点预测、过程参数、水库账本和质量标记。',
        '干月浓度未定义，不使用正流量下限掩盖无水状态。单控制水库与多控制臂的坝前/坝后边界分别标注。','']
    for a in [f24,f25]:
        for item in a['exports']:
            path=Path(item['path']);text.append(f'- [{path.name}]({path.as_posix()})')
    text += ['','## 6. 必须保留的解释限制','',
        '- 现有月观测没有可靠采样日和完整原始记录链接；与模型月流量加权浓度的统计口径一致性尚未证实。',
        '- reach内观测算子假定局地水与负荷按河段位置分配。白盆珠位置未定及开放湖泊域仍作为单独敏感性。',
        '- 2025保留PET/source延伸、未完整12月TN、气温来源切换等标记。',
        '- 温度探针有部分条件信号，但未通过预先定义的跨年重复性门槛。本轮没有据此增加温度参数。',
        '- WWTP旧账本构建通过质量检查，后续预测试验非劣性失败；本轮未重新加入这套估算账本。',
        '- 拟合度不足不是已证明的理论结构上限。结果只支持或排除本轮实际测试的方程、输入和约束。',
        '', '## 7. 技术依据与可追溯记录','',
        '参数区域化、可微过程拟合、多站损失、氮遗留时间尺度及温度机制的核查程度见下列登记。只读元数据/摘要的文献没有被写成已读全文。','']
    refs=[ROOT/'5_Test/20260905_1/reports/literature_and_history.md',ROOT/'5_Test/20260905_1/reports/point_source_history_recheck.json',
        ROOT/'5_Test/20260905_4/process_equations.md',run/'reports/completion_audit.json',ROOT/'5_Test/20260905_5/reports/identifiability.json']
    text += [f'- [{p.name}]({p.as_posix()})' for p in refs]
    text += ['',f'汇总生成时间（UTC）：{utc_now()}。环境：conda sparrow。','']
    path=run/'final_report_draft.md';path.write_text('\n'.join(text),encoding='utf-8')
    atomic_json(dict(status='DRAFT_REQUIRES_SCIENTIFIC_REVIEW',created_utc=utc_now(),runtime=RUNTIME,
        report_path=str(path),report_sha256=sha256(path),statistical_development_and_spatial_gates_passed=statistical_pass,
        sources=[dict(path=str(p),sha256=sha256(p)) for p in sources]),run/'reports/final_report_manifest.json')
    print('FINAL_REPORT_DRAFT_WRITTEN',str(path),flush=True)


if __name__=='__main__':main()
