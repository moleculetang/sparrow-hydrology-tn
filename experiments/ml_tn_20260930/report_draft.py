"""Evidence-filled report draft. Final expert signoff must inspect the generated numbers."""
import json,datetime,numpy as np,pandas as pd
from mltn.common import ROOT,read,write

def fmt(value):
    return f'{float(value):.4g}' if pd.notna(value) and np.isfinite(float(value)) else '不可定义'
def markdown(frame,cols):
    if frame.empty:return '当前没有已冻结的合格结果。'
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |',*['| '+' | '.join(str(row[c]).replace('|','／') for c in cols)+' |' for _,row in frame.iterrows()]])
def main():
    signed=ROOT/'outputs/expert_interpretation_receipt.json'
    if signed.exists() and read(signed).get('interpretation_authored'):
        raise RuntimeError('FINAL_INTERPRETATION_EXISTS: do not overwrite the interpreted report with an automatic draft')
    e=ROOT/'outputs/evaluation';selection=read(ROOT/'outputs/frozen_selection.json');roles=read(ROOT/'outputs/result_roles.json');literature=read(ROOT/'literature/detailed_review_receipt.json');coverage=read(ROOT/'outputs/coverage/identity.json')
    pairs=read(e/'paired_summary.json');rows=[]
    for p in pairs:
        if 'seedmean' not in p['candidate'] and 'reference_' not in p['candidate'] and 'convex_ensemble' not in p['candidate']:continue
        rows.append({'支持':p['context']+'／'+p['task'],'基准':p['baseline'],'候选':p['candidate'],'共同站数':p['common_stations'],'基准NSE中位数':fmt(p.get('baseline_median',np.nan)),'候选NSE中位数':fmt(p.get('candidate_median',np.nan)),'总体中位数差':fmt(p['median_difference']),'逐站差中位数':fmt(p['median_paired_difference']),'改善比例':fmt(p['improved_fraction']),'月内NSE差中位数':fmt(p.get('centered',{}).get('median_paired_difference',np.nan))})
    pair_table=pd.DataFrame(rows);core=pair_table[pair_table['支持'].str.startswith(('F23','F24'))] if len(pair_table) else pair_table
    metrics=pd.read_csv(e/'all_station_metrics.csv');stats=[]
    for (cfg,context,task),g in metrics.groupby(['configuration','context','task']):
        if 'seedmean' not in cfg and 'reference_' not in cfg and 'convex_ensemble' not in cfg:continue
        g=g[g.nse_eligible]
        stats.append({'配置':cfg,'支持':context+'／'+task,'合格站数':len(g),**{key:fmt(g[key].median()) for key in ['NSE','RMSE','bias','correlation','amplitude_ratio']}})
    stat_table=pd.DataFrame(stats)
    incomplete=len(roles['missing']);report=ROOT/'reports/专家机器学习与全域验证报告.md';report.parent.mkdir(exist_ok=True)
    text=[
        '# 专家机器学习与全域验证报告（待最终独立解读签署）',
        f'生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。本文件由冻结数据生成，结果解读尚需逐项复核；不能把草稿生成当作科学验收完成。',
        f'已登记正式路径尚缺{incomplete}条。所有尝试、替代、失败与缺项见[结果角色登记](../outputs/result_roles.json)。',
        '## 任务身份与结果边界',
        '本轮直接预测TN，主线没有D29/LAND1预测、库存或反推源，也没有可训练站号；NH₄、DO排除。H1从1961演化，机器学习用2015至2024驱动和2016至2024月报；2015只承担窗口暖机。空间支持230河段，认证月度站116、高频站15；230出口外推不等于230站验证。',
        '月报采用用户确认的自动监测有效读数算术均值。缺完整官方读数支持时，联合模型月值使用等日近似。高频日标签不复制月报；同站月的水平仅由官方月报提供，HF只增加去月均异常。',
        '预测期H1、年度产品和当月沉降作为已知回顾性驱动。F23训练截至2022、评价2023；F24训练截至2023、评价2024。2023/2024已在旧实验多次查看，本轮属于冻结设计的回顾性验证。',
        '历史TN辅助另行读取起点前已获得的TN：日均至少等到日终，并采用所有选定有效读数的实际最新捕获时刻；迟到读数不能提前可见。按观测支持时间选择最新TN，不能让迟到的旧记录覆盖较新的观测。月报次月初可见只是条件性假设，逐条发布时间仍未知，不能认证实际业务预报。严格排除起点当刻读数，所以TN可见记录比预测提前量还旧。滚动辅助与年度冻结闭环分别报告，不把反馈收益说成外部源预测能力。',
        '## 文献与技术路线',
        f"去重筛查{literature['deduplicated_screen']}条，详细核查{literature['interpreted_studies']}个研究和{literature['public_implementations']}个公开实现；其中全文{literature['full_text_interpreted']}篇、其余明确标为摘要证据。不是25篇全文综述。",
        '见[文献证据表](../literature/detailed_evidence_table.csv)。Pandit的时间／空间差异、Fang的大样本反例和依赖同期其他水质的高分案例都表明：复杂模型不保证迁移，水质反馈必须与独立驱动分开。',
        'RF、ExtraTrees、XGBoost、LightGBM、CatBoost、MLP、LSTM、GRU、因果TCN、因果Transformer及受限一跳Graph-TCN分别作官方月值和HF日值预测。每任务8个登记配置；2022年1—9月选配置，10—12月只估凸集成。',
        '内部验证冻结的优胜路线：'+json.dumps(selection['winners'],ensure_ascii=False)+'；联合路线：'+json.dumps(selection['joint_routes'],ensure_ascii=False)+'。评价年从未用来选择此处路线。',
        '联合目标的数据平方项系数为0.4月水平和0.1 HF月内异常，来源于0.8／0.2各乘半平方误差；正则由各算法配置单列。树的聚合梯度精确，曲率采用观测算子行L1构造的Gauss–Newton对角上界，避免月共同叶的曲率被低估约30倍；不称完整非线性Newton。网络小批次保留完整站月。',
        '## 时间主结果',
        markdown(core,list(core.columns)) if len(core) else '暂无完整主结果。',
        '逐站结果及共支持RMSE、偏差、相关和幅度见[完整效果表](../outputs/evaluation/all_station_metrics.csv)。以下补充指标均以该配置合格NSE站为支持，不能拿不同站群的中位数差冒充配对增益。',
        markdown(stat_table,list(stat_table.columns)) if len(stat_table) else '暂无补充指标。',
        '## 空间与时空迁移',
        'S56/S113/S191是冻结支流块。所有年份留出及下游缓冲水质从训练移除。S23用允许区截至2023资料，报告2023同年空间；S24复用相同合法检查点，只读出2024时空联合。不是从全域最优参数初始化，也不是全新河网迁移。',
        '为防止全域调参与epoch选择间接看过留出站，空间固定预登记配置0、网络120epoch。另有同配置F24全域参照；完整配置差异和区域排除必须分开讨论。技术家族仍由全域内部验证选定，空间结论以该家族为条件。',
        markdown(pair_table[pair_table['支持'].str.startswith(('S23','S24'))],list(pair_table.columns)) if len(pair_table) else '暂无空间结果。',
        '## 幅度、相位与输入信息的解读',
        '误差分解见[逐站偏差／幅度／去相关表](../outputs/evaluation/error_decomposition_station.csv)：MSE=偏差²+(预测标准差−观测标准差)²+2(两标准差乘积−协方差)。最后一项含日期错配及其他波形差异，不能唯一归因为输运时滞。月内分解按读数次数中心化、月份等权。',
        '事件表固定观测支持，按共同日期比较峰值、背景、振幅和峰日。仅幅度比接近1不代表波形恢复；须同时检查NSE、相关与峰日。四小时日内变幅另见[未解析尺度](../outputs/evaluation/unresolved_four_hour_variation.parquet)，它不是日模型可以逐读数拟合的状态。',
        '源、年活动和水文是研究输入，不是独立认证的日排放及同站界实测流量。纯ML仍共享这些限制。特征重要性和误差分层只揭示统计关联，不能认证物理因果。直接月与联合日读出目标、分辨率及算法正则有所不同，二者差值是完整配置差异，不能唯一归因于增加HF资料。',
        '## 工程修正与数值状态',
        '修正了TF32造成的同权重差异、365日配置的TCN感受野不足、全域图输入临时大数组、2015暖机不必要缺项、上游NaN×零传播、空间调参间接污染风险及只恢复权重的检查点缺陷。旧数组和失败回执保留；受影响旧月序列在安全边界同网格复算。',
        '序列缓存与原变换逐元素一致，改善重复预处理，不扩模型。恢复包含优化器、随机状态、epoch和输入身份；错身份拒绝。CUDA训练FP32，评价及目标复核FP64，禁用TF32与混合精度。patience/迭代停止不认证数值充分，灰箱投影梯度门不适用于神经网络。',
        '本机小批量与工作站批量实测分别登记，单作业远端不一定更快。完整步骤及并发验收后，远端上限4×4 CPU和3个GPU作业，本机上限1个GPU作业；GPU宿主每任务1线程，互斥真实物理核租约并允许其他实验使用余量。依赖等待与资源暂停时实际并发可低于上限。输入预处理局部加速不是完整训练加速。',
        '## 交付入口与尚待签署',
        '[实际方法与偏离](实际方法与偏离.md)；[训练重建指标](../outputs/evaluation/training_station_metrics.csv)；[全站覆盖](../outputs/coverage/station_year_support.csv)；[配对汇总](../outputs/evaluation/paired_summary.json)；[独立读数复核](../evidence/independent_HF_reading_mean.json)；[预测独立复核](../evidence/independent_predictions_review.json)。',
        '同步整月与连续两月块各1000次，seed1729。零方差及覆盖不足不强算NSE；逐站配对差中位数与总体中位数之差分别给出。缺项和失败必须参与最终判读，不用训练下降或单篇文献高分替代验证。',
        '最终签署待办：核对全部正式/替代身份，审核时间与空间量化收益、事件及幅度/相位一致性，检查预算缺项，填写具体的下一步建议。'
    ]
    report.write_text('\n\n'.join(text)+'\n',encoding='utf-8');write(ROOT/'outputs/report_draft_receipt.json',dict(path=str(report),final_scientific_signoff=False))
if __name__=='__main__':main()
