"""Render a source-linked expert readout only after frozen campaign evaluation.

The report describes numerical and heldout evidence separately. It does not
choose parameters, weights or starts and cannot be imported by training workers.
"""
import sys,json,math
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from d29_training.experiment_context import ExperimentContext
import pandas as pd
import numpy as np


def link(path,label=None):
    p=Path(path).resolve();return f'[{label or p.name}]({p.as_posix()})'


def number(x,digits=4):
    if x is None or not math.isfinite(float(x)):return '不可定义'
    return f'{float(x):.{digits}g}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in rows])


def main():
    context=ExperimentContext.load(ROOT)
    out=ROOT/'outputs/evaluation';done=json.loads((out/'evaluation_completed.json').read_text(encoding='utf-8'))
    manifest_path=ROOT/'outputs/campaign_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('experiment_id')!=context.experiment_id or manifest.get('experiment_context_sha256')!=sha(ROOT/'config/experiment_context.json'):
        raise RuntimeError('REPORT_EXPERIMENT_CONTEXT_MISMATCH')
    if done['manifest_sha256']!=sha(manifest_path):raise RuntimeError('EVALUATED_MANIFEST_CHANGED')
    if done['paired_summary_sha256']!=sha(out/'paired_summaries.json'):raise RuntimeError('PAIRED_RESULTS_CHANGED')
    results=json.loads((out/'paired_summaries.json').read_text(encoding='utf-8'))
    jobs={j['id']:j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))}
    core=[r for r in results if r.get('role') in ('time','same_year_spatial','space_time')]
    lines=['# 专家训练策略与全域验证报告','',
        '本报告仅使用全部选定预测封存后的统一评价。月报按用户确认的自动监测有效读数算术平均理解；完整官方读数计数未取得，模拟月报使用等日均值近似。',
        '',f'完整模拟230河段，月度平台覆盖116个已认证站；高频按身份门纳入全部合格站，实际可用TN为15站，不把该子集当成全域结论。选定配置数为{len(manifest["selected_jobs"])}，逐路径排除数为{len(manifest["excluded_jobs"])}。',
        '',f'可追溯入口：{link(manifest_path,"训练选点与32路径去向")}；{link(out/"paired_summaries.json","统一配对汇总")}；{link(ROOT/"reports/文献证据表.md")}；{link(ROOT/"reports/实际方法与偏离.md")}；{link(ROOT/"reports/PR5数值审查对照与年度负值定位.md","年度负值数值审查")}。',
        '', '## 路径编号与比较含义', '',
        '路径格式为`模型_折次_策略_入口`，空间路径再插入支流块，例如`LAND1_F23_T2_s1`和`LAND1_S56_T2_s0`。`s0/s1`表示同一科学配置的两个优化起点，不是站点；只按本路径训练总目标从合法入口中选点。',
        '',table(['字段','含义'],[
            ['U / LAND1','U为原统一来源校正D29/LAND0（31参数）；LAND1为植物—土壤—可用氮完整配置（23个开放坐标，其他生地化参数固定为研究情景）。两者输入/植物活动也不同，直接差值不是纯陆地结构效应。'],
            ['F23','T0用2021—2022训练，T1/T2用2016—2022训练；统一回顾性评价2023。'],
            ['F24','T0用2021—2023训练，T1/T2用2016—2023训练；统一回顾性评价2024。'],
            ['T0','近期月报目标0.8 + HF月内异常0.2。'],
            ['T1','保持T0权重，加入2016—2020月报；不是增加日来源信息。'],
            ['T2','长记录站月报0.4 + 其他站月报0.4 + HF月内异常0.2；54个长记录站仅由训练期资料定义。'],
            ['S56 / S113 / S191','冻结的三个支流空间留出块；路径不使用留出及影响缓冲站的训练水质，分别报告2023同年空间和2024时空迁移。'],
            ['NSE与目标','训练目标是按站/年/月分组的标准化半平方误差加先验；下表原尺度NSE是独立评价指标。不同策略训练目标值不能直接互比。']]),
        '',f'2016—2022真实训练支持和逐站NSE见{link(ROOT/"reports/训练期2016-2022TN效果核查.md","训练期专门核查")}。T0在2016—2020没有训练记录，其同期模拟只能叫历史回算。',
        '', '## 1．数值证据与完整性', '',
        '原坐标投影梯度≤1e-5才称数值充分。目标项的数值不是水质NSE；优化、梯度和质量量的NSE不适用。一次数值充分也不证明全局最优或来源／水文真实。LAND1当前实现的完整历史23坐标门已通过；极小年度负高位由近完全动员时的大通量差额消差造成，舍入许可没有消除此表示缺陷，见上方独立定位。', '']
    audit_rows=[]
    for name in manifest['selected_jobs']:
        r=manifest['training_records'][name]
        folder=context.folder(name);context.check_worker(name)
        audit_path=folder/'independent_audit.json'
        audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
        audit_rows.append([name,number(r['objective']),number(r['projected_gradient']),
            '充分' if r['numerically_sufficient'] else '未充分',
            number(audit.get('source_multiplier')),
            link(folder/'independent_audit.json','独立复算'),
            link(folder/'physical_ledger.json','物理／来源账本')])
    lines.append(table(['配置','训练总目标（仅同配置选入口）','投影梯度','数值状态','历史外源统一倍率','复算','账本'],audit_rows))
    if manifest['excluded_jobs']:
        lines+=['',table(['未纳入路径','明确原因'],list(manifest['excluded_jobs'].items()))]
    lines+=['','## 2．冻结留出NSE：同输入配置内比较训练策略','',
        '每行使用同站、同日期的共同支持。总体中位数之差与逐站配对差中位数是不同统计量；改善比例只在双方NSE均可定义的站中计算。月报、HF月值与HF日值分别列示。','']
    rows=[]
    for r in core:
        if 'NSE' not in r:continue
        n=r['NSE'];c=r.get('month_centered_NSE',{})
        rows.append([r['job'],r['year'],r['role'],r['scope'],n['common_stations'],
            number(n.get('baseline_median')),number(n.get('candidate_median')),number(n.get('median_difference')),
            number(n.get('median_paired_difference')),number(n.get('improved_fraction')),
            number(c.get('median_paired_difference')) if c else '不适用'])
    lines.append(table(['候选（对本配置T0）','年','验证支持','尺度','共同NSE站数','T0 NSE中位数','候选NSE中位数','中位数之差','配对差中位数','改善比例','月内NSE配对差'],rows))
    lines+=['','月内指标在每站每月按有效日读数次数中心化，月内按读数次数加权，各月等权累积误差平方与观测异常平方；零方差不加epsilon。日原尺度指标按共同合格日等权。沿用年度NSE覆盖门：至少30个日值且跨3个月，月尺度至少8个有效月值。覆盖不足的NSE标为不可定义，其他误差量保留；抽样固定原共同支持的站点资格。NSE同时反映偏差、幅度与相关，不能单独称为动态恢复。','',
        '## 3．误差、相关和振幅及抽样不确定性','']
    rows=[];intervals=[]
    for r in core:
        if 'NSE' not in r:continue
        folder=out/r['job'];path=folder/f"{r['scope']}_{r['year']}_heldout_stations.csv"
        if not path.exists():raise RuntimeError('MISSING_STATION_TABLE '+str(path))
        t=pd.read_csv(path)
        for config,g in t.groupby('configuration'):
            rows.append([r['job'],r['year'],r['scope'],config,number(g.NSE.median()),number(g.RMSE.median()),
                number(g.bias.median()),number(g.correlation.median()),number(g.amplitude_ratio.median()),link(path,'逐站')])
        for block in [1,2]:
            for centered in ([False,True] if r['scope']=='daily' else [False]):
                p=folder/f"{r['scope']}_{r['year']}_{'centered' if centered else 'raw'}_block{block}_bootstrap_receipt.json"
                b=json.loads(p.read_text(encoding='utf-8'));q=b['intervals']['median_paired_difference']
                intervals.append([r['job'],r['year'],r['scope'],'月内异常' if centered else '原尺度',block,
                    number(q.get('0.025')),number(q.get('0.5')),number(q.get('0.975')),b['undefined_replicates'],link(p,'抽样回执')])
    lines.append(table(['候选比较','年','尺度','配置','NSE中位数','RMSE中位数','偏差中位数','相关中位数','幅度比中位数','入口'],rows))
    lines+=['','下列区间来自seed1729、1000次同步月块抽样，各站及两个候选共享抽中的月份。区间不包含来源产品、历史初态和水文误差的不确定性。','',
        table(['候选','年','尺度','指标','月块长度','配对差2.5%','配对差中位数','配对差97.5%','未定义抽样数','入口'],intervals)]
    lines+=['','## 4．事件对应与全域支持','']
    event_rows=[]
    for name in manifest['selected_jobs']:
        if jobs[name]['strategy']=='T0':continue
        for year in jobs[name]['evaluate']:
            p=out/name/f'events_{year}.csv'
            if not p.exists():continue
            try:t=pd.read_csv(p)
            except pd.errors.EmptyDataError:continue
            if 'configuration' not in t:continue
            for config,g in t[t.status.eq('common_observed_dates')].groupby('configuration'):
                event_rows.append([name,year,config,len(g),g.station_key.nunique(),number(g.NSE.median()),number(g.RMSE.median()),
                    number(g.bias.median()),number(g.correlation.median()),number(g.amplitude_ratio.median()),
                    number(g.amplitude_error.abs().median()),number(g.peak_error.abs().median()),
                    number(g.background_error.abs().median()),number(g.peak_day_offset.abs().median()),link(p,'逐事件')])
    lines.append(table(['比较','年','配置','可比事件数','共同站数','事件NSE中位数','事件RMSE中位数','偏差','相关','幅度比','振幅绝对误差','峰值绝对误差','背景绝对误差','峰日绝对偏移（日）','入口'],event_rows))
    lines+=['','原文件2024年有83个候选事件，原日期覆盖门筛后为57个；2023年为40个合格事件。本轮冻结这些身份，当前共同支持若不足则单列，不按预测误差取舍。至少4个背景日、1个事件日；背景采用中位数，事件只有1个日值时保留峰值／背景误差而NSE不可定义。事件背景不能跨入评价年前的训练时期；峰日在共同实测日期上确定，缺测并不提供未观测峰日的真值。空间检验只使用冻结留出区，缓冲站不训练也不混入留出站成绩；全域地理分组仍只是描述。',
        '',f'2016—2024逐站逐年表见{link(out/"absolute_results","全时期逐站结果目录")}；T0的2016—2020回算明确标为未参与训练，不能计作早期训练改善。',
        '', '## 5．训练信息、参数补偿与固定情景边界','',
        'T0/T1/T2使用相同物理标准化参考，区别在合法观测时期与预登记权重。T2将长记录和其他站各赋0.4，HF异常为0.2；这不是文献推导的最优权重。训练目标总值跨策略不直接比较，须结合共同支持的重建和留出成绩。', '']
    block_rows=[]
    for name in manifest['selected_jobs']:
        p=context.folder(name)/'training_block_gradients.json'
        if not p.exists():
            block_rows.append([name,'未完成','不可用','不可用','缺项']);continue
        b=json.loads(p.read_text(encoding='utf-8'))
        for pair in b['cosines']:
            block_rows.append([name,pair['first']+' / '+pair['second'],number(pair['raw_coordinate_cosine']),
                number(pair['fixed_native_scaled_coordinate_cosine']),link(p,'梯度明细')])
    lines.append(table(['配置','训练块／先验','原坐标余弦','固定尺度余弦','证据'],block_rows))
    lines+=['','负余弦表示该点附近的下降方向冲突，不单独证明观测错误或某个过程错误。来源倍率、库存和植物活动缺额须与独立账本同时看；降低加权训练损失可能只是重新分配数据块误差。',
        '', 'U与LAND1使用不同完整来源／植物活动配置，只能比较配置表现。LAND1的SON初态、矿化、植物转换和有效损失仍是固定研究情景；年度农业量及植物活动均匀分日，沉降月量均匀分日，未恢复施肥日期或湿沉降雨日信息。早期NPP使用参考情景。潜在植物活动存在未满足量，不宣称重现了真实收获。',
        '', '## 6．文献证据与本轮可以得出的结论','',
        '文献登记有57项去重候选、15项定向核查，深度为3项全文方法／讨论与12项摘要；不能称全部15项都是独立预测成功案例。长期记录有机会约束慢状态，多站资料有机会分辨空间映射，高频异常增加月均之外的信息，但这些都要求相容的观测算子、驱动与过程表达。Taylor等的日硝酸盐负荷研究完整验证NSE为0.46，不能将删去困难事件后的结果当成本项目删除事件的依据。', '']
    for model in ['U','LAND1']:
        r=[x for x in core if x.get('role')=='time' and x.get('scope')=='monthly_report' and jobs[x['job']]['model']==model and jobs[x['job']]['strategy']=='T2' and 'NSE' in x]
        if len(r)<2:
            lines.append(f'- {model}：T2对T0尚无两折完整月报留出证据，不能签发两折一致收益结论。')
        else:
            positive=all((x['NSE'].get('median_paired_difference') or 0)>0 for x in r)
            lines.append(f'- {model}：T2对T0的两折月报逐站配对NSE中位差'+('均为正。仍需同时检查日动态、空间迁移、区间及事件代价，不能据此单独宣布策略成功。' if positive else '未共同为正。现有结果不支持“两折月报均获益”的结论；日动态及空间结果按上表分别判读。'))
    insufficient=[name for name in manifest['selected_jobs'] if not manifest['training_records'][name]['numerically_sufficient']]
    lines+=['',('数值未充分的选定配置：'+', '.join(insufficient)+'。这些结果先报告求解限制，不能当成训练策略或结构失败。') if insufficient else '所有选定配置均满足登记的原坐标投影梯度门槛；这不消除固定输入情景、参数不可识别或局部最优风险。',
        '', '本轮不认证H1水文准确性，不识别真实排放，不将回顾性评价称为首次盲测。若训练改善但两折或支流迁移不保持，优先解释为信息约束尚未转化为可迁移规则；若多个数据块持续冲突，还需独立源日历、站界水量和过程状态证据定位原因，不能仅继续增加权重或自由参数。',
        '', '依照本轮研究决定，氨氮与DO均不进入训练目标、预测变量或常规残差分层。四小时尺度变幅仍只作为日模型未解析尺度的限制。', '']
    report=ROOT/'reports/专家训练策略与全域验证报告.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    write_json(ROOT/'outputs/report_receipt.json',dict(report_sha256=sha(report),manifest_sha256=sha(manifest_path),
        evaluation_complete_sha256=sha(out/'evaluation_completed.json'),generated=datetime.now().astimezone().isoformat(),
        scientific_review_still_required=True,not_a_model_selection_action=True))
    print(report)


if __name__=='__main__':main()
