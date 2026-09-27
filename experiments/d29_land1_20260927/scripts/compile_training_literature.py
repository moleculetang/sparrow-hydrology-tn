"""Evidence-level case register after targeted abstract/full-text reading."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
OUT=ROOT/'evidence/literature'
reg=json.loads((OUT/'deduplicated_candidates.json').read_text(encoding='utf-8'))
previous=json.loads((ROOT.parent/'20260926_1/retrospective_v1/literature/literature_cases.json').read_text(encoding='utf-8'))
lookup={w['doi']:w for w in reg['candidates']}
inherited={w['doi']:w for w in previous}
notes=[
('10.1029/2022wr032329','河段氮过程观测','15分钟两站NO3和DO；五个高阶河段','上下游质量差需要侧向输入和同边界流量支持；DO可提供代谢约束，但需独立观测算子','同期质量平衡与过程解释；摘要未提供独立跨年预测设计'),
('10.2166/nh.2013.087','水文—硝酸盐软约束','高频Q和NO3、稳定同位素支持近保守输移','浓度—流量模式可约束流路；仅在生地化转化较弱时成立，不能把TN事件都解释为流路','WaSiM多目标校准；五类流路为软数据；不属于本项目TN结构验证'),
('10.2134/jeq2015.10.0531','长期观测关联，非过程校准','1976—2014 Illinois河流NO3；年度浓度、负荷和多年农业N剩余','长记录能辨认年际趋势与累积输入关联，但不能据相关唯一识别慢库初态','年度回归/趋势分析；未验证逐日动态或空间迁移'),
('10.1016/j.jenvman.2016.05.002','流域水质过程模型','SWAT；Wensum/Blackwater高频资料转日流量、NO3、TP','日尺度多变量校准可行；多污染物存在权衡；其情景管理收益不能外推为D29收益','摘要确认日校准与验证；精确划分年份和观测算子尚未全文核实'),
('10.1029/2010wr009525','纯水文方法证据','四个概念模型，30分钟至3日分辨率','更细数据的参数收益受数值积分和误差模型影响；不得把同日六读数当六个独立过程状态','慢过程参数与快过程信息尺度不同；不是氮模型性能证据'),
('10.1016/j.watres.2023.120347','高频数据与输移融合','Bode河段；15分钟，2015—2017常水与2018—2020干旱','保守传播与实测反应信号之差用于反推滞留；有助解释季节和日内过程','明确为同期反演；不得混入本轮跨年预测成绩'),
('10.1002/wat2.1155','水文—水质输移综述','流域水压传播、质量输移和行程时间','增加TN记录不能自动认证H1水量或溶质输移；改善拟合后仍须过程支持','概念/方法综述，不计作一个新的校准成功案例'),
('10.13031/2013.25407','纯水文多站方法证据','SWAT三流量站；单目标GA与多目标SPEA2','多站约束与求解方法共同改变最优参数；训练优势不保证验证优势','摘要分别讨论校准和验证；无TN结果，不外推氮收益'),
('10.2166/wst.2005.0062','城市雨水过程模型','地表积累、侵蚀、管网输移；四种管内沉积初态','相同训练信息下初态不同可导致不同响应及低预测能力；本轮须保留初态身份','Bayesian/MCMC校准与不确定性；与流域D29尺度不同'),
('10.5194/hess-6-395-2002','INCA长期过程情景','1995—1999均值和季节NO3校准；1993—1999气候重复构造百年情景','全文显示土壤池固定、矿化/固定/固氮设零；地下水状态强烈决定长期迟滞，不能把长情景当独立长序列验证','全文定向核查pp396—400的参数化与讨论；百年驱动为重复情景，非百年实测留出'),
('10.1016/j.watres.2019.02.059','mHM-Nitrate河网过程模型','Selke 2011—2015两河段15分钟NO3和GPP派生摄取','辐射与遮蔽约束日摄取并支持区域化；独立过程信息与更多同类浓度样本不同','摘要核对两河段验证；旧代码证据保留来源，本轮未重新审查全部源码'),
('10.1029/2021wr031587','ELEMeNT长期遗留N模型','Weser 1960—2015；N负荷、浓度、土壤N逐步加入','额外状态资料可减少输入、反硝化、行程时间和保护系数的等效性；长期TN仍非初态认证','摘要：9参数渐进约束；N剩余输入不得套用毛外源后再扣作物需求'),
('10.1002/hyp.15154','SAS行程时间水质模型','日NO3、月稳定同位素，三组Monte Carlo信息对照','月同位素与日NO3联合减少输移—反应参数交互；支持独立信息而非单纯提高采样次数','摘要明确仅NO3、仅同位素、联合三实验；不能据此直接新增本项目NH4/DO损失项'),
('10.5194/hessd-12-1279-2015','采样方法证据，讨论稿','四河流小时P、DO、pH和水温的降采样','日内周期会使抽样时刻影响统计；官方自动月均不能与零星月采样等同','本轮获得讨论稿摘要；尚未核对最终版本，不作为水质模型校准成功案例')]
cases=[]
for doi,role,support,insight,validation in notes:
    w=lookup.get(doi)
    inherited_evidence=None
    if doi in inherited:
        old=inherited[doi]
        verified={}
        for p,h in old.get('evidence_sha256',{}).items():
            path=Path(p)
            if not path.exists() or sha(path)!=h:raise RuntimeError('INHERITED_EVIDENCE_CHANGED '+p)
            verified[p]=h
        inherited_evidence={'case':old['case_id'],'verified_paths':verified}
        if w is None:
            w={'doi':doi,'title':old['title'],'abstracts':{'inherited_verified':old['abstract_reviewed']},'sources':['verified_local_prior_audit']}
            reg['candidates'].append(w);lookup[doi]=w
        elif not w['abstracts'] and old.get('abstract_reviewed'):w['abstracts']={'inherited_verified':old['abstract_reviewed']}
    assert w is not None and w['abstracts']
    cases.append({'doi':doi,'title':w['title'],'role':role,'target_and_support':support,'transferable_information':insight,'validation_and_limit':validation,'abstract_reviewed':True,'fulltext_targeted_review':doi=='10.5194/hess-6-395-2002','fulltext_scope':'parameterisation and limitations pp396–400' if doi=='10.5194/hess-6-395-2002' else 'not reviewed this round','code_reviewed_this_round':False,'inherited_evidence':inherited_evidence,'NSE':'not re-reported: no independently checked prediction vectors/denominator; do not infer TN performance from method evidence'})
overrides=OUT/'fulltext_review_overrides.json'
if overrides.exists():
    for override in json.loads(overrides.read_text(encoding='utf-8')):
        if not (ROOT/override['fulltext_review_file']).exists():raise RuntimeError('FULLTEXT_REVIEW_NOTE_MISSING')
        cases=[c for c in cases if c['doi']!=override['doi']]+[override]
assert len(reg['candidates'])<=60
reg['candidate_count']=len(reg['candidates']);write_json(OUT/'deduplicated_candidates.json',reg)
write_json(OUT/'reviewed_cases.json',{'screened':len(reg['candidates']),'detailed_related_evidence_records':len(cases),'not_all_records_are_greybox_calibration_cases':True,'cases':cases})
nfull=sum(c['fulltext_targeted_review'] for c in cases)
md=['# 训练策略文献证据表','',f'去重后{len(reg["candidates"])}条候选，定向核查{len(cases)}项相关证据。这些不全是物理灰箱成功案例：纯水文、采样、观测关联和综述分别标记。{len(cases)-nfull}项为摘要层面，{nfull}项完成全文方法或讨论定向核查。Rode2007、Piniewski2019等题录入口未获得充分方法证据，仍作为待全文核查，不能据题名补写结论。','', '| DOI | 类型与资料 | 对本轮的含义与限制 | 核查层级 |','|---|---|---|---|']
for c in cases:md.append(f'| [{c["doi"]}](https://doi.org/{c["doi"]}) | {c["role"]}；{c["target_and_support"]} | {c["transferable_information"]}；{c["validation_and_limit"]} | '+('全文定向＋摘要' if c['fulltext_targeted_review'] else '摘要；未核全文')+' |')
md+=['','这些证据支持检验信息互补、参数补偿和跨尺度观测算子，未提供本项目0.8/0.2权重的最优性证据。权重仍为预登记研究设定。没有资料支持把NH4或DO直接当作TN同期驱动，或把水文模型的NSE改善当作本项目TN收益。']
(ROOT/'reports/文献证据表.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print('reviewed',len(cases),'screened',len(reg['candidates']))
