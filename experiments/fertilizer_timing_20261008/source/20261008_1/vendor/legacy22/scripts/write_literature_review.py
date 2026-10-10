"""Human-reviewed case register; unknown evidence stays unknown."""
import json,csv,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'evidence/literature'
CASES=[
('10.1016/0022-1694(95)02951-6','输出系数/活动量','TN/TP到地表水负荷','活动量与到水体输出系数；不是土壤库存输入系数','元数据；尺度、独立验证和失败细节未核实','不能把文中输出系数直接塞入D29再重复计算到河损失'),
('10.1029/97WR02171','统计—过程耦合','TN/TP输送量','源强与陆地输送、河道衰减、库滞留结合；输出年度产率','摘要：414美国站；空间关系拟合，未从摘要认证独立时空留出','年度空间解释支持活动量思路，不提供施肥日历'),
('10.5194/hess-6-559-2002','过程氮模型','日NO3/NH4浓度','日氮质量、施肥多次、植被生长期；降水控制沉降；土壤保留体积','全文定向核查：Kennet 1998；不是独立跨年TN验证','日输入、作物期和库存共同作用；1994沉降估计用于1998，本案例也含年份概化'),
('10.5194/hess-8-695-2004','过程结构修正反例','NO3浓度及峰时','经验积雪/土温模块修正转化速率；非源总量校正','全文定向核查：芬兰Simojoki和挪威Dalelva；各处适用限制','浓度水平未显著改变而高浓度时间改善，说明动态误差也可能在转换结构'),
('10.2166/nh.2010.007','分区过程水质','流量/N/P','土壤—土地利用分区；水路径伴随营养物转化','摘要+官方说明：小流域率定和两个较大流域参数迁移；未核实所有输入实测性','可迁移共享参数有先例，不能推断本项目15HF站支持真实230河段输入反演'),
('10.1016/j.jhydrol.2007.05.020','确定性—模糊混合候选','硝酸盐动态','标题确认hybrid deterministic–fuzzy；内部校正位置未核实','仅元数据；全文/摘要缺项','不因标题含hybrid就补造守恒、合成测试或成绩'),
('10.1029/2005WR004368','贝叶斯输入误差理论','径流','暴雨潜在乘性输入误差与过程参数同时推断','摘要：理论；输入/输出/结构误差需显式假设','TN也需区分来源误差与结构误差；不能只以拟合确定真实输入'),
('10.1029/2005WR004376','输入误差应用','径流','BATEA+VIC，降雨和输出不确定性','摘要：French Broad/Potomac两流域，含严重结构误差讨论','输入校正会改变过程参数；错误误差模型和计算维数限制仍在'),
('10.1029/2007WR006720','反演输入与参数','径流','DREAM估计降雨误差与五参数过程模型','摘要：两流域；后验分布与预测界变化','提供反演方法依据，不证明TN单变量足以识别源强'),
('10.1002/hyp.14847','物理约束机器学习','径流','两个总雨量不同的驱动产品；严格质量闭合对照','摘要：闭合限制损害部分技巧，但仅解释DL与概念模型差距小部分','直接反驳“只要修准输入就足够”的强结论'),
('10.1029/2019WR024922','过程引导深度学习','湖泊水温','物理模拟合成预训练+能量违反惩罚','摘要：2详细湖+68湖；测试中位RMSE 1.65°C，DL1.78，PB2.03；需预训练覆盖更大变幅','合成数据要覆盖状态变化；温度效果不能当作TN改善数字'),
('10.1038/s41467-021-26107-z','可微参数学习','土壤水分/径流','共享映射输出过程参数；过程模型传递梯度','全文定向核查+摘要：空间/未率定变量推广与数据规模效应','支持通过过程链学习；不赋予本地来源校正独立真实性'),
('10.1029/2022WR032404','可微区域过程模型','径流及未训练状态','HBV模块由网络参数化、增强或替代','摘要+冻结代码：671流域；Daymet中位NSE .732 vs LSTM .748，另一驱动 .715 vs .722','更换驱动仍有结构差距；大样本水文不是小样本TN承诺'),
('10.1029/2023WR036461','守恒可学习模型','径流','MCP库存/流量门；仓库另含输入偏差校正实现','摘要+代码：Leaf River概念验证；InputBiasCorr.py属于仓库证据，不冒充本文已验证所有扩展','校正输入可与守恒共存；代码加性校正、FloatTensor和观测接口不原样移植'),
('10.1137/1.9781611976700.69','过程引导图循环网络','流量/水温','河网图网络+过程知识迁移；平衡各河段损失','OpenAlex摘要：Delaware子网，稀疏数据、跨季节/河段流量范围；硬质量守恒未核实','网络传播和合成知识有用的案例；不可直接证明氮源估计'),
('10.1016/j.agwat.2013.08.003','SWAT管理措施候选','硝酸盐淋失','标题包含氮肥施用与排水配置','元数据；实际试验设计和准确率未核实','保留为后续全文线索，不作为改日历已证实有效的证据'),
('10.1016/j.jhydrol.2022.127675','过程模型模拟器','库水温/细菌/Zn/Pb','气象水文驱动物理模型，再用其输出训练LSTM','OpenAlex摘要：挪威Brusdalsvatnet；主要是仿真结果仿真','重现过程模拟器不同于提高独立实测TN准确性'),
('10.1371/journal.pone.0125971','简约过程氮记忆模型','出口氮浓度响应','土壤累积氮与地下水行程时间分布共同产生响应滞后','全文定向核查：Iowa流域41%农田转草地；理想化空间干预和实测浓度轨迹比较','更准确输入也经库存与路径时间滤波；干预空间位置改变响应速度，不只由总量决定'),
('10.1029/2018wr023815','SWAT-LAG结构反例','流域氮输出','修改土壤C-N积累和地下水行程时间分布','摘要：502平方公里Iowa流域；1950—2016，未来减肥情景；非独立未来观测验证','相同100%减肥情景，达到79%负荷下降所需84年与原SWAT的2年不同；属于模型条件结果，说明输入正确仍不足'),
('10.1029/2021wr031587','ELEMeNT输入—库存反演','氮负荷/浓度/库存','九参数长期遗留氮模型，逐步加入不同观测约束','正式期刊摘要：Weser 1960—2015；土壤氮与浓度可降低等效性；已与ESSOAR预印本去重','直接支持补充土壤氮/水量等独立证据；不能只用TN与负荷反演真实农业输入'),
('https://proceedings.mlr.press/v139/hoedt21a.html','结构守恒神经网络','合成加法/径流等','明确区分质量输入xm与辅助量xa；归一分配与库存输出','官方论文页面+冻结核心代码；PDF下载超时','辅助信息可控门，质量必须单独记账；不能把任意强度直接当守恒质量')]

def main():
    records=json.loads((P/'screening_candidates.json').read_text(encoding='utf-8'));by={a['doi']:a for a in records}
    for filename in ['nitrogen_legacy_search.json','nitrogen_legacy_final_search.json']:
        for a in json.loads((P/filename).read_text())['message']['items']:
            doi=a['DOI'].lower()
            if doi not in by:by[doi]=dict(doi=doi,title=a['title'][0],abstract=a.get('abstract'),evidence='abstract' if a.get('abstract') else 'metadata',file=filename,selected_seed=False)
    by.pop('10.1002/essoar.10510519.1',None)
    by['10.1371/journal.pone.0125971']['evidence']='full_text_targeted_review'
    by['10.1371/journal.pone.0125971']['file']='legacy_plos_full.html'
    for p in P.glob('openalex_*.json'):
        if p.name=='openalex_search.json':continue
        a=json.loads(p.read_text());doi=(a.get('doi') or '').replace('https://doi.org/','').lower();inv=a.get('abstract_inverted_index') or {}
        abstract=' '.join(k for _,k in sorted((v,k) for k,vs in inv.items() for v in vs))
        if doi in by and abstract:by[doi].update(abstract=abstract,evidence='abstract',secondary_evidence_file=p.name)
    extra=json.loads((P/'openalex_search.json').read_text())['results']
    for a in extra:
        key=(a.get('doi') or a['id']).replace('https://doi.org/','').lower()
        if key not in by:by[key]=dict(doi=key,title=a['title'],selected_seed=False,evidence='metadata',query='OpenAlex independent search')
    mc=CASES[-1][0];by[mc]=dict(doi=mc,title='MC-LSTM: Mass-Conserving LSTM',selected_seed=True,evidence='official abstract and archived code')
    assert len(by)<=80
    detailed=[];pending=[]
    for doi,family,target,position,validation,implication in CASES:
        a=by[doi];metadata=a.get('evidence')=='metadata';row=dict(identifier=doi,title=a['title'],family=family,target=target,correction_position_and_resolution=position,validation_and_evidence=validation,implication=implication,evidence_level=a['evidence'],synthetic_test='见具体案例文字；未明确报道的字段为未核实',independent_input_measurement='未从当前证据认证；降雨/监测亦有误差',local_files=a.get('file',a.get('secondary_evidence_file','archived code / official page')),status='metadata_only_pending' if metadata else 'reviewed_with_limited_evidence')
        row['conservation']='未从当前证据核实'
        row['input_resolution']='未从当前证据核实'
        row['time_validation']='未从当前证据核实独立时间留出'
        row['space_validation']='未从当前证据核实独立空间留出'
        row['failure_or_limit']=implication
        if doi in ['10.5194/hess-6-559-2002','10.5194/hess-8-695-2004']:
            row.update(evidence_level='full_text_targeted_review',conservation='氮质量方程；有机氮无限池等假设须区分真实全氮闭合',input_resolution='日水文、氮负荷；部分年均估计',synthetic_test='未核实已知真值源恢复实验')
        if doi=='10.1038/s41467-021-26107-z':row.update(evidence_level='full_text_targeted_review',conservation='继承过程模型；本文没有证明所有输入无偏',input_resolution='日尺度水文/土壤水分应用',space_validation='训练网格邻域及未率定变量ET；河流案例531美国流域',time_validation='文中有独立时段检验；具体本地可迁移性未知')
        if doi=='10.1029/97WR02171':row.update(input_resolution='年输送率/源与空间属性',conservation='网络输送与经验衰减回归；不是本地日库存核')
        if doi in ['10.1029/2005WR004368','10.1029/2005WR004376','10.1029/2007WR006720']:row.update(input_resolution='降雨驱动；理论允许逐暴雨输入误差',conservation='过程模型内部；不等于强制使用错误观测输入闭合',synthetic_test='输入误差反演/结构误差实验讨论；具体真值恢复数值未核实')
        if doi=='10.1029/2019WR024922':row.update(conservation='能量违反惩罚；并非日氮质量硬约束',synthetic_test='过程模拟水温用于预训练；仅覆盖较大变幅时优势明显',time_validation='独立测试及训练范围外情况',space_validation='68湖扩展；不据摘要认定为完全未训练湖泊留出')
        if doi=='10.1029/2022WR032404':row.update(input_resolution='日气象/径流',conservation='可微HBV过程骨架',space_validation='671美国流域；具体留出设置以完整论文为准')
        if doi=='10.1029/2023WR036461':row.update(evidence_level='abstract_and_archived_code',conservation='MCP显式库存流量守恒；输入校正额外登记',input_resolution='降雨径流时序',synthetic_test='本文概念模型功能表达；本地TN合成恢复无直接证据')
        if doi==mc:row.update(conservation='核心归一门重分配，输出从库存扣除',input_resolution='序列质量输入与辅助输入分离',synthetic_test='加法、摆动等合成任务；官方摘要不等于TN输入识别')
        if doi=='10.1016/j.jhydrol.2022.127675':row.update(synthetic_test='以校准过程模型的时空输出训练模拟器',time_validation='模型测试；独立观测与模拟器标签比较需区分',conservation='摘要未证明LSTM硬守恒')
        (pending if metadata else detailed).append(row)
    with (R/'reports/literature_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=detailed[0].keys());w.writeheader();w.writerows(detailed+pending)
    selected={r[0] for r in CASES}
    for key,a in by.items():a['screening_decision']='case_register' if key in selected else 'not_core_case';a['reason']='直接输入/过程/合成方法证据' if key in selected else '非目标过程灰箱证据、仅农业试验、书目附件或外围背景；不为数量纳入核心案例'
    (P/'screening_reviewed.json').write_text(json.dumps(list(by.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 输入误差与灰箱可行性：案例证据核查','',f'去重筛查 {len(by)} 项；详细登记 {len(detailed)} 项（证据层级见表），另保留 {len(pending)} 项仅元数据线索。详细登记不等于全文通读。Crossref与OpenAlex检索原始响应、下载失败及哈希均在 evidence/literature。','',
    '## 对本轮最直接的启示','',
    '1. 系数的入口位置决定含义。SPARROW/经典输出系数讨论到河负荷或产率；本轮A在陆地库存之前，称活动量代理有效农业输入，不能把两种系数混用。',
    '2. INCA全文明确把施肥作为日质量序列并支持多次施用，同时保留土壤保留体积、植物生长期和沉降洗脱过程（PDF第12页）。它也用了由1994气象/排放估计的沉降支持1998模拟，说明过程模型案例并不等于原始输入完全实测。',
    '3. Frame等的对照显示输入质量与硬守恒的矛盾只解释部分性能差距；INCA积雪修正可改变高浓度出现时间而不明显改变水平。这两类反例要求同时检查输入到浓度的转换。',
    '4. Read等合成预训练依赖合成覆盖范围；库水质LSTM模拟器主要复现过程模型结果。合成实验应检验可恢复性与失败方式，不能代替真实留出验证。',
    '5. BATEA/DREAM把输入误差作为依赖过程模型的潜变量。仅用TN估计出的源强，可能吸收流量分母、漏源和结构误差。','',
    '## 逐案例记录','']
    for a in detailed+pending:
        url=a['identifier'] if a['identifier'].startswith('https') else 'https://doi.org/'+a['identifier']
        text += [f"### [{a['title']}]({url})",'',f"类别：{a['family']}；目标：{a['target']}。",'',a['correction_position_and_resolution']+'。', '',a['validation_and_evidence']+'。', '',a['implication']+'。','']
    text += ['## 代码核查与边界','',
    'MCP固定版本InputBiasCorr.py第108—133行：先形成u1_bc，再送入库存；这支持“校正质量后守恒”，不是把任意辅助量叫质量。该实现有FloatTensor、加性校正与y_obs接口；本实验未导入或执行这些外部核心。',
    'MC-LSTM核心第105—115行：归一重分配、输入门及输出门划分质量；xm与xa分别作为质量与辅助输入。本轮保留D29库存与来源账本，仅借鉴分离质量和调节信息的原则。',
    'HydroDL2代码/固定commit与许可原始回执沿用已核验归档；许可未明确的文件只作阅读证据。本轮没有移植第三方科学核。',
    '三份仅元数据线索（Johnes、确定性模糊硝酸盐、SWAT施肥排水）不提供本次实验成功的证据；其未知细节未补写。','']
    text += ['## 氮记忆与独立证据的追加定向检索','',
    'SWAT-LAG和ELEMeNT的氮案例提供了比水温类比更直接的约束：输入施加之后，土壤遗留量与地下水输送时间仍可控制响应；额外土壤氮资料能减少源与过程的等效解。它们不说明本地D29应立即增加年龄结构，而是要求本轮脉冲试验和错误归因反例回答输入信号能否被利用。',
    'ESSOAR 10.1002/essoar.10510519.1与正式发表10.1029/2021WR031587按同一研究合并；未把预印本重复算成独立案例。','']
    (R/'reports/文献与代码证据.md').write_text('\n'.join(text),encoding='utf-8')
    (R/'reports/literature_audit.json').write_text(json.dumps(dict(status='COMPLETE_WITH_ACCESS_LIMITS',screened=len(by),reviewed_cases=len(detailed),metadata_only=len(pending),large_data_downloaded=False),indent=2))
    print('LITERATURE',len(by),len(detailed),len(pending),flush=True)
if __name__=='__main__':main()
