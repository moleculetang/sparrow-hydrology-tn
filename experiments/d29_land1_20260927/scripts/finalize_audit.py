"""Build an evidence-linked closeout without converting unresolved gaps into passes."""
from pathlib import Path
import hashlib, json, re
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'20260926_1'
OUT=ROOT/'outputs/final_audit';OUT.mkdir(exist_ok=True)
REPORT=ROOT/'reports';REPORT.mkdir(exist_ok=True)
RUN=ROOT/'outputs/supply_limited_reference'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def link(p,label=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return f'[{label or p.name}](<{p.as_posix()}>)'
def table(df):
    def cell(x):
        if isinstance(x,(float,np.floating)):return '不可定义' if not np.isfinite(x) else f'{x:.6g}'
        return str(x).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

# Every original identifier is reviewed, with a concrete new-model action and limit.
# Evidence paths are verified below. Status is deliberately not a scientific success flag.
rows=[
('land_inventory_filtering','候选已改，科学未关闭','以Sa/Sp矿化、植物及可用池替代原M；退出旧M寿命；完整历史运行。','活性矿化仍为固定90天半衰期情景；未做同参数单因素归因或证明动态最优。','d29_platform/land1.py'),
('observed_amplitude_phase','已评价，未解决','日/月内NSE、幅度、相关、峰日按共同支持重算。','振幅提高到1以上而偏差和多数动态误差扩大；不能把增幅称为修复。','outputs/supply_limited_reference/evaluation/2024_daily_summary.csv'),
('monthly_support_45','覆盖接口已扩展','新独立源覆盖230河段，输出全部116站界，不再限旧13区反推支持。','未删45个月站；新的完整源仍不等于已确认真实负荷，旧反推结论不改写。','outputs/input_contract_audit/receipt.json'),
('historical_initial_state','明确情景，未识别','登记现代SOC/14初态、64年连续源和植物活动、初态来源标签。','2024局地出口约25%仍来自初态标签；1961真实SON及历史植物活动未知。','outputs/supply_limited_reference/audit/annual_local_export_source_fractions.csv'),
('network_time','保留诊断，未修改','沿冻结H1路由和OU逐来源重放，检查网络账本。','普通河段同日传递保留；没有独立旅行时间证据，不能按残差加全网时滞。','outputs/supply_limited_reference/audit/independent_result_audit.json'),
('boundary_water','接口核验，观测证据缺失','同日同OU质量/水量及C=1000M/Q复算，零水不定义浓度。','9河段H1容量面积与几何差>1%；同站界实测Q未认证。','outputs/input_contract_audit/receipt.json'),
('optimizer_convergence','旧限制保留，本轮不适用','本轮为冻结参数前向，拟合调用为0；不把前向完成标成收敛。','旧四联合点未充分状态未改；新28路径未派发。','outputs/supply_limited_reference/evaluation/receipt.json'),
('depletion_kinks','新增分支已修验','精确满足植物目标时用代数等价目标分支消除负末位；严格不可行仍报错；供氮受限新分支独立验导。','耗尽边界不声称光滑；计划短缺是情景不相容，不能用浮点修正消除。','outputs/precision_revision/potential_activity_validation.json'),
('input_bounds','本轮不使用旧反推输入族','不使用W7/D1或TN辅助源；独立资料与研究默认构造kg N/day。','国家趋势、缺率代理和年度均匀日历不是真实管理日期；不宣称准确输入。','d29_platform/conditional_inputs.py'),
('source_eta_inactive','新运行已避开','不移植旧theta30及来源先验，不保留只影响先验的校准坐标。','本次未优化；未来引入倍率必须全历史作用外源并重新验导。','config/conditional_reference.json'),
('sampling_and_aggregation','已实施共同支持','日读数加权、月内等月、HF月与原月报分开；事件支持缺项明列。','原月报实际采样时间仍未认证；日平均无法重建日内尖峰。','outputs/supply_limited_reference/evaluation/receipt.json'),
('causal_comparison','归因门保留','明确候选同时改变源、历史、植物活动和方程，仅描述完整配置差异。','不将新旧NSE差解释为某个单独过程的作用。','outputs/supply_limited_reference/evaluation/receipt.json'),
('land_units_and_cross_class_uptake','实现修正并验证','230河段×12物理类及被动隔离槽；作物源、活动只进入耕地。','LUH/CLCD交叉映射仍是支持适配；转地后的质量搬移不代表化学转换已识别。','outputs/input_contract_audit/receipt.json'),
('physical_area_vs_multiple_harvests_and_CCD','库存重复问题已防止；CCD未接入','LUH物理面积建库存；收获面积只估源与活动，复种不复制土壤；CCD退出驱动。','CCD近邻重叠非精确并集，省界去重未完成，不宣称已消除原产品冲突。','outputs/agriculture_reference/receipt.json'),
('harvest_proxy_vs_total_plant_demand','语义拆分并新增可行性分支','产出N、残体归还、植物目标分别登记；严格运行失败保留，供氮受限另命名。','2024植物归还计划7.40%未实现；不能声称产量/NPP目标已全部达成。','outputs/supply_limited_reference/audit/annual_balance_and_activity_deficits.csv'),
('external_inputs_and_internal_transfers','账本路径已修验','矿化、摄取和归还为内部；四入口外源互斥，MOD17不是外源。','大库存绝对质量门仍未通过，不能由小核通过推定正式账本合格。','outputs/precision_revision/potential_activity_validation.json'),
('old_effective_M_lifetime_reuse','已退出','LAND1不接收旧M寿命及七项空间寿命映射；动员和快慢/H1保留。','新矿化率为研究改编；旧水文映射迁移未证明已适合新氮库。','config/conditional_reference.json'),
('SON_initialization_and_modern_soil_TN','单位已修，初态情景化','SOC换算、容重、砾石和深度审计；C:N14，活性比例.02显式配置。','SOC/14与独立土壤TN有最大1.457倍比值；不裁剪强行一致；现代不能识别1961。','outputs/soil_reference/son_soc14_receipt.json'),
('residue_fates_and_animal_recycling','互斥处理，实际去向未知','.76直接归还，.24登记未解析域外去向；不再重复回加粪肥。','根残体、放牧/焚烧细分和动物再循环缺资料；.76跨历史/区域使用是情景。','config/conditional_reference.json'),
('land_use_change_stock_transfer','守恒接口已实现','LUH年y算子于y+1元旦搬移全部库存和来源；城市转入质量被动保留。','面积转换不提供收获/燃烧/清理氮去向，城市封存只是边界情景。','outputs/luh3_adapter/validation.json'),
('old_source_vintages_and_2024_agriculture','年份/代理已逐项登记','2021—24面积从2020空间型按对应国家趋势调整；2024施肥用全国N比例一次；粪肥独立强度。','未获得2021—24河段实测农业活动；不把false替代标志当新资料。','outputs/agriculture_reference/annual_crop_source_reference.parquet'),
('annual_national_scaling_and_real_input_calendars','重复缩放已防止，日期未知','固定28商品完整篮子，面积仅分配总量；年度源和活动均匀日分。','全国不是流域实测；均摊不增加施肥/收获事件信息，宽作物组代理仍粗。','scripts/prepare_agriculture_reference.py'),
('new_deposition_product_and_units','已接入并独立复核','新input4MIPs四分量取增量月量；保留ScenarioMIP与土地掩码年份。','2023—24为情景非观测，上游暴露表不作第二份来源。','outputs/input_contract_audit/receipt.json'),
('deposition_daily_and_surface_eligibility','严格冲突保留，单列研究版本','统一月内日分为独立情景；非植被/未知表面进入被动隔离，水面排除。','93河段月正湿沉降/零H1雨未物理解决；隔离不是城市冲刷模型。','outputs/deposition_reference/receipt.json'),
('BNF_seed_manure_forms_and_total_form_double_count','入口和总量互斥已实现','豆科BNF入植物，稻/甘蔗入土；田施粪肥矿质/有机互斥；不叠加总N和形态。','自然BNF、种子、其他豆科及牧场排泄未量化；矿质.5及有机分配是改编情景。','config/conditional_reference.json'),
('direct_river_sources_and_full_TN_boundary','缺项显式保留','直接入河接口不暗加土壤；缺负荷及位置的污水/工业/养殖源不启用。','不能称来源齐全；这也限制TN诊断归因。','config/conditional_reference.json'),
('MOD17_noncropland_plant_production','已接入，默认假设可追溯','仅非耕地碳→器官N活动；QC百分数保留，数值有效值不误作bitmask。','2024 QC≤50仅34.28%支持；1961—2000均值情景、缺地类借值、年度器官周转皆非观测。','d29_platform/conditional_inputs.py'),
('H1_full_chain_and_prediction_claim','完整运行已完成','1961—2024真实资料/情景组合经陆地—河网—OU到116站；输入生成不读TN。','未经校准，且绝对账本门失败；2024为F23参数回放，不是F24预测或2025预报。','outputs/supply_limited_reference/receipt.json'),
('ENG_gradient_acceptance','修正保留且复验','72项继承测试；多步长需两相邻通过，保留更小步长消差失败；新分支对照Torch。','不是逐个穷举所有真实历史坐标，真实运行年块前向不是截断梯度训练。','outputs/precision_revision/receipt.json'),
('ENG_source_responsibility','当前参数职责已登记','外源五类标签和初态责任分离，旧失活来源参数退出。','来源标签为来历不是化学形态。','outputs/supply_limited_reference/audit/annual_local_export_source_fractions.csv'),
('ENG_full_land1_accounting','部分通过，严格门失败','新分支、补偿求和及补偿检查点完成；小核/合成全历史导数通过。','实际大库存局地1.1444e-5kg、标签5.4479e-5kg均超1e-6kg；正式运行仍阻塞。','outputs/supply_limited_reference/receipt.json'),
('ENG_comparison_identity','规则通过','72项测试复验历史身份拒绝门，冻结输入/代码/预测哈希；无单因素增益主张。','描述比较可以做，因果贡献尚不能隔离。','outputs/precision_revision/inherited_72_tests.log'),
('ENG_evaluation_support','已复算','逐站NSE/月内NSE、两汇总口径、比例及1000月块/双月块完成。','2024日合格14站，事件有支持不足；不与旧轮不同支持排行榜混合。','outputs/supply_limited_reference/evaluation/receipt.json'),
('ENG_missing_data_gate','严格门保留，条件分支单列','未知不默认零；条件版本明确排除未知边界、被动隔离和估计值；正式28路径不派发。','授权研究默认不等于所有未知量获独立证实，正式来源完备为false。','outputs/input_contract_audit/receipt.json'),
('fast_pool_same_platform','不重复旧局部快库试验','保留既有阴性范围，新LAND1不再新增第三快速库存。','此次均值/幅度变化不能归因为旧快库获胜或失败。','config/conditional_reference.json'),
('carrier_alignment','代码层检查通过，物理层未认证','正快路N无正快路水为0例，慢路同为0；同OU路由和分母重算。','这只排除零载体等实现错误；不证明供体混合体积、水龄、氮随水日期均真实。','outputs/input_contract_audit/receipt.json'),
('nitrogen_species_observation','保留未解决','不把来历标签冒称NO3/NH4/颗粒N；有效N与实测TN区别列明。','没有形态/悬沙证据前不增加不可识别形态参数。','config/conditional_reference.json')]

old=read(PARENT/'retrospective_v1/planning/gap_register_v2.json')['items']
assert {v['id'] for v in old}=={r[0] for r in rows} and len(rows)==37
claims={v['id']:v['claim'] for v in old}
# Locate the actual named LUH receipt, keeping missing evidence a hard failure.
for i,r in enumerate(rows):
    if r[0]=='land_use_change_stock_transfer' and not (ROOT/r[-1]).exists():
        files=list((ROOT/'outputs/luh3_adapter').glob('*receipt*.json'))
        if not files:files=list((ROOT/'outputs/luh3_adapter').glob('*.json'))
        assert files
        rows[i]=(*r[:-1],files[0].relative_to(ROOT).as_posix())
assert all((ROOT/r[-1]).is_file() for r in rows)
items=[dict(id=k,original_claim=claims[k],status=s,action=a,remaining=l,evidence=str(ROOT/e),scientific_closed=False) for k,s,a,l,e in rows]
save(OUT/'gap_crosswalk_37.json',dict(reviewed=37,all_original_ids_covered=True,items=items))
cross='# 37项历史缺口对新模型的逐项审查\n\n2026-09-27。依据上一轮37项登记逐一核对。工程通过、条件情景、未解决科学问题分开；没有将真实TN问题标为科学闭合。\n\n'
for n,r in enumerate(items,1):
    cross+=f"## {n}. {r['id']}\n\n原问题：{r['original_claim']}\n\n**{r['status']}**。本轮：{r['action']}\n\n仍缺：{r['remaining']}\n\n证据：{link(r['evidence'])}。\n\n"
(REPORT/'历史37项问题逐项修正对照.md').write_text(cross,encoding='utf-8')

# Summaries always originate from saved paired outputs, not hand-copied values.
core=[];secondary=[]
for year in (2023,2024):
    d=pd.read_csv(RUN/f'evaluation/{year}_daily_summary.csv')
    for _,r in d.iterrows():
        if r.metric in ['nse','month_centered_nse']:
            core.append([year,r.metric,int(r.n_common_stations),r.baseline_median,r.candidate_median,r.difference_of_medians,r.median_paired_difference,r.improvement_fraction])
        else:secondary.append([year,r.metric,int(r.n_common_stations),r.baseline_median,r.candidate_median])
core=pd.DataFrame(core,columns=['年','指标','共同站数','U中位数','候选中位数','中位数之差','配对差中位数','改善比例'])
core.to_csv(OUT/'core_nse.csv',index=False)
second=pd.DataFrame(secondary,columns=['年','指标','共同站数','U中位数','候选中位数'])
decomp=pd.read_csv(RUN/'audit/station_year_nse_decomposition.csv');diagnostic=[]
for year in (2023,2024):
    st=pd.read_csv(RUN/f'evaluation/{year}_daily_station.csv')
    ok=st[st.candidate_nse_status.eq('defined')]
    w=decomp[(decomp.year==year)&decomp.model.eq('candidate')&decomp.station_key.isin(ok.station_key)]
    diagnostic.append(dict(year=year,stations=len(w),amplitude_exceeds_one=int((w.amplitude_ratio>1).sum()),
        amplitude_exceeds_correlation=int((w.amplitude_ratio>w.correlation).sum()),
        median_bias_fraction_of_normalized_mse=float(np.median(w.bias_penalty/(1-w.nse))),
        excluded_station_records=st[~st.candidate_nse_status.eq('defined')][['station_key','n_common_days','n_common_months','candidate_nse_status']].to_dict('records')))
save(OUT/'waveform_interpretation.json',diagnostic)
events=pd.read_csv(RUN/'evaluation/frozen_events.csv');er=[]
for year,g in events.groupby('year'):
    g1=g[g.status.eq('defined')]
    pair=g1[['baseline_peak_date_offset_days','candidate_peak_date_offset_days']].dropna()
    er.append(dict(year=int(year),registered=len(g),defined=len(g1),insufficient=len(g)-len(g1),
        peak_date_improved=int((pair.candidate_peak_date_offset_days.abs()<pair.baseline_peak_date_offset_days.abs()).sum()),
        peak_date_worse=int((pair.candidate_peak_date_offset_days.abs()>pair.baseline_peak_date_offset_days.abs()).sum()),
        peak_date_unchanged=int((pair.candidate_peak_date_offset_days.abs()==pair.baseline_peak_date_offset_days.abs()).sum())))
save(OUT/'event_support.json',er)

# Verify parent source snapshot and exact implementation that produced the run.
snap=read(ROOT/'evidence/parent_snapshot.json')
parent_checks=[dict(path=k,matched=(Path(snap['parent'])/k).is_file() and sha(Path(snap['parent'])/k)==v) for k,v in snap['sha256'].items()]
frozen=read(RUN/'configuration_frozen.json')
implementation=[dict(path=k,matched=sha(k)==v) for k,v in frozen['implementation_hashes'].items()]
testlog=(ROOT/'outputs/precision_revision/inherited_72_tests.log').read_text(encoding='utf-8',errors='replace')
passed72='Ran 72 tests' in testlog and testlog.rstrip().endswith('OK')
run=read(RUN/'receipt.json');evaluation=read(RUN/'evaluation/receipt.json')
receipt=dict(task_scope='new conditional model run plus all 37 previous gaps reviewed; not all scientific gaps solved',
    historical_gaps_reviewed=37,new_forward_complete=run['status']=='completed_conditional_forward',
    days=run['days'],reaches=run['reaches'],old_snapshot_files_checked=len(parent_checks),
    old_snapshot_all_unchanged=all(x['matched'] for x in parent_checks),old_snapshot_checks=parent_checks,
    legacy_readonly_deviation='An earlier inherited test invocation created Python bytecode/test temporary artifacts under the old test tree. Frozen source/data/report snapshot hashes verified unchanged; tests relocated and PYTHONDONTWRITEBYTECODE=1 thereafter. No old files deleted.',
    run_implementation_unchanged=all(x['matched'] for x in implementation),implementation_checks=implementation,
    inherited_72_tests_passed=passed72,
    precision_validation_passed=read(ROOT/'outputs/precision_revision/receipt.json')['passed'],
    potential_activity_validation_passed=read(ROOT/'outputs/precision_revision/potential_activity_validation.json')['passed'],
    input_contract_validation_passed=read(ROOT/'outputs/input_contract_audit/receipt.json')['passed'],
    strict_mass_pass=run['strict_mass_pass'],formal_calibration_allowed=False,formal_paths_started=0,
    all_input_uncertainty_resolved=False,TN_prediction_improved=False,
    unresolved=['wet deposition versus zero H1 rain','independent station-boundary Q','modern SON versus historical state','real agricultural daily timing and reach quantities','CCD exact deduplication (not a driver)','plant fate and complete direct/noncrop nitrogen sources','large-stock absolute ledger tolerance'],
    reports=['专家新模型与全链条审计报告.md','实际方法与偏离.md','历史37项问题逐项修正对照.md'])
assert passed72 and receipt['old_snapshot_all_unchanged'] and receipt['run_implementation_unchanged']
save(OUT/'completion_receipt.json',receipt)

expert=f'''# 专家新模型与全链条审计报告

2026-09-27；实验目录20260927_1。**新模型已完成一次1961—2024、230河段全历史条件性运行，37项历史缺口已逐项审查；并非所有输入矛盾或真实动态问题已解决。** 本轮无新拟合，不能认定LAND1优于或劣于公平校准后的U。绝对质量验收尚有失败，正式校准继续阻塞。

## 1. 做了什么、修正了什么

新源使用LUH3物理土地状态/转换、新沉降、HaNi农业参考及FAOSTAT趋势、MOD17非耕地活动和土壤参考。资料不足项依据文献建立有身份的研究默认，不读取同期TN反推源。新骨架区分植物、活性/保护SON、可用N和慢水N；矿化与归还是内部转移，旧M寿命及七项寿命映射退出。

实际修改包括：土地转换守恒搬移及导数；矿化固定/温度接口与单位门；缺肥率不再默认零；国家比例只作用一次；QC百分数不作位码；作物与非耕地活动分开；精确植物需求处的消差修正；大SON补偿求和与补偿检查点；供氮受限植物去向分支及完整伴随。后者是单列科学情景，不是把原不可行计划悄悄修成可行。

所有旧问题的“证据—行动—未解决”见{link(REPORT/'历史37项问题逐项修正对照.md')}，机器登记见{link(OUT/'gap_crosswalk_37.json')}。**振幅不足没有被作为一个可独立放大的目标。** 新结果表明库存/入口改变可以增大振幅，但均值和日期对应必须同时约束。

## 2. 真实TN条件性评价

U采用冻结F23合法参数覆盖所有年份；2024不是重新拟合F24。新旧的源、初态、植物活动和方程同时改变，因此下表为完整配置的描述比较，不是来源或某个库存的单因素增益。候选未校准；评价后没有调参数再选成绩。

{table(core)}

{table(second)}

RMSE、偏差单位mg/L；幅度比为预测/观测标准差。日指标按读数次数加权；月内先在站月内按读数次数去均值，再等月汇总误差平方与观测方差。覆盖门为至少120日、6个月，零方差不强算。逐站及HF月/原月报完整表见{link(RUN/'evaluation/2023_daily_station.csv','2023逐站')}、{link(RUN/'evaluation/2024_daily_station.csv','2024逐站')}、{link(RUN/'evaluation/2023_HF_monthly_summary.csv','2023 HF月')}、{link(RUN/'evaluation/2024_HF_monthly_summary.csv','2024 HF月')}、{link(RUN/'evaluation/2023_original_monthly_summary.csv','2023原月报')}、{link(RUN/'evaluation/2024_original_monthly_summary.csv','2024原月报')}。2021没有满足上述日覆盖门的站，不补出可比NSE。

2024缺资格记录为：{json.dumps(diagnostic[1]['excluded_station_records'],ensure_ascii=False)}。不能为了保留15站而改变门槛。原月报实际采样时间未认证，模型仍使用冻结日历权重。

seed1729、1000次同步月块和连续两月块结果已生成，例如{link(RUN/'evaluation/2024_bootstrap_1month_intervals.csv')}与{link(RUN/'evaluation/2024_bootstrap_2month_intervals.csv')}。区间只描述冻结结果对观测月份的稳定性，不能认证预测能力或输入精度。

## 3. 为什么振幅起来了，动态仍不好

同支持有NSE=2ra−a²−b²，其中r为相关、a为幅度比、b为标准化偏差。独立代数复算最大误差1.82×10⁻¹²。候选2023/2024日幅度中位数分别1.495/1.375，已不再是原先约0.54/0.53的普遍低幅度；但日偏差中位数为+3.766/+2.681mg/L。相关只从0.380/0.529变为0.444/0.553，增幅不足以抵消均值与幅度误差。月内去均值后NSE仍整体恶化，说明问题不只是均值偏高。

按逐站分解，2023/2024有{diagnostic[0]['amplitude_exceeds_correlation']}/{diagnostic[1]['amplitude_exceeds_correlation']}个合格站a>r，继续只放大同一异常曲线会增加误差；偏差项占标准化MSE的逐站中位比例为{diagnostic[0]['median_bias_fraction_of_normalized_mse']:.1%}/{diagnostic[1]['median_bias_fraction_of_normalized_mse']:.1%}。这是描述分解，不能由此唯一识别水量或源日期是主因。证据：{link(RUN/'audit/station_year_nse_decomposition.csv')}。

事件支持清单：{json.dumps(er,ensure_ascii=False)}。原冻结记录保留；背景和事件日期交集不足者不算成功事件。2023登记51、可评价34，不能冒称旧40事件支持；2024登记83、可评价57，并列26个不足记录。峰日并列取共同日期中的首日。逐事件NSE、幅度/峰值/背景误差与峰日见{link(RUN/'evaluation/frozen_events.csv')}。这些峰日结果没有支持统一全网延迟。

空间上，新来源覆盖全部230河段，全部116站界有读出，解决了旧反推只改变部分区域的接口范围问题；但其空间量仍含国家到河段代理，不能由覆盖完整推断空间分配准确。快慢路正N没有零水载体的实现错误（各0例），仍不证明每股水的混合库存、年龄和质量对应已正确。缺少同站界实测Q是未关闭的科学证据口。

## 4. 源、植物和初态解释

2024外源合计33.522亿kgN；其中农业肥料16.472亿、田施粪肥4.358亿、植物BNF1.799亿、土壤BNF0.943亿、陆地加未知表面沉降9.951亿。水面沉降0.0963亿被排除，0.4223亿沉降进入被动隔离；后两者不能伪装成土壤可用源。沉降原已知陆地约9.948亿与本账本含未知表面的9.951亿口径不同。

局地出口来源标签：沉降16.94%、肥料46.57%、田施粪肥7.93%、作物BNF3.55%、初态研究库存25.02%。这是固定条件下的来历追踪，不是可识别的真实源贡献，也不是各源干预的因果贡献。历史初态影响在2024仍明显，不能称暖机后自然消失。不同标签经过同一河网/OU后，浓度总和与实体浓度最大差仅1.29×10⁻¹²mg/L；微小标签数值误差不能解释数mg/L的浓度偏高。

本配置可用池非出口损失为0，是无额外未知损失的上界参考，不是本流域矿化/反硝化实测。现代SOC/14与慢库释放、历史来源、非耕地活动及旧动员映射共同决定供氮，不能凭此次过高浓度就指定其中一项为主因。下一次科学对照应先独立限定这些量，不能用TN任意拟合损失来掩盖不相容。

原严格植物计划在1968-01-24、reach34牧场缺278.865kgN而中止；该失败保留。另行冻结“潜在计划、供氮受限”情景后才完成全历史。2024植物归还计划仍缺3.222亿kgN，占7.40%，当年计划作物运离缺额为0。后者不意味着植物目标库存、所有NPP及真实收获均获证实；当前短缺表度量计划去向，尚不能替代完整器官生产误差诊断。

## 5. 数值审计与不能签发的部分

72项继承测试、28项新供氮受限分支检查、24项输入/载体检查通过；全历史合成H1/OU及7个梯度方向复验通过。完整历史导数在合成条件验证，不宣称逐一覆盖每个真实历史折点。最小差分步长出现消差失败仍保留，判定使用至少两个相邻步长通过；不再挑单一最好误差。

实际大库存局地质量误差最大1.1444×10⁻⁵kg、来源状态差5.4479×10⁻⁵kg，**均未满足1×10⁻⁶kg绝对门槛**。网络相对误差4.82×10⁻¹⁷通过。补偿求和改善稳定性但未闭合绝对门；年度继续时仅允许显式诊断重启误差，不改质量、不提高正式门槛，回执仍strict_mass_pass=false。年度相对残差很小不能替代绝对标准。新正式拟合不允许启动。

这次没有优化，因此不报告投影梯度收敛，也不改变旧四联合点未充分的结论。新运行接口按年度流式前向，未据此建立截断暖机梯度训练。源代码冻结检查、父文件哈希及全部限制见{link(OUT/'completion_receipt.json')}。

## 6. 文献如何指导修正与下一步

沿用上一轮22案例/9代码参照并补查SWAT、CTSM、mHM-Nitrate、FAOSTAT氮预算、残体和再吸收资料。没有把22案例宣称为22篇全文新综述。SWAT支撑SOC/C:N与有机库初始化作为方法参考；mHM支撑管理入口和矿质/有机分账；CTSM支撑器官配置与再循环。它们的原系数不能跨库存定义直接等同；本轮的90天、270年、.76及器官设置均有改编边界。详见{link(REPORT/'实际方法与偏离.md')}及{link('config/conditional_reference.json')}。

建议优先顺序：①先解决大库存绝对账本误差及真实规模检查点验收，保留全部失败；②独立约束SON活性/慢库、植物去向、可用氮有效损失和完整来源，避免先用TN统一补偿；③补独立农业日期和同站界Q，检验局地快慢载体与采样支持；④在共同目标和同历史配置下公平校准，再考虑有上游支持站的有限传播核。当前不增加快库、不统一加延迟、不用仅增幅作为晋级条件。

**本轮支持“新骨架可以运行并显著改变幅度”，不支持“动态已恢复”“新输入已准确”或“所有审计缺点补齐”。** 正式运行仍有绝对质量验收和独立输入支持两道缺口；已形成可复现结果和具体未解决项，而非把不确定性隐藏在默认零或成功标志中。
'''
(REPORT/'专家新模型与全链条审计报告.md').write_text(expert,encoding='utf-8')
print('37 gaps, reports and closeout written',len(parent_checks),'parent files checked')

