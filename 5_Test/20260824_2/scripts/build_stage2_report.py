from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


CURATED = [
    ("10.1371/journal.pone.0125971", "framework", "Biogeochemical SON depletion is convolved with a groundwater travel-time distribution.", "Iowa prairie-conversion concentration trajectory plus groundwater-flow/travel-time modelling.", "Supports separating source-zone memory from hydrologic delay; does not identify a manure-only pool."),
    ("10.1088/1748-9326/11/3/035014", "review", "Anthropogenic landscapes accumulate nitrogen in soil and groundwater.", "Synthesis of mass-balance, soil, groundwater and watershed evidence.", "Supports a Legacy state; not a local PRB parameter prior."),
    ("10.1002/2016gb005498", "reconstruction", "ELEMeNT combines a 214-year N-input history with soil and groundwater storage/transport.", "Historical input reconstruction and outlet nitrate-yield hindcasting in two major basins.", "Long source histories and outlet loads are the main evidence; internal age fractions are model diagnostics."),
    ("10.1126/science.aar4462", "management", "Accumulated basin N can delay achievement of Gulf water-quality targets for decades.", "Historical mass balance, Legacy modelling and management counterfactuals.", "Justifies long planning horizons, not a universal lag time."),
    ("10.1038/s41561-021-00889-9", "synthesis", "Legacy-aware management must address stored N as well as current inputs.", "Cross-system synthesis of legacy mitigation and policy options.", "Supports decision framing, not source-specific calibration."),
    ("10.1088/1748-9326/ac0d7b", "watershed_application", "Legacy stores help explain delayed Chesapeake tributary responses.", "ELEMeNT-N hindcasting and scenario analysis across nine tributaries.", "Demonstrates between-watershed heterogeneity and the need for spatial validation."),
    ("10.1038/s41893-024-01369-9", "global_groundwater", "Groundwater N accumulation persists in the Rhine, Mississippi, Yangtze and Pearl basins.", "Reconstructed and projected groundwater N dynamics across four major basins.", "Directly establishes Pearl-basin relevance but remains basin-scale model evidence."),
    ("10.1073/pnas.1305372110", "tracer", "Thirty years after labelled fertilizer application, 12-15% remained in soil organic matter and 8-12% had leaked toward groundwater.", "Long-term isotopically labelled fertilizer field experiment.", "Strong independent evidence for slow soil retention and later release; it is fertilizer-specific, not a manure fraction estimate."),
    ("10.1038/s41467-017-01321-w", "vadose_storage", "Large nitrate stores in thick vadose zones create long delays.", "Global leaching-depth reconstruction validated against basin/national estimates and observed groundwater nitrate.", "Supports an unsaturated-zone storage pathway and explicit validation against external observations."),
    ("10.1088/1748-9326/ac55b5", "soil_process", "SON accumulation and mineralization depend on yield, residues, climate, soil and management; equal SON stock can produce different mineralization fluxes.", "CENTURY simulations calibrated to crop yield and long-term soil-N accumulation data.", "Argues against interpreting a fixed 12-month pool as a universal manure mineralization constant."),
    ("10.1088/1748-9326/ac243c", "nested_spatial", "Soil and groundwater Legacy fractions vary across nested agricultural basins and co-vary with travel time.", "ELEMeNT-N application across 14 nested basins.", "Shows why temporal fit alone cannot establish spatial transfer."),
    ("10.1088/1748-9326/acd1a2", "china_application", "Time-varying SON mineralization improved an eastern-China ELEMeNT-N application.", "Thirty-one-year river-N record; reported NSE and R2 comparisons; temperature-driven annual mineralization modification.", "Shows potential value of climate-dependent mineralization only after a basic Legacy structure is identifiable."),
    ("10.1088/1748-9326/acea34", "groundwater_observation", "Groundwater nitrate inventories can constrain Legacy mass independently of river TN.", "Approximately 49,000 well nitrate values plus spatial prediction and mass-balance comparison.", "A template for independent subsurface validation absent from the current PRB data."),
    ("10.1029/2020gb006626", "source_history", "Long, spatially explicit N mass-balance histories are needed to interpret Legacy trajectories.", "County-scale 1930-2017 TREND-Nitrogen source/sink synthesis.", "Supports the present source-history audit and explicit source-product uncertainty."),
    ("10.1016/j.scitotenv.2021.146698", "policy_review", "Policy evaluation should include storage dynamics and time lags.", "Review and policy synthesis.", "Supports communicating delayed response without claiming that a fitted lag is a measured age."),
]


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(OUT / "legacy_literature_crossref_records.parquet")
    evidence = pd.DataFrame(CURATED, columns=["doi", "evidence_role", "mechanism_or_result", "validation_basis", "prb_model_lesson"])
    evidence = evidence.merge(
        metadata[["doi", "title", "authors", "year", "journal", "url", "abstract", "metadata_source"]],
        on="doi", how="left", validate="one_to_one",
    )
    if evidence.title.isna().any():
        raise RuntimeError(f"missing curated metadata: {evidence.loc[evidence.title.isna(), 'doi'].tolist()}")
    evidence.to_parquet(OUT / "legacy_literature_evidence_table.parquet", index=False)

    decision = json.loads((REPORTS / "synthetic_identifiability_decision.json").read_text(encoding="utf-8"))
    source = json.loads((REPORTS / "source_history_quality_audit.json").read_text(encoding="utf-8"))
    hydro = json.loads((REPORTS / "hydrologic_provenance_audit.json").read_text(encoding="utf-8"))
    wwtp = json.loads((REPORTS / "wwtp_source_nonoverlap_audit.json").read_text(encoding="utf-8"))
    search = json.loads((REPORTS / "legacy_literature_search_audit.json").read_text(encoding="utf-8"))
    fulltext = json.loads((REPORTS / "open_access_fulltext_audit.json").read_text(encoding="utf-8"))
    recovery = pd.read_parquet(OUT / "synthetic_recovery_summary.parquet")
    noiseless = pd.read_parquet(OUT / "synthetic_noiseless_recovery.parquet")
    signature = pd.read_parquet(OUT / "source_location_signature_metrics.parquet")
    mass = pd.read_parquet(OUT / "operator_mass_balance_audit.parquet")
    spin = pd.read_parquet(OUT / "operator_spinup_audit.parquet")

    c = source["checks"]
    gate = decision["synthetic_recovery_gate"]
    practical = decision["practical_misselection"]
    sig = decision["source_location_signal"]
    lines = [
        "# 20260824_2：农业源区氮 Legacy 文献、数据与可识别性审计",
        "",
        "## 技术结论",
        "",
        "本阶段的正式状态是：",
        "",
        f"```text\n{decision['status']}\n```",
        "",
        "现有1961–2021农业氮账本足以构造一个质量守恒的**有效源区慢库假设**，但不足以在当前监测支持与合理噪声下识别“粪肥关联慢库”相对于“同质量generic慢库”的来源身份。因此，不应进入观察TN上的正式 `phi_M` 搜索；这不是否定农业SON Legacy，而是拒绝用当前数据为一个无法识别的source label背书。",
        "",
        "关键证据如下：",
        "",
        f"- 无噪声、完全同模型的profile可100%恢复结构、`phi_M`和六个运输记忆尺度，说明代码与方程不是代数退化；",
        f"- 加入预注册的0.20 log噪声后，各`phi_M`层的最低结构恢复率只有{gate['structure_recovery_rate_minimum']:.1%}，低于80%门槛；",
        f"- 最大median `|phi_hat-phi_true|`为{gate['median_phi_absolute_error_maximum']:.2f}，高于0.10门槛；",
        f"- manure与equal-mass generic的监测域log指纹中位相关系数为{sig['median_log_signature_correlation_phi_ge_0_25']:.6f}，归一化距离仅{sig['median_normalized_log_distance_phi_ge_0_25']:.4f}；",
        f"- source结构误选率为{practical['source_structure_misselection_rate']:.1%}，而`mu`误选率为{practical['transport_mu_misselection_rate']:.1%}。主要问题是source identity，而不是运输记忆普遍无法恢复。",
        "",
        "## 当前模型与本阶段边界",
        "",
        "当前TN主线仍是修复后的Q72耦合Parent：Q72 `structural_canonical main`提供全部quick/GW水量与河道流量；固定F00完成source–water接触；固定T1表示额外有效TN delivery memory；H1以",
        "",
        "$$H=\\frac{LW}{86400Q}$$",
        "",
        "表示河床/面积水力暴露。Andreadis只提供河宽，参考流量不进入模型。H1是暴露指标，不是严格停留时间。本阶段没有改变Q72、H1、F00、T1、六个`mu`、R0或readout，也没有读取任何TN值。",
        "",
        "第一阶段已经证明H1适合作为监测网络的时间过程Parent，但station-blind LOSO绝对空间skill未通过，LOTO又受8棵tree小样本影响。因此，即使未来Legacy结构获得时间证据，也必须单独证明空间外推；P2不能替代这个证明。",
        "",
        "## 农业源历史审计",
        "",
        f"账本覆盖230个Reach、1961–2021共{c['reach_year_rows']:,}个Reach-year和{c['reach_month_rows']:,}个Reach-month。主键无重复、五个组成项无缺失或负值，净盈余恒等式最大误差为{c['surplus_identity_max_abs_kg_n']:.3g} kg N，annual-to-month闭合误差为{c['annual_to_month_closure_max_abs_kg_n']:.3g} kg N。",
        "",
        "注册的净盈余为",
        "",
        "$$G=F+M+BNF+D,\qquad S=G-R,$$",
        "",
        "粪肥关联的有效正盈余定义为",
        "",
        "$$M^*=S^+\\frac{M}{G}\\quad(G>0).$$",
        "",
        "这个比例分配避免了把gross manure优先塞进净盈余，但它仍只是账本归因，不是分子示踪。全历史`M*`约占正盈余的"
        f"{c['manure_associated_fraction_full_history']:.1%}。其与总正盈余的年度Reach空间相关系数中位数达到{c['median_year_spatial_correlation']:.3f}，这正是来源身份难以识别的第一层原因。",
        "",
        "数据可以用于注册的effective source-zone sensitivity，但有五个硬边界：",
        "",
        "1. manure是cropland manure，不包括无法安全空间化的grazing manure；",
        "2. 它是重建产品，不是土地现场监测；",
        "3. 月输入仍严格是annual/12，最大月CV接近0，不能解释为真实施肥或矿化月份；",
        "4. 2021收获面积和沉降空间场沿用2020；",
        "5. `M*`是正净盈余的比例归因，不能声称追踪了真实粪肥N原子。",
        "",
        "## 文献真正支持什么",
        "",
        "### Legacy研究的核心是两个不同的储存—释放过程",
        "",
        "Van Meter与Basu（2015）的开放全文明确把源区SON衰减与地下水旅行时间分开：源区中过量SON按一阶过程矿化，矿质N进入可淋溶库；该source function再与地下水travel-time distribution卷积。由此，`额外TN lag`的合理物理解释是**未被月尺度水文响应直接表示的有效溶质输送记忆**，而不是第二份水量、也不是自动等于地下水年龄。",
        "",
        "后续ELEMeNT工作把主要注意力放在：",
        "",
        "- 历史净N盈余如何累积为土壤有机N；",
        "- 硝酸盐如何存于深层包气带和地下水；",
        "- 土壤生物地球化学延迟与地下水水文延迟如何分开；",
        "- 管理变化以后，河流负荷为何需要多年或数十年才能响应；",
        "- 不同source-zone位置和travel time如何改变治理收益。",
        "",
        "这与“给每一种源自由拟合一个lag”不同。多数高影响研究首先用总N盈余、土壤/地下水库存和长记录约束总Legacy，然后才讨论来源和管理差异。",
        "",
        "### 农田有机N缓慢矿化是可信机制，但不等于manure-only",
        "",
        "Sebilo等（2013）的长期15N试验提供了最强的独立证据之一：标记肥料施用30年后仍有12–15%留在土壤有机质中，8–12%已向地下水泄漏，并预计继续释放数十年。Van Meter与Basu据此把过量SON作为biogeochemical Legacy。Ilampooranan、Van Meter与Basu（2022）进一步指出，SON积累和矿化受作物残体、产量、土壤、气候和管理共同控制；相同SON量不保证相同矿化通量。",
        "",
        "因此，本项目增加一个固定12个月SON池作为低自由度挑战是科学上可辩护的；但把该池专门命名为‘粪肥矿化池’，并把12个月当成文献参数，则证据不足。更准确的名称应是`manure-associated effective SON12 allocation`。",
        "",
        "### 珠江相关性存在，但不是本地率定",
        "",
        "Liu等（2024）在Nature Sustainability中明确把珠江列入四个大流域的地下水N Legacy重建，并报告珠江仍处于积累阶段。这说明研究问题对珠江高度相关，但该结果是大尺度模型重建，不能替代本项目230 Reach、月尺度TN的本地结构验证。",
        "",
        "## 没有土地监测时如何验证",
        "",
        "仅用河流TN可以验证的是：某个预注册Legacy算子是否改善时间外预测、是否保留空间稳健性，以及其管理响应是否合理。它不能独立验证内部的manure fraction、SON stock或真实矿化时间。证据层应固定为：",
        "",
        "1. **预测证据**：严格temporal OOF、nested LOSO/LOTO、状态/月份残差和质量守恒；",
        "2. **独立过程证据**：重复土壤有机N清查、深层土壤/包气带硝酸盐、lysimeter或tile-drain通量、地下水井硝酸盐；",
        "3. **年龄与来源证据**：长期15N试验、硝酸盐δ15N/δ18O结合土地利用和粪肥指标、3H/3He、CFC或SF6等地下水年龄示踪；",
        "4. **管理干预证据**：有明确源削减日期的长期前后序列或多流域自然实验。",
        "",
        "在这些数据缺失时，不确定性必须通过source-product ensemble、结构ensemble、先验敏感性、block bootstrap和posterior predictive checks表达；但统计不确定性不能补救结构不可识别。当前最关键的不是换成Bayesian sampler，而是获得能让manure与total surplus空间或时间指纹真正分开的信息。",
        "",
        "## 合成可识别性结果",
        "",
        "四个basis结构均独立spin-up，所有24个`structure × mu`组合收敛。最大相对质量平衡误差为"
        f"{mass.max_relative_mass_balance_error.max():.3g}，最小状态/通量为{mass.minimum_state_or_flux_kg_n.min():.3g} kg N。post-bypass generic与manure慢库逐月流域质量匹配误差不超过{c['post_bypass_mass_match_max_abs_kg_n']:.3g} kg N。",
        "",
        "无噪声时，真正的MANURE候选在24/24个生成组合中恢复，说明候选在无限精度下可分。但在0.20 log噪声、3895个锁定OOF支持键和每候选独立重估`eta_quick/eta_gw`的注册实验中，结果为：",
        "",
        "| true phi_M | structure recovery | median absolute phi error | mu recovery |",
        "|---:|---:|---:|---:|",
    ]
    for row in recovery.itertuples(index=False):
        lines.append(f"| {row.generating_phi_M:.2f} | {row.structure_recovery_rate:.1%} | {row.median_phi_absolute_error:.2f} | {row.mu_recovery_rate:.1%} |")
    lines.extend([
        "",
        "失败的原因不是算子无效，而是信息几何：`M*`只占正盈余约12%，且其Reach空间分布与总盈余高度共线；经过水文接触、慢路径记忆、H1河网暴露和上游累积后，MANURE与generic的差异进一步缩小到远低于注册噪声的量级。",
        "",
        "## WWTP与水文来源复核",
        "",
        f"WWTP source non-overlap为`{wwtp['status']}`。旧diffuse账本的point-source TN全部缺失且不进入soil Legacy；正式WWTP产品有{wwtp['checks']['formal_wwtp_product_rows']:,}行、112个Reach、2006–2019三个TN情景，但上一轮科学裁决为`contradictory`，本阶段保持关闭。",
        "",
        f"水文来源审计为`{hydro['status']}`：所有水量字段来自Q72，`q_local=quick_release+gw_discharge`最大误差{hydro['q_local_identity_max_abs_mm']:.3g} mm，Andreadis reference discharge未出现，温度未使用。",
        "",
        "## 正式裁决与后续边界",
        "",
        "本阶段不授权 `FORMAL_SOURCE_SELECTIVE_RETENTION_EXPERIMENT`，也不读取观察TN来‘试试看’。程序在以下终止状态停止：",
        "",
        f"```text\n{decision['status']}\n```",
        "",
        "这意味着：",
        "",
        "- 保留当前Q72–F00–T1–H1 TN主线；",
        "- 保留农业SON Legacy作为科学上可信但当前source identity未识别的解释；",
        "- 不拟合`phi_M`，不增加第二土壤池、温度、SAS、更多`mu`或空间随机`phi_M`；",
        "- 不把generic慢库的可能改善解释成manure evidence；",
        "- 只有获得独立manure application/soil-N/vadose-zone/groundwater证据，或形成与total surplus明显不共线的新源产品后，才能注册新的来源识别实验。",
        "",
        "如果未来新增数据，最小重启条件应是：先在不读TN的条件下把本合成门重新跑到结构恢复率≥80%且median `phi`误差≤0.10，然后才允许观察TN进入正式OOF。",
        "",
        "## 文献证据表",
        "",
        "下列文献均已用CrossRef DOI元数据核验；Van Meter与Basu（2015）另通过PLOS开放JATS全文核验。",
        "",
        "| 年份 | 文献 | 证据角色 | 对本项目的直接含义 |",
        "|---:|---|---|---|",
    ])
    for row in evidence.sort_values(["year", "doi"]).itertuples(index=False):
        lines.append(f"| {int(row.year)} | [{row.title}](https://doi.org/{row.doi}) | {row.evidence_role} | {row.prb_model_lesson} |")
    lines.extend([
        "",
        "## 检索与证据边界",
        "",
        f"CrossRef多主题检索去重后保存{search['unique_records']}条记录，其中{search['records_with_abstract']}条含publisher-deposited abstract；15条核心DOI全部核验成功。PLOS开放全文验证状态为`{fulltext['status']}`。CrossRef引用数仅作检索排序，不用于科学裁决。中文数据库未自动检索，全文未获得的论文只使用已核验元数据和出版社提交摘要，不虚构方法细节。",
        "",
        "本报告没有修改任何父实验，也没有读取2022 TN。",
    ])
    (REPORTS / "nitrogen_legacy_literature_and_modeling_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stage = {
        "experiment_id": "20260824_2",
        "status": decision["status"],
        "formal_source_selective_retention_experiment_authorized": False,
        "source_history_status": source["status"],
        "synthetic_identifiability_status": decision["status"],
        "hydrologic_provenance_status": hydro["status"],
        "wwtp_source_nonoverlap_status": wwtp["status"],
        "literature_search_status": search["status"],
        "open_access_fulltext_status": fulltext["status"],
        "next_folder_created": False,
        "TN_values_read": False,
        "TN_2022_values_read": False,
        "stop_reason": "The registered manure-associated source identity is practically unidentifiable against the equal-mass generic control at current monitoring support.",
    }
    (REPORTS / "stage2_decision.json").write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
