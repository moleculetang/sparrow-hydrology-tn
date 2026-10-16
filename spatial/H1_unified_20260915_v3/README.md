# 已确认的水文／旧水质统一空间包：20260915_2 v3

**此前最终统一版本是 `5_Test/20260915_2` 的 v3 包，底层增量汇水区来自 `5_Test/20260814_1/inputs/spatial/reach_catchments.shp`。** 水文和当时的水质支持确实已经统一；不是尚未做过统一。230个增量汇水区几何面积合计 **410866.1378872674 km²**，与当前冻结 H1 的230坐标最大面积差为 **9.094947017729282e-13 km²**，属于浮点末位差。

## 文件与使用

`H1_unified_spatial_files.zip` 无损保存原始 SHP、SHX、DBF、PRJ、CPG 五组件，以及 v3 的拓扑、河段登记、水库连接、面积对账和30米支持标签。解压后保持文件前缀与目录关系；原始文件没有重投影或重写。

| ZIP内路径 | 职责 |
|---|---|
| `vectors/reach_catchments.*` | 230个增量汇水区，`reach_id=1…230`；局地水文和来源的空间支持，不是累计上游范围。|
| `canonical/hydrology_model_topology_v3.json` | 冻结水文兼容模型入口，`downstream`键和值使用零起始索引。|
| `canonical/hydrology_edges_v3.parquet` | 216条有向连接，表内使用一始河段ID。|
| `canonical/hydrology_reach_registry_v3.parquet` | 河段、上下游、终端和连续水文面积。|
| `canonical/hydrology_reservoir_links_v3.parquet` | 13个共享水库的控制臂、共同下游和局地捕获份额；共享库存只释放一次。|
| `canonical/training_area_crosscheck.parquet` | 230河段连续几何与水文面积对账。|
| `canonical/hydrology_topology_edges_compatible_v3.csv` | 原模型兼容连接表。|
| `canonical/hydrology_support_labels_30m_v3.tif` | 与原网格对齐的多边形中心归属标签；不是重算D8流向或汇水区。|
| `evidence/`（ZIP外） | 原v3审计和本次面积继承复核。|

图层使用 WGS84 基准的自定义 Albers 等积投影，单位米，完整 WKT 以原 PRJ 为准；不能因投影名称为 `unknown` 将其重新指定为经纬度。面积继续使用连续多边形面积，不能以30米像元数替换水文面积。

历史 v3 同时确认源网格8930条交叠可复算，并统一站支持。原267个无方向像元与所选水文域交叠为零，属于另一边界引入的域外范围；该结论不认证域外真实排水，也不批准新子河段过程。原报告保留于 `evidence/topology_alignment_v3.md`。本公开包不含原始TN、站点读数或驱动时序；历史源权重和站支持的一致性以原审计为证据，不冒充本包重新完成其验收。

## 后续新来源为何又出现面积差

后续新S1来源处理选中了 [20260810_1修正几何](../PRB230_20260810/README.md)，合计414345.6372136284 km²，重新引入支持分叉。当前施肥时序实验继承了这条新来源支路，而 H1 仍继承上述统一水文面积。

因此必须区分“此前已经统一”和“后续新来源未遵循该统一支持”。此前统一结论有效；当前新来源全链统一尚未恢复。本次上传修正版本说明并提供原始空间证据，没有热改现役训练。后续应重新按已确认空间支持聚合受影响土地、沉降及农业来源，逐项验证质量、河段和站界身份，再登记独立运行；不能只按面积比缩放氮量或水量。既有结果保留其原输入身份。

## 独立复核

在具备 GeoPandas、Pandas、NumPy、NetworkX、PyArrow 和 Rasterio 的环境中运行：

```bash
python spatial/H1_unified_20260915_v3/verify.py
```

核验器直接读取发布ZIP并验证每个成员SHA256，不依赖本机原始SHP；检查230个唯一河段、正面积、逐河段面积一致、216条无环边、13水库和支持栅格全部河段标签。`manifest.json`记录完整字节身份。通过空间核验不等于真实TN改善；空间QA的NSE不适用。
