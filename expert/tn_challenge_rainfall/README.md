# 旧专家完整案例对应的逐日降雨

本包补齐 [PR #2 纯文本专家案例](https://github.com/moleculetang/sparrow-hydrology-tn/tree/32a1a94c34fa34722b8f64d858ad5ae0212865de/expert/tn_challenge_plain) 的降雨输入：北江终端56及东江终端22的两个完整上游连通分量，共39河段，1961-01-01至2024-12-31共23,376天、911,664个降雨值。案例仍在上述冻结提交；PR #10本次只补充对应降雨，不复制或改写旧案例模型。

## 文件与读取

- `data/precipitation_1961.csv` 至 `precipitation_2024.csv`：普通文本年度分片，首列`date`，其余39列`reach_<global ID>`。列顺序严格沿用原案例`topology.json/global_reach_ids`，并非全局编号升序。
- `case_scope.json`：完整日期、全局河段ID、本地索引、所属终端、案例原面积及列顺序。案例数组索引为零基，案例观测中的本地河段编号为一基；两者不可与全局ID混用。
- `annual_totals.csv`：逐年、逐河段降雨深度总和（mm）及日数，用于回算核查。
- `manifest.json`：各CSV的SHA256、形状、原驱动与原生成器哈希、浮点数组逻辑哈希及来源网页。
- `verification.json`：真实原文件/案例对照的提取回执；`verify.py`为随包可运行复核器。

在仓库根目录，用既有`sparrow`环境执行：

```sh
python -B expert/tn_challenge_rainfall/verify.py
```

`verify.load_rainfall()`返回`dates, global_reach_ids, precipitation`，最后一项为float64的`[23376,39]`数组，可与原案例日驱动按日期、列顺序直接对应。CSV采用17位有效数字；pandas读取时使用`float_precision='round_trip'`以精确保留原float64值。无需ZIP、NPZ或Git LFS。

## 来源、单位和范围

原产品为CHM_PRE V2逐日降雨，通过当时冻结的格网交叠权重聚合为**局地增量汇水区平均降雨深度**。单位为mm/day，作为日时间步的累计深度使用；不是流量、不是上游累计降雨质量，也不是沿站点位置插值得到的雨量。原生成器直接采用CHM日历日期，本包不平移日期、不重采样、不修正雨量。

提取来自Stage29冻结`daily_hbv_forcing_1961_2024.parquet`，并与案例实际使用的Stage38 sensitivity驱动1961—2024前缀逐值核对。两者降雨完全一致；原案例`fast_water`和`area_ha`也逐值一致。2025未发布，Stage38的2025 PET桥接限制不被本次降雨子集认证消除。上游格网缺测的覆盖处理沿用原生成器，本次没有重新检查全部NetCDF；因此“子集无缺失”不等于原格网无缺测。

这里保留**旧案例当时的空间与水文身份**，不换用后来`20260915 v3`统一包，不按面积比缩放。本包不能证明当前44条施肥时序路径的新S1来源已与H1完成全链空间统一，也没有更改这些路径的训练输入。

来源网页：[CHM_PRE产品入口](https://www.tpdc.ac.cn/zh-hans/data/e5c335d9-cbb9-48a6-ba35-d67dd614bb8c)。用户明确授权本案例范围的降雨供专家审阅；底层第三方资料许可仍按源产品规定，本次不重新授予许可。

原TN模型以冻结水文响应驱动库存、输运和浓度，不直接把降雨作为独立待拟合参数。新增文件用于专家核查、复算或另行登记模型改造，不会自动改变原案例的预测。验证量为文件与物理驱动一致性，NSE不适用；没有启动新校准或读取训练/评价TN。
