# TN实验专家诊断报告

生成时间：2026-09-09T17:40:04.715841+00:00。实验20260910_1，终态`STOPPED_DIAGNOSTIC_GATE`。

## 1. 实际工作与结论边界

已完成拟合0条；最多48条。统计数据集4个。过程诊断门、数值收敛、精度门槛及交付完成分别判定。正式主线没有替换。

冻结水文、S1源、全域共享参数映射、矿质M的lifetime/legacy与慢水L、既有河网水库；每个参数点从1961年重演。κ改变L释放，kL是L有效净损失；无第三氮库、无输出校正。

新增logκ界±log4/先验SD log2；kL界0–2 yr^-1/半正态SD0.5 yr^-1。原RAW站点归一化平方损失加原MAP先验，新增惩罚按站数归一化一次。MAP不是完整后验。

## 2. 工作方法与复核规则

拟合路径及运行证据（缺失字段显示未定义，完整求解器/检查点证据见reports/fits与work）：

没有满足条件的数据；见缺项说明。

求解器沿用有界TRF及解析梯度精修，两个预设起点；最低训练目标须通过原数值可靠性门。开发集只用于配置选择，确认失败不尝试次优者。残差定义为预测−实测，正值为高估。库存关联按同一参数点、年月、reach连接；仅是局地过程代理，不能当作站点总上游归因。

主评价始终保留原始全部观测。逐站NSE恒等式独立复核；低/高TN以训练q25/q75定义，少于12条本站训练记录使用训练汇总阈值并标记。流量阈值为训练观测月份水量分位，缺本站训练支持保留未定义。

季节相位使用常数+年度正余弦最小二乘；至少8条、8个不同月份、非退化振幅且谐波解释比例≥0.1才解释相位误差。该稳定性标记不改模型或评价门槛。

Spearman关联分原始、站内中心化、训练站点—月份异常；异常需至少2个训练参考月。静态属性按站级偏差/绝对误差，动态单位是站月。lag固定0/1/3/12个月，无搜索。上游属性是无标签面积支持汇总，不等于严格的氮来源贡献。

95%区间用seed1729、1000次水系内分层的站点整段配对重采样。全站时间段一起抽取；同时保存逐水系敏感性。不把这些区间当新流域独立不确定性。

## 3. 定量偏差与输入关系

### D0_M0_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_absolute_bias | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 2606 | 116 | 116 | -0.678117 | -1.91097 | 0.096328 | 0.387931 | 0.7648 | 0.508405 | 0.240925 | 0.960403 |

表现较差站点（完整站表另附）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 官渡 | 24 | -22.6084 | 0.275436 | -1.6744 | 0.423554 |
| 厂房大桥 | 24 | -18.5766 | -0.102876 | 1.65458 | 0.874044 |
| 赵家渡 | 23 | -18.0461 | -0.25818 | -2.33497 | 0.263393 |
| 八角电站 | 19 | -13.9495 | -0.00892932 | 1.03053 | 0.658689 |
| 青岩 | 19 | -13.117 | -0.0749528 | -1.83192 | 0.213751 |
| 兴宁电站 | 24 | -12.64 | -0.0263212 | -1.95 | 0.227431 |

分层偏差（浓度阈值只取训练集）：

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 630 | 116 | -0.863639 | 1.01147 | 0.941134 |
| concentration_band | low | 625 | 116 | 0.302706 | 0.595173 | 0.553523 |
| concentration_band | middle | 1351 | 116 | -0.187041 | 0.593381 | 0.547014 |
| season | DJF | 654 | 116 | -0.312643 | 0.701023 | 0.64106 |
| season | JJA | 630 | 116 | -0.31368 | 0.829195 | 0.729218 |
| season | MAM | 692 | 116 | -0.160777 | 0.691777 | 0.600572 |
| season | SON | 630 | 116 | -0.142993 | 0.674145 | 0.607627 |
| year | 2021 | 1391 | 116 | -0.216271 | 0.793848 | 0.674593 |
| year | 2022 | 1215 | 116 | -0.244907 | 0.701387 | 0.602215 |
| flow_band | high | 660 | 116 | -0.477552 | 0.812986 | 0.715818 |
| flow_band | low | 660 | 116 | -0.157529 | 0.724779 | 0.634274 |
| flow_band | middle | 1286 | 116 | -0.142 | 0.693882 | 0.611409 |

方差/平方误差贡献较大的观测，仅用于定位，不自动删除：

| station_key | year | month | tn_mg_l | prediction_mg_l | residual_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|---|
| 底先 | 2021 | 7 | 2.19 | 0.961981 | -1.22802 | 0.811536 | 0.625952 |
| 武林渡口 | 2022 | 6 | 4.4 | 1.76549 | -2.63451 | 0.717675 | 0.622783 |
| 龙归 | 2021 | 3 | 5.29 | 0.611343 | -4.67866 | 0.68764 | 0.627303 |
| 八角电站 | 2021 | 11 | 1.97 | 2.03365 | 0.063653 | 0.60802 | 0.000181384 |
| 锁龙桥 | 2021 | 6 | 9.03 | 4.25867 | -4.77133 | 0.566362 | 0.489844 |
| 小云尚大桥 | 2022 | 10 | 1.22 | 2.70351 | 1.48351 | 0.553982 | 0.0954946 |

输入关联摘要，属于描述统计且可能受季节及共同驱动混杂：

| feature | adjustment | target | spearman | units | unit |
|---|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | absolute_error | -0.413039 | 116 | station |
| upstream_log_dem_slope__static | between_station | absolute_error | 0.343562 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | absolute_error | -0.317054 | 116 | station |
| upstream_glhymps_porosity__static | between_station | absolute_error | -0.311488 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | absolute_error | 0.306077 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | absolute_error | 0.285741 | 116 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | absolute_error | -0.283405 | 116 | station |
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | signed_residual | -0.275899 | 2428 | station_month |

对应reach的局地库存/通量与残差的描述关联（不等于全上游来源贡献或站点边界端元）：
| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | signed_residual | 0.270898 | 2428 |
| local_reach_river_official_kg_n | training_station_month_anomaly | signed_residual | -0.262382 | 2428 |
| local_reach_fast_kg_n | training_station_month_anomaly | signed_residual | -0.256914 | 2428 |
| local_reach_river_official_kg_n | within_station | signed_residual | -0.231063 | 2606 |
| local_reach_river_inlet_kg_n | within_station | signed_residual | -0.206331 | 2606 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | signed_residual | -0.201788 | 2428 |
| local_reach_fast_kg_n | within_station | signed_residual | -0.177181 | 2606 |
| local_reach_river_channel_removed_kg_n | within_station | signed_residual | 0.161562 | 2606 |
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | signed_residual | 0.126871 | 2428 |
| local_reach_demand_kg_n | raw | signed_residual | -0.106261 | 2606 |
| local_reach_uptake_kg_n | raw | signed_residual | -0.106261 | 2606 |
| local_reach_fast_kg_n | raw | signed_residual | -0.104374 | 2606 |

完整表：E:\SPARROW\5_Test\20260910_1\outputs\expert\D0_M0_training_stations.parquet；关联、分层、相位、OOD与共线性同前缀。

### D0_M0_development

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_absolute_bias | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 1090 | 116 | 78 | -0.989531 | -2.79064 | 0.094452 | 0.397436 | 0.843382 | 0.568872 | 0.253139 | 0.953846 |

表现较差站点（完整站表另附）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 杨民 | 12 | -53.4228 | -0.161702 | -1.00895 | 0.543929 |
| 厂房大桥 | 12 | -36.2519 | -0.142658 | 2.74897 | 1.28986 |
| 赵家渡 | 12 | -18.6435 | -0.572057 | -2.73756 | 0.229294 |
| 庙咀里 | 12 | -17.8826 | -0.423111 | -1.06421 | 0.324753 |
| 滃江大站 | 12 | -17.7385 | -0.331835 | -1.98163 | 0.178231 |
| 青岩 | 8 | -13.48 | -0.0278041 | -1.82858 | 0.171583 |

分层偏差（浓度阈值只取训练集）：

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 297 | 93 | -0.869234 | 1.07556 | 1.01291 |
| concentration_band | low | 310 | 95 | 0.514295 | 0.701447 | 0.667267 |
| concentration_band | middle | 483 | 106 | -0.0771491 | 0.610989 | 0.569962 |
| season | DJF | 271 | 116 | -0.213272 | 0.683512 | 0.656729 |
| season | JJA | 273 | 116 | -0.141107 | 0.904201 | 0.84346 |
| season | MAM | 272 | 116 | 0.0764171 | 0.772531 | 0.731764 |
| season | SON | 274 | 116 | -0.171957 | 0.698246 | 0.663122 |
| year | 2023 | 1090 | 116 | -0.11282 | 0.843382 | 0.722192 |
| flow_band | high | 203 | 88 | -0.721183 | 1.04047 | 1.00279 |
| flow_band | low | 507 | 109 | -0.0694881 | 0.798105 | 0.728133 |
| flow_band | middle | 380 | 115 | -0.200189 | 0.727051 | 0.675491 |

方差/平方误差贡献较大的观测，仅用于定位，不自动删除：

| station_key | year | month | tn_mg_l | prediction_mg_l | residual_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|---|
| 禄丰村 | 2023 | 8 | 7.9 | 5.63607 | -2.26393 | 0.751227 | 0.0508253 |
| 布龙 | 2023 | 7 | 0.33 | 1.00212 | 0.672122 | 0.748666 | 0.515331 |
| 乌都河 | 2023 | 7 | 3.44 | 1.84169 | -1.59831 | 0.732061 | 0.938249 |
| 思留口 | 2023 | 10 | 2.52 | 2.09588 | -0.424122 | 0.729388 | 0.0251877 |
| 甲洋 | 2023 | 7 | 1.54 | 1.25884 | -0.281156 | 0.727015 | 0.103034 |
| 上洞 | 2023 | 7 | 6.43 | 1.75344 | -4.67656 | 0.721508 | 0.968686 |

输入关联摘要，属于描述统计且可能受季节及共同驱动混杂：

| feature | adjustment | target | spearman | units | unit |
|---|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | absolute_error | -0.454143 | 116 | station |
| upstream_log_dem_slope__static | between_station | absolute_error | 0.423734 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | absolute_error | -0.386853 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | absolute_error | 0.380096 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | absolute_error | -0.378994 | 116 | station |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.378893 | 1090 | station_month |
| upstream_glhymps_porosity__static | between_station | absolute_error | -0.357995 | 116 | station |
| local_log_dem_slope__static | between_station | absolute_error | 0.352357 | 116 | station |

对应reach的局地库存/通量与残差的描述关联（不等于全上游来源贡献或站点边界端元）：
| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| local_reach_river_official_kg_n | training_station_month_anomaly | signed_residual | -0.339231 | 1065 |
| local_reach_fast_kg_n | training_station_month_anomaly | signed_residual | -0.305193 | 1065 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | signed_residual | -0.28125 | 1065 |
| local_reach_river_official_kg_n | within_station | signed_residual | -0.270872 | 1090 |
| local_reach_river_inlet_kg_n | within_station | signed_residual | -0.250516 | 1090 |
| local_reach_river_channel_removed_kg_n | within_station | signed_residual | 0.227498 | 1090 |
| local_reach_fast_kg_n | raw | signed_residual | -0.220384 | 1090 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | signed_residual | 0.210913 | 1065 |
| local_reach_mineral_M_ending_kg_n | within_station | signed_residual | 0.200017 | 1090 |
| local_reach_injected_kg_n | training_station_month_anomaly | signed_residual | -0.180922 | 1065 |
| local_reach_fast_kg_n | within_station | signed_residual | -0.149461 | 1090 |
| local_reach_mineral_loss_kg_n | within_station | signed_residual | 0.144617 | 1090 |

完整表：E:\SPARROW\5_Test\20260910_1\outputs\expert\D0_M0_development_stations.parquet；关联、分层、相位、OOD与共线性同前缀。

### D0_MG_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_absolute_bias | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 2606 | 116 | 116 | -0.796932 | -2.17522 | 0.0919626 | 0.37931 | 0.754395 | 0.496683 | 0.234441 | 0.93674 |

表现较差站点（完整站表另附）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 官渡 | 24 | -19.3116 | 0.174861 | -1.54317 | 0.479611 |
| 青岩 | 19 | -13.7439 | -0.0641907 | -1.87634 | 0.198077 |
| 赵家渡 | 23 | -13.1962 | -0.165081 | -1.99409 | 0.294942 |
| 厂房大桥 | 24 | -12.1564 | -0.222158 | 1.31494 | 0.811756 |
| 八角电站 | 19 | -12.0171 | -0.0536254 | 0.951622 | 0.656878 |
| 谷拉河大桥 | 24 | -10.4939 | -0.212513 | 0.519368 | 0.427311 |

分层偏差（浓度阈值只取训练集）：

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 630 | 116 | -0.851638 | 0.991384 | 0.922152 |
| concentration_band | low | 625 | 116 | 0.326789 | 0.600336 | 0.562726 |
| concentration_band | middle | 1351 | 116 | -0.173292 | 0.57808 | 0.528949 |
| season | DJF | 654 | 116 | -0.294822 | 0.70037 | 0.638864 |
| season | JJA | 630 | 116 | -0.304725 | 0.806065 | 0.707771 |
| season | MAM | 692 | 116 | -0.117623 | 0.686055 | 0.598626 |
| season | SON | 630 | 116 | -0.151457 | 0.662677 | 0.592241 |
| year | 2021 | 1391 | 116 | -0.197999 | 0.789254 | 0.670529 |
| year | 2022 | 1215 | 116 | -0.231142 | 0.684286 | 0.584755 |
| flow_band | high | 660 | 116 | -0.46789 | 0.789978 | 0.691952 |
| flow_band | low | 660 | 116 | -0.121507 | 0.736296 | 0.648651 |
| flow_band | middle | 1286 | 116 | -0.13316 | 0.678206 | 0.594285 |

方差/平方误差贡献较大的观测，仅用于定位，不自动删除：

| station_key | year | month | tn_mg_l | prediction_mg_l | residual_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|---|
| 底先 | 2021 | 7 | 2.19 | 1.11915 | -1.07085 | 0.811536 | 0.332835 |
| 武林渡口 | 2022 | 6 | 4.4 | 1.91616 | -2.48384 | 0.717675 | 0.69552 |
| 龙归 | 2021 | 3 | 5.29 | 0.733201 | -4.5568 | 0.68764 | 0.677391 |
| 八角电站 | 2021 | 11 | 1.97 | 1.93453 | -0.0354659 | 0.60802 | 6.46688e-05 |
| 锁龙桥 | 2021 | 6 | 9.03 | 4.16 | -4.87 | 0.566362 | 0.416275 |
| 小云尚大桥 | 2022 | 10 | 1.22 | 2.28469 | 1.06469 | 0.553982 | 0.0327317 |

输入关联摘要，属于描述统计且可能受季节及共同驱动混杂：

| feature | adjustment | target | spearman | units | unit |
|---|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | absolute_error | -0.526791 | 116 | station |
| upstream_log_dem_slope__static | between_station | absolute_error | 0.475954 | 116 | station |
| upstream_glhymps_porosity__static | between_station | absolute_error | -0.453372 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | absolute_error | 0.44743 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | absolute_error | -0.429353 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | absolute_error | -0.404518 | 116 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | absolute_error | -0.399148 | 116 | station |
| local_log_dem_slope__static | between_station | absolute_error | 0.369553 | 116 | station |

本数据集没有独立保存的同参数库存账本；不借用其他模型库存解释其残差。

完整表：E:\SPARROW\5_Test\20260910_1\outputs\expert\D0_MG_training_stations.parquet；关联、分层、相位、OOD与共线性同前缀。

### D0_MG_development

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_absolute_bias | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 1090 | 116 | 78 | -0.912167 | -2.75395 | 0.0294428 | 0.487179 | 0.847447 | 0.545879 | 0.249434 | 0.997733 |

表现较差站点（完整站表另附）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 杨民 | 12 | -40.3877 | -0.285389 | -0.871668 | 0.687959 |
| 罗村口 | 12 | -29.1328 | 0.0207792 | 0.561543 | 1.10696 |
| 厂房大桥 | 12 | -28.9557 | -0.218846 | 2.41946 | 1.36087 |
| 谷拉河大桥 | 12 | -18.0661 | -0.499212 | 0.749876 | 0.646495 |
| 滃江大站 | 12 | -15.9512 | -0.311483 | -1.87839 | 0.183162 |
| 赵家渡 | 12 | -14.0905 | -0.655398 | -2.35603 | 0.316377 |

分层偏差（浓度阈值只取训练集）：

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 297 | 93 | -0.854177 | 1.05104 | 0.994446 |
| concentration_band | low | 310 | 95 | 0.560997 | 0.739399 | 0.702653 |
| concentration_band | middle | 483 | 106 | -0.0428836 | 0.595932 | 0.555594 |
| season | DJF | 271 | 116 | -0.183348 | 0.684496 | 0.659489 |
| season | JJA | 273 | 116 | -0.0963046 | 0.911647 | 0.8488 |
| season | MAM | 272 | 116 | 0.143945 | 0.799488 | 0.753296 |
| season | SON | 274 | 116 | -0.162372 | 0.668294 | 0.635998 |
| year | 2023 | 1090 | 116 | -0.0748673 | 0.847447 | 0.723154 |
| flow_band | high | 203 | 88 | -0.723876 | 1.00576 | 0.969157 |
| flow_band | low | 507 | 109 | -0.00517812 | 0.834698 | 0.76032 |
| flow_band | middle | 380 | 115 | -0.184163 | 0.699447 | 0.650727 |

方差/平方误差贡献较大的观测，仅用于定位，不自动删除：

| station_key | year | month | tn_mg_l | prediction_mg_l | residual_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|---|
| 禄丰村 | 2023 | 8 | 7.9 | 5.93353 | -1.96647 | 0.751227 | 0.0254156 |
| 布龙 | 2023 | 7 | 0.33 | 0.995094 | 0.665094 | 0.748666 | 0.501149 |
| 乌都河 | 2023 | 7 | 3.44 | 1.62067 | -1.81933 | 0.732061 | 0.872858 |
| 思留口 | 2023 | 10 | 2.52 | 1.65824 | -0.861763 | 0.729388 | 0.207427 |
| 甲洋 | 2023 | 7 | 1.54 | 1.1619 | -0.378104 | 0.727015 | 0.212151 |
| 上洞 | 2023 | 7 | 6.43 | 1.77891 | -4.65109 | 0.721508 | 0.966237 |

输入关联摘要，属于描述统计且可能受季节及共同驱动混杂：

| feature | adjustment | target | spearman | units | unit |
|---|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | absolute_error | -0.482634 | 116 | station |
| upstream_log_dem_slope__static | between_station | absolute_error | 0.465389 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | absolute_error | -0.461113 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | absolute_error | -0.433944 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | absolute_error | 0.422363 | 116 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | absolute_error | -0.420086 | 116 | station |
| upstream_glhymps_porosity__static | between_station | absolute_error | -0.406138 | 116 | station |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.396437 | 1090 | station_month |

本数据集没有独立保存的同参数库存账本；不借用其他模型库存解释其残差。

完整表：E:\SPARROW\5_Test\20260910_1\outputs\expert\D0_MG_development_stations.parquet；关联、分层、相位、OOD与共线性同前缀。

## 4. 状态、路径、敏感度与错误定位

### D0

训练内慢库准入：False

| kappa | loss_year | affected_stations | response_supported | centered_sse | centered_sse_gain | dynamic_evidence | low_flow_bias_gain | loss_evidence | passed |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | 113 | True | 1.18338 | -0.0941838 | False | -0.0907488 | False | False |
| 0.5 | 0.25 | 115 | True | 1.07276 | 0.00809176 | False | 0.0220065 | False | False |
| 0.5 | 1 | 116 | True | 1.15701 | -0.0698093 | False | -0.449525 | False | False |
| 1 | 0.25 | 115 | True | 1.06095 | 0.0190181 | False | 0.019949 | False | False |
| 1 | 1 | 116 | True | 1.12893 | -0.043838 | False | -0.237523 | False | False |
| 2 | 0 | 114 | True | 1.20442 | -0.11364 | False | -0.0157707 | False | False |
| 2 | 0.25 | 115 | True | 1.2213 | -0.129248 | False | -0.0586659 | False | False |
| 2 | 1 | 115 | True | 1.29047 | -0.193202 | False | -0.243923 | False | False |

物理参数方向数465，共享方向数21；K×G相对误差3.10739e-09。

K局部线性残差投影比例：0.7509803649920974；J对应比例：0.19342722405636292。这些不是可部署自由参数拟合或全局可识别性证明。

慢水主导低流期：{'qualified_stations': 106, 'median_standardized_bias': 0.07285877631346507, 'positive_bias_fraction': 0.5094339622641509, 'supports_excess_slow_concentration': False, 'baseline_mean_absolute_standardized_bias': 1.2058878563286781, 'bias_reduction_statistic': 'mean absolute station standardized bias', 'flow_threshold_support': 'training observation months at each registered station'}

过程状态摘要：{'contact_mapping_saturation_fraction': 0.0, 'lifetime_mapping_saturation_fraction': 0.0, 'hazard_cap_fraction': 0.0, 'modifier_minimum': 1.0, 'modifier_maximum': 1.0, 'log_alpha_minimum': -8.561716923216473, 'log_alpha_maximum': -6.609495762144716, 'lifetime_days_minimum': 935.4257058901082, 'lifetime_days_maximum': 1592.067826376418, 'kappa_slow': 1.0, 'loss_slow_per_year': 0.0, 'mineral_empty_fraction': 0.0, 'slow_empty_fraction': 1.6181590929381304e-05, 'uptake_clears_supply_fraction': 0.0}

全历史质量平衡：-3.814697265625e-06 kg N；路径标签检查：PASS_BOUNDARY_PATH_TAGS

## 5. 配对不确定性与水系差异

D0_M0_training → D0_MG_training

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.412697 | 0.0967098 | 1000 |
| q25_nse | candidate_minus_baseline | -0.707937 | 1.1013 | 1000 |
| median_r | candidate_minus_baseline | -0.0474901 | 0.0372955 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0431034 | 0.0258621 | 1000 |
| log_rmse | candidate_minus_baseline | -0.0133343 | 0.00025368 | 1000 |
| rmse | candidate_minus_baseline | -0.0333685 | 0.0119912 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0449428 | 0.0228854 | 1000 |

| terminal_tree_id | stations | median_nse | q25_nse | median_r | negative_r_fraction | log_rmse | rmse | absolute_bias |
|---|---|---|---|---|---|---|---|---|
| 20 | 93 | -0.130328 | -0.25761 | 0.0218042 | -0.0107527 | -0.00341111 | -0.00201934 | -0.00442778 |
| 22 | 5 | -0.968835 | 2.01441 | 0.000659028 | 0 | -0.0424547 | -0.0851266 | -0.112081 |
| 23 | 1 | -1.07622 | -1.07622 | -0.0629589 | 0 | 0.100432 | 0.272099 | 0.391542 |
| 26 | 1 | 1.94069 | 1.94069 | 0.0372348 | 0 | -0.0611205 | -0.229465 | -0.253058 |
| 56 | 12 | -0.116551 | 1.42039 | 0.0166128 | 0 | -0.0262263 | -0.0612735 | -0.0533937 |
| 212 | 2 | -0.176877 | -0.401903 | -0.0498426 | 0 | 0.00549817 | 0.0107934 | -0.00419212 |
| 217 | 2 | -0.504028 | -0.567887 | 0.00532469 | 0 | 0.0208514 | 0.0387241 | 0.0615601 |

D0_M0_development → D0_MG_development

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.24211 | 0.255028 | 1000 |
| q25_nse | candidate_minus_baseline | -1.70114 | 1.01787 | 1000 |
| median_r | candidate_minus_baseline | -0.147814 | -0.0199995 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.0263158 | 0.152964 | 1000 |
| log_rmse | candidate_minus_baseline | -0.0122055 | 0.00489581 | 1000 |
| rmse | candidate_minus_baseline | -0.026553 | 0.0383287 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0639344 | 0.0201416 | 1000 |

| terminal_tree_id | stations | median_nse | q25_nse | median_r | negative_r_fraction | log_rmse | rmse | absolute_bias |
|---|---|---|---|---|---|---|---|---|
| 20 | 93 | -0.0889812 | 0.664402 | -0.115037 | 0.112903 | 0.000880044 | 0.0181858 | -0.016186 |
| 22 | 5 | 2.33631 | 2.51975 | 0.0512251 | 0 | -0.0468342 | -0.0915039 | -0.104208 |
| 23 | 1 | -1.7622 | -1.7622 | -0.0147176 | 1 | 0.109389 | 0.263652 | 0.385746 |
| 26 | 1 | 未定义 | 未定义 | 未定义 | 0 | -0.0734528 | -0.272584 | -0.280328 |
| 56 | 12 | 0.866052 | 0.691436 | 0.097844 | 0 | -0.0252765 | -0.061394 | -0.0566018 |
| 212 | 2 | -0.221492 | -0.333481 | -0.0402388 | -0.5 | 0.00629231 | 0.0143471 | 0.00157554 |
| 217 | 2 | -0.136181 | -0.136181 | -0.201363 | 0 | -0.0113557 | -0.0226309 | -0.0350574 |

## 6. 条件阶段、异常与限制

{'G2': 'D0 did not support L', 'CONFIRM': 'development absent', 'G3': 'confirmation absent'}

没有登记运行异常。

现有年份曾用于项目，均非新盲测。训练116站与2023动态78站不直接当相同队列；原始逐站计数、固定71/45/2站群统计保存在清单。2025有水文/PET延伸、源沿用和缺月；早期为本轮未拟合回报。

水文快/慢/direct标签来自冻结导出，氮标签从进入河网分配并在相同水库及站点边界传播；标签不是实测端元或真实水龄。零水浓度未定义。Cobs×Qsim只能叫等效负荷。月采样日和聚合口径未核实的部分不擅自修正。

## 7. 给专家的下一轮问题

请先依据原始验证集的均值、幅度、相位与相关是否同时改善评价收益；不能仅凭训练目标下降或删极值后的变化认定改善。

若D0未通过：本轮有限κ/kL范围未提供充分的慢库定位支持；请结合K/G响应、M清空、快慢端元与河网前后账本，区分源端摄取/动员、共享映射和下游边界的替代解释，而非推断所有两库存模型不可行。

若D0通过而开发/确认失败：机制可产生响应与能否泛化分开；请比较逐站反向、参数补偿、触界、幅度和高低流偏差。进一步试验必须针对这些证据，不能自动增加家族或调权。

若确认和空间流程完成：逐水系结果仍限定外推范围。TN有效净损失不能直接当反硝化率；建议的独立端元/氮形态/采样信息用于区分竞争解释，不预先证明任一解释。

## 8. 可复核材料

登记方法：E:\SPARROW\5_Test\20260910_1\PLAN.md；完整统计清单：E:\SPARROW\5_Test\20260910_1\reports\expert_manifest.json。

逐条误差、站表、分层、谐波相位、关联、OOD、共线性、1000次抽样及水系差异均在outputs/expert。原始预测、处理账本、拟合状态和完整过程导出均保留，清单逐文件记录SHA256。
