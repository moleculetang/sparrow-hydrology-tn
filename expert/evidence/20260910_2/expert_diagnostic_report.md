# TN内部响应关系联合辨识：专家诊断报告

生成：2026-09-10T04:11:00.592347+00:00；实验20260910_2；状态COMPLETE_EXPERIMENTAL。

## 技术结论

全部六条拟合已执行；预测支持情况与数值可靠性分别如下。

training（116站、5885条）：NSE中位数 -0.614912 → -0.648313；相关中位数 0.142224 → 0.164499；注册条件 未满足全部条件。

hindcast（71站、3795条）：NSE中位数 -0.328027 → -0.400347；相关中位数 0.133481 → 0.075233；注册条件 未满足全部条件。

## 实际工作、数值可靠性与未执行项

| tag | status | numerical_adequate | train_objective | data_loss | prior_penalty | projected_gradient_max | value_calls | execution_seconds | resource_resumes |
|---|---|---|---|---|---|---|---|---|---|
| B0 | COMPLETE | True | 1.85649 | 1.43584 | 0.420651 | 7.43039e-07 | 1143 | 648.456 | 0 |
| B1 | COMPLETE | True | 1.85649 | 1.43584 | 0.42065 | 8.28092e-07 | 1059 | 601.17 | 0 |
| F0 | COMPLETE | True | 1.83662 | 1.40525 | 0.431369 | 3.68984e-08 | 107 | 76.1445 | 0 |
| F1 | COMPLETE | True | 1.83662 | 1.40525 | 0.431369 | 1.26888e-07 | 108 | 77.7955 | 0 |
| J0 | COMPLETE | True | 1.82414 | 1.43592 | 0.388222 | 6.99423e-07 | 968 | 662.525 | 0 |
| J1 | COMPLETE | True | 1.82414 | 1.43592 | 0.388222 | 7.3736e-07 | 1444 | 961.202 | 0 |

原参数固定组F无论是否改善均进入J；仅B不存在可靠基准会阻止依赖阶段。保存的最低可行点不自动等于收敛解。

## 内部响应曲面与两起点

![响应曲面](../outputs/expert/response_surfaces.png)

黑线为训练日、训练站对应reach的水文状态支持边界；范围外形状仅为方程外推，不是数据识别。图使用统一±log(4)尺度。

| group | weighted_log_multiplier_rms | supported_cells | supported_cell_correlation | two_starts_do_not_prove_identifiability |
|---|---|---|---|---|
| F | 1.82043e-08 | 138 | 1 | True |
| J | 1.54065e-07 | 138 | 1 | True |

## 四角回放与参数补偿

| corner | objective | data_loss | prior_penalty | centered_weighted_sse | physical_pass | M_final_kg | L_final_kg |
|---|---|---|---|---|---|---|---|
| B_zero | 1.85649 | 1.43584 | 0.420651 | 1.04879 | True | 6.00847e+09 | 5.69713e+07 |
| B_Jresponse | 1.86733 | 1.4258 | 0.441525 | 0.988098 | True | 5.72642e+09 | 6.04891e+07 |
| J_zero | 1.89249 | 1.52514 | 0.367347 | 1.15498 | True | 5.93445e+09 | 5.27793e+07 |
| J_Jresponse | 1.82414 | 1.43592 | 0.388222 | 1.04935 | True | 5.68925e+09 | 5.60873e+07 |

四角分别改变原参数和新增项；报告目标、原始数据项、先验及中心化误差。MAP不是完整后验，有限起点不是全局最优保证。

## 单站、时期、输入与内部过程偏差

### B1_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.614913 | -1.46376 | 0.142224 | 0.232759 | 0.832055 | 0.254376 | 0.921628 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.901632 | 1.09419 | 0.997717 |
| concentration_band | low | 1455 | 116 | 0.393299 | 0.691515 | 0.63807 |
| concentration_band | middle | 2967 | 116 | -0.126314 | 0.621983 | 0.565247 |
| season | DJF | 1410 | 116 | -0.254104 | 0.723693 | 0.652491 |
| season | JJA | 1476 | 116 | -0.305956 | 0.924863 | 0.787525 |
| season | MAM | 1528 | 116 | -0.0517122 | 0.778331 | 0.66717 |
| season | SON | 1471 | 116 | -0.160816 | 0.737886 | 0.651496 |
| year | 2021 | 1391 | 116 | -0.153352 | 0.785281 | 0.666407 |
| year | 2022 | 1215 | 116 | -0.149601 | 0.680234 | 0.584737 |
| year | 2023 | 1090 | 116 | -0.0730615 | 0.81121 | 0.699812 |
| year | 2024 | 1138 | 116 | -0.233893 | 0.853646 | 0.733778 |
| year | 2025 | 1051 | 116 | -0.344628 | 0.89056 | 0.765292 |
| flow_band | high | 1497 | 116 | -0.515961 | 0.913582 | 0.783972 |
| flow_band | low | 1497 | 116 | -0.0718774 | 0.796969 | 0.695243 |
| flow_band | middle | 2891 | 116 | -0.0849983 | 0.736765 | 0.638367 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.4844 | -0.23453 | -2.39079 | 0.339765 |
| 八角电站 | 31 | -18.4006 | 0.246566 | 1.0904 | 0.882082 |
| 官渡 | 36 | -15.4932 | -0.0545268 | -1.69552 | 0.367803 |
| 滃江大站 | 59 | -13.2028 | 0.186005 | -1.93452 | 0.292593 |
| 平而关 | 31 | -12.8697 | -0.0278874 | 0.692021 | 0.868083 |
| 杨民 | 59 | -12.619 | 0.00749299 | -0.966978 | 0.425829 |
| 厂房大桥 | 58 | -10.8013 | 0.0435305 | 2.5934 | 0.717723 |
| 庙咀里 | 59 | -10.4739 | 0.112419 | -1.1252 | 0.251723 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.618505 | 0.644702 | 0.57322 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.08554 | 0.476705 | 0.0851964 |
| 八角电站 | 2021 | 11 | 1.97 | 2.10539 | 0.456679 | 0.000462921 |
| 双苏村 | 2025 | 10 | 3.56 | 2.24343 | 0.445703 | 0.423138 |
| 底先 | 2024 | 10 | 2.49 | 0.9863 | 0.393227 | 0.274954 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.369159 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.325195 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.323277 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.323247 | 116 | station |
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | -0.306598 | 5778 | station_month |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.303975 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.297487 | 116 | station |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | -0.27931 | 5778 | station_month |
| upstream_glhymps_porosity__static | between_station | -0.273313 | 116 | station |
| local_log_dem_slope__static | between_station | 0.268917 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.294757 | 5778 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.285769 | 5778 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.284815 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.273162 | 5778 |
| local_reach_frozen_wetness_mean | within_station | -0.257489 | 5885 |
| local_reach_river_official_kg_n | within_station | -0.255738 | 5885 |
| local_reach_river_inlet_kg_n | within_station | -0.223727 | 5885 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.222356 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.214272 | 5778 |
| local_reach_frozen_wetness_mean | training_station_month_anomaly | -0.212499 | 5778 |

完整表前缀：outputs/expert/B1_training_；删可疑评价记录后的指标仅在reports/metrics/B1_training_supplementary.json补充。

### B1_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.389233 | -1.42978 | 0.109399 | 0.273973 | 0.800268 | 0.232631 | 0.929528 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -1.00513 | 1.2601 | 1.05519 |
| concentration_band | low | 1304 | 73 | 0.367339 | 0.601504 | 0.547344 |
| concentration_band | middle | 1772 | 73 | -0.154643 | 0.535288 | 0.480337 |
| season | DJF | 971 | 73 | -0.209719 | 0.714363 | 0.605395 |
| season | JJA | 962 | 73 | -0.207818 | 0.80251 | 0.680482 |
| season | MAM | 961 | 73 | -0.0410915 | 0.713401 | 0.607982 |
| season | SON | 972 | 73 | -0.120738 | 0.760131 | 0.568324 |
| year | 2016 | 734 | 66 | -0.136038 | 0.708861 | 0.59602 |
| year | 2017 | 782 | 73 | -0.242345 | 0.824406 | 0.628343 |
| year | 2018 | 791 | 66 | -0.20763 | 0.689812 | 0.581512 |
| year | 2019 | 778 | 66 | -0.151838 | 0.65007 | 0.547598 |
| year | 2020 | 781 | 66 | -0.177328 | 0.730324 | 0.620463 |
| flow_band | high | 1287 | 71 | -0.300895 | 0.903226 | 0.679103 |
| flow_band | low | 358 | 64 | -0.138048 | 0.789291 | 0.703241 |
| flow_band | middle | 2150 | 71 | -0.0895426 | 0.706558 | 0.587303 |
| flow_band | no_training_threshold | 71 | 2 | -0.194967 | 0.419375 | 0.320449 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 兴宁电站 | 59 | -14.0842 | -0.125752 | -1.67647 | 0.270012 |
| 五马岗 | 21 | -13.5644 | 0.214097 | 0.875099 | 0.696894 |
| 马头福水 | 21 | -12.3616 | 0.358765 | 0.872637 | 0.604784 |
| 爽底坪 | 21 | -10.1939 | 0.187115 | 0.843348 | 0.437174 |
| 厂房大桥 | 39 | -7.61519 | 0.0567649 | 2.01703 | 0.617355 |
| 十里亭 | 60 | -6.49239 | 0.177602 | 0.995973 | 0.64898 |
| 杨民 | 60 | -5.09738 | 0.188675 | -0.624869 | 0.39474 |
| 三溪桥 | 58 | -4.8566 | 0.0151665 | 0.976179 | 0.538725 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.04286 | 0.970543 | 0.986545 |
| 界牌 | 2016 | 9 | 4.38 | 1.14281 | 0.679474 | 0.657611 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.10048 | 0.522775 | 0.416396 |
| 扒齿 | 2016 | 9 | 3.59 | 2.09535 | 0.5128 | 0.159842 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.68275 | 0.463755 | 0.00306402 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.503896 | 73 | station |
| local_log_dem_slope__static | between_station | 0.425163 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.419038 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.415339 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.397408 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.346312 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.32766 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.305434 | 73 | station |
| downstream_fraction__static | between_station | 0.299858 | 73 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.233178 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.238613 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.236449 | 3784 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.236206 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.233225 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.21043 | 3866 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.16193 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.156483 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.124312 | 3784 |
| local_reach_demand_kg_n | raw | -0.121523 | 3866 |
| local_reach_uptake_kg_n | raw | -0.121523 | 3866 |

完整表前缀：outputs/expert/B1_hindcast_；删可疑评价记录后的指标仅在reports/metrics/B1_hindcast_supplementary.json补充。

### B0_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.614912 | -1.46376 | 0.142224 | 0.232759 | 0.832055 | 0.254376 | 0.921629 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.901632 | 1.09419 | 0.997717 |
| concentration_band | low | 1455 | 116 | 0.393299 | 0.691515 | 0.63807 |
| concentration_band | middle | 2967 | 116 | -0.126314 | 0.621983 | 0.565247 |
| season | DJF | 1410 | 116 | -0.254104 | 0.723693 | 0.652491 |
| season | JJA | 1476 | 116 | -0.305956 | 0.924863 | 0.787525 |
| season | MAM | 1528 | 116 | -0.0517123 | 0.778331 | 0.66717 |
| season | SON | 1471 | 116 | -0.160816 | 0.737886 | 0.651496 |
| year | 2021 | 1391 | 116 | -0.153352 | 0.785281 | 0.666407 |
| year | 2022 | 1215 | 116 | -0.149601 | 0.680234 | 0.584737 |
| year | 2023 | 1090 | 116 | -0.0730616 | 0.81121 | 0.699812 |
| year | 2024 | 1138 | 116 | -0.233893 | 0.853646 | 0.733778 |
| year | 2025 | 1051 | 116 | -0.344628 | 0.89056 | 0.765292 |
| flow_band | high | 1497 | 116 | -0.51596 | 0.913582 | 0.783972 |
| flow_band | low | 1497 | 116 | -0.0718776 | 0.796969 | 0.695243 |
| flow_band | middle | 2891 | 116 | -0.0849983 | 0.736765 | 0.638367 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.4844 | -0.23453 | -2.39079 | 0.339765 |
| 八角电站 | 31 | -18.4006 | 0.246567 | 1.0904 | 0.882082 |
| 官渡 | 36 | -15.4932 | -0.054527 | -1.69552 | 0.367803 |
| 滃江大站 | 59 | -13.2028 | 0.186005 | -1.93452 | 0.292593 |
| 平而关 | 31 | -12.8697 | -0.0278872 | 0.69202 | 0.868082 |
| 杨民 | 59 | -12.619 | 0.00749303 | -0.966978 | 0.425829 |
| 厂房大桥 | 58 | -10.8013 | 0.0435314 | 2.5934 | 0.717722 |
| 庙咀里 | 59 | -10.4739 | 0.112419 | -1.1252 | 0.251723 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.618505 | 0.644702 | 0.57322 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.08554 | 0.476705 | 0.0851964 |
| 八角电站 | 2021 | 11 | 1.97 | 2.10539 | 0.456679 | 0.000462921 |
| 双苏村 | 2025 | 10 | 3.56 | 2.24343 | 0.445703 | 0.423138 |
| 底先 | 2024 | 10 | 2.49 | 0.9863 | 0.393227 | 0.274954 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.369159 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.325195 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.323277 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.323247 | 116 | station |
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | -0.306598 | 5778 | station_month |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.303975 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.297487 | 116 | station |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | -0.279309 | 5778 | station_month |
| upstream_glhymps_porosity__static | between_station | -0.273313 | 116 | station |
| local_log_dem_slope__static | between_station | 0.268917 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.294756 | 5778 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.285769 | 5778 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.284815 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.273162 | 5778 |
| local_reach_frozen_wetness_mean | within_station | -0.257489 | 5885 |
| local_reach_river_official_kg_n | within_station | -0.255738 | 5885 |
| local_reach_river_inlet_kg_n | within_station | -0.223727 | 5885 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.222356 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.214272 | 5778 |
| local_reach_frozen_wetness_mean | training_station_month_anomaly | -0.212498 | 5778 |

完整表前缀：outputs/expert/B0_training_；删可疑评价记录后的指标仅在reports/metrics/B0_training_supplementary.json补充。

### B0_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.389233 | -1.42978 | 0.109399 | 0.273973 | 0.800268 | 0.232631 | 0.929528 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -1.00513 | 1.2601 | 1.05519 |
| concentration_band | low | 1304 | 73 | 0.367339 | 0.601505 | 0.547344 |
| concentration_band | middle | 1772 | 73 | -0.154643 | 0.535288 | 0.480337 |
| season | DJF | 971 | 73 | -0.209719 | 0.714363 | 0.605395 |
| season | JJA | 962 | 73 | -0.207818 | 0.802509 | 0.680482 |
| season | MAM | 961 | 73 | -0.0410914 | 0.713401 | 0.607982 |
| season | SON | 972 | 73 | -0.120737 | 0.760131 | 0.568324 |
| year | 2016 | 734 | 66 | -0.136038 | 0.708861 | 0.59602 |
| year | 2017 | 782 | 73 | -0.242344 | 0.824406 | 0.628343 |
| year | 2018 | 791 | 66 | -0.207629 | 0.689812 | 0.581512 |
| year | 2019 | 778 | 66 | -0.151838 | 0.65007 | 0.547598 |
| year | 2020 | 781 | 66 | -0.177328 | 0.730324 | 0.620463 |
| flow_band | high | 1287 | 71 | -0.300895 | 0.903226 | 0.679103 |
| flow_band | low | 358 | 64 | -0.138048 | 0.789291 | 0.703241 |
| flow_band | middle | 2150 | 71 | -0.0895425 | 0.706558 | 0.587303 |
| flow_band | no_training_threshold | 71 | 2 | -0.194967 | 0.419375 | 0.320449 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 兴宁电站 | 59 | -14.0842 | -0.125752 | -1.67647 | 0.270012 |
| 五马岗 | 21 | -13.5644 | 0.214097 | 0.875099 | 0.696894 |
| 马头福水 | 21 | -12.3616 | 0.358765 | 0.872636 | 0.604784 |
| 爽底坪 | 21 | -10.1939 | 0.187115 | 0.843348 | 0.437173 |
| 厂房大桥 | 39 | -7.61519 | 0.0567656 | 2.01703 | 0.617355 |
| 十里亭 | 60 | -6.49239 | 0.177602 | 0.995973 | 0.64898 |
| 杨民 | 60 | -5.09738 | 0.188675 | -0.624869 | 0.39474 |
| 三溪桥 | 58 | -4.8566 | 0.0151664 | 0.976179 | 0.538724 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.04286 | 0.970543 | 0.986545 |
| 界牌 | 2016 | 9 | 4.38 | 1.14281 | 0.679474 | 0.65761 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.10048 | 0.522775 | 0.416396 |
| 扒齿 | 2016 | 9 | 3.59 | 2.09535 | 0.5128 | 0.159842 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.68275 | 0.463755 | 0.00306402 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.503896 | 73 | station |
| local_log_dem_slope__static | between_station | 0.425163 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.419038 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.415339 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.397408 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.346312 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.32766 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.305434 | 73 | station |
| downstream_fraction__static | between_station | 0.299858 | 73 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.233178 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.238613 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.236449 | 3784 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.236206 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.233225 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.210429 | 3866 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.16193 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.156483 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.124312 | 3784 |
| local_reach_demand_kg_n | raw | -0.121523 | 3866 |
| local_reach_uptake_kg_n | raw | -0.121523 | 3866 |

完整表前缀：outputs/expert/B0_hindcast_；删可疑评价记录后的指标仅在reports/metrics/B0_hindcast_supplementary.json补充。

### F0_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.566176 | -1.46491 | 0.240669 | 0.206897 | 0.819868 | 0.249633 | 1.00593 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.881582 | 1.07443 | 0.981855 |
| concentration_band | low | 1455 | 116 | 0.387294 | 0.680531 | 0.629652 |
| concentration_band | middle | 2967 | 116 | -0.124991 | 0.61391 | 0.559758 |
| season | DJF | 1410 | 116 | -0.270137 | 0.719763 | 0.648427 |
| season | JJA | 1476 | 116 | -0.267841 | 0.910424 | 0.781048 |
| season | MAM | 1528 | 116 | -0.0667485 | 0.759744 | 0.650961 |
| season | SON | 1471 | 116 | -0.15136 | 0.727871 | 0.643262 |
| year | 2021 | 1391 | 116 | -0.169039 | 0.779089 | 0.661992 |
| year | 2022 | 1215 | 116 | -0.135834 | 0.677175 | 0.584906 |
| year | 2023 | 1090 | 116 | -0.0915954 | 0.808954 | 0.69835 |
| year | 2024 | 1138 | 116 | -0.213548 | 0.830118 | 0.71711 |
| year | 2025 | 1051 | 116 | -0.318933 | 0.870698 | 0.745081 |
| flow_band | high | 1497 | 116 | -0.441707 | 0.889596 | 0.763244 |
| flow_band | low | 1497 | 116 | -0.105097 | 0.786117 | 0.683537 |
| flow_band | middle | 2891 | 116 | -0.097729 | 0.735175 | 0.637178 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.369 | -0.275517 | -2.38566 | 0.28818 |
| 八角电站 | 31 | -19.0239 | 0.234848 | 1.11137 | 0.800992 |
| 官渡 | 36 | -15.5225 | -0.215081 | -1.69273 | 0.333946 |
| 滃江大站 | 59 | -13.1428 | 0.0737268 | -1.92603 | 0.268903 |
| 平而关 | 31 | -12.9842 | 0.10637 | 0.705401 | 0.782127 |
| 杨民 | 59 | -12.3646 | 0.064213 | -0.958964 | 0.424866 |
| 庙咀里 | 59 | -10.2947 | 0.0627411 | -1.11457 | 0.230952 |
| 厂房大桥 | 58 | -10.0117 | 0.20507 | 2.52227 | 0.719896 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.631469 | 0.644702 | 0.586548 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.01497 | 0.476705 | 0.0914734 |
| 八角电站 | 2021 | 11 | 1.97 | 2.11995 | 0.456679 | 0.000550117 |
| 双苏村 | 2025 | 10 | 3.56 | 2.32218 | 0.445703 | 0.397241 |
| 底先 | 2024 | 10 | 2.49 | 1.03309 | 0.393227 | 0.276734 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.393439 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.348617 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.34239 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.322461 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.321332 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.316021 | 116 | station |
| upstream_glhymps_porosity__static | between_station | -0.291709 | 116 | station |
| local_log_dem_slope__static | between_station | 0.289593 | 116 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.263152 | 116 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.245508 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.23775 | 5778 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.223956 | 5885 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.223821 | 5778 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.212712 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.2122 | 5778 |
| local_reach_river_official_kg_n | within_station | -0.202047 | 5885 |
| local_reach_frozen_wetness_mean | within_station | -0.189663 | 5885 |
| local_reach_river_inlet_kg_n | within_station | -0.184963 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.176053 | 5778 |
| local_reach_wetness_minus_antecedent_mean | raw | -0.15591 | 5885 |

完整表前缀：outputs/expert/F0_training_；删可疑评价记录后的指标仅在reports/metrics/F0_training_supplementary.json补充。

### F0_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.355086 | -1.39192 | 0.121439 | 0.260274 | 0.798388 | 0.23153 | 0.964212 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.982453 | 1.24482 | 1.04148 |
| concentration_band | low | 1304 | 73 | 0.389841 | 0.612349 | 0.559061 |
| concentration_band | middle | 1772 | 73 | -0.126245 | 0.534095 | 0.479191 |
| season | DJF | 971 | 73 | -0.18905 | 0.722517 | 0.611019 |
| season | JJA | 962 | 73 | -0.170869 | 0.793786 | 0.674009 |
| season | MAM | 961 | 73 | -0.0253282 | 0.71105 | 0.605069 |
| season | SON | 972 | 73 | -0.0928234 | 0.75845 | 0.568413 |
| year | 2016 | 734 | 66 | -0.0770202 | 0.713966 | 0.600747 |
| year | 2017 | 782 | 73 | -0.208346 | 0.816086 | 0.6209 |
| year | 2018 | 791 | 66 | -0.202931 | 0.68567 | 0.576614 |
| year | 2019 | 778 | 66 | -0.124453 | 0.641384 | 0.540804 |
| year | 2020 | 781 | 66 | -0.166864 | 0.718277 | 0.611482 |
| flow_band | high | 1287 | 71 | -0.234438 | 0.901016 | 0.677493 |
| flow_band | low | 358 | 64 | -0.154946 | 0.776661 | 0.691107 |
| flow_band | middle | 2150 | 71 | -0.0813067 | 0.709226 | 0.589417 |
| flow_band | no_training_threshold | 71 | 2 | -0.13956 | 0.39628 | 0.304468 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -17.2959 | 0.0356573 | 0.982254 | 0.701535 |
| 马头福水 | 21 | -14.7347 | 0.221124 | 0.949223 | 0.497262 |
| 兴宁电站 | 59 | -13.4167 | -0.139687 | -1.63529 | 0.271884 |
| 爽底坪 | 21 | -12.2088 | 0.163239 | 0.923703 | 0.361821 |
| 厂房大桥 | 39 | -7.21978 | 0.173956 | 1.98262 | 0.61318 |
| 十里亭 | 60 | -7.10479 | 0.267655 | 1.05359 | 0.62055 |
| 三溪桥 | 58 | -5.3867 | -0.0932827 | 1.0238 | 0.502223 |
| 杨民 | 60 | -4.59027 | 0.1961 | -0.593641 | 0.381283 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.10502 | 0.970543 | 0.986894 |
| 界牌 | 2016 | 9 | 4.38 | 1.20239 | 0.679474 | 0.633671 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.11315 | 0.522775 | 0.389034 |
| 扒齿 | 2016 | 9 | 3.59 | 2.13324 | 0.5128 | 0.124864 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.75182 | 0.463755 | 0.0026523 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.525933 | 73 | station |
| local_log_dem_slope__static | between_station | 0.442087 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.439978 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.407471 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.393113 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.34418 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.339415 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.307033 | 73 | station |
| downstream_fraction__static | between_station | 0.299334 | 73 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.245435 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.227707 | 3784 |
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.22654 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.225809 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.222809 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.192587 | 3866 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.166513 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.160398 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.133426 | 3784 |
| local_reach_slow_water_L_ending_kg_n | within_station | 0.118321 | 3866 |
| local_reach_uptake_kg_n | raw | -0.116765 | 3866 |

完整表前缀：outputs/expert/F0_hindcast_；删可疑评价记录后的指标仅在reports/metrics/F0_hindcast_supplementary.json补充。

### F1_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.566175 | -1.46491 | 0.240669 | 0.206897 | 0.819868 | 0.249633 | 1.00593 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.881582 | 1.07443 | 0.981855 |
| concentration_band | low | 1455 | 116 | 0.387294 | 0.680531 | 0.629652 |
| concentration_band | middle | 2967 | 116 | -0.124991 | 0.61391 | 0.559758 |
| season | DJF | 1410 | 116 | -0.270137 | 0.719763 | 0.648427 |
| season | JJA | 1476 | 116 | -0.267841 | 0.910424 | 0.781048 |
| season | MAM | 1528 | 116 | -0.0667485 | 0.759744 | 0.650961 |
| season | SON | 1471 | 116 | -0.15136 | 0.727871 | 0.643262 |
| year | 2021 | 1391 | 116 | -0.169039 | 0.779089 | 0.661992 |
| year | 2022 | 1215 | 116 | -0.135834 | 0.677175 | 0.584906 |
| year | 2023 | 1090 | 116 | -0.0915955 | 0.808954 | 0.69835 |
| year | 2024 | 1138 | 116 | -0.213548 | 0.830118 | 0.71711 |
| year | 2025 | 1051 | 116 | -0.318933 | 0.870698 | 0.745081 |
| flow_band | high | 1497 | 116 | -0.441707 | 0.889596 | 0.763244 |
| flow_band | low | 1497 | 116 | -0.105097 | 0.786117 | 0.683537 |
| flow_band | middle | 2891 | 116 | -0.0977291 | 0.735175 | 0.637178 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.369 | -0.275517 | -2.38566 | 0.28818 |
| 八角电站 | 31 | -19.0239 | 0.234848 | 1.11137 | 0.800992 |
| 官渡 | 36 | -15.5225 | -0.215081 | -1.69273 | 0.333946 |
| 滃江大站 | 59 | -13.1428 | 0.0737269 | -1.92603 | 0.268903 |
| 平而关 | 31 | -12.9842 | 0.10637 | 0.705401 | 0.782127 |
| 杨民 | 59 | -12.3646 | 0.064213 | -0.958964 | 0.424866 |
| 庙咀里 | 59 | -10.2947 | 0.0627411 | -1.11457 | 0.230952 |
| 厂房大桥 | 58 | -10.0117 | 0.20507 | 2.52227 | 0.719896 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.631469 | 0.644702 | 0.586548 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.01497 | 0.476705 | 0.0914734 |
| 八角电站 | 2021 | 11 | 1.97 | 2.11995 | 0.456679 | 0.000550117 |
| 双苏村 | 2025 | 10 | 3.56 | 2.32218 | 0.445703 | 0.397241 |
| 底先 | 2024 | 10 | 2.49 | 1.03309 | 0.393227 | 0.276734 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.393439 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.348617 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.34239 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.322461 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.321332 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.316021 | 116 | station |
| upstream_glhymps_porosity__static | between_station | -0.291709 | 116 | station |
| local_log_dem_slope__static | between_station | 0.289593 | 116 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.263152 | 116 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.245508 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.23775 | 5778 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.223956 | 5885 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.223821 | 5778 |
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.212712 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.2122 | 5778 |
| local_reach_river_official_kg_n | within_station | -0.202047 | 5885 |
| local_reach_frozen_wetness_mean | within_station | -0.189663 | 5885 |
| local_reach_river_inlet_kg_n | within_station | -0.184963 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.176053 | 5778 |
| local_reach_wetness_minus_antecedent_mean | raw | -0.15591 | 5885 |

完整表前缀：outputs/expert/F1_training_；删可疑评价记录后的指标仅在reports/metrics/F1_training_supplementary.json补充。

### F1_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.355086 | -1.39192 | 0.121439 | 0.260274 | 0.798388 | 0.23153 | 0.964212 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.982453 | 1.24482 | 1.04148 |
| concentration_band | low | 1304 | 73 | 0.389841 | 0.612349 | 0.559061 |
| concentration_band | middle | 1772 | 73 | -0.126245 | 0.534095 | 0.479191 |
| season | DJF | 971 | 73 | -0.18905 | 0.722517 | 0.611019 |
| season | JJA | 962 | 73 | -0.170869 | 0.793786 | 0.674009 |
| season | MAM | 961 | 73 | -0.0253282 | 0.71105 | 0.605069 |
| season | SON | 972 | 73 | -0.0928234 | 0.75845 | 0.568413 |
| year | 2016 | 734 | 66 | -0.0770202 | 0.713966 | 0.600747 |
| year | 2017 | 782 | 73 | -0.208347 | 0.816086 | 0.6209 |
| year | 2018 | 791 | 66 | -0.202931 | 0.68567 | 0.576614 |
| year | 2019 | 778 | 66 | -0.124453 | 0.641384 | 0.540804 |
| year | 2020 | 781 | 66 | -0.166864 | 0.718277 | 0.611482 |
| flow_band | high | 1287 | 71 | -0.234438 | 0.901016 | 0.677493 |
| flow_band | low | 358 | 64 | -0.154946 | 0.776661 | 0.691107 |
| flow_band | middle | 2150 | 71 | -0.0813068 | 0.709226 | 0.589417 |
| flow_band | no_training_threshold | 71 | 2 | -0.13956 | 0.39628 | 0.304468 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -17.2959 | 0.0356573 | 0.982254 | 0.701535 |
| 马头福水 | 21 | -14.7347 | 0.221124 | 0.949223 | 0.497262 |
| 兴宁电站 | 59 | -13.4167 | -0.139687 | -1.63529 | 0.271884 |
| 爽底坪 | 21 | -12.2088 | 0.163239 | 0.923702 | 0.361821 |
| 厂房大桥 | 39 | -7.21978 | 0.173956 | 1.98262 | 0.61318 |
| 十里亭 | 60 | -7.10479 | 0.267655 | 1.05359 | 0.62055 |
| 三溪桥 | 58 | -5.3867 | -0.0932826 | 1.0238 | 0.502223 |
| 杨民 | 60 | -4.59027 | 0.1961 | -0.593641 | 0.381283 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.10502 | 0.970543 | 0.986894 |
| 界牌 | 2016 | 9 | 4.38 | 1.20239 | 0.679474 | 0.633671 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.11315 | 0.522775 | 0.389034 |
| 扒齿 | 2016 | 9 | 3.59 | 2.13324 | 0.5128 | 0.124864 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.75182 | 0.463755 | 0.0026523 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.525933 | 73 | station |
| local_log_dem_slope__static | between_station | 0.442087 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.439978 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.407471 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.393113 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.34418 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.339415 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.307033 | 73 | station |
| downstream_fraction__static | between_station | 0.299334 | 73 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.245435 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.227707 | 3784 |
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.22654 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.225809 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.222809 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.192587 | 3866 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.166513 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.160398 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.133426 | 3784 |
| local_reach_slow_water_L_ending_kg_n | within_station | 0.118321 | 3866 |
| local_reach_uptake_kg_n | raw | -0.116765 | 3866 |

完整表前缀：outputs/expert/F1_hindcast_；删可疑评价记录后的指标仅在reports/metrics/F1_hindcast_supplementary.json补充。

### J0_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.648313 | -1.39692 | 0.1645 | 0.258621 | 0.834278 | 0.254065 | 0.992372 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.910955 | 1.10046 | 1.00441 |
| concentration_band | low | 1455 | 116 | 0.397716 | 0.692136 | 0.640324 |
| concentration_band | middle | 2967 | 116 | -0.125929 | 0.619575 | 0.56279 |
| season | DJF | 1410 | 116 | -0.255398 | 0.723051 | 0.650718 |
| season | JJA | 1476 | 116 | -0.307215 | 0.931435 | 0.794848 |
| season | MAM | 1528 | 116 | -0.0554989 | 0.783675 | 0.672301 |
| season | SON | 1471 | 116 | -0.158489 | 0.731282 | 0.644367 |
| year | 2021 | 1391 | 116 | -0.171052 | 0.784941 | 0.666113 |
| year | 2022 | 1215 | 116 | -0.142288 | 0.683492 | 0.587798 |
| year | 2023 | 1090 | 116 | -0.0887512 | 0.824604 | 0.709708 |
| year | 2024 | 1138 | 116 | -0.218117 | 0.849767 | 0.731394 |
| year | 2025 | 1051 | 116 | -0.334259 | 0.8941 | 0.763412 |
| flow_band | high | 1497 | 116 | -0.499372 | 0.915269 | 0.783764 |
| flow_band | low | 1497 | 116 | -0.0758195 | 0.798665 | 0.697176 |
| flow_band | middle | 2891 | 116 | -0.0937098 | 0.739032 | 0.639427 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.2597 | -0.168697 | -2.38166 | 0.305774 |
| 八角电站 | 31 | -18.2993 | 0.219626 | 1.08609 | 0.877256 |
| 官渡 | 36 | -16.0564 | -0.173054 | -1.72294 | 0.345784 |
| 滃江大站 | 59 | -13.4913 | 0.0973388 | -1.95225 | 0.274234 |
| 杨民 | 59 | -12.555 | -0.0255981 | -0.963827 | 0.413247 |
| 平而关 | 31 | -12.1902 | 0.102861 | 0.680495 | 0.832051 |
| 厂房大桥 | 58 | -10.6267 | 0.0860179 | 2.57485 | 0.743993 |
| 庙咀里 | 59 | -10.606 | 0.0122301 | -1.12988 | 0.239592 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.621106 | 0.644702 | 0.574848 |
| 新坪山桥 | 2025 | 4 | 3.32 | 1.94397 | 0.476705 | 0.102646 |
| 八角电站 | 2021 | 11 | 1.97 | 2.10956 | 0.456679 | 0.000494396 |
| 双苏村 | 2025 | 10 | 3.56 | 2.26757 | 0.445703 | 0.424452 |
| 底先 | 2024 | 10 | 2.49 | 1.02473 | 0.393227 | 0.269551 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.377765 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.333254 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.328039 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.322736 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.308794 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.304167 | 116 | station |
| upstream_glhymps_porosity__static | between_station | -0.28085 | 116 | station |
| local_log_dem_slope__static | between_station | 0.276526 | 116 | station |
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | -0.259215 | 5778 | station_month |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.252441 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.266796 | 5778 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.251795 | 5885 |
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.250855 | 5778 |
| local_reach_river_official_kg_n | within_station | -0.244501 | 5885 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.236959 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.228791 | 5778 |
| local_reach_river_inlet_kg_n | within_station | -0.218683 | 5885 |
| local_reach_frozen_wetness_mean | within_station | -0.216451 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.182551 | 5778 |
| local_reach_wetness_minus_antecedent_mean | raw | -0.179031 | 5885 |

完整表前缀：outputs/expert/J0_training_；删可疑评价记录后的指标仅在reports/metrics/J0_training_supplementary.json补充。

### J0_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.402112 | -1.42666 | 0.0752331 | 0.369863 | 0.804727 | 0.233805 | 0.878777 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.999839 | 1.26152 | 1.05687 |
| concentration_band | low | 1304 | 73 | 0.397822 | 0.62237 | 0.566148 |
| concentration_band | middle | 1772 | 73 | -0.131797 | 0.535791 | 0.478875 |
| season | DJF | 971 | 73 | -0.166414 | 0.72467 | 0.610499 |
| season | JJA | 962 | 73 | -0.210727 | 0.810488 | 0.688762 |
| season | MAM | 961 | 73 | -0.0104868 | 0.713314 | 0.607874 |
| season | SON | 972 | 73 | -0.0998382 | 0.759745 | 0.566907 |
| year | 2016 | 734 | 66 | -0.0818631 | 0.719927 | 0.606154 |
| year | 2017 | 782 | 73 | -0.211581 | 0.82144 | 0.624692 |
| year | 2018 | 791 | 66 | -0.206011 | 0.691977 | 0.579711 |
| year | 2019 | 778 | 66 | -0.125778 | 0.649468 | 0.546611 |
| year | 2020 | 781 | 66 | -0.16655 | 0.732552 | 0.621782 |
| flow_band | high | 1287 | 71 | -0.275408 | 0.915762 | 0.689437 |
| flow_band | low | 358 | 64 | -0.118451 | 0.782254 | 0.698586 |
| flow_band | middle | 2150 | 71 | -0.0678448 | 0.707935 | 0.58732 |
| flow_band | no_training_threshold | 71 | 2 | -0.151823 | 0.416754 | 0.316615 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -16.014 | 0.0354912 | 0.946861 | 0.630642 |
| 马头福水 | 21 | -14.1781 | 0.17255 | 0.928397 | 0.545044 |
| 兴宁电站 | 59 | -13.8344 | -0.143563 | -1.65921 | 0.300456 |
| 爽底坪 | 21 | -11.1835 | 0.158269 | 0.883506 | 0.383895 |
| 厂房大桥 | 39 | -7.33773 | 0.0330389 | 1.97854 | 0.590497 |
| 十里亭 | 60 | -6.81337 | 0.274035 | 1.03047 | 0.651667 |
| 三溪桥 | 58 | -5.34976 | -0.149111 | 1.00831 | 0.545585 |
| 杨民 | 60 | -4.63229 | 0.189531 | -0.594856 | 0.423954 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.06955 | 0.970543 | 0.986297 |
| 界牌 | 2016 | 9 | 4.38 | 1.2067 | 0.679474 | 0.61528 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.0751 | 0.522775 | 0.409913 |
| 扒齿 | 2016 | 9 | 3.59 | 2.10848 | 0.5128 | 0.128162 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.45115 | 0.463755 | 0.00539882 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.521199 | 73 | station |
| local_log_dem_slope__static | between_station | 0.440267 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.437251 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.404336 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.401421 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.355402 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.338725 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.31393 | 73 | station |
| downstream_fraction__static | between_station | 0.299611 | 73 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.24252 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.260532 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.234256 | 3866 |
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.224278 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.223079 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.219698 | 3784 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.176469 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.168027 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.142834 | 3784 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.135087 | 3866 |
| local_reach_uptake_kg_n | raw | -0.124631 | 3866 |

完整表前缀：outputs/expert/J0_hindcast_；删可疑评价记录后的指标仅在reports/metrics/J0_hindcast_supplementary.json补充。

### J1_training

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 5885 | 116 | 116 | -0.648313 | -1.39692 | 0.164499 | 0.258621 | 0.834278 | 0.254065 | 0.992372 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.910955 | 1.10046 | 1.00441 |
| concentration_band | low | 1455 | 116 | 0.397716 | 0.692136 | 0.640324 |
| concentration_band | middle | 2967 | 116 | -0.125929 | 0.619575 | 0.56279 |
| season | DJF | 1410 | 116 | -0.255398 | 0.723051 | 0.650718 |
| season | JJA | 1476 | 116 | -0.307215 | 0.931435 | 0.794848 |
| season | MAM | 1528 | 116 | -0.0554989 | 0.783675 | 0.672301 |
| season | SON | 1471 | 116 | -0.158489 | 0.731282 | 0.644367 |
| year | 2021 | 1391 | 116 | -0.171052 | 0.784941 | 0.666113 |
| year | 2022 | 1215 | 116 | -0.142288 | 0.683492 | 0.587798 |
| year | 2023 | 1090 | 116 | -0.0887512 | 0.824604 | 0.709708 |
| year | 2024 | 1138 | 116 | -0.218117 | 0.849767 | 0.731394 |
| year | 2025 | 1051 | 116 | -0.33426 | 0.8941 | 0.763413 |
| flow_band | high | 1497 | 116 | -0.499372 | 0.915269 | 0.783764 |
| flow_band | low | 1497 | 116 | -0.0758195 | 0.798665 | 0.697176 |
| flow_band | middle | 2891 | 116 | -0.0937098 | 0.739032 | 0.639427 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.2597 | -0.168696 | -2.38166 | 0.305774 |
| 八角电站 | 31 | -18.2993 | 0.219626 | 1.08609 | 0.877257 |
| 官渡 | 36 | -16.0564 | -0.173054 | -1.72294 | 0.345784 |
| 滃江大站 | 59 | -13.4913 | 0.097339 | -1.95225 | 0.274234 |
| 杨民 | 59 | -12.555 | -0.0255984 | -0.963827 | 0.413248 |
| 平而关 | 31 | -12.1902 | 0.102861 | 0.680495 | 0.832051 |
| 厂房大桥 | 58 | -10.6267 | 0.0860177 | 2.57485 | 0.743994 |
| 庙咀里 | 59 | -10.606 | 0.0122299 | -1.12988 | 0.239592 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.621106 | 0.644702 | 0.574848 |
| 新坪山桥 | 2025 | 4 | 3.32 | 1.94397 | 0.476705 | 0.102646 |
| 八角电站 | 2021 | 11 | 1.97 | 2.10956 | 0.456679 | 0.000494396 |
| 双苏村 | 2025 | 10 | 3.56 | 2.26757 | 0.445703 | 0.424452 |
| 底先 | 2024 | 10 | 2.49 | 1.02473 | 0.393227 | 0.269551 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.377765 | 116 | station |
| upstream_log_dem_slope__static | between_station | 0.333254 | 116 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.328039 | 116 | station |
| local_log_predev_annual_pet_mm__static | between_station | 0.322736 | 116 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.308794 | 116 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.304167 | 116 | station |
| upstream_glhymps_porosity__static | between_station | -0.28085 | 116 | station |
| local_log_dem_slope__static | between_station | 0.276526 | 116 | station |
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | -0.259215 | 5778 | station_month |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.252441 | 116 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.266797 | 5778 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.251795 | 5885 |
| local_reach_river_official_kg_n | training_station_month_anomaly | -0.250855 | 5778 |
| local_reach_river_official_kg_n | within_station | -0.244502 | 5885 |
| local_reach_mobilization_probability_mean | training_station_month_anomaly | -0.236959 | 5778 |
| local_reach_fast_kg_n | training_station_month_anomaly | -0.228791 | 5778 |
| local_reach_river_inlet_kg_n | within_station | -0.218684 | 5885 |
| local_reach_frozen_wetness_mean | within_station | -0.216451 | 5885 |
| local_reach_river_inlet_kg_n | training_station_month_anomaly | -0.182551 | 5778 |
| local_reach_wetness_minus_antecedent_mean | raw | -0.179031 | 5885 |

完整表前缀：outputs/expert/J1_training_；删可疑评价记录后的指标仅在reports/metrics/J1_training_supplementary.json补充。

### J1_hindcast

| rows | stations | defined_nse | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|---|---|---|---|
| 3866 | 73 | 73 | -0.402111 | -1.42666 | 0.0752329 | 0.369863 | 0.804727 | 0.233805 | 0.878776 |

残差定义为预测−实测；下表按站平均，正值为高估。浓度/流量分层阈值只使用2021–2025训练记录。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.99984 | 1.26152 | 1.05687 |
| concentration_band | low | 1304 | 73 | 0.397822 | 0.62237 | 0.566148 |
| concentration_band | middle | 1772 | 73 | -0.131797 | 0.535791 | 0.478875 |
| season | DJF | 971 | 73 | -0.166414 | 0.72467 | 0.610499 |
| season | JJA | 962 | 73 | -0.210727 | 0.810488 | 0.688762 |
| season | MAM | 961 | 73 | -0.0104868 | 0.713314 | 0.607874 |
| season | SON | 972 | 73 | -0.0998383 | 0.759745 | 0.566907 |
| year | 2016 | 734 | 66 | -0.0818632 | 0.719927 | 0.606154 |
| year | 2017 | 782 | 73 | -0.211581 | 0.82144 | 0.624692 |
| year | 2018 | 791 | 66 | -0.206011 | 0.691977 | 0.579711 |
| year | 2019 | 778 | 66 | -0.125778 | 0.649468 | 0.546611 |
| year | 2020 | 781 | 66 | -0.16655 | 0.732552 | 0.621782 |
| flow_band | high | 1287 | 71 | -0.275409 | 0.915762 | 0.689437 |
| flow_band | low | 358 | 64 | -0.118451 | 0.782254 | 0.698586 |
| flow_band | middle | 2150 | 71 | -0.0678447 | 0.707935 | 0.58732 |
| flow_band | no_training_threshold | 71 | 2 | -0.151823 | 0.416754 | 0.316615 |

误差较差的站点（全部逐站、逐年和队列结果保存在源表）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -16.014 | 0.0354913 | 0.946861 | 0.630641 |
| 马头福水 | 21 | -14.1781 | 0.17255 | 0.928397 | 0.545044 |
| 兴宁电站 | 59 | -13.8344 | -0.143563 | -1.65921 | 0.300456 |
| 爽底坪 | 21 | -11.1835 | 0.158269 | 0.883506 | 0.383896 |
| 厂房大桥 | 39 | -7.33773 | 0.0330386 | 1.97854 | 0.590497 |
| 十里亭 | 60 | -6.81336 | 0.274035 | 1.03047 | 0.651667 |
| 三溪桥 | 58 | -5.34976 | -0.149111 | 1.00831 | 0.545586 |
| 杨民 | 60 | -4.63229 | 0.189532 | -0.594856 | 0.423954 |

极值贡献诊断；以下记录未被自动删除或纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.06955 | 0.970543 | 0.986297 |
| 界牌 | 2016 | 9 | 4.38 | 1.2067 | 0.679474 | 0.61528 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.0751 | 0.522775 | 0.409913 |
| 扒齿 | 2016 | 9 | 3.59 | 2.10847 | 0.5128 | 0.128162 |
| 厂房大桥 | 2018 | 6 | 5.44 | 4.45115 | 0.463755 | 0.00539883 |

输入关联的较大绝对值如下。完整表同时列出正负、无关联/未定义和样本数；不能据此认定因果。

| feature | adjustment | spearman | units | unit |
|---|---|---|---|---|
| log_upstream_area_ha__static | between_station | -0.521199 | 73 | station |
| local_log_dem_slope__static | between_station | 0.440267 | 73 | station |
| upstream_log_dem_slope__static | between_station | 0.437251 | 73 | station |
| upstream_log_awc_0_200_mm__static | between_station | -0.404336 | 73 | station |
| upstream_log_predev_annual_precipitation_mm__static | between_station | -0.401421 | 73 | station |
| upstream_bulk_density_0_30_g_cm3__static | between_station | -0.355402 | 73 | station |
| upstream_glhymps_porosity__static | between_station | -0.338725 | 73 | station |
| upstream_glhymps_log10_permeability_m2__static | between_station | 0.31393 | 73 | station |
| downstream_fraction__static | between_station | 0.299611 | 73 | station |
| upstream_log_predev_annual_pet_mm__static | between_station | -0.24252 | 73 | station |

本地reach库存和通量的关联是局地代理，不是整个上游的独立归因：

| feature | adjustment | spearman | units |
|---|---|---|---|
| local_reach_river_channel_removed_kg_n | training_station_month_anomaly | 0.260532 | 3784 |
| local_reach_river_channel_removed_kg_n | within_station | 0.234256 | 3866 |
| local_reach_mineral_M_ending_kg_n | training_station_month_anomaly | 0.224278 | 3784 |
| local_reach_available_mean_kg_n | training_station_month_anomaly | 0.223078 | 3784 |
| local_reach_loss_kg_n | training_station_month_anomaly | 0.219698 | 3784 |
| local_reach_slow_water_L_ending_kg_n | training_station_month_anomaly | 0.176469 | 3784 |
| local_reach_slow_kg_n | training_station_month_anomaly | 0.168026 | 3784 |
| local_reach_injected_kg_n | training_station_month_anomaly | 0.142834 | 3784 |
| local_reach_wetness_minus_antecedent_mean | within_station | -0.135087 | 3866 |
| local_reach_uptake_kg_n | raw | -0.124631 | 3866 |

完整表前缀：outputs/expert/J1_hindcast_；删可疑评价记录后的指标仅在reports/metrics/J1_hindcast_supplementary.json补充。

## 配对改善、不确定性及门槛

### B0 → F0 / training / all
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | True |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | True |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0753054 | 0.161441 | 1000 |
| q25_nse | candidate_minus_baseline | -0.266825 | 0.248932 | 1000 |
| median_r | candidate_minus_baseline | 0.0312905 | 0.129762 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0775862 | 0.0258621 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00627648 | -0.00319547 | 1000 |
| rmse | candidate_minus_baseline | -0.0181289 | -0.00648202 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00849863 | 0.00752519 | 1000 |

### B0 → F0 / training / common_71
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | True |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | False |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0617189 | 0.215722 | 1000 |
| q25_nse | candidate_minus_baseline | -0.143252 | 0.270629 | 1000 |
| median_r | candidate_minus_baseline | 0.072207 | 0.163764 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0422535 | 0.0704225 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00825251 | -0.00431823 | 1000 |
| rmse | candidate_minus_baseline | -0.0246155 | -0.00839956 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0109701 | 0.0111965 | 1000 |

### B0 → F0 / hindcast / common_71
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | True |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0970489 | 0.148858 | 1000 |
| q25_nse | candidate_minus_baseline | -0.30224 | 0.318621 | 1000 |
| median_r | candidate_minus_baseline | -0.0412487 | 0.0513058 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0704225 | 0.0422535 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00314094 | 0.00147436 | 1000 |
| rmse | candidate_minus_baseline | -0.00861641 | 0.0064584 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0119362 | 0.0115322 | 1000 |

### B0 → F0 / hindcast / all
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | True |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0806416 | 0.187334 | 1000 |
| q25_nse | candidate_minus_baseline | -0.272144 | 0.399064 | 1000 |
| median_r | candidate_minus_baseline | -0.0356684 | 0.0513058 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0684932 | 0.0410959 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00338866 | 0.00124403 | 1000 |
| rmse | candidate_minus_baseline | -0.00934933 | 0.00574126 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0136026 | 0.0102933 | 1000 |

### B0 → J1 / training / all
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | False |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.101105 | 0.104318 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0544867 | 0.0938855 | 1000 |
| median_r | candidate_minus_baseline | -0.0193933 | 0.0427472 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0344828 | 0.0862069 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00113691 | 0.00052875 | 1000 |
| rmse | candidate_minus_baseline | -0.000730423 | 0.00537115 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00335144 | 0.00332794 | 1000 |

### B0 → J1 / training / common_71
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | False |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0932995 | 0.101146 | 1000 |
| q25_nse | candidate_minus_baseline | -0.110311 | 0.0878117 | 1000 |
| median_r | candidate_minus_baseline | -0.0242133 | 0.0518395 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.028169 | 0.15493 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00159125 | 0.000252515 | 1000 |
| rmse | candidate_minus_baseline | -0.00201773 | 0.00588366 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00475845 | 0.00245752 | 1000 |

### B0 → J1 / hindcast / common_71
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | False |
| negative_r_fraction_not_increased | False |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.140372 | 0.079644 | 1000 |
| q25_nse | candidate_minus_baseline | -0.264903 | 0.211963 | 1000 |
| median_r | candidate_minus_baseline | -0.0883481 | 0.00506485 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.0140845 | 0.183099 | 1000 |
| log_rmse | candidate_minus_baseline | -0.000487395 | 0.00301065 | 1000 |
| rmse | candidate_minus_baseline | -0.00048069 | 0.010033 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0107065 | 0.00405213 | 1000 |

### B0 → J1 / hindcast / all
| check | passed |
|---|---|
| median_nse_gain_010 | False |
| median_r_gain_005 | False |
| positive_median_r | True |
| q25_nse_decrease_at_most_005 | True |
| negative_r_fraction_not_increased | False |
| undefined_r_fraction_not_increased | True |
| mean_station_log_rmse_at_most_105_percent | True |
| mean_station_raw_rmse_at_most_105_percent | True |
| mean_station_absolute_bias_at_most_105_percent | True |
| annual_uptake_ratio_drop_at_most_010 | True |
| numerically_adequate | True |
| physical_audit_passed | True |
| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.112391 | 0.132073 | 1000 |
| q25_nse | candidate_minus_baseline | -0.264903 | 0.21209 | 1000 |
| median_r | candidate_minus_baseline | -0.0819097 | 0.00412017 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.0136986 | 0.178082 | 1000 |
| log_rmse | candidate_minus_baseline | -0.000613705 | 0.00298687 | 1000 |
| rmse | candidate_minus_baseline | -0.000742185 | 0.0098326 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0122166 | 0.00271874 | 1000 |

## 数据定义、模型方程和完整工作方法

# TN内部响应关系联合辨识：一个位置、六条拟合路径

登记：用户确认的2026-09-10计划。旧实验和正式主线只读；本轮全程conda sparrow。

## 研究与数据

不追改20260910_1：8组慢库、984次非拟合调用、0候选拟合，STOPPED_DIAGNOSTIC_GATE已独立审计。固定原参数敏感性不等于新增方程的联合校准能力。旧P确实完成21+6参数联合拟合，不将其重命名重做。

训练2021–2025：原始5885条、116站；回报2016–2020：3866条、73站。71共有、45近期新增、2早期独有（勒马、白额）分列。2023/2024是训练，早期曾被项目查看，不能称盲验证。2025为11个月，冻结水文/PET延伸与S1氮源沿用另标。所有配置使用sensitivity输入产品及相同训练权重/预处理。

冻结水文和S1源→原21个全域共享环境—物理参数→M lifetime/额外legacy、慢水氮L、河网/水库→原TN观测算子。每个参数点从1961完整演化；无拟合初始库存、第三库存、当地TN历史输入/偏置、输出修正、未知源、WWTP估算、NN或XGBoost重训。

## 唯一新方程

M动员风险lambda0乘m=exp(log(4)*tanh(sum(c_ij B_ij)/log(4)))。W为冻结soil_wetness；A为此前30天W均值（不含当天，1961前用首日W补齐）。B_ij=P_i(2W-1)P_j(2A-1)，P0=1,P1=x,P2=(3x²-1)/2；顺序(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)。无记忆长度搜索。

8个全域常数系数界[-1,1]、先验SD=0.25/sqrt(8)，先验残差除sqrt(nstation)一次；零系数恢复M0。保留原RAW站点归一化平方损失和原MAP先验、TRF三点差分+最多200次解析精修。无新增对M/L的直接函数依赖；原历史反馈、库存扣减、摄取上限及完整梯度保留。零载体水量零动员，m在[0.25,4]，原日步风险上限700不改。

## 六条主要拟合

B0/B1：21参数M0，原起点0/1。F0/F1：冻结选定B的21参数，拟合8系数，起点为零/按基顺序交替±先验SD。J0/J1：联合29参数，起点为B+F最佳可行系数/原起点1+零系数。按B→F→J每组两路径运行，不根据F收益阻止J。B无可靠解则数值终止；F不收敛仍可用其最低合法可行点进入J并标记数值不确定。

自由坐标投影梯度<=1e-5；原目标容差1e-8*(1+abs(lowest))。保存嵌套可行点，更差终点不叫机制失败。选择仅用训练目标，所有选择和参数哈希冻结后才评价早期TN。保留全部两起点结果。

四角无拟合回放：(thetaB,0),(thetaB,cJ),(thetaJ,0),(thetaJ,cJ)，分数据损失、先验、中心化误差及物理账本。若thetaJ零扩展优于B，则提示基准搜索不足，不新增第7条拟合。本比较不是完整参数剖面/似然置信区间。

## 判读与交付

完整原始观测为主。NSE中位数/下四分位、r中位数/负相关/未定义率、站均RMSE/logRMSE/绝对bias、幅度、相位、中心化误差；逐站逐年、71/45/2、高低TN/流、季节、极值贡献。阈值仅用训练，删极值仅补充。1000次seed1729按水系分层配对整站bootstrap及水系敏感性。

完成拟合后才判支持进一步验证：NSE中位数增加>=0.10、r增加>=0.05且>0、NSE下四分位下降<=0.05、负相关/未定义率不增、RMSE/logRMSE/绝对bias<=1.05倍、逐年全域摄取/需求比例下降<=0.10。训练和早期71站分开判读，不自动晋级。

专家报告reports/expert_diagnostic_report.md为必交：公式/旧P差异、证据支持与反证、样本数、6路径/收敛/预算、逐站统计、曲面与两起点稳定性、四角回放、输入关联/OOD/混杂、M/L与路径/河网账本、失败/替代解释、可复核源表及SHA256。预测改善不等于legacy可识别。旧K/G/J和984回放不重跑；本轮实际新预测必须做专家后处理。任何终态都报告缺项。

## 数值与运行合同

输入/训练ID无泄漏；训练worker只接收近期TN，预测接口无TN标签；改早期标签不影响拟合；未来水文不改变过去状态。零扩展回归、29参数完整历史差分、独立NumPy前向、边界/零水/零源/干湿/摄取、全历史及河网相对守恒1e-10、负值及精度沿用已审计容差。测试自由掩码、依赖、重复启动、检查点、累计预算、退出与报告篡改。

每worker一线程、每阶段至多两worker。新冷启动和完整计算实测峰×1.2预留；CPU或整机RAM90%停派发，RAM90%请求检查点退让，相关资源低于85%恢复。独立控制器正常静默，原生进程身份/PID创建时间/累计CPU/调用数可查；每小时最多一次模型巡检，资源采样不调用模型，不向外部聊天发消息。

每路径最多4000完整调用（含差分、解析精修和终核）、4小时累计执行；总24小时含等待，最后2小时报告审计。失败保留检查点，不改变方程/先验/门槛/起点；资源恢复累计预算不重置。独立completion_audit.json实际核对模型/预测/专家报告/final_report.md及哈希；科学止损不等于精度改善，数值或预算未完成不冒称目标全完成。到本轮六路径和审计结束，不自动开空间分支/大实验/替换主线。


补充统计定义：r为Pearson；NSE逐站计算并核验 r²−(sd_ratio−r)²−标准化bias²。每站至少8条且观测非恒定才有动态指标；常预测的相关保留未定义。logRMSE使用log1p(TN)。幅度误差为abs(log(sd_pred/sd_obs))，无波动不偷偷填零。相位来自固定一年周期谐波，至少8个不同月份、解释度>=0.1且非退化才定义。

输入关系分原始、站内中心化和训练站点—月份异常；固定0/1/3/12月滞后，不搜索最优。静态属性用站级误差。高低流来自冻结水文、按本站训练分位，早期无训练站不借用其他站流量阈值。关联和共线性仅为描述，不当因果；本地库存与上游输入分列。

1000次seed1729水系内配对整站bootstrap，整站时序保留，区间是这些已观测站的稳定性诊断，不能当新流域置信保证。全部队列、逐年指标及逐站变化见all_metrics和对应parquet；负相关/无定义分母均保留。

## 支持、反证和机制不确定性

已有诊断高TN对应局地可用M并未清空，原训练630条和2023年297条；湿润动态关联支持本轮有限假设，但河网相位和源输入错误仍可能解释误差。旧K/G/J只提供无先验无限幅线性方向，不是可实现精度。本轮没有重新运行旧984前向筛选。

旧P已联合拟合21+6参数，并包含即时湿润/快流/可用M及交互；本轮的增量是此前30天湿润历史及二次曲率。不能把其结果归因为旧P从未获得联合拟合机会。

预测改善不等于M寿命、L释放或源贡献被独立识别。高库存、强损失或相似预测下不同库存只构成机制不确定性证据；守恒通过不能证明库存现实。月TN不能直接识别日尺度迟滞。

2025水文/PET延伸、源量沿用、仅11个月观测；早期2016–2020曾被项目查看，为本轮不参与拟合的回报，非盲验证。无早期标签参与特征、阈值、参数及起点选择；Cobs×Qsim只能叫等效负荷。

## 给专家的下一步问题

请结合训练与早期配对结果，区分协同调整收益、基准搜索不足、响应曲面不足、时期/驱动变化及legacy不可识别。若训练改善但回报不改善，先解释支持域与源输入差异；若输出相近但库存差异大，按冲突选择土壤无机氮、年龄/反应示踪或氮形态约束，不能用土壤总氮直接替代M。此处不自动启动后续拟合。

## 文献、代码与复核入口

Raue et al. 2009 https://doi.org/10.1093/bioinformatics/btp358；Reichert et al. 2021 https://doi.org/10.1029/2020WR028400；Ammann et al. 2021 https://doi.org/10.1029/2020WR028311；Höge et al. 2022 https://doi.org/10.5194/hess-26-5085-2022；Huynh et al. 2025 https://doi.org/10.5194/hess-29-3589-2025；Sarrazin et al. 2022 https://doi.org/10.1029/2021WR031587。题录/摘要与全文阅读层级、pyPESTO/SMASH/timedeppar源码版本见provenance.json。

运行入口scripts/ir_controller.py；无新拟合统计入口scripts/ir_postprocess.py；原始源表和SHA256见expert_manifest.json；训练权重逐条账本见outputs/weights；收敛轨迹和已知可行点见reports/fits及work/replay、work/refinement。专家报告生成并不替代独立终态审计。

## 错误与未完成项

没有后处理错误。
