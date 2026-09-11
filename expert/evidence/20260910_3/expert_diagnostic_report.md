# TN先验预算与源到站传递：专家审计报告

## 技术结论

四条约束拟合与河网边界诊断均已交付；数据收益、预测收益与数值可靠性分开判读。

OLD_B→A_B，training/all：NSE中位数-0.614912→-0.596068；r中位数0.142224→0.160378；完整预测条件未全部满足。

OLD_B→A_B，hindcast/common_71：NSE中位数-0.328027→-0.315850；r中位数0.133481→0.131101；完整预测条件未全部满足。

OLD_J→A_J，training/all：NSE中位数-0.648313→-0.570214；r中位数0.164499→0.216155；完整预测条件未全部满足。

OLD_J→A_J，hindcast/common_71：NSE中位数-0.400347→-0.353114；r中位数0.075233→0.093848；完整预测条件未全部满足。

## 实际工作与约束数值可靠性

| tag | status | D | R | KKT | adequate | calls | seconds |
|---|---|---|---|---|---|---|---|
| AB0 | COMPLETE | 1.42539 | 0.431369 | 3.60645e-06 | True | 49 | 74.0547 |
| AB1 | COMPLETE | 1.42539 | 0.431369 | 3.47411e-06 | True | 61 | 92.4955 |
| AJ0 | COMPLETE | 1.39718 | 0.431369 | 2.68523e-06 | True | 69 | 115.996 |
| AJ1 | COMPLETE | 1.39718 | 0.431369 | 1.67443e-06 | True | 65 | 108.191 |

新B零扩展的D=1.4253909622，R=0.4313694506；本次未发现它优于选定J；此检查不证明全局最优，也不是新增拟合。

D为原RAW数据损失；R为原先验代价；选点按D并受R预算约束。KKT包括先验约束与参数边界乘子，不能用原MAP梯度替代。

## 数据—先验权衡与分组代价

AB0：相对旧B，D改变-0.01045309（-0.728%），R改变+0.01071891；相对已知F的数据损失差+0.02014468。

AB1：相对旧B，D改变-0.01045309（-0.728%），R改变+0.01071891；相对已知F的数据损失差+0.02014468。

AJ0：相对旧J，D改变-0.03873938（-2.698%），R改变+0.04314761；相对已知F的数据损失差-0.00806458。

AJ1：相对旧J，D改变-0.03873938（-2.698%），R改变+0.04314761；相对已知F的数据损失差-0.00806458。

![数据与先验](../outputs/expert/data_prior_tradeoff.png)

| tag | group | R |
|---|---|---|
| OLD_B0 | unpenalized | 0 |
| OLD_B0 | contact_exponent | 0.00642579 |
| OLD_B0 | mineral_lifetime | 0.016258 |
| OLD_B0 | contact_mapping | 0.0278754 |
| OLD_B0 | lifetime_mapping | 0.0184999 |
| OLD_B0 | old_dynamic | 0.350776 |
| OLD_B0 | path_partition | 0.000815572 |
| OLD_F0 | unpenalized | 0 |
| OLD_F0 | contact_exponent | 0.00642579 |
| OLD_F0 | mineral_lifetime | 0.016258 |
| OLD_F0 | contact_mapping | 0.0278754 |
| OLD_F0 | lifetime_mapping | 0.0184999 |
| OLD_F0 | old_dynamic | 0.350776 |
| OLD_F0 | path_partition | 0.000815572 |
| OLD_F0 | wet_response | 0.0107189 |
| OLD_J1 | unpenalized | 0 |
| OLD_J1 | contact_exponent | 0.00783901 |
| OLD_J1 | mineral_lifetime | 0.0149988 |
| OLD_J1 | contact_mapping | 0.0275987 |
| OLD_J1 | lifetime_mapping | 0.0197129 |
| OLD_J1 | old_dynamic | 0.295999 |
| OLD_J1 | path_partition | 0.00119882 |
| OLD_J1 | wet_response | 0.0208746 |
| AB0 | unpenalized | 0 |
| AB0 | contact_exponent | 0.00597953 |
| AB0 | mineral_lifetime | 0.0167421 |
| AB0 | contact_mapping | 0.0287425 |
| AB0 | lifetime_mapping | 0.0196207 |
| AB0 | old_dynamic | 0.359497 |
| AB0 | path_partition | 0.000787526 |
| AB1 | unpenalized | 0 |
| AB1 | contact_exponent | 0.00597893 |
| AB1 | mineral_lifetime | 0.0167437 |
| AB1 | contact_mapping | 0.0287416 |
| AB1 | lifetime_mapping | 0.0196162 |
| AB1 | old_dynamic | 0.359501 |
| AB1 | path_partition | 0.000787608 |
| AJ0 | unpenalized | 0 |
| AJ0 | contact_exponent | 0.00612894 |
| AJ0 | mineral_lifetime | 0.0170206 |
| AJ0 | contact_mapping | 0.031679 |
| AJ0 | lifetime_mapping | 0.0256978 |
| AJ0 | old_dynamic | 0.328554 |
| AJ0 | path_partition | 0.00101692 |
| AJ0 | wet_response | 0.0212719 |
| AJ1 | unpenalized | 0 |
| AJ1 | contact_exponent | 0.00612972 |
| AJ1 | mineral_lifetime | 0.0170211 |
| AJ1 | contact_mapping | 0.0316744 |
| AJ1 | lifetime_mapping | 0.0257027 |
| AJ1 | old_dynamic | 0.328553 |
| AJ1 | path_partition | 0.00101689 |
| AJ1 | wet_response | 0.0212719 |

以下为训练时期与训练站reach上的过程量；抵消分母为零时不定义，不能解释为零抵消。

| point | zero_carrier_fraction | old_log_hazard_mean | old_log_hazard_rms | old_log_hazard_valid | new_log_multiplier_mean | new_log_multiplier_rms | new_log_multiplier_valid | total_log_hazard_mean | total_log_hazard_rms | total_log_hazard_valid | cancellation_fraction_mean | cancellation_fraction_rms | cancellation_fraction_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OLD_B | 0.155809 | -8.4799 | 8.54331 | 149099 | 0 | 0 | 149099 | -8.4799 | 8.54331 | 149099 | 未定义 | 未定义 | 0 |
| OLD_F | 0.155809 | -8.4799 | 8.54331 | 149099 | 0.00895949 | 0.0551082 | 149099 | -8.47094 | 8.53988 | 149099 | 0 | 0 | 149099 |
| OLD_J | 0.155809 | -8.5002 | 8.55411 | 149099 | 0.0732625 | 0.125558 | 149099 | -8.42694 | 8.49087 | 149099 | 0.539742 | 0.639782 | 149099 |
| A_B | 0.155809 | -8.5007 | 8.56505 | 149099 | 0 | 0 | 149099 | -8.5007 | 8.56505 | 149099 | 0 | 0 | 149099 |
| A_J | 0.155809 | -8.58327 | 8.64123 | 149099 | 0.0669713 | 0.115939 | 149099 | -8.5163 | 8.58395 | 149099 | 0.520073 | 0.643616 | 149099 |

总先验预算相同不等于相同物理可信度；v_f与全局接触截距没有高斯惩罚。新旧部分的过程抵消及零载体水统计见process_effects.json，完整月/河段表见outputs/process_effects。

## 单站、输入、来源与河网偏差

### AB0_training

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 5885 | 116 | -0.596068 | -1.41983 | 0.160378 | 0.232759 | 0.828958 | 0.253221 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.898293 | 1.08942 | 0.993861 |
| concentration_band | low | 1455 | 116 | 0.394298 | 0.690018 | 0.637253 |
| concentration_band | middle | 2967 | 116 | -0.125473 | 0.618947 | 0.562883 |
| season | DJF | 1410 | 116 | -0.254635 | 0.722285 | 0.651306 |
| season | JJA | 1476 | 116 | -0.300192 | 0.920081 | 0.784008 |
| season | MAM | 1528 | 116 | -0.0518755 | 0.775485 | 0.66492 |
| season | SON | 1471 | 116 | -0.159869 | 0.734975 | 0.649103 |
| year | 2021 | 1391 | 116 | -0.154445 | 0.783443 | 0.664913 |
| year | 2022 | 1215 | 116 | -0.14788 | 0.67829 | 0.583455 |
| year | 2023 | 1090 | 116 | -0.0738244 | 0.809498 | 0.698558 |
| year | 2024 | 1138 | 116 | -0.230662 | 0.849217 | 0.730369 |
| year | 2025 | 1051 | 116 | -0.339599 | 0.885646 | 0.76096 |
| flow_band | high | 1497 | 116 | -0.504917 | 0.907608 | 0.778761 |
| flow_band | low | 1497 | 116 | -0.0727477 | 0.795574 | 0.693968 |
| flow_band | middle | 2891 | 116 | -0.0872067 | 0.735072 | 0.636927 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.4735 | -0.243743 | -2.39052 | 0.328727 |
| 八角电站 | 31 | -18.2406 | 0.248176 | 1.08674 | 0.852977 |
| 官渡 | 36 | -15.4687 | -0.0735496 | -1.69381 | 0.359191 |
| 滃江大站 | 59 | -13.1699 | 0.173929 | -1.9317 | 0.286369 |
| 平而关 | 31 | -12.8081 | -0.0180393 | 0.692017 | 0.842717 |
| 杨民 | 59 | -12.554 | 0.0231519 | -0.965186 | 0.418897 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | signed_residual | -0.300452 | 5778 |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.273523 | 5778 |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.254453 | 5885 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.219597 | 116 |
| current_mean_local_lower_release__lag0 | training_station_month_anomaly | signed_residual | -0.216122 | 5778 |
| current_mean_upstream_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.211452 | 5778 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.210336 | 5885 |
| current_mean_local_soil_wetness__lag0 | training_station_month_anomaly | signed_residual | -0.208342 | 5778 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.619345 | 0.644702 | 0.575157 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.08216 | 0.476705 | 0.0854043 |
| 八角电站 | 2021 | 11 | 1.97 | 2.09781 | 0.456679 | 0.000415935 |
| 双苏村 | 2025 | 10 | 3.56 | 2.25967 | 0.445703 | 0.417367 |

### AB0_hindcast

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 3866 | 73 | -0.36633 | -1.40535 | 0.117651 | 0.273973 | 0.798504 | 0.231932 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -1.00107 | 1.25599 | 1.05157 |
| concentration_band | low | 1304 | 73 | 0.371284 | 0.602479 | 0.548721 |
| concentration_band | middle | 1772 | 73 | -0.150209 | 0.533016 | 0.478292 |
| season | DJF | 971 | 73 | -0.207326 | 0.713448 | 0.604608 |
| season | JJA | 962 | 73 | -0.200448 | 0.799004 | 0.677454 |
| season | MAM | 961 | 73 | -0.0380291 | 0.712209 | 0.606891 |
| season | SON | 972 | 73 | -0.117023 | 0.758562 | 0.56715 |
| year | 2016 | 734 | 66 | -0.130364 | 0.707799 | 0.595167 |
| year | 2017 | 782 | 73 | -0.237245 | 0.821898 | 0.626164 |
| year | 2018 | 791 | 66 | -0.205296 | 0.688349 | 0.580025 |
| year | 2019 | 778 | 66 | -0.147092 | 0.648064 | 0.546004 |
| year | 2020 | 781 | 66 | -0.174082 | 0.727651 | 0.618131 |
| flow_band | high | 1287 | 71 | -0.291037 | 0.90038 | 0.676518 |
| flow_band | low | 358 | 64 | -0.136496 | 0.788511 | 0.702568 |
| flow_band | middle | 2150 | 71 | -0.0886177 | 0.705519 | 0.586294 |
| flow_band | no_training_threshold | 71 | 2 | -0.186021 | 0.41373 | 0.315822 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 兴宁电站 | 59 | -14.041 | -0.127768 | -1.67414 | 0.262712 |
| 五马岗 | 21 | -13.8697 | 0.196641 | 0.884336 | 0.693972 |
| 马头福水 | 21 | -12.495 | 0.355704 | 0.877548 | 0.58228 |
| 爽底坪 | 21 | -10.2251 | 0.186016 | 0.84498 | 0.418163 |
| 厂房大桥 | 39 | -7.61886 | 0.0683886 | 2.02049 | 0.610871 |
| 十里亭 | 60 | -6.54783 | 0.184853 | 1.00225 | 0.632728 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| upstream_log_dem_slope__static | between_station | bias | 0.203289 | 73 |
| log_upstream_area_ha__static | between_station | bias | -0.191502 | 73 |
| local_bulk_density_0_30_g_cm3__static | between_station | bias | 0.191353 | 73 |
| upstream_glhymps_porosity__static | between_station | bias | -0.189182 | 73 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.177711 | 3866 |
| current_mean_local_cropland_bnf_kg_n__lag3 | within_station | signed_residual | 0.168992 | 3866 |
| local_log_dem_slope__static | between_station | bias | 0.168519 | 73 |
| current_mean_upstream_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.166436 | 3866 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.05394 | 0.970543 | 0.986648 |
| 界牌 | 2016 | 9 | 4.38 | 1.14857 | 0.679474 | 0.657913 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.09916 | 0.522775 | 0.417093 |
| 扒齿 | 2016 | 9 | 3.59 | 2.09626 | 0.5128 | 0.158631 |

### AB1_training

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 5885 | 116 | -0.596125 | -1.41982 | 0.160373 | 0.232759 | 0.828959 | 0.253221 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.898297 | 1.08943 | 0.993864 |
| concentration_band | low | 1455 | 116 | 0.394297 | 0.690016 | 0.637251 |
| concentration_band | middle | 2967 | 116 | -0.125476 | 0.618947 | 0.562884 |
| season | DJF | 1410 | 116 | -0.254637 | 0.722285 | 0.651306 |
| season | JJA | 1476 | 116 | -0.300197 | 0.920082 | 0.78401 |
| season | MAM | 1528 | 116 | -0.0518774 | 0.775484 | 0.664919 |
| season | SON | 1471 | 116 | -0.159871 | 0.734976 | 0.649104 |
| year | 2021 | 1391 | 116 | -0.154449 | 0.783442 | 0.664914 |
| year | 2022 | 1215 | 116 | -0.147882 | 0.678293 | 0.583457 |
| year | 2023 | 1090 | 116 | -0.0738283 | 0.809497 | 0.698557 |
| year | 2024 | 1138 | 116 | -0.230662 | 0.849218 | 0.730371 |
| year | 2025 | 1051 | 116 | -0.339601 | 0.885649 | 0.76096 |
| flow_band | high | 1497 | 116 | -0.50492 | 0.907612 | 0.778765 |
| flow_band | low | 1497 | 116 | -0.0727513 | 0.795573 | 0.693967 |
| flow_band | middle | 2891 | 116 | -0.0872087 | 0.735073 | 0.636927 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.473 | -0.243742 | -2.39048 | 0.328731 |
| 八角电站 | 31 | -18.2394 | 0.248179 | 1.08671 | 0.852959 |
| 官渡 | 36 | -15.4688 | -0.0735521 | -1.69381 | 0.359185 |
| 滃江大站 | 59 | -13.1699 | 0.173931 | -1.9317 | 0.286364 |
| 平而关 | 31 | -12.8082 | -0.0180391 | 0.692018 | 0.842714 |
| 杨民 | 59 | -12.5541 | 0.023154 | -0.96519 | 0.418893 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | signed_residual | -0.300448 | 5778 |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.27352 | 5778 |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.254455 | 5885 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.219597 | 116 |
| current_mean_local_lower_release__lag0 | training_station_month_anomaly | signed_residual | -0.216119 | 5778 |
| current_mean_upstream_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.211449 | 5778 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.210336 | 5885 |
| current_mean_local_soil_wetness__lag0 | training_station_month_anomaly | signed_residual | -0.208339 | 5778 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.619348 | 0.644702 | 0.575158 |
| 新坪山桥 | 2025 | 4 | 3.32 | 2.08214 | 0.476705 | 0.0854122 |
| 八角电站 | 2021 | 11 | 1.97 | 2.09777 | 0.456679 | 0.000415729 |
| 双苏村 | 2025 | 10 | 3.56 | 2.25968 | 0.445703 | 0.417365 |

### AB1_hindcast

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 3866 | 73 | -0.366335 | -1.40544 | 0.117643 | 0.273973 | 0.798506 | 0.231933 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -1.00108 | 1.25599 | 1.05157 |
| concentration_band | low | 1304 | 73 | 0.37128 | 0.602477 | 0.548719 |
| concentration_band | middle | 1772 | 73 | -0.150214 | 0.533019 | 0.478294 |
| season | DJF | 971 | 73 | -0.20733 | 0.71345 | 0.60461 |
| season | JJA | 962 | 73 | -0.200455 | 0.799005 | 0.677454 |
| season | MAM | 961 | 73 | -0.0380338 | 0.712212 | 0.606894 |
| season | SON | 972 | 73 | -0.117028 | 0.758562 | 0.567151 |
| year | 2016 | 734 | 66 | -0.130363 | 0.707799 | 0.595168 |
| year | 2017 | 782 | 73 | -0.23725 | 0.821898 | 0.626163 |
| year | 2018 | 791 | 66 | -0.205305 | 0.688351 | 0.580027 |
| year | 2019 | 778 | 66 | -0.1471 | 0.648064 | 0.546005 |
| year | 2020 | 781 | 66 | -0.174089 | 0.727652 | 0.618131 |
| flow_band | high | 1287 | 71 | -0.291042 | 0.900383 | 0.67652 |
| flow_band | low | 358 | 64 | -0.136501 | 0.788507 | 0.702564 |
| flow_band | middle | 2150 | 71 | -0.0886226 | 0.705521 | 0.586295 |
| flow_band | no_training_threshold | 71 | 2 | -0.186025 | 0.413732 | 0.315823 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 兴宁电站 | 59 | -14.0408 | -0.127766 | -1.67414 | 0.262714 |
| 五马岗 | 21 | -13.8685 | 0.196635 | 0.884297 | 0.693932 |
| 马头福水 | 21 | -12.4958 | 0.355696 | 0.877576 | 0.582271 |
| 爽底坪 | 21 | -10.2253 | 0.186013 | 0.844989 | 0.418149 |
| 厂房大桥 | 39 | -7.61875 | 0.0683789 | 2.02047 | 0.610867 |
| 十里亭 | 60 | -6.54755 | 0.184862 | 1.00223 | 0.632717 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| upstream_log_dem_slope__static | between_station | bias | 0.203289 | 73 |
| log_upstream_area_ha__static | between_station | bias | -0.191502 | 73 |
| local_bulk_density_0_30_g_cm3__static | between_station | bias | 0.191353 | 73 |
| upstream_glhymps_porosity__static | between_station | bias | -0.189182 | 73 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.177711 | 3866 |
| current_mean_local_cropland_bnf_kg_n__lag3 | within_station | signed_residual | 0.168993 | 3866 |
| local_log_dem_slope__static | between_station | bias | 0.168519 | 73 |
| current_mean_upstream_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.166434 | 3866 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.05395 | 0.970543 | 0.986648 |
| 界牌 | 2016 | 9 | 4.38 | 1.14857 | 0.679474 | 0.657916 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.09911 | 0.522775 | 0.417115 |
| 扒齿 | 2016 | 9 | 3.59 | 2.09632 | 0.5128 | 0.158606 |

### AJ0_training

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 5885 | 116 | -0.570206 | -1.33084 | 0.216154 | 0.215517 | 0.822215 | 0.249858 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.896484 | 1.08196 | 0.988364 |
| concentration_band | low | 1455 | 116 | 0.39964 | 0.685981 | 0.636255 |
| concentration_band | middle | 2967 | 116 | -0.123102 | 0.60868 | 0.554421 |
| season | DJF | 1410 | 116 | -0.256111 | 0.717487 | 0.645934 |
| season | JJA | 1476 | 116 | -0.287586 | 0.913154 | 0.780997 |
| season | MAM | 1528 | 116 | -0.055591 | 0.772151 | 0.663018 |
| season | SON | 1471 | 116 | -0.15529 | 0.720922 | 0.635685 |
| year | 2021 | 1391 | 116 | -0.173095 | 0.777851 | 0.660586 |
| year | 2022 | 1215 | 116 | -0.136511 | 0.675599 | 0.582499 |
| year | 2023 | 1090 | 116 | -0.0903322 | 0.815888 | 0.703175 |
| year | 2024 | 1138 | 116 | -0.208194 | 0.833468 | 0.719093 |
| year | 2025 | 1051 | 116 | -0.317265 | 0.874708 | 0.746571 |
| flow_band | high | 1497 | 116 | -0.460047 | 0.89364 | 0.765244 |
| flow_band | low | 1497 | 116 | -0.0793903 | 0.792925 | 0.69185 |
| flow_band | middle | 2891 | 116 | -0.101005 | 0.731517 | 0.632993 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.236 | -0.211292 | -2.38085 | 0.268789 |
| 八角电站 | 31 | -17.7369 | 0.220228 | 1.07221 | 0.790585 |
| 官渡 | 36 | -15.8955 | -0.240541 | -1.71311 | 0.321693 |
| 滃江大站 | 59 | -13.3309 | 0.0555312 | -1.93942 | 0.257955 |
| 杨民 | 59 | -12.3246 | 0.0407366 | -0.957543 | 0.392645 |
| 平而关 | 31 | -12.027 | 0.141998 | 0.68065 | 0.752533 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | signed_residual | -0.238057 | 5778 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.221877 | 116 |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.218224 | 5778 |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.203711 | 5885 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.202624 | 5885 |
| current_mean_local_cropland_bnf_kg_n__lag3 | within_station | signed_residual | 0.198858 | 5885 |
| current_mean_local_lower_release__lag0 | training_station_month_anomaly | signed_residual | -0.191139 | 5778 |
| current_mean_upstream_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.182197 | 5778 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.624151 | 0.644702 | 0.581866 |
| 新坪山桥 | 2025 | 4 | 3.32 | 1.94942 | 0.476705 | 0.101022 |
| 八角电站 | 2021 | 11 | 1.97 | 2.08315 | 0.456679 | 0.000334787 |
| 双苏村 | 2025 | 10 | 3.56 | 2.32624 | 0.445703 | 0.397641 |

### AJ0_hindcast

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 3866 | 73 | -0.353117 | -1.34606 | 0.0938286 | 0.342466 | 0.79787 | 0.231295 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.983522 | 1.24476 | 1.04166 |
| concentration_band | low | 1304 | 73 | 0.409579 | 0.62462 | 0.569668 |
| concentration_band | middle | 1772 | 73 | -0.116756 | 0.528233 | 0.472704 |
| season | DJF | 971 | 73 | -0.159505 | 0.719952 | 0.60682 |
| season | JJA | 962 | 73 | -0.183783 | 0.796321 | 0.676809 |
| season | MAM | 961 | 73 | -0.00182393 | 0.710075 | 0.605272 |
| season | SON | 972 | 73 | -0.0869337 | 0.754505 | 0.563272 |
| year | 2016 | 734 | 66 | -0.066161 | 0.716351 | 0.603274 |
| year | 2017 | 782 | 73 | -0.194327 | 0.81314 | 0.617407 |
| year | 2018 | 791 | 66 | -0.196014 | 0.685359 | 0.57426 |
| year | 2019 | 778 | 66 | -0.109973 | 0.642979 | 0.54173 |
| year | 2020 | 781 | 66 | -0.154717 | 0.722104 | 0.613874 |
| flow_band | high | 1287 | 71 | -0.240951 | 0.904878 | 0.680074 |
| flow_band | low | 358 | 64 | -0.114383 | 0.779347 | 0.695647 |
| flow_band | middle | 2150 | 71 | -0.0652233 | 0.703421 | 0.583363 |
| flow_band | no_training_threshold | 71 | 2 | -0.121814 | 0.399258 | 0.304598 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -16.8717 | -0.00932914 | 0.970379 | 0.641111 |
| 马头福水 | 21 | -14.4469 | 0.151069 | 0.938134 | 0.473583 |
| 兴宁电站 | 59 | -13.7207 | -0.138563 | -1.65342 | 0.282434 |
| 爽底坪 | 21 | -11.1617 | 0.144546 | 0.882814 | 0.350365 |
| 厂房大桥 | 39 | -7.37187 | 0.0938286 | 1.99654 | 0.567257 |
| 十里亭 | 60 | -6.99517 | 0.293758 | 1.04826 | 0.611124 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| local_bulk_density_0_30_g_cm3__static | between_station | bias | 0.19919 | 73 |
| upstream_log_dem_slope__static | between_station | bias | 0.19053 | 73 |
| current_mean_local_soil_wetness__lag3 | within_station | signed_residual | 0.181296 | 3866 |
| upstream_glhymps_porosity__static | between_station | bias | -0.180311 | 73 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.179504 | 73 |
| log_upstream_area_ha__static | between_station | bias | -0.177019 | 73 |
| current_mean_local_slow_mm__lag3 | training_station_month_anomaly | signed_residual | 0.169703 | 3784 |
| current_mean_local_soil_wetness__lag3 | raw | signed_residual | 0.168231 | 3866 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.1089 | 0.970543 | 0.986692 |
| 界牌 | 2016 | 9 | 4.38 | 1.22016 | 0.679474 | 0.617636 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.0726 | 0.522775 | 0.412661 |
| 扒齿 | 2016 | 9 | 3.59 | 2.10457 | 0.5128 | 0.129343 |

### AJ1_training

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 5885 | 116 | -0.570214 | -1.33076 | 0.216155 | 0.215517 | 0.822213 | 0.249858 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 1463 | 116 | -0.896484 | 1.08196 | 0.988364 |
| concentration_band | low | 1455 | 116 | 0.399639 | 0.68598 | 0.636253 |
| concentration_band | middle | 2967 | 116 | -0.123104 | 0.608678 | 0.554418 |
| season | DJF | 1410 | 116 | -0.256114 | 0.717484 | 0.645931 |
| season | JJA | 1476 | 116 | -0.287586 | 0.913152 | 0.780995 |
| season | MAM | 1528 | 116 | -0.0555952 | 0.772151 | 0.663017 |
| season | SON | 1471 | 116 | -0.155289 | 0.72092 | 0.635682 |
| year | 2021 | 1391 | 116 | -0.173096 | 0.777849 | 0.660584 |
| year | 2022 | 1215 | 116 | -0.136516 | 0.675597 | 0.582497 |
| year | 2023 | 1090 | 116 | -0.0903322 | 0.815888 | 0.703175 |
| year | 2024 | 1138 | 116 | -0.208196 | 0.833466 | 0.71909 |
| year | 2025 | 1051 | 116 | -0.317264 | 0.874705 | 0.746568 |
| flow_band | high | 1497 | 116 | -0.460046 | 0.893638 | 0.765241 |
| flow_band | low | 1497 | 116 | -0.0793906 | 0.792923 | 0.691848 |
| flow_band | middle | 2891 | 116 | -0.101007 | 0.731515 | 0.63299 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 赵家渡 | 58 | -19.2361 | -0.211306 | -2.38085 | 0.268793 |
| 八角电站 | 31 | -17.7375 | 0.220233 | 1.07223 | 0.790593 |
| 官渡 | 36 | -15.8957 | -0.240532 | -1.71312 | 0.32169 |
| 滃江大站 | 59 | -13.331 | 0.0555414 | -1.93942 | 0.257952 |
| 杨民 | 59 | -12.3245 | 0.0407301 | -0.957539 | 0.392646 |
| 平而关 | 31 | -12.0271 | 0.141993 | 0.680653 | 0.75254 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| current_mean_log_boundary_mm__lag0 | training_station_month_anomaly | signed_residual | -0.238056 | 5778 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.221877 | 116 |
| current_mean_local_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.218223 | 5778 |
| current_mean_local_soil_wetness__lag0 | within_station | signed_residual | -0.203713 | 5885 |
| current_mean_local_crop_demand_kg_n_ha__lag3 | within_station | signed_residual | 0.202625 | 5885 |
| current_mean_local_cropland_bnf_kg_n__lag3 | within_station | signed_residual | 0.198859 | 5885 |
| current_mean_local_lower_release__lag0 | training_station_month_anomaly | signed_residual | -0.191138 | 5778 |
| current_mean_upstream_fast_mm__lag0 | training_station_month_anomaly | signed_residual | -0.182197 | 5778 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 龙归 | 2021 | 3 | 5.29 | 0.624147 | 0.644702 | 0.581864 |
| 新坪山桥 | 2025 | 4 | 3.32 | 1.94942 | 0.476705 | 0.101024 |
| 八角电站 | 2021 | 11 | 1.97 | 2.08318 | 0.456679 | 0.000334935 |
| 双苏村 | 2025 | 10 | 3.56 | 2.32625 | 0.445703 | 0.397637 |

### AJ1_hindcast

| rows | stations | median_nse | q25_nse | median_time_r | negative_r_fraction | mean_station_raw_rmse | mean_station_log_rmse |
|---|---|---|---|---|---|---|---|
| 3866 | 73 | -0.353114 | -1.34606 | 0.0938478 | 0.342466 | 0.797867 | 0.231294 |

偏差=预测−实测；以下按站平均，mg/L。分层阈值只由2021–2025训练数据产生。

| field | label | rows | stations | station_mean_bias_mg_l | station_mean_rmse_mg_l | station_mean_absolute_error_mg_l |
|---|---|---|---|---|---|---|
| concentration_band | high | 790 | 69 | -0.98352 | 1.24475 | 1.04166 |
| concentration_band | low | 1304 | 73 | 0.409581 | 0.62462 | 0.569667 |
| concentration_band | middle | 1772 | 73 | -0.116754 | 0.52823 | 0.472701 |
| season | DJF | 971 | 73 | -0.159503 | 0.719948 | 0.606815 |
| season | JJA | 962 | 73 | -0.183782 | 0.796318 | 0.676806 |
| season | MAM | 961 | 73 | -0.0018256 | 0.710073 | 0.605269 |
| season | SON | 972 | 73 | -0.0869276 | 0.754503 | 0.56327 |
| year | 2016 | 734 | 66 | -0.0661626 | 0.716351 | 0.603274 |
| year | 2017 | 782 | 73 | -0.194324 | 0.81314 | 0.617406 |
| year | 2018 | 791 | 66 | -0.196009 | 0.685355 | 0.574256 |
| year | 2019 | 778 | 66 | -0.109971 | 0.642976 | 0.541726 |
| year | 2020 | 781 | 66 | -0.154716 | 0.722101 | 0.613872 |
| flow_band | high | 1287 | 71 | -0.240949 | 0.904875 | 0.680071 |
| flow_band | low | 358 | 64 | -0.114382 | 0.779347 | 0.695646 |
| flow_band | middle | 2150 | 71 | -0.0652216 | 0.703418 | 0.58336 |
| flow_band | no_training_threshold | 71 | 2 | -0.12181 | 0.399259 | 0.304597 |

较差单站（完整逐站、逐年表已输出）：

| station_key | rows | nse | time_r | bias_mg_l | sd_ratio |
|---|---|---|---|---|---|
| 五马岗 | 21 | -16.8709 | -0.00932355 | 0.970357 | 0.641094 |
| 马头福水 | 21 | -14.4463 | 0.151078 | 0.938114 | 0.473586 |
| 兴宁电站 | 59 | -13.7208 | -0.138567 | -1.65342 | 0.282433 |
| 爽底坪 | 21 | -11.1615 | 0.144545 | 0.882808 | 0.350367 |
| 厂房大桥 | 39 | -7.3718 | 0.0938478 | 1.99653 | 0.567263 |
| 十里亭 | 60 | -6.99514 | 0.293749 | 1.04826 | 0.611123 |

有符号残差与输入的关联（不是因果证明）：

| feature | adjustment | target | spearman | units |
|---|---|---|---|---|
| local_bulk_density_0_30_g_cm3__static | between_station | bias | 0.19919 | 73 |
| upstream_log_dem_slope__static | between_station | bias | 0.19053 | 73 |
| current_mean_local_soil_wetness__lag3 | within_station | signed_residual | 0.181295 | 3866 |
| upstream_glhymps_porosity__static | between_station | bias | -0.180311 | 73 |
| local_log_predev_annual_pet_mm__static | between_station | bias | -0.179504 | 73 |
| log_upstream_area_ha__static | between_station | bias | -0.177019 | 73 |
| current_mean_local_slow_mm__lag3 | training_station_month_anomaly | signed_residual | 0.169701 | 3784 |
| current_mean_local_soil_wetness__lag3 | raw | signed_residual | 0.168231 | 3866 |

极值保留，未自动纠正：

| station_key | year | month | tn_mg_l | prediction_mg_l | station_variance_share | station_sse_share |
|---|---|---|---|---|---|---|
| 棉江 | 2017 | 10 | 40.3067 | 1.10889 | 0.970543 | 0.986692 |
| 界牌 | 2016 | 9 | 4.38 | 1.22018 | 0.679474 | 0.617629 |
| 自良渡口 | 2020 | 1 | 6.55 | 2.07261 | 0.522775 | 0.41266 |
| 扒齿 | 2016 | 9 | 3.59 | 2.10458 | 0.5128 | 0.129341 |

### OLD_B：来源及损失位置

以下为单个来源完整历史的河道损失，单位kg N；嵌套站点的到站量未相加。

| source_reach_id | entry_path | input_kg_n | removed_kg_n | terminal_kg_n | ending_reservoir_kg_n |
|---|---|---|---|---|---|
| 190 | slow | 2.52766e+08 | 8.42726e+07 | 1.62624e+08 | 5.86962e+06 |
| 190 | fast | 3.1414e+08 | 5.02727e+07 | 2.48463e+08 | 1.54041e+07 |
| 158 | slow | 9.81789e+07 | 3.14299e+07 | 6.48361e+07 | 1.91292e+06 |
| 161 | slow | 7.89e+07 | 2.15231e+07 | 5.56558e+07 | 1.72106e+06 |
| 158 | fast | 1.15747e+08 | 1.82636e+07 | 9.32075e+07 | 4.27546e+06 |
| 81 | fast | 4.22235e+08 | 1.75554e+07 | 4.0468e+08 | 0 |

来源→站点及快慢路径的各时期比例见 source_station_contributions；分母仅为该站该时期模型到站量。

### OLD_B / training：固定入河氮的河网边界

原始5885条记录，无河道去除后仍低于实测2898条（49.24%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.614912 | 0.142224 | 0.832055 | 0.254376 | 0.921629 |
| no_channel_loss | -0.729955 | 0.01394 | 0.874572 | 0.261911 | 0.852475 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1497 | 116 | 0.713427 | 0.62059 | 0.0293765 | 0.0410492 |
| flow_band | low | 1497 | 116 | 0.360721 | 0.29488 | 0.0886806 | 0.294122 |
| flow_band | middle | 2891 | 116 | 0.446212 | 0.308126 | 0.0535567 | 0.127839 |
| area_quartile | 2 | 2699 | 58 | 0.521304 | 0.508825 | 0.0367992 | 0.1088 |
| area_quartile | 3 | 1590 | 29 | 0.401887 | 0.244266 | 0.0487026 | 0.159665 |
| area_quartile | 4 | 1596 | 29 | 0.533835 | 0.27411 | 0.103048 | 0.215039 |
| reservoir_affected | False | 4460 | 90 | 0.463901 | 0.410658 | 0.0452191 | 0.14037 |
| reservoir_affected | True | 1425 | 26 | 0.581754 | 0.291753 | 0.0948227 | 0.174751 |

### OLD_B / hindcast：固定入河氮的河网边界

原始3866条记录，无河道去除后仍低于实测1919条（49.64%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.389233 | 0.109399 | 0.800268 | 0.232631 | 0.929528 |
| no_channel_loss | -0.449351 | 0.0623616 | 0.811398 | 0.232703 | 0.876952 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1287 | 71 | 0.618493 | 0.458697 | 0.0313018 | 0.0484395 |
| flow_band | low | 358 | 64 | 0.413408 | 0.325688 | 0.0949562 | 0.280136 |
| flow_band | middle | 2150 | 71 | 0.438605 | 0.269806 | 0.0686172 | 0.144233 |
| flow_band | no_training_threshold | 71 | 2 | 0.450704 | 0.175329 | 0.0823786 | 0.148199 |
| area_quartile | 2 | 1064 | 23 | 0.532895 | 0.528681 | 0.0382846 | 0.113007 |
| area_quartile | 3 | 1325 | 24 | 0.383396 | 0.202622 | 0.0375611 | 0.116715 |
| area_quartile | 4 | 1477 | 26 | 0.571429 | 0.243902 | 0.0969469 | 0.155106 |
| reservoir_affected | False | 2544 | 50 | 0.433569 | 0.346334 | 0.0478459 | 0.130423 |
| reservoir_affected | True | 1322 | 23 | 0.617247 | 0.262929 | 0.0830582 | 0.126606 |

### OLD_F：来源及损失位置

以下为单个来源完整历史的河道损失，单位kg N；嵌套站点的到站量未相加。

| source_reach_id | entry_path | input_kg_n | removed_kg_n | terminal_kg_n | ending_reservoir_kg_n |
|---|---|---|---|---|---|
| 190 | slow | 2.46988e+08 | 8.2234e+07 | 1.58951e+08 | 5.80317e+06 |
| 190 | fast | 3.17321e+08 | 5.02658e+07 | 2.51284e+08 | 1.57714e+07 |
| 158 | slow | 9.57335e+07 | 3.06202e+07 | 6.32336e+07 | 1.87962e+06 |
| 161 | slow | 7.69539e+07 | 2.09657e+07 | 5.42921e+07 | 1.69613e+06 |
| 158 | fast | 1.16329e+08 | 1.82184e+07 | 9.37705e+07 | 4.34059e+06 |
| 81 | fast | 4.51529e+08 | 1.82086e+07 | 4.3332e+08 | 0 |

来源→站点及快慢路径的各时期比例见 source_station_contributions；分母仅为该站该时期模型到站量。

### OLD_F / training：固定入河氮的河网边界

原始5885条记录，无河道去除后仍低于实测2828条（48.05%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.566176 | 0.240669 | 0.819868 | 0.249633 | 1.00593 |
| no_channel_loss | -0.671622 | 0.0813123 | 0.857187 | 0.256277 | 1.01231 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1497 | 116 | 0.670675 | 0.574041 | 0.0284349 | 0.0414994 |
| flow_band | low | 1497 | 116 | 0.369405 | 0.302571 | 0.0917459 | 0.287997 |
| flow_band | middle | 2891 | 116 | 0.43964 | 0.313388 | 0.0540653 | 0.126076 |
| area_quartile | 2 | 2699 | 58 | 0.515376 | 0.508069 | 0.0378136 | 0.106507 |
| area_quartile | 3 | 1590 | 29 | 0.386792 | 0.234687 | 0.0495639 | 0.157223 |
| area_quartile | 4 | 1596 | 29 | 0.515038 | 0.256058 | 0.10334 | 0.212828 |
| reservoir_affected | False | 4460 | 90 | 0.456502 | 0.407463 | 0.0466506 | 0.137654 |
| reservoir_affected | True | 1425 | 26 | 0.555789 | 0.270307 | 0.093417 | 0.173845 |

### OLD_F / hindcast：固定入河氮的河网边界

原始3866条记录，无河道去除后仍低于实测1800条（46.56%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.355086 | 0.121439 | 0.798388 | 0.23153 | 0.964212 |
| no_channel_loss | -0.47843 | 0.0714178 | 0.80792 | 0.232017 | 0.978015 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1287 | 71 | 0.554002 | 0.425826 | 0.0301393 | 0.0489983 |
| flow_band | low | 358 | 64 | 0.407821 | 0.324283 | 0.0987434 | 0.276088 |
| flow_band | middle | 2150 | 71 | 0.424186 | 0.267006 | 0.0683556 | 0.143687 |
| flow_band | no_training_threshold | 71 | 2 | 0.408451 | 0.148505 | 0.0735093 | 0.150396 |
| area_quartile | 2 | 1064 | 23 | 0.521617 | 0.521862 | 0.0402909 | 0.111092 |
| area_quartile | 3 | 1325 | 24 | 0.353962 | 0.190793 | 0.0383283 | 0.116249 |
| area_quartile | 4 | 1477 | 26 | 0.525389 | 0.224428 | 0.0937346 | 0.155477 |
| reservoir_affected | False | 2544 | 50 | 0.411164 | 0.338767 | 0.0493345 | 0.128977 |
| reservoir_affected | True | 1322 | 23 | 0.570348 | 0.238202 | 0.0789976 | 0.127767 |

### OLD_J：来源及损失位置

以下为单个来源完整历史的河道损失，单位kg N；嵌套站点的到站量未相加。

| source_reach_id | entry_path | input_kg_n | removed_kg_n | terminal_kg_n | ending_reservoir_kg_n |
|---|---|---|---|---|---|
| 190 | slow | 2.46567e+08 | 7.46416e+07 | 1.66095e+08 | 5.83032e+06 |
| 190 | fast | 3.16293e+08 | 4.48935e+07 | 2.55623e+08 | 1.57768e+07 |
| 158 | slow | 9.67845e+07 | 2.80583e+07 | 6.68081e+07 | 1.91803e+06 |
| 161 | slow | 7.69932e+07 | 1.89439e+07 | 5.63469e+07 | 1.70232e+06 |
| 158 | fast | 1.17413e+08 | 1.64718e+07 | 9.6547e+07 | 4.39384e+06 |
| 81 | fast | 4.38664e+08 | 1.58136e+07 | 4.2285e+08 | 0 |

来源→站点及快慢路径的各时期比例见 source_station_contributions；分母仅为该站该时期模型到站量。

### OLD_J / training：固定入河氮的河网边界

原始5885条记录，无河道去除后仍低于实测2912条（49.48%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.648313 | 0.164499 | 0.834278 | 0.254065 | 0.992372 |
| no_channel_loss | -0.719614 | 0.0143601 | 0.869487 | 0.260084 | 0.946785 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1497 | 116 | 0.702071 | 0.616126 | 0.025442 | 0.0358726 |
| flow_band | low | 1497 | 116 | 0.380762 | 0.307147 | 0.0793508 | 0.257005 |
| flow_band | middle | 2891 | 116 | 0.446558 | 0.319252 | 0.0473162 | 0.111362 |
| area_quartile | 2 | 2699 | 58 | 0.519822 | 0.519341 | 0.0326226 | 0.0944374 |
| area_quartile | 3 | 1590 | 29 | 0.40566 | 0.24797 | 0.0431544 | 0.139186 |
| area_quartile | 4 | 1596 | 29 | 0.541353 | 0.279216 | 0.0912122 | 0.188827 |
| reservoir_affected | False | 4460 | 90 | 0.46435 | 0.418925 | 0.0399577 | 0.122275 |
| reservoir_affected | True | 1425 | 26 | 0.590175 | 0.296419 | 0.0843288 | 0.153268 |

### OLD_J / hindcast：固定入河氮的河网边界

原始3866条记录，无河道去除后仍低于实测1878条（48.58%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.402111 | 0.0752329 | 0.804727 | 0.233805 | 0.878776 |
| no_channel_loss | -0.50027 | 0.0491873 | 0.815473 | 0.234525 | 0.852182 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1287 | 71 | 0.599068 | 0.455162 | 0.0272609 | 0.0425441 |
| flow_band | low | 358 | 64 | 0.416201 | 0.324708 | 0.0838108 | 0.245962 |
| flow_band | middle | 2150 | 71 | 0.430698 | 0.268334 | 0.0592487 | 0.127035 |
| flow_band | no_training_threshold | 71 | 2 | 0.450704 | 0.167205 | 0.0670146 | 0.132442 |
| area_quartile | 2 | 1064 | 23 | 0.528195 | 0.524946 | 0.0341326 | 0.0985731 |
| area_quartile | 3 | 1325 | 24 | 0.368302 | 0.202096 | 0.0324512 | 0.102415 |
| area_quartile | 4 | 1477 | 26 | 0.560596 | 0.2418 | 0.0830272 | 0.137395 |
| reservoir_affected | False | 2544 | 50 | 0.422956 | 0.345498 | 0.0417685 | 0.114264 |
| reservoir_affected | True | 1322 | 23 | 0.606657 | 0.258085 | 0.0710505 | 0.112356 |

### A_B：来源及损失位置

以下为单个来源完整历史的河道损失，单位kg N；嵌套站点的到站量未相加。

| source_reach_id | entry_path | input_kg_n | removed_kg_n | terminal_kg_n | ending_reservoir_kg_n |
|---|---|---|---|---|---|
| 190 | slow | 2.52176e+08 | 8.26245e+07 | 1.63644e+08 | 5.90809e+06 |
| 190 | fast | 3.14244e+08 | 4.92131e+07 | 2.49518e+08 | 1.55125e+07 |
| 158 | slow | 9.80348e+07 | 3.08412e+07 | 6.52682e+07 | 1.92539e+06 |
| 161 | slow | 7.8765e+07 | 2.10932e+07 | 5.59412e+07 | 1.73061e+06 |
| 158 | fast | 1.15755e+08 | 1.78852e+07 | 9.357e+07 | 4.30021e+06 |
| 81 | fast | 4.25762e+08 | 1.72297e+07 | 4.08532e+08 | 0 |

来源→站点及快慢路径的各时期比例见 source_station_contributions；分母仅为该站该时期模型到站量。

### A_B / training：固定入河氮的河网边界

原始5885条记录，无河道去除后仍低于实测2895条（49.19%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.596068 | 0.160378 | 0.828958 | 0.253221 | 0.950606 |
| no_channel_loss | -0.698281 | 0.0244731 | 0.870149 | 0.260538 | 0.883093 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1497 | 116 | 0.708083 | 0.613207 | 0.0286315 | 0.0402612 |
| flow_band | low | 1497 | 116 | 0.364061 | 0.296539 | 0.0868188 | 0.287275 |
| flow_band | middle | 2891 | 116 | 0.446212 | 0.309611 | 0.0524557 | 0.124876 |
| area_quartile | 2 | 2699 | 58 | 0.521675 | 0.508988 | 0.0360858 | 0.106187 |
| area_quartile | 3 | 1590 | 29 | 0.401887 | 0.243025 | 0.047763 | 0.15593 |
| area_quartile | 4 | 1596 | 29 | 0.531328 | 0.272125 | 0.100599 | 0.210407 |
| reservoir_affected | False | 4460 | 90 | 0.463901 | 0.410395 | 0.0443378 | 0.137061 |
| reservoir_affected | True | 1425 | 26 | 0.579649 | 0.289426 | 0.0925027 | 0.171044 |

### A_B / hindcast：固定入河氮的河网边界

原始3866条记录，无河道去除后仍低于实测1912条（49.46%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.36633 | 0.117651 | 0.798504 | 0.231932 | 0.954964 |
| no_channel_loss | -0.428025 | 0.0660314 | 0.809522 | 0.232093 | 0.883679 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1287 | 71 | 0.609946 | 0.453299 | 0.0304786 | 0.0474954 |
| flow_band | low | 358 | 64 | 0.416201 | 0.326681 | 0.0928505 | 0.273891 |
| flow_band | middle | 2150 | 71 | 0.44 | 0.270501 | 0.0669552 | 0.141092 |
| flow_band | no_training_threshold | 71 | 2 | 0.450704 | 0.171581 | 0.0793403 | 0.145367 |
| area_quartile | 2 | 1064 | 23 | 0.534774 | 0.528424 | 0.0374251 | 0.11043 |
| area_quartile | 3 | 1325 | 24 | 0.381132 | 0.201044 | 0.0366587 | 0.114182 |
| area_quartile | 4 | 1477 | 26 | 0.567366 | 0.241751 | 0.0944198 | 0.151902 |
| reservoir_affected | False | 2544 | 50 | 0.433176 | 0.345446 | 0.046748 | 0.127546 |
| reservoir_affected | True | 1322 | 23 | 0.612708 | 0.260525 | 0.080787 | 0.124018 |

### A_J：来源及损失位置

以下为单个来源完整历史的河道损失，单位kg N；嵌套站点的到站量未相加。

| source_reach_id | entry_path | input_kg_n | removed_kg_n | terminal_kg_n | ending_reservoir_kg_n |
|---|---|---|---|---|---|
| 190 | slow | 2.45083e+08 | 6.90399e+07 | 1.70049e+08 | 5.99331e+06 |
| 190 | fast | 3.16963e+08 | 4.12851e+07 | 2.59498e+08 | 1.61801e+07 |
| 158 | slow | 9.66442e+07 | 2.60776e+07 | 6.8593e+07 | 1.97371e+06 |
| 161 | slow | 7.65725e+07 | 1.7469e+07 | 5.73625e+07 | 1.74094e+06 |
| 158 | fast | 1.17721e+08 | 1.51865e+07 | 9.80414e+07 | 4.49309e+06 |
| 81 | fast | 4.49611e+08 | 1.46298e+07 | 4.34981e+08 | 0 |

来源→站点及快慢路径的各时期比例见 source_station_contributions；分母仅为该站该时期模型到站量。

### A_J / training：固定入河氮的河网边界

原始5885条记录，无河道去除后仍低于实测2909条（49.43%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.570214 | 0.216155 | 0.822213 | 0.249858 | 1.0555 |
| no_channel_loss | -0.611477 | 0.0753597 | 0.853348 | 0.255247 | 1.05122 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1497 | 116 | 0.684035 | 0.589542 | 0.0231015 | 0.0332527 |
| flow_band | low | 1497 | 116 | 0.386774 | 0.312873 | 0.0727468 | 0.234219 |
| flow_band | middle | 2891 | 116 | 0.451747 | 0.323377 | 0.0436217 | 0.101672 |
| area_quartile | 2 | 2699 | 58 | 0.52538 | 0.518958 | 0.03024 | 0.0859026 |
| area_quartile | 3 | 1590 | 29 | 0.404403 | 0.243216 | 0.0398956 | 0.126854 |
| area_quartile | 4 | 1596 | 29 | 0.531328 | 0.271631 | 0.0828719 | 0.173332 |
| reservoir_affected | False | 4460 | 90 | 0.468386 | 0.4172 | 0.0369723 | 0.11138 |
| reservoir_affected | True | 1425 | 26 | 0.575439 | 0.287775 | 0.0764103 | 0.140904 |

### A_J / hindcast：固定入河氮的河网边界

原始3866条记录，无河道去除后仍低于实测1843条（47.67%）。这是数值边界检查，不是测量显著性检验。

| scenario | median_nse | median_time_r | mean_station_raw_rmse | mean_station_log_rmse | d_sigma |
|---|---|---|---|---|---|
| fitted | -0.353114 | 0.0938478 | 0.797867 | 0.231294 | 0.960161 |
| no_channel_loss | -0.421872 | 0.0712495 | 0.80824 | 0.232233 | 0.961594 |

| field | label | rows | stations | still_below_fraction | station_mean_remaining_deficit_mg_l | station_mean_reducible_deficit_mg_l | station_mean_no_loss_gain_mg_l |
|---|---|---|---|---|---|---|---|
| flow_band | high | 1287 | 71 | 0.571873 | 0.436165 | 0.0243453 | 0.0393814 |
| flow_band | low | 358 | 64 | 0.413408 | 0.328294 | 0.0767202 | 0.225021 |
| flow_band | middle | 2150 | 71 | 0.432558 | 0.270513 | 0.0537778 | 0.116551 |
| flow_band | no_training_threshold | 71 | 2 | 0.408451 | 0.154609 | 0.0585942 | 0.122596 |
| area_quartile | 2 | 1064 | 23 | 0.526316 | 0.523609 | 0.0311627 | 0.0901411 |
| area_quartile | 3 | 1325 | 24 | 0.362264 | 0.196752 | 0.0292876 | 0.0939682 |
| area_quartile | 4 | 1477 | 26 | 0.54367 | 0.234311 | 0.0749702 | 0.126509 |
| reservoir_affected | False | 2544 | 50 | 0.418239 | 0.342118 | 0.0379491 | 0.104738 |
| reservoir_affected | True | 1322 | 23 | 0.589259 | 0.250056 | 0.0639747 | 0.103515 |

## 配对不确定性及注册预测条件

### OLD_B→A_B / training / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.00196018 | 0.0526135 | 1000 |
| q25_nse | candidate_minus_baseline | -0.00736287 | 0.064879 | 1000 |
| median_r | candidate_minus_baseline | -0.00351205 | 0.024659 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00135989 | -0.000968915 | 1000 |
| rmse | candidate_minus_baseline | -0.00363932 | -0.00256154 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00230925 | -0.000959209 | 1000 |

| condition | passed |
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

### OLD_B→A_B / training / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | 8.33473e-05 | 0.0577761 | 1000 |
| q25_nse | candidate_minus_baseline | 0.0055399 | 0.0703891 | 1000 |
| median_r | candidate_minus_baseline | -0.00649985 | 0.0277881 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00163661 | -0.00118009 | 1000 |
| rmse | candidate_minus_baseline | -0.00439989 | -0.00318929 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00290188 | -0.000885464 | 1000 |

| condition | passed |
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

### OLD_B→A_B / hindcast / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.00807418 | 0.0209482 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0188243 | 0.0528457 | 1000 |
| median_r | candidate_minus_baseline | -0.00260279 | 0.0196459 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0 | 1000 |
| log_rmse | candidate_minus_baseline | -0.000841794 | -0.000447992 | 1000 |
| rmse | candidate_minus_baseline | -0.0021729 | -0.00108405 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00263696 | -0.000522302 | 1000 |

| condition | passed |
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

### OLD_B→A_B / hindcast / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.00807418 | 0.0525793 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0188243 | 0.0528457 | 1000 |
| median_r | candidate_minus_baseline | -0.00260279 | 0.0196459 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0 | 1000 |
| log_rmse | candidate_minus_baseline | -0.000902196 | -0.000501902 | 1000 |
| rmse | candidate_minus_baseline | -0.00230009 | -0.0012301 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00282165 | -0.000759012 | 1000 |

| condition | passed |
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

### OLD_J→A_J / training / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | 0.00855929 | 0.152703 | 1000 |
| q25_nse | candidate_minus_baseline | -0.00864004 | 0.231496 | 1000 |
| median_r | candidate_minus_baseline | 0.00188944 | 0.0832659 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0862069 | -0.00840517 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00492543 | -0.0035401 | 1000 |
| rmse | candidate_minus_baseline | -0.0139734 | -0.00997536 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00921001 | -0.00402519 | 1000 |

| condition | passed |
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

### OLD_J→A_J / training / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | 0.0156435 | 0.182477 | 1000 |
| q25_nse | candidate_minus_baseline | 0.0282292 | 0.227683 | 1000 |
| median_r | candidate_minus_baseline | 0.0430775 | 0.0943028 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.112676 | -0.0140845 | 1000 |
| log_rmse | candidate_minus_baseline | -0.0058868 | -0.00433089 | 1000 |
| rmse | candidate_minus_baseline | -0.017042 | -0.0125813 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0113383 | -0.00353935 | 1000 |

| condition | passed |
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

### OLD_J→A_J / hindcast / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.00131092 | 0.0884505 | 1000 |
| q25_nse | candidate_minus_baseline | 0.00874397 | 0.172838 | 1000 |
| median_r | candidate_minus_baseline | -0.0142732 | 0.0524055 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.084507 | 0.028169 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00302421 | -0.00176837 | 1000 |
| rmse | candidate_minus_baseline | -0.00855656 | -0.00455677 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0104506 | -0.00247896 | 1000 |

| condition | passed |
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

### OLD_J→A_J / hindcast / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.00131092 | 0.105134 | 1000 |
| q25_nse | candidate_minus_baseline | 0.00845803 | 0.19355 | 1000 |
| median_r | candidate_minus_baseline | -0.0122421 | 0.0524055 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0821918 | 0.0273973 | 1000 |
| log_rmse | candidate_minus_baseline | -0.0031274 | -0.00191359 | 1000 |
| rmse | candidate_minus_baseline | -0.00862834 | -0.00498321 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0108195 | -0.0035511 | 1000 |

| condition | passed |
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

### OLD_F→A_J / training / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0898497 | 0.0845913 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0594235 | 0.373954 | 1000 |
| median_r | candidate_minus_baseline | -0.0654836 | -0.005687 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0172414 | 0.0344828 | 1000 |
| log_rmse | candidate_minus_baseline | -0.000976836 | 0.00137925 | 1000 |
| rmse | candidate_minus_baseline | -0.00315291 | 0.0082615 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0137117 | 0.00126778 | 1000 |

| condition | passed |
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

### OLD_F→A_J / training / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.108096 | 0.0994197 | 1000 |
| q25_nse | candidate_minus_baseline | -0.147599 | 0.213761 | 1000 |
| median_r | candidate_minus_baseline | -0.0594397 | -0.00970317 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0.0422535 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00105769 | 0.00210898 | 1000 |
| rmse | candidate_minus_baseline | -0.00441638 | 0.0115357 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0197071 | 0.00124038 | 1000 |

| condition | passed |
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

### OLD_F→A_J / hindcast / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0815173 | 0.109811 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0932734 | 0.182799 | 1000 |
| median_r | candidate_minus_baseline | -0.0625548 | 0.0110222 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.028169 | 0.140845 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00140451 | 0.000968748 | 1000 |
| rmse | candidate_minus_baseline | -0.00541807 | 0.00474867 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0175507 | -0.00201908 | 1000 |

| condition | passed |
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

### OLD_F→A_J / hindcast / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0800045 | 0.0951507 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0898168 | 0.179772 | 1000 |
| median_r | candidate_minus_baseline | -0.062546 | 0.00915411 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0.0273973 | 0.136986 | 1000 |
| log_rmse | candidate_minus_baseline | -0.0014352 | 0.000922306 | 1000 |
| rmse | candidate_minus_baseline | -0.0058259 | 0.00449578 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0176972 | -0.00241671 | 1000 |

| condition | passed |
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

### A_B→A_J / training / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0577758 | 0.137237 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0151388 | 0.199953 | 1000 |
| median_r | candidate_minus_baseline | -0.00923329 | 0.0954526 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.0689655 | 0.0344828 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00445767 | -0.00233315 | 1000 |
| rmse | candidate_minus_baseline | -0.010022 | -0.0033435 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.00881028 | -0.00109792 | 1000 |

| condition | passed |
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

### A_B→A_J / training / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0195512 | 0.182189 | 1000 |
| q25_nse | candidate_minus_baseline | -0.0733648 | 0.228248 | 1000 |
| median_r | candidate_minus_baseline | 0.0268808 | 0.112861 | 1000 |
| negative_r_fraction | candidate_minus_baseline | -0.028169 | 0.0848592 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00561948 | -0.00315078 | 1000 |
| rmse | candidate_minus_baseline | -0.0136786 | -0.00513689 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0117805 | -0.00170732 | 1000 |

| condition | passed |
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

### A_B→A_J / hindcast / common_71

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.094397 | 0.155862 | 1000 |
| q25_nse | candidate_minus_baseline | -0.247981 | 0.289615 | 1000 |
| median_r | candidate_minus_baseline | -0.0786094 | 0.0098567 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0.140845 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00239896 | 0.00137399 | 1000 |
| rmse | candidate_minus_baseline | -0.00548857 | 0.00511123 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0168316 | 0.000106231 | 1000 |

| condition | passed |
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

### A_B→A_J / hindcast / all

| metric | difference | low95 | high95 | finite_replicates |
|---|---|---|---|---|
| median_nse | candidate_minus_baseline | -0.0775418 | 0.150663 | 1000 |
| q25_nse | candidate_minus_baseline | -0.252772 | 0.310276 | 1000 |
| median_r | candidate_minus_baseline | -0.0768833 | 0.0197258 | 1000 |
| negative_r_fraction | candidate_minus_baseline | 0 | 0.150685 | 1000 |
| log_rmse | candidate_minus_baseline | -0.00252931 | 0.00131729 | 1000 |
| rmse | candidate_minus_baseline | -0.00583779 | 0.0050964 | 1000 |
| absolute_bias | candidate_minus_baseline | -0.0183885 | -0.00160949 | 1000 |

| condition | passed |
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

## 条件先验前向

完成记录64/64；数值失败保留，没有用观测筛选替补。

| group | quantity | successful_draws | q05 | median | q95 |
|---|---|---|---|---|---|
| B | reach_month_tn_median | 32 | 0.108487 | 0.288842 | 0.646846 |
| B | reach_month_tn_q95 | 32 | 0.748074 | 1.76468 | 4.27012 |
| B | uptake_demand_ratio | 32 | 1 | 1 | 1 |
| B | M_2025_kg_n | 32 | 1.32685e+09 | 2.18019e+09 | 5.29377e+09 |
| B | L_2025_kg_n | 32 | 8.1596e+06 | 1.95074e+07 | 4.48473e+07 |
| B | terminal_2021_2025_kg_n | 32 | 1.19665e+08 | 3.35391e+08 | 7.25694e+08 |
| B | M_loss_2021_2025_kg_n | 32 | 1.02359e+10 | 1.06758e+10 | 1.08934e+10 |
| J | reach_month_tn_median | 32 | 0.106961 | 0.27846 | 0.662953 |
| J | reach_month_tn_q95 | 32 | 0.751608 | 1.79915 | 4.4208 |
| J | uptake_demand_ratio | 32 | 1 | 1 | 1 |
| J | M_2025_kg_n | 32 | 1.32555e+09 | 2.18291e+09 | 5.26445e+09 |
| J | L_2025_kg_n | 32 | 8.36831e+06 | 1.87836e+07 | 4.79169e+07 |
| J | terminal_2021_2025_kg_n | 32 | 1.23234e+08 | 3.21646e+08 | 7.3786e+08 |
| J | M_loss_2021_2025_kg_n | 32 | 1.02124e+10 | 1.06973e+10 | 1.08895e+10 |

没有满足条件的数据；见缺项说明。

以上为实际成功前向的经验分位数，失败数量同时保留；TN按全域reach×月等权，非主评价单站中位数。B/J使用配对的旧参数抽样，逐次差异另表保存。

按名义高斯与硬边界抽样，未惩罚参数固定旧B0；数据和先验的1/nstation共同缩放不被偷偷解释为新概率模型。此处不是完整后验、不是先验可信区间，也不是拟合优劣排序。

## 方法、数据边界与可复核实现

# TN先验预算与源到站传递诊断（A＋B）

2026-09-10用户确认：本轮完成A＋B，河道新方程C审查后另行登记。本轮不自动启动C，不替换主线。

## 固定科学定义

冻结水文、S1氮源、全域共享参数映射、M矿质氮legacy与慢水氮L、河网及水库记忆。所有参数点从1961连续演化至2025。沿用20260910_2的sensitivity输入、训练设计及RAW逐条权重。训练2021–2025共5885条116站；回报2016–2020共3866条73站（71共有站3795条，2早期独有站71条）。早期曾被项目查看，是本轮未拟合的时间外回报，不是盲验证。2025仅11个月，冻结水文/PET延伸及源量沿用限制保留。2023/2024已参与训练。

不增加函数、库存、来源、站点/年份偏置；不重训XGBoost，不重跑旧K/G/J及984次筛选。保留新函数现有8系数，但仅检验校准偏好，不扩展其表达。

## A：四条约束拟合

D=0.5*sum(w*(TN_pred-TN_raw)^2)，R=0.5*sum(prior_residual^2)。预算从旧F0完整参数重算，Rmax=0.4313694506152499；旧F0数据损失=1.4052462780620552。两者不可用报告舍入数替代。原先验均值、SD、硬边界及nstation归一化不变。v_f和全局接触截距无高斯惩罚，只受原边界约束；分组报告，不能称全参数受到同等概率约束。预算为诊断条件，不是可信区间。旧B/F/J在D+lambda*R下解析重新评分，F/J交点约0.710927，不搜索或采用新lambda。

AB0：21参数M0，从旧B0；AB1：21参数M0，从旧J1的前21参数。AJ0：29参数MH，从旧F0；AJ1：29参数MH，从旧J1。四条独立并行。各自min D，s.t. R<=Rmax及原边界。SLSQP SciPy1.17.1，解析完整历史梯度，原variable_scale缩放；ftol1e-10，maxiter500。每路径最多4000实际完整调用、4小时累计执行；失败/中断调用/恢复/终核均计入。约束代数评价另计，不伪装完整模型调用。保留全部合法点，按最小D选，不按D+R。

R容差1e-10*max(1,Rmax)；独立缩放坐标KKT残差<=1e-5，非负乘子及互补残差<=1e-7。保留旧B0/旧F0可行锚点；J最终D<=D_F+1e-8*(1+abs(D_F))。额外检查新B最佳点零扩展；若优于J提示搜索不足，不追加第五条拟合。锚点不是自动收敛解。

分解旧B/F/J及新B/J的原log风险、新log乘数、总log风险及其变化/抵消；零载体水单列。分组先验代价明确。B/J各32次seed1729非拟合条件前向：原名义高斯及硬边界、共享原参数随机数，无高斯先验参数固定旧B0；不依据TN筛样，数值失败保留。称名义先验条件前向，不冒称完整先验或后验。共64次单独登记，不是新增拟合。

## B：固定入河氮的累计传递

对象为旧B0/F0/J1及A选定B/J。固定各自完整历史快/慢入河氮，以230个真实reach、水库捕获/旁路/释放及现有月暴露驱动的每日传播重放。来源reach、快慢入河路径、到站年月分解；损失按发生reach/年月，水库留存/释放单列。标签批处理并落盘，零贡献隐式编码须注明。快慢不是肥料/粪肥标签，也不是真实水龄。嵌套站点不相加当全域输出。

无损参照只令河网和站点算子v_f=0；M原损失、水库记忆、冻结水文、边界及入河N保持。1961完整历史重演。检查非负传递/释放独立于N及有损<=无损。按相同历史输入有损/无损到站量比解释，不用本月到站/本月入河假称存活率。

逐站逐月计算可增加TN、仍低于原始观测的缺口/比例、均值/中心化误差/r/幅度/相位。按真实上下游、训练面积四分位、本站训练流量四分位、水库影响分组；早期沿用阈值，无训练站不从早期标签造阈值。相容/部分不足/明显不足需报告支持反证和连续数值，不设固定1%SSE拟合准入，不启动C。

## 测试与交付

分别核对D/R及梯度；三种已知解约束问题（活跃/非活跃/边界）；运行身份、重复启动、坏缓存、确定性重放、累计预算和最优可行点。原始损失/分组和重现；四个起点合法。训练不可读早期标签；冻结模型后评分。完整历史独立物理核查，守恒相对1e-10、非负原容差、摄取上限。源标签求和、水库跨月/启停、无损边界和v_f=0恢复。

四路径全部保留预测和权重；主评价完整RAW。NSE中位/q25、r中位、负相关/未定义率、站均RMSE/logRMSE/绝对bias、幅度/相位，71/45/2、逐年/逐站、1000次seed1729水系分层配对整站bootstrap。支持进一步验证沿用：NSE+0.10、r+0.05且>0、q25下降<=0.05、负/无定义率不增、RMSE/logRMSE/absbias<=1.05倍、年度摄取/需求比下降<=0.10。训练/回报分开；数据项改善不能替代预测门槛。

必交reports/expert_diagnostic_report.md：目标权衡、参数组代价转移、具体残差/输入/上下游联系、无损仍无法解释的月份、来源传递证据/反证/混杂/样本数、全部方法/预算/源表SHA256。交付简短final_report.md、完整拟合/来源账本/预测、独立completion_audit.json。数值未完成不宣称机制无效。任何终态必须列缺项。

## 运行合同

全程conda sparrow，旧实验及主线只读；本轮只写自身目录。每compute worker一线程，无两个worker上限。CPU数量上限floor(0.9*逻辑核数)，实际按冷启动/完整峰值*1.2、就绪任务派发。CPU/RAM任一90%停派发；RAM90%检查点退让；相关资源<85%恢复。4拟合和独立诊断可并行。总24h，最后2h审计。所有真实调用分类计数；超时保留未完成项，科学/数值失败分开。

独立后台控制器执行全程；目标保持，正常训练静默，原生PID/创建时间核实后等待句柄退出或1小时；退出立即查，一小时最多一次模型巡检，不建重复定时任务、不外发聊天、不自动重启或新科学起点。完成A+B及审计后结束。


NSE逐站至少8条、非恒定实测才定义；r为Pearson，相关未定义保留。logRMSE使用log1p；幅度误差为abs(log(sd_pred/sd_obs))。相位用一年谐波，至少8个不同月份、R²≥0.1且非退化才定义。

输入关联区分站间、站内和训练站点—月份异常，滞后0/1/3/12月固定。展示有符号残差时不混用绝对误差相关；完整表保留两者及样本数、共线性和支持域。水系分层整站bootstrap为已观测站稳定性诊断，不是新流域保证。

来源标签依据入河reach及快/慢路径，精确零稀疏省略；同一标签在水库内跨日跨月保存。到站量是模拟量，观测TN×冻结模拟水量不能称独立实测负荷。各嵌套站不相加当全域输出；同reach不同坝边界不凭reach编号推断上下游。

水库影响分组表示该月启用水库的释放路径可能到达站点，按真实target和上下游拓扑及坝前/坝后边界计算；它不是水库因果效应或精确水库水占比。

## 支持、反证与未辨识问题

A中的数据项降低不自动证明迁移改善；若仅先验代价重新分配，或可行点KKT不合格，必须分别解释。无损参照仍低于实测的月份排除了单独减少非负河道损失补足这些缺口的解释，但不能排除其他入河过程或来源支持问题。

无损传播是固定入河序列与水库释放机制条件下的上限参照，不是整个模型性能上界。若主要改善均值而相关/相位不改善，不称动态机制成立。相关性不定位因果责任；TN预测相近而库存不同仍体现legacy不确定性。

## 给专家的下一轮决策

请分别审查先验组间代价转移、未惩罚的v_f变化、源到站损失位置/时间和剩余缺口。只有证据支持后再登记河道候选C的混合、反应与站点算子；本轮不加参数、不启动C、不改变主线。

## 文献与证据来源

PEST Tikhonov官方帮助区分测量失配和正则化；arXiv:1902.00242讨论结构相关的联合先验；10.1038/nature06686与10.1007/s10533-008-9274-8涉及硝酸盐去除效率，不能直接等同珠江TN。阅读层级见provenance.json。

全部产物、父代来源和SHA256见expert_manifest.json；源标签完整性见source_tag_reconciliation.json；正式完成以completion_audit.json为准。

## 错误与未完成项

无后处理错误。
