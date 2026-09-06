"""Pre-registered settings for the 20260727_13 nonlinear-recession experiment."""

REFERENCE_RUN_ID = "20260727_6"
INPUT_ANCHOR_RUN_ID = "20260727_6"
Q78_OBSERVATION_START_YEAR = 2010
Q78_OBSERVATION_END_YEAR = 2022

# The only fitted-model change in this run. The exponent is fixed before
# execution and is not selected with 2019-2022 observations.
LINEAR_RECESSION_REFERENCE_EXPONENT = 1.0
NONLINEAR_RECESSION_EXPONENT = 1.5

# Frozen engineering effect sizes from 20260727_6/config/metric_spec.yaml.
NSELOG_EFFECT_SIZE = 0.005
KGE_EFFECT_SIZE = 0.005
ABS_PBIAS_EFFECT_SIZE_PCT_POINTS = 0.5
LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS = 5.0
LOW_FLOW_LOG_RMSE_EFFECT_SIZE = 0.02
