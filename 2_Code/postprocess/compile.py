"""SAS sparrow_compile.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as t_dist
except Exception:  # pragma: no cover - optional
    t_dist = None


@dataclass
class CompileCalibrateResult:
    boot_betaest_all: pd.DataFrame | None
    summary_betaest: pd.DataFrame | None
    cov_betaest: pd.DataFrame | None
    resids: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SummarizeCalibrateResult:
    summary_betaest: pd.DataFrame | None
    summary_boot_betaest: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CompilePredictState:
    predict: pd.DataFrame | None = None
    upmonload: pd.DataFrame | None = None
    predict_stats: pd.DataFrame | None = None
    test_data: pd.DataFrame | None = None
    test_predict: pd.DataFrame | None = None
    store: dict[str, pd.DataFrame] = field(default_factory=dict)
    model_parm_predict: pd.DataFrame | None = None
    boot_detail: pd.DataFrame | None = None


@dataclass
class CompilePredictResult:
    state: CompilePredictState
    predict: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SummarizePredictResult:
    predict: pd.DataFrame | None
    summary_predict: pd.DataFrame | None
    lu_yield_percentiles: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _split_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                tokens.extend(item.split())
            else:
                tokens.append(str(item))
        return tokens
    return str(value).split()


def _first_token(value: object) -> str:
    tokens = _split_tokens(value)
    return tokens[0] if tokens else ""


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def _locin(search_list: Iterable[str], master_list: Iterable[str]) -> list[int]:
    master_upper = [item.upper() for item in master_list]
    out: list[int] = []
    for item in search_list:
        if not item:
            out.append(0)
            continue
        target = item.upper()
        try:
            out.append(master_upper.index(target) + 1)
        except ValueError:
            out.append(0)
    return out or [0]


def _choose(cond: np.ndarray, if_true: np.ndarray | float, if_false: np.ndarray | float) -> np.ndarray:
    return np.where(cond, if_true, if_false)


def _append_rows(
    base: pd.DataFrame | None, new_rows: pd.DataFrame | None
) -> pd.DataFrame | None:
    if new_rows is None:
        return base
    if base is None or base.empty:
        return new_rows.copy()
    cols = list(dict.fromkeys(list(base.columns) + list(new_rows.columns)))
    left = base.reindex(columns=cols)
    right = new_rows.reindex(columns=cols)
    return pd.concat([left, right], axis=0, ignore_index=True)


def _reorder_columns(df: pd.DataFrame, priority: Iterable[str]) -> pd.DataFrame:
    priority_list = [col for col in priority if col in df.columns]
    remaining = [col for col in df.columns if col not in priority_list]
    return df.loc[:, priority_list + remaining]


def compile_calibrate(
    *,
    iter_val: int,
    boot_betaest: pd.DataFrame | None,
    summary_betaest: pd.DataFrame | None,
    cov_betaest: pd.DataFrame | None,
    resids: pd.DataFrame | None,
    station_data: pd.DataFrame | None,
    config: MutableMapping[str, object],
    boot_betaest_all: pd.DataFrame | None = None,
) -> CompileCalibrateResult:
    warnings: list[str] = []
    errors: list[str] = []

    boot_betaest_all = _append_rows(boot_betaest_all, boot_betaest)

    summary_out = summary_betaest.copy() if summary_betaest is not None else None
    cov_out = cov_betaest.copy() if cov_betaest is not None else None
    resids_out: pd.DataFrame | None = None

    if iter_val == 0 and resids is not None and station_data is not None:
        staid = _first_token(config.get("staid"))
        if staid and staid in resids.columns and staid in station_data.columns:
            resids_sorted = resids.sort_values(by=[staid], kind="stable", na_position="first")
            station_sorted = station_data.sort_values(by=[staid], kind="stable", na_position="first")
            merged = resids_sorted.merge(station_sorted, on=staid, how="outer", sort=False)
            merged = merged.reset_index(drop=True)
            merged["id"] = np.arange(1, len(merged) + 1)
            station_priority_list = _split_tokens(config.get("station_priority_list"))
            resids_out = _reorder_columns(merged, station_priority_list)
        else:
            resids_out = resids.copy()

    return CompileCalibrateResult(
        boot_betaest_all=boot_betaest_all,
        summary_betaest=summary_out,
        cov_betaest=cov_out,
        resids=resids_out,
        warnings=warnings,
        errors=errors,
    )


def summarize_calibrate(
    *,
    boot_betaest_all: pd.DataFrame | None,
    summary_betaest: pd.DataFrame | None,
    config: Mapping[str, object],
) -> SummarizeCalibrateResult:
    warnings: list[str] = []
    errors: list[str] = []

    if boot_betaest_all is None or summary_betaest is None:
        return SummarizeCalibrateResult(summary_betaest, None, warnings, errors)

    betalst = _split_tokens(config.get("betalst"))
    if not betalst:
        return SummarizeCalibrateResult(summary_betaest, None, warnings, errors)

    parm_beta = boot_betaest_all.loc[boot_betaest_all["iter"] == 0, betalst]
    boot_beta = boot_betaest_all.loc[boot_betaest_all["iter"] > 0, betalst]
    if parm_beta.empty or boot_beta.empty:
        return SummarizeCalibrateResult(summary_betaest, None, warnings, errors)

    parm_beta = parm_beta.iloc[0].to_numpy(dtype=float)
    boot_beta_vals = boot_beta.to_numpy(dtype=float)

    n_lo = int(np.floor((1 - float(config.get("cov_prob", 0)) / 100.0) * boot_beta_vals.shape[0] / 2.0) + 1)
    n_hi = int(
        boot_beta_vals.shape[0]
        - np.floor((1 - float(config.get("cov_prob", 0)) / 100.0) * boot_beta_vals.shape[0])
        - 1
        + n_lo
    )

    def _varcol(mat: np.ndarray) -> np.ndarray:
        if mat.shape[0] <= 1:
            return np.full((mat.shape[1],), np.nan)
        return (np.sum(mat**2, axis=0) - mat.shape[0] * np.mean(mat, axis=0) ** 2) / (mat.shape[0] - 1)

    sorted_boot = np.sort(boot_beta_vals, axis=0)
    idx_lo = max(n_lo - 1, 0)
    idx_hi = max(n_hi - 1, 0)
    b_lo = sorted_boot[idx_lo, :]
    b_hi = sorted_boot[idx_hi, :]
    ci_lo = 2 * parm_beta - b_hi
    ci_hi = 2 * parm_beta - b_lo

    unbias_parm = 2 * parm_beta - np.mean(boot_beta_vals, axis=0)
    boot_std_parm = np.sqrt(_varcol(boot_beta_vals))

    # Noreen-style bootstrap p-value based on mean-centered bootstrap draws.
    mean_boot = np.mean(boot_beta_vals, axis=0)
    boot_pval = np.zeros((boot_beta_vals.shape[1],), dtype=float)
    for i in range(boot_beta_vals.shape[1]):
        thresh = parm_beta[i] + mean_boot[i]
        p = float(np.mean(boot_beta_vals[:, i] >= thresh))
        if parm_beta[i] < 0:
            p = 1.0 - p
        boot_pval[i] = p

    t_stat_boot = np.divide(
        unbias_parm,
        boot_std_parm,
        out=np.full_like(unbias_parm, np.nan, dtype=float),
        where=boot_std_parm > 0,
    )

    std_cols = [f"sd_{p}" for p in betalst]
    df_error = np.nan
    if "df_error" in summary_betaest.columns:
        df_error = float(summary_betaest["df_error"].iloc[0])
    parm_std = np.full((len(betalst),), np.nan, dtype=float)
    if set(std_cols).issubset(summary_betaest.columns):
        parm_std = summary_betaest.loc[summary_betaest.index[0], std_cols].to_numpy(dtype=float)
    t_stat_parm = np.divide(
        parm_beta,
        parm_std,
        out=np.full_like(parm_beta, np.nan, dtype=float),
        where=parm_std > 0,
    )
    if t_dist is not None and np.isfinite(df_error) and df_error > 0:
        p_value_parm = 2 * (1 - t_dist.cdf(np.abs(t_stat_parm), df_error))
    else:
        p_value_parm = np.full_like(t_stat_parm, np.nan, dtype=float)
    p_value_boot = np.full_like(t_stat_boot, np.nan, dtype=float)
    valid_boot = boot_std_parm > 0
    p_value_boot[valid_boot] = boot_pval[valid_boot]

    summary_boot_betaest = pd.DataFrame(
        [
            np.concatenate(
                [
                    unbias_parm,
                    boot_std_parm,
                    ci_lo,
                    ci_hi,
                    boot_pval,
                    t_stat_parm,
                    p_value_parm,
                    t_stat_boot,
                    p_value_boot,
                ]
            )
        ],
        columns=[
            *[f"unbias_{p}" for p in betalst],
            *[f"stdev_{p}" for p in betalst],
            *[f"ci_lo_{p}" for p in betalst],
            *[f"ci_hi_{p}" for p in betalst],
            *[f"boot_pval_{p}" for p in betalst],
            *[f"t_stat_parm_{p}" for p in betalst],
            *[f"p_value_parm_{p}" for p in betalst],
            *[f"t_stat_boot_{p}" for p in betalst],
            *[f"p_value_boot_{p}" for p in betalst],
        ],
    )

    summary_out = summary_betaest.reset_index(drop=True)
    summary_out = pd.concat([summary_out, summary_boot_betaest], axis=1)

    return SummarizeCalibrateResult(summary_out, summary_boot_betaest, warnings, errors)


def compile_predict(
    *,
    iter_val: int,
    jter_val: int,
    n_boot_iter: int,
    predict: pd.DataFrame,
    upmonload: pd.DataFrame | None,
    test_data: pd.DataFrame | None,
    boot_betaest: pd.DataFrame,
    resids: pd.DataFrame,
    summary_betaest: pd.DataFrame,
    indata: pd.DataFrame,
    config: MutableMapping[str, object],
    state: CompilePredictState | None = None,
    ranuni: Callable[[int, int], np.ndarray] | None = None,
) -> CompilePredictResult:
    warnings: list[str] = []
    errors: list[str] = []

    if state is None:
        state = CompilePredictState()

    if_test_predict = _is_yes(config.get("if_test_predict"))
    if_adjust = _is_yes(config.get("if_adjust"))

    if if_test_predict and test_data is not None:
        state.test_data = _append_rows(state.test_data, test_data)

    if iter_val == 0:
        state.predict = predict.copy()
        if if_adjust and upmonload is not None:
            state.upmonload = upmonload.copy()
        return CompilePredictResult(state=state, predict=state.predict, warnings=warnings, errors=errors)

    if if_test_predict:
        state.test_predict = _append_test_predict(state.test_predict, predict, iter_val, jter_val, config)

    state.predict_stats = _compile_stats(
        iter_val=iter_val,
        n_boot_iter=n_boot_iter,
        predict=predict,
        upmonload=upmonload,
        config=config,
        predict_stats=state.predict_stats,
        predict_parametric=state.predict,
        upmonload_parametric=state.upmonload,
        summary_betaest=summary_betaest,
    )

    state, updated_predict = _compile_ci(
        iter_val=iter_val,
        n_boot_iter=n_boot_iter,
        predict=predict,
        upmonload=upmonload,
        boot_betaest=boot_betaest,
        resids=resids,
        summary_betaest=summary_betaest,
        indata=indata,
        config=config,
        state=state,
        ranuni=ranuni,
    )

    return CompilePredictResult(state=state, predict=updated_predict, warnings=warnings, errors=errors)


def _append_test_predict(
    test_predict: pd.DataFrame | None,
    predict: pd.DataFrame,
    iter_val: int,
    jter_val: int,
    config: Mapping[str, object],
) -> pd.DataFrame:
    test_obs = _parse_test_obs(config)
    if test_obs is None:
        return test_predict if test_predict is not None else pd.DataFrame()
    idx = max(test_obs - 1, 0)
    if idx >= len(predict):
        return test_predict if test_predict is not None else pd.DataFrame()

    row = predict.iloc[[idx]].copy()
    row.insert(0, "iter", iter_val)
    row.insert(1, "jter", jter_val)
    if test_predict is None or test_predict.empty:
        return row.reset_index(drop=True)
    return _append_rows(test_predict, row)


def _parse_test_obs(config: Mapping[str, object]) -> int | None:
    tokens = _split_tokens(config.get("test_obs"))
    if not tokens:
        return None
    text = str(tokens[0]).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _compile_stats(
    *,
    iter_val: int,
    n_boot_iter: int,
    predict: pd.DataFrame,
    upmonload: pd.DataFrame | None,
    config: Mapping[str, object],
    predict_stats: pd.DataFrame | None,
    predict_parametric: pd.DataFrame | None,
    upmonload_parametric: pd.DataFrame | None,
    summary_betaest: pd.DataFrame | None,
) -> pd.DataFrame:
    predlst = _split_tokens(config.get("predlst"))
    if not predlst:
        predlst = [col for col in predict.columns if col.startswith("pload_") or col == "del_frac"]

    vpredlst = list(predlst)
    pred_values = predict.loc[:, predlst].to_numpy(dtype=float)

    if _is_yes(config.get("if_adjust")) and upmonload is not None:
        umnames = _build_um_names(_split_tokens(config.get("srcvar")))
        um_values = upmonload.loc[:, umnames].to_numpy(dtype=float)
        pred_values = np.column_stack([pred_values, um_values])
        predlst = predlst + umnames

    n_var = len(vpredlst)
    statnames = ["iter"] + [f"mean_{name}" for name in predlst] + [f"var_{name}" for name in vpredlst]

    if iter_val == 1 or predict_stats is None or predict_stats.empty:
        stats = np.column_stack(
            [
                np.full((pred_values.shape[0], 1), iter_val),
                pred_values,
                np.zeros((pred_values.shape[0], n_var)),
            ]
        )
    else:
        stats_old = predict_stats.loc[:, statnames].to_numpy(dtype=float)
        stats_old[:, 0] = iter_val
        mean_old = stats_old[:, 1 : 1 + pred_values.shape[1]]
        var_old = stats_old[:, 1 + pred_values.shape[1] :]
        mean_new = (1 - 1 / iter_val) * mean_old + pred_values / iter_val
        var_new = (1 - 1 / (iter_val - 1)) * var_old + ((mean_old[:, :n_var] - pred_values[:, :n_var]) ** 2) / iter_val
        stats = np.column_stack([stats_old[:, [0]], mean_new, var_new])

    predict_stats_out = pd.DataFrame(stats, columns=statnames)

    if iter_val == n_boot_iter:
        predict_stats_out = _make_stats(
            predict_stats=predict_stats_out,
            config=config,
            predict_parametric=predict_parametric if predict_parametric is not None else predict,
            upmonload_parametric=upmonload_parametric,
            summary_betaest=summary_betaest,
        )

    return predict_stats_out


def _make_stats(
    *,
    predict_stats: pd.DataFrame,
    config: Mapping[str, object],
    predict_parametric: pd.DataFrame,
    upmonload_parametric: pd.DataFrame | None,
    summary_betaest: pd.DataFrame | None,
) -> pd.DataFrame:
    predlst = _split_tokens(config.get("predlst"))
    if not predlst:
        predlst = [col for col in predict_parametric.columns if col.startswith("pload_") or col == "del_frac"]

    exclude_list = _locin(_split_tokens(config.get("retrans_exclude_list")), predlst)
    exclude_idx = [idx - 1 for idx in exclude_list if idx and idx > 0]

    m_statnames = [f"mean_{name}" for name in predlst]
    v_statnames = [f"var_{name}" for name in predlst]
    statnames = m_statnames + [f"se_{name}" for name in predlst]

    if_adjust = _is_yes(config.get("if_adjust"))
    if if_adjust:
        srcvar = _split_tokens(config.get("srcvar"))
        modnames = _build_mod_names(srcvar)
        umnames = _build_um_names(srcvar)
        m_umnames = [f"mean_{name}" for name in umnames]
        mod_list = _locin(modnames, predlst)
        mod_idx = np.array([idx - 1 for idx in mod_list if idx and idx > 0], dtype=int)
    else:
        umnames = []
        m_umnames = []
        mod_idx = np.array([], dtype=int)

    parm_predict = predict_parametric.loc[:, predlst].to_numpy(dtype=float)
    depvar = _first_token(config.get("depvar"))
    dep = predict_parametric[depvar].to_numpy(dtype=float) if depvar in predict_parametric.columns else np.full((parm_predict.shape[0],), np.nan)

    m_boot_predict = predict_stats.loc[:, m_statnames].to_numpy(dtype=float)
    v_boot_predict = predict_stats.loc[:, v_statnames].to_numpy(dtype=float)

    if if_adjust and umnames:
        m_boot_upmonload = predict_stats.loc[:, m_umnames].to_numpy(dtype=float)
        if upmonload_parametric is not None:
            parm_upmonload = upmonload_parametric.loc[:, umnames].to_numpy(dtype=float)
        else:
            parm_upmonload = np.zeros((parm_predict.shape[0], len(umnames)))
        parm_upmonload = parm_predict[:, mod_idx] - parm_upmonload
        m_boot_upmonload = m_boot_predict[:, mod_idx] - m_boot_upmonload
        m_boot_nonupmonload = (parm_upmonload**2) / _choose(m_boot_upmonload == 0, 1, m_boot_upmonload)
    else:
        m_boot_nonupmonload = None

    m_boot_predict = (parm_predict**2) / _choose(m_boot_predict == 0, 1, m_boot_predict)

    if summary_betaest is not None and not summary_betaest.empty:
        mean_exp_weighted_error = float(summary_betaest["mean_exp_weighted_error"].iloc[0])
        var_exp_weighted_error = float(summary_betaest["var_exp_weighted_error"].iloc[0])
    else:
        mean_exp_weighted_error = float(config.get("mean_exp_weighted_error", np.nan))
        var_exp_weighted_error = float(config.get("var_exp_weighted_error", np.nan))

    cv_model_error = var_exp_weighted_error / (mean_exp_weighted_error**2)
    cv_error = np.full_like(v_boot_predict, cv_model_error, dtype=float)
    if exclude_idx:
        cv_error[:, exclude_idx] = 0

    if if_adjust and mod_idx.size > 0 and m_boot_nonupmonload is not None:
        monitored = ~np.isnan(dep)
        monitored_rows = np.where(monitored)[0]
        if monitored_rows.size > 0:
            cv_error[np.ix_(monitored_rows, mod_idx)] = 0
        scale = (
            _choose(m_boot_nonupmonload < 0, 0, m_boot_nonupmonload)
            / _choose(m_boot_predict[:, mod_idx] == 0, 1, m_boot_predict[:, mod_idx])
        ) ** 2
        cv_error[:, mod_idx] = cv_error[:, mod_idx] * scale

    cv_error = (v_boot_predict * (1 / _choose(parm_predict == 0, 1, parm_predict) ** 2)) + cv_error
    se_boot_predict = m_boot_predict * np.sqrt(cv_error)

    boot_stats = np.column_stack([m_boot_predict, se_boot_predict])
    return pd.DataFrame(boot_stats, columns=statnames)


def _build_mod_names(srcvar: Iterable[str]) -> list[str]:
    return [f"pload_{name}" for name in ["total", *srcvar]]


def _build_um_names(srcvar: Iterable[str]) -> list[str]:
    return [f"umpload_{name}" for name in ["total", *srcvar]]


def _compile_ci(
    *,
    iter_val: int,
    n_boot_iter: int,
    predict: pd.DataFrame,
    upmonload: pd.DataFrame | None,
    boot_betaest: pd.DataFrame,
    resids: pd.DataFrame,
    summary_betaest: pd.DataFrame,
    indata: pd.DataFrame,
    config: MutableMapping[str, object],
    state: CompilePredictState,
    ranuni: Callable[[int, int], np.ndarray] | None,
) -> tuple[CompilePredictState, pd.DataFrame | None]:
    predlst = _split_tokens(config.get("predlst"))
    if not predlst:
        predlst = [col for col in predict.columns if col.startswith("pload_") or col == "del_frac"]

    if ranuni is None:
        raise ValueError("ranuni function is required for compile_ci.")
    seed_2 = _to_int(config.get("seed_2"), 0)
    rand_draw = ranuni(seed_2, len(predict))

    n_store = _to_int(config.get("n_store"), 0)
    n_low = _to_int(config.get("n_low"), 0)

    if iter_val <= n_store:
        state = _compile_ci_set_storage(
            iter_val=iter_val,
            predlst=predlst,
            predict=predict,
            upmonload=upmonload,
            boot_betaest=boot_betaest,
            resids=resids,
            summary_betaest=summary_betaest,
            indata=indata,
            config=config,
            state=state,
            rand_draw=rand_draw,
        )

    if iter_val >= n_store and n_store > 0:
        for i_pred, pred_var in enumerate(predlst, start=1):
            if iter_val == n_store:
                state = _compile_ci_sort_storage(
                    pred_var=pred_var,
                    predlst=predlst,
                    config=config,
                    state=state,
                )
            else:
                state = _compile_ci_add_data(
                    pred_var=pred_var,
                    predlst=predlst,
                    predict=predict,
                    upmonload=upmonload,
                    boot_betaest=boot_betaest,
                    resids=resids,
                    summary_betaest=summary_betaest,
                    indata=indata,
                    config=config,
                    state=state,
                    rand_draw=rand_draw,
                    i_pred=i_pred,
                    last_i_pred=len(predlst),
                )

            if iter_val == n_boot_iter:
                if i_pred == 1 or state.model_parm_predict is None:
                    state.model_parm_predict = _set_model_parm_predict(
                        predlst=predlst,
                        summary_betaest=summary_betaest,
                        indata=indata,
                        config=config,
                        predict_parametric=state.predict,
                        upmonload_parametric=state.upmonload,
                    )
                if not (_skip_del_frac_bounds(pred_var, config)):
                    state.predict = _compile_ci_get_bounds(
                        pred_var=pred_var,
                        n_low=n_low,
                        predlst=predlst,
                        config=config,
                        state=state,
                    )

    return state, state.predict


def _skip_del_frac_bounds(pred_var: str, config: Mapping[str, object]) -> bool:
    target = _first_token(config.get("target"))
    return (not target) and pred_var.upper() == "DEL_FRAC"


def _compile_ci_set_storage(
    *,
    iter_val: int,
    predlst: list[str],
    predict: pd.DataFrame,
    upmonload: pd.DataFrame | None,
    boot_betaest: pd.DataFrame,
    resids: pd.DataFrame,
    summary_betaest: pd.DataFrame,
    indata: pd.DataFrame,
    config: Mapping[str, object],
    state: CompilePredictState,
    rand_draw: np.ndarray,
) -> CompilePredictState:
    pred_values_unadj = predict.loc[:, predlst].to_numpy(dtype=float)

    errors = resids["boot_resid"].to_numpy(dtype=float)
    mean_exp_weighted_error = float(boot_betaest["mean_exp_weighted_error"].iloc[0])

    depvar = _first_token(config.get("depvar"))
    dep = indata[depvar].to_numpy(dtype=float) if depvar in indata.columns else np.full((pred_values_unadj.shape[0],), np.nan)

    rand = np.asarray(rand_draw, dtype=float).reshape(-1)
    if rand.shape[0] != pred_values_unadj.shape[0]:
        raise ValueError("rand_draw length mismatch in _compile_ci_set_storage.")
    idx = np.ceil(len(errors) * rand).astype(int)
    idx[idx < 1] = 1
    boot_errors_column = np.exp(errors[idx - 1])
    boot_errors = np.tile(1.0 / (boot_errors_column * mean_exp_weighted_error), (pred_values_unadj.shape[1], 1)).T

    exclude_list = _locin(_split_tokens(config.get("retrans_exclude_list")), predlst)
    exclude_idx = [i - 1 for i in exclude_list if i and i > 0]
    if exclude_idx:
        boot_errors[:, exclude_idx] = 1

    pred_values = pred_values_unadj.copy()

    if _is_yes(config.get("if_adjust")):
        srcvar = _split_tokens(config.get("srcvar"))
        modnames = _build_mod_names(srcvar)
        mod_list = _locin(modnames, predlst)
        mod_idx = np.array([i - 1 for i in mod_list if i and i > 0], dtype=int)
        if mod_idx.size > 0:
            parm_pload = state.predict.loc[:, modnames].to_numpy(dtype=float)
            parm_upmonload = state.upmonload.loc[:, _build_um_names(srcvar)].to_numpy(dtype=float)
            parm_mewe = float(summary_betaest["mean_exp_weighted_error"].iloc[0])
            boot_upmonload = upmonload.loc[:, _build_um_names(srcvar)].to_numpy(dtype=float) if upmonload is not None else np.zeros_like(parm_upmonload)

            parm_upmonload = parm_upmonload / _choose(parm_pload == 0, 1, parm_pload)
            monitored = ~np.isnan(dep)
            monitored_rows = np.where(monitored)[0]
            if monitored_rows.size > 0:
                boot_errors[np.ix_(monitored_rows, mod_idx)] = 1
            boot_errors[:, mod_idx] = boot_errors[:, mod_idx] * (
                (1 + parm_upmonload * (parm_mewe - 1))
                / (1 + parm_upmonload * (parm_mewe * mean_exp_weighted_error * boot_errors[:, mod_idx] - 1))
            )
            pred_values[:, mod_idx] = pred_values[:, mod_idx] + boot_upmonload * (mean_exp_weighted_error - 1)

    pred_values = pred_values * boot_errors

    for i_var, var in enumerate(predlst):
        colname = f"value_{iter_val}"
        values = pred_values[:, [i_var]]
        if iter_val == 1 or var not in state.store:
            state.store[var] = pd.DataFrame(values, columns=[colname])
        else:
            existing = state.store[var]
            state.store[var] = pd.concat(
                [existing, pd.DataFrame(values, columns=[colname])],
                axis=1,
                ignore_index=False,
            )

    state = _update_boot_detail_initial(
        iter_val=iter_val,
        predlst=predlst,
        pred_values_unadj=pred_values_unadj,
        boot_errors_column=boot_errors_column,
        dep=dep,
        upmonload=upmonload,
        config=config,
        state=state,
    )

    return state


def _compile_ci_sort_storage(
    *,
    pred_var: str,
    predlst: list[str],
    config: Mapping[str, object],
    state: CompilePredictState,
) -> CompilePredictState:
    if pred_var not in state.store:
        return state

    test_obs = _parse_test_obs(config)
    if _is_yes(config.get("if_test_predict")) and test_obs is not None:
        idx = max(test_obs - 1, 0)
        store_df = state.store[pred_var]
        if idx < len(store_df):
            boot_vals = store_df.iloc[idx].to_numpy(dtype=float)
            if state.test_predict is not None and not state.test_predict.empty:
                n = min(len(state.test_predict), len(boot_vals))
                col = f"boot_{pred_var}"
                if col not in state.test_predict.columns:
                    state.test_predict[col] = np.nan
                state.test_predict.loc[state.test_predict.index[:n], col] = boot_vals[:n]

    if _skip_del_frac_bounds(pred_var, config):
        return state

    values = state.store[pred_var].to_numpy(dtype=float)
    for i in range(values.shape[0]):
        row = values[i, :]
        if np.all(~np.isnan(row)):
            values[i, :] = np.sort(row)
        else:
            values[i, :] = np.nan
    state.store[pred_var] = pd.DataFrame(values, columns=state.store[pred_var].columns)
    return state


def _compile_ci_add_data(
    *,
    pred_var: str,
    predlst: list[str],
    predict: pd.DataFrame,
    upmonload: pd.DataFrame | None,
    boot_betaest: pd.DataFrame,
    resids: pd.DataFrame,
    summary_betaest: pd.DataFrame,
    indata: pd.DataFrame,
    config: Mapping[str, object],
    state: CompilePredictState,
    rand_draw: np.ndarray,
    i_pred: int,
    last_i_pred: int,
) -> CompilePredictState:
    if pred_var not in state.store:
        return state

    depvar = _first_token(config.get("depvar"))
    dep = indata[depvar].to_numpy(dtype=float) if depvar in indata.columns else np.full((len(predict),), np.nan)

    new_value_unadj = predict[pred_var].to_numpy(dtype=float)
    errors = resids["boot_resid"].to_numpy(dtype=float)
    mean_exp_weighted_error = float(boot_betaest["mean_exp_weighted_error"].iloc[0])

    rand = np.asarray(rand_draw, dtype=float).reshape(-1)
    if rand.shape[0] != new_value_unadj.shape[0]:
        raise ValueError("rand_draw length mismatch in _compile_ci_add_data.")
    idx = np.ceil(len(errors) * rand).astype(int)
    idx[idx < 1] = 1
    boot_error = np.exp(errors[idx - 1])

    new_value = new_value_unadj.copy()

    retrans_raw = str(config.get("retrans_exclude_list") or "")
    if pred_var.upper() not in retrans_raw.upper():
        boot_errors = 1.0 / (boot_error * mean_exp_weighted_error)
        if _is_yes(config.get("if_adjust")) and _is_cum_pload(pred_var):
            umname = f"um{pred_var}"
            parm_pload = state.predict[pred_var].to_numpy(dtype=float)
            parm_upmonload = state.upmonload[umname].to_numpy(dtype=float) if state.upmonload is not None and umname in state.upmonload.columns else np.zeros_like(parm_pload)
            parm_mewe = float(summary_betaest["mean_exp_weighted_error"].iloc[0])
            um_pred = upmonload[umname].to_numpy(dtype=float) if upmonload is not None and umname in upmonload.columns else np.zeros_like(parm_pload)

            monitored = ~np.isnan(dep)
            boot_errors[monitored] = 1
            parm_upmonload = parm_upmonload / _choose(parm_pload == 0, 1, parm_pload)
            boot_errors = boot_errors * (
                (1 + parm_upmonload * (parm_mewe - 1))
                / (1 + parm_upmonload * (parm_mewe * mean_exp_weighted_error * boot_errors - 1))
            )
            new_value = new_value + um_pred * (mean_exp_weighted_error - 1)
        new_value = new_value * boot_errors

    if _is_yes(config.get("if_test_predict")):
        test_obs = _parse_test_obs(config)
        if test_obs is not None and state.test_predict is not None and not state.test_predict.empty:
            idx = max(test_obs - 1, 0)
            if idx < len(new_value):
                col = f"boot_{pred_var}"
                state.test_predict.loc[state.test_predict.index[-1], col] = new_value[idx]

    value = state.store[pred_var].to_numpy(dtype=float)
    n_low = _to_int(config.get("n_low"), 0)
    n_hi = _to_int(config.get("n_hi"), 0)
    n_drop = n_low + 1

    keep_idx = [i for i in range(n_low + n_hi + 1) if i != n_drop - 1]
    idx_low = n_low - 1
    idx_drop = n_drop - 1

    if idx_low >= 0 and idx_drop >= 0:
        cond = (value[:, idx_low] > new_value) | (new_value > value[:, idx_drop])
        cond = cond & ~np.isnan(value[:, idx_drop])
        rows = np.where(cond)[0]
        for row in rows:
            y = np.concatenate([value[row, :], [new_value[row]]])
            if np.isnan(y[-1]):
                value[row, :] = np.nan
            else:
                y_sorted = np.sort(y)
                value[row, :] = y_sorted[keep_idx]

    state.store[pred_var] = pd.DataFrame(value, columns=state.store[pred_var].columns)

    state = _update_boot_detail_incremental(
        pred_var=pred_var,
        predlst=predlst,
        new_value_unadj=new_value_unadj,
        boot_error=boot_error,
        dep=dep,
        upmonload=upmonload,
        config=config,
        state=state,
        i_pred=i_pred,
        last_i_pred=last_i_pred,
    )

    return state


def _compile_ci_get_bounds(
    *,
    pred_var: str,
    n_low: int,
    predlst: list[str],
    config: Mapping[str, object],
    state: CompilePredictState,
) -> pd.DataFrame | None:
    if state.predict is None or state.model_parm_predict is None or state.predict_stats is None:
        return state.predict
    if pred_var not in state.store:
        return state.predict

    model_pred = state.model_parm_predict[pred_var].to_numpy(dtype=float)
    mean_col = f"mean_{pred_var}"
    se_col = f"se_{pred_var}"

    mean_vals = state.predict_stats[mean_col].to_numpy(dtype=float)
    se_vals = state.predict_stats[se_col].to_numpy(dtype=float)

    store = state.store[pred_var]
    col_low = f"value_{n_low}"
    col_hi = f"value_{n_low + 1}"
    if col_low not in store.columns or col_hi not in store.columns:
        return state.predict
    val_low = store[col_low].to_numpy(dtype=float)
    val_hi = store[col_hi].to_numpy(dtype=float)

    ci_lo = np.where(val_hi != 0, (model_pred**2) / val_hi, 0)
    ci_hi = np.where(val_low != 0, (model_pred**2) / val_low, 0)

    out = state.predict.copy()
    out[mean_col] = mean_vals
    out[se_col] = se_vals
    out[f"ci_lo_{pred_var}"] = ci_lo
    out[f"ci_hi_{pred_var}"] = ci_hi
    return out


def _set_model_parm_predict(
    *,
    predlst: list[str],
    summary_betaest: pd.DataFrame,
    indata: pd.DataFrame,
    config: Mapping[str, object],
    predict_parametric: pd.DataFrame | None,
    upmonload_parametric: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if predict_parametric is None:
        return None

    pred_values = predict_parametric.loc[:, predlst].to_numpy(dtype=float)
    exclude_list = _locin(_split_tokens(config.get("retrans_exclude_list")), predlst)
    exclude_idx = [i - 1 for i in exclude_list if i and i > 0]

    mean_exp_weighted_error = float(summary_betaest["mean_exp_weighted_error"].iloc[0])

    inv_mean_boot_error = np.full_like(pred_values, 1.0 / mean_exp_weighted_error, dtype=float)
    if exclude_idx:
        inv_mean_boot_error[:, exclude_idx] = 1

    if _is_yes(config.get("if_adjust")) and upmonload_parametric is not None:
        srcvar = _split_tokens(config.get("srcvar"))
        modnames = _build_mod_names(srcvar)
        mod_list = _locin(modnames, predlst)
        mod_idx = np.array([i - 1 for i in mod_list if i and i > 0], dtype=int)
        depvar = _first_token(config.get("depvar"))
        dep = indata[depvar].to_numpy(dtype=float) if depvar in indata.columns else np.full((pred_values.shape[0],), np.nan)
        monitored = ~np.isnan(dep)
        monitored_rows = np.where(monitored)[0]
        if monitored_rows.size > 0:
            inv_mean_boot_error[np.ix_(monitored_rows, mod_idx)] = 1

    pred_values = pred_values * inv_mean_boot_error

    if _is_yes(config.get("if_adjust")) and upmonload_parametric is not None:
        modnames = _build_mod_names(_split_tokens(config.get("srcvar")))
        mod_list = _locin(modnames, predlst)
        mod_idx = np.array([i - 1 for i in mod_list if i and i > 0], dtype=int)
        umnames = _build_um_names(_split_tokens(config.get("srcvar")))
        upmonload_vals = upmonload_parametric.loc[:, umnames].to_numpy(dtype=float)
        pred_values[:, mod_idx] = pred_values[:, mod_idx] + upmonload_vals * (
            (mean_exp_weighted_error - 1) / mean_exp_weighted_error
        )

    return pd.DataFrame(pred_values, columns=predlst)


def _is_cum_pload(pred_var: str) -> bool:
    parts = pred_var.upper().split("_")
    if not parts or parts[0] != "PLOAD":
        return False
    if len(parts) >= 2 and parts[1] in {"INC", "ND"}:
        return False
    return True


def _update_boot_detail_initial(
    *,
    iter_val: int,
    predlst: list[str],
    pred_values_unadj: np.ndarray,
    boot_errors_column: np.ndarray,
    dep: np.ndarray,
    upmonload: pd.DataFrame | None,
    config: Mapping[str, object],
    state: CompilePredictState,
) -> CompilePredictState:
    boot_detail_reaches = str(config.get("boot_detail_reaches") or "").strip()
    if not boot_detail_reaches:
        return state

    waterid = _first_token(config.get("waterid"))
    if not waterid or state.predict is None:
        return state
    water_vals = state.predict[waterid].to_numpy(dtype=float)
    select = _select_reaches(water_vals, boot_detail_reaches)
    if select is None or select.size == 0:
        return state

    boot_detail_predvars = str(config.get("boot_detail_predvars") or "").upper()
    sel_pred_indices, varnames = _boot_detail_var_selection(predlst, config, include_del_frac=True)

    values = pred_values_unadj[select][:, sel_pred_indices]
    if _is_yes(config.get("if_adjust")) and (
        (not boot_detail_predvars) or ("CUM" in boot_detail_predvars)
    ):
        srcvar = _split_tokens(config.get("srcvar"))
        umnames = [name for name in _build_um_names(srcvar) if upmonload is not None and name in upmonload.columns]
        if umnames:
            umvals = upmonload.loc[:, umnames].to_numpy(dtype=float)[select]
            values = np.column_stack([values, umvals])
            varnames = [*varnames, *umnames]

    detail = pd.DataFrame(values, columns=varnames)
    detail.insert(0, "boot_error", boot_errors_column[select])
    detail.insert(0, "if_mon", (~np.isnan(dep[select])).astype(int))
    detail.insert(0, waterid, water_vals[select])
    detail.insert(0, "iter", iter_val)

    if _is_yes(config.get("if_adjust")):
        if "INC" not in boot_detail_predvars:
            detail.loc[detail["if_mon"] == 1, "boot_error"] = 1
        if boot_detail_predvars and "CUM" not in boot_detail_predvars:
            detail["if_mon"] = 0
    else:
        detail["if_mon"] = 0

    state.boot_detail = _append_rows(state.boot_detail, detail)
    return state


def _update_boot_detail_incremental(
    *,
    pred_var: str,
    predlst: list[str],
    new_value_unadj: np.ndarray,
    boot_error: np.ndarray,
    dep: np.ndarray,
    upmonload: pd.DataFrame | None,
    config: Mapping[str, object],
    state: CompilePredictState,
    i_pred: int,
    last_i_pred: int,
) -> CompilePredictState:
    pending_key = "__boot_detail_adddata__"
    if i_pred == 1:
        state.store.pop(pending_key, None)

    def _finalize_pending(curr: CompilePredictState) -> CompilePredictState:
        if i_pred != last_i_pred:
            return curr
        pending = curr.store.pop(pending_key, None)
        if isinstance(pending, pd.DataFrame) and not pending.empty:
            curr.boot_detail = _append_rows(curr.boot_detail, pending)
        return curr

    boot_detail_reaches = str(config.get("boot_detail_reaches") or "").strip()
    if not boot_detail_reaches:
        return _finalize_pending(state)

    waterid = _first_token(config.get("waterid"))
    if not waterid or state.predict is None:
        return _finalize_pending(state)
    water_vals = state.predict[waterid].to_numpy(dtype=float)
    select = _select_reaches(water_vals, boot_detail_reaches)
    if select is None or select.size == 0:
        return _finalize_pending(state)

    boot_detail_predvars = str(config.get("boot_detail_predvars") or "").upper()
    _, varnames = _boot_detail_var_selection(predlst, config, include_del_frac=True)
    pred_upper = pred_var.upper()
    if pred_upper not in [name.upper() for name in varnames]:
        return _finalize_pending(state)

    pending = state.store.get(pending_key)
    if not isinstance(pending, pd.DataFrame) or pending.empty:
        pending = pd.DataFrame(
            {
                "iter": np.full((select.size,), _to_int(config.get("iter"), 0)),
                waterid: water_vals[select],
                "if_mon": (~np.isnan(dep[select])).astype(int),
                "boot_error": boot_error[select],
            }
        )

        if _is_yes(config.get("if_adjust")):
            if "INC" not in boot_detail_predvars:
                pending.loc[pending["if_mon"] == 1, "boot_error"] = 1
            if boot_detail_predvars and "CUM" not in boot_detail_predvars:
                pending["if_mon"] = 0
        else:
            pending["if_mon"] = 0

    pending[pred_var] = new_value_unadj[select]
    if _is_yes(config.get("if_adjust")) and _is_cum_pload(pred_var):
        um_name = f"um{pred_var}"
        if upmonload is not None and um_name in upmonload.columns:
            pending[um_name] = upmonload[um_name].to_numpy(dtype=float)[select]

    if _is_yes(config.get("if_adjust")):
        if "INC" not in boot_detail_predvars:
            pending.loc[pending["if_mon"] == 1, "boot_error"] = 1
        if boot_detail_predvars and "CUM" not in boot_detail_predvars:
            pending["if_mon"] = 0
    else:
        pending["if_mon"] = 0

    state.store[pending_key] = pending
    return _finalize_pending(state)


def _select_reaches(values: np.ndarray, spec: str) -> np.ndarray | None:
    if spec.upper() == "ALL":
        return np.arange(len(values))
    tokens = _split_tokens(spec)
    if not tokens:
        return None
    selected: list[int] = []
    for token in tokens:
        try:
            val = float(token)
        except ValueError:
            continue
        selected.extend(list(np.where(values == val)[0]))
    return np.array(selected, dtype=int)


def _boot_detail_var_selection(
    predlst: list[str],
    config: Mapping[str, object],
    *,
    include_del_frac: bool,
) -> tuple[list[int], list[str]]:
    pred_upper = [p.upper() for p in predlst]
    sel_idx: list[int] = []
    varnames: list[str] = []
    if include_del_frac and "DEL_FRAC" in pred_upper:
        idx = pred_upper.index("DEL_FRAC")
        sel_idx.append(idx)
        varnames.append(predlst[idx])

    boot_detail_predvars = str(config.get("boot_detail_predvars") or "").upper()
    srcvar = _split_tokens(config.get("srcvar"))
    if not boot_detail_predvars or "CUM" in boot_detail_predvars:
        modnames = _build_mod_names(srcvar)
        for name in modnames:
            if name in predlst:
                sel_idx.append(predlst.index(name))
                varnames.append(name)
    if "INC" in boot_detail_predvars:
        modinc = [f"pload_inc_{name}" for name in ["total", *srcvar]]
        for name in modinc:
            if name in predlst:
                sel_idx.append(predlst.index(name))
                varnames.append(name)
    return sel_idx, varnames


def summarize_predict(
    *,
    predict: pd.DataFrame,
    ancillary: pd.DataFrame,
    config: Mapping[str, object],
) -> SummarizePredictResult:
    warnings: list[str] = []
    errors: list[str] = []

    waterid = _first_token(config.get("waterid"))
    if not waterid or waterid not in predict.columns or waterid not in ancillary.columns:
        errors.append("waterid is missing from predict or ancillary.")
        return SummarizePredictResult(predict, None, None, warnings, errors)

    predict_sorted = predict.sort_values(by=[waterid], kind="stable", na_position="first")
    ancillary_sorted = ancillary.sort_values(by=[waterid], kind="stable", na_position="first")
    merged = predict_sorted.merge(ancillary_sorted, on=waterid, how="left", sort=False)

    inc_area = _first_token(config.get("inc_area"))
    tot_area = _first_token(config.get("tot_area"))
    mean_flow = _first_token(config.get("mean_flow"))
    hydseq = _first_token(config.get("hydseq"))
    predict_prefix = str(config.get("predict_prefix") or "")

    pload_total = f"{predict_prefix}pload_total"
    pload_inc_total = f"{predict_prefix}pload_inc_total"
    del_frac_col = f"{predict_prefix}del_frac"

    total_yield = np.where(
        merged[tot_area] > 0,
        merged[pload_total] / (merged[tot_area] * 100),
        np.nan,
    )
    merged["total_yield"] = total_yield

    inc_total_yield = np.where(
        merged[inc_area] > 0,
        merged[pload_inc_total] / (merged[inc_area] * 100),
        np.nan,
    )
    merged["inc_total_yield"] = inc_total_yield

    adjust_units = float(config.get("adjust_units", 1))
    adjust_conc_units = float(config.get("adjust_conc_units", 1))
    ndt_per_year = float(config.get("ndt_per_year", 1))
    concentration = np.where(
        merged[mean_flow] > 0,
        adjust_units
        * adjust_conc_units
        * ndt_per_year
        * merged[pload_total]
        / (893.585 * merged[mean_flow]),
        np.nan,
    )
    if _is_yes(config.get("if_flow_units_metric")):
        concentration = concentration * 893.585 / 3153.6
    merged["concentration"] = concentration

    map_del_frac = np.where(merged[del_frac_col] == 0, np.nan, 100 * merged[del_frac_col])
    merged["map_del_frac"] = map_del_frac

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(_summary_rows(merged["total_yield"], 1))
    summary_rows.extend(_summary_rows(merged["inc_total_yield"], 2))
    summary_rows.extend(_summary_rows(merged["concentration"], 3))
    summary_rows.extend(_summary_rows(merged["map_del_frac"], 4))

    srcvar = _split_tokens(config.get("srcvar"))
    varnum = 4
    for src in srcvar:
        varnum += 1
        inc_src = f"{predict_prefix}pload_inc_{src}"
        share = np.where(
            merged[pload_inc_total] > 0,
            100 * merged[inc_src] / merged[pload_inc_total],
            np.nan,
        )
        merged[f"sh_{src}"] = share
        summary_rows.extend(_summary_rows(share, varnum))

    summary_predict = _summarize_distribution(summary_rows, srcvar, config)

    if _is_yes(config.get("if_distribute_yield_by_land_use")) and "LU_class" in merged.columns:
        lu_df = merged.loc[merged["LU_class"].notna(), ["LU_class", "inc_total_yield"]]
        lu_yield_percentiles = _summarize_lu_yield(lu_df)
    else:
        lu_yield_percentiles = None

    reach_priority_list = _split_tokens(config.get("reach_priority_list"))
    merged = _reorder_columns(merged, reach_priority_list)

    if hydseq and hydseq in merged.columns:
        merged = merged.sort_values(by=[hydseq], kind="stable", na_position="first")
    output_schema = str(config.get("predict_output_schema") or "").strip().lower()
    if output_schema == "sas_q_adjust":
        merged = _project_sas_q_adjust(merged, config, pload_total)
    elif output_schema in {"sas_tp", "sas_tn"}:
        merged = _project_sas_template(merged, config)

    return SummarizePredictResult(merged, summary_predict, lu_yield_percentiles, warnings, errors)


def _project_sas_template(
    merged: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    template_path = str(config.get("sas_predict_template") or "").strip()
    if not template_path:
        return merged

    path = Path(template_path)
    if not path.exists():
        return merged

    template_cols = list(pd.read_csv(path, nrows=0).columns)
    out = merged.copy()

    # Normalize duplicated merge artifacts before projection.
    if "load" not in out.columns:
        if "load_x" in out.columns:
            out["load"] = out["load_x"]
        elif "load_y" in out.columns:
            out["load"] = out["load_y"]
    if "STAID" not in out.columns and "staid" in out.columns:
        out["STAID"] = out["staid"]

    lower_to_col = {str(c).lower(): c for c in out.columns}
    proj_cols: dict[str, pd.Series | float] = {}
    for col in template_cols:
        src = lower_to_col.get(str(col).lower())
        if src is not None:
            proj_cols[col] = out[src]
        else:
            proj_cols[col] = np.nan

    return pd.DataFrame(proj_cols, index=out.index)


def _project_sas_q_adjust(
    merged: pd.DataFrame,
    config: Mapping[str, object],
    pload_total_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=merged.index)

    waterid = _first_token(config.get("waterid"))

    if "comid" in merged.columns:
        out["comid"] = merged["comid"]
    elif waterid and waterid in merged.columns:
        out["comid"] = merged[waterid]
    else:
        out["comid"] = np.nan

    for col in ("year", "quarter", "period"):
        out[col] = merged[col] if col in merged.columns else np.nan

    out["PLOAD_TOTAL"] = merged[pload_total_col] if pload_total_col in merged.columns else np.nan

    if "Q_calc_cfs" in merged.columns:
        q_calc = merged["Q_calc_cfs"]
    elif "Q_calc" in merged.columns:
        q_calc = merged["Q_calc"]
    elif "flow_cfs" in merged.columns:
        q_calc = merged["flow_cfs"]
    else:
        q_calc = np.nan
    out["Q_calc"] = q_calc

    if "Q_ma_cfs" in merged.columns:
        q_ma = merged["Q_ma_cfs"]
    elif "Q_ma" in merged.columns:
        q_ma = merged["Q_ma"]
    elif "maflow_cfs" in merged.columns:
        q_ma = merged["maflow_cfs"]
    else:
        q_ma = np.nan
    out["Q_ma"] = q_ma

    out["difQ"] = pd.to_numeric(out["Q_calc"], errors="coerce") - pd.to_numeric(
        out["PLOAD_TOTAL"], errors="coerce"
    )

    out = out.loc[:, ["comid", "year", "quarter", "period", "PLOAD_TOTAL", "Q_calc", "difQ", "Q_ma"]]
    out = out.sort_values(by=["comid", "year", "quarter", "period"], kind="stable", na_position="first")
    out = out.reset_index(drop=True)
    return out


def _summary_rows(values: pd.Series, varnum: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for val in values:
        if pd.notna(val):
            out.append({"varnum": varnum, "value": float(val)})
    return out


def _summarize_distribution(
    rows: list[dict[str, object]], srcvar: list[str], config: Mapping[str, object]
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = df.groupby("varnum")["value"]
    summary = grouped.agg(
        n_watersheds="count",
        mean="mean",
        std="std",
        p_10=lambda x: np.percentile(x, 10),
        p_25=lambda x: np.percentile(x, 25),
        p_50=lambda x: np.percentile(x, 50),
        p_75=lambda x: np.percentile(x, 75),
        p_90=lambda x: np.percentile(x, 90),
        range=lambda x: np.max(x) - np.min(x),
    ).reset_index()

    varnames = _build_varnames(srcvar, config)
    summary = summary.merge(varnames, on="varnum", how="left")
    summary = summary.drop(columns=["varnum"])
    return summary


def _build_varnames(srcvar: list[str], config: Mapping[str, object]) -> pd.DataFrame:
    numerator_units = str(config.get("numerator_load_units") or "")
    concentration_units = str(config.get("concentration_units") or "")
    names = [
        "Upstream Yield (" + numerator_units + "/ha/yr)",
        "Incremental Yield (" + numerator_units + "/ha/yr)",
        "Flow-Weighted Conc (" + concentration_units + ")",
        "Reach Flux Share Delivered (%)",
    ]
    for src in srcvar:
        names.append(f"{src.upper()} Source Share (%)")
    return pd.DataFrame({"varnum": np.arange(1, len(names) + 1), "varname": names})


def _summarize_lu_yield(lu_df: pd.DataFrame) -> pd.DataFrame:
    grouped = lu_df.groupby("LU_class")["inc_total_yield"]
    summary = grouped.agg(
        inc_total_yield_n="count",
        inc_total_yield_range=lambda x: np.max(x) - np.min(x),
        inc_total_yield_p10=lambda x: np.percentile(x, 10),
        inc_total_yield_p25=lambda x: np.percentile(x, 25),
        inc_total_yield_p50=lambda x: np.percentile(x, 50),
        inc_total_yield_p75=lambda x: np.percentile(x, 75),
        inc_total_yield_p90=lambda x: np.percentile(x, 90),
    ).reset_index()
    return summary
