"""SAS revise_covbetaest_sercor.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as t_dist
except Exception:  # pragma: no cover - optional
    t_dist = None

from ..cli import FileTableStore


@dataclass
class SerCorrResult:
    revcov_betaest: pd.DataFrame | None
    revbetastat: pd.DataFrame | None
    rho: float | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def revise_covbetaest_sercor(
    config: MutableMapping[str, object],
    *,
    resids: pd.DataFrame | None = None,
    cov_betaest: pd.DataFrame | None = None,
    summary_betaest: pd.DataFrame | None = None,
    results_store: FileTableStore | None = None,
) -> SerCorrResult:
    """Revise coefficient covariance matrix to account for serial correlation."""
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    if "if_error" not in config:
        config["if_error"] = "no"

    home_results = _first_path(config, "home_results", "results_dir")
    if results_store is None:
        results_store = FileTableStore(home_results, config.get("results_tables"))

    if resids is None:
        if results_store.exists("resids"):
            resids = results_store.read("resids")
        else:
            error("resids table not found - stop processing.")
    if cov_betaest is None:
        if results_store.exists("cov_betaest"):
            cov_betaest = results_store.read("cov_betaest")
        else:
            error("cov_betaest table not found - stop processing.")
    if summary_betaest is None:
        if results_store.exists("summary_betaest"):
            summary_betaest = results_store.read("summary_betaest")
        else:
            error("summary_betaest table not found - stop processing.")

    if config.get("if_error") == "yes":
        return SerCorrResult(None, None, None, warnings, errors)

    if resids is None or cov_betaest is None or summary_betaest is None:
        return SerCorrResult(None, None, None, warnings, errors)

    siteid = _first_token(config.get("siteid"))
    period = _first_token(config.get("period"))
    if not siteid or siteid not in resids.columns:
        error("siteid column not found in resids - stop processing.")
        return SerCorrResult(None, None, None, warnings, errors)
    if not period or period not in resids.columns:
        error("period column not found in resids - stop processing.")
        return SerCorrResult(None, None, None, warnings, errors)
    if "ln_resid" not in resids.columns:
        error("ln_resid column not found in resids - stop processing.")
        return SerCorrResult(None, None, None, warnings, errors)

    gradients = _select_gradients(cov_betaest.columns.tolist())
    if not gradients:
        error("No gradient columns found in cov_betaest - stop processing.")
        return SerCorrResult(None, None, None, warnings, errors)

    missing_grad = [g for g in gradients if g not in resids.columns]
    if missing_grad:
        error(
            "The following gradient columns are missing from resids: "
            + " ".join(missing_grad)
            + "."
        )
        return SerCorrResult(None, None, None, warnings, errors)

    resids_df = resids.loc[resids["ln_resid"].notna()].copy()
    resids_df = resids_df.sort_values(
        by=[siteid, period], kind="stable", na_position="first"
    ).reset_index(drop=True)

    period_values = pd.to_numeric(resids_df[period], errors="coerce")
    if period_values.isna().any():
        error("period values must be numeric for serial correlation correction.")
        return SerCorrResult(None, None, None, warnings, errors)
    resids_df[period] = period_values.astype(float)

    if_mean_pd = _is_yes(config.get("ifmean_pd"))
    regdata = _make_regdata(resids_df, siteid, period, if_mean_pd)

    rho = _estimate_rho(regdata)

    grads = resids_df.loc[:, gradients].to_numpy(dtype=float)
    cov0 = cov_betaest.loc[:, gradients].to_numpy(dtype=float)

    mse0 = _get_scalar(summary_betaest, ["MSE", "mse"])
    df_error = _get_scalar(summary_betaest, ["df_error", "DF_ERROR"])
    beta = summary_betaest.loc[:, gradients].iloc[0].to_numpy(dtype=float)

    veet = _build_veet(resids_df, siteid, period, float(rho))

    loc_cnstrn = np.where(np.diag(cov0) == 0)[0]
    loc_uncnstrn = np.array(
        [i for i in range(cov0.shape[0]) if i not in set(loc_cnstrn)],
        dtype=int,
    )

    cov_new = np.zeros_like(cov0)
    if loc_uncnstrn.size:
        gtg = grads[:, loc_uncnstrn].T @ grads[:, loc_uncnstrn]
        invgtg = np.linalg.pinv(gtg)
        cov_new[np.ix_(loc_uncnstrn, loc_uncnstrn)] = invgtg

    covbeta_revised = mse0 * cov_new @ grads.T @ veet @ grads @ cov_new

    std_err0 = np.sqrt(np.diag(cov0))
    std_err_rev = np.sqrt(np.diag(covbeta_revised))

    t0 = np.full_like(std_err0, np.nan, dtype=float)
    t_rev = np.full_like(std_err_rev, np.nan, dtype=float)
    pval0 = np.full_like(std_err0, np.nan, dtype=float)
    pval_rev = np.full_like(std_err_rev, np.nan, dtype=float)

    if loc_uncnstrn.size:
        t0[loc_uncnstrn] = beta[loc_uncnstrn] / std_err0[loc_uncnstrn]
        t_rev[loc_uncnstrn] = beta[loc_uncnstrn] / std_err_rev[loc_uncnstrn]
        if t_dist is not None and not np.isnan(df_error):
            pval0[loc_uncnstrn] = 2 * (
                1 - t_dist.cdf(np.abs(t0[loc_uncnstrn]), df_error)
            )
            pval_rev[loc_uncnstrn] = 2 * (
                1 - t_dist.cdf(np.abs(t_rev[loc_uncnstrn]), df_error)
            )

    revbetastat = pd.DataFrame(
        {
            "parameter": gradients,
            "estimate": beta,
            "std_err0": std_err0,
            "t0": t0,
            "pval0": pval0,
            "std_err_rev": std_err_rev,
            "t_rev": t_rev,
            "pval_rev": pval_rev,
        }
    )
    revcov_betaest = pd.DataFrame(covbeta_revised, columns=gradients)

    results_store.write("revcov_betaest", revcov_betaest)
    results_store.write("revbetastat", revbetastat)

    return SerCorrResult(
        revcov_betaest=revcov_betaest,
        revbetastat=revbetastat,
        rho=float(rho) if rho is not None else None,
        warnings=warnings,
        errors=errors,
    )


def _select_gradients(columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        if col and col[:1].upper() == "B":
            out.append(col)
    return out


def _first_token(value: object) -> str:
    tokens = _split_tokens(value)
    return tokens[0] if tokens else ""


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


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _first_path(config: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        val = config.get(key)
        if val:
            return str(val)
    return None


def _make_regdata(
    resids: pd.DataFrame, siteid: str, period: str, ifmean_pd: bool
) -> pd.DataFrame:
    out = resids.loc[:, [siteid, period, "ln_resid"]].copy()
    if ifmean_pd:
        out["l1ln_resid"] = out.groupby(siteid, sort=False)["ln_resid"].shift(1)
        first_period = out.groupby(siteid, sort=False)[period].transform("first")
        last_lag = out.groupby(siteid, sort=False)["l1ln_resid"].transform("last")
        out.loc[out[period] == first_period, "l1ln_resid"] = last_lag
        return out
    out["l1ln_resid"] = out.groupby(siteid, sort=False)["ln_resid"].shift(1)
    return out.loc[out.groupby(siteid, sort=False).cumcount() > 0].copy()


def _estimate_rho(regdata: pd.DataFrame) -> float:
    valid = regdata["ln_resid"].notna() & regdata["l1ln_resid"].notna()
    x = regdata.loc[valid, "l1ln_resid"].to_numpy(dtype=float)
    y = regdata.loc[valid, "ln_resid"].to_numpy(dtype=float)
    denom = float(np.sum(x**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _build_veet(
    resids: pd.DataFrame, siteid: str, period: str, rho: float
) -> np.ndarray:
    n = len(resids)
    veet = np.zeros((n, n), dtype=float)
    start = 0
    for _, group in resids.groupby(siteid, sort=False):
        periods = group[period].to_numpy(dtype=float)
        dif = np.abs(periods.reshape(-1, 1) - periods.reshape(1, -1)) / 10.0
        vblock = np.power(rho, dif)
        end = start + len(group)
        veet[start:end, start:end] = vblock
        start = end
    return veet


def _get_scalar(df: pd.DataFrame, names: list[str]) -> float:
    for name in names:
        if name in df.columns:
            return float(df[name].iloc[0])
    return float("nan")
