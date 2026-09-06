"""SAS sparrow_calibrate.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, sqrt
import re
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize
    from scipy.stats import norm, t as t_dist
except Exception:  # pragma: no cover - scipy optional
    Bounds = None
    LinearConstraint = None
    least_squares = None
    minimize = None
    norm = None
    t_dist = None


@dataclass
class CalibrationResult:
    boot_betaest: pd.DataFrame | None
    summary_betaest: pd.DataFrame | None
    cov_betaest: pd.DataFrame | None
    resids: pd.DataFrame | None
    error_report: pd.DataFrame | None
    test_resids: pd.DataFrame | None
    temp_beta: pd.DataFrame | None
    temp_rc: pd.DataFrame | None
    n_npos_flux: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_SAS_MISSING_SENTINEL = -np.inf
_NORMAL_DIST = NormalDist()


def _norm_cdf(x: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if norm is not None:
        return np.asarray(norm.cdf(arr), dtype=float)
    return np.vectorize(_NORMAL_DIST.cdf)(arr).astype(float)


def _norm_ppf(x: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    if norm is not None:
        return np.asarray(norm.ppf(arr), dtype=float)
    return np.vectorize(_NORMAL_DIST.inv_cdf)(arr).astype(float)


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


def _to_float_list(values: Iterable[object]) -> list[float | None]:
    out: list[float | None] = []
    for item in values:
        if item is None:
            out.append(None)
            continue
        text = str(item).strip()
        if text == "" or text == ".":
            out.append(None)
            continue
        try:
            out.append(float(text))
        except ValueError:
            out.append(None)
    return out


def _parse_float_mapping(value: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if value is None:
        return out
    if isinstance(value, Mapping):
        for key, raw_value in value.items():
            if key is None or raw_value is None:
                continue
            try:
                out[str(key).strip()] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.update(_parse_float_mapping(item))
        return out
    text = str(value).strip()
    if not text:
        return out
    for token in re.split(r"[\s,;]+", text):
        if not token or "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        try:
            out[key] = float(raw_value)
        except ValueError:
            continue
    return out


def _eval_condition(expr: str, df: pd.DataFrame) -> pd.Series:
    expr = expr.strip()
    expr = re.sub(r"(?<![\w.])\.(?![\w.])", "_MISSING_", expr)
    expr = re.sub(r"\bne\b", "!=", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\beq\b", "==", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bgt\b", ">", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bge\b", ">=", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\blt\b", "<", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\ble\b", "<=", expr, flags=re.IGNORECASE)
    expr = expr.replace("^=", "!=")
    expr = expr.replace("<>", "!=")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = re.sub(r"\band\b", "&", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "|", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bnot\b", "~", expr, flags=re.IGNORECASE)
    temp = df.copy()
    numeric_cols = temp.select_dtypes(include=[np.number]).columns
    temp[numeric_cols] = temp[numeric_cols].fillna(_SAS_MISSING_SENTINEL)
    local_dict: dict[str, object] = {col: temp[col] for col in temp.columns}
    local_dict["_MISSING_"] = _SAS_MISSING_SENTINEL
    return pd.eval(expr, local_dict=local_dict, engine="python")


def _translate_iml(expr: str) -> str:
    binary_term = r"(?:-?\([^()]+\)|-?[A-Za-z_][\w\[\],:]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    pattern_max = re.compile(rf"(?P<a>{binary_term})\s*<>\s*(?P<b>{binary_term})")
    pattern_min = re.compile(rf"(?P<a>{binary_term})\s*><\s*(?P<b>{binary_term})")
    while pattern_max.search(expr):
        expr = pattern_max.sub(r"__sas_max(\g<a>, \g<b>)", expr)
    while pattern_min.search(expr):
        expr = pattern_min.sub(r"__sas_min(\g<a>, \g<b>)", expr)
    expr = expr.replace("##", "**")
    expr = expr.replace("#", "*")
    expr = re.sub(r"\bdata\s*\[,\s*([A-Za-z_]\w*)\s*\]", r"data[:, \1]", expr)
    expr = re.sub(r"\bbeta\s*\[,\s*([A-Za-z_]\w*)\s*\]", r"beta[\1]", expr)
    return expr


def _eval_spec(spec: object, context: Mapping[str, object], default: np.ndarray) -> np.ndarray:
    if spec is None:
        return default
    if isinstance(spec, np.ndarray):
        return spec
    if callable(spec):
        return np.asarray(spec(context))
    text = str(spec).strip()
    if not text:
        return default
    expr = _translate_iml(text)
    local_dict = dict(context)
    local_dict.update(
        {
            "np": np,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "floor": np.floor,
            "ceil": np.ceil,
            "repeat": lambda x, n: np.tile(x, (int(n), 1)),
            "j": lambda r, c, v: np.full((int(r), int(c)), v),
            "choose": lambda cond, if_true, if_false: np.where(cond, if_true, if_false),
            "probit": _norm_ppf,
            "__sas_max": np.maximum,
            "__sas_min": np.minimum,
        }
    )
    return np.asarray(eval(expr, {"__builtins__": {}}, local_dict))


def _parse_dlvdsgn(
    value: object, nsrc: int, ndlv: int
) -> np.ndarray:
    """Parse SAS dlvdsgn text into an nsrc x ndlv matrix."""
    if not value or nsrc <= 0 or ndlv <= 0:
        return np.empty((0, 0), dtype=float)
    rows = [row.strip() for row in str(value).split(",") if row.strip()]
    matrix = np.zeros((nsrc, ndlv), dtype=float)
    for i, row in enumerate(rows[:nsrc]):
        parts = row.split()
        for j, part in enumerate(parts[:ndlv]):
            try:
                matrix[i, j] = float(part)
            except ValueError:
                matrix[i, j] = 0.0
    return matrix


def _build_linear_constraints(
    constraints: pd.DataFrame | None, betalst: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (A, op, b) where op in {-1,0,1} for >=,=,<=."""
    if constraints is None or constraints.empty or not betalst:
        return None
    required = {"constraint_op", "constraint_val"}
    if not required.issubset(constraints.columns):
        return None
    coef_cols = [c for c in betalst if c in constraints.columns]
    if not coef_cols:
        return None
    A = constraints.loc[:, coef_cols].to_numpy(dtype=float)
    b = constraints["constraint_val"].to_numpy(dtype=float)
    op = constraints["constraint_op"].to_numpy(dtype=float)
    # Expand to full beta order (columns missing in constraints treated as 0).
    if len(coef_cols) != len(betalst):
        full = np.zeros((A.shape[0], len(betalst)), dtype=float)
        col_map = {name: i for i, name in enumerate(coef_cols)}
        for i, name in enumerate(betalst):
            idx = col_map.get(name)
            if idx is not None:
                full[:, i] = A[:, idx]
        A = full
    return A, op, b


def _numerical_jacobian(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Finite-difference Jacobian of residuals."""
    x = np.asarray(x0, dtype=float)
    r0 = np.asarray(residuals_fn(x), dtype=float).reshape(-1)
    nobs = r0.shape[0]
    npar = x.shape[0]
    jac = np.zeros((nobs, npar), dtype=float)
    for j in range(npar):
        step = np.zeros_like(x)
        step[j] = eps * max(abs(x[j]), 1.0)
        rp = np.asarray(residuals_fn(x + step), dtype=float).reshape(-1)
        rm = np.asarray(residuals_fn(x - step), dtype=float).reshape(-1)
        jac[:, j] = (rp - rm) / (2.0 * step[j])
    return jac


def _matrix_root_psd(matrix: np.ndarray) -> np.ndarray:
    """Return a stable square-root for symmetric positive semi-definite matrices."""
    sym = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals))


def _constraint_residual_jacobian(
    x: np.ndarray,
    linear_constraints: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    weight: float = 1e3,
) -> tuple[np.ndarray, np.ndarray]:
    if linear_constraints is None:
        return np.empty((0,), dtype=float), np.empty((0, x.shape[0]), dtype=float)
    A, op, b = linear_constraints
    op = op.astype(int)
    ax = A @ x

    rows_r: list[np.ndarray] = []
    rows_j: list[np.ndarray] = []
    scale = sqrt(weight)

    eq = np.where(op == 0)[0]
    if eq.size > 0:
        rows_r.append(scale * (ax[eq] - b[eq]))
        rows_j.append(scale * A[eq, :])

    le = np.where(op == 1)[0]
    if le.size > 0:
        viol = ax[le] - b[le]
        active = viol > 0
        if np.any(active):
            rows_r.append(scale * viol[active])
            rows_j.append(scale * A[le[active], :])

    ge = np.where(op == -1)[0]
    if ge.size > 0:
        viol = b[ge] - ax[ge]
        active = viol > 0
        if np.any(active):
            rows_r.append(scale * viol[active])
            rows_j.append(-scale * A[ge[active], :])

    if not rows_r:
        return np.empty((0,), dtype=float), np.empty((0, x.shape[0]), dtype=float)
    return np.concatenate(rows_r), np.vstack(rows_j)


def _clip_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def _sanitize_residuals(resid: np.ndarray) -> np.ndarray:
    out = np.asarray(resid, dtype=float).reshape(-1)
    out[~np.isfinite(out)] = 1e12
    return out


def _least_squares_fallback(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    linear_constraints: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    max_iter: int = 1500,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bounded least-squares fallback used when SciPy optimizers are unavailable."""
    x = _clip_to_bounds(np.asarray(x0, dtype=float).copy(), lb, ub)
    npar = x.shape[0]
    damp = 1e-3
    rc = -8

    def objective(xv: np.ndarray) -> float:
        r = _sanitize_residuals(residuals_fn(xv))
        r_c, _ = _constraint_residual_jacobian(xv, linear_constraints)
        if r_c.size > 0:
            r = np.concatenate([r, r_c])
        return float(0.5 * np.dot(r, r))

    prev_obj = objective(x)
    for _ in range(max_iter):
        r = _sanitize_residuals(residuals_fn(x))
        j = _numerical_jacobian(lambda z: _sanitize_residuals(residuals_fn(z)), x)
        r_c, j_c = _constraint_residual_jacobian(x, linear_constraints)
        if r_c.size > 0:
            r = np.concatenate([r, r_c])
            j = np.vstack([j, j_c])

        g = j.T @ r
        h = j.T @ j + damp * np.eye(npar)
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(h) @ g

        if np.linalg.norm(step) <= xtol * (1 + np.linalg.norm(x)):
            rc = 1
            break

        accepted = False
        alpha = 1.0
        trial_obj = prev_obj
        trial_x = x
        while alpha >= 1e-6:
            cand = _clip_to_bounds(x + alpha * step, lb, ub)
            cand_obj = objective(cand)
            if np.isfinite(cand_obj) and cand_obj <= prev_obj:
                trial_x = cand
                trial_obj = cand_obj
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            damp *= 10.0
            if damp > 1e12:
                rc = 0
                break
            continue

        x = trial_x
        if abs(prev_obj - trial_obj) <= ftol * (1 + prev_obj):
            rc = 1
            prev_obj = trial_obj
            break

        prev_obj = trial_obj
        damp = max(damp * 0.7, 1e-8)
    else:
        rc = -8

    jac = _numerical_jacobian(lambda z: _sanitize_residuals(residuals_fn(z)), x)
    return x, jac, rc


def _solve_least_squares_problem(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    use_scipy: bool,
    linear_constraints: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    solver: str = "least_squares",
    max_nfev: int = 1500,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-8,
    max_iter: int = 15000,
    nlp_print: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    x0 = _clip_to_bounds(np.asarray(x0, dtype=float), lb, ub)
    if use_scipy:
        if linear_constraints is None and solver != "trust-constr":
            result = least_squares(
                residuals_fn,
                x0,
                bounds=(lb, ub),
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=0 if nlp_print <= 0 else 2,
            )
            estimate = np.asarray(result.x, dtype=float)
            jac_seed = np.asarray(result.jac, dtype=float)
            if result.success:
                rc = 1
            elif int(getattr(result, "status", -1)) == 0:
                rc = -8
            else:
                rc = 0
            return estimate, jac_seed, rc

        lin_cons: list[LinearConstraint] = []
        if linear_constraints is not None:
            A, op, b = linear_constraints
            op = op.astype(int)
            eq = op == 0
            le = op == 1
            ge = op == -1
            if np.any(eq):
                lin_cons.append(LinearConstraint(A[eq, :], b[eq], b[eq]))
            if np.any(le):
                lin_cons.append(LinearConstraint(A[le, :], -np.inf, b[le]))
            if np.any(ge):
                lin_cons.append(LinearConstraint(A[ge, :], b[ge], np.inf))

        def objective(beta: np.ndarray) -> float:
            resid = residuals_fn(beta)
            if np.any(~np.isfinite(resid)):
                return float("inf")
            return float(0.5 * np.dot(resid, resid))

        result = minimize(
            objective,
            x0,
            method="trust-constr",
            bounds=Bounds(lb, ub),
            constraints=lin_cons,
            options={
                "maxiter": max_iter,
                "xtol": xtol,
                "verbose": 0 if nlp_print <= 0 else 2,
            },
        )
        estimate = np.asarray(result.x, dtype=float)
        if bool(getattr(result, "success", False)):
            rc = 1
        else:
            msg = str(getattr(result, "message", "")).lower()
            rc = -8 if "max" in msg else 0
        jac_seed = _numerical_jacobian(residuals_fn, estimate)
        return estimate, jac_seed, rc

    estimate, jac_seed, rc = _least_squares_fallback(
        residuals_fn,
        x0,
        lb,
        ub,
        linear_constraints=linear_constraints,
        max_iter=max_nfev,
        ftol=ftol,
        xtol=xtol,
    )
    if rc > 0:
        rc = 1
    else:
        rc = -8 if rc == -8 else 0
    return estimate, jac_seed, rc


def _reduce_linear_constraints(
    linear_constraints: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    fixed_beta: np.ndarray,
    free_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if linear_constraints is None:
        return None
    free_idx = np.asarray(free_idx, dtype=int)
    if free_idx.size == 0:
        return None
    A, op, b = linear_constraints
    free_set = set(free_idx.tolist())
    fixed_idx = np.array(
        [idx for idx in range(fixed_beta.shape[0]) if idx not in free_set],
        dtype=int,
    )
    A_free = A[:, free_idx]
    rhs = b.copy()
    if fixed_idx.size > 0:
        rhs = rhs - A[:, fixed_idx] @ fixed_beta[fixed_idx]
    keep_rows = np.any(np.abs(A_free) > 0, axis=1)
    if not np.any(keep_rows):
        return None
    return A_free[keep_rows, :], op[keep_rows], rhs[keep_rows]


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def _swilk(x_in: np.ndarray, n: int) -> tuple[float, float]:
    x_in = x_in.astype(float)
    n1 = x_in.shape[0]
    if n1 < 3:
        return (np.nan, np.nan)
    rng = np.nanmax(x_in) - np.nanmin(x_in)
    ncens = n - n1
    if n > 5000 or (n1 < n and n < 20) or n < n1 or n > 5 * n1 or rng == 0 or n1 < 3:
        return (np.nan, np.nan)

    c12 = np.array(
        [
            [0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056],
            [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633],
        ]
    )
    c34 = np.array(
        [
            [0.5440, -0.39978, 0.025054, -0.0006714],
            [1.3822, -0.77857, 0.062767, -0.0020322],
        ]
    )
    c56 = np.array(
        [
            [-1.5861, -0.31082, -0.083751, 0.0038915],
            [-0.4803, -0.082676, 0.0030302, 0.0],
        ]
    )
    c7 = np.array([0.164, 0.533])
    c8 = np.array([0.1736, 0.315])
    c9 = np.array([0.256, -0.00635])
    g = np.array([-2.273, 0.459])
    z = _norm_ppf([0.9, 0.95, 0.99])
    zm = z.mean()
    zss = np.sum((z - zm) ** 2)
    bf1 = 0.8378
    xx90 = 0.556
    xx95 = 0.622
    pi6 = 1.909859
    stqr = 1.047198
    small = 1e-19
    log_n = np.log(n)
    delta = ncens / n

    x = -x_in
    order = np.argsort(x, kind="mergesort")
    x = (-x_in / rng)[order]

    n2 = int(floor(n / 2))
    if_n_even = (((-1) ** n) > 0)

    if n == 3:
        a = np.array([np.sqrt(0.5)], dtype=float)
    else:
        a = _norm_ppf(((np.arange(1, n2 + 1) - 0.375) / (n + 0.25)))
        summ2 = 2 * np.sum(a**2)
        i1 = 1 + (n > 5)
        a_1 = _poly(c12[:i1, :], 1 / np.sqrt(n)) - a[:i1] / np.sqrt(summ2)
        fac = np.sqrt((summ2 - 2 * np.sum(a[:i1] ** 2)) / (1 - 2 * np.sum(a_1**2)))
        a[:i1] = a_1
        a[i1:n2] = -a[i1:n2] / fac

    if if_n_even:
        a = np.concatenate([a, -a[n2 - 1 :: -1]])[:n1]
    else:
        a = np.concatenate([a, [0], -a[n2 - 1 :: -1]])[:n1]

    a = a - np.mean(a)
    x = x - np.mean(x)
    ssa = np.sum(a**2)
    ssx = np.sum(x**2)
    sax = np.dot(a, x)
    sr_ssa_ssx = np.sqrt(ssa * ssx)
    w = 1 - (sr_ssa_ssx - sax) * (sr_ssa_ssx + sax) / (ssa * ssx)

    if n == 3:
        w_scalar = float(np.asarray(w).reshape(-1)[0])
        return (w_scalar, float(pi6 * (np.arcsin(np.sqrt(w_scalar)) - stqr)))

    y = np.log(1 - w)
    if n <= 11:
        gamma = float(np.asarray(_poly(g, n)).reshape(-1)[0])
        if y >= gamma:
            return (w, small)
        y = -np.log(gamma - y)
        parm = np.asarray(_poly(c34, n)).reshape(-1)
    else:
        parm = np.asarray(_poly(c56, log_n)).reshape(-1)
    m = float(parm[0])
    s = float(np.exp(parm[1]))

    if ncens > 0:
        ld = -np.log(delta)
        bf = 1 + log_n * bf1
        z_f = z + bf * np.concatenate(
            [
                _poly(c7, xx90**log_n),
                _poly(c8, xx95**log_n),
                _poly(c9, log_n),
            ]
        ) * ld
        z_fm = z_f.mean()
        zsd = np.dot(z, (z_f - z_fm)) / zss
        zbar = z_fm - zsd * zm
        m = m + zbar * s
        s = s * zsd

    pval = float(np.asarray(_norm_cdf((m - y) / s)).reshape(-1)[0])
    w_scalar = float(np.asarray(w).reshape(-1)[0])
    return (w_scalar, pval)


def _poly(c: np.ndarray, x: float | np.ndarray) -> np.ndarray:
    coeff = np.asarray(c, dtype=float)
    if coeff.ndim == 1:
        coeff = coeff.reshape(1, -1)

    x_arr = np.asarray(x, dtype=float)
    exponents = np.arange(coeff.shape[1], dtype=int)

    if x_arr.ndim == 0:
        powers = np.power(float(x_arr), exponents)
        return coeff @ powers

    x_flat = x_arr.reshape(-1)
    powers = np.power.outer(x_flat, exponents).T
    out = coeff @ powers
    return out.reshape((coeff.shape[0],) + x_arr.shape)


def calibrate(
    indata: pd.DataFrame,
    config: MutableMapping[str, object],
    betahat0: pd.DataFrame,
    *,
    iter_val: int = 0,
    jter_val: int = 0,
    constraints: pd.DataFrame | None = None,
    ranuni: Callable[[int, int], np.ndarray] | None = None,
    rannor: Callable[[int, int], np.ndarray] | None = None,
    cov_estimate_input: np.ndarray | None = None,
) -> CalibrationResult:
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    def empty_result() -> CalibrationResult:
        return CalibrationResult(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            warnings,
            errors,
        )

    if "if_error" not in config:
        config["if_error"] = "no"

    if_test_calibrate = _is_yes(config.get("if_test_calibrate"))
    use_scipy = least_squares is not None and minimize is not None
    if not use_scipy:
        warn("SciPy not available; using internal fallback optimizer.")

    linear_constraints = _build_linear_constraints(
        constraints, _split_tokens(config.get("betalst"))
    )
    if constraints is not None and linear_constraints is None:
        warn("constraint_file was provided but could not be parsed; constraints not applied.")

    datalst = _split_tokens(config.get("datalst"))
    if not datalst:
        error("datalst is missing in config; run makemacros first.")
        return empty_result()

    selection = config.get("calibrate_selection_criteria")
    if selection:
        mask = pd.Series(_eval_condition(str(selection), indata), index=indata.index)
        data_df = indata.loc[mask].copy()
    else:
        data_df = indata.copy()

    data_df = data_df.loc[:, [c for c in datalst if c in data_df.columns]]
    data = data_df.to_numpy()

    ls_weight = _first_token(config.get("ls_weight"))
    depvar = _first_token(config.get("depvar"))

    weights_all = None
    if ls_weight and ls_weight in data_df.columns:
        weights_all = data_df[ls_weight].to_numpy()

    if weights_all is None:
        error("ls_weight column not found in indata for calibration.")
        return empty_result()

    if depvar and depvar in data_df.columns:
        obsloc = np.where(~np.isnan(data_df[depvar].to_numpy()))[0]
    else:
        obsloc = np.array([], dtype=int)
    nobs = len(obsloc)

    if nobs == 0:
        error("No observed depvar values found for calibration.")
        return empty_result()

    weights = weights_all[obsloc]
    if weights.shape[0] != nobs:
        error("Weights vector length does not match number of observations.")
        return empty_result()

    inv_mean = np.mean(np.divide(1.0, weights, out=np.zeros_like(weights), where=weights != 0))
    weights = weights * inv_mean

    boot_weights = np.ones_like(weights)
    if iter_val > 0 and not _is_yes(config.get("if_parm_bootstrap")):
        seed_1 = int(config.get("seed_1", 0))
        if ranuni is None:
            error("ranuni function required for nonparametric bootstrap.")
            return empty_result()
        samp = np.ceil(nobs * ranuni(seed_1, nobs)).astype(int)
        samp[samp < 1] = 1
        counts = np.bincount(samp, minlength=nobs + 1)[1:]
        boot_weights = counts.astype(float)

    weights = weights * boot_weights

    makecol = config.get("makecol") or {}
    def _idx(name: str) -> int:
        vals = makecol.get(name) or []
        return int(vals[0] - 1) if vals and vals[0] > 0 else -1

    def _idx_list(name: str) -> np.ndarray:
        vals = makecol.get(name) or []
        arr = np.array([v - 1 for v in vals if v and v > 0], dtype=int)
        return arr

    jwaterid = _idx("jwaterid")
    jstaid = _idx("jstaid")
    jfnode = _idx("jfnode")
    jtnode = _idx("jtnode")
    jfrac = _idx("jfrac")
    jtarget = _idx("jtarget")
    jiftran = _idx("jiftran")
    jdepvar = _idx("jdepvar")
    jsrcvar = _idx_list("jsrcvar")
    jdlvvar = _idx_list("jdlvvar")
    jdecvar = _idx_list("jdecvar")
    jresvar = _idx_list("jresvar")
    jtotarea = _idx("jtotarea")
    jbsrcvar = _idx_list("jbsrcvar")
    jbdlvvar = _idx_list("jbdlvvar")
    jbdecvar = _idx_list("jbdecvar")
    jbresvar = _idx_list("jbresvar")

    nreach = data.shape[0]
    nnode = int(np.nanmax(data[:, [jfnode, jtnode]])) if jfnode >= 0 and jtnode >= 0 else 0

    n_periods = int(config.get("n_periods", 0) or 0)
    nrch = int(nreach / n_periods) if n_periods else nreach

    catchment_storage_source = str(config.get("catchment_storage_source") or "").strip()
    catchment_storage_source_exclude = _split_tokens(config.get("catchment_storage_source_exclude"))
    catchment_storage_source_exclud0 = _split_tokens(config.get("catchment_storage_source_exclud0"))

    jstoreincldsrcs = np.arange(len(jsrcvar))
    jstoreincldsrcs0 = np.arange(len(jsrcvar))
    if catchment_storage_source_exclude:
        exclude_vals = [int(v) - 1 for v in catchment_storage_source_exclude]
        jstoreincldsrcs = np.array([i for i in jstoreincldsrcs if i not in exclude_vals], dtype=int)
    if catchment_storage_source_exclud0:
        exclude_vals = [int(v) - 1 for v in catchment_storage_source_exclud0]
        jstoreincldsrcs0 = np.array([i for i in jstoreincldsrcs0 if i not in exclude_vals], dtype=int)

    beta0 = betahat0.loc[:, _split_tokens(config.get("betalst"))].iloc[0].to_numpy()

    reach_spec = config.get("reach_decay_specification")
    res_spec = config.get("reservoir_decay_specification")
    incr_spec = config.get("incr_delivery_specification")
    convert_spec = config.get("convert_specification")
    dlvdsgn = _parse_dlvdsgn(
        config.get("dlvdsgn"), int(len(jsrcvar)), int(len(jdlvvar))
    )

    error_rows: list[dict[str, object]] = []
    n_npos_flux_final = 0
    temp_beta = None
    temp_rc = None

    def feval(
        beta: np.ndarray,
        *,
        if_final_pass: bool = False,
        include_test_n_rch: bool = False,
    ) -> np.ndarray:
        nonlocal n_npos_flux_final
        beta = beta.astype(float)
        rchdcayf = _eval_spec(
            reach_spec,
            {
                "data": data,
                "beta": beta,
                "jdecvar": jdecvar,
                "jbdecvar": jbdecvar,
            },
            np.ones((nreach,)),
        )
        resdcayf = _eval_spec(
            res_spec,
            {
                "data": data,
                "beta": beta,
                "jresvar": jresvar,
                "jbresvar": jbresvar,
            },
            np.ones((nreach,)),
        )
        rchdcayf = np.asarray(rchdcayf).reshape(-1)
        resdcayf = np.asarray(resdcayf).reshape(-1)

        carryf = data[:, jfrac] * rchdcayf * resdcayf if jfrac >= 0 else np.zeros((nreach,))
        incdcayf = (rchdcayf ** 0.5) * resdcayf

        incddsrc = np.zeros((nreach,))
        if jsrcvar.size > 0:
            if incr_spec:
                del2strm_factors = _eval_spec(
                    incr_spec,
                    {
                        "data": data,
                        "beta": beta,
                        "jbsrcvar": jbsrcvar,
                        "jsrcvar": jsrcvar,
                        "jdlvvar": jdlvvar,
                        "jbdlvvar": jbdlvvar,
                        "dlvdsgn": dlvdsgn,
                        "nreach": nreach,
                    },
                    np.tile(beta[jbsrcvar], (nreach, 1)),
                )
                del2strm_factors = np.asarray(del2strm_factors, dtype=float)
                if del2strm_factors.ndim == 1:
                    del2strm_factors = del2strm_factors.reshape(-1, 1)
                if del2strm_factors.shape[0] != nreach:
                    del2strm_factors = np.tile(del2strm_factors.reshape(1, -1), (nreach, 1))
                del2strm_factors = del2strm_factors * beta[jbsrcvar]
            else:
                del2strm_factors = np.tile(beta[jbsrcvar], (nreach, 1))

            data_src = data[:, jsrcvar].copy()
            if catchment_storage_source and n_periods:
                incdelbysrc = np.zeros_like(data_src)
                if _is_yes(config.get("if_estimate_ic")):
                    if jstoreincldsrcs0.size > 0:
                        incdelbysrc0 = data_src[:, jstoreincldsrcs0] * del2strm_factors[:, jstoreincldsrcs0]
                        incload0 = incdelbysrc0.sum(axis=1)
                        src_idx = int(float(catchment_storage_source)) - 1
                        bt = del2strm_factors[:, src_idx]
                        K = 0.97
                        start0 = 0
                        end0 = nrch
                        slices = []
                        for _ in range(4):
                            b = np.minimum(bt[start0:end0], K)
                            x = incload0[start0:end0]
                            slices.append((b, x))
                            start0 += nrch
                            end0 += nrch
                        b1, X1 = slices[0]
                        b2, X2 = slices[1]
                        b3, X3 = slices[2]
                        b4, X4 = slices[3]
                        pb = np.minimum(b1 * b2 * b3 * b4, K)
                        denom = np.where((1 - pb) == 0, np.nan, (1 - pb))
                        L1 = X1 + (b4 * b3 * b1 * X2 + b4 * b1 * X3 + b1 * X4) / denom
                        L1 = np.nan_to_num(L1, nan=0.0, posinf=0.0, neginf=0.0)
                        data_src[:nrch, src_idx] = L1

                start0 = 0
                end0 = nrch
                for t in range(1, n_periods + 1):
                    incdelbysrc[start0:end0, :] = data_src[start0:end0, :] * del2strm_factors[start0:end0, :]
                    if t < n_periods:
                        start1 = start0 + nrch
                        end1 = end0 + nrch
                        src_idx = int(float(catchment_storage_source)) - 1
                        data_src[start1:end1, src_idx] = incdelbysrc[start0:end0, jstoreincldsrcs].sum(axis=1)
                        start0 = start1
                        end0 = end1
            else:
                incdelbysrc = data_src * del2strm_factors

            incload = incdelbysrc.sum(axis=1)
            incddsrc = incdcayf * incload

        convert = None
        if convert_spec:
            convert = _eval_spec(convert_spec, {"data": data, "beta": beta}, np.ones((nreach,)))
            convert = np.asarray(convert).reshape(-1)

        e = np.zeros((nobs,))
        node = np.zeros((nnode + 1,))
        i_obs = 0
        n_rch = np.zeros((nobs,), dtype=float) if include_test_n_rch else None
        e_flag = 1 if (jbsrcvar.size > 0 and np.all(beta[jbsrcvar] <= 0)) else 0
        n_npos_flux = 0
        for i in range(nreach):
            if n_rch is not None and i_obs < nobs:
                n_rch[i_obs] += 1
            upload = node[int(data[i, jfnode])] if jfnode >= 0 else 0
            rchld = incddsrc[i] + carryf[i] * upload

            if jdepvar >= 0 and not np.isnan(data[i, jdepvar]):
                if rchld <= 0:
                    n_npos_flux += 1
                    if (not include_test_n_rch) and e_flag <= 1:
                        row = {
                            "iter": iter_val,
                            "jter": jter_val,
                            "if_final_est": int(if_final_pass),
                            "staid": data[i, jstaid] if jstaid >= 0 else np.nan,
                            "waterid": data[i, jwaterid] if jwaterid >= 0 else np.nan,
                            "rchld": rchld,
                        }
                        for idx, name in enumerate(_split_tokens(config.get("srcvar"))):
                            row[name] = data[i, jsrcvar[idx]] if idx < len(jsrcvar) else np.nan
                        row["upnode_flux"] = upload
                        row["inc_flux"] = incddsrc[i]
                        row["rch_deliv_factor"] = rchdcayf[i]
                        row["res_deliv_factor"] = resdcayf[i]
                        for idx, name in enumerate(_split_tokens(config.get("betalst"))):
                            row[name] = beta[idx]
                        error_rows.append(row)
                        e_flag = 2
                    if include_test_n_rch:
                        e[i_obs] = 0.0
                    else:
                        rchld = np.nan

                if include_test_n_rch and rchld <= 0:
                    pass
                else:
                    if convert is not None:
                        denom = convert[i] * rchld
                    else:
                        denom = rchld
                    if denom <= 0 or np.isnan(denom):
                        e[i_obs] = np.nan
                    else:
                        e[i_obs] = np.log(data[i, jdepvar] / denom)

                rchld = data[i, jdepvar]
                i_obs += 1

            if jtnode >= 0:
                node[int(data[i, jtnode])] = node[int(data[i, jtnode])] + data[i, jiftran] * rchld

        if if_final_pass:
            n_npos_flux_final = int(n_npos_flux)

        f = np.sqrt(weights) * e
        if if_final_pass:
            lactual = np.log(data[obsloc, jdepvar])
            lpredict = lactual - e
            lpredyld = lpredict - np.log(data[obsloc, jtotarea])
            f = np.column_stack(
                [
                    data[obsloc, jdepvar],
                    np.exp(lpredict),
                    lactual,
                    lpredict,
                    lpredyld,
                    e,
                    np.sqrt(weights) * e,
                ]
            )
            if n_rch is not None:
                f = np.column_stack([f, n_rch])
        return f

    def _count_test_overflow(beta: np.ndarray) -> tuple[int, int]:
        n_rch_decay = 0
        n_incddsrc = 0
        if jdecvar.size > 0 and jdecvar.size == jbdecvar.size:
            term = -data[:, jdecvar] * beta[jbdecvar]
            n_rch_decay = int(np.sum(np.abs(term) > 709))
        if jdlvvar.size > 0 and jdlvvar.size == jbdlvvar.size:
            term = data[:, jdlvvar] * beta[jbdlvvar]
            if dlvdsgn.size > 0:
                try:
                    if dlvdsgn.shape[1] == term.shape[1]:
                        term = term @ dlvdsgn.T
                    elif dlvdsgn.shape[0] == term.shape[1]:
                        term = term @ dlvdsgn
                except Exception:
                    pass
            n_incddsrc = int(np.sum(np.abs(term) > 709))
        return n_rch_decay, n_incddsrc

    if if_test_calibrate:
        n_rch_decay, n_incddsrc = _count_test_overflow(beta0)
        test_resids = None
        if n_rch_decay > 0:
            warn(
                "Test calibration detected large reach-decay terms (abs(value) > 709); "
                "test_resids was not written."
            )
        if n_incddsrc > 0:
            warn(
                "Test calibration detected large incremental-delivery terms (abs(value) > 709); "
                "test_resids was not written."
            )
        if n_rch_decay == 0 and n_incddsrc == 0:
            outdat = feval(beta0, if_final_pass=True, include_test_n_rch=True)
            waterid = _first_token(config.get("waterid"))
            staid = _first_token(config.get("staid"))
            water_col = data[obsloc, jwaterid] if jwaterid >= 0 else np.full((len(obsloc),), np.nan)
            staid_col = data[obsloc, jstaid] if jstaid >= 0 else np.full((len(obsloc),), np.nan)
            out_cols = np.column_stack(
                [
                    water_col,
                    staid_col,
                    outdat[:, [0, 1, 2, 3, 5, 6, 7]],
                ]
            )
            test_resids = pd.DataFrame(
                out_cols,
                columns=[
                    waterid,
                    staid,
                    "actual",
                    "predict",
                    "ln_actual",
                    "ln_predict",
                    "ln_resid",
                    "weighted_ln_resid",
                    "n_rch",
                ],
            )
        return CalibrationResult(
            boot_betaest=None,
            summary_betaest=None,
            cov_betaest=None,
            resids=None,
            error_report=None,
            test_resids=test_resids,
            temp_beta=None,
            temp_rc=None,
            n_npos_flux=0,
            warnings=warnings,
            errors=errors,
        )

    bounds = config.get("blubnd")
    blbnd = _to_float_list(bounds[0]) if bounds else []
    bubnd = _to_float_list(bounds[1]) if bounds else []

    lb = np.array([b if b is not None else -np.inf for b in blbnd])
    ub = np.array([b if b is not None else np.inf for b in bubnd])
    fixed_mask = lb >= ub
    if np.any(fixed_mask):
        ub = ub.copy()
        ub[fixed_mask] = lb[fixed_mask] + 1e-12

    estimate = beta0.copy()
    beta0 = np.minimum(np.maximum(beta0, lb), ub)
    rc = 1
    jac_seed = None
    nlp_print = int(float(config.get("NLP_printing_option", 0) or 0))
    solver = str(config.get("calibration_solver") or "least_squares").strip().lower()
    calib_max_nfev = int(float(config.get("calibration_max_nfev", 1500) or 1500))
    calib_ftol = float(config.get("calibration_ftol", 1e-4) or 1e-4)
    calib_xtol = float(config.get("calibration_xtol", 1e-4) or 1e-4)
    calib_gtol = float(config.get("calibration_gtol", 1e-8) or 1e-8)
    calib_max_iter = int(float(config.get("calibration_max_iter", 15000) or 15000))

    if iter_val > 0 and _is_yes(config.get("if_parm_bootstrap")):
        if cov_estimate_input is None:
            error("cov_estimate_input is required for parametric bootstrap.")
            return empty_result()
        seed_1 = int(config.get("seed_1", 0))
        if rannor is None:
            error("rannor function required for parametric bootstrap.")
            return empty_result()
        z = rannor(seed_1, beta0.shape[0]).reshape(1, -1)
        root = _matrix_root_psd(cov_estimate_input)
        estimate = beta0 + (z @ root).reshape(-1)
        estimate = np.maximum(estimate, lb)
        estimate = np.minimum(estimate, ub)
    else:
        def residuals_fn(beta: np.ndarray) -> np.ndarray:
            return feval(beta, if_final_pass=False)

        estimate, jac_seed, rc = _solve_least_squares_problem(
            residuals_fn,
            beta0,
            lb,
            ub,
            use_scipy=use_scipy,
            linear_constraints=linear_constraints,
            solver=solver,
            max_nfev=calib_max_nfev,
            ftol=calib_ftol,
            xtol=calib_xtol,
            gtol=calib_gtol,
            max_iter=calib_max_iter,
            nlp_print=nlp_print,
        )

        if rc > 0 and iter_val == 0 and _is_yes(config.get("if_tp_local_refine")):
            parameter_names = _split_tokens(config.get("betalst"))
            selected_names = [
                name
                for name in _split_tokens(config.get("tp_local_refine_betas"))
                if name in parameter_names
            ]
            if not selected_names:
                warn(
                    "TP local refinement was requested but tp_local_refine_betas did not match any model coefficients."
                )
            else:
                free_idx = np.array(
                    [
                        parameter_names.index(name)
                        for name in selected_names
                    ],
                    dtype=int,
                )
                reduced_constraints = _reduce_linear_constraints(
                    linear_constraints, estimate, free_idx
                )
                anchor_map = _parse_float_mapping(config.get("tp_local_refine_anchor_targets"))
                anchor_weight = float(
                    config.get("tp_local_refine_anchor_weight", 0.0) or 0.0
                )
                anchor_positions: list[int] = []
                anchor_values: list[float] = []
                if anchor_weight > 0 and anchor_map:
                    for pos, name in enumerate(selected_names):
                        if name in anchor_map:
                            anchor_positions.append(pos)
                            anchor_values.append(float(anchor_map[name]))
                anchor_idx = np.array(anchor_positions, dtype=int)
                anchor_arr = np.array(anchor_values, dtype=float)
                local_solver = str(
                    config.get("tp_local_refine_solver") or solver
                ).strip().lower()
                local_max_nfev = int(
                    float(config.get("tp_local_refine_max_nfev", calib_max_nfev) or calib_max_nfev)
                )
                local_ftol = float(config.get("tp_local_refine_ftol", calib_ftol) or calib_ftol)
                local_xtol = float(config.get("tp_local_refine_xtol", calib_xtol) or calib_xtol)
                local_gtol = float(config.get("tp_local_refine_gtol", calib_gtol) or calib_gtol)
                local_max_iter = int(
                    float(config.get("tp_local_refine_max_iter", calib_max_iter) or calib_max_iter)
                )
                local_max_obj_increase_frac = float(
                    config.get("tp_local_refine_max_obj_increase_frac", 1e-3) or 1e-3
                )

                estimate_before = estimate.copy()
                obj_before = float(np.dot(residuals_fn(estimate_before), residuals_fn(estimate_before)))

                def local_residuals(free_beta: np.ndarray) -> np.ndarray:
                    beta_full = estimate_before.copy()
                    beta_full[free_idx] = free_beta
                    resid = residuals_fn(beta_full)
                    if anchor_weight > 0 and anchor_idx.size > 0:
                        anchor_resid = np.sqrt(anchor_weight) * (free_beta[anchor_idx] - anchor_arr)
                        resid = np.concatenate([resid, anchor_resid])
                    return resid

                local_estimate, local_jac, local_rc = _solve_least_squares_problem(
                    local_residuals,
                    estimate_before[free_idx],
                    lb[free_idx],
                    ub[free_idx],
                    use_scipy=use_scipy,
                    linear_constraints=reduced_constraints,
                    solver=local_solver,
                    max_nfev=local_max_nfev,
                    ftol=local_ftol,
                    xtol=local_xtol,
                    gtol=local_gtol,
                    max_iter=local_max_iter,
                    nlp_print=nlp_print,
                )

                if local_rc > 0:
                    candidate = estimate_before.copy()
                    candidate[free_idx] = local_estimate
                    resid_after = residuals_fn(candidate)
                    obj_after = float(np.dot(resid_after, resid_after))
                    if np.isfinite(obj_after) and obj_after <= obj_before * (1.0 + local_max_obj_increase_frac):
                        estimate = candidate
                        if (
                            anchor_weight <= 0
                            and jac_seed is not None
                            and local_jac is not None
                        ):
                            jac_seed = np.asarray(jac_seed, dtype=float).copy()
                            if jac_seed.shape[0] == local_jac.shape[0]:
                                jac_seed[:, free_idx] = local_jac
                            else:
                                jac_seed = None
                        else:
                            jac_seed = None
                    else:
                        warn(
                            "TP local refinement completed but did not improve the objective; retained the global estimate."
                        )
                else:
                    warn(
                        "TP local refinement failed; retained the global estimate."
                    )

    if iter_val == 0 and linear_constraints is not None:
        temp_beta = pd.DataFrame([estimate], columns=_split_tokens(config.get("betalst")))
        temp_rc = pd.DataFrame([{"rc": int(rc)}])

    if rc <= 0:
        error("Calibration optimization failed.")
        return CalibrationResult(
            None,
            None,
            None,
            None,
            pd.DataFrame(error_rows) if error_rows else None,
            None,
            temp_beta,
            temp_rc,
            0,
            warnings,
            errors,
        )

    outdat = feval(estimate, if_final_pass=True)
    error_report = pd.DataFrame(error_rows) if error_rows else None
    if n_npos_flux_final > 0:
        if iter_val > 0:
            warn(
                "Non-positive loads were predicted for monitored reaches in bootstrap "
                "iteration; current seed is invalidated."
            )
            return CalibrationResult(
                None,
                None,
                None,
                None,
                error_report,
                None,
                temp_beta,
                temp_rc,
                int(n_npos_flux_final),
                warnings,
                errors,
            )
        warn(
            "Non-positive loads were predicted for monitored reaches during calibration; "
            "check error_report/resids for diagnostics."
        )

    weighted_resid = outdat[:, 6] if outdat.shape[1] > 6 else outdat[:, -1]

    # Jacobian / leverage approximation
    def residuals_fn(beta: np.ndarray) -> np.ndarray:
        return feval(beta, if_final_pass=False)

    g_full = np.zeros((nobs, estimate.shape[0]))
    if jac_seed is not None:
        g_full = np.asarray(jac_seed, dtype=float)
    else:
        try:
            g_full = _numerical_jacobian(residuals_fn, estimate)
        except Exception:
            g_full = np.zeros((nobs, estimate.shape[0]))

    tol = 1e-6
    free_mask = np.ones((estimate.shape[0],), dtype=bool)
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    free_mask[finite_lb] &= np.abs(estimate[finite_lb] - lb[finite_lb]) > tol
    free_mask[finite_ub] &= np.abs(estimate[finite_ub] - ub[finite_ub]) > tol
    free_idx = np.where(free_mask)[0]
    if free_idx.size > 0:
        g_sub = np.nan_to_num(g_full[:, free_idx], nan=0.0, posinf=0.0, neginf=0.0)
        h_sub = g_sub.T @ g_sub
        try:
            h_sub_inv = np.linalg.pinv(h_sub)
            h_lev = np.sum((g_sub @ h_sub_inv) * g_sub, axis=1)
        except Exception:
            h_lev = np.zeros((nobs,), dtype=float)
    else:
        g_full[:, :] = 0.0
        h_lev = np.zeros((nobs,), dtype=float)

    boot_resid = weighted_resid / np.sqrt(np.maximum(boot_weights, 1) - h_lev)
    mean_exp_weighted_error = float((boot_weights @ np.exp(boot_resid)) / np.sum(boot_weights))
    var_exp_weighted_error = float(
        (boot_weights @ (np.exp(boot_resid) - mean_exp_weighted_error) ** 2) / np.sum(boot_weights)
    )

    parameters = _split_tokens(config.get("betalst"))
    boot_betaest = pd.DataFrame(
        [[iter_val, jter_val, *estimate.tolist(), mean_exp_weighted_error]],
        columns=["iter", "jter", *parameters, "mean_exp_weighted_error"],
    )

    summary_betaest = None
    cov_betaest = None
    resids = None
    test_resids = None

    if iter_val == 0:
        depvar_log = np.log(data[obsloc, jdepvar])
        larea = np.log(data[obsloc, jtotarea])
        lyield = depvar_log - larea

        n_linear_constraints = int(linear_constraints[0].shape[0]) if linear_constraints is not None else 0
        df_model = max(int(free_idx.size) - n_linear_constraints, 0)
        df_error = nobs - df_model
        sse = float(np.sum(weighted_resid**2))
        mse = float(sse / df_error)
        rmse = float(np.sqrt(mse))
        r_square = 1 - sse / (np.sum(depvar_log**2) - (np.sum(depvar_log) ** 2) / nobs)
        adj_r_square = 1 - ((nobs - 1) / df_error) * (1 - r_square)
        r_square_yield = 1 - sse / (np.sum(lyield**2) - (np.sum(lyield) ** 2) / nobs)

        v0 = np.zeros((estimate.shape[0], estimate.shape[0]), dtype=float)
        if free_idx.size > 0:
            g_sub = np.nan_to_num(g_full[:, free_idx], nan=0.0, posinf=0.0, neginf=0.0)
            h_sub = g_sub.T @ g_sub
            try:
                v0[np.ix_(free_idx, free_idx)] = np.linalg.pinv(h_sub)
            except Exception:
                v0[np.ix_(free_idx, free_idx)] = 0.0
        if linear_constraints is not None:
            A, _, _ = linear_constraints
            try:
                middle = A @ v0 @ A.T
                v0 = v0 - v0 @ A.T @ np.linalg.pinv(middle) @ A @ v0
            except Exception:
                pass
        cov_estimate = mse * v0
        sd_estimate = np.sqrt(np.diag(cov_estimate))

        t_stat = np.full_like(sd_estimate, np.nan, dtype=float)
        p_value = np.full_like(sd_estimate, np.nan, dtype=float)
        if free_idx.size > 0:
            valid = free_idx[sd_estimate[free_idx] > 0]
            t_stat[valid] = estimate[valid] / sd_estimate[valid]
            if t_dist is not None and df_error > 0:
                p_value[valid] = 2 * (1 - t_dist.cdf(np.abs(t_stat[valid]), df_error))

        map_resid = boot_resid / rmse
        z_map_resid = _norm_ppf(((_rank(map_resid) - 0.4) / (nobs + 0.2)))
        ppcc = np.corrcoef(map_resid, z_map_resid)[0, 1] if map_resid.size > 1 else np.nan
        if nobs > 3:
            z_up = 2 * _norm_ppf((1 + float(config.get("cov_prob", 0)) / 100) / 2) / np.sqrt(nobs - 3)
            lb = ((1 + ppcc) * np.exp(-z_up) - (1 - ppcc)) / ((1 + ppcc) * np.exp(-z_up) + (1 - ppcc))
            ub = ((1 + ppcc) * np.exp(z_up) - (1 - ppcc)) / ((1 + ppcc) * np.exp(z_up) + (1 - ppcc))
        else:
            lb = np.nan
            ub = np.nan

        swilk_stat, swilk_pval = _swilk(map_resid, nobs)

        vif = np.full(estimate.shape[0], np.nan, dtype=float)
        e_val = np.full(estimate.shape[0], np.nan, dtype=float)
        e_val_spread = np.nan
        if free_idx.size > 0:
            gc = g_full[:, free_idx].copy()
            var = np.mean(gc**2, axis=0) - (np.mean(gc, axis=0) ** 2)
            if np.any((var == 0) & (np.mean(gc, axis=0) != 0)):
                gc = gc - np.mean(gc, axis=0)
            c = gc.T @ gc
            diag = np.diag(c)
            if np.all(diag > 0):
                scale = np.diag(diag ** -0.5)
                c_corr = scale @ c @ scale
                vif_sub = np.diag(np.linalg.pinv(c_corr))
                eigval, eigvec = np.linalg.eig(c_corr)
                vif[free_idx[: len(vif_sub)]] = vif_sub
                e_val[free_idx[: len(eigval)]] = eigval
                if eigval.size and np.min(eigval) != 0:
                    e_val_spread = np.max(eigval) / np.min(eigval)

        summary_betaest = pd.DataFrame(
            [
                [
                    *estimate.tolist(),
                    *sd_estimate.tolist(),
                    mean_exp_weighted_error,
                    var_exp_weighted_error,
                    nobs,
                    df_error,
                    df_model,
                    sse,
                    mse,
                    rmse,
                    r_square,
                    adj_r_square,
                    r_square_yield,
                    e_val_spread,
                    ppcc,
                    swilk_stat,
                    swilk_pval,
                ]
            ],
            columns=[
                *parameters,
                *[f"sd_{p}" for p in parameters],
                "mean_exp_weighted_error",
                "var_exp_weighted_error",
                "nobs",
                "df_error",
                "df_model",
                "sse",
                "mse",
                "rmse",
                "r_square",
                "adj_r_square",
                "r_square_yield",
                "e_val_spread",
                "ppcc",
                "swilk_stat",
                "swilk_pval",
            ],
        )

        cov_betaest = pd.DataFrame(
            np.column_stack([cov_estimate, vif, e_val]),
            columns=[*parameters, "vif", "e_val"],
        )

        waterid = _first_token(config.get("waterid"))
        staid = _first_token(config.get("staid"))
        resid_names = [
            waterid,
            staid,
            "norm_weight",
            "actual",
            "predict",
            "ln_actual",
            "ln_predict",
            "ln_pred_yield",
            "ln_resid",
            "weighted_ln_resid",
            "map_resid",
            "boot_resid",
            "leverage",
            "z_map_resid",
            *parameters,
        ]

        outmat = np.column_stack(
            [
                data[obsloc][:, [jwaterid, jstaid]],
                weights.reshape(-1, 1),
                outdat,
                map_resid.reshape(-1, 1),
                boot_resid.reshape(-1, 1),
                h_lev.reshape(-1, 1),
                z_map_resid.reshape(-1, 1),
                g_full,
            ]
        )
        resids = pd.DataFrame(outmat, columns=resid_names)

        if _is_yes(config.get("if_test_calibrate")):
            test_cols = [
                waterid,
                staid,
                "actual",
                "predict",
                "ln_actual",
                "ln_predict",
                "ln_resid",
                "weighted_ln_resid",
                "n_rch",
            ]
            keep = [c for c in test_cols if c in resids.columns]
            test_resids = resids.loc[:, keep].copy()

    return CalibrationResult(
        boot_betaest=boot_betaest,
        summary_betaest=summary_betaest,
        cov_betaest=cov_betaest,
        resids=resids,
        error_report=error_report,
        test_resids=test_resids,
        temp_beta=temp_beta,
        temp_rc=temp_rc,
        n_npos_flux=int(n_npos_flux_final),
        warnings=warnings,
        errors=errors,
    )
