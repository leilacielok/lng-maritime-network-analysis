"""Shared panel preparation, response specifications and bounded likelihoods."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


PANEL_DIR = BASE_DIR / "data" / "processed" / "model_ready"


OUTPUT_DIR = BASE_DIR / "outputs" / "model"


BINARY_RESPONSES = [
    "is_articulation_point", "has_incident_global_bridge",
    "connected_to_articulation_point",
]


COMMON_PREDICTORS = [
    "neighbor_activity_lag1", "log1p_mean_voyage_distance_lag1",
    "month_sin", "month_cos",
]


SPECS = {
    "terminal": {
        "id": "terminal_id",
        "continuous": ["role_specific_pagerank", "counterparty_hhi_terminal",
                       "role_specific_dependence"],
        "binary": BINARY_RESPONSES.copy(),
        "predictors": COMMON_PREDICTORS + ["log1p_capacity_mtpa", "terminal_age"],
    },
    "chokepoint": {
        "id": "node_id",
        "continuous": ["pagerank", "betweenness", "share_monthly_network_flow"],
        "binary": ["is_articulation_point", "connected_to_articulation_point"],
        "predictors": COMMON_PREDICTORS,
    },
}


def standardize(frame):
    """Fit scaling on the estimation sample; save it for interpretation/reuse."""
    mean, sd = frame.mean(), frame.std(ddof=0)
    if (sd <= 0).any() or not np.isfinite(sd).all():
        raise ValueError(f"Constant/invalid columns: {sd.index[sd <= 0].tolist()}")
    values = ((frame - mean) / sd).to_numpy(dtype=float)
    return values, pd.DataFrame({"mean": mean, "sd": sd})


def prepare_panel(path, kind):
    spec = SPECS[kind]
    frame = pd.read_csv(path, dtype={spec["id"]: "string"})
    needed = [spec["id"], "node_name", "period_month", "active",
              "activity_lag1", "lag_available", *spec["predictors"],
              *spec["continuous"], *spec["binary"]]
    if kind == "terminal":
        needed += ["terminal_role_lag1"]
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    keys = [spec["id"], "period_month"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{path.name}: missing or duplicate node-month keys")
    frame["period_month"] = pd.to_datetime(frame["period_month"], errors="raise")
    if not frame["period_month"].dt.is_month_start.all():
        raise ValueError("period_month must contain first-of-month dates")
    for col in ["active", "lag_available"]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary without missing values")
    if not frame["activity_lag1"].dropna().isin([0, 1]).all():
        raise ValueError("activity_lag1 must be binary where observed")
    # Verify the supplied lag refers to the preceding CALENDAR month, not
    # the previous selected active row. Do not recalculate any panel metric.
    ordered = frame.sort_values(keys)
    grouped = ordered.groupby(spec["id"], sort=False)
    previous_month = grouped["period_month"].shift()
    previous_active = grouped["active"].shift()
    expected = previous_month.eq(ordered["period_month"] - pd.offsets.MonthBegin(1))
    if not ordered["lag_available"].eq(expected.astype(int)).all():
        raise ValueError("lag_available disagrees with calendar-month availability")
    if not ordered.loc[expected, "activity_lag1"].eq(previous_active[expected]).all():
        raise ValueError("activity_lag1 disagrees with previous calendar activity")
    audit = {"panel_rows": len(frame), "panel_nodes": frame[spec["id"]].nunique()}
    selected = frame.loc[frame["active"].eq(1)].copy()
    audit["excluded_inactive_current"] = len(frame) - len(selected)
    valid_lag = selected["lag_available"].eq(1)
    audit["excluded_no_calendar_lag"] = int((~valid_lag).sum())
    selected = selected.loc[valid_lag].copy()
    previous_active_mask = selected["activity_lag1"].eq(1)
    audit["excluded_inactive_previous"] = int((~previous_active_mask).sum())
    selected = selected.loc[previous_active_mask].copy()
    audit["consecutive_active_rows"] = len(selected)
    audit["excluded_missing_age"] = 0
    if kind == "terminal":
        audit["excluded_missing_age"] = int(selected["terminal_age"].isna().sum())
        selected = selected.loc[selected["terminal_age"].notna()].copy()
        if selected["terminal_age"].lt(0).any():
            raise ValueError("Negative terminal age: review the panel metadata")
        if not selected["terminal_role_lag1"].isin(["importer", "exporter"]).all():
            raise ValueError("Unexpected lagged role in consecutive-active sample")
        selected["exporter_lag1"] = selected["terminal_role_lag1"].eq("exporter").astype(int)
    selected = selected.sort_values(["period_month", spec["id"]]).reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No observations remain for {kind}")
    predictors = spec["predictors"] + (["exporter_lag1"] if kind == "terminal" else [])
    model_columns = predictors + spec["continuous"] + spec["binary"]
    values = selected[model_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Unexpected missing/nonfinite model values: "
                         f"{selected[model_columns].isna().sum().to_dict()}")
    if not selected[spec["binary"]].isin([0, 1]).all().all():
        raise ValueError("Structural responses must be binary")
    if not selected[spec["continuous"]].ge(0).all().all() or not selected[spec["continuous"]].le(1).all().all():
        raise ValueError("Continuous responses must lie in [0, 1]")
    if selected[spec["binary"]].nunique().lt(2).any():
        raise ValueError("A binary response is constant in the estimation sample")
    # Center/scale numeric covariates; keep role dummy on its 0/1 scale.
    x, x_scaling = standardize(selected[spec["predictors"]])
    if kind == "terminal":
        x = np.column_stack([x, selected["exporter_lag1"].to_numpy()])
        x_scaling.loc["exporter_lag1"] = [0.0, 1.0]
    yc, y_scaling = standardize(selected[spec["continuous"]])
    audit.update(estimation_rows=len(selected), estimation_nodes=selected[spec["id"]].nunique(),
                 predictors=predictors, continuous_responses=spec["continuous"],
                 binary_responses=spec["binary"],
                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return {"frame": selected, "x": x, "yc": yc,
            "yb": selected[spec["binary"]].to_numpy(dtype=int),
            "x_scaling": x_scaling, "y_scaling": y_scaling, "audit": audit}


def one_inflated_beta_logp(value, alpha, beta, boundary_logit, inflated):
    """Mixed measure: Lebesgue density on (0,1), point probability at one.

    Substitute 0.5 ONLY in the unselected Beta branch for exact boundaries;
    observations are never clipped, shifted, or replaced in the likelihood.
    """
    import pymc as pm
    import pytensor.tensor as pt
    inside = (value > 0) & (value < 1)
    safe_value = pt.switch(inside, value, 0.5)
    log_beta = pm.logp(pm.Beta.dist(alpha=alpha, beta=beta), safe_value)
    log_one = -pt.logaddexp(0.0, -boundary_logit)
    log_interior = -pt.logaddexp(0.0, boundary_logit)
    continuous = log_beta + pt.switch(inflated, log_interior, 0.0)
    return pt.switch(inside, continuous,
                     pt.switch(pt.eq(value, 1) & inflated, log_one, -np.inf))


def one_inflated_beta_random(alpha, beta, boundary_logit, inflated, rng=None, size=None):
    rng = np.random.default_rng() if rng is None else rng
    # Stable sigmoid, including extreme logits in predictive draws.
    prob = np.exp(-np.logaddexp(0.0, -np.asarray(boundary_logit)))
    prob = np.where(inflated, prob, 0.0)
    interior = rng.beta(alpha, beta, size=size)
    # Floating-point beta routines may round to 0/1; keep interior draws interior.
    interior = np.clip(interior, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return np.where(rng.uniform(size=np.shape(interior)) < prob, 1.0, interior)


def zero_inflated_beta_logp(value, alpha, beta, boundary_logit, inflated):
    # Reflecting the Beta and boundary gives exactly a zero-inflated Beta.
    return one_inflated_beta_logp(1 - value, beta, alpha, boundary_logit, inflated)


def zero_inflated_beta_random(alpha, beta, boundary_logit, inflated, rng=None, size=None):
    rng = np.random.default_rng() if rng is None else rng
    prob = np.exp(-np.logaddexp(0.0, -np.asarray(boundary_logit)))
    prob = np.where(inflated, prob, 0.0)
    interior = np.clip(rng.beta(alpha, beta, size=size),
                       np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return np.where(rng.uniform(size=np.shape(interior)) < prob, 0.0, interior)


