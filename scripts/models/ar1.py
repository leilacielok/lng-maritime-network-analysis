"""Stationary, non-centred AR(1) residuals on irregular node-month panels."""
import numpy as np


def build_ar1_factor(data, factor_mean):
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    n = len(data["frame"])
    if not all(k in data for k in ("ar1_order", "ar1_restore_index", "ar1_gaps", "ar1_same_node")):
        raise ValueError("Call prepare_ar1_index(data, kind) before building an AR(1) model")
    rho = pm.Uniform("rho", lower=-1, upper=1)
    u = pm.Normal("ar1_innovation", mu=0, sigma=1, shape=n)
    gaps = pt.as_tensor_variable(np.asarray(data["ar1_gaps"], dtype="int64"))
    same = pt.as_tensor_variable(np.asarray(data["ar1_same_node"], dtype="bool"))
    ordered_u = u[data["ar1_order"]]

    def transition(innovation, gap, continued, previous, coefficient):
        persistence = coefficient ** gap
        return pt.switch(continued,
                         persistence * previous +
                         pt.sqrt(pt.maximum(0.0, 1.0 - persistence**2)) * innovation,
                         innovation)

    ordered_z, _ = pytensor.scan(
        fn=transition,
        sequences=[ordered_u, gaps, same],
        outputs_info=pt.as_tensor_variable(np.float64(0.0)),
        non_sequences=[rho], strict=True)
    residual = ordered_z[data["ar1_restore_index"]]
    pm.Deterministic("factor_residual", residual, dims="obs")
    return pm.Deterministic("factor", factor_mean + residual, dims="obs")


def ar1_initial_values(starts, data):
    """Replace IID factor starts with non-centred innovation starts."""
    import numpy as np
    for start in starts:
        start.pop("factor", None)  # deterministic in AR(1), not a free variable
        start["rho"] = 0.0
        start["ar1_innovation"] = np.zeros(len(data["frame"]), dtype=float)
    return starts
