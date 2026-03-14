import numpy as np
from scipy.integrate import solve_ivp

from .config import DEFAULT_PARAMS, ODE_TOLERANCES
from .reduced_model import REDUCED_SPECIES, reduced_vkb_ode


def run_reduced_simulation(
    s_R=0.2,
    t_max=400,
    y0=None,
    method="BDF",
    rtol=None,
    atol=None,
    num_points=1000,
):
    params = {**DEFAULT_PARAMS, "s_R": s_R}
    
    # Use unified tolerances if not explicitly provided
    if rtol is None:
        rtol = ODE_TOLERANCES['rtol']
    if atol is None:
        atol = ODE_TOLERANCES['atol']

    if y0 is None:
        y0 = [0.0, 0.0]

    t_eval = np.linspace(0, t_max, int(num_points))
    sol = solve_ivp(
        lambda t, y: reduced_vkb_ode(t, y, params),
        (0, t_max),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    result = {"time": sol.t}
    for index, name in enumerate(REDUCED_SPECIES):
        result[name] = sol.y[index]
    result["solver_stats"] = {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "method": method,
        "nfev": int(getattr(sol, "nfev", -1)),
        "njev": int(getattr(sol, "njev", -1)),
        "nlu": int(getattr(sol, "nlu", -1)),
    }
    return result
